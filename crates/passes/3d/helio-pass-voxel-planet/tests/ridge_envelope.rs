//! Display generation keeps mountain mass; canonical field queries stay exact.
mod common;
use common::*;
use helio_pass_voxel_planet::{
    edits::FaceBrush,
    grid::Grid,
    landform::{Landform, LandformConstants, LandformField},
    noise::{hash3, mul_fine, noise_fine, FINE_ONE},
    TerrainField,
};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct World {
    grid: [i32; 4],
    scale: [u32; 4],
    bounds: [[i32; 4]; 6],
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Probe {
    cell: [i32; 4],
    chart: [u32; 4],
}

// Only continents and region masks determine unresolved mountain placement.
// This independently computes those masks, not the production display loop.
fn masks(k: &LandformConstants, p: glam::IVec3) -> (i32, i32) {
    let mut q = p;
    for o in &k.octaves[..6] {
        q[(o.kind - 4) as usize] += mul_fine(o.amplitude, noise_fine(p, o.shift, o.seed));
    }
    let mut continent = 0i32;
    let mut mask = 0i32;
    for o in &k.octaves[6..k.header[0] as usize] {
        if o.kind == 0 {
            continent += mul_fine(noise_fine(q, o.shift, o.seed), o.amplitude << 8);
        }
        if o.kind == 1 {
            mask += mul_fine(noise_fine(q, o.shift, o.seed), o.amplitude << 8);
        }
    }
    (
        (continent * 3).clamp(0, FINE_ONE),
        ((mask - (k.shape[0] << 8)) * 3).clamp(0, FINE_ONE),
    )
}

#[test]
fn production_generation_retains_ridge_envelope_and_canonical_queries() {
    let Some(gpu) = gpu() else {
        panic!("ridge envelope GPU regression needs an adapter")
    };
    let source_engine = include_str!("../src/engine.rs");
    let wrapper = source_engine
        .lines()
        .find(|line| {
            line.contains("s.push_str(\"fn generation_height")
                && line.contains("terrain_display_height")
        })
        .unwrap()
        .trim()
        .strip_prefix("s.push_str(\"")
        .unwrap()
        .strip_suffix("\");")
        .unwrap()
        .replace("\\n", "\n");
    let generation = include_str!("../shaders/generate.wgsl");
    // Execute the actual production gate and height invocation. An active
    // topology brush must only disable fraction metadata, not neighbor height.
    let gates = [
        "let display_base =",
        "let requested_relief =",
        "let height = generation_height",
    ]
    .map(|prefix| {
        generation
            .lines()
            .find(|line| line.trim_start().starts_with(prefix))
            .unwrap()
    })
    .join("\n");
    let world_source = include_str!("../shaders/world.wgsl");
    let common_source = include_str!("../shaders/common.wgsl");
    let brush_start = common_source.find("struct FaceBrush").unwrap();
    let brush_struct = &common_source
        [brush_start..brush_start + common_source[brush_start..].find('\n').unwrap() + 1];
    let brush_struct = if brush_struct.contains('}') {
        brush_struct
    } else {
        &common_source
            [brush_start..brush_start + common_source[brush_start..].find("\n}").unwrap() + 2]
    };
    let function = |name: &str| {
        let start = common_source.find(&format!("fn {name}(")).unwrap();
        &common_source[start..start + common_source[start..].find("\n}").unwrap() + 2]
    };
    let contains = function("brush_contains");
    let apply = function("apply_edits");
    let noise_source = include_str!("../shaders/noise.wgsl");
    let landform_source = include_str!("../shaders/landform.wgsl");
    let source = format!(
        r#"
        {noise_source}
        {world_source}
        {landform_source}
        var<private> material_footprint:f32=0.0;
        var<private> material_radial_span:f32=0.0;
        var<private> material_stone_coverage:f32=-1.0;
        var<private> material_snow_mix:vec4<f32>=vec4<f32>(-1.0,0.0,0.0,0.0);
        var<private> material_rock_id:u32=0u;
        var<private> material_rock_base_id:u32=0u;
        struct Frame {{hints:vec4<u32>}}
        struct Probe {{cell:vec4<i32>,chart:vec4<u32>}}
        @group(0) @binding(0) var<uniform> frame:Frame;
        @group(0) @binding(1) var<uniform> world:World;
        @group(0) @binding(2) var<uniform> terrain:TerrainConstants;
        @group(0) @binding(3) var<storage,read> probes:array<Probe>;
        @group(0) @binding(4) var<storage,read_write> answers:array<vec4<i32>>;
        {brush_struct}
        @group(0) @binding(5) var<storage,read> brushes:array<FaceBrush>;
        @group(0) @binding(6) var<storage,read> edit_refs:array<u32>;
        {contains}
        {apply}
        fn is_plane()->bool {{return false;}}
        {wrapper}
        @compute @workgroup_size(64)
        fn probe(@builtin(global_invocation_id) id:vec3<u32>) {{
            if id.x>=arrayLength(&probes) {{return;}}
            let p=probes[id.x];
            let face=p.chart.x;
            let i=p.cell.x;
            let j=p.cell.y;
            let level=u32(p.cell.z);
            let brush=brushes[p.chart.y];
            let op=(brush.flags>>4u)&3u;
            let topology_flags=select(0u,1u,p.chart.w!=0u && brush.radius_half>=(1u<<level) && op<2u);
            {gates}
            answers[id.x*2u]=vec4<i32>(field_height(face,i,j,level),height,
                i32(requested_relief),i32(display_base));
            let kind_in=select(1u,0u,op==1u);
            let edited=apply_edits(p.chart.w,level,brush.center.xyz,kind_in).x;
            let untouched=apply_edits(p.chart.w,level,brush.center.xyz+vec3<i32>(i32(brush.radius_half)*2+1,0,0),kind_in).x;
            answers[id.x*2u+1u]=vec4<i32>(i32(edited),i32(untouched),i32(kind_in),i32(topology_flags));
        }}
    "#
    );
    let module = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("production ridge envelope"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let pipeline = gpu
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("display generation mode"),
            layout: None,
            module: &module,
            entry_point: Some("probe"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[("RIDGE_DISPLAY_GENERATION", 1.0)],
                ..Default::default()
            },
            cache: None,
        });
    let canonical_pipeline = gpu
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("canonical verification mode"),
            layout: None,
            module: &module,
            entry_point: Some("probe"),
            compilation_options: Default::default(),
            cache: None,
        });
    let grid = Grid::new(6_371_000.0, 0.1).unwrap();
    let mut retained = 0;
    let mut exact = 0;
    let mut edits = 0;
    for (mountain_km, mountain_m) in [(20.0, 2400.0), (20.0, -2400.0), (0.000001, 20.0)] {
        let field = LandformField::new(
            &grid,
            &Landform {
                mountain_km,
                mountain_m,
                ..Default::default()
            },
            7,
        );
        let k = field.constants();
        let max_shift = k.octaves[..k.header[0] as usize]
            .iter()
            .filter(|o| o.kind == 2)
            .map(|o| o.shift)
            .max()
            .unwrap();
        let min_shift = k.octaves[..k.header[0] as usize]
            .iter()
            .filter(|o| o.kind == 2)
            .map(|o| o.shift)
            .min()
            .unwrap();
        let far = max_shift.saturating_sub(1).max(1);
        let fine = min_shift.saturating_sub(3);
        let levels = [
            0,
            fine,
            max_shift.saturating_sub(3),
            max_shift.saturating_sub(2),
            far,
            4,
        ];
        let program = field.program();
        assert_eq!(program.constants.len(), 1616);
        let lo = i32::from_le_bytes(
            program.constants[560 + 31 * 4..560 + 32 * 4]
                .try_into()
                .unwrap(),
        );
        let hi = i32::from_le_bytes(
            program.constants[560 + 32 * 4..560 + 33 * 4]
                .try_into()
                .unwrap(),
        );
        // Initial weight is 1 minus one Q24 unit. Independently interpolate
        // the final interval using i64 multiplication, not the WGSL helper.
        let fraction = (((393216 - 1) << 7) / 3) as i64;
        let mean = lo + (i64::from(hi - lo) * fraction / i64::from(FINE_ONE)) as i32;
        let mut probes = Vec::new();
        let mut brush_values = Vec::new();
        let mut refs = Vec::new();
        for (n, level) in levels.into_iter().enumerate() {
            let edge = grid.cells() >> level;
            for index in 0..384 {
                let i = (hash3(index, n as i32, 0, 991) & 0x7fffffff) as i32 % edge;
                let j = (hash3(index, n as i32, 1, 991) & 0x7fffffff) as i32 % edge;
                let face = index as u32 % 6;
                // Pair identical untouched lanes with and without an active
                // topology brush elsewhere in the same 8x8 column.
                for topology in [0, 1, 2] {
                    let brush_index = brush_values.len() as u32;
                    // Radius is active at the tested level, including the
                    // tiny-wavelength fixture where a one-metre brush is active.
                    let radius = (1u32 << level).max(20);
                    let op = u32::from(topology == 2);
                    brush_values.push(FaceBrush {
                        flags: face | (op << 4),
                        radius_half: radius,
                        pad: [0; 2],
                        center: [0, 0, 0, 0],
                    });
                    let list = refs.len() as u32 + 1;
                    refs.extend_from_slice(&[1, brush_index]);
                    probes.push(Probe {
                        cell: [i, j, level as i32, topology],
                        chart: [face, brush_index, 0, if topology == 0 { 0 } else { list }],
                    });
                }
            }
        }
        let bounds = field.render_bound_margins();
        let world = World {
            grid: [
                grid.reference_cells(),
                grid.layer_mm() as i32,
                grid.cells(),
                grid.level_offset() as i32,
            ],
            scale: [grid.domain_scale(), 0, 0, 0],
            bounds: std::array::from_fn(|i| std::array::from_fn(|j| bounds[i * 4 + j])),
        };
        let input = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&probes),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let uniform = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: &program.constants,
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let world_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&world),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let hints = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&[0u32, 0, 0, 8]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let brush_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&brush_values),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let refs_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&refs),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let bytes = probes.len() as u64 * 32;
        let output = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let run = |pipeline: &wgpu::ComputePipeline| {
            let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: hints.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: world_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: input.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: output.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: brush_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: refs_buffer.as_entire_binding(),
                    },
                ],
            });
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups((probes.len() as u32 + 63) / 64, 1, 1);
            }
            gpu.queue.submit([encoder.finish()]);
            let data = read_buffer(&gpu, &output, bytes);
            bytemuck::cast_slice::<u8, [i32; 4]>(&data).to_vec()
        };
        let canonical = run(&canonical_pipeline);
        let display = run(&pipeline);
        for (index, probe) in probes.iter().enumerate() {
            let canonical = canonical[index * 2];
            let height = display[index * 2];
            let occupancy = display[index * 2 + 1];
            let level = probe.cell[2] as u32;
            let point =
                grid.domain_point(probe.chart[0] as u8, probe.cell[0], probe.cell[1], level);
            let cpu = field.height(point, level + grid.level_offset());
            assert_eq!(canonical[0], cpu, "canonical GPU field parity");
            assert_eq!(
                canonical[1], cpu,
                "default override must preserve canonical entry"
            );
            assert_eq!(
                height[0], cpu,
                "display compilation must not contaminate canonical queries"
            );
            if level == 0 || level + 3 <= min_shift {
                assert_eq!(
                    height[1], cpu,
                    "fully resolved and L0 generation stay exact"
                );
                exact += 1;
            }
            if level > 0 && level + 2 > max_shift {
                let (land, region) = masks(k, point);
                let expected = cpu + mul_fine(mul_fine(mean, region), land);
                assert!(
                    (i64::from(height[1]) - i64::from(expected)).abs() <= 2,
                    "coarse field must retain signed mask-scaled mountain mass: {} vs {expected}",
                    height[1]
                );
                if expected != cpu {
                    retained += 1;
                }
            }
            if probe.cell[3] != 0 {
                assert_eq!(
                    height[1],
                    display[(index - probe.cell[3] as usize) * 2][1],
                    "editing another lane must not drop the envelope"
                );
                assert_eq!(
                    height[2], 0,
                    "mixed topology cannot claim heightfield fractions"
                );
                let expected = u32::from(probe.cell[3] == 2) as i32;
                assert_eq!(
                    occupancy[0], expected,
                    "ordered Dig/Build must update the contained cell"
                );
                assert_eq!(
                    occupancy[1], occupancy[2],
                    "ordered Dig/Build must preserve an untouched neighbor cell"
                );
                edits += 1;
            }
        }
    }
    assert!(
        retained > 100,
        "fixture must expose old missing-mountain failure"
    );
    assert!(exact > 100);
    assert!(edits > 100);
    eprintln!("production GPU probes: retained={retained}, canonical/fully-resolved={exact}, untouched edited-column neighbors={edits}");
}
