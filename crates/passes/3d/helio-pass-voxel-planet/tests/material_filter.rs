//! Appearance-only procedural filtering; canonical material queries remain exact.
mod common;
use common::*;
use wgpu::util::DeviceExt;

#[test]
fn unresolved_outcrop_preserves_canonical_ids_and_mean_palette() {
    let Some(gpu) = gpu() else { return };
    let noise = include_str!("../shaders/noise.wgsl");
    let landform = include_str!("../shaders/landform.wgsl");
    let world = include_str!("../shaders/world.wgsl");
    let materials = world
        .split("const M_AIR")
        .nth(1)
        .unwrap()
        .split("// Face bases")
        .next()
        .unwrap();
    let source = format!(
        r#"
        {noise}
        const M_AIR{materials}
        {landform}
        @group(0) @binding(0) var<uniform> terrain:TerrainConstants;
        @group(0) @binding(1) var<storage,read> points:array<vec4<i32>>;
        @group(0) @binding(2) var<storage,read_write> answers:array<vec4<u32>>;
        @group(0) @binding(3) var<storage,read_write> coverage:array<vec4<f32>>;
        var<private> material_footprint:f32=0.0;
        var<private> material_snow_mix:vec4<f32>=vec4<f32>(-1.0,0.0,0.0,0.0);
        var<private> material_rock_id:u32=0u;
        @compute @workgroup_size(64) fn probe(@builtin(global_invocation_id) id:vec3<u32>) {{
            if id.x>=arrayLength(&points) {{return;}}
            let p=points[id.x].xyz;
            material_footprint=0.0;
            let canonical=ground_material(p,4000000,0,5,39999);
            material_footprint=0.1;
            let near=ground_material(p,4000000,0,5,39999);
            let near_disabled=u32(material_snow_mix.x<0.0);
            material_footprint=128.0;
            let far=ground_material(p,4000000,0,5,39999);
            coverage[id.x*2u]=material_snow_mix;
            let far_rock=material_rock_id;
            // The 6.4m octave is unresolved here; the 51.2m octave remains.
            material_footprint=6.4;
            let middle=ground_material(p,4000000,0,5,39999);
            coverage[id.x*2u+1u]=material_snow_mix;
            let middle_rock=material_rock_id;
            material_footprint=0.0;
            let reset=ground_material(p,4000000,0,5,39999);
            answers[id.x*2u]=vec4<u32>(canonical,near,far,middle);
            answers[id.x*2u+1u]=vec4<u32>(far_rock,middle_rock,near_disabled,u32(material_snow_mix.x<0.0 && reset==canonical));
        }}
    "#
    );
    let mut constants = [0i32; 140];
    constants[1] = 100;
    constants[2] = 7;
    constants[3] = 123;
    constants[6] = 2_000_000;
    constants[9] = 16;
    constants[12 + 6 * 4] = 10;
    let points: Vec<[i32; 4]> = (0..16384i32)
        .map(|i| {
            // Actual cube-face/plane slices: one domain coordinate is fixed.
            let x = (helio_pass_voxel_planet::noise::hash3(i, 0, 0, 123) & 0xffffff) as i32 * 2 + 1;
            let z = (helio_pass_voxel_planet::noise::hash3(i, 2, 0, 123) & 0xffffff) as i32 * 2 + 1;
            [x, 1 << 27, z, 0]
        })
        .collect();
    let terrain = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&constants),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let input = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&points),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let output = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (points.len() * 32) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let coverage_output = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (points.len() * 32) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let shader = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("material appearance filtering"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let pipeline = gpu
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("probe"),
            compilation_options: Default::default(),
            cache: None,
        });
    let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: terrain.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: coverage_output.as_entire_binding(),
            },
        ],
    });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups((points.len() as u32 + 63) / 64, 1, 1);
    }
    gpu.queue.submit([encoder.finish()]);
    let bytes = read_buffer(&gpu, &output, (points.len() * 32) as u64);
    let answers: &[[u32; 4]] = bytemuck::cast_slice(&bytes);
    let coverage_bytes = read_buffer(&gpu, &coverage_output, (points.len() * 32) as u64);
    let coverage: &[[f32; 4]] = bytemuck::cast_slice(&coverage_bytes);
    let colours = [
        [200., 0., 200.],
        [91., 125., 65.],
        [120., 87., 61.],
        [133., 139., 142.],
        [203., 188., 151.],
        [217., 228., 236.],
        [28., 72., 92.],
        [116., 111., 102.],
        [185., 142., 104.],
        [82., 88., 95.],
    ];
    let linear = |id: u32, channel: usize| (colours[id as usize][channel] / 255f32).powf(2.2);
    let mut canonical_mean = [0f32; 3];
    let mut filtered_mean = [[0f32; 3]; 2];
    let mut canonical_snow = 0usize;
    let mut filtered_snow = 0f32;
    let mut middle_range = [1f32, 0f32];
    for index in 0..points.len() {
        let ids = answers[index * 2];
        let meta = answers[index * 2 + 1];
        assert_eq!(
            ids, [ids[0]; 4],
            "canonical material changed at point {index}"
        );
        assert_eq!(
            meta[2..],
            [1, 1],
            "near/default queries leaked coverage at point {index}"
        );
        canonical_snow += usize::from(ids[0] == 5);
        filtered_snow += coverage[index * 2][0];
        middle_range[0] = middle_range[0].min(coverage[index * 2 + 1][0]);
        middle_range[1] = middle_range[1].max(coverage[index * 2 + 1][0]);
        for channel in 0..3 {
            canonical_mean[channel] += linear(ids[0] & 255, channel);
            for filter in 0..2 {
                let w = coverage[index * 2 + filter];
                assert!(w.iter().all(|x| x.is_finite() && *x >= 0.0 && *x <= 1.0));
                assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-5);
                filtered_mean[filter][channel] += w[0] * linear(5, channel)
                    + w[1] * linear(meta[filter], channel)
                    + w[2] * linear(9, channel)
                    + w[3] * linear(2, channel);
            }
        }
    }
    let n = points.len() as f32;
    assert!(canonical_snow > 0 && canonical_snow < points.len());
    assert!(
        (filtered_snow / n - canonical_snow as f32 / n).abs() < 0.02,
        "unresolved snow/rock coverage must retain the canonical class mean"
    );
    assert!(
        middle_range[1] - middle_range[0] > 0.5,
        "resolved broad outcrops must keep spatial variation"
    );
    for mean in filtered_mean {
        for channel in 0..3 {
            assert!(
                (mean[channel] / n - canonical_mean[channel] / n).abs() < 0.02,
                "filtered palette mean differs: {} vs {}",
                mean[channel] / n,
                canonical_mean[channel] / n
            );
        }
    }
}
