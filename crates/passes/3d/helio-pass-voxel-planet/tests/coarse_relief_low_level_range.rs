//! Direct trace-range regression; no renderer API additions.
mod common;
use common::*;
use helio_pass_voxel_planet::grid::Shape;
use helio_pass_voxel_planet::{Planet, PlanetRecipe};

#[test]
fn l1_to_l5_fractional_top_hits_respect_requested_trace_range() {
    let Some(gpu) = gpu() else {
        eprintln!("SKIP: no GPU adapter available for this rendering fixture");
        return;
    };
    for level in 1u32..=5 {
        for ceil_top in [16i32, -8i32] {
            let quantum = 0.1 * (1u32 << level) as f32;
            let physical = (ceil_top - 1) as f32 * quantum + 0.1;
            let expected = 100.0 - physical;
            let short = expected - 0.05;
            let long = expected + 0.05;
            let planet = Planet::new(PlanetRecipe {
                shape: Shape::Plane,
                terrain: helio_pass_voxel_planet::layers::TerrainLayers::flat().source(7),
                ..Default::default()
            })
            .unwrap();
            let mut source = String::from(include_str!("../shaders/noise.wgsl"));
            source.push_str(include_str!("../shaders/world.wgsl"));
            source.push_str(&planet.field().program().wgsl);
            source.push_str(
                &include_str!("../shaders/common.wgsl")
                    .replace("ACCESS", "read")
                    .replace("LEVEL_TOP", "i32")
                    .replace("BLOCK_ENTRY", "vec4<i32>")
                    .replace("SHAPE_ID", "1u"),
            );
            source.push_str(include_str!("../shaders/trace.wgsl"));
            source.push_str(&format!(
                r#"
        @group(0) @binding(17) var<storage,read_write> range_hits:array<Hit>;
        @compute @workgroup_size(1) fn range_probe() {{
            let r=make_ray(vec3<f32>(0.0),vec3<f32>(0.0,-1.0,0.0));
            range_hits[0]=trace(r,0.0,{short},0.0,1.0,0.0);
            range_hits[1]=trace(r,0.0,{long},0.0,1.0,0.0);
        }}
    "#
            ));
            let mut frame = vec![0u8; 2144];
            fn ints(bytes: &mut [u8], offset: usize, values: &[i32]) {
                for (i, v) in values.iter().enumerate() {
                    bytes[offset + i * 4..offset + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
                }
            }
            fn floats(bytes: &mut [u8], offset: usize, values: &[f32]) {
                for (i, v) in values.iter().enumerate() {
                    bytes[offset + i * 4..offset + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
                }
            }
            let face = 2 * 80;
            floats(&mut frame, face, &[1.0, 0.0, 0.0, 0.0]);
            floats(&mut frame, face + 32, &[0.0, 0.0, -1.0, 0.0]);
            ints(&mut frame, face + 64, &[512, 512, 1, 0]);
            floats(&mut frame, 480, &[0.0, 1.0, 0.0, 100.0]);
            floats(&mut frame, 496, &[0.0, 0.1, 0.1, 0.0]);
            ints(&mut frame, 512, &[1000, 1024, 7, 2]);
            floats(
                &mut frame,
                528,
                &[
                    expected / (1.5 * (1u32 << (level - 1)) as f32),
                    0.0,
                    -1000.0,
                    1000.0,
                ],
            );
            floats(&mut frame, 544, &[1.0, 1.0, 0.0, 0.0]);
            ints(&mut frame, 576, &[0, 0, 0, 4]);
            let mut world = vec![0u8; 144];
            ints(&mut world, 0, &[1024, 100, 1024, 0]);
            let ci = (512 >> level) / 8;
            let k_lo = ceil_top / 8 - 1;
            let mut record = vec![0u8; 32];
            ints(
                &mut record,
                0,
                &[
                    (ci | (2 << 24) | (level << 27)) as i32,
                    ci as i32,
                    k_lo,
                    0x90000001u32 as i32,
                    0,
                    0,
                    1,
                    0,
                ],
            );
            let fraction = 1u32 << (16 - level);
            let mut pool = vec![0u32; 64];
            pool[..16].fill(0x08080808);
            pool[16..48].fill(fraction | (fraction << 16));
            let buffer = |label: &str, data: &[u8], uniform: bool| {
                let b = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: data.len() as u64,
                    usage: (if uniform {
                        wgpu::BufferUsages::UNIFORM
                    } else {
                        wgpu::BufferUsages::STORAGE
                    }) | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                gpu.queue.write_buffer(&b, 0, data);
                b
            };
            let frame = buffer("range frame", &frame, true);
            let world = buffer("range world", &world, true);
            let table = buffer("range table", &[0; 4], false);
            let records = buffer("range record", &record, false);
            let pool = buffer("range pool", bytemuck::cast_slice(&pool), false);
            let brushes = buffer("range empty brushes", &[0; 32], false);
            let refs = buffer("range empty refs", &[0; 4], false);
            let tops = buffer(
                "range tops",
                bytemuck::cast_slice(&[ceil_top << level; 64]),
                false,
            );
            let blocks = buffer("range empty blocks", &vec![0; 1048576], false);
            let constants = buffer("range constants", &planet.field().program().constants, true);
            let out = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("range output"),
                size: 64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let bindings = [
                (0, &frame, true, false),
                (1, &world, true, false),
                (2, &table, false, false),
                (3, &records, false, false),
                (4, &pool, false, false),
                (5, &brushes, false, false),
                (6, &refs, false, false),
                (14, &tops, false, false),
                (15, &blocks, false, false),
                (16, &constants, true, false),
                (17, &out, false, true),
            ];
            let entries: Vec<_> = bindings
                .iter()
                .map(
                    |(binding, _, uniform, writable)| wgpu::BindGroupLayoutEntry {
                        binding: *binding,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: if *uniform {
                                wgpu::BufferBindingType::Uniform
                            } else {
                                wgpu::BufferBindingType::Storage {
                                    read_only: !*writable,
                                }
                            },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                )
                .collect();
            let layout = gpu
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &entries,
                });
            let bind_entries: Vec<_> = bindings
                .iter()
                .map(|(binding, b, _, _)| wgpu::BindGroupEntry {
                    binding: *binding,
                    resource: b.as_entire_binding(),
                })
                .collect();
            let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &bind_entries,
            });
            let pl = gpu
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[Some(&layout)],
                    immediate_size: 0,
                });
            let shader = gpu
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("range trace shader"),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                });
            let pipeline = gpu
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: Some(&pl),
                    module: &shader,
                    entry_point: Some("range_probe"),
                    compilation_options: Default::default(),
                    cache: None,
                });
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            gpu.queue.submit([encoder.finish()]);
            let data = read_buffer(&gpu, &out, 64);
            let word = |n: usize| u32::from_le_bytes(data[n * 4..n * 4 + 4].try_into().unwrap());
            assert_eq!(
                word(4) & 3,
                0,
                "clipped range incorrectly hit: {:?}",
                &data[..32]
            );
            assert_eq!(
                word(12) & 3,
                1,
                "long range failed to hit: {:?}",
                &data[32..]
            );
            assert_eq!(
                (word(12) >> 5) & 31,
                level,
                "synthetic hit did not select requested level"
            );
            let t = f32::from_bits(word(8));
            assert!((t-expected).abs()<0.002,"L{level} ceil_top={ceil_top} stored radial top was lost: t={t} expected={expected}");
            eprintln!(
                "range L{level} ceil_top={ceil_top}: clipped MISS + exact full-brick top HIT"
            );
        }
    }
}
