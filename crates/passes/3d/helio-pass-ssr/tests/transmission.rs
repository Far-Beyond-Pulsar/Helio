use helio_core::{BlasGeometry, BlasManager, TlasInstanceInput, TlasManager};
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[test]
#[ignore = "requires hardware ray queries"]
fn reflected_segment_filters_only_sheets_before_nearest_opaque_hit() {
    pollster::block_on(async {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::EXPERIMENTAL_RAY_QUERY,
                required_limits: adapter.limits(),
                experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
                ..Default::default()
            })
            .await
            .unwrap();
        let device = Arc::new(device);
        // Exercise the production query function verbatim, independently of
        // screen radiance availability or the reflection composition pass.
        let source = include_str!("../shaders/ssr_trace_rt.wgsl");
        let start = source.find("struct RayTransmissionData").unwrap();
        let end = source.find("// Screen color").unwrap();
        let query = source[start..end].replace("@group(2) @binding(2)", "@group(0) @binding(1)");
        let shader=device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("reflection transmission contract"),
            source: wgpu::ShaderSource::Wgsl(format!(r#"
                enable wgpu_ray_query;
                @group(0) @binding(0) var acc_struct: acceleration_structure;
                @group(0) @binding(2) var<storage,read_write> result: array<vec4<f32>,2>;
                const MAX_RAY_DIST: f32=100.0;
                {query}
                @compute @workgroup_size(1) fn main() {{
                    let hit=ray_query_hit_position(vec3<f32>(0.0),vec3<f32>(0.0),vec3<f32>(1.0,0.0,0.0));
                    result[0]=hit.position;
                    result[1]=vec4<f32>(hit.throughput,1.0);
                }}
            "#).into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[
                0.0f32, -10.0, -10.0, 0.0, 10.0, -10.0, 0.0, 0.0, 10.0,
            ]),
            usage: wgpu::BufferUsages::BLAS_INPUT,
        });
        let mut blas = BlasManager::new(device.clone());
        let mut tlas = TlasManager::new(device.clone(), 16);
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 32,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // Reversed instance order and both BLAS classifications exercise the
        // generic material override and SceneDB classified geometry paths.
        for classified in [false, true] {
            for reverse in [false, true] {
                for mode in 0..5 {
                    let mut encoder = device.create_command_encoder(&Default::default());
                    for (id, opaque) in [(1, false), (2, true)] {
                        blas.build_from_buffers_with_opacity(
                            id,
                            &mut encoder,
                            BlasGeometry {
                                revision: 0,
                                vertices: &vertices,
                                first_vertex: 0,
                                vertex_count: 3,
                                vertex_stride: 12,
                                indices: None,
                                first_index: 0,
                                index_count: 0,
                            },
                            if classified { opaque } else { true },
                        )
                        .unwrap();
                    }
                    let mut sheets = vec![
                        (2.0, [0.8f32, 0.2, 0.05, 0.0]),
                        (4.0, [0.5, 0.6, 0.8, 0.0]),
                        (6.0, [0.0; 4]),
                        // Behind the endpoint: must never darken the result.
                        (8.0, [0.01, 0.01, 0.01, 0.0]),
                        (10.0, [0.0; 4]),
                    ];
                    if mode == 1 {
                        sheets[0].1 = [1.0; 4];
                        sheets[1].1 = [1.0; 4];
                    }
                    if mode == 2 {
                        sheets.insert(0, (1.0, [0.0; 4]));
                    }
                    if mode == 3 {
                        sheets.retain(|(_, t)| t[0] != 0.0);
                    }
                    if reverse {
                        sheets.reverse();
                    }
                    let instances: Vec<_> = sheets
                        .iter()
                        .map(|(x, t)| TlasInstanceInput {
                            mesh_id: if t[0] == 0.0 { 2 } else { 1 },
                            transform: [1.0, 0.0, 0.0, *x, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        })
                        .collect();
                    tlas.build(&mut encoder, &instances, &blas).unwrap();
                    // Mode 4 simulates removing transmission metadata entirely.
                    let count = if mode == 4 { 0 } else { sheets.len() as u32 };
                    let mut bytes =
                        bytemuck::cast_slice(&[classified as u32, count, 0, 0]).to_vec();
                    for (_, tint) in &sheets {
                        bytes.extend_from_slice(bytemuck::cast_slice(tint));
                    }
                    let metadata = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: &bytes,
                        usage: wgpu::BufferUsages::STORAGE,
                    });
                    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &pipeline.get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: tlas.tlas().unwrap().as_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: metadata.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: output.as_entire_binding(),
                            },
                        ],
                    });
                    {
                        let mut pass = encoder.begin_compute_pass(&Default::default());
                        pass.set_pipeline(&pipeline);
                        pass.set_bind_group(0, &bg, &[]);
                        pass.dispatch_workgroups(1, 1, 1);
                    }
                    let readback = device.create_buffer(&wgpu::BufferDescriptor {
                        label: None,
                        size: 32,
                        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                        mapped_at_creation: false,
                    });
                    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, 32);
                    queue.submit([encoder.finish()]);
                    let (tx, rx) = std::sync::mpsc::channel();
                    readback
                        .slice(..)
                        .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
                    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                    rx.recv().unwrap().unwrap();
                    let data = readback.slice(..).get_mapped_range().unwrap();
                    let result: &[f32] = bytemuck::cast_slice(&data);
                    let (distance, tint) = match mode {
                        0 => (6.0, [0.4, 0.12, 0.04]),
                        1 => (6.0, [1.0; 3]),
                        2 => (1.0, [1.0; 3]),
                        3 => (0.0, [1.0; 3]),
                        4 => (2.0, [1.0; 3]),
                        _ => unreachable!(),
                    };
                    assert!(
                        (result[0] - distance).abs() < 0.0001,
                        "endpoint: {result:?}, mode {mode}"
                    );
                    assert_eq!(result[3], if mode == 3 { 0.0 } else { 1.0 });
                    for channel in 0..3 {
                        assert!((result[4+channel]-tint[channel]).abs()<0.0001,
                            "tint: {result:?}, mode {mode}, classified {classified}, reverse {reverse}");
                    }
                }
            }
        }
    });
}
