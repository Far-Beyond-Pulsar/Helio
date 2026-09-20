use super::*;
use std::sync::Arc;

#[test]
fn prepare_uploads_the_actual_camera_sample_and_supplied_frame_time() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).unwrap();
    let device = Arc::new(device);
    let queue = Arc::new(queue);
    let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: std::mem::size_of::<helio_core::GpuCameraUniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let scene_buffers = helio_core::SceneBufferProjection::empty();
    let mut pass = TsrPass::new(
        &device,
        32,
        16,
        64,
        32,
        wgpu::TextureFormat::Rgba8Unorm,
        TsrQuality::Quality,
    );
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("read actual TSR uniform upload"),
        source: wgpu::ShaderSource::Wgsl(
            r#"
@group(0) @binding(0) var<uniform> input_data: array<vec4<u32>, 6>;
@group(0) @binding(1) var<storage, read_write> output_data: array<vec4<u32>, 6>;
@compute @workgroup_size(1) fn read_uniform() {
    for(var i=0u;i<6u;i++) { output_data[i] = input_data[i]; }
}
"#
            .into(),
        ),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some("read_uniform"),
        compilation_options: Default::default(),
        cache: None,
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 96,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 96,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: pass.uniform_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
        ],
    });
    let mut previous_uv = [0.0f32; 2];
    let mut previous_view = [0.0f32; 16];
    for (index, (size, expected, time, frame)) in [
        ([32, 16], [0.0_f32, 0.0_f32], 1.0_f32 / 60.0, 17),
        ([32, 16], [0.375, -0.25], 1.0 / 30.0, 991),
        ([19, 27], [-0.125, 0.375], 1.0 / 120.0, 0),
    ]
    .into_iter()
    .enumerate()
    {
        let mut camera = helio_core::GpuCameraUniforms::zeroed();
        for i in [0, 5, 10, 15] {
            camera.view[i] = 1.0;
        }
        camera.view[14] = -(index as f32) * 2.0;
        camera.jitter_frame = [
            expected[0] * 2.0 / size[0] as f32,
            expected[1] * 2.0 / size[1] as f32,
            123.0,
            0.0,
        ];
        queue.write_buffer(&camera_buffer, 0, bytemuck::bytes_of(&camera));
        pass.prepare(&PrepareContext {
            device: &device,
            queue: &queue,
            camera: &camera_buffer,
            camera_data: &camera,
            camera_generation: 0,
            scene_buffers: &scene_buffers,
            registry: &helio_core::ResourceRegistry::empty(),
            pass_resources: &helio_core::ResourceRegistry::empty(),
            width: size[0],
            height: size[1],
            frame_num: frame,
            resize: false,
            delta_time: time,
        })
        .unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut compute = encoder.begin_compute_pass(&Default::default());
            compute.set_pipeline(&pipeline);
            compute.set_bind_group(0, &group, &[]);
            compute.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, 96);
        queue.submit([encoder.finish()]);
        let (send, receive) = std::sync::mpsc::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                send.send(result).unwrap();
            });
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        receive.recv().unwrap().unwrap();
        let data = staging.slice(..).get_mapped_range().unwrap();
        let words: &[u32] = bytemuck::cast_slice(&data);
        assert!((f32::from_bits(words[0]) - expected[0]).abs() < 1e-6);
        assert!((f32::from_bits(words[1]) - expected[1]).abs() < 1e-6);
        assert_eq!(words[3], u32::from(index == 0));
        assert_eq!(words[4], time.to_bits());
        assert_eq!(words[6], previous_uv[0].to_bits());
        assert_eq!(words[7], previous_uv[1].to_bits());
        assert_eq!(&words[8..24], &previous_view.map(f32::to_bits));
        previous_view = camera.view;
        previous_uv = [camera.jitter_frame[0] * 0.5, -camera.jitter_frame[1] * 0.5];
        drop(data);
        staging.unmap();
    }
}

#[test]
fn resolved_view_publication_follows_resize() {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
    let (device, _) = pollster::block_on(adapter.request_device(&Default::default())).unwrap();
    let mut pass = TsrPass::new(
        &device,
        32,
        16,
        64,
        32,
        wgpu::TextureFormat::Rgba8Unorm,
        TsrQuality::Quality,
    )
    .with_intermediate_output();
    let old = pass.output_view.clone();
    for size in [None, Some((96, 48))] {
        if let Some((width, height)) = size {
            pass.on_resize(&device, width, height);
        }
        let mut frame = helio_core::ResourceRegistry::empty();
        pass.publish(&mut frame);
        let published: &wgpu::TextureView = frame
            .get(helio_core::ResourceKey::new("tsr_color"))
            .unwrap();
        assert_eq!(published, &pass.output_view);
        assert_eq!(pass.history_depth.size(), pass.output_texture.size());
        assert_eq!(pass.output_depth.size(), pass.output_texture.size());
        if size.is_some() {
            assert_ne!(published, &old);
        }
    }
}

#[test]
fn reprojection_removes_both_jitters_without_removing_camera_motion() {
    use wgpu::util::DeviceExt;
    pollster::block_on(async {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();
        let source = include_str!("../shaders/tsr_main.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(format!("{source}\n\
                @group(0) @binding(9) var<storage,read_write> answer: vec4<f32>;\n\
                @compute @workgroup_size(1) fn probe() {{\n\
                  let raster_uv=vec2<f32>(0.4,0.6)+cameras[0].jitter_frame.xy*vec2<f32>(0.5,-0.5);\n\
                  answer=reproject_history(raster_uv,0.5);\n\
                }}").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("probe"),
            compilation_options: Default::default(),
            cache: None,
        });
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        for size in [[33.0, 17.0], [2560.0, 1440.0]] {
            for phase in 0..4 {
                let current = [
                    (phase as f32 * 0.21 - 0.37) * 2.0 / size[0],
                    (0.31 - phase as f32 * 0.17) * 2.0 / size[1],
                ];
                let previous = [
                    (0.27 - phase as f32 * 0.13) * 2.0 / size[0],
                    (-0.41 + phase as f32 * 0.19) * 2.0 / size[1],
                ];
                let mut camera = helio_core::GpuCameraUniforms::zeroed();
                let mut identity = [0.0; 16];
                for i in [0, 5, 10, 15] {
                    identity[i] = 1.0;
                }
                camera.inv_view_proj = identity;
                camera.prev_view_proj = identity;
                camera.inv_view_proj[12] = -current[0];
                camera.inv_view_proj[13] = -current[1];
                camera.prev_view_proj[12] = previous[0] - 0.12;
                camera.prev_view_proj[13] = previous[1] + 0.08;
                camera.jitter_frame = [current[0], current[1], 0.0, 0.0];
                let cameras = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&[camera, camera]),
                    usage: wgpu::BufferUsages::STORAGE,
                });
                let mut uniform = TsrUniform::zeroed();
                uniform.previous_view = identity;
                uniform.previous_view[14] = -2.0;
                uniform.previous_jitter_uv = [previous[0] * 0.5, -previous[1] * 0.5];
                let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::bytes_of(&uniform),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: cameras.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: params.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 9,
                            resource: output.as_entire_binding(),
                        },
                    ],
                });
                let mut encoder = device.create_command_encoder(&Default::default());
                {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &bind, &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                }
                encoder.copy_buffer_to_buffer(&output, 0, &read, 0, 16);
                queue.submit([encoder.finish()]);
                let (tx, rx) = std::sync::mpsc::channel();
                read.slice(..)
                    .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                rx.recv().unwrap().unwrap();
                let mapped = read.slice(..).get_mapped_range().unwrap();
                let values = bytemuck::cast_slice::<u8, f32>(&mapped);
                assert!(
                    (values[0] - 0.34).abs() < 0.000001 && (values[1] - 0.56).abs() < 0.000001,
                    "{size:?}/{phase}: {values:?}"
                );
                assert!(
                    (values[3] - 1.5).abs() < 0.000001,
                    "previous view depth: {values:?}"
                );
                drop(mapped);
                read.unmap();
            }
        }
    });
}

#[test]
fn depth_history_rejects_disocclusion_and_invalid_samples() {
    pollster::block_on(async {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();
        let source = include_str!("../shaders/tsr_main.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{source}\n\
            @group(0) @binding(9) var<storage,read_write> verdicts: array<u32,6>;\n\
            @compute @workgroup_size(1) fn test_depth() {{\n\
                verdicts[0]=u32(history_depth_matches(10.0,10.0,0.02));\n\
                verdicts[1]=u32(history_depth_matches(10.0,10.009,0.02));\n\
                verdicts[2]=u32(history_depth_matches(10.0,10.1,0.02));\n\
                verdicts[3]=u32(history_depth_matches(10.0,2.0,0.02));\n\
                verdicts[4]=u32(history_depth_matches(10.0,0.0,0.02));\n\
                verdicts[5]=u32(history_depth_matches(-10.0,10.0,0.02));\n\
            }}"
                )
                .into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("test_depth"),
            compilation_options: Default::default(),
            cache: None,
        });
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 24,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 24,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 9,
                resource: output.as_entire_binding(),
            }],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &read, 0, 24);
        queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        read.slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let data = read.slice(..).get_mapped_range().unwrap();
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&data), [1, 1, 0, 0, 0, 0]);
    });
}
