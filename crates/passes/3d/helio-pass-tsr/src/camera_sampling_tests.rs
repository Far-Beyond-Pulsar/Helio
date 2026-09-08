use super::*;
use std::sync::Arc;

#[test]
fn prepare_uploads_the_actual_camera_sample_and_supplied_frame_time() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).unwrap();
    let device = Arc::new(device);
    let queue = Arc::new(queue);
    let mut scene = helio_core::GpuScene::new(device.clone(), queue.clone());
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
@group(0) @binding(0) var<uniform> input_data: array<vec4<u32>, 2>;
@group(0) @binding(1) var<storage, read_write> output_data: array<vec4<u32>, 2>;
@compute @workgroup_size(1) fn read_uniform() {
    output_data[0] = input_data[0]; output_data[1] = input_data[1];
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
        size: 32,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 32,
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
    for (index, (size, expected, time, frame)) in [
        ([32, 16], [0.0_f32, 0.0_f32], 1.0_f32 / 60.0, 17),
        ([32, 16], [0.375, -0.25], 1.0 / 30.0, 991),
        ([19, 27], [-0.125, 0.375], 1.0 / 120.0, 0),
    ]
    .into_iter()
    .enumerate()
    {
        let mut camera = libhelio::GpuCameraUniforms::zeroed();
        camera.jitter_frame = [
            expected[0] * 2.0 / size[0] as f32,
            expected[1] * 2.0 / size[1] as f32,
            123.0,
            0.0,
        ];
        scene.camera.update(camera);
        pass.prepare(&PrepareContext {
            device: &device,
            queue: &queue,
            scene: &scene,
            frame_resources: &libhelio::FrameResources::empty(),
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
        encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, 32);
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
        drop(data);
        staging.unmap();
    }
}
