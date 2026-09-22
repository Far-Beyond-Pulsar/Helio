use glam::{Mat4, Vec3};

#[test]
fn sky_rays_remain_finite_at_planetary_depth_ranges_and_large_local_positions() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).unwrap();
    let source = format!(
        "{}\n{}",
        include_str!("../shaders/sky.wgsl"),
        r#"
        @group(1) @binding(1) var<storage,read_write> rays:array<vec4f>;
        @compute @workgroup_size(1) fn ray_test(@builtin(global_invocation_id) id:vec3u) {
            let xy=vec2f(f32(id.x%4u)/1.5-1.0,f32(id.x/4u)/1.5-1.0);
            rays[id.x]=vec4f(sky_camera_ray(xy),1.0);
        }
    "#
    );
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some("ray_test"),
        compilation_options: Default::default(),
        cache: None,
    });
    let buffer = |size, usage| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage,
            mapped_at_creation: false,
        })
    };
    let camera_buffer = buffer(
        2 * std::mem::size_of::<helio_core::GpuCameraUniforms>() as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    let output = buffer(
        256,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let staging = buffer(
        256 * 4,
        wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
    );
    let group = |index, binding, b: &wgpu::Buffer| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(index),
            entries: &[wgpu::BindGroupEntry {
                binding,
                resource: b.as_entire_binding(),
            }],
        })
    };
    let cameras = group(0, 0, &camera_buffer);
    let outputs = group(1, 1, &output);
    let mut references = Vec::new();
    for (index, position) in [
        Vec3::ZERO,
        Vec3::splat(1024.0),
        Vec3::splat(8_000_000.0),
        Vec3::ZERO,
    ]
    .into_iter()
    .enumerate()
    {
        let view = Mat4::look_to_rh(position, Vec3::new(0.1, 0.2, -1.0), Vec3::Y);
        let proj = if index == 3 {
            Mat4::orthographic_rh(-10.0, 10.0, -8.0, 8.0, 0.1, 30_000_000.0)
        } else {
            Mat4::from_translation(Vec3::new(0.0003, -0.0007, 0.0))
                * Mat4::perspective_rh(1.0, 16.0 / 9.0, 0.1, 30_000_000.0)
        };
        let camera = helio_core::GpuCameraUniforms::new(
            view,
            proj,
            position,
            0.1,
            30_000_000.0,
            0,
            [0.0003, -0.0007],
            proj * view,
        );
        queue.write_buffer(&camera_buffer, 0, bytemuck::cast_slice(&[camera, camera]));
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &cameras, &[]);
            pass.set_bind_group(1, &outputs, &[]);
            pass.dispatch_workgroups(16, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &staging, index as u64 * 256, 256);
        queue.submit(Some(encoder.finish()));
        let inverse_proj = proj.as_dmat4().inverse();
        let inverse_rotation = glam::DMat3::from_mat4(view.as_dmat4()).transpose();
        for i in 0..16 {
            let xy = glam::DVec2::new((i % 4) as f64 / 1.5 - 1.0, (i / 4) as f64 / 1.5 - 1.0);
            let ray = if index == 3 {
                glam::DVec3::NEG_Z
            } else {
                (inverse_proj * glam::DVec4::new(xy.x, xy.y, 0.5, 1.0)).truncate()
            };
            references.push((inverse_rotation * ray).normalize());
        }
    }
    let (tx, rx) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let data = staging.slice(..).get_mapped_range().unwrap();
    let values: &[[f32; 4]] = bytemuck::cast_slice(&data);
    for (actual, reference) in values.iter().zip(references) {
        let ray = glam::DVec3::new(actual[0] as f64, actual[1] as f64, actual[2] as f64);
        assert!(
            ray.is_finite() && ray.distance(reference) < 0.000001,
            "sky ray differs from finite f64 reference: {ray} / {reference}"
        );
    }
}
