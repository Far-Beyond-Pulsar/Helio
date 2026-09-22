use wgpu::util::DeviceExt;

#[test]
fn gpu_shadow_orthographic_projection_matches_right_handed_depth() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).unwrap();
    let source = format!(
        "{}\n{}",
        include_str!("../shaders/shadow_matrices.wgsl"),
        r#"
        @group(1) @binding(0) var<storage,read_write> results:array<vec4f>;
        @compute @workgroup_size(1) fn projection_test() {
            let m=mat4_orthographic_rh(-16.0,16.0,-8.0,8.0,0.1,8000.0);
            results[0]=m*vec4f(-16.0,-8.0,-0.1,1.0);
            results[1]=m*vec4f(16.0,8.0,-8000.0,1.0);
            results[2]=m*vec4f(0.0,0.0,-4000.05,1.0);
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
        entry_point: Some("projection_test"),
        compilation_options: Default::default(),
        cache: None,
    });
    let output = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: &[0; 48],
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(1),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: output.as_entire_binding(),
        }],
    });
    let empty = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[],
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 48,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &empty, &[]);
        pass.set_bind_group(1, &group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, 48);
    queue.submit(Some(encoder.finish()));
    let (tx, rx) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let data = staging.slice(..).get_mapped_range().unwrap();
    let actual: &[f32] = bytemuck::cast_slice(&data);
    let expected = [-1.0, -1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.5, 1.0];
    for (a, b) in actual.iter().zip(expected) {
        assert!((a - b).abs() < 0.000001, "GPU shadow clip {a} != {b}");
    }
}

#[test]
fn cascades_ignore_projection_jitter_and_snap_in_world_light_coordinates() {
    use glam::{Mat4, Vec3};
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).unwrap();
    let source = include_str!("../shaders/shadow_matrices.wgsl");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some("compute_shadow_matrices"),
        compilation_options: Default::default(),
        cache: None,
    });
    let make = |bytes: &[u8], usage| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytes,
            usage,
        })
    };
    let light = helio_pass_forward_lit::GpuLight {
        direction_outer: [-0.45, -0.82, -0.35, 0.0],
        shadow_index: 0,
        light_type: 0,
        ..Default::default()
    };
    let lights = make(bytemuck::bytes_of(&light), wgpu::BufferUsages::STORAGE);
    let matrices = make(
        &[0; 384],
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let cameras = make(
        &vec![0; std::mem::size_of::<helio_core::GpuCameraUniforms>() * 2],
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    let params = make(
        bytemuck::cast_slice(&[1u32, 1024, 0, 0]),
        wgpu::BufferUsages::UNIFORM,
    );
    let dirty = make(&[0; 4], wgpu::BufferUsages::STORAGE);
    let hashes = make(&[0; 4], wgpu::BufferUsages::STORAGE);
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[&lights, &matrices, &cameras, &params, &dirty, &hashes]
            .into_iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect::<Vec<_>>(),
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 384 * 18,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    for frame in 0..18 {
        let jitter = if frame < 16 {
            [(frame % 4) as f32 / 2000.0, (frame / 4) as f32 / 2000.0]
        } else {
            [0.0; 2]
        };
        let position = if frame == 17 {
            Vec3::new(0.00001, 0.00001, 0.00001)
        } else {
            Vec3::ZERO
        };
        let view = Mat4::look_at_rh(position, position + Vec3::new(0.1, -0.2, -1.0), Vec3::Y);
        let proj = Mat4::from_translation(Vec3::new(jitter[0], jitter[1], 0.0))
            * Mat4::perspective_rh(1.0, 16.0 / 9.0, 0.01, 10000.0);
        let camera = helio_core::GpuCameraUniforms::new(
            view,
            proj,
            position,
            0.01,
            10000.0,
            frame,
            jitter,
            proj * view,
        );
        queue.write_buffer(&cameras, 0, bytemuck::cast_slice(&[camera, camera]));
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&matrices, 0, &staging, 384 * frame as u64, 384);
        queue.submit(Some(encoder.finish()));
    }
    let (tx, rx) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let data = staging.slice(..).get_mapped_range().unwrap();
    let actual: &[f32] = bytemuck::cast_slice(&data);
    for frame in 1..17 {
        for (i, (a, b)) in actual[..96]
            .iter()
            .zip(&actual[frame * 96..(frame + 1) * 96])
            .enumerate()
        {
            assert!((a-b).abs()<0.000001,"stationary cascade changed with AA jitter: frame={frame} component={i}: {a} -> {b}");
        }
    }
    for cascade in 0..4 {
        for column in 0..4 {
            for row in 0..2 {
                let i = cascade * 16 + column * 4 + row;
                assert_eq!(
                    actual[i].to_bits(),
                    actual[17 * 96 + i].to_bits(),
                    "subtexel movement shifted light XY grid"
                );
            }
        }
    }
    for cascade in 0..4 {
        let m = Mat4::from_cols_array(actual[cascade * 16..cascade * 16 + 16].try_into().unwrap());
        let p = m.project_point3(Vec3::new(0.0, 0.0, -8.0));
        assert!(
            p.z > 0.0 && p.z < 1.0,
            "visible ground must lie inside shadow depth"
        );
    }
}
