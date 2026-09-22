use glam::Vec3;

#[test]
fn planetary_sky_follows_altitude_and_rotates_with_the_observer() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).unwrap();
    let source = format!(
        "{}\n{}",
        include_str!("../../helio-pass-sky-lut/shaders/sky_lut.wgsl"),
        r#"
@group(1) @binding(1) var<storage,read> probe_rays:array<vec4<f32>>;
@group(1) @binding(2) var<storage,read_write> probe_colors:array<vec4<f32>>;
@compute @workgroup_size(4) fn probe_atmosphere(@builtin(global_invocation_id) id:vec3<u32>) {
    let rd=probe_rays[id.x].xyz;let uv=planet_lut_uv(rd);
    let recovered=planet_lut_direction(vec2<f32>(uv.x,1.0-uv.y));
    probe_colors[id.x]=vec4<f32>(atmosphere(atmosphere_observer(),rd),length(rd-recovered));
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
        entry_point: Some("probe_atmosphere"),
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
    let uniform = buffer(
        128,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );
    let rays = buffer(
        64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    let output = buffer(
        64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let staging = buffer(
        64,
        wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
    );
    let empty = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[],
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(1),
        entries: &[(0, &uniform), (1, &rays), (2, &output)].map(|(binding, b)| {
            wgpu::BindGroupEntry {
                binding,
                resource: b.as_entire_binding(),
            }
        }),
    });
    for height in [0.001f32, 20.0, 80.0, 8000.0] {
        let mut reference = Vec::new();
        for rotation in 0..4 {
            let rotate = |v: Vec3| match rotation {
                0 => v,
                1 => Vec3::new(v.y, v.z, v.x),
                2 => Vec3::new(v.z, v.x, v.y),
                _ => Vec3::new(-v.x, -v.y, v.z),
            };
            let eye = rotate(Vec3::Y * (6371.0 + height));
            let sun = rotate(Vec3::new(0.45, 0.82, 0.35).normalize());
            let mut params = [0.0f32; 32];
            params[..3].copy_from_slice(&sun.to_array());
            params[3] = 22.0;
            params[4..8].copy_from_slice(&[0.0058, 0.0135, 0.0331, 0.1]);
            params[8..12].copy_from_slice(&[0.0021, 0.075, 0.76, 0.9998]);
            params[12..15].copy_from_slice(&[6371.0, 6431.0, 1.0]);
            params[28..31].copy_from_slice(&eye.to_array());
            params[31] = 1.0;
            let tangent_sin = (6391.0 / (6371.0 + height)).min(1.0);
            let limb = Vec3::new(tangent_sin, -(1.0 - tangent_sin * tangent_sin).sqrt(), 0.0);
            let input =
                [Vec3::Y, Vec3::X, limb, -Vec3::Y].map(|v| rotate(v).extend(0.0).to_array());
            queue.write_buffer(&uniform, 0, bytemuck::cast_slice(&params));
            queue.write_buffer(&rays, 0, bytemuck::cast_slice(&input));
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &empty, &[]);
                pass.set_bind_group(1, &group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, 64);
            queue.submit(Some(encoder.finish()));
            let (tx, rx) = std::sync::mpsc::channel();
            staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            rx.recv().unwrap().unwrap();
            let mapped = staging.slice(..).get_mapped_range().unwrap();
            let colors: &[[f32; 4]] = bytemuck::cast_slice(&mapped);
            for color in colors {
                assert!(
                    color.iter().all(|c| c.is_finite() && *c >= 0.0),
                    "non-finite scattering at {height} km"
                );
            }
            for (ray, color) in colors.iter().enumerate() {
                assert!(color[3]<0.00001,"planet lookup mapping does not roundtrip at {height} km, rotation {rotation}, ray {ray}: {}",color[3]);
            }
            if height > 60.0 {
                assert_eq!(
                    &colors[0][..3],
                    &[0.0; 3],
                    "looking away from the atmosphere must see space"
                );
            } else {
                assert!(
                    colors[0][2] > 0.001,
                    "ground/high-altitude sky must retain atmospheric scattering"
                );
            }
            if height == 8000.0 {
                assert!(
                    colors[2][2] > 0.001,
                    "orbital atmospheric limb must remain visible"
                );
            }
            if rotation == 0 {
                reference = colors.to_vec();
            } else {
                for (a, b) in colors.iter().flatten().zip(reference.iter().flatten()) {
                    assert!(
                        (a - b).abs() < 0.001 + 0.01 * b.abs(),
                        "atmosphere changed with global orientation: {a} vs {b}"
                    );
                }
            }
            drop(mapped);
            staging.unmap();
        }
    }
}
