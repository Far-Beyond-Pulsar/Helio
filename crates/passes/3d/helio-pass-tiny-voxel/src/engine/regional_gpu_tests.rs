use super::*;

#[test]
fn clipped_ancestor_payload_preserves_sampling_and_region_ownership() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&Default::default())
            .await
            .expect("GPU required");
        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffers_per_shader_stage = 8;
        limits.max_color_attachments = 8;
        limits.max_color_attachment_bytes_per_sample = 64;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: limits,
                ..Default::default()
            })
            .await
            .unwrap();
        let terrain = StoredTerrain::new(&device, &queue, 64, 1, DepthConvention::Forward);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("clipped ancestor query regression"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{SHADER}\n{}\n{}",
                    include_str!("stored.wgsl"),
                    r#"
@compute @workgroup_size(1)
fn regional_query(@builtin(global_invocation_id) id:vec3<u32>) {
    var rd=vec3<f32>(f32(id.x%3u)-1.0,f32((id.x/3u)%3u)-1.0,f32((id.x/9u)%3u)-1.0);
    if all(rd==vec3<f32>(0.0)) {rd.y=-1.0;}
    if id.x>=27u {rd.x*=0.00001;}
    var ro=vec3<f32>(f32(id.x%5u)*0.013,0.0,f32(id.x%7u)*0.017);
    if p.lighting.w>0.0 {
        let horizontal=select(-16.5,16.5,(id.x&1u)==0u);
        ro=(vec3<f32>(horizontal,40.5,horizontal)-vec3<f32>(p.origin.xyz)-p.fraction.xyz)*0.1;
        rd=vec3<f32>(0.0,-1.0,0.0);
    }
    primary_hits[id.x]=stored_trace(ro,normalize(rd),30000000.0);
}
"#
                )
                .into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("regional_query"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[("STORED_REGIONAL", 1.0)],
                ..Default::default()
            },
            cache: None,
        });
        let group = terrain.group(
            &pipeline.get_bind_group_layout(0),
            &[
                (0, &terrain.uniform),
                (9, &terrain.hits),
                (24, &terrain.nodes),
                (25, &terrain.materials),
                (28, &terrain.exact_occupied),
            ],
        );
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 64 * 32,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let capture = || {
            let mut encoder = device.create_command_encoder(&Default::default());
            terrain.compute(&mut encoder, &pipeline, &[group.clone()], [54, 1, 1]);
            encoder.copy_buffer_to_buffer(&terrain.hits, 0, &readback, 0, 54 * 32);
            queue.submit([encoder.finish()]);
            let (tx, rx) = std::sync::mpsc::channel();
            readback
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            rx.recv().unwrap().unwrap();
            let bytes = readback.slice(..).get_mapped_range().unwrap();
            let values = bytemuck::cast_slice::<u8, u32>(&bytes)[..54 * 8].to_vec();
            drop(bytes);
            readback.unmap();
            values
        };
        for level in [1u32, 4, 21] {
            let low = [-(16i32 << level); 3];
            let root = residency::Node {
                low,
                level,
                child: 0x80000000,
            };
            let mut payload = [1.0f32.to_bits(); 2048];
            for i in 0..729 {
                let value = if level == 21 {
                    -1.0
                } else {
                    4.25 - ((i / 9) % 9) as f32
                };
                payload[i] = value.to_bits() | 1;
            }
            if level == 21 {
                payload[729..1241].fill((-1.0f32).to_bits());
            }
            queue.write_buffer(&terrain.materials, 0, bytemuck::cast_slice(&payload));
            let mut params: Params = bytemuck::Zeroable::zeroed();
            params.origin[1] = 8 << level;
            params.fraction = [0.37, 0.51, 0.73, 0.0];
            params.settings[3] = 1.0;
            queue.write_buffer(&terrain.uniform, 0, bytemuck::bytes_of(&params));
            terrain.upload_nodes(&[root]);
            let original = capture();
            let mut clipped = vec![residency::Node { child: 1, ..root }];
            for octant in 0..8 {
                clipped.push(residency::Node {
                    low: std::array::from_fn(|a| low[a] + ((octant >> a) & 1) * (16 << level)),
                    level: level - 1,
                    child: 0x80010000,
                });
            }
            terrain.upload_nodes(&clipped);
            let divided = capture();
            for (ray, (a, b)) in original
                .chunks_exact(8)
                .zip(divided.chunks_exact(8))
                .enumerate()
            {
                assert!(a[3] & 3 < 2 && b[3] & 3 < 2, "exhausted at {level}/{ray}");
                assert_eq!(a[3] & 3, b[3] & 3, "coverage at {level}/{ray}");
                if a[3] & 3 == 1 {
                    assert_eq!(&a[..3], &b[..3], "cell at {level}/{ray}");
                    let before = f32::from_bits(a[7]);
                    let after = f32::from_bits(b[7]);
                    assert!(
                        (before - after).abs() <= 0.00002 * before.abs().max(1.0),
                        "depth at {level}/{ray}: {before} -> {after}"
                    );
                }
            }
            if level == 1 {
                // Publish an air child while its neighbours still use the
                // parent field. A ray through that child must not return the
                // old parent surface inside the replaced region.
                clipped[8].child = residency::AIR;
                terrain.upload_nodes(&clipped);
                params.origin = [0; 4];
                params.fraction = [0.5; 4];
                params.lighting[3] = 1.0;
                queue.write_buffer(&terrain.uniform, 0, bytemuck::bytes_of(&params));
                for (ray, hit) in capture().chunks_exact(8).enumerate() {
                    let positive = ray % 2 == 0;
                    let horizontal = if positive { 16 } else { -17 };
                    let y = if positive { -1 } else { 2 };
                    assert_eq!(hit[3] & 3, 1);
                    assert_eq!(
                        [hit[0] as i32, hit[1] as i32, hit[2] as i32],
                        [horizontal, y, horizontal]
                    );
                    let expected = if positive { 4.05 } else { 3.75 };
                    assert!((f32::from_bits(hit[7]) - expected).abs() < 0.00002);
                }
            }
        }
    });
}
