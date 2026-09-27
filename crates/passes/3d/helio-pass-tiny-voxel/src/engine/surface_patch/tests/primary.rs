use super::*;

fn material(q: [i32; 3]) -> u32 {
    if q.iter().any(|v| !(0..32).contains(v)) {
        return 0;
    }
    if q[2] < 8 || (q[0] >= 21 && q[1] < 15) || (q[1] >= 24 && q[2] < 20) {
        1 + (q[0] + q[1] + q[2]) as u32 % 3
    } else {
        0
    }
}

// Independently enumerate every plane and sample its open intervals. Integer
// anchors stay separate from the fractional camera, including on the CPU.
fn oracle(
    origin: [i32; 3],
    fraction: [f32; 3],
    ray: [f32; 3],
    step: i32,
) -> Option<([i32; 3], u32, u32, f64)> {
    let mut events = vec![(0.0, 0u32)];
    for a in 0..3 {
        if ray[a] == 0.0 {
            continue;
        }
        for plane in 0..=64 {
            let t =
                (f64::from(plane * step - origin[a]) - f64::from(fraction[a])) / f64::from(ray[a]);
            if t >= 0.0 {
                events.push((t, 1 + a as u32 * 2 + u32::from(ray[a] > 0.0)));
            }
        }
    }
    events.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    events.dedup_by(|a, b| a.0 == b.0);
    for pair in events.windows(2) {
        let t = pair[0].0 + (pair[1].0 - pair[0].0) * 0.5;
        let q = std::array::from_fn(|a| {
            (origin[a] + (f64::from(fraction[a]) + f64::from(ray[a]) * t).floor() as i32)
                .div_euclid(step)
        });
        let m = material(q);
        if m != 0 {
            return Some((q, m, pair[0].1, pair[0].0 * 0.1));
        }
    }
    None
}

#[test]
fn prepared_primary_rays_preserve_near_coincident_planes_at_planetary_anchors() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&Default::default())
            .await
            .expect("GPU required");
        let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();
        {
            let source = shader_source()
                + r#"
@compute @workgroup_size(1)
fn predicate_probe() {
    let rd=abs(primary_hits[0].normal);
    let a=patch_delta(16,0,0.0,1);let b=patch_delta(20,0,0.0,1);
    primary_hits[0]=Hit(vec3<i32>(patch_compare(a,rd.x,b,rd.z),0,0),0u,rd,0.0);
}"#;
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &shader,
                entry_point: Some("predicate_probe"),
                compilation_options: Default::default(),
                cache: None,
            });
            let words = [
                0u32,
                0,
                0,
                3,
                (-0.54815096f32).to_bits(),
                (-0.47963208f32).to_bits(),
                (-0.6851887f32).to_bits(),
                0,
            ];
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&words),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
            let read = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 32,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[wgpu::BindGroupEntry {
                    binding: 9,
                    resource: buffer.as_entire_binding(),
                }],
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&buffer, 0, &read, 0, 32);
            queue.submit([encoder.finish()]);
            let (tx, rx) = mpsc::channel();
            read.slice(..)
                .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            rx.recv().unwrap().unwrap();
            let mapped = read.slice(..).get_mapped_range().unwrap();
            let result: &[u32] = bytemuck::cast_slice(&mapped);
            // 16/z versus 20/x differ by 2^-22 before division. Both rounded
            // f32 products are equal; the exact predicate must retain order.
            assert_eq!(result[0] as i32, 1);
        }
        let mut patch = Patch::with_mesh(&device, false);
        patch.stats.enabled = true;
        let brick = Brick::from_samples(material);
        queue.write_buffer(&patch.words, 0, bytemuck::cast_slice(brick.words()));
        queue.write_buffer(&patch.directory, 0, bytemuck::bytes_of(&0u32));
        // The next Z page is certified entirely empty. Start in either page
        // so the same independent oracle covers both 4-cell and 32-cell skips.
        let empty_base = brick.words().len() as u32;
        queue.write_buffer(
            &patch.words,
            u64::from(empty_base) * 4,
            bytemuck::bytes_of(&0u32),
        );
        queue.write_buffer(
            &patch.directory,
            (SIDE * SIDE * 4) as u64,
            bytemuck::bytes_of(&empty_base),
        );
        let mut checked = 0;
        let mut hits_checked = 0;
        for (disable_skip, step, start_z) in [0u32, 1].into_iter().flat_map(|skip| {
            [1i32, 3, 10]
                .into_iter()
                .flat_map(move |step| [28, 60].map(|z| (skip, step, z)))
        }) {
            for key in [Key([-3, 1_991_171, -7]), Key([3, -1_991_171, 7])] {
                let low = key.low(step as u32);
                queue.write_buffer(
                    &patch.settings,
                    0,
                    bytemuck::cast_slice(&[
                        key.0[0] as u32,
                        key.0[1] as u32,
                        key.0[2] as u32,
                        8u32,
                        step as u32,
                        0,
                        disable_skip,
                        0,
                    ]),
                );
                for fraction in [[0.0; 3], [0.25, 0.999990, 0.0], [0.5, 0.125, 0.99999994]] {
                    let origin = [16 * step, 14 * step, start_z * step];
                    let mut rays = Vec::<[f32; 3]>::new();
                    for y in (0..=32).step_by(4) {
                        for x in (0..=32).step_by(4) {
                            let target = [x * step, y * step, 8 * step];
                            let ray = glam::Vec3::from_array(std::array::from_fn(|a| {
                                (f64::from(target[a] - origin[a]) - f64::from(fraction[a])) as f32
                            }))
                            .normalize()
                            .to_array();
                            for axis in 0..3 {
                                for perturb in -1..=1 {
                                    let mut r = ray;
                                    r[axis] = match perturb {
                                        -1 => r[axis].next_down(),
                                        1 => r[axis].next_up(),
                                        _ => r[axis],
                                    };
                                    rays.push(r);
                                }
                            }
                        }
                    }
                    let mut params: Params = bytemuck::Zeroable::zeroed();
                    params.origin = [
                        low[0] + origin[0],
                        low[1] + origin[1],
                        low[2] + origin[2],
                        0,
                    ];
                    params.fraction = [fraction[0], fraction[1], fraction[2], 0.0];
                    params.screen = [rays.len() as f32, 1.0, 0.0, 0.0];
                    params.settings = [1000.0, 0.0, 1.0, step as f32];
                    let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: bytemuck::bytes_of(&params),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
                    let prepared: Vec<[u32; 8]> = rays
                        .iter()
                        .map(|r| {
                            [
                                0,
                                0,
                                0,
                                3,
                                r[0].to_bits(),
                                r[1].to_bits(),
                                r[2].to_bits(),
                                0,
                            ]
                        })
                        .collect();
                    let hits = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: bytemuck::cast_slice(&prepared),
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    });
                    let read = device.create_buffer(&wgpu::BufferDescriptor {
                        label: None,
                        size: hits.size(),
                        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                        mapped_at_creation: false,
                    });
                    let mut encoder = device.create_command_encoder(&Default::default());
                    patch.encode_primary(
                        &device,
                        &uniform,
                        &uniform, // No raster path in this arbitrary-ray audit.
                        &hits,
                        &mut encoder,
                        [rays.len() as u32, 1],
                        None,
                    );
                    encoder.copy_buffer_to_buffer(&hits, 0, &read, 0, hits.size());
                    queue.submit([encoder.finish()]);
                    let (tx, rx) = mpsc::channel();
                    read.slice(..)
                        .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
                    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                    rx.recv().unwrap().unwrap();
                    let bytes = read.slice(..).get_mapped_range().unwrap();
                    let output: &[[u32; 8]] = bytemuck::cast_slice(&bytes);
                    for (i, (r, gpu)) in rays.iter().zip(output).enumerate() {
                        assert_eq!(&gpu[4..7], &prepared[i][4..7]);
                        let expected = oracle(origin, fraction, *r, step);
                        if let Some((q, m, face, t)) = expected {
                            let cell: [u32; 3] = std::array::from_fn(|a| {
                                (low[a] + q[a] * step + (step - 1) / 2) as u32
                            });
                            assert_eq!(
                                gpu[3] & 3,
                                1,
                                "step={step} key={key:?} fraction={fraction:?} ray={r:?}"
                            );
                            assert_eq!(
                                &gpu[..3],
                                &cell,
                                "step={step} fraction={fraction:?} ray={r:?}"
                            );
                            assert_eq!((gpu[3] >> 8) & 3, m);
                            assert_eq!(
                                (gpu[3] >> 28) & 7,
                                face,
                                "step={step} fraction={fraction:?} ray={r:?}"
                            );
                            assert!(
                                (f64::from(f32::from_bits(gpu[7])) - t).abs()
                                    <= 4.0 * f64::from(f32::EPSILON) * t.max(1.0)
                            );
                            hits_checked += 1;
                        } else {
                            assert_eq!(
                                gpu, &prepared[i],
                                "missing prefix/patch exit must retain prepared ray"
                            );
                        }
                        checked += 1;
                    }
                    drop(bytes);
                    read.unmap();
                }
            }
        }
        eprintln!("SURFACE_PATCH_PRIMARY rays={checked} occupied={hits_checked}");
    });
}
