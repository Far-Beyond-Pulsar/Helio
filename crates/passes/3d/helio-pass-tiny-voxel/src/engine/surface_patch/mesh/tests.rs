use super::*;
use glam::{Mat4, Vec3};

fn material(q: [i32; 3]) -> u32 {
    if q.iter().any(|&v| !(0..32).contains(&v)) {
        return 0;
    }
    if q[2] < 8 + q[0] / 4 && !(q[0] > 11 && q[0] < 20 && q[1] > 10 && q[1] < 22) {
        1 + (q[0] / 8) as u32 % 3
    } else {
        0
    }
}
fn first(eye: [f64; 3], rd: [f32; 3]) -> Option<([i32; 3], u32)> {
    let mut events = vec![0.0];
    for a in 0..3 {
        if rd[a] != 0.0 {
            for plane in 0..=96 {
                let t = (f64::from(plane) - eye[a]) / f64::from(rd[a]);
                if t > 0.0 {
                    events.push(t);
                }
            }
        }
    }
    events.sort_by(f64::total_cmp);
    events.dedup();
    for pair in events.windows(2) {
        let t = (pair[0] + pair[1]) * 0.5;
        let cell = std::array::from_fn(|a| (eye[a] + t * f64::from(rd[a])).floor() as i32);
        let m = material(cell);
        if m != 0 {
            return Some((cell, m));
        }
    }
    None
}

#[test]
fn planetary_mesh_hits_require_current_complete_prefix_and_bounded_admission() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: adapter.features() & wgpu::Features::CONSERVATIVE_RASTERIZATION,
                ..Default::default()
            })
            .await
            .unwrap();
        if !device
            .features()
            .contains(wgpu::Features::CONSERVATIVE_RASTERIZATION)
        {
            eprintln!("SURFACE_ENGINE_MESH skipped: conservative rasterization unavailable");
            return;
        }
        let mut raster = RasterPatch::new(&device);
        let buffer = |label, bytes: &[u8], usage| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage,
            })
        };
        // Exhaustion is an atomic rejection: no partial ready page or writes.
        raster.counts.fill(CAPACITY);
        raster.upload(&queue, 0, Prepared::new(Mesh::from_samples(material)));
        assert_eq!(raster.counts, [CAPACITY; BUCKETS]);
        assert_eq!(
            (raster.accepted, raster.rejected, raster.uploaded_bytes),
            (0, 1, 0)
        );
        raster.reset(&queue);
        let mut checked = 0;
        for step in [1, 3, 10] {
            for sign in [-1, 1] {
                let low = [sign * 1_900_000, -sign * 1_900_000, sign * 1_900_000];
                let anchor = low.map(|v| v * 32 * step);
                let settings = buffer(
                    "mesh test domain",
                    bytemuck::cast_slice(&[
                        low[0] as u32,
                        low[1] as u32,
                        low[2] as u32,
                        8,
                        step as u32,
                        0,
                        0,
                        0,
                    ]),
                    wgpu::BufferUsages::UNIFORM,
                );
                let mut pages = [u32::MAX; TILES];
                let mut payload = vec![0u32; TILES * WORDS];
                // Independent dense-format payload, accepted by the existing cache shader.
                let occupied = crate::surface_cache::Brick::from_samples(material);
                for slot in [0usize, 64, 128] {
                    pages[slot] = (slot * WORDS) as u32;
                    if slot == 0 {
                        payload[..occupied.words().len()].copy_from_slice(occupied.words());
                    }
                }
                let directory = buffer(
                    "mesh test directory",
                    bytemuck::cast_slice(&pages),
                    wgpu::BufferUsages::STORAGE,
                );
                let words = buffer(
                    "mesh test words",
                    bytemuck::cast_slice(&payload),
                    wgpu::BufferUsages::STORAGE,
                );
                for stage in 0..6 {
                    raster.reset(&queue);
                    if stage != 2 {
                        let mesh = Mesh::from_samples(material);
                        let prepared = if stage == 3 {
                            Prepared::new(Mesh {
                                quads: vec![mesh.quads[0]; MAX_TILE_QUADS + 1],
                                exposed_faces: 0,
                            })
                        } else {
                            Prepared::new(mesh)
                        };
                        raster.upload(&queue, 0, prepared);
                    }
                    if stage != 1 {
                        raster.upload(&queue, 64, Prepared::new(Mesh::default()));
                    }
                    raster.upload(&queue, 128, Prepared::new(Mesh::default()));
                    // A failed admission must not issue draws for invalid arena bytes.
                    if stage == 3 {
                        assert_eq!(raster.rejected, 1);
                    }
                    let size = if stage % 2 == 0 { [64, 48] } else { [48, 32] };
                    let eye = [
                        if stage == 5 { 6.31 } else { 16.31 },
                        16.67,
                        if stage == 4 {
                            96.2
                        } else if stage == 5 {
                            4.2
                        } else {
                            80.2
                        },
                    ];
                    let mut params: Params = bytemuck::Zeroable::zeroed();
                    for a in 0..3 {
                        let relative = eye[a] * f64::from(step);
                        params.origin[a] = anchor[a] + relative.floor() as i32;
                        params.fraction[a] = relative.fract() as f32;
                    }
                    params.screen = [size[0] as f32, size[1] as f32, 0.0, 0.0];
                    params.settings = [1000.0, 0.0, 1.0, step as f32];
                    let uniform = buffer(
                        "mesh test params",
                        bytemuck::bytes_of(&params),
                        wgpu::BufferUsages::UNIFORM,
                    );
                    let proj = glam::camera::rh::proj::directx::perspective_infinite_reverse(
                        0.55,
                        size[0] as f32 / size[1] as f32,
                        0.1,
                    );
                    let mut camera = [0f32; 92];
                    camera[..16].copy_from_slice(&Mat4::IDENTITY.to_cols_array());
                    camera[16..32].copy_from_slice(&proj.to_cols_array());
                    camera[72] = 0.0003;
                    camera[73] = -0.0007;
                    let cameras = buffer(
                        "engine mesh camera",
                        bytemuck::cast_slice(&camera),
                        wgpu::BufferUsages::STORAGE,
                    );
                    let mut rays = Vec::new();
                    let mut input = Vec::new();
                    for y in 0..size[1] {
                        for x in 0..size[0] {
                            let ndc = [
                                (x as f32 + 0.5) / size[0] as f32 * 2.0 - 1.0 - camera[72],
                                1.0 - (y as f32 + 0.5) / size[1] as f32 * 2.0 - camera[73],
                            ];
                            let rd =
                                Vec3::new(ndc[0] / proj.x_axis.x, ndc[1] / proj.y_axis.y, -1.0)
                                    .normalize()
                                    .to_array();
                            rays.push(rd);
                            input.extend_from_slice(&[
                                0u32,
                                0,
                                0,
                                3,
                                rd[0].to_bits(),
                                rd[1].to_bits(),
                                rd[2].to_bits(),
                                0,
                            ]);
                        }
                    }
                    let hits = buffer(
                        "mesh prepared rays",
                        bytemuck::cast_slice(&input),
                        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    );
                    let read = device.create_buffer(&wgpu::BufferDescriptor {
                        label: None,
                        size: hits.size(),
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    let mut encoder = device.create_command_encoder(&Default::default());
                    raster.encode(
                        &device,
                        &uniform,
                        &cameras,
                        &hits,
                        &settings,
                        &directory,
                        &words,
                        &mut encoder,
                        size,
                    );
                    encoder.copy_buffer_to_buffer(&hits, 0, &read, 0, hits.size());
                    queue.submit([encoder.finish()]);
                    let (tx, rx) = mpsc::channel();
                    read.slice(..)
                        .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
                    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                    rx.recv().unwrap().unwrap();
                    let data = read.slice(..).get_mapped_range().unwrap();
                    let output: &[u32] = bytemuck::cast_slice(&data);
                    let mut accepted = 0;
                    let actual_eye = std::array::from_fn(|a| {
                        (f64::from(params.origin[a] - anchor[a]) + f64::from(params.fraction[a]))
                            / f64::from(step)
                    });
                    for (i, hit) in output.chunks_exact(8).enumerate() {
                        assert_eq!(&hit[4..7], &input[i * 8 + 4..i * 8 + 7]);
                        if hit[3] == 3 {
                            continue;
                        }
                        assert_eq!(
                            stage, 0,
                            "stage {stage} accepted unavailable/inside-solid terrain"
                        );
                        let cell =
                            std::array::from_fn(|a| (hit[a] as i32 - anchor[a]).div_euclid(step));
                        assert_eq!(
                            Some((cell, (hit[3] >> 8) & 3)),
                            first(actual_eye, rays[i]),
                            "step={step} sign={sign} pixel={i}"
                        );
                        accepted += 1;
                        checked += 1;
                    }
                    if stage == 0 {
                        assert!(accepted > 200, "raster did no useful work: {accepted}");
                    }
                    drop(data);
                    read.unmap();
                }
            }
        }
        eprintln!("SURFACE_ENGINE_MESH checked_first_hits={checked} grids=3 anchors=2 states=6");
    });
}

#[test]
fn merged_rectangle_internal_plane_tie_keeps_the_ray_for_exact_traversal() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("merged face interval-ownership regression"),
            source: wgpu::ShaderSource::Wgsl(
                (source()
                    + r#"
@compute @workgroup_size(1) fn tie_probe() {
    mesh_candidate(0u,vec4<u32>((1u<<12u)|(4u<<18u)|(1u<<21u),2u|(1u<<6u),1u,0u));
}"#)
                .into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("tie_probe"),
            compilation_options: Default::default(),
            cache: None,
        });
        let buffer = |bytes: &[u8], usage| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytes,
                usage,
            })
        };
        let mut params: Params = bytemuck::Zeroable::zeroed();
        params.origin = [2, 0, 2, 0];
        params.fraction = [0.0, 0.5, 0.0, 0.0];
        params.settings = [1000.0, 0.0, 1.0, 1.0];
        let uniform = buffer(
            bytemuck::bytes_of(&params),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let settings = buffer(
            bytemuck::cast_slice(&[0u32, 0, 0, 8, 1, 0, 0, 0]),
            wgpu::BufferUsages::UNIFORM,
        );
        let mut pages = [u32::MAX; TILES];
        pages[0] = 0;
        let directory = buffer(bytemuck::cast_slice(&pages), wgpu::BufferUsages::STORAGE);
        let mut availability = [0u32; TILES];
        availability[0] = 1;
        let ready = buffer(
            bytemuck::cast_slice(&availability),
            wgpu::BufferUsages::STORAGE,
        );
        let brick = crate::surface_cache::Brick::from_samples(|q| {
            u32::from((0..2).contains(&q[0]) && q[1] == 0 && q[2] == 0)
        });
        let words = buffer(
            bytemuck::cast_slice(brick.words()),
            wgpu::BufferUsages::STORAGE,
        );
        let d = -std::f32::consts::FRAC_1_SQRT_2;
        let input = [0u32, 0, 0, 3, d.to_bits(), 0, d.to_bits(), 0];
        let hits = buffer(
            bytemuck::cast_slice(&input),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );
        let read = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 32,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let inputs = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                (0, &uniform),
                (9, &hits),
                (31, &settings),
                (32, &directory),
                (33, &words),
                (35, &ready),
            ]
            .map(|(binding, b)| wgpu::BindGroupEntry {
                binding,
                resource: b.as_entire_binding(),
            }),
        });
        for offset in [0.0, 0.0002, 0.01] {
            params.fraction[0] = offset;
            queue.write_buffer(&uniform, 0, bytemuck::bytes_of(&params));
            queue.write_buffer(&hits, 0, bytemuck::cast_slice(&input));
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &inputs, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&hits, 0, &read, 0, 32);
            queue.submit([encoder.finish()]);
            let (tx, rx) = mpsc::channel();
            read.slice(..)
                .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            rx.recv().unwrap().unwrap();
            let data = read.slice(..).get_mapped_range().unwrap();
            let output: &[u32] = bytemuck::cast_slice(&data);
            if offset < 0.001 {
                assert_eq!(output, &input, "ambiguous interval must remain prepared");
            } else {
                assert_eq!(&output[..3], &[1, 0, 0]);
                assert_eq!(output[3] & 3, 1);
            }
            drop(data);
            read.unmap();
        }
    });
}
