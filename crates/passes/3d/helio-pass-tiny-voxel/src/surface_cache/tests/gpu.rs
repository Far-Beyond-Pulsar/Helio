use super::*;
use wgpu::util::DeviceExt;

#[derive(Clone, Copy, Debug)]
struct Ray {
    origin: [f32; 3],
    direction: [f32; 3],
}

/// Independent reference: sort all grid-plane events, then sample the open
/// intervals between them. No DDA stepping, packed lookup, or skipping is used.
fn oracle(ray: Ray, dense: &[u32]) -> Option<([i32; 3], u32, f64, Option<u32>)> {
    let ro = ray.origin.map(f64::from);
    let rd = ray.direction.map(f64::from);
    if rd == [0.0; 3] {
        return None;
    }
    let mut events = vec![(0.0, None)];
    for a in 0..3 {
        if rd[a] == 0.0 {
            continue;
        }
        for plane in 0..=32 {
            let t = (f64::from(plane) - ro[a]) / rd[a];
            if t >= 0.0 {
                events.push((t, Some(1 + a as u32 * 2 + u32::from(rd[a] > 0.0))));
            }
        }
    }
    events.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    events.dedup_by(|a, b| a.0 == b.0);
    for pair in events.windows(2) {
        let t = pair[0].0 + (pair[1].0 - pair[0].0) * 0.5;
        let q: [i32; 3] = std::array::from_fn(|a| (ro[a] + rd[a] * t).floor() as i32);
        if q.iter().any(|v| !(0..32).contains(v)) {
            continue;
        }
        let material = dense[(q[0] + q[1] * 32 + q[2] * 1024) as usize];
        if material != 0 {
            // An origin already in a solid voxel has no entry surface normal.
            let face = if pair[0].0 == 0.0 { None } else { pair[0].1 };
            return Some((q, material, pair[0].0, face));
        }
    }
    None
}

fn rays() -> Vec<Ray> {
    let mut out = Vec::new();
    // Axis-parallel, negative, boundary, grazing, edge/corner ties, and inside
    // starts. Inputs are f32; the reference uses their exact f64 conversion.
    for axis in 0..3 {
        for sign in [-1.0, 1.0] {
            for p in [-0.125, 0.0, 0.5, 3.0, 4.0, 15.5, 16.0, 31.5, 32.0] {
                for shift in [0.0, 0.125] {
                    let mut origin = [p; 3];
                    origin[axis] = if sign > 0.0 { -2.0 } else { 34.0 };
                    origin[(axis + 1) % 3] += shift;
                    let mut direction = [0.0; 3];
                    direction[axis] = sign;
                    out.push(Ray { origin, direction });
                }
            }
        }
    }
    for x in -1..=1 {
        for y in -1..=1 {
            for z in -1..=1 {
                let direction = [x as f32, y as f32, z as f32];
                for p in [0.0, 4.0, 16.0, 32.0] {
                    out.push(Ray {
                        origin: [p; 3],
                        direction,
                    });
                    out.push(Ray {
                        origin: direction.map(|v| 16.0 - v * 20.0),
                        direction,
                    });
                }
            }
        }
    }
    let mut seed = 17u32;
    let mut random = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed >> 8) as f32 / 16_777_216.0
    };
    for i in 0..2048 {
        let origin = std::array::from_fn(|_| random() * 48.0 - 8.0);
        let target: [f32; 3] = std::array::from_fn(|_| random() * 32.0);
        let mut direction = std::array::from_fn(|a| target[a] - origin[a]);
        if i % 4 == 0 {
            direction[i % 3] = 0.0;
        }
        if i % 4 == 1 {
            direction[i % 3] *= 0.00001;
        }
        out.push(Ray { origin, direction });
    }
    out
}

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
}
impl Gpu {
    fn new() -> Self {
        pollster::block_on(async {
            let instance =
                wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
            let adapter = instance
                .request_adapter(&Default::default())
                .await
                .expect("GPU required");
            eprintln!("SURFACE_CACHE_GPU adapter={:?}", adapter.get_info());
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    required_features: adapter.features() & wgpu::Features::TIMESTAMP_QUERY,
                    ..Default::default()
                })
                .await
                .unwrap();
            Self { device, queue }
        })
    }

    fn run(
        &self,
        words: &[u32],
        inputs: &[u32],
        source: &str,
        count: usize,
        stride: usize,
    ) -> Vec<u32> {
        self.dispatch(words, inputs, source, count, stride, None)
    }

    fn dispatch(
        &self,
        words: &[u32],
        inputs: &[u32],
        source: &str,
        count: usize,
        stride: usize,
        timing: Option<&str>,
    ) -> Vec<u32> {
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("experimental exact surface cache"),
                source: wgpu::ShaderSource::Wgsl(format!("{}\n{}", GPU_SHADER, source).into()),
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let buffer = |data: &[u32]| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(data),
                    usage: wgpu::BufferUsages::STORAGE,
                })
        };
        let words = buffer(words);
        let input = buffer(inputs);
        let size = (count * stride * 4) as u64;
        let repeats = if timing.is_some() { 36 } else { 1 };
        let queries = timing.map(|_| {
            assert!(self
                .device
                .features()
                .contains(wgpu::Features::TIMESTAMP_QUERY));
            self.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: None,
                ty: wgpu::QueryType::Timestamp,
                count: repeats * 2,
            })
        });
        let timing_size = if queries.is_some() {
            u64::from(repeats) * 16
        } else {
            0
        };
        let resolve = queries.as_ref().map(|_| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: timing_size,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        });
        let output = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size + timing_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: words.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        for repeat in 0..repeats {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: queries.as_ref().map(|q| wgpu::ComputePassTimestampWrites {
                    query_set: q,
                    beginning_of_pass_write_index: Some(repeat * 2),
                    end_of_pass_write_index: Some(repeat * 2 + 1),
                }),
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups((count as u32).div_ceil(64), 1, 1);
        }
        if let Some(q) = &queries {
            encoder.resolve_query_set(q, 0..repeats * 2, resolve.as_ref().unwrap(), 0);
            encoder.copy_buffer_to_buffer(
                resolve.as_ref().unwrap(),
                0,
                &readback,
                size,
                timing_size,
            );
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, size);
        self.queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        rx.recv().unwrap().unwrap();
        let bytes = readback.slice(..).get_mapped_range().unwrap();
        let result = bytemuck::cast_slice::<u8, u32>(&bytes[..size as usize]).to_vec();
        if let Some(label) = timing {
            let times: Vec<_> = bytes[size as usize..]
                .chunks_exact(16)
                .skip(4)
                .map(|pair| {
                    let a = u64::from_le_bytes(pair[..8].try_into().unwrap());
                    let b = u64::from_le_bytes(pair[8..].try_into().unwrap());
                    assert!(b >= a);
                    (b - a) as f64 * f64::from(self.queue.get_timestamp_period()) / 1e6
                })
                .collect();
            eprintln!("SURFACE_CACHE_GPU_TIMING label={label} rays={count} milliseconds={times:?}");
        }
        drop(bytes);
        readback.unmap();
        result
    }
}

fn fixtures() -> Vec<(String, Brick, Vec<u32>)> {
    let samples: [(&str, Box<dyn Fn([i32; 3]) -> u32>); 6] = [
        ("air", Box::new(|_| 0)),
        ("solid", Box::new(|_| 3)),
        ("plane", Box::new(|q| u32::from(q[1] < 17) * 2)),
        (
            "checker",
            Box::new(|q| (q[0] + q[1] + q[2]).rem_euclid(4) as u32),
        ),
        (
            "shell",
            Box::new(|q| {
                let r = q.map(|v| v - 16).iter().map(|v| v * v).sum::<i32>();
                u32::from((100..144).contains(&r))
            }),
        ),
        (
            "walls",
            Box::new(|q| {
                if q[2] == 25 {
                    1
                } else if q[2] == 7 {
                    3
                } else {
                    0
                }
            }),
        ),
    ];
    let mut fixtures: Vec<_> = samples
        .into_iter()
        .map(|(name, sample)| {
            let brick = Brick::from_samples(&sample);
            let dense = (0..32768)
                .map(|i| sample([i % 32, i / 32 % 32, i / 1024]))
                .collect();
            (name.to_string(), brick, dense)
        })
        .collect();
    for step in [1, 3, 10] {
        let mut world = World::default();
        world.set_voxel_size(f64::from(step) * 0.1).unwrap();
        let ground = world.ground_spawn(-0.3, -0.3, 0.0);
        let cell = crate::world::cell_of(ground);
        world
            .apply_edit(Edit {
                cell,
                radius: 0.61,
                material: 3,
            })
            .unwrap();
        world
            .apply_edit(Edit {
                cell: [cell[0] + 1, cell[1], cell[2]],
                radius: 0.31,
                material: 0,
            })
            .unwrap();
        let key = Key::containing(cell, step);
        let low = key.low(step);
        let brick = Brick::from_world(&world, key);
        let dense = (0..32768)
            .map(|i| {
                let q = [i % 32, i / 32 % 32, i / 1024];
                world.material(std::array::from_fn(|a| low[a] + q[a] * step as i32))
            })
            .collect();
        fixtures.push((format!("planet-{step}"), brick, dense));
    }
    fixtures
}

const TRACE: &str = r#"
struct Input { origin:vec3<f32>, base:u32, direction:vec3<f32>, skip:u32 }
@group(0) @binding(0) var<storage,read> surface_words:array<u32>;
@group(0) @binding(1) var<storage,read> inputs:array<Input>;
@group(0) @binding(2) var<storage,read_write> outputs:array<SurfaceHit>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x>=arrayLength(&inputs) {return;}
    let i=inputs[id.x]; outputs[id.x]=surface_trace(i.base,i.origin,i.direction,i.skip!=0u);
}"#;

#[test]
fn packed_gpu_materials_and_skipping_match_independent_grid_intervals() {
    let gpu = Gpu::new();
    let rays = rays();
    let mut checked = 0;
    let mut ordinary_iterations = 0u64;
    let mut skipping_iterations = 0u64;
    for (name, brick, dense) in fixtures() {
        // A nonzero offset also validates variable-length concatenated payloads.
        let mut words = vec![0xdeadbeef; 7];
        words.extend_from_slice(brick.words());
        let lookup = gpu.run(
            &words,
            &[7],
            r#"
@group(0) @binding(0) var<storage,read> surface_words:array<u32>;
@group(0) @binding(1) var<storage,read> bases:array<u32>;
@group(0) @binding(2) var<storage,read_write> outputs:array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x>=arrayLength(&outputs) {return;}
    outputs[id.x]=surface_material(bases[0],vec3<u32>(id.x%32u,id.x/32u%32u,id.x/1024u));
}"#,
            32768,
            1,
        );
        assert_eq!(lookup, dense, "packed material {name}");
        let mut inputs = Vec::new();
        for ray in &rays {
            for skip in [0, 1] {
                inputs.extend(ray.origin.map(f32::to_bits));
                inputs.push(7);
                inputs.extend(ray.direction.map(f32::to_bits));
                inputs.push(skip);
            }
        }
        let hits = gpu.run(&words, &inputs, TRACE, rays.len() * 2, 8);
        for (i, ray) in rays.iter().enumerate() {
            let reference = oracle(*ray, &dense);
            for skip in 0..2 {
                let hit = &hits[(i * 2 + skip) * 8..(i * 2 + skip + 1) * 8];
                if skip == 0 {
                    ordinary_iterations += u64::from(hit[5]);
                } else {
                    skipping_iterations += u64::from(hit[5]);
                }
                if let Some((cell, material, distance, face)) = reference {
                    assert_eq!(hit[3], material, "material {name}/{i}/{skip} {ray:?}");
                    assert_eq!(
                        [hit[0] as i32, hit[1] as i32, hit[2] as i32],
                        cell,
                        "cell {name}/{i}/{skip} {ray:?}"
                    );
                    if let Some(face) = face {
                        assert_eq!(hit[4], face, "face {name}/{i}/{skip} {ray:?}");
                    }
                    let error = (f64::from(f32::from_bits(hit[6])) - distance).abs();
                    assert!(
                        error <= 0.00001 * distance.max(1.0),
                        "distance {name}/{i}/{skip} error={error}"
                    );
                } else {
                    assert_eq!(hit[3], 0, "miss or exhaustion {name}/{i}/{skip} {ray:?}");
                }
                checked += 1;
            }
        }
        eprintln!(
            "SURFACE_CACHE_FIXTURE name={name} bytes={} faces={}",
            brick.material_bytes(),
            brick.summary.face_count()
        );
    }
    eprintln!("SURFACE_CACHE_CHECK materials={} rays={checked} plain_iterations={ordinary_iterations} skip_iterations={skipping_iterations}",9*32768);
    assert!(skipping_iterations < ordinary_iterations);
}

#[test]
#[ignore = "isolated warm GPU traversal diagnostic; not a full-frame benchmark"]
fn measure_warm_gpu_brick_traversal() {
    let gpu = Gpu::new();
    for (name, brick, dense) in fixtures() {
        let mut inputs = Vec::new();
        let mut expected = Vec::new();
        // Coherent orthographic oblique view. Each invocation performs one
        // brick intersection; hierarchy, shading and uploads are not measured.
        for y in 0..512 {
            for x in 0..512 {
                let ray = Ray {
                    origin: [
                        (x as f32 + 0.5) * 32.0 / 512.0,
                        (y as f32 + 0.5) * 32.0 / 512.0,
                        34.0,
                    ],
                    direction: [0.125, -0.25, -1.0],
                };
                if x % 32 == 0 && y % 32 == 0 {
                    expected.push((x + y * 512, oracle(ray, &dense)));
                }
                inputs.extend(ray.origin.map(f32::to_bits));
                inputs.push(0);
                inputs.extend(ray.direction.map(f32::to_bits));
                inputs.push(0);
            }
        }
        let mut previous: Option<Vec<u32>> = None;
        // ABBA order reveals some clock/order sensitivity without claiming
        // these short, warm component measurements qualify engine performance.
        for (repeat, skip) in [false, true, true, false].into_iter().enumerate() {
            for i in inputs.chunks_exact_mut(8) {
                i[7] = u32::from(skip);
            }
            let label = format!("{name}-skip{}-r{repeat}", u32::from(skip));
            let hits = gpu.dispatch(brick.words(), &inputs, TRACE, 512 * 512, 8, Some(&label));
            for (i, reference) in &expected {
                let hit = &hits[i * 8..(i + 1) * 8];
                assert_eq!(hit[3], reference.map_or(0, |r| r.1), "{label}/{i}");
                if let Some((cell, _, _, _)) = reference {
                    assert_eq!(
                        [hit[0] as i32, hit[1] as i32, hit[2] as i32],
                        *cell,
                        "{label}/{i}"
                    );
                }
            }
            if let Some(previous) = &previous {
                for (i, (a, b)) in previous
                    .chunks_exact(8)
                    .zip(hits.chunks_exact(8))
                    .enumerate()
                {
                    assert_eq!(a[3], b[3], "coverage {label}/{i}");
                    if a[3] != 0 {
                        assert_eq!(&a[..5], &b[..5], "cell/material/face {label}/{i}");
                        assert_eq!(a[6], b[6], "distance {label}/{i}");
                    }
                }
            }
            previous = Some(hits);
        }
    }
}

#[test]
fn unoccluded_face_histograms_fail_hidden_wall_appearance() {
    // Both walls have equal area, but an orthographic camera at +Z sees only
    // the front material. Area mixtures lose the ordering needed by filtering.
    let sample = |q: [i32; 3]| {
        if q[2] == 25 {
            1
        } else if q[2] == 7 {
            3
        } else {
            0
        }
    };
    let brick = Brick::from_samples(sample);
    let weights = brick.summary.projected_area_weights([0.0, 0.0, 1.0]);
    assert_eq!(weights[4][1], 0.5);
    assert_eq!(weights[4][3], 0.5);
    let dense: Vec<u32> = (0..32768)
        .map(|i| sample([i % 32, i / 32 % 32, i / 1024]))
        .collect();
    for y in 0..32 {
        for x in 0..32 {
            let hit = oracle(
                Ray {
                    origin: [x as f32 + 0.5, y as f32 + 0.5, 34.0],
                    direction: [0.0, 0.0, -1.0],
                },
                &dense,
            )
            .unwrap();
            assert_eq!(hit.1, 1);
        }
    }
}
