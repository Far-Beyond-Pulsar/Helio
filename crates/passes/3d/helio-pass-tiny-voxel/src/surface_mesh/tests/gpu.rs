use super::*;
use crate::surface_mesh::raster::{Camera, Output, Raster};
use glam::{DVec3, Vec3};

mod control;

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
            eprintln!("SURFACE_MESH_ADAPTER {:?}", adapter.get_info());
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
    fn map(&self, buffer: &wgpu::Buffer) -> Vec<u8> {
        let (tx, rx) = std::sync::mpsc::channel();
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        rx.recv().unwrap().unwrap();
        let bytes = buffer.slice(..).get_mapped_range().unwrap().to_vec();
        buffer.unmap();
        bytes
    }
    fn buffer(&self, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage,
            mapped_at_creation: false,
        })
    }
}
struct Target {
    color: wgpu::Texture,
    depth: wgpu::Texture,
    resolved: Option<wgpu::Texture>,
    size: [u32; 2],
    bytes: u32,
}
impl Target {
    fn new(gpu: &Gpu, size: [u32; 2], output: Output, samples: u32) -> Self {
        let texture = |format, sample_count, usage| {
            gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: size[0],
                    height: size[1],
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let color = texture(
            output.format(),
            samples,
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | if samples == 1 {
                    wgpu::TextureUsages::COPY_SRC
                        | if matches!(output, Output::Lit) {
                            wgpu::TextureUsages::STORAGE_BINDING
                        } else {
                            wgpu::TextureUsages::empty()
                        }
                } else {
                    wgpu::TextureUsages::empty()
                },
        );
        let depth = texture(
            wgpu::TextureFormat::Depth32Float,
            samples,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        let resolved = (samples > 1).then(|| {
            texture(
                output.format(),
                1,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            )
        });
        Self {
            color,
            depth,
            resolved,
            size,
            bytes: if matches!(output, Output::Identity) {
                16
            } else {
                8
            },
        }
    }
    fn encode(
        &self,
        raster: &Raster,
        encoder: &mut wgpu::CommandEncoder,
        timestamps: Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) {
        let color = self.color.create_view(&Default::default());
        let depth = self.depth.create_view(&Default::default());
        let resolve = self
            .resolved
            .as_ref()
            .map(|t| t.create_view(&Default::default()));
        raster.encode(encoder, &color, &depth, resolve.as_ref(), timestamps);
    }
    fn read(&self, gpu: &Gpu) -> Vec<u8> {
        let row = (self.size[0] * self.bytes).div_ceil(256) * 256;
        let buffer = gpu.buffer(
            u64::from(row * self.size[1]),
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            self.resolved
                .as_ref()
                .unwrap_or(&self.color)
                .as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(self.size[1]),
                },
            },
            wgpu::Extent3d {
                width: self.size[0],
                height: self.size[1],
                depth_or_array_layers: 1,
            },
        );
        gpu.queue.submit([encoder.finish()]);
        gpu.map(&buffer)
            .chunks_exact(row as usize)
            .flat_map(|r| r[..(self.size[0] * self.bytes) as usize].iter().copied())
            .collect()
    }
    fn render(&self, gpu: &Gpu, raster: &Raster, camera: &Camera) -> Vec<u8> {
        raster.camera(&gpu.queue, camera);
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        self.encode(raster, &mut encoder, None);
        gpu.queue.submit([encoder.finish()]);
        self.read(gpu)
    }
}

fn material(fixture: usize, q: [i32; 3]) -> u32 {
    if q.iter().any(|v| !(0..32).contains(v)) {
        return 0;
    }
    match fixture {
        0 => {
            if q[1] < 3 + q[0] / 2 + q[2] / 4 {
                1
            } else {
                0
            }
        }
        1 => {
            if q[2] == 25 {
                2
            } else if q[2] == 7 {
                3
            } else {
                0
            }
        }
        2 => {
            if q[2] == 25 && !(12..20).contains(&q[0]) || q[2] == 25 && !(10..22).contains(&q[1]) {
                2
            } else if q[2] == 7 {
                3
            } else {
                0
            }
        }
        3 => {
            let r = q.map(|v| v - 16).iter().map(|v| v * v).sum::<i32>();
            let cut = (q[0] - 22).pow(2) + (q[1] - 16).pow(2) + (q[2] - 26).pow(2);
            if (121..196).contains(&r) && cut > 36 {
                3
            } else {
                0
            }
        }
        _ => {
            if q[1] < 20 {
                (q[0] + q[1] + q[2]).rem_euclid(4) as u32
            } else {
                0
            }
        }
    }
}
fn ray(camera: &Camera, size: [u32; 2], pixel: [f64; 2]) -> DVec3 {
    let v = |a: [f32; 4]| DVec3::new(f64::from(a[0]), f64::from(a[1]), f64::from(a[2]));
    (v(camera.forward)
        + v(camera.right)
            * ((pixel[0] / f64::from(size[0]) * 2.0 - 1.0)
                * f64::from(camera.eye[3])
                * f64::from(camera.right[3]))
        + v(camera.up) * ((1.0 - pixel[1] / f64::from(size[1]) * 2.0) * f64::from(camera.eye[3])))
    .normalize()
}
// Independent sorted grid-plane intervals, with no rectangle or DDA traversal.
fn oracle(fixture: usize, eye: DVec3, direction: DVec3) -> Option<([i32; 3], u32, u32, f64)> {
    let mut events = vec![(0.0, 0)];
    for a in 0..3 {
        if direction[a] == 0.0 {
            continue;
        }
        for plane in 0..=32 {
            let t = (f64::from(plane) - eye[a]) / direction[a];
            if t >= 0.0 {
                events.push((t, 1 + a as u32 * 2 + u32::from(direction[a] > 0.0)));
            }
        }
    }
    events.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    events.dedup_by(|a, b| a.0 == b.0);
    for pair in events.windows(2) {
        let p = eye + direction * ((pair[0].0 + pair[1].0) * 0.5);
        let cell = p.floor().as_ivec3().to_array();
        let m = material(fixture, cell);
        if m != 0 {
            return Some((cell, m, pair[0].1, pair[0].0));
        }
    }
    None
}
fn identity(hit: &[u32]) -> Option<([i32; 3], u32, u32)> {
    (hit[1] != 0).then(|| {
        (
            [
                (hit[0] % 32) as i32,
                (hit[0] / 32 % 32) as i32,
                (hit[0] / 1024) as i32,
            ],
            hit[1] & 255,
            hit[1] >> 8,
        )
    })
}
fn eye(camera: &Camera) -> DVec3 {
    DVec3::new(
        f64::from(camera.eye[0]),
        f64::from(camera.eye[1]),
        f64::from(camera.eye[2]),
    )
}

#[test]
fn rasterized_faces_match_canonical_visibility_including_hidden_and_destroyed_walls() {
    let gpu = Gpu::new();
    let size = [96, 64];
    let target = Target::new(&gpu, size, Output::Identity, 1);
    let directions = [
        DVec3::new(0.0, 0.0, -1.0),
        DVec3::new(0.25, -0.35, -1.0),
        DVec3::new(-0.75, -0.2, -0.9),
        DVec3::new(0.5, 0.3, 1.0),
        DVec3::new(1.0, -0.02, 0.05),
        DVec3::new(0.01, -1.0, 0.02),
    ];
    let mut checked = 0;
    let mut boundary = 0;
    for fixture in 0..5 {
        let mesh = Mesh::from_samples(|q| material(fixture, q));
        let raster = Raster::new(&gpu.device, &mesh, Output::Identity, 1);
        for direction in directions {
            let center = DVec3::new(16.137, 15.781, 16.219);
            let camera = Camera::new(
                center - direction.normalize() * 48.0,
                direction,
                size,
                Vec3::new(-0.4, 0.7, 0.5),
            );
            let pixels = target.render(&gpu, &raster, &camera);
            for (i, hit) in bytemuck::cast_slice::<u8, u32>(&pixels)
                .chunks_exact(4)
                .enumerate()
            {
                let pixel = [
                    (i % size[0] as usize) as f64 + 0.5,
                    (i / size[0] as usize) as f64 + 0.5,
                ];
                let expected = oracle(fixture, eye(&camera), ray(&camera, size, pixel));
                let expected_id = expected.map(|r| (r.0, r.1, r.2));
                let actual = identity(hit);
                if actual != expected_id {
                    // Fixed-function raster coverage is quantized. Declare and
                    // audit a 0.01-pixel boundary tolerance, not arbitrary count
                    // tolerance or permission to leak hidden material.
                    let mut found = false;
                    for y in [-0.01, 0.0, 0.01] {
                        for x in [-0.01, 0.0, 0.01] {
                            found |= oracle(
                                fixture,
                                eye(&camera),
                                ray(&camera, size, [pixel[0] + x, pixel[1] + y]),
                            )
                            .map(|r| (r.0, r.1, r.2))
                                == actual;
                        }
                    }
                    // A grazing face can have a visible wedge between those
                    // probes. Find the nearest points of its PROJECTED edges;
                    // clamping in world space is not screen-distance minimization.
                    // Every accepted point still needs independent first-hit
                    // visibility within the unchanged 0.01-pixel bound.
                    if !found {
                        if let Some((cell, _, face)) = actual {
                            let axis = (face as usize - 1) / 2;
                            let u = (axis + 1) % 3;
                            let v = (axis + 2) % 3;
                            let project = |point: DVec3| {
                                let delta = point - eye(&camera);
                                let vec = |a: [f32; 4]| {
                                    DVec3::new(f64::from(a[0]), f64::from(a[1]), f64::from(a[2]))
                                };
                                let distance = delta.dot(vec(camera.forward));
                                glam::DVec2::new(
                                    (delta.dot(vec(camera.right))
                                        / (distance
                                            * f64::from(camera.eye[3])
                                            * f64::from(camera.right[3]))
                                        * 0.5
                                        + 0.5)
                                        * f64::from(size[0]),
                                    (0.5 - delta.dot(vec(camera.up))
                                        / (distance * f64::from(camera.eye[3]))
                                        * 0.5)
                                        * f64::from(size[1]),
                                )
                            };
                            let corners =
                                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]].map(|uv| {
                                    let mut p = DVec3::from_array(cell.map(f64::from));
                                    p[axis] += f64::from(face % 2 == 1);
                                    p[u] += uv[0];
                                    p[v] += uv[1];
                                    project(p)
                                });
                            let center = corners.iter().copied().sum::<glam::DVec2>() * 0.25;
                            let pixel = glam::DVec2::from_array(pixel);
                            for edge in 0..4 {
                                let a = corners[edge];
                                let b = corners[(edge + 1) % 4];
                                let direction = b - a;
                                let t = ((pixel - a).dot(direction) / direction.length_squared())
                                    .clamp(0.0, 1.0);
                                let candidate = (a + direction * t).lerp(center, 1e-6);
                                if (candidate - pixel).abs().max_element() <= 0.01 {
                                    found |= oracle(
                                        fixture,
                                        eye(&camera),
                                        ray(&camera, size, candidate.to_array()),
                                    )
                                    .map(|r| (r.0, r.1, r.2))
                                        == actual;
                                }
                            }
                        }
                    }
                    assert!(found,"visibility mismatch fixture={fixture} pixel={pixel:?} actual={actual:?} expected={expected:?} direction={direction:?}");
                    boundary += 1;
                } else if let Some((_, _, _, distance)) = expected {
                    assert!(
                        (f64::from(f32::from_bits(hit[2])) - distance).abs() < 0.002,
                        "depth mismatch fixture={fixture} pixel={pixel:?} actual={} expected={expected:?} direction={direction:?}", f32::from_bits(hit[2])
                    );
                }
                checked += 1;
            }
            // At the front-facing wall's interior, no rear material may leak.
            if fixture == 1 && direction == directions[0] {
                let center = (size[0] / 2 + size[0] * (size[1] / 2)) as usize;
                let words = bytemuck::cast_slice::<u8, u32>(&pixels);
                assert_eq!(words[center * 4 + 1] & 255, 2);
            }
        }
        eprintln!(
            "SURFACE_MESH_STORAGE fixture={fixture} faces={} quads={} bytes={}",
            mesh.exposed_faces,
            mesh.quads.len(),
            mesh.logical_bytes()
        );
    }
    eprintln!("SURFACE_MESH_VISIBILITY pixels={checked} boundary_tolerance_pixels={boundary} max_screen_tolerance=0.01");
}

#[test]
#[ignore = "local visibility/lighting and GPU component diagnostic; not an engine acceptance benchmark"]
fn measure_and_capture_independently_shaded_raster_samples() {
    let gpu = Gpu::new();
    let output = std::env::var_os("HELIO_MESH_AUDIT_DIR")
        .map(std::path::PathBuf::from)
        .expect("audit output directory required");
    std::fs::create_dir_all(&output).unwrap();
    let light = [Vec3::new(-0.4, 0.7, 0.5), Vec3::new(0.8, 0.15, -0.4)];
    for fixture in 0..5 {
        let start = std::time::Instant::now();
        let mesh = Mesh::from_samples(|q| material(fixture, q));
        let build_ms = start.elapsed().as_secs_f64() * 1000.0;
        eprintln!("SURFACE_MESH_BUILD fixture={fixture} milliseconds={build_ms:.4} faces={} quads={} bytes={}",mesh.exposed_faces,mesh.quads.len(),mesh.logical_bytes());
        for samples in [1, 4] {
            let raster = Raster::new(&gpu.device, &mesh, Output::Lit, samples);
            for (lighting, light) in light.into_iter().enumerate() {
                for movement in 0..2 {
                    let position = DVec3::new(16.137 + f64::from(movement) * 0.19, 25.17, 52.219);
                    for size in [[128, 72], [1024, 576]] {
                        if samples == 4 && size[0] == 1024 {
                            continue;
                        }
                        let target = Target::new(&gpu, size, Output::Lit, samples);
                        let camera =
                            Camera::new(position, DVec3::new(0.0, -0.18, -1.0), size, light);
                        let pixels = target.render(&gpu, &raster, &camera);
                        let name = format!(
                            "f{fixture}-l{lighting}-m{movement}-s{samples}-{}x{}",
                            size[0], size[1]
                        );
                        std::fs::write(output.join(name + ".rgba16"), pixels).unwrap();
                    }
                }
            }
            let size = [1280, 720];
            let target = Target::new(&gpu, size, Output::Lit, samples);
            let camera = Camera::new(
                DVec3::new(16.137, 25.17, 52.219),
                DVec3::new(0.0, -0.18, -1.0),
                size,
                light[0],
            );
            raster.camera(&gpu.queue, &camera);
            assert!(gpu
                .device
                .features()
                .contains(wgpu::Features::TIMESTAMP_QUERY));
            let count = 144;
            let query = gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: None,
                ty: wgpu::QueryType::Timestamp,
                count: count * 2,
            });
            let resolved = gpu.buffer(
                u64::from(count) * 16,
                wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            );
            let readback = gpu.buffer(
                u64::from(count) * 16,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            );
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            for i in 0..count {
                target.encode(
                    &raster,
                    &mut encoder,
                    Some(wgpu::RenderPassTimestampWrites {
                        query_set: &query,
                        beginning_of_pass_write_index: Some(i * 2),
                        end_of_pass_write_index: Some(i * 2 + 1),
                    }),
                );
            }
            encoder.resolve_query_set(&query, 0..count * 2, &resolved, 0);
            encoder.copy_buffer_to_buffer(&resolved, 0, &readback, 0, u64::from(count) * 16);
            gpu.queue.submit([encoder.finish()]);
            let times: Vec<_> = gpu
                .map(&readback)
                .chunks_exact(16)
                .skip(16)
                .map(|v| {
                    let a = u64::from_le_bytes(v[..8].try_into().unwrap());
                    let b = u64::from_le_bytes(v[8..].try_into().unwrap());
                    assert!(b >= a);
                    (b - a) as f64 * f64::from(gpu.queue.get_timestamp_period()) / 1e6
                })
                .collect();
            eprintln!(
                "SURFACE_MESH_GPU fixture={fixture} samples={samples} milliseconds={times:?}"
            );
        }
        control::run(&gpu, fixture, &output);
    }
}
