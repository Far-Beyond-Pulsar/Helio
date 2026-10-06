#![allow(dead_code)]
//! Headless GPU helpers for voxel planet tests.
use glam::{DVec3, Mat4, Vec3};
use helio_pass_voxel_planet::engine::{PlanetFrame, PlanetRenderer, Settings, GBUFFER_FORMATS};
use helio_pass_voxel_planet::Planet;
use std::sync::Arc;

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

pub fn gpu() -> Option<Gpu> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
    let limits = adapter.limits();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: adapter.features() & wgpu::Features::TIMESTAMP_QUERY,
        required_limits: wgpu::Limits {
            max_buffer_size: limits.max_buffer_size.min(u32::MAX as u64),
            ..limits
        },
        ..Default::default()
    }))
    .ok()?;
    Some(Gpu { device, queue })
}

pub fn read_buffer(gpu: &Gpu, buffer: &wgpu::Buffer, size: u64) -> Vec<u8> {
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    gpu.queue.submit([encoder.finish()]);
    staging.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
    gpu.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let data = staging.slice(..).get_mapped_range().unwrap().to_vec();
    data
}

/// Minimal offscreen frame: camera buffer, GBuffer targets and depth.
pub struct Target {
    pub size: [u32; 2],
    pub camera: wgpu::Buffer,
    pub colors: Vec<wgpu::Texture>,
    pub views: Vec<wgpu::TextureView>,
    pub depth: wgpu::Texture,
    pub depth_view: wgpu::TextureView,
    pub fov_y: f32,
}

impl Target {
    pub fn new(gpu: &Gpu, size: [u32; 2]) -> Self {
        let tex = |format, usage| {
            gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let colors: Vec<_> = GBUFFER_FORMATS
            .iter()
            .map(|f| tex(*f, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC))
            .collect();
        let views = colors.iter().map(|t| t.create_view(&Default::default())).collect();
        let depth = tex(
            wgpu::TextureFormat::Depth32Float,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
        );
        let depth_view = depth.create_view(&Default::default());
        let camera = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: std::mem::size_of::<helio_core::GpuCameraUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { size, camera, colors, views, depth, depth_view, fov_y: std::f32::consts::FRAC_PI_4 }
    }

    /// Camera at the local origin looking along `forward` (planet frame axes).
    pub fn camera(&self, forward: Vec3, up_hint: Vec3) -> helio_core::GpuCameraUniforms {
        let f = forward.normalize();
        let right = f.cross(up_hint).normalize();
        let up = right.cross(f);
        let view = Mat4::look_to_rh(Vec3::ZERO, f, up);
        let aspect = self.size[0] as f32 / self.size[1] as f32;
        let proj = Mat4::perspective_rh(self.fov_y, aspect, 0.05, 30_000_000.0);
        helio_core::GpuCameraUniforms::new(view, proj, Vec3::ZERO, 0.05, 30_000_000.0, 0, [0.0, 0.0], proj * view)
    }

    pub fn clear(&self, gpu: &Gpu, encoder: &mut wgpu::CommandEncoder) {
        let attachments: Vec<_> = self
            .views
            .iter()
            .map(|view| {
                Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                })
            })
            .collect();
        let _ = gpu;
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &attachments,
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_view,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    }

    /// Render one frame and wait for completion.
    pub fn render(&self, gpu: &Gpu, renderer: &mut PlanetRenderer, frame: &PlanetFrame, forward: Vec3, frame_num: u64) {
        let up = frame.planet.grid().up(frame.eye).as_vec3();
        let camera = self.camera(forward, if forward.normalize().dot(up).abs() > 0.99 { up.any_orthonormal_vector() } else { up });
        gpu.queue.write_buffer(&self.camera, 0, bytemuck::bytes_of(&camera));
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        self.clear(gpu, &mut encoder);
        let v: Vec<&wgpu::TextureView> = self.views.iter().collect();
        renderer.encode(
            &mut encoder,
            &camera,
            frame,
            self.size,
            [v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]],
            &self.depth_view,
            frame_num,
        );
        gpu.queue.submit([encoder.finish()]);
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    }
}

pub fn renderer(gpu: &Gpu, planet: Arc<Planet>, size: [u32; 2]) -> PlanetRenderer {
    PlanetRenderer::new(&gpu.device, &gpu.queue, planet, Settings::default(), size)
}

pub fn frame(planet: &Arc<Planet>, eye: DVec3) -> PlanetFrame {
    PlanetFrame { eye, planet: planet.clone(), sun: Vec3::new(0.3, 0.8, 0.4), shadows: false, picks: None }
}

/// Decoded primary hit.
#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub t: f32,
    pub i: i32,
    pub j: i32,
    pub k: i32,
    pub status: u32,
    pub face: u8,
    pub level: u32,
    pub normal: u32,
    /// Traversal loop steps (diagnostic counter).
    pub steps: u32,
}

pub fn hits(gpu: &Gpu, renderer: &PlanetRenderer) -> Vec<Hit> {
    let [w, h] = renderer.screen_size();
    let data = read_buffer(gpu, renderer.hit_buffer(), u64::from(w) * u64::from(h) * 32);
    data.chunks_exact(32)
        .map(|c| {
            let w = |i: usize| u32::from_le_bytes(c[i * 4..i * 4 + 4].try_into().unwrap());
            let info = w(4);
            Hit {
                t: f32::from_bits(w(0)),
                i: w(1) as i32,
                j: w(2) as i32,
                k: w(3) as i32,
                status: info & 3,
                face: ((info >> 2) & 7) as u8,
                level: (info >> 5) & 31,
                normal: (info >> 10) & 7,
                steps: w(6) & 0xffff,
            }
        })
        .collect()
}

/// World direction of pixel centre (x, y) for the test camera.
/// Whether accelerated hit `b` skipped no geometry that plain hit `a`
/// met (`rise`: vertical component of the pixel ray). A later start can
/// only skip geometry: a wrong miss, or a hit farther on. Level choice
/// depends on the path, though: the LOD dither hashes the column the ray is
/// in, a missing finer column leaves a ray on the coarser level it came
/// from, and a large empty box of a coarse level can pass over a finer
/// level's terrain, while a ray starting there locates on the finer level.
/// So `b` may meet the same surface a level or two apart, up to two cells
/// of the next coarser level away in height, or meet terrain `a` passed
/// over.
pub fn skipped_nothing(a: &Hit, b: &Hit, rise: f32, voxel: f32) -> bool {
    match (a.status, b.status) {
        (x, y) if x == y && x != 1 => true,
        (0, 1) => true,
        (1, 1) => b.t - a.t <= 0.0 || (b.t - a.t) * rise.max(1e-3) <= 2.0 * voxel * (2u32 << a.level.max(b.level)) as f32,
        _ => false,
    }
}

pub fn pixel_dir(target: &Target, camera: &helio_core::GpuCameraUniforms, x: u32, y: u32) -> DVec3 {
    let inv = Mat4::from_cols_array(&camera.inv_view_proj);
    let ndc = glam::Vec4::new(
        (x as f32 + 0.5) / target.size[0] as f32 * 2.0 - 1.0,
        1.0 - (y as f32 + 0.5) / target.size[1] as f32 * 2.0,
        0.5,
        1.0,
    );
    let w = inv * ndc;
    (w.truncate() / w.w).as_dvec3().normalize()
}

/// A land direction (surface above sea level) near `face` coordinates.
pub fn land(planet: &Planet, face: u8, fi: f64, fj: f64) -> DVec3 {
    let grid = planet.grid();
    let n = f64::from(grid.cells());
    for step in 0..400 {
        let a = fi + 0.013 * f64::from(step % 20);
        let b = fj + 0.017 * f64::from(step / 20);
        let (i, j) = ((a.fract() * n) as i32, (b.fract() * n) as i32);
        let top = planet.column_top(face, i, j, 0);
        if top > 40 {
            return grid.direction(face, f64::from(i) + 0.5, f64::from(j) + 0.5);
        }
    }
    panic!("no land found");
}

/// A land column whose face fractions lie in `[lo, hi)` on both axes.
pub fn land_within(planet: &Planet, face: u8, lo: f64, hi: f64) -> DVec3 {
    let grid = planet.grid();
    let n = f64::from(grid.cells());
    for step in 0..1600 {
        let a = lo + (hi - lo) * f64::from(step % 40) / 40.0;
        let b = lo + (hi - lo) * f64::from(step / 40) / 40.0;
        let (i, j) = ((a * n) as i32, (b * n) as i32);
        if planet.column_top(face, i, j, 0) > 40 {
            return grid.direction(face, f64::from(i) + 0.5, f64::from(j) + 0.5);
        }
    }
    panic!("no land found in [{lo}, {hi})");
}

/// The highest of a coarse sample of columns: a mountain peak.
pub fn peak(planet: &Planet) -> DVec3 {
    let grid = planet.grid();
    let n = grid.cells();
    let mut best = (i32::MIN, 0u8, 0, 0);
    for face in 0..6u8 {
        for a in 0..120 {
            for b in 0..120 {
                let (i, j) = (n / 120 * a + n / 240, n / 120 * b + n / 240);
                let h = planet.column_height(face, i, j, 0);
                if h > best.0 {
                    best = (h, face, i, j);
                }
            }
        }
    }
    grid.direction(best.1, f64::from(best.2) + 0.5, f64::from(best.3) + 0.5)
}
