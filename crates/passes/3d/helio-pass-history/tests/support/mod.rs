//! What the history's and the denoiser's GPU tests render with: an analytic
//! scene ray-cast on the CPU in f64 into the frame's G-buffer (depth,
//! normals, motion vectors), cameras uploaded as the renderer does (frame
//! space, jittered, the previous view-projection rebased into this frame's
//! origin), a graph run one frame at a time, and read-backs. The denoiser's
//! tests include this file (`#[path]`), so both crates test against one
//! fixture.

#![allow(dead_code)] // Each test binary uses its own part of the fixture.

use std::sync::Arc;

use bytemuck::Zeroable;
use glam::{DMat4, DVec2, DVec3, DVec4, Mat4};
use helio_core::{GpuCameraUniforms, RenderGraph, SceneBufferProjection, SceneInput};
use helio_feature_api::{BuildEnv, FrameDesc, NodeBuild};

pub struct Gpu {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
}

/// The machine's GPU (no adapter: the tests skip).
pub fn gpu() -> Option<&'static Gpu> {
    static GPU: std::sync::OnceLock<Option<Gpu>> = std::sync::OnceLock::new();
    GPU.get_or_init(|| {
        let instance = wgpu::Instance::new(helio_core::wgpu_instance_descriptor());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
        let limits = adapter.limits();
        let features = adapter.features()
            & (wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: features,
            required_limits: wgpu::Limits {
                max_buffer_size: limits.max_buffer_size.min(u32::MAX as u64),
                ..limits
            },
            ..Default::default()
        }))
        .ok()?;
        Some(Gpu {
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    })
    .as_ref()
}

/// What a camera ray meets.
#[derive(Clone, Copy, Debug)]
pub struct Hit {
    /// World position (f64).
    pub point: DVec3,
    pub normal: DVec3,
    /// Where the same surface point was last frame, for geometry that moves
    /// by itself (the drawer's motion vector).
    pub previous: Option<DVec3>,
    /// The scene's quad it lies on.
    pub quad: usize,
}

/// An axis-aligned rectangle facing +z (a wall seen from the front) or +y
/// (ground), in world space, optionally moving.
#[derive(Clone, Copy, Debug)]
pub struct Quad {
    pub centre: DVec3,
    pub half: DVec2,
    /// 2: faces +z (extent in x, y); 1: faces +y (extent in x, z).
    pub axis: usize,
    /// How far it moved since last frame (world).
    pub moved: DVec3,
}

impl Quad {
    pub fn wall(centre: DVec3, half: DVec2) -> Self {
        Self {
            centre,
            half,
            axis: 2,
            moved: DVec3::ZERO,
        }
    }

    pub fn ground(centre: DVec3, half: DVec2) -> Self {
        Self {
            centre,
            half,
            axis: 1,
            moved: DVec3::ZERO,
        }
    }

    fn hit(&self, origin: DVec3, dir: DVec3) -> Option<(f64, Hit)> {
        let a = self.axis;
        if dir[a].abs() < 1e-12 {
            return None;
        }
        let t = (self.centre[a] - origin[a]) / dir[a];
        if t <= 0.0 {
            return None;
        }
        let p = origin + dir * t;
        let (u, v) = if a == 2 { (0, 1) } else { (0, 2) };
        let d = p - self.centre;
        if d[u].abs() > self.half.x || d[v].abs() > self.half.y {
            return None;
        }
        let mut normal = DVec3::ZERO;
        normal[a] = if dir[a] < 0.0 { 1.0 } else { -1.0 };
        let moves = self.moved != DVec3::ZERO;
        Some((
            t,
            Hit {
                point: p,
                normal,
                previous: moves.then(|| p - self.moved),
                quad: 0,
            },
        ))
    }
}

/// A scene: quads, and a normal field that bends each hit's shading normal
/// (`None`: geometric normals).
#[derive(Clone)]
pub struct Scene {
    pub quads: Vec<Quad>,
    pub bumps: Option<Arc<dyn Fn(DVec3, DVec3) -> DVec3 + Send + Sync>>,
}

impl Scene {
    pub fn new(quads: Vec<Quad>) -> Self {
        Self { quads, bumps: None }
    }

    pub fn cast(&self, origin: DVec3, dir: DVec3) -> Option<Hit> {
        let mut best: Option<(f64, Hit)> = None;
        for (i, q) in self.quads.iter().enumerate() {
            if let Some((t, hit)) = q.hit(origin, dir) {
                if best.is_none_or(|(b, _)| t < b) {
                    best = Some((t, Hit { quad: i, ..hit }));
                }
            }
        }
        best.map(|(_, mut hit)| {
            if let Some(bumps) = &self.bumps {
                hit.normal = bumps(hit.point, hit.normal).normalize();
            }
            hit
        })
    }
}

/// A camera in world space (f64).
#[derive(Clone, Copy, Debug)]
pub struct View {
    pub eye: DVec3,
    pub forward: DVec3,
    pub up: DVec3,
    /// Vertical field of view, radians.
    pub fov: f64,
    /// The projection's jitter, in pixels (TSR's convention: NDC = 2 x / w).
    pub jitter: DVec2,
}

impl View {
    pub fn looking(eye: DVec3, forward: DVec3) -> Self {
        Self {
            eye,
            forward: forward.normalize(),
            up: DVec3::Y,
            fov: 1.0,
            jitter: DVec2::ZERO,
        }
    }

    pub fn with_jitter(mut self, jitter: DVec2) -> Self {
        self.jitter = jitter;
        self
    }
}

pub const NEAR: f64 = 0.05;

/// The world-to-view matrix with the eye at `eye - origin` (frame space).
fn view_matrix(view: &View, origin: DVec3) -> DMat4 {
    glam::dcamera::rh::view::look_to_mat4(view.eye - origin, view.forward, view.up)
}

/// The jittered reversed infinite projection of `view` at `size`.
fn projection(view: &View, size: [u32; 2]) -> DMat4 {
    let aspect = f64::from(size[0]) / f64::from(size[1]);
    let p = glam::dcamera::rh::proj::directx::perspective_infinite_reverse(view.fov, aspect, NEAR);
    let j = DVec3::new(
        view.jitter.x * 2.0 / f64::from(size[0]),
        view.jitter.y * 2.0 / f64::from(size[1]),
        0.0,
    );
    DMat4::from_translation(j) * p
}

/// The world point `x`'s position on `view`'s raster, in pixels, and its
/// view depth.
pub fn project(view: &View, size: [u32; 2], x: DVec3) -> (DVec2, f64) {
    let clip = projection(view, size) * view_matrix(view, DVec3::ZERO) * x.extend(1.0);
    let ndc = clip.truncate() / clip.w;
    (
        DVec2::new(
            (ndc.x + 1.0) * 0.5 * f64::from(size[0]),
            (1.0 - ndc.y) * 0.5 * f64::from(size[1]),
        ),
        clip.w,
    )
}

/// The world ray through pixel `p`'s centre of `view`'s jittered raster.
pub fn ray(view: &View, size: [u32; 2], p: [u32; 2]) -> DVec3 {
    let uv = DVec2::new(
        (f64::from(p[0]) + 0.5) / f64::from(size[0]),
        (f64::from(p[1]) + 0.5) / f64::from(size[1]),
    );
    let ndc = DVec4::new(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.5, 1.0);
    let inv = (projection(view, size) * view_matrix(view, DVec3::ZERO)).inverse();
    let far = inv * ndc;
    (far.truncate() / far.w - view.eye).normalize()
}

/// One pixel of the G-buffer as drawn.
#[derive(Clone, Copy, Debug, Default)]
pub struct Texel {
    pub hit: Option<Hit>,
}

struct Previous {
    view: View,
}

/// The fixture: a graph over the analytic G-buffer, one view.
pub struct Fixture {
    pub gpu: &'static Gpu,
    pub size: [u32; 2],
    pub graph: RenderGraph,
    camera_buffer: wgpu::Buffer,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    target_view: wgpu::TextureView,
    depth_source: wgpu::Texture,
    depth_pipeline: wgpu::RenderPipeline,
    depth_group: wgpu::BindGroup,
    targets: [wgpu::Texture; 5],
    views: [wgpu::TextureView; 5],
    previous: Option<Previous>,
    pub frame: u64,
    pub view_id: u32,
    /// This frame's G-buffer, as cast (row-major).
    pub texels: Vec<Texel>,
    /// This frame's camera uniforms.
    pub camera: GpuCameraUniforms,
    pub origin: DVec3,
    stats: helio_core::FrameStats,
}

/// `Rgba32Float` and `Depth32Float` texel writes.
const DEPTH_WGSL: &str = "
@group(0) @binding(0) var source: texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    return vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
}
@fragment fn fs(@builtin(position) p: vec4<f32>) -> @builtin(frag_depth) f32 {
    return textureLoad(source, vec2<i32>(p.xy), 0).r;
}
";

struct Input {
    gpu: &'static Gpu,
    camera_buffer: wgpu::Buffer,
    camera: GpuCameraUniforms,
    origin: DVec3,
    frame: u64,
    buffers: SceneBufferProjection,
}

impl SceneInput for Input {
    fn world_origin(&self) -> DVec3 {
        self.origin
    }
    fn device(&self) -> &Arc<wgpu::Device> {
        &self.gpu.device
    }
    fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.gpu.queue
    }
    fn frame_count(&self) -> u64 {
        self.frame
    }
    fn camera(&self) -> &wgpu::Buffer {
        &self.camera_buffer
    }
    fn camera_data(&self) -> &GpuCameraUniforms {
        &self.camera
    }
    fn camera_generation(&self) -> u64 {
        0
    }
    fn scene_buffers(&self) -> &SceneBufferProjection {
        &self.buffers
    }
}

impl Fixture {
    /// A fixture of `size` whose graph `build` makes (with a build
    /// environment over the fixture's device and camera, any settings
    /// `env` adds).
    pub fn new(
        gpu: &'static Gpu,
        size: [u32; 2],
        env: impl FnOnce(&mut BuildEnv<'_>),
        build: impl FnOnce(&mut NodeBuild<'_, '_>) -> Result<(), String>,
    ) -> Self {
        let device = &gpu.device;
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test cameras"),
            size: 2 * std::mem::size_of::<GpuCameraUniforms>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let stats = helio_core::FrameStats::new(device);
        let frame = FrameDesc {
            width: size[0],
            height: size[1],
            internal_width: size[0],
            internal_height: size[1],
            surface_format: wgpu::TextureFormat::Rgba8Unorm,
            xr: false,
        };
        let mut graph = RenderGraph::new(device, &gpu.queue);
        {
            let mut build_env = BuildEnv::new(
                device,
                &gpu.queue,
                frame,
                &camera_buffer,
                &camera_buffer,
                &stats,
                true,
            );
            env(&mut build_env);
            let params = Default::default();
            build(&mut NodeBuild {
                graph: &mut graph,
                env: &mut build_env,
                params: &params,
                node_id: "test",
            })
            .expect("the test graph builds");
        }
        graph.lock(size[0], size[1]);
        let extent = wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        };
        let texture = |label: &str, format, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let depth_texture = texture(
            "test depth",
            wgpu::TextureFormat::Depth32Float,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let target = texture(
            "test target",
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let sampled = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
        let depth_source = texture("test depth source", wgpu::TextureFormat::R32Float, sampled);
        // albedo, normal, orm, emissive, velocity
        let targets = ["albedo", "normal", "orm", "emissive", "velocity"]
            .map(|label| texture(label, wgpu::TextureFormat::Rgba32Float, sampled));
        let views = targets
            .each_ref()
            .map(|t| t.create_view(&Default::default()));
        let module = helio_core::shader::module(device, "test depth", DEPTH_WGSL);
        let depth_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("test depth"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                targets: &[],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let depth_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &depth_pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(
                    &depth_source.create_view(&Default::default()),
                ),
            }],
        });
        Self {
            gpu,
            size,
            graph,
            camera_buffer,
            depth_view: depth_texture.create_view(&Default::default()),
            depth_texture,
            target_view: target.create_view(&Default::default()),
            depth_source,
            depth_pipeline,
            depth_group,
            targets,
            views,
            previous: None,
            frame: 0,
            view_id: 0,
            texels: Vec::new(),
            camera: GpuCameraUniforms::zeroed(),
            origin: DVec3::ZERO,
            stats,
        }
    }

    /// Casts `scene` from `view` with the frame origin at `origin` into the
    /// G-buffer and uploads the camera, as a renderer's frame does.
    pub fn draw(&mut self, scene: &Scene, view: &View, origin: DVec3) {
        let size = self.size;
        let (w, h) = (size[0] as usize, size[1] as usize);
        let forward = view.forward;
        let mut depth = vec![0f32; w * h];
        let mut normal = vec![[0f32; 4]; w * h];
        let mut velocity = vec![[0f32; 4]; w * h];
        let orm = vec![[1.0f32, 0.5, 0.0, 0.0]; w * h];
        let albedo = vec![[0.5f32, 0.5, 0.5, 1.0]; w * h];
        let mut texels = vec![Texel::default(); w * h];
        let previous = self.previous.as_ref().map(|p| p.view).unwrap_or(*view);
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let dir = ray(view, size, [x as u32, y as u32]);
                let Some(hit) = scene.cast(view.eye, dir) else {
                    continue;
                };
                let view_depth = (hit.point - view.eye).dot(forward);
                depth[i] = (NEAR / view_depth) as f32;
                normal[i] = [
                    hit.normal.x as f32,
                    hit.normal.y as f32,
                    hit.normal.z as f32,
                    0.0,
                ];
                if let Some(before) = hit.previous {
                    let (now, _) = project(view, size, hit.point);
                    let (then, _) = project(&previous, size, before);
                    let m = now - then;
                    velocity[i] = [m.x as f32, m.y as f32, 0.0, 1.0];
                }
                texels[i] = Texel { hit: Some(hit) };
            }
        }
        let queue = &self.gpu.queue;
        let write = |texture: &wgpu::Texture, bytes: &[u8], texel: u32| {
            queue.write_texture(
                texture.as_image_copy(),
                bytes,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(size[0] * texel),
                    rows_per_image: Some(size[1]),
                },
                wgpu::Extent3d {
                    width: size[0],
                    height: size[1],
                    depth_or_array_layers: 1,
                },
            );
        };
        write(&self.depth_source, bytemuck::cast_slice(&depth), 4);
        write(&self.targets[0], bytemuck::cast_slice(&albedo), 16);
        write(&self.targets[1], bytemuck::cast_slice(&normal), 16);
        write(&self.targets[2], bytemuck::cast_slice(&orm), 16);
        write(
            &self.targets[3],
            bytemuck::cast_slice(&vec![[0f32; 4]; w * h]),
            16,
        );
        write(&self.targets[4], bytemuck::cast_slice(&velocity), 16);
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("test depth"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.depth_pipeline);
            pass.set_bind_group(0, &self.depth_group, &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit([encoder.finish()]);

        // The camera as the renderer uploads it: frame space, jittered, the
        // previous view-projection in this frame's origin.
        let proj = projection(view, size);
        let view_matrix = view_matrix(view, origin);
        let previous_view_proj = projection(&previous, size) * view_matrix_of(&previous, origin);
        let f = |m: DMat4| m.as_mat4();
        let eye = (view.eye - origin).as_vec3();
        let mut camera = GpuCameraUniforms::new(
            f(view_matrix),
            f(proj),
            eye,
            NEAR as f32,
            1000.0,
            self.frame as u32,
            [
                (view.jitter.x * 2.0 / f64::from(size[0])) as f32,
                (view.jitter.y * 2.0 / f64::from(size[1])) as f32,
            ],
            f(previous_view_proj),
        )
        .with_view_id(self.view_id);
        // The renderer composes view-projection in f32 from these.
        camera.view_proj = (Mat4::from_cols_array(&camera.proj)
            * Mat4::from_cols_array(&camera.view))
        .to_cols_array();
        camera.inv_view_proj = Mat4::from_cols_array(&camera.view_proj)
            .inverse()
            .to_cols_array();
        queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[camera, camera]),
        );
        self.camera = camera;
        self.origin = origin;
        self.texels = texels;
        self.previous = Some(Previous { view: *view });
    }

    /// Runs the graph over the frame drawn, with `seeds` the textures a
    /// test's producer wrote (by resource name), and waits for it.
    pub fn run(&mut self, seeds: &[(&'static str, &wgpu::TextureView)]) {
        self.submit(seeds);
        self.gpu
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
    }

    /// Runs the graph as a renderer's frames run: at most `in_flight`
    /// frames queued on the GPU, never waiting for the last (GPU timings of
    /// a synced loop read low clocks).
    pub fn run_pipelined(
        &mut self,
        seeds: &[(&'static str, &wgpu::TextureView)],
        done: &Arc<std::sync::atomic::AtomicU64>,
        in_flight: u64,
    ) {
        self.submit(seeds);
        let finished = done.clone();
        self.gpu.queue.on_submitted_work_done(move || {
            finished.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        });
        while done.load(std::sync::atomic::Ordering::Relaxed) + in_flight < self.frame {
            self.gpu.device.poll(wgpu::PollType::Poll).unwrap();
            std::thread::yield_now();
        }
    }

    fn submit(&mut self, seeds: &[(&'static str, &wgpu::TextureView)]) {
        let mut registry = helio_core::ResourceRegistry::empty();
        registry.write_view_group(
            helio_core::ResourceKey::new(helio_frame_api::gbuffer::GROUP),
            [
                &self.views[0],
                &self.views[1],
                &self.views[2],
                &self.views[3],
            ],
            "test",
        );
        registry.write(
            helio_core::ResourceKey::new(helio_frame_api::gbuffer::VELOCITY),
            &self.views[4],
            "test",
        );
        for (name, view) in seeds {
            registry.write(helio_core::ResourceKey::new(name), *view, "test producer");
        }
        let input = Input {
            gpu: self.gpu,
            camera_buffer: self.camera_buffer.clone(),
            camera: self.camera,
            origin: self.origin,
            frame: self.frame,
            buffers: SceneBufferProjection::empty(),
        };
        self.graph
            .execute_with_registry(&input, &self.target_view, &self.depth_view, &mut registry)
            .expect("the frame renders");
        self.frame += 1;
    }

    pub fn frame_stats(&self) -> &helio_core::FrameStats {
        &self.stats
    }
}

fn view_matrix_of(view: &View, origin: DVec3) -> DMat4 {
    view_matrix(view, origin)
}

/// Reads `texture` back as `f32` texels of `channels` (`Rgba16Float`,
/// `Rgba32Float` or `R32Float`).
pub fn read(gpu: &Gpu, texture: &wgpu::Texture) -> Vec<[f32; 4]> {
    let size = texture.size();
    let (bytes, channels, half) = match texture.format() {
        wgpu::TextureFormat::Rgba16Float => (8, 4, true),
        wgpu::TextureFormat::Rgba32Float => (16, 4, false),
        wgpu::TextureFormat::R32Float => (4, 1, false),
        other => panic!("cannot read {other:?}"),
    };
    let row = (size.width * bytes).div_ceil(256) * 256;
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::from(row * size.height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: Some(size.height),
            },
        },
        size,
    );
    gpu.queue.submit([encoder.finish()]);
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, |r| r.unwrap());
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    let data = staging.slice(..).get_mapped_range().unwrap();
    let mut out = Vec::with_capacity((size.width * size.height) as usize);
    for y in 0..size.height {
        let line = &data[(y * row) as usize..];
        for x in 0..size.width as usize {
            let mut texel = [0f32; 4];
            for (c, value) in texel.iter_mut().enumerate().take(channels) {
                *value = if half {
                    let at = x * 8 + c * 2;
                    f16_to_f32(u16::from_le_bytes([line[at], line[at + 1]]))
                } else {
                    let at = x * bytes as usize + c * 4;
                    f32::from_le_bytes(line[at..at + 4].try_into().unwrap())
                };
            }
            out.push(texel);
        }
    }
    out
}

/// A half float's value.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let mantissa = f32::from(bits & 0x3ff);
    sign * match exponent {
        0 => mantissa * (-24.0f32).exp2(),
        31 => {
            if mantissa == 0.0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        }
        e => (1.0 + mantissa / 1024.0) * ((e - 15) as f32).exp2(),
    }
}

/// A half float's bits (round to nearest; finite values in range).
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let a = value.abs();
    if a.is_nan() {
        return 0x7e00;
    }
    if a >= 65520.0 {
        return sign | 0x7c00;
    }
    if a < 6.1e-5 {
        return sign | (a / (-24.0f32).exp2()).round() as u16;
    }
    let exponent = a.log2().floor() as i32;
    let mut e = exponent;
    let mut m = (a / (e as f32).exp2() - 1.0) * 1024.0;
    m = m.round();
    if m >= 1024.0 {
        m = 0.0;
        e += 1;
    }
    sign | (((e + 15) as u16) << 10) | m as u16
}

/// Writes `texels` (`Rgba16Float`) into `texture`.
pub fn write_half(gpu: &Gpu, texture: &wgpu::Texture, texels: &[[f32; 4]]) {
    let size = texture.size();
    let bits: Vec<u16> = texels.iter().flatten().map(|v| f32_to_f16(*v)).collect();
    gpu.queue.write_texture(
        texture.as_image_copy(),
        bytemuck::cast_slice(&bits),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(size.width * 8),
            rows_per_image: Some(size.height),
        },
        size,
    );
}

/// Runs `step` until it says done, failing past `seconds` of wall clock.
pub fn until(seconds: u64, what: &str, mut step: impl FnMut() -> bool) {
    let start = std::time::Instant::now();
    while !step() {
        assert!(
            start.elapsed().as_secs() < seconds,
            "{what}: not done after {seconds} s"
        );
    }
}
