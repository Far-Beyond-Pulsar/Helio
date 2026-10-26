//! Image-quality metrics of the whole pass, measured on the GPU through a real
//! render graph: flicker of a still, jittered scene with sub-pixel wires, and
//! of foliage (texel-sized high-contrast texture, depth edges, ground at a
//! grazing angle, lit by stochastic per-frame samples as the voxel terrain's
//! sun and sky are); response to a light switching without anything moving;
//! trails behind moving edges (object motion without motion vectors, and
//! camera motion); and behind an emissive object over foliage, moving by
//! itself or by parallax, with and without motion vectors.
//!
//! The input is a procedural scene point-sampled at each frame's jittered
//! raster positions, exactly as a rasterizer would, so every number here is
//! what a viewer would see from the pass, sharpening included.

use super::*;
use glam::{Mat4, Vec3};
use std::sync::Arc;

const SCENE_WGSL: &str = r#"
struct Scene {
    inv_view_proj: mat4x4<f32>,
    // brightness, bar left edge (world x), scene id (0 detail, 1 bar, 2 foliage), supersampling
    params: vec4<f32>,
    // target size in pixels, world units per output pixel, bar width in output pixels
    frame_size: vec4<f32>,
    // frame index, background depth (near edge), foreground depth, background depth (far edge)
    extra: vec4<f32>,
    prev_view_proj: mat4x4<f32>,
    // the bar's world x shift since last frame; 1 to write its motion vectors
    motion: vec4<f32>,
}
@group(0) @binding(0) var<uniform> s: Scene;

@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    return vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
}

// Whether `p` lies on an infinite wire through `o` at `degrees`, `width` wide.
fn wire(p: vec2<f32>, o: vec2<f32>, degrees: f32, width: f32) -> bool {
    let a = radians(degrees);
    let n = vec2<f32>(-sin(a), cos(a));
    return abs(dot(p - o, n)) < width * 0.5;
}

fn hash(c: vec2<f32>, salt: u32) -> f32 {
    var h = bitcast<u32>(vec2<i32>(c).x) * 0x8da6b343u ^ bitcast<u32>(vec2<i32>(c).y) * 0xd8163841u ^ salt * 0xcb1ab31fu;
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    return f32((h ^ (h >> 16u)) & 0xffffffu) / 16777216.0;
}

// Foliage: a leaf texture of one-texel cells (nearest-filtered, high
// contrast) at 2.5, 1.0 and 0.6 output pixels per texel, with nearer leaf
// blocks in front of it.
fn foreground(p: vec2<f32>) -> bool {
    let px = s.frame_size.z;
    let block = floor(p / (14.0 * px));
    let inside = fract(p / (14.0 * px));
    return hash(block, 7u) > 0.6 && all(inside > vec2<f32>(0.15)) && all(inside < vec2<f32>(0.85));
}

fn albedo(p: vec2<f32>) -> vec3<f32> {
    let px = s.frame_size.z;
    let texel = select(select(0.6, 1.0, p.x < 0.33), 2.5, p.x < -0.33) * px;
    let near = foreground(p);
    let h = hash(floor(p / texel), select(1u, 2u, near));
    var c = mix(vec3<f32>(0.03, 0.08, 0.02), vec3<f32>(0.22, 0.45, 0.09), h);
    if h > 0.9 { c = vec3<f32>(0.55, 0.65, 0.25); }
    if near { c *= vec3<f32>(0.8, 0.9, 0.7); }
    return c;
}

// Sun visibility: a canopy's shadow, cells 5 output pixels wide.
fn sun(p: vec2<f32>) -> f32 {
    let px = s.frame_size.z;
    return select(0.0, 1.0, hash(floor(p / (5.0 * px) + 0.37), 3u) > 0.45);
}

// Sky visibility: smooth, between 0.4 and 1.
fn sky(p: vec2<f32>) -> f32 {
    return 0.7 + 0.3 * sin(p.x * 9.0 + 1.3) * cos(p.y * 7.0);
}

// The bar: `frame_size.w` output pixels wide from world x `params.y` on
// (none if 0). Over the foliage it is an emissive block 1 m in front of the
// leaf blocks, many times as bright as their sunlit texels.
const EMISSIVE_BAR: f32 = 30.0;
fn on_bar(p: vec2<f32>) -> bool {
    return s.frame_size.w > 0.0 && p.x >= s.params.y && p.x < s.params.y + s.frame_size.w * s.frame_size.z;
}

fn scene(p: vec2<f32>) -> vec3<f32> {
    let px = s.frame_size.z;
    var c = vec3<f32>(0.05);
    if s.params.z > 1.5 {
        if on_bar(p) { return vec3<f32>(EMISSIVE_BAR); }
        return albedo(p) * (4.0 * sun(p) + 0.6 * sky(p)) * s.params.x;
    }
    if s.params.z < 0.5 {
        // A filled square rotated 12 degrees: long, nearly axis-aligned edges.
        let a = radians(12.0);
        let d = p - vec2<f32>(-0.5, -0.5);
        let q = vec2<f32>(cos(a) * d.x + sin(a) * d.y, -sin(a) * d.x + cos(a) * d.y);
        if all(abs(q) < vec2<f32>(0.3)) { c = vec3<f32>(0.9, 0.5, 0.2); }
        // A fan of 0.35 px wires: every orientation of sub-pixel geometry.
        if p.y > -0.1 && p.x < 0.1 {
            for (var i = 0; i < 9; i++) {
                if wire(p, vec2<f32>(-0.95, -0.1), 6.0 + f32(i) * 10.0, 0.35 * px) { c = vec3<f32>(1.0); }
            }
        }
        // A fence: 0.6 px slats every 3 px, slightly rotated.
        if p.x > 0.2 && p.y < -0.1 {
            let a = radians(7.0);
            let u = cos(a) * p.x + sin(a) * p.y;
            if fract(u / (3.0 * px)) < 0.2 { c = vec3<f32>(0.8, 0.85, 1.0); }
        }
        // Single near-horizontal wires 0.3, 0.5 and 0.8 px wide.
        if p.x > 0.2 && p.y > 0.0 {
            for (var i = 0; i < 3; i++) {
                if wire(p, vec2<f32>(0.0, 0.2 + f32(i) * 0.25), 1.5, (0.3 + f32(i) * 0.25) * px) { c = vec3<f32>(1.0); }
            }
        }
    } else if on_bar(p) {
        c = vec3<f32>(1.0);
    }
    return c * s.params.x;
}

fn world(uv: vec2<f32>) -> vec2<f32> {
    let w = s.inv_view_proj * vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.5, 1.0);
    return w.xy / w.w;
}

// Foliage lit the way the voxel terrain is: sun visibility sampled once per
// 2x2 pixel block at a pixel that rotates every frame, sky visibility once
// per 4x4 block at one of its 16 pixels. Only temporal accumulation turns
// either into a filtered term.
fn stochastic_foliage(frag: vec2<f32>) -> vec3<f32> {
    let size = s.frame_size.xy;
    let frame = u32(s.extra.x);
    let pixel = floor(frag);
    let turn = frame & 3u;
    let sun_pixel = floor(pixel / 2.0) * 2.0 + vec2<f32>(f32(turn & 1u), f32(turn >> 1u));
    let sky_turn = (frame * 7u) & 15u;
    let sky_pixel = floor(pixel / 4.0) * 4.0 + vec2<f32>(f32(sky_turn & 3u), f32(sky_turn >> 2u));
    let light = 4.0 * sun(world((sun_pixel + 0.5) / size)) + 0.6 * sky(world((sky_pixel + 0.5) / size));
    return albedo(world(frag / size)) * light * s.params.x;
}

struct Out {
    @location(0) colour: vec4<f32>,
    // helio-pass-tsr MOTION_VECTORS: the bar's, when it writes them.
    @location(1) motion: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

@fragment fn fs(@builtin(position) frag: vec4<f32>) -> Out {
    let n = i32(s.params.w);
    var out: Out;
    let centre = world(frag.xy / s.frame_size.xy);
    // The foliage's ground recedes from the bottom of the frame to the top,
    // steeply, as ground seen at a grazing angle does.
    let ground = select(s.extra.y, mix(s.extra.w, s.extra.y, frag.y / s.frame_size.y), s.params.z > 1.5);
    out.depth = select(ground, s.extra.z, s.params.z > 1.5 && foreground(centre));
    out.motion = vec4<f32>(0.0);
    if s.params.z > 1.5 && on_bar(centre) {
        // Depth is linear in view z (orthographic): the plane 3 m away is at
        // extra.y, the leaf blocks 2 m away at extra.z, the bar 1 m away.
        out.depth = 2.0 * s.extra.z - s.extra.y;
        if s.motion.y > 0.0 {
            // Where this point of the bar was last frame, in last frame's
            // (jittered) raster.
            let previous = s.prev_view_proj * vec4<f32>(centre.x - s.motion.x, centre.y, 0.0, 1.0);
            let ndc = previous.xy / previous.w;
            let raster = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * s.frame_size.xy;
            out.motion = vec4<f32>(frag.xy - raster, 0.0, 1.0);
        }
    }
    if n <= 1 {
        // One sample at the jittered pixel centre, as rasterization takes it.
        if s.params.z > 1.5 && on_bar(centre) {
            out.colour = vec4<f32>(vec3<f32>(EMISSIVE_BAR), 1.0);
        } else if s.params.z > 1.5 {
            out.colour = vec4<f32>(stochastic_foliage(frag.xy), 1.0);
        } else {
            out.colour = vec4<f32>(scene(centre), 1.0);
        }
        return out;
    }
    // Ground truth: the pixel's box footprint, n x n samples.
    var sum = vec3<f32>(0.0);
    for (var y = 0; y < n; y++) {
        for (var x = 0; x < n; x++) {
            let at = floor(frag.xy) + (vec2<f32>(f32(x), f32(y)) + 0.5) / f32(n);
            sum += scene(world(at / s.frame_size.xy));
        }
    }
    out.colour = vec4<f32>(sum / f32(n * n), 1.0);
    return out;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SceneUniform {
    inv_view_proj: [f32; 16],
    params: [f32; 4],
    frame_size: [f32; 4],
    extra: [f32; 4],
    prev_view_proj: [f32; 16],
    motion: [f32; 4],
}

#[derive(Clone, Copy)]
struct Content {
    brightness: f32,
    /// `None`: the detail scene. `Some(x)`: a bar with its left edge at world x.
    bar: Option<f32>,
    /// The bar's width in output pixels.
    bar_width: f32,
    /// Camera position along world x.
    pan: f32,
    /// The foliage scene instead: high-contrast texel-sized texture, leaf
    /// blocks in front of it, and stochastic per-frame-sampled lighting
    /// (with a bar: an emissive block in front of the leaf blocks).
    foliage: bool,
    /// Mark the foliage's stochastic lighting in the pass's stochastic mask.
    masked: bool,
    /// The bar writes motion vectors (`MOTION_VECTORS`) for its own motion.
    motion_vectors: bool,
}

impl Content {
    const DETAIL: Self = Self {
        brightness: 1.0,
        bar: None,
        bar_width: 0.0,
        pan: 0.0,
        foliage: false,
        masked: false,
        motion_vectors: false,
    };
    const FOLIAGE: Self = Self {
        foliage: true,
        ..Self::DETAIL
    };
    const FOLIAGE_MASKED: Self = Self {
        masked: true,
        ..Self::FOLIAGE
    };
}

struct Input {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    camera: wgpu::Buffer,
    camera_data: helio_core::GpuCameraUniforms,
    frame: u64,
    projection: helio_core::SceneBufferProjection,
}

impl helio_core::SceneInput for Input {
    fn world_origin(&self) -> glam::DVec3 {
        glam::DVec3::ZERO
    }
    fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }
    fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }
    fn frame_count(&self) -> u64 {
        self.frame
    }
    fn camera(&self) -> &wgpu::Buffer {
        &self.camera
    }
    fn camera_data(&self) -> &helio_core::GpuCameraUniforms {
        &self.camera_data
    }
    fn camera_generation(&self) -> u64 {
        0
    }
    fn scene_buffers(&self) -> &helio_core::SceneBufferProjection {
        &self.projection
    }
}

/// Fixtures create their own devices; several waiting on the GPU at once
/// deadlock now and then on NVIDIA Vulkan (4 of the quality tests hung in 2 of
/// 6 parallel runs), so tests run one at a time, as helio-pass-flare's do. A
/// test may hold several fixtures: the first takes the lock for its thread.
static GPU_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

thread_local! {
    static SERIAL_HELD: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

struct Serial(Option<std::sync::MutexGuard<'static, ()>>);

impl Serial {
    fn take() -> Self {
        let held = SERIAL_HELD.with(|n| n.replace(n.get() + 1));
        Self((held == 0).then(|| {
            GPU_SERIAL
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
        }))
    }
}

impl Drop for Serial {
    fn drop(&mut self) {
        SERIAL_HELD.with(|n| n.set(n.get() - 1));
    }
}

struct Fixture {
    /// Held for the fixture's life (`GPU_SERIAL`).
    _serial: Serial,
    input: Input,
    graph: helio_core::RenderGraph,
    internal: [u32; 2],
    output: [u32; 2],
    color: wgpu::Texture,
    color_view: wgpu::TextureView,
    /// The stochastic mask the fixture publishes (`STOCHASTIC_MASK`).
    mask_view: wgpu::TextureView,
    /// The motion vectors it publishes (`MOTION_VECTORS`), and the reference
    /// draw's (unused).
    motion_view: wgpu::TextureView,
    reference_motion_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    target_view: wgpu::TextureView,
    reference: wgpu::Texture,
    reference_view: wgpu::TextureView,
    reference_depth_view: wgpu::TextureView,
    /// Depth of the background plane, of the foliage's nearer blocks and of
    /// the far edge of the foliage's receding ground.
    plane_depths: [f32; 3],
    scene_pipeline: wgpu::RenderPipeline,
    scene_uniform: wgpu::Buffer,
    scene_bind: wgpu::BindGroup,
    previous_view_proj: Option<Mat4>,
    /// The bar's world x last frame.
    previous_bar: Option<f32>,
    /// The bar's shift since last frame as its motion vectors say, for the
    /// next frame only (a writer's bad vectors).
    bar_shift_override: Option<f32>,
}

/// One output pixel in world units: the camera spans world [-1, 1].
fn world_per_pixel(output: [u32; 2]) -> f32 {
    2.0 / output[0] as f32
}

fn luma(rgba: &[f32; 4]) -> f32 {
    0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
}

impl Fixture {
    fn new(quality: TsrQuality, output: [u32; 2]) -> Option<Self> {
        let serial = Serial::take();
        let instance = wgpu::Instance::new(helio_core::wgpu_instance_descriptor().with_env());
        let Ok(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else {
            eprintln!("SKIP: no GPU adapter for TSR quality metrics");
            return None;
        };
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default())).unwrap();
        let device = Arc::new(device);
        let queue = Arc::new(queue);
        let scale = quality.render_scale();
        let internal = output.map(|v| ((v as f32 * scale).ceil() as u32).max(1));
        let texture = |size: [u32; 2], format, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("TSR quality fixture"),
                size: wgpu::Extent3d {
                    width: size[0],
                    height: size[1],
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let color_usage = wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC;
        let color = texture(internal, wgpu::TextureFormat::Rgba16Float, color_usage);
        let mask = texture(internal, wgpu::TextureFormat::R8Unorm, color_usage);
        let motion = texture(internal, wgpu::TextureFormat::Rgba16Float, color_usage);
        let reference_motion = texture(output, wgpu::TextureFormat::Rgba16Float, color_usage);
        let reference = texture(output, wgpu::TextureFormat::Rgba16Float, color_usage);
        let target = texture(output, wgpu::TextureFormat::Rgba16Float, color_usage);
        let depth = texture(
            internal,
            wgpu::TextureFormat::Depth32Float,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        let depth_view = depth.create_view(&Default::default());
        let reference_depth = texture(
            output,
            wgpu::TextureFormat::Depth32Float,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        // A plane 3 m in front of the camera fills the frame; the foliage's
        // leaf blocks stand 1 m in front of it.
        let proj = helio_core::depth::orthographic(-1.0, 1.0, -1.0, 1.0, 0.1, 10.0);
        let plane_depths = [-3.0, -2.0, -9.0].map(|z| {
            let p = proj * glam::Vec4::new(0.0, 0.0, z, 1.0);
            p.z / p.w
        });

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("TSR fixture scene"),
            source: wgpu::ShaderSource::Wgsl(SCENE_WGSL.into()),
        });
        let scene_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("TSR fixture scene"),
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
                compilation_options: Default::default(),
                targets: &[
                    Some(wgpu::TextureFormat::Rgba16Float.into()),
                    Some(wgpu::TextureFormat::Rgba16Float.into()),
                ],
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
        let scene_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TSR fixture scene"),
            size: std::mem::size_of::<SceneUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let scene_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &scene_pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: scene_uniform.as_entire_binding(),
            }],
        });
        let camera = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TSR fixture cameras"),
            size: 2 * std::mem::size_of::<helio_core::GpuCameraUniforms>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut graph = helio_core::RenderGraph::new(&device, &queue);
        graph.add_pass(Box::new(
            TsrPass::new(
                &device,
                internal[0],
                internal[1],
                output[0],
                output[1],
                wgpu::TextureFormat::Rgba16Float,
                quality,
            )
            .with_intermediate_output()
            .with_stochastic_mask()
            .with_motion_vectors(),
        ));
        graph.lock(internal[0], internal[1]);
        graph.set_delta_time(1.0 / 60.0);
        Some(Self {
            _serial: serial,
            input: Input {
                device: device.clone(),
                queue: queue.clone(),
                camera,
                camera_data: helio_core::GpuCameraUniforms::zeroed(),
                frame: 0,
                projection: helio_core::SceneBufferProjection::empty(),
            },
            graph,
            internal,
            output,
            color_view: color.create_view(&Default::default()),
            mask_view: mask.create_view(&Default::default()),
            motion_view: motion.create_view(&Default::default()),
            reference_motion_view: reference_motion.create_view(&Default::default()),
            color,
            depth_view,
            target_view: target.create_view(&Default::default()),
            reference_view: reference.create_view(&Default::default()),
            reference,
            reference_depth_view: reference_depth.create_view(&Default::default()),
            plane_depths,
            scene_pipeline,
            scene_uniform,
            scene_bind,
            previous_view_proj: None,
            previous_bar: None,
            bar_shift_override: None,
        })
    }

    fn device(&self) -> &wgpu::Device {
        &self.input.device
    }
    fn queue(&self) -> &wgpu::Queue {
        &self.input.queue
    }

    fn view(pan: f32) -> Mat4 {
        glam::camera::rh::view::look_at_mat4(
            Vec3::new(pan, 0.0, 3.0),
            Vec3::new(pan, 0.0, 0.0),
            Vec3::Y,
        )
    }

    /// Draws `content` into `view`, `depth` and `motion` (the bar's motion
    /// vectors, `bar_shift` world units since `prev_view_proj`'s frame).
    #[allow(clippy::too_many_arguments)]
    fn draw_scene(
        &self,
        view: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        motion: &wgpu::TextureView,
        inv_view_proj: Mat4,
        prev_view_proj: Mat4,
        bar_shift: f32,
        content: Content,
        size: [u32; 2],
        samples: f32,
    ) {
        let scene = if content.foliage {
            2.0
        } else if content.bar.is_some() {
            1.0
        } else {
            0.0
        };
        let uniform = SceneUniform {
            inv_view_proj: inv_view_proj.to_cols_array(),
            params: [
                content.brightness,
                content.bar.unwrap_or(0.0),
                scene,
                samples,
            ],
            frame_size: [
                size[0] as f32,
                size[1] as f32,
                world_per_pixel(self.output),
                content.bar_width,
            ],
            extra: [
                self.input.frame as f32,
                self.plane_depths[0],
                self.plane_depths[1],
                self.plane_depths[2],
            ],
            prev_view_proj: prev_view_proj.to_cols_array(),
            motion: [
                bar_shift,
                if content.motion_vectors { 1.0 } else { 0.0 },
                0.0,
                0.0,
            ],
        };
        self.queue()
            .write_buffer(&self.scene_uniform, 0, bytemuck::bytes_of(&uniform));
        let mut encoder = self.device().create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("TSR fixture scene"),
                color_attachments: &[view, motion].map(|view| {
                    Some(wgpu::RenderPassColorAttachment {
                        view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })
                }),
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&self.scene_pipeline);
            pass.set_bind_group(0, &self.scene_bind, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue().submit([encoder.finish()]);
    }

    /// Render one jittered frame of `content` and resolve it; returns the
    /// luminance of the pass's published output.
    fn frame(&mut self, content: Content) -> Vec<f32> {
        let frame = self.input.frame;
        // The renderer's own sequence, starting where its frame cursor does.
        let jitter = helio_core::temporal::r1_r2_jitter(frame + 1);
        let ndc = [
            jitter[0] * 2.0 / self.internal[0] as f32,
            jitter[1] * 2.0 / self.internal[1] as f32,
        ];
        let view = Self::view(content.pan);
        let proj = Mat4::from_translation(Vec3::new(ndc[0], ndc[1], 0.0))
            * helio_core::depth::orthographic(-1.0, 1.0, -1.0, 1.0, 0.1, 10.0);
        let view_proj = proj * view;
        let previous = self.previous_view_proj.unwrap_or(view_proj);
        self.previous_view_proj = Some(view_proj);
        let camera = helio_core::GpuCameraUniforms::new(
            view,
            proj,
            Vec3::new(content.pan, 0.0, 3.0),
            0.1,
            10.0,
            frame as u32,
            ndc,
            previous,
        );
        self.queue().write_buffer(
            &self.input.camera,
            0,
            bytemuck::cast_slice(&[camera, camera]),
        );
        self.input.camera_data = camera;
        let bar = content.bar.unwrap_or(0.0);
        let bar_shift = self
            .bar_shift_override
            .take()
            .unwrap_or(bar - self.previous_bar.unwrap_or(bar));
        self.previous_bar = Some(bar);
        self.draw_scene(
            &self.color_view,
            &self.depth_view,
            &self.motion_view,
            view_proj.inverse(),
            previous,
            bar_shift,
            content,
            self.internal,
            1.0,
        );

        let mut registry = helio_core::ResourceRegistry::empty();
        // The whole of the foliage's lighting is sampled stochastically.
        let marked = if content.foliage && content.masked {
            1.0
        } else {
            0.0
        };
        let mut encoder = self.device().create_command_encoder(&Default::default());
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("TSR fixture stochastic mask"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.mask_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: marked,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        self.queue().submit([encoder.finish()]);
        registry.write(
            helio_core::ResourceKey::new("pre_aa"),
            &self.color_view,
            "Fixture",
        );
        registry.write(
            helio_core::ResourceKey::new(STOCHASTIC_MASK),
            &self.mask_view,
            "Fixture",
        );
        registry.write(
            helio_core::ResourceKey::new(MOTION_VECTORS),
            &self.motion_view,
            "Fixture",
        );
        self.graph
            .execute_with_registry(
                &self.input,
                &self.target_view,
                &self.depth_view,
                &mut registry,
            )
            .unwrap();
        self.input.frame += 1;
        let output = &self.graph.find_pass::<TsrPass>().unwrap().output_texture;
        self.read(output).iter().map(luma).collect()
    }

    /// Box-filtered ground truth of `content` at output resolution.
    fn reference(&self, content: Content) -> Vec<f32> {
        let proj = helio_core::depth::orthographic(-1.0, 1.0, -1.0, 1.0, 0.1, 10.0);
        let view_proj = proj * Self::view(content.pan);
        self.draw_scene(
            &self.reference_view,
            &self.reference_depth_view,
            &self.reference_motion_view,
            view_proj.inverse(),
            view_proj,
            0.0,
            content,
            self.output,
            8.0,
        );
        self.read(&self.reference).iter().map(luma).collect()
    }

    fn read(&self, texture: &wgpu::Texture) -> Vec<[f32; 4]> {
        let size = texture.size();
        let stride = (size.width * 8).div_ceil(256) * 256;
        let buffer = self.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("TSR fixture readback"),
            size: (stride * size.height) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device().create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(stride),
                    rows_per_image: Some(size.height),
                },
            },
            size,
        );
        self.queue().submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        self.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        rx.recv().unwrap().unwrap();
        let data = buffer.slice(..).get_mapped_range().unwrap();
        let mut texels = Vec::with_capacity((size.width * size.height) as usize);
        for row in data.chunks(stride as usize) {
            for texel in row[..(size.width * 8) as usize].chunks(8) {
                let h: [u16; 4] = bytemuck::pod_read_unaligned(texel);
                texels.push(h.map(|v| half::f16::from_bits(v).to_f32()));
            }
        }
        texels
    }
}

fn mean_abs_difference(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| (a - b).abs()).sum::<f32>() / a.len() as f32
}

struct StillMetrics {
    /// Mean |dL| per frame over the whole image.
    flicker: f32,
    /// The same over pixels within a pixel of an edge or wire, where all of
    /// the flicker is.
    edge_flicker: f32,
    /// The jittered input's own mean |dL| per frame: what no history gives.
    input_flicker: f32,
    /// Mean |L - truth| of the time-averaged output: a pass cannot buy
    /// stability by blurring.
    error: f32,
    /// Display-referred, as a viewer (and voxel_craft's burst tool) sees
    /// it: mean |dT| per frame of the tonemapped luma T = L / (1 + L)...
    display_flicker: f32,
    /// ...and the fraction of pixels whose T changes by more than 2 %.
    changing: f32,
}

/// Display-referred luma.
fn display(l: f32) -> f32 {
    l / (1.0 + l)
}

/// Flicker of a converged still frame of `content`.
fn still_metrics(quality: TsrQuality, content: Content) -> Option<StillMetrics> {
    let mut fixture = Fixture::new(quality, [128, 128])?;
    let truth = fixture.reference(content);
    // Pixels whose 3x3 footprint holds any contrast.
    let edges: Vec<bool> = (0..truth.len())
        .map(|i| {
            let (x, y) = ((i % 128) as i32, (i / 128) as i32);
            let at = |dx: i32, dy: i32| {
                truth[((y + dy).clamp(0, 127) * 128 + (x + dx).clamp(0, 127)) as usize]
            };
            (-1..=1)
                .flat_map(|dy| (-1..=1).map(move |dx| (dx, dy)))
                .any(|(dx, dy)| (at(dx, dy) - truth[i]).abs() > 0.02)
        })
        .collect();
    let edge_count = edges.iter().filter(|e| **e).count() as f32;
    let mut previous: Option<(Vec<f32>, Vec<f32>)> = None;
    let mut metrics = StillMetrics {
        flicker: 0.0,
        edge_flicker: 0.0,
        input_flicker: 0.0,
        error: 0.0,
        display_flicker: 0.0,
        changing: 0.0,
    };
    let mut average = vec![0.0; truth.len()];
    const WARMUP: usize = 64;
    const MEASURED: usize = 64;
    for frame in 0..WARMUP + MEASURED {
        let output = fixture.frame(content);
        let input: Vec<f32> = fixture.read(&fixture.color).iter().map(luma).collect();
        if frame >= WARMUP {
            let (last_output, last_input) = previous.as_ref().unwrap();
            metrics.flicker += mean_abs_difference(&output, last_output) / MEASURED as f32;
            let changes: Vec<f32> = output
                .iter()
                .zip(last_output)
                .map(|(a, b)| (display(*a) - display(*b)).abs())
                .collect();
            metrics.display_flicker +=
                changes.iter().sum::<f32>() / changes.len() as f32 / MEASURED as f32;
            metrics.changing += changes.iter().filter(|c| **c > 0.02).count() as f32
                / changes.len() as f32
                / MEASURED as f32;
            metrics.input_flicker += mean_abs_difference(&input, last_input) / MEASURED as f32;
            let edge_change: f32 = output
                .iter()
                .zip(last_output)
                .zip(&edges)
                .filter(|(_, e)| **e)
                .map(|((a, b), _)| (a - b).abs())
                .sum();
            metrics.edge_flicker += edge_change / edge_count / MEASURED as f32;
            for (a, o) in average.iter_mut().zip(&output) {
                *a += o / MEASURED as f32;
            }
        }
        previous = Some((output, input));
    }
    metrics.error = mean_abs_difference(&average, &truth);
    Some(metrics)
}

/// Frames until a converged still image follows a light switching by
/// `factor` without stale history, or `None` if it has not within 64 frames,
/// with the stale fraction left at frame 64. The outer `None` is a missing GPU.
///
/// The reference is the same pass shown the new lighting all along, in
/// lockstep (same input from the switch on, same jitter every frame). Stale
/// history is the fraction of the old image still in the output: the
/// difference from the reference projected on the difference between the
/// old image and the reference. Re-converging the aliasing of sub-pixel
/// detail after dropping history is pixel-scale noise that projects to
/// almost nothing. Settled is under 2 % toward the old image: a negative
/// projection is not old light, and with stochastic lighting (whose sample
/// pattern is the same for the whole frame) two histories of different
/// lengths average its cycle differently, a few percent either way.
fn light_switch_frames(
    quality: TsrQuality,
    content: Content,
    factor: f32,
) -> Option<(Option<usize>, f32)> {
    let switched = Content {
        brightness: factor,
        ..content
    };
    let mut always = Fixture::new(quality, [128, 128])?;
    let mut fixture = Fixture::new(quality, [128, 128])?;
    let mut old = vec![0.0; 128 * 128];
    for frame in 0..64 {
        always.frame(switched);
        let output = fixture.frame(content);
        if frame >= 56 {
            for (o, v) in old.iter_mut().zip(&output) {
                *o += v / 8.0;
            }
        }
    }
    let mut settled = None;
    let mut stale = 0.0;
    for frame in 1..=64 {
        let reference = always.frame(switched);
        let output = fixture.frame(switched);
        let (mut along, mut norm) = (0.0, 0.0);
        for ((o, r), v) in output.iter().zip(&reference).zip(&old) {
            along += (o - r) * (v - r);
            norm += (v - r) * (v - r);
        }
        stale = along / norm;
        if settled.is_none() && stale < 0.02 {
            settled = Some(frame);
        }
    }
    Some((settled, stale))
}

/// A bar `width` output pixels wide sweeping right at `speed` output pixels
/// per frame, moved either
/// by itself (no motion vectors: only the history rectification can drop the
/// stale bar) or by the camera (reprojection carries it). Returns the mean
/// count, per frame, of trail pixels: wrong by more than 10 % of the bar's
/// contrast, and more than 1.5 px away from either of its edges; and the
/// mean |L - truth| over the frame, which also catches an edge smeared by
/// resampling.
fn trail_pixels(
    quality: TsrQuality,
    width: f32,
    speed: f32,
    camera_motion: bool,
) -> Option<(f32, f32)> {
    let mut fixture = Fixture::new(quality, [128, 128])?;
    let px = world_per_pixel(fixture.output);
    let mut trails = 0.0;
    let mut error = 0.0;
    const FRAMES: usize = 48;
    const MEASURED: usize = 16;
    for frame in 0..FRAMES {
        let travelled = frame as f32 * speed * px;
        // The bar starts near the left edge; camera motion moves the camera
        // left so the world-fixed bar travels right on screen.
        let content = if camera_motion {
            Content {
                bar: Some(-0.8),
                bar_width: width,
                pan: -travelled,
                ..Content::DETAIL
            }
        } else {
            Content {
                bar: Some(-0.8 + travelled),
                bar_width: width,
                ..Content::DETAIL
            }
        };
        let output = fixture.frame(content);
        if frame < FRAMES - MEASURED {
            continue;
        }
        let reference = fixture.reference(content);
        // Screen-space left edge of the bar, in output pixels.
        let left = (-0.8 + travelled + 1.0) / px;
        let right = left + width;
        let mut count = 0;
        for (i, (o, r)) in output.iter().zip(&reference).enumerate() {
            let x = (i % 128) as f32 + 0.5;
            if (x - left).abs() > 1.5 && (x - right).abs() > 1.5 && (o - r).abs() > 0.1 * 0.95 {
                count += 1;
            }
        }
        trails += count as f32 / MEASURED as f32;
        error += mean_abs_difference(&output, &reference) / MEASURED as f32;
    }
    Some((trails, error))
}

/// The fixture shader's `EMISSIVE_BAR`.
const EMISSIVE_BAR: f32 = 30.0;

/// An emissive bar `width` output pixels wide 1 m in front of the foliage,
/// moving right on screen at `speed` output pixels per frame, either by
/// itself with the camera still, or by parallax: the camera pans at half
/// that speed and the bar, nearer than anything, moves twice as fast as the
/// foliage (the orthographic fixture has no parallax of its own, so the bar
/// moves in the world by the pan; without motion vectors reprojection, which
/// follows the camera, then misses it as it missed glowstone flown past).
/// With `motion_vectors` the bar writes them. The foliage's thin features
/// lock pixels (step 7), and a lock must not keep the bar's light once the
/// foliage shows there again. Returns the mean count, per frame, of pixels
/// more than 1.5 px from the bar brighter by over a tenth of the bar than the
/// same pass shown the foliage alone, in lockstep (same camera, input and
/// jitter but for the bar). Under that, a fresh pixel's stochastic lighting
/// differs from the converged one's by up to about 2.
fn emissive_trail_pixels(
    quality: TsrQuality,
    width: f32,
    speed: f32,
    camera_motion: bool,
    motion_vectors: bool,
) -> Option<f32> {
    let mut fixture = Fixture::new(quality, [128, 128])?;
    let mut plain = Fixture::new(quality, [128, 128])?;
    let px = world_per_pixel(fixture.output);
    let mut trails = 0.0;
    const FRAMES: usize = 48;
    const MEASURED: usize = 16;
    for frame in 0..FRAMES {
        let (bar, pan) = if camera_motion {
            let pan = frame as f32 * speed * 0.5 * px;
            (-0.8 + pan, -pan)
        } else {
            (-0.8 + frame as f32 * speed * px, 0.0)
        };
        let foliage = Content {
            pan,
            ..Content::FOLIAGE
        };
        let output = fixture.frame(Content {
            bar: Some(bar),
            bar_width: width,
            motion_vectors,
            ..foliage
        });
        let without = plain.frame(foliage);
        if frame < FRAMES - MEASURED {
            continue;
        }
        // Screen-space left edge of the bar, in output pixels.
        let left = (bar - pan + 1.0) / px;
        let right = left + width;
        let count = output
            .iter()
            .zip(&without)
            .enumerate()
            .filter(|(i, (o, w))| {
                let x = (i % 128) as f32 + 0.5;
                (x < left - 1.5 || x > right + 1.5) && *o - *w > 0.1 * EMISSIVE_BAR
            })
            .count();
        trails += count as f32 / MEASURED as f32;
    }
    Some(trails)
}

// Thresholds, from the metrics on an RTX 3060 (Vulkan). The still-pixel
// resolve that preceded this one (1.75x box and a 1/64 blend for pixels
// moving under a pixel, no sharpening on them) and the first rebuild of this
// resolve (40b19cab: +/-1 sigma box with weight loss, a single ridge texel's
// luma releasing locks, disocclusion on the dilated depth alone), which
// shimmered on real foliage and stochastic shadows:
//
//                                still-pixel     40b19cab        this resolve
//   wires    flicker N / Q       0.00065 / -     0.00065/0.00067 0.00053/0.00051
//            error   N / Q       0.0249 / -      0.0210/0.0212   0.0209/0.0210
//   foliage  >2 % changing       0.00 / 0.03 /   26 / 49 /       0.23 / 0.17 /
//   (N / Q / Balanced)           1.26 %          55 %            0.19 %
//            display |dT|        0.0010/0.0015/  0.029/0.061/    0.0014/0.0013/
//                                0.0027          0.074           0.0015
//            error vs truth      0.153/0.206/    0.133/0.165/    0.141/0.168/
//                                0.223           0.179           0.180
//   light    detail x0.2 / x5    never (7.5 %    4-5 / 1         2-3 / 2
//   (frames, N and Q)            stale at 64)
//            foliage x0.2 / x5   -               7 / never       3 / 3
//   trails   6 px bar, object    96 px           0 px            0 px
//   camera   6 px bar, error N/Q -               0.0026/0.0034   0.0031/0.0039
//
// Foliage error is dominated by the stochastic sun's 2x2 blocks, which no
// accumulation can undo (the truth samples the sun exactly), so it bounds
// blur rather than measuring sharpness.

#[test]
fn still_jittered_wires_do_not_flicker() {
    for quality in [TsrQuality::Native, TsrQuality::Quality] {
        let Some(m) = still_metrics(quality, Content::DETAIL) else {
            return;
        };
        eprintln!("TSR {quality:?}: still flicker {:.5} (edges {:.5}, input {:.5}) mean |dL|/frame, error vs truth {:.5}",
            m.flicker, m.edge_flicker, m.input_flicker, m.error);
        assert!(
            m.flicker < 0.001,
            "{quality:?}: converged still image flickers: {:.5} mean |dL| per frame",
            m.flicker
        );
        assert!(
            m.error < 0.023,
            "{quality:?}: converged image strays from ground truth (blur?): {:.5}",
            m.error
        );
    }
}

/// Texel-sized, high-contrast texture under jitter, lit by stochastic
/// per-frame samples, with depth edges and ground at a grazing angle: what
/// the first rebuild of this resolve shimmered on in voxel_craft (26-55 % of
/// pixels changing by over 2 % a frame here). It must stay as still as the
/// still-pixel resolve was, without its blur; the stochastic mask may only
/// help.
#[test]
fn still_foliage_does_not_shimmer() {
    for quality in [
        TsrQuality::Native,
        TsrQuality::Quality,
        TsrQuality::Balanced,
    ] {
        let Some(plain) = still_metrics(quality, Content::FOLIAGE) else {
            return;
        };
        let Some(masked) = still_metrics(quality, Content::FOLIAGE_MASKED) else {
            return;
        };
        for (m, mask) in [(&plain, "unmasked"), (&masked, "masked")] {
            eprintln!("TSR {quality:?} {mask}: foliage flicker {:.5} (input {:.5}) mean |dL|/frame, display {:.5}, \
                {:.2} % changing >2 %, error vs truth {:.5}",
                m.flicker, m.input_flicker, m.display_flicker, m.changing * 100.0, m.error);
            assert!(
                m.changing < 0.005,
                "{quality:?} {mask}: {:.2} % of foliage pixels change by over 2 % a frame",
                m.changing * 100.0
            );
            assert!(
                m.display_flicker < 0.002,
                "{quality:?} {mask}: foliage shimmers: {:.5} mean |dT| per frame",
                m.display_flicker
            );
            assert!(
                m.error < 0.19,
                "{quality:?} {mask}: foliage strays from ground truth (blur?): {:.5}",
                m.error
            );
        }
        assert!(
            masked.changing <= plain.changing + 0.0005
                && masked.display_flicker <= plain.display_flicker * 1.05,
            "{quality:?}: the stochastic mask made foliage less stable"
        );
    }
}

#[test]
fn light_switch_without_motion_does_not_ghost() {
    for content in [Content::DETAIL, Content::FOLIAGE, Content::FOLIAGE_MASKED] {
        for quality in [TsrQuality::Native, TsrQuality::Quality] {
            for factor in [0.2, 5.0] {
                let Some((frames, stale)) = light_switch_frames(quality, content, factor) else {
                    return;
                };
                let name = match (content.foliage, content.masked) {
                    (false, _) => "detail",
                    (true, false) => "foliage",
                    (true, true) => "masked foliage",
                };
                eprintln!("TSR {quality:?} {name}: light x{factor} settles in {frames:?} frames ({:.1} % stale after 64)",
                    stale * 100.0);
                assert!(frames.is_some_and(|f| f <= 8),
                    "{quality:?} {name}: light x{factor} left stale history for {frames:?} frames ({:.1} % at 64)",
                    stale * 100.0);
            }
        }
    }
}

#[test]
fn moving_edges_leave_no_trails() {
    for quality in [TsrQuality::Native, TsrQuality::Quality] {
        for (width, camera_motion) in [(6.0, false), (6.0, true), (0.5, false), (0.5, true)] {
            let Some((trails, error)) = trail_pixels(quality, width, 1.5, camera_motion) else {
                return;
            };
            let motion = if camera_motion { "camera" } else { "object" };
            eprintln!("TSR {quality:?}: {width} px {motion} motion trail {trails:.1} px/frame, error vs truth {error:.5}");
            // A 128-row image: 4 pixels is a stray pixel, not a trail.
            assert!(
                trails <= 4.0,
                "{quality:?}: {width} px bar under {motion} motion trails {trails:.1} px per frame"
            );
            assert!(
                error < 0.01,
                "{quality:?}: {width} px bar under {motion} motion: error {error:.5} vs truth"
            );
        }
    }
}

/// The ghosting seen flying past glowstone in voxel_craft: locks on the
/// foliage's thin features skipped the clip for history that had passed the
/// disocclusion test only by its depth slope allowance, and kept the
/// emissive block's light in streaks behind it. With motion vectors, history
/// the bar uncovered is found by the reconstruction; without, under camera
/// motion, locks yield to motion. (Object motion without motion vectors, a
/// still camera, is the one case left to the box and the change detectors.)
#[test]
fn emissive_object_over_foliage_leaves_no_trail() {
    for quality in [TsrQuality::Native, TsrQuality::Balanced] {
        for (speed, camera_motion, motion_vectors) in [
            (1.0, false, true),
            (2.0, false, true),
            (2.0, true, true),
            (4.0, true, true),
            (2.0, true, false),
        ] {
            let Some(trails) =
                emissive_trail_pixels(quality, 6.0, speed, camera_motion, motion_vectors)
            else {
                return;
            };
            let motion = if camera_motion { "parallax" } else { "object" };
            let vectors = if motion_vectors { "with" } else { "without" };
            eprintln!("TSR {quality:?}: emissive bar over foliage, {motion} motion at {speed} px/frame {vectors} motion \
                vectors, trails {trails:.1} px/frame");
            assert!(
                trails <= 4.0,
                "{quality:?}: emissive bar, {motion} motion at {speed} px/frame {vectors} motion \
                vectors, trails {trails:.1} px per frame"
            );
        }
    }
}

/// A frame whose motion vectors are NaN (a writer dividing by a zero w)
/// reads as a frame without them: through the history position they would
/// reach the accumulated state, and a NaN there kept a pixel's output NaN,
/// black, for good. The run must match one whose bar wrote no vectors that
/// frame, and stay finite.
#[test]
fn non_finite_motion_vectors_read_as_none() {
    let Some(mut fixture) = Fixture::new(TsrQuality::Balanced, [128, 128]) else {
        return;
    };
    let Some(mut reference) = Fixture::new(TsrQuality::Balanced, [128, 128]) else {
        return;
    };
    let px = world_per_pixel(fixture.output);
    let content = |frame: usize| Content {
        bar: Some(-0.8 + frame as f32 * px),
        bar_width: 6.0,
        motion_vectors: true,
        ..Content::FOLIAGE
    };
    let (mut output, mut expected) = (Vec::new(), Vec::new());
    for frame in 0..32 {
        if frame == 16 {
            fixture.bar_shift_override = Some(f32::NAN);
        }
        output = fixture.frame(content(frame));
        expected = reference.frame(Content {
            motion_vectors: frame != 16,
            ..content(frame)
        });
    }
    let non_finite = output.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(non_finite, 0, "{non_finite} pixels non-finite");
    let difference = mean_abs_difference(&output, &expected);
    assert!(
        difference < 1.0e-4,
        "mean difference from a run without the frame's vectors {difference}"
    );
}

/// A frame whose input holds NaN, then one holding infinities (a lighting
/// bug upstream for a frame), must not stay in history: a NaN survives
/// every blend after it. After clean frames the output is finite and
/// matches a run that never saw them.
#[test]
fn a_non_finite_frame_does_not_stick() {
    let Some(mut fixture) = Fixture::new(TsrQuality::Balanced, [128, 128]) else {
        return;
    };
    let Some(mut clean) = Fixture::new(TsrQuality::Balanced, [128, 128]) else {
        return;
    };
    for _ in 0..16 {
        fixture.frame(Content::DETAIL);
        clean.frame(Content::DETAIL);
    }
    fixture.frame(Content {
        brightness: f32::NAN,
        ..Content::DETAIL
    });
    clean.frame(Content::DETAIL);
    fixture.frame(Content {
        brightness: f32::INFINITY,
        ..Content::DETAIL
    });
    clean.frame(Content::DETAIL);
    let (mut output, mut expected) = (Vec::new(), Vec::new());
    for _ in 0..48 {
        output = fixture.frame(Content::DETAIL);
        expected = clean.frame(Content::DETAIL);
    }
    let non_finite = output.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(non_finite, 0, "{non_finite} pixels still non-finite");
    let difference = mean_abs_difference(&output, &expected);
    assert!(
        difference < 0.01,
        "mean difference from a clean run {difference}"
    );
}
