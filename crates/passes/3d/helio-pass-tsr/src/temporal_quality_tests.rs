//! Image-quality metrics of the whole pass, measured on the GPU through a real
//! render graph: flicker of a still, jittered scene with sub-pixel wires,
//! response to a light switching without anything moving, and trails behind
//! moving edges (object motion without motion vectors, and camera motion).
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
    // brightness, bar left edge (world x), scene id (0 detail, 1 bar), supersampling
    params: vec4<f32>,
    // target size in pixels, world units per output pixel, bar width in output pixels
    frame_size: vec4<f32>,
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

fn scene(p: vec2<f32>) -> vec3<f32> {
    let px = s.frame_size.z;
    var c = vec3<f32>(0.05);
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
    } else if p.x >= s.params.y && p.x < s.params.y + s.frame_size.w * px {
        c = vec3<f32>(1.0);
    }
    return c * s.params.x;
}

fn world(uv: vec2<f32>) -> vec2<f32> {
    let w = s.inv_view_proj * vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.5, 1.0);
    return w.xy / w.w;
}

@fragment fn fs(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let n = i32(s.params.w);
    if n <= 1 {
        // One sample at the jittered pixel centre, as rasterization takes it.
        return vec4<f32>(scene(world(frag.xy / s.frame_size.xy)), 1.0);
    }
    // Ground truth: the pixel's box footprint, n x n samples.
    var sum = vec3<f32>(0.0);
    for (var y = 0; y < n; y++) {
        for (var x = 0; x < n; x++) {
            let at = floor(frag.xy) + (vec2<f32>(f32(x), f32(y)) + 0.5) / f32(n);
            sum += scene(world(at / s.frame_size.xy));
        }
    }
    return vec4<f32>(sum / f32(n * n), 1.0);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SceneUniform {
    inv_view_proj: [f32; 16],
    params: [f32; 4],
    frame_size: [f32; 4],
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
}

impl Content {
    const DETAIL: Self = Self { brightness: 1.0, bar: None, bar_width: 0.0, pan: 0.0 };
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
    fn device(&self) -> &Arc<wgpu::Device> { &self.device }
    fn queue(&self) -> &Arc<wgpu::Queue> { &self.queue }
    fn frame_count(&self) -> u64 { self.frame }
    fn camera(&self) -> &wgpu::Buffer { &self.camera }
    fn camera_data(&self) -> &helio_core::GpuCameraUniforms { &self.camera_data }
    fn camera_generation(&self) -> u64 { 0 }
    fn scene_buffers(&self) -> &helio_core::SceneBufferProjection { &self.projection }
}

struct Fixture {
    input: Input,
    graph: helio_core::RenderGraph,
    internal: [u32; 2],
    output: [u32; 2],
    color: wgpu::Texture,
    color_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    target_view: wgpu::TextureView,
    reference: wgpu::Texture,
    reference_view: wgpu::TextureView,
    scene_pipeline: wgpu::RenderPipeline,
    scene_uniform: wgpu::Buffer,
    scene_bind: wgpu::BindGroup,
    previous_view_proj: Option<Mat4>,
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
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
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
                size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
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
        let reference = texture(output, wgpu::TextureFormat::Rgba16Float, color_usage);
        let target = texture(output, wgpu::TextureFormat::Rgba16Float, color_usage);
        let depth = texture(
            internal,
            wgpu::TextureFormat::Depth32Float,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        let depth_view = depth.create_view(&Default::default());
        // A plane 3 m in front of the camera fills the frame.
        let proj = helio_core::depth::orthographic(-1.0, 1.0, -1.0, 1.0, 0.1, 10.0);
        let plane = proj * glam::Vec4::new(0.0, 0.0, -3.0, 1.0);
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("TSR fixture depth"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(plane.z / plane.w),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        queue.submit([encoder.finish()]);

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
                targets: &[Some(wgpu::TextureFormat::Rgba16Float.into())],
            }),
            primitive: Default::default(),
            depth_stencil: None,
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
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: scene_uniform.as_entire_binding() }],
        });
        let camera = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TSR fixture cameras"),
            size: 2 * std::mem::size_of::<helio_core::GpuCameraUniforms>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut graph = helio_core::RenderGraph::new(&device, &queue);
        graph.add_pass(Box::new(
            TsrPass::new(&device, internal[0], internal[1], output[0], output[1],
                wgpu::TextureFormat::Rgba16Float, quality)
            .with_intermediate_output(),
        ));
        graph.lock(internal[0], internal[1]);
        graph.set_delta_time(1.0 / 60.0);
        Some(Self {
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
            color,
            depth_view,
            target_view: target.create_view(&Default::default()),
            reference_view: reference.create_view(&Default::default()),
            reference,
            scene_pipeline,
            scene_uniform,
            scene_bind,
            previous_view_proj: None,
        })
    }

    fn device(&self) -> &wgpu::Device { &self.input.device }
    fn queue(&self) -> &wgpu::Queue { &self.input.queue }

    fn view(pan: f32) -> Mat4 {
        glam::camera::rh::view::look_at_mat4(Vec3::new(pan, 0.0, 3.0), Vec3::new(pan, 0.0, 0.0), Vec3::Y)
    }

    fn draw_scene(&self, view: &wgpu::TextureView, inv_view_proj: Mat4, content: Content, size: [u32; 2], samples: f32) {
        let uniform = SceneUniform {
            inv_view_proj: inv_view_proj.to_cols_array(),
            params: [content.brightness, content.bar.unwrap_or(0.0), content.bar.map_or(0.0, |_| 1.0), samples],
            frame_size: [size[0] as f32, size[1] as f32, world_per_pixel(self.output), content.bar_width],
        };
        self.queue().write_buffer(&self.scene_uniform, 0, bytemuck::bytes_of(&uniform));
        let mut encoder = self.device().create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("TSR fixture scene"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
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
        let ndc = [jitter[0] * 2.0 / self.internal[0] as f32, jitter[1] * 2.0 / self.internal[1] as f32];
        let view = Self::view(content.pan);
        let proj = Mat4::from_translation(Vec3::new(ndc[0], ndc[1], 0.0))
            * helio_core::depth::orthographic(-1.0, 1.0, -1.0, 1.0, 0.1, 10.0);
        let view_proj = proj * view;
        let previous = self.previous_view_proj.unwrap_or(view_proj);
        self.previous_view_proj = Some(view_proj);
        let camera = helio_core::GpuCameraUniforms::new(
            view, proj, Vec3::new(content.pan, 0.0, 3.0), 0.1, 10.0, frame as u32, ndc, previous,
        );
        self.queue().write_buffer(&self.input.camera, 0, bytemuck::cast_slice(&[camera, camera]));
        self.input.camera_data = camera;
        self.draw_scene(&self.color_view, view_proj.inverse(), content, self.internal, 1.0);

        let mut registry = helio_core::ResourceRegistry::empty();
        registry.write(helio_core::ResourceKey::new("pre_aa"), &self.color_view, "Fixture");
        self.graph
            .execute_with_registry(&self.input, &self.target_view, &self.depth_view, &mut registry)
            .unwrap();
        self.input.frame += 1;
        let output = &self.graph.find_pass::<TsrPass>().unwrap().output_texture;
        self.read(output).iter().map(luma).collect()
    }

    /// Box-filtered ground truth of `content` at output resolution.
    fn reference(&self, content: Content) -> Vec<f32> {
        let proj = helio_core::depth::orthographic(-1.0, 1.0, -1.0, 1.0, 0.1, 10.0);
        self.draw_scene(&self.reference_view, (proj * Self::view(content.pan)).inverse(), content, self.output, 8.0);
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
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(stride), rows_per_image: Some(size.height) },
            },
            size,
        );
        self.queue().submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        self.device().poll(wgpu::PollType::wait_indefinitely()).unwrap();
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
}

/// Flicker of a converged still frame of the detail scene.
fn still_metrics(quality: TsrQuality) -> Option<StillMetrics> {
    let mut fixture = Fixture::new(quality, [128, 128])?;
    let truth = fixture.reference(Content::DETAIL);
    // Pixels whose 3x3 footprint holds any contrast.
    let edges: Vec<bool> = (0..truth.len())
        .map(|i| {
            let (x, y) = ((i % 128) as i32, (i / 128) as i32);
            let at = |dx: i32, dy: i32| truth[((y + dy).clamp(0, 127) * 128 + (x + dx).clamp(0, 127)) as usize];
            (-1..=1).flat_map(|dy| (-1..=1).map(move |dx| (dx, dy))).any(|(dx, dy)| (at(dx, dy) - truth[i]).abs() > 0.02)
        })
        .collect();
    let edge_count = edges.iter().filter(|e| **e).count() as f32;
    let mut previous: Option<(Vec<f32>, Vec<f32>)> = None;
    let mut metrics = StillMetrics { flicker: 0.0, edge_flicker: 0.0, input_flicker: 0.0, error: 0.0 };
    let mut average = vec![0.0; truth.len()];
    const WARMUP: usize = 64;
    const MEASURED: usize = 64;
    for frame in 0..WARMUP + MEASURED {
        let output = fixture.frame(Content::DETAIL);
        let input: Vec<f32> = fixture.read(&fixture.color).iter().map(luma).collect();
        if frame >= WARMUP {
            let (last_output, last_input) = previous.as_ref().unwrap();
            metrics.flicker += mean_abs_difference(&output, last_output) / MEASURED as f32;
            metrics.input_flicker += mean_abs_difference(&input, last_input) / MEASURED as f32;
            let edge_change: f32 = output.iter().zip(last_output).zip(&edges)
                .filter(|(_, e)| **e).map(|((a, b), _)| (a - b).abs()).sum();
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
/// almost nothing. Settled is within 2 %.
fn light_switch_frames(quality: TsrQuality, factor: f32) -> Option<(Option<usize>, f32)> {
    let switched = Content { brightness: factor, ..Content::DETAIL };
    let mut always = Fixture::new(quality, [128, 128])?;
    let mut fixture = Fixture::new(quality, [128, 128])?;
    let mut old = vec![0.0; 128 * 128];
    for frame in 0..64 {
        always.frame(switched);
        let output = fixture.frame(Content::DETAIL);
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
        if settled.is_none() && stale.abs() < 0.02 {
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
fn trail_pixels(quality: TsrQuality, width: f32, speed: f32, camera_motion: bool) -> Option<(f32, f32)> {
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
            Content { brightness: 1.0, bar: Some(-0.8), bar_width: width, pan: -travelled }
        } else {
            Content { brightness: 1.0, bar: Some(-0.8 + travelled), bar_width: width, pan: 0.0 }
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

// Thresholds, from the metrics on an RTX 3060 (Vulkan). Main's TSR (still
// AABB clamp, fixed blend floor) and the still-pixel variant that preceded
// this resolve (1.75x box and 1/64 blend for pixels moving under a pixel):
//
//                          main             still-pixel      this resolve
//   flicker  Native        0.00172          0.00065          0.00065
//            Quality       0.00184          0.00070          0.00067
//   error    Native        0.0204           0.0249           0.0210
//            Quality       0.0267           0.0317           0.0212
//   light    x0.2 / x5     35 / 25 frames   never (7-16 %    4 / 3 frames
//   (Native, Quality)      39 / 27          stale at 64)     5 / 5
//   trails   6 px bar      96 / 288 px      96 / 288 px      0 / 0 px
//   (object, Native / Q)
//
// Flicker must stay at the still-pixel variant's level without its blur or
// its ghosts; a light must settle within a few frames; nothing may trail.

#[test]
fn still_jittered_wires_do_not_flicker() {
    for quality in [TsrQuality::Native, TsrQuality::Quality] {
        let Some(m) = still_metrics(quality) else { return };
        eprintln!("TSR {quality:?}: still flicker {:.5} (edges {:.5}, input {:.5}) mean |dL|/frame, error vs truth {:.5}",
            m.flicker, m.edge_flicker, m.input_flicker, m.error);
        assert!(m.flicker < 0.001, "{quality:?}: converged still image flickers: {:.5} mean |dL| per frame", m.flicker);
        assert!(m.error < 0.023, "{quality:?}: converged image strays from ground truth (blur?): {:.5}", m.error);
    }
}

#[test]
fn light_switch_without_motion_does_not_ghost() {
    for quality in [TsrQuality::Native, TsrQuality::Quality] {
        for factor in [0.2, 5.0] {
            let Some((frames, stale)) = light_switch_frames(quality, factor) else { return };
            eprintln!("TSR {quality:?}: light x{factor} settles in {frames:?} frames ({:.1} % stale after 64)", stale * 100.0);
            assert!(frames.is_some_and(|f| f <= 8),
                "{quality:?}: light x{factor} left stale history for {frames:?} frames ({:.1} % at 64)", stale * 100.0);
        }
    }
}

#[test]
fn moving_edges_leave_no_trails() {
    for quality in [TsrQuality::Native, TsrQuality::Quality] {
        for (width, camera_motion) in [(6.0, false), (6.0, true), (0.5, false), (0.5, true)] {
            let Some((trails, error)) = trail_pixels(quality, width, 1.5, camera_motion) else { return };
            let motion = if camera_motion { "camera" } else { "object" };
            eprintln!("TSR {quality:?}: {width} px {motion} motion trail {trails:.1} px/frame, error vs truth {error:.5}");
            // A 128-row image: 4 pixels is a stray pixel, not a trail.
            assert!(trails <= 4.0, "{quality:?}: {width} px bar under {motion} motion trails {trails:.1} px per frame");
            assert!(error < 0.01, "{quality:?}: {width} px bar under {motion} motion: error {error:.5} vs truth");
        }
    }
}
