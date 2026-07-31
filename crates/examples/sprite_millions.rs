//! 10 million sprite galaxy — the GPU cull/sort path at extreme pool size.
//!
//! 10,000,000 sprites are laid out on a golden-angle spiral (`index ≈ radius`,
//! so a contiguous run of indices is a contiguous *radial* band), and an
//! orthographic camera slowly orbits the galaxy. A narrow band of "awake"
//! sprites shimmers as it sweeps outward and back across the disk.
//!
//! This exercises the architecture's real wins at a scale that would crush a
//! CPU-culled bunnymark:
//!
//! - `SpriteCullPass` culls the entire 10M pool against the view rect on the
//!   GPU every frame (~39k threads) and radix-sorts only what survives. The
//!   CPU never does any per-instance culling/sorting, regardless of pool size.
//! - The batch pass delta-uploads only the awake band's contiguous byte range
//!   (`band_width × 80 B`), not the ~800 MB pool. Sprites outside the band
//!   aren't touched at all after the initial upload, so per-frame CPU→GPU
//!   traffic stays tiny even with 10M sprites in the pool.
//!
//! `HELIO_SPRITE_MILLIONS_COUNT` overrides the sprite count (default
//! 10,000,000); `HELIO_SPRITE_MILLIONS_BAND` the awake-band width (default
//! 300,000). Run in release mode for the full effect:
//!
//! ```text
//! cargo run --release -p examples --bin sprite_millions
//! ```

use std::sync::Arc;
use std::time::Instant;

use helio_core::{GpuScene, RenderGraph};
use helio_pass_sprite_batch::{SpriteBatchPass, SpriteHandle, SpriteInstance};
use helio_pass_sprite_cull::SpriteCullPass;

use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

const TAU: f32 = std::f32::consts::TAU;
/// Golden angle (radians): spiral arm pitch such that consecutive sprites
/// never line up on the same ray — gives a uniform-density disk.
const GOLDEN_ANGLE: f32 = 2.3999632297286533;

/// Byte size of one `SpriteInstance` (`#[repr(C)]`, 16-byte aligned).
const INSTANCE_BYTES: u64 = 80;

/// Radius of the galaxy disk, in world units. The camera's half-extent is
/// smaller than this, so the view always sits *inside* the galaxy.
const GALAXY_RADIUS: f32 = 5000.0;
/// Orthographic view half-extent, in world units.
const VIEW_HALF: [f32; 2] = [1100.0, 620.0];
/// Camera orbit radius, in world units.
const ORBIT_RADIUS: f32 = 2500.0;
/// Sprite size, in world units (~1 px at the view above).
const SPRITE_SIZE: [f32; 2] = [2.0, 2.0];

/// Position of sprite `i` on the golden-angle spiral. `index ≈ radius`
/// (sqrt fill), so a contiguous index range is a contiguous radial band.
fn spiral_pos(i: u32, count: u32, r_max: f32) -> [f32; 2] {
    let t = i as f32 / count as f32;
    let r = r_max * t.sqrt();
    let theta = i as f32 * GOLDEN_ANGLE;
    [r * theta.cos(), r * theta.sin()]
}

fn hue_for(i: u32) -> f32 {
    (i as f32 * GOLDEN_ANGLE / TAU).fract()
}

/// Small deterministic hue→RGB, same as the other sprite demos.
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 4] {
    let i = (h * 6.0).floor();
    let f = h * 6.0 - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    let (r, g, b) = match (i as i32).rem_euclid(6) {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    [r, g, b, 1.0]
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().expect("event loop");
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("run");
}

struct App {
    state: Option<AppState>,
}

impl App {
    fn new() -> Self {
        Self { state: None }
    }
}

struct AppState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    surface_format: wgpu::TextureFormat,
    graph: RenderGraph,
    /// `RenderGraph::execute()` takes a `&GpuScene` for its API shape only —
    /// neither 2D pass reads it.
    scene: GpuScene,
    dummy_depth_view: wgpu::TextureView,
    dot_layer: u32,

    count: u32,
    band_width: u32,
    band_center: f32,
    band_dir: f32,
    handles: Vec<SpriteHandle>,

    start_time: Instant,
    last_frame: Instant,
    fps_frames: u32,
    fps_last_print: Instant,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let count = std::env::var("HELIO_SPRITE_MILLIONS_COUNT")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(10_000_000);
        let band_width = std::env::var("HELIO_SPRITE_MILLIONS_BAND")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(300_000)
            .min(count);
        eprintln!(
            "[sprite_millions] {count} sprites, awake band {band_width} \
             (~{:.0} MiB instance pool, ~{:.0} MiB/frame delta upload)",
            count as f64 * 80.0 / (1024.0 * 1024.0),
            band_width as f64 * 80.0 / (1024.0 * 1024.0),
        );

        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Helio – 10M Sprite Galaxy")
                        .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32)),
                )
                .expect("window"),
        );

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::empty(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let surface = instance.create_surface(window.clone()).expect("surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .expect("adapter");
        // The instance pool is ~count × 80 B (763 MiB at 10M) — far past
        // wgpu's default `max_buffer_size` of 256 MiB. Request the adapter's
        // real limits, clamped to 4 GiB − 1 exactly like the engine's own
        // `required_wgpu_limits` does (wgpu-core asserts
        // `max_buffer_size <= u32::MAX` at device creation).
        let adapter_limits = adapter.limits();
        let storage_bytes = count as u64 * INSTANCE_BYTES;
        let required_limits = wgpu::Limits {
            max_buffer_size: adapter_limits.max_buffer_size.min(u32::MAX as u64),
            ..adapter_limits
        };
        if storage_bytes > required_limits.max_buffer_size
            || storage_bytes > required_limits.max_storage_buffer_binding_size
        {
            eprintln!(
                "[sprite_millions] adapter can't hold {count} sprites: pool needs {:.0} MiB \
                 but max_buffer_size={:.0} MiB, max_storage_buffer_binding_size={:.0} MiB",
                storage_bytes / (1024 * 1024),
                required_limits.max_buffer_size / (1024 * 1024),
                required_limits.max_storage_buffer_binding_size / (1024 * 1024),
            );
            event_loop.exit();
            return;
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Device"),
            required_features: wgpu::Features::empty(),
            required_limits,
            ..Default::default()
        }))
        .expect("device");
        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);
        let size = window.inner_size();
        surface.configure(
            &device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                width: size.width.max(1),
                height: size.height.max(1),
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: caps.alpha_modes[0],
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
                color_space: wgpu::SurfaceColorSpace::Auto,
            },
        );

        let mut graph = RenderGraph::new(&device, &queue);
        let mut sprite_pass = SpriteBatchPass::new(&device, &queue, format);
        sprite_pass.set_camera([0.0, 0.0], Some(VIEW_HALF));
        sprite_pass.set_clear_color(Some(wgpu::Color { r: 0.005, g: 0.008, b: 0.02, a: 1.0 }));
        let dot_layer = sprite_pass.add_atlas_layer(&device, &queue, 8, 8, &make_dot_atlas());

        // GPU cull/sort pass, wired in *before* the batch pass it feeds. It
        // binds the pool's instance/alive buffers once at construction, so
        // `reserve()` must fix the pool size before any sprite is inserted.
        // `max_visible` is sized for the densest patch the view can ever
        // cover (a fraction of the disk), not the whole pool.
        sprite_pass.reserve(&device, count as usize);
        let max_visible = 1_000_000u32;
        let mut sprite_cull = SpriteCullPass::new(
            &device,
            &queue,
            sprite_pass.instances_buffer(),
            sprite_pass.alive_buffer(),
            count,
            max_visible,
        );
        sprite_cull.set_view_rect([0.0, 0.0], VIEW_HALF);
        sprite_pass.use_gpu_culling(sprite_cull.draw_order_buf.clone(), sprite_cull.indirect_buf.clone());

        // Insert the whole galaxy once. Everything after this is delta-uploaded
        // in contiguous radial bands — static sprites are never touched again.
        let mut handles = Vec::with_capacity(count as usize);
        for i in 0..count {
            let pos = spiral_pos(i, count, GALAXY_RADIUS);
            let hue = hue_for(i);
            let handle = sprite_pass.insert_sprite(
                SpriteInstance::new(pos, SPRITE_SIZE)
                    .with_depth(pos[1])
                    .with_color(hsv_to_rgb(hue, 0.8, 1.0))
                    .with_atlas_layer(dot_layer),
            );
            handles.push(handle);
        }

        graph.add_pass(Box::new(sprite_cull));
        graph.add_pass(Box::new(sprite_pass));
        graph.lock(size.width.max(1), size.height.max(1));

        let scene = GpuScene::new(device.clone(), queue.clone());
        let dummy_depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Dummy Depth (unused by 2D passes)"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let dummy_depth_view = dummy_depth.create_view(&wgpu::TextureViewDescriptor::default());

        self.state = Some(AppState {
            window,
            surface,
            device,
            queue,
            surface_format: format,
            graph,
            scene,
            dummy_depth_view,
            dot_layer,
            count,
            band_width,
            band_center: 0.0,
            band_dir: 1.0,
            handles,
            start_time: Instant::now(),
            last_frame: Instant::now(),
            fps_frames: 0,
            fps_last_print: Instant::now(),
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(s) if s.width > 0 && s.height > 0 => {
                state.surface.configure(
                    &state.device,
                    &wgpu::SurfaceConfiguration {
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                        format: state.surface_format,
                        width: s.width,
                        height: s.height,
                        present_mode: wgpu::PresentMode::Fifo,
                        alpha_mode: wgpu::CompositeAlphaMode::Auto,
                        view_formats: vec![],
                        desired_maximum_frame_latency: 2,
                        color_space: wgpu::SurfaceColorSpace::Auto,
                    },
                );
                state.graph.set_render_size(s.width, s.height);
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - state.last_frame).as_secs_f32().min(0.05);
                state.last_frame = now;
                let t = state.start_time.elapsed().as_secs_f32();

                // Camera: slow orbit around the galaxy center. World-space view
                // is fixed (`VIEW_HALF`), so only the center pans.
                let angle = t * 0.09;
                let center = [ORBIT_RADIUS * angle.cos(), ORBIT_RADIUS * angle.sin()];
                state
                    .graph
                    .find_pass_mut::<SpriteBatchPass>()
                    .expect("sprite batch pass missing from graph")
                    .set_camera(center, Some(VIEW_HALF));
                state
                    .graph
                    .find_pass_mut::<SpriteCullPass>()
                    .expect("sprite cull pass missing from graph")
                    .set_view_rect(center, VIEW_HALF);

                // Awake band: sweeps outward and back across the disk. Index
                // ≈ radius (spiral), so the band is a contiguous index run →
                // one contiguous dirty byte range uploaded this frame.
                state.band_center += state.band_dir * 2_000_000.0 * dt;
                let max_center = (state.count - state.band_width) as f32;
                if state.band_center >= max_center {
                    state.band_center = max_center;
                    state.band_dir = -1.0;
                } else if state.band_center <= 0.0 {
                    state.band_center = 0.0;
                    state.band_dir = 1.0;
                }
                let band_start = state.band_center as u32;

                let dot_layer = state.dot_layer;
                let count = state.count;
                let sprite_pass = state
                    .graph
                    .find_pass_mut::<SpriteBatchPass>()
                    .expect("sprite batch pass missing from graph");
                for k in 0..state.band_width {
                    let i = band_start + k;
                    let base = spiral_pos(i, count, GALAXY_RADIUS);
                    let r_scale = 1.0 + 0.04 * (t * 1.6 + i as f32 * 0.0021).sin();
                    let pos = [base[0] * r_scale, base[1] * r_scale];
                    sprite_pass.update_sprite(
                        state.handles[i as usize],
                        SpriteInstance::new(pos, SPRITE_SIZE)
                            .with_depth(pos[1])
                            .with_color(hsv_to_rgb(hue_for(i), 0.8, 1.0))
                            .with_atlas_layer(dot_layer),
                    );
                }

                state.fps_frames += 1;
                if state.fps_last_print.elapsed().as_secs_f32() >= 1.0 {
                    let elapsed = state.fps_last_print.elapsed().as_secs_f32();
                    log::info!(
                        "[sprite_millions] {:.0} fps | pool={} awake={} band={}",
                        state.fps_frames as f32 / elapsed,
                        sprite_pass.sprite_count(),
                        state.band_width,
                        state.band_center as u32,
                    );
                    state.fps_frames = 0;
                    state.fps_last_print = Instant::now();
                }

                let output = match state.surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(texture)
                    | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
                    _ => {
                        state.window.request_redraw();
                        return;
                    }
                };
                let view = output.texture.create_view(&Default::default());
                if let Err(e) = state.graph.execute(&state.scene, &view, &state.dummy_depth_view) {
                    log::error!("graph execute error: {e:?}");
                }
                state.queue.present(output);
                state.window.request_redraw();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(s) = &self.state {
            s.window.request_redraw();
        }
    }
}

/// A small soft-edged 8×8 dot, straight alpha, on a transparent background.
fn make_dot_atlas() -> Vec<u8> {
    const SIZE: usize = 8;
    let mut pixels = vec![0u8; SIZE * SIZE * 4];
    for y in 0..SIZE {
        for x in 0..SIZE {
            let cx = (x as f32 + 0.5) / SIZE as f32 * 2.0 - 1.0;
            let cy = (y as f32 + 0.5) / SIZE as f32 * 2.0 - 1.0;
            let dist = (cx * cx + cy * cy).sqrt();
            let a = ((1.0 - dist).clamp(0.0, 1.0) * 255.0) as u8;
            let i = (y * SIZE + x) * 4;
            pixels[i] = 255;
            pixels[i + 1] = 255;
            pixels[i + 2] = 255;
            pixels[i + 3] = a;
        }
    }
    pixels
}
