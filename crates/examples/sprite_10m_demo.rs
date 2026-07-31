//! 10 million sprite demo — the full GPU-driven 2D pipeline at scale.
//!
//! Three passes, in graph order, and the CPU touches none of them per frame:
//!
//!   1. `SpriteSimulatePass` (`helio-pass-sprite-simulate`) — bounces every
//!      sprite around a world *much* larger than the camera's view, entirely
//!      on the GPU. Positions are written once at startup and never touched
//!      by the CPU again (see that crate's module doc comment on why that
//!      rule matters).
//!   2. `SpriteCullPass` (`helio-pass-sprite-cull`) — culls the 10M against
//!      the (small, fixed) camera view rect and radix-sorts the survivors,
//!      also entirely on the GPU.
//!   3. `SpriteBatchPass` (`helio-pass-sprite-batch`) — one
//!      `draw_indexed_indirect` reading the GPU-computed visible count.
//!
//! The reason this is smooth at 10M and the naive version wouldn't be: the
//! world is ~27x the camera's area, so at any moment only a few hundred
//! thousand of the 10M sprites are actually visible (see `CAMERA_HALF`'s doc
//! comment for the exact ratio and why it isn't zoomed in tighter) — culling
//! isn't optional set-dressing here, it's what keeps the draw call and the
//! sort's `O(n)` cost bounded by *visible* count, not pool size. Simulating
//! and culling the full 10M every frame is still real GPU work, but it's GPU
//! work: thousands of parallel threads doing a few flops each, not 10
//! million sequential CPU iterations.
//!
//! `HELIO_SPRITE_10M_COUNT` overrides the sprite count (default 10,000,000).
//! Startup inserts the whole pool before the first frame, which takes a
//! visible pause (a few seconds) — that's one-time CPU cost building the
//! initial `Vec<SpriteInstance>` and its one big upload, not a per-frame cost.

use std::sync::Arc;
use std::time::Instant;

use helio_core::{GpuScene, RenderGraph};
use helio_pass_sprite_batch::{SpriteBatchPass, SpriteInstance};
use helio_pass_sprite_cull::SpriteCullPass;
use helio_pass_sprite_simulate::SpriteSimulatePass;

use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

/// Half-extent of the simulated world sprites bounce around in — much larger
/// than the camera's view (see the module doc comment on why).
const WORLD_HALF: [f32; 2] = [10_000.0, 10_000.0];
/// Half-extent of the fixed, static camera view. A 1:1 world-unit-to-pixel
/// camera (e.g. `[640, 360]` for a 1280×720 window) only shows ~23,000 of
/// the 10M sprites at once — visually underwhelming for a "look at all these
/// sprites" demo even though it's culling correctly. Zoomed out 4x per axis
/// (16x the visible area) so there's an actual field of sprites on screen,
/// while still only showing a small fraction of the world (~3.7% of its
/// area) — culling is still doing real work, just not *maximally aggressive*
/// work.
const CAMERA_HALF: [f32; 2] = [2560.0, 1440.0];
/// Safety cap on how many sprites the cull pass will compact + sort in one
/// frame. Expected visible count is roughly
/// `count * (camera area / world area)` — for the defaults that's
/// `10e6 * (5120*2880)/(20000*20000) ≈ 369,000`; this leaves headroom for
/// clustering without the cull shader's `slot < max_visible` cap silently
/// dropping the overflow.
const MAX_VISIBLE: u32 = 600_000;
const SPRITE_SIZE: [f32; 2] = [8.0, 8.0];

struct Rng(u64);
impl Rng {
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_f32() * (hi - lo)
    }
}

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
    /// none of the three sprite passes read it.
    scene: GpuScene,
    dummy_depth_view: wgpu::TextureView,
    sprite_count: usize,
    last_frame: Instant,
    fps_frames: u32,
    fps_last_print: Instant,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let count = std::env::var("HELIO_SPRITE_10M_COUNT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(10_000_000);
        eprintln!("[sprite_10m_demo] spawning {count} sprites (this takes a moment)...");

        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Helio – 10M Sprite Demo")
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
        // `wgpu::Limits::default()` caps a single buffer at 256 MiB (the
        // WebGPU spec minimum) — the 10M-sprite instance buffer alone needs
        // ~800 MiB, so request the adapter's actual supported limits instead.
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Device"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
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
        // Fixed, static camera — small relative to the simulated world, so
        // only a fraction of the pool is ever visible at once. Not derived
        // from window size (`Some(..)`), so resizing the window doesn't
        // change which sprites the (separately configured) cull pass and
        // this camera agree are visible.
        sprite_pass.set_camera([0.0, 0.0], Some(CAMERA_HALF));
        sprite_pass.set_clear_color(Some(wgpu::Color { r: 0.02, g: 0.02, b: 0.04, a: 1.0 }));
        let dot_layer = sprite_pass.add_atlas_layer(&device, &queue, 8, 8, &make_dot_atlas());

        // Pre-size the pool's GPU buffers *before* wiring GPU culling/
        // simulation — both bind to these buffers once at construction and
        // don't follow reallocation.
        sprite_pass.reserve(&device, count);

        let mut sprite_cull =
            SpriteCullPass::new(&device, &queue, sprite_pass.instances_buffer(), sprite_pass.alive_buffer(), count as u32, MAX_VISIBLE);
        sprite_cull.set_view_rect([0.0, 0.0], CAMERA_HALF);

        // Spawn the whole pool once. Positions/velocities scatter across
        // `WORLD_HALF` (not `CAMERA_HALF` — see the module doc comment);
        // after this, `SpriteBatchPass`'s dirty range goes permanently
        // clean and `SpriteSimulatePass` owns every slot's position from
        // here on, entirely on the GPU.
        let mut rng = Rng(0x10DA_5EED_C0FF_EE10);
        let mut velocities: Vec<[f32; 2]> = Vec::with_capacity(count);
        for _ in 0..count {
            let pos = [rng.range(-WORLD_HALF[0], WORLD_HALF[0]), rng.range(-WORLD_HALF[1], WORLD_HALF[1])];
            let vel = [rng.range(-120.0, 120.0), rng.range(-120.0, 120.0)];
            let hue = rng.next_f32();
            velocities.push(vel);
            sprite_pass.insert_sprite(
                SpriteInstance::new(pos, SPRITE_SIZE)
                    .with_depth(pos[1])
                    .with_color(hsv_to_rgb(hue, 0.75, 1.0))
                    .with_atlas_layer(dot_layer),
            );
        }

        let sprite_simulate = SpriteSimulatePass::new(
            &device,
            sprite_pass.instances_buffer(),
            sprite_pass.alive_buffer(),
            &velocities,
            [-WORLD_HALF[0], -WORLD_HALF[1]],
            WORLD_HALF,
        );

        sprite_pass.use_gpu_culling(sprite_cull.draw_order_buf.clone(), sprite_cull.indirect_buf.clone());

        // Order matters: simulate writes positions the cull pass reads, and
        // cull writes the draw order/indirect args the batch pass reads.
        graph.add_pass(Box::new(sprite_simulate));
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

        eprintln!("[sprite_10m_demo] ready");
        self.state = Some(AppState {
            window,
            surface,
            device,
            queue,
            surface_format: format,
            graph,
            scene,
            dummy_depth_view,
            sprite_count: count,
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
                // Mirror is resized but the render graph's *internal*
                // resolution deliberately isn't tied to it here — the camera
                // is fixed (`Some(CAMERA_HALF)`), so there's nothing for a
                // resize to invalidate about what's visible.
                state.graph.set_render_size(s.width, s.height);
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - state.last_frame).as_secs_f32().min(0.05);
                state.last_frame = now;
                // The one thing the CPU still does every frame: tell the
                // graph how much time passed, so `SpriteSimulatePass::prepare`
                // can integrate by the real `dt` instead of a fixed step.
                state.graph.set_delta_time(dt);

                state.fps_frames += 1;
                if state.fps_last_print.elapsed().as_secs_f32() >= 1.0 {
                    let elapsed = state.fps_last_print.elapsed().as_secs_f32();
                    log::info!(
                        "[sprite_10m_demo] {:.1} fps | pool={}",
                        state.fps_frames as f32 / elapsed,
                        state.sprite_count,
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
