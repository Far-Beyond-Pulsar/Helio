//! Minimal standalone 2D sprite-batch demo.
//!
//! Deliberately bypasses `helio::Renderer`/`Scene` — those are 3D-specific
//! (see `helio-pass-sprite-batch`'s module docs) — and drives
//! `helio_core::RenderGraph` directly against a single `SpriteBatchPass`,
//! proving out the independent 2D render-graph path end to end: its own
//! device/window setup, its own orthographic camera, no 3D scene graph.
//!
//! A ring of spinning, color-cycled sprites; nothing to click or drag.

use std::sync::Arc;
use std::time::Instant;

use helio_core::{GpuScene, RenderGraph};
use helio_pass_sprite_batch::{SpriteBatchPass, SpriteInstance};

use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

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
    /// `SpriteBatchPass` never reads it. Constructed once and otherwise
    /// ignored, so this demo can skip `helio::Renderer`/`Scene` entirely.
    scene: GpuScene,
    dummy_depth_view: wgpu::TextureView,
    start_time: Instant,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Helio – 2D Sprite Batch Demo")
                        .with_inner_size(winit::dpi::LogicalSize::new(960u32, 540u32)),
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
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
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
        graph.add_pass(Box::new(SpriteBatchPass::new(&device, &queue, format)));
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
            start_time: Instant::now(),
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
                let time = state.start_time.elapsed().as_secs_f32();

                let sprite = state
                    .graph
                    .find_pass_mut::<SpriteBatchPass>()
                    .expect("sprite batch pass missing from graph");
                sprite.clear();
                const COUNT: usize = 12;
                for i in 0..COUNT {
                    let t = i as f32 / COUNT as f32;
                    let angle = t * std::f32::consts::TAU + time * 0.5;
                    let radius = 160.0;
                    let pos = [angle.cos() * radius, angle.sin() * radius];
                    let hue = (t + time * 0.1).fract();
                    sprite.push_sprite(
                        SpriteInstance::new(pos, [48.0, 48.0])
                            .with_rotation(time * 2.0 + t * std::f32::consts::TAU)
                            .with_color(hsv_to_rgb(hue, 0.7, 1.0)),
                    );
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
