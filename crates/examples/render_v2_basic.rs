//! Feature showcase example using helio-render-v2
//!
//! All scene content is driven by a `Scene` struct — no hardcoded lights
//! or geometry in the renderer.

use helio_render_v2::{Renderer, RendererConfig, Camera, GpuMesh, Scene, SceneLight};
use helio_render_v2::features::{
    FeatureRegistry,
    LightingFeature,
    BloomFeature, ShadowsFeature,
    BillboardsFeature, BillboardInstance,
};
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};
use std::sync::Arc;

fn main() {
    env_logger::init();
    log::info!("Starting Helio Render V2 Basic Example");

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut app = App::new();

    event_loop.run_app(&mut app).expect("Event loop error");
}

struct App {
    state: Option<AppState>,
}

struct AppState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>,
    surface_format: wgpu::TextureFormat,
    renderer: Renderer,
    last_frame: std::time::Instant,
    cube1: GpuMesh,
    cube2: GpuMesh,
    cube3: GpuMesh,
    ground: GpuMesh,
}

impl App {
    fn new() -> Self {
        Self { state: None }
    }
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
                        .with_title("Helio Render V2 – Scene-Driven")
                        .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32)),
                )
                .expect("Failed to create window"),
        );

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let surface = instance
            .create_surface(window.clone())
            .expect("Failed to create surface");

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("Failed to find adapter");

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Main Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        ))
        .expect("Failed to create device");

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // Features — data-free: all content comes from the Scene
        let feature_registry = FeatureRegistry::builder()
            .with_feature(LightingFeature::new())
            .with_feature(BloomFeature::new().with_intensity(0.4).with_threshold(1.2))
            .with_feature(ShadowsFeature::new().with_atlas_size(1024).with_max_lights(4))
            .with_feature(BillboardsFeature::new())
            .build();

        let renderer = Renderer::new(
            device.clone(),
            queue.clone(),
            RendererConfig {
                width: size.width,
                height: size.height,
                surface_format,
                features: feature_registry,
            },
        )
        .expect("Failed to create renderer");

        let cube1  = GpuMesh::cube(&device, [ 0.0, 0.5,  0.0], 0.5);
        let cube2  = GpuMesh::cube(&device, [-2.0, 0.4, -1.0], 0.4);
        let cube3  = GpuMesh::cube(&device, [ 2.0, 0.3,  0.5], 0.3);
        let ground = GpuMesh::plane(&device, [0.0, 0.0, 0.0], 5.0);

        self.state = Some(AppState {
            window,
            surface,
            device,
            surface_format,
            renderer,
            last_frame: std::time::Instant::now(),
            cube1, cube2, cube3, ground,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };

        match event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                event: KeyEvent {
                    state: ElementState::Pressed,
                    physical_key: PhysicalKey::Code(KeyCode::Escape),
                    ..
                },
                ..
            } => {
                log::info!("Shutting down");
                event_loop.exit();
            }
            WindowEvent::Resized(size) if size.width > 0 && size.height > 0 => {
                let config = wgpu::SurfaceConfiguration {
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    format: state.surface_format,
                    width: size.width,
                    height: size.height,
                    present_mode: wgpu::PresentMode::Fifo,
                    alpha_mode: wgpu::CompositeAlphaMode::Auto,
                    view_formats: vec![],
                    desired_maximum_frame_latency: 2,
                };
                state.surface.configure(&state.device, &config);
                state.renderer.resize(size.width, size.height);
            }
            WindowEvent::RedrawRequested => {
                let now = std::time::Instant::now();
                let dt = (now - state.last_frame).as_secs_f32();
                state.last_frame = now;
                state.render(dt);
                state.window.request_redraw();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }
}

impl AppState {
    fn render(&mut self, delta_time: f32) {
        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(e) => { log::warn!("Surface error: {:?}", e); return; }
        };
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let time = self.renderer.frame_count() as f32 * 0.016;
        let angle = time * 0.3;
        let camera = Camera::perspective(
            glam::Vec3::new(angle.sin() * 6.0, 3.0, angle.cos() * 6.0),
            glam::Vec3::new(0.0, 0.5, 0.0),
            glam::Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            1280.0 / 720.0,
            0.1,
            100.0,
            time,
        );

        // Build the scene – this is the ONLY place scene content is defined
        let p0 = [0.0f32, 2.2 + (time * 0.7).sin() * 0.3, 0.0];
        let p1 = [-3.5f32, 2.0, -1.5];
        let p2 = [3.5f32, 1.5, 1.5];

        let scene = Scene::new()
            // Orange point light hovering above center, slowly bobbing
            .add_light(SceneLight::point(p0, [1.0, 0.55, 0.15], 6.0, 5.0))
            // Cool blue light off to one side
            .add_light(SceneLight::point(p1, [0.25, 0.5, 1.0], 5.0, 6.0))
            // Warm pink light on the other side
            .add_light(SceneLight::point(p2, [1.0, 0.3, 0.5], 5.0, 6.0))
            // Objects
            .add_object(self.cube1.clone())
            .add_object(self.cube2.clone())
            .add_object(self.cube3.clone())
            .add_object(self.ground.clone())
            // Billboards co-located with each light so you can see their source
            .add_billboard(BillboardInstance::new(p0, [0.35, 0.35]).with_color([1.0, 0.55, 0.15, 1.0]))
            .add_billboard(BillboardInstance::new(p1, [0.35, 0.35]).with_color([0.25, 0.5, 1.0, 1.0]))
            .add_billboard(BillboardInstance::new(p2, [0.35, 0.35]).with_color([1.0, 0.3, 0.5, 1.0]));

        if let Err(e) = self.renderer.render_scene(&scene, &camera, &view, delta_time) {
            log::error!("Render error: {:?}", e);
        }

        output.present();
    }
}

