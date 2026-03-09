//! Sky atmosphere example – three boxes + roof on a ground plane, sun
//! atmosphere, and three lights.  Q/E rotate the sun.
//!
//! Controls:
//!   WASD        — move
//!   Space/Shift — up/down
//!   Q / E       — rotate sun left / right
//!   Mouse drag  — look (click to grab)
//!   Escape      — release / exit

mod demo_portal;
mod primitives;

use helio_render_v3::{
    Renderer, RendererConfig, AntiAliasingMode, ShadowConfig,
    Camera, HismRegistry, Material,
    SceneLight, SkyAtmosphere, LightId,
};
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId, CursorGrabMode},
};
use std::collections::HashSet;
use std::sync::Arc;

const SPEED: f32 = 5.0;
const LOOK_SENS: f32 = 0.003;

fn main() {
    env_logger::init();
    EventLoop::new().expect("el").run_app(&mut App { state: None }).expect("run");
}

struct App { state: Option<AppState> }

struct AppState {
    window:         Arc<Window>,
    surface:        wgpu::Surface<'static>,
    device:         Arc<wgpu::Device>,
    queue:          Arc<wgpu::Queue>,
    surface_format: wgpu::TextureFormat,
    renderer:       Renderer,
    last_frame:     std::time::Instant,
    time:           f32,
    sun_angle:      f32,
    sun_light:      LightId,
    cam_pos:        glam::Vec3,
    cam_yaw:        f32,
    cam_pitch:      f32,
    keys:           HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta:    (f32, f32),
}

fn make_renderer(device: &wgpu::Device, queue: &wgpu::Queue, w: u32, h: u32, fmt: wgpu::TextureFormat) -> Renderer {
    Renderer::new(device, queue, RendererConfig {
        width: w, height: h, surface_format: fmt,
        anti_aliasing: AntiAliasingMode::Fxaa,
        shadows: Some(ShadowConfig { atlas_size: 1024, max_shadow_lights: 4 }),
        radiance_cascades: None, billboards: None, bloom: None, ssao: None,
        gpu_driven: false, debug_printout: false,
    }, HismRegistry::new())
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() { return; }
        let window = Arc::new(event_loop.create_window(
            Window::default_attributes().with_title("Helio – Sky")
                .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32))
        ).expect("window"));
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { backends: wgpu::Backends::all(), ..Default::default() });
        let surface = instance.create_surface(window.clone()).expect("surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface), force_fallback_adapter: false,
        })).expect("adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("device"),
            required_features: wgpu::Features::EXPERIMENTAL_RAY_QUERY | wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
            required_limits: wgpu::Limits::default().using_minimum_supported_acceleration_structure_values(),
            memory_hints: wgpu::MemoryHints::default(),
            experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
            trace: wgpu::Trace::Off,
        }, )).expect("device");
        let device = Arc::new(device);
        let queue  = Arc::new(queue);
        let caps = surface.get_capabilities(&adapter);
        let fmt  = caps.formats.iter().find(|f| f.is_srgb()).copied().unwrap_or(caps.formats[0]);
        let size = window.inner_size();
        surface.configure(&device, &wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT, format: fmt,
            width: size.width, height: size.height,
            present_mode: wgpu::PresentMode::Fifo, alpha_mode: caps.alpha_modes[0],
            view_formats: vec![], desired_maximum_frame_latency: 2,
        });

        let mut renderer = make_renderer(&device, &queue, size.width, size.height, fmt);
        demo_portal::enable_live_dashboard(&mut renderer);

        let mat_grey  = renderer.create_material(&device, &queue, &Material { base_color: [0.7,0.7,0.7,1.0], roughness: 0.6, ..Default::default() });
        let mat_red   = renderer.create_material(&device, &queue, &Material { base_color: [0.8,0.15,0.1,1.0], roughness: 0.5, ..Default::default() });
        let mat_green = renderer.create_material(&device, &queue, &Material { base_color: [0.1,0.7,0.2,1.0],  roughness: 0.5, ..Default::default() });
        let mat_blue  = renderer.create_material(&device, &queue, &Material { base_color: [0.1,0.3,0.9,1.0],  roughness: 0.5, ..Default::default() });
        let mat_roof  = renderer.create_material(&device, &queue, &Material { base_color: [0.55,0.45,0.35,1.0], roughness: 0.8, ..Default::default() });

        let cube1  = primitives::build_box(&device, &queue,  0.0, 0.5,  0.0, 0.5, 0.5, 0.5);
        let cube2  = primitives::build_box(&device, &queue, -2.0, 0.4, -1.0, 0.4, 0.4, 0.4);
        let cube3  = primitives::build_box(&device, &queue,  2.0, 0.3,  0.5, 0.3, 0.3, 0.3);
        let ground = primitives::build_plane(&device, &queue, 0.0, 0.0, 0.0, 5.0, 5.0);
        let roof   = primitives::build_box(&device, &queue,  0.0, 4.0, 0.0, 5.0, 0.15, 5.0);

        let h1 = renderer.register_hism(cube1,  mat_red);
        let h2 = renderer.register_hism(cube2,  mat_green);
        let h3 = renderer.register_hism(cube3,  mat_blue);
        let hg = renderer.register_hism(ground, mat_grey);
        let hr = renderer.register_hism(roof,   mat_roof);
        renderer.add_instance(h1, glam::Mat4::IDENTITY);
        renderer.add_instance(h2, glam::Mat4::IDENTITY);
        renderer.add_instance(h3, glam::Mat4::IDENTITY);
        renderer.add_instance(hg, glam::Mat4::IDENTITY);
        renderer.add_instance(hr, glam::Mat4::IDENTITY);

        let sun_angle = 0.8_f32;
        let sun_dir = glam::Vec3::new(sun_angle.cos() * 0.6, -sun_angle.sin(), -0.8).normalize();
        renderer.add_light(SceneLight::point([-3.5, 2.0, -1.5].into(), [0.25, 0.5, 1.0].into(), 4.0, 6.0));
        renderer.add_light(SceneLight::point([3.5, 1.5, 1.5].into(),   [1.0, 0.3, 0.5].into(),  4.0, 6.0));
        let sun_light = renderer.add_light(SceneLight::directional(
            [sun_dir.x, sun_dir.y, sun_dir.z].into(), [1.0, 0.95, 0.85].into(), 3.0));

        renderer.set_sky(Some(SkyAtmosphere { sun_direction: [sun_dir.x, sun_dir.y, sun_dir.z].into(), ..Default::default() }));

        self.state = Some(AppState {
            window, surface, device, queue, surface_format: fmt,
            renderer, last_frame: std::time::Instant::now(), time: 0.0,
            sun_angle, sun_light,
            cam_pos: glam::Vec3::new(0.0, 2.5, 9.0), cam_yaw: 0.0, cam_pitch: -0.2,
            keys: HashSet::new(), cursor_grabbed: false, mouse_delta: (0.0, 0.0),
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event: KeyEvent { state: ElementState::Pressed, physical_key: PhysicalKey::Code(KeyCode::Escape), .. }, .. } => {
                if state.cursor_grabbed {
                    state.cursor_grabbed = false;
                    let _ = state.window.set_cursor_grab(CursorGrabMode::None);
                    state.window.set_cursor_visible(true);
                } else { event_loop.exit(); }
            }
            WindowEvent::KeyboardInput { event: KeyEvent { physical_key: PhysicalKey::Code(k), state: es, .. }, .. } => {
                match es { ElementState::Pressed => { state.keys.insert(k); } ElementState::Released => { state.keys.remove(&k); } }
            }
            WindowEvent::MouseInput { state: ElementState::Pressed, button: MouseButton::Left, .. } => {
                if !state.cursor_grabbed {
                    state.cursor_grabbed = true;
                    let _ = state.window.set_cursor_grab(CursorGrabMode::Locked)
                        .or_else(|_| state.window.set_cursor_grab(CursorGrabMode::Confined));
                    state.window.set_cursor_visible(false);
                }
            }
            WindowEvent::Resized(s) => {
                if s.width > 0 && s.height > 0 {
                    state.surface.configure(&state.device, &wgpu::SurfaceConfiguration {
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT, format: state.surface_format,
                        width: s.width, height: s.height, present_mode: wgpu::PresentMode::Fifo,
                        alpha_mode: wgpu::CompositeAlphaMode::Auto, view_formats: vec![],
                        desired_maximum_frame_latency: 2,
                    });
                }
            }
            WindowEvent::RedrawRequested => {
                let now = std::time::Instant::now();
                let dt = now.duration_since(state.last_frame).as_secs_f32();
                state.last_frame = now; state.time += dt;
                state.render(dt); state.window.request_redraw();
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _el: &ActiveEventLoop, _: winit::event::DeviceId, event: DeviceEvent) {
        let Some(state) = &mut self.state else { return };
        if let DeviceEvent::MouseMotion { delta } = event {
            if state.cursor_grabbed { state.mouse_delta.0 += delta.0 as f32; state.mouse_delta.1 += delta.1 as f32; }
        }
    }
}

impl AppState {
    fn render(&mut self, dt: f32) {
        self.cam_yaw   += self.mouse_delta.0 * LOOK_SENS;
        self.cam_pitch  = (self.cam_pitch - self.mouse_delta.1 * LOOK_SENS).clamp(-1.5, 1.5);
        self.mouse_delta = (0.0, 0.0);
        let (sy, cy) = self.cam_yaw.sin_cos();
        let (sp, cp) = self.cam_pitch.sin_cos();
        let forward = glam::Vec3::new(sy * cp, sp, -cy * cp);
        let right   = glam::Vec3::new(cy, 0.0, sy);
        if self.keys.contains(&KeyCode::KeyW) { self.cam_pos += forward * SPEED * dt; }
        if self.keys.contains(&KeyCode::KeyS) { self.cam_pos -= forward * SPEED * dt; }
        if self.keys.contains(&KeyCode::KeyA) { self.cam_pos -= right   * SPEED * dt; }
        if self.keys.contains(&KeyCode::KeyD) { self.cam_pos += right   * SPEED * dt; }
        if self.keys.contains(&KeyCode::Space)     { self.cam_pos.y += SPEED * dt; }
        if self.keys.contains(&KeyCode::ShiftLeft) { self.cam_pos.y -= SPEED * dt; }
        if self.keys.contains(&KeyCode::KeyQ) { self.sun_angle += 0.8 * dt; }
        if self.keys.contains(&KeyCode::KeyE) { self.sun_angle -= 0.8 * dt; }

        let sun_dir = glam::Vec3::new(self.sun_angle.cos() * 0.6, -self.sun_angle.sin(), -0.8).normalize();
        self.renderer.update_light(self.sun_light, SceneLight::directional(
            [sun_dir.x, sun_dir.y, sun_dir.z].into(), [1.0, 0.95, 0.85].into(), 3.0));
        self.renderer.set_sky(Some(SkyAtmosphere {
            sun_direction: [sun_dir.x, sun_dir.y, sun_dir.z].into(),
            ..Default::default()
        }));

        let size = self.window.inner_size();
        let aspect = size.width as f32 / size.height.max(1) as f32;
        self.renderer.set_camera(Camera::perspective(
            self.cam_pos, self.cam_pos + forward, glam::Vec3::Y,
            std::f32::consts::FRAC_PI_4, aspect, 0.1, 1000.0, self.time,
        ));
        let output = match self.surface.get_current_texture() {
            Ok(t) => t, Err(e) => { log::warn!("surface: {:?}", e); return; }
        };
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        if let Err(e) = self.renderer.render(&self.device, &self.queue, &view, dt) {
            log::error!("render: {:?}", e);
        }
        output.present();
    }
}
