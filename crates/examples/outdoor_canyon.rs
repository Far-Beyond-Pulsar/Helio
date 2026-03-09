//! Outdoor canyon – rocky valley with campfire lights and sky atmosphere.
//! Q / E rotate the sun.
//!
//! Controls: WASD / Space / Shift, mouse drag, Escape.

mod demo_portal;
mod primitives;

use helio_render_v3::{
    Renderer, RendererConfig, AntiAliasingMode, ShadowConfig,
    Camera, HismRegistry, LightId, Material, SceneLight, SkyAtmosphere,
};
use winit::{
    application::ApplicationHandler, event::*, event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey}, window::{Window, WindowId, CursorGrabMode},
};
use std::collections::HashSet;
use std::sync::Arc;

const SPEED: f32 = 8.0;
const LOOK_SENS: f32 = 0.003;

fn main() { env_logger::init(); EventLoop::new().expect("el").run_app(&mut App { state: None }).expect("run"); }

struct App { state: Option<AppState> }
struct AppState {
    window: Arc<Window>, surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>,
    surface_format: wgpu::TextureFormat, renderer: Renderer,
    last_frame: std::time::Instant, time: f32,
    sun_angle: f32, sun_light: LightId,
    cam_pos: glam::Vec3, cam_yaw: f32, cam_pitch: f32,
    keys: HashSet<KeyCode>, cursor_grabbed: bool, mouse_delta: (f32, f32),
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() { return; }
        let window = Arc::new(event_loop.create_window(Window::default_attributes().with_title("Helio – Outdoor Canyon").with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32))).expect("w"));
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { backends: wgpu::Backends::all(), ..Default::default() });
        let surface = instance.create_surface(window.clone()).expect("surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: Some(&surface), force_fallback_adapter: false })).expect("adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { label: Some("device"), required_features: wgpu::Features::EXPERIMENTAL_RAY_QUERY | wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS, required_limits: wgpu::Limits::default().using_minimum_supported_acceleration_structure_values(), memory_hints: wgpu::MemoryHints::default(), experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() }, trace: wgpu::Trace::Off })).expect("device");
        let device = Arc::new(device); let queue = Arc::new(queue);
        let caps = surface.get_capabilities(&adapter);
        let fmt = caps.formats.iter().find(|f| f.is_srgb()).copied().unwrap_or(caps.formats[0]);
        let size = window.inner_size();
        surface.configure(&device, &wgpu::SurfaceConfiguration { usage: wgpu::TextureUsages::RENDER_ATTACHMENT, format: fmt, width: size.width, height: size.height, present_mode: wgpu::PresentMode::Fifo, alpha_mode: caps.alpha_modes[0], view_formats: vec![], desired_maximum_frame_latency: 2 });

        let mut renderer = Renderer::new(&device, &queue, RendererConfig { width: size.width, height: size.height, surface_format: fmt, anti_aliasing: AntiAliasingMode::Fxaa, shadows: Some(ShadowConfig { atlas_size: 2048, max_shadow_lights: 4 }), radiance_cascades: None, billboards: None, bloom: None, ssao: None, gpu_driven: false, debug_printout: false }, HismRegistry::new());
        demo_portal::enable_live_dashboard(&mut renderer);

        let mat_valley = renderer.create_material(&device, &queue, &Material { base_color: [0.48,0.38,0.28,1.0], roughness: 0.95, ..Default::default() });
        let mat_rock   = renderer.create_material(&device, &queue, &Material { base_color: [0.55,0.42,0.30,1.0], roughness: 0.90, ..Default::default() });
        let mat_mesa   = renderer.create_material(&device, &queue, &Material { base_color: [0.62,0.50,0.35,1.0], roughness: 0.88, ..Default::default() });
        let mat_tent   = renderer.create_material(&device, &queue, &Material { base_color: [0.70,0.55,0.35,1.0], roughness: 0.8, ..Default::default() });
        let mat_fire   = renderer.create_material(&device, &queue, &Material { base_color: [1.0,0.5,0.1,1.0], roughness: 1.0, emissive_color: [5.0,2.5,0.5], ..Default::default() });

        // Valley floor
        let floor = primitives::build_plane(&device, &queue, 0.0, 0.0, 0.0, 25.0, 20.0);
        let hf = renderer.register_hism(floor, mat_valley.clone());
        renderer.add_instance(hf, glam::Mat4::IDENTITY);

        // Canyon walls: 3 on each side + back
        let canyon_walls: &[(f32, f32, f32, f32, f32, f32)] = &[
            (-22.0, 8.0,  0.0, 3.0, 8.0, 20.0),
            (-18.0, 5.0, -16.0, 3.0, 5.0,  6.0),
            (-20.0, 6.0,  14.0, 3.0, 6.0,  6.0),
            ( 22.0, 8.0,  0.0, 3.0, 8.0, 20.0),
            ( 18.0, 5.0, -16.0, 3.0, 5.0,  6.0),
            ( 20.0, 6.0,  14.0, 3.0, 6.0,  6.0),
        ];
        for &(cx, cy, cz, hx, hy, hz) in canyon_walls {
            let m = primitives::build_box(&device, &queue, cx, cy, cz, hx, hy, hz);
            let h = renderer.register_hism(m, mat_rock.clone());
            renderer.add_instance(h, glam::Mat4::IDENTITY);
        }

        // Terraces
        let terraces: &[(f32, f32, f32, f32, f32, f32)] = &[
            (-14.0, 1.5, 5.0, 5.0, 1.5, 3.5),
            ( 14.0, 2.0, -3.0, 5.0, 2.0, 4.0),
            (-10.0, 1.0, -8.0, 4.0, 1.0, 4.0),
            ( 10.0, 1.2,  8.0, 4.0, 1.2, 3.5),
        ];
        for &(cx, cy, cz, hx, hy, hz) in terraces {
            let m = primitives::build_box(&device, &queue, cx, cy, cz, hx, hy, hz);
            let h = renderer.register_hism(m, mat_rock.clone());
            renderer.add_instance(h, glam::Mat4::IDENTITY);
        }

        // Mesa
        let mesa = primitives::build_box(&device, &queue, 0.0, 3.0, -18.0, 8.0, 3.0, 4.0);
        let hm = renderer.register_hism(mesa, mat_mesa);
        renderer.add_instance(hm, glam::Mat4::IDENTITY);

        // Tents
        for &(tx, tz) in &[(-4.0f32, -2.0f32), (4.0, -4.0), (0.0, 5.0)] {
            let tent = primitives::build_box(&device, &queue, tx, 0.8, tz, 0.9, 0.8, 0.7);
            let ht = renderer.register_hism(tent, mat_tent.clone());
            renderer.add_instance(ht, glam::Mat4::IDENTITY);
        }

        // Firepit emissive
        let fire = primitives::build_box(&device, &queue, 0.0, 0.2, 0.0, 0.3, 0.2, 0.3);
        let hfire = renderer.register_hism(fire, mat_fire);
        renderer.add_instance(hfire, glam::Mat4::IDENTITY);

        // Campfire lights
        renderer.add_light(SceneLight::point([-0.2, 0.6, -0.1].into(), [1.0,0.65,0.2].into(), 6.0, 8.0));
        renderer.add_light(SceneLight::point([ 0.2, 0.6,  0.1].into(), [1.0,0.55,0.15].into(), 5.0, 7.0));
        renderer.add_light(SceneLight::point([ 0.0, 0.8,  0.0].into(), [1.0,0.7, 0.25].into(), 4.0, 6.0));
        // Fill light
        renderer.add_light(SceneLight::point([0.0, 6.0, 0.0].into(), [0.35,0.45,0.7].into(), 2.0, 40.0));

        // Sun directional
        let sun_angle: f32 = 0.8;
        let sun_dir = glam::Vec3::new(sun_angle.cos() * 0.6, -sun_angle.sin(), -0.8).normalize();
        let sun_light = renderer.add_light(SceneLight::directional(sun_dir, [1.0,0.95,0.80].into(), 3.0));
        renderer.set_sky(Some(SkyAtmosphere { sun_direction: sun_dir, ..Default::default() }));

        self.state = Some(AppState { window, surface, device, queue, surface_format: fmt, renderer, last_frame: std::time::Instant::now(), time: 0.0, sun_angle, sun_light, cam_pos: glam::Vec3::new(4.0, 4.0, 18.0), cam_yaw: std::f32::consts::PI, cam_pitch: -0.15, keys: HashSet::new(), cursor_grabbed: false, mouse_delta: (0.0, 0.0) });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event: KeyEvent { state: ElementState::Pressed, physical_key: PhysicalKey::Code(KeyCode::Escape), .. }, .. } => { if state.cursor_grabbed { state.cursor_grabbed = false; let _ = state.window.set_cursor_grab(CursorGrabMode::None); state.window.set_cursor_visible(true); } else { event_loop.exit(); } }
            WindowEvent::KeyboardInput { event: KeyEvent { physical_key: PhysicalKey::Code(k), state: es, .. }, .. } => { match es { ElementState::Pressed => { state.keys.insert(k); } ElementState::Released => { state.keys.remove(&k); } } }
            WindowEvent::MouseInput { state: ElementState::Pressed, button: MouseButton::Left, .. } => { if !state.cursor_grabbed { state.cursor_grabbed = true; let _ = state.window.set_cursor_grab(CursorGrabMode::Locked).or_else(|_| state.window.set_cursor_grab(CursorGrabMode::Confined)); state.window.set_cursor_visible(false); } }
            WindowEvent::Resized(s) => { if s.width > 0 && s.height > 0 { state.surface.configure(&state.device, &wgpu::SurfaceConfiguration { usage: wgpu::TextureUsages::RENDER_ATTACHMENT, format: state.surface_format, width: s.width, height: s.height, present_mode: wgpu::PresentMode::Fifo, alpha_mode: wgpu::CompositeAlphaMode::Auto, view_formats: vec![], desired_maximum_frame_latency: 2 }); } }
            WindowEvent::RedrawRequested => { let now = std::time::Instant::now(); let dt = now.duration_since(state.last_frame).as_secs_f32(); state.last_frame = now; state.time += dt; state.render(dt); state.window.request_redraw(); }
            _ => {}
        }
    }
    fn device_event(&mut self, _el: &ActiveEventLoop, _: winit::event::DeviceId, event: DeviceEvent) {
        let Some(state) = &mut self.state else { return };
        if let DeviceEvent::MouseMotion { delta } = event { if state.cursor_grabbed { state.mouse_delta.0 += delta.0 as f32; state.mouse_delta.1 += delta.1 as f32; } }
    }
}

impl AppState {
    fn render(&mut self, dt: f32) {
        if self.keys.contains(&KeyCode::KeyQ) { self.sun_angle += dt * 0.5; }
        if self.keys.contains(&KeyCode::KeyE) { self.sun_angle -= dt * 0.5; }
        let sun_dir = glam::Vec3::new(self.sun_angle.cos() * 0.6, -self.sun_angle.sin(), -0.8).normalize();
        self.renderer.update_light(self.sun_light, SceneLight::directional(sun_dir, [1.0,0.95,0.80].into(), 3.0));
        self.renderer.set_sky(Some(SkyAtmosphere { sun_direction: sun_dir, ..Default::default() }));

        self.cam_yaw += self.mouse_delta.0 * LOOK_SENS; self.cam_pitch = (self.cam_pitch - self.mouse_delta.1 * LOOK_SENS).clamp(-1.5, 1.5); self.mouse_delta = (0.0, 0.0);
        let (sy, cy) = self.cam_yaw.sin_cos(); let (sp, cp) = self.cam_pitch.sin_cos();
        let forward = glam::Vec3::new(sy * cp, sp, -cy * cp); let right = glam::Vec3::new(cy, 0.0, sy);
        if self.keys.contains(&KeyCode::KeyW) { self.cam_pos += forward * SPEED * dt; }
        if self.keys.contains(&KeyCode::KeyS) { self.cam_pos -= forward * SPEED * dt; }
        if self.keys.contains(&KeyCode::KeyA) { self.cam_pos -= right * SPEED * dt; }
        if self.keys.contains(&KeyCode::KeyD) { self.cam_pos += right * SPEED * dt; }
        if self.keys.contains(&KeyCode::Space)     { self.cam_pos.y += SPEED * dt; }
        if self.keys.contains(&KeyCode::ShiftLeft) { self.cam_pos.y -= SPEED * dt; }
        let size = self.window.inner_size(); let aspect = size.width as f32 / size.height.max(1) as f32;
        self.renderer.set_camera(Camera::perspective(self.cam_pos, self.cam_pos + forward, glam::Vec3::Y, std::f32::consts::FRAC_PI_4, aspect, 0.1, 300.0, self.time));
        let output = match self.surface.get_current_texture() { Ok(t) => t, Err(e) => { log::warn!("{:?}", e); return; } };
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        if let Err(e) = self.renderer.render(&self.device, &self.queue, &view, dt) { log::error!("{:?}", e); }
        output.present();
    }
}
