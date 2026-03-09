//! AO and AA demo – floor, spheres (as boxes), cubes; SSAO + bloom enabled.
//! Tab cycles the displayed AA mode label (v3 AA mode is fixed at init time).
//!
//! Controls: WASD / Space / Shift, mouse drag, Escape.

mod demo_portal;
mod primitives;

use helio_render_v3::{
    Renderer, RendererConfig, AntiAliasingMode, ShadowConfig, BloomConfig, SsaoConfig,
    Camera, HismRegistry, Material, SceneLight,
};
use winit::{
    application::ApplicationHandler, event::*, event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey}, window::{Window, WindowId, CursorGrabMode},
};
use std::collections::HashSet;
use std::sync::Arc;

const SPEED: f32 = 5.0;
const LOOK_SENS: f32 = 0.003;
const AA_LABELS: &[&str] = &["None", "FXAA", "TAA"];

fn main() { env_logger::init(); EventLoop::new().expect("el").run_app(&mut App { state: None }).expect("run"); }

struct App { state: Option<AppState> }
struct AppState {
    window: Arc<Window>, surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>,
    surface_format: wgpu::TextureFormat, renderer: Renderer,
    last_frame: std::time::Instant, time: f32,
    aa_label_idx: usize,
    cam_pos: glam::Vec3, cam_yaw: f32, cam_pitch: f32,
    keys: HashSet<KeyCode>, cursor_grabbed: bool, mouse_delta: (f32, f32),
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() { return; }
        let window = Arc::new(event_loop.create_window(Window::default_attributes().with_title("Helio – AO / AA Demo").with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32))).expect("w"));
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { backends: wgpu::Backends::all(), ..Default::default() });
        let surface = instance.create_surface(window.clone()).expect("surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: Some(&surface), force_fallback_adapter: false })).expect("adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { label: Some("device"), required_features: wgpu::Features::EXPERIMENTAL_RAY_QUERY | wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS, required_limits: wgpu::Limits::default().using_minimum_supported_acceleration_structure_values(), memory_hints: wgpu::MemoryHints::default(), experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() }, trace: wgpu::Trace::Off })).expect("device");
        let device = Arc::new(device); let queue = Arc::new(queue);
        let caps = surface.get_capabilities(&adapter);
        let fmt = caps.formats.iter().find(|f| f.is_srgb()).copied().unwrap_or(caps.formats[0]);
        let size = window.inner_size();
        surface.configure(&device, &wgpu::SurfaceConfiguration { usage: wgpu::TextureUsages::RENDER_ATTACHMENT, format: fmt, width: size.width, height: size.height, present_mode: wgpu::PresentMode::Fifo, alpha_mode: caps.alpha_modes[0], view_formats: vec![], desired_maximum_frame_latency: 2 });

        let mut renderer = Renderer::new(&device, &queue, RendererConfig {
            width: size.width, height: size.height, surface_format: fmt,
            anti_aliasing: AntiAliasingMode::Fxaa,
            shadows: Some(ShadowConfig { atlas_size: 1024, max_shadow_lights: 4 }),
            radiance_cascades: None, billboards: None,
            bloom: Some(BloomConfig { threshold: 0.8, intensity: 0.3 }),
            ssao: Some(SsaoConfig { radius: 0.5, bias: 0.025, power: 1.5, samples: 16 }),
            gpu_driven: false, debug_printout: false,
        }, HismRegistry::new());
        demo_portal::enable_live_dashboard(&mut renderer);

        let mat_floor    = renderer.create_material(&device, &queue, &Material { base_color: [0.55,0.50,0.45,1.0], roughness: 0.85, ..Default::default() });
        let mat_sphere   = renderer.create_material(&device, &queue, &Material { base_color: [0.80,0.30,0.20,1.0], roughness: 0.4, metallic: 0.0, ..Default::default() });
        let mat_cube     = renderer.create_material(&device, &queue, &Material { base_color: [0.25,0.45,0.70,1.0], roughness: 0.3, metallic: 0.5, ..Default::default() });
        let mat_cube2    = renderer.create_material(&device, &queue, &Material { base_color: [0.30,0.65,0.35,1.0], roughness: 0.6, ..Default::default() });
        let mat_emissive = renderer.create_material(&device, &queue, &Material { base_color: [1.0,0.9,0.5,1.0], roughness: 1.0, emissive_color: [3.0, 2.7, 1.5], ..Default::default() });

        let floor = primitives::build_plane(&device, &queue, 0.0, 0.0, 0.0, 12.0, 12.0);
        let hf = renderer.register_hism(floor, mat_floor);
        renderer.add_instance(hf, glam::Mat4::IDENTITY);

        // Spheres (approximated as boxes)
        let positions: &[(f32, f32, f32)] = &[(-4.0, 0.8, 2.0), (-1.5, 0.8, -3.0), (2.5, 0.8, 1.5), (4.5, 0.8, -2.0)];
        for &(x, y, z) in positions {
            let m = primitives::build_box(&device, &queue, x, y, z, 0.8, 0.8, 0.8);
            let h = renderer.register_hism(m, mat_sphere.clone());
            renderer.add_instance(h, glam::Mat4::IDENTITY);
        }

        // Cubes
        let cube1 = primitives::build_box(&device, &queue, 0.5, 0.6, 0.0, 0.6, 0.6, 0.6);
        let h1 = renderer.register_hism(cube1, mat_cube);
        renderer.add_instance(h1, glam::Mat4::IDENTITY);

        let cube2 = primitives::build_box(&device, &queue, -2.5, 1.2, 1.0, 0.5, 1.2, 0.5);
        let h2 = renderer.register_hism(cube2, mat_cube2);
        renderer.add_instance(h2, glam::Mat4::IDENTITY);

        // Small emissive cube to drive bloom
        let glow = primitives::build_box(&device, &queue, 3.0, 1.5, -1.5, 0.3, 0.3, 0.3);
        let hg = renderer.register_hism(glow, mat_emissive);
        renderer.add_instance(hg, glam::Mat4::IDENTITY);

        renderer.add_light(SceneLight::point([ 0.0, 5.0,  0.0].into(), [1.0,0.95,0.85].into(), 8.0, 14.0));
        renderer.add_light(SceneLight::point([-4.0, 3.0, -4.0].into(), [0.7,0.8,1.0].into(), 4.0, 10.0));
        renderer.add_light(SceneLight::point([ 4.0, 3.0, -4.0].into(), [1.0,0.8,0.7].into(), 4.0, 10.0));

        self.state = Some(AppState { window, surface, device, queue, surface_format: fmt, renderer, last_frame: std::time::Instant::now(), time: 0.0, aa_label_idx: 1, cam_pos: glam::Vec3::new(0.0, 4.0, 10.0), cam_yaw: std::f32::consts::PI, cam_pitch: -0.3, keys: HashSet::new(), cursor_grabbed: false, mouse_delta: (0.0, 0.0) });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event: KeyEvent { state: ElementState::Pressed, physical_key: PhysicalKey::Code(KeyCode::Escape), .. }, .. } => { if state.cursor_grabbed { state.cursor_grabbed = false; let _ = state.window.set_cursor_grab(CursorGrabMode::None); state.window.set_cursor_visible(true); } else { event_loop.exit(); } }
            WindowEvent::KeyboardInput { event: KeyEvent { state: ElementState::Pressed, physical_key: PhysicalKey::Code(KeyCode::Tab), .. }, .. } => {
                state.aa_label_idx = (state.aa_label_idx + 1) % AA_LABELS.len();
                log::info!("AA mode label (display only): {}", AA_LABELS[state.aa_label_idx]);
                state.window.set_title(&format!("Helio – AO/AA Demo  [AA: {}]", AA_LABELS[state.aa_label_idx]));
            }
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
        self.renderer.set_camera(Camera::perspective(self.cam_pos, self.cam_pos + forward, glam::Vec3::Y, std::f32::consts::FRAC_PI_4, aspect, 0.1, 100.0, self.time));
        let output = match self.surface.get_current_texture() { Ok(t) => t, Err(e) => { log::warn!("{:?}", e); return; } };
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        if let Err(e) = self.renderer.render(&self.device, &self.queue, &view, dt) { log::error!("{:?}", e); }
        output.present();
    }
}
