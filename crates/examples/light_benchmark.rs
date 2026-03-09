//! Light benchmark – warehouse with 150 point lights.
//!
//! Controls: WASD / Space / Shift, mouse drag, Escape.

mod demo_portal;
mod primitives;

use helio_render_v3::{
    Renderer, RendererConfig, AntiAliasingMode, ShadowConfig,
    Camera, HismRegistry, LightId, Material, SceneLight,
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
    light_ids: Vec<LightId>, light_positions: Vec<glam::Vec3>, light_colors: Vec<[f32; 3]>,
    cam_pos: glam::Vec3, cam_yaw: f32, cam_pitch: f32,
    keys: HashSet<KeyCode>, cursor_grabbed: bool, mouse_delta: (f32, f32),
}

fn build_lights() -> Vec<(glam::Vec3, [f32; 3])> {
    let mut lights = Vec::new();
    let colors: &[[f32; 3]] = &[
        [1.0, 0.85, 0.6], [0.6, 0.85, 1.0], [1.0, 0.6, 0.6], [0.6, 1.0, 0.6], [1.0, 1.0, 0.8],
    ];
    for row in 0..10_i32 {
        for col in 0..15_i32 {
            let x = (col as f32 - 7.0) * 4.0;
            let z = (row as f32 - 4.5) * 5.0;
            let color = colors[((row * 15 + col) as usize) % colors.len()];
            lights.push((glam::Vec3::new(x, 4.5, z), color));
        }
    }
    lights
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() { return; }
        let window = Arc::new(event_loop.create_window(Window::default_attributes().with_title("Helio – Light Benchmark").with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32))).expect("w"));
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { backends: wgpu::Backends::all(), ..Default::default() });
        let surface = instance.create_surface(window.clone()).expect("surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: Some(&surface), force_fallback_adapter: false })).expect("adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { label: Some("device"), required_features: wgpu::Features::EXPERIMENTAL_RAY_QUERY | wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS, required_limits: wgpu::Limits::default().using_minimum_supported_acceleration_structure_values(), memory_hints: wgpu::MemoryHints::default(), experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() }, trace: wgpu::Trace::Off })).expect("device");
        let device = Arc::new(device); let queue = Arc::new(queue);
        let caps = surface.get_capabilities(&adapter);
        let fmt = caps.formats.iter().find(|f| f.is_srgb()).copied().unwrap_or(caps.formats[0]);
        let size = window.inner_size();
        surface.configure(&device, &wgpu::SurfaceConfiguration { usage: wgpu::TextureUsages::RENDER_ATTACHMENT, format: fmt, width: size.width, height: size.height, present_mode: wgpu::PresentMode::Fifo, alpha_mode: caps.alpha_modes[0], view_formats: vec![], desired_maximum_frame_latency: 2 });

        let mut renderer = Renderer::new(&device, &queue, RendererConfig { width: size.width, height: size.height, surface_format: fmt, anti_aliasing: AntiAliasingMode::Fxaa, shadows: None, radiance_cascades: None, billboards: None, bloom: None, ssao: None, gpu_driven: false, debug_printout: false }, HismRegistry::new());
        demo_portal::enable_live_dashboard(&mut renderer);

        let mat_floor   = renderer.create_material(&device, &queue, &Material { base_color: [0.30,0.28,0.25,1.0], roughness: 0.9, ..Default::default() });
        let mat_pillar  = renderer.create_material(&device, &queue, &Material { base_color: [0.50,0.48,0.45,1.0], roughness: 0.8, ..Default::default() });
        let mat_crate   = renderer.create_material(&device, &queue, &Material { base_color: [0.55,0.40,0.25,1.0], roughness: 0.85, ..Default::default() });

        // Floor
        let floor = primitives::build_plane(&device, &queue, 0.0, 0.0, 0.0, 30.0, 25.0);
        let hf = renderer.register_hism(floor, mat_floor);
        renderer.add_instance(hf, glam::Mat4::IDENTITY);

        // 6×6 pillar grid
        let pillar_mesh = primitives::build_box(&device, &queue, 0.0, 2.5, 0.0, 0.4, 2.5, 0.4);
        let hp = renderer.register_hism(pillar_mesh, mat_pillar);
        for row in 0..6_i32 {
            for col in 0..6_i32 {
                let x = (col as f32 - 2.5) * 9.0;
                let z = (row as f32 - 2.5) * 8.0;
                renderer.add_instance(hp, glam::Mat4::from_translation(glam::Vec3::new(x, 0.0, z)));
            }
        }

        // Scattered crates
        let crate_mesh = primitives::build_box(&device, &queue, 0.0, 0.5, 0.0, 0.5, 0.5, 0.5);
        let hc = renderer.register_hism(crate_mesh, mat_crate);
        let crate_positions: &[(f32, f32)] = &[
            (-12.0, 8.0), (0.0, 15.0), (8.0, -12.0), (-5.0, -10.0),
            (15.0, 5.0), (-15.0, -5.0), (3.0, 3.0), (-10.0, 12.0),
        ];
        for &(x, z) in crate_positions {
            renderer.add_instance(hc, glam::Mat4::from_translation(glam::Vec3::new(x, 0.0, z)));
        }

        // 150 point lights
        let light_data = build_lights();
        let mut light_ids = Vec::with_capacity(light_data.len());
        let mut light_positions = Vec::with_capacity(light_data.len());
        let mut light_colors = Vec::with_capacity(light_data.len());
        for (pos, color) in &light_data {
            let id = renderer.add_light(SceneLight::point(*pos, glam::Vec3::from(*color), 3.5, 9.0));
            light_ids.push(id);
            light_positions.push(*pos);
            light_colors.push(*color);
        }

        self.state = Some(AppState { window, surface, device, queue, surface_format: fmt, renderer, last_frame: std::time::Instant::now(), time: 0.0, light_ids, light_positions, light_colors, cam_pos: glam::Vec3::new(0.0, 8.0, 40.0), cam_yaw: std::f32::consts::PI, cam_pitch: -0.25, keys: HashSet::new(), cursor_grabbed: false, mouse_delta: (0.0, 0.0) });
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
        // Animate lights slowly
        for (i, id) in self.light_ids.iter().enumerate() {
            let base = self.light_positions[i];
            let offset_y = (self.time * 0.5 + i as f32 * 0.3).sin() * 0.4;
            let pos = base + glam::Vec3::Y * offset_y;
            self.renderer.update_light(*id, SceneLight::point(pos, glam::Vec3::from(self.light_colors[i]), 3.5, 9.0));
        }
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
