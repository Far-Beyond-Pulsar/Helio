//! Space station – large modular station built from boxes, no sky atmosphere.
//!
//! Controls: WASD / Space / Shift, mouse drag, Escape.

mod demo_portal;
mod primitives;

use helio_render_v3::{
    Renderer, RendererConfig, AntiAliasingMode, ShadowConfig,
    Camera, HismRegistry, Material, SceneLight,
};
use winit::{
    application::ApplicationHandler, event::*, event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey}, window::{Window, WindowId, CursorGrabMode},
};
use std::collections::HashSet;
use std::sync::Arc;

const SPEED: f32 = 15.0;
const LOOK_SENS: f32 = 0.003;

/// Modular part descriptor (offset_x, offset_y, offset_z, half_x, half_y, half_z, material_idx)
struct Part { x: f32, y: f32, z: f32, hx: f32, hy: f32, hz: f32, mat: usize }

fn part(x: f32, y: f32, z: f32, hx: f32, hy: f32, hz: f32, mat: usize) -> Part {
    Part { x, y, z, hx, hy, hz, mat }
}

fn build_station() -> Vec<Part> {
    let mut p = Vec::new();
    // Central hub
    p.push(part( 0.0, 0.0,  0.0, 12.0, 5.0, 12.0, 0));   // main deck
    p.push(part( 0.0, 8.0,  0.0,  7.0, 3.0,  7.0, 0));   // upper tier
    p.push(part( 0.0,-8.0,  0.0,  7.0, 3.0,  7.0, 1));   // lower tier

    // Four radial arms
    for &(sx, sz) in &[(1f32, 0f32), (-1f32, 0f32), (0f32, 1f32), (0f32, -1f32)] {
        let ax = sx * 28.0; let az = sz * 28.0;
        let ahx = if sz == 0.0 { 15.0 } else { 2.5 };
        let ahz = if sx == 0.0 { 15.0 } else { 2.5 };
        p.push(part(ax, 0.0, az, ahx, 2.5, ahz, 1));    // arm
        // End module
        let ex = sx * 45.0; let ez = sz * 45.0;
        p.push(part(ex, 0.0, ez, 5.0, 5.0, 5.0, 0));    // docking ring
        p.push(part(ex, 0.0, ez, 3.0, 3.0, 3.0, 2));    // inner core
        // Attachment ring
        p.push(part(ax, 4.5, az, ahx, 0.4, ahz, 2));
        p.push(part(ax,-4.5, az, ahx, 0.4, ahz, 2));
    }

    // Solar panel booms (two perpendicular sets)
    for &boom_z in &[-1f32, 1f32] {
        for i in 0..4_i32 {
            let bx = (i as f32 - 1.5) * 16.0;
            let by = if boom_z > 0.0 { 12.0 } else { -12.0 };
            // Boom strut
            p.push(part(bx, by, boom_z * 55.0, 1.0, 0.4, 22.0, 1));
            // Solar panels
            for &px in &[-1f32, 1f32] {
                p.push(part(bx + px * 7.0, by, boom_z * 55.0, 5.5, 0.1, 11.0, 3));
            }
        }
    }

    // Docking bay on +X arm extended
    p.push(part(50.0, -6.0, 0.0, 4.0, 2.5, 8.0, 0));
    p.push(part(50.0, -6.0, 0.0, 2.5, 0.2, 8.2, 2));

    // Central spine (vertical column)
    p.push(part( 0.0, 20.0, 0.0, 3.0, 12.0, 3.0, 2));
    p.push(part( 0.0,-22.0, 0.0, 3.0, 14.0, 3.0, 1));
    // Top observation dome (flat box approximation)
    p.push(part( 0.0, 32.0, 0.0, 5.0, 2.0, 5.0, 0));
    // Bottom thruster cluster
    p.push(part( 0.0,-36.0, 0.0, 4.0, 0.8, 4.0, 2));
    for &(tx, tz) in &[(-2f32,-2f32),(2f32,-2f32),(-2f32,2f32),(2f32,2f32)] {
        p.push(part(tx, -37.5, tz, 0.6, 1.5, 0.6, 1));
    }

    // Cross-brace rings at +/- 14 on Z
    for &rz in &[-14f32, 14f32] {
        p.push(part( 12.0, 0.0, rz, 0.3, 0.3, 0.3, 2));
        p.push(part(-12.0, 0.0, rz, 0.3, 0.3, 0.3, 2));
        p.push(part( 0.0, 0.0, rz + 12.0, 0.3, 0.3, 0.3, 2));
        p.push(part( 0.0, 0.0, rz - 12.0, 0.3, 0.3, 0.3, 2));
    }

    p
}

fn main() { env_logger::init(); EventLoop::new().expect("el").run_app(&mut App { state: None }).expect("run"); }

struct App { state: Option<AppState> }
struct AppState {
    window: Arc<Window>, surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>,
    surface_format: wgpu::TextureFormat, renderer: Renderer,
    last_frame: std::time::Instant, time: f32,
    cam_pos: glam::Vec3, cam_yaw: f32, cam_pitch: f32,
    keys: HashSet<KeyCode>, cursor_grabbed: bool, mouse_delta: (f32, f32),
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() { return; }
        let window = Arc::new(event_loop.create_window(Window::default_attributes().with_title("Helio – Space Station").with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32))).expect("w"));
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { backends: wgpu::Backends::all(), ..Default::default() });
        let surface = instance.create_surface(window.clone()).expect("surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: Some(&surface), force_fallback_adapter: false })).expect("adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { label: Some("device"), required_features: wgpu::Features::EXPERIMENTAL_RAY_QUERY | wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS, required_limits: wgpu::Limits::default().using_minimum_supported_acceleration_structure_values(), memory_hints: wgpu::MemoryHints::default(), experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() }, trace: wgpu::Trace::Off })).expect("device");
        let device = Arc::new(device); let queue = Arc::new(queue);
        let caps = surface.get_capabilities(&adapter);
        let fmt = caps.formats.iter().find(|f| f.is_srgb()).copied().unwrap_or(caps.formats[0]);
        let size = window.inner_size();
        surface.configure(&device, &wgpu::SurfaceConfiguration { usage: wgpu::TextureUsages::RENDER_ATTACHMENT, format: fmt, width: size.width, height: size.height, present_mode: wgpu::PresentMode::Fifo, alpha_mode: caps.alpha_modes[0], view_formats: vec![], desired_maximum_frame_latency: 2 });

        let mut renderer = Renderer::new(&device, &queue, RendererConfig { width: size.width, height: size.height, surface_format: fmt, anti_aliasing: AntiAliasingMode::Fxaa, shadows: Some(ShadowConfig { atlas_size: 2048, max_shadow_lights: 8 }), radiance_cascades: None, billboards: None, bloom: None, ssao: None, gpu_driven: false, debug_printout: false }, HismRegistry::new());
        demo_portal::enable_live_dashboard(&mut renderer);

        let materials = [
            renderer.create_material(&device, &queue, &Material { base_color: [0.70,0.68,0.65,1.0], roughness: 0.5, metallic: 0.5, ..Default::default() }), // hull plate
            renderer.create_material(&device, &queue, &Material { base_color: [0.40,0.40,0.45,1.0], roughness: 0.3, metallic: 0.9, ..Default::default() }), // structural metal
            renderer.create_material(&device, &queue, &Material { base_color: [0.60,0.62,0.65,1.0], roughness: 0.6, metallic: 0.4, ..Default::default() }), // detail panels
            renderer.create_material(&device, &queue, &Material { base_color: [0.10,0.12,0.10,1.0], roughness: 0.1, metallic: 0.0, ..Default::default() }), // solar panels
        ];

        let parts = build_station();
        for p in &parts {
            let mat = materials[p.mat.min(3)].clone();
            let mesh = primitives::build_box(&device, &queue, p.x, p.y, p.z, p.hx, p.hy, p.hz);
            let h = renderer.register_hism(mesh, mat);
            renderer.add_instance(h, glam::Mat4::IDENTITY);
        }

        // Docking bay lights (warm white)
        renderer.add_light(SceneLight::point([ 45.0,  0.0, 12.0].into(), [1.0,0.95,0.85].into(), 6.0, 20.0));
        renderer.add_light(SceneLight::point([-45.0,  0.0, 12.0].into(), [1.0,0.95,0.85].into(), 6.0, 20.0));
        renderer.add_light(SceneLight::point([ 0.0,  0.0, 45.0].into(), [1.0,0.95,0.85].into(), 6.0, 20.0));
        renderer.add_light(SceneLight::point([ 0.0,  0.0,-45.0].into(), [1.0,0.95,0.85].into(), 6.0, 20.0));
        // Hub core lights (cool blue-white)
        renderer.add_light(SceneLight::point([ 0.0, 10.0, 0.0].into(), [0.85,0.90,1.0].into(), 8.0, 30.0));
        renderer.add_light(SceneLight::point([ 0.0,-10.0, 0.0].into(), [0.85,0.90,1.0].into(), 6.0, 25.0));
        // Distant star directional
        renderer.add_light(SceneLight::directional([-0.5, -0.3, -0.8].into(), [1.0,0.98,0.95].into(), 2.5));

        self.state = Some(AppState { window, surface, device, queue, surface_format: fmt, renderer, last_frame: std::time::Instant::now(), time: 0.0, cam_pos: glam::Vec3::new(0.0, 55.0, 175.0), cam_yaw: std::f32::consts::PI, cam_pitch: -0.25, keys: HashSet::new(), cursor_grabbed: false, mouse_delta: (0.0, 0.0) });
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
        self.renderer.set_camera(Camera::perspective(self.cam_pos, self.cam_pos + forward, glam::Vec3::Y, std::f32::consts::FRAC_PI_4, aspect, 0.5, 1000.0, self.time));
        let output = match self.surface.get_current_texture() { Ok(t) => t, Err(e) => { log::warn!("{:?}", e); return; } };
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        if let Err(e) = self.renderer.render(&self.device, &self.queue, &view, dt) { log::error!("{:?}", e); }
        output.present();
    }
}
