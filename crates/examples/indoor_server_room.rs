//! Indoor server room – raised floor, server racks, cable trays, cold-aisle containment.
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

const SPEED: f32 = 6.0;
const LOOK_SENS: f32 = 0.003;

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
        let window = Arc::new(event_loop.create_window(Window::default_attributes().with_title("Helio – Server Room").with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32))).expect("w"));
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { backends: wgpu::Backends::all(), ..Default::default() });
        let surface = instance.create_surface(window.clone()).expect("surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: Some(&surface), force_fallback_adapter: false })).expect("adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { label: Some("device"), required_features: wgpu::Features::EXPERIMENTAL_RAY_QUERY | wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS, required_limits: wgpu::Limits::default().using_minimum_supported_acceleration_structure_values(), memory_hints: wgpu::MemoryHints::default(), experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() }, trace: wgpu::Trace::Off })).expect("device");
        let device = Arc::new(device); let queue = Arc::new(queue);
        let caps = surface.get_capabilities(&adapter);
        let fmt = caps.formats.iter().find(|f| f.is_srgb()).copied().unwrap_or(caps.formats[0]);
        let size = window.inner_size();
        surface.configure(&device, &wgpu::SurfaceConfiguration { usage: wgpu::TextureUsages::RENDER_ATTACHMENT, format: fmt, width: size.width, height: size.height, present_mode: wgpu::PresentMode::Fifo, alpha_mode: caps.alpha_modes[0], view_formats: vec![], desired_maximum_frame_latency: 2 });

        let mut renderer = Renderer::new(&device, &queue, RendererConfig { width: size.width, height: size.height, surface_format: fmt, anti_aliasing: AntiAliasingMode::Fxaa, shadows: Some(ShadowConfig { atlas_size: 1024, max_shadow_lights: 8 }), radiance_cascades: None, billboards: None, bloom: None, ssao: None, gpu_driven: false, debug_printout: false }, HismRegistry::new());
        demo_portal::enable_live_dashboard(&mut renderer);

        let mat_floor   = renderer.create_material(&device, &queue, &Material { base_color: [0.55,0.55,0.55,1.0], roughness: 0.4, metallic: 0.2, ..Default::default() });
        let mat_tile    = renderer.create_material(&device, &queue, &Material { base_color: [0.35,0.35,0.38,1.0], roughness: 0.7, metallic: 0.1, ..Default::default() });
        let mat_ceiling = renderer.create_material(&device, &queue, &Material { base_color: [0.80,0.80,0.82,1.0], roughness: 0.9, ..Default::default() });
        let mat_wall    = renderer.create_material(&device, &queue, &Material { base_color: [0.72,0.72,0.70,1.0], roughness: 0.85, ..Default::default() });
        let mat_rack    = renderer.create_material(&device, &queue, &Material { base_color: [0.12,0.12,0.14,1.0], roughness: 0.4, metallic: 0.8, ..Default::default() });
        let mat_cable   = renderer.create_material(&device, &queue, &Material { base_color: [0.18,0.18,0.20,1.0], roughness: 0.8, ..Default::default() });
        let mat_contain = renderer.create_material(&device, &queue, &Material { base_color: [0.25,0.55,0.35,1.0], roughness: 0.7, ..Default::default() });
        let mat_door    = renderer.create_material(&device, &queue, &Material { base_color: [0.40,0.38,0.35,1.0], roughness: 0.6, metallic: 0.3, ..Default::default() });

        let hw = 12.0f32;  // room half-width / half-depth
        let rh = 4.0f32;   // room height

        // Room shell
        let floor   = primitives::build_plane(&device, &queue, 0.0, 0.0, 0.0, hw, hw);
        let ceiling  = primitives::build_box(&device, &queue, 0.0, rh + 0.05, 0.0, hw, 0.05, hw);
        let wall_n   = primitives::build_box(&device, &queue, 0.0, rh * 0.5, -hw, hw, rh * 0.5, 0.05);
        let wall_s   = primitives::build_box(&device, &queue, 0.0, rh * 0.5,  hw, hw, rh * 0.5, 0.05);
        let wall_e   = primitives::build_box(&device, &queue,  hw, rh * 0.5, 0.0, 0.05, rh * 0.5, hw);
        let wall_w   = primitives::build_box(&device, &queue, -hw, rh * 0.5, 0.0, 0.05, rh * 0.5, hw);
        let hfl  = renderer.register_hism(floor,   mat_floor.clone());  renderer.add_instance(hfl,  glam::Mat4::IDENTITY);
        let hce  = renderer.register_hism(ceiling, mat_ceiling);        renderer.add_instance(hce,  glam::Mat4::IDENTITY);
        let hwn  = renderer.register_hism(wall_n,  mat_wall.clone());   renderer.add_instance(hwn,  glam::Mat4::IDENTITY);
        let hws  = renderer.register_hism(wall_s,  mat_wall.clone());   renderer.add_instance(hws,  glam::Mat4::IDENTITY);
        let hwe  = renderer.register_hism(wall_e,  mat_wall.clone());   renderer.add_instance(hwe,  glam::Mat4::IDENTITY);
        let hww  = renderer.register_hism(wall_w,  mat_wall.clone());   renderer.add_instance(hww,  glam::Mat4::IDENTITY);

        // Raised floor tiles (4×4 grid of panels)
        let tile = primitives::build_box(&device, &queue, 0.0, 0.05, 0.0, 1.45, 0.05, 1.45);
        let htile = renderer.register_hism(tile, mat_tile);
        for row in -5_i32..5 {
            for col in -5_i32..5 {
                let tx = col as f32 * 3.0 + 1.5;
                let tz = row as f32 * 3.0 + 1.5;
                renderer.add_instance(htile, glam::Mat4::from_translation(glam::Vec3::new(tx, 0.0, tz)));
            }
        }

        // Server racks (4 rows × 5 racks)
        let rack = primitives::build_box(&device, &queue, 0.0, 1.0, 0.0, 0.4, 1.0, 0.7);
        let hrack = renderer.register_hism(rack, mat_rack.clone());
        for row in 0..4_i32 {
            let rz = row as f32 * 4.5 - 6.75;
            for col in 0..5_i32 {
                let rx = col as f32 * 2.2 - 4.4;
                renderer.add_instance(hrack, glam::Mat4::from_translation(glam::Vec3::new(rx, 0.12, rz)));
            }
            // Row overhead light
            let mid_x = 0.0;
            renderer.add_light(SceneLight::point([mid_x, 3.6, rz].into(), [0.85,0.90,1.0].into(), 5.0, 10.0));
            renderer.add_light(SceneLight::point([mid_x - 3.0, 3.6, rz].into(), [0.85,0.90,1.0].into(), 3.5, 8.0));
            renderer.add_light(SceneLight::point([mid_x + 3.0, 3.6, rz].into(), [0.85,0.90,1.0].into(), 3.5, 8.0));
        }

        // Cable trays (horizontal channels near ceiling)
        let cable = primitives::build_box(&device, &queue, 0.0, 3.5, 0.0, hw - 1.0, 0.08, 0.3);
        let hcable = renderer.register_hism(cable, mat_cable);
        for z_off in &[-4.5f32, -1.5, 1.5, 4.5] {
            renderer.add_instance(hcable, glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, *z_off)));
        }

        // Cold-aisle containment panels (2 walls between rack rows)
        let contain = primitives::build_box(&device, &queue, 0.0, 2.0, 0.0, 5.5, 2.0, 0.05);
        let hcont = renderer.register_hism(contain, mat_contain);
        renderer.add_instance(hcont, glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, -2.2)));
        renderer.add_instance(hcont, glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0,  2.2)));

        // Door frame + panel
        let door_frame = primitives::build_box(&device, &queue, -hw + 0.1, rh * 0.5, -6.0, 0.15, rh * 0.5, 0.6);
        let hdf = renderer.register_hism(door_frame, mat_door.clone());
        renderer.add_instance(hdf, glam::Mat4::IDENTITY);
        let door_panel = primitives::build_box(&device, &queue, -hw + 0.25, 1.1, -6.0, 0.05, 1.1, 0.55);
        let hdp = renderer.register_hism(door_panel, mat_door);
        renderer.add_instance(hdp, glam::Mat4::IDENTITY);

        // Emergency point lights (red, near exit)
        renderer.add_light(SceneLight::point([-hw + 0.5, 3.2, -6.0].into(), [1.0,0.1,0.1].into(), 1.0, 3.5));
        renderer.add_light(SceneLight::point([ hw - 0.5, 3.2,  6.0].into(), [1.0,0.1,0.1].into(), 1.0, 3.5));

        self.state = Some(AppState { window, surface, device, queue, surface_format: fmt, renderer, last_frame: std::time::Instant::now(), time: 0.0, cam_pos: glam::Vec3::new(0.0, 1.8, 11.0), cam_yaw: std::f32::consts::PI, cam_pitch: -0.05, keys: HashSet::new(), cursor_grabbed: false, mouse_delta: (0.0, 0.0) });
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
        self.renderer.set_camera(Camera::perspective(self.cam_pos, self.cam_pos + forward, glam::Vec3::Y, std::f32::consts::FRAC_PI_4, aspect, 0.1, 100.0, self.time));
        let output = match self.surface.get_current_texture() { Ok(t) => t, Err(e) => { log::warn!("{:?}", e); return; } };
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        if let Err(e) = self.renderer.render(&self.device, &self.queue, &view, dt) { log::error!("{:?}", e); }
        output.present();
    }
}
