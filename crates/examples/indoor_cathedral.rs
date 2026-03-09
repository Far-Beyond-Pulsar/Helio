//! Indoor cathedral – nave, aisles, pillars, altar, pews, chandeliers, stained glass light.
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

// Candle light positions (chandeliers + altar candles)
const CANDLES: &[[f32; 3]] = &[
    [0.0, 19.0, -10.0], [0.0, 19.0, 0.0], [0.0, 19.0, 10.0],
    [-5.0, 9.5, -10.0], [-5.0, 9.5, 0.0], [-5.0, 9.5, 10.0],
    [ 5.0, 9.5, -10.0], [ 5.0, 9.5, 0.0], [ 5.0, 9.5, 10.0],
];

// Stained glass: (x, y, z, r, g, b) – side wall windows
const STAINED: &[(f32, f32, f32, f32, f32, f32)] = &[
    (-10.5, 9.0, -16.0, 1.0, 0.2, 0.1),
    (-10.5, 9.0, -8.0,  0.2, 0.6, 1.0),
    (-10.5, 9.0,  0.0,  0.2, 1.0, 0.3),
    (-10.5, 9.0,  8.0,  1.0, 0.8, 0.1),
    ( 10.5, 9.0, -16.0, 0.6, 0.2, 1.0),
    ( 10.5, 9.0, -8.0,  1.0, 0.5, 0.2),
    ( 10.5, 9.0,  0.0,  0.2, 0.8, 1.0),
    ( 10.5, 9.0,  8.0,  0.8, 1.0, 0.2),
];

fn main() { env_logger::init(); EventLoop::new().expect("el").run_app(&mut App { state: None }).expect("run"); }

struct App { state: Option<AppState> }
struct AppState {
    window: Arc<Window>, surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>,
    surface_format: wgpu::TextureFormat, renderer: Renderer,
    last_frame: std::time::Instant, time: f32,
    candle_ids: Vec<LightId>,
    cam_pos: glam::Vec3, cam_yaw: f32, cam_pitch: f32,
    keys: HashSet<KeyCode>, cursor_grabbed: bool, mouse_delta: (f32, f32),
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() { return; }
        let window = Arc::new(event_loop.create_window(Window::default_attributes().with_title("Helio – Cathedral").with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32))).expect("w"));
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

        let mat_stone     = renderer.create_material(&device, &queue, &Material { base_color: [0.65,0.60,0.52,1.0], roughness: 0.95, ..Default::default() });
        let mat_floor_s   = renderer.create_material(&device, &queue, &Material { base_color: [0.45,0.42,0.38,1.0], roughness: 0.7, metallic: 0.05, ..Default::default() });
        let mat_marble    = renderer.create_material(&device, &queue, &Material { base_color: [0.85,0.82,0.78,1.0], roughness: 0.35, metallic: 0.1, ..Default::default() });
        let mat_wood      = renderer.create_material(&device, &queue, &Material { base_color: [0.40,0.26,0.14,1.0], roughness: 0.8, ..Default::default() });
        let mat_gold      = renderer.create_material(&device, &queue, &Material { base_color: [0.85,0.70,0.25,1.0], roughness: 0.3, metallic: 0.9, ..Default::default() });
        let mat_chandelier= renderer.create_material(&device, &queue, &Material { base_color: [0.70,0.65,0.50,1.0], roughness: 0.4, metallic: 0.7, ..Default::default() });

        // Floor (wide: 11.0 half-size)
        let floor = primitives::build_plane(&device, &queue, 0.0, 0.0, 0.0, 11.0, 22.0);
        let hfl = renderer.register_hism(floor, mat_floor_s);
        renderer.add_instance(hfl, glam::Mat4::IDENTITY);

        // Nave ceiling
        let nave_ceil = primitives::build_box(&device, &queue, 0.0, 21.0, 0.0, 6.0, 0.5, 22.0);
        let hnc = renderer.register_hism(nave_ceil, mat_stone.clone());
        renderer.add_instance(hnc, glam::Mat4::IDENTITY);

        // Aisle ceilings (left + right)
        for sx in &[-8.5f32, 8.5] {
            let ac = primitives::build_box(&device, &queue, *sx, 11.0, 0.0, 2.5, 0.4, 22.0);
            let h = renderer.register_hism(ac, mat_stone.clone());
            renderer.add_instance(h, glam::Mat4::IDENTITY);
        }

        // Outer walls (left/right)
        for sx in &[-11.0f32, 11.0] {
            let ow = primitives::build_box(&device, &queue, *sx, 6.0, 0.0, 0.4, 6.0, 22.0);
            let h = renderer.register_hism(ow, mat_stone.clone());
            renderer.add_instance(h, glam::Mat4::IDENTITY);
        }
        // Front / back walls
        let front = primitives::build_box(&device, &queue, 0.0, 11.0, -22.0, 11.0, 11.0, 0.4);
        let hfw = renderer.register_hism(front, mat_stone.clone());
        renderer.add_instance(hfw, glam::Mat4::IDENTITY);
        let back = primitives::build_box(&device, &queue, 0.0, 11.0, 22.0, 11.0, 11.0, 0.4);
        let hbw = renderer.register_hism(back, mat_stone.clone());
        renderer.add_instance(hbw, glam::Mat4::IDENTITY);

        // Inner nave wall segments (7 arched sections per side, simplified as boxes)
        for side in &[-6.0f32, 6.0] {
            for i in 0..7_i32 {
                let z = i as f32 * 6.0 - 18.0;
                let seg = primitives::build_box(&device, &queue, *side, 11.0, z, 0.5, 11.0, 2.0);
                let h = renderer.register_hism(seg, mat_stone.clone());
                renderer.add_instance(h, glam::Mat4::IDENTITY);
            }
        }

        // Pillar pairs at z intervals
        for i in 0..6_i32 {
            let pz = i as f32 * 7.0 - 17.5;
            for px in &[-5.5f32, 5.5] {
                let pillar = primitives::build_box(&device, &queue, *px, 10.5, pz, 0.55, 10.5, 0.55);
                let hp = renderer.register_hism(pillar, mat_marble.clone());
                renderer.add_instance(hp, glam::Mat4::IDENTITY);
            }
        }

        // Altar step + plinth
        let step  = primitives::build_box(&device, &queue, 0.0, 0.15, -18.0, 3.5, 0.15, 2.5);
        let hs    = renderer.register_hism(step, mat_marble.clone());
        renderer.add_instance(hs, glam::Mat4::IDENTITY);
        let plinth= primitives::build_box(&device, &queue, 0.0, 0.7, -19.0, 1.0, 0.7, 0.7);
        let hpl   = renderer.register_hism(plinth, mat_marble.clone());
        renderer.add_instance(hpl, glam::Mat4::IDENTITY);
        // Cross: vertical + horizontal bar
        let cross_v = primitives::build_box(&device, &queue, 0.0, 2.5, -19.3, 0.08, 1.5, 0.08);
        let hcv = renderer.register_hism(cross_v, mat_gold.clone());
        renderer.add_instance(hcv, glam::Mat4::IDENTITY);
        let cross_h = primitives::build_box(&device, &queue, 0.0, 3.2, -19.3, 0.55, 0.08, 0.08);
        let hch = renderer.register_hism(cross_h, mat_gold.clone());
        renderer.add_instance(hch, glam::Mat4::IDENTITY);

        // Pews: left + right of nave, 8 rows
        let pew_mesh = primitives::build_box(&device, &queue, 0.0, 0.45, 0.0, 1.8, 0.45, 0.4);
        let hpew = renderer.register_hism(pew_mesh, mat_wood.clone());
        for row in 0..8_i32 {
            let pz = row as f32 * 4.0 - 14.0;
            for side in &[-3.0f32, 3.0] {
                renderer.add_instance(hpew, glam::Mat4::from_translation(glam::Vec3::new(*side, 0.0, pz)));
            }
        }

        // Chandeliers: rod + disc hanging from nave ceiling
        let rod  = primitives::build_box(&device, &queue, 0.0, 0.0, 0.0, 0.05, 1.5, 0.05);
        let disc = primitives::build_box(&device, &queue, 0.0, 0.0, 0.0, 0.6, 0.05, 0.6);
        let hrod  = renderer.register_hism(rod,  mat_chandelier.clone());
        let hdisc = renderer.register_hism(disc, mat_chandelier);
        for i in 0..3_i32 {
            let cz = i as f32 * 10.0 - 10.0;
            renderer.add_instance(hrod,  glam::Mat4::from_translation(glam::Vec3::new(0.0, 18.5, cz)));
            renderer.add_instance(hdisc, glam::Mat4::from_translation(glam::Vec3::new(0.0, 17.0, cz)));
        }

        // Lights: flickering candles (stored for animation) + stained glass
        let mut candle_ids = Vec::new();
        for &p in CANDLES {
            let id = renderer.add_light(SceneLight::point(p.into(), [1.0, 0.92, 0.78].into(), 8.0, 22.0));
            candle_ids.push(id);
        }
        // Altar candle flames
        for &(x, z) in [(-0.6f32, -18.5f32), (0.6, -18.5)].iter() {
            let id = renderer.add_light(SceneLight::point([x, 1.7, z].into(), [1.0, 0.6, 0.15].into(), 1.2, 4.0));
            candle_ids.push(id);
        }
        // Stained glass window lights
        for &(x, y, z, r, g, b) in STAINED {
            renderer.add_light(SceneLight::point([x, y, z].into(), [r, g, b].into(), 1.8, 8.0));
        }

        self.state = Some(AppState { window, surface, device, queue, surface_format: fmt, renderer, last_frame: std::time::Instant::now(), time: 0.0, candle_ids, cam_pos: glam::Vec3::new(0.0, 3.0, 18.0), cam_yaw: std::f32::consts::PI, cam_pitch: -0.1, keys: HashSet::new(), cursor_grabbed: false, mouse_delta: (0.0, 0.0) });
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
        // Animate chandelier candles with flicker
        for (i, &id) in self.candle_ids.iter().enumerate() {
            let flicker = 0.85 + 0.15 * (self.time * 7.3 + i as f32 * 1.7).sin()
                        + 0.05 * (self.time * 13.1 + i as f32 * 3.3).cos();
            let (color, base_int, range) = if i < CANDLES.len() {
                ([1.0f32, 0.92, 0.78], 8.0f32, 22.0f32)
            } else {
                ([1.0f32, 0.6,  0.15], 1.2f32,  4.0f32)
            };
            let p = if i < CANDLES.len() { CANDLES[i] } else {
                let idx = i - CANDLES.len();
                [[-0.6f32, 1.7, -18.5], [0.6, 1.7, -18.5]][idx]
            };
            self.renderer.update_light(id, SceneLight::point(p.into(), color.into(), base_int * flicker, range));
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
        self.renderer.set_camera(Camera::perspective(self.cam_pos, self.cam_pos + forward, glam::Vec3::Y, std::f32::consts::FRAC_PI_4, aspect, 0.1, 200.0, self.time));
        let output = match self.surface.get_current_texture() { Ok(t) => t, Err(e) => { log::warn!("{:?}", e); return; } };
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        if let Err(e) = self.renderer.render(&self.device, &self.queue, &view, dt) { log::error!("{:?}", e); }
        output.present();
    }
}
