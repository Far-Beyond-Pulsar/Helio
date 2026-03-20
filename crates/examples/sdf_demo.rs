//! SDF Clipmap Demo — helio v3
//!
//! Fullscreen volumetric ray march through a sparse SDF brick atlas.
//! No triangle meshes are rendered — the entire scene is SDF geometry:
//!   - Procedural rolling terrain via FBM noise (CPU + GPU match)
//!   - A few sphere/capsule edits to show boolean operations
//!
//! The clipmap has 8 nested LOD levels (level 0 = finest at 0.5 wu/voxel).
//! Only the newly-visible toroidal shell is reclassified each frame -> O(1) CPU.
//!
//! Controls:
//!   WASD / Space / Shift  -- free-fly (auto-orbit disabled on first input)
//!   Mouse drag            -- look (click window to grab cursor)
//!   Escape                -- release cursor / exit

use helio::{required_wgpu_features, required_wgpu_limits, Camera, Renderer, RendererConfig};
use helio_pass_sdf::{
    SdfClipmapPass,
    edit_list::{SdfEdit, BooleanOp},
    terrain::TerrainConfig,
    primitives::{SdfShapeType, SdfShapeParams},
};
use glam::{Vec3, Mat4};
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId, CursorGrabMode},
};
use std::collections::HashSet;
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────────
// APP BOILERPLATE
// ─────────────────────────────────────────────────────────────────────────────

fn main() {
    env_logger::init();
    EventLoop::new().expect("event loop")
        .run_app(&mut App { state: None })
        .expect("run");
}

struct App { state: Option<AppState> }

struct AppState {
    window:         Arc<Window>,
    surface:        wgpu::Surface<'static>,
    device:         Arc<wgpu::Device>,
    surface_format: wgpu::TextureFormat,
    renderer:       Renderer,
    last_frame:     std::time::Instant,
    // Camera
    cam_pos:        Vec3,
    cam_yaw:        f32,
    cam_pitch:      f32,
    orbit_mode:     bool,
    elapsed:        f32,
    // Input
    keys:           HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta:    (f32, f32),
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() { return; }

        let window = Arc::new(event_loop.create_window(
            Window::default_attributes()
                .with_title("Helio SDF Clipmap Demo (v3)")
                .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32)),
        ).expect("window"));

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(), ..Default::default()
        });
        let surface = instance.create_surface(window.clone()).expect("surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })).expect("adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: required_wgpu_features(adapter.features()),
                required_limits:   required_wgpu_limits(adapter.limits()),
                ..Default::default()
            },
            None,
        )).expect("device");
        device.on_uncaptured_error(Box::new(|e| panic!("[GPU] {:?}", e)));
        let device = Arc::new(device);
        let queue  = Arc::new(queue);
        let caps   = surface.get_capabilities(&adapter);
        let format = caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(caps.formats[0]);
        let size   = window.inner_size();
        surface.configure(&device, &wgpu::SurfaceConfiguration {
            usage:  wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width:  size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode:   caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        });

        let mut renderer = Renderer::new(
            device.clone(),
            queue.clone(),
            RendererConfig::new(size.width, size.height, format),
        );

        // Build the SDF clipmap pass using the renderer's camera buffer.
        let mut sdf = {
            let camera_buf = renderer.camera_buffer();
            SdfClipmapPass::new(&device, camera_buf, format)
        };

        // Rolling terrain base
        sdf.set_terrain(TerrainConfig::rolling());

        // Sphere on a hill
        sdf.add_edit(SdfEdit {
            shape:        SdfShapeType::Sphere,
            op:           BooleanOp::Union,
            transform:    Mat4::from_translation(Vec3::new(0.0, 12.0, 0.0)),
            params:       SdfShapeParams::sphere(6.0),
            blend_radius: 2.0,
        });

        // Tall capsule pillar
        sdf.add_edit(SdfEdit {
            shape:        SdfShapeType::Capsule,
            op:           BooleanOp::Union,
            transform:    Mat4::from_translation(Vec3::new(20.0, 8.0, 5.0)),
            params:       SdfShapeParams::capsule(2.0, 8.0),
            blend_radius: 1.0,
        });

        // Subtracted hollow in the terrain
        sdf.add_edit(SdfEdit {
            shape:        SdfShapeType::Sphere,
            op:           BooleanOp::Subtraction,
            transform:    Mat4::from_translation(Vec3::new(0.0, 2.0, 0.0)),
            params:       SdfShapeParams::sphere(5.0),
            blend_radius: 1.5,
        });

        renderer.add_pass(Box::new(sdf));

        eprintln!("[SDF] Clipmap ready. Auto-orbiting -- press WASD to fly freely.");

        self.state = Some(AppState {
            window, surface, device, surface_format: format, renderer,
            last_frame:     std::time::Instant::now(),
            cam_pos:        Vec3::new(0.0, 30.0, 80.0),
            cam_yaw:        std::f32::consts::PI,
            cam_pitch:      -0.18,
            orbit_mode:     true,
            elapsed:        0.0,
            keys:           HashSet::new(),
            cursor_grabbed: false,
            mouse_delta:    (0.0, 0.0),
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    state: ElementState::Pressed,
                    physical_key: PhysicalKey::Code(KeyCode::Escape),
                    ..
                },
                ..
            } => {
                if state.cursor_grabbed {
                    state.cursor_grabbed = false;
                    let _ = state.window.set_cursor_grab(CursorGrabMode::None);
                    state.window.set_cursor_visible(true);
                } else {
                    event_loop.exit();
                }
            }
            WindowEvent::KeyboardInput {
                event: KeyEvent { state: ks, physical_key: PhysicalKey::Code(key), .. }, ..
            } => {
                match ks {
                    ElementState::Pressed  => { state.keys.insert(key); }
                    ElementState::Released => { state.keys.remove(&key); }
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed, button: MouseButton::Left, ..
            } => {
                if !state.cursor_grabbed {
                    let ok = state.window
                        .set_cursor_grab(CursorGrabMode::Confined)
                        .or_else(|_| state.window.set_cursor_grab(CursorGrabMode::Locked))
                        .is_ok();
                    if ok {
                        state.window.set_cursor_visible(false);
                        state.cursor_grabbed = true;
                    }
                }
            }
            WindowEvent::Resized(sz) if sz.width > 0 && sz.height > 0 => {
                state.surface.configure(&state.device, &wgpu::SurfaceConfiguration {
                    usage:  wgpu::TextureUsages::RENDER_ATTACHMENT,
                    format: state.surface_format,
                    width:  sz.width,
                    height: sz.height,
                    present_mode: wgpu::PresentMode::AutoVsync,
                    alpha_mode:   wgpu::CompositeAlphaMode::Auto,
                    view_formats: vec![],
                    desired_maximum_frame_latency: 2,
                });
                state.renderer.set_render_size(sz.width, sz.height);
            }
            WindowEvent::RedrawRequested => {
                let now = std::time::Instant::now();
                let dt  = (now - state.last_frame).as_secs_f32();
                state.last_frame = now;
                state.render(dt);
                state.window.request_redraw();
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _: &ActiveEventLoop,
        _: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        let Some(s) = &mut self.state else { return };
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            if s.cursor_grabbed {
                s.mouse_delta.0 += dx as f32;
                s.mouse_delta.1 += dy as f32;
            }
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(s) = &self.state { s.window.request_redraw(); }
    }
}

impl AppState {
    fn render(&mut self, dt: f32) {
        const SPEED: f32 = 25.0;
        const SENS:  f32 = 0.0020;

        self.elapsed += dt;

        // Exit orbit mode on first movement input
        if self.keys.contains(&KeyCode::KeyW)
            || self.keys.contains(&KeyCode::KeyS)
            || self.keys.contains(&KeyCode::KeyA)
            || self.keys.contains(&KeyCode::KeyD)
            || self.keys.contains(&KeyCode::Space)
        {
            self.orbit_mode = false;
        }

        // Apply mouse look
        self.cam_yaw   += self.mouse_delta.0 * SENS;
        self.cam_pitch  = (self.cam_pitch - self.mouse_delta.1 * SENS).clamp(-1.55, 1.55);
        self.mouse_delta = (0.0, 0.0);

        if self.orbit_mode {
            let angle  = self.elapsed * 0.07;
            let radius = 100.0_f32;
            let height = 30.0 + 15.0 * (self.elapsed * 0.022).sin();
            self.cam_pos = Vec3::new(radius * angle.cos(), height, radius * angle.sin());
            let dir = (-self.cam_pos).normalize();
            self.cam_yaw   = dir.z.atan2(dir.x);
            self.cam_pitch = dir.y.asin();
        } else {
            let (sy, cy) = self.cam_yaw.sin_cos();
            let (sp, cp) = self.cam_pitch.sin_cos();
            let fwd   = Vec3::new(sy * cp, sp, -cy * cp);
            let right = Vec3::new(cy, 0.0, sy);
            if self.keys.contains(&KeyCode::KeyW)      { self.cam_pos += fwd   * SPEED * dt; }
            if self.keys.contains(&KeyCode::KeyS)      { self.cam_pos -= fwd   * SPEED * dt; }
            if self.keys.contains(&KeyCode::KeyA)      { self.cam_pos -= right * SPEED * dt; }
            if self.keys.contains(&KeyCode::KeyD)      { self.cam_pos += right * SPEED * dt; }
            if self.keys.contains(&KeyCode::Space)     { self.cam_pos.y += SPEED * dt; }
            if self.keys.contains(&KeyCode::ShiftLeft) { self.cam_pos.y -= SPEED * dt; }
        }

        let (sy, cy) = self.cam_yaw.sin_cos();
        let (sp, cp) = self.cam_pitch.sin_cos();
        let fwd = Vec3::new(sy * cp, sp, -cy * cp);
        let sz  = self.window.inner_size();
        let asp = sz.width as f32 / sz.height.max(1) as f32;
        let camera = Camera::perspective_look_at(
            self.cam_pos,
            self.cam_pos + fwd,
            Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            asp,
            0.5,
            2000.0,
        );

        let output = match self.surface.get_current_texture() {
            Ok(t)  => t,
            Err(e) => { log::warn!("surface: {:?}", e); return; }
        };
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        if let Err(e) = self.renderer.render(&camera, &view) {
            log::error!("render: {:?}", e);
        }
        output.present();
    }
}