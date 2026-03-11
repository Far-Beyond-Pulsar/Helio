//! SDF Game Engine Demo
//!
//! Interactive SDF rendering with constructive solid geometry.
//!
//! Controls:
//!   WASD          — move forward/left/back/right
//!   Space/Shift   — move up/down
//!   Mouse drag    — look around (click to grab cursor)
//!   Left click    — use active tool (dig/build) at surface
//!   Scroll wheel  — adjust tool radius
//!   1             — Dig tool (subtract spheres)
//!   2             — Build tool (add spheres)
//!   3             — No tool (camera only)
//!   F3            — toggle debug visualization (brick/clip level overlay)
//!   Ctrl+Z        — undo last edit
//!   Ctrl+Y        — redo
//!   Escape        — release cursor / exit

use helio_render_v2::{Renderer, RendererConfig, Camera};

use helio_render_v2::features::{
    FeatureRegistry,
    LightingFeature,
    SdfFeature, SdfEdit, SdfShapeType, SdfShapeParams, BooleanOp,
    TerrainConfig,
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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Tool {
    None,
    Dig,
    Build,
}

fn main() {
    env_logger::init();
    log::info!("Starting Helio SDF Demo");

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

    // Free-camera state
    cam_pos: glam::Vec3,
    cam_yaw: f32,
    cam_pitch: f32,
    keys: HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta: (f32, f32),

    // Tool state
    active_tool: Tool,
    tool_radius: f32,
    ctrl_held: bool,

    // Undo/Redo
    edit_history: Vec<SdfEdit>,
    edit_history_cursor: usize,
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
                        .with_title("Helio SDF Demo")
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
                label: Some("SDF Demo Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    max_storage_buffers_per_shader_stage: 10,
                    ..wgpu::Limits::default()
                },
                memory_hints: wgpu::MemoryHints::default(),
                experimental_features: Default::default(),
                trace: wgpu::Trace::Off,
            },
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

        // ── SDF Feature Setup ──────────────────────────────────────────────────
        let sdf_feature = SdfFeature::new()
            .with_grid_dim(128)
            .with_volume_bounds([-10.0, -10.0, -10.0], [10.0, 10.0, 10.0])
            .with_terrain(TerrainConfig::rolling());

        let feature_registry = FeatureRegistry::builder()
            .with_feature(LightingFeature::new())
            .with_feature(sdf_feature)
            .build();

        let renderer = Renderer::new(
            device.clone(),
            queue.clone(),
            RendererConfig::new(size.width, size.height, surface_format, feature_registry),
        )
        .expect("Failed to create renderer");

        self.state = Some(AppState {
            window,
            surface,
            device,
            surface_format,
            renderer,
            last_frame: std::time::Instant::now(),
            cam_pos: glam::Vec3::new(0.0, 8.0, 20.0),
            cam_yaw: 0.0,
            cam_pitch: -0.2,
            keys: HashSet::new(),
            cursor_grabbed: false,
            mouse_delta: (0.0, 0.0),
            active_tool: Tool::Dig,
            tool_radius: 1.0,
            ctrl_held: false,
            edit_history: Vec::new(),
            edit_history_cursor: 0,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
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
                event: KeyEvent { state: ks, physical_key: PhysicalKey::Code(key), .. },
                ..
            } => {
                match ks {
                    ElementState::Pressed  => {
                        state.keys.insert(key);
                        if key == KeyCode::ControlLeft || key == KeyCode::ControlRight {
                            state.ctrl_held = true;
                        }
                        // F3: toggle SDF debug visualization
                        if key == KeyCode::F3 {
                            if let Some(sdf) = state.renderer.get_feature_mut::<SdfFeature>("sdf") {
                                sdf.toggle_debug();
                            }
                        }
                        // 1/2/3: tool switching
                        if key == KeyCode::Digit1 {
                            state.active_tool = Tool::Dig;
                            log::info!("Tool: Dig (radius={:.1})", state.tool_radius);
                        }
                        if key == KeyCode::Digit2 {
                            state.active_tool = Tool::Build;
                            log::info!("Tool: Build (radius={:.1})", state.tool_radius);
                        }
                        if key == KeyCode::Digit3 {
                            state.active_tool = Tool::None;
                            log::info!("Tool: None (camera only)");
                        }
                        // Ctrl+Z: undo
                        if key == KeyCode::KeyZ && state.ctrl_held {
                            state.undo();
                        }
                        // Ctrl+Y: redo
                        if key == KeyCode::KeyY && state.ctrl_held {
                            state.redo();
                        }
                    }
                    ElementState::Released => {
                        state.keys.remove(&key);
                        if key == KeyCode::ControlLeft || key == KeyCode::ControlRight {
                            state.ctrl_held = false;
                        }
                    }
                }
            }

            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                if !state.cursor_grabbed {
                    let grabbed = state.window.set_cursor_grab(CursorGrabMode::Confined)
                        .or_else(|_| state.window.set_cursor_grab(CursorGrabMode::Locked))
                        .is_ok();
                    if grabbed {
                        state.window.set_cursor_visible(false);
                        state.cursor_grabbed = true;
                    }
                } else {
                    state.use_tool();
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                if state.cursor_grabbed {
                    let scroll = match delta {
                        winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                        winit::event::MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.01,
                    };
                    state.tool_radius = (state.tool_radius + scroll * 0.2).clamp(0.2, 5.0);
                    log::info!("Tool radius: {:.1}", state.tool_radius);
                }
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

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _id: winit::event::DeviceId, event: DeviceEvent) {
        let Some(state) = &mut self.state else { return };
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            if state.cursor_grabbed {
                state.mouse_delta.0 += dx as f32;
                state.mouse_delta.1 += dy as f32;
            }
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }
}

impl AppState {
    /// Perform a center-screen pick and apply the active tool.
    fn use_tool(&mut self) {
        if self.active_tool == Tool::None {
            return;
        }

        let (sy, cy) = self.cam_yaw.sin_cos();
        let (sp, cp) = self.cam_pitch.sin_cos();
        let forward = glam::Vec3::new(sy * cp, sp, -cy * cp);

        let sdf = match self.renderer.get_feature_mut::<SdfFeature>("sdf") {
            Some(s) => s as *mut SdfFeature,
            None => return,
        };

        // Safety: we hold exclusive access to renderer/sdf for this scope.
        let sdf = unsafe { &mut *sdf };

        let pick = match sdf.pick_surface(self.cam_pos, forward, 100.0) {
            Some(p) => p,
            None => {
                log::debug!("Pick miss");
                return;
            }
        };

        let edit = match self.active_tool {
            Tool::Dig => {
                log::info!("Dig at ({:.1}, {:.1}, {:.1}) r={:.1}",
                    pick.position.x, pick.position.y, pick.position.z, self.tool_radius);
                SdfEdit {
                    shape: SdfShapeType::Sphere,
                    op: BooleanOp::Subtraction,
                    transform: glam::Mat4::from_translation(pick.position),
                    params: SdfShapeParams::sphere(self.tool_radius),
                    blend_radius: self.tool_radius * 0.3,
                }
            }
            Tool::Build => {
                let place_pos = pick.position + pick.normal * self.tool_radius * 0.5;
                log::info!("Build at ({:.1}, {:.1}, {:.1}) r={:.1}",
                    place_pos.x, place_pos.y, place_pos.z, self.tool_radius);
                SdfEdit {
                    shape: SdfShapeType::Sphere,
                    op: BooleanOp::Union,
                    transform: glam::Mat4::from_translation(place_pos),
                    params: SdfShapeParams::sphere(self.tool_radius),
                    blend_radius: self.tool_radius * 0.3,
                }
            }
            Tool::None => unreachable!(),
        };

        // Record in undo history (truncate any future redo entries)
        self.edit_history.truncate(self.edit_history_cursor);
        self.edit_history.push(edit.clone());
        self.edit_history_cursor += 1;

        sdf.add_edit(edit);
        log::info!("Edit count: {}", sdf.edit_list().len());
    }

    fn undo(&mut self) {
        if self.edit_history_cursor == 0 {
            log::info!("Nothing to undo");
            return;
        }

        let sdf = match self.renderer.get_feature_mut::<SdfFeature>("sdf") {
            Some(s) => s,
            None => return,
        };

        self.edit_history_cursor -= 1;

        // Remove all tool edits, then re-add up to cursor
        sdf.clear_edits();
        for edit in &self.edit_history[..self.edit_history_cursor] {
            sdf.add_edit(edit.clone());
        }
        log::info!("Undo (edit count: {})", sdf.edit_list().len());
    }

    fn redo(&mut self) {
        if self.edit_history_cursor >= self.edit_history.len() {
            log::info!("Nothing to redo");
            return;
        }

        let sdf = match self.renderer.get_feature_mut::<SdfFeature>("sdf") {
            Some(s) => s,
            None => return,
        };

        self.edit_history_cursor += 1;

        sdf.clear_edits();
        for edit in &self.edit_history[..self.edit_history_cursor] {
            sdf.add_edit(edit.clone());
        }
        log::info!("Redo (edit count: {})", sdf.edit_list().len());
    }

    fn render(&mut self, dt: f32) {
        const SPEED: f32 = 20.0;
        const LOOK_SENS: f32 = 0.002;

        self.cam_yaw += self.mouse_delta.0 * LOOK_SENS;
        self.cam_pitch = (self.cam_pitch - self.mouse_delta.1 * LOOK_SENS).clamp(-1.5, 1.5);
        self.mouse_delta = (0.0, 0.0);

        let (sy, cy) = self.cam_yaw.sin_cos();
        let (sp, cp) = self.cam_pitch.sin_cos();
        let forward = glam::Vec3::new(sy * cp, sp, -cy * cp);
        let right = glam::Vec3::new(cy, 0.0, sy);
        let up = glam::Vec3::Y;

        if self.keys.contains(&KeyCode::KeyW)      { self.cam_pos += forward * SPEED * dt; }
        if self.keys.contains(&KeyCode::KeyS)      { self.cam_pos -= forward * SPEED * dt; }
        if self.keys.contains(&KeyCode::KeyA)      { self.cam_pos -= right * SPEED * dt; }
        if self.keys.contains(&KeyCode::KeyD)      { self.cam_pos += right * SPEED * dt; }
        if self.keys.contains(&KeyCode::Space)     { self.cam_pos += up * SPEED * dt; }
        if self.keys.contains(&KeyCode::ShiftLeft) { self.cam_pos -= up * SPEED * dt; }

        let size = self.window.inner_size();
        let aspect = size.width as f32 / size.height.max(1) as f32;
        let time = self.renderer.frame_count() as f32 * 0.016;

        let camera = Camera::perspective(
            self.cam_pos,
            self.cam_pos + forward,
            glam::Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            aspect,
            0.1,
            5000.0,
            time,
        );

        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(e) => { log::warn!("Surface error: {:?}", e); return; }
        };
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        if let Err(e) = self.renderer.render(&camera, &view, dt) {
            log::error!("Render error: {:?}", e);
        }

        output.present();
    }
}
