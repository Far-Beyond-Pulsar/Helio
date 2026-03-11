//! Feature showcase example using helio-render-v2
//!
//! All scene content is driven by a `Scene` struct — no hardcoded lights
//! or geometry in the renderer.
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

mod demo_portal;

use helio_render_v2::{Renderer, RendererConfig, Camera, GpuMesh, SceneLight, LightId, BillboardId};


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

    // Free-camera state
    cam_pos:   glam::Vec3,
    cam_yaw:   f32,   // radians, horizontal rotation
    cam_pitch: f32,   // radians, vertical rotation (clamped)
    keys:      HashSet<KeyCode>,
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
                        .with_title("Helio Render V2 – Scene-Driven")
                        .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32)),
                )
                .expect("Failed to create window"),
        );

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            flags: wgpu::InstanceFlags::VALIDATION | wgpu::InstanceFlags::GPU_BASED_VALIDATION | wgpu::InstanceFlags::DEBUG,
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
                // SAFETY: We acknowledge EXPERIMENTAL_RAY_QUERY may have implementation bugs.
                experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
                trace: wgpu::Trace::Off,
            },
        ))
        .expect("Failed to create device (ray tracing required)");

        device.on_uncaptured_error(std::sync::Arc::new(|e| {
            panic!("[GPU UNCAPTURED ERROR] {:?}", e);
        }));
        let info = adapter.get_info();
        println!("[WGPU] Backend: {:?}, Device: {}, Driver: {}", info.backend, info.name, info.driver);
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
            .with_volume_bounds([-3.0, -1.0, -3.0], [3.0, 3.0, 3.0])
            .with_terrain(TerrainConfig::rolling());

        let feature_registry = FeatureRegistry::builder()
            .with_feature(LightingFeature::new())
            .with_feature(BloomFeature::new().with_intensity(0.4).with_threshold(1.2))
            .with_feature(ShadowsFeature::new().with_atlas_size(1024).with_max_lights(4))
            .with_feature(BillboardsFeature::new().with_sprite(sprite_rgba.clone(), sprite_w, sprite_h).with_max_instances(5000))
            .with_feature(
                RadianceCascadesFeature::new()
                    .with_world_bounds(RC_WORLD_MIN, RC_WORLD_MAX),
            )
            .with_feature(sdf_feature)
            .build();

        let mut renderer = Renderer::new(
            device.clone(),
            queue.clone(),
            RendererConfig::new(size.width, size.height, surface_format, feature_registry),
        )
        .expect("Failed to create renderer");

        let cube1  = renderer.create_mesh_cube([ 0.0, 0.5,  0.0], 0.5);
        let cube2  = renderer.create_mesh_cube([-2.0, 0.4, -1.0], 0.4);
        let cube3  = renderer.create_mesh_cube([ 2.0, 0.3,  0.5], 0.3);
        let ground = renderer.create_mesh_plane([0.0, 0.0, 0.0], 5.0);
        demo_portal::enable_live_dashboard(&mut renderer);

        renderer.add_object(&cube1,  None, glam::Mat4::IDENTITY);
        renderer.add_object(&cube2,  None, glam::Mat4::IDENTITY);
        renderer.add_object(&cube3,  None, glam::Mat4::IDENTITY);
        renderer.add_object(&ground, None, glam::Mat4::IDENTITY);

        // p0 bobs up/down (animated), p1 and p2 are static
        let p0_init = [0.0f32, 2.2, 0.0];
        let p1 = [-3.5f32, 2.0, -1.5];
        let p2 = [3.5f32, 1.5, 1.5];
        let light_p0_id = renderer.add_light(SceneLight::point(p0_init, [1.0, 0.55, 0.15], 6.0, 5.0));
        let light_p1_id = renderer.add_light(SceneLight::point(p1, [0.25, 0.5, 1.0], 5.0, 6.0));
        let light_p2_id = renderer.add_light(SceneLight::point(p2, [1.0, 0.3, 0.5], 5.0, 6.0));

        let mut billboard_ids = Vec::new();
        billboard_ids.push(renderer.add_billboard(BillboardInstance::new(p0_init, [0.35, 0.35]).with_color([1.0, 0.55, 0.15, 1.0])));
        billboard_ids.push(renderer.add_billboard(BillboardInstance::new(p1, [0.35, 0.35]).with_color([0.25, 0.5, 1.0, 1.0])));
        billboard_ids.push(renderer.add_billboard(BillboardInstance::new(p2, [0.35, 0.35]).with_color([1.0, 0.3, 0.5, 1.0])));

        self.state = Some(AppState {
            window,
            surface,
            device,
            surface_format,
            renderer,
            last_frame: std::time::Instant::now(),
            cube1, cube2, cube3, ground,
            cam_pos:   glam::Vec3::new(0.0, 2.5, 7.0),
            cam_yaw:   0.0,         // yaw=0 looks down -Z toward the scene
            cam_pitch: -0.2,
            keys:      HashSet::new(),
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
            // ── Exit ──────────────────────────────────────────────────────────
            WindowEvent::CloseRequested => {
                log::info!("Shutting down");
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
                    // First Escape releases the cursor
                    state.cursor_grabbed = false;
                    let _ = state.window.set_cursor_grab(CursorGrabMode::None);
                    state.window.set_cursor_visible(true);
                } else {
                    event_loop.exit();
                }
            }

            // ── Probe visualization toggle ────────────────────────────────────
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    state: ElementState::Pressed,
                    physical_key: PhysicalKey::Code(KeyCode::Digit3),
                    ..
                },
                ..
            } => {
                state.probe_vis = !state.probe_vis;
                let raw: &[u8] = if state.probe_vis {
                    include_bytes!("../../probe.png")
                } else {
                    include_bytes!("../../spotlight.png")
                };
                let img = image::load_from_memory(raw)
                    .unwrap_or_else(|_| image::DynamicImage::new_rgba8(state.sprite_w, state.sprite_h))
                    .resize_exact(state.sprite_w, state.sprite_h, image::imageops::FilterType::Triangle)
                    .into_rgba8();
                if let Some(bb) = state.renderer.get_feature_mut::<BillboardsFeature>("billboards") {
                    bb.set_sprite(img.into_raw(), state.sprite_w, state.sprite_h);
                }
                for id in state.billboard_ids.drain(..) { state.renderer.remove_billboard(id); }
                if state.probe_vis {
                    for b in probe_billboards(RC_WORLD_MIN, RC_WORLD_MAX) {
                        state.billboard_ids.push(state.renderer.add_billboard(b));
                    }
                } else {
                    // Re-register light marker billboards (p0 position at init; will be updated per-frame)
                    let p0 = [0.0f32, 2.2, 0.0];
                    let p1 = [-3.5f32, 2.0, -1.5];
                    let p2 = [3.5f32, 1.5, 1.5];
                    state.billboard_ids.push(state.renderer.add_billboard(BillboardInstance::new(p0, [0.35, 0.35]).with_color([1.0, 0.55, 0.15, 1.0])));
                    state.billboard_ids.push(state.renderer.add_billboard(BillboardInstance::new(p1, [0.35, 0.35]).with_color([0.25, 0.5, 1.0, 1.0])));
                    state.billboard_ids.push(state.renderer.add_billboard(BillboardInstance::new(p2, [0.35, 0.35]).with_color([1.0, 0.3, 0.5, 1.0])));
                }
            }
            // ── Live profiler portal ──────────────────────────────────────────
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    state: ElementState::Pressed,
                    physical_key: PhysicalKey::Code(KeyCode::Digit4),
                    ..
                },
                ..
            } => { let _ = state.renderer.start_live_portal_default(); }

            // ── Keyboard held state ───────────────────────────────────────────
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

            // ── Mouse button — grab cursor on click ───────────────────────────
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                if !state.cursor_grabbed {
                    // Try confined first, fall back to locked
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

        // Differential undo: remove just the last edit instead of
        // clear+replay, which avoids a full reclassify of all bricks.
        let last_idx = sdf.edit_list().len().saturating_sub(1);
        sdf.remove_edit(last_idx);
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

        // Differential redo: add back just the one edit.
        let edit = self.edit_history[self.edit_history_cursor].clone();
        self.edit_history_cursor += 1;
        sdf.add_edit(edit);
        log::info!("Redo (edit count: {})", sdf.edit_list().len());
    }

    fn render(&mut self, dt: f32) {
        const SPEED: f32 = 20.0;
        const LOOK_SENS: f32 = 0.002;

        // Apply mouse look — yaw left/right, pitch up/down (non-inverted)
        self.cam_yaw   += self.mouse_delta.0 * LOOK_SENS;
        self.cam_pitch  = (self.cam_pitch - self.mouse_delta.1 * LOOK_SENS).clamp(-1.5, 1.5);
        self.mouse_delta = (0.0, 0.0);

        // Standard FPS basis: yaw=0 looks down -Z
        let (sy, cy) = self.cam_yaw.sin_cos();
        let (sp, cp) = self.cam_pitch.sin_cos();
        let forward = glam::Vec3::new(sy * cp, sp, -cy * cp);
        let right   = glam::Vec3::new(cy, 0.0, sy);
        let up      = glam::Vec3::Y;

        if self.keys.contains(&KeyCode::KeyW)      { self.cam_pos += forward * SPEED * dt; }
        if self.keys.contains(&KeyCode::KeyS)      { self.cam_pos -= forward * SPEED * dt; }
        if self.keys.contains(&KeyCode::KeyA)      { self.cam_pos -= right   * SPEED * dt; }
        if self.keys.contains(&KeyCode::KeyD)      { self.cam_pos += right   * SPEED * dt; }
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

        // p0 bobs up/down per-frame; update its light and billboard position
        let p0 = [0.0f32, 2.2 + (time * 0.7).sin() * 0.3, 0.0];
        self.renderer.update_light(self.light_p0_id, SceneLight::point(p0, [1.0, 0.55, 0.15], 6.0, 5.0));
        if !self.probe_vis && !self.billboard_ids.is_empty() {
            self.renderer.update_billboard(self.billboard_ids[0], BillboardInstance::new(p0, [0.35, 0.35]).with_color([1.0, 0.55, 0.15, 1.0]));
        }
        if let Err(e) = self.renderer.render(&camera, &view, dt) {
            log::error!("Render error: {:?}", e);
        }

        output.present();
    }
}

