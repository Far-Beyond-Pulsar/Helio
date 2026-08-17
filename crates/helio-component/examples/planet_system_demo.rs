//! Live validation for Pulsar's production smooth 3D SDF planet path.
//!
//! This executable deliberately owns no terrain planner, page fixture, or
//! fallback mesh. `PlanetTerrainRuntime` produces the canonical sparse SDF
//! pages and publishes them through Helio's graph-owned planetary pass.
//!
//! Controls:
//! - click: capture mouse
//! - W/A/S/D + Space/Shift: free flight
//! - Control: fast travel
//! - F2: switch page/meshlet draw path
//! - F3: cycle planetary debug views
//! - F6: cycle low flight, orbit, and far-orbit positions
//! - Escape: release mouse, then exit

use std::{
    collections::HashSet,
    sync::Arc,
    time::{Duration, Instant},
};

use glam::{EulerRot, Quat, Vec3};
use helio::{
    Camera, DebugDrawState, Renderer, RendererConfig, Scene, required_experimental_features,
    required_wgpu_features, required_wgpu_limits,
};
use helio_component::{PlanetTerrainComponent, PlanetTerrainFrameInput, PlanetTerrainRuntime};
use helio_default_graphs::build_default_graph_external_with_planetary_voxels;
use helio_pass_planetary_voxel::PlanetaryVoxelRenderPass;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, ElementState, KeyEvent, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, DeviceEvents, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

const EARTH_RADIUS_M: f64 = 6_371_000.0;
const LOOK_SENSITIVITY: f32 = 0.002;
const FIELD_OF_VIEW: f32 = std::f32::consts::FRAC_PI_3;
const INITIAL_YAW: f32 = std::f32::consts::FRAC_PI_2;
const VALIDATION_POSITIONS: [[f64; 3]; 3] = [
    [EARTH_RADIUS_M + 25_000.0, 0.0, 0.0],
    [EARTH_RADIUS_M * 2.5, 0.0, 0.0],
    [EARTH_RADIUS_M * 20.0, 0.0, 0.0],
];

fn main() {
    let event_loop = EventLoop::new().expect("planet demo event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop
        .run_app(&mut App { state: None })
        .expect("planet demo application");
}

struct App {
    state: Option<AppState>,
}

struct AppState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    surface_format: wgpu::TextureFormat,
    alpha_mode: wgpu::CompositeAlphaMode,
    renderer: Renderer,
    terrain: PlanetTerrainRuntime,
    canonical_camera_m: [f64; 3],
    camera_velocity_mps: [f64; 3],
    validation_position: usize,
    yaw: f32,
    pitch: f32,
    keys: HashSet<KeyCode>,
    mouse_delta: (f32, f32),
    cursor_grabbed: bool,
    graph_rebuilt: bool,
    frame_index: u64,
    last_frame: Instant,
    last_title_update: Instant,
    stream_status: String,
}

impl AppState {
    fn reset_input(&mut self) {
        self.keys.clear();
        self.mouse_delta = (0.0, 0.0);
        self.camera_velocity_mps = [0.0; 3];
        self.cursor_grabbed = false;
        let _ = self.window.set_cursor_grab(CursorGrabMode::None);
        self.window.set_cursor_visible(true);
        self.last_frame = Instant::now();
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.reset_input();
        if width == 0 || height == 0 {
            return;
        }
        self.surface.configure(
            &self.device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: self.surface_format,
                color_space: wgpu::SurfaceColorSpace::Auto,
                width,
                height,
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: self.alpha_mode,
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            },
        );
        self.renderer.set_render_size(width, height);
        self.graph_rebuilt = true;
    }

    fn orientation(&self) -> Quat {
        Quat::from_euler(EulerRot::YXZ, self.yaw, self.pitch, 0.0)
    }

    fn forward_up(&self) -> (Vec3, Vec3) {
        let orientation = self.orientation();
        (orientation * -Vec3::Z, orientation * Vec3::Y)
    }

    fn update_camera(&mut self, dt: f64) {
        self.yaw -= self.mouse_delta.0 * LOOK_SENSITIVITY;
        self.pitch = (self.pitch - self.mouse_delta.1 * LOOK_SENSITIVITY).clamp(-1.5, 1.5);
        self.mouse_delta = (0.0, 0.0);

        let orientation = self.orientation();
        let mut direction = Vec3::ZERO;
        if self.keys.contains(&KeyCode::KeyW) {
            direction += orientation * -Vec3::Z;
        }
        if self.keys.contains(&KeyCode::KeyS) {
            direction += orientation * Vec3::Z;
        }
        if self.keys.contains(&KeyCode::KeyA) {
            direction += orientation * -Vec3::X;
        }
        if self.keys.contains(&KeyCode::KeyD) {
            direction += orientation * Vec3::X;
        }
        if self.keys.contains(&KeyCode::Space) {
            direction += orientation * Vec3::Y;
        }
        if self.keys.contains(&KeyCode::ShiftLeft) {
            direction += orientation * -Vec3::Y;
        }

        let altitude = (vector_length(self.canonical_camera_m) - EARTH_RADIUS_M).max(0.0);
        let base_speed = (altitude * 0.35).clamp(10.0, 5_000_000.0);
        let boosted =
            self.keys.contains(&KeyCode::ControlLeft) || self.keys.contains(&KeyCode::ControlRight);
        let speed = if boosted {
            (base_speed * 12.0).min(25_000_000.0)
        } else {
            base_speed
        };
        let step = direction.normalize_or_zero().as_dvec3() * speed * dt;
        self.camera_velocity_mps = if dt > 0.0 {
            (step / dt).to_array()
        } else {
            [0.0; 3]
        };
        for axis in 0..3 {
            self.canonical_camera_m[axis] += step[axis];
        }
    }

    fn clip_planes(&self) -> (f64, f64) {
        let distance = vector_length(self.canonical_camera_m);
        let altitude = (distance - EARTH_RADIUS_M).max(0.0);
        let near = (altitude * 1.0e-5).clamp(0.1, 10_000.0);
        let far = (distance + EARTH_RADIUS_M * 2.5).max(100_000.0);
        (near, far)
    }

    fn camera(&self, width: u32, height: u32) -> Camera {
        let (forward, up) = self.forward_up();
        let (near, far) = self.clip_planes();
        Camera::perspective_look_at(
            Vec3::ZERO,
            forward,
            up,
            FIELD_OF_VIEW,
            width as f32 / height.max(1) as f32,
            near as f32,
            far as f32,
        )
    }

    fn advance_terrain(&mut self, width: u32, height: u32, dt: f32) {
        self.frame_index = self.frame_index.wrapping_add(1);
        let (forward, up) = self.forward_up();
        let (near, far) = self.clip_planes();
        let input = PlanetTerrainFrameInput {
            camera_m: self.canonical_camera_m,
            forward: forward.as_dvec3().to_array(),
            up: up.as_dvec3().to_array(),
            vertical_fov_radians: f64::from(FIELD_OF_VIEW),
            viewport_px: [width.max(1), height.max(1)],
            near_m: near,
            far_m: far,
            velocity_mps: self.camera_velocity_mps,
            delta_time_s: dt,
            tick: self.frame_index,
            frame_index: self.frame_index,
            graph_rebuilt: std::mem::take(&mut self.graph_rebuilt),
        };
        self.stream_status =
            match self
                .terrain
                .advance(&mut self.renderer, &self.device, &self.queue, input)
            {
                Ok(report) if report.planning_failures.is_empty() => format!(
                    "ok plans{} up{} ev{} vis{:?}",
                    report.plans_applied,
                    report.render.uploads.len(),
                    report.render.evictions.len(),
                    report.visibility,
                ),
                Ok(report) => format!("FAIL {}", report.planning_failures.join("; ")),
                Err(error) => format!("FAIL {error}"),
            };
    }

    fn update_title(&mut self) {
        if self.last_title_update.elapsed() < Duration::from_millis(250) {
            return;
        }
        self.last_title_update = Instant::now();
        let pass = self
            .renderer
            .find_pass_mut::<PlanetaryVoxelRenderPass>()
            .expect("production planetary pass");
        let diagnostics = pass.poll_diagnostics(&self.device, &self.queue);
        let counters = pass.counters();
        let residency = pass.residency().counters();
        let lods = diagnostics
            .resident_lods
            .iter()
            .map(u8::to_string)
            .collect::<Vec<_>>()
            .join(",");
        self.window.set_title(&format!(
            "Pulsar 3D SDF Planet | {:?}/{} | {} | pos[{:+.0},{:+.0},{:+.0}]m | pages{} lod[{lods}] active{} candidate{}/{} target{} missing{} flight{} state np{} iv{} g{} r{} t{} jobs{}/{} reject s{} o{} i{} queued{}",
            pass.draw_path(),
            pass.debug_view().label(),
            self.stream_status,
            self.canonical_camera_m[0],
            self.canonical_camera_m[1],
            self.canonical_camera_m[2],
            residency.resident_pages,
            diagnostics.active_surface_pages,
            diagnostics.ready_candidate_surface_pages,
            diagnostics.candidate_surface_pages,
            diagnostics.resident_candidate_targets,
            diagnostics.missing_candidate_dependencies,
            diagnostics.cpu_surface_jobs_in_flight,
            diagnostics.candidate_without_publication,
            diagnostics.candidate_invalid_state,
            diagnostics.candidate_generation_mismatches,
            diagnostics.candidate_revision_mismatches,
            diagnostics.candidate_transition_mismatches,
            diagnostics.gpu_published_jobs,
            diagnostics.gpu_submitted_jobs,
            diagnostics.gpu_stale_rejections,
            diagnostics.gpu_overflow_rejections,
            diagnostics.gpu_incomplete_rejections,
            counters.queued_surfaces,
        ));
    }

    fn cycle_validation_position(&mut self) {
        self.validation_position = (self.validation_position + 1) % VALIDATION_POSITIONS.len();
        self.canonical_camera_m = VALIDATION_POSITIONS[self.validation_position];
        self.camera_velocity_mps = [0.0; 3];
        self.yaw = INITIAL_YAW;
        self.pitch = 0.0;
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        event_loop.listen_device_events(DeviceEvents::WhenFocused);
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Pulsar 3D SDF Planet")
                        .with_inner_size(winit::dpi::PhysicalSize::new(1280, 720)),
                )
                .expect("planet demo window"),
        );
        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window.clone())
            .expect("planet demo surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .expect("planet demo GPU adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Pulsar 3D SDF Planet Device"),
            required_features: required_wgpu_features(adapter.features()),
            required_limits: required_wgpu_limits(adapter.limits()),
            experimental_features: required_experimental_features(adapter.features()),
            ..Default::default()
        }))
        .expect("planet demo GPU device");
        device.on_uncaptured_error(Arc::new(|error| {
            eprintln!("planet demo uncaptured GPU error: {error}");
        }));
        let device = Arc::new(device);
        let queue = Arc::new(queue);
        let size = window.inner_size();
        let capabilities = surface.get_capabilities(&adapter);
        let surface_format = capabilities
            .formats
            .iter()
            .copied()
            .find(wgpu::TextureFormat::is_srgb)
            .unwrap_or(capabilities.formats[0]);
        let alpha_mode = capabilities.alpha_modes[0];
        surface.configure(
            &device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: surface_format,
                color_space: wgpu::SurfaceColorSpace::Auto,
                width: size.width,
                height: size.height,
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode,
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            },
        );

        let renderer_config =
            RendererConfig::new(size.width, size.height, surface_format).with_render_scale(1.0);
        let scene = Scene::new(device.clone(), queue.clone());
        let debug_camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Planet Demo Debug Camera"),
            size: core::mem::size_of::<helio::DebugCameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cull_stats_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Planet Demo Cull Stats"),
            size: 32,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let debug_state = Arc::new(std::sync::Mutex::new(DebugDrawState::default()));
        let graph = build_default_graph_external_with_planetary_voxels(
            &device,
            &queue,
            &scene,
            renderer_config,
            debug_state.clone(),
            &debug_camera_buffer,
            &cull_stats_buffer,
            None,
            PlanetTerrainRuntime::renderer_config(),
        )
        .expect("bounded production planet graph");
        let mut renderer = Renderer::new(
            device.clone(),
            queue.clone(),
            renderer_config.surface_format,
            renderer_config.width,
            renderer_config.height,
            renderer_config.render_scale,
            renderer_config,
            scene,
            graph,
            debug_state,
            debug_camera_buffer,
            cull_stats_buffer,
        );
        renderer.set_jitter_enabled(false);

        let mut terrain = PlanetTerrainRuntime::new().expect("production terrain runtime");
        let component = PlanetTerrainComponent::default();
        let definition = component
            .definition("planet-system-demo:0")
            .expect("valid Earth-sized SDF planet");
        let planet_id = definition.planet_id;
        let (runtime, cache) = terrain.component_context_mut();
        runtime
            .upsert_component("planet-system-demo:0".into(), definition)
            .expect("register demo component");
        cache.record("planet-system-demo:0".into(), planet_id);

        self.state = Some(AppState {
            window,
            surface,
            device,
            queue,
            surface_format,
            alpha_mode,
            renderer,
            terrain,
            canonical_camera_m: VALIDATION_POSITIONS[1],
            camera_velocity_mps: [0.0; 3],
            validation_position: 1,
            yaw: INITIAL_YAW,
            pitch: 0.0,
            keys: HashSet::new(),
            mouse_delta: (0.0, 0.0),
            cursor_grabbed: false,
            graph_rebuilt: false,
            frame_index: 0,
            last_frame: Instant::now(),
            last_title_update: Instant::now() - Duration::from_secs(1),
            stream_status: "starting".into(),
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size.width, size.height),
            WindowEvent::Focused(false) => state.reset_input(),
            WindowEvent::Focused(true) => state.last_frame = Instant::now(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        ..
                    },
                ..
            } => {
                if state.cursor_grabbed {
                    state.reset_input();
                } else {
                    event_loop.exit();
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: key_state,
                        physical_key: PhysicalKey::Code(key),
                        repeat,
                        ..
                    },
                ..
            } => match (key_state, key, repeat) {
                (ElementState::Pressed, KeyCode::F2, false) => {
                    state
                        .renderer
                        .find_pass_mut::<PlanetaryVoxelRenderPass>()
                        .expect("production planetary pass")
                        .toggle_draw_path(&state.queue);
                }
                (ElementState::Pressed, KeyCode::F3, false) => {
                    state
                        .renderer
                        .find_pass_mut::<PlanetaryVoxelRenderPass>()
                        .expect("production planetary pass")
                        .cycle_debug_view(&state.queue);
                }
                (ElementState::Pressed, KeyCode::F6, false) => {
                    state.cycle_validation_position();
                }
                (ElementState::Pressed, _, _) => {
                    state.keys.insert(key);
                }
                (ElementState::Released, _, _) => {
                    state.keys.remove(&key);
                }
            },
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } if !state.cursor_grabbed => {
                state.window.focus_window();
                let grabbed = state
                    .window
                    .set_cursor_grab(CursorGrabMode::Confined)
                    .or_else(|_| state.window.set_cursor_grab(CursorGrabMode::Locked))
                    .is_ok();
                if grabbed {
                    state.cursor_grabbed = true;
                    state.window.set_cursor_visible(false);
                }
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = now.duration_since(state.last_frame).as_secs_f64().min(0.05);
                state.last_frame = now;
                state.update_camera(dt);
                let size = state.window.inner_size();
                if size.width == 0 || size.height == 0 {
                    return;
                }
                state.advance_terrain(size.width, size.height, dt as f32);
                state.update_title();
                let output = match state.surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(texture)
                    | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
                    _ => return,
                };
                let view = output
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                if let Err(error) = state
                    .renderer
                    .render(&state.camera(size.width, size.height), &view)
                {
                    eprintln!("planet render failed: {error:?}");
                }
                state.queue.present(output);
                state.window.request_redraw();
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: DeviceId, event: DeviceEvent) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        if let DeviceEvent::MouseMotion { delta: (x, y) } = event
            && state.cursor_grabbed
        {
            state.mouse_delta.0 += x as f32;
            state.mouse_delta.1 += y as f32;
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(state) = self.state.as_ref() {
            state.window.request_redraw();
        }
    }
}

fn vector_length(value: [f64; 3]) -> f64 {
    value.iter().map(|axis| axis * axis).sum::<f64>().sqrt()
}
