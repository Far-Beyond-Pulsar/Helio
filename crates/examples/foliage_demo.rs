//! Foliage demo — GPU-driven grass over an open field.
//!
//! Exercises the whole foliage authoring path: a registered foliage type, a layer, the
//! global wind clock, and a moving interactor that pushes grass aside. Placement, culling
//! and LOD selection all happen on the GPU; the per-frame CPU cost here is a camera
//! update, one wind tick and one interactor move, regardless of how many blades are drawn.
//!
//! Controls:
//!   WASD        — move forward/left/back/right
//!   Space/Shift — move up/down
//!   Q/E         — decrease/increase wind speed
//!   R           — toggle the roaming interactor
//!   Mouse drag  — look around (click to grab cursor)
//!   Escape      — release cursor / exit

mod v3_demo_common;

use helio::{
    required_experimental_features, required_wgpu_features, required_wgpu_limits, Camera,
    DebugDrawState, FoliageInteractor, FoliageInteractorId, FoliageLayer, FoliageTypeDescriptor,
    LightId, Renderer, RendererConfig, Scene,
};
use helio_default_graphs::build_default_graph;
use libhelio::Wind;
use v3_demo_common::{directional_light, make_material, plane_mesh, sphere_mesh};

use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

use std::collections::HashSet;
use std::sync::Arc;

/// Half-extent of the ground plane in metres.
const FIELD_HALF_EXTENT: f32 = 120.0;

fn main() {
    env_logger::init();
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
    queue: Arc<wgpu::Queue>,
    surface_format: wgpu::TextureFormat,
    renderer: Renderer,
    last_frame: std::time::Instant,
    start_time: std::time::Instant,

    cam_pos: glam::Vec3,
    cam_yaw: f32,
    cam_pitch: f32,
    keys: HashSet<KeyCode>,
    prev_keys: HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta: (f32, f32),

    wind_speed: f32,
    interactor_enabled: bool,
    interactor_id: FoliageInteractorId,
    interactor_prev_pos: glam::Vec3,
    marker_object: helio::ObjectId,

    _sun_light_id: LightId,
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
                        .with_title("Helio – Foliage")
                        .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32)),
                )
                .expect("window"),
        );

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::empty(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let surface = instance.create_surface(window.clone()).expect("surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .expect("adapter");

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Device"),
            required_features: required_wgpu_features(adapter.features()),
            required_limits: required_wgpu_limits(adapter.limits()),
            experimental_features: required_experimental_features(adapter.features()),
            ..Default::default()
        }))
        .expect("device");

        device.on_uncaptured_error(std::sync::Arc::new(|e: wgpu::Error| {
            panic!("[GPU UNCAPTURED ERROR] {:?}", e);
        }));
        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);
        let size = window.inner_size();
        surface.configure(
            &device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                width: size.width,
                height: size.height,
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: caps.alpha_modes[0],
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
                color_space: wgpu::SurfaceColorSpace::Auto,
            },
        );

        let config = RendererConfig::new(size.width, size.height, format);
        let scene = Scene::new(device.clone(), queue.clone());
        let debug_camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Debug Camera Buffer"),
            size: std::mem::size_of::<helio::DebugCameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cull_stats_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cull Stats Buffer"),
            size: 32,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let debug_state = Arc::new(std::sync::Mutex::new(DebugDrawState::default()));
        let graph = build_default_graph(
            &device,
            &queue,
            &scene,
            config,
            debug_state.clone(),
            &debug_camera_buf,
            &cull_stats_buf,
            None,
        );
        let mut renderer = Renderer::new(
            device.clone(),
            queue.clone(),
            config.surface_format,
            config.width,
            config.height,
            config.render_scale,
            config,
            scene,
            graph,
            debug_state,
            debug_camera_buf,
            cull_stats_buf,
        );

        // ── Ground ───────────────────────────────────────────────────────────
        // Flat for now: `FoliageTerrainPass` (the top-down height/slope capture the
        // placement shader samples) is a later phase, and until it exists placement falls
        // back to a plane at y=0. This mesh is what that fallback is pretending to be, so
        // the two agree and the grass sits on the ground rather than floating.
        let ground_mat = renderer.scene_mut().insert_material(make_material(
            [0.16, 0.22, 0.10, 1.0],
            0.95,
            0.0,
            [0.0, 0.0, 0.0],
            0.0,
        ));
        let ground_mesh = renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::mesh(plane_mesh(
                [0.0, 0.0, 0.0],
                FIELD_HALF_EXTENT,
            )))
            .as_mesh()
            .unwrap();
        let _ = v3_demo_common::insert_object(
            &mut renderer,
            ground_mesh,
            ground_mat,
            glam::Mat4::IDENTITY,
            FIELD_HALF_EXTENT,
        );

        // A visible marker for the roaming interactor, so the grass displacement has
        // something obviously attached to it.
        let marker_mat = renderer.scene_mut().insert_material(make_material(
            [0.8, 0.2, 0.15, 1.0],
            0.4,
            0.0,
            [0.5, 0.05, 0.0],
            2.0,
        ));
        let marker_mesh = renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::mesh(sphere_mesh([0.0, 0.0, 0.0], 0.6)))
            .as_mesh()
            .unwrap();
        let marker_object = v3_demo_common::insert_object(
            &mut renderer,
            marker_mesh,
            marker_mat,
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.6, 0.0)),
            0.6,
        )
        .expect("marker object");

        // ── Foliage ──────────────────────────────────────────────────────────
        let grass_mat = renderer.scene_mut().insert_material(make_material(
            [0.28, 0.46, 0.14, 1.0],
            0.85,
            0.0,
            [0.0, 0.0, 0.0],
            0.0,
        ));

        let grass = renderer
            .scene_mut()
            .add_foliage_type(FoliageTypeDescriptor {
                density: 40.0,
                height_range: [0.18, 0.5],
                width_range: [0.012, 0.03],
                // Everything up to 35° of slope. Flat ground here, but this is the knob
                // that keeps grass off cliff faces once real terrain is under it.
                slope_range: [0.0, 35f32.to_radians()],
                lod_distances: [8.0, 20.0, 45.0, 120.0],
                // Blades have no trunk, so the sway band is off; flutter carries the
                // body of the motion and jitter the tips.
                wind_response: [0.0, 0.35, 1.0],
                interaction_stiffness: 6.0,
                material_id: grass_mat.slot(),
                receives_interaction: true,
                casts_shadow: false,
                ..Default::default()
            });

        renderer.scene_mut().add_foliage_layer(FoliageLayer {
            types: vec![grass],
            bounds: [
                glam::Vec3::new(-FIELD_HALF_EXTENT, -1.0, -FIELD_HALF_EXTENT),
                glam::Vec3::new(FIELD_HALF_EXTENT, 4.0, FIELD_HALF_EXTENT),
            ],
            seed: 0x5EED,
        });

        let wind_speed = 4.0;
        renderer.scene_mut().set_wind(Wind {
            direction: glam::Vec3::new(1.0, 0.0, 0.35).normalize(),
            speed: wind_speed,
            gust_amplitude: 0.6,
            gust_frequency: 0.25,
            turbulence_scale: 0.05,
            ..Default::default()
        });

        let interactor_id = renderer
            .scene_mut()
            .add_foliage_interactor(FoliageInteractor {
                position: glam::Vec3::ZERO,
                radius: 1.2,
                velocity: glam::Vec3::ZERO,
            });

        // ── Lighting ─────────────────────────────────────────────────────────
        let sun_light_id = renderer
            .scene_mut()
            .insert_light(directional_light(
                [-0.35, -0.8, -0.5],
                [1.0, 0.96, 0.88],
                3.0,
            ))
            .expect("sun");

        renderer.scene_mut().flush();

        self.state = Some(AppState {
            window,
            surface,
            device,
            queue,
            surface_format: format,
            renderer,
            last_frame: std::time::Instant::now(),
            start_time: std::time::Instant::now(),
            cam_pos: glam::Vec3::new(0.0, 1.7, 12.0),
            cam_yaw: 0.0,
            cam_pitch: -0.1,
            keys: HashSet::new(),
            prev_keys: HashSet::new(),
            cursor_grabbed: false,
            mouse_delta: (0.0, 0.0),
            wind_speed,
            interactor_enabled: true,
            interactor_id,
            interactor_prev_pos: glam::Vec3::ZERO,
            marker_object,
            _sun_light_id: sun_light_id,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
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
                    state.cursor_grabbed = false;
                    let _ = state.window.set_cursor_grab(CursorGrabMode::None);
                    state.window.set_cursor_visible(true);
                } else {
                    event_loop.exit();
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ks,
                        physical_key: PhysicalKey::Code(key),
                        ..
                    },
                ..
            } => match ks {
                ElementState::Pressed => {
                    state.keys.insert(key);
                }
                ElementState::Released => {
                    state.keys.remove(&key);
                }
            },
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                if !state.cursor_grabbed {
                    let ok = state
                        .window
                        .set_cursor_grab(CursorGrabMode::Confined)
                        .or_else(|_| state.window.set_cursor_grab(CursorGrabMode::Locked))
                        .is_ok();
                    if ok {
                        state.window.set_cursor_visible(false);
                        state.cursor_grabbed = true;
                    }
                }
            }
            WindowEvent::Resized(s) if s.width > 0 && s.height > 0 => {
                state.surface.configure(
                    &state.device,
                    &wgpu::SurfaceConfiguration {
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                        format: state.surface_format,
                        width: s.width,
                        height: s.height,
                        present_mode: wgpu::PresentMode::Fifo,
                        alpha_mode: wgpu::CompositeAlphaMode::Auto,
                        view_formats: vec![],
                        desired_maximum_frame_latency: 2,
                        color_space: wgpu::SurfaceColorSpace::Auto,
                    },
                );
                state.renderer.set_render_size(s.width, s.height);
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

    fn device_event(&mut self, _: &ActiveEventLoop, _: winit::event::DeviceId, event: DeviceEvent) {
        let Some(state) = &mut self.state else { return };
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            if state.cursor_grabbed {
                state.mouse_delta.0 += dx as f32;
                state.mouse_delta.1 += dy as f32;
            }
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(s) = &self.state {
            s.window.request_redraw();
        }
    }
}

impl AppState {
    fn render(&mut self, dt: f32) {
        const SPEED: f32 = 8.0;
        const SENS: f32 = 0.002;
        const WIND_STEP: f32 = 4.0;

        // Guard against a hitch (or a breakpoint) producing a huge dt, which would jump
        // the wind clock far enough that the motion-vector pair describes a teleport and
        // TAA smears the whole screen for a frame.
        let dt = dt.clamp(0.0, 0.1);

        if self.keys.contains(&KeyCode::KeyQ) {
            self.wind_speed = (self.wind_speed - WIND_STEP * dt).max(0.0);
        }
        if self.keys.contains(&KeyCode::KeyE) {
            self.wind_speed = (self.wind_speed + WIND_STEP * dt).min(30.0);
        }
        if self.keys.contains(&KeyCode::KeyR) && !self.prev_keys.contains(&KeyCode::KeyR) {
            self.interactor_enabled = !self.interactor_enabled;
        }
        self.prev_keys = self.keys.clone();

        self.cam_yaw += self.mouse_delta.0 * SENS;
        self.cam_pitch = (self.cam_pitch - self.mouse_delta.1 * SENS).clamp(-1.4, 1.4);
        self.mouse_delta = (0.0, 0.0);

        let (sy, cy) = self.cam_yaw.sin_cos();
        let (sp, cp) = self.cam_pitch.sin_cos();
        let forward = glam::Vec3::new(sy * cp, sp, -cy * cp);
        let right = glam::Vec3::new(cy, 0.0, sy);

        if self.keys.contains(&KeyCode::KeyW) {
            self.cam_pos += forward * SPEED * dt;
        }
        if self.keys.contains(&KeyCode::KeyS) {
            self.cam_pos -= forward * SPEED * dt;
        }
        if self.keys.contains(&KeyCode::KeyA) {
            self.cam_pos -= right * SPEED * dt;
        }
        if self.keys.contains(&KeyCode::KeyD) {
            self.cam_pos += right * SPEED * dt;
        }
        if self.keys.contains(&KeyCode::Space) {
            self.cam_pos += glam::Vec3::Y * SPEED * dt;
        }
        if self.keys.contains(&KeyCode::ShiftLeft) {
            self.cam_pos -= glam::Vec3::Y * SPEED * dt;
        }

        let time = self.start_time.elapsed().as_secs_f32();

        // ── Drive the foliage frame state ────────────────────────────────────
        // Three O(1) calls. Nothing here scales with the number of blades on screen —
        // that is the whole claim the design makes, and this loop is what it looks like.
        {
            let scene = self.renderer.scene_mut();

            let mut wind = scene.wind();
            wind.speed = self.wind_speed;
            scene.set_wind(wind);
            scene.advance_wind(dt);

            let marker_pos = if self.interactor_enabled {
                let radius = 9.0;
                glam::Vec3::new(
                    (time * 0.45).cos() * radius,
                    0.6,
                    (time * 0.45).sin() * radius,
                )
            } else {
                glam::Vec3::new(0.0, -50.0, 0.0)
            };

            // Velocity is passed rather than differenced GPU-side so a fast pass leaves a
            // continuous track instead of one splat per frame.
            let velocity = if dt > 0.0 {
                (marker_pos - self.interactor_prev_pos) / dt
            } else {
                glam::Vec3::ZERO
            };
            self.interactor_prev_pos = marker_pos;

            let _ = scene.update_foliage_interactor(self.interactor_id, marker_pos, velocity);
            let _ = scene.update_object_transform(
                self.marker_object,
                glam::Mat4::from_translation(marker_pos),
            );

            scene.flush();
        }

        let size = self.window.inner_size();
        let aspect = size.width as f32 / size.height.max(1) as f32;
        let camera = Camera::perspective_look_at(
            self.cam_pos,
            self.cam_pos + forward,
            glam::Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            aspect,
            0.1,
            1000.0,
        );

        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            _ => return,
        };
        let view = output.texture.create_view(&Default::default());

        if let Err(e) = self.renderer.render(&camera, &view) {
            log::error!("Render: {:?}", e);
        }
        self.queue.present(output);
    }
}
