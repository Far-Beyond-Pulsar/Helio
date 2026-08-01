//! Foliage demo — GPU-driven grass over an open field.
//!
//! Exercises the whole foliage authoring path: a registered foliage type, a layer, the
//! global wind clock, and a moving interactor that pushes grass aside. Placement, culling
//! and LOD selection all happen on the GPU; the per-frame CPU cost here is a camera
//! update, one wind tick and one interactor move, regardless of how many blades are drawn.
//!
//! Rendered through the OpenXR path when a headset is connected (see `examples/vr`), and
//! a plain desktop mirror with WASD + mouse look when OpenXR initialisation fails. The
//! foliage graph (`build_default_graph`) is used unchanged in both modes: `render_xr()`
//! drives the normal 2D graph twice, once per eye, so no foliage-specific XR code is
//! needed beyond the bootstrap and per-frame branch below.
//!
//! Controls:
//!   WASD        — move forward/left/back/right (desktop mirror mode)
//!   Space/Shift — move up/down (desktop mirror mode)
//!   Q/E         — decrease/increase wind speed
//!   R           — toggle the roaming interactor
//!   Mouse drag  — look around (click to grab cursor)
//!   Escape      — release cursor / exit
//!   Left stick  — move (XR)
//!   Right stick — turn (XR)

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

// ── OpenXR state (native only) ───────────────────────────────────────────────
//
// Mirrors `examples/vr/main.rs` exactly: a headset session requires the Vulkan
// instance/device to be created *through* OpenXR, so this has to happen before
// any wgpu device exists, and it either fully succeeds or the demo falls back
// to the desktop mirror path.

#[cfg(not(target_arch = "wasm32"))]
struct XrBundle {
    instance: helio_xr::XrInstance,
    session: helio_xr::XrSession,
    swapchain: helio_xr::XrSwapchain,
    input: helio_xr::XrInput,
    wgpu_instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

/// The features the XR device is asked for. `create_wgpu_device` masks this
/// down to what the HMD's Vulkan adapter actually supports.
#[cfg(not(target_arch = "wasm32"))]
fn xr_features() -> wgpu::Features {
    let required = wgpu::Features::TEXTURE_BINDING_ARRAY
        | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
        | wgpu::Features::INDIRECT_FIRST_INSTANCE
        | wgpu::Features::MULTIVIEW
        | wgpu::Features::MULTISAMPLE_ARRAY;
    let optional = wgpu::Features::MULTI_DRAW_INDIRECT_COUNT
        | wgpu::Features::TIMESTAMP_QUERY
        | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
        | wgpu::Features::VERTEX_WRITABLE_STORAGE;
    required | optional
}

/// Bring up the full OpenXR stack: OpenXR instance → wgpu Vulkan instance →
/// wgpu device → session → swapchain. Any failure degrades to `None`, which
/// switches the demo to desktop mirror mode.
#[cfg(not(target_arch = "wasm32"))]
fn try_init_xr() -> Option<XrBundle> {
    let result = (|| -> helio_xr::Result<XrBundle> {
        let instance = helio_xr::XrInstance::create("helio_foliage_demo")?;
        let wgpu_instance =
            helio_xr::create_wgpu_instance(&instance.instance, instance.system)?;
        let (adapter, device, queue) =
            helio_xr::create_wgpu_device(&instance.instance, instance.system, &wgpu_instance, xr_features())?;
        // Actions must be declared and their bindings suggested BEFORE the session is
        // created; the runtime resolves them at session creation and will not accept new
        // suggestions afterwards.
        let input = helio_xr::XrInput::new(&instance.instance)?;
        let session = helio_xr::XrSession::create(
            &instance.instance,
            instance.system,
            &wgpu_instance,
            &device,
            &queue,
        )?;
        // Attach exactly once, after creation and before the first sync.
        input.attach(&session.session)?;
        let swapchain = helio_xr::XrSwapchain::create(
            &device,
            &session.session,
            session.width,
            session.height,
            wgpu::TextureFormat::Rgba8UnormSrgb,
        )?;
        log::info!(
            "[XR] OpenXR ready — {}x{} eye buffer, {} array layer(s), format {:?}",
            session.width,
            session.height,
            swapchain.array_size,
            swapchain.format,
        );
        Ok(XrBundle {
            instance,
            session,
            swapchain,
            input,
            wgpu_instance,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    })();

    match result {
        Ok(bundle) => Some(bundle),
        Err(e) => {
            log::warn!("[XR] OpenXR init failed ({e}); running in desktop mirror mode instead");
            None
        }
    }
}

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
    /// Present only in desktop mode; idle/absent while the headset is driven.
    surface: Option<wgpu::Surface<'static>>,
    alpha_mode: wgpu::CompositeAlphaMode,
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

    /// True when a live OpenXR session is driving `renderer.render_xr()`.
    xr_active: bool,
    /// Controller actions, and a handle to the session to sync them against.
    #[cfg(not(target_arch = "wasm32"))]
    xr_input: Option<(helio_xr::XrInput, helio_xr::XrSessionHandle)>,
    /// Player locomotion, applied as the XR stage transform. Position is the stage
    /// origin in world space; yaw is snap/smooth turn about it.
    player_position: glam::Vec3,
    player_yaw: f32,
}

impl AppState {
    fn configure_surface(&self, width: u32, height: u32) {
        let Some(surface) = &self.surface else { return };
        surface.configure(
            &self.device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: self.surface_format,
                color_space: wgpu::SurfaceColorSpace::Auto,
                width: width.max(1),
                height: height.max(1),
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: self.alpha_mode,
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            },
        );
    }

    /// Read the controllers and walk the player, then publish the result as the XR
    /// stage transform. Movement is head-relative on the yaw axis only, matching
    /// `examples/vr`.
    #[cfg(not(target_arch = "wasm32"))]
    fn update_locomotion(&mut self, dt: f32) {
        const MOVE_SPEED: f32 = 2.5;
        const TURN_SPEED: f32 = 1.8;
        const DEADZONE: f32 = 0.15;

        let Some((input, session)) = &self.xr_input else {
            return;
        };
        let Ok(controls) = input.sync(session) else {
            return;
        };

        let deadzone = |v: glam::Vec2| -> glam::Vec2 {
            if v.length() < DEADZONE {
                glam::Vec2::ZERO
            } else {
                v
            }
        };
        let move_stick = deadzone(controls.left_stick);
        let turn_stick = deadzone(controls.right_stick);

        self.player_yaw -= turn_stick.x * TURN_SPEED * dt;

        let (sin, cos) = self.player_yaw.sin_cos();
        let forward = glam::Vec3::new(-sin, 0.0, -cos);
        let right = glam::Vec3::new(cos, 0.0, -sin);
        self.player_position += (forward * move_stick.y + right * move_stick.x) * MOVE_SPEED * dt;

        self.renderer.set_xr_stage_transform(
            glam::Mat4::from_translation(self.player_position)
                * glam::Mat4::from_rotation_y(self.player_yaw),
        );
    }
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

        // `HELIO_FOLIAGE=0` builds the graph with no foliage passes at all, so the ground
        // and the rest of the scene can be judged without them. Worth having as a runtime
        // switch rather than a rebuild: "is this the foliage or the base renderer?" is the
        // first question to answer for any artefact in this demo, and answering it should
        // not cost a recompile.
        let foliage_enabled = std::env::var("HELIO_FOLIAGE")
            .map(|value| value != "0")
            .unwrap_or(true);
        eprintln!("[foliage_demo] foliage passes: {}", if foliage_enabled { "ON" } else { "OFF" });

        // Density is an allocation ceiling, so it is stated up front and the arena is
        // sized to hold it — 256 blades/m² over a 128 m ring is ~256 MiB of blade arena.
        // Overridable so the cost/quality trade is visible rather than baked in.
        let blades_per_m2 = std::env::var("HELIO_FOLIAGE_DENSITY")
            .ok()
            .and_then(|value| value.parse::<f32>().ok())
            .unwrap_or(256.0);

        // Try OpenXR before creating wgpu: a headset session requires the
        // Vulkan instance/device to be created through OpenXR.
        #[cfg(not(target_arch = "wasm32"))]
        let xr_owned = try_init_xr();
        #[cfg(target_arch = "wasm32")]
        let xr_owned: Option<XrBundle> = None;

        let (device, queue, surface, surface_format, alpha_mode, mut config, xr_owned) = match xr_owned
        {
            Some(bundle) => {
                let config = RendererConfig::new(
                    bundle.session.width,
                    bundle.session.height,
                    bundle.swapchain.format,
                )
                // The graph's internal resolution must match the XR eye buffer exactly
                // (it becomes the swapchain target); no scaling. The graph is built in
                // single-layer (non-multiview) mode — render_xr renders each eye
                // separately (dual-pass stereo) so every existing pass and shader,
                // including the whole foliage path, works unchanged.
                .with_render_scale(1.0);
                // PC mirror surface: the OpenXR-created device presents both eye
                // buffers side-by-side into this window.
                let mirror_surface = bundle
                    .wgpu_instance
                    .create_surface(window.clone())
                    .expect("Failed to create mirror surface");
                let caps = mirror_surface.get_capabilities(&bundle.adapter);
                let mirror_format = caps
                    .formats
                    .iter()
                    .find(|f| f.is_srgb())
                    .copied()
                    .unwrap_or(caps.formats[0]);
                let mirror_alpha = caps.alpha_modes[0];
                let mirror_size = window.inner_size();
                mirror_surface.configure(
                    &bundle.device,
                    &wgpu::SurfaceConfiguration {
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                        format: mirror_format,
                        color_space: wgpu::SurfaceColorSpace::Auto,
                        width: mirror_size.width.max(1),
                        height: mirror_size.height.max(1),
                        present_mode: wgpu::PresentMode::Fifo,
                        alpha_mode: mirror_alpha,
                        view_formats: vec![],
                        desired_maximum_frame_latency: 2,
                    },
                );
                (
                    bundle.device.clone(),
                    bundle.queue.clone(),
                    Some(mirror_surface),
                    mirror_format,
                    mirror_alpha,
                    config,
                    Some(bundle),
                )
            }
            None => {
                log::warn!("[XR] no headset — running desktop mirror");
                let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                    backends: wgpu::Backends::all(),
                    flags: wgpu::InstanceFlags::empty(),
                    ..wgpu::InstanceDescriptor::new_without_display_handle()
                });
                let surface = instance.create_surface(window.clone()).expect("surface");
                let adapter =
                    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::HighPerformance,
                        compatible_surface: Some(&surface),
                        force_fallback_adapter: false,
                        apply_limit_buckets: false,
                    }))
                    .expect("adapter");

                let (device, queue) =
                    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                        label: Some("Device"),
                        required_features: required_wgpu_features(adapter.features()),
                        required_limits: required_wgpu_limits(adapter.limits()),
                        experimental_features: required_experimental_features(adapter.features()),
                        ..Default::default()
                    }))
                    .expect("device");
                let device = Arc::new(device);
                let queue = Arc::new(queue);

                let caps = surface.get_capabilities(&adapter);
                let format = caps
                    .formats
                    .iter()
                    .find(|f| f.is_srgb())
                    .copied()
                    .unwrap_or(caps.formats[0]);
                let alpha_mode = caps.alpha_modes[0];
                let size = window.inner_size();
                let config = RendererConfig::new(size.width, size.height, format);
                (device, queue, Some(surface), format, alpha_mode, config, None)
            }
        };
        config.enable_foliage = foliage_enabled;
        config.foliage_blades_per_m2 = Some(blades_per_m2);

        device.on_uncaptured_error(std::sync::Arc::new(|e: wgpu::Error| {
            panic!("[GPU UNCAPTURED ERROR] {:?}", e);
        }));

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

        #[cfg(not(target_arch = "wasm32"))]
        let mut xr_input = None;
        #[cfg(not(target_arch = "wasm32"))]
        let xr_active = if let Some(bundle) = xr_owned {
            let template = Camera::perspective_look_at(
                glam::Vec3::new(0.0, 1.6, 12.0),
                glam::Vec3::new(0.0, 1.6, 11.0),
                glam::Vec3::Y,
                std::f32::consts::FRAC_PI_4,
                1.0,
                0.1,
                1000.0,
            );
            renderer.set_xr_camera(template);
            renderer.set_xr_mirror_format(surface_format);
            let session_handle = bundle.session.session.clone();
            xr_input = Some((bundle.input, session_handle));
            renderer.set_xr_session(
                Some(bundle.instance),
                Some(bundle.session),
                Some(bundle.swapchain),
            );
            // No temporal AA/TSR jitter in VR (render_xr disables it anyway).
            renderer.set_jitter_enabled(false);
            true
        } else {
            false
        };
        #[cfg(target_arch = "wasm32")]
        let xr_active = false;

        // Indoors, but the sky still drives ambient — and `SkyPass` is what establishes the
        // colour target each frame, so its absence is what made geometry smear over itself.
        // See `Renderer::rebuild_graph_if_sky_changed`.
        renderer.scene_mut().insert_actor(helio::SceneActor::sky(
            helio::SkyActor::new().with_sky_color([0.05, 0.07, 0.11]),
        ));

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
        // Culling is opted out of for the ground, via `INSTANCE_FLAG_ALWAYS_VISIBLE`.
        //
        // A 240 m plane is the case a single bounding sphere describes worst: the sphere's
        // radius is set by the diagonal, so it is enormous next to the geometry actually
        // inside it. It culls essentially nothing useful, and getting the radius even
        // slightly wrong deletes the entire ground the moment a corner leaves the frustum.
        // One object always being submitted costs a single draw; the alternative is a
        // whole-screen artefact.
        let _ = renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::object(helio::ObjectDescriptor {
                mesh: ground_mesh,
                material: ground_mat,
                transform: glam::Mat4::IDENTITY,
                bounds: [0.0, 0.0, 0.0, FIELD_HALF_EXTENT * std::f32::consts::SQRT_2],
                flags: libhelio::INSTANCE_FLAG_ALWAYS_VISIBLE,
                groups: helio::GroupMask::NONE,
                movability: None,
                user_tag: 0,
            }));

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
                density: blades_per_m2,
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
            has_infinite_extent: true,
        });

        let wind_speed = 2.0;
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
        let sun_light_id = renderer.scene_mut().insert_light(directional_light(
            [-0.35, -0.8, -0.5],
            [1.0, 0.96, 0.88],
            3.0,
        ));

        renderer.scene_mut().flush();

        let state = AppState {
            window,
            surface,
            alpha_mode,
            device,
            queue,
            surface_format,
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
            xr_active,
            #[cfg(not(target_arch = "wasm32"))]
            xr_input,
            player_position: glam::Vec3::ZERO,
            player_yaw: 0.0,
        };
        state.configure_surface(state.window.inner_size().width, state.window.inner_size().height);
        self.state = Some(state);
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
                state.configure_surface(s.width, s.height);
                // In XR mode the graph resolution is fixed by the headset's eye
                // buffer; resizing the mirror window must not rebuild the graph
                // at the window's resolution (that destroys resources cached in
                // pass bind groups and breaks the XR composite).
                if !state.xr_active {
                    state.renderer.set_render_size(s.width, s.height);
                }
            }
            WindowEvent::RedrawRequested => {
                let now = std::time::Instant::now();
                let dt = (now - state.last_frame).as_secs_f32();
                state.last_frame = now;

                #[cfg(not(target_arch = "wasm32"))]
                if state.xr_active {
                    state.update_locomotion(dt);
                    state.update_foliage_frame(dt);
                    // Headset path: render_xr() polls session events, locates the
                    // per-eye poses, uploads the stereo camera and renders both
                    // eyes. The mirror surface (if any) receives both eye buffers
                    // side by side via the renderer's mirror blit.
                    let mirror = match &state.surface {
                        Some(surface) => match surface.get_current_texture() {
                            wgpu::CurrentSurfaceTexture::Success(texture)
                            | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => Some(texture),
                            _ => None,
                        },
                        None => None,
                    };
                    let mirror_view = mirror.as_ref().map(|t| {
                        t.texture
                            .create_view(&wgpu::TextureViewDescriptor::default())
                    });
                    if let Err(e) = state.renderer.render_xr(mirror_view.as_ref()) {
                        log::error!("[XR] render_xr error: {e:?}");
                    }
                    if let Some(output) = mirror {
                        state.queue.present(output);
                    }
                    state.mouse_delta = (0.0, 0.0);
                    state.window.request_redraw();
                    return;
                }

                state.render_desktop(dt);
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
    /// Advances wind + the roaming interactor and flushes the scene. Shared between
    /// the desktop and XR redraw paths — nothing here depends on how the camera for
    /// this frame is being driven.
    fn update_foliage_frame(&mut self, dt: f32) {
        const WIND_STEP: f32 = 4.0;

        // Guard against a hitch (or a breakpoint) producing a huge dt, which would jump
        // the wind clock far enough that the motion-vector pair describes a teleport and
        // TAA/TSR smears the whole screen for a frame.
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

        let time = self.start_time.elapsed().as_secs_f32();

        // ── Drive the foliage frame state ────────────────────────────────────
        // Three O(1) calls. Nothing here scales with the number of blades on screen —
        // that is the whole claim the design makes, and this loop is what it looks like.
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

    /// Desktop mirror path: WASD + mouse free camera, then a normal render.
    fn render_desktop(&mut self, dt: f32) {
        const SPEED: f32 = 8.0;
        const SENS: f32 = 0.002;

        let dt = dt.clamp(0.0, 0.1);

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

        self.update_foliage_frame(dt);

        let Some(surface) = &self.surface else {
            return;
        };
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

        let output = match surface.get_current_texture() {
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
