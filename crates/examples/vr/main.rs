//! VR demo: a long showcase hallway (one renderer feature per bay) rendered
//! through the OpenXR render path when a headset is connected, and a plain
//! desktop mirror (forward opaque) with WASD + mouse look when OpenXR
//! initialisation fails.
//!
//! In XR mode, two small emissive cubes are parented to the controller grip
//! poses every frame, so they follow the player's hands exactly.
//!
//! When a headset is present, the Vulkan instance and device are created
//! *through* OpenXR (`xrCreateVulkanInstanceKHR` / `xrCreateVulkanDeviceKHR`)
//! via `helio_xr::create_wgpu_instance` / `create_wgpu_device` so the runtime's
//! required extensions are enabled and the HMD's GPU is used. The mirror window
//! stays idle in XR mode.
//!
//! Controls:
//!   WASD / Space / Shift — fly (desktop mirror mode)
//!   Mouse drag           — look around (click to grab cursor)
//!   Escape               — release cursor / exit
//!   Left stick           — move (XR)
//!   Right stick          — turn (XR)
//!
//! Build / run:
//!   cargo run -p examples --bin vr_demo
//!
//! Without a headset (or without OpenXR + a Vulkan-capable GPU) the demo logs a
//! warning and runs the desktop mirror path; with one, `renderer.render_xr()`
//! drives the headset each frame.

mod input;
mod scene;
#[path = "../v3_demo_common.rs"]
mod v3_demo_common;

use std::sync::Arc;
use std::time::Instant;

use glam::{Mat4, Vec3};
use helio::{
    required_experimental_features, required_wgpu_features, required_wgpu_limits, Camera,
    DebugDrawState, RenderMode, Renderer, RendererConfig, Scene,
};
use helio_default_graphs::build_forward_opaque_graph;
use input::FreeCam;
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

// ── OpenXR state (native only) ───────────────────────────────────────────────

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
        let instance = helio_xr::XrInstance::create("helio_vr_demo")?;
        let wgpu_instance = helio_xr::create_wgpu_instance(&instance.instance, instance.system)?;
        let (adapter, device, queue) = helio_xr::create_wgpu_device(
            &instance.instance,
            instance.system,
            &wgpu_instance,
            xr_features(),
        )?;
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

// ── FPS counter ──────────────────────────────────────────────────────────────

struct FpsCounter {
    frames: u32,
    last_print: Instant,
}

impl FpsCounter {
    fn new() -> Self {
        Self {
            frames: 0,
            last_print: Instant::now(),
        }
    }

    /// Counts frames; logs the rolling FPS once per second.
    fn tick(&mut self) {
        self.frames += 1;
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_print).as_secs_f32();
        if elapsed >= 1.0 {
            log::info!("[fps] {:.0}", self.frames as f32 / elapsed);
            self.frames = 0;
            self.last_print = now;
        }
    }
}

// ── App ──────────────────────────────────────────────────────────────────────

struct App {
    state: Option<AppState>,
}

impl App {
    fn new() -> Self {
        Self { state: None }
    }
}

struct AppState {
    window: Arc<Window>,
    /// Present only in desktop mode; idle/absent while the headset is driven.
    surface: Option<wgpu::Surface<'static>>,
    surface_format: wgpu::TextureFormat,
    alpha_mode: wgpu::CompositeAlphaMode,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    renderer: Renderer,
    input: FreeCam,
    /// True when a live OpenXR session is driving `renderer.render_xr()`.
    xr_active: bool,
    /// Controller actions, and a handle to the session to sync them against.
    ///
    /// The session is cloned rather than borrowed from the renderer: `openxr::Session` is
    /// reference-counted, and the renderer owns the original.
    #[cfg(not(target_arch = "wasm32"))]
    xr_input: Option<(helio_xr::XrInput, helio_xr::XrSessionHandle)>,
    /// Player locomotion, applied as the XR stage transform. Position is the stage
    /// origin in world space; yaw is snap/smooth turn about it.
    player_position: Vec3,
    player_yaw: f32,
    animated: scene::Animated,
    start_time: Instant,
    last_frame: Instant,
    fps: FpsCounter,
}

impl AppState {
    /// Read the controllers and walk the player, then publish the result as the XR
    /// stage transform.
    ///
    /// Movement is head-relative on the yaw axis only: pitching your head must not send
    /// you into the floor or the ceiling, which is both disorienting and the fastest way
    /// to make someone sick. Turning is smooth here for simplicity; snap turn is the
    /// gentler default for people prone to motion sickness and is a one-line change.
    #[cfg(not(target_arch = "wasm32"))]
    fn update_locomotion(&mut self, dt: f32) {
        const MOVE_SPEED: f32 = 2.5;
        const TURN_SPEED: f32 = 1.8;
        /// Sticks rest slightly off-centre on most hardware; without a deadzone the
        /// player drifts continuously while standing still.
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

        // Yaw-only basis, so "forward" is where you are facing on the floor plane.
        let (sin, cos) = self.player_yaw.sin_cos();
        let forward = Vec3::new(-sin, 0.0, -cos);
        let right = Vec3::new(cos, 0.0, -sin);
        self.player_position += (forward * move_stick.y + right * move_stick.x) * MOVE_SPEED * dt;

        self.renderer.set_xr_stage_transform(
            glam::Mat4::from_translation(self.player_position)
                * glam::Mat4::from_rotation_y(self.player_yaw),
        );
    }

    /// Reparent the controller cubes to the current grip poses. Located at the
    /// renderer's most recent XR display time, so the cubes stay glued to the
    /// controllers. Hands that are not tracked keep their last transform.
    #[cfg(not(target_arch = "wasm32"))]
    fn update_hands(&mut self) {
        // Offset each cube a little forward and up in grip space so it floats just
        // past the palm instead of being hidden inside the physical controller.
        const HAND_OFFSET: Vec3 = Vec3::new(0.0, 0.03, -0.12);

        let Some(time) = self.renderer.xr_last_display_time() else {
            return;
        };
        let world_from_stage =
            Mat4::from_translation(self.player_position) * Mat4::from_rotation_y(self.player_yaw);
        let Some((input, session)) = &mut self.xr_input else {
            return;
        };
        let Ok(poses) = input.grip_pose_matrices(session, time, &world_from_stage) else {
            return;
        };
        for (i, pose) in poses.into_iter().enumerate() {
            if let Some(world) = pose {
                let transform = world * Mat4::from_translation(HAND_OFFSET);
                let _ = self
                    .renderer
                    .scene_mut()
                    .update_object_transform(self.animated.hand_cubes[i], transform);
            }
        }
    }

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
                        .with_title("Helio — VR Demo")
                        .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32)),
                )
                .expect("Failed to create window"),
        );

        // Try OpenXR before creating wgpu: a headset session requires the
        // Vulkan instance/device to be created through OpenXR.
        #[cfg(not(target_arch = "wasm32"))]
        let xr_bundle = try_init_xr();
        #[cfg(target_arch = "wasm32")]
        let xr_bundle: Option<XrBundle> = None;

        // ── Device / queue / surface / config ────────────────────────────────
        #[cfg(not(target_arch = "wasm32"))]
        let xr_owned = xr_bundle;
        #[cfg(target_arch = "wasm32")]
        let xr_owned: Option<XrBundle> = xr_bundle;

        let (device, queue, surface, surface_format, alpha_mode, config, xr_owned) = match xr_owned
        {
            Some(bundle) => {
                let config = RendererConfig::new(
                    bundle.session.width,
                    bundle.session.height,
                    bundle.swapchain.format,
                )
                .with_render_mode(RenderMode::ForwardOpaque)
                // The graph's internal resolution must match the XR eye buffer
                // exactly (it becomes the swapchain target); no scaling. Note:
                // the graph is intentionally built in single-layer (non-multiview)
                // mode — render_xr renders each eye separately (dual-pass stereo)
                // so every existing pass and shader works unchanged.
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
                log::warn!("[XR] no headset — running desktop mirror (forward opaque)");
                let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                    backends: wgpu::Backends::all(),
                    flags: wgpu::InstanceFlags::empty(),
                    ..wgpu::InstanceDescriptor::new_without_display_handle()
                });
                let surface = instance
                    .create_surface(window.clone())
                    .expect("Failed to create surface");
                let adapter =
                    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::HighPerformance,
                        compatible_surface: Some(&surface),
                        force_fallback_adapter: false,
                        apply_limit_buckets: false,
                    }))
                    .expect("Failed to find adapter");
                let (device, queue) =
                    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                        label: Some("Main Device"),
                        required_features: required_wgpu_features(adapter.features()),
                        required_limits: required_wgpu_limits(adapter.limits()),
                        experimental_features: required_experimental_features(adapter.features()),
                        ..Default::default()
                    }))
                    .expect("Failed to create device");
                let caps = surface.get_capabilities(&adapter);
                let surface_format = caps
                    .formats
                    .iter()
                    .find(|f| f.is_srgb())
                    .copied()
                    .unwrap_or(caps.formats[0]);
                let alpha_mode = caps.alpha_modes[0];
                let size = window.inner_size();
                let config = RendererConfig::new(size.width, size.height, surface_format)
                    .with_render_mode(RenderMode::ForwardOpaque);
                (
                    Arc::new(device),
                    Arc::new(queue),
                    Some(surface),
                    surface_format,
                    alpha_mode,
                    config,
                    None,
                )
            }
        };
        let device_arc = Arc::clone(&device);
        device.on_uncaptured_error(std::sync::Arc::new(move |e: wgpu::Error| {
            let _ = &device_arc;
            log::error!("[GPU UNCAPTURED ERROR] {e:?}");
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

        let graph = build_forward_opaque_graph(
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
        renderer.set_editor_mode(true);

        #[cfg(not(target_arch = "wasm32"))]
        let mut xr_input = None;
        #[cfg(not(target_arch = "wasm32"))]
        let xr_active = if let Some(bundle) = xr_owned {
            let template = Camera::perspective_look_at(
                Vec3::new(0.0, 1.6, 0.0),
                Vec3::new(0.0, 1.6, -1.0),
                glam::Vec3::Y,
                std::f32::consts::FRAC_PI_4,
                1.0,
                0.05,
                200.0,
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
            // No temporal AA in VR (jitter is disabled in render_xr anyway).
            renderer.set_jitter_enabled(false);
            true
        } else {
            false
        };
        #[cfg(target_arch = "wasm32")]
        let xr_active = false;

        let animated = scene::build(&mut renderer);

        let state = AppState {
            window,
            surface,
            surface_format,
            alpha_mode,
            device,
            queue,
            renderer,
            input: FreeCam::new(),
            xr_active,
            #[cfg(not(target_arch = "wasm32"))]
            xr_input,
            player_position: Vec3::ZERO,
            player_yaw: 0.0,
            animated,
            start_time: Instant::now(),
            last_frame: Instant::now(),
            fps: FpsCounter::new(),
        };
        state.configure_surface(
            state.window.inner_size().width,
            state.window.inner_size().height,
        );
        self.state = Some(state);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };

        match event {
            WindowEvent::CloseRequested => {
                log::info!("Shutting down");
                event_loop.exit();
            }

            WindowEvent::Resized(size) if size.width > 0 && size.height > 0 => {
                state.configure_surface(size.width, size.height);
                // In XR mode the graph resolution is fixed by the headset's eye
                // buffer; resizing the mirror window must not rebuild the graph
                // at the window's resolution (that destroys resources cached in
                // pass bind groups and breaks the XR composite).
                if !state.xr_active {
                    state.renderer.set_render_size(size.width, size.height);
                }
            }

            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        ..
                    },
                ..
            } => {
                if state.input.cursor_grabbed {
                    state.input.cursor_grabbed = false;
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
                    state.input.keys.insert(key);
                }
                ElementState::Released => {
                    state.input.keys.remove(&key);
                }
            },

            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                if !state.input.cursor_grabbed {
                    let grabbed = state
                        .window
                        .set_cursor_grab(CursorGrabMode::Confined)
                        .or_else(|_| state.window.set_cursor_grab(CursorGrabMode::Locked))
                        .is_ok();
                    if grabbed {
                        state.window.set_cursor_visible(false);
                        state.input.cursor_grabbed = true;
                    }
                }
            }

            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = now.duration_since(state.last_frame).as_secs_f32().min(0.05);
                state.last_frame = now;

                #[cfg(not(target_arch = "wasm32"))]
                if state.xr_active {
                    state.update_locomotion(dt);
                    let scene_time = state.start_time.elapsed().as_secs_f32();
                    scene::animate(&mut state.renderer, &mut state.animated, scene_time);
                    state.update_hands();
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
                    state.input.mouse_delta = (0.0, 0.0);
                    state.fps.tick();
                    state.window.request_redraw();
                    return;
                }

                // Desktop mirror path: WASD + mouse free camera.
                state.input.update(dt);
                let scene_time = state.start_time.elapsed().as_secs_f32();
                scene::animate(&mut state.renderer, &mut state.animated, scene_time);
                let size = state.window.inner_size();
                let aspect = size.width as f32 / size.height.max(1) as f32;
                let camera = state.input.camera(aspect);

                let Some(surface) = &state.surface else {
                    state.window.request_redraw();
                    return;
                };
                let output = match surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(texture)
                    | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
                    _ => {
                        log::warn!("surface acquire failed");
                        state.window.request_redraw();
                        return;
                    }
                };
                let view = output
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                if let Err(e) = state.renderer.render(&camera, &view) {
                    log::error!("Render error: {e:?}");
                }
                state.queue.present(output);
                state.fps.tick();
                state.window.request_redraw();
            }

            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        let Some(state) = &mut self.state else { return };
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            if state.input.cursor_grabbed {
                state.input.mouse_delta.0 += dx as f32;
                state.input.mouse_delta.1 += dy as f32;
            }
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }
}

fn main() {
    env_logger::init();
    log::info!("Starting Helio VR demo (OpenXR)");

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("Event loop error");
}
