//! VR demo: a folder-based demo that renders a small scene (3 cubes + ground +
//! 3 lights + sky) through the OpenXR multiview render path when a headset is
//! connected, and falls back to a plain desktop mirror (forward opaque) with
//! WASD + mouse look when OpenXR initialisation fails.
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

use glam::Vec3;
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
        let wgpu_instance =
            helio_xr::create_wgpu_instance(&instance.instance, instance.system)?;
        let (device, queue) =
            helio_xr::create_wgpu_device(&instance.instance, instance.system, &wgpu_instance, xr_features())?;
        let session = helio_xr::XrSession::create(
            &instance.instance,
            instance.system,
            &wgpu_instance,
            &device,
            &queue,
        )?;
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
    last_frame: Instant,
    fps: FpsCounter,
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

        let (device, queue, surface, surface_format, alpha_mode, config, xr_owned) = match xr_owned {
            Some(bundle) => {
                let config = RendererConfig::new(
                    bundle.session.width,
                    bundle.session.height,
                    bundle.swapchain.format,
                )
                .with_render_mode(RenderMode::ForwardOpaque)
                // The graph's internal resolution must match the XR eye buffer
                // exactly (it becomes the swapchain target); no scaling.
                .with_render_scale(1.0)
                .with_xr_mode(true);
                let format = bundle.swapchain.format;
                (
                    bundle.device.clone(),
                    bundle.queue.clone(),
                    None,
                    format,
                    wgpu::CompositeAlphaMode::Opaque,
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
                let adapter = pollster::block_on(instance.request_adapter(
                    &wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::HighPerformance,
                        compatible_surface: Some(&surface),
                        force_fallback_adapter: false,
                        apply_limit_buckets: false,
                    },
                ))
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

        scene::build(&mut renderer);

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
            last_frame: Instant::now(),
            fps: FpsCounter::new(),
        };
        state.configure_surface(state.window.inner_size().width, state.window.inner_size().height);
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
                    // Headset path: render_xr() polls session events, locates the
                    // per-eye poses, uploads the stereo camera and renders both
                    // eyes in one multiview pass.
                    if let Err(e) = state.renderer.render_xr() {
                        log::error!("[XR] render_xr error: {e:?}");
                    }
                    state.input.mouse_delta = (0.0, 0.0);
                    state.fps.tick();
                    state.window.request_redraw();
                    return;
                }

                // Desktop mirror path: WASD + mouse free camera.
                state.input.update(dt);
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
