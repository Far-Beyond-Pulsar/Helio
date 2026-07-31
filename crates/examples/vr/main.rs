//! VR demo: a folder-based demo that renders a small scene (3 cubes + ground +
//! 3 lights + sky) through the OpenXR multiview render path when a headset is
//! connected, and falls back to a plain desktop mirror (forward opaque) with
//! WASD + mouse look when OpenXR initialisation fails.
//!
//! Controls:
//!   WASD / Space / Shift — fly (desktop mirror mode)
//!   Mouse drag           — look around (click to grab cursor)
//!   Escape               — release cursor / exit
//!
//! Build / run:
//!   cargo run -p examples --bin vr_demo
//!
//! Without a headset (or without a Vulkan-backed wgpu adapter) the demo logs a
//! warning and runs the desktop mirror path; with one, `renderer.render_xr()`
//! drives the headset each frame and the window stays idle.

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
struct XrHmd {
    instance: helio_xr::XrInstance,
    session: helio_xr::XrSession,
    swapchain: helio_xr::XrSwapchain,
}

/// Try to bring up OpenXR. Any failure (no runtime, no headset, non-Vulkan
/// wgpu adapter, ...) degrades to `None`, which switches the demo to desktop
/// mirror mode. `session.width/height` are the runtime-recommended per-eye
/// resolution and become the graph's internal resolution.
#[cfg(not(target_arch = "wasm32"))]
fn try_init_xr(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    surface_format: wgpu::TextureFormat,
) -> Option<XrHmd> {
    let result = (|| -> helio_xr::Result<XrHmd> {
        let instance = helio_xr::XrInstance::create("helio_vr_demo")?;
        let session =
            helio_xr::XrSession::create(&instance.instance, instance.system, device, queue)?;
        let swapchain = helio_xr::XrSwapchain::create(
            device,
            &session.session,
            session.width,
            session.height,
            surface_format,
        )?;
        Ok(XrHmd {
            instance,
            session,
            swapchain,
        })
    })();

    match result {
        Ok(hmd) => {
            log::info!(
                "[XR] OpenXR ready — {}x{} eye buffer, {} array layer(s), format {:?}",
                hmd.session.width,
                hmd.session.height,
                hmd.swapchain.array_size,
                hmd.swapchain.format,
            );
            Some(hmd)
        }
        Err(e) => {
            log::warn!(
                "[XR] OpenXR init failed ({e}); running in desktop mirror mode instead"
            );
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
    surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    surface_format: wgpu::TextureFormat,
    alpha_mode: wgpu::CompositeAlphaMode,
    renderer: Renderer,
    input: FreeCam,
    /// True when a live OpenXR session is driving `renderer.render_xr()`.
    xr_active: bool,
    last_frame: Instant,
    fps: FpsCounter,
}

impl AppState {
    fn configure_surface(&self, width: u32, height: u32) {
        self.surface.configure(
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

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Main Device"),
            required_features: required_wgpu_features(adapter.features()),
            required_limits: required_wgpu_limits(adapter.limits()),
            experimental_features: required_experimental_features(adapter.features()),
            ..Default::default()
        }))
        .expect("Failed to create device");

        device.on_uncaptured_error(std::sync::Arc::new(|e: wgpu::Error| {
            log::error!("[GPU UNCAPTURED ERROR] {e:?}");
        }));
        let info = adapter.get_info();
        log::info!("[WGPU] Backend: {:?}, Device: {}", info.backend, info.name);

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);
        let alpha_mode = surface_caps.alpha_modes[0];

        let size = window.inner_size();

        // Try OpenXR before building the renderer so the config's resolution /
        // surface format can follow the runtime's recommendation.
        #[cfg(not(target_arch = "wasm32"))]
        let hmd = try_init_xr(&device, &queue, surface_format);
        #[cfg(target_arch = "wasm32")]
        let hmd: Option<XrHmd> = None;

        let render_mode = RenderMode::ForwardOpaque;
        let config = match &hmd {
            #[cfg(not(target_arch = "wasm32"))]
            Some(hmd) => RendererConfig::new(hmd.session.width, hmd.session.height, hmd.swapchain.format)
                .with_render_mode(render_mode)
                // The graph's internal resolution must match the XR eye buffer
                // exactly (it becomes the swapchain target); no scaling.
                .with_render_scale(1.0)
                .with_xr_mode(true),
            _ => RendererConfig::new(size.width, size.height, surface_format)
                .with_render_mode(render_mode),
        };

        let mut scene = Scene::new(device.clone(), queue.clone());
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
        let xr_active = if let Some(hmd) = hmd {
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
            renderer.set_xr_session(Some(hmd.instance), Some(hmd.session), Some(hmd.swapchain));
            // No temporal AA in VR (jitter is disabled in render_xr anyway).
            renderer.set_jitter_enabled(false);
            true
        } else {
            false
        };
        #[cfg(target_arch = "wasm32")]
        let xr_active = false;

        scene::build(&mut renderer);

        let mut state = AppState {
            window,
            surface,
            device,
            queue,
            surface_format,
            alpha_mode,
            renderer,
            input: FreeCam::new(),
            xr_active,
            last_frame: Instant::now(),
            fps: FpsCounter::new(),
        };
        state.configure_surface(size.width, size.height);
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
                state.renderer.set_render_size(size.width, size.height);
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

                let output = match state.surface.get_current_texture() {
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
