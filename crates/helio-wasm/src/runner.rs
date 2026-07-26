//! Internal `WasmRunner<T>` — implements `ApplicationHandler` for
//! any `HelioWasmApp`.  Handles async wgpu init, surface management, input
//! collection, frame-timing, and cursor locking.

use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;
use std::sync::Arc;

use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

use helio::{DebugDrawState, Renderer, RendererConfig, Scene};

use crate::{HelioWasmApp, InputState};

// ── Platform-specific time helper ─────────────────────────────────────────────

#[cfg(not(target_arch = "wasm32"))]
fn now_secs() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

#[cfg(target_arch = "wasm32")]
fn now_secs() -> f64 {
    js_sys::Date::now() / 1000.0
}

// ── Cursor helpers ────────────────────────────────────────────────────────────

fn grab_cursor(window: &Window) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        window
            .set_cursor_grab(winit::window::CursorGrabMode::Locked)
            .or_else(|_| window.set_cursor_grab(winit::window::CursorGrabMode::Confined))
            .unwrap_or_default();
        window.set_cursor_visible(false);
    }
    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::WindowExtWebSys;
        if let Some(canvas) = window.canvas() {
            canvas.request_pointer_lock();
        }
    }
}

fn release_cursor(window: &Window) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = window.set_cursor_grab(winit::window::CursorGrabMode::None);
        window.set_cursor_visible(true);
    }
    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::WindowExtWebSys;

        if let Some(web_window) = web_sys::window() {
            if let Some(document) = web_window.document() {
                // Release pointer lock explicitly so demos can implement
                // hold-to-fly behaviour on right mouse button release.
                document.exit_pointer_lock();
            }
        }

        if let Some(canvas) = window.canvas() {
            let _ = canvas.style().set_property("cursor", "default");
        }
    }
}

// ── Per-frame state ───────────────────────────────────────────────────────────

struct RunnerState<T: HelioWasmApp> {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    surface_format: wgpu::TextureFormat,
    renderer: Renderer,
    demo: T,

    // Input
    keys: HashSet<KeyCode>,
    mouse_delta: (f32, f32),
    cursor_grabbed: bool,
    cursor_pos: (f32, f32),
    mouse_left_just_pressed: bool,
    mouse_left_just_released: bool,

    // Timing
    start_time: f64,
    last_time: f64,
}

// ── WasmRunner ────────────────────────────────────────────────────────────────

pub(crate) struct WasmRunner<T: HelioWasmApp> {
    state: Rc<RefCell<Option<RunnerState<T>>>>,
}

impl<T: HelioWasmApp> WasmRunner<T> {
    fn new() -> Self {
        Self {
            state: Rc::new(RefCell::new(None)),
        }
    }
}

impl<T: HelioWasmApp> ApplicationHandler for WasmRunner<T> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Only initialise once.
        if self.state.borrow().is_some() {
            return;
        }

        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title(T::title())
                        .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32)),
                )
                .expect("helio-wasm: failed to create window"),
        );

        // Attach the canvas to <body> when running in the browser.
        #[cfg(target_arch = "wasm32")]
        attach_canvas_to_body(&window);

        let state_cell = self.state.clone();
        let init_future = init_wgpu::<T>(window, state_cell);

        #[cfg(not(target_arch = "wasm32"))]
        pollster::block_on(init_future);

        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_futures::spawn_local(init_future);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(ref mut state) = *self.state.borrow_mut() else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            WindowEvent::KeyboardInput {
                event:
                    winit::event::KeyEvent {
                        physical_key: PhysicalKey::Code(key),
                        state: elem_state,
                        ..
                    },
                ..
            } => match elem_state {
                ElementState::Pressed => {
                    state.keys.insert(key);
                    if key == KeyCode::Escape && state.cursor_grabbed {
                        state.cursor_grabbed = false;
                        release_cursor(&state.window);
                    }
                }
                ElementState::Released => {
                    state.keys.remove(&key);
                }
            },

            WindowEvent::MouseInput { button, state: elem_state, .. } => {
                let grab_button = T::grab_cursor_button();

                // Handle cursor grab / release via the configured button.
                if button == grab_button {
                    match elem_state {
                        ElementState::Pressed => {
                            if !state.cursor_grabbed {
                                state.cursor_grabbed = true;
                                grab_cursor(&state.window);
                            }
                        }
                        ElementState::Released => {
                            if state.cursor_grabbed && T::release_cursor_on_grab_button_release() {
                                state.cursor_grabbed = false;
                                release_cursor(&state.window);
                            }
                        }
                    }
                }

                // Track left-button press/release for demos that need click events.
                if button == MouseButton::Left {
                    match elem_state {
                        ElementState::Pressed  => { state.mouse_left_just_pressed  = true; }
                        ElementState::Released => { state.mouse_left_just_released = true; }
                    }
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                state.cursor_pos = (position.x as f32, position.y as f32);
            }

            WindowEvent::Resized(new_size) => {
                if new_size.width > 0 && new_size.height > 0 {
                    let config = wgpu::SurfaceConfiguration {
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                        format: state.surface_format,
                        width: new_size.width,
                        height: new_size.height,
                        color_space: wgpu::SurfaceColorSpace::Auto,
                        present_mode: wgpu::PresentMode::Fifo,
                        desired_maximum_frame_latency: 2,
                        alpha_mode: wgpu::CompositeAlphaMode::Auto,
                        view_formats: vec![],
                    };
                    state.surface.configure(&state.device, &config);
                    state
                        .renderer
                        .set_render_size(new_size.width, new_size.height);
                    state
                        .demo
                        .on_resize(&mut state.renderer, new_size.width, new_size.height);
                }
            }

            WindowEvent::RedrawRequested => {
                render_frame(state);
                state.window.request_redraw();
            }

            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            if let Some(ref mut state) = *self.state.borrow_mut() {
                if state.cursor_grabbed {
                    state.mouse_delta.0 += dx as f32;
                    state.mouse_delta.1 += dy as f32;
                }
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(ref state) = *self.state.borrow() {
            state.window.request_redraw();
        }
    }
}

// ── One-shot async wgpu initialisation ────────────────────────────────────────

async fn init_wgpu<T: HelioWasmApp>(
    window: Arc<Window>,
    state_cell: Rc<RefCell<Option<RunnerState<T>>>>,
) {
    if let Some(message) = browser_webgpu_preflight_error() {
        show_startup_error(&message);
        return;
    }

    #[cfg(not(target_arch = "wasm32"))]
    let backends = wgpu::Backends::VULKAN | wgpu::Backends::GL;
    #[cfg(target_arch = "wasm32")]
    let backends = wgpu::Backends::BROWSER_WEBGPU;

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends,
        flags: wgpu::InstanceFlags::empty(),
        ..wgpu::InstanceDescriptor::new_with_display_handle(Box::new(window.clone()))
    });

    let surface = match instance.create_surface(window.clone()) {
        Ok(surface) => surface,
        Err(error) => {
            show_startup_error(&format!(
                "Helio could not create a browser WebGPU canvas surface.\n\n{error}"
            ));
            return;
        }
    };

    let adapter = match instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        })
        .await
    {
        Ok(adapter) => adapter,
        Err(error) => {
            show_startup_error(&format!(
                "Helio could not obtain a browser WebGPU adapter.\n\n\
                 navigator.gpu is present, but the browser returned no adapter. The GPU, \
                 driver, or browser configuration may be unsupported or blocklisted.\n\n{error}"
            ));
            return;
        }
    };

    let (device, queue) = match adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("helio-wasm device"),
            required_features: helio::required_wgpu_features(adapter.features()),
            required_limits: helio::required_wgpu_limits(adapter.limits()),
            experimental_features: helio::required_experimental_features(adapter.features()),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        })
        .await
    {
        Ok(device) => device,
        Err(error) => {
            show_startup_error(&format!(
                "Helio found a WebGPU adapter but could not create the required device.\n\n\
                 This adapter may not expose Helio's required indirect-draw feature or limits.\n\n{error}"
            ));
            return;
        }
    };

    device.on_uncaptured_error(std::sync::Arc::new(|e: wgpu::Error| {
        log::error!("[GPU uncaptured error] {:?}", e);
    }));

    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let caps = surface.get_capabilities(&adapter);
    let surface_format = caps
        .formats
        .iter()
        .find(|f| f.is_srgb())
        .copied()
        .unwrap_or(caps.formats[0]);

    // Browser layout can transiently report a zero-sized canvas while the newly
    // attached element is being laid out. WebGPU does not permit configuring a
    // surface (or creating its backing texture) with an empty extent.
    let size = window.inner_size();
    let width = size.width.max(1);
    let height = size.height.max(1);
    surface.configure(
        &device,
        &wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            color_space: wgpu::SurfaceColorSpace::Auto,
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        },
    );

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
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let debug_state = Arc::new(std::sync::Mutex::new(DebugDrawState::default()));

    // The demo may swap in a custom render graph (voxel meshing, injected
    // post-process effects, …); otherwise fall back to the default deferred
    // graph. `render_scale` is threaded into both the graph config and the
    // renderer so a custom graph with no TAA upscale can pin it to 1.0.
    let render_scale = T::render_scale();
    let graph = T::build_graph(
        &device,
        &queue,
        &scene,
        RendererConfig::new(width, height, surface_format).with_render_scale(render_scale),
        debug_state.clone(),
        &debug_camera_buf,
        &cull_stats_buf,
    )
    .unwrap_or_else(|| {
        helio_default_graphs::build_default_graph(
            &device,
            &queue,
            &scene,
            RendererConfig::new(width, height, surface_format).with_render_scale(render_scale),
            debug_state.clone(),
            &debug_camera_buf,
            &cull_stats_buf,
            None, // debug_overlay
        )
    });

    let mut renderer = Renderer::new(
        device.clone(),
        queue.clone(),
        surface_format,
        width,
        height,
        render_scale,
        RendererConfig::new(width, height, surface_format).with_render_scale(render_scale),
        scene,
        graph,
        debug_state,
        debug_camera_buf,
        cull_stats_buf,
    );

    let demo = T::init(
        &mut renderer,
        device.clone(),
        queue.clone(),
        width,
        height,
    );

    let now = now_secs();
    *state_cell.borrow_mut() = Some(RunnerState {
        window,
        surface,
        device,
        queue,
        surface_format,
        renderer,
        demo,
        keys: HashSet::new(),
        mouse_delta: (0.0, 0.0),
        cursor_grabbed: false,
        cursor_pos: (0.0, 0.0),
        mouse_left_just_pressed: false,
        mouse_left_just_released: false,
        start_time: now,
        last_time: now,
    });
    hide_loading_overlay();
}

// ── Per-frame render helper ───────────────────────────────────────────────────

fn render_frame<T: HelioWasmApp>(state: &mut RunnerState<T>) {
    let now = now_secs();
    let dt = (now - state.last_time).min(0.1) as f32;
    let elapsed = (now - state.start_time) as f32;
    state.last_time = now;

    let delta = state.mouse_delta;
    state.mouse_delta = (0.0, 0.0);

    let just_left_pressed  = state.mouse_left_just_pressed;
    let just_left_released = state.mouse_left_just_released;
    state.mouse_left_just_pressed  = false;
    state.mouse_left_just_released = false;

    let viewport = state.window.inner_size();
    let input = InputState {
        viewport_size: (viewport.width, viewport.height),
        keys: state.keys.clone(),
        mouse_delta: delta,
        cursor_grabbed: state.cursor_grabbed,
        cursor_pos: state.cursor_pos,
        mouse_left_just_pressed: just_left_pressed,
        mouse_left_just_released: just_left_released,
    };

    let camera = state.demo.update(&mut state.renderer, dt, elapsed, &input);

    let output = match state.surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(texture)
        | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
        error => {
            log::warn!("helio-wasm: surface error: {:?}", error);
            return;
        }
    };
    let view = output
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    if let Err(e) = state.renderer.render(&camera, &view) {
        log::error!("helio-wasm: render error: {:?}", e);
    }
    state.queue.present(output);
}

// ── Canvas helper (WASM only) ─────────────────────────────────────────────────

#[cfg(target_arch = "wasm32")]
fn attach_canvas_to_body(window: &Window) {
    use winit::platform::web::WindowExtWebSys;
    let canvas = match window.canvas() {
        Some(c) => c,
        None => return,
    };
    let web_window = match web_sys::window() {
        Some(w) => w,
        None => return,
    };
    let document = match web_window.document() {
        Some(d) => d,
        None => return,
    };
    let body = match document.body() {
        Some(b) => b,
        None => return,
    };

    // Style: full-page canvas, black background
    let style = canvas.style();
    let _ = style.set_property("width", "100%");
    let _ = style.set_property("height", "100%");
    let _ = style.set_property("display", "block");
    let _ = style.set_property("background", "#000");

    let body_style = body.style();
    let _ = body_style.set_property("margin", "0");
    let _ = body_style.set_property("overflow", "hidden");
    let _ = body_style.set_property("background", "#000");

    let _ = body.append_child(&web_sys::Element::from(canvas));
}

#[cfg(target_arch = "wasm32")]
fn browser_webgpu_preflight_error() -> Option<String> {
    let window = web_sys::window()?;
    let location = window.location();
    let href = location
        .href()
        .unwrap_or_else(|_| "the current page".to_string());

    if !window.is_secure_context() {
        if location.hostname().as_deref() == Ok("0.0.0.0") {
            let redirect = href.replacen("://0.0.0.0", "://127.0.0.1", 1);
            if location.replace(&redirect).is_ok() {
                return Some(format!(
                    "Redirecting the server bind address to a WebGPU-capable loopback origin:\n{redirect}"
                ));
            }
        }

        return Some(format!(
            "Helio requires browser WebGPU, but this page is not a secure context:\n{href}\n\n\
             Open the demo through http://localhost or http://127.0.0.1 instead of \
             http://0.0.0.0. Production deployments must use HTTPS."
        ));
    }

    let navigator = window.navigator();
    let gpu =
        js_sys::Reflect::get(navigator.as_ref(), &wasm_bindgen::JsValue::from_str("gpu")).ok();
    if gpu.is_none_or(|gpu| gpu.is_null() || gpu.is_undefined()) {
        return Some(format!(
            "Helio requires browser WebGPU, but navigator.gpu is unavailable at:\n{href}\n\n\
             Use a WebGPU-capable browser and confirm that WebGPU is enabled for this GPU and driver."
        ));
    }

    None
}

#[cfg(not(target_arch = "wasm32"))]
fn browser_webgpu_preflight_error() -> Option<String> {
    None
}

#[cfg(target_arch = "wasm32")]
fn show_startup_error(message: &str) {
    log::error!("helio-wasm startup error: {message}");

    let Some(document) = web_sys::window().and_then(|window| window.document()) else {
        return;
    };
    let panel = document
        .get_element_by_id("loading")
        .or_else(|| document.create_element("div").ok());
    let Some(panel) = panel else {
        return;
    };

    panel.set_id("helio-startup-error");
    panel.set_class_name("");
    panel.set_text_content(Some(message));
    let _ = panel.set_attribute(
        "style",
        "position:fixed;inset:0;z-index:1000;display:flex;align-items:center;\
         justify-content:center;padding:clamp(24px,8vw,96px);background:#09090b;\
         color:#e88;font:14px/1.6 system-ui,sans-serif;white-space:pre-wrap;\
         text-align:left;",
    );

    if panel.parent_node().is_none() {
        if let Some(body) = document.body() {
            let _ = body.append_child(&panel);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn show_startup_error(message: &str) {
    log::error!("helio-wasm startup error: {message}");
}

#[cfg(target_arch = "wasm32")]
fn hide_loading_overlay() {
    let Some(document) = web_sys::window().and_then(|window| window.document()) else {
        return;
    };
    if let Some(overlay) = document.get_element_by_id("loading") {
        overlay.set_class_name("hidden");
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn hide_loading_overlay() {}

/// Android entry: receive the `AndroidApp` from `android_main` and run.
#[cfg(target_os = "android")]
pub fn launch_android<T: HelioWasmApp>(app: winit::platform::android::activity::AndroidApp) {
    use winit::platform::android::EventLoopBuilderExtAndroid;

    #[cfg(not(target_arch = "wasm32"))]
    env_logger::try_init().ok();

    let event_loop = winit::event_loop::EventLoop::builder()
        .with_android_app(app)
        .build()
        .expect("helio-wasm: failed to create Android EventLoop");

    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut runner = WasmRunner::<T>::new();
    event_loop
        .run_app(&mut runner)
        .expect("helio-wasm: event loop error");
}

// ── Public launch function ────────────────────────────────────────────────────

/// Launch the demo.  Works on both native (blocking) and WASM (non-blocking).
///
/// On native this is equivalent to the standard winit run-loop.  
/// On WASM this spawns a `spawn_local` future and returns immediately; the
/// browser drives the frame loop via `requestAnimationFrame`.
pub fn launch<T: HelioWasmApp>() {
    // Logging setup
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::try_init().ok();
    }
    #[cfg(target_arch = "wasm32")]
    {
        console_error_panic_hook::set_once();
        console_log::init_with_level(log::Level::Debug)
            .expect("helio-wasm: failed to init console_log");
    }

    let event_loop = EventLoop::new().expect("helio-wasm: failed to create EventLoop");
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut runner = WasmRunner::<T>::new();
        event_loop
            .run_app(&mut runner)
            .expect("helio-wasm: event loop error");
    }

    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::EventLoopExtWebSys;
        let runner = WasmRunner::<T>::new();
        event_loop.spawn_app(runner);
    }
}
