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

use helio::{Camera, Renderer, RendererConfig};

use crate::{HelioWasmApp, InputState};

fn now_secs() -> f64 {
    js_sys::Date::now() / 1000.0
}

// ── Cursor helpers ────────────────────────────────────────────────────────────

fn grab_cursor(window: &Window) {
    use winit::platform::web::WindowExtWebSys;
    if let Some(canvas) = window.canvas() {
        canvas.request_pointer_lock();
    }
}

fn release_cursor(window: &Window) {
    use winit::platform::web::WindowExtWebSys;

    if let Some(web_window) = web_sys::window() {
        if let Some(document) = web_window.document() {
            document.exit_pointer_lock();
        }
    }

    if let Some(canvas) = window.canvas() {
        let _ = canvas.style().set_property("cursor", "default");
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

        attach_canvas_to_body(&window);

        let state_cell = self.state.clone();
        let init_future = init_wgpu::<T>(window, state_cell);

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

            WindowEvent::MouseInput {
                button,
                state: elem_state,
                ..
            } => {
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
                        ElementState::Pressed => {
                            state.mouse_left_just_pressed = true;
                        }
                        ElementState::Released => {
                            state.mouse_left_just_released = true;
                        }
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
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::BROWSER_WEBGPU,
        flags: wgpu::InstanceFlags::empty(),
        ..wgpu::InstanceDescriptor::new_with_display_handle(Box::new(window.clone()))
    });

    let surface = instance
        .create_surface(window.clone())
        .expect("helio-wasm: failed to create surface");

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        })
        .await
        .expect("helio-wasm: no suitable wgpu adapter");

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("helio-wasm device"),
            required_features: helio::required_wgpu_features(adapter.features()),
            required_limits: helio::required_wgpu_limits(adapter.limits()),
            ..Default::default()
        })
        .await
        .expect("helio-wasm: failed to create device");

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

    let size = window.inner_size();
    surface.configure(
        &device,
        &wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            color_space: wgpu::SurfaceColorSpace::Auto,
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        },
    );

    let mut renderer = Renderer::new(
        device.clone(),
        queue.clone(),
        RendererConfig::new(size.width, size.height, surface_format),
    );

    let demo = T::init(
        &mut renderer,
        device.clone(),
        queue.clone(),
        size.width,
        size.height,
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
}

// ── Per-frame render helper ───────────────────────────────────────────────────

fn render_frame<T: HelioWasmApp>(state: &mut RunnerState<T>) {
    let now = now_secs();
    let dt = (now - state.last_time).min(0.1) as f32;
    let elapsed = (now - state.start_time) as f32;
    state.last_time = now;

    let delta = state.mouse_delta;
    state.mouse_delta = (0.0, 0.0);

    let just_left_pressed = state.mouse_left_just_pressed;
    let just_left_released = state.mouse_left_just_released;
    state.mouse_left_just_pressed = false;
    state.mouse_left_just_released = false;

    let input = InputState {
        keys: state.keys.clone(),
        mouse_delta: delta,
        cursor_grabbed: state.cursor_grabbed,
        cursor_pos: state.cursor_pos,
        mouse_left_just_pressed: just_left_pressed,
        mouse_left_just_released: just_left_released,
    };

    let camera = state.demo.update(&mut state.renderer, dt, elapsed, &input);

    let output = match state.surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
        e => {
            log::warn!("helio-wasm: surface error: {:?}", e);
            return;
        }
    };
    let view = output
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    if let Err(e) = state.renderer.render(&camera, &view) {
        log::error!("helio-wasm: render error: {:?}", e);
    }
    state.renderer.present(output);
}

// ── Canvas helper (WASM only) ─────────────────────────────────────────────────

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

// ── Public launch function ────────────────────────────────────────────────────

/// Launch the application and let the browser drive it through `requestAnimationFrame`.
pub fn launch<T: HelioWasmApp>() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Debug)
        .expect("helio-wasm: failed to init console_log");

    let event_loop = EventLoop::new().expect("helio-wasm: failed to create EventLoop");
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    use winit::platform::web::EventLoopExtWebSys;
    event_loop.spawn_app(WasmRunner::<T>::new());
}
