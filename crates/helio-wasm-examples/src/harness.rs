//! Shared WASM render harness — canvas setup, wgpu init, input, render loop.
//!
//! Callers implement [`WasmScene`] and pass an instance to [`run_scene`].

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::spawn_local;
use web_sys::{HtmlCanvasElement, KeyboardEvent, MouseEvent};
use std::sync::Arc;
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashSet;

use raw_window_handle::{
    HasWindowHandle, HasDisplayHandle,
    WindowHandle, DisplayHandle, HandleError,
    WebCanvasWindowHandle, WebDisplayHandle,
    RawWindowHandle, RawDisplayHandle,
};

use helio_render_v2::{Renderer, RendererConfig, Camera};
use helio_render_v2::features::FeatureRegistryBuilder;

/// Implement this trait for each demo. The harness owns the wgpu state;
/// the scene only sees `Renderer` + time.
pub trait WasmScene: 'static {
    /// Called with a fresh [`FeatureRegistryBuilder`] so the scene can add
    /// its required features before the renderer is created.
    fn configure_features(&mut self, builder: FeatureRegistryBuilder) -> FeatureRegistryBuilder;

    /// Called once after the renderer exists so the scene can add meshes,
    /// lights, etc.
    fn setup_scene(&mut self, renderer: &mut Renderer);

    /// Called every animation frame.  `time` is seconds since the first frame.
    fn update_scene(&mut self, renderer: &mut Renderer, time: f32);
}

// ── CanvasHandle newtype ───────────────────────────────────────────────────────

struct CanvasHandle<'a>(&'a HtmlCanvasElement);

impl<'a> HasWindowHandle for CanvasHandle<'a> {
    fn window_handle(&self) -> Result<WindowHandle<'_>, HandleError> {
        let js_val: &wasm_bindgen::JsValue = self.0.as_ref();
        let web_canvas = WebCanvasWindowHandle::from_wasm_bindgen_0_2(js_val);
        let raw = RawWindowHandle::WebCanvas(web_canvas);
        Ok(unsafe { WindowHandle::borrow_raw(raw) })
    }
}

impl<'a> HasDisplayHandle for CanvasHandle<'a> {
    fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
        let web_disp = WebDisplayHandle::new();
        let raw = RawDisplayHandle::Web(web_disp);
        Ok(unsafe { DisplayHandle::borrow_raw(raw) })
    }
}

// ── Helper to load spotlight sprite ──────────────────────────────────────────

pub(crate) fn load_sprite() -> (Vec<u8>, u32, u32) {
    let img = image::load_from_memory(include_bytes!("../../../spotlight.png"))
        .unwrap_or_else(|_| image::DynamicImage::new_rgba8(1, 1))
        .into_rgba8();
    let (w, h) = img.dimensions();
    (img.into_raw(), w, h)
}

// ── Public entry-point ────────────────────────────────────────────────────────

/// Spin up the canvas, wgpu, and the animation loop with the given scene.
/// Call this from `#[wasm_bindgen(start)]`.
pub fn run_scene<S: WasmScene>(scene: S) {
    console_error_panic_hook::set_once();
    std::panic::set_hook(Box::new(|info| {
        console_error_panic_hook::hook(info);
        let _ = display_error(&format!("panic: {}", info));
    }));
    wasm_logger::init(wasm_logger::Config::default());

    spawn_local(async move {
        if let Err(e) = run(scene).await {
            let msg = format!("wasm run failure: {:?}", e);
            log::error!("{}", msg);
            let _ = display_error(&msg);
        }
    });
}

fn display_error(msg: &str) -> Result<(), JsValue> {
    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    let body = document.body().unwrap();
    let el = document.create_element("p")?;
    el.set_text_content(Some(msg));
    body.append_child(&el)?;
    Ok(())
}

// ── Core async init + loop ────────────────────────────────────────────────────

async fn run<S: WasmScene>(mut scene: S) -> Result<(), JsValue> {
    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    let body = document.body().unwrap();

    // ── Canvas ───────────────────────────────────────────────────────────────
    let canvas: HtmlCanvasElement = document
        .create_element("canvas")?
        .dyn_into::<HtmlCanvasElement>()?;

    let dpr = window.device_pixel_ratio();
    let initial_width  = (window.inner_width()?.as_f64().unwrap_or(800.0) * dpr) as u32;
    let initial_height = (window.inner_height()?.as_f64().unwrap_or(600.0) * dpr) as u32;
    canvas.set_width(initial_width);
    canvas.set_height(initial_height);
    canvas.set_attribute("style", "position:absolute;top:0;left:0;width:100%;height:100%;")?;
    body.set_attribute("style", "margin:0")?;
    canvas.set_attribute("tabindex", "0")?;
    body.append_child(&canvas)?;
    canvas.focus().ok();

    // ── Input ─────────────────────────────────────────────────────────────────
    #[derive(Default)]
    struct InputState {
        keys: HashSet<String>,
        last_keys: HashSet<String>,
        mouse_dx: f32,
        mouse_dy: f32,
        locked: bool,
    }
    let input = Rc::new(RefCell::new(InputState::default()));

    {
        let input = input.clone();
        let cb = Closure::wrap(Box::new(move |e: KeyboardEvent| {
            e.prevent_default();
            input.borrow_mut().keys.insert(e.code());
        }) as Box<dyn FnMut(_)>);
        window.add_event_listener_with_callback("keydown", cb.as_ref().unchecked_ref())?;
        cb.forget();
    }
    {
        let input = input.clone();
        let cb = Closure::wrap(Box::new(move |e: KeyboardEvent| {
            e.prevent_default();
            input.borrow_mut().keys.remove(&e.code());
        }) as Box<dyn FnMut(_)>);
        window.add_event_listener_with_callback("keyup", cb.as_ref().unchecked_ref())?;
        cb.forget();
    }
    {
        let input = input.clone();
        let cb = Closure::wrap(Box::new(move |e: MouseEvent| {
            let mut i = input.borrow_mut();
            if i.locked {
                i.mouse_dx += e.movement_x() as f32;
                i.mouse_dy += e.movement_y() as f32;
            }
        }) as Box<dyn FnMut(_)>);
        canvas.add_event_listener_with_callback("mousemove", cb.as_ref().unchecked_ref())?;
        cb.forget();
    }
    {
        let canvas2 = canvas.clone();
        let cb = Closure::wrap(Box::new(move |_e: MouseEvent| {
            let _ = canvas2.request_pointer_lock();
        }) as Box<dyn FnMut(_)>);
        canvas.add_event_listener_with_callback("click", cb.as_ref().unchecked_ref())?;
        cb.forget();
    }
    {
        let input = input.clone();
        let cb = Closure::wrap(Box::new(move |_e: web_sys::Event| {
            let locked = web_sys::window()
                .unwrap()
                .document()
                .unwrap()
                .pointer_lock_element()
                .is_some();
            input.borrow_mut().locked = locked;
        }) as Box<dyn FnMut(_)>);
        web_sys::window().unwrap().document().unwrap()
            .add_event_listener_with_callback("pointerlockchange", cb.as_ref().unchecked_ref())?;
        cb.forget();
    }

    // ── wgpu ─────────────────────────────────────────────────────────────────
    let width  = canvas.width();
    let height = canvas.height();

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .map_err(|_| JsValue::from_str("No WebGPU adapter — check browser support"))?;

    let canvas_handle = CanvasHandle(&canvas);
    let surface = unsafe { instance.create_surface(&canvas_handle) }
        .map_err(|e| JsValue::from_str(&format!("create_surface: {:?}", e)))?;
    // canvas_handle must stay alive for the duration of the function because
    // Surface<'_> borrows the window handle through its lifetime parameter.
    let surface = Rc::new(surface);

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await
        .map_err(|e| JsValue::from_str(&format!("device: {:?}", e)))?;

    let surface_caps = surface.get_capabilities(&adapter);
    let surface_format = surface_caps
        .formats
        .iter()
        .find(|f| f.is_srgb())
        .copied()
        .unwrap_or(surface_caps.formats[0]);

    // ── Feature registry ──────────────────────────────────────────────────────
    let base_builder = helio_render_v2::features::FeatureRegistry::builder();
    let feature_registry = scene.configure_features(base_builder).build();

    // ── Surface config ────────────────────────────────────────────────────────
    let config = Rc::new(RefCell::new(wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_format,
        width,
        height,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: surface_caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    }));
    surface.configure(&device, &config.borrow());

    // ── Renderer ──────────────────────────────────────────────────────────────
    let mut renderer = Renderer::new(
        Arc::new(device.clone()),
        Arc::new(queue.clone()),
        RendererConfig::new(width, height, surface_format, feature_registry),
    ).map_err(|e| JsValue::from_str(&format!("renderer: {:?}", e)))?;

    scene.setup_scene(&mut renderer);

    let renderer = Rc::new(RefCell::new(renderer));
    let scene    = Rc::new(RefCell::new(scene));

    // ── Resize listener ───────────────────────────────────────────────────────
    {
        let canvas   = canvas.clone();
        let surface  = surface.clone();
        let device   = device.clone();
        let renderer = renderer.clone();
        let config   = config.clone();
        let win      = web_sys::window().unwrap();
        let win2     = win.clone();
        let cb = Closure::wrap(Box::new(move |_e: web_sys::UiEvent| {
            let dpr   = win2.device_pixel_ratio();
            let new_w = (win2.inner_width().unwrap().as_f64().unwrap_or(canvas.width() as f64) * dpr) as u32;
            let new_h = (win2.inner_height().unwrap().as_f64().unwrap_or(canvas.height() as f64) * dpr) as u32;
            canvas.set_width(new_w);
            canvas.set_height(new_h);
            { let mut cfg = config.borrow_mut(); cfg.width = new_w; cfg.height = new_h; surface.configure(&device, &cfg); }
            renderer.borrow_mut().set_render_size(new_w, new_h);
        }) as Box<dyn FnMut(_)>);
        win.add_event_listener_with_callback("resize", cb.as_ref().unchecked_ref()).unwrap();
        cb.forget();
    }

    // ── Camera state ──────────────────────────────────────────────────────────
    let mut eye   = glam::Vec3::new(0.0, 2.5, 7.0);
    let mut yaw   = 0.0_f32;
    let mut pitch = -0.2_f32;
    let auto_rotate = Rc::new(RefCell::new(0u32));
    let start_time  = Rc::new(RefCell::new(
        js_sys::Date::now()
    ));

    let proj = glam::Mat4::perspective_rh_gl(
        std::f32::consts::FRAC_PI_2,
        width as f32 / height as f32,
        0.1, 100.0,
    );
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let forward  = glam::Vec3::new(sy * cp, sp, -cy * cp);
    let view     = glam::Mat4::look_at_rh(eye, eye + forward, glam::Vec3::Y);
    let mut camera = Camera::new(proj * view, eye, 0.0);

    // ── Render first frame ────────────────────────────────────────────────────
    {
        let frame = surface.get_current_texture()
            .map_err(|e| JsValue::from_str(&format!("frame: {:?}", e)))?;
        let view_tex = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        renderer.borrow_mut().render(&camera, &view_tex, 0.0)
            .map_err(|e| JsValue::from_str(&format!("render: {:?}", e)))?;
        frame.present();
    }

    // ── Animation loop ────────────────────────────────────────────────────────
    // Clone canvas so the closure owns its own JS ref; the original stays
    // borrowed by canvas_handle (which must outlive the Surface).
    let canvas = canvas.clone();
    let f: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let g = f.clone();

    *g.borrow_mut() = Some(Closure::wrap(Box::new(move || {
        // input
        {
            let mut i = input.borrow_mut();

            if *auto_rotate.borrow() < 300 {
                yaw += 0.005;
                *auto_rotate.borrow_mut() += 1;
            }
            yaw   += i.mouse_dx * 0.002;
            pitch -= i.mouse_dy * 0.002;
            i.mouse_dx = 0.0;
            i.mouse_dy = 0.0;
            pitch = pitch.clamp(-1.5, 1.5);

            let (sy, cy) = yaw.sin_cos();
            let (sp, cp) = pitch.sin_cos();
            let forward = glam::Vec3::new(sy * cp, sp, -cy * cp);
            let right   = glam::Vec3::new(cy, 0.0, sy);
            let up      = glam::Vec3::Y;
            let mut dir = glam::Vec3::ZERO;
            if i.keys.contains("KeyW")     { dir += forward; }
            if i.keys.contains("KeyS")     { dir -= forward; }
            if i.keys.contains("KeyA")     { dir -= right; }
            if i.keys.contains("KeyD")     { dir += right; }
            if i.keys.contains("Space")    { dir += up; }
            if i.keys.contains("ShiftLeft"){ dir -= up; }
            if dir.length_squared() > 0.0 { eye += dir.normalize() * 0.05; }

            i.last_keys = i.keys.clone();
        }

        // sync canvas backing-store
        let dpr      = web_sys::window().unwrap().device_pixel_ratio();
        let target_w = (canvas.client_width()  as f64 * dpr).max(1.0) as u32;
        let target_h = (canvas.client_height() as f64 * dpr).max(1.0) as u32;
        if target_w != canvas.width() || target_h != canvas.height() {
            canvas.set_width(target_w);
            canvas.set_height(target_h);
            { let mut cfg = config.borrow_mut(); cfg.width = target_w; cfg.height = target_h;
              surface.configure(&device, &cfg); }
            renderer.borrow_mut().set_render_size(target_w, target_h);
        }

        // camera
        let cur_w = canvas.width();
        let cur_h = canvas.height();
        let proj  = glam::Mat4::perspective_rh_gl(
            std::f32::consts::FRAC_PI_2,
            cur_w as f32 / cur_h as f32,
            0.1, 100.0,
        );
        let (sy, cy) = yaw.sin_cos();
        let (sp, cp) = pitch.sin_cos();
        let forward  = glam::Vec3::new(sy * cp, sp, -cy * cp);
        let view     = glam::Mat4::look_at_rh(eye, eye + forward, glam::Vec3::Y);
        camera = Camera::new(proj * view, eye, 0.0);

        // time
        let now  = js_sys::Date::now();
        let time = ((now - *start_time.borrow()) / 1000.0) as f32;

        // scene update
        {
            let mut r = renderer.borrow_mut();
            scene.borrow_mut().update_scene(&mut r, time);
        }

        // render
        let frame = surface.get_current_texture().unwrap();
        let view  = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        renderer.borrow_mut().render(&camera, &view, 0.0).unwrap();
        frame.present();

        web_sys::window().unwrap()
            .request_animation_frame(f.borrow().as_ref().unwrap().as_ref().unchecked_ref())
            .unwrap();
    }) as Box<dyn FnMut()>));

    web_sys::window().unwrap()
        .request_animation_frame(g.borrow().as_ref().unwrap().as_ref().unchecked_ref())?;

    Ok(())
}
