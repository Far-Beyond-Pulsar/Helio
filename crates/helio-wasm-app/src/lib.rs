use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::spawn_local;
use web_sys::HtmlCanvasElement;
use std::sync::Arc;

// math types used by Camera
use glam::{Mat4, Vec3};

// re-export log macros for convenience
use log::{info, error};

// When the `console_error_panic_hook` feature is enabled, we can call the
// `set_once` function at the beginning of `main` or `start` to register a
// panic hook that prints panics to the browser console.
#[wasm_bindgen(start)]
pub fn start() -> Result<(), JsValue> {
    // install panic hook for better error messages
    console_error_panic_hook::set_once();
    wasm_logger::init(wasm_logger::Config::default());

    info!("starting helio wasm app");

    // run the async renderer logic
    spawn_local(async move {
        if let Err(e) = run().await {
            error!("wasm run failure: {:?}", e);
        }
    });

    Ok(())
}

async fn run() -> Result<(), JsValue> {
    // create a canvas and attach to document body
    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    let body = document.body().unwrap();

    let canvas: HtmlCanvasElement = document
        .create_element("canvas")?
        .dyn_into::<HtmlCanvasElement>()?;
    canvas.set_width(800);
    canvas.set_height(600);
    body.append_child(&canvas)?;

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    // request_adapter returns a `Result<Option<Adapter>, _>` on wasm targets
    // so handle both the error and the missing adapter case.  annotate the
    // type to help inference later.
    let adapter: wgpu::Adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .map_err(|e| JsValue::from_str("Failed to request adapter"))?;

    let device_queue = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await
        .map_err(|e| JsValue::from_str(&format!("device error: {:?}", e)))?;
    let (device, queue): (wgpu::Device, wgpu::Queue) = device_queue;

    // construct a renderer with minimal config (surface is unused)
    let mut renderer = helio_render_v2::Renderer::new(
        Arc::<wgpu::Device>::new(device.clone()),
        Arc::<wgpu::Queue>::new(queue.clone()),
        helio_render_v2::RendererConfig::new(
            0,
            0,
            wgpu::TextureFormat::Bgra8UnormSrgb,
            helio_render_v2::features::FeatureRegistry::new(),
        ),
    ).map_err(|e| JsValue::from_str(&format!("renderer init failed: {:?}", e)))?;

    // set up a trivial camera for rendering
    // create a trivial camera (identity view_proj)
    let camera = helio_render_v2::Camera::new(glam::Mat4::IDENTITY, glam::Vec3::ZERO, 0.0);

    // create a tiny dummy texture target so we can call render
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("wasm dummy"),
        size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Bgra8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());

    // perform a single frame render to prove that it compiles and runs
    renderer
        .render(&camera, &view, 0.0)
        .map_err(|e| JsValue::from_str(&format!("render error: {:?}", e)))?;

    info!("first frame rendered (no visible content, but code executed)");

    Ok(())
}
