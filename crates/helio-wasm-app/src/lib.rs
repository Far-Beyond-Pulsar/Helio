use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::spawn_local;
use web_sys::{HtmlCanvasElement, KeyboardEvent, MouseEvent};
use std::sync::Arc;
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashSet;

// world bounds used when generating probe billboards (same as sky example)
const RC_WORLD_MIN: [f32; 3] = [-10.0, -0.3, -10.0];
const RC_WORLD_MAX: [f32; 3] = [10.0, 8.0, 10.0];

// math types used by Camera (qualified to avoid unused warnings)

// import raw-window-handle helpers used by our wrapper type below.
use raw_window_handle::{
    HasWindowHandle, HasDisplayHandle,
    WindowHandle, DisplayHandle, HandleError,
    WebCanvasWindowHandle, WebDisplayHandle,
    RawWindowHandle, RawDisplayHandle,
};

// helper to load the spotlight image used for billboards; mirrors example
fn load_sprite() -> (Vec<u8>, u32, u32) {
    let img = image::load_from_memory(include_bytes!("../../../spotlight.png"))
        .unwrap_or_else(|_| image::DynamicImage::new_rgba8(1, 1))
        .into_rgba8();
    let (w, h) = img.dimensions();
    (img.into_raw(), w, h)
}

// `HtmlCanvasElement` is foreign, so we cannot implement external traits on it
// due to Rust's orphan rules.  Instead we create a thin newtype wrapper which
// we control and implement the two required traits for that wrapper.

// helper for building a grid of radiance probe billboards; copied from the
// render_v2_sky example.  We toggle between these and the point-light
// billboards when the user presses Digit3.
#[allow(dead_code)]
fn probe_billboards(world_min: [f32; 3], world_max: [f32; 3]) -> Vec<helio_render_v2::features::BillboardInstance> {
    use helio_render_v2::features::radiance_cascades::PROBE_DIMS;
    const COLORS: [[f32; 4]; 4] = [
        [0.0, 1.0, 1.0, 0.85],
        [0.0, 1.0, 0.0, 0.80],
        [1.0, 1.0, 0.0, 0.75],
        [1.0, 0.35, 0.0, 0.70],
    ];
    // screen_scale=true: sizes are angular (multiplied by distance), giving constant apparent size
    const SIZES: [[f32; 2]; 4] = [
        [0.035, 0.035],  // cascade 0 — finest (4096 probes) — tiny dots
        [0.075, 0.075],  // cascade 1
        [0.140, 0.140],  // cascade 2
        [0.260, 0.260],  // cascade 3 — coarsest (8 probes) — large markers
    ];
    let mut out = Vec::new();
    for (c, &dim) in PROBE_DIMS.iter().enumerate() {
        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    let x = world_min[0] + (i as f32 + 0.5) / dim as f32 * (world_max[0] - world_min[0]);
                    let y = world_min[1] + (j as f32 + 0.5) / dim as f32 * (world_max[1] - world_min[1]);
                    let z = world_min[2] + (k as f32 + 0.5) / dim as f32 * (world_max[2] - world_min[2]);
                    out.push(helio_render_v2::features::BillboardInstance::new([x, y, z], SIZES[c])
                        .with_color(COLORS[c])
                        .with_screen_scale(true));
                }
            }
        }
    }
    out
}
struct CanvasHandle<'a>(&'a HtmlCanvasElement);

impl<'a> HasWindowHandle for CanvasHandle<'a> {
    fn window_handle(&self) -> Result<WindowHandle<'_>, HandleError> {
        let js_val: &wasm_bindgen::JsValue = self.0.as_ref();
        let web_canvas = WebCanvasWindowHandle::from_wasm_bindgen_0_2(js_val);
        let raw = RawWindowHandle::WebCanvas(web_canvas);
        // SAFETY: handle points back to the same canvas element we hold a
        // reference to, so its pointers are valid for the lifetime of the
        // handle.
        Ok(unsafe { WindowHandle::borrow_raw(raw) })
    }
}

impl<'a> HasDisplayHandle for CanvasHandle<'a> {
    fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
        // the web display handle doesn't carry any meaningful state; a default
        // constructed value is acceptable for wgpu on web.
        let web_disp = WebDisplayHandle::new();
        let raw = RawDisplayHandle::Web(web_disp);
        Ok(unsafe { DisplayHandle::borrow_raw(raw) })
    }
}

// we can create a surface directly from the HtmlCanvasElement on wasm

// renderer/scene types we will exercise
use helio_render_v2::{
    Renderer, RendererConfig, Camera, SceneLight, SkyAtmosphere,
    VolumetricClouds,
};
use helio_render_v2::features::{
    FeatureRegistry, LightingFeature, ShadowsFeature,
    BloomFeature, BillboardsFeature,
    radiance_cascades::RadianceCascadesFeature,
};

// re-export log macros for convenience
use log::{info, error};

// When the `console_error_panic_hook` feature is enabled, we can call the
// `set_once` function at the beginning of `main` or `start` to register a
// panic hook that prints panics to the browser console.
#[wasm_bindgen(start)]
pub fn start() -> Result<(), JsValue> {
    // install panic hook for better error messages
    console_error_panic_hook::set_once();
    // also write any panic message into the document body so that it is
    // visible even if the browser console is hidden.
    std::panic::set_hook(Box::new(|info| {
        console_error_panic_hook::hook(info);
        let _ = display_error(&format!("panic: {}", info));
    }));
    wasm_logger::init(wasm_logger::Config::default());

    info!("starting helio wasm app");

    // run the async renderer logic and surface errors to the page
    spawn_local(async move {
        if let Err(e) = run().await {
            let msg = format!("wasm run failure: {:?}", e);
            error!("{}", msg);
            let _ = display_error(&msg);
        }
    });

    Ok(())
}

// helper that appends a paragraph to the document body containing the
// provided message.  Used for surfacing errors that the console log might not
// be visible to end users.
fn display_error(msg: &str) -> Result<(), JsValue> {
    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    let body = document.body().unwrap();

    let error_el = document.create_element("p")?;
    error_el.set_text_content(Some(msg));
    body.append_child(&error_el)?;
    Ok(())
}

async fn run() -> Result<(), JsValue> {
    // --- setup DOM and canvas ------------------------------------------------
    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    let body = document.body().unwrap();

    let canvas: HtmlCanvasElement = document
        .create_element("canvas")?
        .dyn_into::<HtmlCanvasElement>()?;
    canvas.set_width(800);
    canvas.set_height(600);
    canvas.set_attribute("style", "border:1px solid red;")?;
    // make the canvas focusable and give it focus so keyboard events fire
    canvas.set_attribute("tabindex", "0")?;
    body.append_child(&canvas)?;
    canvas.focus().ok();

    // input tracking state shared across event callbacks & animation loop
    #[derive(Default)]
    struct InputState {
        keys: HashSet<String>,
        last_keys: HashSet<String>,
        mouse_dx: f32,
        mouse_dy: f32,
        locked: bool,
    }
    let input = std::rc::Rc::new(std::cell::RefCell::new(InputState::default()));

    // keyboard listeners (use physical `code` instead of localized key string)
    {
        let input = input.clone();
        let keydown = Closure::wrap(Box::new(move |e: KeyboardEvent| {
            let code = e.code();
            web_sys::console::log_1(&wasm_bindgen::JsValue::from_str(&format!("keydown {}", code)));
            e.prevent_default();
            input.borrow_mut().keys.insert(code);
        }) as Box<dyn FnMut(_)>);
        window.add_event_listener_with_callback("keydown", keydown.as_ref().unchecked_ref())?;
        keydown.forget();
    }
    {
        let input = input.clone();
        let keyup = Closure::wrap(Box::new(move |e: KeyboardEvent| {
            let code = e.code();
            web_sys::console::log_1(&wasm_bindgen::JsValue::from_str(&format!("keyup {}", code)));
            e.prevent_default();
            input.borrow_mut().keys.remove(&code);
        }) as Box<dyn FnMut(_)>);
        window.add_event_listener_with_callback("keyup", keyup.as_ref().unchecked_ref())?;
        keyup.forget();
    }

    // mouse movement for camera look
    {
        let input = input.clone();
        let move_cb = Closure::wrap(Box::new(move |e: MouseEvent| {
            let mut i = input.borrow_mut();
            if i.locked {
                i.mouse_dx += e.movement_x() as f32;
                i.mouse_dy += e.movement_y() as f32;
            }
        }) as Box<dyn FnMut(_)>);
        canvas.add_event_listener_with_callback("mousemove", move_cb.as_ref().unchecked_ref())?;
        move_cb.forget();
    }
    // request pointer lock on click
    {
        let canvas2 = canvas.clone();
        let click = Closure::wrap(Box::new(move |_e: web_sys::MouseEvent| {
            let _ = canvas2.request_pointer_lock();
        }) as Box<dyn FnMut(_)>);
        canvas.add_event_listener_with_callback("click", click.as_ref().unchecked_ref())?;
        click.forget();
    }
    // track lock state
    {
        let input = input.clone();
        let lockchange = Closure::wrap(Box::new(move |_e: web_sys::Event| {
            let locked = web_sys::window()
                .unwrap()
                .document()
                .unwrap()
                .pointer_lock_element()
                .is_some();
            input.borrow_mut().locked = locked;
        }) as Box<dyn FnMut(_)>);
        web_sys::window().unwrap().document().unwrap().add_event_listener_with_callback("pointerlockchange", lockchange.as_ref().unchecked_ref())?;
        lockchange.forget();
    }

    // record the canvas dimensions for renderer configuration
    let width: u32 = canvas.width();
    let height: u32 = canvas.height();

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    // request_adapter returns an Adapter or fails with an error.
    let adapter: wgpu::Adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .map_err(|_e| JsValue::from_str("No WebGPU adapter available – check your browser support or flags"))?;

    // wrap the canvas in our newtype so it satisfies the trait bounds for
    // surface creation.  the `unsafe` is required by wgpu but there's nothing
    // unsafe about passing our wrapper.
    let canvas_handle = CanvasHandle(&canvas);
    let surface = unsafe { instance.create_surface(&canvas_handle) }
        .map_err(|e| JsValue::from_str(&format!("create_surface failed: {:?}", e)))?;
    // shareable reference for the animation callback
    let surface = Rc::new(surface);

    let device_queue = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await
        .map_err(|e| JsValue::from_str(&format!("device error: {:?}", e)))?;
    let (device, queue): (wgpu::Device, wgpu::Queue) = device_queue;

    // choose a sane configuration from the adapter's surface capabilities
    // unlike native code we can't rely on the `get_default_config` helper
    // (it currently returns an `alpha_mode` that ends up being transparent on
    // some browsers which makes the canvas appear white). replicate the logic
    // used in the desktop examples to pick a sensible format & alpha mode.
    let surface_caps = surface.get_capabilities(&adapter);
    let surface_format = surface_caps
        .formats
        .iter()
        .find(|f| f.is_srgb())
        .copied()
        .unwrap_or(surface_caps.formats[0]);

    // register features just like the sky example, including bloom and a
    // billboards sprite; ray queries not available on web so this will be
    // false, but we keep the logic here for parity.
    let has_ray = adapter.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY);
    let (sprite_rgba, sprite_w, sprite_h) = load_sprite();
    let mut registry_builder = FeatureRegistry::builder()
        .with_feature(LightingFeature::new())
        .with_feature(BloomFeature::new().with_intensity(0.3).with_threshold(1.2))
        .with_feature(ShadowsFeature::new()
            .with_atlas_size(1024)
            .with_max_lights(4))
        .with_feature(BillboardsFeature::new().with_sprite(sprite_rgba, sprite_w, sprite_h).with_max_instances(5000));
    if has_ray {
        registry_builder = registry_builder.with_feature(
            RadianceCascadesFeature::new()
                .with_world_bounds(RC_WORLD_MIN, RC_WORLD_MAX),
        );
    }
    let feature_registry = registry_builder.build();

    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_format,
        width,
        height,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: surface_caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    info!("surface config: {:?}", config);
    surface.configure(&device, &config);


    let mut renderer = helio_render_v2::Renderer::new(
        Arc::<wgpu::Device>::new(device.clone()),
        Arc::<wgpu::Queue>::new(queue.clone()),
        helio_render_v2::RendererConfig::new(
            width,
            height,
            surface_format,
            feature_registry,
        ),
    ).map_err(|e| JsValue::from_str(&format!("renderer init failed: {:?}", e)))?;

    // add a few cubes and a ground/roof like the sky example
    info!("creating meshes");
    // deliberately spread cubes wide so they can't all be hidden by camera
    let cube1  = renderer.create_mesh_cube([ 0.0, 0.5,  0.0], 0.5);
    let cube2  = renderer.create_mesh_cube([-4.0, 0.4, -2.0], 0.4);
    let cube3  = renderer.create_mesh_cube([ 4.0, 0.3,  2.0], 0.3);
    let ground = renderer.create_mesh_plane([0.0, 0.0, 0.0], 20.0);
    let roof   = renderer.create_mesh_rect3d([0.0, 2.85, 0.0], [4.5, 0.15, 4.5]);
    info!("cube meshes created");
    renderer.add_object(&cube1,  None, glam::Mat4::IDENTITY);
    info!("count after cube1: {}", renderer.object_count());
    renderer.add_object(&cube2,  None, glam::Mat4::IDENTITY);
    info!("count after cube2: {}", renderer.object_count());
    renderer.add_object(&cube3,  None, glam::Mat4::IDENTITY);
    info!("count after cube3: {}", renderer.object_count());
    renderer.add_object(&ground, None, glam::Mat4::IDENTITY);
    info!("count after ground: {}", renderer.object_count());
    renderer.add_object(&roof,   None, glam::Mat4::IDENTITY);
    info!("count after roof: {}", renderer.object_count());
    info!("meshes added");

    // lights + sky from sky example (static noon).  record the sun light
    // id and make room for a varying sun angle that the user will control.
    let init_sun_dir = glam::Vec3::new(1.0_f32.cos() * 0.3, 1.0_f32.sin(), 0.5).normalize();
    let init_light_dir = [-init_sun_dir.x, -init_sun_dir.y, -init_sun_dir.z];
    let init_elev = init_sun_dir.y.clamp(-1.0, 1.0);
    let init_lux = (init_elev * 3.0).clamp(0.0, 1.0);
    let mut sun_angle = 1.0_f32; // matches the value used above
    let sun_light_id = renderer.add_light(SceneLight::directional(init_light_dir, [1.0, 0.85, 0.7], (init_lux * 0.35).max(0.01)));
    renderer.add_light(SceneLight::point([ 0.0, 2.5,  0.0], [1.0, 0.85, 0.6],  4.0, 8.0));
    renderer.add_light(SceneLight::point([-2.5, 2.0, -1.5], [0.4, 0.6,  1.0],  3.5, 7.0));
    renderer.add_light(SceneLight::point([ 2.5, 1.8,  1.5], [1.0, 0.3,  0.3],  3.0, 6.0));
    renderer.set_sky_atmosphere(Some(
        SkyAtmosphere::new()
            .with_sun_intensity(22.0)
            .with_exposure(4.0)
            .with_mie_g(0.76)
            .with_clouds(
                VolumetricClouds::new()
                    .with_coverage(0.30)
                    .with_density(0.7)
                    .with_layer(800.0, 1800.0)
                    .with_wind([1.0, 0.0], 0.08),
            ),
    ));
    renderer.set_skylight(Some(helio_render_v2::Skylight::new().with_intensity(0.08).with_tint([1.0,1.0,1.0])));
    renderer.set_sky_atmosphere(Some(
        SkyAtmosphere::new()
            .with_sun_intensity(22.0)
            .with_exposure(4.0)
            .with_mie_g(0.76)
    ));
    // add light billboards (one per point light); keep handles so we can swap
    // them out when the user toggles Digit3.
    let mut billboard_ids = Vec::new();
    billboard_ids.push(renderer.add_billboard(helio_render_v2::features::BillboardInstance::new([0.0, 2.5, 0.0],[0.35,0.35]).with_color([1.0,0.85,0.6,1.0])));
    billboard_ids.push(renderer.add_billboard(helio_render_v2::features::BillboardInstance::new([-2.5,2.0,-1.5],[0.35,0.35]).with_color([0.4,0.6,1.0,1.0])));
    billboard_ids.push(renderer.add_billboard(helio_render_v2::features::BillboardInstance::new([2.5,1.8,1.5],[0.35,0.35]).with_color([1.0,0.3,0.3,1.0])));
    let mut probe_vis = false;

    // camera state that we'll update each frame (match example start)
    let mut eye = glam::Vec3::new(0.0, 2.5, 7.0);
    let mut yaw = 0.0_f32;
    let mut pitch = -0.2_f32;
    // automatic rotation counter wrapped in Rc so closure can mutate it
    let auto_rotate = Rc::new(RefCell::new(0u32));
    let proj = glam::Mat4::perspective_rh_gl(std::f32::consts::FRAC_PI_2, width as f32 / height as f32, 0.1, 100.0);
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let forward = glam::Vec3::new(sy * cp, sp, -cy * cp);
    let view = glam::Mat4::look_at_rh(eye, eye + forward, glam::Vec3::Y);
    let view_proj = proj * view;
    let mut camera = helio_render_v2::Camera::new(view_proj, eye, 0.0);

    // convert renderer to refcell for mutable access in the animation closure
    let renderer = std::rc::Rc::new(std::cell::RefCell::new(renderer));

    // render a first frame immediately
    let frame = surface
        .get_current_texture()
        .map_err(|e| JsValue::from_str(&format!("failed to acquire frame: {:?}", e)))?;
    let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
    renderer
        .borrow_mut()
        .render(&camera, &view, 0.0)
        .map_err(|e| JsValue::from_str(&format!("render error: {:?}", e)))?;
    frame.present();
    info!("first frame rendered to canvas");

    // begin animation loop -------------------------------------------------
    let input = input.clone();            // already declared earlier
    let surface = surface.clone();        // Rc<Surface> cloned for the closure
    let width = width;
    let height = height;
    let f: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let g = f.clone();
    *g.borrow_mut() = Some(Closure::wrap(Box::new(move || {
        // update camera and handle input commands
        {
            let mut i = input.borrow_mut();

            // automatic yaw during the first couple hundred frames so the user
            // sees the other cubes without needing to move the mouse.
            if *auto_rotate.borrow() < 300 {
                yaw += 0.005;
                *auto_rotate.borrow_mut() += 1;
            }

            // look around with mouse (after or during auto phase)
            yaw += i.mouse_dx * 0.002;
            pitch += i.mouse_dy * 0.002;
            i.mouse_dx = 0.0;
            i.mouse_dy = 0.0;
            pitch = pitch.clamp(-1.5, 1.5);

            // compute forward/right using same trig as the native example
            let (sy, cy) = yaw.sin_cos();
            let (sp, cp) = pitch.sin_cos();
            let forward = glam::Vec3::new(sy * cp, sp, -cy * cp);
            let right = glam::Vec3::new(cy, 0.0, sy);
            let up = glam::Vec3::Y;
            let mut dir = glam::Vec3::ZERO;
            if i.keys.contains("KeyW") { dir += forward; }
            if i.keys.contains("KeyS") { dir -= forward; }
            if i.keys.contains("KeyA") { dir -= right; }
            if i.keys.contains("KeyD") { dir += right; }
            if i.keys.contains("Space") { dir += up; }
            if i.keys.contains("ShiftLeft") { dir -= up; }
            if dir.length_squared() > 0.0 {
                eye += dir.normalize() * 0.05;
            }

            // continuous sun control
            if i.keys.contains("KeyQ") { sun_angle -= 0.002; }
            if i.keys.contains("KeyE") { sun_angle += 0.002; }

            // figure out which keys were just pressed this frame so we can
            // perform one-shot toggles rather than repeating while held.
            let mut just_pressed = Vec::new();
            for k in i.keys.iter() {
                if !i.last_keys.contains(k) {
                    just_pressed.push(k.clone());
                }
            }

            if just_pressed.contains(&"Digit3".to_string()) {
                probe_vis = !probe_vis;
                // rebuild billboard list
                let mut r = renderer.borrow_mut();
                for id in billboard_ids.drain(..) { r.remove_billboard(id); }
                if probe_vis {
                    for b in probe_billboards(RC_WORLD_MIN, RC_WORLD_MAX) {
                        billboard_ids.push(r.add_billboard(b));
                    }
                } else {
                    billboard_ids.push(r.add_billboard(helio_render_v2::features::BillboardInstance::new([0.0, 2.5, 0.0],[0.35,0.35]).with_color([1.0,0.85,0.6,1.0])));
                    billboard_ids.push(r.add_billboard(helio_render_v2::features::BillboardInstance::new([-2.5,2.0,-1.5],[0.35,0.35]).with_color([0.4,0.6,1.0,1.0])));
                    billboard_ids.push(r.add_billboard(helio_render_v2::features::BillboardInstance::new([2.5,1.8,1.5],[0.35,0.35]).with_color([1.0,0.3,0.3,1.0])));
                }
            }
            if just_pressed.contains(&"Digit4".to_string()) {
                let mut r = renderer.borrow_mut();
                let _ = r.start_live_portal_default();
            }

            // remember keys for next frame
            i.last_keys = i.keys.clone();
        }

        let proj = glam::Mat4::perspective_rh_gl(std::f32::consts::FRAC_PI_2, width as f32 / height as f32, 0.1, 100.0);
        // build view using forward vector derived from yaw/pitch
        let (sy, cy) = yaw.sin_cos();
        let (sp, cp) = pitch.sin_cos();
        let forward = glam::Vec3::new(sy * cp, sp, -cy * cp);
        let view = glam::Mat4::look_at_rh(eye, eye + forward, glam::Vec3::Y);
        let view_proj = proj * view;
        camera = helio_render_v2::Camera::new(view_proj, eye, 0.0);

        // update directional sun light before rendering
        {
            let sun_dir = glam::Vec3::new(
                sun_angle.cos() * 0.3,
                sun_angle.sin(),
                0.5,
            ).normalize();
            let light_dir = [-sun_dir.x, -sun_dir.y, -sun_dir.z];
            let sun_elev = sun_dir.y.clamp(-1.0, 1.0);
            let sun_lux = (sun_elev * 3.0).clamp(0.0, 1.0);
            let sun_color = [
                1.0_f32.min(1.0 + (1.0 - sun_elev) * 0.3),
                (0.85 + sun_elev * 0.15).clamp(0.0, 1.0),
                (0.7  + sun_elev * 0.3 ).clamp(0.0, 1.0),
            ];
            renderer.borrow_mut().update_light(sun_light_id, SceneLight::directional(light_dir, sun_color, (sun_lux * 0.35).max(0.01)));
        }

        // draw
        let frame = surface.get_current_texture().unwrap();
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        // debug: log number of registered scene objects so we can tell if the
        // extra cubes are actually present in the renderer.
        web_sys::console::log_1(&wasm_bindgen::JsValue::from_str(&format!("obj count {}", renderer.borrow().object_count())));
        renderer.borrow_mut().render(&camera, &view, 0.0).unwrap();
        frame.present();

        web_sys::window().unwrap()
            .request_animation_frame(f.borrow().as_ref().unwrap().as_ref().unchecked_ref()).unwrap();
    }) as Box<dyn FnMut()>));
    web_sys::window().unwrap()
        .request_animation_frame(g.borrow().as_ref().unwrap().as_ref().unchecked_ref())?;

    Ok(())
}
