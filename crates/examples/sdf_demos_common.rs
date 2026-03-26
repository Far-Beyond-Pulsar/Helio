//! Common support code for a family of simple SDF demos.
//!
//! The demos are all almost identical to `sdf_demo.rs` except for the
//! way the `SdfFeature` is initialized and updated each frame.  This module
//! exposes a generic `App` driven by a user-provided `SdfUpdater` object
//! which handles those per-demo details.
//!
//! Individual binaries in `crates/examples` can then be a handful of lines:
//!
//! ```rust
//! struct MyUpdater;
//! impl SdfUpdater for MyUpdater { ... }
//! fn main() { run_demo("my demo", MyUpdater::default()); }
//! ```

use helio_render_v2::{Renderer, RendererConfig, Camera, GpuMesh, SceneLight, LightId, BillboardId};
use helio_render_v2::features::{
    FeatureRegistry,
    LightingFeature,
    BloomFeature, ShadowsFeature,
    BillboardsFeature, BillboardInstance,
    RadianceCascadesFeature,
    SdfFeature, SdfMode,
};
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId, CursorGrabMode},
};
use std::collections::HashSet;
use std::sync::Arc;

const RC_WORLD_MIN: [f32; 3] = [-3.5, -0.3, -3.5];
const RC_WORLD_MAX: [f32; 3] = [3.5, 5.0, 3.5];

pub trait SdfUpdater {
    /// Called once during startup so the demo can add whatever edits it wants.
    fn init(&mut self, sdf: &mut SdfFeature);

    /// Called every frame with the current frame time (approx. seconds).
    /// The updater may modify existing edits (via `set_edit`) or adjust
    /// terrain parameters, boolean ops, etc.
    fn update(&mut self, sdf: &mut SdfFeature, time: f32);
}

/// Run a demo given a title string and an updater value.
pub fn run_demo<U: SdfUpdater + 'static>(title: &str, updater: U) {
    env_logger::init();
    log::info!("Starting Helio Render V2 {}", title);

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut app = DemoApp::new(title.to_string(), updater);
    event_loop.run_app(&mut app).expect("Event loop error");
}

struct DemoApp<U: SdfUpdater> {
    state: Option<DemoState<U>>,
    updater: Option<U>, // stored until resumed() when moved into state
    title: String,
}

struct DemoState<U: SdfUpdater> {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>,
    surface_format: wgpu::TextureFormat,
    renderer: Renderer,
    last_frame: std::time::Instant,
    cube1: GpuMesh,
    cube2: GpuMesh,
    cube3: GpuMesh,
    ground: GpuMesh,

    cam_pos:   glam::Vec3,
    cam_yaw:   f32,
    cam_pitch: f32,
    keys:      HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta: (f32, f32),

    light_p0_id: LightId,
    light_p1_id: LightId,
    light_p2_id: LightId,
    billboard_ids: Vec<BillboardId>,

    probe_vis: bool,
    sprite_w: u32,
    sprite_h: u32,

    updater: U,
}

impl<U: SdfUpdater> DemoApp<U> {
    fn new(title: String, updater: U) -> Self {
        Self { state: None, updater: Some(updater), title }
    }
}

impl<U: SdfUpdater + 'static> ApplicationHandler for DemoApp<U> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title(self.title.clone())
                        .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32)),
                )
                .expect("Failed to create window"),
        );

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            flags: wgpu::InstanceFlags::empty(),
            ..Default::default()
        });
        let surface = instance
            .create_surface(window.clone())
            .expect("Failed to create surface");

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("Failed to find adapter");

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Main Device"),
                required_features: wgpu::Features::EXPERIMENTAL_RAY_QUERY | wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
                required_limits: wgpu::Limits::default()
                    .using_minimum_supported_acceleration_structure_values(),
                memory_hints: wgpu::MemoryHints::default(),
                experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
                trace: wgpu::Trace::Off,
            },
        ))
        .expect("Failed to create device (ray tracing required)");

        device.on_uncaptured_error(std::sync::Arc::new(|e| {
            panic!("[GPU UNCAPTURED ERROR] {:?}", e);
        }));
        let info = adapter.get_info();
        println!("[WGPU] Backend: {:?}, Device: {}, Driver: {}", info.backend, info.name, info.driver);
        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let (sprite_rgba, sprite_w, sprite_h) = load_sprite();

        // build registry later with SDF included
        // create feature and let updater initialize it
        let mut sdf_feature = SdfFeature::new()
            .with_mode(SdfMode::ClipMap)
            .with_grid_dim(128)
            .with_volume_bounds([-3.0, -1.0, -3.0], [3.0, 3.0, 3.0]);

        if let Some(mut u) = self.updater.take() {
            u.init(&mut sdf_feature);
            // move updater into state
            let updater = u;

            let feature_registry = FeatureRegistry::builder()
                .with_feature(LightingFeature::new())
                .with_feature(BloomFeature::new().with_intensity(0.4).with_threshold(1.2))
                .with_feature(ShadowsFeature::new().with_atlas_size(1024).with_max_lights(4))
                .with_feature(BillboardsFeature::new().with_sprite(sprite_rgba.clone(), sprite_w, sprite_h).with_max_instances(5000))
                .with_feature(
                    RadianceCascadesFeature::new()
                        .with_world_bounds(RC_WORLD_MIN, RC_WORLD_MAX),
                )
                .with_feature(sdf_feature)
                .build();

            let mut renderer = Renderer::new(
                device.clone(),
                queue.clone(),
                RendererConfig::new(size.width, size.height, surface_format, feature_registry),
            )
            .expect("Failed to create renderer");
            // show light billboards in all of the demo apps
            renderer.set_editor_mode(true);

            let cube1  = renderer.create_mesh_cube([ 0.0, 0.5,  0.0], 0.5);
            let cube2  = renderer.create_mesh_cube([-2.0, 0.4, -1.0], 0.4);
            let cube3  = renderer.create_mesh_cube([ 2.0, 0.3,  0.5], 0.3);
            let ground = renderer.create_mesh_plane([0.0, 0.0, 0.0], 5.0);
            crate::demo_portal::enable_live_dashboard(&mut renderer);

            renderer.add_object(&cube1,  None, glam::Mat4::IDENTITY);
            renderer.add_object(&cube2,  None, glam::Mat4::IDENTITY);
            renderer.add_object(&cube3,  None, glam::Mat4::IDENTITY);
            renderer.add_object(&ground, None, glam::Mat4::IDENTITY);

            let p0_init = [0.0f32, 2.2, 0.0];
            let p1 = [-3.5f32, 2.0, -1.5];
            let p2 = [3.5f32, 1.5, 1.5];
            let light_p0_id = renderer.add_light(SceneLight::point(p0_init, [1.0, 0.55, 0.15], 6.0, 5.0));
            let light_p1_id = renderer.add_light(SceneLight::point(p1, [0.25, 0.5, 1.0], 5.0, 6.0));
            let light_p2_id = renderer.add_light(SceneLight::point(p2, [1.0, 0.3, 0.5], 5.0, 6.0));

            let mut billboard_ids = Vec::new();
            billboard_ids.push(renderer.add_billboard(BillboardInstance::new(p0_init, [0.35, 0.35]).with_color([1.0, 0.55, 0.15, 1.0])));
            billboard_ids.push(renderer.add_billboard(BillboardInstance::new(p1, [0.35, 0.35]).with_color([0.25, 0.5, 1.0, 1.0])));
            billboard_ids.push(renderer.add_billboard(BillboardInstance::new(p2, [0.35, 0.35]).with_color([1.0, 0.3, 0.5, 1.0])));

            self.state = Some(DemoState {
                window,
                surface,
                device,
                surface_format,
                renderer,
                last_frame: std::time::Instant::now(),
                cube1, cube2, cube3, ground,
                cam_pos:   glam::Vec3::new(0.0, 2.5, 7.0),
                cam_yaw:   0.0,
                cam_pitch: -0.2,
                keys:      HashSet::new(),
                cursor_grabbed: false,
                mouse_delta: (0.0, 0.0),
                light_p0_id,
                light_p1_id,
                light_p2_id,
                billboard_ids,
                probe_vis: false,
                sprite_w,
                sprite_h,
                updater,
            });
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };

        match event {
            // ── Exit ──────────────────────────────────────────────────────────
            WindowEvent::CloseRequested => {
                log::info!("Shutting down");
                event_loop.exit();
            }
            WindowEvent::KeyboardInput {
                event: KeyEvent {
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
                event: KeyEvent {
                    state: ElementState::Pressed,
                    physical_key: PhysicalKey::Code(KeyCode::Digit3),
                    ..
                },
                ..
            } => {
                state.probe_vis = !state.probe_vis;
                let raw: &[u8] = if state.probe_vis {
                    include_bytes!("../../probe.png")
                } else {
                    include_bytes!("../../spotlight.png")
                };
                let img = image::load_from_memory(raw)
                    .unwrap_or_else(|_| image::DynamicImage::new_rgba8(state.sprite_w, state.sprite_h))
                    .resize_exact(state.sprite_w, state.sprite_h, image::imageops::FilterType::Triangle)
                    .into_rgba8();
                if let Some(bb) = state.renderer.get_feature_mut::<BillboardsFeature>("billboards") {
                    bb.set_sprite(img.into_raw(), state.sprite_w, state.sprite_h);
                }
                for id in state.billboard_ids.drain(..) { state.renderer.remove_billboard(id); }
                if state.probe_vis {
                    for b in probe_billboards(RC_WORLD_MIN, RC_WORLD_MAX) {
                        state.billboard_ids.push(state.renderer.add_billboard(b));
                    }
                } else {
                    let p0 = [0.0f32, 2.2, 0.0];
                    let p1 = [-3.5f32, 2.0, -1.5];
                    let p2 = [3.5f32, 1.5, 1.5];
                    state.billboard_ids.push(state.renderer.add_billboard(BillboardInstance::new(p0, [0.35, 0.35]).with_color([1.0, 0.55, 0.15, 1.0])));
                    state.billboard_ids.push(state.renderer.add_billboard(BillboardInstance::new(p1, [0.35, 0.35]).with_color([0.25, 0.5, 1.0, 1.0])));
                    state.billboard_ids.push(state.renderer.add_billboard(BillboardInstance::new(p2, [0.35, 0.35]).with_color([1.0, 0.3, 0.5, 1.0])));
                }
            }
            WindowEvent::KeyboardInput {
                event: KeyEvent { state: ks, physical_key: PhysicalKey::Code(key), .. },
                ..
            } => {
                    // global F3 debug overlay toggle
                    if ks == ElementState::Pressed && key == KeyCode::F3 {
                        state.renderer.debug_viz_mut().enabled ^= true;
                    }
            }

            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                if !state.cursor_grabbed {
                    let grabbed = state.window.set_cursor_grab(CursorGrabMode::Confined)
                        .or_else(|_| state.window.set_cursor_grab(CursorGrabMode::Locked))
                        .is_ok();
                    if grabbed {
                        state.window.set_cursor_visible(false);
                        state.cursor_grabbed = true;
                    }
                }
            }

            WindowEvent::Resized(size) if size.width > 0 && size.height > 0 => {
                let config = wgpu::SurfaceConfiguration {
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    format: state.surface_format,
                    width: size.width,
                    height: size.height,
                    present_mode: wgpu::PresentMode::Fifo,
                    alpha_mode: wgpu::CompositeAlphaMode::Auto,
                    view_formats: vec![],
                    desired_maximum_frame_latency: 2,
                };
                state.surface.configure(&state.device, &config);
                state.renderer.resize(size.width, size.height);
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

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _id: winit::event::DeviceId, event: DeviceEvent) {
        let Some(state) = &mut self.state else { return };
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            if state.cursor_grabbed {
                state.mouse_delta.0 += dx as f32;
                state.mouse_delta.1 += dy as f32;
            }
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }
}

impl<U: SdfUpdater> DemoState<U> {
    fn render(&mut self, dt: f32) {
        const SPEED: f32 = 5.0;
        const LOOK_SENS: f32 = 0.002;

        self.cam_yaw   += self.mouse_delta.0 * LOOK_SENS;
        self.cam_pitch  = (self.cam_pitch - self.mouse_delta.1 * LOOK_SENS).clamp(-1.5, 1.5);
        self.mouse_delta = (0.0, 0.0);

        let (sy, cy) = self.cam_yaw.sin_cos();
        let (sp, cp) = self.cam_pitch.sin_cos();
        let forward = glam::Vec3::new(sy * cp, sp, -cy * cp);
        let right   = glam::Vec3::new(cy, 0.0, sy);
        let up      = glam::Vec3::Y;

        if self.keys.contains(&KeyCode::KeyW)      { self.cam_pos += forward * SPEED * dt; }
        if self.keys.contains(&KeyCode::KeyS)      { self.cam_pos -= forward * SPEED * dt; }
        if self.keys.contains(&KeyCode::KeyA)      { self.cam_pos -= right   * SPEED * dt; }
        if self.keys.contains(&KeyCode::KeyD)      { self.cam_pos += right   * SPEED * dt; }
        if self.keys.contains(&KeyCode::Space)     { self.cam_pos += up * SPEED * dt; }
        if self.keys.contains(&KeyCode::ShiftLeft) { self.cam_pos -= up * SPEED * dt; }

        let size = self.window.inner_size();
        let aspect = size.width as f32 / size.height.max(1) as f32;
        let time = self.renderer.frame_count() as f32 * 0.016;

        // allow updater to adjust edits
        if let Some(sdf) = self.renderer.get_feature_mut::<SdfFeature>("sdf") {
            self.updater.update(sdf, time);
        }

        let camera = Camera::perspective(
            self.cam_pos,
            self.cam_pos + forward,
            glam::Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            aspect,
            0.1,
            200.0,
            time,
        );

        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(e) => { log::warn!("Surface error: {:?}", e); return; }
        };
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let p0 = [0.0f32, 2.2 + (time * 0.7).sin() * 0.3, 0.0];
        self.renderer.scene_mut().update_light(self.light_p0_id, SceneLight::point(p0, [1.0, 0.55, 0.15], 6.0, 5.0));
        if !self.probe_vis && !self.billboard_ids.is_empty() {
            self.renderer.update_billboard(self.billboard_ids[0], BillboardInstance::new(p0, [0.35, 0.35]).with_color([1.0, 0.55, 0.15, 1.0]));
        }
        if let Err(e) = self.renderer.render(&camera, &view, dt) {
            log::error!("Render error: {:?}", e);
        }

        output.present();
    }
}

// convenience helpers copied from sdf_move earlier

fn load_sprite() -> (Vec<u8>, u32, u32) {
    let img = image::load_from_memory(include_bytes!("../../spotlight.png"))
        .unwrap_or_else(|_| image::DynamicImage::new_rgba8(1, 1))
        .into_rgba8();
    let (w, h) = img.dimensions();
    (img.into_raw(), w, h)
}

#[allow(dead_code)]
fn load_probe_sprite() -> (Vec<u8>, u32, u32) {
    let img = image::load_from_memory(include_bytes!("../../probe.png"))
        .unwrap_or_else(|_| image::DynamicImage::new_rgba8(1, 1))
        .into_rgba8();
    let (w, h) = img.dimensions();
    (img.into_raw(), w, h)
}

fn probe_billboards(world_min: [f32; 3], world_max: [f32; 3]) -> Vec<helio_render_v2::features::BillboardInstance> {
    use helio_render_v2::features::radiance_cascades::PROBE_DIMS;
    const COLORS: [[f32; 4]; 4] = [
        [0.0, 1.0, 1.0, 0.85],
        [0.0, 1.0, 0.0, 0.80],
        [1.0, 1.0, 0.0, 0.75],
        [1.0, 0.35, 0.0, 0.70],
    ];
    const SIZES: [[f32; 2]; 4] = [
        [0.035, 0.035],
        [0.075, 0.075],
        [0.140, 0.140],
        [0.260, 0.260],
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




