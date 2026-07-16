// Planar Reflection Demo
// A glossy floor plane with coloured objects above it, demonstrating the
// planar reflection capture system.
//
// Controls:
//   WASD        — move
//   Space/Shift — up/down
//   Mouse       — look (click to grab)
//   Escape      — release cursor / quit

mod v3_demo_common;

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use glam::{EulerRot, Mat4, Quat, Vec3};
use helio::{
    required_wgpu_features, required_wgpu_limits, Camera, DebugDrawState, ObjectDescriptor,
    Renderer, RendererConfig, Scene, SceneActor, SkyActor, VolumetricClouds,
};
use helio_default_graphs::build_default_graph;
use v3_demo_common::{
    box_mesh, cube_mesh, directional_light, make_material, plane_mesh, sphere_mesh,
};
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

const LOOK_SENS: f32 = 0.002;
const FLY_SPEED: f32 = 8.0;
const DRAG: f32 = 6.0;

struct App {
    state: Option<AppState>,
}

struct AppState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    adapter: wgpu::Adapter,
    surface_format: wgpu::TextureFormat,
    renderer: Renderer,
    last_frame: Instant,
    cam_pos: Vec3,
    yaw: f32,
    pitch: f32,
    velocity: Vec3,
    keys: HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta: (f32, f32),
    spinning_cube: helio::ObjectId,
    orbiting_sphere: helio::ObjectId,
    sphere_angle: f32,
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
                        .with_title("Helio — Planar Reflection Demo")
                        .with_inner_size(winit::dpi::PhysicalSize::new(1600, 900)),
                )
                .expect("create window"),
        );

        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window.clone())
            .expect("create surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            apply_limit_buckets: true,
        }))
        .expect("no adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: required_wgpu_features(adapter.features()),
            required_limits: required_wgpu_limits(adapter.limits()),
            ..Default::default()
        }))
        .expect("no device");
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
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: caps.alpha_modes[0],
                color_space: wgpu::SurfaceColorSpace::Auto,
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            },
        );

        let mut renderer_config = RendererConfig::new(size.width, size.height, surface_format);
        // HELIO_DEBUG_MODE=30 -> SSR confidence, 31 -> SSR colour. See the debug
        // blocks in deferred_lighting.wgsl for the full list.
        if let Ok(mode) = std::env::var("HELIO_DEBUG_MODE") {
            if let Ok(mode) = mode.parse::<u32>() {
                renderer_config.debug_mode = mode;
            }
        }

        let mut scene = Scene::new(device.clone(), queue.clone());

        // ── Materials ──────────────────────────────────────────────────────
        let floor_mat = scene.insert_material(make_material(
            [0.97, 0.97, 0.97, 1.0],  // silver-tinted mirror
            0.001,                      // near-zero roughness = mirror-sharp
            1.0,                        // metallic mirror
            [0.0; 3],
            0.0,
        ));
        let red_mat =
            scene.insert_material(make_material([1.0, 0.2, 0.2, 1.0], 0.3, 0.0, [0.0; 3], 0.0));
        let blue_mat =
            scene.insert_material(make_material([0.2, 0.4, 1.0, 1.0], 0.2, 0.1, [0.0; 3], 0.0));
        let green_mat =
            scene.insert_material(make_material([0.2, 0.9, 0.3, 1.0], 0.6, 0.0, [0.0; 3], 0.0));
        let gold_mat = scene.insert_material(make_material(
            [1.0, 0.85, 0.4, 1.0],
            0.15,
            1.0,
            [0.0; 3],
            0.0,
        ));
        let white_mat = scene.insert_material(make_material(
            [0.95, 0.95, 0.98, 1.0],
            0.4,
            0.0,
            [0.0; 3],
            0.0,
        ));

        // ── Meshes ─────────────────────────────────────────────────────────
        let cube_mesh = scene
            .insert_actor(SceneActor::mesh(cube_mesh([0.0; 3], 0.5)))
            .as_mesh()
            .unwrap();
        let sphere_mesh = scene
            .insert_actor(SceneActor::mesh(sphere_mesh([0.0; 3], 0.5)))
            .as_mesh()
            .unwrap();
        let floor_mesh = scene
            .insert_actor(SceneActor::mesh(plane_mesh([0.0; 3], 6.0)))
            .as_mesh()
            .unwrap();
        let pillar_mesh = scene
            .insert_actor(SceneActor::mesh(box_mesh([0.0; 3], [0.15, 1.5, 0.15])))
            .as_mesh()
            .unwrap();

        // ── Floor (mirror) ─────────────────────────────────────────────────
        scene.insert_actor(SceneActor::object(ObjectDescriptor {
            mesh: floor_mesh,
            material: floor_mat,
            transform: Mat4::from_translation(Vec3::new(0.0, -1.5, 0.0)),
            bounds: [0.0, -1.5, 0.0, 6.0],
            flags: 0,
            groups: helio::GroupMask::NONE,
            movability: None,
            user_tag: 0,
        }));

        // ── Pillars ────────────────────────────────────────────────────────
        for x in [-2.0_f32, 2.0] {
            for z in [-2.0_f32, 2.0] {
                scene.insert_actor(SceneActor::object(ObjectDescriptor {
                    mesh: pillar_mesh,
                    material: white_mat,
                    transform: Mat4::from_translation(Vec3::new(x, -0.75, z)),
                    bounds: [x, -0.75, z, 0.5],
                    flags: 0,
                    groups: helio::GroupMask::NONE,
                    movability: None,
                    user_tag: 0,
                }));
            }
        }

        // ── Central objects ────────────────────────────────────────────────
        let cube_id = scene
            .insert_actor(SceneActor::object(ObjectDescriptor {
                mesh: cube_mesh,
                material: red_mat,
                transform: Mat4::from_translation(Vec3::new(-1.2, 0.5, 0.0)),
                bounds: [-1.2, 0.5, 0.0, 0.5],
                flags: 0,
                groups: helio::GroupMask::NONE,
                movability: Some(helio::Movability::Movable),
                user_tag: 0,
            }))
            .as_object()
            .unwrap();

        scene.insert_actor(SceneActor::object(ObjectDescriptor {
            mesh: sphere_mesh,
            material: blue_mat,
            transform: Mat4::from_translation(Vec3::new(1.2, 0.5, -0.8)),
            bounds: [1.2, 0.5, -0.8, 0.5],
            flags: 0,
            groups: helio::GroupMask::NONE,
            movability: Some(helio::Movability::Movable),
            user_tag: 0,
        }));

        scene.insert_actor(SceneActor::object(ObjectDescriptor {
            mesh: sphere_mesh,
            material: gold_mat,
            transform: Mat4::from_translation(Vec3::new(0.0, 0.5, 1.2)),
            bounds: [0.0, 0.5, 1.2, 0.5],
            flags: 0,
            groups: helio::GroupMask::NONE,
            movability: Some(helio::Movability::Movable),
            user_tag: 0,
        }));

        let sphere_id = scene
            .insert_actor(SceneActor::object(ObjectDescriptor {
                mesh: sphere_mesh,
                material: green_mat,
                transform: Mat4::from_translation(Vec3::new(-1.5, 1.8, 1.5)),
                bounds: [-1.5, 1.8, 1.5, 0.5],
                flags: 0,
                groups: helio::GroupMask::NONE,
                movability: Some(helio::Movability::Movable),
                user_tag: 0,
            }))
            .as_object()
            .unwrap();

        // ── Reflection capture ─────────────────────────────────────────────
        // A box capture spanning the room, centred on the box's own centre.
        // Its cubemap layer is assigned by the probe bake, so this contributes
        // nothing until the scene has been baked.
        scene
            .insert_reflection_capture(
                helio::ReflectionCaptureDescriptor::boxed(
                    Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0)),
                    [6.0, 3.0, 6.0],
                )
                .with_transition_distance(1.0),
            )
            .unwrap();

        // ── Sky ────────────────────────────────────────────────────────────
        scene.insert_actor(SceneActor::sky(
            SkyActor::new()
                .with_sky_color([0.6, 0.7, 1.0])
                .with_ambient_color([0.15, 0.18, 0.25])
                .with_clouds(VolumetricClouds {
                    coverage: 0.3,
                    density: 0.4,
                    base: 500.0,
                    top: 800.0,
                    wind_x: 0.3,
                    wind_z: 0.1,
                    speed: 2.0,
                    skylight_intensity: 0.5,
                }),
        ));

        // ── Lights ─────────────────────────────────────────────────────────
        scene.insert_actor(SceneActor::light(directional_light(
            [-0.5, -1.0, -0.3],
            [1.0, 0.95, 0.9],
            12.0,
        )));

        scene.insert_actor(SceneActor::light(helio::GpuLight {
            position_range: [3.0, 4.0, 2.0, 10.0],
            direction_outer: [0.0; 4],
            color_intensity: [1.0, 0.85, 0.6, 30.0],
            shadow_index: 0,
            light_type: helio::LightType::Point as u32,
            inner_angle: 0.0,
            _pad: 0,
        }));

        scene.insert_actor(SceneActor::light(helio::GpuLight {
            position_range: [-3.0, 3.0, -2.0, 10.0],
            direction_outer: [0.0; 4],
            color_intensity: [0.4, 0.6, 1.0, 20.0],
            shadow_index: 0,
            light_type: helio::LightType::Point as u32,
            inner_angle: 0.0,
            _pad: 0,
        }));

        // ── Build renderer ─────────────────────────────────────────────────
        let debug_state = Arc::new(std::sync::Mutex::new(DebugDrawState::default()));
        let debug_camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Debug Camera Buffer"),
            size: std::mem::size_of::<helio::DebugCameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cull_stats_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cull Stats"),
            size: 32,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let graph = build_default_graph(
            &device,
            &queue,
            &scene,
            renderer_config,
            debug_state.clone(),
            &debug_camera_buf,
            &cull_stats_buf,
            None,
        );

        let renderer = Renderer::new(
            device.clone(),
            queue.clone(),
            surface_format,
            size.width,
            size.height,
            renderer_config.render_scale,
            renderer_config,
            scene,
            graph,
            debug_state,
            debug_camera_buf,
            cull_stats_buf,
        );

        self.state = Some(AppState {
            window,
            surface,
            device,
            queue,
            adapter,
            surface_format,
            renderer,
            last_frame: Instant::now(),
            cam_pos: Vec3::new(0.0, 2.0, 6.0),
            yaw: 0.0,
            pitch: -0.2,
            velocity: Vec3::ZERO,
            keys: HashSet::new(),
            cursor_grabbed: false,
            mouse_delta: (0.0, 0.0),
            spinning_cube: cube_id,
            orbiting_sphere: sphere_id,
            sphere_angle: 0.0,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let state = self.state.as_mut().unwrap();

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(s) if s.width > 0 && s.height > 0 => {
                let w = s.width;
                let h = s.height;
                let caps = state.surface.get_capabilities(&state.adapter);
                state.surface.configure(
                    &state.device,
                    &wgpu::SurfaceConfiguration {
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                        format: state.surface_format,
                        width: w,
                        height: h,
                        present_mode: wgpu::PresentMode::Fifo,
                        alpha_mode: caps.alpha_modes[0],
                        color_space: wgpu::SurfaceColorSpace::Auto,
                        view_formats: vec![],
                        desired_maximum_frame_latency: 2,
                    },
                );
                state.renderer.set_render_size(w, h);
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key),
                        state: key_state,
                        ..
                    },
                ..
            } => match (key, key_state) {
                (KeyCode::Escape, ElementState::Pressed) => {
                    if state.cursor_grabbed {
                        state.cursor_grabbed = false;
                        let _ = state.window.set_cursor_grab(CursorGrabMode::None);
                        state.window.set_cursor_visible(true);
                    } else {
                        event_loop.exit();
                    }
                }
                _ => {
                    if key_state == ElementState::Pressed {
                        state.keys.insert(key);
                    } else {
                        state.keys.remove(&key);
                    }
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
            WindowEvent::RedrawRequested => {
                let dt = state.last_frame.elapsed().as_secs_f32().min(0.05);
                state.last_frame = Instant::now();

                // ---- Input ----
                let (dx, dy) = state.mouse_delta;
                state.mouse_delta = (0.0, 0.0);
                state.yaw -= dx * LOOK_SENS;
                state.pitch = (state.pitch - dy * LOOK_SENS).clamp(-1.5, 1.5);

                let orientation = Quat::from_euler(EulerRot::YXZ, state.yaw, state.pitch, 0.0);
                let forward = orientation * -Vec3::Z;
                let right = orientation * Vec3::X;

                let mut accel = Vec3::ZERO;
                if state.keys.contains(&KeyCode::KeyW) {
                    accel += forward;
                }
                if state.keys.contains(&KeyCode::KeyS) {
                    accel -= forward;
                }
                if state.keys.contains(&KeyCode::KeyA) {
                    accel -= right;
                }
                if state.keys.contains(&KeyCode::KeyD) {
                    accel += right;
                }
                if state.keys.contains(&KeyCode::Space) {
                    accel += Vec3::Y;
                }
                if state.keys.contains(&KeyCode::ShiftLeft) {
                    accel -= Vec3::Y;
                }
                if accel.length_squared() > 0.0 {
                    accel = accel.normalize();
                }
                state.velocity += accel * FLY_SPEED * dt;
                state.velocity /= 1.0 + DRAG * dt;
                state.cam_pos += state.velocity * dt;

                // ---- Animate ----
                let angle = state.renderer.scene_mut().gpu_scene().frame_count as f32 * 0.02;
                let cube_transform = Mat4::from_axis_angle(Vec3::Y, angle)
                    * Mat4::from_axis_angle(Vec3::X, angle * 0.5)
                    * Mat4::from_translation(Vec3::new(-1.2, 0.5, 0.0));
                let _ = state
                    .renderer
                    .scene_mut()
                    .update_object_transform(state.spinning_cube, cube_transform);

                state.sphere_angle += dt * 0.6;
                let orbit_x = 2.5 * state.sphere_angle.cos();
                let orbit_z = 2.5 * state.sphere_angle.sin();
                let sphere_pos = Vec3::new(orbit_x, 1.0 + 0.5 * state.sphere_angle.sin(), orbit_z);
                let _ = state.renderer.scene_mut().update_object_transform(
                    state.orbiting_sphere,
                    Mat4::from_translation(sphere_pos),
                );

                // ---- Camera ----
                let target = state.cam_pos + forward;
                let up = orientation * Vec3::Y;
                let size = state.window.inner_size();
                let camera = Camera::perspective_look_at(
                    state.cam_pos,
                    target,
                    up,
                    std::f32::consts::FRAC_PI_4,
                    size.width as f32 / size.height.max(1) as f32,
                    0.1,
                    500.0,
                );

                // ---- Render ----
                let output = match state.surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(t) => t,
                    wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
                    _ => {
                        log::warn!("surface acquire failed");
                        return;
                    }
                };
                let view = output.texture.create_view(&Default::default());

                if let Err(e) = state.renderer.render(&camera, &view) {
                    log::error!("Render: {:?}", e);
                }
                state.renderer.queue().present(output);
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

fn main() {
    env_logger::init();
    EventLoop::new().unwrap().run_app(&mut App::new()).unwrap();
}
