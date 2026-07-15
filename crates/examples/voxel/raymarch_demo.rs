//! Voxel Demo — procedurally generated voxel world, per-frame raymarched.
//!
//! Same world/controls as `voxel_demo` (mesh_demo.rs), but rendered through
//! `VoxelRayMarchPass` (`VoxelMode::Dynamic`) instead of `VoxelMeshPass`: a
//! fullscreen compute shader DDA-marches the brick grid every frame instead
//! of extracting real triangles. Useful for comparing the two rendering
//! paths, or for volumes under heavy per-frame editing where re-meshing on
//! every edit would be too expensive.
//!
//! Controls:
//!   Mouse click        – grab cursor / look around
//!   W/A/S/D             – fly
//!   Space/Shift         – up / down
//!   Left mouse button   – add voxel sphere
//!   Right mouse button  – subtract voxel sphere
//!   1-4                 – select material (grass/dirt/stone/ore)
//!   R                   – regenerate the world with a new random seed
//!   Escape              – release cursor / quit

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use glam::{EulerRot, Quat, Vec3};
use helio::{
    required_wgpu_features, required_wgpu_limits, Camera, GpuLight, LightType, Renderer,
    RendererConfig, Scene, SceneActor, RenderGraph, RenderPass, VoxelTerrain,
    VoxelVolumeDescriptor, VoxelMode, VoxelVolumeId, VOXEL_TERRAIN_GRID_DIM,
};
use helio_pass_fxaa::FxaaPass;
use helio_pass_voxel_raymarch::VoxelRayMarchPass;
use helio_voxel_core::GpuVoxelMaterial;
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

// ── constants ─────────────────────────────────────────────────────────────────

const LOOK_SENS: f32 = 0.002;
const FLY_SPEED: f32 = 10.0;
const DRAG: f32 = 6.0;
// The GPU-side voxel volume is always a dense 64^3 grid (fixed by the engine's
// BRICK_SIZE constant); `VOXEL_SIZE` just scales that grid into world units.
const VOXEL_SIZE: f32 = 0.75;
const ROOT_EXTENT: f32 = (VOXEL_TERRAIN_GRID_DIM as f32) * VOXEL_SIZE;

// ── app ───────────────────────────────────────────────────────────────────────

struct App {
    state: Option<AppState>,
}

struct AppState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    surface_format: wgpu::TextureFormat,
    alpha_mode: wgpu::CompositeAlphaMode,
    renderer: Renderer,
    last_frame: Instant,
    cam_pos: Vec3,
    yaw: f32,
    pitch: f32,
    velocity: Vec3,
    keys: HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta: (f32, f32),
    current_material: u8,
    vol_id: VoxelVolumeId,
    world: VoxelTerrain,
    world_seed: u32,
}

impl AppState {
    fn reset_transient_input(&mut self) {
        self.keys.clear();
        self.mouse_delta = (0.0, 0.0);
        self.velocity = Vec3::ZERO;
        self.cursor_grabbed = false;
        let _ = self.window.set_cursor_grab(CursorGrabMode::None);
        self.window.set_cursor_visible(true);
        self.last_frame = Instant::now();
    }

    fn resize(&mut self, width: u32, height: u32) {
        // Native resize loops can swallow input-release events. Release cursor
        // capture before rebuilding targets so the next click can reacquire it.
        self.reset_transient_input();
        if width == 0 || height == 0 {
            return;
        }

        self.surface.configure(
            &self.device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: self.surface_format,
                width,
                height,
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: self.alpha_mode,
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            },
        );
        self.renderer.set_render_size(width, height);
    }

    fn update(&mut self, dt: f32) {
        let (dx, dy) = self.mouse_delta;
        self.mouse_delta = (0.0, 0.0);
        self.yaw -= dx * LOOK_SENS;
        self.pitch = (self.pitch - dy * LOOK_SENS).clamp(-1.5, 1.5);

        let orientation = Quat::from_euler(EulerRot::YXZ, self.yaw, self.pitch, 0.0);
        let forward = orientation * -Vec3::Z;
        let right = orientation * Vec3::X;

        let mut accel = Vec3::ZERO;
        if self.keys.contains(&KeyCode::KeyW) { accel += forward; }
        if self.keys.contains(&KeyCode::KeyS) { accel -= forward; }
        if self.keys.contains(&KeyCode::KeyA) { accel -= right; }
        if self.keys.contains(&KeyCode::KeyD) { accel += right; }
        if self.keys.contains(&KeyCode::Space) { accel += Vec3::Y; }
        if self.keys.contains(&KeyCode::ShiftLeft) { accel -= Vec3::Y; }

        self.velocity += accel * FLY_SPEED * dt;
        self.velocity /= 1.0 + DRAG * dt;
        self.cam_pos += self.velocity * dt;
    }

    fn camera(&self, width: u32, height: u32) -> Camera {
        let orientation = Quat::from_euler(EulerRot::YXZ, self.yaw, self.pitch, 0.0);
        let target = self.cam_pos + orientation * -Vec3::Z;
        let up = orientation * Vec3::Y;
        Camera::perspective_look_at(
            self.cam_pos,
            target,
            up,
            std::f32::consts::FRAC_PI_4,
            width as f32 / height.max(1) as f32,
            0.01,
            2000.0,
        )
    }

    /// Converts a world-space position into the voxel volume's grid coordinates.
    /// The volume is centered on the world origin, with voxels scaled by `VOXEL_SIZE`.
    fn world_to_grid(pos: Vec3) -> [f32; 3] {
        let half = VOXEL_TERRAIN_GRID_DIM as f32 / 2.0;
        [
            pos.x / VOXEL_SIZE + half,
            pos.y / VOXEL_SIZE + half,
            pos.z / VOXEL_SIZE + half,
        ]
    }

    fn place_edit(
        add: bool,
        material: u8,
        cam_pos: Vec3,
        yaw: f32,
        pitch: f32,
        world: &mut VoxelTerrain,
        queue: &Arc<wgpu::Queue>,
        brick_pool: &wgpu::Buffer,
        data_pool: &wgpu::Buffer,
    ) {
        let orientation = Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0);
        let forward = orientation * -Vec3::Z;
        let center_world = cam_pos + forward * 5.0;
        let center_grid = Self::world_to_grid(center_world);
        let radius_grid = 2.0 / VOXEL_SIZE;

        if let Some(range) = world.paint_sphere(center_grid, radius_grid, material, add) {
            world.upload_range_raymarch(queue, brick_pool, data_pool, range);
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() { return; }

        let attrs = Window::default_attributes()
            .with_title("Helio Voxel Demo (Raymarch)")
            .with_inner_size(winit::dpi::PhysicalSize::new(1280, 720));
        let window = Arc::new(event_loop.create_window(attrs).unwrap());

        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
        }))
        .expect("No suitable GPU adapter");

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: required_wgpu_features(adapter.features()),
                required_limits: required_wgpu_limits(adapter.limits()),
                ..Default::default()
            },
        ))
        .expect("Device request failed");

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        device.on_uncaptured_error(Arc::new(|error| {
            log::error!("wgpu uncaptured error: {}", error);
        }));

        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        let surface_format = caps.formats.iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);
        let alpha_mode = caps.alpha_modes[0];

        surface.configure(
            &device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: surface_format,
                width: size.width,
                height: size.height,
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode,
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            },
        );

        // ── Renderer + custom graph with voxel pass ──────────────────────────
        // render_scale defaults to 0.75 (Renderer's shared depth texture is sized
        // from it), but our graph has no TAA upscale step and locks pre_aa at
        // full window resolution — a scaled depth buffer here would mismatch the
        // full-res color attachment, so pin it to 1.0.
        let config = RendererConfig::new(size.width, size.height, surface_format).with_render_scale(1.0);
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
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let debug_state = Arc::new(std::sync::Mutex::new(helio::DebugDrawState::default()));

        // Create a voxel volume with some initial structure
        let voxel_desc = VoxelVolumeDescriptor {
            voxel_size: VOXEL_SIZE,
            root_extent: ROOT_EXTENT,
            local_to_world: glam::Mat4::IDENTITY,
            movability: Some(libhelio::Movability::Stationary),
            // Dynamic (raymarch) mode: re-traced from scratch every frame, no
            // meshing cost on edit. Auto (VoxelMeshPass) is the mesh_demo.rs
            // sibling of this file.
            mode: Some(VoxelMode::Dynamic),
            material_palette: vec![
                GpuVoxelMaterial { color: [0.0, 0.0, 0.0], roughness: 1.0, metalness: 0.0, emissive: 0.0, _pad: [0; 2] }, // air (unused)
                GpuVoxelMaterial { color: [0.3, 0.7, 0.25], roughness: 0.8, metalness: 0.0, emissive: 0.0, _pad: [0; 2] }, // grass
                GpuVoxelMaterial { color: [0.45, 0.3, 0.15], roughness: 0.9, metalness: 0.0, emissive: 0.0, _pad: [0; 2] }, // dirt
                GpuVoxelMaterial { color: [0.5, 0.5, 0.52], roughness: 0.85, metalness: 0.0, emissive: 0.0, _pad: [0; 2] }, // stone
                GpuVoxelMaterial { color: [0.9, 0.75, 0.2], roughness: 0.4, metalness: 0.8, emissive: 0.0, _pad: [0; 2] }, // ore
            ],
        };
        let vol_id = scene.insert_voxel_volume(voxel_desc).expect("Failed to create voxel volume");

        // Real scene lighting — VoxelRayMarchPass sums the scene's lights buffer
        // directly (see voxel_raymarch.wgsl), the same infrastructure the default
        // render graphs feed their deferred lighting pass with.
        scene.insert_actor(SceneActor::light(GpuLight {
            position_range: [0.0, 0.0, 0.0, f32::MAX],
            direction_outer: [0.35, -0.8, 0.25, 0.0],
            color_intensity: [1.0, 0.95, 0.85, 3.0],
            shadow_index: u32::MAX,
            light_type: LightType::Directional as u32,
            inner_angle: 0.0,
            _pad: 0,
        }));
        scene.insert_actor(SceneActor::light(GpuLight {
            position_range: [0.0, 0.0, 0.0, f32::MAX],
            direction_outer: [-0.4, -0.2, -0.6, 0.0],
            color_intensity: [0.5, 0.6, 0.8, 0.6],
            shadow_index: u32::MAX,
            light_type: LightType::Directional as u32,
            inner_angle: 0.0,
            _pad: 0,
        }));

        // Procedurally generate the world on the CPU and bake it straight to the
        // shared GPU voxel pools. This bypasses Scene::edit_voxel_volume because
        // the edit ring does not currently have a GPU consumer.
        let world_seed = 1;
        let mut world = VoxelTerrain::empty();
        world.generate(world_seed);
        world.upload_all_raymarch(&queue, &scene.gpu_scene().voxel_brick_pool, &scene.gpu_scene().voxel_data_pool);

        // Build a custom graph: VoxelRayMarchPass (writes "pre_aa") then
        // FxaaPass (reads "pre_aa", writes directly to the swapchain target —
        // doing double duty as anti-aliasing and the terminal blit;
        // PostProcessPass would instead clear+rewrite the target straight from
        // "pre_aa" and discard FXAA's result if chained after it).
        let mut graph = RenderGraph::new(&device, &queue);
        let mut voxel_rm_pass = VoxelRayMarchPass::new(&device, surface_format);
        // VoxelRayMarchPass allocates its output textures at a placeholder 1x1 and
        // only resizes them in on_resize(), which the engine normally calls from a
        // window-resize event. Since RenderGraph::lock() never calls it, we have to
        // size the pass explicitly here or it ray marches into a 1x1 texture forever.
        voxel_rm_pass.on_resize(&device, size.width, size.height);
        graph.add_pass(Box::new(voxel_rm_pass));
        graph.add_pass(Box::new(FxaaPass::new(&device, surface_format)));
        graph.lock(size.width, size.height);

        let mut renderer = Renderer::new(
            device.clone(), queue.clone(),
            config.surface_format, config.width, config.height, config.render_scale,
            config, scene, graph, debug_state, debug_camera_buf, cull_stats_buf,
        );
        // Renderer applies TAA-style subpixel camera jitter every frame
        // unconditionally; without a TaaPass to resolve it (we only have
        // FXAA, which is spatial-only), that jitter just makes the image
        // visibly shimmer/flicker frame to frame — and for a per-pixel
        // raymarch it also changes which voxel each pixel samples, making it
        // noisier still.
        renderer.set_jitter_enabled(false);

        self.state = Some(AppState {
            window,
            surface,
            device,
            queue,
            surface_format,
            alpha_mode,
            renderer,
            last_frame: Instant::now(),
            // Start well above the terrain (world y can reach ~±10 given the
            // generator's amplitude) looking down, so the camera doesn't spawn
            // clipped inside solid ground.
            cam_pos: Vec3::new(0.0, 30.0, 45.0),
            yaw: 0.0,
            pitch: -0.5,
            velocity: Vec3::ZERO,
            keys: HashSet::new(),
            cursor_grabbed: false,
            mouse_delta: (0.0, 0.0),
            current_material: 1,
            vol_id,
            world,
            world_seed,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(new_size) => state.resize(new_size.width, new_size.height),

            WindowEvent::Focused(false) | WindowEvent::CursorLeft { .. } => {
                state.reset_transient_input();
            }

            WindowEvent::Focused(true) => state.last_frame = Instant::now(),

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
                    state.window.set_cursor_visible(true);
                    let _ = state.window.set_cursor_grab(CursorGrabMode::None);
                } else {
                    event_loop.exit();
                }
            }

            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    state: ElementState::Pressed,
                    physical_key: PhysicalKey::Code(code),
                    ..
                },
                ..
            } => {
                let _ = state.keys.insert(code);
                match code {
                    KeyCode::Digit1 => state.current_material = 1,
                    KeyCode::Digit2 => state.current_material = 2,
                    KeyCode::Digit3 => state.current_material = 3,
                    KeyCode::Digit4 => state.current_material = 4,
                    KeyCode::KeyR => {
                        state.world_seed = state.world_seed.wrapping_add(1).wrapping_mul(2654435761).wrapping_add(1);
                        state.world.generate(state.world_seed);
                        let scene = state.renderer.scene();
                        state.world.upload_all_raymarch(
                            &state.queue,
                            &scene.gpu_scene().voxel_brick_pool,
                            &scene.gpu_scene().voxel_data_pool,
                        );
                    }
                    _ => {}
                }
            }

            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    state: ElementState::Released,
                    physical_key: PhysicalKey::Code(code),
                    ..
                },
                ..
            } => {
                state.keys.remove(&code);
            }

            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } if !state.cursor_grabbed => {
                let ok = state.window
                    .set_cursor_grab(CursorGrabMode::Confined)
                    .or_else(|_| state.window.set_cursor_grab(CursorGrabMode::Locked))
                    .is_ok();
                if ok {
                    state.cursor_grabbed = true;
                    state.window.set_cursor_visible(false);
                }
            }

            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } if state.cursor_grabbed => {
                let mat = state.current_material;
                let pos = state.cam_pos;
                let yaw = state.yaw;
                let pitch = state.pitch;
                let scene = state.renderer.scene();
                let (brick_pool, data_pool) = (
                    scene.gpu_scene().voxel_brick_pool.clone(),
                    scene.gpu_scene().voxel_data_pool.clone(),
                );
                AppState::place_edit(true, mat, pos, yaw, pitch, &mut state.world, &state.queue, &brick_pool, &data_pool);
            }

            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Right,
                ..
            } if state.cursor_grabbed => {
                let mat = state.current_material;
                let pos = state.cam_pos;
                let yaw = state.yaw;
                let pitch = state.pitch;
                let scene = state.renderer.scene();
                let (brick_pool, data_pool) = (
                    scene.gpu_scene().voxel_brick_pool.clone(),
                    scene.gpu_scene().voxel_data_pool.clone(),
                );
                AppState::place_edit(false, mat, pos, yaw, pitch, &mut state.world, &state.queue, &brick_pool, &data_pool);
            }

            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = now.duration_since(state.last_frame).as_secs_f32().min(0.05);
                state.last_frame = now;
                state.update(dt);

                let size = state.window.inner_size();
                let camera = state.camera(size.width, size.height);

                let output = match state.surface.get_current_texture() {
                    Ok(t) => t,
                    Err(e) => {
                        log::warn!("surface error: {:?}", e);
                        return;
                    }
                };
                let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
                if let Err(e) = state.renderer.render(&camera, &view) {
                    log::error!("render error: {:?}", e);
                }
                output.present();
                state.window.request_redraw();
            }

            _ => {}
        }
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: DeviceId, event: DeviceEvent) {
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

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut app = App { state: None };
    event_loop.run_app(&mut app).unwrap();
}
