//! Shape Battle Royale — helio v3
//!
//! 4+ shapes (adjustable) are launched from arena edges into the center and collide with
//! high restitution. The last moving shape still inside the arena wins.
//! Eliminated shapes explode into temporary blast particles.
//!
//! Controls:
//!   WASD / Space / Shift — fly
//!   +/-                  — adjust shape count and restart round (auto-reset 2s after end)
//!   Escape               — release cursor / exit

mod v3_demo_common;
use v3_demo_common::{box_mesh, cube_mesh, insert_object, make_material, plane_mesh};

use helio::{required_wgpu_features, required_wgpu_limits, Camera, ObjectId, Renderer, RendererConfig};
use rapier3d::prelude::*;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::{application::ApplicationHandler, event::*, event_loop::{ActiveEventLoop, EventLoop}, keyboard::{KeyCode, PhysicalKey}, window::{CursorGrabMode, Window, WindowId}};

const ARENA_RADIUS: f32 = 17.5;
const WALL_HEIGHT: f32 = 6.0;
const WALL_THICKNESS: f32 = 1.0;
const MIN_SHAPES: usize = 4;
const MAX_SHAPES: usize = 16;
const ROUND_RESET_DELAY: Duration = Duration::from_secs(2);

struct BattleShape {
    body_handle: RigidBodyHandle,
    collider_handle: ColliderHandle,
    object_id: ObjectId,
    eliminated: bool,
}

struct BlastParticle {
    object_id: ObjectId,
    birth: Instant,
}

struct App {
    state: Option<AppState>,
}

struct AppState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>,
    surface_format: wgpu::TextureFormat,
    renderer: Renderer,
    last_frame: Instant,
    frame_count: u64,

    cam_pos: glam::Vec3,
    cam_yaw: f32,
    cam_pitch: f32,
    keys: HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta: (f32, f32),

    physics_integration: IntegrationParameters,
    physics_bodies: RigidBodySet,
    physics_colliders: ColliderSet,
    physics_forces: IslandManager,
    physics_broad_phase: BroadPhase,
    physics_narrow_phase: NarrowPhase,
    physics_joint_set: JointSet,
    physics_ccd_solver: CCDSolver,

    battle_shapes: Vec<BattleShape>,
    explosion_particles: Vec<BlastParticle>,

    shape_count: usize,
    round_active: bool,
    round_end_instant: Option<Instant>,

    mats: [helio::MaterialId; 4],
    meshes: [helio::MeshId; 4],

    time_render_end: Option<Instant>,
    time_about_to_wait_start: Option<Instant>,
    time_redraw_requested: Option<Instant>,
}

fn main() {
    env_logger::init();
    log::info!("Starting Shape Battle Royale");
    EventLoop::new().expect("event loop").run_app(&mut App { state: None }).expect("run");
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() { return; }

        let window = Arc::new(event_loop.create_window(Window::default_attributes().with_title("Helio — Shape Battle Royale").with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32))).expect("window"));

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { backends: wgpu::Backends::all(), flags: wgpu::InstanceFlags::empty(), ..Default::default() });
        let surface = instance.create_surface(window.clone()).expect("surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: Some(&surface), force_fallback_adapter: false })).expect("adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { required_features: required_wgpu_features(adapter.features()), required_limits: required_wgpu_limits(adapter.limits()), ..Default::default() })).expect("device");
        device.on_uncaptured_error(Arc::new(|e: wgpu::Error| { panic!("[GPU] {:?}", e) }));
        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let caps = surface.get_capabilities(&adapter);
        let fmt = caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(caps.formats[0]);
        let size = window.inner_size();
        surface.configure(&device, &wgpu::SurfaceConfiguration { usage: wgpu::TextureUsages::RENDER_ATTACHMENT, format: fmt, width: size.width, height: size.height, present_mode: wgpu::PresentMode::Fifo, alpha_mode: caps.alpha_modes[0], view_formats: vec![], desired_maximum_frame_latency: 1 });

        let mut renderer = Renderer::new(device.clone(), queue.clone(), RendererConfig::new(size.width, size.height, fmt));
        renderer.set_ambient([0.05, 0.05, 0.07], 1.0);

        let flooring = renderer.scene_mut().insert_material(make_material([0.15, 0.15, 0.18, 1.0], 0.86, 0.05, [0.0, 0.0, 0.0], 0.0));
        let red = renderer.scene_mut().insert_material(make_material([0.84, 0.14, 0.14, 1.0], 0.45, 0.0, [0.0,0.0,0.0], 0.0));
        let green = renderer.scene_mut().insert_material(make_material([0.18, 0.85, 0.25, 1.0], 0.45, 0.0, [0.0,0.0,0.0], 0.0));
        let blue = renderer.scene_mut().insert_material(make_material([0.2, 0.38, 0.90, 1.0], 0.45, 0.0, [0.0,0.0,0.0], 0.0));
        let yellow = renderer.scene_mut().insert_material(make_material([0.95, 0.85, 0.17, 1.0], 0.45, 0.0, [0.0,0.0,0.0], 0.0));

        let floor_mesh = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(plane_mesh([0.0,0.0,0.0], ARENA_RADIUS))).as_mesh().unwrap();
        let _ = insert_object(&mut renderer, floor_mesh, flooring, glam::Mat4::from_translation(glam::Vec3::new(0.0,0.0,0.0)), ARENA_RADIUS);

        let cube_mesh_id = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(cube_mesh([0.0,0.0,0.0], 1.0))).as_mesh().unwrap();
        let small_cube_mesh = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(cube_mesh([0.0,0.0,0.0], 0.2))).as_mesh().unwrap();

        let meshes = [cube_mesh_id, cube_mesh_id, cube_mesh_id, cube_mesh_id];
        let materials = [red, green, blue, yellow];

        let mut state = AppState {
            window,
            surface,
            device,
            surface_format: fmt,
            renderer,
            last_frame: Instant::now(),
            frame_count: 0,
            cam_pos: glam::Vec3::new(0.0, 16.0, 32.0),
            cam_yaw: 0.0,
            cam_pitch: -0.45,
            keys: HashSet::new(),
            cursor_grabbed: false,
            mouse_delta: (0.0, 0.0),
            physics_integration: IntegrationParameters::default(),
            physics_bodies: RigidBodySet::new(),
            physics_colliders: ColliderSet::new(),
            physics_forces: IslandManager::new(),
            physics_broad_phase: BroadPhase::new(),
            physics_narrow_phase: NarrowPhase::new(),
            physics_joint_set: JointSet::new(),
            physics_ccd_solver: CCDSolver::new(),
            battle_shapes: Vec::new(),
            explosion_particles: Vec::new(),
            shape_count: MIN_SHAPES,
            round_active: false,
            round_end_instant: None,
            mats: [red, green, blue, yellow],
            meshes,
            time_render_end: None,
            time_about_to_wait_start: None,
            time_redraw_requested: None,
        };

        state.spawn_arena_walls();
        state.start_new_round();

        self.state = Some(state);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event: KeyEvent { state: ElementState::Pressed, physical_key: PhysicalKey::Code(KeyCode::Escape), ..}, .. } => {
                if state.cursor_grabbed {
                    state.cursor_grabbed = false;
                    let _ = state.window.set_cursor_grab(CursorGrabMode::None);
                    state.window.set_cursor_visible(true);
                } else {
                    event_loop.exit();
                }
            }
            WindowEvent::KeyboardInput { event: KeyEvent { state: ElementState::Pressed, physical_key: PhysicalKey::Code(KeyCode::Plus), ..}, .. } => {
                state.shape_count = (state.shape_count + 1).min(MAX_SHAPES);
                eprintln!("shape_count={}", state.shape_count);
                state.start_new_round();
            }
            WindowEvent::KeyboardInput { event: KeyEvent { state: ElementState::Pressed, physical_key: PhysicalKey::Code(KeyCode::Minus), ..}, .. } => {
                state.shape_count = (state.shape_count.saturating_sub(1)).max(MIN_SHAPES);
                eprintln!("shape_count={}", state.shape_count);
                state.start_new_round();
            }
            WindowEvent::KeyboardInput { event: KeyEvent { state: ks, physical_key: PhysicalKey::Code(key), .. }, .. } => {
                match ks {
                    ElementState::Pressed => { state.keys.insert(key); }
                    ElementState::Released => { state.keys.remove(&key); }
                }
            }
            WindowEvent::MouseInput { state: ElementState::Pressed, button: MouseButton::Left, .. } => {
                if !state.cursor_grabbed {
                    let ok = state.window.set_cursor_grab(CursorGrabMode::Confined).or_else(|_| state.window.set_cursor_grab(CursorGrabMode::Locked)).is_ok();
                    if ok {
                        state.window.set_cursor_visible(false);
                        state.cursor_grabbed = true;
                    }
                }
            }
            WindowEvent::Resized(s) if s.width > 0 && s.height > 0 => {
                state.surface.configure(&state.device, &wgpu::SurfaceConfiguration { usage: wgpu::TextureUsages::RENDER_ATTACHMENT, format: state.surface_format, width: s.width, height: s.height, present_mode: wgpu::PresentMode::Fifo, alpha_mode: wgpu::CompositeAlphaMode::Auto, view_formats: vec![], desired_maximum_frame_latency: 1, });
                state.renderer.set_render_size(s.width, s.height);
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                if let Some(last) = state.time_render_end {
                    let full_cycle_ms = last.elapsed().as_secs_f32() * 1000.0;
                    if state.frame_count % 60 == 0 { eprintln!("render_end -> next: {:.2}ms", full_cycle_ms); }
                }
                if let Some(about) = state.time_about_to_wait_start {
                    let gap_ms = about.elapsed().as_secs_f32() * 1000.0;
                    if gap_ms > 2.0 { eprintln!("about_to_wait -> redraw: {:.2}ms", gap_ms); }
                }
                state.time_redraw_requested = Some(now);
                let dt = (now - state.last_frame).as_secs_f32();
                state.last_frame = now;
                state.render(dt);
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
        if let Some(s) = &mut self.state {
            let now = Instant::now();
            if let Some(end) = s.time_render_end {
                let gap_ms = end.elapsed().as_secs_f32() * 1000.0;
                if gap_ms > 2.0 { eprintln!("render_end -> about_to_wait: {:.2}ms", gap_ms); }
            }
            s.time_about_to_wait_start = Some(now);
            s.window.request_redraw();
        }
    }
}

impl AppState {
    fn spawn_arena_walls(&mut self) {
        let wall_material = self.mats[0];
        let wall_mesh = self.renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([0.0,0.0,0.0],[ARENA_RADIUS, WALL_HEIGHT, WALL_THICKNESS]))).as_mesh().unwrap();

        let positions = [
            (0.0, WALL_HEIGHT/2.0, ARENA_RADIUS),
            (0.0, WALL_HEIGHT/2.0, -ARENA_RADIUS),
            (ARENA_RADIUS, WALL_HEIGHT/2.0, 0.0),
            (-ARENA_RADIUS, WALL_HEIGHT/2.0, 0.0),
        ];
        for (x, y, z) in positions {
            let transform = glam::Mat4::from_translation(glam::Vec3::new(x, y, z));
            let _ = insert_object(&mut self.renderer, wall_mesh, wall_material, transform, ARENA_RADIUS);
        }

        let static_body = RigidBodyBuilder::fixed().build();
        let body_handle = self.physics_bodies.insert(static_body);
        let wall_collider = ColliderBuilder::cuboid(ARENA_RADIUS + WALL_THICKNESS, WALL_HEIGHT, ARENA_RADIUS + WALL_THICKNESS)
            .friction(0.0).restitution(0.95).build();
        self.physics_colliders.insert_with_parent(wall_collider, body_handle, &mut self.physics_bodies);
    }

    fn start_new_round(&mut self) {
        // clear old objects
        for shape in self.battle_shapes.drain(..) {
            let _ = self.renderer.scene_mut().remove_object(shape.object_id);
            self.physics_colliders.remove(shape.collider_handle, &mut self.physics_bodies, false);
            self.physics_bodies.remove(shape.body_handle, &mut self.physics_forces, &mut self.physics_colliders, &mut self.physics_joint_set);
        }
        for part in self.explosion_particles.drain(..) {
            let _ = self.renderer.scene_mut().remove_object(part.object_id);
        }
        self.round_active = true;
        self.round_end_instant = None;

        let center = glam::Vec3::new(0.0, 1.2, 0.0);
        for i in 0..self.shape_count {
            let angle = i as f32 * 2.0 * std::f32::consts::PI / self.shape_count as f32;
            let radius = ARENA_RADIUS * 0.75;
            let floor = glam::Vec3::new(angle.cos() * radius, 1.0, angle.sin() * radius);
            let direction = (center - floor).normalize();
            let velocity = direction * 16.0 + glam::Vec3::new(0.0, 2.0, 0.0);
            let body = RigidBodyBuilder::dynamic().translation(floor.into()).linvel(velocity.into()).angvel([0.0, 5.0, 0.0].into()).build();
            let body_handle = self.physics_bodies.insert(body);

            let size = 1.0 + (i as f32 * 0.05);
            let collider = if i % 2 == 0 {
                ColliderBuilder::ball(size * 0.45)
            } else {
                ColliderBuilder::cuboid(size*0.4, size*0.4, size*0.4)
            }.restitution(0.95).friction(0.0).build();
            let collider_handle = self.physics_colliders.insert_with_parent(collider, body_handle, &mut self.physics_bodies);

            let mesh_id = self.meshes[i % self.meshes.len()];
            let mat_id = self.mats[i % self.mats.len()];
            let transform = glam::Mat4::from_translation(floor) * glam::Mat4::from_scale(glam::Vec3::splat(size * 0.8));
            let obj = insert_object(&mut self.renderer, mesh_id, mat_id, transform, size * 1.2).expect("insert object");

            self.battle_shapes.push(BattleShape { body_handle, collider_handle, object_id: obj, eliminated: false });
        }
    }

    fn create_explosion(&mut self, position: glam::Vec3) {
        for i in 0..16 {
            let angle = i as f32 * 2.0 * std::f32::consts::PI / 16.0;
            let offset = glam::Vec3::new(angle.cos(), 0.4, angle.sin()) * 0.4;
            let pos = position + offset;
            let mesh = self.renderer.scene_mut().insert_actor(helio::SceneActor::mesh(cube_mesh([0.0,0.0,0.0],0.12))).as_mesh().unwrap();
            let mat = self.mats[(i % self.mats.len())];
            let obj = insert_object(&mut self.renderer, mesh, mat, glam::Mat4::from_translation(pos), 0.2).expect("insert explosion");
            self.explosion_particles.push(BlastParticle { object_id: obj, birth: Instant::now() });
        }
    }

    fn step_physics(&mut self, dt: f32) {
        self.physics_integration.dt = dt;
        // Single step
        rapier3d::pipeline::PhysicsPipeline::new().step(
            &Vector::y_axis(),
            &self.physics_integration,
            &mut self.physics_forces,
            &mut self.physics_broad_phase,
            &mut self.physics_narrow_phase,
            &mut self.physics_bodies,
            &mut self.physics_colliders,
            &mut self.physics_joint_set,
            &mut self.physics_ccd_solver,
            &(),
            &(),
        );
    }

    fn update_battle_state(&mut self) {
        let mut alive = 0;
        let mut last_alive_i = None;

        for i in 0..self.battle_shapes.len() {
            let shape = &mut self.battle_shapes[i];
            if shape.eliminated { continue; }
            if let Some(body) = self.physics_bodies.get(shape.body_handle) {
                let pos = body.position().translation;
                let trans = glam::Mat4::from_cols_array(&body.position().to_homogeneous().to_cols_array());
                let _ = self.renderer.scene_mut().update_object_transform(shape.object_id, trans);

                let radial_dist = glam::Vec3::new(pos.x, 0.0, pos.z).length();
                let speed = body.linvel.norm();
                if radial_dist > ARENA_RADIUS || speed < 0.8 {
                    shape.eliminated = true;
                    self.create_explosion(glam::Vec3::new(pos.x, pos.y, pos.z));
                    let _ = self.renderer.scene_mut().remove_object(shape.object_id);
                    self.physics_colliders.remove(shape.collider_handle, &mut self.physics_bodies, false);
                    self.physics_bodies.remove(shape.body_handle, &mut self.physics_forces, &mut self.physics_colliders, &mut self.physics_joint_set);
                    continue;
                }

                alive += 1;
                last_alive_i = Some(i);
            }
        }

        if alive <= 1 {
            if self.round_active {
                self.round_active = false;
                self.round_end_instant = Some(Instant::now());
                if let Some(i) = last_alive_i {
                    log::info!("Round ended, winner: shape {}", i);
                } else {
                    log::info!("Round ended with no winner");
                }
            }
        }

        self.explosion_particles.retain(|p| {
            let alive = p.birth.elapsed() < Duration::from_millis(700);
            if !alive {
                let _ = self.renderer.scene_mut().remove_object(p.object_id);
            }
            alive
        });

        if !self.round_active {
            if let Some(end) = self.round_end_instant {
                if end.elapsed() >= ROUND_RESET_DELAY {
                    self.start_new_round();
                }
            }
        }
    }

    fn render(&mut self, dt: f32) {
        const SPEED: f32 = 11.0;
        const SENS: f32 = 0.002;
        self.cam_yaw += self.mouse_delta.0 * SENS;
        self.cam_pitch = (self.cam_pitch - self.mouse_delta.1 * SENS).clamp(-1.5, 1.5);
        self.mouse_delta = (0.0, 0.0);

        let (sy, cy) = self.cam_yaw.sin_cos();
        let (sp, cp) = self.cam_pitch.sin_cos();
        let forward = glam::Vec3::new(sy * cp, sp, -cy * cp);
        let right = glam::Vec3::new(cy, 0.0, sy);

        if self.keys.contains(&KeyCode::KeyW) { self.cam_pos += forward * SPEED * dt; }
        if self.keys.contains(&KeyCode::KeyS) { self.cam_pos -= forward * SPEED * dt; }
        if self.keys.contains(&KeyCode::KeyA) { self.cam_pos -= right * SPEED * dt; }
        if self.keys.contains(&KeyCode::KeyD) { self.cam_pos += right * SPEED * dt; }
        if self.keys.contains(&KeyCode::Space) { self.cam_pos.y += SPEED * dt; }
        if self.keys.contains(&KeyCode::ShiftLeft) { self.cam_pos.y -= SPEED * dt; }

        let size = self.window.inner_size();
        let aspect = size.width as f32 / size.height.max(1) as f32;
        let camera = Camera::perspective_look_at(self.cam_pos, self.cam_pos + forward, glam::Vec3::Y, std::f32::consts::FRAC_PI_4, aspect, 0.1, 200.0);

        self.step_physics(dt);
        self.update_battle_state();

        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(e) => { log::warn!("Surface error: {:?}", e); return; }
        };
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        if let Err(e) = self.renderer.render(&camera, &view) { log::error!("Render error: {:?}", e); }
        output.present();

        self.time_render_end = Some(Instant::now());
        self.frame_count += 1;

        if self.frame_count % 60 == 0 {
            let live = self.battle_shapes.iter().filter(|b| !b.eliminated).count();
            eprintln!("Frame {}: live={} particles={} shapes={}, active={}", self.frame_count, live, self.explosion_particles.len(), self.shape_count, self.round_active);
        }
    }
}
