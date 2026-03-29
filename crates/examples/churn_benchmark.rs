//! Churn Benchmark — helio v3
//!
//! Stress test with objects continuously spawning, despawning, and moving.
//!
//! Controls:
//!   WASD / Space / Shift — fly
//!   +/-                  — open/close spawn pressure (number of objects added each frame)
//!   C                    — toggle object-object collisions on/off
//!   Escape               — release cursor / exit

mod v3_demo_common;
use v3_demo_common::{box_mesh, cube_mesh, insert_object, make_material, point_light, plane_mesh};

use helio::{required_wgpu_features, required_wgpu_limits, Camera, MaterialId, MeshId, ObjectId, Renderer, RendererConfig};
use rapier3d::prelude::*;

use crate::nalgebra::UnitQuaternion;
use std::collections::HashSet;
use std::sync::Arc;
use winit::{application::ApplicationHandler, event::*, event_loop::{ActiveEventLoop, EventLoop}, keyboard::{KeyCode, PhysicalKey}, window::{CursorGrabMode, Window, WindowId}};

const MAX_DYNAMIC_OBJECTS: usize = 2200;
const START_SPAWN_RATE: usize = 8;
const MIN_SPAWN_RATE: usize = 1;
const MAX_SPAWN_RATE: usize = 64;

struct SpawnedObject {
    id: ObjectId,
    seed: f32,
    speed: f32,
    scale: f32,
    mesh: MeshId,
    material: MaterialId,
    body_handle: RigidBodyHandle,
    collider_handle: ColliderHandle,
}

struct SimpleRng(u64);
impl SimpleRng {
    fn new(seed: u64) -> Self { Self(seed | 1) }
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 32) as u32
    }
    fn next_f32(&mut self, min: f32, max: f32) -> f32 {
        let r = self.next_u32() as f32 / 4294967295.0;
        min + (max - min) * r
    }
    fn next_usize(&mut self, max: usize) -> usize {
        (self.next_u32() as usize) % max
    }
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
    last_frame: std::time::Instant,
    frame_count: u64,

    cam_pos: glam::Vec3,
    cam_yaw: f32,
    cam_pitch: f32,
    keys: HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta: (f32, f32),

    spawn_rate: usize,
    dynamic_objects: Vec<SpawnedObject>,
    meshes: Vec<MeshId>,
    materials: Vec<MaterialId>,
    rng: SimpleRng,
    collisions_enabled: bool,

    time_render_end: Option<std::time::Instant>,
    time_about_to_wait_start: Option<std::time::Instant>,

    physics_integration: IntegrationParameters,
    physics_bodies: RigidBodySet,
    physics_colliders: ColliderSet,
    physics_forces: IslandManager,
    physics_broad_phase: DefaultBroadPhase,
    physics_narrow_phase: NarrowPhase,
    physics_impulse_joints: ImpulseJointSet,
    physics_multibody_joint_set: MultibodyJointSet,
    physics_ccd_solver: CCDSolver,

    time_redraw_requested: Option<std::time::Instant>,
}

fn main() {
    env_logger::init();
    log::info!("Starting Churn Benchmark");
    EventLoop::new().expect("event loop").run_app(&mut App { state: None }).expect("run");
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() { return; }

        let window = Arc::new(event_loop.create_window(Window::default_attributes().with_title("Helio — Churn Benchmark").with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32))).expect("window"));

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { backends: wgpu::Backends::all(), flags: wgpu::InstanceFlags::empty(), ..Default::default() });
        let surface = instance.create_surface(window.clone()).expect("surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: Some(&surface), force_fallback_adapter: false, })).expect("adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { required_features: required_wgpu_features(adapter.features()), required_limits: required_wgpu_limits(adapter.limits()), ..Default::default() })).expect("device");
        device.on_uncaptured_error(Arc::new(|e: wgpu::Error| { panic!("[GPU] {:?}", e) }));
        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let caps = surface.get_capabilities(&adapter);
        let fmt = caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(caps.formats[0]);
        let size = window.inner_size();
        surface.configure(&device, &wgpu::SurfaceConfiguration { usage: wgpu::TextureUsages::RENDER_ATTACHMENT, format: fmt, width: size.width, height: size.height, present_mode: wgpu::PresentMode::Fifo, alpha_mode: caps.alpha_modes[0], view_formats: vec![], desired_maximum_frame_latency: 1, });

        let mut renderer = Renderer::new(device.clone(), queue.clone(), RendererConfig::new(size.width, size.height, fmt));
        renderer.set_ambient([0.04, 0.04, 0.05], 1.0);

        let mat_floor = renderer.scene_mut().insert_material(make_material([0.25, 0.25, 0.30, 1.0], 0.85, 0.03, [0.0, 0.0, 0.0], 0.0));
        let mat_red = renderer.scene_mut().insert_material(make_material([0.85, 0.12, 0.12, 1.0], 0.65, 0.00, [0.0, 0.0, 0.0], 0.0));
        let mat_green = renderer.scene_mut().insert_material(make_material([0.17, 0.82, 0.28, 1.0], 0.60, 0.00, [0.0, 0.0, 0.0], 0.0));
        let mat_blue = renderer.scene_mut().insert_material(make_material([0.16, 0.40, 0.90, 1.0], 0.70, 0.00, [0.0, 0.0, 0.0], 0.0));
        let mat_steel = renderer.scene_mut().insert_material(make_material([0.7, 0.7, 0.75, 1.0], 0.15, 0.80, [0.0, 0.0, 0.0], 0.0));

        let floor_mesh = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(plane_mesh([0.0, 0.0, 0.0], 40.0))).as_mesh().unwrap();
        let _ = insert_object(&mut renderer, floor_mesh, mat_floor, glam::Mat4::from_translation(glam::Vec3::new(0.0, -0.01, 0.0)), 40.0);

        let mut mesh_list = Vec::new();
        mesh_list.push(renderer.scene_mut().insert_actor(helio::SceneActor::mesh(cube_mesh([0.0, 0.0, 0.0], 0.35))).as_mesh().unwrap());
        mesh_list.push(renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [0.15, 0.65, 0.15]))).as_mesh().unwrap());
        mesh_list.push(renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [0.60, 0.20, 0.20]))).as_mesh().unwrap());

        let offset = 20.0;
        let _ = renderer.scene_mut().insert_actor(helio::SceneActor::light(point_light([-offset, 5.0, -offset], [0.8, 0.7, 0.55], 7.0, 40.0))).as_light().unwrap();
        let _ = renderer.scene_mut().insert_actor(helio::SceneActor::light(point_light([offset, 5.0, offset], [0.5, 0.7, 1.0], 7.0, 40.0))).as_light().unwrap();

        self.state = Some(AppState {
            window,
            surface,
            device,
            surface_format: fmt,
            renderer,
            last_frame: std::time::Instant::now(),
            frame_count: 0,
            cam_pos: glam::Vec3::new(0.0, 7.0, 25.0),
            cam_yaw: 0.0,
            cam_pitch: -0.26,
            keys: HashSet::new(),
            cursor_grabbed: false,
            mouse_delta: (0.0, 0.0),
            spawn_rate: START_SPAWN_RATE,
            dynamic_objects: Vec::new(),
            meshes: mesh_list,
            materials: vec![mat_red, mat_green, mat_blue, mat_steel],
            rng: SimpleRng::new(0x59A0_D3E4_B2CA_1897),
            collisions_enabled: false,
            time_render_end: None,
            time_about_to_wait_start: None,
            time_redraw_requested: None,
            physics_integration: IntegrationParameters::default(),
            physics_bodies: RigidBodySet::new(),
            physics_colliders: ColliderSet::new(),
            physics_forces: IslandManager::new(),
            physics_broad_phase: DefaultBroadPhase::new(),
            physics_narrow_phase: NarrowPhase::new(),
            physics_impulse_joints: ImpulseJointSet::new(),
            physics_multibody_joint_set: MultibodyJointSet::new(),
            physics_ccd_solver: CCDSolver::new(),
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return }; 
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event: KeyEvent { state: ElementState::Pressed, physical_key: PhysicalKey::Code(KeyCode::Escape), .. }, .. } => {
                if state.cursor_grabbed {
                    state.cursor_grabbed = false;
                    let _ = state.window.set_cursor_grab(CursorGrabMode::None);
                    state.window.set_cursor_visible(true);
                } else {
                    event_loop.exit();
                }
            }
            WindowEvent::KeyboardInput { event: KeyEvent { state: ElementState::Pressed, physical_key: PhysicalKey::Code(KeyCode::Equal), .. }, .. } => {
                state.spawn_rate = (state.spawn_rate + 1).min(MAX_SPAWN_RATE);
                eprintln!("spawn_rate={}", state.spawn_rate);
            }
            WindowEvent::KeyboardInput { event: KeyEvent { state: ElementState::Pressed, physical_key: PhysicalKey::Code(KeyCode::Minus), .. }, .. } => {
                state.spawn_rate = state.spawn_rate.saturating_sub(1).max(MIN_SPAWN_RATE);
                eprintln!("spawn_rate={}", state.spawn_rate);
            }
            WindowEvent::KeyboardInput { event: KeyEvent { state: ElementState::Pressed, physical_key: PhysicalKey::Code(KeyCode::KeyC), .. }, .. } => {
                state.collisions_enabled = !state.collisions_enabled;
                eprintln!("collisions={}", state.collisions_enabled);
            }
            WindowEvent::KeyboardInput { event: KeyEvent { state: ks, physical_key: PhysicalKey::Code(key), .. }, .. } => match ks {
                ElementState::Pressed => { state.keys.insert(key); }
                ElementState::Released => { state.keys.remove(&key); }
            },
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
                state.surface.configure(&state.device, &wgpu::SurfaceConfiguration {
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    format: state.surface_format,
                    width: s.width,
                    height: s.height,
                    present_mode: wgpu::PresentMode::Fifo,
                    alpha_mode: wgpu::CompositeAlphaMode::Auto,
                    view_formats: vec![],
                    desired_maximum_frame_latency: 1,
                });
                state.renderer.set_render_size(s.width, s.height);
            }
            WindowEvent::RedrawRequested => {
                let now = std::time::Instant::now();

                if let Some(last_render_end) = state.time_render_end {
                    let full_cycle_ms = last_render_end.elapsed().as_secs_f32() * 1000.0;
                    if state.frame_count % 60 == 0 {
                        eprintln!("render_end -> next RedrawRequested: {:.2}ms", full_cycle_ms);
                    }
                }

                if let Some(about_to_wait_start) = state.time_about_to_wait_start {
                    let gap_ms = about_to_wait_start.elapsed().as_secs_f32() * 1000.0;
                    if gap_ms > 2.0 { eprintln!("about_to_wait -> RedrawRequested: {:.2}ms", gap_ms); }
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
            let now = std::time::Instant::now();
            if let Some(render_end) = s.time_render_end {
                let gap_ms = render_end.elapsed().as_secs_f32() * 1000.0;
                if gap_ms > 2.0 { eprintln!("render_end -> about_to_wait: {:.2}ms", gap_ms); }
            }
            s.time_about_to_wait_start = Some(now);
            s.window.request_redraw();
        }
    }
}

impl AppState {
    fn spawn_objects(&mut self) {
        let max_count = MAX_DYNAMIC_OBJECTS;
        for _ in 0..self.spawn_rate {
            let mesh = self.meshes[self.rng.next_usize(self.meshes.len())];
            let material = self.materials[self.rng.next_usize(self.materials.len())];
            let radius = self.rng.next_f32(5.0, 20.0);
            let angle = self.rng.next_f32(0.0, std::f32::consts::TAU);
            let height = self.rng.next_f32(0.35, 2.1);
            let scale = self.rng.next_f32(0.25, 1.0);
            let pos = glam::Vec3::new(angle.cos() * radius, height, angle.sin() * radius);
            let transform = glam::Mat4::from_translation(pos) * glam::Mat4::from_scale(glam::Vec3::splat(scale));

            let body = RigidBodyBuilder::dynamic()
                .translation([pos.x, pos.y, pos.z].into())
                .linvel(Vector::new(self.rng.next_f32(-4.0, 4.0), self.rng.next_f32(-1.0, 1.0), self.rng.next_f32(-4.0, 4.0)))
                .angvel(Vector::new(self.rng.next_f32(-2.0, 2.0), self.rng.next_f32(-2.0, 2.0), self.rng.next_f32(-2.0, 2.0)))
                .build();
            let body_handle = self.physics_bodies.insert(body);

            let collider = ColliderBuilder::ball((scale * 0.35).max(0.2))
                .restitution(0.3)
                .friction(0.2)
                .build();
            let collider_handle = self.physics_colliders.insert_with_parent(collider, body_handle, &mut self.physics_bodies);

            if let Ok(obj_id) = insert_object(&mut self.renderer, mesh, material, transform, (scale * 1.3).max(0.15)) {
                self.dynamic_objects.push(SpawnedObject { id: obj_id, seed: self.rng.next_f32(0.0, std::f32::consts::TAU), speed: self.rng.next_f32(0.4, 1.6), scale, mesh, material, body_handle, collider_handle });
            } else {
                self.physics_colliders.remove(collider_handle, &mut self.physics_forces, &mut self.physics_bodies, false);
                self.physics_bodies.remove(body_handle, &mut self.physics_forces, &mut self.physics_colliders, &mut self.physics_impulse_joints, &mut self.physics_multibody_joint_set, true);
            }

            if self.dynamic_objects.len() > max_count {
                if let Some(dead) = self.dynamic_objects.first() {
                    let _ = self.renderer.scene_mut().remove_object(dead.id);
                    self.physics_colliders.remove(dead.collider_handle, &mut self.physics_forces, &mut self.physics_bodies, false);
                    self.physics_bodies.remove(dead.body_handle, &mut self.physics_forces, &mut self.physics_colliders, &mut self.physics_impulse_joints, &mut self.physics_multibody_joint_set, true);
                }
                self.dynamic_objects.remove(0);
            }
        }
    }

    fn animate_objects(&mut self) {
        let t = (self.frame_count as f32) * 0.01;

        for variant in &mut self.dynamic_objects {
            let phase = variant.seed + t * variant.speed;
            let radius = 8.0 + (phase * 0.25).sin() * 2.0;
            let x = phase.cos() * radius;
            let z = phase.sin() * radius;
            let y = 0.5 + (phase * 1.3).sin() * 0.8;
            let pos = glam::Vec3::new(x, y, z);
            let transform = glam::Mat4::from_translation(pos)
                * glam::Mat4::from_rotation_y(phase * 1.37)
                * glam::Mat4::from_scale(glam::Vec3::splat(variant.scale));
            let _ = self.renderer.scene_mut().update_object_transform(variant.id, transform);

            if let Some(body) = self.physics_bodies.get_mut(variant.body_handle) {
                body.set_position(
                    Isometry::from_parts(
                        Translation::from(Vector::new(pos.x, pos.y, pos.z)),
                        UnitQuaternion::from_euler_angles(0.0, phase * 1.37, 0.0),
                    ),
                    true,
                );
                body.set_linvel(Vector::zeros(), true);
                body.set_angvel(Vector::zeros(), true);
            }
        }
    }

    fn step_physics(&mut self, dt: f32) {
        self.physics_integration.dt = dt;
        PhysicsPipeline::new().step(
            &Vector::y_axis(),
            &self.physics_integration,
            &mut self.physics_forces,
            &mut self.physics_broad_phase,
            &mut self.physics_narrow_phase,
            &mut self.physics_bodies,
            &mut self.physics_colliders,
            &mut self.physics_impulse_joints,
            &mut self.physics_multibody_joint_set,
            &mut self.physics_ccd_solver,
            None,
            &(),
            &(),
        );
    }

    fn sync_transforms_from_physics(&mut self) {
        for variant in &self.dynamic_objects {
            if let Some(body) = self.physics_bodies.get(variant.body_handle) {
                let pos = body.position();
                let translation = glam::Vec3::new(pos.translation.vector.x, pos.translation.vector.y, pos.translation.vector.z);
                let rotation = glam::Quat::from_xyzw(pos.rotation.i, pos.rotation.j, pos.rotation.k, pos.rotation.w);
                let transform = glam::Mat4::from_translation(translation) * glam::Mat4::from_quat(rotation) * glam::Mat4::from_scale(glam::Vec3::splat(variant.scale));
                let _ = self.renderer.scene_mut().update_object_transform(variant.id, transform);
            }
        }
    }

    fn render(&mut self, dt: f32) {
        const SPEED: f32 = 8.0;
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
        let camera = Camera::perspective_look_at(self.cam_pos, self.cam_pos + forward, glam::Vec3::Y, std::f32::consts::FRAC_PI_4, aspect, 0.1, 300.0);

        if self.frame_count % 2 == 0 {
            self.spawn_objects();
        }

        if self.collisions_enabled {
            self.step_physics(dt);
            self.sync_transforms_from_physics();
        } else {
            self.animate_objects();
        }

        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(e) => { log::warn!("Surface error: {:?}", e); return; }
        };
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        if let Err(e) = self.renderer.render(&camera, &view) { log::error!("Render error: {:?}", e); }
        output.present();

        self.frame_count += 1;
        self.time_render_end = Some(std::time::Instant::now());

        if self.frame_count % 60 == 0 {
            eprintln!("Churn: frame {} objects={} spawn_rate={} dt={:.3}ms", self.frame_count, self.dynamic_objects.len(), self.spawn_rate, dt * 1000.0);
        }
    }
}
