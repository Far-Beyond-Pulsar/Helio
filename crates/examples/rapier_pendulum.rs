//! Rapier Pendulum Demo — helio v3
//!
//! A chain of spheres connected by ball joints, swinging under gravity.
//! Controls:
//!   C — toggle physics
//!   R — reset chain
//!   WASD / Space / Shift — fly camera
//!   Escape — release cursor / exit

mod v3_demo_common;
use v3_demo_common::{box_mesh, cube_mesh, insert_object, make_material, point_light, sphere_mesh};

use helio::{required_wgpu_features, required_wgpu_limits, Camera, MaterialId, ObjectId, Renderer, RendererConfig};
use rapier3d::prelude::*;
use std::collections::HashSet;
use std::sync::Arc;
use winit::{application::ApplicationHandler, event::*, event_loop::{ActiveEventLoop, EventLoop}, keyboard::{KeyCode, PhysicalKey}, window::{CursorGrabMode, Window, WindowId}};

struct PendulumSegment {
    id: ObjectId,
    body_handle: RigidBodyHandle,
    collider_handle: ColliderHandle,
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

    segments: Vec<PendulumSegment>,
    sphere_mat: MaterialId,
    physics_enabled: bool,

    physics_integration: IntegrationParameters,
    physics_bodies: RigidBodySet,
    physics_colliders: ColliderSet,
    physics_forces: IslandManager,
    physics_broad_phase: DefaultBroadPhase,
    physics_narrow_phase: NarrowPhase,
    physics_impulse_joints: ImpulseJointSet,
    physics_multibody_joint_set: MultibodyJointSet,
    physics_ccd_solver: CCDSolver,

    time_render_end: Option<std::time::Instant>,
    time_about_to_wait_start: Option<std::time::Instant>,
    time_redraw_requested: Option<std::time::Instant>,
}

struct App { state: Option<AppState> }

fn main() {
    env_logger::init();
    log::info!("Starting Rapier Pendulum Demo");
    EventLoop::new().expect("event loop").run_app(&mut App { state: None }).expect("run");
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() { return; }

        let window = Arc::new(event_loop.create_window(Window::default_attributes().with_title("Helio — Rapier Pendulum Demo").with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32))).expect("window"));
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

        let floor_mat = renderer.scene_mut().insert_material(make_material([0.25, 0.25, 0.3, 1.0], 0.85, 0.03, [0.0, 0.0, 0.0], 0.0));
        let sphere_mat = renderer.scene_mut().insert_material(make_material([0.2, 0.5, 0.8, 1.0], 0.45, 0.05, [0.0, 0.0, 0.0], 0.0));

        let floor_mesh = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(plane_mesh([0.0, 0.0, 0.0], 50.0))).as_mesh().unwrap();
        let _ = insert_object(&mut renderer, floor_mesh, floor_mat, glam::Mat4::from_translation(glam::Vec3::new(0.0, -0.01, 0.0)), 50.0);

        let _ = renderer.scene_mut().insert_actor(helio::SceneActor::light(point_light([20.0, 24.0, 20.0], [0.9, 0.85, 0.8], 14.0, 60.0))).as_light().unwrap();
        let _ = renderer.scene_mut().insert_actor(helio::SceneActor::light(point_light([-20.0, 20.0, -20.0], [0.6, 0.7, 1.0], 12.0, 60.0))).as_light().unwrap();

        let mut state = AppState {
            window,
            surface,
            device,
            surface_format: fmt,
            renderer,
            last_frame: std::time::Instant::now(),
            frame_count: 0,
            cam_pos: glam::Vec3::new(0.0, 8.0, 35.0),
            cam_yaw: 0.0,
            cam_pitch: -0.2,
            keys: HashSet::new(),
            cursor_grabbed: false,
            mouse_delta: (0.0, 0.0),
            segments: Vec::new(),
            sphere_mat,
            physics_enabled: true,
            physics_integration: IntegrationParameters::default(),
            physics_bodies: RigidBodySet::new(),
            physics_colliders: ColliderSet::new(),
            physics_forces: IslandManager::new(),
            physics_broad_phase: DefaultBroadPhase::new(),
            physics_narrow_phase: NarrowPhase::new(),
            physics_impulse_joints: ImpulseJointSet::new(),
            physics_multibody_joint_set: MultibodyJointSet::new(),
            physics_ccd_solver: CCDSolver::new(),
            time_render_end: None,
            time_about_to_wait_start: None,
            time_redraw_requested: None,
        };

        state.reset_chain(sphere_mat);
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
            WindowEvent::KeyboardInput { event: KeyEvent { state: ElementState::Pressed, physical_key: PhysicalKey::Code(KeyCode::KeyC), ..}, .. } => {
                state.physics_enabled = !state.physics_enabled;
                eprintln!("rapier physics={}", state.physics_enabled);
            }
            WindowEvent::KeyboardInput { event: KeyEvent { state: ElementState::Pressed, physical_key: PhysicalKey::Code(KeyCode::KeyR), ..}, .. } => {
                state.clear_chain();
                state.reset_chain(state.sphere_mat);
            }
            WindowEvent::KeyboardInput { event: KeyEvent { state: ks, physical_key: PhysicalKey::Code(key), ..}, .. } => {
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
                state.surface.configure(&state.device, &wgpu::SurfaceConfiguration { usage: wgpu::TextureUsages::RENDER_ATTACHMENT, format: state.surface_format, width: s.width, height: s.height, present_mode: wgpu::PresentMode::Fifo, alpha_mode: wgpu::CompositeAlphaMode::Auto, view_formats: vec![], desired_maximum_frame_latency: 1 });
                state.renderer.set_render_size(s.width, s.height);
            }
            WindowEvent::RedrawRequested => {
                let now = std::time::Instant::now();
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
            let now = std::time::Instant::now();
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
    fn clear_chain(&mut self) {
        for seg in self.segments.drain(..) {
            let _ = self.renderer.scene_mut().remove_object(seg.id);
            self.physics_colliders.remove(seg.collider_handle, &mut self.physics_forces, &mut self.physics_bodies, false);
            self.physics_bodies.remove(seg.body_handle, &mut self.physics_forces, &mut self.physics_colliders, &mut self.physics_impulse_joints, &mut self.physics_multibody_joint_set, true);
        }
    }

    fn reset_chain(&mut self, sphere_mat: MaterialId) {
        self.clear_chain();

        let pivot_body = RigidBodyBuilder::fixed().translation([0.0, 12.0, 0.0].into()).build();
        let pivot_handle = self.physics_bodies.insert(pivot_body);
        let pivot_collider = ColliderBuilder::ball(0.25).build();
        self.physics_colliders.insert_with_parent(pivot_collider, pivot_handle, &mut self.physics_bodies);

        let chain_length = 13;
        let seg_length = 1.0;

        let mut parent_handle = pivot_handle;
        let mut last_body_handle = pivot_handle;

        for i in 0..chain_length {
            let y = 12.0 - (i as f32 + 1.0) * seg_length;
            let sphere_mesh_id = self.renderer.scene_mut().insert_actor(helio::SceneActor::mesh(sphere_mesh([0.0,0.0,0.0], 0.4))).as_mesh().unwrap();
            let obj = insert_object(&mut self.renderer, sphere_mesh_id, sphere_mat, glam::Mat4::from_translation(glam::Vec3::new(0.0, y, 0.0)) * glam::Mat4::from_scale(glam::Vec3::splat(0.8)), 0.9).expect("insert sphere");

            let body = RigidBodyBuilder::dynamic().translation([0.0,y,0.0].into()).build();
            let body_handle = self.physics_bodies.insert(body);
            let collider = ColliderBuilder::ball(0.4).restitution(0.2).friction(0.4).build();
            let collider_handle = self.physics_colliders.insert_with_parent(collider, body_handle, &mut self.physics_bodies);

            let joint = BallJoint::new(Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, seg_length, 0.0));
            self.physics_impulse_joints.insert(parent_handle, body_handle, joint, true);

            self.segments.push(PendulumSegment { id: obj, body_handle, collider_handle });
            parent_handle = body_handle;
            last_body_handle = body_handle;
        }

        let ground_body = RigidBodyBuilder::fixed().translation([0.0, -0.2, 0.0].into()).build();
        let ground_handle = self.physics_bodies.insert(ground_body);
        let ground_collider = ColliderBuilder::cuboid(25.0, 0.5, 25.0).build();
        self.physics_colliders.insert_with_parent(ground_collider, ground_handle, &mut self.physics_bodies);
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
        for seg in &self.segments {
            if let Some(body) = self.physics_bodies.get(seg.body_handle) {
                let pos = body.position();
                let translation = glam::Vec3::new(pos.translation.x, pos.translation.y, pos.translation.z);
                let rotation = glam::Quat::from_xyzw(pos.rotation.i, pos.rotation.j, pos.rotation.k, pos.rotation.w);
                let transform = glam::Mat4::from_translation(translation) * glam::Mat4::from_quat(rotation) * glam::Mat4::from_scale(glam::Vec3::splat(0.8));
                let _ = self.renderer.scene_mut().update_object_transform(seg.id, transform);
            }
        }
    }

    fn render(&mut self, dt: f32) {
        const SPEED: f32 = 12.0;
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

        if self.physics_enabled {
            self.step_physics(dt);
            self.sync_transforms_from_physics();
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
            eprintln!("Rapier Pendulum: frame {} segments={} sim={}", self.frame_count, self.segments.len(), self.physics_enabled);
        }
    }
}
