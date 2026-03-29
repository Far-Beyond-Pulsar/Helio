//! Rapier + Salva3D fluid demo — helio v3
//!
//! Controls:
//!   R — reset fluid and rigid scene
//!   C — toggle rapier physics
//!   WASD / Space / Shift — flight cam
//!   Escape — exit

mod v3_demo_common;
use v3_demo_common::{box_mesh, insert_object, make_material, plane_mesh, point_light, sphere_mesh};

use helio::{required_wgpu_features, required_wgpu_limits, Camera, ObjectId, Renderer, RendererConfig};
use rapier3d::prelude::*;
use salva3d::integrations::rapier::FluidsPipeline;
use salva3d::math::{Point, Vector};
use crate::nalgebra::Vector3;
use salva3d::object::{Boundary, Fluid};
use salva3d::solver::{Akinci2013SurfaceTension, XSPHViscosity};
use std::collections::HashSet;
use std::sync::Arc;
use winit::{application::ApplicationHandler, event::*, event_loop::{ActiveEventLoop, EventLoop}, keyboard::{KeyCode, PhysicalKey}, window::{CursorGrabMode, Window, WindowId}};

const PARTICLE_RADIUS: f32 = 0.06;
const SMOOTHING_FACTOR: f32 = 2.2;
const MAX_PARTICLES: usize = 600;

struct RigidBox {
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
    keys: HashSet<PhysicalKey>,
    cursor_grabbed: bool,
    mouse_delta: (f32, f32),

    physics_enabled: bool,
    fluid_spawn_timer: f32,

    physics_integration: IntegrationParameters,
    physics_bodies: RigidBodySet,
    physics_colliders: ColliderSet,
    physics_forces: IslandManager,
    physics_broad_phase: DefaultBroadPhase,
    physics_narrow_phase: NarrowPhase,
    physics_impulse_joints: ImpulseJointSet,
    physics_multibody_joint_set: MultibodyJointSet,
    physics_ccd_solver: CCDSolver,

    rigid_boxes: Vec<RigidBox>,

    fluids_pipeline: FluidsPipeline,
    fluid_handle: Option<salva3d::object::FluidHandle>,
    boundary_handle: Option<salva3d::object::BoundaryHandle>,
    fluid_particle_ids: Vec<ObjectId>,
    fluid_material: helio::MaterialId,
    fluid_mesh: helio::MeshId,

    time_render_end: Option<std::time::Instant>,
    time_about_to_wait_start: Option<std::time::Instant>,
    time_redraw_requested: Option<std::time::Instant>,
}

struct App { state: Option<AppState> }

fn main() {
    env_logger::init();
    log::info!("Starting Rapier+Salva demo");
    EventLoop::new().expect("event loop").run_app(&mut App { state: None }).expect("run");
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() { return; }

        let window = Arc::new(event_loop.create_window(Window::default_attributes().with_title("Helio — Rapier+Salva3D").with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32))).expect("window"));
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
        renderer.set_ambient([0.07, 0.07, 0.09], 1.0);

        let floor_mat = renderer.scene_mut().insert_material(make_material([0.22, 0.25, 0.28, 1.0], 0.9, 0.03, [0.0, 0.0, 0.0], 0.0));
        let box_mat = renderer.scene_mut().insert_material(make_material([0.67, 0.42, 0.24, 1.0], 0.55, 0.2, [0.0, 0.0, 0.0], 0.0));
        let fluid_mat = renderer.scene_mut().insert_material(make_material([0.20, 0.70, 0.98, 0.6], 0.2, 0.0, [0.0, 0.0, 0.0], 0.0));

        let floor_mesh = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(plane_mesh([0.0, 0.0, 0.0], 40.0))).as_mesh().unwrap();
        let _ = insert_object(&mut renderer, floor_mesh, floor_mat, glam::Mat4::from_translation(glam::Vec3::new(0.0, -0.02, 0.0)), 40.0);

        let _ = renderer.scene_mut().insert_actor(helio::SceneActor::light(point_light([22.0, 26.0, 20.0], [0.94, 0.88, 0.80], 12.0, 90.0))).as_light().unwrap();
        let _ = renderer.scene_mut().insert_actor(helio::SceneActor::light(point_light([-20.0, 24.0, -20.0], [0.72, 0.82, 1.0], 10.0, 90.0))).as_light().unwrap();

        let fluid_mesh = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(sphere_mesh([0.0, 0.0, 0.0], PARTICLE_RADIUS))).as_mesh().unwrap();

        let mut state = AppState {
            window,
            surface,
            device,
            surface_format: fmt,
            renderer,
            last_frame: std::time::Instant::now(),
            frame_count: 0,
            cam_pos: glam::Vec3::new(0.0, 6.0, 15.0),
            cam_yaw: 0.0,
            cam_pitch: -0.3,
            keys: HashSet::new(),
            cursor_grabbed: false,
            mouse_delta: (0.0, 0.0),
            physics_enabled: true,
            fluid_spawn_timer: 0.0,
            physics_integration: IntegrationParameters::default(),
            physics_bodies: RigidBodySet::new(),
            physics_colliders: ColliderSet::new(),
            physics_forces: IslandManager::new(),
            physics_broad_phase: DefaultBroadPhase::new(),
            physics_narrow_phase: NarrowPhase::new(),
            physics_impulse_joints: ImpulseJointSet::new(),
            physics_multibody_joint_set: MultibodyJointSet::new(),
            physics_ccd_solver: CCDSolver::new(),
            rigid_boxes: Vec::new(),
            fluids_pipeline: FluidsPipeline::new(PARTICLE_RADIUS, SMOOTHING_FACTOR),
            fluid_handle: None,
            boundary_handle: None,
            fluid_particle_ids: Vec::new(),
            fluid_material: fluid_mat,
            fluid_mesh,
            time_render_end: None,
            time_about_to_wait_start: None,
            time_redraw_requested: None,
        };

        state.reset_scene(box_mat);

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
                eprintln!("Rapier physics={}", state.physics_enabled);
            }
            WindowEvent::KeyboardInput { event: KeyEvent { state: ElementState::Pressed, physical_key: PhysicalKey::Code(KeyCode::KeyR), ..}, .. } => {
                state.reset_scene(state.fluid_material);
            }
            WindowEvent::KeyboardInput { event: ks, .. } => match ks.state {
                ElementState::Pressed => { state.keys.insert(ks.physical_key); }
                ElementState::Released => { state.keys.remove(&ks.physical_key); }
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
    fn reset_scene(&mut self, box_mat: helio::MaterialId) {
        self.clear_rigid_boxes();
        self.fluids_pipeline = FluidsPipeline::new(PARTICLE_RADIUS, SMOOTHING_FACTOR);
        self.fluid_handle = None;
        self.boundary_handle = None;
        self.fluid_particle_ids.clear();
        self.fluid_spawn_timer = 0.0;

        // Reset physics world to floor + a few boxes.
        let ground_body = RigidBodyBuilder::fixed().translation(vector![0.0, 0.0, 0.0]).build();
        let ground_handle = self.physics_bodies.insert(ground_body);
        let ground_collider = ColliderBuilder::cuboid(40.0, 0.5, 40.0).friction(0.9).restitution(0.2).build();
        self.physics_colliders.insert_with_parent(ground_collider, ground_handle, &mut self.physics_bodies);

        for x in -2..=2 {
            for z in -2..=2 {
                let pos = glam::Vec3::new(x as f32 * 2.2, 1.0, z as f32 * 2.2);
                let transform = glam::Mat4::from_translation(pos);
                let box_mesh = self.renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [0.8, 0.8, 0.8]))).as_mesh().unwrap();
                let obj = insert_object(&mut self.renderer, box_mesh, box_mat, transform, 1.4).expect("insert box");

                let body = RigidBodyBuilder::dynamic().translation([pos.x, pos.y, pos.z].into()).build();
                let body_handle = self.physics_bodies.insert(body);
                let collider = ColliderBuilder::cuboid(0.8, 0.8, 0.8).friction(0.6).restitution(0.05).build();
                let collider_handle = self.physics_colliders.insert_with_parent(collider, body_handle, &mut self.physics_bodies);

                self.rigid_boxes.push(RigidBox { id: obj, body_handle, collider_handle });
            }
        }

        // Create floor boundary for fluid in Salva3D.
        let mut floor_points = Vec::new();
        let spacing = PARTICLE_RADIUS * 1.5;
        for x in (-40..=40).step_by(4) {
            for z in (-40..=40).step_by(4) {
                floor_points.push(Point::new(x as f32 * spacing, 0.0, z as f32 * spacing));
            }
        }
        let boundary = Boundary::new(floor_points);
        self.boundary_handle = Some(self.fluids_pipeline.liquid_world.add_boundary(boundary));

        let init_positions = Self::make_fluid_block(10, 10, 10, 0.8, 6.0);

        let mut fluid = Fluid::new(init_positions, PARTICLE_RADIUS, 1000.0);
        fluid.nonpressure_forces.push(Box::new(XSPHViscosity::new(0.5, 0.0)));
        fluid.nonpressure_forces.push(Box::new(Akinci2013SurfaceTension::new(1.0, 10.0)));

        self.fluid_handle = Some(self.fluids_pipeline.liquid_world.add_fluid(fluid));

        self.spawn_fluid_render_objects();
    }

    fn clear_rigid_boxes(&mut self) {
        for item in self.rigid_boxes.drain(..) {
            let _ = self.renderer.scene_mut().remove_object(item.id);
            self.physics_colliders.remove(item.collider_handle, &mut self.physics_forces, &mut self.physics_bodies, false);
            self.physics_bodies.remove(item.body_handle, &mut self.physics_forces, &mut self.physics_colliders, &mut self.physics_impulse_joints, &mut self.physics_multibody_joint_set, true);
        }
    }

    fn make_fluid_block(nx: usize, ny: usize, nz: usize, spacing: f32, y_base: f32) -> Vec<Point<f32>> {
        let mut positions = Vec::new();
        let start_x = -((nx as f32 - 1.0) * spacing) / 2.0;
        let start_z = -((nz as f32 - 1.0) * spacing) / 2.0;
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    positions.push(Point::new(start_x + (i as f32) * spacing, y_base + (j as f32) * spacing, start_z + (k as f32) * spacing));
                }
            }
        }
        positions
    }

    fn spawn_fluid_render_objects(&mut self) {
        self.fluid_particle_ids.clear();
        if let Some(fluid_handle) = self.fluid_handle {
            let fluid = self.fluids_pipeline.liquid_world.fluids_mut().get_mut(fluid_handle).unwrap();
            for p in &fluid.positions {
                let transform = glam::Mat4::from_translation(glam::Vec3::new(p.x, p.y, p.z)) * glam::Mat4::from_scale(glam::Vec3::splat(PARTICLE_RADIUS));
                let object = insert_object(&mut self.renderer, self.fluid_mesh, self.fluid_material, transform, PARTICLE_RADIUS);
                if let Ok(id) = object { self.fluid_particle_ids.push(id); }
            }
        }
    }

    fn sync_fluid_render(&mut self) {
        if let Some(fluid_handle) = self.fluid_handle {
            let fluid = self.fluids_pipeline.liquid_world.fluids().get(fluid_handle).unwrap();
            let n = fluid.num_particles();

            while self.fluid_particle_ids.len() > n {
                if let Some(id) = self.fluid_particle_ids.pop() {
                    let _ = self.renderer.scene_mut().remove_object(id);
                }
            }

            while self.fluid_particle_ids.len() < n {
                let transform = glam::Mat4::from_translation(glam::Vec3::new(0.0, -9.0, 0.0)) * glam::Mat4::from_scale(glam::Vec3::splat(PARTICLE_RADIUS));
                if let Ok(id) = insert_object(&mut self.renderer, self.fluid_mesh, self.fluid_material, transform, PARTICLE_RADIUS) {
                    self.fluid_particle_ids.push(id);
                }
            }

            for (i, p) in fluid.positions.iter().enumerate() {
                let transform = glam::Mat4::from_translation(glam::Vec3::new(p.x, p.y, p.z)) * glam::Mat4::from_scale(glam::Vec3::splat(PARTICLE_RADIUS));
                let _ = self.renderer.scene_mut().update_object_transform(self.fluid_particle_ids[i], transform);
            }
        }
    }

    fn step_physics(&mut self, dt: f32) {
        self.physics_integration.dt = dt;
        let gravity = Vector3::new(0.0, -9.81, 0.0);
        PhysicsPipeline::new().step(
            &gravity,
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

        for item in &self.rigid_boxes {
            if let Some(body) = self.physics_bodies.get(item.body_handle) {
                let p = body.position();
                let pos = glam::Vec3::new(p.translation.x, p.translation.y, p.translation.z);
                let rot = glam::Quat::from_xyzw(p.rotation.i, p.rotation.j, p.rotation.k, p.rotation.w);
                let transform = glam::Mat4::from_translation(pos) * glam::Mat4::from_quat(rot);
                let _ = self.renderer.scene_mut().update_object_transform(item.id, transform);
            }
        }
    }

    fn spawn_fluid_emitter(&mut self, dt: f32) {
        self.fluid_spawn_timer += dt;
        if self.fluid_spawn_timer < 0.08 { return; }
        self.fluid_spawn_timer = 0.0;

        if let Some(fluid_handle) = self.fluid_handle {
            let fluid = self.fluids_pipeline.liquid_world.fluids_mut().get_mut(fluid_handle).unwrap();
            let new_particles = Self::make_fluid_block(5, 5, 6, PARTICLE_RADIUS * 1.8, 10.0);
            if fluid.num_particles() + new_particles.len() > MAX_PARTICLES { return; }
            let velocities = vec![Vector::new(0.0, -1.5, 0.0); new_particles.len()];
            fluid.add_particles(&new_particles, Some(&velocities));
        }
    }

    fn prune_fluid(&mut self) {
        if let Some(fluid_handle) = self.fluid_handle {
            let fluid = self.fluids_pipeline.liquid_world.fluids_mut().get_mut(fluid_handle).unwrap();
            for i in 0..fluid.num_particles() {
                if fluid.positions[i].y < -1.5 {
                    fluid.delete_particle_at_next_timestep(i);
                }
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

        if self.keys.contains(&PhysicalKey::Code(KeyCode::KeyW)) { self.cam_pos += forward * SPEED * dt; }
        if self.keys.contains(&PhysicalKey::Code(KeyCode::KeyS)) { self.cam_pos -= forward * SPEED * dt; }
        if self.keys.contains(&PhysicalKey::Code(KeyCode::KeyA)) { self.cam_pos -= right * SPEED * dt; }
        if self.keys.contains(&PhysicalKey::Code(KeyCode::KeyD)) { self.cam_pos += right * SPEED * dt; }
        if self.keys.contains(&PhysicalKey::Code(KeyCode::Space)) { self.cam_pos.y += SPEED * dt; }
        if self.keys.contains(&PhysicalKey::Code(KeyCode::ShiftLeft)) { self.cam_pos.y -= SPEED * dt; }

        let size = self.window.inner_size();
        let aspect = size.width as f32 / size.height.max(1) as f32;
        let camera = Camera::perspective_look_at(self.cam_pos, self.cam_pos + forward, glam::Vec3::Y, std::f32::consts::FRAC_PI_4, aspect, 0.1, 300.0);

        if self.physics_enabled { self.step_physics(dt); }

        self.spawn_fluid_emitter(dt);
        self.prune_fluid();
        self.fluids_pipeline.liquid_world.step(dt, &Vector3::new(0.0, -9.81, 0.0));
        self.sync_fluid_render();

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
            eprintln!("rapier_salva: frame {} rigid={} fluid={}", self.frame_count, self.rigid_boxes.len(), self.fluid_particle_ids.len());
        }
    }
}
