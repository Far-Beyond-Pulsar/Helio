//! Salva3d Fluid Demo — helio v3
//!
//! An angled faucet emits SPH water particles into a container box.
//! Particles fall, pool on the floor, and collide with boundary particles.
//!
//! Note: This uses salva3d for fluid simulation with manual boundaries.
//! We avoid rapier integration due to version conflicts (salva3d 0.9 uses rapier 0.18).
//!
//! Controls:
//!   C — toggle physics
//!   R — reset scene
//!   = (Plus) — start tap for 5 seconds
//!   WASD / Space / Shift — fly camera
//!   Escape — release cursor / exit

mod v3_demo_common;
use v3_demo_common::{box_mesh, insert_object, make_material, point_light, sphere_mesh};

use helio::{required_wgpu_features, required_wgpu_limits, Camera, MaterialId, MeshId, ObjectId, Renderer, RendererConfig};
use salva3d::{
    kernel::CubicSplineKernel,
    object::{Boundary, Fluid},
    solver::{Akinci2013SurfaceTension, XSPHViscosity, DFSPHSolver},
    LiquidWorld,
};
use std::collections::HashSet;
use std::sync::Arc;
use winit::{application::ApplicationHandler, event::*, event_loop::{ActiveEventLoop, EventLoop}, keyboard::{KeyCode, PhysicalKey}, window::{CursorGrabMode, Window, WindowId}};

struct FluidParticle {
    id: ObjectId,
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

    physics_enabled: bool,

    liquid_world: LiquidWorld,
    particle_radius: f32,
    fluid_density: f32,

    fluid_particles: Vec<FluidParticle>,
    water_material: MaterialId,
    sphere_mesh: MeshId,

    spawn_timer: f32,
    spawn_interval: f32,
    max_particles: usize,

    tap_active: bool,
    tap_timer: f32,
    tap_duration: f32,
}

struct App { state: Option<AppState> }

fn main() {
    env_logger::init();
    log::info!("Starting Salva3d Fluid Demo");
    EventLoop::new().expect("event loop").run_app(&mut App { state: None }).expect("run");
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() { return; }

        let window = Arc::new(event_loop.create_window(Window::default_attributes().with_title("Helio — Faucet Fill Box (Salva3d)").with_inner_size(winit::dpi::LogicalSize::new(1600u32, 900u32))).expect("window"));
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
        renderer.set_ambient([0.08, 0.08, 0.1], 1.0);

        let _ = renderer.scene_mut().insert_actor(helio::SceneActor::light(point_light([15.0, 25.0, 15.0], [0.9, 0.9, 0.85], 20.0, 80.0))).as_light().unwrap();
        let _ = renderer.scene_mut().insert_actor(helio::SceneActor::light(point_light([-15.0, 20.0, -10.0], [0.7, 0.8, 1.0], 15.0, 70.0))).as_light().unwrap();

        let floor_mat = renderer.scene_mut().insert_material(make_material([0.3, 0.3, 0.35, 1.0], 0.8, 0.05, [0.0, 0.0, 0.0], 0.0));
        let wall_mat = renderer.scene_mut().insert_material(make_material([0.35, 0.3, 0.28, 1.0], 0.7, 0.02, [0.0, 0.0, 0.0], 0.0));
        let water_material = renderer.scene_mut().insert_material(make_material([0.15, 0.4, 0.75, 0.85], 0.1, 0.0, [0.0, 0.0, 0.0], 0.0));
        let faucet_mat = renderer.scene_mut().insert_material(make_material([0.6, 0.6, 0.65, 1.0], 0.3, 0.6, [0.0, 0.0, 0.0], 0.0));

        let sphere_mesh = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(sphere_mesh([0.0, 0.0, 0.0], 1.0))).as_mesh().unwrap();

        let particle_radius = 0.2;
        let fluid_density = 1000.0;
        let smoothing_factor = 2.0;

        // Create liquid world with DFSPH solver
        let mut liquid_world = LiquidWorld::new(DFSPHSolver::<CubicSplineKernel>::new(), particle_radius, smoothing_factor);

        // Create boundary particles for container walls and floor
        let mut boundary_positions = Vec::new();
        let container_size = 20.0;
        let wall_height = 6.0;
        let spacing = particle_radius * 2.0;

        // Floor boundary particles (y = 0)
        for x in (-100..=100).map(|i| i as f32 * spacing) {
            for z in (-100..=100).map(|i| i as f32 * spacing) {
                if x.abs() <= container_size && z.abs() <= container_size {
                    boundary_positions.push(salva3d::math::Point::new(x, 0.0, z));
                }
            }
        }

        // Wall boundary particles
        for y in (0..=30).map(|i| i as f32 * spacing) {
            if y > wall_height { break; }

            // Left wall (x = -20)
            for z in (-100..=100).map(|i| i as f32 * spacing) {
                if z.abs() <= container_size {
                    boundary_positions.push(salva3d::math::Point::new(-container_size, y, z));
                }
            }

            // Right wall (x = 20)
            for z in (-100..=100).map(|i| i as f32 * spacing) {
                if z.abs() <= container_size {
                    boundary_positions.push(salva3d::math::Point::new(container_size, y, z));
                }
            }

            // Front wall (z = -20)
            for x in (-100..=100).map(|i| i as f32 * spacing) {
                if x.abs() <= container_size {
                    boundary_positions.push(salva3d::math::Point::new(x, y, -container_size));
                }
            }

            // Back wall (z = 20)
            for x in (-100..=100).map(|i| i as f32 * spacing) {
                if x.abs() <= container_size {
                    boundary_positions.push(salva3d::math::Point::new(x, y, container_size));
                }
            }
        }

        let boundary = Boundary::new(boundary_positions);
        liquid_world.add_boundary(boundary);

        // Render container floor
        let floor_mesh = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [20.0, 0.5, 20.0]))).as_mesh().unwrap();
        let _ = insert_object(&mut renderer, floor_mesh, floor_mat, glam::Mat4::from_translation(glam::Vec3::new(0.0, -0.5, 0.0)), 20.0);

        // Render container walls
        let wall_thickness = 0.5;

        // Right wall
        let wall_mesh = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [wall_thickness * 0.5, wall_height * 0.5, container_size]))).as_mesh().unwrap();
        let _ = insert_object(&mut renderer, wall_mesh, wall_mat, glam::Mat4::from_translation(glam::Vec3::new(container_size + wall_thickness * 0.5, wall_height * 0.5, 0.0)), 10.0);

        // Left wall
        let wall_mesh = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [wall_thickness * 0.5, wall_height * 0.5, container_size]))).as_mesh().unwrap();
        let _ = insert_object(&mut renderer, wall_mesh, wall_mat, glam::Mat4::from_translation(glam::Vec3::new(-container_size - wall_thickness * 0.5, wall_height * 0.5, 0.0)), 10.0);

        // Back wall
        let wall_mesh = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [container_size, wall_height * 0.5, wall_thickness * 0.5]))).as_mesh().unwrap();
        let _ = insert_object(&mut renderer, wall_mesh, wall_mat, glam::Mat4::from_translation(glam::Vec3::new(0.0, wall_height * 0.5, container_size + wall_thickness * 0.5)), 10.0);

        // Front wall
        let wall_mesh = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [container_size, wall_height * 0.5, wall_thickness * 0.5]))).as_mesh().unwrap();
        let _ = insert_object(&mut renderer, wall_mesh, wall_mat, glam::Mat4::from_translation(glam::Vec3::new(0.0, wall_height * 0.5, -container_size - wall_thickness * 0.5)), 10.0);

        // Render simple faucet indicator
        let faucet_mesh = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [0.5, 0.5, 0.5]))).as_mesh().unwrap();
        let _ = insert_object(&mut renderer, faucet_mesh, faucet_mat, glam::Mat4::from_translation(glam::Vec3::new(0.0, 10.0, 0.0)), 1.0);

        let state = AppState {
            window,
            surface,
            device,
            surface_format: fmt,
            renderer,
            last_frame: std::time::Instant::now(),
            frame_count: 0,
            cam_pos: glam::Vec3::new(-15.0, 12.0, 25.0),
            cam_yaw: 0.5,
            cam_pitch: -0.3,
            keys: HashSet::new(),
            cursor_grabbed: false,
            mouse_delta: (0.0, 0.0),
            physics_enabled: true,
            liquid_world,
            particle_radius,
            fluid_density,
            fluid_particles: Vec::new(),
            water_material,
            sphere_mesh,
            spawn_timer: 0.0,
            spawn_interval: 0.04,
            max_particles: 2000,
            tap_active: false,
            tap_timer: 0.0,
            tap_duration: 5.0,
        };

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
                eprintln!("physics enabled={}", state.physics_enabled);
            }
            WindowEvent::KeyboardInput { event: KeyEvent { state: ElementState::Pressed, physical_key: PhysicalKey::Code(KeyCode::KeyR), ..}, .. } => {
                state.reset_scene();
            }
            WindowEvent::KeyboardInput { event: KeyEvent { state: ElementState::Pressed, physical_key: PhysicalKey::Code(KeyCode::Equal), ..}, .. } => {
                state.tap_active = true;
                state.tap_timer = 0.0;
                state.spawn_timer = 0.0; // Reset spawn timer so particles start immediately
                eprintln!("Tap activated for {} seconds (particles: {})", state.tap_duration, state.fluid_particles.len());
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
                let dt = (now - state.last_frame).as_secs_f32().min(0.033);
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
            s.window.request_redraw();
        }
    }
}

impl AppState {
    fn reset_scene(&mut self) {
        eprintln!("Resetting scene...");

        // Clear all fluid particles
        for particle in self.fluid_particles.drain(..) {
            let _ = self.renderer.scene_mut().remove_object(particle.id);
        }

        // Reset liquid world
        self.liquid_world = LiquidWorld::new(DFSPHSolver::<CubicSplineKernel>::new(), self.particle_radius, 2.0);

        // Recreate boundaries
        let mut boundary_positions = Vec::new();
        let container_size = 20.0;
        let wall_height = 6.0;
        let spacing = self.particle_radius * 2.0;

        // Floor
        for x in (-100..=100).map(|i| i as f32 * spacing) {
            for z in (-100..=100).map(|i| i as f32 * spacing) {
                if x.abs() <= container_size && z.abs() <= container_size {
                    boundary_positions.push(salva3d::math::Point::new(x, 0.0, z));
                }
            }
        }

        // Walls
        for y in (0..=30).map(|i| i as f32 * spacing) {
            if y > wall_height { break; }
            for z in (-100..=100).map(|i| i as f32 * spacing) {
                if z.abs() <= container_size {
                    boundary_positions.push(salva3d::math::Point::new(-container_size, y, z));
                    boundary_positions.push(salva3d::math::Point::new(container_size, y, z));
                }
            }
            for x in (-100..=100).map(|i| i as f32 * spacing) {
                if x.abs() <= container_size {
                    boundary_positions.push(salva3d::math::Point::new(x, y, -container_size));
                    boundary_positions.push(salva3d::math::Point::new(x, y, container_size));
                }
            }
        }

        let boundary = Boundary::new(boundary_positions);
        self.liquid_world.add_boundary(boundary);

        self.spawn_timer = 0.0;
    }

    fn spawn_fluid_parcel(&mut self) {
        if self.fluid_particles.len() >= self.max_particles {
            if self.tap_active {
                eprintln!("Max particles ({}) reached! Press R to reset.", self.max_particles);
                self.tap_active = false; // Stop trying to spawn
            }
            return;
        }

        // Spawn position at the faucet (center, above container)
        let spawn_pos = glam::Vec3::new(0.0, 9.0, 0.0);

        // Create a tight vertical stream (2x2 horizontal, 1 vertical)
        let horizontal_size = 2;
        let spacing = self.particle_radius * 1.8;

        let mut positions = Vec::new();
        let mut velocities = Vec::new();

        // Downward initial velocity for stream effect
        let initial_velocity = salva3d::math::Vector::new(0.0, -5.0, 0.0);

        for x in 0..horizontal_size {
            for z in 0..horizontal_size {
                if self.fluid_particles.len() + positions.len() >= self.max_particles {
                    break;
                }

                let offset = glam::Vec3::new(
                    (x as f32 - horizontal_size as f32 * 0.5 + 0.5) * spacing,
                    0.0,
                    (z as f32 - horizontal_size as f32 * 0.5 + 0.5) * spacing,
                );
                let pos = spawn_pos + offset;
                positions.push(salva3d::math::Point::new(pos.x, pos.y, pos.z));
                velocities.push(initial_velocity);
            }
        }

        if positions.is_empty() {
            return;
        }

        let particle_count = positions.len();

        // Create fluid and add to liquid world
        let mut fluid = Fluid::new(positions, self.particle_radius, self.fluid_density);
        fluid.nonpressure_forces.push(Box::new(XSPHViscosity::new(0.5, 0.0)));
        fluid.nonpressure_forces.push(Box::new(Akinci2013SurfaceTension::new(0.5, 0.0)));

        let fluid_handle = self.liquid_world.add_fluid(fluid);

        // Set initial velocities
        if let Some(fluid_obj) = self.liquid_world.fluids_mut().get_mut(fluid_handle) {
            for (i, vel) in velocities.iter().enumerate() {
                if i < fluid_obj.velocities.len() {
                    fluid_obj.velocities[i] = *vel;
                }
            }
        }

        // Create visual representation for each particle
        for _ in 0..particle_count {
            let transform = glam::Mat4::from_scale(glam::Vec3::splat(self.particle_radius));
            let obj = insert_object(&mut self.renderer, self.sphere_mesh, self.water_material, transform, self.particle_radius).expect("insert particle");
            self.fluid_particles.push(FluidParticle { id: obj });
        }
    }

    fn step_physics(&mut self, dt: f32) {
        let gravity = salva3d::math::Vector::new(0.0, -9.81, 0.0);
        self.liquid_world.step(dt, &gravity);
    }

    fn sync_fluid_particles(&mut self) {
        let mut particle_idx = 0;

        for (_, fluid) in self.liquid_world.fluids().iter() {
            for pos in fluid.positions.iter() {
                if particle_idx >= self.fluid_particles.len() {
                    return;
                }

                let transform = glam::Mat4::from_translation(glam::Vec3::new(pos.x, pos.y, pos.z))
                    * glam::Mat4::from_scale(glam::Vec3::splat(self.particle_radius));
                let _ = self.renderer.scene_mut().update_object_transform(
                    self.fluid_particles[particle_idx].id,
                    transform
                );

                particle_idx += 1;
            }
        }
    }

    fn render(&mut self, dt: f32) {
        const SPEED: f32 = 15.0;
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

        if self.physics_enabled {
            // Update tap timer
            if self.tap_active {
                self.tap_timer += dt;
                if self.tap_timer >= self.tap_duration {
                    self.tap_active = false;
                    self.tap_timer = 0.0;
                    eprintln!("Tap deactivated (total particles: {})", self.fluid_particles.len());
                }
            }

            // Spawn fluid parcels periodically when tap is active
            if self.tap_active {
                self.spawn_timer += dt;
                if self.spawn_timer >= self.spawn_interval {
                    self.spawn_timer = 0.0;
                    self.spawn_fluid_parcel();

                    // Show progress every 1 second
                    if (self.tap_timer * 2.0) as i32 != ((self.tap_timer - dt) * 2.0) as i32 {
                        let remaining = self.tap_duration - self.tap_timer;
                        if remaining > 0.0 {
                            eprintln!("  Tap running: {:.1}s remaining, {} particles", remaining, self.fluid_particles.len());
                        }
                    }
                }
            }

            self.step_physics(dt);
            self.sync_fluid_particles();
        }

        let size = self.window.inner_size();
        let aspect = size.width as f32 / size.height.max(1) as f32;
        let camera = Camera::perspective_look_at(self.cam_pos, self.cam_pos + forward, glam::Vec3::Y, std::f32::consts::FRAC_PI_3, aspect, 0.1, 300.0);

        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(e) => { log::warn!("Surface error: {:?}", e); return; }
        };
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        if let Err(e) = self.renderer.render(&camera, &view) {
            log::error!("Render error: {:?}", e);
        }
        output.present();

        self.frame_count += 1;

        if self.frame_count % 120 == 0 {
            eprintln!("Frame {} | Particles: {} | Physics: {}",
                self.frame_count,
                self.fluid_particles.len(),
                self.physics_enabled
            );
        }
    }
}
