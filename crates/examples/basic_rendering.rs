use helio_core::{
    gpu, Camera, Scene, Transform, Entity, Vertex, create_cube_mesh, create_sphere_mesh, create_plane_mesh,
};
use helio_render::{Renderer, RendererConfig, RenderPath};
use helio_lighting::{LightingSystem, DirectionalLight, PointLight};
use glam::{Vec3, Vec4, Quat};
use std::sync::Arc;
use winit::{
    event::{Event, WindowEvent, KeyEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{PhysicalKey, KeyCode},
};

struct Application {
    window: Arc<winit::window::Window>,
    context: Arc<gpu::Context>,
    surface: gpu::Surface,
    renderer: Renderer,
    scene: Scene,
    lighting: LightingSystem,
    last_frame_time: std::time::Instant,
    camera_rotation: f32,
    camera_distance: f32,
    frame_count: u64,
    fps_timer: std::time::Instant,
    frame_times: Vec<f32>,
}

impl Application {
    fn new(event_loop: &EventLoop<()>) -> Self {
        log::info!("Initializing Helio Rendering Engine Example");

        let window_attr = winit::window::Window::default_attributes()
            .with_title("Helio - Basic Rendering Example")
            .with_inner_size(winit::dpi::LogicalSize::new(1920, 1080));
        
        let window = Arc::new(event_loop.create_window(window_attr).unwrap());

        let context = Arc::new(unsafe {
            gpu::Context::init(gpu::ContextDesc {
                presentation: true,
                validation: cfg!(debug_assertions),
                timing: true,
                capture: false,
                overlay: false,
                device_id: 0,
            })
            .unwrap()
        });

        log::info!("GPU Context initialized");
        let caps = context.capabilities();
        log::info!("GPU Capabilities: ray_query={:?}, sample_count_mask={}",
            caps.ray_query, caps.sample_count_mask);

        let size = window.inner_size();
        let surface_config = gpu::SurfaceConfig {
            size: gpu::Extent {
                width: size.width,
                height: size.height,
                depth: 1,
            },
            usage: gpu::TextureUsage::TARGET,
            display_sync: gpu::DisplaySync::Block,
            ..Default::default()
        };

        let surface = context
            .create_surface_configured(&*window, surface_config)
            .unwrap();

        log::info!("Surface created: {:?}", surface.info());

        let config = RendererConfig {
            render_path: RenderPath::Deferred,
            width: size.width,
            height: size.height,
            hdr_enabled: true,
            msaa_samples: 1,
            vsync_enabled: true,
            max_lights_per_tile: 256,
            tile_size: 16,
        };

        let renderer = Renderer::new(context.clone(), config);
        log::info!("Renderer initialized with {:?} render path", RenderPath::Deferred);

        let mut camera = Camera::new_perspective(
            std::f32::consts::FRAC_PI_3,
            size.width as f32 / size.height as f32,
            0.1,
            1000.0,
        );
        camera.position = Vec3::new(0.0, 5.0, 10.0);
        camera.look_at(Vec3::ZERO, Vec3::Y);

        let mut scene = Scene::new(camera);
        log::info!("Scene created");

        let mut cube_mesh = create_cube_mesh(2.0);
        cube_mesh.upload_to_gpu(&context);
        log::info!("Cube mesh created with {} vertices", cube_mesh.vertices.len());

        let mut sphere_mesh = create_sphere_mesh(1.5, 32, 32);
        sphere_mesh.upload_to_gpu(&context);
        log::info!("Sphere mesh created with {} vertices", sphere_mesh.vertices.len());

        let mut plane_mesh = create_plane_mesh(20.0, 20.0);
        plane_mesh.upload_to_gpu(&context);
        log::info!("Plane mesh created with {} vertices", plane_mesh.vertices.len());

        let cube_entity_id = scene.create_entity();
        if let Some(entity) = scene.get_entity_mut(cube_entity_id) {
            entity.transform = Transform::from_position(Vec3::new(-3.0, 1.0, 0.0));
            entity.visible = true;
            entity.cast_shadows = true;
            entity.receive_shadows = true;
        }
        log::info!("Added cube entity");

        let sphere_entity_id = scene.create_entity();
        if let Some(entity) = scene.get_entity_mut(sphere_entity_id) {
            entity.transform = Transform::from_position(Vec3::new(3.0, 1.5, 0.0));
            entity.visible = true;
            entity.cast_shadows = true;
            entity.receive_shadows = true;
        }
        log::info!("Added sphere entity");

        let plane_entity_id = scene.create_entity();
        if let Some(entity) = scene.get_entity_mut(plane_entity_id) {
            entity.transform = Transform::from_position(Vec3::new(0.0, 0.0, 0.0));
            entity.visible = true;
            entity.cast_shadows = false;
            entity.receive_shadows = true;
        }
        log::info!("Added ground plane entity");

        let mut lighting = LightingSystem::new(&context);

        let sun = DirectionalLight::new(
            Vec3::new(-0.5, -1.0, -0.3),
            Vec3::new(1.0, 0.95, 0.9),
            2.0,
        );
        lighting.add_directional_light(sun);
        log::info!("Added directional light (sun)");

        let point_light1 = PointLight::new(
            Vec3::new(-5.0, 3.0, -5.0),
            Vec3::new(1.0, 0.3, 0.3),
            10.0,
            15.0,
        );
        lighting.add_point_light(point_light1);

        let point_light2 = PointLight::new(
            Vec3::new(5.0, 3.0, 5.0),
            Vec3::new(0.3, 0.3, 1.0),
            10.0,
            15.0,
        );
        lighting.add_point_light(point_light2);
        log::info!("Added {} point lights", 2);

        lighting.update_gpu_data(&context);

        log::info!("Scene setup complete!");
        log::info!("Total entities: {}", scene.entities.len());
        log::info!("Directional lights: {}", lighting.directional_lights.len());
        log::info!("Point lights: {}", lighting.point_lights.len());

        Self {
            window,
            context,
            surface,
            renderer,
            scene,
            lighting,
            last_frame_time: std::time::Instant::now(),
            camera_rotation: 0.0,
            camera_distance: 10.0,
            frame_count: 0,
            fps_timer: std::time::Instant::now(),
            frame_times: Vec::with_capacity(10000),
        }
    }

    fn update(&mut self) {
        let now = std::time::Instant::now();
        let delta_time = (now - self.last_frame_time).as_secs_f32();
        self.last_frame_time = now;

        self.frame_times.push(delta_time);
        self.frame_count += 1;

        if self.frame_count % 10000 == 0 {
            let elapsed = self.fps_timer.elapsed().as_secs_f32();
            let avg_fps = 10000.0 / elapsed;
            let avg_frame_time = elapsed / 10000.0 * 1000.0;
            
            let min_frame_time = self.frame_times.iter().copied().fold(f32::INFINITY, f32::min) * 1000.0;
            let max_frame_time = self.frame_times.iter().copied().fold(0.0f32, f32::max) * 1000.0;
            
            println!("Frame {}: avg {:.2} FPS ({:.2}ms), min {:.2}ms, max {:.2}ms", 
                self.frame_count, avg_fps, avg_frame_time, min_frame_time, max_frame_time);
            
            self.fps_timer = std::time::Instant::now();
            self.frame_times.clear();
        }

        self.camera_rotation += delta_time * 0.3;

        let x = self.camera_distance * self.camera_rotation.cos();
        let z = self.camera_distance * self.camera_rotation.sin();
        self.scene.camera.position = Vec3::new(x, 5.0, z);
        self.scene.camera.look_at(Vec3::ZERO, Vec3::Y);

        for (i, entity) in self.scene.entities.values_mut().enumerate() {
            if i == 0 {
                let angle = now.elapsed().as_secs_f32() * 2.0;
                entity.transform.rotation = Quat::from_rotation_y(angle);
            } else if i == 1 {
                let angle = now.elapsed().as_secs_f32() * -1.5;
                entity.transform.rotation = Quat::from_rotation_y(angle) * Quat::from_rotation_x(angle);
            }
        }
    }

    fn render(&mut self) {
        let frame = self.surface.acquire_frame();
        let target_view = frame.texture_view();

        // Clear the screen to a visible color
        let mut encoder = self.context.create_command_encoder(gpu::CommandEncoderDesc {
            name: "render",
            buffer_count: 1,
        });
        encoder.start();
        
        // Clear to blue so we can see something
        let mut pass = encoder.render(
            "clear",
            gpu::RenderTargetSet {
                colors: &[gpu::RenderTarget {
                    view: target_view,
                    init_op: gpu::InitOp::Clear(gpu::TextureColor::OpaqueBlack),
                    finish_op: gpu::FinishOp::Store,
                }],
                depth_stencil: None,
            },
        );
        drop(pass);

        // TODO: Actual rendering will go here
        // self.renderer.render(&self.scene, &mut self.lighting, target_view);
        
        encoder.present(frame);
        let _sync = self.context.submit(&mut encoder);
        self.context.destroy_command_encoder(&mut encoder);
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }

        log::info!("Resizing to {}x{}", width, height);

        let config = gpu::SurfaceConfig {
            size: gpu::Extent {
                width,
                height,
                depth: 1,
            },
            usage: gpu::TextureUsage::TARGET,
            display_sync: gpu::DisplaySync::Block,
            ..Default::default()
        };

        self.context.reconfigure_surface(&mut self.surface, config);
        self.renderer.resize(width, height);
        self.scene.camera.set_aspect_ratio(width as f32 / height as f32);
    }

    fn cleanup(&mut self) {
        log::info!("Cleaning up resources...");
        self.renderer.cleanup();
        self.lighting.cleanup(&self.context);
        self.context.destroy_surface(&mut self.surface);
        log::info!("Cleanup complete");
    }
}

fn main() {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();
    
    println!("=== Helio Rendering Engine ===");
    println!("Initializing...");
    
    let event_loop = EventLoop::new().unwrap();
    let mut app = Application::new(&event_loop);

    println!("Starting event loop...");
    println!("Performance logging every 10,000 frames. Press ESC to exit.");
    println!("You should see a blue screen (rendering system is a stub for now).");

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    log::info!("Close requested");
                    app.cleanup();
                    elwt.exit();
                }
                WindowEvent::Resized(size) => {
                    app.resize(size.width, size.height);
                }
                WindowEvent::KeyboardInput {
                    event: KeyEvent {
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        ..
                    },
                    ..
                } => {
                    log::info!("Escape pressed");
                    app.cleanup();
                    elwt.exit();
                }
                WindowEvent::RedrawRequested => {
                    app.update();
                    app.render();
                    app.window.request_redraw();
                }
                _ => {}
            },
            Event::AboutToWait => {
                app.window.request_redraw();
            }
            _ => {}
        }
    }).unwrap();
}
