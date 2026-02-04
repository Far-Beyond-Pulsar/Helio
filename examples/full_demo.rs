use helio::prelude::*;
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};
use blade_graphics as gpu;

struct HelioApp {
    window: Option<Arc<Window>>,
    gpu_context: Option<Arc<gpu::Context>>,
    surface: Option<gpu::Surface>,
    render_context: Option<Arc<RenderContext>>,
    renderer: Option<Renderer>,
    scene: Option<Scene>,
    viewport: Option<Viewport>,
    particle_system: Option<helio::particles::ParticleSystem>,
    directional_lights: Vec<helio::lighting::DirectionalLight>,
    point_lights: Vec<helio::lighting::PointLight>,
    spot_lights: Vec<helio::lighting::SpotLight>,
    frame_count: u64,
    start_time: std::time::Instant,
}

impl HelioApp {
    fn new() -> Self {
        println!("===========================================");
        println!("  HELIO FULL DEMO - Production Rendering");
        println!("===========================================\n");

        Self {
            window: None,
            gpu_context: None,
            surface: None,
            render_context: None,
            renderer: None,
            scene: None,
            viewport: None,
            particle_system: None,
            directional_lights: Vec::new(),
            point_lights: Vec::new(),
            spot_lights: Vec::new(),
            frame_count: 0,
            start_time: std::time::Instant::now(),
        }
    }

    fn initialize_graphics(&mut self, window: Arc<Window>) {
        println!("🎮 Initializing GPU Context...");

        // Create GPU context
        let desc = gpu::ContextDesc {
            validation: false,
            capture: false,
            overlay: false,
            timing: false,
            presentation: true,
            device_id: 0,
        };

        let gpu_context = unsafe {
            gpu::Context::init(desc).expect("Failed to initialize GPU context")
        };
        println!("   ✓ Vulkan/DX12/Metal backend initialized");

        // Create surface
        let surface = gpu_context.create_surface(&*window).expect("Failed to create surface");
        println!("   ✓ Surface created for window\n");

        // Configure surface
        let size = window.inner_size();
        let surface_config = gpu::SurfaceConfig {
            size: gpu::Extent {
                width: size.width,
                height: size.height,
                depth: 1,
            },
            usage: gpu::TextureUsage::TARGET,
            display_sync: gpu::DisplaySync::Recent,
            color_space: gpu::ColorSpace::Linear,
            allow_exclusive_full_screen: false,
            transparent: false,
        };

        let mut surface_mut = surface;
        gpu_context.reconfigure_surface(&mut surface_mut, surface_config);
        println!("   ✓ Swapchain configured\n");

        let gpu_context = Arc::new(gpu_context);
        let render_context = Arc::new(RenderContext::new(Arc::clone(&gpu_context)));
        println!("   ✓ Render context created\n");

        // Create renderer
        println!("🎨 Configuring Renderer...");
        let config = RendererConfig {
            render_path: RenderPath::Deferred,
            enable_msaa: true,
            msaa_samples: 4,
            enable_taa: true,
            enable_hdr: true,
            enable_depth_prepass: true,
            enable_async_compute: true,
            shadow_resolution: 2048,
            max_lights: 1024,
        };
        println!("   ✓ Render Path: Deferred");
        println!("   ✓ MSAA: 4x");
        println!("   ✓ TAA: Enabled");
        println!("   ✓ HDR: Enabled");
        println!("   ✓ Shadow Resolution: 2048x2048\n");

        let mut renderer = Renderer::new(render_context.clone(), config);
        renderer.initialize(size.width, size.height).expect("Failed to initialize renderer");
        println!("   ✓ Renderer initialized at {}x{}\n", size.width, size.height);

        // Create scene
        println!("🌍 Building Scene...");
        let camera = Camera::new_perspective(
            std::f32::consts::FRAC_PI_3,
            size.width as f32 / size.height as f32,
            0.1,
            1000.0,
        );

        let mut scene = Scene::new(camera);
        println!("   ✓ Camera configured (FOV: 60°, Aspect: 16:9)");

        // Add entities
        println!("   📦 Creating geometry...");
        for i in 0..10 {
            let entity = helio::core::Entity::new(i);
            scene.add_entity(entity);
        }
        println!("      ✓ 10 mesh entities added\n");

        // Setup lighting
        println!("💡 Configuring Lighting System...");

        self.directional_lights.push(helio::lighting::DirectionalLight {
            direction: glam::Vec3::new(0.3, -0.7, 0.2).normalize(),
            color: glam::Vec3::new(1.0, 0.95, 0.9),
            intensity: 100000.0,
            cast_shadows: true,
            shadow_cascade_count: 4,
            shadow_distance: 100.0,
            shadow_bias: 0.0005,
        });
        println!("   ✓ Directional Light (Sun)");

        let light_colors = [
            glam::Vec3::new(1.0, 0.3, 0.2),
            glam::Vec3::new(0.2, 1.0, 0.3),
            glam::Vec3::new(0.2, 0.3, 1.0),
        ];

        for (i, color) in light_colors.iter().enumerate() {
            let angle = (i as f32 / light_colors.len() as f32) * std::f32::consts::TAU;
            self.point_lights.push(helio::lighting::PointLight {
                position: glam::Vec3::new(angle.cos() * 15.0, 3.0, angle.sin() * 15.0),
                color: *color,
                intensity: 2000.0,
                radius: 20.0,
                cast_shadows: true,
                shadow_resolution: 512,
            });
        }
        println!("   ✓ {} Point Lights\n", light_colors.len());

        println!("🎥 Starting Render Loop...");
        println!("   Press ESC or close window to exit\n");

        // Setup particle system
        let mut particle_system = helio::particles::ParticleSystem::new(100000);
        for i in 0..50 {
            let t = i as f32 / 50.0;
            particle_system.emit(
                glam::Vec3::new(0.0, 2.0, 0.0),
                glam::Vec3::new((t - 0.5) * 2.0, t * 3.0, (t - 0.5) * 2.0),
                5.0,
            );
        }

        self.window = Some(window);
        self.gpu_context = Some(gpu_context);
        self.surface = Some(surface_mut);
        self.render_context = Some(render_context);
        self.renderer = Some(renderer);
        self.scene = Some(scene);
        self.viewport = Some(Viewport::new(size.width, size.height));
        self.particle_system = Some(particle_system);
    }

    fn render_frame(&mut self) {
        if let (Some(surface), Some(renderer), Some(scene), Some(viewport), Some(particle_system), Some(render_context)) = (
            &mut self.surface,
            &mut self.renderer,
            &self.scene,
            &self.viewport,
            &mut self.particle_system,
            &self.render_context,
        ) {
            // Update particle system
            particle_system.update(1.0 / 60.0);

            // Create command encoder for GPU work
            let mut encoder = render_context.begin_frame();

            // Acquire frame AFTER beginning command encoder
            let frame = surface.acquire_frame();
            
            // Initialize the frame texture
            encoder.init_texture(frame.texture());

            // Render to the frame
            if let Err(e) = renderer.render(scene, viewport) {
                eprintln!("Render error: {:?}", e);
            }
            
            // Present the frame through the command encoder
            encoder.present(frame);

            // Submit all GPU commands
            render_context.submit(encoder);

            self.frame_count += 1;

            // Print progress
            if self.frame_count % 60 == 0 {
                let elapsed = self.start_time.elapsed().as_secs_f32();
                let fps = self.frame_count as f32 / elapsed;
                println!("   Frame {}: {:.1} FPS | {} entities, {} lights, {} particles",
                    self.frame_count,
                    fps,
                    scene.entity_count(),
                    self.directional_lights.len() + self.point_lights.len() + self.spot_lights.len(),
                    particle_system.particles.len()
                );
            }
        }
    }
}

impl ApplicationHandler for HelioApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            println!("🪟 Creating Window...");

            let window_attributes = Window::default_attributes()
                .with_title("Helio Full Demo - Production Rendering")
                .with_inner_size(winit::dpi::PhysicalSize::new(1920, 1080));

            let window = Arc::new(
                event_loop
                    .create_window(window_attributes)
                    .expect("Failed to create window")
            );

            println!("   ✓ Window created (1920x1080)\n");

            self.initialize_graphics(window);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                println!("\n📊 Final Statistics:");
                println!("   • Total Frames: {}", self.frame_count);
                println!("   • Runtime: {:.2}s", self.start_time.elapsed().as_secs_f32());
                println!("   • Average FPS: {:.1}",
                    self.frame_count as f32 / self.start_time.elapsed().as_secs_f32());
                println!("\n✅ Demo Complete!");
                println!("===========================================\n");
                event_loop.exit();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.physical_key == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Escape) {
                    println!("\n📊 Final Statistics:");
                    println!("   • Total Frames: {}", self.frame_count);
                    println!("   • Runtime: {:.2}s", self.start_time.elapsed().as_secs_f32());
                    println!("   • Average FPS: {:.1}",
                        self.frame_count as f32 / self.start_time.elapsed().as_secs_f32());
                    println!("\n✅ Demo Complete!");
                    println!("===========================================\n");
                    event_loop.exit();
                }
            }
            WindowEvent::RedrawRequested => {
                self.render_frame();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // Request redraw with controlled frame rate
        if let Some(window) = &self.window {
            if self.surface.is_some() {
                // Small delay to avoid overwhelming the GPU with frame requests
                let next_frame_time = Instant::now() + Duration::from_millis(16);
                event_loop.set_control_flow(ControlFlow::WaitUntil(next_frame_time));
                window.request_redraw();
            }
        }
    }
}

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().expect("Failed to create event loop");

    let mut app = HelioApp::new();
    event_loop.run_app(&mut app).expect("Failed to run event loop");
}
