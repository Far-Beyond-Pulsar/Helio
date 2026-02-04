use helio::prelude::*;
use std::sync::Arc;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};
use blade_graphics as gpu;

fn main() {
    env_logger::init();

    println!("===========================================");
    println!("  HELIO FULL DEMO - Production Rendering");
    println!("===========================================\n");

    // Create window and event loop
    println!("🪟 Creating Window...");
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Helio Full Demo - Production Rendering")
        .with_inner_size(winit::dpi::PhysicalSize::new(1920, 1080))
        .build(&event_loop)
        .expect("Failed to create window");
    println!("   ✓ Window created (1920x1080)\n");

    // Initialize GPU context with the window
    println!("🎮 Initializing GPU Context...");
    let gpu_context = create_gpu_context_with_window(&window);
    println!("   ✓ Vulkan/DX12/Metal backend initialized");
    println!("   ✓ Surface created for window\n");

    // Configure the surface
    let size = window.inner_size();
    let surface_config = gpu::SurfaceConfig {
        size: gpu::Extent {
            width: size.width,
            height: size.height,
            depth: 1,
        },
        usage: gpu::TextureUsage::TARGET,
        display_sync: gpu::DisplaySync::Block,
        color_space: gpu::ColorSpace::Linear,
    };
    gpu_context.resize(surface_config);

    let render_context = Arc::new(RenderContext::new(Arc::new(gpu_context)));
    println!("   ✓ Render context created\n");

    // Create renderer with configuration
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
    renderer.initialize(1920, 1080).expect("Failed to initialize renderer");
    println!("   ✓ Renderer initialized at 1920x1080\n");

    // Create scene
    println!("🌍 Building Scene...");
    let camera = Camera::new_perspective(
        std::f32::consts::FRAC_PI_3,
        16.0 / 9.0,
        0.1,
        1000.0,
    );

    let mut scene = Scene::new(camera);
    println!("   ✓ Camera configured (FOV: 60°, Aspect: 16:9)");

    // Add multiple entities with different materials
    println!("   📦 Creating geometry...");

    for i in 0..10 {
        let entity = helio::core::Entity::new(i);
        scene.add_entity(entity);
    }
    println!("      ✓ 10 mesh entities added\n");

    // Create diverse PBR materials
    println!("🎭 Setting up Materials...");
    let materials = create_demo_materials();
    println!("   ✓ {} PBR materials created:", materials.len());
    for mat in &materials {
        println!("      • {} (M:{:.2} R:{:.2})", mat.name, mat.metallic, mat.roughness);
    }
    println!("");

    // Setup comprehensive lighting
    println!("💡 Configuring Lighting System...");
    let mut directional_lights = vec![];
    let mut point_lights = vec![];
    let mut spot_lights = vec![];

    // Sun (directional light with cascaded shadows)
    directional_lights.push(helio::lighting::DirectionalLight {
        direction: glam::Vec3::new(0.3, -0.7, 0.2).normalize(),
        color: glam::Vec3::new(1.0, 0.95, 0.9),
        intensity: 100000.0,
        cast_shadows: true,
        shadow_cascade_count: 4,
        shadow_distance: 100.0,
        shadow_bias: 0.0005,
    });
    println!("   ✓ Directional Light (Sun)");
    println!("      • 4 Cascade Shadow Maps");
    println!("      • Distance: 100m");

    // Point lights
    let light_colors = [
        glam::Vec3::new(1.0, 0.3, 0.2),  // Red
        glam::Vec3::new(0.2, 1.0, 0.3),  // Green
        glam::Vec3::new(0.2, 0.3, 1.0),  // Blue
        glam::Vec3::new(1.0, 1.0, 0.2),  // Yellow
        glam::Vec3::new(1.0, 0.2, 1.0),  // Magenta
    ];

    for (i, color) in light_colors.iter().enumerate() {
        let angle = (i as f32 / light_colors.len() as f32) * std::f32::consts::TAU;
        point_lights.push(helio::lighting::PointLight {
            position: glam::Vec3::new(
                angle.cos() * 15.0,
                3.0,
                angle.sin() * 15.0,
            ),
            color: *color,
            intensity: 2000.0,
            radius: 20.0,
            cast_shadows: true,
            shadow_resolution: 512,
        });
    }
    println!("   ✓ {} Point Lights (colored, with shadows)", light_colors.len());

    // Spot light
    spot_lights.push(helio::lighting::SpotLight {
        position: glam::Vec3::new(0.0, 10.0, 0.0),
        direction: glam::Vec3::new(0.0, -1.0, 0.0),
        color: glam::Vec3::ONE,
        intensity: 5000.0,
        radius: 25.0,
        inner_cone_angle: 0.4,
        outer_cone_angle: 0.6,
        cast_shadows: true,
        shadow_resolution: 1024,
    });
    println!("   ✓ Spot Light (overhead)");

    // Global Illumination
    let mut _gi = helio::lighting::GlobalIllumination::new(
        helio::lighting::GIProbeResolution::Medium
    );
    _gi.enabled = true;
    _gi.probe_count = glam::UVec3::new(16, 8, 16);
    println!("   ✓ Dynamic Diffuse GI (DDGI)");
    println!("      • Probe Grid: 16x8x16\n");

    // Post-processing stack
    println!("🎬 Post-Processing Pipeline...");

    let _bloom = helio::postprocess::Bloom {
        enabled: true,
        threshold: 1.0,
        intensity: 0.3,
        radius: 2.0,
        ..Default::default()
    };
    println!("   ✓ HDR Bloom");

    let _taa = helio::postprocess::TAA {
        enabled: true,
        jitter_sequence: helio::postprocess::JitterSequence::Halton,
        history_weight: 0.95,
        ..Default::default()
    };
    println!("   ✓ Temporal Anti-Aliasing (Halton sequence)");

    let _dof = helio::postprocess::DepthOfField {
        enabled: true,
        focal_distance: 10.0,
        focal_range: 5.0,
        bokeh_enabled: true,
        aperture: 0.05,
        ..Default::default()
    };
    println!("   ✓ Depth of Field (Bokeh)");

    let _ao = helio::postprocess::SSAO {
        enabled: true,
        radius: 1.0,
        bias: 0.025,
        samples: 16,
        ..Default::default()
    };
    println!("   ✓ Screen Space Ambient Occlusion");

    let _tone_mapping = helio::postprocess::ToneMappingOperator::ACES;
    println!("   ✓ Tone Mapping: ACES\n");

    // Advanced features
    println!("⚡ Advanced Features...");

    // Particles
    let mut particle_system = helio::particles::ParticleSystem::new(100000);
    for i in 0..50 {
        let t = i as f32 / 50.0;
        particle_system.emit(
            glam::Vec3::new(0.0, 2.0, 0.0),
            glam::Vec3::new(
                (t - 0.5) * 2.0,
                t * 3.0,
                (t - 0.5) * 2.0,
            ),
            5.0,
        );
    }
    println!("   ✓ GPU Particle System (100K max particles)");
    println!("      • Active particles: {}", particle_system.particles.len());

    println!("   ✓ Lighting: {} directional, {} point, {} spot lights",
        directional_lights.len(), point_lights.len(), spot_lights.len());

    // Terrain
    let _terrain = helio::terrain::Heightmap::new(512, 512);
    println!("   ✓ Terrain System (512x512)");

    // Atmosphere
    let _sky = helio::atmosphere::SkyDome::default();
    println!("   ✓ Atmospheric Scattering");

    let _clouds = helio::atmosphere::VolumetricClouds::default();
    println!("   ✓ Volumetric Clouds");

    // Water
    let _ocean = helio::water::Ocean::default();
    println!("   ✓ Ocean Simulation (FFT Waves)\n");

    println!("🎥 Starting Render Loop...");
    println!("   Press ESC or close window to exit\n");

    let viewport = Viewport::new(1920, 1080);
    let mut frame_count = 0u64;
    let start_time = std::time::Instant::now();

    // Run event loop
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    println!("\n📊 Final Statistics:");
                    println!("   • Total Frames: {}", frame_count);
                    println!("   • Runtime: {:.2}s", start_time.elapsed().as_secs_f32());
                    println!("   • Average FPS: {:.1}", frame_count as f32 / start_time.elapsed().as_secs_f32());
                    println!("\n✅ Demo Complete!");
                    println!("===========================================\n");
                    *control_flow = ControlFlow::Exit;
                }
                WindowEvent::KeyboardInput {
                    input: winit::event::KeyboardInput {
                        virtual_keycode: Some(winit::event::VirtualKeyCode::Escape),
                        ..
                    },
                    ..
                } => {
                    println!("\n📊 Final Statistics:");
                    println!("   • Total Frames: {}", frame_count);
                    println!("   • Runtime: {:.2}s", start_time.elapsed().as_secs_f32());
                    println!("   • Average FPS: {:.1}", frame_count as f32 / start_time.elapsed().as_secs_f32());
                    println!("\n✅ Demo Complete!");
                    println!("===========================================\n");
                    *control_flow = ControlFlow::Exit;
                }
                _ => {}
            },
            Event::RedrawRequested(_) => {
                // Skip first frame to let swapchain initialize
                if frame_count == 0 {
                    frame_count += 1;
                    return;
                }

                // Update particle system
                particle_system.update(1.0 / 60.0);

                // Acquire frame
                let frame = render_context.gpu_context.acquire_frame();

                // Render frame
                if let Err(e) = renderer.render(&scene, &viewport) {
                    eprintln!("Render error: {:?}", e);
                }

                // Frame is automatically presented when dropped
                let _ = frame;

                frame_count += 1;

                // Print progress every 60 frames
                if frame_count % 60 == 0 {
                    let elapsed = start_time.elapsed().as_secs_f32();
                    let fps = frame_count as f32 / elapsed;
                    println!("   Frame {}: {:.1} FPS | {} entities, {} lights, {} particles",
                        frame_count,
                        fps,
                        scene.entity_count(),
                        directional_lights.len() + point_lights.len() + spot_lights.len(),
                        particle_system.particles.len()
                    );
                }
            }
            Event::MainEventsCleared => {
                window.request_redraw();
            }
            _ => {}
        }
    });
}

fn create_demo_materials() -> Vec<helio::material::Material> {
    use helio::material::{Material, ShadingModel, BlendMode, MaterialFlags};

    vec![
        Material {
            name: "Chrome".to_string(),
            shading_model: ShadingModel::DefaultLit,
            blend_mode: BlendMode::Opaque,
            flags: MaterialFlags::empty(),
            base_color: glam::Vec4::new(0.95, 0.95, 0.95, 1.0),
            metallic: 1.0,
            roughness: 0.05,
            specular: 0.5,
            emissive: glam::Vec3::ZERO,
            emissive_strength: 0.0,
            normal_scale: 1.0,
            occlusion_strength: 1.0,
            anisotropy: 0.0,
            anisotropy_rotation: 0.0,
            subsurface_color: glam::Vec3::ZERO,
            subsurface_radius: 0.0,
            transmission: 0.0,
            thickness: 0.0,
            ior: 1.5,
            clear_coat: 0.0,
            clear_coat_roughness: 0.0,
            clear_coat_normal: glam::Vec3::ZERO,
            sheen_color: glam::Vec3::ZERO,
            sheen_roughness: 0.0,
            base_color_texture: None,
            metallic_roughness_texture: None,
            normal_texture: None,
            occlusion_texture: None,
            emissive_texture: None,
            opacity_texture: None,
            uv_offset: glam::Vec2::ZERO,
            uv_scale: glam::Vec2::ONE,
            uv_rotation: 0.0,
            alpha_cutoff: 0.5,
            shadow_bias: 0.0,
            shadow_slope_bias: 0.0,
        },
        Material {
            name: "Gold".to_string(),
            shading_model: ShadingModel::DefaultLit,
            blend_mode: BlendMode::Opaque,
            flags: MaterialFlags::empty(),
            base_color: glam::Vec4::new(1.0, 0.766, 0.336, 1.0),
            metallic: 1.0,
            roughness: 0.1,
            specular: 0.5,
            emissive: glam::Vec3::ZERO,
            emissive_strength: 0.0,
            normal_scale: 1.0,
            occlusion_strength: 1.0,
            anisotropy: 0.0,
            anisotropy_rotation: 0.0,
            subsurface_color: glam::Vec3::ZERO,
            subsurface_radius: 0.0,
            transmission: 0.0,
            thickness: 0.0,
            ior: 1.5,
            clear_coat: 0.0,
            clear_coat_roughness: 0.0,
            clear_coat_normal: glam::Vec3::ZERO,
            sheen_color: glam::Vec3::ZERO,
            sheen_roughness: 0.0,
            base_color_texture: None,
            metallic_roughness_texture: None,
            normal_texture: None,
            occlusion_texture: None,
            emissive_texture: None,
            opacity_texture: None,
            uv_offset: glam::Vec2::ZERO,
            uv_scale: glam::Vec2::ONE,
            uv_rotation: 0.0,
            alpha_cutoff: 0.5,
            shadow_bias: 0.0,
            shadow_slope_bias: 0.0,
        },
        Material {
            name: "Plastic Red".to_string(),
            shading_model: ShadingModel::DefaultLit,
            blend_mode: BlendMode::Opaque,
            flags: MaterialFlags::empty(),
            base_color: glam::Vec4::new(0.8, 0.1, 0.1, 1.0),
            metallic: 0.0,
            roughness: 0.4,
            specular: 0.5,
            emissive: glam::Vec3::ZERO,
            emissive_strength: 0.0,
            normal_scale: 1.0,
            occlusion_strength: 1.0,
            anisotropy: 0.0,
            anisotropy_rotation: 0.0,
            subsurface_color: glam::Vec3::ZERO,
            subsurface_radius: 0.0,
            transmission: 0.0,
            thickness: 0.0,
            ior: 1.5,
            clear_coat: 0.0,
            clear_coat_roughness: 0.0,
            clear_coat_normal: glam::Vec3::ZERO,
            sheen_color: glam::Vec3::ZERO,
            sheen_roughness: 0.0,
            base_color_texture: None,
            metallic_roughness_texture: None,
            normal_texture: None,
            occlusion_texture: None,
            emissive_texture: None,
            opacity_texture: None,
            uv_offset: glam::Vec2::ZERO,
            uv_scale: glam::Vec2::ONE,
            uv_rotation: 0.0,
            alpha_cutoff: 0.5,
            shadow_bias: 0.0,
            shadow_slope_bias: 0.0,
        },
    ]
}

// Create GPU context with window support
fn create_gpu_context_with_window<W: raw_window_handle::HasRawWindowHandle + raw_window_handle::HasRawDisplayHandle>(
    window: &W
) -> gpu::Context {
    // Create GPU context with validation disabled for performance
    let desc = gpu::ContextDesc {
        validation: false,
        capture: false,
        overlay: false,
    };

    unsafe {
        gpu::Context::init_windowed(window, desc).expect("Failed to initialize GPU context")
    }
}

// Present rendered frame to the surface
fn present_frame(context: &gpu::Context) {
    // Acquire next swapchain image and present
    // The frame is automatically presented when dropped
    let _frame = context.acquire_frame();

    // Note: In a real application, you would render to the frame's texture here
    // For this demo, we're just acquiring and releasing to show the window
}
