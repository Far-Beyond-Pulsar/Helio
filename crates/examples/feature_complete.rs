use blade_graphics as gpu;
use glam::{Mat4, Vec3};
use helio_core::{create_cube_mesh, create_plane_mesh, create_sphere_mesh, MeshBuffer};
use helio_feature_base_geometry::BaseGeometry;
use helio_feature_bloom::Bloom;
use helio_feature_lighting::BasicLighting;
use helio_feature_materials::BasicMaterials;
use helio_feature_procedural_shadows::{ProceduralShadows};
use helio_features::FeatureRegistry;
use helio_render::{CameraUniforms, FeatureRenderer, TransformUniforms};
use std::{sync::Arc, time::Instant};

struct Example {
    context: Arc<gpu::Context>,
    surface: gpu::Surface,
    renderer: FeatureRenderer,
    command_encoder: gpu::CommandEncoder,
    window_size: winit::dpi::PhysicalSize<u32>,

    cube_mesh: MeshBuffer,
    sphere_mesh: MeshBuffer,
    plane_mesh: MeshBuffer,

    start_time: Instant,
    last_frame_time: Instant,
}

impl Example {
    fn make_surface_config(size: winit::dpi::PhysicalSize<u32>) -> gpu::SurfaceConfig {
        gpu::SurfaceConfig {
            size: gpu::Extent {
                width: size.width,
                height: size.height,
                depth: 1,
            },
            usage: gpu::TextureUsage::TARGET,
            display_sync: gpu::DisplaySync::Recent,
            ..Default::default()
        }
    }

    fn adjust_light_brightness(&mut self, factor: f32) {
        if let Some(shadows) = self
            .renderer
            .registry_mut()
            .get_feature_as_mut::<ProceduralShadows>("procedural_shadows")
        {
            for light in shadows.lights_mut() {
                light.intensity *= factor;
            }
            println!("Light brightness adjusted by {}x", factor);
        }
    }

    fn new(window: &winit::window::Window) -> Self {
        let context = Arc::new(unsafe {
            gpu::Context::init(gpu::ContextDesc {
                presentation: true,
                validation: cfg!(debug_assertions),
                timing: false,
                capture: false,
                overlay: true,
                device_id: 0,
            })
            .unwrap()
        });

        let window_size = window.inner_size();
        let surface = context
            .create_surface_configured(window, Self::make_surface_config(window_size))
            .unwrap();

        let cube_mesh = MeshBuffer::from_mesh(&context, "cube", &create_cube_mesh(1.0));
        let sphere_mesh = MeshBuffer::from_mesh(&context, "sphere", &create_sphere_mesh(0.5, 32, 32));
        let plane_mesh = MeshBuffer::from_mesh(&context, "plane", &create_plane_mesh(20.0, 20.0));

        let base_geometry = BaseGeometry::new();
        let base_shader = base_geometry.shader_template().to_string();

        let registry = FeatureRegistry::builder()
            .with_feature(base_geometry)
            .with_feature(BasicLighting::new())
            .with_feature(BasicMaterials::new())
            .with_feature(ProceduralShadows::new().with_ambient(0.0))
            .with_feature(Bloom::new())
            .build();

        let renderer = FeatureRenderer::new(
            context.clone(),
            surface.info().format,
            window_size.width,
            window_size.height,
            registry,
            &base_shader,
        )
        .expect("Failed to create renderer");

        let command_encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
            name: "main",
            buffer_count: 2,
        });

        let now = Instant::now();
        Self {
            context,
            surface,
            renderer,
            command_encoder,
            window_size,
            cube_mesh,
            sphere_mesh,
            plane_mesh,
            start_time: now,
            last_frame_time: now,
        }
    }

    fn render(&mut self) {
        let frame = self.surface.acquire_frame();

        let now = Instant::now();
        let delta_time = (now - self.last_frame_time).as_secs_f32();
        self.last_frame_time = now;

        let elapsed = (now - self.start_time).as_secs_f32();
        let elapsed_wrapped = elapsed % (2.0 * std::f32::consts::PI);

        self.command_encoder.start();
        self.command_encoder.init_texture(frame.texture());

        let aspect = self.window_size.width as f32 / self.window_size.height as f32;
        let camera_pos = Vec3::new(
            5.0 * elapsed_wrapped.sin(),
            4.0,
            5.0 * elapsed_wrapped.cos(),
        );
        let view = Mat4::look_at_rh(camera_pos, Vec3::new(0.0, 0.5, 0.0), Vec3::Y);
        let proj = Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.1, 100.0);
        let camera = CameraUniforms::new(proj * view, camera_pos);

        let mut meshes = Vec::new();

        // Ground plane
        meshes.push((TransformUniforms::from_matrix(Mat4::IDENTITY), &self.plane_mesh));

        // Central rotating cube
        meshes.push((
            TransformUniforms::from_matrix(
                Mat4::from_rotation_y(elapsed_wrapped)
                    * Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0)),
            ),
            &self.cube_mesh,
        ));

        // Orbiting spheres
        let orbit_radius = 3.0;
        meshes.push((
            TransformUniforms::from_matrix(Mat4::from_translation(Vec3::new(
                orbit_radius * elapsed_wrapped.cos(),
                1.5,
                orbit_radius * elapsed_wrapped.sin(),
            ))),
            &self.sphere_mesh,
        ));
        meshes.push((
            TransformUniforms::from_matrix(Mat4::from_translation(Vec3::new(
                orbit_radius * (-elapsed_wrapped).cos(),
                1.0,
                orbit_radius * (-elapsed_wrapped).sin(),
            ))),
            &self.sphere_mesh,
        ));

        // Bobbing cube
        let bob_height = 2.0 + (elapsed * 2.0).sin() * 1.5;
        meshes.push((
            TransformUniforms::from_matrix(
                Mat4::from_rotation_x(elapsed_wrapped * 0.5)
                    * Mat4::from_translation(Vec3::new(0.0, bob_height, 0.0)),
            ),
            &self.cube_mesh,
        ));

        // Figure-8 small cube
        meshes.push((
            TransformUniforms::from_matrix(
                Mat4::from_rotation_z(elapsed_wrapped * 1.5)
                    * Mat4::from_translation(Vec3::new(
                        2.5 * (elapsed * 0.7).sin(),
                        0.75,
                        1.5 * (elapsed * 1.4).sin(),
                    ))
                    * Mat4::from_scale(Vec3::splat(0.6)),
            ),
            &self.cube_mesh,
        ));

        // Vertical bouncing sphere
        meshes.push((
            TransformUniforms::from_matrix(Mat4::from_translation(Vec3::new(
                -1.5,
                0.5 + (elapsed * 1.5).sin() * 2.5,
                1.5,
            ))),
            &self.sphere_mesh,
        ));

        // Side spinning cube
        meshes.push((
            TransformUniforms::from_matrix(
                Mat4::from_translation(Vec3::new(-3.0, 1.5, -2.0))
                    * Mat4::from_rotation_y(elapsed_wrapped * 2.0)
                    * Mat4::from_rotation_x(elapsed_wrapped),
            ),
            &self.cube_mesh,
        ));

        // Floating arc sphere
        meshes.push((
            TransformUniforms::from_matrix(Mat4::from_translation(Vec3::new(
                2.0 * (elapsed * 0.5).cos(),
                2.5,
                2.0 * (elapsed * 0.5).sin(),
            ))),
            &self.sphere_mesh,
        ));

        // Stacked wobbling cubes
        for (i, offset) in [(0.0f32, 0.5f32), (1.0, 1.0)].iter() {
            let y = offset + (elapsed * 1.2 + i).sin() * 0.3;
            meshes.push((
                TransformUniforms::from_matrix(
                    Mat4::from_translation(Vec3::new(2.5, y, -2.0))
                        * Mat4::from_scale(Vec3::splat(0.8)),
                ),
                &self.cube_mesh,
            ));
        }

        self.renderer.render(
            &mut self.command_encoder,
            frame.texture_view(),
            camera,
            &meshes,
            delta_time,
        );

        self.command_encoder.present(frame);
        let sync_point = self.context.submit(&mut self.command_encoder);
        self.context.wait_for(&sync_point, !0);
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.window_size = new_size;
        self.context
            .reconfigure_surface(&mut self.surface, Self::make_surface_config(new_size));
        self.renderer.resize(new_size.width, new_size.height);
    }
}

fn main() {
    env_logger::init();

    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    let window_attr = winit::window::Window::default_attributes()
        .with_title("Helio - Shadow Stress Test (1: Geometry, 2: Lighting, 3: Materials, 4: Shadows)")
        .with_inner_size(winit::dpi::LogicalSize::new(1920, 1080));

    #[allow(deprecated)]
    let window = Arc::new(event_loop.create_window(window_attr).unwrap());
    let mut app = Example::new(&window);

    #[allow(deprecated)]
    event_loop
        .run(move |event, elwt| match event {
            winit::event::Event::WindowEvent { event, .. } => match event {
                winit::event::WindowEvent::CloseRequested => elwt.exit(),
                winit::event::WindowEvent::KeyboardInput {
                    event: winit::event::KeyEvent {
                        physical_key,
                        state: winit::event::ElementState::Pressed,
                        ..
                    },
                    ..
                } => match physical_key {
                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Escape) => {
                        elwt.exit();
                    }
                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Digit1) => {
                        if let Ok(on) = app.renderer.toggle_and_rebuild("base_geometry") {
                            println!("[1] Base Geometry: {}", if on { "ON" } else { "OFF" });
                        }
                    }
                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Digit2) => {
                        if let Ok(on) = app.renderer.toggle_and_rebuild("basic_lighting") {
                            println!("[2] Basic Lighting: {}", if on { "ON" } else { "OFF" });
                        }
                    }
                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Digit3) => {
                        if let Ok(on) = app.renderer.toggle_and_rebuild("basic_materials") {
                            println!("[3] Basic Materials: {}", if on { "ON" } else { "OFF" });
                        }
                    }
                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Digit4) => {
                        if let Ok(on) = app.renderer.toggle_and_rebuild("procedural_shadows") {
                            println!("[4] Procedural Shadows: {}", if on { "ON" } else { "OFF" });
                        }
                    }
                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Equal)
                    | winit::keyboard::PhysicalKey::Code(
                        winit::keyboard::KeyCode::NumpadAdd,
                    ) => {
                        app.adjust_light_brightness(1.2);
                    }
                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Minus)
                    | winit::keyboard::PhysicalKey::Code(
                        winit::keyboard::KeyCode::NumpadSubtract,
                    ) => {
                        app.adjust_light_brightness(0.8);
                    }
                    _ => {}
                },
                winit::event::WindowEvent::Resized(new_size) => {
                    app.resize(new_size);
                }
                winit::event::WindowEvent::RedrawRequested => {
                    app.render();
                    window.request_redraw();
                }
                _ => {}
            },
            winit::event::Event::AboutToWait => {
                window.request_redraw();
            }
            _ => {}
        })
        .unwrap();
}
