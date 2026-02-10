use blade_graphics as gpu;
use glam::{Mat4, Vec3};
use helio_core::{create_cube_mesh, create_plane_mesh, create_sphere_mesh};
use helio_feature_base_geometry::BaseGeometry;
use helio_feature_bloom::Bloom;
use helio_feature_lighting::BasicLighting;
use helio_feature_materials::BasicMaterials;
use helio_feature_procedural_shadows::ProceduralShadows;
use helio_features::FeatureRegistry;
use helio_render::{CameraUniforms, FeatureRenderer, TransformUniforms};
use std::{ptr, sync::Arc, time::Instant};

struct Example {
    context: Arc<gpu::Context>,
    surface: gpu::Surface,
    renderer: FeatureRenderer,
    command_encoder: gpu::CommandEncoder,
    window_size: winit::dpi::PhysicalSize<u32>,

    cube_vertices: gpu::Buffer,
    cube_indices: gpu::Buffer,
    cube_index_count: u32,

    sphere_vertices: gpu::Buffer,
    sphere_indices: gpu::Buffer,
    sphere_index_count: u32,

    plane_vertices: gpu::Buffer,
    plane_indices: gpu::Buffer,
    plane_index_count: u32,

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
        // Get the shadow feature
        let shadows = if let Some(feature) = self.renderer.registry_mut().get_feature_mut("procedural_shadows") {
            unsafe {
                &mut *(feature.as_mut() as *mut dyn helio_features::Feature as *mut ProceduralShadows)
            }
        } else {
            return;
        };

        // Adjust intensity of all lights
        for light in shadows.lights_mut() {
            light.intensity *= factor;
        }
        
        println!("Light brightness adjusted by {}x", factor);
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

        let cube_mesh = create_cube_mesh(1.0);
        let sphere_mesh = create_sphere_mesh(0.5, 32, 32);
        let plane_mesh = create_plane_mesh(20.0, 20.0);

        let cube_vertices = context.create_buffer(gpu::BufferDesc {
            name: "cube_vertices",
            size: (cube_mesh.vertices.len() * std::mem::size_of::<helio_core::PackedVertex>())
                as u64,
            memory: gpu::Memory::Shared,
        });
        unsafe {
            ptr::copy_nonoverlapping(
                cube_mesh.vertices.as_ptr(),
                cube_vertices.data() as *mut helio_core::PackedVertex,
                cube_mesh.vertices.len(),
            );
        }
        context.sync_buffer(cube_vertices);

        let cube_indices = context.create_buffer(gpu::BufferDesc {
            name: "cube_indices",
            size: (cube_mesh.indices.len() * std::mem::size_of::<u32>()) as u64,
            memory: gpu::Memory::Shared,
        });
        unsafe {
            ptr::copy_nonoverlapping(
                cube_mesh.indices.as_ptr(),
                cube_indices.data() as *mut u32,
                cube_mesh.indices.len(),
            );
        }
        context.sync_buffer(cube_indices);

        let sphere_vertices = context.create_buffer(gpu::BufferDesc {
            name: "sphere_vertices",
            size: (sphere_mesh.vertices.len() * std::mem::size_of::<helio_core::PackedVertex>())
                as u64,
            memory: gpu::Memory::Shared,
        });
        unsafe {
            ptr::copy_nonoverlapping(
                sphere_mesh.vertices.as_ptr(),
                sphere_vertices.data() as *mut helio_core::PackedVertex,
                sphere_mesh.vertices.len(),
            );
        }
        context.sync_buffer(sphere_vertices);

        let sphere_indices = context.create_buffer(gpu::BufferDesc {
            name: "sphere_indices",
            size: (sphere_mesh.indices.len() * std::mem::size_of::<u32>()) as u64,
            memory: gpu::Memory::Shared,
        });
        unsafe {
            ptr::copy_nonoverlapping(
                sphere_mesh.indices.as_ptr(),
                sphere_indices.data() as *mut u32,
                sphere_mesh.indices.len(),
            );
        }
        context.sync_buffer(sphere_indices);

        let plane_vertices = context.create_buffer(gpu::BufferDesc {
            name: "plane_vertices",
            size: (plane_mesh.vertices.len() * std::mem::size_of::<helio_core::PackedVertex>())
                as u64,
            memory: gpu::Memory::Shared,
        });
        unsafe {
            ptr::copy_nonoverlapping(
                plane_mesh.vertices.as_ptr(),
                plane_vertices.data() as *mut helio_core::PackedVertex,
                plane_mesh.vertices.len(),
            );
        }
        context.sync_buffer(plane_vertices);

        let plane_indices = context.create_buffer(gpu::BufferDesc {
            name: "plane_indices",
            size: (plane_mesh.indices.len() * std::mem::size_of::<u32>()) as u64,
            memory: gpu::Memory::Shared,
        });
        unsafe {
            ptr::copy_nonoverlapping(
                plane_mesh.indices.as_ptr(),
                plane_indices.data() as *mut u32,
                plane_mesh.indices.len(),
            );
        }
        context.sync_buffer(plane_indices);

        let base_geometry = BaseGeometry::new();
        let base_shader = base_geometry.shader_template().to_string();

        let mut registry = FeatureRegistry::new();
        registry.register(base_geometry);
        registry.register(BasicLighting::new());
        registry.register(BasicMaterials::new());
        registry.register(ProceduralShadows::new().with_ambient(0.0));
        registry.register(Bloom::new());

        let renderer = FeatureRenderer::new(
            context.clone(),
            surface.info().format,
            window_size.width,
            window_size.height,
            registry,
            &base_shader,
        );

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
            cube_vertices,
            cube_indices,
            cube_index_count: cube_mesh.indices.len() as u32,
            sphere_vertices,
            sphere_indices,
            sphere_index_count: sphere_mesh.indices.len() as u32,
            plane_vertices,
            plane_indices,
            plane_index_count: plane_mesh.indices.len() as u32,
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

        let aspect_ratio = self.window_size.width as f32 / self.window_size.height as f32;
        let projection = Mat4::perspective_rh(45.0f32.to_radians(), aspect_ratio, 0.1, 100.0);

        let camera_pos = Vec3::new(5.0 * elapsed_wrapped.sin(), 4.0, 5.0 * elapsed_wrapped.cos());
        let view = Mat4::look_at_rh(camera_pos, Vec3::new(0.0, 0.5, 0.0), Vec3::Y);
        let view_proj = projection * view;

        let camera = CameraUniforms {
            view_proj: view_proj.to_cols_array_2d(),
            position: camera_pos.to_array(),
            _pad: 0.0,
        };

        let mut meshes = Vec::new();

        // Ground plane
        let plane_transform = Mat4::from_translation(Vec3::ZERO);
        meshes.push((
            TransformUniforms {
                model: plane_transform.to_cols_array_2d(),
            },
            self.plane_vertices.into(),
            self.plane_indices.into(),
            self.plane_index_count,
        ));

        // Central rotating cube
        let cube_transform =
            Mat4::from_rotation_y(elapsed_wrapped) * Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0));
        meshes.push((
            TransformUniforms {
                model: cube_transform.to_cols_array_2d(),
            },
            self.cube_vertices.into(),
            self.cube_indices.into(),
            self.cube_index_count,
        ));

        // Sphere orbiting around center - circular path
        let orbit_radius = 3.0;
        let sphere1_x = orbit_radius * elapsed_wrapped.cos();
        let sphere1_z = orbit_radius * elapsed_wrapped.sin();
        let sphere1_transform = Mat4::from_translation(Vec3::new(sphere1_x, 1.5, sphere1_z));
        meshes.push((
            TransformUniforms {
                model: sphere1_transform.to_cols_array_2d(),
            },
            self.sphere_vertices.into(),
            self.sphere_indices.into(),
            self.sphere_index_count,
        ));

        // Second sphere orbiting opposite direction
        let sphere2_x = orbit_radius * (-elapsed_wrapped).cos();
        let sphere2_z = orbit_radius * (-elapsed_wrapped).sin();
        let sphere2_transform = Mat4::from_translation(Vec3::new(sphere2_x, 1.0, sphere2_z));
        meshes.push((
            TransformUniforms {
                model: sphere2_transform.to_cols_array_2d(),
            },
            self.sphere_vertices.into(),
            self.sphere_indices.into(),
            self.sphere_index_count,
        ));

        // Vertically bobbing cube that passes through the rotating cube
        let bob_height = 2.0 + (elapsed * 2.0).sin() * 1.5;
        let bobbing_cube_transform =
            Mat4::from_rotation_x(elapsed_wrapped * 0.5) *
            Mat4::from_translation(Vec3::new(0.0, bob_height, 0.0));
        meshes.push((
            TransformUniforms {
                model: bobbing_cube_transform.to_cols_array_2d(),
            },
            self.cube_vertices.into(),
            self.cube_indices.into(),
            self.cube_index_count,
        ));

        // Small cube moving in figure-8 pattern
        let figure8_x = 2.5 * (elapsed * 0.7).sin();
        let figure8_z = 1.5 * (elapsed * 1.4).sin();
        let figure8_cube_transform =
            Mat4::from_rotation_z(elapsed_wrapped * 1.5) *
            Mat4::from_translation(Vec3::new(figure8_x, 0.75, figure8_z)) *
            Mat4::from_scale(Vec3::splat(0.6));
        meshes.push((
            TransformUniforms {
                model: figure8_cube_transform.to_cols_array_2d(),
            },
            self.cube_vertices.into(),
            self.cube_indices.into(),
            self.cube_index_count,
        ));

        // Sphere moving up and down through other objects
        let vertical_sphere_y = 0.5 + (elapsed * 1.5).sin() * 2.5;
        let vertical_sphere_transform = Mat4::from_translation(Vec3::new(-1.5, vertical_sphere_y, 1.5));
        meshes.push((
            TransformUniforms {
                model: vertical_sphere_transform.to_cols_array_2d(),
            },
            self.sphere_vertices.into(),
            self.sphere_indices.into(),
            self.sphere_index_count,
        ));

        // Spinning cube on the side
        let side_cube_transform =
            Mat4::from_translation(Vec3::new(-3.0, 1.5, -2.0)) *
            Mat4::from_rotation_y(elapsed_wrapped * 2.0) *
            Mat4::from_rotation_x(elapsed_wrapped);
        meshes.push((
            TransformUniforms {
                model: side_cube_transform.to_cols_array_2d(),
            },
            self.cube_vertices.into(),
            self.cube_indices.into(),
            self.cube_index_count,
        ));

        // Floating sphere moving in a slow arc
        let arc_angle = elapsed * 0.5;
        let arc_x = 2.0 * arc_angle.cos();
        let arc_z = 2.0 * arc_angle.sin();
        let arc_sphere_transform = Mat4::from_translation(Vec3::new(arc_x, 2.5, arc_z));
        meshes.push((
            TransformUniforms {
                model: arc_sphere_transform.to_cols_array_2d(),
            },
            self.sphere_vertices.into(),
            self.sphere_indices.into(),
            self.sphere_index_count,
        ));

        // Stack of cubes that periodically intersect
        let stack1_y = 0.5 + (elapsed * 1.2).sin() * 0.3;
        let stack1_transform =
            Mat4::from_translation(Vec3::new(2.5, stack1_y, -2.0)) *
            Mat4::from_scale(Vec3::splat(0.8));
        meshes.push((
            TransformUniforms {
                model: stack1_transform.to_cols_array_2d(),
            },
            self.cube_vertices.into(),
            self.cube_indices.into(),
            self.cube_index_count,
        ));

        let stack2_y = 1.0 + (elapsed * 1.2 + 1.0).sin() * 0.3;
        let stack2_transform =
            Mat4::from_translation(Vec3::new(2.5, stack2_y, -2.0)) *
            Mat4::from_scale(Vec3::splat(0.8));
        meshes.push((
            TransformUniforms {
                model: stack2_transform.to_cols_array_2d(),
            },
            self.cube_vertices.into(),
            self.cube_indices.into(),
            self.cube_index_count,
        ));

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
        .with_title("Helio - Shadow Stress Test (1: Geometry, 2: Lighting, 3: Materials, 4: Procedural Shadows)")
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
                    event:
                        winit::event::KeyEvent {
                            physical_key,
                            state: winit::event::ElementState::Pressed,
                            ..
                        },
                    ..
                } => {
                    match physical_key {
                        winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Escape) => {
                            elwt.exit();
                        }
                        winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Digit1) => {
                            if let Ok(_) = app.renderer.registry_mut().toggle_feature("base_geometry") {
                                let enabled = app.renderer.registry().get_feature("base_geometry").unwrap().is_enabled();
                                let status = if enabled { "ON" } else { "OFF" };
                                println!("[1] Base Geometry: {}", status);
                                log::info!("[1] Base Geometry: {}", status);
                                app.renderer.rebuild_pipeline();
                            }
                        }
                        winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Digit2) => {
                            if let Ok(_) = app.renderer.registry_mut().toggle_feature("basic_lighting") {
                                let enabled = app.renderer.registry().get_feature("basic_lighting").unwrap().is_enabled();
                                let status = if enabled { "ON" } else { "OFF" };
                                println!("[2] Basic Lighting: {}", status);
                                log::info!("[2] Basic Lighting: {}", status);
                                app.renderer.rebuild_pipeline();
                            }
                        }
                        winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Digit3) => {
                            if let Ok(_) = app.renderer.registry_mut().toggle_feature("basic_materials") {
                                let enabled = app.renderer.registry().get_feature("basic_materials").unwrap().is_enabled();
                                let status = if enabled { "ON" } else { "OFF" };
                                println!("[3] Basic Materials: {}", status);
                                log::info!("[3] Basic Materials: {}", status);
                                app.renderer.rebuild_pipeline();
                            }
                        }
                        winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Digit4) => {
                            if let Ok(_) = app.renderer.registry_mut().toggle_feature("procedural_shadows") {
                                let enabled = app.renderer.registry().get_feature("procedural_shadows").unwrap().is_enabled();
                                let status = if enabled { "ON" } else { "OFF" };
                                println!("[4] Procedural Shadows: {}", status);
                                log::info!("[4] Procedural Shadows: {}", status);
                                app.renderer.rebuild_pipeline();
                            }
                        }
                        winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Equal) |
                        winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::NumpadAdd) => {
                            // Brighten lights by 20%
                            app.adjust_light_brightness(1.2);
                        }
                        winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Minus) |
                        winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::NumpadSubtract) => {
                            // Darken lights by 20%
                            app.adjust_light_brightness(0.8);
                        }
                        _ => {}
                    }
                }
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
