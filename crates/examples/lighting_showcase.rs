use blade_graphics as gpu;
use glam::{Mat4, Vec3};
use helio_core::{create_cube_mesh, create_plane_mesh, create_sphere_mesh};
use helio_feature_base_geometry::BaseGeometry;
use helio_feature_lighting::BasicLighting;
use helio_feature_materials::BasicMaterials;
use helio_feature_procedural_shadows::{ProceduralShadows, LightType, LightConfig};
use helio_features::FeatureRegistry;
use helio_render::{CameraUniforms, FeatureRenderer, TransformUniforms};
use std::{collections::HashSet, ptr, sync::Arc, time::Instant};

struct CameraController {
    position: Vec3,
    yaw: f32,
    pitch: f32,
    move_speed: f32,
    look_speed: f32,
    keys_pressed: HashSet<winit::keyboard::KeyCode>,
}

impl CameraController {
    fn new(position: Vec3) -> Self {
        Self {
            position,
            yaw: -90.0_f32.to_radians(),
            pitch: -20.0_f32.to_radians(),
            move_speed: 5.0,
            look_speed: 0.1,
            keys_pressed: HashSet::new(),
        }
    }

    fn get_forward(&self) -> Vec3 {
        Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
        .normalize()
    }

    fn get_right(&self) -> Vec3 {
        self.get_forward().cross(Vec3::Y).normalize()
    }

    fn update(&mut self, delta_time: f32) {
        let forward = self.get_forward();
        let right = self.get_right();
        let speed = self.move_speed * delta_time;

        if self.keys_pressed.contains(&winit::keyboard::KeyCode::KeyW) {
            self.position += forward * speed;
        }
        if self.keys_pressed.contains(&winit::keyboard::KeyCode::KeyS) {
            self.position -= forward * speed;
        }
        if self.keys_pressed.contains(&winit::keyboard::KeyCode::KeyA) {
            self.position -= right * speed;
        }
        if self.keys_pressed.contains(&winit::keyboard::KeyCode::KeyD) {
            self.position += right * speed;
        }
        if self.keys_pressed.contains(&winit::keyboard::KeyCode::Space) {
            self.position.y += speed;
        }
        if self.keys_pressed.contains(&winit::keyboard::KeyCode::ShiftLeft) {
            self.position.y -= speed;
        }
    }

    fn handle_mouse_motion(&mut self, delta_x: f32, delta_y: f32) {
        self.yaw += delta_x * self.look_speed * 0.01;
        self.pitch -= delta_y * self.look_speed * 0.01;

        // Clamp pitch to prevent camera flipping
        self.pitch = self.pitch.clamp(-89.0_f32.to_radians(), 89.0_f32.to_radians());
    }

    fn get_view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.position + self.get_forward(), Vec3::Y)
    }
}

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
    camera: CameraController,
    cursor_grabbed: bool,
    current_light_type: usize,  // 0: spotlight, 1: point, 2: rect, 3: sun
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

        // Start with spotlight shadows active
        let spotlight = ProceduralShadows::new()
            .with_spot_light(
                Vec3::new(0.0, 7.0, -2.0),
                Vec3::new(0.0, -1.0, 0.0),
                30.0_f32.to_radians(),
                45.0_f32.to_radians(),
                15.0, // attenuation radius
            );
        registry.register(spotlight);

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
            camera: CameraController::new(Vec3::new(0.0, 3.0, 10.0)),
            cursor_grabbed: false,
            current_light_type: 0,
        }
    }

    fn update_light_type(&mut self) {
        // Create the light configuration based on current type
        let config = match self.current_light_type {
            0 => {
                println!("[5] Red Spotlight (indoor center)");
                LightConfig {
                    light_type: LightType::Spot {
                        inner_angle: 30.0_f32.to_radians(),
                        outer_angle: 45.0_f32.to_radians(),
                    },
                    position: Vec3::new(0.0, 7.0, -2.0),
                    direction: Vec3::new(0.0, -1.0, 0.0),
                    intensity: 1.0,
                    color: Vec3::new(1.0, 0.3, 0.3), // Red tint
                    attenuation_radius: 15.0,
                    attenuation_falloff: 2.0,
                }
            }
            1 => {
                println!("[6] Green Point Light (indoor corner)");
                LightConfig {
                    light_type: LightType::Point,
                    position: Vec3::new(-3.0, 4.0, -3.0),
                    direction: Vec3::new(0.0, -1.0, 0.0),
                    intensity: 1.0,
                    color: Vec3::new(0.3, 1.0, 0.3), // Green tint
                    attenuation_radius: 12.0,
                    attenuation_falloff: 2.0,
                }
            }
            2 => {
                println!("[7] Blue Rectangular Light (indoor ceiling)");
                LightConfig {
                    light_type: LightType::Rect {
                        width: 3.0,
                        height: 3.0,
                    },
                    position: Vec3::new(1.0, 4.8, -1.0),
                    direction: Vec3::new(0.0, -1.0, 0.0),
                    intensity: 1.0,
                    color: Vec3::new(0.3, 0.3, 1.0), // Blue tint
                    attenuation_radius: 10.0,
                    attenuation_falloff: 2.0,
                }
            }
            3 => {
                println!("[8] Directional Sun (outdoor)");
                LightConfig {
                    light_type: LightType::Directional,
                    position: Vec3::new(10.0, 15.0, 10.0),
                    direction: Vec3::new(0.5, -1.0, 0.3).normalize(),
                    intensity: 1.0,
                    color: Vec3::new(1.0, 0.95, 0.8), // Warm sunlight
                    attenuation_radius: 100.0, // Not used for directional
                    attenuation_falloff: 1.0,  // Not used for directional
                }
            }
            _ => LightConfig::default(),
        };

        // Update the shadow feature's light configuration
        if let Some(feature) = self.renderer.registry_mut().get_feature_mut("procedural_shadows") {
            // Downcast to ProceduralShadows to access set_light_config
            let shadows: &mut ProceduralShadows = unsafe {
                &mut *(feature.as_mut() as *mut dyn helio_features::Feature as *mut ProceduralShadows)
            };
            shadows.set_light_config(config);
        }
    }

    fn render(&mut self) {
        let frame = self.surface.acquire_frame();

        let now = Instant::now();
        let delta_time = (now - self.last_frame_time).as_secs_f32();
        self.last_frame_time = now;

        // Update camera
        self.camera.update(delta_time);

        self.command_encoder.start();
        self.command_encoder.init_texture(frame.texture());

        let aspect_ratio = self.window_size.width as f32 / self.window_size.height as f32;
        let projection = Mat4::perspective_rh(60.0f32.to_radians(), aspect_ratio, 0.1, 100.0);
        let view = self.camera.get_view_matrix();
        let view_proj = projection * view;

        let camera = CameraUniforms {
            view_proj: view_proj.to_cols_array_2d(),
            position: self.camera.position.to_array(),
            _pad: 0.0,
        };

        let mut meshes = Vec::new();

        // Ground plane
        let ground_transform = Mat4::from_translation(Vec3::ZERO);
        meshes.push((
            TransformUniforms {
                model: ground_transform.to_cols_array_2d(),
            },
            self.plane_vertices.into(),
            self.plane_indices.into(),
            self.plane_index_count,
        ));

        // Build a shelter/room structure
        // Back wall
        let back_wall = Mat4::from_translation(Vec3::new(0.0, 2.5, -5.0)) *
            Mat4::from_scale(Vec3::new(10.0, 5.0, 0.2));
        meshes.push((
            TransformUniforms {
                model: back_wall.to_cols_array_2d(),
            },
            self.cube_vertices.into(),
            self.cube_indices.into(),
            self.cube_index_count,
        ));

        // Left wall
        let left_wall = Mat4::from_translation(Vec3::new(-5.0, 2.5, 0.0)) *
            Mat4::from_scale(Vec3::new(0.2, 5.0, 10.0));
        meshes.push((
            TransformUniforms {
                model: left_wall.to_cols_array_2d(),
            },
            self.cube_vertices.into(),
            self.cube_indices.into(),
            self.cube_index_count,
        ));

        // Right wall (partial, with opening)
        let right_wall_back = Mat4::from_translation(Vec3::new(5.0, 2.5, -3.0)) *
            Mat4::from_scale(Vec3::new(0.2, 5.0, 4.0));
        meshes.push((
            TransformUniforms {
                model: right_wall_back.to_cols_array_2d(),
            },
            self.cube_vertices.into(),
            self.cube_indices.into(),
            self.cube_index_count,
        ));

        // Roof
        let roof = Mat4::from_translation(Vec3::new(0.0, 5.0, 0.0)) *
            Mat4::from_scale(Vec3::new(10.0, 0.2, 10.0));
        meshes.push((
            TransformUniforms {
                model: roof.to_cols_array_2d(),
            },
            self.cube_vertices.into(),
            self.cube_indices.into(),
            self.cube_index_count,
        ));

        // Indoor objects
        let elapsed = (now - self.start_time).as_secs_f32();

        // Rotating cube indoors
        let indoor_cube = Mat4::from_translation(Vec3::new(-2.0, 1.0, -2.0)) *
            Mat4::from_rotation_y(elapsed);
        meshes.push((
            TransformUniforms {
                model: indoor_cube.to_cols_array_2d(),
            },
            self.cube_vertices.into(),
            self.cube_indices.into(),
            self.cube_index_count,
        ));

        // Sphere indoors
        let indoor_sphere = Mat4::from_translation(Vec3::new(2.0, 1.5, -2.0));
        meshes.push((
            TransformUniforms {
                model: indoor_sphere.to_cols_array_2d(),
            },
            self.sphere_vertices.into(),
            self.sphere_indices.into(),
            self.sphere_index_count,
        ));

        // Outdoor objects
        // Cube stack outside
        let outdoor_cube1 = Mat4::from_translation(Vec3::new(7.0, 0.5, 2.0));
        meshes.push((
            TransformUniforms {
                model: outdoor_cube1.to_cols_array_2d(),
            },
            self.cube_vertices.into(),
            self.cube_indices.into(),
            self.cube_index_count,
        ));

        let outdoor_cube2 = Mat4::from_translation(Vec3::new(7.0, 1.5, 2.0)) *
            Mat4::from_rotation_y(elapsed * 0.5);
        meshes.push((
            TransformUniforms {
                model: outdoor_cube2.to_cols_array_2d(),
            },
            self.cube_vertices.into(),
            self.cube_indices.into(),
            self.cube_index_count,
        ));

        // Moving sphere outside
        let outdoor_sphere_x = 8.0 + (elapsed * 0.8).sin() * 3.0;
        let outdoor_sphere = Mat4::from_translation(Vec3::new(outdoor_sphere_x, 2.0, 5.0));
        meshes.push((
            TransformUniforms {
                model: outdoor_sphere.to_cols_array_2d(),
            },
            self.sphere_vertices.into(),
            self.sphere_indices.into(),
            self.sphere_index_count,
        ));

        // Visual indicators for light positions (small colored spheres)
        // Red spotlight indicator
        let spotlight_indicator = Mat4::from_translation(Vec3::new(0.0, 7.0, -2.0)) *
            Mat4::from_scale(Vec3::splat(0.2));
        meshes.push((
            TransformUniforms {
                model: spotlight_indicator.to_cols_array_2d(),
            },
            self.sphere_vertices.into(),
            self.sphere_indices.into(),
            self.sphere_index_count,
        ));

        // Green point light indicator
        let point_light_indicator = Mat4::from_translation(Vec3::new(-3.0, 4.0, -3.0)) *
            Mat4::from_scale(Vec3::splat(0.2));
        meshes.push((
            TransformUniforms {
                model: point_light_indicator.to_cols_array_2d(),
            },
            self.sphere_vertices.into(),
            self.sphere_indices.into(),
            self.sphere_index_count,
        ));

        // Blue rect light indicator (small cube)
        let rect_light_indicator = Mat4::from_translation(Vec3::new(1.0, 4.8, -1.0)) *
            Mat4::from_scale(Vec3::new(1.5, 0.1, 1.5));
        meshes.push((
            TransformUniforms {
                model: rect_light_indicator.to_cols_array_2d(),
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
        .with_title("Helio - Lighting Showcase | WASD: Move, Mouse: Look | 1-4: Features | 5: Spot 6: Point 7: Rect 8: Sun")
        .with_inner_size(winit::dpi::LogicalSize::new(1920, 1080));

    #[allow(deprecated)]
    let window = Arc::new(event_loop.create_window(window_attr).unwrap());
    let mut app = Example::new(&window);

    // Grab cursor for FPS controls
    let _ = window.set_cursor_grab(winit::window::CursorGrabMode::Confined);
    window.set_cursor_visible(false);
    app.cursor_grabbed = true;

    #[allow(deprecated)]
    event_loop
        .run(move |event, elwt| match event {
            winit::event::Event::WindowEvent { event, .. } => match event {
                winit::event::WindowEvent::CloseRequested => elwt.exit(),
                winit::event::WindowEvent::KeyboardInput {
                    event:
                        winit::event::KeyEvent {
                            physical_key,
                            state,
                            ..
                        },
                    ..
                } => {
                    if let winit::keyboard::PhysicalKey::Code(keycode) = physical_key {
                        match state {
                            winit::event::ElementState::Pressed => {
                                app.camera.keys_pressed.insert(keycode);

                                match keycode {
                                    winit::keyboard::KeyCode::Escape => {
                                        if app.cursor_grabbed {
                                            let _ = window.set_cursor_grab(winit::window::CursorGrabMode::None);
                                            window.set_cursor_visible(true);
                                            app.cursor_grabbed = false;
                                        } else {
                                            elwt.exit();
                                        }
                                    }
                                    winit::keyboard::KeyCode::Digit1 => {
                                        if let Ok(_) = app.renderer.registry_mut().toggle_feature("base_geometry") {
                                            let enabled = app.renderer.registry().get_feature("base_geometry").unwrap().is_enabled();
                                            println!("[1] Base Geometry: {}", if enabled { "ON" } else { "OFF" });
                                            app.renderer.rebuild_pipeline();
                                        }
                                    }
                                    winit::keyboard::KeyCode::Digit2 => {
                                        if let Ok(_) = app.renderer.registry_mut().toggle_feature("basic_lighting") {
                                            let enabled = app.renderer.registry().get_feature("basic_lighting").unwrap().is_enabled();
                                            println!("[2] Basic Lighting: {}", if enabled { "ON" } else { "OFF" });
                                            app.renderer.rebuild_pipeline();
                                        }
                                    }
                                    winit::keyboard::KeyCode::Digit3 => {
                                        if let Ok(_) = app.renderer.registry_mut().toggle_feature("basic_materials") {
                                            let enabled = app.renderer.registry().get_feature("basic_materials").unwrap().is_enabled();
                                            println!("[3] Basic Materials: {}", if enabled { "ON" } else { "OFF" });
                                            app.renderer.rebuild_pipeline();
                                        }
                                    }
                                    winit::keyboard::KeyCode::Digit4 => {
                                        if let Ok(_) = app.renderer.registry_mut().toggle_feature("procedural_shadows") {
                                            let enabled = app.renderer.registry().get_feature("procedural_shadows").unwrap().is_enabled();
                                            println!("[4] Procedural Shadows: {}", if enabled { "ON" } else { "OFF" });
                                            app.renderer.rebuild_pipeline();
                                        }
                                    }
                                    winit::keyboard::KeyCode::Digit5 => {
                                        app.current_light_type = 0;
                                        app.update_light_type();
                                    }
                                    winit::keyboard::KeyCode::Digit6 => {
                                        app.current_light_type = 1;
                                        app.update_light_type();
                                    }
                                    winit::keyboard::KeyCode::Digit7 => {
                                        app.current_light_type = 2;
                                        app.update_light_type();
                                    }
                                    winit::keyboard::KeyCode::Digit8 => {
                                        app.current_light_type = 3;
                                        app.update_light_type();
                                    }
                                    _ => {}
                                }
                            }
                            winit::event::ElementState::Released => {
                                app.camera.keys_pressed.remove(&keycode);
                            }
                        }
                    }
                }
                winit::event::WindowEvent::MouseInput {
                    button: winit::event::MouseButton::Left,
                    state: winit::event::ElementState::Pressed,
                    ..
                } => {
                    if !app.cursor_grabbed {
                        let _ = window.set_cursor_grab(winit::window::CursorGrabMode::Confined);
                        window.set_cursor_visible(false);
                        app.cursor_grabbed = true;
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
            winit::event::Event::DeviceEvent { event, .. } => {
                if app.cursor_grabbed {
                    if let winit::event::DeviceEvent::MouseMotion { delta } = event {
                        app.camera.handle_mouse_motion(delta.0 as f32, delta.1 as f32);
                    }
                }
            }
            winit::event::Event::AboutToWait => {
                window.request_redraw();
            }
            _ => {}
        })
        .unwrap();
}
