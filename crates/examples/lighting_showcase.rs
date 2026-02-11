use blade_graphics as gpu;
use glam::{Mat4, Vec3};
use helio_core::{create_cube_mesh, create_plane_mesh, create_sphere_mesh};
use helio_feature_base_geometry::BaseGeometry;
use helio_feature_lighting::BasicLighting;
use helio_feature_materials::BasicMaterials;
use helio_feature_bloom::Bloom;
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
    demo_mode: usize,  // 0: single light cycle, 1: multi-light dance, 2: spotlight array, 3: color party
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

        // Start with an impressive multi-light setup
        let mut shadows = ProceduralShadows::new().with_ambient(0.0);
        
        // Add multiple overlapping colored lights for a dramatic showcase
        shadows.add_light(LightConfig {
            light_type: LightType::Spot {
                inner_angle: 25.0_f32.to_radians(),
                outer_angle: 40.0_f32.to_radians(),
            },
            position: Vec3::new(0.0, 8.0, 0.0),
            direction: Vec3::new(0.0, -1.0, 0.0),
            intensity: 1.5,
            color: Vec3::new(1.0, 0.2, 0.2), // Red
            attenuation_radius: 12.0,
            attenuation_falloff: 2.0,
        })
        .expect("Failed to add light");
        
        shadows.add_light(LightConfig {
            light_type: LightType::Point,
            position: Vec3::new(-4.0, 3.0, -4.0),
            direction: Vec3::new(0.0, -1.0, 0.0),
            intensity: 1.2,
            color: Vec3::new(0.2, 1.0, 0.2), // Green
            attenuation_radius: 10.0,
            attenuation_falloff: 2.5,
        })
        .expect("Failed to add light");
        
        shadows.add_light(LightConfig {
            light_type: LightType::Point,
            position: Vec3::new(4.0, 3.0, -4.0),
            direction: Vec3::new(0.0, -1.0, 0.0),
            intensity: 1.2,
            color: Vec3::new(0.2, 0.2, 1.0), // Blue
            attenuation_radius: 10.0,
            attenuation_falloff: 2.5,
        })
        .expect("Failed to add light");
        
        registry.register(shadows);
        registry.register(Bloom::new());

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
            camera: CameraController::new(Vec3::new(0.0, 5.0, 15.0)),
            cursor_grabbed: false,
            demo_mode: 1, // Start with multi-light dance
        }
    }

    fn update_demo_lights(&mut self, time: f32) {
        // Get the shadow feature
        let shadows = if let Some(feature) = self.renderer.registry_mut().get_feature_mut("procedural_shadows") {
            unsafe {
                &mut *(feature.as_mut() as *mut dyn helio_features::Feature as *mut ProceduralShadows)
            }
        } else {
            return;
        };

        shadows.clear_lights();

        match self.demo_mode {
            0 => {
                // Single rotating spotlight with pulsing intensity
                let angle = time * 0.5;
                let pulse = (time * 2.0).sin() * 0.3 + 1.2;
                
                let _ = shadows.add_light(LightConfig {
                    light_type: LightType::Spot {
                        inner_angle: 30.0_f32.to_radians(),
                        outer_angle: 50.0_f32.to_radians(),
                    },
                    position: Vec3::new(angle.cos() * 5.0, 8.0, angle.sin() * 5.0),
                    direction: Vec3::new(-angle.cos(), -1.0, -angle.sin()).normalize(),
                    intensity: pulse,
                    color: Vec3::new(1.0, 0.9, 0.7),
                    attenuation_radius: 18.0,
                    attenuation_falloff: 2.0,
                });
            }
            1 => {
                // Multi-light dance - RGB lights circling with different speeds
                let r_angle = time * 0.8;
                let g_angle = time * 1.2 + 2.0;
                let b_angle = time * 1.0 + 4.0;
                
                // Red spotlight from above
                let _ = shadows.add_light(LightConfig {
                    light_type: LightType::Spot {
                        inner_angle: 25.0_f32.to_radians(),
                        outer_angle: 40.0_f32.to_radians(),
                    },
                    position: Vec3::new(r_angle.cos() * 3.0, 7.0, r_angle.sin() * 3.0),
                    direction: Vec3::new(0.0, -1.0, 0.0),
                    intensity: 1.5,
                    color: Vec3::new(1.0, 0.1, 0.1),
                    attenuation_radius: 12.0,
                    attenuation_falloff: 2.0,
                });
                
                // Green point light
                let _ = shadows.add_light(LightConfig {
                    light_type: LightType::Point,
                    position: Vec3::new(g_angle.cos() * 5.0, 3.0, g_angle.sin() * 5.0),
                    direction: Vec3::new(0.0, -1.0, 0.0),
                    intensity: 1.3,
                    color: Vec3::new(0.1, 1.0, 0.1),
                    attenuation_radius: 10.0,
                    attenuation_falloff: 2.5,
                });
                
                // Blue point light
                let _ = shadows.add_light(LightConfig {
                    light_type: LightType::Point,
                    position: Vec3::new(b_angle.cos() * 4.0, 4.0, b_angle.sin() * 4.0),
                    direction: Vec3::new(0.0, -1.0, 0.0),
                    intensity: 1.3,
                    color: Vec3::new(0.1, 0.1, 1.0),
                    attenuation_radius: 10.0,
                    attenuation_falloff: 2.5,
                });
                
                // Cyan accent light
                let _ = shadows.add_light(LightConfig {
                    light_type: LightType::Point,
                    position: Vec3::new((time * 1.5).cos() * 2.0, 2.0, (time * 1.5).sin() * 2.0),
                    direction: Vec3::new(0.0, -1.0, 0.0),
                    intensity: 0.8,
                    color: Vec3::new(0.3, 1.0, 1.0),
                    attenuation_radius: 6.0,
                    attenuation_falloff: 3.0,
                });
            }
            2 => {
                // Spotlight array - 6 spotlights in a grid pattern
                let colors = [
                    Vec3::new(1.0, 0.3, 0.3),  // Red
                    Vec3::new(1.0, 0.8, 0.2),  // Orange
                    Vec3::new(0.3, 1.0, 0.3),  // Green
                    Vec3::new(0.3, 0.8, 1.0),  // Cyan
                    Vec3::new(0.4, 0.3, 1.0),  // Blue
                    Vec3::new(1.0, 0.3, 0.8),  // Magenta
                ];
                
                let wave = (time * 2.0).sin() * 0.3 + 1.0;
                
                for i in 0..6 {
                    let angle = (i as f32 / 6.0) * std::f32::consts::TAU;
                    let phase_offset = i as f32 * 0.5;
                    let height_wave = ((time * 1.5 + phase_offset).sin() * 2.0 + 7.0).max(5.0);
                    
                    let _ = shadows.add_light(LightConfig {
                        light_type: LightType::Spot {
                            inner_angle: 20.0_f32.to_radians(),
                            outer_angle: 35.0_f32.to_radians(),
                        },
                        position: Vec3::new(
                            angle.cos() * 6.0,
                            height_wave,
                            angle.sin() * 6.0
                        ),
                        direction: Vec3::new(-angle.cos() * 0.3, -1.0, -angle.sin() * 0.3).normalize(),
                        intensity: wave + (i as f32 * 0.1),
                        color: colors[i],
                        attenuation_radius: 15.0,
                        attenuation_falloff: 2.0,
                    });
                }
            }
            3 => {
                // Color party - 8 overlapping lights with pulsing colors
                for i in 0..8 {
                    let angle = (i as f32 / 8.0) * std::f32::consts::TAU + time * 0.3;
                    let height = ((time * 2.0 + i as f32).sin() * 1.5 + 4.0).max(2.0);
                    let radius = 3.0 + i as f32 * 0.3;
                    
                    // Create rainbow colors with time variation
                    let hue = (i as f32 / 8.0 + time * 0.2) % 1.0;
                    let color = Self::hue_to_rgb(hue);
                    
                    let light_type = if i % 3 == 0 {
                        LightType::Spot {
                            inner_angle: 30.0_f32.to_radians(),
                            outer_angle: 45.0_f32.to_radians(),
                        }
                    } else {
                        LightType::Point
                    };
                    
                    let _ = shadows.add_light(LightConfig {
                        light_type,
                        position: Vec3::new(angle.cos() * radius, height, angle.sin() * radius),
                        direction: Vec3::new(0.0, -1.0, 0.0),
                        intensity: 1.0 + (time * 3.0 + i as f32).sin() * 0.5,
                        color,
                        attenuation_radius: 8.0 + (i as f32 * 0.5),
                        attenuation_falloff: 2.0 + (time * 0.5).sin().abs(),
                    });
                }
            }
            _ => {}
        }
    }
    
    // Convert HSV to RGB for rainbow colors
    fn hue_to_rgb(hue: f32) -> Vec3 {
        let h = hue * 6.0;
        let x = 1.0 - (h % 2.0 - 1.0).abs();
        
        let (r, g, b) = if h < 1.0 {
            (1.0, x, 0.0)
        } else if h < 2.0 {
            (x, 1.0, 0.0)
        } else if h < 3.0 {
            (0.0, 1.0, x)
        } else if h < 4.0 {
            (0.0, x, 1.0)
        } else if h < 5.0 {
            (x, 0.0, 1.0)
        } else {
            (1.0, 0.0, x)
        };
        
        Vec3::new(r, g, b)
    }

    fn update_light_type(&mut self) {
        println!("\n=== Demo Mode {} ===", self.demo_mode);
        match self.demo_mode {
            0 => println!("Single Rotating Spotlight - Watch it circle and pulse!"),
            1 => println!("RGB Multi-Light Dance - Multiple colored lights with overlapping shadows!"),
            2 => println!("Spotlight Array - 6 spotlights in formation with wave motion!"),
            3 => println!("Color Party - 8 lights with rainbow colors and dynamic intensity!"),
            _ => {}
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
        let elapsed = (now - self.start_time).as_secs_f32();

        // Ground plane - larger and more interesting
        let ground_transform = Mat4::from_translation(Vec3::new(0.0, -0.1, 0.0)) *
            Mat4::from_scale(Vec3::new(1.5, 1.0, 1.5));
        meshes.push((
            TransformUniforms {
                model: ground_transform.to_cols_array_2d(),
            },
            self.plane_vertices.into(),
            self.plane_indices.into(),
            self.plane_index_count,
        ));

        // Central rotating pillar with spheres
        for i in 0..5 {
            let height = i as f32 * 1.5;
            let angle = elapsed * 0.5 + i as f32 * 0.6;
            let radius = 2.0 + (elapsed * 0.3 + i as f32).sin() * 0.5;
            
            let sphere_pos = Mat4::from_translation(Vec3::new(
                angle.cos() * radius,
                height + 1.0,
                angle.sin() * radius
            )) * Mat4::from_scale(Vec3::splat(0.8 + (elapsed * 2.0 + i as f32).sin().abs() * 0.3));
            
            meshes.push((
                TransformUniforms {
                    model: sphere_pos.to_cols_array_2d(),
                },
                self.sphere_vertices.into(),
                self.sphere_indices.into(),
                self.sphere_index_count,
            ));
        }

        // Orbiting cubes at different heights
        for i in 0..8 {
            let orbit_angle = elapsed * 0.8 + (i as f32 / 8.0) * std::f32::consts::TAU;
            let orbit_radius = 6.0;
            let height = 2.0 + (elapsed * 1.5 + i as f32).sin() * 2.0;
            let rotation_speed = 1.0 + i as f32 * 0.2;
            
            let cube_transform = Mat4::from_translation(Vec3::new(
                orbit_angle.cos() * orbit_radius,
                height,
                orbit_angle.sin() * orbit_radius
            )) * Mat4::from_rotation_y(elapsed * rotation_speed) * 
            Mat4::from_rotation_x(elapsed * rotation_speed * 0.7) *
            Mat4::from_scale(Vec3::splat(0.6));
            
            meshes.push((
                TransformUniforms {
                    model: cube_transform.to_cols_array_2d(),
                },
                self.cube_vertices.into(),
                self.cube_indices.into(),
                self.cube_index_count,
            ));
        }

        // Dancing spheres on the ground
        for i in 0..12 {
            let dance_angle = (i as f32 / 12.0) * std::f32::consts::TAU;
            let dance_radius = 4.0 + (elapsed * 0.5).sin() * 1.0;
            let bounce_height = ((elapsed * 3.0 + i as f32).sin().abs() * 2.0 + 0.5).max(0.5);
            
            let sphere_transform = Mat4::from_translation(Vec3::new(
                dance_angle.cos() * dance_radius,
                bounce_height,
                dance_angle.sin() * dance_radius
            )) * Mat4::from_scale(Vec3::splat(0.4 + (elapsed + i as f32).cos().abs() * 0.2));
            
            meshes.push((
                TransformUniforms {
                    model: sphere_transform.to_cols_array_2d(),
                },
                self.sphere_vertices.into(),
                self.sphere_indices.into(),
                self.sphere_index_count,
            ));
        }

        // Spinning double helix of cubes
        for i in 0..16 {
            let helix_height = i as f32 * 0.6;
            let helix_angle1 = elapsed * 2.0 + i as f32 * 0.4;
            let helix_angle2 = helix_angle1 + std::f32::consts::PI;
            let helix_radius = 2.5;
            
            // First helix strand
            let helix1 = Mat4::from_translation(Vec3::new(
                helix_angle1.cos() * helix_radius,
                helix_height + 0.5,
                helix_angle1.sin() * helix_radius
            )) * Mat4::from_rotation_y(elapsed * 3.0) * Mat4::from_scale(Vec3::splat(0.3));
            
            meshes.push((
                TransformUniforms {
                    model: helix1.to_cols_array_2d(),
                },
                self.cube_vertices.into(),
                self.cube_indices.into(),
                self.cube_index_count,
            ));
            
            // Second helix strand
            let helix2 = Mat4::from_translation(Vec3::new(
                helix_angle2.cos() * helix_radius,
                helix_height + 0.5,
                helix_angle2.sin() * helix_radius
            )) * Mat4::from_rotation_y(elapsed * 3.0) * Mat4::from_scale(Vec3::splat(0.3));
            
            meshes.push((
                TransformUniforms {
                    model: helix2.to_cols_array_2d(),
                },
                self.cube_vertices.into(),
                self.cube_indices.into(),
                self.cube_index_count,
            ));
        }

        // Pulsing towers at corners
        let tower_positions = [
            Vec3::new(-8.0, 0.0, -8.0),
            Vec3::new(8.0, 0.0, -8.0),
            Vec3::new(-8.0, 0.0, 8.0),
            Vec3::new(8.0, 0.0, 8.0),
        ];
        
        for (idx, pos) in tower_positions.iter().enumerate() {
            let pulse = (elapsed * 2.0 + idx as f32 * 1.5).sin() * 2.0 + 3.0;
            let tower = Mat4::from_translation(*pos + Vec3::new(0.0, pulse / 2.0, 0.0)) *
                Mat4::from_scale(Vec3::new(0.8, pulse, 0.8));
            
            meshes.push((
                TransformUniforms {
                    model: tower.to_cols_array_2d(),
                },
                self.cube_vertices.into(),
                self.cube_indices.into(),
                self.cube_index_count,
            ));
        }

        // Rotating ring of spheres
        for i in 0..20 {
            let ring_angle = (i as f32 / 20.0) * std::f32::consts::TAU + elapsed * 1.5;
            let ring_radius = 10.0;
            let ring_height = 3.0 + (ring_angle * 2.0).sin() * 1.0;
            
            let sphere = Mat4::from_translation(Vec3::new(
                ring_angle.cos() * ring_radius,
                ring_height,
                ring_angle.sin() * ring_radius
            )) * Mat4::from_scale(Vec3::splat(0.5));
            
            meshes.push((
                TransformUniforms {
                    model: sphere.to_cols_array_2d(),
                },
                self.sphere_vertices.into(),
                self.sphere_indices.into(),
                self.sphere_index_count,
            ));
        }


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
                                    winit::keyboard::KeyCode::Equal | winit::keyboard::KeyCode::NumpadAdd => {
                                        // Brighten lights by 20%
                                        app.adjust_light_brightness(1.2);
                                    }
                                    winit::keyboard::KeyCode::Minus | winit::keyboard::KeyCode::NumpadSubtract => {
                                        // Darken lights by 20%
                                        app.adjust_light_brightness(0.8);
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
