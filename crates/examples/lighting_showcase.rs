use glam::{Mat4, Vec3};
use helio_core::{create_cube_mesh, create_plane_mesh, create_sphere_mesh, MeshBuffer};
use helio_feature_base_geometry::BaseGeometry;
use helio_feature_billboards::BillboardFeature;
use helio_feature_bloom::Bloom;
use helio_feature_lighting::BasicLighting;
use helio_feature_materials::BasicMaterials;
use helio_feature_procedural_shadows::{LightConfig, LightType, ProceduralShadows};
use helio_features::FeatureRegistry;
use helio_render::{FpsCamera, FeatureRenderer, TransformUniforms};
use std::{collections::HashSet, sync::Arc, time::Instant};

struct Example {
    context: Arc<blade_graphics::Context>,
    surface: blade_graphics::Surface,
    renderer: FeatureRenderer,
    command_encoder: blade_graphics::CommandEncoder,
    window_size: winit::dpi::PhysicalSize<u32>,

    cube_mesh: MeshBuffer,
    sphere_mesh: MeshBuffer,
    plane_mesh: MeshBuffer,

    start_time: Instant,
    last_frame_time: Instant,
    camera: FpsCamera,
    keys_pressed: HashSet<winit::keyboard::KeyCode>,
    cursor_grabbed: bool,
    demo_mode: usize,
}

impl Example {
    fn make_surface_config(size: winit::dpi::PhysicalSize<u32>) -> blade_graphics::SurfaceConfig {
        blade_graphics::SurfaceConfig {
            size: blade_graphics::Extent {
                width: size.width,
                height: size.height,
                depth: 1,
            },
            usage: blade_graphics::TextureUsage::TARGET,
            display_sync: blade_graphics::DisplaySync::Recent,
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
        use blade_graphics as gpu;

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

        // Upload meshes — one line each instead of ~15 lines of unsafe buffer code
        let cube_mesh = MeshBuffer::from_mesh(&context, "cube", &create_cube_mesh(1.0));
        let sphere_mesh = MeshBuffer::from_mesh(&context, "sphere", &create_sphere_mesh(0.5, 32, 32));
        let plane_mesh = MeshBuffer::from_mesh(&context, "plane", &create_plane_mesh(20.0, 20.0));

        // Texture manager for billboard icons
        let mut texture_manager = helio_core::TextureManager::new(context.clone());
        let spotlight_texture_id = match texture_manager.load_png("spotlight.png") {
            Ok(id) => {
                log::info!("Loaded spotlight.png for light billboards");
                Some(id)
            }
            Err(e) => {
                log::warn!(
                    "Failed to load spotlight.png: {} - light billboards will not be available",
                    e
                );
                None
            }
        };
        let texture_manager = Arc::new(texture_manager);

        let mut base_geometry = BaseGeometry::new();
        base_geometry.set_texture_manager(texture_manager.clone());
        let base_shader = base_geometry.shader_template().to_string();

        // Start with an impressive multi-light setup
        let mut shadows = ProceduralShadows::new().with_ambient(0.0);
        shadows
            .add_light(LightConfig {
                light_type: LightType::Spot {
                    inner_angle: 25.0_f32.to_radians(),
                    outer_angle: 40.0_f32.to_radians(),
                },
                position: Vec3::new(0.0, 8.0, 0.0),
                direction: Vec3::new(0.0, -1.0, 0.0),
                intensity: 1.5,
                color: Vec3::new(1.0, 0.2, 0.2),
                attenuation_radius: 12.0,
                attenuation_falloff: 2.0,
            })
            .expect("Failed to add light");
        shadows
            .add_light(LightConfig {
                light_type: LightType::Point,
                position: Vec3::new(-4.0, 3.0, -4.0),
                direction: Vec3::new(0.0, -1.0, 0.0),
                intensity: 1.2,
                color: Vec3::new(0.2, 1.0, 0.2),
                attenuation_radius: 10.0,
                attenuation_falloff: 2.5,
            })
            .expect("Failed to add light");
        shadows
            .add_light(LightConfig {
                light_type: LightType::Point,
                position: Vec3::new(4.0, 3.0, -4.0),
                direction: Vec3::new(0.0, -1.0, 0.0),
                intensity: 1.2,
                color: Vec3::new(0.2, 0.2, 1.0),
                attenuation_radius: 10.0,
                attenuation_falloff: 2.5,
            })
            .expect("Failed to add light");
        shadows.set_texture_manager(texture_manager.clone());
        if let Some(texture_id) = spotlight_texture_id {
            shadows.set_spotlight_icon(texture_id);
        }

        let mut billboards = BillboardFeature::new();
        billboards.set_texture_manager(texture_manager.clone());

        let registry = FeatureRegistry::builder()
            .with_feature(base_geometry)
            .with_feature(BasicLighting::new())
            .with_feature(BasicMaterials::new())
            .with_feature(shadows)
            .with_feature(Bloom::new())
            .with_feature(billboards)
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

        let command_encoder = context.create_command_encoder(blade_graphics::CommandEncoderDesc {
            name: "main",
            buffer_count: 2,
        });

        let now = Instant::now();
        let mut camera = FpsCamera::new(Vec3::new(0.0, 5.0, 15.0));
        camera.pitch = -20.0_f32.to_radians();

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
            camera,
            keys_pressed: HashSet::new(),
            cursor_grabbed: false,
            demo_mode: 1,
        }
    }

    fn update_demo_lights(&mut self, time: f32) {
        let Some(shadows) = self
            .renderer
            .registry_mut()
            .get_feature_as_mut::<ProceduralShadows>("procedural_shadows")
        else {
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
                let _ = shadows.add_light(LightConfig {
                    light_type: LightType::Point,
                    position: Vec3::new(g_angle.cos() * 5.0, 3.0, g_angle.sin() * 5.0),
                    direction: Vec3::new(0.0, -1.0, 0.0),
                    intensity: 1.3,
                    color: Vec3::new(0.1, 1.0, 0.1),
                    attenuation_radius: 10.0,
                    attenuation_falloff: 2.5,
                });
                let _ = shadows.add_light(LightConfig {
                    light_type: LightType::Point,
                    position: Vec3::new(b_angle.cos() * 4.0, 4.0, b_angle.sin() * 4.0),
                    direction: Vec3::new(0.0, -1.0, 0.0),
                    intensity: 1.3,
                    color: Vec3::new(0.1, 0.1, 1.0),
                    attenuation_radius: 10.0,
                    attenuation_falloff: 2.5,
                });
                let _ = shadows.add_light(LightConfig {
                    light_type: LightType::Point,
                    position: Vec3::new(
                        (time * 1.5).cos() * 2.0,
                        2.0,
                        (time * 1.5).sin() * 2.0,
                    ),
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
                    Vec3::new(1.0, 0.3, 0.3),
                    Vec3::new(1.0, 0.8, 0.2),
                    Vec3::new(0.3, 1.0, 0.3),
                    Vec3::new(0.3, 0.8, 1.0),
                    Vec3::new(0.4, 0.3, 1.0),
                    Vec3::new(1.0, 0.3, 0.8),
                ];
                let wave = (time * 2.0).sin() * 0.3 + 1.0;
                for i in 0..6 {
                    let angle = (i as f32 / 6.0) * std::f32::consts::TAU;
                    let phase_offset = i as f32 * 0.5;
                    let height_wave =
                        ((time * 1.5 + phase_offset).sin() * 2.0 + 7.0).max(5.0);
                    let _ = shadows.add_light(LightConfig {
                        light_type: LightType::Spot {
                            inner_angle: 20.0_f32.to_radians(),
                            outer_angle: 35.0_f32.to_radians(),
                        },
                        position: Vec3::new(
                            angle.cos() * 6.0,
                            height_wave,
                            angle.sin() * 6.0,
                        ),
                        direction: Vec3::new(
                            -angle.cos() * 0.3,
                            -1.0,
                            -angle.sin() * 0.3,
                        )
                        .normalize(),
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
                    let angle =
                        (i as f32 / 8.0) * std::f32::consts::TAU + time * 0.3;
                    let height =
                        ((time * 2.0 + i as f32).sin() * 1.5 + 4.0).max(2.0);
                    let radius = 3.0 + i as f32 * 0.3;
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
                        position: Vec3::new(
                            angle.cos() * radius,
                            height,
                            angle.sin() * radius,
                        ),
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
            1 => println!(
                "RGB Multi-Light Dance - Multiple colored lights with overlapping shadows!"
            ),
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

        // Camera movement from key state
        use winit::keyboard::KeyCode;
        let fwd = self.keys_pressed.contains(&KeyCode::KeyW) as i8
            - self.keys_pressed.contains(&KeyCode::KeyS) as i8;
        let rgt = self.keys_pressed.contains(&KeyCode::KeyD) as i8
            - self.keys_pressed.contains(&KeyCode::KeyA) as i8;
        let up = self.keys_pressed.contains(&KeyCode::Space) as i8
            - self.keys_pressed.contains(&KeyCode::ShiftLeft) as i8;
        self.camera
            .update_movement(fwd as f32, rgt as f32, up as f32, delta_time);

        self.command_encoder.start();
        self.command_encoder.init_texture(frame.texture());

        let aspect = self.window_size.width as f32 / self.window_size.height as f32;
        let camera = self.camera.build_camera_uniforms(60.0, aspect);

        let elapsed = (now - self.start_time).as_secs_f32();
        self.update_demo_lights(elapsed);

        let mut meshes = Vec::new();

        // Ground plane
        let ground = Mat4::from_translation(Vec3::new(0.0, -0.1, 0.0))
            * Mat4::from_scale(Vec3::new(1.5, 1.0, 1.5));
        meshes.push((TransformUniforms::from_matrix(ground), &self.plane_mesh));

        // Central rotating pillar of spheres
        for i in 0..5 {
            let height = i as f32 * 1.5;
            let angle = elapsed * 0.5 + i as f32 * 0.6;
            let radius = 2.0 + (elapsed * 0.3 + i as f32).sin() * 0.5;
            let t = Mat4::from_translation(Vec3::new(
                angle.cos() * radius,
                height + 1.0,
                angle.sin() * radius,
            )) * Mat4::from_scale(Vec3::splat(
                0.8 + (elapsed * 2.0 + i as f32).sin().abs() * 0.3,
            ));
            meshes.push((TransformUniforms::from_matrix(t), &self.sphere_mesh));
        }

        // Orbiting cubes
        for i in 0..8 {
            let orbit_angle = elapsed * 0.8 + (i as f32 / 8.0) * std::f32::consts::TAU;
            let height = 2.0 + (elapsed * 1.5 + i as f32).sin() * 2.0;
            let rs = 1.0 + i as f32 * 0.2;
            let t = Mat4::from_translation(Vec3::new(
                orbit_angle.cos() * 6.0,
                height,
                orbit_angle.sin() * 6.0,
            )) * Mat4::from_rotation_y(elapsed * rs)
                * Mat4::from_rotation_x(elapsed * rs * 0.7)
                * Mat4::from_scale(Vec3::splat(0.6));
            meshes.push((TransformUniforms::from_matrix(t), &self.cube_mesh));
        }

        // Dancing spheres on the ground
        for i in 0..12 {
            let dance_angle = (i as f32 / 12.0) * std::f32::consts::TAU;
            let dance_radius = 4.0 + (elapsed * 0.5).sin() * 1.0;
            let bounce =
                ((elapsed * 3.0 + i as f32).sin().abs() * 2.0 + 0.5).max(0.5);
            let t = Mat4::from_translation(Vec3::new(
                dance_angle.cos() * dance_radius,
                bounce,
                dance_angle.sin() * dance_radius,
            )) * Mat4::from_scale(Vec3::splat(
                0.4 + (elapsed + i as f32).cos().abs() * 0.2,
            ));
            meshes.push((TransformUniforms::from_matrix(t), &self.sphere_mesh));
        }

        // Spinning double helix of cubes
        for i in 0..16 {
            let helix_height = i as f32 * 0.6;
            let a1 = elapsed * 2.0 + i as f32 * 0.4;
            let a2 = a1 + std::f32::consts::PI;
            let hr = 2.5;
            for a in [a1, a2] {
                let t = Mat4::from_translation(Vec3::new(
                    a.cos() * hr,
                    helix_height + 0.5,
                    a.sin() * hr,
                )) * Mat4::from_rotation_y(elapsed * 3.0)
                    * Mat4::from_scale(Vec3::splat(0.3));
                meshes.push((TransformUniforms::from_matrix(t), &self.cube_mesh));
            }
        }

        // Pulsing corner towers
        for (idx, pos) in [
            Vec3::new(-8.0, 0.0, -8.0),
            Vec3::new(8.0, 0.0, -8.0),
            Vec3::new(-8.0, 0.0, 8.0),
            Vec3::new(8.0, 0.0, 8.0),
        ]
        .iter()
        .enumerate()
        {
            let pulse = (elapsed * 2.0 + idx as f32 * 1.5).sin() * 2.0 + 3.0;
            let t = Mat4::from_translation(*pos + Vec3::new(0.0, pulse / 2.0, 0.0))
                * Mat4::from_scale(Vec3::new(0.8, pulse, 0.8));
            meshes.push((TransformUniforms::from_matrix(t), &self.cube_mesh));
        }

        // Rotating ring of spheres
        for i in 0..20 {
            let ring_angle = (i as f32 / 20.0) * std::f32::consts::TAU + elapsed * 1.5;
            let ring_height = 3.0 + (ring_angle * 2.0).sin() * 1.0;
            let t = Mat4::from_translation(Vec3::new(
                ring_angle.cos() * 10.0,
                ring_height,
                ring_angle.sin() * 10.0,
            )) * Mat4::from_scale(Vec3::splat(0.5));
            meshes.push((TransformUniforms::from_matrix(t), &self.sphere_mesh));
        }

        // A few extra standalone objects
        let t = Mat4::from_translation(Vec3::new(7.0, 1.5, 2.0))
            * Mat4::from_rotation_y(elapsed * 0.5);
        meshes.push((TransformUniforms::from_matrix(t), &self.cube_mesh));

        let outdoor_x = 8.0 + (elapsed * 0.8).sin() * 3.0;
        meshes.push((
            TransformUniforms::from_matrix(Mat4::from_translation(Vec3::new(
                outdoor_x, 2.0, 5.0,
            ))),
            &self.sphere_mesh,
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
        .with_title(
            "Helio - Lighting Showcase | WASD: Move, Mouse: Look | 1-4: Toggle features | +/-: Brightness",
        )
        .with_inner_size(winit::dpi::LogicalSize::new(1920, 1080));

    #[allow(deprecated)]
    let window = Arc::new(event_loop.create_window(window_attr).unwrap());
    let mut app = Example::new(&window);

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
                                app.keys_pressed.insert(keycode);
                                match keycode {
                                    winit::keyboard::KeyCode::Escape => {
                                        if app.cursor_grabbed {
                                            let _ = window.set_cursor_grab(
                                                winit::window::CursorGrabMode::None,
                                            );
                                            window.set_cursor_visible(true);
                                            app.cursor_grabbed = false;
                                        } else {
                                            elwt.exit();
                                        }
                                    }
                                    winit::keyboard::KeyCode::Digit1 => {
                                        if let Ok(on) =
                                            app.renderer.toggle_and_rebuild("base_geometry")
                                        {
                                            println!("[1] Base Geometry: {}", if on { "ON" } else { "OFF" });
                                        }
                                    }
                                    winit::keyboard::KeyCode::Digit2 => {
                                        if let Ok(on) =
                                            app.renderer.toggle_and_rebuild("basic_lighting")
                                        {
                                            println!("[2] Basic Lighting: {}", if on { "ON" } else { "OFF" });
                                        }
                                    }
                                    winit::keyboard::KeyCode::Digit3 => {
                                        if let Ok(on) =
                                            app.renderer.toggle_and_rebuild("basic_materials")
                                        {
                                            println!("[3] Basic Materials: {}", if on { "ON" } else { "OFF" });
                                        }
                                    }
                                    winit::keyboard::KeyCode::Digit4 => {
                                        if let Ok(on) =
                                            app.renderer.toggle_and_rebuild("procedural_shadows")
                                        {
                                            println!("[4] Shadows: {}", if on { "ON" } else { "OFF" });
                                        }
                                    }
                                    winit::keyboard::KeyCode::Digit5 => {
                                        app.demo_mode = (app.demo_mode + 1) % 4;
                                        app.update_light_type();
                                    }
                                    winit::keyboard::KeyCode::Equal
                                    | winit::keyboard::KeyCode::NumpadAdd => {
                                        app.adjust_light_brightness(1.2);
                                    }
                                    winit::keyboard::KeyCode::Minus
                                    | winit::keyboard::KeyCode::NumpadSubtract => {
                                        app.adjust_light_brightness(0.8);
                                    }
                                    _ => {}
                                }
                            }
                            winit::event::ElementState::Released => {
                                app.keys_pressed.remove(&keycode);
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
                        let _ = window
                            .set_cursor_grab(winit::window::CursorGrabMode::Confined);
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
                        app.camera
                            .handle_mouse_delta(delta.0 as f32, delta.1 as f32);
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
