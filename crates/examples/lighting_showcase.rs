use glam::{Mat4, Vec3};
use helio_core::{create_cube_mesh, create_plane_mesh, create_sphere_mesh, MeshBuffer, TextureManager};
use helio_feature_base_geometry::BaseGeometry;
use helio_feature_billboards::BillboardFeature;
use helio_feature_bloom::Bloom;
use helio_feature_materials::BasicMaterials;
use helio_feature_radiance_cascades::{LightConfig, LightType, RadianceCascades, GIQuality, RadianceCascadesConfig};
use helio_features::FeatureRegistry;
use helio_render::{FpsCamera, FeatureRenderer, TransformUniforms};
use std::{collections::HashSet, sync::Arc, time::Instant};
use wgpu::SurfaceTarget;

struct Example {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    renderer: FeatureRenderer,
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
    fn new(window: Arc<winit::window::Window>) -> Self {
        let window_size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // Safety: window is Arc so its lifetime is managed; we keep arc alive in struct
        let surface = instance.create_surface(SurfaceTarget::from(window)).unwrap();

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })).expect("No suitable GPU adapter found");

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("helio_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        )).expect("Failed to create device");

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let caps = surface.get_capabilities(&adapter);
        let surface_format = caps.formats.iter().copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: window_size.width,
            height: window_size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        let cube_mesh = MeshBuffer::from_mesh(&device, "cube", &create_cube_mesh(1.0));
        let sphere_mesh = MeshBuffer::from_mesh(&device, "sphere", &create_sphere_mesh(0.5, 32, 32));
        let plane_mesh = MeshBuffer::from_mesh(&device, "plane", &create_plane_mesh(20.0, 20.0));

        let mut texture_manager = TextureManager::new(device.clone(), queue.clone());
        let spotlight_texture_id = match texture_manager.load_png("spotlight.png") {
            Ok(id) => { log::info!("Loaded spotlight.png"); Some(id) }
            Err(e) => { log::warn!("Failed to load spotlight.png: {}", e); None }
        };
        let texture_manager = Arc::new(texture_manager);

        let mut base_geometry = BaseGeometry::new();
        base_geometry.set_texture_manager(texture_manager.clone());
        let base_shader = base_geometry.shader_template().to_string();

        let mut radiance_cascades = RadianceCascades::with_config(
            RadianceCascadesConfig::new()
                .with_quality(GIQuality::Medium)
                .with_gi_intensity(1.0)
                .with_ambient(0.02)
        );
        // Initial lights — these get replaced every frame by update_demo_lights
        radiance_cascades.add_light(LightConfig {
            light_type: LightType::Spot { inner_angle: 25.0_f32.to_radians(), outer_angle: 40.0_f32.to_radians() },
            position: Vec3::new(0.0, 8.0, 0.0), direction: Vec3::new(0.0, -1.0, 0.0),
            intensity: 1.5, color: Vec3::new(1.0, 0.2, 0.2), attenuation_radius: 12.0, attenuation_falloff: 2.0,
        }).expect("Failed to add light");

        let mut billboards = BillboardFeature::new();
        billboards.set_texture_manager(texture_manager.clone());

        let registry = FeatureRegistry::builder()
            .with_feature(base_geometry)
            .with_feature(BasicMaterials::new())
            .with_feature(radiance_cascades)
            .with_feature(Bloom::new())
            .with_feature(billboards)
            .debug_output(true)
            .build();

        let renderer = FeatureRenderer::new(
            device.clone(), queue.clone(),
            surface_format, window_size.width, window_size.height,
            registry, &base_shader,
        ).expect("Failed to create renderer");

        let now = Instant::now();
        let mut camera = FpsCamera::new(Vec3::new(0.0, 5.0, 15.0));
        camera.pitch = -20.0_f32.to_radians();

        Self {
            device, queue, surface, surface_config, renderer, window_size,
            cube_mesh, sphere_mesh, plane_mesh,
            start_time: now, last_frame_time: now,
            camera, keys_pressed: HashSet::new(), cursor_grabbed: false, demo_mode: 1,
        }
    }

    fn adjust_light_brightness(&mut self, factor: f32) {
        if let Some(rc) = self.renderer.registry_mut().get_feature_as_mut::<RadianceCascades>("radiance_cascades") {
            for light in rc.lights_mut() { light.intensity *= factor; }
            println!("Light brightness adjusted by {}x", factor);
        }
    }

    fn update_demo_lights(&mut self, time: f32) {
        let Some(rc) = self.renderer.registry_mut().get_feature_as_mut::<RadianceCascades>("radiance_cascades") else { return };
        rc.clear_lights();
        match self.demo_mode {
            0 => {
                // Single slow overhead spotlight — best for observing shadow shapes
                let a = time * 0.25;
                let _ = rc.add_light(LightConfig {
                    light_type: LightType::Spot { inner_angle: 18.0_f32.to_radians(), outer_angle: 32.0_f32.to_radians() },
                    position: Vec3::new(a.cos() * 4.0, 10.0, a.sin() * 4.0),
                    direction: Vec3::new(-a.cos() * 0.3, -1.0, -a.sin() * 0.3).normalize(),
                    intensity: 120.0, color: Vec3::new(1.0, 0.95, 0.82),
                    attenuation_radius: 18.0, attenuation_falloff: 2.0,
                });
            }
            1 => {
                // 4 large coloured point lights, one per quadrant.
                // With UE4 inverse-square falloff, each dominates only its quadrant.
                // NE: red, NW: green, SE: blue, SW: amber
                let corners = [
                    (Vec3::new( 7.5, 2.5,  7.5), Vec3::new(1.0, 0.08, 0.08)),
                    (Vec3::new(-7.5, 2.5,  7.5), Vec3::new(0.08, 1.0, 0.08)),
                    (Vec3::new( 7.5, 2.5, -7.5), Vec3::new(0.08, 0.18, 1.0)),
                    (Vec3::new(-7.5, 2.5, -7.5), Vec3::new(1.0, 0.72, 0.05)),
                ];
                for (pos, color) in corners {
                    let _ = rc.add_light(LightConfig {
                        light_type: LightType::Point,
                        position: pos, direction: Vec3::NEG_Y,
                        intensity: 10.0, color,
                        attenuation_radius: 9.0, attenuation_falloff: 2.0,
                    });
                }
            }
            2 => {
                // Same 4 quadrant lights, now slowly orbiting their pillar — shows dynamic shadows
                let corners = [
                    (Vec3::new( 7.5, 2.5,  7.5), Vec3::new(1.0, 0.08, 0.08)),
                    (Vec3::new(-7.5, 2.5,  7.5), Vec3::new(0.08, 1.0, 0.08)),
                    (Vec3::new( 7.5, 2.5, -7.5), Vec3::new(0.08, 0.18, 1.0)),
                    (Vec3::new(-7.5, 2.5, -7.5), Vec3::new(1.0, 0.72, 0.05)),
                ];
                for (i, (base, color)) in corners.into_iter().enumerate() {
                    let a = time * 0.7 + i as f32 * std::f32::consts::FRAC_PI_2;
                    let pos = Vec3::new(base.x + a.cos() * 2.0,
                                       2.5 + (time * 1.3 + i as f32).sin() * 1.0,
                                       base.z + a.sin() * 2.0);
                    let pulse = 1.0 + (time * 2.5 + i as f32).sin() * 0.25;
                    let _ = rc.add_light(LightConfig {
                        light_type: LightType::Point,
                        position: pos, direction: Vec3::NEG_Y,
                        intensity: 10.0 * pulse, color,
                        attenuation_radius: 9.0, attenuation_falloff: 2.0,
                    });
                }
            }
            3 => {
                // 8 theatrical coloured spotlights from high above, aimed at the floor
                let colors = [
                    Vec3::new(1.0, 0.08, 0.08), Vec3::new(1.0, 0.55, 0.0),
                    Vec3::new(0.9, 0.9, 0.05),   Vec3::new(0.08, 1.0, 0.08),
                    Vec3::new(0.0, 0.85, 0.85),  Vec3::new(0.08, 0.18, 1.0),
                    Vec3::new(0.6, 0.0, 1.0),    Vec3::new(1.0, 0.08, 0.6),
                ];
                for i in 0..8usize {
                    let a = (i as f32 / 8.0) * std::f32::consts::TAU + time * 0.18;
                    let lpos = Vec3::new(a.cos() * 5.0, 10.0, a.sin() * 5.0);
                    let target = Vec3::new(
                        (a + 0.5).cos() * 4.0 + (time * 0.5 + i as f32).cos() * 1.5,
                        0.0,
                        (a + 0.5).sin() * 4.0 + (time * 0.5 + i as f32).sin() * 1.5,
                    );
                    let dir = (target - lpos).normalize();
                    let pulse = 1.0 + (time * 2.8 + i as f32 * 0.9).sin() * 0.2;
                    let _ = rc.add_light(LightConfig {
                        light_type: LightType::Spot { inner_angle: 8.0_f32.to_radians(), outer_angle: 20.0_f32.to_radians() },
                        position: lpos, direction: dir,
                        intensity: 200.0 * pulse, color: colors[i],
                        attenuation_radius: 16.0, attenuation_falloff: 2.0,
                    });
                }
            }
            _ => {}
        }
    }

    fn update_light_type(&mut self) {
        println!("\n=== Demo Mode {} ===", self.demo_mode);
        match self.demo_mode {
            0 => println!("Single Rotating Spotlight"),
            1 => println!("4 Coloured Quadrant Point Lights (static)"),
            2 => println!("4 Coloured Quadrant Point Lights (dynamic)"),
            3 => println!("8 Theatrical Coloured Spotlights"),
            _ => {}
        }
    }

    fn render(&mut self) {
        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.surface.configure(&self.device, &self.surface_config);
                return;
            }
            Err(e) => { log::error!("Surface error: {:?}", e); return; }
        };

        let now = Instant::now();
        let delta_time = (now - self.last_frame_time).as_secs_f32();
        self.last_frame_time = now;

        use winit::keyboard::KeyCode;
        let fwd = self.keys_pressed.contains(&KeyCode::KeyW) as i8 - self.keys_pressed.contains(&KeyCode::KeyS) as i8;
        let rgt = self.keys_pressed.contains(&KeyCode::KeyD) as i8 - self.keys_pressed.contains(&KeyCode::KeyA) as i8;
        let up  = self.keys_pressed.contains(&KeyCode::Space) as i8 - self.keys_pressed.contains(&KeyCode::ShiftLeft) as i8;
        self.camera.update_movement(fwd as f32, rgt as f32, up as f32, delta_time);

        let aspect = self.window_size.width as f32 / self.window_size.height as f32;
        let elapsed = (now - self.start_time).as_secs_f32();
        let camera = self.camera.build_camera_uniforms(60.0, aspect, elapsed);
        self.update_demo_lights(elapsed);

        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });

        let mut meshes: Vec<(TransformUniforms, &MeshBuffer)> = Vec::new();

        // Floor
        let ground = Mat4::from_translation(Vec3::new(0.0, -0.05, 0.0)) * Mat4::from_scale(Vec3::new(1.2, 1.0, 1.2));
        meshes.push((TransformUniforms::from_matrix(ground), &self.plane_mesh));

        // 4 corner pillars — mark the light zones and cast strong shadows
        for (sx, sz) in [(-1.0f32, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)] {
            let t = Mat4::from_translation(Vec3::new(sx * 9.0, 4.0, sz * 9.0))
                * Mat4::from_scale(Vec3::new(1.0, 8.0, 1.0));
            meshes.push((TransformUniforms::from_matrix(t), &self.cube_mesh));
        }

        // Low pedestals at quadrant centres for objects to rest on
        for (sx, sz) in [(-1.0f32, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)] {
            let t = Mat4::from_translation(Vec3::new(sx * 6.0, 0.75, sz * 6.0))
                * Mat4::from_scale(Vec3::new(1.4, 1.5, 1.4));
            meshes.push((TransformUniforms::from_matrix(t), &self.cube_mesh));
        }

        // Central tower — lit by whichever light reaches the centre
        for i in 0..5usize {
            let h = 0.6 + i as f32 * 1.4;
            let s = 0.7 - i as f32 * 0.07;
            let t = Mat4::from_translation(Vec3::new(0.0, h, 0.0))
                * Mat4::from_rotation_y(elapsed * 0.4 + i as f32 * 0.5)
                * Mat4::from_scale(Vec3::splat(s));
            meshes.push((TransformUniforms::from_matrix(t), if i % 2 == 0 { &self.sphere_mesh } else { &self.cube_mesh }));
        }

        // Per-quadrant orbiting objects — each cluster sits under its coloured light
        let quads: [(f32, f32, f32); 4] = [
            ( 6.0,  6.0, 0.0),
            (-6.0,  6.0, 0.8),
            ( 6.0, -6.0, 1.6),
            (-6.0, -6.0, 2.4),
        ];
        for (qx, qz, phase) in quads {
            // Sphere sitting on pedestal
            let sit = Mat4::from_translation(Vec3::new(qx, 2.0, qz))
                * Mat4::from_scale(Vec3::splat(0.75));
            meshes.push((TransformUniforms::from_matrix(sit), &self.sphere_mesh));
            // Three small cubes orbiting the pedestal at different heights
            for j in 0..3usize {
                let a = elapsed * (0.5 + j as f32 * 0.15) + phase + (j as f32 / 3.0) * std::f32::consts::TAU;
                let r = 2.0 + j as f32 * 0.3;
                let h = 1.2 + j as f32 * 0.8;
                let t = Mat4::from_translation(Vec3::new(qx + a.cos() * r, h, qz + a.sin() * r))
                    * Mat4::from_rotation_y(elapsed * 2.0 + j as f32)
                    * Mat4::from_scale(Vec3::splat(0.38));
                meshes.push((TransformUniforms::from_matrix(t), &self.cube_mesh));
            }
            // One extra sphere bouncing in the quadrant
            let bounce_a = elapsed * 1.2 + phase;
            let bounce_h = ((elapsed * 2.5 + phase).sin().abs() * 2.5 + 0.4).max(0.4);
            let bt = Mat4::from_translation(Vec3::new(qx + bounce_a.cos() * 2.8, bounce_h, qz + bounce_a.sin() * 2.8))
                * Mat4::from_scale(Vec3::splat(0.4));
            meshes.push((TransformUniforms::from_matrix(bt), &self.sphere_mesh));
        }

        // Mid-ring of objects between quadrants — straddle multiple light zones
        for i in 0..8usize {
            let a = (i as f32 / 8.0) * std::f32::consts::TAU + elapsed * 0.15;
            let h = ((elapsed * 1.8 + i as f32).sin() * 0.6 + 1.4).max(0.3);
            let t = Mat4::from_translation(Vec3::new(a.cos() * 4.2, h, a.sin() * 4.2))
                * Mat4::from_rotation_y(elapsed * 1.8 + i as f32 * 0.7)
                * Mat4::from_scale(Vec3::splat(0.45));
            meshes.push((TransformUniforms::from_matrix(t), if i % 2 == 0 { &self.sphere_mesh } else { &self.cube_mesh }));
        }

        self.renderer.render(&mut encoder, &view, camera, &meshes, delta_time);
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 { return; }
        self.window_size = new_size;
        self.surface_config.width = new_size.width;
        self.surface_config.height = new_size.height;
        self.surface.configure(&self.device, &self.surface_config);
        self.renderer.resize(new_size.width, new_size.height);
    }
}

fn main() {
    env_logger::init();
    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    let window_attr = winit::window::Window::default_attributes()
        .with_title("Helio - Lighting Showcase | WASD: Move, Mouse: Look | 1-4: Toggle features | 5: Demo Mode | 6: Toggle GI | +/-: Brightness")
        .with_inner_size(winit::dpi::LogicalSize::new(1920u32, 1080u32));
    #[allow(deprecated)]
    let window = Arc::new(event_loop.create_window(window_attr).unwrap());
    let mut app = Example::new(window.clone());
    let _ = window.set_cursor_grab(winit::window::CursorGrabMode::Confined);
    window.set_cursor_visible(false);
    app.cursor_grabbed = true;

    #[allow(deprecated)]
    event_loop.run(move |event, elwt| match event {
        winit::event::Event::WindowEvent { event, .. } => match event {
            winit::event::WindowEvent::CloseRequested => elwt.exit(),
            winit::event::WindowEvent::KeyboardInput { event: winit::event::KeyEvent { physical_key, state, .. }, .. } => {
                if let winit::keyboard::PhysicalKey::Code(kc) = physical_key {
                    match state {
                        winit::event::ElementState::Pressed => {
                            app.keys_pressed.insert(kc);
                            match kc {
                                winit::keyboard::KeyCode::Escape => {
                                    if app.cursor_grabbed { let _ = window.set_cursor_grab(winit::window::CursorGrabMode::None); window.set_cursor_visible(true); app.cursor_grabbed = false; }
                                    else { elwt.exit(); }
                                }
                                winit::keyboard::KeyCode::Digit1 => { if let Ok(on) = app.renderer.toggle_and_rebuild("base_geometry") { println!("[1] Base Geometry: {}", if on{"ON"}else{"OFF"}); } }
                                winit::keyboard::KeyCode::Digit2 => { if let Ok(on) = app.renderer.toggle_and_rebuild("basic_materials") { println!("[2] Materials: {}", if on{"ON"}else{"OFF"}); } }
                                winit::keyboard::KeyCode::Digit3 => { if let Ok(on) = app.renderer.toggle_and_rebuild("bloom") { println!("[3] Bloom: {}", if on{"ON"}else{"OFF"}); } }
                                winit::keyboard::KeyCode::Digit4 => { if let Ok(on) = app.renderer.toggle_and_rebuild("radiance_cascades") { println!("[4] GI: {}", if on{"ON"}else{"OFF"}); } }
                                winit::keyboard::KeyCode::Digit5 => { app.demo_mode = (app.demo_mode + 1) % 4; app.update_light_type(); }
                                winit::keyboard::KeyCode::Digit6 => { if let Ok(on) = app.renderer.toggle_and_rebuild("radiance_cascades") { println!("[6] Radiance Cascades GI: {}", if on{"ON"}else{"OFF"}); } }
                                winit::keyboard::KeyCode::Equal | winit::keyboard::KeyCode::NumpadAdd => app.adjust_light_brightness(1.2),
                                winit::keyboard::KeyCode::Minus | winit::keyboard::KeyCode::NumpadSubtract => app.adjust_light_brightness(0.8),
                                _ => {}
                            }
                        }
                        winit::event::ElementState::Released => { app.keys_pressed.remove(&kc); }
                    }
                }
            }
            winit::event::WindowEvent::MouseInput { button: winit::event::MouseButton::Left, state: winit::event::ElementState::Pressed, .. } => {
                if !app.cursor_grabbed { let _ = window.set_cursor_grab(winit::window::CursorGrabMode::Confined); window.set_cursor_visible(false); app.cursor_grabbed = true; }
            }
            winit::event::WindowEvent::Resized(sz) => app.resize(sz),
            winit::event::WindowEvent::RedrawRequested => { app.render(); window.request_redraw(); }
            _ => {}
        },
        winit::event::Event::DeviceEvent { event, .. } => {
            if app.cursor_grabbed { if let winit::event::DeviceEvent::MouseMotion { delta } = event { app.camera.handle_mouse_delta(delta.0 as f32, delta.1 as f32); } }
        }
        winit::event::Event::AboutToWait => window.request_redraw(),
        _ => {}
    }).unwrap();
}
