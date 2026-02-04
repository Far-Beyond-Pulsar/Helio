use helio_core::{
    gpu, Camera, Scene, Transform, Entity, Vertex, create_cube_mesh, create_sphere_mesh, create_plane_mesh,
    MeshHandle, Mesh,
};
use helio_render::{Renderer, RendererConfig, RenderPath};
use helio_lighting::{LightingSystem, DirectionalLight, PointLight};
use glam::{Vec3, Vec4, Quat, Mat4};
use std::sync::Arc;
use std::collections::HashMap;
use bytemuck::{Pod, Zeroable};
use winit::{
    event::{Event, WindowEvent, KeyEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{PhysicalKey, KeyCode},
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ModelUniform {
    transform: [[f32; 4]; 4],
    color: [f32; 4],
}

#[derive(blade_macros::ShaderData)]
struct CameraData {
    camera_data: CameraUniform,
}

#[derive(blade_macros::ShaderData)]
struct ModelData {
    model_data: ModelUniform,
}

#[derive(blade_macros::Vertex, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct SimpleVertex {
    position: [f32; 3],
    normal: [f32; 3],
}

struct SimpleMesh {
    vertex_buffer: gpu::BufferPiece,
    index_buffer: gpu::BufferPiece,
    index_count: u32,
}

struct Application {
    window: Arc<winit::window::Window>,
    context: Arc<gpu::Context>,
    surface: gpu::Surface,
    renderer: Renderer,
    scene: Scene,
    lighting: LightingSystem,
    
    // Frame pacing
    frame_index: usize,
    command_encoder: gpu::CommandEncoder,
    prev_sync_point: Option<gpu::SyncPoint>,
    
    // Simple rendering pipeline
    render_pipeline: gpu::RenderPipeline,
    meshes: HashMap<MeshHandle, SimpleMesh>,
    mesh_list: Vec<Mesh>,
    
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
            display_sync: gpu::DisplaySync::Recent,
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

        let command_encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
            name: "main_encoder",
            buffer_count: 1,
        });

        // Create rendering pipeline
        let camera_layout = <CameraData as gpu::ShaderData>::layout();
        let model_layout = <ModelData as gpu::ShaderData>::layout();
        
        let shader_source = std::fs::read_to_string("shaders/simple_forward.wgsl")
            .expect("Failed to load shader");
        
        let shader = context.create_shader(gpu::ShaderDesc {
            source: &shader_source,
        });

        let render_pipeline = context.create_render_pipeline(gpu::RenderPipelineDesc {
            name: "simple_forward",
            data_layouts: &[&camera_layout, &model_layout],
            vertex: shader.at("vs_main"),
            vertex_fetches: &[gpu::VertexFetchState {
                layout: &<SimpleVertex as gpu::Vertex>::layout(),
                instanced: false,
            }],
            primitive: gpu::PrimitiveState {
                topology: gpu::PrimitiveTopology::TriangleList,
                front_face: gpu::FrontFace::Ccw,
                cull_mode: Some(gpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(gpu::DepthStencilState {
                format: gpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: gpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            fragment: Some(shader.at("fs_main")),
            color_targets: &[gpu::ColorTargetState {
                format: surface.info().format,
                blend: None,
                write_mask: gpu::ColorWrites::default(),
            }],
            multisample_state: gpu::MultisampleState::default(),
        });

        log::info!("Render pipeline created");

        // Upload meshes to GPU
        let mut meshes = HashMap::new();
        let mut mesh_list = vec![cube_mesh, sphere_mesh, plane_mesh];
        
        for (idx, mesh) in mesh_list.iter().enumerate() {
            let vertices: Vec<SimpleVertex> = mesh.vertices.iter().map(|v| SimpleVertex {
                position: v.position,
                normal: v.normal,
            }).collect();
            
            let vertex_data = bytemuck::cast_slice(&vertices);
            let index_data = bytemuck::cast_slice(&mesh.indices);
            
            let vertex_buffer = context.create_buffer(gpu::BufferDesc {
                name: "vertex_buffer",
                size: vertex_data.len() as u64,
                memory: gpu::Memory::Shared,
            });
            
            let index_buffer = context.create_buffer(gpu::BufferDesc {
                name: "index_buffer",
                size: index_data.len() as u64,
                memory: gpu::Memory::Shared,
            });
            
            unsafe {
                std::ptr::copy_nonoverlapping(
                    vertex_data.as_ptr(),
                    vertex_buffer.data(),
                    vertex_data.len(),
                );
                std::ptr::copy_nonoverlapping(
                    index_data.as_ptr(),
                    index_buffer.data(),
                    index_data.len(),
                );
            }
            context.sync_buffer(vertex_buffer);
            context.sync_buffer(index_buffer);
            
            meshes.insert(
                MeshHandle(idx as u32),
                SimpleMesh {
                    vertex_buffer: vertex_buffer.into(),
                    index_buffer: index_buffer.into(),
                    index_count: mesh.indices.len() as u32,
                }
            );
        }
        
        // Assign mesh handles to entities
        let entities: Vec<_> = scene.entities.keys().copied().collect();
        for (idx, entity_id) in entities.iter().enumerate() {
            if let Some(entity) = scene.entities.get_mut(entity_id) {
                entity.mesh = Some(MeshHandle(idx as u32));
            }
        }

        log::info!("Uploaded {} meshes to GPU", meshes.len());

        Self {
            window,
            context,
            surface,
            renderer,
            scene,
            lighting,
            frame_index: 0,
            command_encoder,
            prev_sync_point: None,
            render_pipeline,
            meshes,
            mesh_list,
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

        // Print FPS every 100 frames
        if self.frame_count % 100 == 0 {
            let recent_avg = self.frame_times.iter().rev().take(100).sum::<f32>() / 100.0;
            let fps = 1.0 / recent_avg;
            println!("FPS: {:.1} ({:.2}ms)", fps, recent_avg * 1000.0);
        }

        if self.frame_count % 10000 == 0 {
            let elapsed = self.fps_timer.elapsed().as_secs_f32();
            let avg_fps = 10000.0 / elapsed;
            let avg_frame_time = elapsed / 10000.0 * 1000.0;
            
            let min_frame_time = self.frame_times.iter().copied().fold(f32::INFINITY, f32::min) * 1000.0;
            let max_frame_time = self.frame_times.iter().copied().fold(0.0f32, f32::max) * 1000.0;
            
            println!("=== 10k Frame Stats ===");
            println!("Average: {:.1} FPS ({:.2}ms)", avg_fps, avg_frame_time);
            println!("Min frame time: {:.2}ms (max FPS: {:.1})", min_frame_time, 1000.0 / min_frame_time);
            println!("Max frame time: {:.2}ms (min FPS: {:.1})", max_frame_time, 1000.0 / max_frame_time);
            
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
        // Wait for the previous frame to complete BEFORE starting the next one
        if let Some(sp) = self.prev_sync_point.take() {
            self.context.wait_for(&sp, !0);
        }
        
        // Now we can safely start the encoder for the new frame
        self.command_encoder.start();
        
        let frame = self.surface.acquire_frame();
        let target_view = frame.texture_view();
        
        // Create camera uniform
        let view_proj = self.scene.camera.view_projection_matrix();
        let camera_uniform = CameraUniform {
            view_proj: view_proj.to_cols_array_2d(),
        };
        
        // Render pass
        if let mut pass = self.command_encoder.render(
            "main_pass",
            gpu::RenderTargetSet {
                colors: &[gpu::RenderTarget {
                    view: target_view,
                    init_op: gpu::InitOp::Clear(gpu::TextureColor::OpaqueBlack),
                    finish_op: gpu::FinishOp::Store,
                }],
                depth_stencil: None,
            },
        ) {
            let mut rc = pass.with(&self.render_pipeline);
            
            // Bind camera data (group 0)
            let camera_data = CameraData {
                camera_data: camera_uniform,
            };
            rc.bind(0, &camera_data);
            
            // Draw each entity
            for entity in self.scene.entities.values() {
                if !entity.visible {
                    continue;
                }
                
                if let Some(mesh_handle) = entity.mesh {
                    if let Some(simple_mesh) = self.meshes.get(&mesh_handle) {
                        // Create model uniform
                        let transform = entity.transform.to_matrix();
                        let model_uniform = ModelUniform {
                            transform: transform.to_cols_array_2d(),
                            color: [0.7, 0.7, 0.8, 1.0],
                        };
                        
                        // Bind model data (group 1)
                        let model_data = ModelData {
                            model_data: model_uniform,
                        };
                        rc.bind(1, &model_data);
                        
                        // Bind vertex buffer
                        rc.bind_vertex(0, simple_mesh.vertex_buffer);
                        
                        // Draw indexed
                        rc.draw_indexed(simple_mesh.index_buffer, gpu::IndexType::U32, 0, simple_mesh.index_count as i32, 0, 1);
                    }
                }
            }
        }
        
        self.command_encoder.present(frame);
        
        // Submit and save the sync point for next frame
        let sync_point = self.context.submit(&mut self.command_encoder);
        self.prev_sync_point = Some(sync_point);
        self.frame_index += 1;
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
            display_sync: gpu::DisplaySync::Recent,
            ..Default::default()
        };

        self.context.reconfigure_surface(&mut self.surface, config);
        self.renderer.resize(width, height);
        self.scene.camera.set_aspect_ratio(width as f32 / height as f32);
    }

    fn cleanup(&mut self) {
        log::info!("Cleaning up resources...");
        if let Some(sp) = self.prev_sync_point.take() {
            self.context.wait_for(&sp, !0);
        }
        self.renderer.cleanup();
        self.lighting.cleanup(&self.context);
        self.context.destroy_command_encoder(&mut self.command_encoder);
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

    println!("Starting render loop...");
    println!("FPS counter updates every 100 frames.");
    println!("Rendering 3D scene with cube, sphere, and plane.");
    println!("Press ESC to exit.\n");

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
                    // Handled in AboutToWait now
                }
                _ => {}
            },
            Event::AboutToWait => {
                app.update();
                app.render();
                app.window.request_redraw();
            }
            _ => {}
        }
    }).unwrap();
}
