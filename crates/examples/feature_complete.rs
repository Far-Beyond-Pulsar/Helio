use blade_graphics as gpu;
use glam::{Mat4, Vec3};
use helio_core::{create_cube_mesh, create_plane_mesh, create_sphere_mesh};
use helio_feature_base_geometry::BaseGeometry;
use helio_feature_lighting::BasicLighting;
use helio_feature_materials::BasicMaterials;
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
        let plane_mesh = create_plane_mesh(10.0, 10.0);

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

        let aspect_ratio = self.window_size.width as f32 / self.window_size.height as f32;
        let projection = Mat4::perspective_rh(45.0f32.to_radians(), aspect_ratio, 0.1, 100.0);

        let camera_pos = Vec3::new(5.0 * elapsed.sin(), 3.0, 5.0 * elapsed.cos());
        let view = Mat4::look_at_rh(camera_pos, Vec3::ZERO, Vec3::Y);
        let view_proj = projection * view;

        let camera = CameraUniforms {
            view_proj: view_proj.to_cols_array_2d(),
            position: camera_pos.to_array(),
            _pad: 0.0,
        };

        let mut meshes = Vec::new();

        let cube_transform =
            Mat4::from_rotation_y(elapsed) * Mat4::from_translation(Vec3::new(-2.0, 1.0, 0.0));
        meshes.push((
            TransformUniforms {
                model: cube_transform.to_cols_array_2d(),
            },
            self.cube_vertices.into(),
            self.cube_indices.into(),
            self.cube_index_count,
        ));

        let sphere_transform = Mat4::from_translation(Vec3::new(2.0, 1.0, 0.0));
        meshes.push((
            TransformUniforms {
                model: sphere_transform.to_cols_array_2d(),
            },
            self.sphere_vertices.into(),
            self.sphere_indices.into(),
            self.sphere_index_count,
        ));

        let plane_transform = Mat4::from_translation(Vec3::ZERO);
        meshes.push((
            TransformUniforms {
                model: plane_transform.to_cols_array_2d(),
            },
            self.plane_vertices.into(),
            self.plane_indices.into(),
            self.plane_index_count,
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
        .with_title("Helio - All Features (Geometry + Lighting + Materials)")
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
                            physical_key:
                                winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Escape),
                            ..
                        },
                    ..
                } => elwt.exit(),
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
