//! Main renderer implementation

use crate::resources::ResourceManager;
use crate::features::{FeatureRegistry, FeatureContext, PrepareContext};
use crate::pipeline::{PipelineCache, PipelineVariant};
use crate::graph::{RenderGraph, GraphContext};
use crate::passes::GeometryPass;
use crate::mesh::{GpuMesh, DrawCall};
use crate::camera::Camera;
use crate::scene::Scene;
use crate::features::lighting::{GpuLight, MAX_LIGHTS};
use crate::features::BillboardsFeature;
use crate::Result;
use std::sync::{Arc, Mutex};
use wgpu::util::DeviceExt;
use bytemuck::Zeroable;

/// Main renderer configuration
pub struct RendererConfig {
    pub width: u32,
    pub height: u32,
    pub surface_format: wgpu::TextureFormat,
    pub features: FeatureRegistry,
}

/// Globals uniform data
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GlobalsUniform {
    frame: u32,
    delta_time: f32,
    light_count: u32,
    ambient_intensity: f32,
    ambient_color: [f32; 4],  // w unused, ensures alignment
}

/// Material uniform data – must match WGSL Material struct (32 bytes)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MaterialUniform {
    base_color: [f32; 4],
    metallic: f32,
    roughness: f32,
    emissive: f32,
    ao: f32,
}

/// Create a Depth32Float texture + view at the given resolution
fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Depth Texture"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

/// Main renderer
pub struct Renderer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    resources: ResourceManager,
    graph: RenderGraph,
    pipelines: PipelineCache,
    features: FeatureRegistry,

    // Uniform buffers
    camera_buffer: wgpu::Buffer,
    globals_buffer: wgpu::Buffer,

    // Bind groups
    global_bind_group: wgpu::BindGroup,
    lighting_bind_group: Arc<wgpu::BindGroup>,
    default_material_bind_group: Arc<wgpu::BindGroup>,

    // Draw list (shared with GeometryPass)
    draw_list: Arc<Mutex<Vec<DrawCall>>>,

    // Light buffer for scene writes
    light_buffer: Arc<wgpu::Buffer>,
    // Current scene ambient (updated by render_scene)
    scene_ambient_color: [f32; 3],
    scene_ambient_intensity: f32,
    scene_light_count: u32,
    scene_sky_color: [f32; 3],

    // Depth buffer (Depth32Float, recreated on resize)
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,

    // Frame state
    frame_count: u64,
    width: u32,
    height: u32,
}

impl Renderer {
    /// Create a new renderer
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        config: RendererConfig,
    ) -> Result<Self> {
        log::info!("Creating Helio Render V2");
        log::info!("  Surface format: {:?}", config.surface_format);
        log::info!("  Resolution: {}x{}", config.width, config.height);

        let mut resources = ResourceManager::new(device.clone());
        let bind_group_layouts = Arc::new(resources.bind_group_layouts.clone());
        let mut pipelines = PipelineCache::new(device.clone(), bind_group_layouts.clone(), config.surface_format);
        let mut graph = RenderGraph::new();
        let mut features = config.features;

        // ── Uniform buffers ──────────────────────────────────────────────────
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera Uniform Buffer"),
            size: std::mem::size_of::<Camera>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let globals_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Globals Uniform Buffer"),
            size: std::mem::size_of::<GlobalsUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let global_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Global Bind Group"),
            layout: &resources.bind_group_layouts.global,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: camera_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: globals_buffer.as_entire_binding() },
            ],
        });

        // ── Register features (shadow passes etc.) ────────────────────────────
        let (feat_light_buf, feat_shadow_view, feat_shadow_sampler) = {
            let mut ctx = FeatureContext::new(
                &device, &queue, &mut graph, &mut resources, config.surface_format,
            );
            features.register_all(&mut ctx)?;
            (ctx.light_buffer, ctx.shadow_atlas_view, ctx.shadow_sampler)
        };

        // ── Default 1×1 textures ──────────────────────────────────────────────

        // White base-color texture
        let white_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Default White Texture"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &white_tex, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
            },
            &[255u8, 255, 255, 255],
            wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        let white_view = white_tex.create_view(&Default::default());

        // Flat normal map (0.5, 0.5, 1.0)
        let normal_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Default Flat Normal"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &normal_tex, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
            },
            &[128u8, 128, 255, 255],
            wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        let normal_view = normal_tex.create_view(&Default::default());

        let default_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Default Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let mat_uniform = MaterialUniform {
            base_color: [1.0, 1.0, 1.0, 1.0],
            metallic: 0.0, roughness: 0.6, emissive: 0.0, ao: 1.0,
        };
        let material_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Default Material Uniform"),
            contents: bytemuck::bytes_of(&mat_uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let default_material_bind_group = Arc::new(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Default Material Bind Group"),
            layout: &resources.bind_group_layouts.material,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: material_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&white_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&normal_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&default_sampler) },
            ],
        }));

        // ── Lighting bind group (group 2) ────────────────────────────────────
        // Fallback null light buffer when LightingFeature is not registered
        let null_light_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Null Light Buffer"),
            size: 64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Fallback 1-layer shadow atlas
        let default_shadow_atlas = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Default Shadow Atlas"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let default_shadow_atlas_view = default_shadow_atlas.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        let default_comparison_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Default Comparison Sampler"),
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        // Fallback black cube (env map)
        let cube_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Default Env Cube"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 6 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        for face in 0..6u32 {
            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &cube_tex, mip_level: 0,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: face },
                    aspect: wgpu::TextureAspect::All,
                },
                &[0u8, 0, 0, 255],
                wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) },
                wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            );
        }
        let cube_view = cube_tex.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
            array_layer_count: Some(6),
            ..Default::default()
        });

        // Pick the real resources if features provided them, otherwise use defaults
        let light_buf     = feat_light_buf.unwrap_or_else(|| Arc::new(null_light_buf));
        let light_buffer  = light_buf.clone();
        let shadow_view   = feat_shadow_view.unwrap_or_else(|| Arc::new(default_shadow_atlas_view));
        let shadow_samp   = feat_shadow_sampler.unwrap_or_else(|| Arc::new(default_comparison_sampler));

        let lighting_bind_group = Arc::new(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Lighting Bind Group"),
            layout: &resources.bind_group_layouts.lighting,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: light_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&shadow_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&shadow_samp) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&cube_view) },
            ],
        }));

        // ── Geometry pass with shared draw list ──────────────────────────────
        let draw_list: Arc<Mutex<Vec<DrawCall>>> = Arc::new(Mutex::new(Vec::new()));

        let defines = features.collect_shader_defines();
        let geometry_pipeline = pipelines.get_or_create(
            include_str!("../shaders/passes/geometry.wgsl"),
            "geometry".to_string(),
            &defines,
            PipelineVariant::Forward,
        )?;

        let mut geometry_pass = GeometryPass::with_draw_list(draw_list.clone());
        geometry_pass.set_pipeline(geometry_pipeline);
        graph.add_pass(geometry_pass);

        // Build the render graph
        graph.build()?;

        let (depth_texture, depth_view) = create_depth_texture(&device, config.width, config.height);

        log::info!("Helio Render V2 initialized successfully");

        Ok(Self {
            device,
            queue,
            resources,
            graph,
            pipelines,
            features,
            camera_buffer,
            globals_buffer,
            global_bind_group,
            lighting_bind_group,
            default_material_bind_group,
            draw_list,
            light_buffer,
            scene_ambient_color: [0.0, 0.0, 0.0],
            scene_ambient_intensity: 0.0,
            scene_light_count: 0,
            scene_sky_color: [0.0, 0.0, 0.0],
            depth_texture,
            depth_view,
            frame_count: 0,
            width: config.width,
            height: config.height,
        })
    }

    // ── Draw submission ───────────────────────────────────────────────────────

    /// Queue a mesh to be drawn this frame using the default white material
    pub fn draw_mesh(&self, mesh: &GpuMesh) {
        self.draw_list.lock().unwrap().push(DrawCall::new(mesh, self.default_material_bind_group.clone()));
    }

    /// Queue a mesh with a custom material bind group
    pub fn draw_mesh_with_material(&self, mesh: &GpuMesh, material: Arc<wgpu::BindGroup>) {
        self.draw_list.lock().unwrap().push(DrawCall::new(mesh, material));
    }

    // ── Feature enable/disable ────────────────────────────────────────────────

    /// Enable a feature at runtime
    pub fn enable_feature(&mut self, name: &str) -> Result<()> {
        self.features.enable(name)?;
        let flags = self.features.active_flags();
        self.pipelines.set_active_features(flags);
        log::info!("Enabled feature: {}", name);
        Ok(())
    }

    /// Disable a feature at runtime
    pub fn disable_feature(&mut self, name: &str) -> Result<()> {
        self.features.disable(name)?;
        let flags = self.features.active_flags();
        self.pipelines.set_active_features(flags);
        log::info!("Disabled feature: {}", name);
        Ok(())
    }

    // ── Frame rendering ───────────────────────────────────────────────────────

    /// Render a frame.  Call `draw_mesh()` BEFORE calling this.
    pub fn render(&mut self, camera: &Camera, target: &wgpu::TextureView, delta_time: f32) -> Result<()> {
        log::trace!("Rendering frame {}", self.frame_count);

        // Update global uniforms
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(camera));
        let globals = GlobalsUniform {
            frame: self.frame_count as u32,
            delta_time,
            light_count: self.scene_light_count,
            ambient_intensity: self.scene_ambient_intensity,
            ambient_color: [self.scene_ambient_color[0], self.scene_ambient_color[1], self.scene_ambient_color[2], 0.0],
        };
        self.queue.write_buffer(&self.globals_buffer, 0, bytemuck::bytes_of(&globals));

        // Prepare features (upload lights etc.)
        let prep_ctx = PrepareContext::new(
            &self.device, &self.queue, &self.resources,
            self.frame_count, delta_time, camera,
        );
        self.features.prepare_all(&prep_ctx)?;

        // Execute render graph
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        let mut graph_ctx = GraphContext {
            encoder: &mut encoder,
            resources: &self.resources,
            target,
            depth_view: &self.depth_view,
            frame: self.frame_count,
            global_bind_group: &self.global_bind_group,
            lighting_bind_group: &self.lighting_bind_group,
            sky_color: self.scene_sky_color,
        };

        self.graph.execute(&mut graph_ctx)?;

        // Submit and clear the draw list for next frame
        self.queue.submit(Some(encoder.finish()));
        self.draw_list.lock().unwrap().clear();

        self.frame_count += 1;
        Ok(())
    }

    // ── Utilities ─────────────────────────────────────────────────────────────

    /// Render the full scene. Everything in the scene is drawn; nothing else.
    pub fn render_scene(&mut self, scene: &Scene, camera: &Camera, target: &wgpu::TextureView, delta_time: f32) -> Result<()> {
        // Queue draw calls for all objects
        for obj in &scene.objects {
            self.draw_mesh(&obj.mesh);
        }

        // Upload scene lights to GPU buffer
        let count = scene.lights.len().min(MAX_LIGHTS as usize);
        let mut gpu_lights: Vec<GpuLight> = scene.lights[..count].iter().map(|l| {
            let light_type = match l.light_type {
                crate::features::LightType::Directional => 0.0,
                crate::features::LightType::Point => 1.0,
                crate::features::LightType::Spot { .. } => 2.0,
            };
            GpuLight {
                position: l.position,
                light_type,
                direction: l.direction,
                range: l.range,
                color: l.color,
                intensity: l.intensity,
                inner_angle: 0.0,
                outer_angle: 0.0,
                _pad: [0.0; 2],
            }
        }).collect();
        gpu_lights.resize(MAX_LIGHTS as usize, GpuLight::zeroed());
        self.queue.write_buffer(&self.light_buffer, 0, bytemuck::cast_slice(&gpu_lights));

        // Update billboard instances from scene
        if let Some(bb) = self.features.get_typed_mut::<BillboardsFeature>("billboards") {
            bb.set_billboards(scene.billboards.clone());
        }

        // Store scene ambient + sky for globals upload in render()
        self.scene_ambient_color = scene.ambient_color;
        self.scene_ambient_intensity = scene.ambient_intensity;
        self.scene_light_count = count as u32;
        self.scene_sky_color = scene.sky_color;

        self.render(camera, target, delta_time)
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        log::info!("Resizing renderer to {}x{}", width, height);
        self.width = width;
        self.height = height;
        let (tex, view) = create_depth_texture(&self.device, width, height);
        self.depth_texture = tex;
        self.depth_view = view;
    }

    pub fn frame_count(&self) -> u64 { self.frame_count }
    pub fn device(&self) -> &wgpu::Device { &self.device }
    pub fn queue(&self) -> &wgpu::Queue { &self.queue }
}
