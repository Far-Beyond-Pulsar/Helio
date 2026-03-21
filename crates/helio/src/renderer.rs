use std::sync::Arc;

use arrayvec::ArrayVec;
use helio_v3::{RenderGraph, RenderPass, Result as HelioResult};
use helio_pass_billboard::BillboardPass;
use helio_pass_virtual_geometry::VirtualGeometryPass;
use helio_pass_deferred_light::DeferredLightPass;
use helio_pass_depth_prepass::DepthPrepassPass;
use helio_pass_fxaa::FxaaPass;
use helio_pass_gbuffer::GBufferPass;
use helio_pass_shadow::ShadowPass;
use helio_pass_shadow_matrix::ShadowMatrixPass;
use helio_pass_simple_cube::SimpleCubePass;
use helio_pass_sky_lut::SkyLutPass;
use helio_pass_transparent::TransparentPass;
// TODO: Add these passes once cross-reference issues are resolved:
// - SkyPass (needs sky_lut_view from SkyLutPass)
// - SsaoPass (needs gbuffer views + depth view)
// - SmaaPass, TaaPass (for higher-quality AA)
use crate::handles::{LightId, MaterialId, MeshId, ObjectId};
use crate::material::{MaterialAsset, MAX_TEXTURES, TextureUpload};
use crate::mesh::{MeshBuffers, MeshUpload};
use crate::scene::{Camera, ObjectDescriptor, Result as SceneResult, Scene};

pub fn required_wgpu_features(adapter_features: wgpu::Features) -> wgpu::Features {
    let required = wgpu::Features::TEXTURE_BINDING_ARRAY
        | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
    let optional = wgpu::Features::MULTI_DRAW_INDIRECT;
    required | (adapter_features & optional)
}

pub fn required_wgpu_limits(adapter_limits: wgpu::Limits) -> wgpu::Limits {
    wgpu::Limits {
        max_sampled_textures_per_shader_stage: MAX_TEXTURES as u32,
        max_samplers_per_shader_stage: MAX_TEXTURES as u32,
        ..adapter_limits
    }
}

/// Global Illumination configuration (AAA dual-tier: RC near, ambient far).
#[derive(Debug, Clone, Copy)]
pub struct GiConfig {
    /// Radiance Cascades volume radius around camera (world units).
    /// GI within this radius uses RC, outside uses cheap ambient fallback.
    /// Default: 80.0 (AAA near-field quality like Unreal Lumen).
    pub rc_radius: f32,
    /// Fade margin for smooth RC→ambient transition (world units).
    /// Default: 20.0 (soft blend zone).
    pub rc_fade_margin: f32,
}

impl Default for GiConfig {
    fn default() -> Self {
        Self {
            rc_radius: 80.0,      // AAA near-field GI
            rc_fade_margin: 20.0, // Smooth transition
        }
    }
}

impl GiConfig {
    /// Disable RC entirely (use only ambient/hemisphere for all distances).
    pub fn ambient_only() -> Self {
        Self {
            rc_radius: 0.0,
            rc_fade_margin: 0.0,
        }
    }

    /// High-quality large-radius RC (for open worlds).
    pub fn large_radius(radius: f32) -> Self {
        Self {
            rc_radius: radius,
            rc_fade_margin: radius * 0.25,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RendererConfig {
    pub width: u32,
    pub height: u32,
    pub surface_format: wgpu::TextureFormat,
    /// Global illumination configuration (RC near, ambient far).
    pub gi_config: GiConfig,
    /// Shadow quality (Low, Medium, High, Ultra).
    pub shadow_quality: libhelio::ShadowQuality,
    /// Debug visualisation mode (0=normal, 10=shadow heatmap, 11=light-space depth).
    pub debug_mode: u32,
}

impl RendererConfig {
    pub fn new(width: u32, height: u32, surface_format: wgpu::TextureFormat) -> Self {
        Self {
            width,
            height,
            surface_format,
            gi_config: GiConfig::default(), // AAA default: RC close, ambient far
            shadow_quality: libhelio::ShadowQuality::Medium, // Default quality
            debug_mode: 0,
        }
    }

    /// Builder: set custom GI configuration.
    pub fn with_gi_config(mut self, gi_config: GiConfig) -> Self {
        self.gi_config = gi_config;
        self
    }

    /// Builder: set shadow quality.
    pub fn with_shadow_quality(mut self, quality: libhelio::ShadowQuality) -> Self {
        self.shadow_quality = quality;
        self
    }
}

pub struct Renderer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    graph: RenderGraph,
    graph_kind: GraphKind,
    scene: Scene,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    surface_format: wgpu::TextureFormat,
    ambient_color: [f32; 3],
    ambient_intensity: f32,
    clear_color: [f32; 4],
    gi_config: GiConfig,
    shadow_quality: libhelio::ShadowQuality,
    debug_mode: u32,
    billboard_instances: Vec<helio_pass_billboard::BillboardInstance>,
}

/// Which graph is currently active — used by `set_render_size` to rebuild correctly.
enum GraphKind {
    /// Full deferred pipeline (default).
    Default,
    /// Minimal single-cube debug graph.
    Simple,
    /// User-provided graph; never rebuilt automatically.
    Custom,
}

impl Renderer {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        config: RendererConfig,
    ) -> Self {
        let mut scene = Scene::new(device.clone(), queue.clone());
        scene.set_render_size(config.width, config.height);

        let graph = build_default_graph(&device, &queue, &scene, config);

        let (depth_texture, depth_view) =
            create_depth_resources(&device, config.width, config.height);
        Self {
            device,
            queue,
            graph,
            graph_kind: GraphKind::Default,
            scene,
            depth_texture,
            depth_view,
            surface_format: config.surface_format,
            ambient_color: [0.05, 0.05, 0.08],
            ambient_intensity: 1.0,
            clear_color: [0.02, 0.02, 0.03, 1.0],
            gi_config: config.gi_config,
            shadow_quality: config.shadow_quality,
            debug_mode: 0,
            billboard_instances: Vec::new(),
        }
    }

    /// Set GI configuration (RC radius, fade margin).
    pub fn set_gi_config(&mut self, gi_config: GiConfig) {
        self.gi_config = gi_config;
    }

    /// Get current GI configuration.
    pub fn gi_config(&self) -> GiConfig {
        self.gi_config
    }

    /// Set shadow quality (Low, Medium, High, Ultra). Rebuilds the render graph.
    pub fn set_shadow_quality(&mut self, quality: libhelio::ShadowQuality) {
        self.shadow_quality = quality;
        // Rebuild the default graph to apply the new shadow quality
        if matches!(self.graph_kind, GraphKind::Default) {
            let config = RendererConfig {
                width: self.depth_texture.size().width,
                height: self.depth_texture.size().height,
                surface_format: self.surface_format,
                gi_config: self.gi_config,
                shadow_quality: self.shadow_quality,
                debug_mode: self.debug_mode,
            };
            self.graph = build_default_graph(&self.device, &self.queue, &self.scene, config);
        }
    }

    /// Set the debug visualisation mode (0=normal, 10=shadow heatmap, 11=light-space depth).
    pub fn set_debug_mode(&mut self, mode: u32) {
        self.debug_mode = mode;
        if matches!(self.graph_kind, GraphKind::Default) {
            let config = RendererConfig {
                width: self.depth_texture.size().width,
                height: self.depth_texture.size().height,
                surface_format: self.surface_format,
                gi_config: self.gi_config,
                shadow_quality: self.shadow_quality,
                debug_mode: self.debug_mode,
            };
            self.graph = build_default_graph(&self.device, &self.queue, &self.scene, config);
        }
    }

    /// Get current shadow quality.
    pub fn shadow_quality(&self) -> libhelio::ShadowQuality {
        self.shadow_quality
    }

    pub fn scene(&self) -> &Scene {
        &self.scene
    }

    pub fn scene_mut(&mut self) -> &mut Scene {
        &mut self.scene
    }

    /// Returns a reference to the GPU camera uniform buffer.
    /// Useful for creating custom render passes that need to read camera data (e.g. SDF ray march).
    pub fn camera_buffer(&self) -> &wgpu::Buffer {
        self.scene.gpu_scene().camera.buffer()
    }

    pub fn mesh_buffers(&self) -> MeshBuffers<'_> {
        self.scene.mesh_buffers()
    }

    pub fn add_pass(&mut self, pass: Box<dyn RenderPass>) {
        self.graph.add_pass(pass);
    }

    pub fn set_render_size(&mut self, width: u32, height: u32) {
        self.scene.set_render_size(width, height);
        let (depth_texture, depth_view) = create_depth_resources(&self.device, width, height);
        self.depth_texture = depth_texture;
        self.depth_view = depth_view;
        let config = RendererConfig::new(width, height, self.surface_format)
            .with_gi_config(self.gi_config)
            .with_shadow_quality(self.shadow_quality);
        let config = RendererConfig { debug_mode: self.debug_mode, ..config };
        match self.graph_kind {
            GraphKind::Default => {
                self.graph = build_default_graph(&self.device, &self.queue, &self.scene, config);
            }
            GraphKind::Simple => {
                self.graph = build_simple_graph(&self.device, &self.queue, self.surface_format);
            }
            GraphKind::Custom => {
                // User-provided graph: do not replace it.
            }
        }
    }

    pub fn set_clear_color(&mut self, color: [f32; 4]) {
        self.clear_color = color;
    }

    pub fn set_ambient(&mut self, color: [f32; 3], intensity: f32) {
        self.ambient_color = color;
        self.ambient_intensity = intensity;
    }

    /// Replace the active render graph. Use [`build_simple_graph`] or
    /// [`build_default_graph`](fn.build_default_graph.html) to construct one.
    /// The graph will NOT be automatically rebuilt on window resize.
    pub fn set_graph(&mut self, graph: RenderGraph) {
        self.graph = graph;
        self.graph_kind = GraphKind::Custom;
    }

    /// Convenience helper: switch to the simple single-cube graph.
    pub fn use_simple_graph(&mut self) {
        self.graph = build_simple_graph(&self.device, &self.queue, self.surface_format);
        self.graph_kind = GraphKind::Simple;
    }

    /// Convenience helper: switch back to the full deferred graph.
    pub fn use_default_graph(&mut self) {
        let config = RendererConfig {
            width:          self.depth_texture.size().width,
            height:         self.depth_texture.size().height,
            surface_format: self.surface_format,
            gi_config:      self.gi_config,
            shadow_quality: self.shadow_quality,
            debug_mode:     self.debug_mode,
        };
        self.graph = build_default_graph(&self.device, &self.queue, &self.scene, config);
        self.graph_kind = GraphKind::Default;
    }

    pub fn insert_mesh(&mut self, mesh: MeshUpload) -> MeshId {
        self.scene.insert_mesh(mesh)
    }

    pub fn insert_texture(&mut self, texture: TextureUpload) -> SceneResult<crate::TextureId> {
        self.scene.insert_texture(texture)
    }

    pub fn insert_material(&mut self, material: crate::GpuMaterial) -> MaterialId {
        self.scene.insert_material(material)
    }

    pub fn insert_material_asset(&mut self, material: MaterialAsset) -> SceneResult<MaterialId> {
        self.scene.insert_material_asset(material)
    }

    pub fn update_material(
        &mut self,
        id: MaterialId,
        material: crate::GpuMaterial,
    ) -> SceneResult<()> {
        self.scene.update_material(id, material)
    }

    pub fn update_material_asset(
        &mut self,
        id: MaterialId,
        material: MaterialAsset,
    ) -> SceneResult<()> {
        self.scene.update_material_asset(id, material)
    }

    pub fn insert_light(&mut self, light: crate::GpuLight) -> LightId {
        self.scene.insert_light(light)
    }

    pub fn update_light(&mut self, id: LightId, light: crate::GpuLight) -> SceneResult<()> {
        self.scene.update_light(id, light)
    }

    pub fn insert_object(&mut self, desc: ObjectDescriptor) -> SceneResult<ObjectId> {
        self.scene.insert_object(desc)
    }

    pub fn update_object_transform(
        &mut self,
        id: ObjectId,
        transform: glam::Mat4,
    ) -> SceneResult<()> {
        self.scene.update_object_transform(id, transform)
    }

    /// Replace the billboard instance list for the next frame.
    ///
    /// Call once per frame (or whenever the billboard set changes).
    /// Billboards are camera-facing quads rendered with alpha blending on top of
    /// the opaque scene.  Pass an empty slice to disable billboards entirely.
    pub fn set_billboard_instances(&mut self, instances: &[helio_pass_billboard::BillboardInstance]) {
        self.billboard_instances.clear();
        self.billboard_instances.extend_from_slice(instances);
    }

    pub fn render(
        &mut self,
        camera: &Camera,
        target: &wgpu::TextureView,
    ) -> HelioResult<()> {
        self.scene.update_camera(*camera);
        self.scene.flush();

        let mut texture_views = ArrayVec::<&wgpu::TextureView, MAX_TEXTURES>::new();
        let mut samplers = ArrayVec::<&wgpu::Sampler, MAX_TEXTURES>::new();
        for slot in 0..MAX_TEXTURES {
            texture_views.push(self.scene.texture_view_for_slot(slot));
            samplers.push(self.scene.texture_sampler_for_slot(slot));
        }

        let mesh_buffers = self.scene.mesh_buffers();
        // Compute RC bounds (AAA dual-tier GI: RC near, ambient far)
        let cam_pos = camera.position; // Camera.position is a field (Vec3), not a method
        let rc_radius = self.gi_config.rc_radius;
        let rc_min = [
            cam_pos.x - rc_radius,
            cam_pos.y - rc_radius,
            cam_pos.z - rc_radius,
        ];
        let rc_max = [
            cam_pos.x + rc_radius,
            cam_pos.y + rc_radius,
            cam_pos.z + rc_radius,
        ];

        let frame_resources = libhelio::FrameResources {
            gbuffer: None,
            shadow_atlas: None,
            shadow_sampler: None,
            hiz: None,
            hiz_sampler: None,
            sky_lut: None,
            sky_lut_sampler: None,
            ssao: None,
            pre_aa: None,
            main_scene: Some(libhelio::MainSceneResources {
                mesh_buffers: libhelio::MeshBuffers {
                    vertices: mesh_buffers.vertices,
                    indices: mesh_buffers.indices,
                },
                material_textures: libhelio::MaterialTextureBindings {
                    material_textures: self.scene.material_texture_buffer(),
                    texture_views: texture_views.as_slice(),
                    samplers: samplers.as_slice(),
                    version: self.scene.texture_binding_version(),
                },
                clear_color: self.clear_color,
                ambient_color: self.ambient_color,
                ambient_intensity: self.ambient_intensity,
                rc_world_min: rc_min,
                rc_world_max: rc_max,
            }),
            sky: libhelio::SkyContext::default(),
            billboards: if self.billboard_instances.is_empty() {
                None
            } else {
                Some(libhelio::BillboardFrameData {
                    instances: bytemuck::cast_slice(&self.billboard_instances),
                    count: self.billboard_instances.len() as u32,
                })
            },
            vg: self.scene.vg_frame_data(),
        };

        self.graph.execute_with_frame_resources(
            self.scene.gpu_scene(),
            target,
            &self.depth_view,
            &frame_resources,
        )?;
        // frame_resources is Copy, no need to drop
        drop(texture_views);
        drop(samplers);
        self.scene.advance_frame();
        Ok(())
    }
}

fn build_default_graph(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    scene: &Scene,
    config: RendererConfig,
) -> RenderGraph {
    let gpu_scene = scene.gpu_scene();
    let mut graph = RenderGraph::new(device, queue);

    let camera_buf = gpu_scene.camera.buffer();
    let _instances_buf = gpu_scene.instances.buffer();

    // 1. ShadowMatrixPass — GPU compute: writes correct view-projection matrices for
    //    every shadow face into the shadow_matrices storage buffer.
    //    Scene::flush() pre-sizes that buffer to N_lights*6 entries so shadow_count
    //    is nonzero and this binding covers the full range.
    let shadow_dirty_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Shadow Dirty Flags"),
        size: 64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let shadow_hashes_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Shadow Hashes"),
        size: 64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    graph.add_pass(Box::new(ShadowMatrixPass::new(
        device,
        gpu_scene.lights.buffer(),
        gpu_scene.shadow_matrices.buffer(),
        camera_buf,
        &shadow_dirty_buf,
        &shadow_hashes_buf,
    )));

    // 2. ShadowPass — renders geometry into the shadow atlas (one face per entry in shadow_matrices)
    // publish()es shadow_atlas + shadow_sampler into FrameResources for DeferredLightPass
    graph.add_pass(Box::new(ShadowPass::new(device)));

    // 3. SkyLutPass — generates atmospheric sky lookup texture
    // Publishes "sky_lut" resource for SkyPass to consume
    graph.add_pass(Box::new(SkyLutPass::new(device, camera_buf)));

    // 3. DepthPrepassPass — early depth pass for better GPU culling
    graph.add_pass(Box::new(DepthPrepassPass::new(
        device,
        wgpu::TextureFormat::Depth32Float,
    )));

    // 4. GBufferPass — fills G-buffer (albedo, normal, ORM, emissive)
    graph.add_pass(Box::new(GBufferPass::new(
        device,
        config.width,
        config.height,
    )));

    // 4b. VirtualGeometryPass — GPU-driven meshlet cull + draw into the same GBuffer
    graph.add_pass(Box::new(VirtualGeometryPass::new(
        device,
        camera_buf,
    )));

    // TODO: Add SsaoPass — needs resource declaration support

    // 5. DeferredLightPass — lighting pass (reads G-buffer, shadow maps)
    // With automatic resource management, this will write to "pre_aa" if declared,
    // or directly to surface if no post-processing passes are present
    let mut deferred_light_pass = DeferredLightPass::new(
        device,
        queue,
        camera_buf,
        config.width,
        config.height,
        config.surface_format,
    );
    deferred_light_pass.set_shadow_quality(config.shadow_quality, queue);
    deferred_light_pass.debug_mode = config.debug_mode;
    graph.add_pass(Box::new(deferred_light_pass));

    // TODO: Enable these passes once they declare resources properly:
    // - SkyPass (reads "sky_lut", writes to scene color)
    // - TransparentPass (reads scene depth, writes to scene color with blending)
    // - FxaaPass (reads "pre_aa", writes to final surface)

    // 6. BillboardPass — camera-facing instanced quads (composited after opaque geometry)
    graph.add_pass(Box::new(BillboardPass::new(
        device,
        queue,
        camera_buf,
        config.surface_format,
    )));

    // Initialize transient textures from pass declarations
    graph.set_render_size(config.width, config.height);

    graph
}

/// A minimal graph with a single geometry-only pass that always renders one
/// hardcoded cube at full brightness. Useful as a sanity-check baseline.
pub fn build_simple_graph(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    surface_format: wgpu::TextureFormat,
) -> RenderGraph {
    let mut graph = RenderGraph::new(device, queue);
    graph.add_pass(Box::new(SimpleCubePass::new(device, surface_format)));
    graph
}

fn create_depth_resources(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Helio Depth Texture"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}
