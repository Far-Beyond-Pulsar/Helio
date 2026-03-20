use std::sync::Arc;

use arrayvec::ArrayVec;
use helio_v3::{RenderGraph, RenderPass, Result as HelioResult};
use helio_pass_billboard::BillboardPass;
use helio_pass_deferred_light::DeferredLightPass;
use helio_pass_depth_prepass::DepthPrepassPass;
use helio_pass_fxaa::FxaaPass;
use helio_pass_gbuffer::GBufferPass;
use helio_pass_shadow::ShadowPass;
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

#[derive(Debug, Clone, Copy)]
pub struct RendererConfig {
    pub width: u32,
    pub height: u32,
    pub surface_format: wgpu::TextureFormat,
}

impl RendererConfig {
    pub fn new(width: u32, height: u32, surface_format: wgpu::TextureFormat) -> Self {
        Self {
            width,
            height,
            surface_format,
        }
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
        }
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
        let config = RendererConfig::new(width, height, self.surface_format);
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
            }),
            sky: libhelio::SkyContext::default(),
        };

        self.graph.execute_with_frame_resources(
            self.scene.gpu_scene(),
            target,
            &self.depth_view,
            &frame_resources,
        )?;
        drop(frame_resources);
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

    // 1. ShadowPass — generates shadow atlas for all shadow-casting lights
    graph.add_pass(Box::new(ShadowPass::new(device)));

    // 2. SkyLutPass — generates atmospheric sky lookup texture
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

    // TODO: Add SsaoPass — needs resource declaration support

    // 5. DeferredLightPass — lighting pass (reads G-buffer, shadow maps)
    // With automatic resource management, this will write to "pre_aa" if declared,
    // or directly to surface if no post-processing passes are present
    graph.add_pass(Box::new(DeferredLightPass::new(
        device,
        queue,
        camera_buf,
        config.width,
        config.height,
        config.surface_format,
    )));

    // TODO: Enable these passes once they declare resources properly:
    // - SkyPass (reads "sky_lut", writes to scene color)
    // - TransparentPass (reads scene depth, writes to scene color with blending)
    // - BillboardPass (reads scene depth, writes to scene color)
    // - FxaaPass (reads "pre_aa", writes to final surface)

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
