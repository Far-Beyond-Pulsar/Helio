//! Context types for features

use crate::buffer_pool::SharedPoolBuffer;
use crate::gpu_scene::MaterialRange;
use crate::resources::ResourceManager;
use crate::graph::RenderGraph;
use crate::camera::Camera;
use crate::mesh::DrawCall;
use crate::passes::ShadowCullLight;
use std::sync::{Arc, Mutex, atomic::AtomicU32};

/// Context provided to features during registration
pub struct FeatureContext<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub graph: &'a mut RenderGraph,
    pub resources: &'a mut ResourceManager,
    /// Surface / swapchain format (used for pipeline creation)
    pub surface_format: wgpu::TextureFormat,
    /// Arc clone of the device (needed by passes that create GPU resources at execution time)
    pub device_arc: Arc<wgpu::Device>,
    /// Arc clone of the queue (needed by passes that write to GPU buffers at execution time)
    pub queue_arc: Arc<wgpu::Queue>,

    // ── Inputs from Renderer (always set) ──────────────────────────────────
    /// Shared draw list — used by TransparentPass / RadianceCascadesPass
    pub draw_list: Arc<Mutex<Vec<DrawCall>>>,
    /// Shadow light-space matrix buffer — ShadowsFeature passes this to ShadowPass
    pub shadow_matrix_buffer: Arc<wgpu::Buffer>,
    /// Shared light count — updated by Renderer each frame; ShadowPass reads it
    pub light_count_arc: Arc<AtomicU32>,
    /// Per-light face counts: 6 for point, 4 for directional (CSM), 1 for spot.
    /// Updated by Renderer each frame so ShadowPass can skip identity-matrix faces.
    pub light_face_counts: Arc<Mutex<Vec<u8>>>,
    /// Per-light culling data (position, range, type) for `ShadowPass::execute`.
    pub shadow_cull_lights: Arc<Mutex<Vec<ShadowCullLight>>>,
    /// GpuScene instance data storage buffer — ShadowPass uses this at binding 2
    /// so the vertex shader can read transforms via `@builtin(instance_index)`.
    pub instance_data_buffer: &'a wgpu::Buffer,
    /// Unified pool vertex buffer — ShadowPass uses pool-allocated meshes.
    pub pool_vertex_buffer: SharedPoolBuffer,
    /// Unified pool index buffer — ShadowPass uses pool-allocated meshes.
    pub pool_index_buffer: SharedPoolBuffer,
    /// Shared opaque draw-call buffer from GpuScene (Arc refreshed by Renderer each frame).
    /// ShadowsFeature passes this to ShadowPass for GPU indirect shadow culling.
    pub shared_draw_call_buf: Arc<Mutex<Option<Arc<wgpu::Buffer>>>>,
    /// Per-material draw ranges — shared with GBufferPass and ShadowPass.
    pub shared_material_ranges: Arc<Mutex<Vec<MaterialRange>>>,
    /// Whether MULTI_DRAW_INDIRECT is available on this device.
    pub has_multi_draw: bool,

    // ── Outputs set by features during register() ───────────────────────────
    /// Light storage buffer set by LightingFeature
    pub light_buffer: Option<Arc<wgpu::Buffer>>,
    /// Shadow atlas texture view set by ShadowsFeature
    pub shadow_atlas_view: Option<Arc<wgpu::TextureView>>,
    /// Shadow comparison sampler set by ShadowsFeature
    pub shadow_sampler: Option<Arc<wgpu::Sampler>>,
    /// Radiance Cascades cascade-0 texture view set by RadianceCascadesFeature
    pub rc_cascade0_view: Option<Arc<wgpu::TextureView>>,
    /// Radiance Cascades world AABB set by RadianceCascadesFeature
    pub rc_world_bounds: Option<([f32; 3], [f32; 3])>,
}

impl<'a> FeatureContext<'a> {
    pub fn new(
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        graph: &'a mut RenderGraph,
        resources: &'a mut ResourceManager,
        surface_format: wgpu::TextureFormat,
        device_arc: Arc<wgpu::Device>,
        queue_arc: Arc<wgpu::Queue>,
        draw_list: Arc<Mutex<Vec<DrawCall>>>,
        shadow_matrix_buffer: Arc<wgpu::Buffer>,
        light_count_arc: Arc<AtomicU32>,
        light_face_counts: Arc<Mutex<Vec<u8>>>,
        shadow_cull_lights: Arc<Mutex<Vec<ShadowCullLight>>>,
        instance_data_buffer: &'a wgpu::Buffer,
        pool_vertex_buffer: SharedPoolBuffer,
        pool_index_buffer: SharedPoolBuffer,
        shared_draw_call_buf: Arc<Mutex<Option<Arc<wgpu::Buffer>>>>,
        shared_material_ranges: Arc<Mutex<Vec<MaterialRange>>>,
        has_multi_draw: bool,
    ) -> Self {
        Self {
            device,
            queue,
            graph,
            resources,
            surface_format,
            device_arc,
            queue_arc,
            draw_list,
            shadow_matrix_buffer,
            light_count_arc,
            light_face_counts,
            shadow_cull_lights,
            instance_data_buffer,
            pool_vertex_buffer,
            pool_index_buffer,
            shared_draw_call_buf,
            shared_material_ranges,
            has_multi_draw,
            light_buffer: None,
            shadow_atlas_view: None,
            shadow_sampler: None,
            rc_cascade0_view: None,
            rc_world_bounds: None,
        }
    }
}

/// Context provided to features during frame preparation
pub struct PrepareContext<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub resources: &'a ResourceManager,
    pub frame: u64,
    pub delta_time: f32,
    pub camera: &'a Camera,
}

impl<'a> PrepareContext<'a> {
    pub fn new(
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        resources: &'a ResourceManager,
        frame: u64,
        delta_time: f32,
        camera: &'a Camera,
    ) -> Self {
        Self {
            device,
            queue,
            resources,
            frame,
            delta_time,
            camera,
        }
    }
}

