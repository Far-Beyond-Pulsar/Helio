//! Context types for features

use crate::resources::ResourceManager;
use crate::graph::RenderGraph;
use crate::camera::Camera;
use crate::mesh::DrawCall;
use std::sync::{Arc, Mutex, atomic::AtomicU32};

/// Context provided to features during registration
pub struct FeatureContext<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub graph: &'a mut RenderGraph,
    pub resources: &'a mut ResourceManager,
    /// Surface / swapchain format (used for pipeline creation)
    pub surface_format: wgpu::TextureFormat,

    // ── Inputs from Renderer (always set) ──────────────────────────────────
    /// Shared draw list — ShadowsFeature passes this to ShadowPass
    pub draw_list: Arc<Mutex<Vec<DrawCall>>>,
    /// Shadow light-space matrix buffer — ShadowsFeature passes this to ShadowPass
    pub shadow_matrix_buffer: Arc<wgpu::Buffer>,
    /// Shared light count — updated by Renderer each frame; ShadowPass reads it
    pub light_count_arc: Arc<AtomicU32>,

    // ── Outputs set by features during register() ───────────────────────────
    /// Light storage buffer set by LightingFeature
    pub light_buffer: Option<Arc<wgpu::Buffer>>,
    /// Shadow atlas texture view set by ShadowsFeature
    pub shadow_atlas_view: Option<Arc<wgpu::TextureView>>,
    /// Shadow comparison sampler set by ShadowsFeature
    pub shadow_sampler: Option<Arc<wgpu::Sampler>>,
}

impl<'a> FeatureContext<'a> {
    pub fn new(
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        graph: &'a mut RenderGraph,
        resources: &'a mut ResourceManager,
        surface_format: wgpu::TextureFormat,
        draw_list: Arc<Mutex<Vec<DrawCall>>>,
        shadow_matrix_buffer: Arc<wgpu::Buffer>,
        light_count_arc: Arc<AtomicU32>,
    ) -> Self {
        Self {
            device,
            queue,
            graph,
            resources,
            surface_format,
            draw_list,
            shadow_matrix_buffer,
            light_count_arc,
            light_buffer: None,
            shadow_atlas_view: None,
            shadow_sampler: None,
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
