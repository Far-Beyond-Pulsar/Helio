use std::{collections::HashMap, sync::Arc};
use crate::{
    mesh::DrawCall,
    debug_draw::DebugDrawBatch,
    profiler::GpuProfiler,
    resources::FrameTextures,
    scene::{GpuLight, SkyAtmosphere, Skylight},
    camera::Camera,
};

/// Per-frame data passed to every pass in the graph.
pub struct PassContext<'frame> {
    pub device:   &'frame wgpu::Device,
    pub queue:    &'frame wgpu::Queue,
    pub encoder:  &'frame mut wgpu::CommandEncoder,
    pub frame_tex: &'frame FrameTextures,

    // Immutable scene state for this frame.
    pub camera:           Camera,
    pub camera_buffer:    &'frame wgpu::Buffer,
    pub globals_buffer:   &'frame wgpu::Buffer,
    pub camera_bg:        &'frame wgpu::BindGroup,
    pub opaque_draws:     &'frame [DrawCall],
    pub transparent_draws: &'frame [DrawCall],
    pub lights:           &'frame [GpuLight],
    pub light_buffer:     &'frame wgpu::Buffer,
    pub sky_atmosphere:   Option<&'frame SkyAtmosphere>,
    pub skylight:         Option<&'frame Skylight>,
    pub debug_batch:      Option<&'frame DebugDrawBatch>,
    pub frame_index:      u64,
    pub width:            u32,
    pub height:           u32,

    // Output surface texture (written last by the post-process pass).
    pub surface_view:     &'frame wgpu::TextureView,

    // Profiler scope name → index map (populated once at renderer init).
    pub profiler_scopes:  &'frame HashMap<String, u32>,
    pub profiler:         &'frame GpuProfiler,
}

impl<'frame> PassContext<'frame> {
    pub(crate) fn profiler_begin_scope_for(&mut self, label: &str) {
        if let Some(&idx) = self.profiler_scopes.get(label) {
            self.profiler.begin_scope(self.encoder, idx);
        }
    }

    pub(crate) fn profiler_end_scope_for(&mut self, label: &str) {
        if let Some(&idx) = self.profiler_scopes.get(label) {
            self.profiler.end_scope(self.encoder, idx);
        }
    }
}

/// The trait every render pass must implement.
pub trait RenderPass: Send {
    fn execute(&mut self, ctx: &mut PassContext);
}
