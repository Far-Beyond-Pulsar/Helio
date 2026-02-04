pub mod renderer;
pub mod frame_graph;
pub mod render_pass;
pub mod gbuffer;
pub mod deferred;
pub mod forward;
pub mod visibility_buffer;
pub mod shader_compiler;

pub use renderer::*;
pub use frame_graph::*;
pub use render_pass::*;
pub use gbuffer::*;
pub use deferred::*;
pub use forward::*;
pub use visibility_buffer::*;
pub use shader_compiler::*;

use helio_core::gpu;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderPath {
    Deferred,
    Forward,
    ForwardPlus,
    VisibilityBuffer,
}

impl Default for RenderPath {
    fn default() -> Self {
        Self::Deferred
    }
}

#[derive(Clone, Debug)]
pub struct RendererConfig {
    pub render_path: RenderPath,
    pub width: u32,
    pub height: u32,
    pub hdr_enabled: bool,
    pub msaa_samples: u32,
    pub vsync_enabled: bool,
    pub max_lights_per_tile: u32,
    pub tile_size: u32,
}

impl Default for RendererConfig {
    fn default() -> Self {
        Self {
            render_path: RenderPath::Deferred,
            width: 1920,
            height: 1080,
            hdr_enabled: true,
            msaa_samples: 1,
            vsync_enabled: true,
            max_lights_per_tile: 256,
            tile_size: 16,
        }
    }
}
