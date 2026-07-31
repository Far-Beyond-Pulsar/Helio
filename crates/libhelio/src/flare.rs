//! GPU flare query data — compacted list of visible flare-enabled lights.

use bytemuck::{Pod, Zeroable};

/// Per-visible-light flare query result.
///
/// Written by the flare query compute pass after the depth prepass, read by
/// the flare render pass.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuFlareQuery {
    /// Light centre on screen (pixels, xy).
    pub screen_pos: [f32; 2],
    /// Depth buffer value at the light centre.
    pub screen_depth: f32,
    /// Luminance for threshold selection.
    pub light_intensity: f32,
    /// Light colour (RGB).
    pub light_color: [f32; 3],
    /// Light index in the main light buffer.
    pub light_index: u32,
}
