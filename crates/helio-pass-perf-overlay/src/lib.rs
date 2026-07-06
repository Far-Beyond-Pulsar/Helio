pub mod analysis;
pub mod rendering;

pub use analysis::*;
pub use rendering::*;

use bytemuck::{Pod, Zeroable};

pub const TILE_SIZE: u32 = 16;

// ── Visualization Modes ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum PerfOverlayMode {
    #[default]
    Disabled = 0,
    PassOverdraw = 1,
    ShaderComplexity = 2,
    TileLightCount = 3,
    PassOutput = 4,
}

// ── GPU-side uniforms ──────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct ColorCompareParams {
    pub screen_width: u32,
    pub screen_height: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct ComputeCostParams {
    pub screen_width: u32,
    pub screen_height: u32,
    pub num_tiles_x: u32,
    pub num_timing_entries: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct AggregateParams {
    pub num_tiles_x: u32,
    pub num_tiles_y: u32,
    pub num_tiles: u32,
    pub screen_width: u32,
    pub screen_height: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct TileMetrics {
    pub pass_overdraw_max: u32,
    pub light_count: u32,
    pub complexity_avg: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct VisualizeParams {
    pub mode: u32,
    pub num_tiles_x: u32,
    pub num_tiles_y: u32,
    pub internal_width: u32,
    pub internal_height: u32,
    pub display_width: u32,
    pub display_height: u32,
    pub heatmap_scale: f32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct MaterialProfileParams {
    pub roughness: f32,
    pub metallic: f32,
    pub num_lights: u32,
    pub num_shadow_lights: u32,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct MaterialTimingEntry {
    pub roughness: f32,
    pub metallic: f32,
    pub num_lights: u32,
    pub gpu_time_ns: u64,
}

#[derive(Clone, Copy)]
pub(crate) struct PerfOverlayRuntime {
    pub frame_num: u64,
    pub snapshot_valid: bool,
}
