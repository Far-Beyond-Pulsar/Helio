pub mod rendering;

pub use rendering::VirtualGeometryPass;

use bytemuck::{Pod, Zeroable};

// ═══════════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════════

pub const LOD_LEVEL_COUNT: u32 = 8;

pub(crate) const INITIAL_MESHLETS: u64 = 1024;
pub(crate) const INITIAL_INSTANCES: u64 = 256;

// ═══════════════════════════════════════════════════════════════════════════════
// Lod quality preset
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LodQuality {
    Low,
    #[default]
    Medium,
    High,
    Ultra,
}

impl LodQuality {
    pub fn thresholds(self) -> [f32; 7] {
        match self {
            LodQuality::Low => [0.180, 0.120, 0.080, 0.050, 0.030, 0.015, 0.006],
            LodQuality::Medium => [0.050, 0.035, 0.022, 0.014, 0.008, 0.004, 0.002],
            LodQuality::High => [0.020, 0.014, 0.009, 0.005, 0.003, 0.0015, 0.0006],
            LodQuality::Ultra => [0.008, 0.005, 0.003, 0.002, 0.001, 0.0005, 0.0002],
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GPU uniform types
// ═══════════════════════════════════════════════════════════════════════════════

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct VgGlobals {
    pub frame: u32,
    pub delta_time: f32,
    pub light_count: u32,
    pub ambient_intensity: f32,
    pub ambient_color: [f32; 4],
    pub rc_world_min: [f32; 4],
    pub rc_world_max: [f32; 4],
    pub csm_splits: [f32; 4],
    pub debug_mode: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct CullUniforms {
    pub meshlet_count: u32,
    pub screen_width: u32,
    pub screen_height: u32,
    pub hiz_mip_count: u32,
    pub lod_thresholds: [f32; 7],
    _pad3: f32,
}
