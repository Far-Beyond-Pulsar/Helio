//! SceneDB component types for Helio's scene object data (Helio #210).
//!
//! GPU-mirrored components carry `#[derive(SceneStore)]` + `#[gpu(layout = packed)]`
//! to pack all `#[gpu]` fields into a single dirty-tracked GPU column. SceneDB's
//! world-mirror bridge handles upload transparently: `World::insert` →
//! dispatch → `write_gpu_columns_at_row` → `mark_gpu_row_dirty` →
//! `flush_gpu_mirror` → coalesced GPU upload.
//!
//! CPU-only bookkeeping lives in SEPARATE companion components (no derive).

use pulsar_scenedb_derive::SceneStore;

use crate::groups::GroupMask;

/// GPU-facing instance data — one packed dirty-tracked column.
///
/// NOTE: only scalar types (`f32`, `u32`, `i32`) and `[f32; 16]` (which has
/// an explicit `Pod` impl in pulsar_scenedb) can be used as `#[gpu]` fields,
/// because the derive generates `Pod` bounds for EVERY field. The normal
/// matrix (`nm_00` … `nm_23`) and bounds are split into individual scalars.
#[repr(C)]
#[derive(Debug, Clone, Copy, SceneStore)]
#[gpu(layout = packed)]
pub(crate) struct HelioGpuInstance {
    #[gpu] pub m_00: f32,  #[gpu] pub m_01: f32,
    #[gpu] pub m_02: f32,  #[gpu] pub m_03: f32,
    #[gpu] pub m_10: f32,  #[gpu] pub m_11: f32,
    #[gpu] pub m_12: f32,  #[gpu] pub m_13: f32,
    #[gpu] pub m_20: f32,  #[gpu] pub m_21: f32,
    #[gpu] pub m_22: f32,  #[gpu] pub m_23: f32,
    #[gpu] pub m_30: f32,  #[gpu] pub m_31: f32,
    #[gpu] pub m_32: f32,  #[gpu] pub m_33: f32,
    // normal matrix (3×3 → 3×vec4 = 12 f32)
    #[gpu] pub nm_00: f32, #[gpu] pub nm_01: f32,
    #[gpu] pub nm_02: f32, #[gpu] pub _pad_nm0: f32,
    #[gpu] pub nm_10: f32, #[gpu] pub nm_11: f32,
    #[gpu] pub nm_12: f32, #[gpu] pub _pad_nm1: f32,
    #[gpu] pub nm_20: f32, #[gpu] pub nm_21: f32,
    #[gpu] pub nm_22: f32, #[gpu] pub _pad_nm2: f32,
    // bounds sphere
    #[gpu] pub bounds_cx: f32, #[gpu] pub bounds_cy: f32,
    #[gpu] pub bounds_cz: f32, #[gpu] pub bounds_r:  f32,
    // prev model matrix (same layout as m_*)
    #[gpu] pub pm_00: f32, #[gpu] pub pm_01: f32,
    #[gpu] pub pm_02: f32, #[gpu] pub pm_03: f32,
    #[gpu] pub pm_10: f32, #[gpu] pub pm_11: f32,
    #[gpu] pub pm_12: f32, #[gpu] pub pm_13: f32,
    #[gpu] pub pm_20: f32, #[gpu] pub pm_21: f32,
    #[gpu] pub pm_22: f32, #[gpu] pub pm_23: f32,
    #[gpu] pub pm_30: f32, #[gpu] pub pm_31: f32,
    #[gpu] pub pm_32: f32, #[gpu] pub pm_33: f32,
    #[gpu] pub mesh_id: u32,
    #[gpu] pub material_id: u32,
    #[gpu] pub flags: u32,
    #[gpu] pub lightmap_index: u32,
}

/// GPU-facing light data — one packed dirty-tracked column.
///
/// All vec4 fields are split into individual `f32` scalars (same Pod constraint
/// as [`HelioGpuInstance`]). The packed view struct re-packs them into the
/// WGSL-matching layout.
#[repr(C)]
#[derive(Debug, Clone, Copy, SceneStore)]
#[gpu(layout = packed)]
pub(crate) struct HelioGpuLight {
    // position_range
    #[gpu] pub pos_x: f32, #[gpu] pub pos_y: f32,
    #[gpu] pub pos_z: f32, #[gpu] pub range: f32,
    // direction_outer
    #[gpu] pub dir_x: f32, #[gpu] pub dir_y: f32,
    #[gpu] pub dir_z: f32, #[gpu] pub outer_cos: f32,
    // color_intensity
    #[gpu] pub color_r: f32, #[gpu] pub color_g: f32,
    #[gpu] pub color_b: f32, #[gpu] pub intensity: f32,
    #[gpu] pub shadow_index: u32,
    #[gpu] pub light_type: u32,
    #[gpu] pub inner_angle: f32,
    #[gpu] pub _pad: u32,
    #[gpu] pub god_rays_enabled: u32,
    #[gpu] pub god_rays_density: f32,
    #[gpu] pub god_rays_weight: f32,
    #[gpu] pub god_rays_decay: f32,
    #[gpu] pub god_rays_exposure: f32,
    #[gpu] pub flare_enabled: u32,
    #[gpu] pub flare_type: u32,
    #[gpu] pub flare_intensity: f32,
    #[gpu] pub flare_scale: f32,
    #[gpu] pub flare_tint_r: f32,
    #[gpu] pub flare_tint_g: f32,
    #[gpu] pub flare_tint_b: f32,
    #[gpu] pub ies_profile_index: i32,
    #[gpu] pub light_function_index: i32,
    #[gpu] pub ies_angle_scale: f32,
    #[gpu] pub ies_angle_offset: f32,
}

// ── CPU-only companion components (no derive, no GPU cost) ──

/// CPU-only bookkeeping attached to the same entity as [`HelioGpuInstance`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct HelioCpuInstance {
    pub groups: GroupMask,
    pub movability: libhelio::Movability,
    pub user_tag: u64,
}

/// CPU-only bookkeeping attached to the same entity as [`HelioGpuLight`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct HelioCpuLight {
    pub movability: libhelio::Movability,
    pub user_tag: u64,
    pub gpu_index: u32,
}

/// Register every Helio component's GPU columns on the world-mirror store.
pub(crate) fn register_gpu_columns(
    store: &mut pulsar_scenedb::gpu::SceneGpuStore,
    device: &std::sync::Arc<wgpu::Device>,
) {
    // HelioGpuInstance::register_gpu_columns_growable(store, 1024, device.clone());
    // HelioGpuLight::register_gpu_columns_growable(store, 256, device.clone());
}
