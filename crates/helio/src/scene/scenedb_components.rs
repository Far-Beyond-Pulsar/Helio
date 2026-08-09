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

/// GPU-facing camera data — singleton entity (index 0) in World.
#[repr(C)]
#[derive(Debug, Clone, Copy, SceneStore)]
#[gpu(layout = packed)]
pub(crate) struct HelioGpuCamera {
    #[gpu] pub view_proj_00: f32, #[gpu] pub view_proj_01: f32,
    #[gpu] pub view_proj_02: f32, #[gpu] pub view_proj_03: f32,
    #[gpu] pub view_proj_10: f32, #[gpu] pub view_proj_11: f32,
    #[gpu] pub view_proj_12: f32, #[gpu] pub view_proj_13: f32,
    #[gpu] pub view_proj_20: f32, #[gpu] pub view_proj_21: f32,
    #[gpu] pub view_proj_22: f32, #[gpu] pub view_proj_23: f32,
    #[gpu] pub view_proj_30: f32, #[gpu] pub view_proj_31: f32,
    #[gpu] pub view_proj_32: f32, #[gpu] pub view_proj_33: f32,
    #[gpu] pub prev_vp_00: f32, #[gpu] pub prev_vp_01: f32,
    #[gpu] pub prev_vp_02: f32, #[gpu] pub prev_vp_03: f32,
    #[gpu] pub prev_vp_10: f32, #[gpu] pub prev_vp_11: f32,
    #[gpu] pub prev_vp_12: f32, #[gpu] pub prev_vp_13: f32,
    #[gpu] pub prev_vp_20: f32, #[gpu] pub prev_vp_21: f32,
    #[gpu] pub prev_vp_22: f32, #[gpu] pub prev_vp_23: f32,
    #[gpu] pub prev_vp_30: f32, #[gpu] pub prev_vp_31: f32,
    #[gpu] pub prev_vp_32: f32, #[gpu] pub prev_vp_33: f32,
    #[gpu] pub pos_x: f32, #[gpu] pub pos_y: f32, #[gpu] pub pos_z: f32,
    #[gpu] pub jitter_x: f32, #[gpu] pub jitter_y: f32,
}

/// GPU-facing decal data.
#[repr(C)]
#[derive(Debug, Clone, Copy, SceneStore)]
#[gpu(layout = packed)]
pub(crate) struct HelioGpuDecal {
    #[gpu] pub transform_00: f32, #[gpu] pub transform_01: f32,
    #[gpu] pub transform_02: f32, #[gpu] pub transform_03: f32,
    #[gpu] pub transform_10: f32, #[gpu] pub transform_11: f32,
    #[gpu] pub transform_12: f32, #[gpu] pub transform_13: f32,
    #[gpu] pub transform_20: f32, #[gpu] pub transform_21: f32,
    #[gpu] pub transform_22: f32, #[gpu] pub transform_23: f32,
    #[gpu] pub transform_30: f32, #[gpu] pub transform_31: f32,
    #[gpu] pub transform_32: f32, #[gpu] pub transform_33: f32,
    #[gpu] pub color_r: f32, #[gpu] pub color_g: f32, #[gpu] pub color_b: f32, #[gpu] pub color_a: f32,
    #[gpu] pub uv_offset_x: f32, #[gpu] pub uv_offset_y: f32,
    #[gpu] pub uv_scale_x: f32, #[gpu] pub uv_scale_y: f32,
    #[gpu] pub angle: f32,
    #[gpu] pub texture_index: u32,
    #[gpu] pub flags: u32,
}

/// GPU-facing reflection capture data.
#[repr(C)]
#[derive(Debug, Clone, Copy, SceneStore)]
#[gpu(layout = packed)]
pub(crate) struct HelioGpuReflectionCapture {
    #[gpu] pub position_x: f32, #[gpu] pub position_y: f32,
    #[gpu] pub position_z: f32, #[gpu] pub influence_radius: f32,
    #[gpu] pub cubemap_index: i32,
    #[gpu] pub blend_distance: f32,
    #[gpu] pub intensity: f32,
    #[gpu] pub flags: u32,
}

/// GPU-facing portal view data.
#[repr(C)]
#[derive(Debug, Clone, Copy, SceneStore)]
#[gpu(layout = packed)]
pub(crate) struct HelioGpuPortalView {
    #[gpu] pub view_00: f32, #[gpu] pub view_01: f32,
    #[gpu] pub view_02: f32, #[gpu] pub view_03: f32,
    #[gpu] pub view_10: f32, #[gpu] pub view_11: f32,
    #[gpu] pub view_12: f32, #[gpu] pub view_13: f32,
    #[gpu] pub view_20: f32, #[gpu] pub view_21: f32,
    #[gpu] pub view_22: f32, #[gpu] pub view_23: f32,
    #[gpu] pub view_30: f32, #[gpu] pub view_31: f32,
    #[gpu] pub view_32: f32, #[gpu] pub view_33: f32,
    #[gpu] pub proj_00: f32, #[gpu] pub proj_01: f32,
    #[gpu] pub proj_02: f32, #[gpu] pub proj_03: f32,
    #[gpu] pub proj_10: f32, #[gpu] pub proj_11: f32,
    #[gpu] pub proj_12: f32, #[gpu] pub proj_13: f32,
    #[gpu] pub proj_20: f32, #[gpu] pub proj_21: f32,
    #[gpu] pub proj_22: f32, #[gpu] pub proj_23: f32,
    #[gpu] pub proj_30: f32, #[gpu] pub proj_31: f32,
    #[gpu] pub proj_32: f32, #[gpu] pub proj_33: f32,
}

/// GPU-facing portal chain data.
#[repr(C)]
#[derive(Debug, Clone, Copy, SceneStore)]
#[gpu(layout = packed)]
pub(crate) struct HelioGpuPortalChain {
    #[gpu] pub clip_plane_x: f32, #[gpu] pub clip_plane_y: f32,
    #[gpu] pub clip_plane_z: f32, #[gpu] pub clip_plane_w: f32,
    #[gpu] pub parent_index: u32,
    #[gpu] pub view_count: u32,
}

/// Register every Helio component's GPU columns on the world-mirror store.
pub(crate) fn register_gpu_columns(
    store: &mut pulsar_scenedb::gpu::SceneGpuStore,
    device: &std::sync::Arc<wgpu::Device>,
) {
    HelioGpuInstance::register_gpu_columns_growable(store, 1024, device);
    HelioGpuLight::register_gpu_columns_growable(store, 256, device);
    HelioGpuCamera::register_gpu_columns_growable(store, 1, device);
    HelioGpuDecal::register_gpu_columns_growable(store, 256, device);
    HelioGpuReflectionCapture::register_gpu_columns_growable(store, 64, device);
    HelioGpuPortalView::register_gpu_columns_growable(store, 64, device);
    HelioGpuPortalChain::register_gpu_columns_growable(store, 32, device);
}
