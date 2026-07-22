//! GPU light types.

use bytemuck::{Pod, Zeroable};

/// GPU light type discriminant.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightType {
    Directional = 0,
    Point = 1,
    Spot = 2,
    Area = 3,
}

/// Per-light GPU data. 96 bytes.
///
/// # WGSL equivalent
/// ```wgsl
/// struct GpuLight {
///     position_range:    vec4<f32>,  // xyz = position, w = range
///     direction_outer:   vec4<f32>,  // xyz = direction, w = spot outer angle cos
///     color_intensity:   vec4<f32>,  // xyz = linear RGB, w = intensity
///     shadow_index:      u32,        // -1 if no shadow
///     light_type:        u32,        // LightType enum
///     inner_angle:       f32,        // spot inner angle cos
///     _pad:              u32,
///     god_rays_enabled:  u32,
///     god_rays_density:  f32,
///     god_rays_weight:   f32,
///     god_rays_decay:    f32,
///     god_rays_exposure: f32,
///     _pad2_0:           u32,
///     _pad2_1:           u32,
///     _pad2_2:           u32,
/// }
/// ```
///
/// The tail padding is three scalars, not a `vec3<u32>`: a WGSL `vec3` has
/// 16-byte alignment, so it would be pushed from offset 84 to 96 and grow the
/// struct to 112 — silently mismatching the 96-byte Rust side.
///
/// # Layout contract
///
/// This struct is mirrored **by hand** in every WGSL shader that binds the scene
/// light storage buffer. A field added here without updating each mirror silently
/// misreads the buffer — no validation error, just wrong lighting. Current mirrors:
///
/// - `helio-pass-deferred-light/shaders/deferred_lighting.wgsl`
/// - `helio-pass-light-cull/shaders/light_cull.wgsl`
/// - `helio-pass-hlfs/shaders/hlfs_shade.wgsl`
/// - `helio-pass-hlfs/shaders/hlfs_importance.wgsl`
/// - `helio-pass-shadow-matrix/shaders/shadow_matrices.wgsl`
/// - `helio-pass-voxel-mesh/shaders/voxel_meshlet.wgsl`
/// - `helio-pass-voxel-raymarch/shaders/voxel_raymarch.wgsl`
///
/// `helio-pass-radiance-cascades/shaders/rc_trace.wgsl` also declares a `GpuLight`,
/// but it is dormant (bundled via `include_str!`, never compiled — it needs
/// `EXPERIMENTAL_RAY_QUERY`) and its layout has already diverged. Reconcile it
/// against this struct before that pass is revived.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuLight {
    /// World-space position (xyz) + effective range (w)
    pub position_range: [f32; 4],
    /// Direction (xyz, normalized) + spot outer cos angle (w)
    pub direction_outer: [f32; 4],
    /// Linear RGB color (xyz) + intensity (w, in candela for point/spot, lux for directional)
    pub color_intensity: [f32; 4],
    /// Shadow map slice index (-1u32 = no shadow)
    pub shadow_index: u32,
    /// LightType discriminant
    pub light_type: u32,
    /// Spot inner cos angle
    pub inner_angle: f32,
    pub _pad: u32,

    // ── Light shafts / god rays (volumetric fog pass) ──
    /// Non-zero to accumulate light shafts for this light in the volumetric fog pass.
    pub god_rays_enabled: u32,
    /// Participating-media density along the shaft, independent of scene fog density.
    pub god_rays_density: f32,
    /// Per-step in-scattering weight.
    pub god_rays_weight: f32,
    /// Per-step attenuation (< 1.0 shortens the shaft).
    pub god_rays_decay: f32,
    /// Final scale applied to the accumulated shaft radiance.
    pub god_rays_exposure: f32,
    pub _pad2: [u32; 3],
}

// The WGSL mirrors above assume this exact size. A storage-buffer array of
// GpuLight strides by size_of::<GpuLight>(), so any drift shifts every light
// after index 0.
const _: () = assert!(std::mem::size_of::<GpuLight>() == 96);
// WGSL rounds the array stride up to the struct's alignment, which is 16 here
// (vec4<f32> members). If the Rust size were not a multiple of 16 the two sides
// would stride differently even at identical field counts.
const _: () = assert!(std::mem::size_of::<GpuLight>() % 16 == 0);

impl Default for GpuLight {
    fn default() -> Self {
        Self {
            position_range: [0.0, 0.0, 0.0, 0.0],
            direction_outer: [0.0, -1.0, 0.0, 0.0],
            color_intensity: [1.0, 1.0, 1.0, 1.0],
            shadow_index: u32::MAX,
            light_type: LightType::Point as u32,
            inner_angle: 0.0,
            _pad: 0,

            // Off by default, but with usable values behind the switch: the fog
            // pass multiplies by density, weight and exposure, so leaving those at
            // 0.0 would make `god_rays_enabled = 1` render nothing and look broken.
            god_rays_enabled: 0,
            god_rays_density: 1.0,
            god_rays_weight: 0.6,
            god_rays_decay: 1.0,
            god_rays_exposure: 0.7,
            _pad2: [0; 3],
        }
    }
}

/// Per-light shadow matrix for the shadow map atlas.
/// Layout: one `mat4x4<f32>` = 64 bytes, matching `LightMatrix` in all WGSL shaders.
/// 6 consecutive entries per light (indices light_idx*6 .. light_idx*6+5):
///   - Point lights: 6 cube-face view-projection matrices (+X/-X/+Y/-Y/+Z/-Z)
///   - Spot lights:  face 0 = perspective view-proj, faces 1-5 = identity (unused)
///   - Directional:  face 0 = ortho view-proj,       faces 1-5 = identity (unused)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuShadowMatrix {
    /// Light-space view-projection matrix (64 bytes, matches `LightMatrix { mat: mat4x4<f32> }`)
    pub light_view_proj: [f32; 16],
}

