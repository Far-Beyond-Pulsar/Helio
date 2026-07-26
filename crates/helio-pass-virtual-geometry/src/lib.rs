pub mod rendering;

pub use rendering::VirtualGeometryPass;

use bytemuck::{Pod, Zeroable};
use helio_core::GpuInstanceData;

// ═══════════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════════

/// Bindless texture array size per shader stage.
/// Capped at 16 on wasm32, Apple native Metal, and Android; 256 on other desktop backends.
#[cfg(not(any(target_arch = "wasm32", target_os = "macos", target_os = "ios", target_os = "android")))]
pub(crate) const MAX_TEXTURES: usize = 256;
#[cfg(any(target_arch = "wasm32", target_os = "macos", target_os = "ios", target_os = "android"))]
pub(crate) const MAX_TEXTURES: usize = 16;

pub const LOD_LEVEL_COUNT: u32 = 8;

/// Draw publication counters written by the GPU cull stages.
///
/// Slot 0 is the opaque meshlet draw count consumed by the first
/// indirect-count draw call, slot 1 is the alpha-test draw count consumed
/// by the second draw call, slot 2 is the capacity-rejection overflow
/// count, slots 3..10 form the selected-LOD object histogram, and slot 11
/// stores the largest LOD available among visible objects.
pub(crate) const DRAW_COUNTER_COUNT: u64 = 12;
pub(crate) const DRAW_COUNTER_BYTES: u64 = DRAW_COUNTER_COUNT * 4;

/// Latest non-blocking GPU readback from the virtual-geometry cull pass.
///
/// Debug readback is only scheduled while the LOD heatmap is active. Values
/// can trail the rendered frame by a few frames rather than stalling the GPU.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VirtualGeometryDebugStats {
    pub visible_meshlets: u32,
    pub rejected_meshlets: u32,
    pub lod_object_counts: [u32; LOD_LEVEL_COUNT as usize],
    pub max_available_lod: u32,
}

impl VirtualGeometryDebugStats {
    pub fn visible_objects(self) -> u32 {
        self.lod_object_counts.iter().copied().sum()
    }

    pub fn selected_lod_range(self) -> Option<(u32, u32)> {
        let first = self.lod_object_counts.iter().position(|&count| count != 0)? as u32;
        let last = self.lod_object_counts.iter().rposition(|&count| count != 0)? as u32;
        Some((first, last))
    }

    fn from_counters(counters: &[u32]) -> Option<Self> {
        if counters.len() < DRAW_COUNTER_COUNT as usize {
            return None;
        }

        let mut lod_object_counts = [0; LOD_LEVEL_COUNT as usize];
        lod_object_counts.copy_from_slice(&counters[3..11]);
        Some(Self {
            visible_meshlets: counters[0] + counters[1],
            rejected_meshlets: counters[2],
            lod_object_counts,
            max_available_lod: counters[11].min(LOD_LEVEL_COUNT - 1),
        })
    }
}

pub(crate) const INITIAL_MESHLETS: u64 = 1024;
pub(crate) const INITIAL_OBJECTS: u64 = 256;
pub(crate) const INITIAL_INSTANCES: u64 = 256;

/// Default ceiling for visible meshlets published as indexed indirect draws.
///
/// At 36 bytes per slot (20-byte indirect command plus 16-byte draw metadata),
/// this bounds the publication buffers to 9 MiB plus the two counters. Callers
/// with a measured platform-specific budget can override it explicitly.
/// Upper bound on how many meshlet draws a single frame can publish to the
/// indirect/draw-metadata buffers. This is a genuine ceiling, not a
/// pre-allocation size — `VirtualGeometryPass` starts small and grows these
/// buffers toward this value only as scenes actually need it (see
/// `prepare()` in `rendering.rs`), so raising it costs nothing for scenes
/// that don't need the extra headroom.
///
/// 262_144 (the previous value) is nowhere near enough for any real scene
/// with more than a couple hundred separate VG object instances: the budget
/// is sized against `vg.max_draw_count`, the *worst-case* sum of every
/// object's LOD0 meshlet count, which scales with instance count, not just
/// unique mesh complexity. A scene with ~1200 instances of a ~90k-triangle,
/// 7-section asset (the shipyard demo's shipping containers) needs
/// ~2.5 million draw slots — once `cull_meshlet()`'s atomic slot counter
/// (`vg_cull.wgsl`) exceeds the budget, every excess meshlet is silently
/// rejected via the overflow counter, which looks exactly like the culling
/// tests themselves are wrongly rejecting on-screen geometry: it isn't the
/// visibility tests, it's the output buffer running out of room, and which
/// meshlets lose that race is effectively fixed frame-to-frame for a static
/// scene, so the drops look like a consistent, reproducible mis-cull.
pub const DEFAULT_MAX_PUBLISHED_MESHLETS: u32 = 4_194_304;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VirtualGeometryBudget {
    max_published_meshlets: u32,
}

impl VirtualGeometryBudget {
    pub const fn new(max_published_meshlets: u32) -> Self {
        assert!(
            max_published_meshlets > 0,
            "virtual geometry publication budget must be non-zero"
        );
        Self {
            max_published_meshlets,
        }
    }

    pub const fn publication_bytes(self) -> u64 {
        self.max_published_meshlets as u64 * 36 + DRAW_COUNTER_BYTES
    }

    pub const fn max_published_meshlets(self) -> u32 {
        self.max_published_meshlets
    }

    pub const fn clamp_draw_count(self, worst_case_draw_count: u32) -> u32 {
        if worst_case_draw_count < self.max_published_meshlets {
            worst_case_draw_count
        } else {
            self.max_published_meshlets
        }
    }
}

impl Default for VirtualGeometryBudget {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_PUBLISHED_MESHLETS)
    }
}

/// Per-instance values used by meshlet culling. Kept separate from
/// `GpuInstanceData` so the cull shader does not recompute matrix norms for
/// every meshlet in an instance.
///
/// `cull_flags` bit layout:
///   bit 0 — cone culling enabled (angle-preserving, non-reflected transform)
///   bit 1 — opaque material (no alpha test)
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct InstanceCullData {
    pub max_scale: f32,
    pub min_scale: f32,
    pub cull_flags: u32,
    pub valid_transform: u32,
}

impl InstanceCullData {
    const CULL_FLAG_CONE_CULL: u32 = 1 << 0;
    const CULL_FLAG_OPAQUE: u32 = 1 << 1;

    pub(crate) fn from_instance(instance: &GpuInstanceData, material_flags: u32) -> Self {
        const FLAG_ALPHA_TEST: u32 = 1 << 2; // libhelio::FLAG_ALPHA_TEST
        let model = &instance.model;
        let scale_x = (model[0] * model[0] + model[1] * model[1] + model[2] * model[2]).sqrt();
        let scale_y = (model[4] * model[4] + model[5] * model[5] + model[6] * model[6]).sqrt();
        let scale_z = (model[8] * model[8] + model[9] * model[9] + model[10] * model[10]).sqrt();
        let max_scale = scale_x.max(scale_y).max(scale_z);
        let min_scale = scale_x.min(scale_y).min(scale_z);
        let dot_xy = model[0] * model[4] + model[1] * model[5] + model[2] * model[6];
        let dot_xz = model[0] * model[8] + model[1] * model[9] + model[2] * model[10];
        let dot_yz = model[4] * model[8] + model[5] * model[9] + model[6] * model[10];
        let determinant = model[0] * (model[5] * model[10] - model[6] * model[9])
            - model[4] * (model[1] * model[10] - model[2] * model[9])
            + model[8] * (model[1] * model[6] - model[2] * model[5]);
        let affine = model[3].abs() <= 1.0e-6
            && model[7].abs() <= 1.0e-6
            && model[11].abs() <= 1.0e-6
            && (model[15] - 1.0).abs() <= 1.0e-4;
        let valid = model.iter().all(|value| value.is_finite())
            && max_scale.is_finite()
            && min_scale > 1.0e-8
            && affine;
        let uniform_tolerance = max_scale * 1.0e-4;
        let orthogonal_tolerance = max_scale * max_scale * 1.0e-4;
        let angle_preserving = max_scale - min_scale <= uniform_tolerance
            && dot_xy.abs() <= orthogonal_tolerance
            && dot_xz.abs() <= orthogonal_tolerance
            && dot_yz.abs() <= orthogonal_tolerance;
        let cone_cull = valid && angle_preserving && determinant > 0.0;
        let opaque = (material_flags & FLAG_ALPHA_TEST) == 0;

        Self {
            max_scale: if valid { max_scale } else { 0.0 },
            min_scale: if valid { min_scale } else { 0.0 },
            cull_flags: (u32::from(cone_cull) * Self::CULL_FLAG_CONE_CULL)
                | (u32::from(opaque) * Self::CULL_FLAG_OPAQUE),
            valid_transform: u32::from(valid),
        }
    }
}

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

    /// Maximum tolerated geometric simplification error in output pixels.
    /// The cull shader selects the coarsest whole-object LOD below this bound.
    pub fn max_error_pixels(self) -> f32 {
        match self {
            LodQuality::Low => 4.0,
            LodQuality::Medium => 2.0,
            LodQuality::High => 1.0,
            LodQuality::Ultra => 0.5,
        }
    }
}

#[cfg(test)]
pub(crate) fn select_object_lod(
    errors: &[f32; LOD_LEVEL_COUNT as usize],
    lod_count: u32,
    max_scale: f32,
    focal_pixels: f32,
    closest_distance: f32,
    max_error_pixels: f32,
) -> u32 {
    if lod_count == 0
        || !max_scale.is_finite()
        || !focal_pixels.is_finite()
        || !closest_distance.is_finite()
        || !max_error_pixels.is_finite()
    {
        return 0;
    }

    let mut selected = 0;
    let denominator = closest_distance.max(1.0e-4);
    for level in 1..lod_count.min(LOD_LEVEL_COUNT) {
        let projected_error = errors[level as usize] * max_scale * focal_pixels / denominator;
        if projected_error <= max_error_pixels {
            selected = level;
        } else {
            break;
        }
    }
    selected
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
    pub object_count: u32,
    pub screen_width: u32,
    pub screen_height: u32,
    pub hiz_mip_count: u32,
    pub draw_capacity: u32,
    pub lod_error_threshold_px: f32,
    pub object_dispatch_width: u32,
    pub work_item_count: u32,
    pub work_dispatch_width: u32,
    /// 0 on frame 0 (or right after a resize rebuilds the graph), when the Hi-Z
    /// pyramid hasn't been built from any real depth yet — an untouched wgpu
    /// texture reads back as 0.0, which this engine's non-reversed depth
    /// convention (near = 0.0) means the shader's `near_z > hiz_depth` test
    /// would otherwise treat as "everything is behind the (nonexistent)
    /// occluder," incorrectly culling the entire visible scene for that frame.
    /// Mirrors the equivalent frame-0 guard in `OcclusionCullPass`.
    pub hiz_valid: u32,
    _pad1: u32,
    _pad2: u32,
}

#[cfg(test)]
mod tests {
    use super::{
        select_object_lod, CullUniforms, InstanceCullData, LodQuality,
        VirtualGeometryBudget, VirtualGeometryDebugStats, DEFAULT_MAX_PUBLISHED_MESHLETS,
    };
    use helio_core::GpuInstanceData;

    fn instance_with_model(model: [f32; 16]) -> GpuInstanceData {
        GpuInstanceData {
            model,
            normal_mat: [0.0; 12],
            bounds: [0.0; 4],
            mesh_id: 0,
            material_id: 0,
            flags: 0,
            lightmap_index: u32::MAX,
        }
    }

    #[test]
    fn instance_cull_data_is_gpu_aligned() {
        assert_eq!(std::mem::size_of::<InstanceCullData>(), 16);
        assert_eq!(std::mem::size_of::<CullUniforms>(), 48);
    }

    #[test]
    fn default_publication_budget_is_144_mib_plus_counters() {
        let budget = VirtualGeometryBudget::default();
        assert_eq!(budget.max_published_meshlets(), DEFAULT_MAX_PUBLISHED_MESHLETS);
        assert_eq!(budget.publication_bytes(), 144 * 1024 * 1024 + 48);
        assert_eq!(budget.clamp_draw_count(65_536), 65_536);
        assert_eq!(budget.clamp_draw_count(u32::MAX), DEFAULT_MAX_PUBLISHED_MESHLETS);
    }

    #[test]
    #[should_panic(expected = "publication budget must be non-zero")]
    fn publication_budget_rejects_zero() {
        let _ = VirtualGeometryBudget::new(0);
    }

    #[test]
    fn uniform_scale_keeps_cone_culling_enabled() {
        let instance = instance_with_model([
            2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 4.0, 5.0, 6.0, 1.0,
        ]);
        let cull = InstanceCullData::from_instance(&instance, 0);

        assert_eq!(cull.valid_transform, 1);
        assert_eq!(cull.cull_flags & 1, 1);
        assert_eq!(cull.cull_flags >> 1 & 1, 1);
        assert_eq!(cull.max_scale, 2.0);
        assert_eq!(cull.min_scale, 2.0);
    }

    #[test]
    fn non_uniform_scale_disables_cone_culling_but_keeps_conservative_radius() {
        let instance = instance_with_model([
            4.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]);
        let cull = InstanceCullData::from_instance(&instance, 0);

        assert_eq!(cull.valid_transform, 1);
        assert_eq!(cull.cull_flags & 1, 0);
        assert_eq!(cull.max_scale, 4.0);
        assert_eq!(cull.min_scale, 1.0);
    }

    #[test]
    fn alpha_test_material_clears_opaque_bit() {
        let instance = instance_with_model([
            2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 4.0, 5.0, 6.0, 1.0,
        ]);
        let cull = InstanceCullData::from_instance(&instance, 0);
        assert_eq!(cull.cull_flags >> 1 & 1, 1, "no flags = opaque");

        let cull_alpha = InstanceCullData::from_instance(&instance, 1 << 2);
        assert_eq!(cull_alpha.cull_flags >> 1 & 1, 0, "FLAG_ALPHA_TEST clears opaque");
    }

    #[test]
    fn singular_or_non_finite_transform_is_rejected() {
        let singular = instance_with_model([0.0; 16]);
        assert_eq!(
            InstanceCullData::from_instance(&singular, 0).valid_transform,
            0
        );

        let mut non_finite_model = [0.0; 16];
        non_finite_model[0] = f32::NAN;
        let non_finite = instance_with_model(non_finite_model);
        assert_eq!(
            InstanceCullData::from_instance(&non_finite, 0).valid_transform,
            0
        );
    }

    #[test]
    fn shear_or_reflection_disables_cone_culling() {
        let shear = instance_with_model([
            1.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]);
        assert_eq!(InstanceCullData::from_instance(&shear, 0).cull_flags & 1, 0);

        let reflection = instance_with_model([
            -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]);
        assert_eq!(
            InstanceCullData::from_instance(&reflection, 0).cull_flags & 1,
            0
        );
    }

    #[test]
    fn object_lod_uses_measured_projected_error() {
        let errors = [0.0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64];

        assert_eq!(select_object_lod(&errors, 8, 1.0, 1000.0, 100.0, 1.0), 4);
        assert_eq!(select_object_lod(&errors, 8, 1.0, 1000.0, 10.0, 1.0), 1);
        assert_eq!(select_object_lod(&errors, 8, 2.0, 1000.0, 100.0, 1.0), 3);
    }

    #[test]
    fn higher_lod_quality_has_a_stricter_pixel_error_bound() {
        assert!(LodQuality::Low.max_error_pixels() > LodQuality::Medium.max_error_pixels());
        assert!(LodQuality::Medium.max_error_pixels() > LodQuality::High.max_error_pixels());
        assert!(LodQuality::High.max_error_pixels() > LodQuality::Ultra.max_error_pixels());
    }

    #[test]
    fn debug_stats_decode_the_gpu_counter_layout() {
        let stats = VirtualGeometryDebugStats::from_counters(&[
            100, 23, 4, 0, 2, 5, 0, 0, 1, 0, 0, 6,
        ])
        .expect("complete counter layout");

        assert_eq!(stats.visible_meshlets, 123);
        assert_eq!(stats.rejected_meshlets, 4);
        assert_eq!(stats.visible_objects(), 8);
        assert_eq!(stats.selected_lod_range(), Some((1, 5)));
        assert_eq!(stats.max_available_lod, 6);
        assert!(VirtualGeometryDebugStats::from_counters(&[0; 11]).is_none());
    }
}
