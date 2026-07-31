//! Uniform blocks shared with `shaders/foliage_place.wgsl` and `shaders/foliage_cull.wgsl`.
//!
//! Both are 64 bytes of plain scalars, and both sizes are pinned by a `const _` assert in
//! the style of `libhelio::meshlet`. A size or field-order change that the WGSL does not
//! follow does not fail loudly: the shader reads every field after the change from the
//! wrong offset, and the symptom is grass at the wrong density in the wrong places, which
//! reads as a placement bug rather than a layout one.
//!
//! Every field is a scalar for the same reason `GpuFoliageType` is: WGSL gives `vec3` a
//! 16-byte alignment and `vec2` an 8-byte one, so a single vector member can silently
//! shift everything after it. Keeping "all scalars, always" unconditionally true means no
//! future reorder can make a vector declaration *accidentally* correct and then break on
//! the reorder after that.

use bytemuck::{Pod, Zeroable};

/// Per-frame constants for the placement dispatch. Exactly 64 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
pub struct PlaceUniforms {
    /// Edge length of one tile in metres — [`helio_foliage_core::FOLIAGE_TILE_SIZE_METERS`].
    pub tile_size: f32,

    /// Resolution `G` of the stratified candidate grid; the shader evaluates `G²`
    /// candidates per tile.
    ///
    /// Derived on the CPU from the *densest* registered type, and clamped so `G²` never
    /// exceeds the tile's arena slab. That clamp is why blades thin out uniformly under
    /// budget pressure instead of the arena filling up and the last candidates in scan
    /// order vanishing — which would leave a bald corner on every tile rather than
    /// slightly sparser grass everywhere.
    pub candidate_grid: u32,
    /// Edge length of one cluster block in candidate cells (`sqrt(cluster_size)`).
    ///
    /// Candidates are mapped block-linearly so that the `cluster_size` consecutive
    /// indices forming one cluster occupy a square patch. Row-major mapping made a
    /// cluster a 1-cell-tall strip, and since the L3 clump card is anchored on its
    /// cluster, the far field rendered as straight rows of cards.
    pub cluster_edge: u32,

    /// Blades one tile may own. Fixed and equal for every ring slot; see
    /// [`crate::FoliagePlacePass::blades_per_tile`].
    pub slab_capacity: u32,

    /// Number of valid entries in the place queue this frame, bounded by
    /// `max_tiles_per_frame`.
    pub queued_tile_count: u32,

    /// [`helio_foliage_core::FoliageQuality::density_multiplier`].
    pub density_multiplier: f32,

    /// `max(type.density) * density_multiplier` over the registered types.
    ///
    /// The denominator of the rejection sampler: one candidate grid serves every type,
    /// and each type accepts with probability `its density / this`. Zero types leaves
    /// this at zero and the shader's `max(..., 1e-6)` keeps the division finite.
    pub max_density: f32,

    /// Number of valid entries in the foliage type table.
    pub type_count: u32,

    /// Tallest `height_range[1]` over the registered types, in metres. Used to dilate the
    /// tile's vertical bounds so wind-displaced blades do not fall outside them.
    pub max_foliage_height: f32,

    /// Non-zero only when the terrain capture exists *and* its ring transform has been
    /// published through [`crate::FoliagePlacePass::set_terrain_transform`].
    ///
    /// Zero in every shipped configuration today, because `FoliageTerrainPass` does not
    /// exist yet. See `sample_terrain` in the placement shader for the temporary fallback.
    pub terrain_valid: u32,

    /// World X of the terrain capture's minimum corner.
    pub terrain_origin_x: f32,
    /// World Z of the terrain capture's minimum corner.
    pub terrain_origin_z: f32,
    /// Side length in metres covered by the terrain capture.
    pub terrain_extent: f32,

    /// Reserved. Must be written as zero so a future build can distinguish "left at the
    /// default" from "predates the field".
    pub _pad: [u32; 3],
}

/// Per-frame constants for the tile cull, cluster cull and finalize dispatches.
/// Exactly 64 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
pub struct FoliageCullUniforms {
    /// Ring slots to test — the tile table's capacity.
    pub tile_count: u32,
    /// Render target width in pixels.
    pub screen_width: u32,
    /// Render target height in pixels.
    pub screen_height: u32,
    /// Mip count of the Hi-Z pyramid.
    pub hiz_mip_count: u32,

    /// Zero on frame 0 and whenever the Hi-Z view is not routed.
    ///
    /// **This is not optional.** An untouched depth texture reads back as 0.0, and this
    /// engine's near-is-0.0 convention would read that as "everything is behind the
    /// occluder", culling the entire world for that frame — and again after every resize
    /// rebuilds the graph. `vg_cull.wgsl` carries the identical guard for the identical
    /// reason.
    pub hiz_valid: u32,

    /// Blades per cull cluster — [`helio_foliage_core::FoliageQuality::cluster_granularity`].
    pub cluster_size: u32,
    /// `ceil(slab_capacity / cluster_size)`; the cluster index space per tile.
    pub clusters_per_tile: u32,
    /// [`crate::FOLIAGE_VISIBLE_PER_LOD_CAPACITY`], duplicated here so the shader can
    /// address regions and clamp `instance_count` without a second constant to keep in
    /// sync by hand.
    pub per_lod_capacity: u32,

    /// Edge length of one tile in metres.
    pub tile_size: f32,
    /// [`helio_foliage_core::FoliageQuality::lod_distance_scale`], the `quality_scale`
    /// argument to `select_blade_lod`.
    pub lod_quality_scale: f32,
    /// Number of valid entries in the foliage type table.
    pub type_count: u32,
    /// Width of the 2D cluster-cull dispatch grid, so the shader can linearise
    /// `workgroup_id` the way `vg_cull.wgsl` does when the workgroup count exceeds the
    /// device's single-dimension limit.
    pub cluster_dispatch_width: u32,

    /// Tallest `height_range[1]` over the registered types, in metres.
    pub max_foliage_height: f32,
    /// Wind displacement extent used to dilate cull bounds — [`crate::DEFAULT_WPO_EXTENT_METERS`].
    pub wpo_extent: f32,

    /// Reserved. Must be written as zero.
    pub lod_fade_band: f32,
    pub _pad: [u32; 1],
}

const _: () = {
    assert!(
        std::mem::size_of::<PlaceUniforms>() == 64,
        "PlaceUniforms must be exactly 64 bytes to match foliage_place.wgsl"
    );
    assert!(
        std::mem::size_of::<FoliageCullUniforms>() == 64,
        "FoliageCullUniforms must be exactly 64 bytes to match foliage_cull.wgsl"
    );
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_sizes_are_stable() {
        assert_eq!(std::mem::size_of::<PlaceUniforms>(), 64);
        assert_eq!(std::mem::size_of::<FoliageCullUniforms>(), 64);
        assert_eq!(std::mem::align_of::<PlaceUniforms>(), 4);
        assert_eq!(std::mem::align_of::<FoliageCullUniforms>(), 4);
    }

    #[test]
    fn uniforms_fit_the_minimum_guaranteed_uniform_binding_size() {
        // WebGPU guarantees at least 16 KiB of uniform buffer binding; 64 bytes is not
        // close to it, but the assert documents that these are *uniform* blocks and must
        // stay small enough to be one, not storage buffers in disguise.
        assert!(std::mem::size_of::<PlaceUniforms>() <= 16 * 1024);
        assert!(std::mem::size_of::<FoliageCullUniforms>() <= 16 * 1024);
    }

    #[test]
    fn place_uniform_field_offsets_match_the_wgsl_mirror() {
        // The WGSL struct is a hand transcription of this one. Pinning the offsets means
        // a reordered field fails in CI rather than in a shader that reads `max_density`
        // out of `type_count` and places no grass at all.
        let value = PlaceUniforms::default();
        let base = &value as *const _ as usize;
        let offset_of = |field: *const u8| field as usize - base;

        assert_eq!(offset_of(&value.tile_size as *const f32 as *const u8), 0);
        assert_eq!(offset_of(&value.candidate_grid as *const u32 as *const u8), 4);
        assert_eq!(offset_of(&value.cluster_edge as *const u32 as *const u8), 8);
        assert_eq!(offset_of(&value.slab_capacity as *const u32 as *const u8), 12);
        assert_eq!(offset_of(&value.queued_tile_count as *const u32 as *const u8), 16);
        assert_eq!(offset_of(&value.density_multiplier as *const f32 as *const u8), 20);
        assert_eq!(offset_of(&value.max_density as *const f32 as *const u8), 24);
        assert_eq!(offset_of(&value.type_count as *const u32 as *const u8), 28);
        assert_eq!(offset_of(&value.max_foliage_height as *const f32 as *const u8), 32);
        assert_eq!(offset_of(&value.terrain_valid as *const u32 as *const u8), 36);
        assert_eq!(offset_of(&value.terrain_origin_x as *const f32 as *const u8), 40);
        assert_eq!(offset_of(&value.terrain_origin_z as *const f32 as *const u8), 44);
        assert_eq!(offset_of(&value.terrain_extent as *const f32 as *const u8), 48);
        assert_eq!(offset_of(value._pad.as_ptr() as *const u8), 52);
    }

    #[test]
    fn cull_uniform_field_offsets_match_the_wgsl_mirror() {
        let value = FoliageCullUniforms::default();
        let base = &value as *const _ as usize;
        let offset_of = |field: *const u8| field as usize - base;

        assert_eq!(offset_of(&value.tile_count as *const u32 as *const u8), 0);
        assert_eq!(offset_of(&value.screen_width as *const u32 as *const u8), 4);
        assert_eq!(offset_of(&value.screen_height as *const u32 as *const u8), 8);
        assert_eq!(offset_of(&value.hiz_mip_count as *const u32 as *const u8), 12);
        assert_eq!(offset_of(&value.hiz_valid as *const u32 as *const u8), 16);
        assert_eq!(offset_of(&value.cluster_size as *const u32 as *const u8), 20);
        assert_eq!(offset_of(&value.clusters_per_tile as *const u32 as *const u8), 24);
        assert_eq!(offset_of(&value.per_lod_capacity as *const u32 as *const u8), 28);
        assert_eq!(offset_of(&value.tile_size as *const f32 as *const u8), 32);
        assert_eq!(offset_of(&value.lod_quality_scale as *const f32 as *const u8), 36);
        assert_eq!(offset_of(&value.type_count as *const u32 as *const u8), 40);
        assert_eq!(offset_of(&value.cluster_dispatch_width as *const u32 as *const u8), 44);
        assert_eq!(offset_of(&value.max_foliage_height as *const f32 as *const u8), 48);
        assert_eq!(offset_of(&value.wpo_extent as *const f32 as *const u8), 52);
        assert_eq!(offset_of(&value.lod_fade_band as *const f32 as *const u8), 56);
        assert_eq!(offset_of(value._pad.as_ptr() as *const u8), 60);
    }

    #[test]
    fn reserved_padding_defaults_to_zero() {
        assert_eq!(PlaceUniforms::default()._pad, [0; 3]);
        assert_eq!(FoliageCullUniforms::default()._pad, [0; 1]);
    }
}
