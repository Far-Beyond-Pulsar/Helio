//! Uniform blocks shared with `shaders/foliage_place.wgsl` and `shaders/foliage_cull.wgsl`.
//!
//! Both are plain scalars (80 and 64 bytes), and both sizes are pinned by a `const _` assert in
//! the style of `helio_pass_virtual_geometry`. A size or field-order change that the WGSL does not
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

/// Per-frame constants for the placement dispatches (`cs_query`, `cs_ground_plane`,
/// `cs_place`). Exactly 80 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
pub struct PlaceUniforms {
    /// Edge length of one tile in chart metres — [`crate::FOLIAGE_TILE_SIZE_METERS`].
    pub tile_size: f32,

    /// Resolution `G` of the stratified candidate grid; the shader evaluates `G²`
    /// candidates per tile.
    ///
    /// Derived on the CPU from the densest ground's density, and clamped so `G²` never
    /// exceeds the tile's arena slab. That clamp is why blades thin out uniformly under
    /// budget pressure instead of the arena filling up and the last candidates in scan
    /// order vanishing — which would leave a bald corner on every tile.
    pub candidate_grid: u32,
    /// Edge length of one cluster block in candidate cells (`sqrt(cluster_size)`).
    ///
    /// Candidates are mapped block-linearly so that the `cluster_size` consecutive
    /// indices forming one cluster occupy a square patch: the L3 clump card is anchored on
    /// its cluster, and row-major strips rendered the far field as straight rows.
    pub cluster_edge: u32,

    /// Blades one tile may own. Fixed and equal for every ring slot; see
    /// [`crate::FoliagePlacePass::blades_per_tile`].
    pub slab_capacity: u32,

    /// Tiles placed this frame from last frame's answers (the place queue's leading
    /// entries).
    pub place_tile_count: u32,
    /// Tiles whose ground is queried this frame (the entries after them).
    pub query_tile_count: u32,

    /// [`crate::FoliageQuality::density_multiplier`].
    pub density_multiplier: f32,

    /// Candidates per square metre of the grid, `candidate_grid² / tile area`.
    ///
    /// The denominator of the acceptance probability: a candidate grows a blade with
    /// probability `(sum of the densities of the types growing on its ground) *
    /// density_multiplier / grid_density`, then takes a type by its share of that sum.
    pub grid_density: f32,

    /// Number of valid entries in the foliage type table.
    pub type_count: u32,

    /// Tallest `height_range[1]` over the registered types, in metres. Used to dilate the
    /// tile's bounds so wind-displaced blades do not fall outside them.
    pub max_foliage_height: f32,

    /// Non-zero only when a terrain capture exists *and* its ring transform has been
    /// published through [`crate::FoliagePlacePass::set_terrain_transform`]: the built-in
    /// ground samples it, else the plane y = 0.
    pub terrain_valid: u32,
    /// World X of the terrain capture's minimum corner.
    pub terrain_origin_x: f32,
    /// World Z of the terrain capture's minimum corner.
    pub terrain_origin_z: f32,
    /// Side length in metres covered by the terrain capture.
    pub terrain_extent: f32,

    /// Number of valid entries in the foliage layer table.
    ///
    /// The layer table buffer is fixed-capacity, so the placement shader must not loop
    /// `arrayLength` entries. Zero means "no layers": carpet everything.
    pub layer_count: u32,

    /// A query's segment: from this far over the eye's height along the tile's up...
    pub search_above: f32,
    /// ...to this far under it.
    pub search_below: f32,
    /// The eye in render space.
    pub eye: [f32; 3],
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
    /// **This is not optional.** Until a pyramid is built from real depth its texels are
    /// whatever the graph's texture pool left there, and testing against them can cull
    /// the entire world for that frame — and again after every resize rebuilds the
    /// graph. `vg_cull.wgsl` carries the identical guard for the identical reason.
    pub hiz_valid: u32,

    /// Blades per cull cluster — [`crate::FoliageQuality::cluster_granularity`].
    pub cluster_size: u32,
    /// `ceil(slab_capacity / cluster_size)`; the cluster index space per tile.
    pub clusters_per_tile: u32,
    /// [`crate::FOLIAGE_VISIBLE_PER_LOD_CAPACITY`], duplicated here so the shader can
    /// address regions and clamp `instance_count` without a second constant to keep in
    /// sync by hand.
    pub per_lod_capacity: u32,

    /// Edge length of one tile in metres.
    pub tile_size: f32,
    /// [`crate::FoliageQuality::lod_distance_scale`], the `quality_scale`
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
        std::mem::size_of::<PlaceUniforms>() == 80,
        "PlaceUniforms must be exactly 80 bytes to match foliage_place.wgsl"
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
        assert_eq!(std::mem::size_of::<PlaceUniforms>(), 80);
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
        // a reordered field fails in CI rather than in a shader that reads `grid_density`
        // out of `type_count` and places no grass at all.
        let value = PlaceUniforms::default();
        let base = &value as *const _ as usize;
        let at = |field: *const u8| field as usize - base;
        let f = |x: &f32| x as *const f32 as *const u8;
        let u = |x: &u32| x as *const u32 as *const u8;
        assert_eq!(at(f(&value.tile_size)), 0);
        assert_eq!(at(u(&value.candidate_grid)), 4);
        assert_eq!(at(u(&value.cluster_edge)), 8);
        assert_eq!(at(u(&value.slab_capacity)), 12);
        assert_eq!(at(u(&value.place_tile_count)), 16);
        assert_eq!(at(u(&value.query_tile_count)), 20);
        assert_eq!(at(f(&value.density_multiplier)), 24);
        assert_eq!(at(f(&value.grid_density)), 28);
        assert_eq!(at(u(&value.type_count)), 32);
        assert_eq!(at(f(&value.max_foliage_height)), 36);
        assert_eq!(at(u(&value.terrain_valid)), 40);
        assert_eq!(at(f(&value.terrain_origin_x)), 44);
        assert_eq!(at(f(&value.terrain_origin_z)), 48);
        assert_eq!(at(f(&value.terrain_extent)), 52);
        assert_eq!(at(u(&value.layer_count)), 56);
        assert_eq!(at(f(&value.search_above)), 60);
        assert_eq!(at(f(&value.search_below)), 64);
        assert_eq!(at(value.eye.as_ptr() as *const u8), 68);
    }

    #[test]
    fn cull_uniform_field_offsets_match_the_wgsl_mirror() {
        let value = FoliageCullUniforms::default();
        let base = &value as *const _ as usize;
        let offset_of = |field: *const u8| field as usize - base;

        assert_eq!(offset_of(&value.tile_count as *const u32 as *const u8), 0);
        assert_eq!(offset_of(&value.screen_width as *const u32 as *const u8), 4);
        assert_eq!(
            offset_of(&value.screen_height as *const u32 as *const u8),
            8
        );
        assert_eq!(
            offset_of(&value.hiz_mip_count as *const u32 as *const u8),
            12
        );
        assert_eq!(offset_of(&value.hiz_valid as *const u32 as *const u8), 16);
        assert_eq!(
            offset_of(&value.cluster_size as *const u32 as *const u8),
            20
        );
        assert_eq!(
            offset_of(&value.clusters_per_tile as *const u32 as *const u8),
            24
        );
        assert_eq!(
            offset_of(&value.per_lod_capacity as *const u32 as *const u8),
            28
        );
        assert_eq!(offset_of(&value.tile_size as *const f32 as *const u8), 32);
        assert_eq!(
            offset_of(&value.lod_quality_scale as *const f32 as *const u8),
            36
        );
        assert_eq!(offset_of(&value.type_count as *const u32 as *const u8), 40);
        assert_eq!(
            offset_of(&value.cluster_dispatch_width as *const u32 as *const u8),
            44
        );
        assert_eq!(
            offset_of(&value.max_foliage_height as *const f32 as *const u8),
            48
        );
        assert_eq!(offset_of(&value.wpo_extent as *const f32 as *const u8), 52);
        assert_eq!(
            offset_of(&value.lod_fade_band as *const f32 as *const u8),
            56
        );
        assert_eq!(offset_of(value._pad.as_ptr() as *const u8), 60);
    }

    #[test]
    fn reserved_padding_defaults_to_zero() {
        assert_eq!(PlaceUniforms::default().layer_count, 0);
        assert_eq!(FoliageCullUniforms::default()._pad, [0; 1]);
    }
}
