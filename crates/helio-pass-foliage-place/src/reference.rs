//! CPU reference implementation of `cs_place`.
//!
//! This is the same relationship `helio_foliage_core::placement` has to the shader math,
//! one level up: it is a line-for-line transcription of the placement shader's candidate
//! loop, and it exists so the determinism contract is enforced by a test that runs in a
//! headless container rather than by hope.
//!
//! # What "deterministic" has to mean here
//!
//! The plan's §6.1 requires that the same `(tile_coord, generation)` produce a
//! **byte-identical** blade list on any GPU. That is stronger than "the same set of
//! blades survives", and it is the reason the shader compacts with a workgroup prefix sum
//! instead of `atomicAdd`: atomic ordering is unspecified, so an atomic append produces
//! the right *set* in an arbitrary *order*, and the arena bytes would differ between two
//! runs on the same machine. With the scan, a blade's slab index is a pure function of
//! its candidate index, and this function can predict it.
//!
//! # Scope
//!
//! Models the flat-plane terrain fallback only — height 0, `cos(slope)` 1 — because that
//! is what `sample_terrain` returns while `FoliageTerrainPass` does not exist. When the
//! capture lands, this function grows a terrain sampler argument; until then a reference
//! that pretended to sample a texture would be testing nothing.

use helio_foliage_core::{
    blade_seed, hash_to_unit, pack_blade, BladeParams, GpuBladeInstance, GpuFoliageType,
};

use crate::uniforms::PlaceUniforms;

/// One stratified candidate, before the accept/reject decision is applied.
///
/// Exposed so a failing determinism test can point at *which* candidate diverged rather
/// than only reporting that two blade lists differ.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReferenceCandidate {
    /// Index into the `G × G` stratified grid. This, not the output slab index, is what
    /// the seed is derived from.
    pub index: u32,
    /// [`blade_seed`] for this candidate.
    pub seed: u32,
    /// Tile-local position in `0.0..1.0`.
    pub tile_uv: [f32; 2],
    /// Foliage type this candidate drew from the rejection sampler.
    pub type_id: u8,
    /// Acceptance probability: the type's share of the densest type's density, gated by
    /// the slope and altitude bands.
    pub weight: f32,
    /// Whether the candidate became a blade.
    pub accepted: bool,
}

/// Result of placing one tile.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ReferencePlacement {
    /// Surviving blades, in candidate order — the exact byte layout the shader writes
    /// into the tile's arena slab.
    pub blades: Vec<GpuBladeInstance>,
    /// Candidates evaluated, i.e. `candidate_grid²`.
    pub candidates: u32,
    /// Blades that survived but did not fit the slab. Mirrors
    /// [`crate::COUNTER_PLACEMENT_OVERFLOW`]; normally zero, because the CPU clamps the
    /// candidate grid so the slab cannot be exceeded.
    pub dropped: u32,
}

/// Evaluate a single stratified candidate.
///
/// Every hash rotation here is load-bearing and must match the shader exactly: the
/// rotations are what decorrelate position from yaw from tint, and using the same
/// rotation twice would visibly couple two properties (all tall blades facing the same
/// way, for instance).
pub fn reference_candidate(
    uniforms: &PlaceUniforms,
    types: &[GpuFoliageType],
    tile_coord: [i32; 2],
    generation: u32,
    index: u32,
) -> ReferenceCandidate {
    let grid = uniforms.candidate_grid.max(1);
    let inv_grid = 1.0 / grid as f32;
    let seed = blade_seed(tile_coord, index, generation);

    let cell_x = index % grid;
    let cell_z = index / grid;
    let u = (cell_x as f32 + hash_to_unit(seed)) * inv_grid;
    let v = (cell_z as f32 + hash_to_unit(seed.rotate_left(11))) * inv_grid;

    // The temporary flat-plane fallback, matching `sample_terrain` when the capture is
    // absent: everything is at y = 0 with a straight-up normal, so every slope band
    // accepts.
    let height = 0.0f32;
    let slope_cos = 1.0f32;

    let type_limit = (uniforms.type_count.min(types.len() as u32)).max(1);
    let type_id = ((hash_to_unit(seed.rotate_left(5)) * type_limit as f32) as u32)
        .min(type_limit - 1);
    let foliage = types
        .get(type_id as usize)
        .copied()
        .unwrap_or_else(GpuFoliageType::default);

    let mut weight = 0.0f32;
    if slope_cos >= foliage.slope_range[0]
        && slope_cos <= foliage.slope_range[1]
        && height >= foliage.altitude_range[0]
        && height <= foliage.altitude_range[1]
    {
        weight = (foliage.density * uniforms.density_multiplier
            / uniforms.max_density.max(1.0e-6))
        .clamp(0.0, 1.0);
    }

    ReferenceCandidate {
        index,
        seed,
        tile_uv: [u, v],
        type_id: type_id as u8,
        weight,
        accepted: hash_to_unit(seed.rotate_left(17)) < weight,
    }
}

/// Place one tile, returning the blade list the placement shader must reproduce
/// byte-for-byte.
///
/// `center_y` is zero under the flat-plane fallback, so blade height offsets are zero.
/// Once the terrain capture exists the shader derives it from a 4×4 probe of the tile and
/// this function will have to do the same — the two must not diverge, because the offset
/// is stored as f16 relative to that centre and a different centre re-quantises every
/// blade in the tile.
pub fn place_tile_reference(
    uniforms: &PlaceUniforms,
    types: &[GpuFoliageType],
    tile_coord: [i32; 2],
    generation: u32,
) -> ReferencePlacement {
    let grid = uniforms.candidate_grid.max(1);
    let candidates = grid.saturating_mul(grid);
    let center_y = 0.0f32;

    let mut placement = ReferencePlacement {
        blades: Vec::new(),
        candidates,
        dropped: 0,
    };

    for index in 0..candidates {
        let candidate = reference_candidate(uniforms, types, tile_coord, generation, index);
        if !candidate.accepted {
            continue;
        }
        if placement.blades.len() as u32 >= uniforms.slab_capacity {
            placement.dropped += 1;
            continue;
        }

        let seed = candidate.seed;
        placement.blades.push(pack_blade(BladeParams {
            tile_uv: candidate.tile_uv,
            height_offset: 0.0 - center_y,
            yaw: hash_to_unit(seed.rotate_left(23)) * std::f32::consts::TAU,
            height_scale: hash_to_unit(seed.rotate_left(29)),
            width_scale: hash_to_unit(seed.rotate_left(3)),
            type_id: candidate.type_id,
            variant: ((seed >> 30) & 3) as u8,
            tint: [
                hash_to_unit(seed.rotate_left(7)),
                hash_to_unit(seed.rotate_left(13)),
            ],
            seed: seed as u16,
        }));
    }

    placement
}

#[cfg(test)]
mod tests {
    use super::*;
    use helio_foliage_core::{unpack_blade, FOLIAGE_TILE_SIZE_METERS};

    fn uniforms(grid: u32, slab: u32, types: &[GpuFoliageType]) -> PlaceUniforms {
        let max_density = types
            .iter()
            .map(|t| t.density)
            .fold(0.0f32, f32::max);
        PlaceUniforms {
            tile_size: FOLIAGE_TILE_SIZE_METERS,
            candidate_grid: grid,
            // The CPU reference walks candidates linearly, so a block edge of 1 keeps its
            // cell mapping identical to the shader's. Block-linear vs row-major only
            // changes which cells a *cluster* spans, not which cells exist.
            cluster_edge: 1,
            slab_capacity: slab,
            queued_tile_count: 1,
            density_multiplier: 1.0,
            max_density,
            type_count: types.len() as u32,
            max_foliage_height: 0.45,
            terrain_valid: 0,
            terrain_origin_x: 0.0,
            terrain_origin_z: 0.0,
            terrain_extent: 256.0,
            _pad: [0; 3],
        }
    }

    #[test]
    fn placing_the_same_tile_twice_is_byte_identical() {
        // The contract. Two independent "dispatches" over the same tile and generation
        // must produce identical arena bytes — this is what lets a tile be evicted and
        // re-placed without its grass moving.
        let types = [GpuFoliageType::default()];
        let uni = uniforms(24, 1024, &types);
        let first = place_tile_reference(&uni, &types, [12, -7], 3);
        let second = place_tile_reference(&uni, &types, [12, -7], 3);
        assert!(!first.blades.is_empty(), "the reference placed nothing to compare");
        assert_eq!(
            bytemuck::cast_slice::<_, u8>(&first.blades),
            bytemuck::cast_slice::<_, u8>(&second.blades),
        );
    }

    #[test]
    fn nothing_frame_dependent_can_reach_the_output() {
        // Placement is a pure function of (tile_coord, generation, candidate index).
        // There is no frame or time input to pass, and this test exists to make adding
        // one an obvious API break rather than a quiet regression: if a future signature
        // grows a frame argument, this call stops compiling.
        let types = [GpuFoliageType::default()];
        let uni = uniforms(16, 1024, &types);
        let baseline = place_tile_reference(&uni, &types, [0, 0], 0);
        for _ in 0..8 {
            assert_eq!(place_tile_reference(&uni, &types, [0, 0], 0), baseline);
        }
    }

    #[test]
    fn a_generation_bump_reshuffles_rather_than_perturbs() {
        // Editing density must visibly re-roll the tile. If the generation only
        // perturbed the tail, an edit would look like it did nothing wherever the terrain
        // stayed dense.
        let types = [GpuFoliageType::default()];
        let uni = uniforms(24, 1024, &types);
        let first = place_tile_reference(&uni, &types, [4, 4], 0);
        let bumped = place_tile_reference(&uni, &types, [4, 4], 1);
        let shared = first
            .blades
            .iter()
            .zip(bumped.blades.iter())
            .filter(|(a, b)| a.packed_pos == b.packed_pos)
            .count();
        assert!(
            shared * 20 < first.blades.len().max(1),
            "{shared} of {} blades kept their position across a generation bump",
            first.blades.len()
        );
    }

    #[test]
    fn neighbouring_tiles_are_independent_draws() {
        let types = [GpuFoliageType::default()];
        let uni = uniforms(24, 1024, &types);
        let here = place_tile_reference(&uni, &types, [0, 0], 0);
        let east = place_tile_reference(&uni, &types, [1, 0], 0);
        let west = place_tile_reference(&uni, &types, [-1, 0], 0);
        // The i32-as-u32 reinterpretation in `blade_seed` is what keeps the negative half
        // of the world from mirroring the positive half.
        assert_ne!(here.blades, east.blades);
        assert_ne!(here.blades, west.blades);
        assert_ne!(east.blades, west.blades);
    }

    #[test]
    fn every_blade_lands_inside_its_own_tile() {
        let types = [GpuFoliageType::default()];
        let uni = uniforms(32, 4096, &types);
        let placement = place_tile_reference(&uni, &types, [-3, 5], 2);
        for blade in &placement.blades {
            let params = unpack_blade(blade);
            assert!((0.0..=1.0).contains(&params.tile_uv[0]));
            assert!((0.0..=1.0).contains(&params.tile_uv[1]));
            assert_eq!(params.type_id, 0);
        }
    }

    #[test]
    fn stratification_spreads_blades_over_the_whole_tile() {
        // A rejection sampler without the stratified grid clumps, and clumps read as bald
        // patches next to fat tufts. Bucket the survivors into a 4x4 grid and require
        // every bucket to be populated.
        let types = [GpuFoliageType::default()];
        let uni = uniforms(32, 4096, &types);
        let placement = place_tile_reference(&uni, &types, [7, -2], 0);
        let mut buckets = [0u32; 16];
        for blade in &placement.blades {
            let params = unpack_blade(blade);
            let bx = ((params.tile_uv[0] * 4.0) as usize).min(3);
            let bz = ((params.tile_uv[1] * 4.0) as usize).min(3);
            buckets[bz * 4 + bx] += 1;
        }
        for (index, count) in buckets.iter().enumerate() {
            assert!(*count > 0, "quadrant {index} of the tile got no blades at all");
        }
    }

    #[test]
    fn the_rejection_sampler_honours_relative_density() {
        // Two types, one ten times denser. One shared candidate grid must still produce
        // roughly a 10:1 split, or the "one grid serves every type" trick is wrong.
        let mut sparse = GpuFoliageType::default();
        sparse.density = 4.0;
        let mut dense = GpuFoliageType::default();
        dense.density = 40.0;
        let types = [sparse, dense];
        let uni = uniforms(64, 8192, &types);

        let mut counts = [0u32; 2];
        for tile in 0..8i32 {
            let placement = place_tile_reference(&uni, &types, [tile, 0], 0);
            for blade in &placement.blades {
                counts[blade.type_id() as usize] += 1;
            }
        }
        assert!(counts[0] > 0 && counts[1] > 0);
        let ratio = counts[1] as f32 / counts[0] as f32;
        assert!(
            (7.0..14.0).contains(&ratio),
            "density ratio came out {ratio:.2}, expected about 10"
        );
    }

    #[test]
    fn a_slope_band_that_rejects_flat_ground_places_nothing() {
        // The fallback terrain is flat with cos(slope) = 1. A type that only accepts
        // steep ground must therefore place nothing — proving the band test is actually
        // consulted rather than being decoration.
        let mut cliff_only = GpuFoliageType::default();
        cliff_only.slope_range = [-1.0, 0.5];
        let types = [cliff_only];
        let uni = uniforms(32, 4096, &types);
        let placement = place_tile_reference(&uni, &types, [0, 0], 0);
        assert!(placement.blades.is_empty());
        assert_eq!(placement.candidates, 32 * 32);
    }

    #[test]
    fn a_full_slab_drops_the_tail_and_counts_it() {
        // The hard-ceiling contract: over-budget candidates are dropped, and the drop is
        // reported rather than silently swallowed.
        let types = [GpuFoliageType::default()];
        let uni = uniforms(32, 16, &types);
        let placement = place_tile_reference(&uni, &types, [0, 0], 0);
        assert_eq!(placement.blades.len(), 16);
        assert!(placement.dropped > 0, "over-budget blades must be counted");
    }
}
