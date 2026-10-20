//! Foliage GPU types, packing helpers and CPU mirrors of the foliage shader math.
//!
//! This is the `*-core` crate for the foliage stack: shared POD types live in a crate
//! with no `wgpu` dependency, so the passes that produce foliage data and the passes
//! that consume it agree on layout without depending on each other, and so the layout
//! and the maths can be tested on a machine with no GPU at all.
//!
//! The no-`wgpu` rule is load-bearing, not stylistic. Everything here is on the critical
//! path of a CI test that must run in a headless container: the layout asserts, the
//! packing round-trips, and — most importantly — the placement determinism reference. If
//! this crate ever grows a graphics dependency, those tests become GPU tests, GPU tests
//! become optional, and the determinism contract stops being enforced.
//!
//! # What lives here
//!
//! - [`gpu_types`] — [`GpuBladeInstance`], [`GpuFoliageTile`] and [`GpuFoliageType`],
//!   with compile-time size asserts. These are mirrored field-for-field in WGSL.
//! - [`packing`] — the bit-level primitives the packed fields are built from, each with
//!   a documented WGSL equivalent.
//! - [`placement`] — CPU reference implementations of the placement and LOD maths. The
//!   WGSL is a transcription of these, and these are what the determinism and
//!   cull-equality tests compare against.
//! - [`quality`] — the [`FoliageQuality`] presets, the only platform difference in the
//!   foliage stack.
//! - [`bands`] — the bands of resident tiles that carry foliage out to its range.
//! - [`ground`] — what foliage grows on.
//!
//! # What does not live here
//!
//! Anything holding a GPU resource, and anything with per-frame state. Buffers, bind
//! groups and residency bookkeeping belong to `helio-pass-foliage-place`; the authoring
//! API belongs to `helio::Scene`.

pub mod bands;
pub mod ground;
pub mod gpu_types;
pub mod packing;
pub mod placement;
pub mod quality;

pub use bands::*;
pub use gpu_types::*;
pub use ground::*;
pub use packing::*;
pub use placement::*;
pub use quality::*;

/// Edge length of one base foliage tile in metres.
///
/// 8 m is the unit of everything downstream: the near band's tiles, the ground's chart
/// (`GroundGeometry::tile_frame` spans whole base tiles), the probe blocks and the
/// placement dispatch (one workgroup per tile). The far bands' tiles span `2^k` of them
/// ([`crate::plan_bands`]).
///
/// Changing this invalidates every cached tile and shifts the tile grid, which re-rolls
/// every tuft in the world via [`blade_seed`].
pub const FOLIAGE_TILE_SIZE_METERS: f32 = 8.0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn re_placing_a_tile_reproduces_the_same_records() {
        // The determinism contract, exercised through the packing layer rather than just
        // the hash: two independent "dispatches" over the same tile must produce
        // byte-identical arena contents.
        fn place(tile_coord: [i32; 2], generation: u32) -> Vec<GpuBladeInstance> {
            (0..128u32)
                .map(|lane| {
                    let seed = blade_seed(tile_coord, lane, generation);
                    pack_blade(BladeParams {
                        tile_uv: [hash_to_unit(seed), hash_to_unit(seed.rotate_left(11))],
                        height_offset: hash_to_unit(seed.rotate_left(19)) * 0.4,
                        rank: hash_to_unit(seed.rotate_left(23)),
                        height_scale: hash_to_unit(seed.rotate_left(5)),
                        type_id: 2,
                        variant: (seed >> 30) as u8,
                        seed: seed as u16,
                        ..BladeParams::default()
                    })
                })
                .collect()
        }

        let first = place([12, -7], 3);
        let second = place([12, -7], 3);
        assert_eq!(first, second, "the same tile and generation must produce a byte-identical record list");

        // A generation bump must reshuffle, not merely perturb.
        let bumped = place([12, -7], 4);
        let unchanged = first.iter().zip(bumped.iter()).filter(|(a, b)| a.packed_pos == b.packed_pos).count();
        assert!(unchanged <= 2, "{unchanged} of 128 tufts kept their position across a generation bump");

        // And a neighbouring tile must be an independent draw, not an offset copy.
        let neighbour = place([13, -7], 3);
        let shared = first.iter().zip(neighbour.iter()).filter(|(a, b)| a.packed_pos == b.packed_pos).count();
        assert!(shared <= 2, "{shared} of 128 tufts are shared with the neighbouring tile");
    }
}
