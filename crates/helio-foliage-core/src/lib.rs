//! Foliage GPU types, packing helpers and CPU mirrors of the foliage shader math.
//!
//! This is the `*-core` crate for the foliage stack, following the same rule as
//! `helio-voxel-core` and `helio-planet-voxel-core`: shared POD types live in a crate
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
//!
//! # What does not live here
//!
//! Anything holding a GPU resource, and anything with per-frame state. Buffers, bind
//! groups and residency bookkeeping belong to `helio-pass-foliage-place`; the authoring
//! API belongs to `helio::Scene`.

pub mod gpu_types;
pub mod packing;
pub mod placement;
pub mod quality;

pub use gpu_types::*;
pub use packing::*;
pub use placement::*;
pub use quality::*;

use glam::{Vec2, Vec3};

/// Edge length of one foliage tile in metres.
///
/// 8 m is the unit of everything downstream: the residency ring, the placement dispatch
/// (one workgroup per tile), the terrain capture's redraw granularity and the tile-level
/// Hi-Z test. It is chosen so one tile is one workgroup's worth of stratified
/// candidates at the reference density — larger tiles under-occupy the GPU on sparse
/// terrain, smaller tiles multiply the per-tile header and cull cost without improving
/// culling, because a tile is already smaller than the Hi-Z footprint it is tested
/// against at any distance where culling matters.
///
/// Changing this invalidates every cached tile and shifts the tile grid, which re-rolls
/// every blade in the world via [`blade_seed`].
pub const FOLIAGE_TILE_SIZE_METERS: f32 = 8.0;

/// Default number of tile slots in the resident ring.
///
/// 4096 tiles at 8 m covers a 512 m square, comfortably past the Ultra ring radius, and
/// costs 128 KiB of tile table. The ring is a hard ceiling: once it is full, tiles
/// entering the ring evict LRU tiles, so an undersized ring does not fail — it thrashes,
/// re-placing the same tiles every frame and turning the amortised placement cost into a
/// per-frame one.
pub const DEFAULT_TILE_RING_CAPACITY: u32 = 4096;

/// Default cap on tiles placed per frame.
///
/// This is what turns a teleport into a few frames of progressive fill-in instead of a
/// hitch. Raising it makes the ring converge faster and makes the worst-case frame
/// longer, in direct proportion; lowering it below the ring's perimeter refill rate
/// means a player running in a straight line outruns placement and sees the ring edge.
pub const DEFAULT_MAX_TILES_PER_FRAME: u32 = 24;

/// World-space XZ origin (minimum corner) of a tile.
#[inline]
pub fn tile_origin(tile_coord: [i32; 2]) -> Vec2 {
    Vec2::new(
        tile_coord[0] as f32 * FOLIAGE_TILE_SIZE_METERS,
        tile_coord[1] as f32 * FOLIAGE_TILE_SIZE_METERS,
    )
}

/// Reconstruct a blade's world position from its tile header and packed record.
///
/// This is the CPU mirror of the first three lines of every foliage vertex shader.
/// Blades store tile-local XZ and a height *offset* from the tile's `bounds_center_y`
/// precisely so this reconstruction is the only place world coordinates appear — see
/// [`GpuBladeInstance::packed_height_yaw`] for why storing absolute Y instead would
/// quantise grass into terraces far from the origin.
#[inline]
pub fn blade_world_position(tile: &GpuFoliageTile, blade: &GpuBladeInstance) -> Vec3 {
    let origin = tile_origin(tile.tile_coord);
    let local = unpack_blade_pos(blade.packed_pos);
    let (height_offset, _yaw) = unpack_blade_height_yaw(blade.packed_height_yaw);
    Vec3::new(
        origin.x + local[0] * FOLIAGE_TILE_SIZE_METERS,
        tile.bounds_center_y + height_offset,
        origin.y + local[1] * FOLIAGE_TILE_SIZE_METERS,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_grid_constants_are_consistent() {
        assert_eq!(FOLIAGE_TILE_SIZE_METERS, 8.0);
        // The ring must be able to hold the widest preset's footprint, or Ultra thrashes
        // instead of caching.
        let ultra_tiles_across =
            (2.0 * FoliageQuality::Ultra.ring_radius() / FOLIAGE_TILE_SIZE_METERS).ceil() as u32;
        assert!(
            ultra_tiles_across * ultra_tiles_across <= DEFAULT_TILE_RING_CAPACITY,
            "the {DEFAULT_TILE_RING_CAPACITY}-slot ring cannot hold Ultra's \
             {ultra_tiles_across}x{ultra_tiles_across} tile footprint"
        );
    }

    #[test]
    fn blade_world_position_is_tile_relative() {
        let mut tile = GpuFoliageTile::EMPTY;
        tile.tile_coord = [-3, 5];
        tile.bounds_center_y = 120.0;

        let blade = pack_blade(BladeParams {
            tile_uv: [0.0, 0.0],
            height_offset: 0.5,
            ..BladeParams::default()
        });
        let position = blade_world_position(&tile, &blade);
        assert!((position.x - (-24.0)).abs() < 1.0e-3);
        assert!((position.z - 40.0).abs() < 1.0e-3);
        assert!((position.y - 120.5).abs() < 1.0e-3);

        // The far corner of the tile lands exactly one tile away, with no gap or overlap
        // against the neighbouring tile's origin — a unorm endpoint that did not reach
        // 1.0 exactly would leave a visible seam of bare ground on every tile boundary.
        let far = pack_blade(BladeParams {
            tile_uv: [1.0, 1.0],
            ..BladeParams::default()
        });
        let far_position = blade_world_position(&tile, &far);
        assert!((far_position.x - (-24.0 + FOLIAGE_TILE_SIZE_METERS)).abs() < 1.0e-3);
        assert!((far_position.z - (40.0 + FOLIAGE_TILE_SIZE_METERS)).abs() < 1.0e-3);
    }

    #[test]
    fn tile_origin_is_signed_and_linear() {
        assert_eq!(tile_origin([0, 0]), Vec2::ZERO);
        assert_eq!(tile_origin([1, -1]), Vec2::new(8.0, -8.0));
        assert_eq!(tile_origin([-100, 100]), Vec2::new(-800.0, 800.0));
    }

    #[test]
    fn a_placed_blade_survives_the_full_pipeline() {
        // End-to-end: seed the blade, pack it, read it back, and pick its LOD. This is
        // the shape the placement shader's inner loop takes, and it is here so a change
        // that breaks the composition rather than any single function still fails.
        let tile_coord = [12, -7];
        let generation = 3;
        for lane in 0..64u32 {
            let seed = blade_seed(tile_coord, lane, generation);
            let params = BladeParams {
                tile_uv: [
                    hash_to_unit(seed),
                    hash_to_unit(seed.rotate_left(11)),
                ],
                height_offset: hash_to_unit(seed.rotate_left(19)) * 0.4,
                yaw: hash_to_unit(seed.rotate_left(23)) * std::f32::consts::TAU,
                height_scale: hash_to_unit(seed.rotate_left(5)),
                width_scale: 0.5,
                type_id: 1,
                variant: (seed & 0x3) as u8,
                tint: [hash_to_unit(seed.rotate_left(3)), 0.5],
                seed: seed as u16,
            };

            let mut tile = GpuFoliageTile::EMPTY;
            tile.tile_coord = tile_coord;
            tile.generation = generation;
            tile.state = TileState::Resident.as_u32();

            let instance = pack_blade(params);
            assert_eq!(instance.seed(), seed as u16);
            assert_eq!(instance.type_id(), 1);

            let position = blade_world_position(&tile, &instance);
            assert!(position.is_finite());
            let tile_min = tile_origin(tile_coord);
            assert!(position.x >= tile_min.x - 1.0e-3);
            assert!(position.x <= tile_min.x + FOLIAGE_TILE_SIZE_METERS + 1.0e-3);
            assert!(position.z >= tile_min.y - 1.0e-3);
            assert!(position.z <= tile_min.y + FOLIAGE_TILE_SIZE_METERS + 1.0e-3);

            let distance = position.length();
            let level = select_blade_lod(
                distance,
                GpuFoliageType::default().lod_distances(),
                FoliageQuality::Medium.lod_distance_scale(),
            );
            assert!(level <= FOLIAGE_LOD_NONE);
        }
    }

    #[test]
    fn re_placing_a_tile_reproduces_the_same_blades() {
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
                        yaw: hash_to_unit(seed.rotate_left(23)) * std::f32::consts::TAU,
                        height_scale: hash_to_unit(seed.rotate_left(5)),
                        width_scale: hash_to_unit(seed.rotate_left(7)),
                        type_id: 2,
                        variant: (seed >> 30) as u8,
                        tint: [hash_to_unit(seed.rotate_left(3)), 0.25],
                        seed: seed as u16,
                    })
                })
                .collect()
        }

        let first = place([12, -7], 3);
        let second = place([12, -7], 3);
        assert_eq!(
            bytemuck::cast_slice::<_, u8>(&first),
            bytemuck::cast_slice::<_, u8>(&second),
            "the same tile and generation must produce a byte-identical blade list"
        );

        // A generation bump must reshuffle, not merely perturb — otherwise a density
        // edit looks like it did nothing wherever the terrain stayed dense.
        let bumped = place([12, -7], 4);
        let unchanged = first
            .iter()
            .zip(bumped.iter())
            .filter(|(a, b)| a.packed_pos == b.packed_pos)
            .count();
        assert!(
            unchanged <= 2,
            "{unchanged} of 128 blades kept their position across a generation bump"
        );

        // And a neighbouring tile must be an independent draw, not an offset copy.
        let neighbour = place([13, -7], 3);
        let shared = first
            .iter()
            .zip(neighbour.iter())
            .filter(|(a, b)| a.packed_pos == b.packed_pos)
            .count();
        assert!(shared <= 2, "{shared} of 128 blades are shared with the neighbouring tile");
    }
}
