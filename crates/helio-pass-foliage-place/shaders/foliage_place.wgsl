// Foliage placement — stage 1 of `FoliagePlacePass`.
//
// One workgroup per queued tile. Each lane evaluates one stratified candidate of a
// jittered grid against the foliage types' density/slope/altitude bands, and the
// survivors are compacted into that tile's fixed arena slab.
//
// ── Determinism is the contract, not a nice-to-have ─────────────────────────────
//
// The same (tile_coord, generation, lane) must produce a byte-identical blade list on
// every GPU, in every build, forever. Residency caching, the stable per-blade seed that
// wind phase and the dithered LOD cross-fade key off, and the CPU reference this shader
// is tested against all rest on it. Two consequences that look like over-engineering and
// are not:
//
//  1. **Nothing frame-dependent may enter the hash.** No frame index, no time, no
//     `atomicAdd` result. A seed that changes between frames turns the stochastic
//     cross-fade into full-screen static that TAA cannot resolve.
//
//  2. **Compaction is a workgroup prefix sum, not `atomicAdd`.** The plan's §6.1 says
//     "writing survivors with atomicAdd", and that is correct for *which* blades survive
//     but not for *where they land*: atomic ordering is unspecified, so the arena
//     contents would differ run to run and the byte-identical requirement would be
//     unenforceable. A 64-lane Hillis-Steele scan costs six barriers per chunk and makes
//     the slab a pure function of the candidate index. That is the whole reason a CPU
//     reference implementation can predict this shader's output.
//
// The packing helpers below are hand-rolled rather than using `pack2x16unorm` /
// `pack4x8unorm` on purpose: the builtins round to nearest even, while
// `helio_foliage_core::packing` rounds half away from zero (`x * 65535.0 + 0.5` then
// truncate). Those differ on exact ties, and a one-ULP disagreement is enough to fail a
// byte-for-byte determinism test against the CPU reference. `pack2x16float` *is* used for
// the f16 height offset because both sides round to nearest even there.

const WG_SIZE: u32 = 64u;
const TAU: f32 = 6.283185307179586;

// Mirrors helio_foliage_core::TileState.
const TILE_STATE_PLACING: u32 = 1u;
const TILE_STATE_RESIDENT: u32 = 2u;

// Mirrors the COUNTER_* constants in src/lib.rs. Slots 0..4 are the per-LOD visible
// counts owned by the cull shader; placement only touches 5 and 6.
const COUNTER_PLACEMENT_OVERFLOW: u32 = 5u;
const COUNTER_PLACED_BLADES: u32 = 6u;

/// Mirrors `PlaceUniforms` (Rust, 64 bytes).
struct PlaceUniforms {
    tile_size:          f32,
    candidate_grid:     u32,
    slab_capacity:      u32,
    queued_tile_count:  u32,

    density_multiplier: f32,
    max_density:        f32,
    type_count:         u32,
    max_foliage_height: f32,

    // 0 until `FoliageTerrainPass` exists *and* its ring transform has been published.
    // See `sample_terrain` for what happens in the meantime.
    terrain_valid:      u32,
    terrain_origin_x:   f32,
    terrain_origin_z:   f32,
    terrain_extent:     f32,

    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

/// Mirrors `helio_foliage_core::GpuFoliageType` (Rust, 96 bytes).
///
/// **Every field is a scalar.** Not one of them may become a vector type: with this
/// field order none of them lands on the alignment WGSL requires for `vec2`/`vec3`/
/// `vec4`, so a single `vec3<f32>` for the wind response would shift every field after
/// it by 12 bytes and nothing would crash — trees would just render with a random
/// material. See the struct doc in `helio-foliage-core/src/gpu_types.rs`.
struct FoliageType {
    density:               f32,
    height_min:            f32,
    height_max:            f32,
    width_min:             f32,
    width_max:             f32,
    slope_min:             f32,
    slope_max:             f32,
    altitude_min:          f32,
    altitude_max:          f32,
    lod0:                  f32,
    lod1:                  f32,
    lod2:                  f32,
    lod3:                  f32,
    wind_trunk:            f32,
    wind_branch:           f32,
    wind_leaf:             f32,
    interaction_stiffness: f32,
    material_id:           u32,
    density_layer:         u32,
    kind_and_flags:        u32,
    mesh_or_impostor_id:   u32,
    _pad0:                 u32,
    _pad1:                 u32,
    _pad2:                 u32,
}

/// Mirrors `helio_foliage_core::GpuFoliageTile` (Rust, 32 bytes).
struct FoliageTile {
    tile_coord_x:    i32,
    tile_coord_z:    i32,
    blade_offset:    u32,
    blade_count:     u32,
    bounds_center_y: f32,
    bounds_half_y:   f32,
    state:           u32,
    generation:      u32,
}

/// Mirrors `helio_foliage_core::GpuBladeInstance` (Rust, 16 bytes).
struct BladeInstance {
    packed_pos:        u32,
    packed_height_yaw: u32,
    packed_scale_type: u32,
    packed_tint_seed:  u32,
}

@group(0) @binding(0) var<uniform> place: PlaceUniforms;
@group(0) @binding(1) var<storage, read> types: array<FoliageType>;
@group(0) @binding(2) var<storage, read_write> tiles: array<FoliageTile>;
@group(0) @binding(3) var<storage, read_write> blades: array<BladeInstance>;
@group(0) @binding(4) var<storage, read> place_queue: array<u32>;
@group(0) @binding(5) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(6) var terrain_height_slope: texture_2d<f32>;
@group(0) @binding(7) var terrain_samp: sampler;

var<workgroup> wg_scan: array<u32, WG_SIZE>;
var<workgroup> wg_base: u32;
var<workgroup> wg_center_y: f32;

// ── Hash, transcribed from helio_foliage_core::placement ────────────────────────

fn rotl(value: u32, amount: u32) -> u32 {
    return (value << amount) | (value >> (32u - amount));
}

/// Byte-for-byte transcription of `helio_foliage_core::blade_seed`.
///
/// `bitcast<u32>` on the coordinates matches Rust's `i32 as u32`, i.e. two's-complement
/// reinterpretation. Do not "fix" this with `abs()`: that aliases tiles across the world
/// origin, and grass west of x = 0 becomes a mirror image of grass east of it.
fn blade_seed(coord_x: i32, coord_z: i32, lane: u32, generation: u32) -> u32 {
    var h = bitcast<u32>(coord_x) * 374761393u
        + bitcast<u32>(coord_z) * 668265263u
        + lane * 2654435761u
        + generation * 2246822519u;
    h = (h ^ (h >> 15u)) * 2246822519u;
    h = (h ^ (h >> 13u)) * 3266489917u;
    h = h ^ (h >> 16u);
    return h;
}

/// Transcription of `helio_foliage_core::hash_to_unit`.
///
/// Top 24 bits over 2^24, so the result is exactly representable and strictly below 1.0.
/// Dividing the full u32 by 0xffffffff can return exactly 1.0, which puts
/// `floor(u * grid)` one cell past the end of the stratified grid.
fn hash_to_unit(hash: u32) -> f32 {
    return f32(hash >> 8u) * (1.0 / 16777216.0);
}

// ── Packing, matching helio_foliage_core::packing exactly ───────────────────────

fn pack_unorm16_u(value: f32) -> u32 {
    return u32(clamp(value, 0.0, 1.0) * 65535.0 + 0.5);
}

fn pack_unorm8_u(value: f32) -> u32 {
    return u32(clamp(value, 0.0, 1.0) * 255.0 + 0.5);
}

/// 16 bits of *turn*, not unorm: 0 and 2π are the same orientation, so dividing by
/// 65536 rather than 65535 keeps the wrap point uniform instead of leaving a seam of
/// over-represented yaw near zero.
fn pack_yaw_u(radians: f32) -> u32 {
    let turns = radians * (1.0 / TAU);
    let wrapped = turns - floor(turns);
    return u32(wrapped * 65536.0 + 0.5) & 0xffffu;
}

fn pack_pos(u: f32, v: f32) -> u32 {
    return pack_unorm16_u(u) | (pack_unorm16_u(v) << 16u);
}

fn pack_height_yaw(height_offset: f32, yaw: f32) -> u32 {
    return (pack2x16float(vec2<f32>(height_offset, 0.0)) & 0xffffu) | (pack_yaw_u(yaw) << 16u);
}

fn pack_scale_type(height: f32, width: f32, type_id: u32, variant: u32) -> u32 {
    return pack_unorm8_u(height)
        | (pack_unorm8_u(width) << 8u)
        | ((type_id & 0xffu) << 16u)
        | ((variant & 0xffu) << 24u);
}

fn pack_tint_seed(tint_x: f32, tint_y: f32, seed: u32) -> u32 {
    return pack_unorm8_u(tint_x) | (pack_unorm8_u(tint_y) << 8u) | ((seed & 0xffffu) << 16u);
}

// ── Terrain ─────────────────────────────────────────────────────────────────────

/// Returns `(height_metres, cos_slope)` at a world XZ.
///
/// **TEMPORARY FALLBACK.** `FoliageTerrainPass` (plan §5) does not exist yet, so
/// `terrain_valid` is 0 in every shipped configuration today and this returns a flat
/// plane at y = 0 with a straight-up normal. That is deliberately the most permissive
/// answer possible — every slope band accepts `cos = 1` — so the pass is testable and the
/// demo shows grass instead of nothing. It is *not* a placement policy: the moment the
/// capture lands, `terrain_valid` flips and this branch becomes dead weight that should
/// be deleted, not kept as a "no terrain" mode. Leaving it live would let a scene whose
/// capture failed to render silently carpet the void at y = 0.
fn sample_terrain(world_xz: vec2<f32>) -> vec2<f32> {
    if place.terrain_valid == 0u {
        return vec2<f32>(0.0, 1.0);
    }
    let uv = (world_xz - vec2<f32>(place.terrain_origin_x, place.terrain_origin_z))
        / max(place.terrain_extent, 1.0e-3);
    let texel = textureSampleLevel(
        terrain_height_slope,
        terrain_samp,
        clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)),
        0.0,
    );
    return vec2<f32>(texel.r, texel.g);
}

// ── Placement ───────────────────────────────────────────────────────────────────

@compute @workgroup_size(64)
fn cs_place(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    // Resolve the tile *without* letting the result gate any barrier. `place_queue` is a
    // storage load, so WGSL's uniformity analysis cannot prove the value is the same in
    // every lane even though it demonstrably is. Every lane therefore runs the full
    // candidate loop and only the *writes* are guarded — an invalid workgroup burns a few
    // microseconds and touches nothing, which is cheaper than fighting the analysis.
    var slot = 0u;
    var valid = false;
    if workgroup_id.x < place.queued_tile_count && workgroup_id.x < arrayLength(&place_queue) {
        slot = place_queue[workgroup_id.x];
        valid = slot < arrayLength(&tiles);
    }

    var tile: FoliageTile;
    if valid {
        tile = tiles[slot];
    }

    let origin_x = f32(tile.tile_coord_x) * place.tile_size;
    let origin_z = f32(tile.tile_coord_z) * place.tile_size;

    // Lane 0 establishes the tile's vertical extent before anyone places into it, because
    // blades store a height *offset* from `bounds_center_y` and every other lane needs
    // that value to encode its own f16 offset. A 4x4 probe over an 8 m tile is one sample
    // every 2.6 m — conservative enough given the cull dilates the result by the tallest
    // foliage height anyway, and 16 taps on one lane of one workgroup per queued tile is
    // free next to the candidate loop below.
    if lane == 0u {
        var y_min = 1.0e30;
        var y_max = -1.0e30;
        for (var probe = 0u; probe < 16u; probe = probe + 1u) {
            let gx = f32(probe % 4u) / 3.0;
            let gz = f32(probe / 4u) / 3.0;
            let height = sample_terrain(vec2<f32>(
                origin_x + gx * place.tile_size,
                origin_z + gz * place.tile_size,
            )).x;
            y_min = min(y_min, height);
            y_max = max(y_max, height);
        }
        wg_center_y = (y_min + y_max) * 0.5;
        wg_base = 0u;
        if valid {
            tiles[slot].bounds_center_y = (y_min + y_max) * 0.5;
            tiles[slot].bounds_half_y = (y_max - y_min) * 0.5 + place.max_foliage_height;
            tiles[slot].blade_count = 0u;
            tiles[slot].state = TILE_STATE_PLACING;
        }
    }
    workgroupBarrier();
    let center_y = wg_center_y;

    let grid = max(place.candidate_grid, 1u);
    let cells = grid * grid;
    let inv_grid = 1.0 / f32(grid);
    let type_limit = max(min(place.type_count, arrayLength(&types)), 1u);

    // `cells` comes from the uniform buffer, so this loop and every barrier inside it are
    // in uniform control flow for the whole workgroup.
    var chunk = 0u;
    loop {
        if chunk >= cells {
            break;
        }

        let candidate = chunk + lane;
        var accepted = 0u;
        var record = BladeInstance(0u, 0u, 0u, 0u);

        if valid && candidate < cells {
            let seed = blade_seed(tile.tile_coord_x, tile.tile_coord_z, candidate, tile.generation);

            // Stratified: one candidate per grid cell, jittered inside it. Pure rejection
            // sampling over the tile would clump, and clumps read as bald patches next to
            // fat tufts at exactly the density where grass is supposed to look uniform.
            let cell_x = candidate % grid;
            let cell_z = candidate / grid;
            let u = (f32(cell_x) + hash_to_unit(seed)) * inv_grid;
            let v = (f32(cell_z) + hash_to_unit(rotl(seed, 11u))) * inv_grid;

            let world_xz = vec2<f32>(
                origin_x + u * place.tile_size,
                origin_z + v * place.tile_size,
            );
            let terrain = sample_terrain(world_xz);
            let height = terrain.x;
            let slope_cos = terrain.y;

            // The candidate grid is sized on the CPU from the *densest* type, and each
            // candidate then accepts with probability `this type's density / max density`.
            // That is a rejection sampler: one grid serves every type at its own authored
            // density, without a dispatch per type and without the grid resolution
            // depending on which types happen to be present.
            let type_id = min(u32(hash_to_unit(rotl(seed, 5u)) * f32(type_limit)), type_limit - 1u);
            let foliage = types[type_id];

            var weight = 0.0;
            if slope_cos >= foliage.slope_min
                && slope_cos <= foliage.slope_max
                && height >= foliage.altitude_min
                && height <= foliage.altitude_max
            {
                weight = clamp(
                    foliage.density * place.density_multiplier / max(place.max_density, 1.0e-6),
                    0.0,
                    1.0,
                );
            }

            if hash_to_unit(rotl(seed, 17u)) < weight {
                accepted = 1u;
                record.packed_pos = pack_pos(u, v);
                record.packed_height_yaw = pack_height_yaw(
                    height - center_y,
                    hash_to_unit(rotl(seed, 23u)) * TAU,
                );
                record.packed_scale_type = pack_scale_type(
                    hash_to_unit(rotl(seed, 29u)),
                    hash_to_unit(rotl(seed, 3u)),
                    type_id,
                    (seed >> 30u) & 3u,
                );
                record.packed_tint_seed = pack_tint_seed(
                    hash_to_unit(rotl(seed, 7u)),
                    hash_to_unit(rotl(seed, 13u)),
                    seed & 0xffffu,
                );
            }
        }

        // Hillis-Steele inclusive scan over the 64 acceptance flags. `offset` is a
        // function-local counter with no data dependence, so both barriers stay in
        // uniform control flow.
        wg_scan[lane] = accepted;
        workgroupBarrier();
        var offset = 1u;
        loop {
            if offset >= WG_SIZE {
                break;
            }
            var addend = 0u;
            if lane >= offset {
                addend = wg_scan[lane - offset];
            }
            workgroupBarrier();
            wg_scan[lane] = wg_scan[lane] + addend;
            workgroupBarrier();
            offset = offset << 1u;
        }

        let inclusive = wg_scan[lane];
        let chunk_total = wg_scan[WG_SIZE - 1u];
        let base = wg_base;

        if accepted == 1u {
            let index = base + inclusive - 1u;
            let arena_index = tile.blade_offset + index;
            if index < place.slab_capacity && arena_index < arrayLength(&blades) {
                blades[arena_index] = record;
            } else {
                // Hard ceiling, never silent truncation: the slab is full and this blade
                // is dropped, but the drop is counted. The CPU normally prevents this
                // outright by clamping the candidate grid so `cells <= slab_capacity`
                // (uniform thinning rather than a bald corner), so a non-zero value here
                // means the two sides have drifted and is worth investigating, not
                // tolerating.
                atomicAdd(&counters[COUNTER_PLACEMENT_OVERFLOW], 1u);
            }
        }

        // Every lane must have consumed `wg_base` before lane 0 advances it.
        workgroupBarrier();
        if lane == 0u {
            wg_base = base + chunk_total;
        }
        workgroupBarrier();

        chunk = chunk + WG_SIZE;
    }

    workgroupBarrier();
    if lane == 0u && valid {
        let count = min(wg_base, place.slab_capacity);
        tiles[slot].blade_count = count;
        // Placing -> Resident is what makes the tile drawable. Until this line runs the
        // slab holds whatever the previous tenant left, which is why `TileState::Placing`
        // is excluded from `is_drawable`.
        tiles[slot].state = TILE_STATE_RESIDENT;
        atomicAdd(&counters[COUNTER_PLACED_BLADES], count);
    }
}
