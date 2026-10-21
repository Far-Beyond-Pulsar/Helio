// Helio foliage common — the records placement, the cull and the draw share, and the
// bands' functions.
//
// Prepended to any shader whose source contains `//!use helio_foliage_common` (see
// helio_core::shader). One copy instead of one per pass: three hand-kept mirrors of a
// 128-byte table drift.
//
// A shader including this declares `foliage_bands: FoliageBands` (a uniform, the
// pass's band table): the bands' functions below read it.

/// Mirrors `GpuFoliageType` (Rust, 128 bytes). Every field a scalar — see the struct's
/// doc in `contract/gpu_types.rs` (a `vec3` would shift every field after it).
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
    base_color:            u32,
    roughness_metallic:    u32,
    ground_mask:           u32,
    blades_per_tuft:       u32,
    lean_min:              f32,
    lean_max:              f32,
    curve:                 f32,
    fall_chance:           f32,
    tip_color:             u32,
    patch_scale:           f32,
    patch_strength:        f32,
}

/// Mirrors `GpuFoliageTile` (Rust, 32 bytes).
struct FoliageTile {
    tile_coord_x:  i32,
    tile_coord_z:  i32,
    blade_offset:  u32,
    blade_count:   u32,
    bounds_center: f32,
    bounds_half:   f32,
    state:         u32,
    generation:    u32,
}

/// Mirrors `GpuTileFrame` (Rust, 64 bytes): where a tile's blades stand.
struct TileFrame {
    origin: vec3<f32>,
    axis_u: vec3<f32>,
    axis_v: vec3<f32>,
    up:     vec3<f32>,
}

/// Mirrors `GpuBladeInstance` (Rust, 16 bytes).
struct BladeInstance {
    packed_pos:         u32,
    packed_height_tuft: u32,
    packed_scale_type:  u32,
    packed_colour_seed: u32,
}

/// Mirrors `GpuFoliageBand` (Rust, 64 bytes).
struct FoliageBand {
    slot_base:      u32,
    slot_count:     u32,
    arena_base:     u32,
    slab:           u32,
    tile_size:      f32,
    tile_span:      u32,
    inner:          f32,
    outer:          f32,
    fade_in:        f32,
    fade_out:       f32,
    share:          f32,
    candidate_grid: u32,
    probe_blocks:   u32,
    lod:            u32,
    grid_density:   f32,
    _pad:           u32,
}

/// Mirrors `GpuFoliageBands` (Rust, 416 bytes).
struct FoliageBands {
    count:         u32,
    keep_distance: f32,
    keep_power:    f32,
    growth_power:  f32,
    growth_max:    f32,
    range:         f32,
    _pad0:         u32,
    _pad1:         u32,
    bands:         array<FoliageBand, 6>,
}

// `FoliageKind` and the `FOLIAGE_FLAG_*` bits.
const FOLIAGE_KIND_BLADE: u32 = 0u;
const FOLIAGE_KIND_CARD: u32 = 1u;
const FOLIAGE_KIND_MESH: u32 = 2u;
const FOLIAGE_KIND_BLOCK: u32 = 3u;
const FOLIAGE_TYPE_FLAG_RECEIVES_INTERACTION: u32 = 1024u;
const FOLIAGE_TYPE_FLAG_GROUND_COLOUR: u32 = 2048u;

// `BAND_LOD_*`: how a band's records are stored and drawn.
const FOLIAGE_BAND_LOD_BLADES: u32 = 0u;
const FOLIAGE_BAND_LOD_TUFT_BLADES: u32 = 2u;
const FOLIAGE_BAND_LOD_TUFT: u32 = 3u;

/// The band tile-table slot `slot` belongs to.
fn foliage_band_of_slot(slot: u32) -> u32 {
    let count = min(foliage_bands.count, 6u);
    for (var b = 0u; b < count; b++) {
        let band = foliage_bands.bands[b];
        if slot < band.slot_base + band.slot_count {
            return b;
        }
    }
    return max(count, 1u) - 1u;
}

/// Share of the tufts drawn at `d` metres (`bands::keep`).
fn foliage_keep(d: f32) -> f32 {
    return min(pow(foliage_bands.keep_distance / max(d, 1.0e-3), foliage_bands.keep_power), 1.0);
}

/// How much a surviving tuft grows at `d` metres (`bands::growth`).
fn foliage_growth(d: f32) -> f32 {
    return clamp(pow(foliage_keep(d), -foliage_bands.growth_power), 1.0, max(foliage_bands.growth_max, 1.0));
}

/// Band `band`'s weight at `d` metres: 1 inside it, fading across its boundaries
/// (`bands::band_weight`).
fn foliage_band_weight(band: FoliageBand, d: f32) -> f32 {
    var w = 1.0;
    if band.inner > 0.0 {
        w = smoothstep(band.inner - band.fade_in, band.inner, d);
    }
    return w * (1.0 - smoothstep(band.outer - band.fade_out, band.outer, d));
}

/// The scale a tuft of `rank` draws at in `band` at `d` metres (0: not drawn): the band's
/// weight, the tuft shrinking out as the kept share falls to its rank, and its growth
/// (`bands::tuft_scale`).
fn foliage_tuft_scale(band: FoliageBand, d: f32, rank: f32) -> f32 {
    let w = foliage_band_weight(band, d);
    if w <= 0.0 {
        return 0.0;
    }
    let rel = foliage_keep(d) / max(band.share, 1.0e-6);
    let fade = clamp((rel - rank) / max(0.2 * rel, 1.0e-3), 0.0, 1.0);
    return w * fade * foliage_growth(d);
}

/// A type's range end (`lod3`): a fifth of it fading out.
fn foliage_type_reach(ty: FoliageType, d: f32) -> f32 {
    let end = select(1.0e30, ty.lod3, ty.lod3 > 0.0);
    return 1.0 - smoothstep(0.8 * end, end, d);
}

/// The record's tile-local `(u, v)` (unorm16 each).
fn foliage_blade_uv(blade: BladeInstance) -> vec2<f32> {
    return vec2<f32>(f32(blade.packed_pos & 0xffffu), f32(blade.packed_pos >> 16u)) * (1.0 / 65535.0);
}

/// Its height offset from the tile's `bounds_center` (f16).
fn foliage_blade_height(blade: BladeInstance) -> f32 {
    return unpack2x16float(blade.packed_height_tuft & 0xffffu).x;
}

/// The blade's index in its tuft, the tuft's blade count and its rank.
fn foliage_blade_index(blade: BladeInstance) -> u32 {
    return (blade.packed_height_tuft >> 16u) & 15u;
}

fn foliage_tuft_blades(blade: BladeInstance) -> u32 {
    return max((blade.packed_height_tuft >> 20u) & 15u, 1u);
}

fn foliage_tuft_rank(blade: BladeInstance) -> f32 {
    return f32(blade.packed_height_tuft >> 24u) * (1.0 / 255.0);
}

/// The ground colour under the tuft (linear, RGB565), `w` 0 where the ground gave none.
fn foliage_ground_colour(blade: BladeInstance) -> vec4<f32> {
    let c = blade.packed_colour_seed & 0xffffu;
    if c == 0u {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(f32(c & 31u) / 31.0, f32((c >> 5u) & 63u) / 63.0, f32(c >> 11u) / 31.0, 1.0);
}

/// The tuft's world root: its tile's frame at the record's `(u, v)` and height.
fn foliage_blade_root(tile: FoliageTile, frame: TileFrame, blade: BladeInstance) -> vec3<f32> {
    let uv = foliage_blade_uv(blade);
    return frame.origin + frame.axis_u * uv.x + frame.axis_v * uv.y
        + frame.up * (tile.bounds_center + foliage_blade_height(blade));
}

/// A 32-bit integer hash (PCG output permutation).
fn foliage_hash(x: u32) -> u32 {
    let state = x * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

/// `hash` to `[0, 1)`: its top 24 bits.
fn foliage_unit(hash: u32) -> f32 {
    return f32(hash >> 8u) * (1.0 / 16777216.0);
}
