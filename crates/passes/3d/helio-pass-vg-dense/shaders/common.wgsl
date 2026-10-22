// Tier-3 shared declarations: params, state-buffer layout, tile math.
// Prepended after the Rust-generated constant prelude (see `src/config.rs`)
// and `helio_vg_format::WGSL`. No bindings here; every kernel declares its own.

struct DenseParams {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    // xyz camera world position, w unused.
    camera_pos: vec4<f32>,
    // x,y viewport pixels; z,w tiles_x, tiles_y.
    viewport: vec4<u32>,
    // x tile_count, y max_dense_slots, z visible_capacity, w unused.
    limits: vec4<u32>,
    // x ray tmin, y ray tmax, z,w unused.
    misc: vec4<f32>,
}

// `state` buffer layout (u32 atomics). The dense tile list starts at STATE_HEADER.
const STATE_DENSE_COUNT: u32 = 0u;
const STAT_LAYERS: u32 = 1u;
const STAT_CLAMPED_TILES: u32 = 2u;
const STAT_CLAMPED_TRIS: u32 = 3u;
const STAT_BIN_OVERFLOW: u32 = 4u;
const STAT_SLOT_OVERFLOW: u32 = 5u;
const STAT_EXTRA_LAYERS: u32 = 6u;
const STAT_TILES: u32 = 7u;
const STATE_HEADER: u32 = 16u;

const SLOT_INVALID: u32 = 0xffffffffu;
const SLOT_STRIDE: u32 = CLUSTER_CAP + 1u;
// Edge-function slack: turns would-be cracks on shared edges into (harmless)
// double coverage resolved by depth.
const EDGE_EPS: f32 = 1.0e-7;

// Rows of the 2D `bin` dispatch: thread = (y * BIN_ROW_GROUPS + x) * 64 + lane.
const BIN_ROW_GROUPS: u32 = 32768u;
