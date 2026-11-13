//! Tile aggregation compute shader.
//!
//! Aggregates per-pixel pass overdraw counters into per-tile metrics (16×16 tiles)
//! and reads the offered per-tile count (a lighting model's light loops).

struct AggregateParams {
    num_tiles_x: u32,
    num_tiles_y: u32,
    num_tiles: u32,
    screen_width: u32,
    screen_height: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct TileMetrics {
    pass_overdraw_max: u32,  // Max pass overwrites in this tile
    light_count: u32,        // The offered per-tile count
}

@group(0) @binding(0) var<uniform> params: AggregateParams;
@group(0) @binding(1) var<storage, read_write> pass_overdraw_counters: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> tile_light_counts: array<u32>;
@group(0) @binding(3) var<storage, read_write> tile_metrics: array<TileMetrics>;

/// Compute shader entry point: one thread per tile.
///
/// Dispatch: (num_tiles, 1, 1) with workgroup size 256.
@compute @workgroup_size(256, 1, 1)
fn cs_aggregate_tiles(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tile_idx = gid.x;
    if tile_idx >= params.num_tiles {
        return;
    }

    let tile_x = tile_idx % params.num_tiles_x;
    let tile_y = tile_idx / params.num_tiles_x;

    // Aggregate 16×16 pixels in this tile
    var overdraw_max: u32 = 0u;

    for (var y = 0u; y < 16u; y++) {
        for (var x = 0u; x < 16u; x++) {
            let px = tile_x * 16u + x;
            let py = tile_y * 16u + y;

            if px >= params.screen_width || py >= params.screen_height {
                continue;
            }

            let pixel_idx = py * params.screen_width + px;
            overdraw_max = max(overdraw_max, atomicLoad(&pass_overdraw_counters[pixel_idx]));
        }
    }

    tile_metrics[tile_idx].pass_overdraw_max = overdraw_max;
    tile_metrics[tile_idx].light_count = tile_light_counts[tile_idx];
}
