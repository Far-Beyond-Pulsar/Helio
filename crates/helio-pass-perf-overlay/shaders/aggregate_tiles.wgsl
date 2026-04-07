//! Tile aggregation compute shader.
//!
//! Aggregates per-pixel pass overdraw counters into per-tile metrics (16×16 tiles).
//! Also computes shader complexity heuristic from GBuffer ORM and reads tile light counts.

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
    light_count: u32,        // From LightCullPass
    complexity_avg: u32,     // GBuffer ORM heuristic
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: AggregateParams;
@group(0) @binding(1) var<storage, read> pass_overdraw_counters: array<atomic<u32>>;
@group(0) @binding(2) var gbuffer_orm: texture_2d<f32>;
@group(0) @binding(3) var<storage, read> tile_light_counts: array<u32>;
@group(0) @binding(4) var<storage, read_write> tile_metrics: array<TileMetrics>;

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
    var complexity_sum: u32 = 0u;
    var pixel_count: u32 = 0u;

    for (var y = 0u; y < 16u; y++) {
        for (var x = 0u; x < 16u; x++) {
            let px = tile_x * 16u + x;
            let py = tile_y * 16u + y;

            if px >= params.screen_width || py >= params.screen_height {
                continue;
            }

            // Read per-pixel pass overdraw counter
            let pixel_idx = py * params.screen_width + px;
            let overdraw = atomicLoad(&pass_overdraw_counters[pixel_idx]);

            if overdraw > 0u {
                pixel_count += 1u;
                overdraw_max = max(overdraw_max, overdraw);

                // Shader complexity heuristic: roughness + metallic (ORM texture)
                // G channel = roughness, B channel = metallic
                let orm = textureLoad(gbuffer_orm, vec2<u32>(px, py), 0);
                complexity_sum += u32((orm.g + orm.b) * 100.0);
            }
        }
    }

    // Write aggregated metrics
    tile_metrics[tile_idx].pass_overdraw_max = overdraw_max;
    tile_metrics[tile_idx].light_count = tile_light_counts[tile_idx];
    tile_metrics[tile_idx].complexity_avg = complexity_sum / max(pixel_count, 1u);
}
