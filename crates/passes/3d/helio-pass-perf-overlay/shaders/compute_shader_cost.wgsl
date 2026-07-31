//! Compute shader to estimate per-pixel rendering cost.
//!
//! Estimates GPU work per pixel based on:
//! - Tile light counts (light loop iterations)
//! - Material complexity (BRDF cost from roughness/metallic)
//! - Shadow samples (estimated from light types)
//! - Texture samples (fixed G-buffer + varying IBL/AO)
//!
//! This provides an operation count estimate without instrumenting deferred lighting shader.

struct ComputeCostParams {
    screen_width: u32,
    screen_height: u32,
    num_tiles_x: u32,
    num_timing_entries: u32,
}

struct MaterialTimingEntry {
    roughness: f32,
    metallic: f32,
    num_lights: u32,
    gpu_time_ns: u32,
}

@group(0) @binding(0) var<uniform> params: ComputeCostParams;
@group(0) @binding(1) var gbuffer_orm: texture_2d<f32>;
@group(0) @binding(2) var gbuffer_depth: texture_2d<f32>;
@group(0) @binding(3) var<storage, read> tile_light_counts: array<u32>;
@group(0) @binding(4) var<storage, read_write> pixel_cost: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read> material_timings: array<MaterialTimingEntry>;

/// Cost weights (calibrated to approximate GPU cycles) - used if profiling data unavailable
const COST_BASE_PIXEL: f32 = 10.0;          // Base cost per pixel (tex loads, position reconstruction)
const COST_PER_LIGHT: f32 = 80.0;           // Cost per light evaluation (BRDF + attenuation)
const COST_SHADOW_SAMPLE: f32 = 40.0;       // Cost per shadow sample (texture compare)
const COST_ROUGHNESS_FACTOR: f32 = 30.0;    // Additional cost for rough surfaces (complex BRDF lobes)
const COST_METALLIC_FACTOR: f32 = 20.0;     // Additional cost for metallic (fresnel, reflections)
const COST_IBL_SAMPLE: f32 = 25.0;          // Image-based lighting sample cost

/// Look up measured GPU time for a material configuration.
/// Finds the closest matching entries and interpolates if needed.
fn lookup_material_cost(roughness: f32, metallic: f32, num_lights: u32) -> f32 {
    // If no timing data available, fall back to heuristic
    if params.num_timing_entries == 0u {
        return 0.0;
    }

    var best_match_cost = 0.0;
    var best_match_distance = 999999.0;

    // Simple nearest-neighbor lookup
    for (var i = 0u; i < params.num_timing_entries; i++) {
        let entry = material_timings[i];
        
        // Calculate distance in parameter space
        let dr = abs(entry.roughness - roughness);
        let dm = abs(entry.metallic - metallic);
        let dl = abs(f32(entry.num_lights) - f32(num_lights)) / 32.0; // Normalize light count
        
        let distance = dr + dm + dl;
        
        if distance < best_match_distance {
            best_match_distance = distance;
            best_match_cost = f32(entry.gpu_time_ns);
        }
    }

    return best_match_cost;
}

@compute @workgroup_size(16, 16, 1)
fn cs_compute_shader_cost(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;    let py = gid.y;

    if px >= params.screen_width || py >= params.screen_height {
        return;
    }

    let pixel_idx = py * params.screen_width + px;

    // Check if pixel is geometry or sky
    let depth = textureLoad(gbuffer_depth, vec2<u32>(px, py), 0).r;
    if depth >= 1.0 {
        // Sky pixel - minimal cost (already rendered by sky pass)
        atomicStore(&pixel_cost[pixel_idx], 0u);
        return;
    }

    // Read material properties
    let orm = textureLoad(gbuffer_orm, vec2<u32>(px, py), 0);
    let roughness = orm.g;
    let metallic = orm.b;

    // Get tile light count
    let tile_x = px / 16u;
    let tile_y = py / 16u;
    let tile_idx = tile_y * params.num_tiles_x + tile_x;
    let num_lights = tile_light_counts[tile_idx];

    var cost: f32;

    // Use measured timings if available, otherwise fall back to heuristic
    if params.num_timing_entries > 0u {
        cost = lookup_material_cost(roughness, metallic, num_lights);
    } else {
        // Estimate total cost using heuristic
        cost = COST_BASE_PIXEL;

        // Light evaluation cost (BRDF + attenuation)
        cost += f32(num_lights) * COST_PER_LIGHT;

        // Shadow sampling cost (assume ~50% of lights cast shadows, 4 samples per light)
        let shadow_lights = max(num_lights / 2u, 1u);
        cost += f32(shadow_lights) * 4.0 * COST_SHADOW_SAMPLE;

        // Material complexity cost
        cost += roughness * COST_ROUGHNESS_FACTOR;   // More diffuse lobes
        cost += metallic * COST_METALLIC_FACTOR;     // Fresnel calculations

        // IBL/environment sampling (all lit pixels)
        cost += COST_IBL_SAMPLE;
    }

    // Store cost (scaled to integer)
    atomicStore(&pixel_cost[pixel_idx], u32(cost));
}
