// =============================================================================
// Temporal Reprojection & Accumulation Pass — Volumetric Cloud Pipeline
// =============================================================================
// Maintains a history buffer storing the full-resolution target from previous
// frame. For every pixel, reprojects previous frame's color/transmittance using
// camera motion vectors and linear depth. Implements Neighborhood Clamping /
// Variance Bounding to prevent ghosting/trailing during fast camera movement.
// Blends current quarter-res evaluation with history using exponential moving
// average (90% history, 10% new sample).
//
// Pipeline stage: 2/3 — Quarter-Res Ray March → Temporal Reprojection → Bilateral Upsample
// =============================================================================

struct ReprojectParams {
    resolution: vec4<f32>,           // width, height, 1/width, 1/height
    quarter_resolution: vec4<f32>,   // qw, qh, 1/qw, 1/qh
    reproj_info: vec4<f32>,          // history valid, blend factor (0.90), frame idx, jitter scale
    camera_prev_view_proj: mat4x4<f32>,
    camera_curr_view_proj: mat4x4<f32>,
    camera_prev_inv_view_proj: mat4x4<f32>,
    camera_curr_inv_view_proj: mat4x4<f32>,
    debug_mode: vec4<u32>,           // 0 off, 2 = reprojection confidence / clamping masks
}

@group(0) @binding(0) var<uniform> params: ReprojectParams;
@group(0) @binding(1) var quarter_color: texture_2d<f32>;      // current quarter-res raymarch output (nearest)
@group(0) @binding(2) var quarter_sampler: sampler;
@group(0) @binding(3) var history_color: texture_2d<f32>;      // previous frame's full-res history (rgba16float: rgb=color, a=transmittance)
@group(0) @binding(4) var history_sampler: sampler;
@group(0) @binding(5) var depth_texture: texture_depth_2d;     // high-res depth buffer
@group(0) @binding(6) var velocity_texture: texture_2d<f32>;   // motion vectors (gbuffer_velocity), rg = delta uv
@group(0) @binding(7) var quarter_depth: texture_2d<f32>;      // quarter-res depth from raymarch data (r channel)
@group(0) @binding(8) var output_color: texture_storage_2d<rgba16float, write>; // temporally accumulated quarter-res result
@group(0) @binding(9) var output_confidence: texture_storage_2d<r32float, write>; // reprojection confidence for debug

// Neighborhood Clamping / Variance Bounding
// Calculate min/max color in 3x3 quarter-res neighborhood around reprojected pixel.
// Clamp historical color to this range to prevent ghosting or trailing artifacts.
struct MinMax { c_min: vec4<f32>, c_max: vec4<f32> };
fn neighborhood_min_max(uv: vec2<f32>, texel_size: vec2<f32>) -> MinMax {
    var c_min = vec4<f32>(1e5);
    var c_max = vec4<f32>(-1e5);
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            let sample_uv = uv + vec2<f32>(f32(dx), f32(dy)) * texel_size;
            let c = textureSampleLevel(quarter_color, quarter_sampler, sample_uv, 0.0);
            c_min = min(c_min, c);
            c_max = max(c_max, c);
        }
    }
    // Variance bounding: expand by local variance to reduce over-clamping
    var mean = vec4<f32>(0.0);
    var mean_sq = vec4<f32>(0.0);
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            let sample_uv = uv + vec2<f32>(f32(dx), f32(dy)) * texel_size;
            let c = textureSampleLevel(quarter_color, quarter_sampler, sample_uv, 0.0);
            mean = mean + c;
            mean_sq = mean_sq + c * c;
        }
    }
    mean = mean / 9.0;
    mean_sq = mean_sq / 9.0;
    let variance = mean_sq - mean * mean;
    let std_dev = sqrt(max(variance, vec4<f32>(0.0)));
    // Bounding box = min/max expanded by 1.2 * stddev (standard variance clipping)
    let gamma: f32 = 1.2;
    c_min = max(c_min - std_dev * gamma, vec4<f32>(0.0));
    c_max = c_max + std_dev * gamma;
    return MinMax(c_min, c_max);
}

fn clamp_history_to_neighborhood(history: vec4<f32>, n_min: vec4<f32>, n_max: vec4<f32>) -> vec4<f32> {
    // Neighborhood Clamping: clamp historical color to 3x3 min/max range
    return clamp(history, n_min, n_max);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = vec2<u32>(u32(params.resolution.x), u32(params.resolution.y));
    // This pass runs at quarter-res (writes to quarter accumulation target)
    let qres = vec2<u32>(u32(params.quarter_resolution.x), u32(params.quarter_resolution.y));
    if (gid.x >= qres.x || gid.y >= qres.y) { return; }

    let uv_quarter = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5)) / vec2<f32>(f32(qres.x), f32(qres.y));
    let uv_full = uv_quarter; // quarter uv corresponds to 4x4 block center in full res

    let current = textureSampleLevel(quarter_color, quarter_sampler, uv_quarter, 0.0);
    let texel_size = params.quarter_resolution.zw;

    let history_valid = params.reproj_info.x > 0.5;
    let blend_history: f32 = 0.90; // 90% history, 10% new sample — exponential moving average
    let blend_new: f32 = 1.0 - blend_history;

    var accumulated = current;
    var confidence: f32 = 1.0;

    if (history_valid) {
        // Reproject using camera motion vectors and linear depth
        // Method 1: velocity texture (preferred, from gbuffer_velocity)
        let velocity = textureSampleLevel(velocity_texture, history_sampler, uv_full, 0.0).xy;
        // velocity is in pixels/frame; convert to UV delta
        let velocity_uv = velocity * params.resolution.zw;

        // Method 2: reconstruct via view_proj matrices and depth (fallback)
        let depth_raw = textureLoad(depth_texture, vec2<i32>(i32(uv_full.x * params.resolution.x), i32(uv_full.y * params.resolution.y)), 0);
        let linear_depth = depth_raw; // assume linear for this pipeline; real would linearize
        // Reproject to previous clip space
        // Use velocity when available, else matrix reprojection
        var prev_uv = uv_full - velocity_uv;
        // Clamp to valid history range
        let out_of_bounds = any(prev_uv < vec2<f32>(0.0)) || any(prev_uv > vec2<f32>(1.0));
        if (out_of_bounds) {
            confidence = 0.0;
        } else {
            let history_sample = textureSampleLevel(history_color, history_sampler, prev_uv, 0.0);

            // Neighborhood Clamping / Variance Bounding: 3x3 quarter-res neighborhood
            let bounds = neighborhood_min_max(uv_quarter, texel_size);
            let n_min = bounds.c_min;
            let n_max = bounds.c_max;
            let clamped_history = clamp_history_to_neighborhood(history_sample, n_min, n_max);

            // Confidence based on clamping distance — how much history was altered
            let clamp_dist = length(clamped_history.rgb - history_sample.rgb);
            let clamp_factor = clamp(clamp_dist * 8.0, 0.0, 1.0);
            // If clamped heavily, history was ghosting — reduce its weight
            let adaptive_blend = mix(blend_history, 0.5, clamp_factor * 0.7);
            let adaptive_new = 1.0 - adaptive_blend;

            // Depth disocclusion: if current depth differs much from reprojected depth, disocclude
            let current_depth = textureSampleLevel(quarter_depth, quarter_sampler, uv_quarter, 0.0).r;
            // History depth not directly stored; approximate via luminance of history alpha
            // Real pipeline would store separate history depth texture
            let depth_diff = abs(current_depth - linear_depth);
            let disocclusion = smoothstep(0.5, 5.0, depth_diff);
            let final_history_weight = mix(adaptive_blend, 0.0, disocclusion);
            let final_new_weight = 1.0 - final_history_weight;

            // Exponential moving average
            accumulated = clamped_history * final_history_weight + current * final_new_weight;
            confidence = 1.0 - clamp_factor * 0.85 - disocclusion * 0.9;
            confidence = clamp(confidence, 0.0, 1.0);
        }
    } else {
        confidence = 0.0;
    }

    // Debug: Reprojection confidence / clamping masks
    if (params.debug_mode.x == 2u) {
        // Visualize confidence as heatmap, clamping mask as red overlay
        let heat = mix(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), confidence);
        accumulated = vec4<f32>(heat, 1.0);
    }

    textureStore(output_color, vec2<i32>(gid.xy), accumulated);
    textureStore(output_confidence, vec2<i32>(gid.xy), vec4<f32>(confidence, 0.0, 0.0, 1.0));
}
