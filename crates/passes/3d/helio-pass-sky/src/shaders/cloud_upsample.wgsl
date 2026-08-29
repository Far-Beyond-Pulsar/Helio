// =============================================================================
// Depth-Aware Bilateral Upsample — Volumetric Cloud Pipeline
// =============================================================================
// Upsamples the quarter-resolution temporally accumulated output to full screen
// resolution. Compares depth of high-res depth buffer against quarter-res depth
// buffer using a cross-bilateral filter weight to prevent edge-bleeding against
// foreground geometry.
//
// Pipeline stage: 3/3 — Quarter-Res Ray March → Temporal Reprojection → Bilateral Upsample
// =============================================================================

struct UpsampleParams {
    resolution: vec4<f32>,          // width, height, 1/width, 1/height
    quarter_resolution: vec4<f32>,  // qw, qh, 1/qw, 1/qh
    bilateral_info: vec4<f32>,      // depth sigma, color sigma, edge threshold, pad
    debug_mode: vec4<u32>,          // debug selector
}

@group(0) @binding(0) var<uniform> params: UpsampleParams;
@group(0) @binding(1) var quarter_color: texture_2d<f32>;   // temporally accumulated quarter-res (rgba16float)
@group(0) @binding(2) var quarter_sampler: sampler;
@group(0) @binding(3) var quarter_depth: texture_2d<f32>;   // quarter-res depth (r channel)
@group(0) @binding(4) var full_depth: texture_depth_2d;     // high-res depth buffer
@group(0) @binding(5) var scene_color: texture_2d<f32>;     // background scene color for compositing
@group(0) @binding(6) var scene_sampler: sampler;
@group(0) @binding(7) var output_texture: texture_storage_2d<rgba16float, write>;

// Cross-bilateral filter weight: compares high-res depth vs quarter-res depth
// Weight = exp( -(depth_diff^2) / (2 * sigma_depth^2) )
fn bilateral_weight(full_d: f32, quarter_d: f32, sigma: f32) -> f32 {
    let diff = full_d - quarter_d;
    return exp(- (diff * diff) / (2.0 * sigma * sigma + 0.0001));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = vec2<u32>(u32(params.resolution.x), u32(params.resolution.y));
    if (gid.x >= res.x || gid.y >= res.y) { return; }

    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5)) / params.resolution.xy;
    let quarter_uv = uv; // same UV space, quarter texture is smaller but sampled with clamp

    // High-res depth at this pixel
    let depth_size = textureDimensions(full_depth);
    let full_d = textureLoad(full_depth, vec2<i32>(gid.xy), 0);

    // Gather 4 nearest quarter-res taps with bilateral weighting
    // Quarter texel size in UV: 4 / full_res per axis
    let q_texel = params.quarter_resolution.zw; // 1/qw, 1/qh
    let q_uv_base = uv; // center

    // 2x2 bilinear neighborhood in quarter-res (covers the 4x4 block's 4 quarter samples relevant to this full pixel)
    // Cross-bilateral: weight by depth similarity to prevent bleeding across geometry edges
    let sigma_depth: f32 = max(params.bilateral_info.x, 0.5); // tunable, default ~1.0
    var sum_color = vec4<f32>(0.0);
    var sum_weight: f32 = 0.0;

    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            let tap_uv = q_uv_base + vec2<f32>(f32(dx), f32(dy)) * q_texel * 0.5;
            let tap_color = textureSampleLevel(quarter_color, quarter_sampler, tap_uv, 0.0);
            let tap_depth = textureSampleLevel(quarter_depth, quarter_sampler, tap_uv, 0.0).r;

            // Cross-bilateral filter weight based on depth comparison
            // Prevent edge-bleeding against foreground geometry
            let w_depth = bilateral_weight(full_d, tap_depth, sigma_depth);
            // Tent weight for spatial distance
            let dist = length(vec2<f32>(f32(dx), f32(dy)));
            let w_spatial = exp(-dist * dist * 0.5);

            let w = w_depth * w_spatial;
            sum_color = sum_color + tap_color * w;
            sum_weight = sum_weight + w;
        }
    }

    var upsampled = sum_color / max(sum_weight, 0.0001);

    // If depth indicates foreground geometry very close (full_d < quarter_d - threshold),
    // reduce cloud contribution to avoid halos — cloud should be occluded
    let edge_threshold: f32 = max(params.bilateral_info.z, 0.1);
    let depth_diff = full_d - textureSampleLevel(quarter_depth, quarter_sampler, quarter_uv, 0.0).r;
    if (full_d < 0.99 && depth_diff < -edge_threshold) {
        // Foreground geometry in front of cloud — increase transmittance (less cloud)
        upsampled.a = mix(upsampled.a, 1.0, clamp((-depth_diff - edge_threshold) * 2.0, 0.0, 1.0));
    }

    // Composite with scene color: scene * transmittance + cloud * (1 - transmittance) is done here
    // But spec's raymarch already composites; this upsample just returns cloud radiance + transmittance
    // Final composite in post: out = cloud.rgb + scene * cloud.a (where a is transmittance)
    var final_color = upsampled;

    // Apply scene compositing if scene_color is valid (not black)
    let scene = textureSampleLevel(scene_color, scene_sampler, vec2<f32>(uv.x, 1.0 - uv.y), 0.0).rgb;
    // If caller wants pre-composited output, do it here (optional — otherwise deferred)
    // We output separate cloud color + transmittance for deferred composite
    // final_color.rgb = final_color.rgb + scene * final_color.a;

    // Debug views
    if (params.debug_mode.x == 4u) {
        // Visualize bilateral weights as edge mask
        let edge = 1.0 - clamp(sum_weight / 9.0, 0.0, 1.0);
        final_color = vec4<f32>(vec3<f32>(edge), 1.0);
    }

    textureStore(output_texture, vec2<i32>(gid.xy), final_color);
}

// Alternative fragment entry for raster path (fullscreen triangle)
// Used when compute path is not desired (e.g., platform without storage writes)
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let pos = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    var out: VertexOutput;
    out.position = vec4<f32>(pos[vi], 0.0, 1.0);
    out.uv = pos[vi] * 0.5 + vec2<f32>(0.5);
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let uv = input.uv;
    let full_d = textureLoad(full_depth, vec2<i32>(i32(uv.x * params.resolution.x), i32(uv.y * params.resolution.y)), 0);
    let q_texel = params.quarter_resolution.zw;
    let sigma_depth: f32 = max(params.bilateral_info.x, 0.5);
    var sum_color = vec4<f32>(0.0);
    var sum_weight: f32 = 0.0;
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            let tap_uv = uv + vec2<f32>(f32(dx), f32(dy)) * q_texel * 0.5;
            let tap_color = textureSampleLevel(quarter_color, quarter_sampler, tap_uv, 0.0);
            let tap_depth = textureSampleLevel(quarter_depth, quarter_sampler, tap_uv, 0.0).r;
            let w_depth = bilateral_weight(full_d, tap_depth, sigma_depth);
            let dist = length(vec2<f32>(f32(dx), f32(dy)));
            let w_spatial = exp(-dist * dist * 0.5);
            let w = w_depth * w_spatial;
            sum_color = sum_color + tap_color * w;
            sum_weight = sum_weight + w;
        }
    }
    return sum_color / max(sum_weight, 0.0001);
}
