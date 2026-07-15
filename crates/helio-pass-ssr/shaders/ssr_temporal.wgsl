// ssr_temporal.wgsl — Temporal reprojection for SSR.
//
// Stabilizes SSR by blending with the previous frame's result.
// Uses:
//   • Luminance-based history rejection (disocclusion detection)
//   • AABB neighborhood clamping for anti-ghosting
//   • Confidence-based blend factor

struct Camera {
    view:           mat4x4<f32>,
    proj:           mat4x4<f32>,
    view_proj:      mat4x4<f32>,
    view_proj_inv:  mat4x4<f32>,
    position_near:  vec4<f32>,
    forward_far:    vec4<f32>,
    jitter_frame:   vec4<f32>,
    prev_view_proj: mat4x4<f32>,
}

struct SsrTemporalUniforms {
    frame_index: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> camera:     Camera;
@group(0) @binding(1) var<uniform> params:     SsrTemporalUniforms;
@group(1) @binding(0) var gbuf_depth:          texture_depth_2d;
@group(1) @binding(1) var ssr_current:         texture_2d<f32>;
@group(1) @binding(2) var ssr_history:         texture_2d<f32>;
@group(1) @binding(3) var linear_sampler:      sampler;
@group(1) @binding(4) var ssr_output:          texture_storage_2d<rgba16float, write>;

const HISTORY_BLEND: f32 = 0.88;
const REJECT_LUMA:   f32 = 0.12;

/// wgpu NDC y+ is up while UV y+ is down — see deferred_lighting.wgsl.
fn uv_to_ndc(uv: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
}

fn ndc_to_uv(ndc: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
}

fn world_from_depth(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let world = camera.view_proj_inv * vec4<f32>(uv_to_ndc(uv), depth, 1.0);
    return world.xyz / world.w;
}

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.299, 0.587, 0.114));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half_dims = textureDimensions(ssr_output);
    if gid.x >= half_dims.x || gid.y >= half_dims.y { return; }

    let full_dims = textureDimensions(gbuf_depth);
    let full_uv = (vec2<f32>(gid.xy) * 2.0 + 0.5) / vec2<f32>(full_dims);

    let current = textureLoad(ssr_current, vec2<i32>(gid.xy), 0);
    let cur_color = current.rgb;
    let cur_conf = current.a;

    var output = current;

    if cur_conf > 0.0 {
        let depth = textureLoad(gbuf_depth, vec2<i32>(gid.xy * 2u), 0);

        if depth < 1.0 {
            let world_pos = world_from_depth(full_uv, depth);

            // Reproject to previous frame
            let prev_clip = camera.prev_view_proj * vec4<f32>(world_pos, 1.0);
            let prev_uv = ndc_to_uv(prev_clip.xy / prev_clip.w);

            if prev_clip.w > 0.0 && all(prev_uv >= vec2<f32>(0.0)) && all(prev_uv <= vec2<f32>(1.0)) {
                // AABB neighborhood clamp: compute min/max of 3x3 neighbourhood
                var neighbourhood_min = vec3<f32>(1.0e6);
                var neighbourhood_max = vec3<f32>(-1.0e6);

                for (var dy = -1; dy <= 1; dy++) {
                    for (var dx = -1; dx <= 1; dx++) {
                        let c = clamp(
                            vec2<i32>(i32(gid.x) + dx, i32(gid.y) + dy),
                            vec2<i32>(0),
                            vec2<i32>(half_dims) - 1
                        );
                        let n = textureLoad(ssr_current, c, 0).rgb;
                        neighbourhood_min = min(neighbourhood_min, n);
                        neighbourhood_max = max(neighbourhood_max, n);
                    }
                }

                let history = textureSampleLevel(ssr_history, linear_sampler, prev_uv, 0.0);

                if history.a > 0.0 {
                    // Clamp history to neighbourhood AABB (anti-ghosting)
                    let clamped_history = clamp(history.rgb, neighbourhood_min, neighbourhood_max);

                    let luma_diff = abs(luma(cur_color) - luma(clamped_history));

                    if luma_diff < REJECT_LUMA {
                        let blend = HISTORY_BLEND * history.a;
                        let blended = mix(cur_color, clamped_history, vec3<f32>(blend));
                        // Confidence is the current frame's alone. Accumulating it
                        // across frames drove alpha to 1 everywhere history existed,
                        // so the composite trusted stale reflections completely.
                        output = vec4<f32>(blended, cur_conf);
                    }
                }
            }
        }
    }

    if params.frame_index == 0u {
        output = vec4<f32>(output.rgb, cur_conf);
    }

    textureStore(ssr_output, vec2<i32>(gid.xy), output);
}
