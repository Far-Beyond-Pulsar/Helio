// ssr_denoise.wgsl — 5x5 bilateral blur on SSR output.
//
// Preserves edges by weighting samples by depth and normal similarity.
// Applies only where SSR confidence > 0; sky and invalid pixels pass through.

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

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var gbuf_normal:  texture_2d<f32>;
@group(1) @binding(1) var gbuf_depth:   texture_depth_2d;
@group(1) @binding(2) var gbuf_orm:     texture_2d<f32>;
@group(1) @binding(3) var ssr_input:    texture_2d<f32>;
@group(1) @binding(4) var ssr_output:   texture_storage_2d<rgba16float, write>;

const KERNEL_HALF: i32 = 2;  // 5x5 kernel
// Blur radius scales with roughness: mirrors stay sharp, gloss spreads.
const RADIUS_SCALE: f32 = 1.5;

/// wgpu NDC z in [0,1] (glam Mat4::perspective_rh) to positive view depth.
fn linearize(depth: f32, near: f32, far: f32) -> f32 {
    return near * far / (far - depth * (far - near));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(ssr_output);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let full_coord = vec2<i32>(gid.xy * 2u);

    let center_ssr = textureLoad(ssr_input, vec2<i32>(gid.xy), 0);
    let center_depth = textureLoad(gbuf_depth, full_coord, 0);

    // Sky has no surface to reflect off.
    if center_depth >= 1.0 {
        textureStore(ssr_output, vec2<i32>(gid.xy), vec4<f32>(0.0));
        return;
    }

    // A missed pixel is NOT skipped: the trace is a dithered point sample, so
    // misses are gaps in sampling rather than evidence of no reflection. This
    // kernel is a reconstruction filter — it pulls colour and confidence from
    // neighbours that did hit, which is what turns sparse hits into a surface.

    // G-buffer stores world normals raw in Rgba16Float — no unorm decode.
    let center_normal = normalize(textureLoad(gbuf_normal, full_coord, 0).xyz);
    let near = camera.position_near.w;
    let far = camera.forward_far.w;
    let center_linear = linearize(center_depth, near, far);
    let roughness = textureLoad(gbuf_orm, full_coord, 0).g;

    const DEPTH_SIGMA: f32 = 0.05;
    const NORMAL_SIGMA: f32 = 8.0;

    // A near-mirror surface should barely blur at all; gloss gets the full kernel.
    let sigma = max(0.5, roughness * RADIUS_SCALE * f32(KERNEL_HALF));

    var total_weight = 0.0;
    var weighted_color = vec3<f32>(0.0);
    var weighted_alpha = 0.0;

    for (var dy = -KERNEL_HALF; dy <= KERNEL_HALF; dy++) {
        for (var dx = -KERNEL_HALF; dx <= KERNEL_HALF; dx++) {
            let sample_coord = vec2<i32>(i32(gid.x) + dx, i32(gid.y) + dy);
            if sample_coord.x < 0 || sample_coord.x >= i32(dims.x) ||
               sample_coord.y < 0 || sample_coord.y >= i32(dims.y) {
                continue;
            }

            let sample_ssr = textureLoad(ssr_input, sample_coord, 0);
            // Misses contribute nothing rather than averaging black in: a miss
            // carries no colour, so it must lower confidence, not darken it.
            if sample_ssr.a <= 0.0 { continue; }

            let sample_full = sample_coord * 2;
            let sample_normal = normalize(textureLoad(gbuf_normal, sample_full, 0).xyz);
            let sample_depth = textureLoad(gbuf_depth, sample_full, 0);

            let sample_linear = linearize(sample_depth, near, far);
            let depth_diff = abs(sample_linear - center_linear) / max(center_linear, 0.001);
            let depth_weight = exp(-depth_diff * depth_diff / (2.0 * DEPTH_SIGMA * DEPTH_SIGMA));

            let normal_weight = pow(max(dot(sample_normal, center_normal), 0.0), NORMAL_SIGMA);

            // Gaussian spatial weight
            let spatial = exp(-f32(dx * dx + dy * dy) / (2.0 * sigma * sigma));

            let weight = spatial * depth_weight * normal_weight;
            total_weight += weight;
            weighted_color += sample_ssr.rgb * weight;
            weighted_alpha += sample_ssr.a * weight;
        }
    }

    if total_weight <= 0.0 {
        textureStore(ssr_output, vec2<i32>(gid.xy), vec4<f32>(0.0));
        return;
    }

    // Colour is normalised over contributing (hit) neighbours only, so it stays
    // full-intensity. Confidence is normalised over the *full* kernel weight, so
    // a pixel whose neighbourhood mostly missed ends up genuinely less confident
    // and the composite leans on the cubemap instead.
    var kernel_weight = 0.0;
    for (var dy = -KERNEL_HALF; dy <= KERNEL_HALF; dy++) {
        for (var dx = -KERNEL_HALF; dx <= KERNEL_HALF; dx++) {
            kernel_weight += exp(-f32(dx * dx + dy * dy) / (2.0 * sigma * sigma));
        }
    }

    let color = weighted_color / total_weight;
    let alpha = clamp(weighted_alpha / max(kernel_weight, 0.001), 0.0, 1.0);
    textureStore(ssr_output, vec2<i32>(gid.xy), vec4<f32>(color, alpha));
}
