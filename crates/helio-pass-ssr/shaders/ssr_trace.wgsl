// ssr_trace.wgsl — Deterministic Hi-Z screen-space ray march.
//
// Marches the `hiz_min` pyramid built by HiZBuildPass. One mirror ray per
// pixel, stable frame to frame — no temporal accumulation needed.
//
// The pyramid MUST be min-reduced. The shared `hiz` resource is max-reduced
// for occlusion culling (conservative farthest); marching that makes rays
// tunnel straight through geometry.
//
// NDC depth is *linear in screen space* (projective z/w), so the reflection
// ray is a straight line in (uv, depth01) space. Nothing in the loop needs
// perspective correction or linearization.
//
// Writes Rgba16Float at full resolution: RGB = colour, A = confidence.
//
// The traversal itself lives in helio_core::shader::HIZ, shared with the water
// pass — see #147.
//!use helio_prelude
//!use helio_hiz

@group(0) @binding(0) var<uniform> camera:      Camera;
@group(1) @binding(0) var gbuf_normal:          texture_2d<f32>;
@group(1) @binding(1) var gbuf_orm:             texture_2d<f32>;
@group(1) @binding(2) var gbuf_depth:           texture_depth_2d;
@group(1) @binding(3) var scene_color:          texture_2d<f32>;
@group(1) @binding(4) var hiz_min:              texture_2d<f32>;
@group(1) @binding(5) var linear_sampler:       sampler;
@group(1) @binding(6) var ssr_output:           texture_storage_2d<rgba16float, write>;

const MAX_ITER:      u32 = 64u;
const START_LEVEL:   i32 = 2;
const MAX_LEVEL:     i32 = 8;
const MAX_RAY_DIST:  f32 = 100.0;
const THICKNESS:     f32 = 0.02;
const NORMAL_OFFSET: f32 = 0.002;
const FADE_START:    f32 = 0.6;

fn linearize_depth(d_01: f32) -> f32 {
    return helio_view_depth(d_01, camera.position_near.w, camera.forward_far.w);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(ssr_output);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let px = vec2<i32>(gid.xy);
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);

    // ── G-buffer reads ──────────────────────────────────────────────────────
    let depth_01 = textureLoad(gbuf_depth, px, 0);
    if depth_01 >= 1.0 {
        textureStore(ssr_output, px, vec4<f32>(0.0));
        return;
    }

    let N = helio_gbuffer_normal(textureLoad(gbuf_normal, px, 0).xyz);
    let roughness = textureLoad(gbuf_orm, px, 0).g;
    let roughness_fade = 1.0 - smoothstep(0.4, 0.7, roughness);
    if roughness_fade <= 0.0 {
        textureStore(ssr_output, px, vec4<f32>(0.0));
        return;
    }

    let world_pos = helio_world_from_depth(camera.view_proj_inv, uv, depth_01);
    let V = normalize(camera.position_near.xyz - world_pos);
    let R = reflect(-V, N);
    if dot(R, N) <= 0.0 {
        textureStore(ssr_output, px, vec4<f32>(0.0));
        return;
    }

    // ── Build the ray in view space ─────────────────────────────────────────
    let near = camera.position_near.w;
    var start_view = (camera.view * vec4<f32>(world_pos, 1.0)).xyz;
    let dir_view = normalize((camera.view * vec4<f32>(R, 0.0)).xyz);
    let n_view = (camera.view * vec4<f32>(N, 0.0)).xyz;
    start_view += n_view * (-start_view.z * NORMAL_OFFSET);

    var ray_len = MAX_RAY_DIST;
    if start_view.z + dir_view.z * ray_len > -near {
        ray_len = (-near - start_view.z) / dir_view.z;
    }
    if ray_len <= 0.0 {
        textureStore(ssr_output, px, vec4<f32>(0.0));
        return;
    }
    let end_view = start_view + dir_view * ray_len;

    let clip0 = camera.proj * vec4<f32>(start_view, 1.0);
    let clip1 = camera.proj * vec4<f32>(end_view, 1.0);
    let p0 = vec3<f32>(helio_ndc_to_uv(clip0.xy / clip0.w), clip0.z / clip0.w);
    let p1 = vec3<f32>(helio_ndc_to_uv(clip1.xy / clip1.w), clip1.z / clip1.w);
    let d = p1 - p0;

    // ── Hi-Z traversal ──────────────────────────────────────────────────────
    let march = helio_hiz_march(hiz_min, p0, p1, START_LEVEL, MAX_LEVEL, MAX_ITER);
    if !march.hit {
        textureStore(ssr_output, px, vec4<f32>(0.0));
        return;
    }
    let ray = march.pos;
    let hit_uv = ray.xy;

    // ── Thickness validation ────────────────────────────────────────────────
    let ray_depth = linearize_depth(ray.z);
    let scene_depth = linearize_depth(
        textureLoad(gbuf_depth, vec2<i32>(hit_uv * vec2<f32>(dims)), 0)
    );
    if ray_depth > scene_depth * (1.0 + THICKNESS) {
        textureStore(ssr_output, px, vec4<f32>(0.0));
        return;
    }

    // ── Validity and confidence ─────────────────────────────────────────────
    let n_hit = helio_gbuffer_normal(
        textureLoad(gbuf_normal, vec2<i32>(hit_uv * vec2<f32>(dims)), 0).xyz
    );
    let arriving = -dot(R, n_hit);
    let backface_fade = smoothstep(-0.15, 0.15, arriving);

    let border = min(min(hit_uv.x, 1.0 - hit_uv.x), min(hit_uv.y, 1.0 - hit_uv.y));
    let edge_fade = smoothstep(0.0, 0.1, border);

    let facing_fade = 1.0 - smoothstep(0.26, 0.5, dot(R, V));

    let travelled = length(hit_uv - p0.xy) / max(length(d.xy), 1e-6);
    let dist_fade = 1.0 - smoothstep(FADE_START, 1.0, travelled);

    let confidence = clamp(
        backface_fade * edge_fade * facing_fade * dist_fade * roughness_fade,
        0.0, 1.0,
    );

    let reflection = textureSampleLevel(scene_color, linear_sampler, hit_uv, 0.0).rgb;
    textureStore(ssr_output, px, vec4<f32>(reflection, confidence));
}
