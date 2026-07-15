// ssr_trace.wgsl — Screen-space reflections, DDA march in screen space.
//
// Marches the reflection ray across the screen in even *pixel* steps rather
// than even world-space steps, so sampling density matches the depth buffer it
// tests against. Perspective-correct interpolation follows McGuire & Mara:
// 1/w and view_z/w are both linear in screen space, so the ray's view depth at
// any point along the screen-space line is exact rather than approximated.
//
// The shared `hiz` pyramid is deliberately NOT used here: hiz_build.wgsl is a
// max-reduction (occlusion culling wants the conservative farthest depth), and
// SSR needs the nearest surface in a tile. Feeding max-depth to a ray march
// makes rays tunnel through geometry.
//
// Writes Rgba16Float (half resolution): RGB = colour, A = confidence.

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

@group(0) @binding(0) var<uniform> camera:      Camera;
@group(1) @binding(0) var gbuf_normal:          texture_2d<f32>;
@group(1) @binding(1) var gbuf_orm:             texture_2d<f32>;
@group(1) @binding(2) var gbuf_depth:           texture_depth_2d;
@group(1) @binding(3) var scene_color:          texture_2d<f32>;
@group(1) @binding(4) var linear_sampler:       sampler;
@group(1) @binding(5) var ssr_output:           texture_storage_2d<rgba16float, write>;

const MAX_STEPS:     u32 = 64u;
const BINARY_STEPS:  u32 = 6u;
const MAX_RAY_DIST:  f32 = 100.0;
const PIXEL_STRIDE:  f32 = 2.0;   // half-res pixels advanced per coarse step
const THICKNESS:     f32 = 0.25;  // surface thickness as a fraction of view depth
const FADE_START:    f32 = 0.6;   // ray-length fraction where confidence starts to drop

/// wgpu NDC: y+ is up, but UV y+ is down — the axes disagree, hence the flip.
/// Matches deferred_lighting.wgsl / hlfs_shade.wgsl.
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

/// Depth buffer (wgpu NDC z in [0,1], via glam Mat4::perspective_rh) to positive
/// view-space distance.
fn linearize_depth(d_01: f32) -> f32 {
    let near = camera.position_near.w;
    let far = camera.forward_far.w;
    return near * far / (far - d_01 * (far - near));
}

fn hash13(p: vec3<f32>) -> f32 {
    var q = fract(p * 0.1031);
    q += dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half_dims = textureDimensions(ssr_output);
    if gid.x >= half_dims.x || gid.y >= half_dims.y { return; }

    let full_dims = textureDimensions(gbuf_depth);
    let full_px = gid.xy * 2u;
    let full_uv = (vec2<f32>(full_px) + 0.5) / vec2<f32>(full_dims);

    // ── G-buffer reads ──────────────────────────────────────────────────────
    let depth_01 = textureLoad(gbuf_depth, vec2<i32>(full_px), 0);
    if depth_01 >= 1.0 {
        textureStore(ssr_output, vec2<i32>(gid.xy), vec4<f32>(0.0));
        return;
    }

    // G-buffer stores world normals raw in Rgba16Float — no unorm decode.
    let N = normalize(textureLoad(gbuf_normal, vec2<i32>(full_px), 0).xyz);
    let roughness = textureLoad(gbuf_orm, vec2<i32>(full_px), 0).g;

    // Matches the composite in deferred_lighting.wgsl; skip what it won't blend.
    let roughness_fade = 1.0 - smoothstep(0.4, 0.7, roughness);
    if roughness_fade <= 0.0 {
        textureStore(ssr_output, vec2<i32>(gid.xy), vec4<f32>(0.0));
        return;
    }

    let world_pos = world_from_depth(full_uv, depth_01);
    let V = normalize(camera.position_near.xyz - world_pos);
    let R = reflect(-V, N);
    if dot(R, N) <= 0.0 {
        textureStore(ssr_output, vec2<i32>(gid.xy), vec4<f32>(0.0));
        return;
    }

    // ── Build the ray in view space, clipped to the near plane ──────────────
    let near = camera.position_near.w;
    let start_view = (camera.view * vec4<f32>(world_pos, 1.0)).xyz;
    let dir_view = normalize((camera.view * vec4<f32>(R, 0.0)).xyz);

    // View space is RH (-z forward), so the near plane sits at z = -near.
    var ray_len = MAX_RAY_DIST;
    if start_view.z + dir_view.z * ray_len > -near {
        ray_len = (-near - start_view.z) / dir_view.z;
    }
    if ray_len <= 0.0 {
        textureStore(ssr_output, vec2<i32>(gid.xy), vec4<f32>(0.0));
        return;
    }
    let end_view = start_view + dir_view * ray_len;

    // ── Project the endpoints to screen space ───────────────────────────────
    let clip0 = camera.proj * vec4<f32>(start_view, 1.0);
    let clip1 = camera.proj * vec4<f32>(end_view, 1.0);
    let k0 = 1.0 / clip0.w;
    let k1 = 1.0 / clip1.w;
    let uv0 = ndc_to_uv(clip0.xy * k0);
    let uv1 = ndc_to_uv(clip1.xy * k1);

    // 1/w and view_z/w interpolate linearly in screen space; view_z does not.
    let z0 = start_view.z * k0;
    let z1 = end_view.z * k1;

    // Step count from the ray's screen-space extent, so steps land ~PIXEL_STRIDE apart.
    let px_span = length((uv1 - uv0) * vec2<f32>(half_dims));
    let steps = clamp(px_span / PIXEL_STRIDE, 1.0, f32(MAX_STEPS));
    let dt = 1.0 / steps;

    // Dither the start offset — breaks banding into noise the temporal pass resolves.
    let jitter = hash13(vec3<f32>(vec2<f32>(gid.xy), camera.jitter_frame.z));

    var t = dt * jitter;
    var prev_t = 0.0;
    var hit = false;
    var hit_t = 0.0;

    for (var i = 0u; i < MAX_STEPS; i++) {
        if t > 1.0 { break; }

        let uv = mix(uv0, uv1, t);
        if any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) { break; }

        let ray_depth = -(mix(z0, z1, t) / mix(k0, k1, t));
        let scene_depth = linearize_depth(
            textureLoad(gbuf_depth, vec2<i32>(uv * vec2<f32>(full_dims)), 0)
        );

        // Thickness scales with depth: a fixed world-space slab is far too thin
        // up close and far too thick in the distance.
        if ray_depth > scene_depth && ray_depth < scene_depth * (1.0 + THICKNESS) {
            hit = true;
            hit_t = t;
            break;
        }

        prev_t = t;
        t += dt;
    }

    if !hit {
        textureStore(ssr_output, vec2<i32>(gid.xy), vec4<f32>(0.0));
        return;
    }

    // ── Binary refinement between the last miss and the hit ─────────────────
    var lo = prev_t;
    var hi = hit_t;
    for (var j = 0u; j < BINARY_STEPS; j++) {
        let mid = (lo + hi) * 0.5;
        let mid_uv = mix(uv0, uv1, mid);
        let mid_ray_depth = -(mix(z0, z1, mid) / mix(k0, k1, mid));
        let mid_scene_depth = linearize_depth(
            textureLoad(gbuf_depth, vec2<i32>(mid_uv * vec2<f32>(full_dims)), 0)
        );
        if mid_ray_depth > mid_scene_depth {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    let final_t = hi;
    let hit_uv = mix(uv0, uv1, final_t);

    // ── Confidence ──────────────────────────────────────────────────────────
    // Per-axis border fade on the *hit* — a radial fade from the source pixel
    // has nothing to do with where the ray actually left the screen.
    let border = min(
        min(hit_uv.x, 1.0 - hit_uv.x),
        min(hit_uv.y, 1.0 - hit_uv.y)
    );
    let edge_fade = smoothstep(0.0, 0.1, border);

    // Rays aimed back at the camera have no on-screen data behind them.
    let facing_fade = 1.0 - smoothstep(0.25, 0.6, dot(R, -V));

    // Fade as the ray runs out of budget, so hits don't pop at MAX_RAY_DIST.
    let dist_fade = 1.0 - smoothstep(FADE_START, 1.0, final_t);

    let confidence = clamp(edge_fade * facing_fade * dist_fade * roughness_fade, 0.0, 1.0);

    // scene_color has no mip chain (pre_aa is MatchSurface, 1 mip), so roughness
    // blur has to come from the denoise pass rather than a mip bias here.
    let reflection = textureSampleLevel(scene_color, linear_sampler, hit_uv, 0.0).rgb;

    textureStore(ssr_output, vec2<i32>(gid.xy), vec4<f32>(reflection, confidence));
}
