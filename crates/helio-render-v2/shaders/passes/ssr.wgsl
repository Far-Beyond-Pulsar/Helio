//! Screen-Space Reflections (SSR).
//!
//! Hierarchical view-space ray-march that traces reflection rays against the
//! depth buffer.  Reads the current frame's HDR scene colour and the G-buffer
//! normal+ORM for direction and confidence calculation.
//!
//! Algorithm:
//!   1. Reconstruct view-space position from depth.
//!   2. Compute reflection direction in view space.
//!   3. Ray-march in view space using variable-step-size (doubles each miss).
//!   4. For each step: project to screen, compare depth.
//!   5. Binary search refinement on hit.
//!   6. Fade confidence by roughness, screen-edge distance, and ray length.
//!   7. Composite over the existing IBL-only specular in the HDR buffer via
//!      additive blend (LoadOp::Load applied in the Rust pass).

// G-buffer inputs
@group(0) @binding(0) var gbuf_normal: texture_2d<f32>;   // world-space normal
@group(0) @binding(1) var gbuf_orm:    texture_2d<f32>;   // AO/roughness/metallic
@group(0) @binding(2) var gbuf_depth:  texture_depth_2d;  // scene depth

// HDR scene colour (current frame, pre-bloom)
@group(0) @binding(3) var scene_color: texture_2d<f32>;
@group(0) @binding(4) var linear_samp: sampler;

// Additional uniforms
struct SsrUniforms {
    view:     mat4x4<f32>,   // view matrix (not view_proj)
    proj:     mat4x4<f32>,   // projection
    proj_inv: mat4x4<f32>,
    max_steps: u32,
    max_dist:  f32,
    thickness: f32,
    _pad:      f32,
}
@group(0) @binding(5) var <uniform> ssr: SsrUniforms;

// ── Fullscreen triangle ───────────────────────────────────────────────────────

struct VSOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0)      uv:       vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
    var pos = array<vec2<f32>, 3>(vec2(-1.0,-1.0), vec2(3.0,-1.0), vec2(-1.0,3.0));
    var uvs = array<vec2<f32>, 3>(vec2(0.0,1.0),   vec2(2.0,1.0),  vec2(0.0,-1.0));
    var o: VSOut;
    o.clip_pos = vec4<f32>(pos[vi], 0.0, 1.0);
    o.uv       = uvs[vi];
    return o;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn depth_to_view_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc     = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let view_h  = ssr.proj_inv * ndc;
    return view_h.xyz / view_h.w;
}

fn view_to_screen(view_pos: vec3<f32>) -> vec3<f32> {
    let clip = ssr.proj * vec4<f32>(view_pos, 1.0);
    let ndc  = clip.xyz / clip.w;
    return vec3<f32>(
        ndc.x * 0.5 + 0.5,
        0.5 - ndc.y * 0.5,
        ndc.z
    );
}

fn screen_depth(uv: vec2<f32>, dim: vec2<f32>) -> f32 {
    let pix = vec2<i32>(clamp(vec2<i32>(uv * dim), vec2(0), vec2<i32>(dim) - 1));
    return textureLoad(gbuf_depth, pix, 0);
}

// Edge fade — dims SSR near screen borders to hide the missing information
fn edge_fade(uv: vec2<f32>) -> f32 {
    let e = 0.1;
    return smoothstep(0.0, e, uv.x)
         * smoothstep(1.0, 1.0 - e, uv.x)
         * smoothstep(0.0, e, uv.y)
         * smoothstep(1.0, 1.0 - e, uv.y);
}

// ── Fragment ──────────────────────────────────────────────────────────────────

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let pix  = vec2<i32>(i32(in.clip_pos.x), i32(in.clip_pos.y));
    let dim  = vec2<f32>(textureDimensions(gbuf_depth));
    let uv   = in.uv;

    let depth = textureLoad(gbuf_depth, pix, 0);
    if depth >= 1.0 { return vec4<f32>(0.0); }   // sky — no reflection

    let orm      = textureLoad(gbuf_orm, pix, 0);
    let roughness = orm.g;
    // Too rough = no SSR contribution (diffuse surfaces shouldn't show SSR)
    if roughness > 0.6 { return vec4<f32>(0.0); }

    // World normal → view normal
    let world_n  = normalize(textureLoad(gbuf_normal, pix, 0).xyz);
    let view_n   = normalize((ssr.view * vec4<f32>(world_n, 0.0)).xyz);

    // View-space position of the current fragment
    let view_pos = depth_to_view_pos(uv, depth);

    // Reflection direction in view space
    let view_dir = normalize(view_pos);          // view origin is 0 in view space
    let refl_dir = reflect(view_dir, view_n);
    if refl_dir.z >= 0.0 { return vec4<f32>(0.0); }  // pointing away from camera in view space

    // Ray march
    var step_size     = ssr.max_dist / f32(ssr.max_steps);
    var ray           = view_pos + refl_dir * step_size;
    var prev_ray      = view_pos;
    var hit_uv        = vec2<f32>(0.0);
    var hit           = false;

    for (var i = 0u; i < ssr.max_steps; i++) {
        let ss    = view_to_screen(ray);
        if ss.x < 0.0 || ss.x > 1.0 || ss.y < 0.0 || ss.y > 1.0 || ss.z > 1.0 { break; }
        let scene_d = screen_depth(ss.xy, dim);
        let diff    = ss.z - scene_d;
        if diff > 0.0 && diff < ssr.thickness {
            // Binary search refinement (4 iterations)
            var lo = prev_ray;
            var hi = ray;
            for (var j = 0; j < 4; j++) {
                let mid  = (lo + hi) * 0.5;
                let mss  = view_to_screen(mid);
                let mid_d = screen_depth(mss.xy, dim);
                if mss.z > mid_d {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            hit_uv = view_to_screen((lo + hi) * 0.5).xy;
            hit = true;
            break;
        }
        prev_ray   = ray;
        step_size  *= 1.05;  // exponential step growth for long rays
        ray       += refl_dir * step_size;
    }

    if !hit { return vec4<f32>(0.0); }

    // Confidence: fade by roughness, edge proximity, and ray length
    let roughness_fade = 1.0 - smoothstep(0.3, 0.6, roughness);
    let ef             = edge_fade(hit_uv);
    let confidence     = roughness_fade * ef;
    if confidence < 0.001 { return vec4<f32>(0.0); }

    let reflected_col = textureSample(scene_color, linear_samp, hit_uv).rgb;
    return vec4<f32>(reflected_col * confidence, confidence);
}
