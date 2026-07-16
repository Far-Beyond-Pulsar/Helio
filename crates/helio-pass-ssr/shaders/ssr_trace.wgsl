// ssr_trace.wgsl — Screen-space reflections, Hi-Z ray march.
//
// Marches the `hiz_min` pyramid built by HiZBuildPass: at each step the nearest
// depth in the current tile answers "could the ray hit anything here?". If the
// ray is in front of that, the whole tile is empty and it skips to the tile
// boundary and coarsens; otherwise it refines. Empty space costs a handful of
// taps instead of one tap per pixel crossed.
//
// The pyramid MUST be min-reduced. The shared `hiz` resource is max-reduced for
// occlusion culling (conservative farthest); marching that makes rays tunnel
// straight through geometry.
//
// Key simplification: NDC depth is *linear in screen space* (it is the projective
// z/w), so the reflection ray is a straight line in (uv, depth01) space. Nothing
// in the loop needs perspective correction or linearization — the march compares
// raw depth-buffer values and only converts to view depth once, at the end.
//
// One deterministic mirror ray per pixel — no roughness sampling, so the result
// is stable frame to frame and needs no temporal pass or denoiser behind it.
//
// Writes Rgba16Float at full resolution: RGB = colour, A = confidence.
//!use helio_prelude

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
// Surface thickness, as a fraction of view depth. Small on purpose: the march
// advances the ray onto the depth plane, so a genuine hit lands with
// ray_depth ~= scene_depth and needs only enough slack to absorb texel
// quantization. Anything larger extrudes objects backward along the view ray —
// the depth buffer has no back faces, so a loose slab makes every ray passing
// *behind* a sphere report a hit on it, smearing it into a cylinder.
const THICKNESS:     f32 = 0.02;
const NORMAL_OFFSET: f32 = 0.002;  // ray origin nudge, relative to view depth
const FADE_START:    f32 = 0.6;    // ray-length fraction where confidence drops

fn linearize_depth(d_01: f32) -> f32 {
    return helio_view_depth(d_01, camera.position_near.w, camera.forward_far.w);
}

fn level_size(level: i32) -> vec2<f32> {
    return vec2<f32>(max(vec2<u32>(1u), textureDimensions(hiz_min) >> vec2<u32>(u32(level))));
}

fn cell_of(uv: vec2<f32>, size: vec2<f32>) -> vec2<f32> {
    return floor(uv * size);
}

/// Nearest scene depth within the tile — the whole point of the min pyramid.
fn min_depth(cell: vec2<f32>, level: i32) -> f32 {
    return textureLoad(hiz_min, vec2<i32>(cell), level).r;
}

fn at_depth(o: vec3<f32>, d: vec3<f32>, z: f32) -> vec3<f32> {
    return o + d * ((z - o.z) / d.z);
}

/// Advance to where the ray leaves `cell`, plus an epsilon so it lands strictly
/// inside the next cell (otherwise floor() snaps back and the march stalls).
fn exit_cell(
    o: vec3<f32>,
    d: vec3<f32>,
    cell: vec2<f32>,
    size: vec2<f32>,
    cross_step: vec2<f32>,
    cross_offset: vec2<f32>,
) -> vec3<f32> {
    let boundary = (cell + cross_step) / size + cross_offset;
    let delta = (boundary - o.xy) / d.xy;
    return o + d * min(delta.x, delta.y);
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

    // Matches the composite in deferred_lighting.wgsl; skip what it won't blend.
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

    // Lift the origin off its own surface along the normal, scaled by view depth.
    // Without this a grazing ray sits within float noise of the surface it came
    // from and the first tile test reports an immediate self-hit.
    let n_view = (camera.view * vec4<f32>(N, 0.0)).xyz;
    start_view += n_view * (-start_view.z * NORMAL_OFFSET);

    // View space is RH (-z forward), so the near plane sits at z = -near.
    var ray_len = MAX_RAY_DIST;
    if start_view.z + dir_view.z * ray_len > -near {
        ray_len = (-near - start_view.z) / dir_view.z;
    }
    if ray_len <= 0.0 {
        textureStore(ssr_output, px, vec4<f32>(0.0));
        return;
    }
    let end_view = start_view + dir_view * ray_len;

    // ── Project to the (uv, depth01) space the march lives in ───────────────
    let clip0 = camera.proj * vec4<f32>(start_view, 1.0);
    let clip1 = camera.proj * vec4<f32>(end_view, 1.0);
    let p0 = vec3<f32>(helio_ndc_to_uv(clip0.xy / clip0.w), clip0.z / clip0.w);
    let p1 = vec3<f32>(helio_ndc_to_uv(clip1.xy / clip1.w), clip1.z / clip1.w);
    var d = p1 - p0;

    // A ray pointing straight at/away from the eye has no screen-space extent, so
    // exit_cell would divide by zero and the march has nothing to walk.
    if abs(d.x) < 1e-7 && abs(d.y) < 1e-7 {
        textureStore(ssr_output, px, vec4<f32>(0.0));
        return;
    }
    d.x = select(d.x, 1e-7, abs(d.x) < 1e-7);
    d.y = select(d.y, 1e-7, abs(d.y) < 1e-7);

    // Which cell edge the ray exits through, and a nudge to land past it.
    let cross_step = vec2<f32>(select(0.0, 1.0, d.x >= 0.0), select(0.0, 1.0, d.y >= 0.0));
    let cross_offset = (cross_step * 2.0 - 1.0) * 1e-6;

    // ── Hi-Z traversal ──────────────────────────────────────────────────────
    var level = START_LEVEL;
    let max_level = min(MAX_LEVEL, i32(textureNumLevels(hiz_min)) - 1);

    // No dither. The traversal finds the crossing exactly rather than sampling at
    // a fixed stride, so there is no stepping pattern to break up — and with no
    // temporal pass downstream, dithering would only add noise nothing resolves.
    var ray = p0;
    {
        // Step out of the origin cell before the first test, or the ray reports a
        // hit against the surface it started on.
        let size = level_size(level);
        ray = exit_cell(p0, d, cell_of(p0.xy, size), size, cross_step, cross_offset);
    }

    var iter = 0u;
    var hit = false;

    while level >= 0 && iter < MAX_ITER {
        iter += 1u;

        if any(ray.xy < vec2<f32>(0.0)) || any(ray.xy > vec2<f32>(1.0)) { break; }
        if ray.z > 1.0 { break; }

        let size = level_size(level);
        let old_cell = cell_of(ray.xy, size);
        let tile_min = min_depth(old_cell, level);

        var next = ray;
        // In front of everything in this tile? Then nothing here can be hit.
        let in_front = ray.z < tile_min;
        if in_front && d.z > 0.0 {
            // Depth increases along the ray, so it can only reach the tile's
            // nearest surface at this depth. Jump straight there.
            next = at_depth(p0, d, tile_min);
        }

        let new_cell = cell_of(next.xy, size);
        // A ray whose depth is flat or decreasing (d.z <= 0, e.g. reflecting back
        // toward the eye) can never meet tile_min from in front, so the depth-plane
        // jump above is invalid for it — skip the tile wholesale instead.
        let skip_tile = in_front && d.z <= 0.0;

        if skip_tile || any(new_cell != old_cell) {
            next = exit_cell(ray, d, old_cell, size, cross_step, cross_offset);
            level = min(max_level, level + 1);
        } else {
            // Stayed in the tile and is not in front of it: refine. Falling below
            // level 0 means the ray is behind an actual depth texel — a hit.
            level -= 1;
            if level < 0 { hit = true; }
        }
        ray = next;
    }

    if !hit {
        textureStore(ssr_output, px, vec4<f32>(0.0));
        return;
    }

    let hit_uv = ray.xy;
    if any(hit_uv < vec2<f32>(0.0)) || any(hit_uv > vec2<f32>(1.0)) {
        textureStore(ssr_output, px, vec4<f32>(0.0));
        return;
    }

    // ── Thickness validation ────────────────────────────────────────────────
    // The depth buffer records front faces only, so a ray that dives behind a
    // thin pillar "crosses" it and would otherwise reflect it as if solid. Scaled
    // by depth: a fixed world-space slab is far too thin up close, too thick far.
    let ray_depth = linearize_depth(ray.z);
    let scene_depth = linearize_depth(
        textureLoad(gbuf_depth, vec2<i32>(hit_uv * vec2<f32>(dims)), 0)
    );
    if ray_depth > scene_depth * (1.0 + THICKNESS) {
        textureStore(ssr_output, px, vec4<f32>(0.0));
        return;
    }

    // ── Confidence ──────────────────────────────────────────────────────────
    // Per-axis border fade on the *hit*: a radial fade from the source pixel has
    // nothing to do with where the ray actually left the screen.
    let border = min(min(hit_uv.x, 1.0 - hit_uv.x), min(hit_uv.y, 1.0 - hit_uv.y));
    let edge_fade = smoothstep(0.0, 0.1, border);

    // Rays aimed back at the camera have no on-screen data behind them.
    let facing_fade = 1.0 - smoothstep(0.25, 0.6, dot(R, -V));

    // Fade as the ray runs out of budget, so hits don't pop at MAX_RAY_DIST.
    let travelled = length(hit_uv - p0.xy) / max(length(d.xy), 1e-6);
    let dist_fade = 1.0 - smoothstep(FADE_START, 1.0, travelled);

    let confidence = clamp(edge_fade * facing_fade * dist_fade * roughness_fade, 0.0, 1.0);

    // scene_color has no mip chain (pre_aa is MatchSurface, 1 mip), so roughness
    // blur comes from the denoise pass rather than a mip bias here.
    let reflection = textureSampleLevel(scene_color, linear_sampler, hit_uv, 0.0).rgb;

    textureStore(ssr_output, px, vec4<f32>(reflection, confidence));
}
