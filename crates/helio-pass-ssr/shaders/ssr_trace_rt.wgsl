// ssr_trace_rt.wgsl — Hybrid SSR with hardware ray queries.
//
// Fully self-contained — no prelude dependency, because `enable wgpu_ray_query;`
// must appear before any other declaration and the prelude-based concatenation
// would place it after the prelude's struct definitions.

enable wgpu_ray_query;

// ── Camera (mirrors helio_core GpuCameraUniforms) ──────────────────────────
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

// ── Screen-space helpers ───────────────────────────────────────────────────
fn helio_uv_to_ndc(uv: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(uv.x * 2.0 - 1.0, -(uv.y * 2.0 - 1.0));
}
fn helio_ndc_to_uv(ndc: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
}
fn helio_view_depth(device_depth01: f32, near: f32, far: f32) -> f32 {
    return near * far / (far + device_depth01 * (near - far));
}
fn helio_world_from_depth(view_proj_inv: mat4x4<f32>, uv: vec2<f32>, depth01: f32) -> vec3<f32> {
    let ndc = helio_uv_to_ndc(uv);
    let clip = vec4<f32>(ndc, depth01, 1.0);
    let world = view_proj_inv * clip;
    return world.xyz / world.w;
}
fn helio_gbuffer_normal(encoded: vec3<f32>) -> vec3<f32> {
    return normalize(encoded * 2.0 - 1.0);
}

@group(0) @binding(0) var<uniform> camera:      Camera;

@group(1) @binding(0) var gbuf_normal:          texture_2d<f32>;
@group(1) @binding(1) var gbuf_orm:             texture_2d<f32>;
@group(1) @binding(2) var gbuf_depth:           texture_depth_2d;
@group(1) @binding(3) var scene_color:          texture_2d<f32>;
@group(1) @binding(4) var hiz_min:              texture_2d<f32>;
@group(1) @binding(5) var linear_sampler:       sampler;
@group(1) @binding(6) var ssr_output:           texture_storage_2d<rgba16float, write>;

@group(2) @binding(0) var acc_struct:           acceleration_structure;
@group(2) @binding(1) var rc_cascades:          texture_2d<f32>;

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

fn level_size(level: i32) -> vec2<f32> {
    return vec2<f32>(max(vec2<u32>(1u), textureDimensions(hiz_min) >> vec2<u32>(u32(level))));
}

fn cell_of(uv: vec2<f32>, size: vec2<f32>) -> vec2<f32> {
    return floor(uv * size);
}

fn min_depth(cell: vec2<f32>, level: i32) -> f32 {
    return textureLoad(hiz_min, vec2<i32>(cell), level).r;
}

fn at_depth(o: vec3<f32>, d: vec3<f32>, z: f32) -> vec3<f32> {
    return o + d * ((z - o.z) / d.z);
}

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

fn ray_query_hit(world_pos: vec3<f32>, R: vec3<f32>) -> bool {
    let origin = world_pos + R * 0.001;
    var rq: ray_query;
    rayQueryInitialize(&rq, acc_struct,
        RayDesc(0x01u, 0xFFu, 0.001, MAX_RAY_DIST, origin, R));
    rayQueryProceed(&rq);
    return rayQueryGetCommittedIntersection(&rq).kind != RAY_QUERY_INTERSECTION_NONE;
}

fn ray_query_hit_world_pos(world_pos: vec3<f32>, R: vec3<f32>) -> vec3<f32> {
    let origin = world_pos + R * 0.001;
    var rq: ray_query;
    rayQueryInitialize(&rq, acc_struct,
        RayDesc(0x01u, 0xFFu, 0.001, MAX_RAY_DIST, origin, R));
    rayQueryProceed(&rq);
    if rayQueryGetCommittedIntersection(&rq).kind != RAY_QUERY_INTERSECTION_NONE {
        let t = rayQueryGetCommittedIntersection(&rq).t;
        return origin + R * t;
    }
    return world_pos;
}

fn sample_rc_reflection(world_pos: vec3<f32>, R: vec3<f32>, roughness: f32) -> vec3<f32> {
    if roughness < 0.6 { return vec3<f32>(0.0); }
    let rc_dims = textureDimensions(rc_cascades);
    if rc_dims.x < 2u || rc_dims.y < 2u { return vec3<f32>(0.0); }

    let f = R / (abs(R.x) + abs(R.y) + abs(R.z));
    let oct_uv = select(
        vec2<f32>(f.z, f.x) * 0.5 + 0.5,
        vec2<f32>(1.0 - abs(f.z), 1.0 - abs(f.x)) * 0.5,
        f.y >= 0.0,
    );
    let sx = f32(rc_dims.x);
    let sy = f32(rc_dims.y);
    let px = clamp(i32(oct_uv.x * sx), 0, i32(rc_dims.x) - 3);
    let py = clamp(i32(oct_uv.y * sy), 0, i32(rc_dims.y) - 3);

    var irradiance = vec3<f32>(0.0);
    for (var dy = 0u; dy < 2u; dy++) {
        for (var dx = 0u; dx < 2u; dx++) {
            let s = textureLoad(rc_cascades, vec2<i32>(px + i32(dx), py + i32(dy)), 0);
            irradiance += s.rgb;
        }
    }
    return irradiance * 0.25;
}

@compute @workgroup_size(8, 8)
fn cs_rt(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(ssr_output);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let px = vec2<i32>(gid.xy);
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
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
    let R2 = reflect(-V, N);
    if dot(R2, N) <= 0.0 {
        textureStore(ssr_output, px, vec4<f32>(0.0));
        return;
    }

    // ── Build ray ───────────────────────────────────────────────────────────
    let near = camera.position_near.w;
    var start_view = (camera.view * vec4<f32>(world_pos, 1.0)).xyz;
    let dir_view = normalize((camera.view * vec4<f32>(R2, 0.0)).xyz);
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
    var d = p1 - p0;

    if abs(d.x) < 1e-7 && abs(d.y) < 1e-7 {
        textureStore(ssr_output, px, vec4<f32>(0.0));
        return;
    }
    d.x = select(d.x, 1e-7, abs(d.x) < 1e-7);
    d.y = select(d.y, 1e-7, abs(d.y) < 1e-7);

    let cross_step = vec2<f32>(select(0.0, 1.0, d.x >= 0.0), select(0.0, 1.0, d.y >= 0.0));
    let cross_offset = (cross_step * 2.0 - 1.0) * 1e-6;

    // ── Hi-Z traversal ──────────────────────────────────────────────────────
    var level = START_LEVEL;
    let max_level = min(MAX_LEVEL, i32(textureNumLevels(hiz_min)) - 1);
    var tr = p0;
    {
        let size = level_size(level);
        tr = exit_cell(p0, d, cell_of(p0.xy, size), size, cross_step, cross_offset);
    }

    var iter = 0u;
    var hiz_hit = false;

    while level >= 0 && iter < MAX_ITER {
        iter += 1u;
        if any(tr.xy < vec2<f32>(0.0)) || any(tr.xy > vec2<f32>(1.0)) { break; }
        if tr.z > 1.0 { break; }

        let size = level_size(level);
        let old_cell = cell_of(tr.xy, size);
        let tile_min = min_depth(old_cell, level);

        var next = tr;
        let in_front = tr.z < tile_min;
        if in_front && d.z > 0.0 {
            next = at_depth(p0, d, tile_min);
        }

        let new_cell = cell_of(next.xy, size);
        let skip_tile = in_front && d.z <= 0.0;

        if skip_tile || any(new_cell != old_cell) {
            next = exit_cell(tr, d, old_cell, size, cross_step, cross_offset);
            level = min(max_level, level + 1);
        } else {
            level -= 1;
            if level < 0 { hiz_hit = true; }
        }
        tr = next;
    }

    // ── Hybrid blend ────────────────────────────────────────────────────────
    var final_color = vec3<f32>(0.0);
    var final_confidence = 0.0;
    let rt_hit = ray_query_hit(world_pos, R2);

    if hiz_hit {
        let hit_uv = tr.xy;
        let r_depth = linearize_depth(tr.z);
        let scene_depth = linearize_depth(
            textureLoad(gbuf_depth, vec2<i32>(hit_uv * vec2<f32>(dims)), 0)
        );

        if r_depth <= scene_depth * (1.0 + THICKNESS) {
            let n_hit = helio_gbuffer_normal(
                textureLoad(gbuf_normal, vec2<i32>(hit_uv * vec2<f32>(dims)), 0).xyz
            );
            let arriving = -dot(R2, n_hit);
            let backface_fade = smoothstep(-0.15, 0.15, arriving);
            let border = min(min(hit_uv.x, 1.0 - hit_uv.x), min(hit_uv.y, 1.0 - hit_uv.y));
            let edge_fade = smoothstep(0.0, 0.1, border);
            let facing_fade = 1.0 - smoothstep(0.26, 0.5, dot(R2, V));
            let travelled = length(hit_uv - p0.xy) / max(length(d.xy), 1e-6);
            let dist_fade = 1.0 - smoothstep(FADE_START, 1.0, travelled);
            let confidence = clamp(
                backface_fade * edge_fade * facing_fade * dist_fade * roughness_fade,
                0.0, 1.0,
            );

            let ssr_col = textureSampleLevel(scene_color, linear_sampler, hit_uv, 0.0).rgb;

            if confidence >= 0.5 || !rt_hit {
                final_color = ssr_col;
                final_confidence = confidence;
            } else {
                let rt_hit_pos = ray_query_hit_world_pos(world_pos, R2);
                let rt_clip = camera.view_proj * vec4<f32>(rt_hit_pos, 1.0);
                let rt_uv = helio_ndc_to_uv(rt_clip.xy / rt_clip.w);
                if all(rt_uv >= vec2<f32>(0.0)) && all(rt_uv <= vec2<f32>(1.0)) {
                    let rt_col = textureSampleLevel(scene_color, linear_sampler, rt_uv, 0.0).rgb;
                    final_color = mix(ssr_col, rt_col, 1.0 - confidence);
                    final_confidence = max(confidence, 0.5);
                } else {
                    let rc_col = sample_rc_reflection(world_pos, R2, roughness);
                    final_color = mix(ssr_col, rc_col, 1.0 - confidence);
                    final_confidence = confidence;
                }
            }
        } else if rt_hit {
            let rt_hit_pos = ray_query_hit_world_pos(world_pos, R2);
            let rt_clip = camera.view_proj * vec4<f32>(rt_hit_pos, 1.0);
            let rt_uv = helio_ndc_to_uv(rt_clip.xy / rt_clip.w);
            if all(rt_uv >= vec2<f32>(0.0)) && all(rt_uv <= vec2<f32>(1.0)) {
                final_color = textureSampleLevel(scene_color, linear_sampler, rt_uv, 0.0).rgb;
                final_confidence = 0.6;
            } else {
                final_color = sample_rc_reflection(world_pos, R2, roughness);
                final_confidence = roughness_fade * 0.3;
            }
        } else {
            textureStore(ssr_output, px, vec4<f32>(0.0));
            return;
        }
    } else if rt_hit {
        let rt_hit_pos = ray_query_hit_world_pos(world_pos, R2);
        let rt_clip = camera.view_proj * vec4<f32>(rt_hit_pos, 1.0);
        let rt_uv = helio_ndc_to_uv(rt_clip.xy / rt_clip.w);
        if all(rt_uv >= vec2<f32>(0.0)) && all(rt_uv <= vec2<f32>(1.0)) {
            final_color = textureSampleLevel(scene_color, linear_sampler, rt_uv, 0.0).rgb;
            final_confidence = 0.5;
        } else {
            final_color = sample_rc_reflection(world_pos, R2, roughness);
            final_confidence = roughness_fade * 0.3;
        }
    } else {
        if roughness > 0.6 {
            final_color = sample_rc_reflection(world_pos, R2, roughness);
            final_confidence = roughness_fade * 0.2;
        } else {
            textureStore(ssr_output, px, vec4<f32>(0.0));
            return;
        }
    }

    textureStore(ssr_output, px, vec4<f32>(final_color, final_confidence));
}
