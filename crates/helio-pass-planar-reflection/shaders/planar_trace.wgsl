// planar_trace.wgsl — Planar reflections via screen-space trace.
//
// For each pixel that sits on a planar reflector surface, the view ray
// is reflected across the plane and the reflected colour is gathered by
// projecting the plane-intersection point back into screen space.
//
// Surfaces are classified as planar reflectors by comparing their
// world-space normal against the plane normal (within a configurable
// angular tolerance).  This avoids needing a separate material flag per
// pixel — a reflective floor, window, or polished tabletop works as
// long as it is flat enough.
//
// Writes Rgba16Float at full resolution: RGB = reflected colour (from
// the fully-lit pre_aa), A = confidence (0 = no valid reflection).
//!use helio_prelude

struct PlanarGlobals {
    // Plane definition (world space).
    plane_pos:    vec4<f32>,   // xyz = position on the plane, w unused
    plane_normal: vec4<f32>,   // xyz = unit normal, w unused
    // Half-extents of the reflector bounding box (world space).
    // A zero or negative value means "unbounded" — treat the whole
    // plane as a reflector regardless of extent.
    half_extents: vec4<f32>,   // x = half-width along plane tangent,
                               // y = half-height along plane bitangent,
                               // z = depth threshold for geometry on the plane,
                               // w unused
    // Normal tolerance: surfaces within this angular range are treated
    // as part of the plane.  The cosine threshold is passed directly:
    // dot(pixel_normal, plane_normal) >= cos_angle_threshold.
    cos_angle_threshold: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<uniform> camera:       Camera;
@group(0) @binding(1) var<uniform> planar:       PlanarGlobals;
@group(1) @binding(0) var gbuf_normal:           texture_2d<f32>;
@group(1) @binding(1) var gbuf_depth:            texture_depth_2d;
@group(1) @binding(2) var scene_color:           texture_2d<f32>;
@group(1) @binding(3) var linear_sampler:        sampler;
@group(1) @binding(4) var planar_output:         texture_storage_2d<rgba16float, write>;

const MAX_RAY_DIST:  f32 = 200.0;
const FADE_START:    f32 = 0.5;

/// Trace the reflected view ray against the scene depth in screen space.
/// Returns the reflected UV and depth, or false if no valid hit.
fn trace_reflected_ray(
    world_pos: vec3<f32>,
    R: vec3<f32>,
    start_uv: vec2<f32>,
) -> bool {
    // Early out: reflection points behind the camera.
    let view_dir = (camera.view * vec4<f32>(R, 0.0)).xyz;
    if view_dir.z >= 0.0 { return false; }

    // March in screen space: step along the reflected ray in the
    // depth buffer.  This is a simplified Hi-Z-like march: take fixed-
    // size steps in UV space and check if the ray depth crossed the
    // scene depth at each sample.
    let near = camera.position_near.w;
    let start_view = (camera.view * vec4<f32>(world_pos, 1.0)).xyz;
    var ray_len = MAX_RAY_DIST;
    if start_view.z + view_dir.z * ray_len > -near {
        ray_len = (-near - start_view.z) / view_dir.z;
    }
    if ray_len <= 0.0 { return false; }

    let end_view = start_view + view_dir * ray_len;
    let clip0 = camera.proj * vec4<f32>(start_view, 1.0);
    let clip1 = camera.proj * vec4<f32>(end_view, 1.0);
    let p0 = vec3<f32>(helio_ndc_to_uv(clip0.xy / clip0.w), clip0.z / clip0.w);
    let p1 = vec3<f32>(helio_ndc_to_uv(clip1.xy / clip1.w), clip1.z / clip1.w);
    var d = p1 - p0;

    if abs(d.x) < 1e-7 && abs(d.y) < 1e-7 { return false; }
    d.x = select(d.x, 1e-7, abs(d.x) < 1e-7);
    d.y = select(d.y, 1e-7, abs(d.y) < 1e-7);

    // Simple linear march: step along the ray checking depth.
    let steps = 32u;
    let step_uv = d.xy / f32(steps);
    let step_d = d.z / f32(steps);
    var ray = p0;
    for (var i = 0u; i < steps; i++) {
        ray.xy += step_uv;
        ray.z += step_d;
        if any(ray.xy < vec2<f32>(0.0)) || any(ray.xy > vec2<f32>(1.0)) { break; }
        if ray.z > 1.0 { break; }

        let dims = textureDimensions(scene_color);
        let scene_d = textureLoad(gbuf_depth, vec2<i32>(ray.xy * vec2<f32>(dims)), 0);
        if ray.z <= scene_d {
            return true;
        }
    }
    return false;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(planar_output);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let px = vec2<i32>(gid.xy);
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);

    // ── G-buffer reads ──────────────────────────────────────────────────
    let depth_01 = textureLoad(gbuf_depth, px, 0);
    if depth_01 >= 1.0 {
        textureStore(planar_output, px, vec4<f32>(0.0));
        return;
    }
    let N = helio_gbuffer_normal(textureLoad(gbuf_normal, px, 0).xyz);

    // Surface must face the plane to be a planar reflector.
    let nd = dot(N, planar.plane_normal.xyz);
    if nd < planar.cos_angle_threshold {
        textureStore(planar_output, px, vec4<f32>(0.0));
        return;
    }

    // ── Bounding box test ───────────────────────────────────────────────
    if planar.half_extents.x > 0.0 && planar.half_extents.y > 0.0 {
        let world_pos = helio_world_from_depth(camera.view_proj_inv, uv, depth_01);
        let to_surface = world_pos - planar.plane_pos.xyz;
        // Build a local 2D frame on the plane.
        let tangent = normalize(select(
            cross(planar.plane_normal.xyz, vec3<f32>(0.0, 1.0, 0.0)),
            cross(planar.plane_normal.xyz, vec3<f32>(0.0, 0.0, 1.0)),
            abs(planar.plane_normal.x) > 0.99,
        ));
        let bitangent = cross(planar.plane_normal.xyz, tangent);
        let proj_x = abs(dot(to_surface, tangent));
        let proj_y = abs(dot(to_surface, bitangent));
        if proj_x > planar.half_extents.x || proj_y > planar.half_extents.y {
            textureStore(planar_output, px, vec4<f32>(0.0));
            return;
        }
    }

    // ── Reflected ray ───────────────────────────────────────────────────
    let world_pos = helio_world_from_depth(camera.view_proj_inv, uv, depth_01);
    let V = normalize(camera.position_near.xyz - world_pos);
    let R = reflect(-V, planar.plane_normal.xyz);
    if dot(R, planar.plane_normal.xyz) <= 0.0 {
        textureStore(planar_output, px, vec4<f32>(0.0));
        return;
    }

    // ── Trace ────────────────────────────────────────────────────────────
    if !trace_reflected_ray(world_pos, R, uv) {
        textureStore(planar_output, px, vec4<f32>(0.0));
        return;
    }

    // Re-acquire the ray endpoint from the trace.
    // For simplicity we re-run the march logic to get the final hit_uv.
    // In practice the trace function would return it; here we recompute.
    let start_view = (camera.view * vec4<f32>(world_pos, 1.0)).xyz;
    let view_dir = (camera.view * vec4<f32>(R, 0.0)).xyz;
    let near = camera.position_near.w;
    var ray_len = MAX_RAY_DIST;
    if start_view.z + view_dir.z * ray_len > -near {
        ray_len = (-near - start_view.z) / view_dir.z;
    }
    if ray_len <= 0.0 {
        textureStore(planar_output, px, vec4<f32>(0.0));
        return;
    }
    let end_view = start_view + view_dir * ray_len;
    let clip0 = camera.proj * vec4<f32>(start_view, 1.0);
    let clip1 = camera.proj * vec4<f32>(end_view, 1.0);
    let p0 = vec3<f32>(helio_ndc_to_uv(clip0.xy / clip0.w), clip0.z / clip0.w);
    let p1 = vec3<f32>(helio_ndc_to_uv(clip1.xy / clip1.w), clip1.z / clip1.w);
    var d2 = p1 - p0;
    d2.x = select(d2.x, 1e-7, abs(d2.x) < 1e-7);
    d2.y = select(d2.y, 1e-7, abs(d2.y) < 1e-7);

    let steps = 32u;
    let step_uv = d2.xy / f32(steps);
    let step_d = d2.z / f32(steps);
    var hit_uv = p0.xy;
    var hit_depth = p0.z;
    for (var i = 0u; i < steps; i++) {
        hit_uv += step_uv;
        hit_depth += step_d;
        if any(hit_uv < vec2<f32>(0.0)) || any(hit_uv > vec2<f32>(1.0)) { break; }
        if hit_depth > 1.0 { break; }
        let dims2 = textureDimensions(scene_color);
        let scene_d = textureLoad(gbuf_depth, vec2<i32>(hit_uv * vec2<f32>(dims2)), 0);
        if hit_depth <= scene_d { break; }
    }

    // ── Validity and confidence ─────────────────────────────────────────
    let border = min(min(hit_uv.x, 1.0 - hit_uv.x), min(hit_uv.y, 1.0 - hit_uv.y));
    let edge_fade = smoothstep(0.0, 0.1, border);

    let travelled = length(hit_uv - uv) / max(length(d2.xy), 1e-6);
    let dist_fade = 1.0 - smoothstep(FADE_START, 1.0, travelled);

    let confidence = clamp(edge_fade * dist_fade, 0.0, 1.0);
    if confidence <= 0.0 {
        textureStore(planar_output, px, vec4<f32>(0.0));
        return;
    }

    let reflection = textureSampleLevel(scene_color, linear_sampler, hit_uv, 0.0).rgb;
    textureStore(planar_output, px, vec4<f32>(reflection, confidence));
}
