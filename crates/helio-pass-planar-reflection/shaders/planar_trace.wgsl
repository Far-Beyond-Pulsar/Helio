// planar_trace.wgsl — Planar reflections via screen-space trace.
//
// For each pixel on a planar reflector surface, the view ray is reflected
// across the plane and the scene colour is gathered by projecting the
// intersection back to screen UV and sampling the fully-lit pre_aa.
//
// Surfaces are classified by comparing their world normal against the
// plane normal (cosine threshold). Bounding-box culling limits the
// reflector to a world-space rectangle on the plane.
//
// Writes Rgba16Float: RGB = reflected colour, A = confidence.
//!use helio_prelude

struct PlanarGlobals {
    plane_pos:    vec4<f32>,
    plane_normal: vec4<f32>,
    half_extents: vec4<f32>,
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

const MAX_STEPS: u32 = 64u;
const MAX_RAY:   f32 = 200.0;
const FADE_UV:   f32 = 0.5;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(planar_output);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let px = vec2<i32>(gid.xy);
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);

    let depth_01 = textureLoad(gbuf_depth, px, 0);
    if depth_01 >= 1.0 {
        textureStore(planar_output, px, vec4<f32>(0.0)); return;
    }

    let N = helio_gbuffer_normal(textureLoad(gbuf_normal, px, 0).xyz);
    // Exit if surface doesn't face the plane.
    if dot(N, planar.plane_normal.xyz) < planar.cos_angle_threshold {
        textureStore(planar_output, px, vec4<f32>(0.0)); return;
    }

    let world_pos = helio_world_from_depth(camera.view_proj_inv, uv, depth_01);

    // Bounding-box test.
    if planar.half_extents.x > 0.0 && planar.half_extents.y > 0.0 {
        let to_surface = world_pos - planar.plane_pos.xyz;
        let up = vec3<f32>(0.0, 1.0, 0.0);
        let tangent = normalize(select(
            cross(planar.plane_normal.xyz, up),
            cross(planar.plane_normal.xyz, vec3<f32>(0.0, 0.0, 1.0)),
            abs(planar.plane_normal.y) > 0.99,
        ));
        let bitangent = cross(planar.plane_normal.xyz, tangent);
        if abs(dot(to_surface, tangent)) > planar.half_extents.x ||
           abs(dot(to_surface, bitangent)) > planar.half_extents.y {
            textureStore(planar_output, px, vec4<f32>(0.0)); return;
        }
    }

    // Reflect view ray across the plane.
    let V = normalize(camera.position_near.xyz - world_pos);
    let R = reflect(-V, planar.plane_normal.xyz);
    if dot(R, planar.plane_normal.xyz) <= 0.0 {
        textureStore(planar_output, px, vec4<f32>(0.0)); return;
    }

    // Project reflected ray into (uv, depth01) space and march.
    let near = camera.position_near.w;
    let start_view = (camera.view * vec4<f32>(world_pos, 1.0)).xyz;
    let dir_view = normalize((camera.view * vec4<f32>(R, 0.0)).xyz);

    var t_max = MAX_RAY;
    if start_view.z + dir_view.z * t_max > -near {
        t_max = (-near - start_view.z) / dir_view.z;
    }
    if t_max <= 0.0 {
        textureStore(planar_output, px, vec4<f32>(0.0)); return;
    }

    let end_view = start_view + dir_view * t_max;
    let c0 = camera.proj * vec4<f32>(start_view, 1.0);
    let c1 = camera.proj * vec4<f32>(end_view, 1.0);
    let p0 = vec3<f32>(helio_ndc_to_uv(c0.xy / c0.w), c0.z / c0.w);
    let p1 = vec3<f32>(helio_ndc_to_uv(c1.xy / c1.w), c1.z / c1.w);
    let d = p1 - p0;

    if abs(d.x) < 1e-7 && abs(d.y) < 1e-7 {
        textureStore(planar_output, px, vec4<f32>(0.0)); return;
    }

    // Linear march in UV space.
    var su = p0.x;
    var sv = p0.y;
    var sd = p0.z;
    let du = select(d.x / f32(MAX_STEPS), 1e-7, abs(d.x) < 1e-7);
    let dv = select(d.y / f32(MAX_STEPS), 1e-7, abs(d.y) < 1e-7);
    let dd = d.z / f32(MAX_STEPS);

    let fdims = vec2<f32>(dims);
    var hit = false;
    var hu = 0.0;
    var hv = 0.0;

    for (var i = 0u; i < MAX_STEPS; i++) {
        su += du;
        sv += dv;
        sd += dd;
        // Bounds check per component.
        if su < 0.0 || su > 1.0 { break; }
        if sv < 0.0 || sv > 1.0 { break; }
        if sd > 1.0 { break; }

        let scene_d = textureLoad(gbuf_depth, vec2<i32>(i32(su * fdims.x), i32(sv * fdims.y)), 0);
        if sd <= scene_d {
            hu = su;
            hv = sv;
            hit = true;
            break;
        }
    }

    if !hit {
        textureStore(planar_output, px, vec4<f32>(0.0)); return;
    }

    // Confidence: edge fade + distance fade.
    let border = min(min(hu, 1.0 - hu), min(hv, 1.0 - hv));
    let edge = smoothstep(0.0, 0.1, border);
    let travelled = length(vec2<f32>(hu - uv.x, hv - uv.y)) /
                    max(length(d.xy), 1e-6);
    let dist = 1.0 - smoothstep(FADE_UV, 1.0, travelled);
    let confidence = clamp(edge * dist, 0.0, 1.0);
    if confidence <= 0.0 {
        textureStore(planar_output, px, vec4<f32>(0.0)); return;
    }

    let reflection = textureSampleLevel(
        scene_color, linear_sampler, vec2<f32>(hu, hv), 0.0
    ).rgb;
    textureStore(planar_output, px, vec4<f32>(reflection, confidence));
}
