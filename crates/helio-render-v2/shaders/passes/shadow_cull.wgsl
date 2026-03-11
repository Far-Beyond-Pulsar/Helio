//! Shadow culling compute pass — GPU-driven per-draw cull for shadow atlas faces.
//!
//! Dispatch: [ceil(draw_count / 64), light_count * 6, 1]
//! Each thread handles one (draw_call, light_face) combination.
//!
//! Output layout in shadow_indirect:
//!   slot    = (light_face * draw_count) + draw_idx
//!   index   = light_face * 6 + face  (light_face = light_idx*6 + face)
//!
//!   instance_count = 0  for culled draws (GPU skips them for free)
//!   otherwise filled with the full draw params

/// GPU draw call data. Must match `GpuDrawCall` in mesh.rs.
struct GpuDrawCall {
    slot:          u32,
    first_index:   u32,
    base_vertex:   i32,
    index_count:   u32,
    bounds_center: vec3<f32>,
    bounds_radius: f32,
}

/// Per-light data for shadow culling.
/// Must match `ShadowCullGpuLight` in passes/shadow.rs.
/// Layout (32 bytes, no implicit padding with these types):
///   [0..12]  position   (vec3f → 12 bytes, align 16 → next field at 12)
///   [12..16] light_type (u32,   align 4   → fits at 12)
///   [16..28] direction  (vec3f → 12 bytes, align 16 → at 16)
///   [28..32] range      (f32,   align 4   → at 28)
struct ShadowCullLight {
    position:   vec3<f32>,
    light_type: u32,     // 0 = point, 1 = directional, 2 = spot
    direction:  vec3<f32>,
    range:      f32,
}

/// DrawIndexedIndirect command layout (20 bytes).
struct DrawCmd {
    index_count:    u32,
    instance_count: u32,
    first_index:    u32,
    base_vertex:    i32,
    first_instance: u32,
}

/// Uniform params for this dispatch.
struct ShadowCullParams {
    draw_count:          u32,
    light_count:         u32,
    shadow_max_distance: f32,
    _pad:                u32,
}

@group(0) @binding(0) var<storage, read>       draw_calls:      array<GpuDrawCall>;
@group(0) @binding(1) var<storage, read>       cull_lights:     array<ShadowCullLight>;
@group(0) @binding(2) var<storage, read_write> shadow_indirect: array<DrawCmd>;
@group(0) @binding(3) var<uniform>             params:          ShadowCullParams;

/// Flat array of per-face frustum plane normals (4 planes × 6 faces = 24 entries).
/// Each plane normal defines the inward-facing half-space for that cube face.
/// Must match CUBE_FACE_PLANES in passes/shadow.rs.
const PLANES: array<vec3<f32>, 24> = array(
    // face 0 (+X)
    vec3<f32>( 1.0,  0.0, -1.0), vec3<f32>( 1.0,  0.0,  1.0),
    vec3<f32>( 1.0, -1.0,  0.0), vec3<f32>( 1.0,  1.0,  0.0),
    // face 1 (-X)
    vec3<f32>(-1.0,  0.0, -1.0), vec3<f32>(-1.0,  0.0,  1.0),
    vec3<f32>(-1.0, -1.0,  0.0), vec3<f32>(-1.0,  1.0,  0.0),
    // face 2 (+Y)
    vec3<f32>( 0.0,  1.0, -1.0), vec3<f32>( 0.0,  1.0,  1.0),
    vec3<f32>(-1.0,  1.0,  0.0), vec3<f32>( 1.0,  1.0,  0.0),
    // face 3 (-Y)
    vec3<f32>( 0.0, -1.0, -1.0), vec3<f32>( 0.0, -1.0,  1.0),
    vec3<f32>(-1.0, -1.0,  0.0), vec3<f32>( 1.0, -1.0,  0.0),
    // face 4 (+Z)
    vec3<f32>(-1.0,  0.0,  1.0), vec3<f32>( 1.0,  0.0,  1.0),
    vec3<f32>( 0.0, -1.0,  1.0), vec3<f32>( 0.0,  1.0,  1.0),
    // face 5 (-Z)
    vec3<f32>(-1.0,  0.0, -1.0), vec3<f32>( 1.0,  0.0, -1.0),
    vec3<f32>( 0.0, -1.0, -1.0), vec3<f32>( 0.0,  1.0, -1.0),
);

/// True if the bounding sphere (center at `delta` relative to light, radius `radius`)
/// overlaps the frustum for cube face `face`.
/// Matches sphere_in_cube_face() in passes/shadow.rs.
fn sphere_in_cube_face(delta: vec3<f32>, radius: f32, face: u32) -> bool {
    let threshold = -radius * sqrt(2.0);
    let base = face * 4u;
    return dot(delta, PLANES[base])       >= threshold
        && dot(delta, PLANES[base + 1u])  >= threshold
        && dot(delta, PLANES[base + 2u])  >= threshold
        && dot(delta, PLANES[base + 3u])  >= threshold;
}

@compute @workgroup_size(64)
fn shadow_cull(@builtin(global_invocation_id) gid: vec3<u32>) {
    let draw_idx   = gid.x;
    let light_face = gid.y;

    if draw_idx   >= params.draw_count          { return; }
    if light_face >= params.light_count * 6u    { return; }

    let light_idx = light_face / 6u;
    let face      = light_face % 6u;

    let dc    = draw_calls[draw_idx];
    let light = cull_lights[light_idx];

    let out_idx = light_face * params.draw_count + draw_idx;

    // ── Cull ─────────────────────────────────────────────────────────────────
    var visible = true;

    // Global distance filter (camera-relative filtering done on CPU; here: light-centric cutoff)
    let to_light = dc.bounds_center - light.position;
    let dist_sq  = dot(to_light, to_light);

    // Point/spot lights: skip if too far beyond 5× range + bounds radius
    if light.light_type != 1u {
        let cutoff = light.range * 5.0 + dc.bounds_radius;
        if dist_sq > cutoff * cutoff {
            visible = false;
        }
    }

    // Global max shadow distance filter
    if visible {
        let max_d = params.shadow_max_distance + dc.bounds_radius;
        if dist_sq > max_d * max_d {
            visible = false;
        }
    }

    // Per-face hemisphere cull for point lights only
    if visible && light.light_type == 0u {
        if !sphere_in_cube_face(to_light, dc.bounds_radius, face) {
            visible = false;
        }
    }

    // ── Write indirect command ────────────────────────────────────────────────
    if visible {
        shadow_indirect[out_idx] = DrawCmd(
            dc.index_count,
            1u,
            dc.first_index,
            dc.base_vertex,
            dc.slot,
        );
    } else {
        shadow_indirect[out_idx] = DrawCmd(0u, 0u, 0u, 0, 0u);
    }
}
