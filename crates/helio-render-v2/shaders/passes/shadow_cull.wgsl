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
//!
//! Culling strategy per light type:
//!
//!   Directional — per-cascade frustum test using the actual shadow matrix.
//!                 The cascade matrix encodes the exact ortho volume, so this
//!                 is pixel-perfect with zero magic distance constants and no
//!                 dependence on world-origin placement of geometry.
//!
//!   Point       — light-range sphere (5× extended) + hemisphere cull per face.
//!
//!   Spot        — light-range sphere (5× extended).

// ── Structs ───────────────────────────────────────────────────────────────────

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
    shadow_max_distance: f32,  // used only by point / spot lights
    _pad:                u32,
}

/// Shadow matrix (one per light face / cascade).
/// Must match GpuShadowMatrix / LightMatrix used by the shadow render pass.
struct LightMatrix {
    mat: mat4x4<f32>,
}

// ── Bindings ──────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<storage, read>       draw_calls:      array<GpuDrawCall>;
@group(0) @binding(1) var<storage, read>       cull_lights:     array<ShadowCullLight>;
@group(0) @binding(2) var<storage, read_write> shadow_indirect: array<DrawCmd>;
@group(0) @binding(3) var<uniform>             params:          ShadowCullParams;
@group(0) @binding(4) var<storage, read>       shadow_matrices: array<LightMatrix>;

// ── Frustum helpers (identical to indirect_dispatch.wgsl) ────────────────────

/// Extract Gribb-Hartmann frustum planes from a combined view-projection matrix.
/// The resulting planes are in the space the matrix maps FROM (world space),
/// so world-space bounding sphere centres can be tested directly against them.
/// wgpu / Vulkan NDC depth is z ∈ [0, 1] — hence the near plane is just r2.
fn extract_frustum_planes(m: mat4x4<f32>) -> array<vec4<f32>, 6> {
    // Build the four row vectors of the matrix.
    // In WGSL m[col][row], so m[c][r] = element at row r, column c.
    let r0 = vec4<f32>(m[0][0], m[1][0], m[2][0], m[3][0]);
    let r1 = vec4<f32>(m[0][1], m[1][1], m[2][1], m[3][1]);
    let r2 = vec4<f32>(m[0][2], m[1][2], m[2][2], m[3][2]);
    let r3 = vec4<f32>(m[0][3], m[1][3], m[2][3], m[3][3]);

    var planes: array<vec4<f32>, 6>;
    planes[0] = r3 + r0;   // left
    planes[1] = r3 - r0;   // right
    planes[2] = r3 + r1;   // bottom
    planes[3] = r3 - r1;   // top
    planes[4] = r2;        // near  (z >= 0)
    planes[5] = r3 - r2;   // far   (z <= 1)
    return planes;
}

/// Conservative sphere-vs-frustum test.
/// Returns false only when the sphere is definitively outside a frustum plane.
/// Uses un-normalised plane normals; multiplies radius by |n| to avoid sqrt.
fn frustum_cull_sphere(planes: array<vec4<f32>, 6>, centre: vec3<f32>, radius: f32) -> bool {
    for (var i = 0u; i < 6u; i++) {
        let n    = planes[i].xyz;
        let dist = dot(n, centre) + planes[i].w;
        if dist < -(radius * length(n)) { return false; }
    }
    return true;
}

// ── Point-light per-face helpers ─────────────────────────────────────────────

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
fn sphere_in_cube_face(delta: vec3<f32>, radius: f32, face: u32) -> bool {
    let threshold = -radius * sqrt(2.0);
    let base = face * 4u;
    return dot(delta, PLANES[base])       >= threshold
        && dot(delta, PLANES[base + 1u])  >= threshold
        && dot(delta, PLANES[base + 2u])  >= threshold
        && dot(delta, PLANES[base + 3u])  >= threshold;
}

// ── Main ──────────────────────────────────────────────────────────────────────

@compute @workgroup_size(64)
fn shadow_cull(@builtin(global_invocation_id) gid: vec3<u32>) {
    let draw_idx   = gid.x;
    let light_face = gid.y;

    if draw_idx   >= params.draw_count       { return; }
    if light_face >= params.light_count * 6u { return; }

    let light_idx = light_face / 6u;
    let face      = light_face % 6u;

    let dc    = draw_calls[draw_idx];
    let light = cull_lights[light_idx];

    let out_idx = light_face * params.draw_count + draw_idx;

    // ── Per-light-type culling ────────────────────────────────────────────────
    var visible = true;

    if light.light_type == 1u {
        // ── Directional light: per-cascade frustum test ───────────────────────
        //
        // The cascade matrix (shadow_matrices[light_idx * 6 + face]) maps world
        // space into the exact ortho clip volume for this cascade slice.
        // Extracting Gribb-Hartmann planes from it gives us a tight world-space
        // frustum that accounts for camera position, cascade extent, and depth
        // range — no magic distance constants, no world-origin bias.
        //
        // For directional lights face_count = 4 (cascades 0-3), so face is
        // always a valid cascade index here.
        let cascade_mat = shadow_matrices[light_idx * 6u + face].mat;
        let planes      = extract_frustum_planes(cascade_mat);
        if !frustum_cull_sphere(planes, dc.bounds_center, dc.bounds_radius) {
            visible = false;
        }

    } else {
        // ── Point / spot light: range sphere + hemisphere (point only) ────────

        let to_light = dc.bounds_center - light.position;
        let dist_sq  = dot(to_light, to_light);

        // Skip if the mesh is beyond 5× the light range (generous margin so
        // objects just outside the nominal range still cast correct shadows).
        let range_cutoff = light.range * 5.0 + dc.bounds_radius;
        if dist_sq > range_cutoff * range_cutoff {
            visible = false;
        }

        // Hard global backstop so extremely large light ranges don't force
        // shadow rendering of the whole world.
        if visible {
            let max_d = params.shadow_max_distance + dc.bounds_radius;
            if dist_sq > max_d * max_d {
                visible = false;
            }
        }

        // Per-face hemisphere cull for point lights: skip faces where the
        // entire bounding sphere lies in the opposite hemisphere.
        if visible && light.light_type == 0u {
            if !sphere_in_cube_face(to_light, dc.bounds_radius, face) {
                visible = false;
            }
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