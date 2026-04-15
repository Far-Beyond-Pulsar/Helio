/// Per-face shadow dirty detection compute shader.
///
/// Runs once per frame AFTER ShadowMatrixPass has updated shadow VP matrices.
/// One thread per movable shadow-caster draw call.
///
/// Algorithm:
///   1. Thread i reads `movable_draws[i].first_instance` to get the instance index.
///   2. Reads current world-space position from `instances[inst_idx].transform` column 3.
///   3. Compares against `prev_positions[i].xyz` (stored from last frame).
///   4. If the object moved more than EPSILON:
///      For each active shadow face, extract 6 frustum planes from the VP matrix
///      (Gribb-Hartmann), and sphere-test the object's bounding sphere.
///      Any face that sees the moved object gets `face_dirty[face] = 1` (atomicOr).
///      Additionally writes `face_geom_count[face] = movable_draw_count` so
///      ShadowPass can use multi_draw_indexed_indirect_count with a GPU count.
///   5. Updates `prev_positions[i]` with the current position for next frame.
///
/// The zeroing of `face_dirty` and `face_geom_count` each frame is done by
/// thread 0 at workgroup 0 before the main loop.

// ── Constants ─────────────────────────────────────────────────────────────────

const MAX_FACES: u32 = 256u;

/// Minimum world-space displacement (metres) that counts as a "move".
/// Set to ~0.1 mm — below floating point noise threshold at scene scale.
const MOVE_EPSILON: f32 = 0.0001;

// ── Structs ───────────────────────────────────────────────────────────────────

/// Must match GpuInstanceData in libhelio/src/instance.rs (144 bytes).
/// We only need the transform and bounds, so we read partial data.
struct GpuInstance {
    transform:    mat4x4f,  // model matrix, 64 bytes
    normal_mat_0: vec4f,    // 16 bytes
    normal_mat_1: vec4f,    // 16 bytes
    normal_mat_2: vec4f,    // 16 bytes
    bounds:       vec4f,    // xyz = world-space bounding sphere center, w = radius
    mesh_id:      u32,
    material_id:  u32,
    flags:        u32,
    lightmap_index: u32,
}

/// Matches wgpu DrawIndexedIndirect layout (20 bytes).
struct DrawIndexedIndirect {
    index_count:    u32,
    instance_count: u32,
    first_index:    u32,
    base_vertex:    i32,
    first_instance: u32,
}

/// Must match GpuShadowMatrix in shadow_matrices.wgsl / libhelio (64 bytes).
struct GpuShadowMatrix {
    mat: mat4x4f,
}

struct ShadowDirtyUniforms {
    /// Number of active draw calls in shadow_movable_indirect.
    movable_draw_count: u32,
    /// Number of active shadow faces (= shadow_count from SceneResources).
    face_count: u32,
    /// Set to 1 on the frame when movable_draw_count changes — dirties all faces.
    force_dirty_all: u32,
    _pad: u32,
}

// ── Bindings ──────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<storage, read>         instances:      array<GpuInstance>;
@group(0) @binding(1) var<storage, read>          movable_draws:  array<DrawIndexedIndirect>;
@group(0) @binding(2) var<storage, read_write>    prev_positions: array<vec4f>;
@group(0) @binding(3) var<storage, read>          shadow_mats:    array<GpuShadowMatrix>;
/// Per-face dirty flag (0 = clean, 1 = dirty). Also used as clear-draw count by ShadowPass.
@group(0) @binding(4) var<storage, read_write>    face_dirty:     array<atomic<u32>>;
/// Per-face geometry draw count written to drive multi_draw_indexed_indirect_count.
/// 0 = clean face (no draws), movable_draw_count = dirty face (draw all movable casters).
@group(0) @binding(5) var<storage, read_write>    face_geom_count: array<u32>;
@group(0) @binding(6) var<uniform>                uniforms:       ShadowDirtyUniforms;

// ── Frustum helpers (Gribb-Hartmann) ─────────────────────────────────────────

/// Extract 6 view-frustum half-space planes from a VP matrix (column-major WGSL mat4x4f).
///
/// Each plane is vec4f(nx, ny, nz, d) where the signed distance of a point P from the
/// plane is:  dot(normal, P) + d.  A positive value means "inside" the frustum.
/// The direction convention is: planes[i].xyz points INWARD into the frustum.
fn extract_frustum_planes(m: mat4x4f) -> array<vec4f, 6> {
    // WGSL mat4x4f: m[col][row], i.e. m[c].r accesses column c, row r.
    // We need the rows of the matrix for Gribb-Hartmann:
    let r0 = vec4f(m[0][0], m[1][0], m[2][0], m[3][0]);
    let r1 = vec4f(m[0][1], m[1][1], m[2][1], m[3][1]);
    let r2 = vec4f(m[0][2], m[1][2], m[2][2], m[3][2]);
    let r3 = vec4f(m[0][3], m[1][3], m[2][3], m[3][3]);

    var planes: array<vec4f, 6>;
    planes[0] = r3 + r0;   // Left   plane
    planes[1] = r3 - r0;   // Right  plane
    planes[2] = r3 + r1;   // Bottom plane
    planes[3] = r3 - r1;   // Top    plane
    planes[4] = r2;         // Near   plane  (depth [0,1] convention)
    planes[5] = r3 - r2;   // Far    plane
    return planes;
}

/// Returns true if the sphere (center, radius) intersects or is inside all 6 planes.
fn sphere_vs_frustum(center: vec3f, radius: f32, planes: array<vec4f, 6>) -> bool {
    for (var i = 0u; i < 6u; i++) {
        let p = planes[i];
        // Signed distance from center to plane (positive = inside half-space).
        let dist = dot(p.xyz, center) + p.w;
        if dist < -radius {
            return false;  // entirely outside this plane → not in frustum
        }
    }
    return true;
}

// ── Main ──────────────────────────────────────────────────────────────────────

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;

    let movable_count = uniforms.movable_draw_count;
    let face_count    = min(uniforms.face_count, MAX_FACES);
    let force_all     = uniforms.force_dirty_all;

    // ── Initialisation: thread 0 zeroes the output arrays ──────────────────
    // Only workgroup-global thread 0 does this to avoid races.
    // The zeroing of face_dirty/face_geom_count happens BEFORE the position
    // comparison below (no barrier needed within a single workgroup-0 thread).
    if tid == 0u {
        for (var f = 0u; f < face_count; f++) {
            atomicStore(&face_dirty[f], 0u);
            face_geom_count[f] = 0u;
        }
    }

    // Ensure all threads see the zeroed arrays before writing dirty flags.
    storageBarrier();

    // force_dirty_all: topology changed (movable count changed) — dirty every face.
    if force_all != 0u {
        if tid == 0u {
            for (var f = 0u; f < face_count; f++) {
                atomicStore(&face_dirty[f], 1u);
                face_geom_count[f] = movable_count;
            }
        }
        // Also update prev_positions for every draw call so next frame is baseline.
        if tid < movable_count {
            let inst_idx  = movable_draws[tid].first_instance;
            let inst      = instances[inst_idx];
            let curr_pos  = vec3f(inst.transform[3][0], inst.transform[3][1], inst.transform[3][2]);
            prev_positions[tid] = vec4f(curr_pos, 0.0);
        }
        return;
    }

    // Per-draw-call dirty detection.
    if tid >= movable_count {
        return;
    }

    // Look up the actual instance this draw call refers to.
    let inst_idx = movable_draws[tid].first_instance;
    let inst     = instances[inst_idx];

    // World-space position = translation column of the model matrix.
    let curr_pos = vec3f(inst.transform[3][0], inst.transform[3][1], inst.transform[3][2]);
    let radius   = inst.bounds.w;

    // Compare against previous frame's position.
    let prev_pos = prev_positions[tid].xyz;
    let delta    = abs(curr_pos - prev_pos);
    let moved    = delta.x > MOVE_EPSILON || delta.y > MOVE_EPSILON || delta.z > MOVE_EPSILON;

    // Always update the stored position (even if unchanged, cost is one write).
    prev_positions[tid] = vec4f(curr_pos, 0.0);

    if !moved {
        return;
    }

    // Object moved → test its bounding sphere against every active shadow face.
    for (var face = 0u; face < face_count; face++) {
        let planes = extract_frustum_planes(shadow_mats[face].mat);
        if sphere_vs_frustum(curr_pos, radius, planes) {
            // Mark face dirty and set geometry draw count.
            atomicOr(&face_dirty[face], 1u);
            // Non-atomic write of movable_count — multiple threads may race on the
            // same face, but they all write the identical value so it is safe.
            face_geom_count[face] = movable_count;
        }
    }
}
