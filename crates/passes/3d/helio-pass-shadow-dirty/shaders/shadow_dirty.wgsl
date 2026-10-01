const MAX_FACES: u32 = 1536u;

/// Minimum world-space displacement (metres) that counts as a "move".
/// Set to ~0.1 mm — below floating point noise threshold at scene scale.
const MOVE_EPSILON: f32 = 0.0001;

// ── Structs ───────────────────────────────────────────────────────────────────

/// Must match GpuInstanceData in libhelio/src/instance.rs (208 bytes).
/// We only need the transform and bounds, so we read partial data.
struct GpuInstance {
    transform:    mat4x4f,  // model matrix, 64 bytes
    normal_mat_0: vec4f,    // 16 bytes
    normal_mat_1: vec4f,    // 16 bytes
    normal_mat_2: vec4f,    // 16 bytes
    bounds:       vec4f,    // xyz = world-space bounding sphere center, w = radius
    prev_model:   mat4x4f,  // 64 bytes
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
    atlas: vec4f,
    policy: vec4u,
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
@group(0) @binding(2) var<storage, read_write>    prev_positions: array<Previous>;
@group(0) @binding(3) var<storage, read>          shadow_mats:    array<GpuShadowMatrix>;
/// Per-face dirty flag (0 = clean, 1 = dirty). Also used as clear-draw count by ShadowPass.
@group(0) @binding(4) var<storage, read_write>    face_dirty:     array<atomic<u32>>;
/// Per-face geometry draw count written to drive multi_draw_indexed_indirect_count.
/// 0 = clean face (no draws), movable_draw_count = dirty face (draw all movable casters).
@group(0) @binding(5) var<storage, read_write>    face_geom_count: array<atomic<u32>>;
@group(0) @binding(6) var<uniform>                uniforms:       ShadowDirtyUniforms;
/// Per-caster flags written by ShadowMatrixPass when a light matrix changes.
@group(0) @binding(7) var<storage, read_write>     light_dirty:    array<atomic<u32>>;

struct Previous { bounds:vec4f, hash:vec4u }
@group(0) @binding(8) var<storage,read> spaces:array<mat4x4f>;

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
        if dist < -radius*length(p.xyz) {
            return false;  // entirely outside this plane → not in frustum
        }
    }
    return true;
}

// ── Main ──────────────────────────────────────────────────────────────────────


fn hash_matrix(m:mat4x4f,h:u32)->u32 {
    var result=h;
    for(var c=0u;c<4u;c++){for(var r=0u;r<4u;r++){result=(result^bitcast<u32>(m[c][r]))*16777619u;}}
    return result;
}
fn dirty(face:u32) {
    atomicStore(&face_dirty[face],1u);
    atomicStore(&face_geom_count[face],uniforms.movable_draw_count);
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3u) {
    let tid=gid.x;let count=min(uniforms.face_count,arrayLength(&shadow_mats));
    if tid==0u {
        for(var caster=0u;caster<(count+5u)/6u;caster++) {
            let moved=atomicExchange(&light_dirty[caster],0u)!=0u;
            if moved || uniforms.force_dirty_all!=0u {
                for(var f=caster*6u;f<min(caster*6u+6u,count);f++){dirty(f);}
            }
        }
    }
    if tid>=min(uniforms.movable_draw_count,arrayLength(&movable_draws)) || tid>=arrayLength(&prev_positions) {return;}
    let draw=movable_draws[tid];var hash=draw.instance_count^draw.index_count^draw.first_instance;
    var lo=vec3f(3e38);var hi=vec3f(-3e38);
    for(var i=0u;i<draw.instance_count;i++) {
        let inst=instances[draw.first_instance+i];
        let space=spaces[(inst.flags>>8u)&255u];
        hash=hash_matrix(space*inst.transform,hash);
        hash=(hash^inst.flags^inst.mesh_id^inst.material_id)*16777619u;
        // Bounds are world-relative to the instance's coordinate space.
        let center=(space*vec4f(inst.bounds.xyz,1)).xyz;
        let radius=inst.bounds.w*max(length(space[0].xyz),max(length(space[1].xyz),length(space[2].xyz)));
        lo=min(lo,center-vec3f(radius));hi=max(hi,center+vec3f(radius));
    }
    var bounds=vec4f(0);
    if draw.instance_count>0u {bounds=vec4f((lo+hi)*0.5,length(hi-lo)*0.5);}
    let previous=prev_positions[tid];
    prev_positions[tid]=Previous(bounds,vec4u(hash,0,0,0));
    if previous.hash.x==hash && all(previous.bounds==bounds) {return;}
    // Test both positions: moving OUT of a tile must erase the old silhouette.
    for(var f=0u;f<count;f++) {
        if shadow_mats[f].atlas.z==0.0 {continue;}
        let planes=extract_frustum_planes(shadow_mats[f].mat);
        if uniforms.force_dirty_all!=0u || sphere_vs_frustum(bounds.xyz,bounds.w,planes)
            || sphere_vs_frustum(previous.bounds.xyz,previous.bounds.w,planes) {dirty(f);}
    }
}
