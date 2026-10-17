const MAX_FACES: u32 = 1536u;

/// Minimum world-space displacement (metres) that counts as a "move".
/// Set to ~0.1 mm — below floating point noise threshold at scene scale.
const MOVE_EPSILON: f32 = 0.0001;

// ── Structs ───────────────────────────────────────────────────────────────────

/// Must match GpuInstanceData in libhelio/src/instance.rs (224 bytes).
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
    skin_vertex_offset: u32,
    bone_palette_offset: u32,
    bone_palette_count: u32,
    _skin_pad: u32,
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
    /// Number of active shadow faces (= shadow_count from SceneResources).
    face_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// ── Bindings ──────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<storage, read>         instances:      array<GpuInstance>;
@group(0) @binding(1) var<storage, read>          movable_draws:  array<DrawIndexedIndirect>;
@group(0) @binding(2) var<storage, read_write>    prev_positions: array<Previous>;
@group(0) @binding(3) var<storage, read>          shadow_mats:    array<GpuShadowMatrix>;
/// Per-face dirty flag (0 = clean, 1 = dirty). Also used as clear-draw count by ShadowPass.
@group(0) @binding(4) var<storage, read_write>    face_dirty:     array<atomic<u32>>;
/// Per-face draw counts that drive multi_draw_indexed_indirect_count, in three
/// MAX_FACES regions: movable, static and transmissive casters. 0 = clean face
/// (no draws); a dirty face gets the list's live count (draw every caster).
/// ShadowPass draws a face it does not know to be dirty with these counts, so
/// a face nothing dirtied runs no vertices at all.
@group(0) @binding(5) var<storage, read_write>    face_geom_count: array<atomic<u32>>;
@group(0) @binding(6) var<uniform>                uniforms:       ShadowDirtyUniforms;
/// Per-caster flags written by ShadowMatrixPass when a light matrix changes.
@group(0) @binding(7) var<storage, read_write>     light_dirty:    array<atomic<u32>>;

struct Previous { bounds:vec4f, hash:vec4u }
@group(0) @binding(8) var<storage,read> spaces:array<mat4x4f>;
/// Object Batch's GPU counts, written this frame (`ObjectBatchFrameData::
/// draw_counts_gpu`); word 2 is the movable shadow-caster count.
@group(0) @binding(9) var<storage,read> batch_counts:array<u32>;
/// The movable count the last dispatch saw. Only invocation 0 touches it.
@group(0) @binding(10) var<storage,read_write> last_movable_count:array<u32>;

/// Live draw calls in `movable_draws` this frame.
fn movable_count()->u32 { return batch_counts[2]; }
/// Live static and transmissive shadow-caster draws this frame.
fn static_count()->u32 { return batch_counts[1]; }
fn transmissive_count()->u32 { return batch_counts[3]; }

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
/// `face_dirty` bits: the layer of the shadow atlas that must be redrawn.
const DIRTY_DYNAMIC: u32 = 1u;
const DIRTY_STATIC: u32 = 2u;
/// A movable caster moved, appeared or vanished: only the dynamic layer holds
/// movable casters, so the cached static layer (and glass) is left alone.
fn dirty_dynamic(face:u32) {
    atomicOr(&face_dirty[face],DIRTY_DYNAMIC);
    atomicStore(&face_geom_count[face],movable_count());
}
/// The light itself changed: both layers are drawn from its matrix.
fn dirty_all(face:u32) {
    atomicOr(&face_dirty[face],DIRTY_DYNAMIC|DIRTY_STATIC);
    atomicStore(&face_geom_count[face],movable_count());
    atomicStore(&face_geom_count[MAX_FACES+face],static_count());
    atomicStore(&face_geom_count[2u*MAX_FACES+face],transmissive_count());
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3u) {
    let tid=gid.x;let count=min(uniforms.face_count,arrayLength(&shadow_mats));
    if tid==0u {
        // A caster added or removed shifts every later draw's slot, and a
        // removed one's silhouette must be erased: dirty every face.
        let force_dirty_all=movable_count()!=last_movable_count[0];
        last_movable_count[0]=movable_count();
        for(var caster=0u;caster<(count+5u)/6u;caster++) {
            let moved=atomicExchange(&light_dirty[caster],0u)!=0u;
            if moved {
                for(var f=caster*6u;f<min(caster*6u+6u,count);f++){dirty_all(f);}
            } else if force_dirty_all {
                for(var f=caster*6u;f<min(caster*6u+6u,count);f++){dirty_dynamic(f);}
            }
        }
        // A face dirtied on an earlier frame and deferred by the shadow
        // budget is drawn to these counts: keep them this frame's, as the
        // movable count is kept by dirtying everything when it changes.
        for(var f=0u;f<count;f++) {
            let pending=atomicLoad(&face_dirty[f]);
            if (pending&DIRTY_DYNAMIC)!=0u {
                atomicStore(&face_geom_count[f],movable_count());
            }
            if (pending&DIRTY_STATIC)!=0u {
                atomicStore(&face_geom_count[MAX_FACES+f],static_count());
                atomicStore(&face_geom_count[2u*MAX_FACES+f],transmissive_count());
            }
        }
    }
    if tid>=min(movable_count(),arrayLength(&movable_draws)) || tid>=arrayLength(&prev_positions) {return;}
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
        if sphere_vs_frustum(bounds.xyz,bounds.w,planes)
            || sphere_vs_frustum(previous.bounds.xyz,previous.bounds.w,planes) {dirty_dynamic(f);}
    }
}
