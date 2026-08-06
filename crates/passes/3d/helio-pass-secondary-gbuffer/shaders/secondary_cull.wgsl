// Per-view frustum cull for the secondary (portal/sublevel) G-buffer fill.
//
// One dispatch per active secondary view, one workgroup per draw-call group
// (mirrors `helio-pass-indirect-dispatch/shaders/indirect_dispatch.wgsl`'s
// per-group cooperative-compaction shape exactly, minus the subpixel/shadow-
// caster bookkeeping this pass doesn't need). Frustum planes are derived
// in-shader from the view's own `cameras[camera_slot].view_proj`
// (`shadow_cull.wgsl`'s per-face inline-plane-extraction idiom) rather than
// CPU-uploaded — secondary camera slots have no CPU-side mirror
// (`GpuCameraBuffer` only mirrors slot 0), so there is nothing to extract
// planes from on the CPU.

const CAMERA_SLOTS: u32 = 7u;

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

struct GpuInstance {
    model_0:      vec4<f32>,
    model_1:      vec4<f32>,
    model_2:      vec4<f32>,
    model_3:      vec4<f32>,
    normal_0:     vec4<f32>,
    normal_1:     vec4<f32>,
    normal_2:     vec4<f32>,
    bounds:       vec4<f32>,
    prev_model_0: vec4<f32>,
    prev_model_1: vec4<f32>,
    prev_model_2: vec4<f32>,
    prev_model_3: vec4<f32>,
    mesh_id:      u32,
    material_id:  u32,
    flags:        u32,
    _pad:         u32,
}

struct GpuDrawCall {
    index_count:    u32,
    first_index:    u32,
    vertex_offset:  i32,
    first_instance: u32,
    instance_count: u32,
}

struct DrawIndexedIndirect {
    index_count:    u32,
    instance_count: u32,
    first_index:    u32,
    base_vertex:    i32,
    first_instance: u32,
}

/// Per-dispatch view parameters. One instance of this uniform per active
/// secondary view (see `SecondaryGBufferPass::view_uniforms` — a plain fixed
/// array of small buffers, not a dynamic-offset ring; `MAX_SECONDARY_VIEWS`
/// is small enough that the extra buffers cost nothing).
struct CullView {
    camera_slot:       u32,
    /// `0` = portal view (no membership filter; render the ordinary
    /// main-visible scene). `1..=15` = sublevel view; only instances whose
    /// membership nibble equals this value pass.
    membership_filter: u32,
    draw_count:         u32,
    _pad0:              u32,
}

@group(0) @binding(0) var<storage, read>       cameras:           array<Camera, CAMERA_SLOTS>;
@group(0) @binding(1) var<storage, read>       instances:         array<GpuInstance>;
@group(0) @binding(2) var<storage, read>       draw_calls:        array<GpuDrawCall>;
@group(0) @binding(3) var<uniform>             view:              CullView;
@group(0) @binding(4) var<storage, read_write> dst_indirect:      array<DrawIndexedIndirect>;
@group(0) @binding(5) var<storage, read_write> compacted_indices: array<u32>;

var<workgroup> wg_counter: atomic<u32>;

/// Mirrors `libhelio::INSTANCE_FLAG_SUBLEVEL_HIDDEN`.
const INSTANCE_FLAG_SUBLEVEL_HIDDEN: u32 = 8u;
/// Mirrors `libhelio::INSTANCE_SUBLEVEL_MEMBERSHIP_SHIFT` / `_MASK`.
const INSTANCE_SUBLEVEL_SHIFT: u32 = 8u;
const INSTANCE_SUBLEVEL_MASK:  u32 = 0xFu << INSTANCE_SUBLEVEL_SHIFT;

fn normalize_plane(p: vec4<f32>) -> vec4<f32> {
    let len = length(p.xyz);
    if len > 1e-10 {
        return vec4<f32>(p.xyz / len, p.w / len);
    }
    return p;
}

fn sphere_in_frustum(vp: mat4x4<f32>, center: vec3<f32>, radius: f32) -> bool {
    let p0 = normalize_plane(vp[3] + vp[0]);
    if dot(p0.xyz, center) + p0.w + radius < 0.0 { return false; }
    let p1 = normalize_plane(vp[3] - vp[0]);
    if dot(p1.xyz, center) + p1.w + radius < 0.0 { return false; }
    let p2 = normalize_plane(vp[3] + vp[1]);
    if dot(p2.xyz, center) + p2.w + radius < 0.0 { return false; }
    let p3 = normalize_plane(vp[3] - vp[1]);
    if dot(p3.xyz, center) + p3.w + radius < 0.0 { return false; }
    let p4 = normalize_plane(vp[2]);
    if dot(p4.xyz, center) + p4.w + radius < 0.0 { return false; }
    let p5 = normalize_plane(vp[3] - vp[2]);
    if dot(p5.xyz, center) + p5.w + radius < 0.0 { return false; }
    return true;
}

fn instance_visible_to_view(flags: u32) -> bool {
    if view.membership_filter != 0u {
        let membership = (flags & INSTANCE_SUBLEVEL_MASK) >> INSTANCE_SUBLEVEL_SHIFT;
        return membership == view.membership_filter;
    }
    // Portal view: the ordinary main-visible scene. Sublevel members are
    // excluded — portals-into-sublevels is a documented v1 limitation
    // (docs/portals_and_sublevels.md's scope decisions).
    return (flags & INSTANCE_FLAG_SUBLEVEL_HIDDEN) == 0u;
}

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let idx = wg_id.x;
    if idx >= view.draw_count { return; }

    let dc = draw_calls[idx];
    let vp = cameras[view.camera_slot].view_proj;

    for (var i = lid.x; i < dc.instance_count; i += 64u) {
        let slot_idx = dc.first_instance + i;
        let inst = instances[slot_idx];
        if instance_visible_to_view(inst.flags) && sphere_in_frustum(vp, inst.bounds.xyz, inst.bounds.w) {
            let slot = atomicAdd(&wg_counter, 1u);
            compacted_indices[dc.first_instance + slot] = slot_idx;
        }
    }

    workgroupBarrier();
    if lid.x != 0u { return; }

    // Always write this group's entry, both branches — unlike the main
    // scene's `indirect_dispatch.wgsl` (which relies on a separate full-
    // buffer clear elsewhere), this pass's per-view `dst_indirect` has no
    // such external clear, so a group that had visible instances last frame
    // and none this frame must be explicitly zeroed here or it would replay
    // stale geometry.
    let visible_count = atomicLoad(&wg_counter);
    if visible_count > 0u {
        dst_indirect[idx] = DrawIndexedIndirect(dc.index_count, visible_count, dc.first_index, dc.vertex_offset, dc.first_instance);
    } else {
        dst_indirect[idx] = DrawIndexedIndirect(0u, 0u, 0u, 0, 0u);
    }
}
