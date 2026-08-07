// Per-portal GPU frustum culling — selects which instances get a duplicate
// draw through each active portal.
//
// Shape mirrors `helio-pass-indirect-dispatch`'s main cull almost exactly:
// one workgroup per (draw-call group, portal) pair, its 64 lanes
// cooperatively test that group's instances — mapped through the portal's
// `pair_map_inverse` coordinate space instead of world space directly — and
// compact survivors. The output buffers give each portal its own reserved
// slice (`portal_idx * capacity`), the same "per-slot slice of one shared
// flat buffer" idiom `helio-pass-shadow`'s atlas already uses, chosen so this
// never has to touch the main scene's `compacted_indices`/`compacted_indices_2`
// (which have zero headroom for extra entries).
//
// No occlusion (Hi-Z) test here — frustum-only. Portal-visible content is
// already bounded by the frustum test; occlusion is a pure fill-rate
// optimization this v1 skips (see docs/).

struct Camera {
    view:           mat4x4<f32>,
    proj:           mat4x4<f32>,
    view_proj:      mat4x4<f32>,
    inv_view_proj:  mat4x4<f32>,
    position_near:  vec4<f32>,
    forward_far:    vec4<f32>,
    jitter_frame:   vec4<f32>,
    prev_view_proj: mat4x4<f32>,
}
@group(0) @binding(0) var<storage, read> cameras: array<Camera, 2>;

struct CullUniforms {
    frustum_planes:    array<vec4<f32>, 6>,
    draw_count:        u32,
    portal_count:      u32,
    // Stride (in DrawIndexedIndirect / u32 units respectively) between one
    // portal's slice and the next in the two output buffers below.
    draw_capacity:     u32,
    instance_capacity: u32,
}
@group(0) @binding(1) var<uniform> cull: CullUniforms;

// Must match GpuInstanceData in libhelio exactly (208 bytes).
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
@group(0) @binding(2) var<storage, read> instances: array<GpuInstance>;

struct GpuDrawCall {
    index_count:    u32,
    first_index:    u32,
    vertex_offset:  i32,
    first_instance: u32,
    instance_count: u32,
}
@group(0) @binding(3) var<storage, read> draw_calls: array<GpuDrawCall>;

// Coordinate-space transforms (current frame). Slot 0 = identity — see
// `libhelio::{coordinate_space, set_coordinate_space}`.
@group(0) @binding(4) var<storage, read> coordinate_spaces: array<mat4x4<f32>>;

// One active portal's render data. Must match libhelio::GpuPortalView (80 bytes).
struct GpuPortalView {
    inverse_transform: mat4x4<f32>,
    half_extent:       vec2<f32>,
    coordinate_space:  u32,
    _pad:              u32,
}
@group(0) @binding(5) var<storage, read> portal_views: array<GpuPortalView>;

struct DrawIndexedIndirect {
    index_count:    u32,
    instance_count: u32,
    first_index:    u32,
    base_vertex:    i32,
    first_instance: u32,
}
// Per-portal indirect draw commands, one slice of `cull.draw_capacity`
// entries per portal (slice `portal_idx`, offset `portal_idx * draw_capacity`).
@group(0) @binding(6) var<storage, read_write> portal_indirect: array<DrawIndexedIndirect>;
// Per-portal compacted original instance slots, one slice of
// `cull.instance_capacity` u32s per portal — within a slice, laid out exactly
// like the main scene's `compacted_indices` (survivors packed starting at
// `dc.first_instance`).
@group(0) @binding(7) var<storage, read_write> portal_compacted_indices: array<u32>;

var<workgroup> wg_counter: atomic<u32>;

/// Mirrors `libhelio::INSTANCE_FLAG_ALWAYS_VISIBLE`.
const INSTANCE_FLAG_ALWAYS_VISIBLE: u32 = 4u;

fn sphere_in_frustum(center: vec3<f32>, radius: f32) -> bool {
    for (var i = 0u; i < 6u; i++) {
        let plane = cull.frustum_planes[i];
        if dot(plane.xyz, center) + plane.w + radius < 0.0 { return false; }
    }
    return true;
}

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let draw_idx = wg_id.x;
    let portal_idx = wg_id.y;
    let is_active = draw_idx < cull.draw_count && portal_idx < cull.portal_count;

    // Every invocation must reach the barrier below regardless of `is_active`
    // — see indirect_dispatch.wgsl for why (FXC barrier-uniformity rejection).
    if is_active {
        let portal = portal_views[portal_idx];
        let portal_space = coordinate_spaces[portal.coordinate_space];
        let dc = draw_calls[draw_idx];
        let compacted_base = portal_idx * cull.instance_capacity;

        for (var i = lid.x; i < dc.instance_count; i += 64u) {
            let slot_idx = dc.first_instance + i;
            let inst = instances[slot_idx];

            // An instance already living in THIS portal's own coordinate
            // space is itself a duplicate (or, for the object that defines
            // the portal's far side, would duplicate onto itself) — skip it.
            // This is also the depth-1 recursion boundary: a duplicate drawn
            // through one portal is never re-duplicated through another.
            let own_space_id = (inst.flags >> 8u) & 0xFFu;
            if own_space_id == portal.coordinate_space {
                continue;
            }

            let own_space = coordinate_spaces[own_space_id];
            let world_center = (portal_space * (own_space * vec4<f32>(inst.bounds.xyz, 1.0))).xyz;

            let visible = (inst.flags & INSTANCE_FLAG_ALWAYS_VISIBLE) != 0u
                || sphere_in_frustum(world_center, inst.bounds.w);
            if visible {
                let slot = atomicAdd(&wg_counter, 1u);
                let write_idx = compacted_base + dc.first_instance + slot;
                // Defensive: `instance_capacity` is a generous fixed cap (see
                // the Rust side), not resized to track scene growth. A scene
                // exceeding it silently drops the excess here rather than
                // writing out of bounds — same accepted degradation shape as
                // helio-pass-shadow-cull's own per-face draw cap.
                if write_idx < arrayLength(&portal_compacted_indices) {
                    portal_compacted_indices[write_idx] = slot_idx;
                }
            }
        }
    }

    workgroupBarrier();

    if lid.x == 0u && is_active {
        // Re-read: `dc` above is scoped to the `is_active` block this needed
        // to stay outside of (the barrier must be reached unconditionally).
        let dc2 = draw_calls[draw_idx];
        let visible_count = atomicLoad(&wg_counter);
        let indirect_base = portal_idx * cull.draw_capacity;
        let write_idx = indirect_base + draw_idx;
        if write_idx < arrayLength(&portal_indirect) {
            portal_indirect[write_idx] = DrawIndexedIndirect(
                dc2.index_count,
                visible_count,
                dc2.first_index,
                dc2.vertex_offset,
                dc2.first_instance,
            );
        }
    }
}
