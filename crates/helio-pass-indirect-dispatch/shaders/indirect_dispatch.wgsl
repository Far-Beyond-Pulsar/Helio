// GPU frustum culling + indirect draw command generation.
// O(1) CPU cost: one dispatch, all culling on GPU.

struct Camera {
    view:          mat4x4<f32>,
    proj:          mat4x4<f32>,
    view_proj:     mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    position_near: vec4<f32>,
    forward_far:   vec4<f32>,
    jitter_frame:  vec4<f32>,
    prev_view_proj: mat4x4<f32>,
}

struct CullUniforms {
    frustum_planes: array<vec4<f32>, 6>,
    draw_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct GpuInstance {
    model_0:     vec4<f32>,
    model_1:     vec4<f32>,
    model_2:     vec4<f32>,
    model_3:     vec4<f32>,
    normal_0:    vec4<f32>,
    normal_1:    vec4<f32>,
    normal_2:    vec4<f32>,
    bounds:      vec4<f32>,  // xyz = world-space center, w = world-space radius
    mesh_id:     u32,
    material_id: u32,
    flags:       u32,
    _pad:        u32,
}

struct GpuDrawCall {
    index_count:    u32,
    first_index:    u32,
    vertex_offset:  i32,
    first_instance: u32,  // base index into instances[] for this batch
    instance_count: u32,  // number of consecutive instances
}

struct GpuAabb {
    min:   vec3<f32>,
    _pad0: f32,
    max:   vec3<f32>,
    _pad1: f32,
}

struct DrawIndexedIndirect {
    index_count:    u32,
    instance_count: u32,
    first_index:    u32,
    base_vertex:    i32,
    first_instance: u32,
}

@group(0) @binding(0) var<uniform>            camera:     Camera;
@group(0) @binding(1) var<uniform>            cull:       CullUniforms;
@group(0) @binding(2) var<storage, read>      instances:  array<GpuInstance>;
@group(0) @binding(3) var<storage, read>      draw_calls: array<GpuDrawCall>;
@group(0) @binding(4) var<storage, read>      aabbs:     array<GpuAabb>;
@group(0) @binding(5) var<storage, read_write> indirect:  array<DrawIndexedIndirect>;
@group(0) @binding(6) var<storage, read_write> stats:   array<atomic<u32>>;
// Per-group compacted list of original instance slots that survive frustum
// culling, packed starting at each group's `first_instance` offset. Consumers
// that draw through `indirect` must index `instances` through this buffer.
@group(0) @binding(7) var<storage, read_write> compacted_indices: array<u32>;

// One workgroup handles one draw-call group. Its 64 lanes cooperatively test
// every instance in the group and compact survivors via workgroup-shared
// atomics, so a group's final `instance_count` reflects only what actually
// passed the frustum test instead of an all-or-nothing pass/fail per batch.
var<workgroup> wg_counter:        atomic<u32>;
var<workgroup> wg_nonsubpixel:    atomic<u32>;
var<workgroup> wg_shadow_caster:  atomic<u32>;

// Stats layout (shared with OcclusionCullPass):
// 0: total_draws
// 1: frustum_culled
// 2: subpixel_culled
// 3: frustum_visible
// 4: occlusion_culled     ← written by occlusion pass only
// 5: shadow_total
// 6: shadow_frustum_visible
// 7: shadow_occlusion_culled ← written by occlusion pass only

fn sphere_in_frustum(center: vec3<f32>, radius: f32) -> bool {
    for (var i = 0u; i < 6u; i++) {
        let plane = cull.frustum_planes[i];
        let dist = dot(plane.xyz, center) + plane.w;
        if dist + radius < 0.0 { return false; }
    }
    return true;
}

fn aabb_in_frustum(min: vec3<f32>, max: vec3<f32>) -> bool {
    for (var i = 0u; i < 6u; i++) {
        let plane = cull.frustum_planes[i];
        let px = select(min.x, max.x, plane.x >= 0.0);
        let py = select(min.y, max.y, plane.y >= 0.0);
        let pz = select(min.z, max.z, plane.z >= 0.0);
        let dist = dot(plane.xyz, vec3<f32>(px, py, pz)) + plane.w;
        if dist < 0.0 { return false; }
    }
    return true;
}

fn test_instance(inst: GpuInstance, aabb: GpuAabb) -> bool {
    let aabb_visible = aabb_in_frustum(aabb.min, aabb.max);
    let aabb_degenerate = all(aabb.min == aabb.max);
    let sphere_visible = sphere_in_frustum(inst.bounds.xyz, inst.bounds.w);
    return select(sphere_visible, aabb_visible, !aabb_degenerate);
}

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let idx = wg_id.x;
    if idx >= cull.draw_count { return; }

    let dc = draw_calls[idx];

    // Cooperatively test every instance in the group across the workgroup's 64
    // lanes (grid-stride loop), compacting survivors into `compacted_indices`
    // via a workgroup-shared atomic counter. `wg_counter`/`wg_nonsubpixel`/
    // `wg_shadow_caster` are zero-initialized per workgroup dispatch by WGSL.
    for (var i = lid.x; i < dc.instance_count; i += 64u) {
        let slot_idx = dc.first_instance + i;
        let inst = instances[slot_idx];
        let aabb = aabbs[slot_idx];
        if test_instance(inst, aabb) {
            let slot = atomicAdd(&wg_counter, 1u);
            compacted_indices[dc.first_instance + slot] = slot_idx;

            // Sub-pixel test: check if this instance projects to ≥ 1 pixel.
            let clip_pos = camera.view_proj * vec4<f32>(inst.bounds.xyz, 1.0);
            if clip_pos.w > 0.0 {
                let r_ndc = abs(inst.bounds.w * camera.proj[1][1] / clip_pos.w);
                if r_ndc >= 0.001 {
                    atomicStore(&wg_nonsubpixel, 1u);
                }
            }
            if (inst.flags & 1u) != 0u {
                atomicStore(&wg_shadow_caster, 1u);
            }
        }
    }

    workgroupBarrier();

    if lid.x != 0u { return; }

    let visible_count = atomicLoad(&wg_counter);
    let any_visible = visible_count > 0u;
    let subpixel_only = any_visible && atomicLoad(&wg_nonsubpixel) == 0u;
    let batch_has_shadow_caster = atomicLoad(&wg_shadow_caster) != 0u;

    if batch_has_shadow_caster {
        atomicAdd(&stats[5u], 1u);
    }

    if any_visible && !subpixel_only {
        indirect[idx] = DrawIndexedIndirect(
            dc.index_count,
            visible_count,
            dc.first_index,
            dc.vertex_offset,
            dc.first_instance,
        );
        atomicAdd(&stats[3u], 1u);
        if batch_has_shadow_caster {
            atomicAdd(&stats[6u], 1u);
        }
    } else {
        indirect[idx] = DrawIndexedIndirect(
            dc.index_count,
            0u,
            dc.first_index,
            dc.vertex_offset,
            dc.first_instance,
        );
        if !any_visible {
            atomicAdd(&stats[1u], 1u);
        } else {
            atomicAdd(&stats[2u], 1u);
        }
    }
    atomicAdd(&stats[0u], 1u);
}
