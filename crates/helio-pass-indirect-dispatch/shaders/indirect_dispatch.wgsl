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
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= cull.draw_count { return; }

    let dc = draw_calls[idx];

    // Iterate all instances in the batch. Cull the entire batch only if
    // EVERY instance is out of view. This prevents flickering when the
    // representative (first) instance is behind the camera but others are visible.
    var any_visible = false;
    var batch_has_shadow_caster = false;
    var subpixel_only = true;
    for (var i = 0u; i < dc.instance_count; i++) {
        let inst = instances[dc.first_instance + i];
        let aabb = aabbs[dc.first_instance + i];
        if test_instance(inst, aabb) {
            any_visible = true;
            // Sub-pixel test: check if this instance projects to ≥ 1 pixel.
            let clip_pos = camera.view_proj * vec4<f32>(inst.bounds.xyz, 1.0);
            if clip_pos.w > 0.0 {
                let r_ndc = abs(inst.bounds.w * camera.proj[1][1] / clip_pos.w);
                if r_ndc >= 0.001 {
                    subpixel_only = false;
                }
            }
            if (inst.flags & 1u) != 0u {
                batch_has_shadow_caster = true;
            }
        }
    }

    if batch_has_shadow_caster {
        atomicAdd(&stats[5u], 1u);
    }

    if any_visible && !subpixel_only {
        indirect[idx] = DrawIndexedIndirect(
            dc.index_count,
            dc.instance_count,
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
