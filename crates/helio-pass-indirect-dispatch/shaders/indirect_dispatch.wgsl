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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= cull.draw_count { return; }

    let dc   = draw_calls[idx];
    // Test the first instance of the group for frustum culling (group-level cull).
    // If the representative passes, all instances in the batch are drawn.
    let inst = instances[dc.first_instance];
    let aabb = aabbs[dc.first_instance];

    // AABB-based frustum culling (tighter than sphere for elongated objects).
    // Fall back to sphere if AABB is degenerate (min == max).
    let aabb_visible = aabb_in_frustum(aabb.min, aabb.max);
    let aabb_degenerate = all(aabb.min == aabb.max);

    // bounds.xyz is the world-space bounding sphere center (pre-computed by CPU).
    // bounds.w is the world-space radius.  Do NOT apply the model matrix here —
    // that would double-transform an already-world-space value.
    let world_center = inst.bounds.xyz;
    let world_radius = inst.bounds.w;

    let sphere_visible = sphere_in_frustum(world_center, world_radius);

    let visible = select(sphere_visible, aabb_visible, !aabb_degenerate);

    // Sub-pixel culling: skip objects projecting to < 1 pixel.
    var should_draw = visible;
    if visible {
        let clip_pos = camera.view_proj * vec4<f32>(world_center, 1.0);
        if clip_pos.w > 0.0 {
            // r_ndc ~ world_radius * cot(fov/2) / clip_w
            let r_ndc = abs(world_radius * camera.proj[1][1] / clip_pos.w);
            if r_ndc < 0.001 {
                should_draw = false;
            }
        }
    }

    let is_shadow_caster = (inst.flags & 1u) != 0u;
    if is_shadow_caster {
        atomicAdd(&stats[5u], 1u);
    }

    if should_draw {
        indirect[idx] = DrawIndexedIndirect(
            dc.index_count,
            dc.instance_count,
            dc.first_index,
            dc.vertex_offset,
            dc.first_instance,
        );
        atomicAdd(&stats[3u], 1u);
        if is_shadow_caster {
            atomicAdd(&stats[6u], 1u);
        }
    } else if !visible {
        indirect[idx] = DrawIndexedIndirect(
            dc.index_count,
            0u,
            dc.first_index,
            dc.vertex_offset,
            dc.first_instance,
        );
        atomicAdd(&stats[1u], 1u);
    } else {
        indirect[idx] = DrawIndexedIndirect(
            dc.index_count,
            0u,
            dc.first_index,
            dc.vertex_offset,
            dc.first_instance,
        );
        atomicAdd(&stats[2u], 1u);
    }
    atomicAdd(&stats[0u], 1u);
}
