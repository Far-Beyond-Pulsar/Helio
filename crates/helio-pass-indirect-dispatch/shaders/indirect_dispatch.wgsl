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
@group(0) @binding(4) var<storage, read_write> indirect:  array<DrawIndexedIndirect>;

fn sphere_in_frustum(center: vec3<f32>, radius: f32) -> bool {
    for (var i = 0u; i < 6u; i++) {
        let plane = cull.frustum_planes[i];
        let dist = dot(plane.xyz, center) + plane.w;
        if dist < -radius { return false; }
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

    // bounds.xyz is the world-space bounding sphere center (pre-computed by CPU).
    // bounds.w is the world-space radius.  Do NOT apply the model matrix here —
    // that would double-transform an already-world-space value.
    let world_center = inst.bounds.xyz;
    let world_radius = inst.bounds.w;

    let visible = sphere_in_frustum(world_center, world_radius);

    indirect[idx] = DrawIndexedIndirect(
        dc.index_count,
        select(0u, dc.instance_count, visible),  // cull entire batch or keep all N instances
        dc.first_index,
        dc.vertex_offset,
        dc.first_instance,
    );
}
