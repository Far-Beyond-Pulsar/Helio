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
    _pad: vec3<u32>,
}

struct GpuInstance {
    model_0:     vec4<f32>,
    model_1:     vec4<f32>,
    model_2:     vec4<f32>,
    model_3:     vec4<f32>,
    normal_0:    vec4<f32>,
    normal_1:    vec4<f32>,
    normal_2:    vec4<f32>,
    bounds:      vec4<f32>,  // xyz = center, w = radius
    mesh_id:     u32,
    material_id: u32,
    flags:       u32,
    _pad:        u32,
}

struct GpuDrawCall {
    index_count:   u32,
    first_index:   u32,
    vertex_offset: i32,
    instance_id:   u32,
    _pad:          u32,
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
    let inst = instances[dc.instance_id];

    // Transform bounding sphere center to world space via model matrix
    let model = mat4x4<f32>(inst.model_0, inst.model_1, inst.model_2, inst.model_3);
    let world_center = (model * vec4<f32>(inst.bounds.xyz, 1.0)).xyz;

    // Scale radius by max scale component
    let sx = length(inst.model_0.xyz);
    let sy = length(inst.model_1.xyz);
    let sz = length(inst.model_2.xyz);
    let world_radius = inst.bounds.w * max(sx, max(sy, sz));

    let visible = sphere_in_frustum(world_center, world_radius);

    indirect[idx] = DrawIndexedIndirect(
        dc.index_count,
        select(0u, 1u, visible),
        dc.first_index,
        dc.vertex_offset,
        dc.instance_id,
    );
}
