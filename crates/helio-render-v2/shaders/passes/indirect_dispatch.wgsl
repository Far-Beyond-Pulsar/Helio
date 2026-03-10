/// Compute shader: GPU-driven frustum culling → non-compacting indirect buffer
///
/// One thread per draw call.  Visible draws write instance_count=1 at their
/// fixed slot; culled draws write instance_count=0.  No atomics, no prefix
/// sum, no GPU readback — O(1) CPU cost for any scene size.
///
/// `first_instance = dc.slot` so the vertex shader reads
/// `instance_data[@builtin(instance_index)]` for the model transform.

struct DrawIndexedIndirect {
    index_count:    u32,
    instance_count: u32,
    first_index:    u32,
    base_vertex:    i32,
    first_instance: u32,
}

/// Per-draw call input data.  Must exactly match `GpuDrawCall` in mesh.rs.
/// 32 bytes, 16-byte struct alignment.
struct GpuDrawCall {
    slot:          u32,    // → first_instance
    first_index:   u32,
    base_vertex:   i32,
    index_count:   u32,
    bounds_center: vec3f,  // offset 16 (vec3f alignment 16, no padding needed)
    bounds_radius: f32,
}

struct CameraUniforms {
    view_proj:     mat4x4f,
    position:      vec3f,
    time:          f32,
    view_proj_inv: mat4x4f,
}

struct DrawCallParams {
    draw_count: u32,
}

// Group 0: per-pass resources (draw calls, indirect output, camera for culling)
@group(0) @binding(0) var<storage, read>       draw_calls: array<GpuDrawCall>;
@group(0) @binding(1) var<storage, read_write> indirect:   array<DrawIndexedIndirect>;
@group(0) @binding(2) var<uniform>             camera:     CameraUniforms;
@group(0) @binding(3) var<uniform>             params:     DrawCallParams;

// ── Frustum plane extraction (Gribb-Hartmann, wgpu/Vulkan NDC z∈[0,1]) ───────
fn extract_frustum_planes(m: mat4x4f) -> array<vec4f, 6> {
    let r0 = vec4f(m[0][0], m[1][0], m[2][0], m[3][0]);
    let r1 = vec4f(m[0][1], m[1][1], m[2][1], m[3][1]);
    let r2 = vec4f(m[0][2], m[1][2], m[2][2], m[3][2]);
    let r3 = vec4f(m[0][3], m[1][3], m[2][3], m[3][3]);
    var planes: array<vec4f, 6>;
    planes[0] = r3 + r0;  // left
    planes[1] = r3 - r0;  // right
    planes[2] = r3 + r1;  // bottom
    planes[3] = r3 - r1;  // top
    planes[4] = r2;        // near (z >= 0 in Vulkan NDC)
    planes[5] = r3 - r2;  // far
    return planes;
}

/// Bounds-based sphere-vs-frustum test.  Returns false only when the sphere
/// is definitively outside a frustum plane (conservative: no false culls).
fn frustum_cull_sphere(planes: array<vec4f, 6>, centre: vec3f, radius: f32) -> bool {
    for (var i = 0u; i < 6u; i++) {
        let n    = planes[i].xyz;
        let dist = dot(n, centre) + planes[i].w;
        if dist < -(radius * length(n)) { return false; }
    }
    return true;
}

@compute @workgroup_size(64)
fn build_indirect_buffers(@builtin(global_invocation_id) gid: vec3u) {
    let draw_idx = gid.x;
    // Bound against the ACTIVE draw count, not the full buffer capacity.
    // draw_call_buffer is pre-allocated to 16K but only draw_count entries are valid.
    if draw_idx >= params.draw_count { return; }

    let dc = draw_calls[draw_idx];
    let planes  = extract_frustum_planes(camera.view_proj);
    let visible = frustum_cull_sphere(planes, dc.bounds_center, dc.bounds_radius);

    // Non-compacting write: every slot is always written.
    // Culled → instance_count=0 (GPU skips the draw for free).
    // Visible → instance_count=1, first_instance=slot (vertex shader reads instance_data[slot]).
    indirect[draw_idx] = DrawIndexedIndirect(
        dc.index_count,
        select(0u, 1u, visible),
        dc.first_index,
        dc.base_vertex,
        select(0u, dc.slot, visible),
    );
}
