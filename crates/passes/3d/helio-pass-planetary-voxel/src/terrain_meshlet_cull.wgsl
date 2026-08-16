struct Camera {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    position_near: vec4<f32>,
    forward_far: vec4<f32>,
    jitter_frame: vec4<f32>,
    prev_view_proj: mat4x4<f32>,
}

struct GpuTerrainCullUniforms {
    meshlet_capacity: u32,
    draw_capacity: u32,
    surface_kind: u32,
    _pad: u32,
}

struct GpuSurfaceState {
    generation_low: u32,
    generation_high: u32,
    valid: u32,
    revision: u32,
    regular_first_vertex: u32,
    regular_vertex_count: u32,
    regular_first_index: u32,
    regular_index_count: u32,
    regular_first_meshlet: u32,
    regular_meshlet_count: u32,
    transition_first_vertex: u32,
    transition_vertex_count: u32,
    transition_first_index: u32,
    transition_index_count: u32,
    transition_first_meshlet: u32,
    transition_meshlet_count: u32,
    transition_mask: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct GpuDrawPage {
    relative_lod0_cell_min_x: vec2<u32>,
    relative_lod0_cell_min_y: vec2<u32>,
    relative_lod0_cell_min_z: vec2<u32>,
    lod: u32,
    _pad0: u32,
    camera_relative_m: vec3<f32>,
    lod0_cell_size_m: f32,
    generation_low: u32,
    generation_high: u32,
    transition_mask: u32,
    visible: u32,
}

fn i64_words_to_f32(value: vec2<u32>) -> f32 {
    if (value.y & 0x80000000u) == 0u {
        return f32(value.y) * 4294967296.0 + f32(value.x);
    }
    let magnitude_low = ~value.x + 1u;
    let carry = select(0u, 1u, magnitude_low == 0u);
    let magnitude_high = ~value.y + carry;
    return -(f32(magnitude_high) * 4294967296.0 + f32(magnitude_low));
}

fn page_relative_min(page: GpuDrawPage) -> vec3<f32> {
    return vec3<f32>(
        i64_words_to_f32(page.relative_lod0_cell_min_x),
        i64_words_to_f32(page.relative_lod0_cell_min_y),
        i64_words_to_f32(page.relative_lod0_cell_min_z),
    );
}

struct GpuTerrainMeshlet {
    first_index: u32,
    index_count: u32,
    first_vertex: u32,
    vertex_count: u32,
    bounds_offset: u32,
    generation_low: u32,
    generation_high: u32,
    flags: u32,
}

struct GpuTerrainMeshletBounds {
    center: vec3<f32>,
    radius: f32,
    cone_apex: vec3<f32>,
    cone_cutoff: f32,
    cone_axis: vec3<f32>,
    _pad: f32,
}

struct DrawIndexedIndirectArgs {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

struct GpuTerrainDraw {
    page_slot: u32,
    meshlet_index: u32,
    surface_kind: u32,
    lod: u32,
}

@group(0) @binding(0) var<storage, read> cameras: array<Camera, 2>;
@group(0) @binding(1) var<uniform> cull: GpuTerrainCullUniforms;
@group(0) @binding(2) var<storage, read> surface_states: array<GpuSurfaceState>;
@group(0) @binding(3) var<storage, read> draw_pages: array<GpuDrawPage>;
@group(0) @binding(4) var<storage, read> meshlets: array<GpuTerrainMeshlet>;
@group(0) @binding(5) var<storage, read> meshlet_bounds: array<GpuTerrainMeshletBounds>;
@group(0) @binding(6) var<storage, read_write> indirect: array<DrawIndexedIndirectArgs>;
@group(0) @binding(7) var<storage, read_write> draws: array<GpuTerrainDraw>;
// [regular draws, transition draws, overflow, stale, frustum rejects,
//  cone rejects, invisible/invalid candidates, reserved]
@group(0) @binding(8) var<storage, read_write> counters: array<atomic<u32>>;

fn normalized_plane(plane: vec4<f32>) -> vec4<f32> {
    let magnitude = length(plane.xyz);
    if magnitude <= 1.0e-20 {
        return vec4<f32>(0.0);
    }
    return plane / magnitude;
}

fn sphere_visible(center: vec3<f32>, radius: f32) -> bool {
    let vp = cameras[0].view_proj;
    let planes = array<vec4<f32>, 6>(
        normalized_plane(vec4<f32>(
            vp[0][3] + vp[0][0],
            vp[1][3] + vp[1][0],
            vp[2][3] + vp[2][0],
            vp[3][3] + vp[3][0],
        )),
        normalized_plane(vec4<f32>(
            vp[0][3] - vp[0][0],
            vp[1][3] - vp[1][0],
            vp[2][3] - vp[2][0],
            vp[3][3] - vp[3][0],
        )),
        normalized_plane(vec4<f32>(
            vp[0][3] + vp[0][1],
            vp[1][3] + vp[1][1],
            vp[2][3] + vp[2][1],
            vp[3][3] + vp[3][1],
        )),
        normalized_plane(vec4<f32>(
            vp[0][3] - vp[0][1],
            vp[1][3] - vp[1][1],
            vp[2][3] - vp[2][1],
            vp[3][3] - vp[3][1],
        )),
        normalized_plane(vec4<f32>(
            vp[0][2],
            vp[1][2],
            vp[2][2],
            vp[3][2],
        )),
        normalized_plane(vec4<f32>(
            vp[0][3] - vp[0][2],
            vp[1][3] - vp[1][2],
            vp[2][3] - vp[2][2],
            vp[3][3] - vp[3][2],
        )),
    );
    for (var plane_index = 0u; plane_index < 6u; plane_index += 1u) {
        let plane = planes[plane_index];
        if dot(plane.xyz, center) + plane.w < -radius {
            return false;
        }
    }
    return true;
}

fn page_local_to_world(page: GpuDrawPage, position: vec3<f32>) -> vec3<f32> {
    let lod_scale = exp2(f32(page.lod));
    let relative_cell = page_relative_min(page) + position * lod_scale;
    return relative_cell * page.lod0_cell_size_m - page.camera_relative_m;
}

@compute @workgroup_size(64)
fn cull_meshlets(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= cull.meshlet_capacity || id.x >= arrayLength(&meshlets) {
        return;
    }
    let meshlet_index = id.x;
    let meshlet = meshlets[meshlet_index];
    let page_slot = meshlet.flags;
    if page_slot >= arrayLength(&surface_states) ||
        page_slot >= arrayLength(&draw_pages) {
        return;
    }

    let state = surface_states[page_slot];
    let page = draw_pages[page_slot];
    if state.valid == 0u || page.visible == 0u {
        atomicAdd(&counters[6], 1u);
        return;
    }
    let first_meshlet = select(
        state.regular_first_meshlet,
        state.transition_first_meshlet,
        cull.surface_kind != 0u,
    );
    let meshlet_count = select(
        state.regular_meshlet_count,
        state.transition_meshlet_count,
        cull.surface_kind != 0u,
    );
    if meshlet_index < first_meshlet || meshlet_index >= first_meshlet + meshlet_count {
        return;
    }
    if meshlet.bounds_offset >= arrayLength(&meshlet_bounds) ||
        meshlet.generation_low != state.generation_low ||
        meshlet.generation_high != state.generation_high ||
        state.generation_low != page.generation_low ||
        state.generation_high != page.generation_high {
        atomicAdd(&counters[3], 1u);
        return;
    }

    let bounds = meshlet_bounds[meshlet.bounds_offset];
    let lod_scale = exp2(f32(page.lod)) * page.lod0_cell_size_m;
    let center_world = page_local_to_world(page, bounds.center);
    let radius_world = max(bounds.radius * lod_scale, 0.0);
    if !sphere_visible(center_world, radius_world) {
        atomicAdd(&counters[4], 1u);
        return;
    }

    let apex_world = page_local_to_world(page, bounds.cone_apex);
    let camera_to_apex = apex_world - cameras[0].position_near.xyz;
    let apex_distance_squared = dot(camera_to_apex, camera_to_apex);
    let guard_radius = radius_world * 1.5;
    let center_distance_squared =
        dot(center_world - cameras[0].position_near.xyz, center_world - cameras[0].position_near.xyz);
    if bounds.cone_cutoff <= 1.0 &&
        center_distance_squared > guard_radius * guard_radius &&
        apex_distance_squared > 1.0e-12 &&
        dot(camera_to_apex * inverseSqrt(apex_distance_squared), bounds.cone_axis)
            >= bounds.cone_cutoff {
        atomicAdd(&counters[5], 1u);
        return;
    }

    let count_index = min(cull.surface_kind, 1u);
    let draw_slot = atomicAdd(&counters[count_index], 1u);
    let capacity = min(
        cull.draw_capacity,
        min(arrayLength(&indirect), arrayLength(&draws)),
    );
    if draw_slot >= capacity {
        atomicAdd(&counters[2], 1u);
        return;
    }
    indirect[draw_slot] = DrawIndexedIndirectArgs(
        meshlet.index_count,
        1u,
        meshlet.first_index,
        i32(meshlet.first_vertex),
        draw_slot,
    );
    draws[draw_slot] = GpuTerrainDraw(
        page_slot,
        meshlet_index,
        cull.surface_kind,
        page.lod,
    );
}
