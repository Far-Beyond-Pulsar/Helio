const BUILD_INDICES: u32 = 63u;
const CONE_EPSILON: f32 = 0.0001;

struct GpuSurfaceJob {
    slot: u32,
    resident_slot: u32,
    transition_mask: u32,
    generation_low: u32,
    generation_high: u32,
    revision: u32,
    regular_first_vertex: u32,
    regular_first_index: u32,
    regular_first_meshlet: u32,
    transition_first_vertex: u32,
    transition_first_index: u32,
    transition_first_meshlet: u32,
    _pad0: u32,
    _pad1: u32,
}

struct GpuPageMeta {
    relative_lod0_cell_min: vec3<i32>,
    lod: u32,
    slot: u32,
    generation_low: u32,
    generation_high: u32,
    transition_mask: u32,
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

struct GpuEmissionCounters {
    required_vertices: u32,
    required_indices: u32,
    emitted_vertices: u32,
    emitted_indices: u32,
    vertex_overflow: u32,
    index_overflow: u32,
    completed: u32,
    _pad: u32,
}

struct GpuTransitionCounters {
    active_cells: u32,
    active_faces: u32,
    required_vertices: u32,
    required_indices: u32,
    emitted_vertices: u32,
    emitted_indices: u32,
    vertex_overflow: u32,
    index_overflow: u32,
    completed: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct GpuTerrainVertex {
    position: vec3<f32>,
    material: u32,
    normal: vec3<f32>,
    flags: u32,
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

struct MeshletBuild {
    vertex_count: u32,
    _pad0: vec3<u32>,
    bounds: GpuTerrainMeshletBounds,
}

@group(0) @binding(0) var<uniform> job: GpuSurfaceJob;
@group(0) @binding(1) var<storage, read> page_metadata: array<GpuPageMeta>;
@group(0) @binding(2) var<storage, read> surface_states: array<GpuSurfaceState>;
@group(0) @binding(3) var<storage, read> regular_counters: GpuEmissionCounters;
@group(0) @binding(4) var<storage, read> regular_vertices: array<GpuTerrainVertex>;
@group(0) @binding(5) var<storage, read> regular_indices: array<u32>;
@group(0) @binding(6) var<storage, read_write> regular_meshlets: array<GpuTerrainMeshlet>;
@group(0) @binding(7) var<storage, read_write> regular_bounds: array<GpuTerrainMeshletBounds>;
@group(0) @binding(8) var<storage, read> transition_counters: GpuTransitionCounters;
@group(0) @binding(9) var<storage, read> transition_vertices: array<GpuTerrainVertex>;
@group(0) @binding(10) var<storage, read> transition_indices: array<u32>;
@group(0) @binding(11) var<storage, read_write> transition_meshlets: array<GpuTerrainMeshlet>;
@group(0) @binding(12) var<storage, read_write> transition_bounds: array<GpuTerrainMeshletBounds>;

fn metadata_is_current() -> bool {
    let page = page_metadata[job.resident_slot];
    return page.slot == job.resident_slot &&
        page.generation_low == job.generation_low &&
        page.generation_high == job.generation_high;
}

fn safe_normal(value: vec3<f32>) -> vec3<f32> {
    let length_squared = dot(value, value);
    if length_squared <= 1.0e-20 { return vec3<f32>(0.0); }
    return value * inverseSqrt(length_squared);
}

fn compute_meshlet(
    positions: array<vec3<f32>, 63>,
    vertex_ids: array<u32, 63>,
    index_count: u32,
) -> MeshletBuild {
    var unique_vertices = 0u;
    var minimum = positions[0];
    var maximum = positions[0];
    for (var i = 0u; i < index_count; i += 1u) {
        minimum = min(minimum, positions[i]);
        maximum = max(maximum, positions[i]);
        var is_unique = true;
        for (var previous = 0u; previous < i; previous += 1u) {
            if vertex_ids[previous] == vertex_ids[i] {
                is_unique = false;
                break;
            }
        }
        if is_unique { unique_vertices += 1u; }
    }

    let center = (minimum + maximum) * 0.5;
    var radius = 0.0;
    for (var i = 0u; i < index_count; i += 1u) {
        radius = max(radius, length(positions[i] - center));
    }

    var normal_sum = vec3<f32>(0.0);
    var valid_triangles = 0u;
    for (var i = 0u; i < index_count; i += 3u) {
        let normal = safe_normal(cross(positions[i + 1u] - positions[i], positions[i + 2u] - positions[i]));
        if dot(normal, normal) > 0.0 {
            normal_sum += normal;
            valid_triangles += 1u;
        }
    }

    let axis = safe_normal(normal_sum);
    var minimum_dot = 1.0;
    if valid_triangles != 0u && dot(axis, axis) > 0.0 {
        for (var i = 0u; i < index_count; i += 3u) {
            let normal = safe_normal(cross(positions[i + 1u] - positions[i], positions[i + 2u] - positions[i]));
            if dot(normal, normal) > 0.0 {
                minimum_dot = min(minimum_dot, dot(normal, axis));
            }
        }
    }

    var apex = center;
    var cutoff = 1.0;
    var stored_axis = vec3<f32>(0.0);
    if valid_triangles != 0u && minimum_dot > 0.1 {
        var apex_distance = 0.0;
        for (var i = 0u; i < index_count; i += 3u) {
            let normal = safe_normal(cross(positions[i + 1u] - positions[i], positions[i + 2u] - positions[i]));
            let denominator = dot(axis, normal);
            if denominator > 0.0 {
                apex_distance = max(apex_distance, dot(center - positions[i], normal) / denominator);
            }
        }
        apex = center - axis * apex_distance;
        cutoff = min(1.0, sqrt(max(0.0, 1.0 - minimum_dot * minimum_dot)) + CONE_EPSILON);
        stored_axis = axis;
    }

    return MeshletBuild(
        unique_vertices,
        vec3<u32>(0u),
        GpuTerrainMeshletBounds(center, radius, apex, cutoff, stored_axis, 0.0),
    );
}

@compute @workgroup_size(64)
fn build_regular(@builtin(global_invocation_id) id: vec3<u32>) {
    if regular_counters.completed == 0u ||
        regular_counters.vertex_overflow != 0u ||
        regular_counters.index_overflow != 0u ||
        !metadata_is_current() { return; }
    let meshlet_count = (regular_counters.emitted_indices + BUILD_INDICES - 1u) / BUILD_INDICES;
    if id.x >= meshlet_count { return; }

    let local_first_index = id.x * BUILD_INDICES;
    let index_count = min(BUILD_INDICES, regular_counters.emitted_indices - local_first_index);
    let first_index = job.regular_first_index + local_first_index;
    let first_vertex = job.regular_first_vertex;
    let output_index = job.regular_first_meshlet + id.x;
    var positions: array<vec3<f32>, 63>;
    var vertex_ids: array<u32, 63>;
    for (var i = 0u; i < index_count; i += 1u) {
        vertex_ids[i] = regular_indices[first_index + i];
        positions[i] = regular_vertices[first_vertex + vertex_ids[i]].position;
    }
    let build = compute_meshlet(positions, vertex_ids, index_count);
    regular_meshlets[output_index] = GpuTerrainMeshlet(
        first_index, index_count, first_vertex, build.vertex_count, output_index,
        job.generation_low, job.generation_high, job.slot,
    );
    regular_bounds[output_index] = build.bounds;
}

@compute @workgroup_size(64)
fn build_transition(@builtin(global_invocation_id) id: vec3<u32>) {
    if transition_counters.completed == 0u ||
        transition_counters.vertex_overflow != 0u ||
        transition_counters.index_overflow != 0u ||
        !metadata_is_current() { return; }
    let meshlet_count = (transition_counters.emitted_indices + BUILD_INDICES - 1u) / BUILD_INDICES;
    if id.x >= meshlet_count { return; }

    let local_first_index = id.x * BUILD_INDICES;
    let index_count = min(BUILD_INDICES, transition_counters.emitted_indices - local_first_index);
    let first_index = job.transition_first_index + local_first_index;
    let first_vertex = job.transition_first_vertex;
    let output_index = job.transition_first_meshlet + id.x;
    var positions: array<vec3<f32>, 63>;
    var vertex_ids: array<u32, 63>;
    for (var i = 0u; i < index_count; i += 1u) {
        vertex_ids[i] = transition_indices[first_index + i];
        positions[i] = transition_vertices[first_vertex + vertex_ids[i]].position;
    }
    let build = compute_meshlet(positions, vertex_ids, index_count);
    transition_meshlets[output_index] = GpuTerrainMeshlet(
        first_index, index_count, first_vertex, build.vertex_count, output_index,
        job.generation_low, job.generation_high, job.slot,
    );
    transition_bounds[output_index] = build.bounds;
}
