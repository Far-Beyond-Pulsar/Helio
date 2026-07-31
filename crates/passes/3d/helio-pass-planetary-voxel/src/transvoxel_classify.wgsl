const PAGE_EDGE: u32 = 32u;
const SAMPLE_EDGE: u32 = 34u;
const PAGE_CELL_COUNT: u32 = 32768u;

struct GpuTransvoxelDispatch {
    dirty_microbricks_low: u32,
    dirty_microbricks_high: u32,
    generation_low: u32,
    generation_high: u32,
    cell_count: u32,
    max_vertices: u32,
    max_indices: u32,
    scan_block_count: u32,
};

struct GpuTransvoxelCell {
    packed_case_class_counts: u32,
    generation_low: u32,
    generation_high: u32,
    _pad: u32,
};

struct GpuTransvoxelClassifyCounters {
    visited_cells: atomic<u32>,
    active_cells: atomic<u32>,
    vertices: atomic<u32>,
    triangles: atomic<u32>,
};

@group(0) @binding(0)
var<uniform> dispatch: GpuTransvoxelDispatch;
@group(0) @binding(1)
var<storage, read> samples: array<u32>;
@group(0) @binding(2)
var<storage, read> regular_cell_class: array<u32>;
@group(0) @binding(3)
var<storage, read> regular_geometry_counts: array<u32>;
@group(0) @binding(4)
var<storage, read_write> cells: array<GpuTransvoxelCell>;
@group(0) @binding(5)
var<storage, read_write> counters: GpuTransvoxelClassifyCounters;

const REGULAR_CORNERS: array<vec3<u32>, 8> = array<vec3<u32>, 8>(
    vec3<u32>(0u, 0u, 0u),
    vec3<u32>(1u, 0u, 0u),
    vec3<u32>(0u, 1u, 0u),
    vec3<u32>(1u, 1u, 0u),
    vec3<u32>(0u, 0u, 1u),
    vec3<u32>(1u, 0u, 1u),
    vec3<u32>(0u, 1u, 1u),
    vec3<u32>(1u, 1u, 1u),
);

fn sample_index(position: vec3<u32>) -> u32 {
    return position.x + position.y * SAMPLE_EDGE + position.z * SAMPLE_EDGE * SAMPLE_EDGE;
}

fn is_solid(word: u32) -> bool {
    let density_bits = word & 0xffffu;
    let sign_extension = select(0u, 0xffff0000u, (density_bits & 0x8000u) != 0u);
    return bitcast<i32>(density_bits | sign_extension) <= 0;
}

fn is_dirty_microbrick(index: u32) -> bool {
    if index < 32u {
        return (dispatch.dirty_microbricks_low & (1u << index)) != 0u;
    }
    return (dispatch.dirty_microbricks_high & (1u << (index - 32u))) != 0u;
}

@compute @workgroup_size(64, 1, 1)
fn classify_regular_cells(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let linear = global_id.x;
    if linear >= dispatch.cell_count || linear >= PAGE_CELL_COUNT {
        return;
    }

    let x = linear % PAGE_EDGE;
    let y = (linear / PAGE_EDGE) % PAGE_EDGE;
    let z = linear / (PAGE_EDGE * PAGE_EDGE);
    let microbrick = x / 8u + (y / 8u) * 4u + (z / 8u) * 16u;
    if !is_dirty_microbrick(microbrick) {
        return;
    }
    atomicAdd(&counters.visited_cells, 1u);

    let sample_origin = vec3<u32>(x + 1u, y + 1u, z + 1u);
    var case_index = 0u;
    for (var corner = 0u; corner < 8u; corner += 1u) {
        let word = samples[sample_index(sample_origin + REGULAR_CORNERS[corner])];
        if is_solid(word) {
            case_index |= 1u << corner;
        }
    }

    let class_index = regular_cell_class[case_index];
    let geometry_counts = regular_geometry_counts[class_index];
    let vertex_count = geometry_counts >> 4u;
    let triangle_count = geometry_counts & 0x0fu;
    let packed_case_class_counts = case_index
        | (class_index << 8u)
        | (vertex_count << 16u)
        | (triangle_count << 24u)
        | 0x80000000u;
    cells[linear] = GpuTransvoxelCell(
        packed_case_class_counts,
        dispatch.generation_low,
        dispatch.generation_high,
        0u,
    );
    if vertex_count != 0u {
        atomicAdd(&counters.active_cells, 1u);
        atomicAdd(&counters.vertices, vertex_count);
        atomicAdd(&counters.triangles, triangle_count);
    }
}
