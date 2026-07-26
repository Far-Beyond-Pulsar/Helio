const PAGE_EDGE: u32 = 32u;
const SAMPLE_EDGE: u32 = 34u;
const PAGE_CELL_COUNT: u32 = 32768u;
const SCAN_WORKGROUP_SIZE: u32 = 256u;
const REGULAR_VERTEX_TABLE_STRIDE: u32 = 12u;
const REGULAR_TOPOLOGY_TABLE_OFFSET: u32 = 3072u;
const REGULAR_TOPOLOGY_TABLE_STRIDE: u32 = 15u;

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

struct GpuTransvoxelCellOffset {
    first_vertex: u32,
    first_index: u32,
    generation_low: u32,
    generation_high: u32,
};

struct GpuTransvoxelScanBlock {
    vertex_count: u32,
    index_count: u32,
    first_vertex: u32,
    first_index: u32,
};

struct GpuTerrainVertex {
    position: vec3<f32>,
    material: u32,
    normal: vec3<f32>,
    flags: u32,
};

struct GpuTransvoxelEmissionCounters {
    required_vertices: atomic<u32>,
    required_indices: atomic<u32>,
    emitted_vertices: atomic<u32>,
    emitted_indices: atomic<u32>,
    vertex_overflow: atomic<u32>,
    index_overflow: atomic<u32>,
    completed: atomic<u32>,
    _pad: u32,
};

@group(0) @binding(0)
var<uniform> dispatch: GpuTransvoxelDispatch;
@group(0) @binding(1)
var<storage, read> samples: array<u32>;
@group(0) @binding(2)
var<storage, read> cells: array<GpuTransvoxelCell>;
@group(0) @binding(3)
var<storage, read_write> cell_offsets: array<GpuTransvoxelCellOffset>;
@group(0) @binding(4)
var<storage, read_write> scan_blocks: array<GpuTransvoxelScanBlock>;
@group(0) @binding(5)
var<storage, read> regular_tables: array<u32>;
@group(0) @binding(6)
var<storage, read_write> vertices: array<GpuTerrainVertex>;
@group(0) @binding(7)
var<storage, read_write> indices: array<u32>;
@group(0) @binding(8)
var<storage, read_write> emission: GpuTransvoxelEmissionCounters;

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

var<workgroup> vertex_scan: array<u32, 256>;
var<workgroup> index_scan: array<u32, 256>;

fn cell_is_current(cell: GpuTransvoxelCell) -> bool {
    return (cell.packed_case_class_counts & 0x80000000u) != 0u
        && cell.generation_low == dispatch.generation_low
        && cell.generation_high == dispatch.generation_high;
}

fn cell_vertex_count(cell: GpuTransvoxelCell) -> u32 {
    return (cell.packed_case_class_counts >> 16u) & 0xffu;
}

fn cell_triangle_count(cell: GpuTransvoxelCell) -> u32 {
    return (cell.packed_case_class_counts >> 24u) & 0x0fu;
}

fn inclusive_scan(local_index: u32) {
    var step = 1u;
    loop {
        if step >= SCAN_WORKGROUP_SIZE {
            break;
        }
        var add_vertices = 0u;
        var add_indices = 0u;
        if local_index >= step {
            add_vertices = vertex_scan[local_index - step];
            add_indices = index_scan[local_index - step];
        }
        workgroupBarrier();
        vertex_scan[local_index] += add_vertices;
        index_scan[local_index] += add_indices;
        workgroupBarrier();
        step <<= 1u;
    }
}

@compute @workgroup_size(256, 1, 1)
fn scan_regular_cells(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let linear = global_id.x;
    let local_index = local_id.x;
    var vertex_count = 0u;
    var index_count = 0u;
    var current = false;
    if linear < dispatch.cell_count && linear < PAGE_CELL_COUNT {
        let cell = cells[linear];
        current = cell_is_current(cell);
        if current {
            vertex_count = cell_vertex_count(cell);
            index_count = cell_triangle_count(cell) * 3u;
        }
    }

    vertex_scan[local_index] = vertex_count;
    index_scan[local_index] = index_count;
    workgroupBarrier();
    inclusive_scan(local_index);

    if current {
        cell_offsets[linear] = GpuTransvoxelCellOffset(
            vertex_scan[local_index] - vertex_count,
            index_scan[local_index] - index_count,
            dispatch.generation_low,
            dispatch.generation_high,
        );
    }
    if local_index == SCAN_WORKGROUP_SIZE - 1u {
        scan_blocks[workgroup_id.x] = GpuTransvoxelScanBlock(
            vertex_scan[local_index],
            index_scan[local_index],
            0u,
            0u,
        );
    }
}

@compute @workgroup_size(256, 1, 1)
fn scan_regular_blocks(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let local_index = local_id.x;
    var vertex_count = 0u;
    var index_count = 0u;
    if local_index < dispatch.scan_block_count {
        let block = scan_blocks[local_index];
        vertex_count = block.vertex_count;
        index_count = block.index_count;
    }
    vertex_scan[local_index] = vertex_count;
    index_scan[local_index] = index_count;
    workgroupBarrier();
    inclusive_scan(local_index);

    if local_index < dispatch.scan_block_count {
        scan_blocks[local_index].first_vertex = vertex_scan[local_index] - vertex_count;
        scan_blocks[local_index].first_index = index_scan[local_index] - index_count;
    }
    if local_index == SCAN_WORKGROUP_SIZE - 1u {
        let required_vertices = vertex_scan[local_index];
        let required_indices = index_scan[local_index];
        atomicStore(&emission.required_vertices, required_vertices);
        atomicStore(&emission.required_indices, required_indices);
        atomicStore(&emission.vertex_overflow, select(0u, 1u, required_vertices > dispatch.max_vertices));
        atomicStore(&emission.index_overflow, select(0u, 1u, required_indices > dispatch.max_indices));
        atomicStore(&emission.completed, 1u);
    }
}

fn sample_index(position: vec3<u32>) -> u32 {
    return position.x + position.y * SAMPLE_EDGE + position.z * SAMPLE_EDGE * SAMPLE_EDGE;
}

fn density(word: u32) -> f32 {
    let density_bits = word & 0xffffu;
    let sign_extension = select(0u, 0xffff0000u, (density_bits & 0x8000u) != 0u);
    return f32(bitcast<i32>(density_bits | sign_extension));
}

fn sample_density(position: vec3<u32>) -> f32 {
    return density(samples[sample_index(position)]);
}

fn density_difference(position: vec3<u32>, axis: u32) -> f32 {
    var lower = position;
    var upper = position;
    if position[axis] == 0u {
        upper[axis] += 1u;
        return sample_density(upper) - sample_density(position);
    }
    if position[axis] + 1u >= SAMPLE_EDGE {
        lower[axis] -= 1u;
        return sample_density(position) - sample_density(lower);
    }
    lower[axis] -= 1u;
    upper[axis] += 1u;
    return (sample_density(upper) - sample_density(lower)) * 0.5;
}

fn sample_gradient(position: vec3<u32>) -> vec3<f32> {
    return vec3<f32>(
        density_difference(position, 0u),
        density_difference(position, 1u),
        density_difference(position, 2u),
    );
}

fn normalized_or_up(value: vec3<f32>) -> vec3<f32> {
    let squared_length = dot(value, value);
    if squared_length <= 1.0e-12 {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    return value * inverseSqrt(squared_length);
}

@compute @workgroup_size(64, 1, 1)
fn emit_regular_cells(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let linear = global_id.x;
    if linear >= dispatch.cell_count || linear >= PAGE_CELL_COUNT {
        return;
    }
    if atomicLoad(&emission.vertex_overflow) != 0u || atomicLoad(&emission.index_overflow) != 0u {
        return;
    }
    let cell = cells[linear];
    if !cell_is_current(cell) {
        return;
    }
    let local_offset = cell_offsets[linear];
    if local_offset.generation_low != dispatch.generation_low
        || local_offset.generation_high != dispatch.generation_high {
        return;
    }
    let block = scan_blocks[linear / SCAN_WORKGROUP_SIZE];
    let first_vertex = block.first_vertex + local_offset.first_vertex;
    let first_index = block.first_index + local_offset.first_index;
    let case_index = cell.packed_case_class_counts & 0xffu;
    let class_index = (cell.packed_case_class_counts >> 8u) & 0xffu;
    let vertex_count = cell_vertex_count(cell);
    let index_count = cell_triangle_count(cell) * 3u;

    let x = linear % PAGE_EDGE;
    let y = (linear / PAGE_EDGE) % PAGE_EDGE;
    let z = linear / (PAGE_EDGE * PAGE_EDGE);
    let cell_position = vec3<f32>(f32(x), f32(y), f32(z));
    let sample_origin = vec3<u32>(x + 1u, y + 1u, z + 1u);
    for (var vertex = 0u; vertex < vertex_count; vertex += 1u) {
        let code = regular_tables[case_index * REGULAR_VERTEX_TABLE_STRIDE + vertex];
        let first_corner = (code >> 4u) & 0x0fu;
        let second_corner = code & 0x0fu;
        let first_sample = sample_origin + REGULAR_CORNERS[first_corner];
        let second_sample = sample_origin + REGULAR_CORNERS[second_corner];
        let first_word = samples[sample_index(first_sample)];
        let second_word = samples[sample_index(second_sample)];
        let first_density = density(first_word);
        let second_density = density(second_word);
        let denominator = first_density - second_density;
        var interpolation = 0.5;
        if abs(denominator) > 1.0e-12 {
            interpolation = clamp(first_density / denominator, 0.0, 1.0);
        }
        let first_position = vec3<f32>(REGULAR_CORNERS[first_corner]);
        let second_position = vec3<f32>(REGULAR_CORNERS[second_corner]);
        let position = cell_position + mix(first_position, second_position, interpolation);
        let gradient = mix(
            sample_gradient(first_sample),
            sample_gradient(second_sample),
            interpolation,
        );
        let material = select(
            (second_word >> 16u) & 0xffu,
            (first_word >> 16u) & 0xffu,
            first_density <= 0.0,
        );
        vertices[first_vertex + vertex] = GpuTerrainVertex(
            position,
            material,
            normalized_or_up(gradient),
            0u,
        );
    }

    for (var index = 0u; index < index_count; index += 1u) {
        let local_vertex = regular_tables[
            REGULAR_TOPOLOGY_TABLE_OFFSET
                + class_index * REGULAR_TOPOLOGY_TABLE_STRIDE
                + index
        ];
        indices[first_index + index] = first_vertex + local_vertex;
    }
    atomicAdd(&emission.emitted_vertices, vertex_count);
    atomicAdd(&emission.emitted_indices, index_count);
}
