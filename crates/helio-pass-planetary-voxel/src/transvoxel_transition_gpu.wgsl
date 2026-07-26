const FACE_COUNT: u32 = 6u;
const FACE_CELL_EDGE: u32 = 32u;
const CELLS_PER_FACE: u32 = 1024u;
const CELL_COUNT: u32 = 6144u;
const SLAB_EDGE: u32 = 67u;
const SLAB_LAYER_STRIDE: u32 = 4489u;
const SLAB_FACE_STRIDE: u32 = 13467u;
const SCAN_WORKGROUP_SIZE: u32 = 256u;

const CLASS_TABLE_OFFSET: u32 = 0u;
const GEOMETRY_TABLE_OFFSET: u32 = 512u;
const VERTEX_TABLE_OFFSET: u32 = 568u;
const VERTEX_TABLE_STRIDE: u32 = 12u;
const TOPOLOGY_TABLE_OFFSET: u32 = 6712u;
const TOPOLOGY_TABLE_STRIDE: u32 = 36u;

const CASE_WEIGHTS: array<u32, 9> = array<u32, 9>(
    0x001u, 0x002u, 0x004u,
    0x080u, 0x100u, 0x008u,
    0x040u, 0x020u, 0x010u,
);

const FULL_SAMPLE_UV: array<vec2<u32>, 9> = array<vec2<u32>, 9>(
    vec2<u32>(0u, 0u), vec2<u32>(1u, 0u), vec2<u32>(2u, 0u),
    vec2<u32>(0u, 1u), vec2<u32>(1u, 1u), vec2<u32>(2u, 1u),
    vec2<u32>(0u, 2u), vec2<u32>(1u, 2u), vec2<u32>(2u, 2u),
);

const DUPLICATE_FULL_CORNERS: array<u32, 4> = array<u32, 4>(0u, 2u, 6u, 8u);

const FACE_ORIGINS: array<vec3<f32>, 6> = array<vec3<f32>, 6>(
    vec3<f32>(0.0, 0.0, 32.0),
    vec3<f32>(32.0, 0.0, 0.0),
    vec3<f32>(32.0, 0.0, 0.0),
    vec3<f32>(0.0, 32.0, 0.0),
    vec3<f32>(0.0, 32.0, 0.0),
    vec3<f32>(0.0, 0.0, 32.0),
);

const FACE_U_AXES: array<vec3<f32>, 6> = array<vec3<f32>, 6>(
    vec3<f32>(0.0, 1.0, 0.0),
    vec3<f32>(0.0, 1.0, 0.0),
    vec3<f32>(0.0, 0.0, 1.0),
    vec3<f32>(0.0, 0.0, 1.0),
    vec3<f32>(1.0, 0.0, 0.0),
    vec3<f32>(1.0, 0.0, 0.0),
);

const FACE_V_AXES: array<vec3<f32>, 6> = array<vec3<f32>, 6>(
    vec3<f32>(0.0, 0.0, -1.0),
    vec3<f32>(0.0, 0.0, 1.0),
    vec3<f32>(-1.0, 0.0, 0.0),
    vec3<f32>(1.0, 0.0, 0.0),
    vec3<f32>(0.0, -1.0, 0.0),
    vec3<f32>(0.0, 1.0, 0.0),
);

const FACE_OUTWARD_AXES: array<vec3<f32>, 6> = array<vec3<f32>, 6>(
    vec3<f32>(-1.0, 0.0, 0.0),
    vec3<f32>(1.0, 0.0, 0.0),
    vec3<f32>(0.0, -1.0, 0.0),
    vec3<f32>(0.0, 1.0, 0.0),
    vec3<f32>(0.0, 0.0, -1.0),
    vec3<f32>(0.0, 0.0, 1.0),
);

struct GpuTransvoxelTransitionDispatch {
    transition_mask: u32,
    generation_low: u32,
    generation_high: u32,
    cell_count: u32,
    max_vertices: u32,
    max_indices: u32,
    scan_block_count: u32,
    _pad: u32,
};

struct GpuTransvoxelTransitionCell {
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

struct GpuTransvoxelTransitionCounters {
    active_cells: atomic<u32>,
    active_faces: atomic<u32>,
    required_vertices: atomic<u32>,
    required_indices: atomic<u32>,
    emitted_vertices: atomic<u32>,
    emitted_indices: atomic<u32>,
    vertex_overflow: atomic<u32>,
    index_overflow: atomic<u32>,
    completed: atomic<u32>,
    _pad: array<u32, 3>,
};

@group(0) @binding(0)
var<uniform> dispatch: GpuTransvoxelTransitionDispatch;
@group(0) @binding(1)
var<storage, read> samples: array<u32>;
@group(0) @binding(2)
var<storage, read> tables: array<u32>;
@group(0) @binding(3)
var<storage, read_write> cells: array<GpuTransvoxelTransitionCell>;
@group(0) @binding(4)
var<storage, read_write> cell_offsets: array<GpuTransvoxelCellOffset>;
@group(0) @binding(5)
var<storage, read_write> scan_blocks: array<GpuTransvoxelScanBlock>;
@group(0) @binding(6)
var<storage, read_write> vertices: array<GpuTerrainVertex>;
@group(0) @binding(7)
var<storage, read_write> indices: array<u32>;
@group(0) @binding(8)
var<storage, read_write> counters: GpuTransvoxelTransitionCounters;

var<workgroup> vertex_scan: array<u32, 256>;
var<workgroup> index_scan: array<u32, 256>;

fn density(word: u32) -> f32 {
    let density_bits = word & 0xffffu;
    let sign_extension = select(0u, 0xffff0000u, (density_bits & 0x8000u) != 0u);
    return f32(bitcast<i32>(density_bits | sign_extension));
}

fn is_solid(word: u32) -> bool {
    return density(word) <= 0.0;
}

fn slab_index(face: u32, u: u32, v: u32, layer: u32) -> u32 {
    return face * SLAB_FACE_STRIDE + u + v * SLAB_EDGE + layer * SLAB_LAYER_STRIDE;
}

fn sample_word(face: u32, u: u32, v: u32, layer: u32) -> u32 {
    return samples[slab_index(face, u, v, layer)];
}

fn cell_is_current(cell: GpuTransvoxelTransitionCell) -> bool {
    return (cell.packed_case_class_counts & 0x80000000u) != 0u
        && cell.generation_low == dispatch.generation_low
        && cell.generation_high == dispatch.generation_high;
}

fn cell_vertex_count(cell: GpuTransvoxelTransitionCell) -> u32 {
    return (cell.packed_case_class_counts >> 17u) & 0x0fu;
}

fn cell_triangle_count(cell: GpuTransvoxelTransitionCell) -> u32 {
    return (cell.packed_case_class_counts >> 21u) & 0x0fu;
}

@compute @workgroup_size(64, 1, 1)
fn classify_transition_cells(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let linear = global_id.x;
    if linear == 0u {
        atomicStore(&counters.active_faces, countOneBits(dispatch.transition_mask & 0x3fu));
    }
    if linear >= dispatch.cell_count || linear >= CELL_COUNT {
        return;
    }
    let face = linear / CELLS_PER_FACE;
    if (dispatch.transition_mask & (1u << face)) == 0u {
        cells[linear] = GpuTransvoxelTransitionCell(0u, 0u, 0u, 0u);
        return;
    }
    let face_cell = linear % CELLS_PER_FACE;
    let cell_u = face_cell % FACE_CELL_EDGE;
    let cell_v = face_cell / FACE_CELL_EDGE;
    var case_index = 0u;
    for (var sample = 0u; sample < 9u; sample += 1u) {
        let uv = FULL_SAMPLE_UV[sample];
        let word = sample_word(face, cell_u * 2u + uv.x + 1u, cell_v * 2u + uv.y + 1u, 1u);
        if is_solid(word) {
            case_index |= CASE_WEIGHTS[sample];
        }
    }
    let class_code = tables[CLASS_TABLE_OFFSET + case_index];
    let class_index = class_code & 0x7fu;
    let geometry_counts = tables[GEOMETRY_TABLE_OFFSET + class_index];
    let vertex_count = geometry_counts >> 4u;
    let triangle_count = geometry_counts & 0x0fu;
    cells[linear] = GpuTransvoxelTransitionCell(
        case_index
            | (class_code << 9u)
            | (vertex_count << 17u)
            | (triangle_count << 21u)
            | 0x80000000u,
        dispatch.generation_low,
        dispatch.generation_high,
        0u,
    );
    if vertex_count != 0u {
        atomicAdd(&counters.active_cells, 1u);
    }
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
fn scan_transition_cells(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let linear = global_id.x;
    let local_index = local_id.x;
    var vertex_count = 0u;
    var index_count = 0u;
    var current = false;
    if linear < dispatch.cell_count && linear < CELL_COUNT {
        let cell = cells[linear];
        let face = linear / CELLS_PER_FACE;
        current = (dispatch.transition_mask & (1u << face)) != 0u
            && cell_is_current(cell);
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
fn scan_transition_blocks(@builtin(local_invocation_id) local_id: vec3<u32>) {
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
        atomicStore(&counters.required_vertices, required_vertices);
        atomicStore(&counters.required_indices, required_indices);
        atomicStore(&counters.vertex_overflow, select(0u, 1u, required_vertices > dispatch.max_vertices));
        atomicStore(&counters.index_overflow, select(0u, 1u, required_indices > dispatch.max_indices));
        atomicStore(&counters.completed, 1u);
    }
}

fn full_corner(corner: u32) -> u32 {
    if corner < 9u {
        return corner;
    }
    return DUPLICATE_FULL_CORNERS[corner - 9u];
}

fn corner_depth(corner: u32) -> f32 {
    return select(1.0, 0.0, corner < 9u);
}

fn sample_gradient(face: u32, u: u32, v: u32) -> vec3<f32> {
    let du = density(sample_word(face, u + 1u, v, 1u))
        - density(sample_word(face, u - 1u, v, 1u));
    let dv = density(sample_word(face, u, v + 1u, 1u))
        - density(sample_word(face, u, v - 1u, 1u));
    let outward = density(sample_word(face, u, v, 2u))
        - density(sample_word(face, u, v, 0u));
    return FACE_U_AXES[face] * du
        + FACE_V_AXES[face] * dv
        + FACE_OUTWARD_AXES[face] * outward;
}

fn normalized_or_outward(value: vec3<f32>, outward: vec3<f32>) -> vec3<f32> {
    let squared_length = dot(value, value);
    if squared_length <= 1.0e-12 {
        return outward;
    }
    return value * inverseSqrt(squared_length);
}

@compute @workgroup_size(64, 1, 1)
fn emit_transition_cells(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let linear = global_id.x;
    if linear >= dispatch.cell_count || linear >= CELL_COUNT {
        return;
    }
    let face = linear / CELLS_PER_FACE;
    if (dispatch.transition_mask & (1u << face)) == 0u {
        return;
    }
    if atomicLoad(&counters.vertex_overflow) != 0u
        || atomicLoad(&counters.index_overflow) != 0u {
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
    let case_index = cell.packed_case_class_counts & 0x1ffu;
    let class_code = (cell.packed_case_class_counts >> 9u) & 0xffu;
    let class_index = class_code & 0x7fu;
    let vertex_count = cell_vertex_count(cell);
    let index_count = cell_triangle_count(cell) * 3u;
    let face_cell = linear % CELLS_PER_FACE;
    let cell_u = face_cell % FACE_CELL_EDGE;
    let cell_v = face_cell / FACE_CELL_EDGE;
    let origin = FACE_ORIGINS[face];
    let u_axis = FACE_U_AXES[face];
    let v_axis = FACE_V_AXES[face];
    let outward_axis = FACE_OUTWARD_AXES[face];

    for (var vertex = 0u; vertex < vertex_count; vertex += 1u) {
        let code = tables[VERTEX_TABLE_OFFSET + case_index * VERTEX_TABLE_STRIDE + vertex];
        let first_corner = (code >> 4u) & 0x0fu;
        let second_corner = code & 0x0fu;
        let first_full = full_corner(first_corner);
        let second_full = full_corner(second_corner);
        let first_uv = FULL_SAMPLE_UV[first_full];
        let second_uv = FULL_SAMPLE_UV[second_full];
        let first_u = cell_u * 2u + first_uv.x + 1u;
        let first_v = cell_v * 2u + first_uv.y + 1u;
        let second_u = cell_u * 2u + second_uv.x + 1u;
        let second_v = cell_v * 2u + second_uv.y + 1u;
        let first_word = sample_word(face, first_u, first_v, 1u);
        let second_word = sample_word(face, second_u, second_v, 1u);
        let first_density = density(first_word);
        let second_density = density(second_word);
        let denominator = first_density - second_density;
        var interpolation = 0.5;
        if abs(denominator) > 1.0e-12 {
            interpolation = clamp(first_density / denominator, 0.0, 1.0);
        }
        let first_face_uv = vec2<f32>(first_uv) * 0.5;
        let second_face_uv = vec2<f32>(second_uv) * 0.5;
        let face_uv = vec2<f32>(f32(cell_u), f32(cell_v))
            + mix(first_face_uv, second_face_uv, interpolation);
        let depth_fraction = mix(
            corner_depth(first_corner),
            corner_depth(second_corner),
            interpolation,
        );
        let normal = normalized_or_outward(
            mix(
                sample_gradient(face, first_u, first_v),
                sample_gradient(face, second_u, second_v),
                interpolation,
            ),
            outward_axis,
        );
        let primary = origin + u_axis * face_uv.x + v_axis * face_uv.y;
        let inward_offset = -outward_axis * 0.25 * depth_fraction;
        let projected_offset = inward_offset - normal * dot(inward_offset, normal);
        let material = select(
            (second_word >> 16u) & 0xffu,
            (first_word >> 16u) & 0xffu,
            first_density <= 0.0,
        );
        vertices[first_vertex + vertex] = GpuTerrainVertex(
            primary + projected_offset,
            material,
            normal,
            0u,
        );
    }

    let inverse_case = (class_code & 0x80u) != 0u;
    for (var index = 0u; index < index_count; index += 1u) {
        var table_index = index;
        if !inverse_case {
            let triangle_corner = index % 3u;
            if triangle_corner == 1u {
                table_index += 1u;
            } else if triangle_corner == 2u {
                table_index -= 1u;
            }
        }
        let local_vertex = tables[
            TOPOLOGY_TABLE_OFFSET + class_index * TOPOLOGY_TABLE_STRIDE + table_index
        ];
        indices[first_index + index] = first_vertex + local_vertex;
    }
    atomicAdd(&counters.emitted_vertices, vertex_count);
    atomicAdd(&counters.emitted_indices, index_count);
}
