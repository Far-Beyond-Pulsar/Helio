const SAMPLE_EDGE: u32 = 34u;
const SAMPLE_LAYER: u32 = 1156u;
const CELL_EDGE: u32 = 33u;
const CELL_LAYER: u32 = 1089u;
const CELL_COUNT: u32 = 35937u;
const OWNER_EDGE: u32 = 32u;
const OWNER_LAYER: u32 = 1024u;
const OWNER_COUNT: u32 = 32768u;
const SCAN_WORKGROUP_SIZE: u32 = 256u;
const INVALID_COMPONENT: u32 = 7u;
const INVALID_FAN: u32 = 15u;
const INVALID_MATERIAL: u32 = 256u;
const INDICES_PER_QUAD: u32 = 24u;
const FACE_PAIRING_TABLE_OFFSET: u32 = 512u;

const REGULAR_CORNERS: array<vec3<u32>, 8> = array<vec3<u32>, 8>(
    vec3<u32>(0u, 0u, 0u), vec3<u32>(1u, 0u, 0u),
    vec3<u32>(0u, 1u, 0u), vec3<u32>(1u, 1u, 0u),
    vec3<u32>(0u, 0u, 1u), vec3<u32>(1u, 0u, 1u),
    vec3<u32>(0u, 1u, 1u), vec3<u32>(1u, 1u, 1u),
);

const FIXTURE_BITS: array<u32, 8> = array<u32, 8>(0u, 1u, 3u, 2u, 4u, 5u, 7u, 6u);

const CUBE_EDGES: array<vec2<u32>, 12> = array<vec2<u32>, 12>(
    vec2<u32>(0u, 1u), vec2<u32>(2u, 3u),
    vec2<u32>(4u, 5u), vec2<u32>(6u, 7u),
    vec2<u32>(0u, 2u), vec2<u32>(1u, 3u),
    vec2<u32>(4u, 6u), vec2<u32>(5u, 7u),
    vec2<u32>(0u, 4u), vec2<u32>(1u, 5u),
    vec2<u32>(2u, 6u), vec2<u32>(3u, 7u),
);

// Corners are indexed by the two projected face coordinates (u + 2*v).
const FACE_CORNERS: array<vec4<u32>, 6> = array<vec4<u32>, 6>(
    vec4<u32>(0u, 2u, 4u, 6u), vec4<u32>(1u, 3u, 5u, 7u),
    vec4<u32>(0u, 1u, 4u, 5u), vec4<u32>(2u, 3u, 6u, 7u),
    vec4<u32>(0u, 1u, 2u, 3u), vec4<u32>(4u, 5u, 6u, 7u),
);

// Face-edge labels match the CPU oracle: 0=[0,1], 1=[1,3],
// 2=[2,3], 3=[0,2] in projected coordinates.
const FACE_CUBE_EDGES: array<vec4<u32>, 6> = array<vec4<u32>, 6>(
    vec4<u32>(4u, 10u, 6u, 8u), vec4<u32>(5u, 11u, 7u, 9u),
    vec4<u32>(0u, 9u, 2u, 8u), vec4<u32>(1u, 11u, 3u, 10u),
    vec4<u32>(0u, 5u, 1u, 4u), vec4<u32>(2u, 7u, 3u, 6u),
);

struct GpuManifoldDcDispatch {
    generation_low: u32,
    generation_high: u32,
    cell_count: u32,
    owner_count: u32,
    max_vertices: u32,
    max_indices: u32,
    cell_scan_blocks: u32,
    owner_scan_blocks: u32,
    position_quantization_steps: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
    _pad4: u32,
    _pad5: u32,
    _pad6: u32,
};

struct GpuManifoldDcCell {
    packed_case_component_fan_vertex_counts: u32,
    edge_fans_low: u32,
    edge_fans_high: u32,
    fan_material_counts_low: u32,
    fan_material_counts_high: u32,
    generation_low: u32,
    generation_high: u32,
    _pad: u32,
};

struct GpuManifoldDcOwner {
    packed_active_materials: u32,
    topology_vertex_count: u32,
    generation_low: u32,
    generation_high: u32,
    _pad: vec4<u32>,
};

struct GpuManifoldDcQuad {
    qef_vertices: vec4<u32>,
    midpoint_vertices: vec4<u32>,
    center_vertex: u32,
    material: u32,
    generation_low: u32,
    generation_high: u32,
};

struct ManifoldDcGridEdge {
    anchor: vec3<i32>,
    axis: u32,
    valid: u32,
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

struct GpuManifoldDcCounters {
    active_cells: atomic<u32>,
    active_components: atomic<u32>,
    active_edges: atomic<u32>,
    required_qef_vertices: atomic<u32>,
    required_topology_vertices: atomic<u32>,
    required_vertices: atomic<u32>,
    required_indices: atomic<u32>,
    emitted_vertices: atomic<u32>,
    emitted_indices: atomic<u32>,
    vertex_overflow: atomic<u32>,
    index_overflow: atomic<u32>,
    completed: atomic<u32>,
    topology_error: atomic<u32>,
    _pad: array<u32, 3>,
};

@group(0) @binding(0)
var<uniform> dispatch: GpuManifoldDcDispatch;
@group(0) @binding(1)
var<storage, read> samples: array<u32>;
@group(0) @binding(2)
var<storage, read> component_tables: array<u32>;
@group(0) @binding(3)
var<storage, read_write> cells: array<GpuManifoldDcCell>;
@group(0) @binding(4)
var<storage, read_write> cell_offsets: array<GpuTransvoxelCellOffset>;
@group(0) @binding(5)
var<storage, read_write> cell_blocks: array<GpuTransvoxelScanBlock>;
@group(0) @binding(6)
var<storage, read_write> owners: array<GpuManifoldDcOwner>;
@group(0) @binding(7)
var<storage, read_write> owner_offsets: array<GpuTransvoxelCellOffset>;
@group(0) @binding(8)
var<storage, read_write> owner_blocks: array<GpuTransvoxelScanBlock>;
@group(0) @binding(9)
var<storage, read_write> vertices: array<GpuTerrainVertex>;
@group(0) @binding(10)
var<storage, read_write> indices: array<u32>;
@group(0) @binding(11)
var<storage, read_write> counters: GpuManifoldDcCounters;
@group(0) @binding(12)
var<storage, read_write> quads: array<GpuManifoldDcQuad>;

var<workgroup> vertex_scan: array<u32, 256>;
var<workgroup> index_scan: array<u32, 256>;

fn density(word: u32) -> f32 {
    let bits = word & 0xffffu;
    let extension = select(0u, 0xffff0000u, (bits & 0x8000u) != 0u);
    return f32(bitcast<i32>(bits | extension));
}

fn material(word: u32) -> u32 {
    return (word >> 16u) & 0xffu;
}

fn is_solid(word: u32) -> bool {
    return density(word) <= 0.0;
}

fn sample_index(local: vec3<i32>) -> u32 {
    let shifted = vec3<u32>(local + vec3<i32>(1));
    return shifted.x + shifted.y * SAMPLE_EDGE + shifted.z * SAMPLE_LAYER;
}

fn sample_word(local: vec3<i32>) -> u32 {
    return samples[sample_index(local)];
}

fn cell_coordinates(linear: u32) -> vec3<i32> {
    let z = linear / CELL_LAYER;
    let remainder = linear - z * CELL_LAYER;
    let y = remainder / CELL_EDGE;
    let x = remainder - y * CELL_EDGE;
    return vec3<i32>(i32(x) - 1, i32(y) - 1, i32(z) - 1);
}

fn cell_index(cell_min: vec3<i32>) -> u32 {
    let shifted = vec3<u32>(cell_min + vec3<i32>(1));
    return shifted.x + shifted.y * CELL_EDGE + shifted.z * CELL_LAYER;
}

fn owner_coordinates(linear: u32) -> vec3<i32> {
    let z = linear / OWNER_LAYER;
    let remainder = linear - z * OWNER_LAYER;
    let y = remainder / OWNER_EDGE;
    let x = remainder - y * OWNER_EDGE;
    return vec3<i32>(i32(x), i32(y), i32(z));
}

fn fixture_case(cell_min: vec3<i32>) -> u32 {
    var case_index = 0u;
    for (var corner = 0u; corner < 8u; corner += 1u) {
        let local = cell_min + vec3<i32>(REGULAR_CORNERS[corner]);
        if is_solid(sample_word(local)) {
            case_index |= 1u << FIXTURE_BITS[corner];
        }
    }
    return case_index;
}

fn component_count(case_index: u32) -> u32 {
    return component_tables[case_index * 2u] & 0x07u;
}

fn edge_component(case_index: u32, edge: u32) -> u32 {
    if edge < 9u {
        return (component_tables[case_index * 2u] >> (3u + edge * 3u)) & 0x07u;
    }
    return (component_tables[case_index * 2u + 1u] >> ((edge - 9u) * 3u)) & 0x07u;
}

fn edge_anchor(cell_min: vec3<i32>, edge: u32) -> vec3<i32> {
    let endpoints = CUBE_EDGES[edge];
    let first = REGULAR_CORNERS[endpoints.x];
    let second = REGULAR_CORNERS[endpoints.y];
    return cell_min + vec3<i32>(min(first, second));
}

fn edge_axis(edge: u32) -> u32 {
    let endpoints = CUBE_EDGES[edge];
    let delta = REGULAR_CORNERS[endpoints.x] ^ REGULAR_CORNERS[endpoints.y];
    return select(select(2u, 1u, delta.y != 0u), 0u, delta.x != 0u);
}

fn is_owned_anchor(anchor: vec3<i32>) -> bool {
    return all(anchor >= vec3<i32>(0)) && all(anchor < vec3<i32>(32));
}

fn edge_solid_material(cell_min: vec3<i32>, edge: u32) -> u32 {
    let endpoints = CUBE_EDGES[edge];
    let first = sample_word(cell_min + vec3<i32>(REGULAR_CORNERS[endpoints.x]));
    let second = sample_word(cell_min + vec3<i32>(REGULAR_CORNERS[endpoints.y]));
    return select(material(second), material(first), is_solid(first));
}

fn face_pattern(cell_min: vec3<i32>, face: u32) -> u32 {
    var pattern = 0u;
    for (var corner = 0u; corner < 4u; corner += 1u) {
        let cube_corner = FACE_CORNERS[face][corner];
        if is_solid(sample_word(cell_min + vec3<i32>(REGULAR_CORNERS[cube_corner]))) {
            pattern |= 1u << corner;
        }
    }
    return pattern;
}

fn paired_edge_on_face(
    cell_min: vec3<i32>,
    current_edge: u32,
    face_axis: u32,
    positive_face: bool,
) -> u32 {
    let face = face_axis * 2u + select(0u, 1u, positive_face);
    var label = 4u;
    for (var candidate = 0u; candidate < 4u; candidate += 1u) {
        if FACE_CUBE_EDGES[face][candidate] == current_edge {
            label = candidate;
        }
    }
    if label >= 4u {
        return 12u;
    }
    let packed = component_tables[FACE_PAIRING_TABLE_OFFSET + face_pattern(cell_min, face)];
    let paired_label = (packed >> (label * 3u)) & 0x07u;
    if paired_label >= 4u {
        return 12u;
    }
    return FACE_CUBE_EDGES[face][paired_label];
}

fn classify_edge_fans(
    cell_min: vec3<i32>,
    case_index: u32,
    output: ptr<function, array<u32, 12>>,
) -> u32 {
    for (var edge = 0u; edge < 12u; edge += 1u) {
        (*output)[edge] = select(
            INVALID_FAN,
            edge,
            edge_component(case_index, edge) != INVALID_COMPONENT
                && is_owned_anchor(edge_anchor(cell_min, edge)),
        );
    }
    for (var iteration = 0u; iteration < 12u; iteration += 1u) {
        for (var face = 0u; face < 6u; face += 1u) {
            let axis = face / 2u;
            let positive = (face & 1u) != 0u;
            for (var label = 0u; label < 4u; label += 1u) {
                let edge = FACE_CUBE_EDGES[face][label];
                if (*output)[edge] == INVALID_FAN {
                    continue;
                }
                let paired = paired_edge_on_face(cell_min, edge, axis, positive);
                if paired < 12u && (*output)[paired] != INVALID_FAN
                    && edge_solid_material(cell_min, edge)
                        == edge_solid_material(cell_min, paired) {
                    let root = min((*output)[edge], (*output)[paired]);
                    (*output)[edge] = root;
                    (*output)[paired] = root;
                }
            }
        }
    }
    var roots: array<u32, 12>;
    for (var edge = 0u; edge < 12u; edge += 1u) {
        roots[edge] = (*output)[edge];
    }
    var fan_count = 0u;
    for (var edge = 0u; edge < 12u; edge += 1u) {
        if roots[edge] == INVALID_FAN {
            continue;
        }
        var existing = INVALID_FAN;
        for (var previous = 0u; previous < edge; previous += 1u) {
            if roots[previous] == roots[edge] {
                existing = (*output)[previous];
            }
        }
        if existing == INVALID_FAN {
            (*output)[edge] = fan_count;
            fan_count += 1u;
        } else {
            (*output)[edge] = existing;
        }
    }
    return fan_count;
}

fn fan_materials_from_labels(
    cell_min: vec3<i32>,
    fan: u32,
    edge_fans: ptr<function, array<u32, 12>>,
    output: ptr<function, array<u32, 12>>,
) -> u32 {
    for (var index = 0u; index < 12u; index += 1u) {
        (*output)[index] = INVALID_MATERIAL;
    }
    var count = 0u;
    for (var edge = 0u; edge < 12u; edge += 1u) {
        if (*edge_fans)[edge] != fan {
            continue;
        }
        let candidate = edge_solid_material(cell_min, edge);
        var duplicate = false;
        for (var index = 0u; index < count; index += 1u) {
            duplicate = duplicate || (*output)[index] == candidate;
        }
        if duplicate {
            continue;
        }
        var insert = count;
        while insert > 0u && (*output)[insert - 1u] > candidate {
            (*output)[insert] = (*output)[insert - 1u];
            insert -= 1u;
        }
        (*output)[insert] = candidate;
        count += 1u;
    }
    return count;
}

fn cell_edge_fan(cell: GpuManifoldDcCell, edge: u32) -> u32 {
    if edge < 8u {
        return (cell.edge_fans_low >> (edge * 4u)) & 0x0fu;
    }
    return (cell.edge_fans_high >> ((edge - 8u) * 4u)) & 0x0fu;
}

fn cell_fan_material_count(cell: GpuManifoldDcCell, fan: u32) -> u32 {
    if fan < 8u {
        return (cell.fan_material_counts_low >> (fan * 4u)) & 0x0fu;
    }
    return (cell.fan_material_counts_high >> ((fan - 8u) * 4u)) & 0x0fu;
}

fn current_cell(cell: GpuManifoldDcCell) -> bool {
    return (cell.packed_case_component_fan_vertex_counts & 0x80000000u) != 0u
        && cell.generation_low == dispatch.generation_low
        && cell.generation_high == dispatch.generation_high;
}

fn current_owner(owner: GpuManifoldDcOwner) -> bool {
    return owner._pad.x != 0u
        && owner.generation_low == dispatch.generation_low
        && owner.generation_high == dispatch.generation_high;
}

@compute @workgroup_size(64, 1, 1)
fn classify_cells(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let linear = global_id.x;
    if linear >= dispatch.cell_count || linear >= CELL_COUNT {
        return;
    }
    let cell_min = cell_coordinates(linear);
    let case_index = fixture_case(cell_min);
    let count_components = component_count(case_index);
    var edge_fans: array<u32, 12>;
    let fan_count = classify_edge_fans(cell_min, case_index, &edge_fans);
    var count_vertices = 0u;
    var material_counts: array<u32, 12>;
    for (var fan = 0u; fan < 12u; fan += 1u) {
        material_counts[fan] = 0u;
    }
    for (var fan = 0u; fan < fan_count; fan += 1u) {
        var fan_material_values: array<u32, 12>;
        material_counts[fan] = fan_materials_from_labels(
            cell_min, fan, &edge_fans, &fan_material_values,
        );
        count_vertices += material_counts[fan];
    }
    var packed_edge_fans_low = 0u;
    var packed_edge_fans_high = 0u;
    for (var edge = 0u; edge < 12u; edge += 1u) {
        if edge < 8u {
            packed_edge_fans_low |= edge_fans[edge] << (edge * 4u);
        } else {
            packed_edge_fans_high |= edge_fans[edge] << ((edge - 8u) * 4u);
        }
    }
    var packed_material_counts_low = 0u;
    var packed_material_counts_high = 0u;
    for (var fan = 0u; fan < 12u; fan += 1u) {
        if fan < 8u {
            packed_material_counts_low |= material_counts[fan] << (fan * 4u);
        } else {
            packed_material_counts_high |= material_counts[fan] << ((fan - 8u) * 4u);
        }
    }
    cells[linear] = GpuManifoldDcCell(
        case_index
            | (count_components << 8u)
            | (fan_count << 11u)
            | (count_vertices << 15u)
            | 0x80000000u,
        packed_edge_fans_low,
        packed_edge_fans_high,
        packed_material_counts_low,
        packed_material_counts_high,
        dispatch.generation_low,
        dispatch.generation_high,
        0u,
    );
    if count_components != 0u {
        atomicAdd(&counters.active_cells, 1u);
        atomicAdd(&counters.active_components, count_components);
    }
}

@compute @workgroup_size(64, 1, 1)
fn classify_owners(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let linear = global_id.x;
    if linear >= dispatch.owner_count || linear >= OWNER_COUNT {
        return;
    }
    let edge_min = owner_coordinates(linear);
    var active_mask = 0u;
    var first_solid_mask = 0u;
    var packed_materials = 0u;
    for (var axis = 0u; axis < 3u; axis += 1u) {
        var edge_max = edge_min;
        edge_max[axis] += 1;
        let first = sample_word(edge_min);
        let second = sample_word(edge_max);
        if is_solid(first) != is_solid(second) {
            active_mask |= 1u << axis;
            if is_solid(first) {
                first_solid_mask |= 1u << axis;
            }
            let edge_material = select(material(second), material(first), is_solid(first));
            packed_materials |= edge_material << (8u + axis * 8u);
        }
    }
    owners[linear] = GpuManifoldDcOwner(
        active_mask | (first_solid_mask << 3u) | packed_materials,
        owner_topology_vertex_count(edge_min, active_mask),
        dispatch.generation_low,
        dispatch.generation_high,
        vec4<u32>(1u, 0u, 0u, 0u),
    );
    atomicAdd(&counters.active_edges, countOneBits(active_mask));
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
fn scan_cells(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let linear = global_id.x;
    let local_index = local_id.x;
    var count = 0u;
    var current = false;
    if linear < dispatch.cell_count && linear < CELL_COUNT {
        let cell = cells[linear];
        current = current_cell(cell);
        if current {
            count = (cell.packed_case_component_fan_vertex_counts >> 15u) & 0x1fu;
        }
    }
    vertex_scan[local_index] = count;
    index_scan[local_index] = 0u;
    workgroupBarrier();
    inclusive_scan(local_index);
    if current {
        cell_offsets[linear] = GpuTransvoxelCellOffset(
            vertex_scan[local_index] - count,
            0u,
            dispatch.generation_low,
            dispatch.generation_high,
        );
    }
    if local_index == SCAN_WORKGROUP_SIZE - 1u {
        cell_blocks[workgroup_id.x] = GpuTransvoxelScanBlock(
            vertex_scan[local_index], 0u, 0u, 0u,
        );
    }
}

@compute @workgroup_size(256, 1, 1)
fn scan_cell_blocks(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let local_index = local_id.x;
    var count = 0u;
    if local_index < dispatch.cell_scan_blocks {
        count = cell_blocks[local_index].vertex_count;
    }
    vertex_scan[local_index] = count;
    index_scan[local_index] = 0u;
    workgroupBarrier();
    inclusive_scan(local_index);
    if local_index < dispatch.cell_scan_blocks {
        cell_blocks[local_index].first_vertex = vertex_scan[local_index] - count;
    }
    if local_index == SCAN_WORKGROUP_SIZE - 1u {
        let required = vertex_scan[local_index];
        atomicStore(&counters.required_qef_vertices, required);
    }
}

@compute @workgroup_size(256, 1, 1)
fn scan_owners(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let linear = global_id.x;
    let local_index = local_id.x;
    var count = 0u;
    var index_count = 0u;
    var current = false;
    if linear < dispatch.owner_count && linear < OWNER_COUNT {
        let owner = owners[linear];
        current = current_owner(owner);
        if current {
            count = owner.topology_vertex_count;
            index_count = countOneBits(owner.packed_active_materials & 0x07u)
                * INDICES_PER_QUAD;
        }
    }
    vertex_scan[local_index] = count;
    index_scan[local_index] = index_count;
    workgroupBarrier();
    inclusive_scan(local_index);
    if current {
        owner_offsets[linear] = GpuTransvoxelCellOffset(
            vertex_scan[local_index] - count,
            index_scan[local_index] - index_count,
            dispatch.generation_low,
            dispatch.generation_high,
        );
    }
    if local_index == SCAN_WORKGROUP_SIZE - 1u {
        owner_blocks[workgroup_id.x] = GpuTransvoxelScanBlock(
            vertex_scan[local_index], index_scan[local_index], 0u, 0u,
        );
    }
}

@compute @workgroup_size(256, 1, 1)
fn scan_owner_blocks(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let local_index = local_id.x;
    var vertex_count = 0u;
    var index_count = 0u;
    if local_index < dispatch.owner_scan_blocks {
        vertex_count = owner_blocks[local_index].vertex_count;
        index_count = owner_blocks[local_index].index_count;
    }
    vertex_scan[local_index] = vertex_count;
    index_scan[local_index] = index_count;
    workgroupBarrier();
    inclusive_scan(local_index);
    if local_index < dispatch.owner_scan_blocks {
        owner_blocks[local_index].first_vertex = vertex_scan[local_index] - vertex_count;
        owner_blocks[local_index].first_index = index_scan[local_index] - index_count;
    }
    if local_index == SCAN_WORKGROUP_SIZE - 1u {
        let qef_vertices = atomicLoad(&counters.required_qef_vertices);
        let topology_vertices = vertex_scan[local_index];
        let required_vertices = qef_vertices + topology_vertices;
        let required_indices = index_scan[local_index];
        atomicStore(&counters.required_topology_vertices, topology_vertices);
        atomicStore(&counters.required_vertices, required_vertices);
        atomicStore(&counters.required_indices, required_indices);
        atomicStore(
            &counters.vertex_overflow,
            select(0u, 1u, required_vertices > dispatch.max_vertices),
        );
        atomicStore(
            &counters.index_overflow,
            select(0u, 1u, required_indices > dispatch.max_indices),
        );
        atomicStore(&counters.completed, 1u);
    }
}

@compute @workgroup_size(64, 1, 1)
fn finalize_offsets(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let linear = global_id.x;
    if linear < dispatch.cell_count && linear < CELL_COUNT {
        let offset = cell_offsets[linear];
        if offset.generation_low == dispatch.generation_low
            && offset.generation_high == dispatch.generation_high {
            cell_offsets[linear].first_vertex = offset.first_vertex
                + cell_blocks[linear / SCAN_WORKGROUP_SIZE].first_vertex;
        }
    }
    if linear < dispatch.owner_count && linear < OWNER_COUNT {
        let offset = owner_offsets[linear];
        if offset.generation_low == dispatch.generation_low
            && offset.generation_high == dispatch.generation_high {
            owner_offsets[linear].first_index = offset.first_index
                + owner_blocks[linear / SCAN_WORKGROUP_SIZE].first_index;
            owner_offsets[linear].first_vertex = offset.first_vertex
                + owner_blocks[linear / SCAN_WORKGROUP_SIZE].first_vertex;
        }
    }
}

fn bilerp(a: f32, b: f32, c: f32, d: f32, u: f32, v: f32) -> f32 {
    return mix(mix(a, b, u), mix(c, d, u), v);
}

fn normalized_or(value: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let length_squared = dot(value, value);
    if length_squared <= 1.0e-24 {
        return fallback;
    }
    return value * inverseSqrt(length_squared);
}

fn trilinear_gradient(d: array<f32, 8>, point: vec3<f32>) -> vec3<f32> {
    let gx = bilerp(d[1] - d[0], d[3] - d[2], d[5] - d[4], d[7] - d[6], point.y, point.z);
    let gy = bilerp(d[2] - d[0], d[3] - d[1], d[6] - d[4], d[7] - d[5], point.x, point.z);
    let gz = bilerp(d[4] - d[0], d[5] - d[1], d[6] - d[2], d[7] - d[3], point.x, point.y);
    return normalized_or(vec3<f32>(gx, gy, gz), vec3<f32>(1.0, 0.0, 0.0));
}

fn qef_error(
    points: ptr<function, array<vec3<f32>, 12>>,
    normals: ptr<function, array<vec3<f32>, 12>>,
    count: u32,
    candidate: vec3<f32>,
) -> f32 {
    var error = 0.0;
    for (var sample = 0u; sample < count; sample += 1u) {
        let distance = dot((*normals)[sample], candidate - (*points)[sample]);
        error += distance * distance;
    }
    return error;
}

fn pseudo_inverse_symmetric(
    input_matrix: array<array<f32, 3>, 3>,
    input_rhs: array<f32, 3>,
    dimension: u32,
) -> array<f32, 3> {
    var matrix = input_matrix;
    var eigenvectors: array<array<f32, 3>, 3>;
    for (var axis = 0u; axis < dimension; axis += 1u) {
        eigenvectors[axis][axis] = 1.0;
    }
    for (var sweep = 0u; sweep < 10u; sweep += 1u) {
        for (var pair = 0u; pair < 3u; pair += 1u) {
            let p = select(0u, 1u, pair == 2u);
            let q = select(1u, 2u, pair != 0u);
            if p >= dimension || q >= dimension {
                continue;
            }
            let off_diagonal = matrix[p][q];
            if abs(off_diagonal) <= 1.0e-15 {
                continue;
            }
            let tau = (matrix[q][q] - matrix[p][p]) / (2.0 * off_diagonal);
            let tangent = select(
                -1.0 / (-tau + sqrt(1.0 + tau * tau)),
                1.0 / (tau + sqrt(1.0 + tau * tau)),
                tau >= 0.0,
            );
            let cosine = inverseSqrt(1.0 + tangent * tangent);
            let sine = tangent * cosine;
            let app = matrix[p][p];
            let aqq = matrix[q][q];
            matrix[p][p] = app - tangent * off_diagonal;
            matrix[q][q] = aqq + tangent * off_diagonal;
            matrix[p][q] = 0.0;
            matrix[q][p] = 0.0;
            var updated_p: array<f32, 3>;
            var updated_q: array<f32, 3>;
            for (var row = 0u; row < dimension; row += 1u) {
                if row == p || row == q {
                    continue;
                }
                updated_p[row] = cosine * matrix[row][p] - sine * matrix[row][q];
                updated_q[row] = sine * matrix[row][p] + cosine * matrix[row][q];
            }
            for (var row = 0u; row < dimension; row += 1u) {
                if row == p || row == q {
                    continue;
                }
                matrix[row][p] = updated_p[row];
                matrix[p][row] = updated_p[row];
                matrix[row][q] = updated_q[row];
                matrix[q][row] = updated_q[row];
            }
            for (var row = 0u; row < dimension; row += 1u) {
                let vrp = eigenvectors[row][p];
                let vrq = eigenvectors[row][q];
                eigenvectors[row][p] = cosine * vrp - sine * vrq;
                eigenvectors[row][q] = sine * vrp + cosine * vrq;
            }
        }
    }
    var largest = 0.0;
    for (var axis = 0u; axis < dimension; axis += 1u) {
        largest = max(largest, abs(matrix[axis][axis]));
    }
    let threshold = max(largest, 1.0) * 1.0e-5;
    var result: array<f32, 3>;
    for (var eigen = 0u; eigen < dimension; eigen += 1u) {
        let value = matrix[eigen][eigen];
        if abs(value) <= threshold {
            continue;
        }
        var projection = 0.0;
        for (var row = 0u; row < dimension; row += 1u) {
            projection += eigenvectors[row][eigen] * input_rhs[row];
        }
        for (var row = 0u; row < dimension; row += 1u) {
            result[row] += eigenvectors[row][eigen] * projection / value;
        }
    }
    return result;
}

fn solve_qef_with_fixed(
    points: ptr<function, array<vec3<f32>, 12>>,
    normals: ptr<function, array<vec3<f32>, 12>>,
    count: u32,
    mass: vec3<f32>,
    fixed: array<i32, 3>,
) -> vec3<f32> {
    var free_axes: array<u32, 3>;
    var dimension = 0u;
    var result = mass;
    for (var axis = 0u; axis < 3u; axis += 1u) {
        if fixed[axis] < 0 {
            free_axes[dimension] = axis;
            dimension += 1u;
        } else {
            result[axis] = f32(fixed[axis]);
        }
    }
    if dimension == 0u {
        return result;
    }
    var matrix: array<array<f32, 3>, 3>;
    var rhs: array<f32, 3>;
    for (var sample = 0u; sample < count; sample += 1u) {
        let normal = (*normals)[sample];
        let plane = dot(normal, (*points)[sample]);
        var fixed_plane = 0.0;
        for (var axis = 0u; axis < 3u; axis += 1u) {
            if fixed[axis] >= 0 {
                fixed_plane += normal[axis] * f32(fixed[axis]);
            }
        }
        for (var row = 0u; row < dimension; row += 1u) {
            let axis = free_axes[row];
            rhs[row] += normal[axis] * (plane - fixed_plane);
            for (var column = 0u; column < dimension; column += 1u) {
                matrix[row][column] += normal[axis] * normal[free_axes[column]];
            }
        }
    }
    var centered_rhs = rhs;
    for (var row = 0u; row < dimension; row += 1u) {
        for (var column = 0u; column < dimension; column += 1u) {
            centered_rhs[row] -= matrix[row][column] * mass[free_axes[column]];
        }
    }
    let offset = pseudo_inverse_symmetric(matrix, centered_rhs, dimension);
    for (var index = 0u; index < dimension; index += 1u) {
        result[free_axes[index]] = mass[free_axes[index]] + offset[index];
    }
    return result;
}

fn solve_bounded_qef(
    points: ptr<function, array<vec3<f32>, 12>>,
    normals: ptr<function, array<vec3<f32>, 12>>,
    count: u32,
) -> vec3<f32> {
    var mass = vec3<f32>(0.0);
    for (var sample = 0u; sample < count; sample += 1u) {
        mass += (*points)[sample];
    }
    mass /= f32(max(count, 1u));
    var best = mass;
    var best_error = qef_error(points, normals, count, best);
    for (var mask = 0u; mask < 27u; mask += 1u) {
        var code = mask;
        var fixed: array<i32, 3>;
        for (var axis = 0u; axis < 3u; axis += 1u) {
            let state = code % 3u;
            fixed[axis] = select(i32(state) - 1, -1, state == 0u);
            code /= 3u;
        }
        let candidate = solve_qef_with_fixed(points, normals, count, mass, fixed);
        if any(candidate < vec3<f32>(-1.0e-6)) || any(candidate > vec3<f32>(1.0 + 1.0e-6)) {
            continue;
        }
        let error = qef_error(points, normals, count, candidate);
        if error + 1.0e-6 < best_error {
            best = candidate;
            best_error = error;
        }
    }
    let steps = f32(dispatch.position_quantization_steps);
    return clamp(round(clamp(best, vec3<f32>(0.0), vec3<f32>(1.0)) * steps), vec3<f32>(1.0), vec3<f32>(steps - 1.0)) / steps;
}

fn cell_fan_materials(
    cell_min: vec3<i32>,
    cell: GpuManifoldDcCell,
    fan: u32,
    output: ptr<function, array<u32, 12>>,
) -> u32 {
    for (var index = 0u; index < 12u; index += 1u) {
        (*output)[index] = INVALID_MATERIAL;
    }
    var count = 0u;
    for (var edge = 0u; edge < 12u; edge += 1u) {
        if cell_edge_fan(cell, edge) != fan {
            continue;
        }
        let candidate = edge_solid_material(cell_min, edge);
        var duplicate = false;
        for (var index = 0u; index < count; index += 1u) {
            duplicate = duplicate || (*output)[index] == candidate;
        }
        if duplicate {
            continue;
        }
        var insert = count;
        while insert > 0u && (*output)[insert - 1u] > candidate {
            (*output)[insert] = (*output)[insert - 1u];
            insert -= 1u;
        }
        (*output)[insert] = candidate;
        count += 1u;
    }
    return count;
}

fn cell_fan_component(cell: GpuManifoldDcCell, case_index: u32, fan: u32) -> u32 {
    for (var edge = 0u; edge < 12u; edge += 1u) {
        if cell_edge_fan(cell, edge) == fan {
            return edge_component(case_index, edge);
        }
    }
    return INVALID_COMPONENT;
}

fn global_cell_vertex_offset(linear: u32) -> u32 {
    return cell_offsets[linear].first_vertex;
}

@compute @workgroup_size(64, 1, 1)
fn emit_vertices(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let linear = global_id.x;
    if linear >= dispatch.cell_count || linear >= CELL_COUNT {
        return;
    }
    if atomicLoad(&counters.vertex_overflow) != 0u
        || atomicLoad(&counters.index_overflow) != 0u {
        return;
    }
    let cell = cells[linear];
    if !current_cell(cell) {
        return;
    }
    let cell_min = cell_coordinates(linear);
    let case_index = cell.packed_case_component_fan_vertex_counts & 0xffu;
    let fan_count = (cell.packed_case_component_fan_vertex_counts >> 11u) & 0x0fu;
    var density_values: array<f32, 8>;
    for (var corner = 0u; corner < 8u; corner += 1u) {
        density_values[corner] = density(sample_word(cell_min + vec3<i32>(REGULAR_CORNERS[corner])));
    }
    var output_offset = global_cell_vertex_offset(linear);
    for (var fan = 0u; fan < fan_count; fan += 1u) {
        let component = cell_fan_component(cell, case_index, fan);
        if component == INVALID_COMPONENT {
            atomicStore(&counters.topology_error, 1u);
            return;
        }
        var points: array<vec3<f32>, 12>;
        var normals: array<vec3<f32>, 12>;
        var hermite_count = 0u;
        for (var edge = 0u; edge < 12u; edge += 1u) {
            if edge_component(case_index, edge) != component {
                continue;
            }
            let endpoints = CUBE_EDGES[edge];
            let first_corner = REGULAR_CORNERS[endpoints.x];
            let second_corner = REGULAR_CORNERS[endpoints.y];
            let first_density = density_values[endpoints.x];
            let second_density = density_values[endpoints.y];
            let denominator = first_density - second_density;
            let interpolation = select(
                0.5,
                clamp(first_density / denominator, 0.0, 1.0),
                abs(denominator) > 1.0e-12,
            );
            let point = mix(vec3<f32>(first_corner), vec3<f32>(second_corner), interpolation);
            points[hermite_count] = point;
            normals[hermite_count] = trilinear_gradient(density_values, point);
            hermite_count += 1u;
        }
        let solution = solve_bounded_qef(&points, &normals, hermite_count);
        var normal_sum = vec3<f32>(0.0);
        for (var sample = 0u; sample < hermite_count; sample += 1u) {
            normal_sum += normals[sample];
        }
        let normal = normalized_or(normal_sum, vec3<f32>(0.0, 1.0, 0.0));
        var fan_material_values: array<u32, 12>;
        let material_count = cell_fan_materials(cell_min, cell, fan, &fan_material_values);
        for (var material_index = 0u; material_index < material_count; material_index += 1u) {
            vertices[output_offset] = GpuTerrainVertex(
                vec3<f32>(cell_min) + solution,
                fan_material_values[material_index],
                normal,
                0u,
            );
            output_offset += 1u;
        }
    }
    atomicAdd(
        &counters.emitted_vertices,
        (cell.packed_case_component_fan_vertex_counts >> 15u) & 0x1fu,
    );
}

fn regular_corner_index(position: vec3<i32>) -> u32 {
    return u32(position.x) + u32(position.y) * 2u + u32(position.z) * 4u;
}

fn cube_edge_index(first: u32, second: u32) -> u32 {
    let sorted = vec2<u32>(min(first, second), max(first, second));
    for (var edge = 0u; edge < 12u; edge += 1u) {
        if all(CUBE_EDGES[edge] == sorted) {
            return edge;
        }
    }
    return 12u;
}

fn incident_cell(edge_min: vec3<i32>, axis: u32, slot: u32) -> vec3<i32> {
    if axis == 0u {
        let offsets = array<vec3<i32>, 4>(
            vec3<i32>(0, -1, -1), vec3<i32>(0, 0, -1),
            vec3<i32>(0, 0, 0), vec3<i32>(0, -1, 0),
        );
        return edge_min + offsets[slot];
    }
    if axis == 1u {
        let offsets = array<vec3<i32>, 4>(
            vec3<i32>(-1, 0, -1), vec3<i32>(-1, 0, 0),
            vec3<i32>(0, 0, 0), vec3<i32>(0, 0, -1),
        );
        return edge_min + offsets[slot];
    }
    let offsets = array<vec3<i32>, 4>(
        vec3<i32>(-1, -1, 0), vec3<i32>(0, -1, 0),
        vec3<i32>(0, 0, 0), vec3<i32>(-1, 0, 0),
    );
    return edge_min + offsets[slot];
}

fn cell_vertex_for_owned_edge(
    cell_min: vec3<i32>,
    edge_min: vec3<i32>,
    edge_max: vec3<i32>,
    target_material: u32,
) -> u32 {
    let linear = cell_index(cell_min);
    let cell = cells[linear];
    let first_corner = regular_corner_index(edge_min - cell_min);
    let second_corner = regular_corner_index(edge_max - cell_min);
    let edge = cube_edge_index(first_corner, second_corner);
    let target_fan = cell_edge_fan(cell, edge);
    var local_offset = 0u;
    for (var fan = 0u; fan < target_fan; fan += 1u) {
        local_offset += cell_fan_material_count(cell, fan);
    }
    var fan_material_values: array<u32, 12>;
    let count = cell_fan_materials(cell_min, cell, target_fan, &fan_material_values);
    for (var material_index = 0u; material_index < count; material_index += 1u) {
        if fan_material_values[material_index] == target_material {
            return global_cell_vertex_offset(linear) + local_offset + material_index;
        }
    }
    return 0xffffffffu;
}

fn owner_index(anchor: vec3<i32>) -> u32 {
    let coordinate = vec3<u32>(anchor);
    return coordinate.x + coordinate.y * OWNER_EDGE + coordinate.z * OWNER_LAYER;
}

fn edge_is_active_and_owned(edge: ManifoldDcGridEdge) -> bool {
    if edge.valid == 0u || !is_owned_anchor(edge.anchor) {
        return false;
    }
    var edge_max = edge.anchor;
    edge_max[edge.axis] += 1;
    return is_solid(sample_word(edge.anchor)) != is_solid(sample_word(edge_max));
}

fn grid_edges_equal(first: ManifoldDcGridEdge, second: ManifoldDcGridEdge) -> bool {
    return first.valid != 0u && second.valid != 0u
        && first.axis == second.axis && all(first.anchor == second.anchor);
}

fn grid_edge_before(first: ManifoldDcGridEdge, second: ManifoldDcGridEdge) -> bool {
    if first.anchor.x != second.anchor.x {
        return first.anchor.x < second.anchor.x;
    }
    if first.anchor.y != second.anchor.y {
        return first.anchor.y < second.anchor.y;
    }
    if first.anchor.z != second.anchor.z {
        return first.anchor.z < second.anchor.z;
    }
    return first.axis < second.axis;
}

fn grid_edge_material(edge: ManifoldDcGridEdge) -> u32 {
    var edge_max = edge.anchor;
    edge_max[edge.axis] += 1;
    let first = sample_word(edge.anchor);
    let second = sample_word(edge_max);
    return select(material(second), material(first), is_solid(first));
}

fn paired_grid_edge(edge_min: vec3<i32>, axis: u32, side: u32) -> ManifoldDcGridEdge {
    let cell_min = incident_cell(edge_min, axis, side);
    let neighbor = incident_cell(edge_min, axis, (side + 1u) & 3u);
    var face_axis = 3u;
    for (var coordinate = 0u; coordinate < 3u; coordinate += 1u) {
        if cell_min[coordinate] != neighbor[coordinate] {
            face_axis = coordinate;
        }
    }
    if face_axis >= 3u {
        return ManifoldDcGridEdge(vec3<i32>(0), 0u, 0u);
    }
    var edge_max = edge_min;
    edge_max[axis] += 1;
    let first_corner = regular_corner_index(edge_min - cell_min);
    let second_corner = regular_corner_index(edge_max - cell_min);
    let current_edge = cube_edge_index(first_corner, second_corner);
    let paired = paired_edge_on_face(
        cell_min,
        current_edge,
        face_axis,
        neighbor[face_axis] > cell_min[face_axis],
    );
    if paired >= 12u {
        return ManifoldDcGridEdge(vec3<i32>(0), 0u, 0u);
    }
    return ManifoldDcGridEdge(edge_anchor(cell_min, paired), edge_axis(paired), 1u);
}

fn segment_is_canonical(edge_min: vec3<i32>, axis: u32, side: u32) -> bool {
    let current = ManifoldDcGridEdge(edge_min, axis, 1u);
    let paired = paired_grid_edge(edge_min, axis, side);
    return !edge_is_active_and_owned(paired) || grid_edge_before(current, paired);
}

fn segment_materials(
    edge_min: vec3<i32>,
    axis: u32,
    side: u32,
    output: ptr<function, array<u32, 2>>,
) -> u32 {
    let current = ManifoldDcGridEdge(edge_min, axis, 1u);
    let first = grid_edge_material(current);
    (*output)[0] = first;
    (*output)[1] = INVALID_MATERIAL;
    let paired = paired_grid_edge(edge_min, axis, side);
    if !edge_is_active_and_owned(paired) {
        return 1u;
    }
    let second = grid_edge_material(paired);
    if first == second {
        return 1u;
    }
    (*output)[0] = min(first, second);
    (*output)[1] = max(first, second);
    return 2u;
}

fn axis_topology_vertex_count(edge_min: vec3<i32>, axis: u32) -> u32 {
    var count = 1u;
    for (var side = 0u; side < 4u; side += 1u) {
        if segment_is_canonical(edge_min, axis, side) {
            var materials: array<u32, 2>;
            count += segment_materials(edge_min, axis, side, &materials);
        }
    }
    return count;
}

fn owner_topology_vertex_count(edge_min: vec3<i32>, active_mask: u32) -> u32 {
    var count = 0u;
    for (var axis = 0u; axis < 3u; axis += 1u) {
        if (active_mask & (1u << axis)) != 0u {
            count += axis_topology_vertex_count(edge_min, axis);
        }
    }
    return count;
}

fn owner_axis_local_offset(edge_min: vec3<i32>, active_mask: u32, axis: u32) -> u32 {
    var offset = 0u;
    for (var previous = 0u; previous < axis; previous += 1u) {
        if (active_mask & (1u << previous)) != 0u {
            offset += axis_topology_vertex_count(edge_min, previous);
        }
    }
    return offset;
}

fn canonical_segment_local_offset(
    edge_min: vec3<i32>,
    active_mask: u32,
    axis: u32,
    side: u32,
) -> u32 {
    var offset = owner_axis_local_offset(edge_min, active_mask, axis) + 1u;
    for (var previous = 0u; previous < side; previous += 1u) {
        if segment_is_canonical(edge_min, axis, previous) {
            var materials: array<u32, 2>;
            offset += segment_materials(edge_min, axis, previous, &materials);
        }
    }
    return offset;
}

fn owner_vertex_base(edge_min: vec3<i32>) -> u32 {
    return atomicLoad(&counters.required_qef_vertices)
        + owner_offsets[owner_index(edge_min)].first_vertex;
}

fn center_vertex_for_owned_edge(edge_min: vec3<i32>, axis: u32) -> u32 {
    let owner = owners[owner_index(edge_min)];
    let active_mask = owner.packed_active_materials & 0x07u;
    return owner_vertex_base(edge_min)
        + owner_axis_local_offset(edge_min, active_mask, axis);
}

fn segment_side_for_pair(
    edge_min: vec3<i32>,
    axis: u32,
    other: ManifoldDcGridEdge,
) -> u32 {
    for (var side = 0u; side < 4u; side += 1u) {
        if grid_edges_equal(paired_grid_edge(edge_min, axis, side), other) {
            return side;
        }
    }
    return 4u;
}

fn segment_vertex_for_owned_edge(
    edge_min: vec3<i32>,
    axis: u32,
    side: u32,
    target_material: u32,
) -> u32 {
    let current = ManifoldDcGridEdge(edge_min, axis, 1u);
    let paired = paired_grid_edge(edge_min, axis, side);
    var canonical = current;
    var canonical_side = side;
    if edge_is_active_and_owned(paired) && grid_edge_before(paired, current) {
        canonical = paired;
        canonical_side = segment_side_for_pair(paired.anchor, paired.axis, current);
    }
    if canonical_side >= 4u {
        return 0xffffffffu;
    }
    let owner = owners[owner_index(canonical.anchor)];
    let active_mask = owner.packed_active_materials & 0x07u;
    var materials: array<u32, 2>;
    let count = segment_materials(
        canonical.anchor, canonical.axis, canonical_side, &materials,
    );
    for (var material_index = 0u; material_index < count; material_index += 1u) {
        if materials[material_index] == target_material {
            return owner_vertex_base(canonical.anchor)
                + canonical_segment_local_offset(
                    canonical.anchor, active_mask, canonical.axis, canonical_side,
                )
                + material_index;
        }
    }
    return 0xffffffffu;
}

fn segment_side_for_cells(
    edge_min: vec3<i32>,
    axis: u32,
    first: vec3<i32>,
    second: vec3<i32>,
) -> u32 {
    for (var side = 0u; side < 4u; side += 1u) {
        let candidate_first = incident_cell(edge_min, axis, side);
        let candidate_second = incident_cell(edge_min, axis, (side + 1u) & 3u);
        if (all(candidate_first == first) && all(candidate_second == second))
            || (all(candidate_first == second) && all(candidate_second == first)) {
            return side;
        }
    }
    return 4u;
}

fn averaged_topology_vertex(
    first: GpuTerrainVertex,
    second: GpuTerrainVertex,
    target_material: u32,
) -> GpuTerrainVertex {
    return GpuTerrainVertex(
        (first.position + second.position) * 0.5,
        target_material,
        normalized_or(first.normal + second.normal, vec3<f32>(0.0, 1.0, 0.0)),
        0u,
    );
}

@compute @workgroup_size(64, 1, 1)
fn emit_topology_vertices(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let linear = global_id.x;
    if linear >= dispatch.owner_count || linear >= OWNER_COUNT
        || atomicLoad(&counters.vertex_overflow) != 0u
        || atomicLoad(&counters.index_overflow) != 0u
        || atomicLoad(&counters.topology_error) != 0u {
        return;
    }
    let owner = owners[linear];
    if !current_owner(owner) {
        return;
    }
    let active_mask = owner.packed_active_materials & 0x07u;
    let edge_min = owner_coordinates(linear);
    for (var axis = 0u; axis < 3u; axis += 1u) {
        if (active_mask & (1u << axis)) == 0u {
            continue;
        }
        var edge_max = edge_min;
        edge_max[axis] += 1;
        let edge_material = (owner.packed_active_materials >> (8u + axis * 8u)) & 0xffu;
        var qef: array<u32, 4>;
        var invalid = false;
        for (var slot = 0u; slot < 4u; slot += 1u) {
            qef[slot] = cell_vertex_for_owned_edge(
                incident_cell(edge_min, axis, slot), edge_min, edge_max, edge_material,
            );
            invalid = invalid || qef[slot] == 0xffffffffu;
        }
        if invalid {
            atomicStore(&counters.topology_error, 1u);
            return;
        }
        let center = center_vertex_for_owned_edge(edge_min, axis);
        let normal = normalized_or(
            vertices[qef[0]].normal + vertices[qef[1]].normal
                + vertices[qef[2]].normal + vertices[qef[3]].normal,
            vec3<f32>(0.0, 1.0, 0.0),
        );
        vertices[center] = GpuTerrainVertex(
            (vertices[qef[0]].position + vertices[qef[1]].position
                + vertices[qef[2]].position + vertices[qef[3]].position) * 0.25,
            edge_material,
            normal,
            0u,
        );
        for (var side = 0u; side < 4u; side += 1u) {
            if !segment_is_canonical(edge_min, axis, side) {
                continue;
            }
            var materials: array<u32, 2>;
            let count = segment_materials(edge_min, axis, side, &materials);
            let first_cell = incident_cell(edge_min, axis, side);
            let second_cell = incident_cell(edge_min, axis, (side + 1u) & 3u);
            let current_segment_edge = ManifoldDcGridEdge(edge_min, axis, 1u);
            let paired_segment_edge = paired_grid_edge(edge_min, axis, side);
            for (var material_index = 0u; material_index < count; material_index += 1u) {
                let target_material = materials[material_index];
                var source_edge = current_segment_edge;
                if grid_edge_material(current_segment_edge) != target_material
                    && edge_is_active_and_owned(paired_segment_edge) {
                    source_edge = paired_segment_edge;
                }
                var source_edge_max = source_edge.anchor;
                source_edge_max[source_edge.axis] += 1;
                let first = cell_vertex_for_owned_edge(
                    first_cell, source_edge.anchor, source_edge_max, target_material,
                );
                let second = cell_vertex_for_owned_edge(
                    second_cell, source_edge.anchor, source_edge_max, target_material,
                );
                if first == 0xffffffffu || second == 0xffffffffu {
                    atomicStore(&counters.topology_error, 1u);
                    return;
                }
                let output = segment_vertex_for_owned_edge(
                    edge_min, axis, side, target_material,
                );
                if output == 0xffffffffu {
                    atomicStore(&counters.topology_error, 1u);
                    return;
                }
                vertices[output] = averaged_topology_vertex(
                    vertices[first], vertices[second], target_material,
                );
            }
        }
    }
    atomicAdd(&counters.emitted_vertices, owner.topology_vertex_count);
}

@compute @workgroup_size(64, 1, 1)
fn emit_quads(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let linear = global_id.x;
    if linear >= dispatch.owner_count || linear >= OWNER_COUNT
        || atomicLoad(&counters.vertex_overflow) != 0u
        || atomicLoad(&counters.index_overflow) != 0u
        || atomicLoad(&counters.topology_error) != 0u {
        return;
    }
    let owner = owners[linear];
    if !current_owner(owner) {
        return;
    }
    let active_mask = owner.packed_active_materials & 0x07u;
    let first_solid_mask = (owner.packed_active_materials >> 3u) & 0x07u;
    let edge_min = owner_coordinates(linear);
    var output_quad = owner_offsets[linear].first_index / INDICES_PER_QUAD;
    for (var axis = 0u; axis < 3u; axis += 1u) {
        if (active_mask & (1u << axis)) == 0u {
            continue;
        }
        var edge_max = edge_min;
        edge_max[axis] += 1;
        let edge_material = (owner.packed_active_materials >> (8u + axis * 8u)) & 0xffu;
        var qef: array<u32, 4>;
        var quad_cells: array<vec3<i32>, 4>;
        for (var slot = 0u; slot < 4u; slot += 1u) {
            quad_cells[slot] = incident_cell(edge_min, axis, slot);
            qef[slot] = cell_vertex_for_owned_edge(
                quad_cells[slot], edge_min, edge_max, edge_material,
            );
        }
        if (first_solid_mask & (1u << axis)) == 0u {
            let qef0 = qef[0];
            let qef1 = qef[1];
            let cell0 = quad_cells[0];
            let cell1 = quad_cells[1];
            qef[0] = qef[3];
            qef[1] = qef[2];
            qef[2] = qef1;
            qef[3] = qef0;
            quad_cells[0] = quad_cells[3];
            quad_cells[1] = quad_cells[2];
            quad_cells[2] = cell1;
            quad_cells[3] = cell0;
        }
        var midpoint: array<u32, 4>;
        var invalid = false;
        for (var side = 0u; side < 4u; side += 1u) {
            let base_side = segment_side_for_cells(
                edge_min,
                axis,
                quad_cells[side],
                quad_cells[(side + 1u) & 3u],
            );
            if base_side >= 4u {
                invalid = true;
            } else {
                midpoint[side] = segment_vertex_for_owned_edge(
                    edge_min, axis, base_side, edge_material,
                );
                invalid = invalid || midpoint[side] == 0xffffffffu;
            }
            invalid = invalid || qef[side] == 0xffffffffu;
        }
        if invalid {
            atomicStore(&counters.topology_error, 1u);
            return;
        }
        quads[output_quad] = GpuManifoldDcQuad(
            vec4<u32>(qef[0], qef[1], qef[2], qef[3]),
            vec4<u32>(midpoint[0], midpoint[1], midpoint[2], midpoint[3]),
            center_vertex_for_owned_edge(edge_min, axis),
            edge_material,
            dispatch.generation_low,
            dispatch.generation_high,
        );
        output_quad += 1u;
    }
}

@compute @workgroup_size(64, 1, 1)
fn emit_indices(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let linear = global_id.x;
    let required_indices = atomicLoad(&counters.required_indices);
    if atomicLoad(&counters.vertex_overflow) != 0u
        || atomicLoad(&counters.index_overflow) != 0u
        || atomicLoad(&counters.topology_error) != 0u
        || linear >= required_indices / INDICES_PER_QUAD {
        return;
    }
    let quad = quads[linear];
    if quad.generation_low != dispatch.generation_low
        || quad.generation_high != dispatch.generation_high {
        atomicStore(&counters.topology_error, 1u);
        return;
    }
    let output_offset = linear * INDICES_PER_QUAD;
    for (var side = 0u; side < 4u; side += 1u) {
        let first = quad.qef_vertices[side];
        let midpoint = quad.midpoint_vertices[side];
        let second = quad.qef_vertices[(side + 1u) & 3u];
        let triangle_offset = output_offset + side * 6u;
        indices[triangle_offset + 0u] = first;
        indices[triangle_offset + 1u] = midpoint;
        indices[triangle_offset + 2u] = quad.center_vertex;
        indices[triangle_offset + 3u] = midpoint;
        indices[triangle_offset + 4u] = second;
        indices[triangle_offset + 5u] = quad.center_vertex;
    }
    atomicAdd(&counters.emitted_indices, INDICES_PER_QUAD);
}
