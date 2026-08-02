const PAGE_EDGE: u32 = 32u;
const REGULAR_SAMPLE_EDGE: u32 = 34u;
const REGULAR_SAMPLE_COUNT: u32 = 39304u;
const TRANSITION_FACE_COUNT: u32 = 6u;
const TRANSITION_SLAB_EDGE: u32 = 67u;
const TRANSITION_SLAB_LAYERS: u32 = 3u;
const TRANSITION_FACE_STRIDE: u32 = 13467u;
const TRANSITION_SAMPLE_COUNT: u32 = 80802u;
const PAGE_TABLE_EMPTY: u32 = 0u;
const PAGE_TABLE_OCCUPIED: u32 = 1u;

struct GpuSurfaceGatherJob {
    planet_id: vec4<u32>,
    relative_lod0_cell_min: vec3<i32>,
    lod: u32,
    generation_low: u32,
    generation_high: u32,
    transition_mask: u32,
    target_slot: u32,
    residency_epoch_low: u32,
    residency_epoch_high: u32,
    _pad: vec2<u32>,
};

struct GpuResidencyUniform {
    table_mask: u32,
    max_probe: u32,
    resident_pages: u32,
    atlas_tiles_x: u32,
    atlas_tiles_y: u32,
    atlas_tiles_z: u32,
    publication_epoch_low: u32,
    publication_epoch_high: u32,
};

struct GpuPageTableEntry {
    planet_id: vec4<u32>,
    relative_lod0_cell_min: vec3<i32>,
    lod: u32,
    slot: u32,
    generation_low: u32,
    generation_high: u32,
    state: u32,
};

struct GpuLookupResult {
    slot: u32,
    generation_low: u32,
    generation_high: u32,
    probes: u32,
    found: u32,
};

struct GpuSurfaceGatherCounters {
    regular_samples: atomic<u32>,
    transition_samples: atomic<u32>,
    table_probes: atomic<u32>,
    page_misses: atomic<u32>,
    stale_targets: atomic<u32>,
    completed: atomic<u32>,
    _pad0: u32,
    _pad1: u32,
};

struct DispatchIndirectArgs {
    x: u32,
    y: u32,
    z: u32,
};

@group(0) @binding(0) var<uniform> job: GpuSurfaceGatherJob;
@group(0) @binding(1) var<uniform> residency: GpuResidencyUniform;
@group(0) @binding(2) var<storage, read> page_table: array<GpuPageTableEntry>;
@group(0) @binding(3) var atlas: texture_3d<u32>;
@group(0) @binding(4) var<storage, read_write> regular_samples: array<u32>;
@group(0) @binding(5) var<storage, read_write> transition_samples: array<u32>;
@group(0) @binding(6) var<storage, read_write> counters: GpuSurfaceGatherCounters;
@group(0) @binding(7) var<storage, read_write> indirect_commands: array<DispatchIndirectArgs>;

const FACE_ORIGIN: array<vec3<i32>, 6> = array<vec3<i32>, 6>(
    vec3<i32>(0, 0, 1), vec3<i32>(1, 0, 0),
    vec3<i32>(1, 0, 0), vec3<i32>(0, 1, 0),
    vec3<i32>(0, 1, 0), vec3<i32>(0, 0, 1),
);
const FACE_U: array<vec3<i32>, 6> = array<vec3<i32>, 6>(
    vec3<i32>(0, 1, 0), vec3<i32>(0, 1, 0),
    vec3<i32>(0, 0, 1), vec3<i32>(0, 0, 1),
    vec3<i32>(1, 0, 0), vec3<i32>(1, 0, 0),
);
const FACE_V: array<vec3<i32>, 6> = array<vec3<i32>, 6>(
    vec3<i32>(0, 0, -1), vec3<i32>(0, 0, 1),
    vec3<i32>(-1, 0, 0), vec3<i32>(1, 0, 0),
    vec3<i32>(0, -1, 0), vec3<i32>(0, 1, 0),
);
const FACE_OUTWARD: array<vec3<i32>, 6> = array<vec3<i32>, 6>(
    vec3<i32>(-1, 0, 0), vec3<i32>(1, 0, 0),
    vec3<i32>(0, -1, 0), vec3<i32>(0, 1, 0),
    vec3<i32>(0, 0, -1), vec3<i32>(0, 0, 1),
);

fn mix_hash(hash: u32, value: u32) -> u32 {
    let mixed = (hash ^ value) * 0x045d9f3bu;
    return mixed ^ (mixed >> 16u);
}

fn page_hash(relative_min: vec3<i32>, lod: u32) -> u32 {
    var hash = 0x811c9dc5u;
    hash = mix_hash(hash, job.planet_id.x);
    hash = mix_hash(hash, job.planet_id.y);
    hash = mix_hash(hash, job.planet_id.z);
    hash = mix_hash(hash, job.planet_id.w);
    hash = mix_hash(hash, bitcast<u32>(relative_min.x));
    hash = mix_hash(hash, bitcast<u32>(relative_min.y));
    hash = mix_hash(hash, bitcast<u32>(relative_min.z));
    return mix_hash(hash, lod);
}

fn keys_equal(entry: GpuPageTableEntry, relative_min: vec3<i32>, lod: u32) -> bool {
    return all(entry.planet_id == job.planet_id)
        && all(entry.relative_lod0_cell_min == relative_min)
        && entry.lod == lod;
}

fn lookup_page(relative_min: vec3<i32>, lod: u32) -> GpuLookupResult {
    let start = page_hash(relative_min, lod) & residency.table_mask;
    var probe = 0u;
    loop {
        if probe >= residency.max_probe {
            break;
        }
        let entry = page_table[(start + probe) & residency.table_mask];
        if entry.state == PAGE_TABLE_EMPTY {
            return GpuLookupResult(0u, 0u, 0u, probe + 1u, 0u);
        }
        if entry.state == PAGE_TABLE_OCCUPIED && keys_equal(entry, relative_min, lod) {
            return GpuLookupResult(
                entry.slot,
                entry.generation_low,
                entry.generation_high,
                probe + 1u,
                1u,
            );
        }
        probe += 1u;
    }
    return GpuLookupResult(0u, 0u, 0u, probe, 0u);
}

fn floor_div(value: i32, divisor: i32) -> i32 {
    var quotient = value / divisor;
    if value % divisor < 0 {
        quotient -= 1;
    }
    return quotient;
}

fn slot_origin(slot: u32) -> vec3<u32> {
    let x = slot % residency.atlas_tiles_x;
    let y = (slot / residency.atlas_tiles_x) % residency.atlas_tiles_y;
    let z = slot / (residency.atlas_tiles_x * residency.atlas_tiles_y);
    return vec3<u32>(x, y, z) * PAGE_EDGE;
}

fn gather_sample(relative_position: vec3<i32>, lod: u32) -> u32 {
    let scale = i32(1u << lod);
    let span = i32(PAGE_EDGE) * scale;
    // Camera-relative zero is snapped to an LOD0 page, not necessarily to
    // this coarser LOD. The target page minimum is aligned to its own LOD and
    // every finer transition LOD, so it is the stable grid anchor for both
    // regular and transition samples.
    let target_offset = relative_position - job.relative_lod0_cell_min;
    let relative_min = job.relative_lod0_cell_min + vec3<i32>(
        floor_div(target_offset.x, span) * span,
        floor_div(target_offset.y, span) * span,
        floor_div(target_offset.z, span) * span,
    );
    let lookup = lookup_page(relative_min, lod);
    atomicAdd(&counters.table_probes, lookup.probes);
    if lookup.found == 0u {
        atomicAdd(&counters.page_misses, 1u);
        return 0x00007fffu;
    }
    let local = vec3<u32>((relative_position - relative_min) / scale);
    return textureLoad(atlas, vec3<i32>(slot_origin(lookup.slot) + local), 0).x;
}

fn epoch_matches() -> bool {
    return residency.publication_epoch_low == job.residency_epoch_low
        && residency.publication_epoch_high == job.residency_epoch_high;
}

@compute @workgroup_size(64, 1, 1)
fn gather_regular(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let linear = invocation.x;
    if linear >= REGULAR_SAMPLE_COUNT || !epoch_matches() {
        return;
    }
    let x = linear % REGULAR_SAMPLE_EDGE;
    let y = (linear / REGULAR_SAMPLE_EDGE) % REGULAR_SAMPLE_EDGE;
    let z = linear / (REGULAR_SAMPLE_EDGE * REGULAR_SAMPLE_EDGE);
    let local = vec3<i32>(i32(x) - 1, i32(y) - 1, i32(z) - 1);
    let scale = i32(1u << job.lod);
    regular_samples[linear] = gather_sample(job.relative_lod0_cell_min + local * scale, job.lod);
    atomicAdd(&counters.regular_samples, 1u);
}

@compute @workgroup_size(64, 1, 1)
fn gather_transition(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let linear = invocation.x;
    if linear >= TRANSITION_SAMPLE_COUNT || job.lod == 0u || !epoch_matches() {
        return;
    }
    let face = linear / TRANSITION_FACE_STRIDE;
    if face >= TRANSITION_FACE_COUNT || (job.transition_mask & (1u << face)) == 0u {
        return;
    }
    let face_linear = linear % TRANSITION_FACE_STRIDE;
    let layer = face_linear / (TRANSITION_SLAB_EDGE * TRANSITION_SLAB_EDGE);
    let layer_linear = face_linear % (TRANSITION_SLAB_EDGE * TRANSITION_SLAB_EDGE);
    let v = layer_linear / TRANSITION_SLAB_EDGE;
    let u = layer_linear % TRANSITION_SLAB_EDGE;
    let fine_scale = i32(1u << (job.lod - 1u));
    let coarse_span = i32(PAGE_EDGE) * fine_scale * 2;
    let relative_position = job.relative_lod0_cell_min
        + FACE_ORIGIN[face] * coarse_span
        + FACE_U[face] * (i32(u) - 1) * fine_scale
        + FACE_V[face] * (i32(v) - 1) * fine_scale
        + FACE_OUTWARD[face] * (i32(layer) - 1) * fine_scale;
    transition_samples[linear] = gather_sample(relative_position, job.lod - 1u);
    atomicAdd(&counters.transition_samples, 1u);
}

fn set_indirect(index: u32, x: u32) {
    indirect_commands[index] = DispatchIndirectArgs(x, 1u, 1u);
}

@compute @workgroup_size(1, 1, 1)
fn finalize_gather() {
    let target_lookup = lookup_page(job.relative_lod0_cell_min, job.lod);
    atomicAdd(&counters.table_probes, target_lookup.probes);
    let target_current = target_lookup.found != 0u
        && target_lookup.slot == job.target_slot
        && target_lookup.generation_low == job.generation_low
        && target_lookup.generation_high == job.generation_high;
    if !target_current || !epoch_matches() {
        atomicStore(&counters.stale_targets, 1u);
        return;
    }
    if atomicLoad(&counters.page_misses) != 0u
        || atomicLoad(&counters.regular_samples) != REGULAR_SAMPLE_COUNT
        || atomicLoad(&counters.transition_samples)
            != countOneBits(job.transition_mask & 0x3fu) * TRANSITION_FACE_STRIDE {
        return;
    }
    atomicStore(&counters.completed, 1u);
    set_indirect(0u, 512u);
    set_indirect(1u, 128u);
    set_indirect(2u, 1u);
    set_indirect(3u, 512u);
    set_indirect(4u, 96u);
    set_indirect(5u, 24u);
    set_indirect(6u, 1u);
    set_indirect(7u, 96u);
}
