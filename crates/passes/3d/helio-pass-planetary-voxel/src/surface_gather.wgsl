const PAGE_EDGE: u32 = 32u;
const REGULAR_SAMPLE_EDGE: u32 = 34u;
const REGULAR_SAMPLE_COUNT: u32 = 39304u;
const TRANSITION_FACE_COUNT: u32 = 6u;
const TRANSITION_SLAB_EDGE: u32 = 69u;
const TRANSITION_SLAB_LAYERS: u32 = 5u;
const TRANSITION_FACE_STRIDE: u32 = 23805u;
const TRANSITION_SAMPLE_COUNT: u32 = 142830u;
const PAGE_TABLE_EMPTY: u32 = 0u;
const PAGE_TABLE_OCCUPIED: u32 = 1u;

struct GpuSurfaceGatherJob {
    planet_id: vec4<u32>,
    lod0_cell_min_x: vec2<u32>,
    lod0_cell_min_y: vec2<u32>,
    lod0_cell_min_z: vec2<u32>,
    lod: u32,
    generation_low: u32,
    generation_high: u32,
    transition_mask: u32,
    target_slot: u32,
    residency_epoch_low: u32,
    residency_epoch_high: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
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
    lod0_cell_min_x: vec2<u32>,
    lod0_cell_min_y: vec2<u32>,
    lod0_cell_min_z: vec2<u32>,
    lod: u32,
    slot: u32,
    generation_low: u32,
    generation_high: u32,
    state: u32,
    _pad: u32,
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

fn page_hash(min_x: vec2<u32>, min_y: vec2<u32>, min_z: vec2<u32>, lod: u32) -> u32 {
    var hash = 0x811c9dc5u;
    hash = mix_hash(hash, job.planet_id.x);
    hash = mix_hash(hash, job.planet_id.y);
    hash = mix_hash(hash, job.planet_id.z);
    hash = mix_hash(hash, job.planet_id.w);
    hash = mix_hash(hash, min_x.x);
    hash = mix_hash(hash, min_x.y);
    hash = mix_hash(hash, min_y.x);
    hash = mix_hash(hash, min_y.y);
    hash = mix_hash(hash, min_z.x);
    hash = mix_hash(hash, min_z.y);
    return mix_hash(hash, lod);
}

fn keys_equal(
    entry: GpuPageTableEntry,
    min_x: vec2<u32>,
    min_y: vec2<u32>,
    min_z: vec2<u32>,
    lod: u32,
) -> bool {
    return all(entry.planet_id == job.planet_id)
        && all(entry.lod0_cell_min_x == min_x)
        && all(entry.lod0_cell_min_y == min_y)
        && all(entry.lod0_cell_min_z == min_z)
        && entry.lod == lod;
}

fn lookup_page(
    min_x: vec2<u32>,
    min_y: vec2<u32>,
    min_z: vec2<u32>,
    lod: u32,
) -> GpuLookupResult {
    let start = page_hash(min_x, min_y, min_z, lod) & residency.table_mask;
    var probe = 0u;
    loop {
        if probe >= residency.max_probe {
            break;
        }
        let entry = page_table[(start + probe) & residency.table_mask];
        if entry.state == PAGE_TABLE_EMPTY {
            return GpuLookupResult(0u, 0u, 0u, probe + 1u, 0u);
        }
        if entry.state == PAGE_TABLE_OCCUPIED && keys_equal(entry, min_x, min_y, min_z, lod) {
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

fn add_i32_to_i64_words(value: vec2<u32>, delta: i32) -> vec2<u32> {
    let delta_low = bitcast<u32>(delta);
    let low = value.x + delta_low;
    let carry = select(0u, 1u, low < value.x);
    let sign_extension = select(0u, 0xffffffffu, delta < 0);
    return vec2<u32>(low, value.y + sign_extension + carry);
}

fn slot_origin(slot: u32) -> vec3<u32> {
    let x = slot % residency.atlas_tiles_x;
    let y = (slot / residency.atlas_tiles_x) % residency.atlas_tiles_y;
    let z = slot / (residency.atlas_tiles_x * residency.atlas_tiles_y);
    return vec3<u32>(x, y, z) * PAGE_EDGE;
}

fn gather_sample(target_offset: vec3<i32>, lod: u32) -> u32 {
    let scale = i32(1u << lod);
    let span = i32(PAGE_EDGE) * scale;
    let page_offset = vec3<i32>(
        floor_div(target_offset.x, span) * span,
        floor_div(target_offset.y, span) * span,
        floor_div(target_offset.z, span) * span,
    );
    let min_x = add_i32_to_i64_words(job.lod0_cell_min_x, page_offset.x);
    let min_y = add_i32_to_i64_words(job.lod0_cell_min_y, page_offset.y);
    let min_z = add_i32_to_i64_words(job.lod0_cell_min_z, page_offset.z);
    let lookup = lookup_page(min_x, min_y, min_z, lod);
    atomicAdd(&counters.table_probes, lookup.probes);
    if lookup.found == 0u {
        atomicAdd(&counters.page_misses, 1u);
        return 0x00007fffu;
    }
    let local = vec3<u32>((target_offset - page_offset) / scale);
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
    regular_samples[linear] = gather_sample(local * scale, job.lod);
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
    let target_offset = FACE_ORIGIN[face] * coarse_span
        + FACE_U[face] * (i32(u) - 2) * fine_scale
        + FACE_V[face] * (i32(v) - 2) * fine_scale
        + FACE_OUTWARD[face] * (i32(layer) - 2) * fine_scale;
    transition_samples[linear] = gather_sample(target_offset, job.lod - 1u);
    atomicAdd(&counters.transition_samples, 1u);
}

fn set_indirect(index: u32, x: u32) {
    indirect_commands[index] = DispatchIndirectArgs(x, 1u, 1u);
}

@compute @workgroup_size(1, 1, 1)
fn finalize_gather() {
    let target_lookup = lookup_page(
        job.lod0_cell_min_x,
        job.lod0_cell_min_y,
        job.lod0_cell_min_z,
        job.lod,
    );
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
