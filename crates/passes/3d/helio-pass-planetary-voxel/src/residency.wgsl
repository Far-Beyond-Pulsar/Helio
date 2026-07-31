const PAGE_TABLE_EMPTY: u32 = 0u;
const PAGE_TABLE_OCCUPIED: u32 = 1u;

struct GpuPageTableEntry {
    planet_id: vec4<u32>,
    relative_lod0_cell_min: vec3<i32>,
    lod: u32,
    slot: u32,
    generation_low: u32,
    generation_high: u32,
    state: u32,
};

struct GpuResidencyUniform {
    table_mask: u32,
    max_probe: u32,
    resident_pages: u32,
    _pad: u32,
};

struct GpuResidencyCounters {
    resident_pages: u32,
    resident_cell_bytes_low: u32,
    resident_cell_bytes_high: u32,
    table_occupied: u32,
    table_tombstones: u32,
    uploads_published: u32,
    evictions_published: u32,
    stale_rejections: u32,
    generation_conflicts: u32,
    backpressure_events: u32,
    table_saturation_events: u32,
    batches_submitted: u32,
    cell_bytes_uploaded_low: u32,
    cell_bytes_uploaded_high: u32,
    peak_resident_pages: u32,
    peak_resident_cell_bytes_low: u32,
    peak_resident_cell_bytes_high: u32,
    allocated_gpu_bytes_low: u32,
    allocated_gpu_bytes_high: u32,
    resource_buffers: u32,
    atlas_shards: u32,
    device_rebuilds: u32,
    _pad: vec2<u32>,
};

struct GpuLookupQuery {
    planet_id: vec4<u32>,
    relative_lod0_cell_min: vec3<i32>,
    lod: u32,
};

struct GpuLookupResult {
    slot: u32,
    generation_low: u32,
    generation_high: u32,
    probes_and_found: u32,
};

@group(0) @binding(0)
var<storage, read> page_table: array<GpuPageTableEntry>;
@group(0) @binding(1)
var<uniform> residency: GpuResidencyUniform;
@group(0) @binding(2)
var<storage, read> queries: array<GpuLookupQuery>;
@group(0) @binding(3)
var<storage, read_write> results: array<GpuLookupResult>;

fn mix_hash(hash: u32, value: u32) -> u32 {
    let mixed = (hash ^ value) * 0x045d9f3bu;
    return mixed ^ (mixed >> 16u);
}

fn page_hash(query: GpuLookupQuery) -> u32 {
    var hash = 0x811c9dc5u;
    hash = mix_hash(hash, query.planet_id.x);
    hash = mix_hash(hash, query.planet_id.y);
    hash = mix_hash(hash, query.planet_id.z);
    hash = mix_hash(hash, query.planet_id.w);
    hash = mix_hash(hash, bitcast<u32>(query.relative_lod0_cell_min.x));
    hash = mix_hash(hash, bitcast<u32>(query.relative_lod0_cell_min.y));
    hash = mix_hash(hash, bitcast<u32>(query.relative_lod0_cell_min.z));
    return mix_hash(hash, query.lod);
}

fn keys_equal(entry: GpuPageTableEntry, query: GpuLookupQuery) -> bool {
    return all(entry.planet_id == query.planet_id)
        && all(entry.relative_lod0_cell_min == query.relative_lod0_cell_min)
        && entry.lod == query.lod;
}

fn lookup_page(query: GpuLookupQuery) -> GpuLookupResult {
    let start = page_hash(query) & residency.table_mask;
    var probe = 0u;
    loop {
        if (probe >= residency.max_probe) {
            break;
        }
        let entry = page_table[(start + probe) & residency.table_mask];
        if (entry.state == PAGE_TABLE_EMPTY) {
            return GpuLookupResult(0u, 0u, 0u, probe + 1u);
        }
        if (entry.state == PAGE_TABLE_OCCUPIED && keys_equal(entry, query)) {
            return GpuLookupResult(
                entry.slot,
                entry.generation_low,
                entry.generation_high,
                0x80000000u | (probe + 1u),
            );
        }
        probe += 1u;
    }
    return GpuLookupResult(0u, 0u, 0u, probe);
}

@compute @workgroup_size(64)
fn validate_lookup(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    if (index >= arrayLength(&queries) || index >= arrayLength(&results)) {
        return;
    }
    results[index] = lookup_page(queries[index]);
}
