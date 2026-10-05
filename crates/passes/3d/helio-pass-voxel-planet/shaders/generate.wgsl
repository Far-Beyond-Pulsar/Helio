// GPU-driven column generation, run allocation and publication.
// Order per frame: evict -> generate -> count -> refill -> allocate -> fixup -> publish.

struct Job {
    key0: u32,
    key1: u32,
    record: u32,
    edits: u32,
    flags: u32,  // 1 = replaces a published column
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct JobOut {
    status: u32,   // 0 ok, 1 overflow, 2 scratch full, 3 pool full, 4 skipped
    k_lo: i32,
    n_band: u32,
    n_mixed: u32,
    scratch: u32,
    size_class: u32,
    run: u32,
    pad: u32,
    mixed: array<u32, 8>,
    solid: array<u32, 8>,
}

const CLASSES: u32 = 10u;
const PAGE_UNITS: u32 = 512u;
const A_TOP: u32 = 0u;
const A_FAILS: u32 = 10u;
const A_NEED: u32 = 16u;
const A_PAGES: u32 = 30u;
const A_SCRATCH: u32 = 31u;
const MAX_BAND: u32 = 256u;

@group(0) @binding(7) var<storage, read> jobs: array<Job>;
@group(0) @binding(8) var<storage, read_write> job_out: array<JobOut>;
@group(0) @binding(9) var<storage, read_write> scratch: array<u32>;
@group(0) @binding(10) var<storage, read_write> alloc: array<atomic<i32>, 32>;
@group(0) @binding(11) var<storage, read_write> free_runs: array<u32>;
@group(0) @binding(12) var<storage, read_write> free_pages: array<u32>;
@group(0) @binding(13) var<storage, read> evictions: array<u32>;
// Failed jobs (key0, key1, status, 0), appended until the CPU copies them.
@group(0) @binding(17) var<storage, read_write> failures: array<u32>;
// Per pool page: free runs and size class. A page whose runs are all free
// returns to the free page stack under pool pressure (allocator_recycle).
struct PageMeta {
    free: atomic<u32>,
    size_class: u32,
}
@group(0) @binding(18) var<storage, read_write> page_meta: array<PageMeta>;

fn class_offset(c: u32) -> u32 {
    let p = frame.counts.w;
    if c == 0u { return 0u; }
    return 2u * p - (p >> (c - 1u));
}

fn job_index(wg: vec3<u32>) -> u32 {
    return wg.x + wg.y * 32768u;
}

var<workgroup> g_band: array<atomic<i32>, 2>;
var<workgroup> g_words: array<atomic<u32>, 16>;
var<workgroup> g_masks: array<atomic<u32>, 16>;
var<workgroup> g_any: array<atomic<u32>, 2>;
var<workgroup> g_base: u32;

@compute @workgroup_size(64)
fn generate(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let index = job_index(wg);
    if index >= frame.counts.x { return; }
    let job = jobs[index];
    let face = (job.key0 >> 24u) & 7u;
    let level = job.key0 >> 27u;
    let ci = i32(job.key0 & 0xffffffu);
    let cj = bitcast<i32>(job.key1);
    let x = i32(li & 7u);
    let y = i32(li >> 3u);
    let i = ci * 8 + x;
    let j = cj * 8 + y;
    let top = top_cells(field_height(face, i, j, level), level);
    if li == 0u {
        atomicStore(&g_band[0], 0x7fffffff);
        atomicStore(&g_band[1], -0x7fffffff);
        atomicStore(&g_any[0], 0u);
        atomicStore(&g_any[1], 0u);
    }
    if li < 16u {
        atomicStore(&g_words[li], 0u);
        atomicStore(&g_masks[li], 0u);
    }
    workgroupBarrier();
    // Everything below the band is solid ground, everything above is air.
    // The band follows the terrain wherever it is: clamping its top to the
    // datum (a sea-level leftover) made every column below datum claim the
    // air up to height 0 as occupied (column tops, summary blocks and level
    // tops), so rays stepped cell by cell through it (5-20x primary cost in
    // lowland below datum) and each column stored the empty bricks.
    atomicMin(&g_band[0], top - 1);
    atomicMax(&g_band[1], top);
    if job.edits != 0u {
        let count = edit_refs[job.edits - 1u];
        for (var e = li; e < count; e += 64u) {
            let b = brushes[edit_refs[job.edits + e]];
            if b.radius_half < (1u << level) { continue; }
            let r = i32(b.radius_half);
            let shift = level + 1u;
            let lo = (b.center.z - r) >> shift;
            let hi = ((b.center.z + r) >> shift) + 1;
            let op = (b.flags >> 4u) & 3u;
            if op == 0u { atomicMin(&g_band[0], lo - 1); }
            if op == 1u { atomicMax(&g_band[1], hi + 1); }
        }
    }
    workgroupBarrier();
    let lo_cell = atomicLoad(&g_band[0]);
    let hi_cell = atomicLoad(&g_band[1]);
    let k_lo = lo_cell >> 3u;
    let k_hi = ((max(hi_cell, lo_cell + 1) - 1) >> 3u) + 1;
    let n_band = u32(k_hi - k_lo);
    if n_band > MAX_BAND {
        if li == 0u { job_out[index].status = 1u; }
        return;
    }
    if li == 0u {
        let need = i32(1u + n_band);
        let base = atomicAdd(&alloc[A_SCRATCH], need);
        if u32(base + need) * UNIT_WORDS > arrayLength(&scratch) {
            g_base = NONE;
        } else {
            g_base = u32(base);
        }
    }
    let base = workgroupUniformLoad(&g_base);
    if base == NONE {
        if li == 0u { job_out[index].status = 2u; }
        return;
    }
    // Column tops relative to the band base, one byte per cell.
    atomicOr(&g_words[li >> 2u], u32(clamp(top - k_lo * 8, 0, 255)) << ((li & 3u) * 8u));
    workgroupBarrier();
    if li < 16u {
        scratch[base * UNIT_WORDS + li] = atomicLoad(&g_words[li]);
        atomicStore(&g_words[li], 0u);
    }
    workgroupBarrier();
    for (var b = 0u; b < n_band; b++) {
        for (var z = 0u; z < 8u; z++) {
            let k = (k_lo + i32(b)) * 8 + i32(z);
            var kind = terrain_kind(top, k);
            if job.edits != 0u {
                let c = vec3<i32>(center_half(i, level), center_half(j, level), center_half(k, level));
                kind = apply_edits(job.edits, level, c, kind).x;
            }
            if kind != 0u {
                let bit = li + z * 64u;
                atomicOr(&g_words[bit >> 5u], 1u << (bit & 31u));
            }
        }
        workgroupBarrier();
        if li < 16u {
            let w = atomicExchange(&g_words[li], 0u);
            scratch[(base + 1u + b) * UNIT_WORDS + li] = w;
            if w != 0u { atomicOr(&g_any[0], 1u); }
            if w != 0xffffffffu { atomicOr(&g_any[1], 1u); }
        }
        workgroupBarrier();
        if li == 0u {
            let some = atomicExchange(&g_any[0], 0u) != 0u;
            let holes = atomicExchange(&g_any[1], 0u) != 0u;
            if some && holes {
                atomicOr(&g_masks[b >> 5u], 1u << (b & 31u));
            } else if some {
                atomicOr(&g_masks[8u + (b >> 5u)], 1u << (b & 31u));
            }
        }
        workgroupBarrier();
    }
    if li == 0u {
        var out: JobOut;
        out.status = 0u;
        out.k_lo = k_lo;
        out.n_band = n_band;
        out.scratch = base;
        var mixed = 0u;
        for (var w = 0u; w < 8u; w++) {
            out.mixed[w] = atomicLoad(&g_masks[w]);
            out.solid[w] = atomicLoad(&g_masks[8u + w]);
            mixed += countOneBits(out.mixed[w]);
        }
        out.n_mixed = mixed;
        job_out[index] = out;
    }
}

fn run_units(o: JobOut) -> u32 {
    return select(1u, 2u, o.n_band > 32u) + o.n_mixed;
}

fn class_of(units: u32) -> u32 {
    return 32u - countLeadingZeros(max(units, 1u) - 1u);
}

@compute @workgroup_size(64)
fn count(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if index >= frame.counts.x || job_out[index].status != 0u { return; }
    let c = class_of(run_units(job_out[index]));
    job_out[index].size_class = c;
    atomicAdd(&alloc[A_NEED + c], 1);
}

@compute @workgroup_size(16)
fn refill(@builtin(local_invocation_index) c: u32) {
    if c >= CLASSES { return; }
    let need = atomicExchange(&alloc[A_NEED + c], 0);
    var top = atomicLoad(&alloc[A_TOP + c]);
    let size = 1u << c;
    let runs = PAGE_UNITS / size;
    let offset = class_offset(c);
    loop {
        if top >= need { break; }
        let page_index = atomicSub(&alloc[A_PAGES], 1) - 1;
        if page_index < 0 {
            atomicAdd(&alloc[A_PAGES], 1);
            break;
        }
        let page = free_pages[page_index];
        page_meta[page].size_class = c;
        atomicStore(&page_meta[page].free, runs);
        for (var r = 0u; r < runs; r++) {
            free_runs[offset + u32(top)] = page * PAGE_UNITS + r * size;
            top++;
        }
    }
    atomicStore(&alloc[A_TOP + c], top);
}

@compute @workgroup_size(64)
fn allocate(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if index >= frame.counts.x || job_out[index].status != 0u { return; }
    let c = job_out[index].size_class;
    let slot = atomicSub(&alloc[A_TOP + c], 1) - 1;
    if slot < 0 {
        job_out[index].status = 3u;
        return;
    }
    let run = free_runs[class_offset(c) + u32(slot)];
    job_out[index].run = run;
    atomicSub(&page_meta[run / PAGE_UNITS].free, 1u);
}

@compute @workgroup_size(16)
fn fixup(@builtin(local_invocation_index) c: u32) {
    if c < CLASSES {
        atomicMax(&alloc[A_TOP + c], 0);
    }
    if c == 0u {
        atomicStore(&alloc[A_SCRATCH], 0);
    }
}

fn free_run(c: Column) {
    let cls = (c.info >> 18u) & 15u;
    let slot = atomicAdd(&alloc[A_TOP + cls], 1);
    free_runs[class_offset(cls) + u32(slot)] = c.run;
    atomicAdd(&page_meta[c.run / PAGE_UNITS].free, 1u);
}

@compute @workgroup_size(64)
fn evict(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= frame.counts.y { return; }
    let r = evictions[id.x];
    let c = records[r];
    if (c.info & INFO_VALID) != 0u {
        free_run(c);
        let ci = i32(c.key0 & 0xffffffu);
        let cj = bitcast<i32>(c.key1);
        for (var tier = 1u; tier <= 3u; tier++) {
            let bi = ci >> (2u * tier);
            let bj = cj >> (2u * tier);
            let slot = block_slot(c.key0 >> 27u, (c.key0 >> 24u) & 7u, tier, bi, bj) * 4u;
            if atomicLoad(&block_state[slot]) == bi && atomicLoad(&block_state[slot + 1u]) == bj {
                atomicSub(&block_state[slot + 3u], 1);
            }
        }
    }
    records[r].info = 0u;
    records[r].key0 = NONE;
}

// Table slot writes, stored as (slot, value) pairs after the eviction list.
@compute @workgroup_size(64)
fn patch_table(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= frame.extra.x { return; }
    let at = frame.counts.y + id.x * 2u;
    table[evictions[at]] = evictions[at + 1u];
}

// Summary block (re)initialisation: (slot, bi, bj) triples after the table
// patches. `bi = -1` releases a slot.
@compute @workgroup_size(64)
fn patch_blocks(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= frame.extra.z { return; }
    let at = frame.counts.y + frame.extra.x * 2u + id.x * 3u;
    let slot = evictions[at] * 4u;
    atomicStore(&block_state[slot], bitcast<i32>(evictions[at + 1u]));
    atomicStore(&block_state[slot + 1u], bitcast<i32>(evictions[at + 2u]));
    atomicStore(&block_state[slot + 2u], -0x3fffffff);
    atomicStore(&block_state[slot + 3u], 0);
}

@compute @workgroup_size(64)
fn publish(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let index = job_index(wg);
    if index >= frame.counts.x { return; }
    let job = jobs[index];
    let o = job_out[index];
    if o.status != 0u {
        // Report the failure (the CPU retries all but band overflows); a new
        // column stays unpublished, a replaced column keeps its old data.
        if li == 0u {
            let at = u32(atomicAdd(&alloc[A_FAILS], 1)) * 4u;
            if at + 3u < arrayLength(&failures) {
                failures[at] = job.key0;
                failures[at + 1u] = job.key1;
                failures[at + 2u] = o.status;
            }
        }
        if li == 0u && (job.flags & 1u) == 0u {
            var c: Column;
            c.key0 = job.key0;
            c.key1 = job.key1;
            c.info = select(0u, INFO_OVERFLOW, o.status == 1u);
            records[job.record] = c;
        }
        return;
    }
    let ext = o.n_band > 32u;
    let header = select(1u, 2u, ext);
    if li < 16u {
        pool[o.run * UNIT_WORDS + li] = scratch[o.scratch * UNIT_WORDS + li];
        if ext {
            pool[(o.run + 1u) * UNIT_WORDS + li] = select(o.solid[li - 8u], o.mixed[li], li < 8u);
        }
    }
    let total = o.n_band * UNIT_WORDS;
    for (var w = li; w < total; w += 64u) {
        let b = w / UNIT_WORDS;
        let bit = b & 31u;
        let word = b >> 5u;
        if ((o.mixed[word] >> bit) & 1u) == 0u { continue; }
        var rank = countOneBits(o.mixed[word] & ((1u << bit) - 1u));
        for (var q = 0u; q < word; q++) { rank += countOneBits(o.mixed[q]); }
        pool[(o.run + header + rank) * UNIT_WORDS + (w % UNIT_WORDS)] =
            scratch[(o.scratch + 1u + b) * UNIT_WORDS + (w % UNIT_WORDS)];
    }
    if li == 0u {
        let previous = records[job.record];
        let was_published = (previous.info & INFO_VALID) != 0u && previous.key0 == job.key0 && previous.key1 == job.key1;
        if (job.flags & 1u) != 0u && (previous.info & INFO_VALID) != 0u {
            free_run(previous);
        }
        // Exact top: the highest occupied layer of the highest non-air band
        // brick (mixed bricks are scanned by z layer, two words each).
        let band_top = (o.k_lo + i32(o.n_band)) * 8;
        var exact = o.k_lo * 8;
        for (var b = i32(o.n_band) - 1; b >= 0; b--) {
            let word = u32(b) >> 5u;
            let bit = u32(b) & 31u;
            if ((o.solid[word] >> bit) & 1u) != 0u {
                exact = (o.k_lo + b + 1) * 8;
                break;
            }
            if ((o.mixed[word] >> bit) & 1u) != 0u {
                let base = (o.scratch + 1u + u32(b)) * UNIT_WORDS;
                var z = 7;
                while z > 0 && (scratch[base + 2u * u32(z)] | scratch[base + 2u * u32(z) + 1u]) == 0u { z -= 1; }
                exact = (o.k_lo + b) * 8 + z + 1;
                break;
            }
        }
        let gap = u32(clamp(band_top - exact, 0, 7));
        let top_cell = band_top - i32(gap);
        let ci = i32(job.key0 & 0xffffffu);
        let cj = bitcast<i32>(job.key1);
        for (var tier = 1u; tier <= 3u; tier++) {
            let bi = ci >> (2u * tier);
            let bj = cj >> (2u * tier);
            let slot = block_slot(job.key0 >> 27u, (job.key0 >> 24u) & 7u, tier, bi, bj) * 4u;
            if atomicLoad(&block_state[slot]) == bi && atomicLoad(&block_state[slot + 1u]) == bj {
                atomicMax(&block_state[slot + 2u], top_cell);
                if !was_published { atomicAdd(&block_state[slot + 3u], 1); }
            }
        }
        var c: Column;
        c.key0 = job.key0;
        c.key1 = job.key1;
        c.k_lo = o.k_lo;
        c.info = o.n_band | (o.n_mixed << 9u) | (o.size_class << 18u) | (gap << 22u) | select(0u, INFO_EXT, ext) | INFO_VALID;
        c.run = o.run;
        c.mixed = o.mixed[0];
        c.solid = o.solid[0];
        c.edits = job.edits;
        records[job.record] = c;
        let level = job.key0 >> 27u;
        atomicMax(&level_tops[level], top_cell << level);
    }
}

// Suffix maxima of the per-level occupied tops (entries 32.. hold the max of
// this level and every coarser one) for early sky exits.
@compute @workgroup_size(1)
fn level_suffix() {
    var best = -0x7fffffff;
    for (var level = 31i; level >= 0; level--) {
        best = max(best, atomicLoad(&level_tops[level]));
        atomicStore(&level_tops[32 + level], best);
    }
}
