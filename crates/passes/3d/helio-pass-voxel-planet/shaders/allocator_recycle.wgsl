// Pressure-only recycling. Ordered after eviction and before generation:
// reclaim whole free pages, compact all class stacks, then publish new tops.
// No live allocation moves, and reclaimed pages cannot be reused until every
// stale run has been removed from the stacks by a separate dispatch.
struct PageMeta {
    free: atomic<u32>,
    size_class: u32,
}

@group(0) @binding(0) var<storage, read_write> alloc: array<atomic<i32>, 32>;
@group(0) @binding(1) var<storage, read_write> pages: array<PageMeta>;
@group(0) @binding(2) var<storage, read> old_runs: array<u32>;
@group(0) @binding(3) var<storage, read_write> new_runs: array<u32>;
@group(0) @binding(4) var<storage, read_write> free_pages: array<u32>;
@group(0) @binding(5) var<storage, read_write> counts: array<atomic<u32>, 16>;

const CLASSES: u32 = 10u;
const PAGE_UNITS: u32 = 512u;
const UNASSIGNED: u32 = 0xffffffffu;

@compute @workgroup_size(128)
fn reclaim_pages(@builtin(global_invocation_id) id: vec3<u32>) {
    let page = id.x;
    if page >= arrayLength(&pages) { return; }
    let c = pages[page].size_class;
    if c >= CLASSES || atomicLoad(&pages[page].free) != (PAGE_UNITS >> c) { return; }
    pages[page].size_class = UNASSIGNED;
    atomicStore(&pages[page].free, 0u);
    let slot = atomicAdd(&alloc[30], 1);
    free_pages[u32(slot)] = page;
    atomicSub(&alloc[26], 1);
    atomicAdd(&alloc[27], 1);
}

@compute @workgroup_size(256)
fn compact_runs(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let units = arrayLength(&old_runs) / 2u;
    let index = (wg.x + wg.y * 32768u) * 256u + li;
    if index >= units * 2u { return; }
    var c = 0u;
    var offset = 0u;
    var end = units;
    loop {
        if index < end { break; }
        c++;
        if c >= CLASSES { return; }
        offset = end;
        end += units >> c;
    }
    if index - offset >= u32(max(atomicLoad(&alloc[c]), 0)) { return; }
    let run = old_runs[index];
    let page = run / PAGE_UNITS;
    if page >= arrayLength(&pages) || pages[page].size_class != c { return; }
    let slot = atomicAdd(&counts[c], 1u);
    new_runs[offset + slot] = run;
}

@compute @workgroup_size(16)
fn finish_recycle(@builtin(local_invocation_index) c: u32) {
    if c < CLASSES { atomicStore(&alloc[c], i32(atomicLoad(&counts[c]))); }
}
