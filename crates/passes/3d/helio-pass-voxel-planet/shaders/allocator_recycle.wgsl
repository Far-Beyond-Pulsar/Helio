// Pool page recycling, run only under pool pressure, between eviction and
// generation. Size classes take whole pages from the free page stack and
// never gave them back, so travel through changing terrain stranded pages in
// classes it no longer needed until generation failed for want of space.
// `reclaim` returns pages whose runs are all free; `compact` then removes
// those pages' runs from the class stacks (into a second buffer that is
// copied back) and `finish` publishes the new stack tops. No live run moves.
struct PageMeta {
    free: atomic<u32>,
    size_class: u32,
}

@group(0) @binding(0) var<storage, read_write> alloc: array<atomic<i32>, 32>;
@group(0) @binding(1) var<storage, read_write> page_meta: array<PageMeta>;
@group(0) @binding(2) var<storage, read> runs: array<u32>;
@group(0) @binding(3) var<storage, read_write> compacted: array<u32>;
@group(0) @binding(4) var<storage, read_write> free_pages: array<u32>;
@group(0) @binding(5) var<storage, read_write> counts: array<atomic<u32>, 16>;

const CLASSES: u32 = 10u;
const PAGE_UNITS: u32 = 512u;
const A_PAGES: u32 = 30u;
const UNASSIGNED: u32 = 0xffffffffu;

@compute @workgroup_size(64)
fn reclaim(@builtin(global_invocation_id) id: vec3<u32>) {
    let page = id.x;
    if page >= arrayLength(&page_meta) { return; }
    let c = page_meta[page].size_class;
    if c >= CLASSES || atomicLoad(&page_meta[page].free) != (PAGE_UNITS >> c) { return; }
    page_meta[page].size_class = UNASSIGNED;
    atomicStore(&page_meta[page].free, 0u);
    free_pages[u32(atomicAdd(&alloc[A_PAGES], 1))] = page;
}

// Class c's stack starts at offset(c) and holds at most units >> c runs
// (the layout of `class_offset` in generate.wgsl).
@compute @workgroup_size(256)
fn compact(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let units = arrayLength(&runs) / 2u;
    let index = (wg.x + wg.y * 32768u) * 256u + li;
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
    let run = runs[index];
    if page_meta[run / PAGE_UNITS].size_class != c { return; }
    compacted[offset + atomicAdd(&counts[c], 1u)] = run;
}

@compute @workgroup_size(16)
fn finish(@builtin(local_invocation_index) c: u32) {
    if c < CLASSES {
        atomicStore(&alloc[c], i32(atomicLoad(&counts[c])));
    }
}
