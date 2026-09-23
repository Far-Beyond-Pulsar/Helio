//! GPU-driven object sort/group/batch pipeline.
//!
//! Turns every live `StaticObjectComponent` SceneDB row into the exact same
//! shape `Scene::rebuild_instance_buffers`'s CPU implementation used to
//! produce: a sorted-by-`(material_class, graph_hash, mesh_id, material_id)`
//! `instances`/`aabbs` array, one `draw_calls` entry per contiguous
//! same-`(mesh_id, material_id)` run, and `(class, graph_hash, start, count)`
//! ranges over those draw-call groups (split into opaque/transparent/
//! forward), plus the static/movable shadow-partitioned indirect lists —
//! with no per-frame CPU iteration over scene objects at all.
//!
//! # Pipeline stages (one `RenderPass::execute()` per frame)
//!
//! 1. `cs_gather` — one thread per `StaticObjectComponent` row capacity slot.
//!    Skips rows nobody has ever written (`mesh_generation == 0u`, the
//!    `Zeroable` default -- never a real generation, see `mesh_generation`'s
//!    field doc in `helio_pass_gbuffer::components`). Live rows compute this
//!    migration's packed sort key and atomically compact directly into
//!    `keys_a`/`indices_a` -- the same buffers the sort's bit-0 pass reads
//!    (same technique as `helio-pass-sprite-cull`'s `cs_cull`: a single
//!    global atomic counter, order doesn't matter yet).
//! 2. `cs_prepare` — turns the gather's live count into `frame_uniform`
//!    (`num_blocks`) and `dispatch_args` (indirect dispatch size for every
//!    later stage) — same technique as `helio-pass-sprite-cull`'s
//!    `sprite_sort.wgsl::cs_prepare`.
//! 3. `cs_histogram` / `cs_scan` / `cs_scatter` — 32-pass LSD radix sort of
//!    `(key, index)` pairs, copied near-verbatim from `helio-pass-sprite-
//!    cull`'s `sprite_sort.wgsl` (a proven, independently-tested, *stable*
//!    sort -- see that file's module doc for why stability specifically
//!    needed a Hillis-Steele rank instead of a per-bucket atomic counter).
//!    Ends in `keys_a`/`indices_a` after 32 (even) passes.
//! 4. `cs_final_gather` — re-reads `static_objects[indices_a[i]]` in final
//!    sorted order and writes `instances[i]`/`aabbs[i]`. (`keys_a`/
//!    `indices_a` only ever carry the sort key + row index, not the full
//!    ~236-byte row, to keep the 32-pass sort's per-element payload small.)
//! 5. `cs_group_local_scan` / `cs_group_block_scan` / `cs_group_write` —
//!    detect draw-call group boundaries (`sorted_keys_a[i] !=
//!    sorted_keys_a[i-1]`) and turn them into *stable* (position-ordered,
//!    not execution-ordered) group indices via the same two-level
//!    Hillis-Steele-per-block + single-thread-cross-block-prefix-sum scan
//!    technique `cs_scatter`'s rank calculation already uses at the
//!    single-block scale -- generalized here to however many blocks the
//!    live object count spans. Writes `group_starts[]` (one entry per draw
//!    group, plus a trailing sentinel `group_starts[group_count] =
//!    live_count`) and per-group `(material_class, graph_hash, is_
//!    transparent, is_forward)` metadata.
//! 6. `cs_build_draw_calls` — one thread per group: assembles `draw_calls[g]`
//!    from `group_starts[g]`/`group_starts[g+1]` and the mesh's static draw
//!    parameters (read from the group's first sorted instance).
//! 7. `cs_range_local_scan` / `cs_range_block_scan` / `cs_range_write` — same
//!    two-level scan technique again, this time detecting boundaries between
//!    *groups* that no longer share `(material_class, graph_hash)`, to build
//!    the `(class, graph_hash, start, count)` range tuples PSO selection
//!    needs -- `start`/`count` here are draw-*group* indices, matching what
//!    `helio-pass-gbuffer`'s `multi_draw_indexed_indirect(indirect, start *
//!    20, count)` call expects (`start`/`count` index the `draw_calls`/
//!    `indirect` array directly, not `instances`). Split into three parallel
//!    output arrays (opaque/transparent/forward). A shading-category change
//!    also splits the run: materials sharing a class and graph can still
//!    require different passes.
//! 8. `cs_shadow_partition` — one thread per LIVE sorted instance: atomically
//!    appends a one-instance `DrawIndexedIndirectArgs` into either
//!    `shadow_static_indirect` or `shadow_movable_indirect` depending on
//!    `libhelio::INSTANCE_FLAG_MOVABLE` in the instance's `flags` (default
//!    unset = static/stationary, the common case for baked level geometry).
//!
//! # Capacity, not live count, sizes every buffer
//!
//! `StaticObjectComponent`'s SceneDB buffer auto-registers at
//! `DEFAULT_AUTO_REGISTER_CAPACITY` (64) and grows unbounded as more entities
//! get the component (`pulsar_scenedb::gpu::world_mirror`'s own doc: "small
//! initial capacity, unbounded growth"), unlike the small, genuinely-fixed-
//! ceiling domains elsewhere in this migration (lights/decals/post-process
//! volumes, capped at `DEFAULT_AUTO_REGISTER_CAPACITY` on purpose). Object
//! counts routinely exceed 64 in any real scene, so `capacity` here is
//! computed fresh each frame from the *live* SceneDB buffer's actual byte
//! size (`buffer.size() / size_of::<StaticObjectRow>()`). Every scratch
//! buffer this pipeline owns (`keys_a`/`keys_b`/`instances_out`/
//! `group_starts`/the range tables/the shadow-partition buffers, all of it)
//! is grown to match -- there is no fixed object-count ceiling anywhere in
//! this pipeline. See `src/lib.rs`'s `ObjectBatchPass::ensure_capacity` for
//! the growth policy (next-power-of-two, rebuilding every affected bind
//! group, exactly the same "grow on demand" seam `GrowableBuffer`/`corona`'s
//! `upload_sort_steps` already use elsewhere in this codebase). `max_objects`/
//! `max_groups`/`max_ranges` above are that grown scratch capacity, not a
//! hardcoded constant -- `max_groups`/`max_ranges` both reuse the SAME grown
//! capacity as `max_objects` because a draw-call group needs at least one
//! live object and a range needs at least one group, so `#ranges <= #groups
//! <= #objects` always holds without a second, independently-guessed ceiling.

const WG: u32 = 256u;

// ── Uniforms ─────────────────────────────────────────────────────────────────

/// Per-frame constants, written once by `prepare()`.
struct BatchUniforms {
    /// `static_objects` buffer's current row capacity (see module doc).
    capacity: u32,
    /// Scratch buffer capacity this frame's dispatches were sized against
    /// (>= capacity; only changes when scratch buffers grow).
    max_objects: u32,
    max_groups: u32,
    max_ranges: u32,
}
@group(0) @binding(0) var<uniform> batch: BatchUniforms;

/// SceneDB's `StaticObjectComponent` row -- byte-for-byte identical to
/// `helio_pass_gbuffer::components::StaticObjectComponent`'s `#[repr(C)]`
/// Rust layout (plain fixed-size arrays throughout, not `mat4x4<f32>`/
/// `vec4<f32>`, for exactly the reason every other SceneDB-mirrored struct
/// in this codebase avoids those: WGSL's struct-member alignment for those
/// types doesn't match Rust's packed layout). Keep in lock-step with that
/// struct; `helio_pass_object_batch`'s own test suite asserts the byte size
/// matches so a drift fails loudly at `cargo test` time, not at draw time.
struct StaticObjectRow {
    mesh_slot: u32,
    mesh_generation: u32,
    material_slot: u32,
    material_generation: u32,
    transform: array<array<f32, 4>, 4>,
    prev_transform: array<array<f32, 4>, 4>,
    normal_mat: array<array<f32, 4>, 3>,
    bounds: array<f32, 4>,
    index_count: u32,
    first_index: u32,
    vertex_offset: i32,
    material_class: u32,
    graph_hash_lo: u32,
    graph_hash_hi: u32,
    flags: u32,
}
@group(0) @binding(1) var<storage, read> static_objects: array<StaticObjectRow>;

// ── Stage 1: cs_gather ──────────────────────────────────────────────────────

// Writes DIRECTLY into `keys_a`/`indices_a` -- the same buffers the radix
// sort's bit-0 pass reads as its source (see stage 3's `src_keys_h`
// binding doc). There is no separate "raw" staging pair: gather's output
// and the sort's input are the same buffer by construction, so there is no
// copy step to forget between them.
@group(0) @binding(2) var<storage, read_write> gathered_keys: array<u32>;
@group(0) @binding(3) var<storage, read_write> gathered_indices: array<u32>;
/// `[_, live_count, _, _]` as a `DrawIndexedIndirectArgs`-shaped buffer so
/// `cs_gather`'s atomic bump doubles as this frame's authoritative live
/// count with no separate readback -- mirrors `helio-pass-sprite-cull`'s
/// `indirect_args[1]` reuse exactly.
@group(0) @binding(4) var<storage, read_write> gather_count: array<atomic<u32>>;

/// A hierarchical per-row sort key matching the CPU reference's tuple sort
/// PRIORITY ORDER `(material_class, graph_hash, mesh_id, material_id)`, not
/// just its highest field: `material_class` occupies the top 8 bits
/// (exact, no hashing -- assumes <=256 material classes, matching every
/// other `& 0xFFu`-masked use of `material_class` in this codebase), a
/// 12-bit hash of `graph_hash` occupies the next 12 bits, and a 12-bit hash
/// of `(mesh_slot, material_slot)` occupies the low 12 bits.
///
/// The field SEPARATION (not just concatenation into one combined hash) is
/// the whole point: two rows sharing the same `(class, graph_hash)` but
/// different mesh/material get the same top 20 bits and therefore always
/// sort into one CONTIGUOUS run (differing only in the low, mesh/material-
/// derived bits) -- exactly what `cs_range_local_scan`/`cs_range_write`
/// need to find every draw-call group belonging to one range without
/// re-scanning the whole array. An earlier version of this function XORed
/// all three fields into a single 24-bit hash instead, which does NOT
/// preserve this property (`graph_hash`'s contribution gets scrambled by
/// whichever mesh/material bits happened to be set) and produced
/// fragmented, non-contiguous ranges for the same `(class, graph_hash)` --
/// caught by `tests/gpu_object_batch_validation.rs`, not by inspection.
///
/// Both 12-bit hashes CAN still collide (two different `graph_hash` values
/// landing in the same 4096-slot bucket, say) -- exactly as before, this
/// never causes a wrong MERGE: `cs_group_local_scan`/`cs_group_write`
/// compare rows' REAL `mesh_slot`/`material_slot` (`same_draw_group`), and
/// `cs_range_local_scan`/`cs_range_write` compare groups' REAL
/// `(material_class, graph_hash)`, never the hash. A collision only costs a
/// slightly less-tidy internal sort order within the affected bucket.
fn hash12(x: u32) -> u32 {
    var h = x;
    h = h ^ (h >> 16u);
    h = h * 0x45D9F3Bu;
    h = h ^ (h >> 13u);
    return h & 0xFFFu;
}

fn compute_sort_key(row: StaticObjectRow) -> u32 {
    let graph_hash_bucket = hash12(row.graph_hash_lo ^ (row.graph_hash_hi * 0x9E3779B1u));
    let mesh_material_bucket = hash12(row.mesh_slot ^ (row.material_slot * 0x85EBCA6Bu));
    return ((row.material_class & 0xFFu) << 24u) | (graph_hash_bucket << 12u) | mesh_material_bucket;
}

@compute @workgroup_size(WG)
fn cs_gather(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= batch.capacity {
        return;
    }
    let row = static_objects[i];
    if row.mesh_generation == 0u {
        return; // Never written -- Zeroable default, not a live entity.
    }
    let slot = atomicAdd(&gather_count[1], 1u);
    if slot < batch.max_objects {
        gathered_keys[slot] = compute_sort_key(row);
        gathered_indices[slot] = i;
    }
}

// ── Stage 2: cs_prepare ──────────────────────────────────────────────────────

struct FrameUniform {
    count: u32,
    num_blocks: u32,
}
@group(0) @binding(0) var<storage, read> prep_gather_count: array<u32>;
@group(0) @binding(1) var<storage, read_write> frame_uniform_rw: FrameUniform;
@group(0) @binding(2) var<storage, read_write> dispatch_args: array<u32>;

@compute @workgroup_size(1)
fn cs_prepare() {
    let count = prep_gather_count[1]; // gather_count[1] -- see cs_gather's atomicAdd target
    frame_uniform_rw.count = count;
    let num_blocks = max((count + WG - 1u) / WG, 1u);
    frame_uniform_rw.num_blocks = num_blocks;
    // `dispatch_workgroups_indirect` reads all three dimensions -- a y or z
    // of 0 dispatches ZERO total workgroups (x*y*z), not "1 workgroup in x
    // only". Must write all three every time, not just x.
    dispatch_args[0] = num_blocks;
    dispatch_args[1] = 1u;
    dispatch_args[2] = 1u;
}

// ── Stage 3: radix sort (cs_histogram / cs_scan / cs_scatter) ──────────────
//
// Copied near-verbatim from `helio-pass-sprite-cull`'s `sprite_sort.wgsl` --
// see that file's module doc for the full stability rationale. Fully
// generic over `(key, index)` pairs; nothing here is object-batch-specific.

struct SortUniforms {
    bit: u32,
}

@group(0) @binding(0) var<uniform> su_h: SortUniforms;
@group(0) @binding(1) var<uniform> fu_h: FrameUniform;
@group(0) @binding(2) var<storage, read> src_keys_h: array<u32>;
@group(0) @binding(3) var<storage, read_write> block_hist_h: array<u32>;

var<workgroup> hist_ones: atomic<u32>;
var<workgroup> hist_total: atomic<u32>;

@compute @workgroup_size(WG)
fn cs_histogram(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    if lid.x == 0u {
        atomicStore(&hist_ones, 0u);
        atomicStore(&hist_total, 0u);
    }
    workgroupBarrier();

    if gid.x < fu_h.count {
        atomicAdd(&hist_total, 1u);
        let bit = (src_keys_h[gid.x] >> su_h.bit) & 1u;
        if bit == 1u {
            atomicAdd(&hist_ones, 1u);
        }
    }
    workgroupBarrier();

    if lid.x == 0u {
        let ones = atomicLoad(&hist_ones);
        let total = atomicLoad(&hist_total);
        block_hist_h[wgid.x * 2u + 0u] = total - ones;
        block_hist_h[wgid.x * 2u + 1u] = ones;
    }
}

@group(0) @binding(0) var<uniform> fu_s: FrameUniform;
@group(0) @binding(1) var<storage, read_write> block_hist_s: array<u32>;

@compute @workgroup_size(1)
fn cs_scan() {
    var total_zeros = 0u;
    var total_ones = 0u;
    for (var blk = 0u; blk < fu_s.num_blocks; blk++) {
        total_zeros += block_hist_s[blk * 2u + 0u];
        total_ones += block_hist_s[blk * 2u + 1u];
    }
    let base_zero = 0u;
    let base_one = total_zeros;

    var running_zero = base_zero;
    var running_one = base_one;
    for (var blk = 0u; blk < fu_s.num_blocks; blk++) {
        let cz = block_hist_s[blk * 2u + 0u];
        let co = block_hist_s[blk * 2u + 1u];
        block_hist_s[blk * 2u + 0u] = running_zero;
        block_hist_s[blk * 2u + 1u] = running_one;
        running_zero += cz;
        running_one += co;
    }
}

@group(0) @binding(0) var<uniform> su_c: SortUniforms;
@group(0) @binding(1) var<uniform> fu_c: FrameUniform;
@group(0) @binding(2) var<storage, read> src_keys_c: array<u32>;
@group(0) @binding(3) var<storage, read> src_indices_c: array<u32>;
@group(0) @binding(4) var<storage, read_write> dst_keys_c: array<u32>;
@group(0) @binding(5) var<storage, read_write> dst_indices_c: array<u32>;
@group(0) @binding(6) var<storage, read> block_offsets_c: array<u32>;

var<workgroup> scan_buf: array<u32, 256>;

@compute @workgroup_size(WG)
fn cs_scatter(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let has_elem = gid.x < fu_c.count;
    let key = select(0u, src_keys_c[gid.x], has_elem);
    let bit = select(0u, (key >> su_c.bit) & 1u, has_elem);

    scan_buf[lid.x] = bit;
    workgroupBarrier();

    var offset = 1u;
    loop {
        if offset >= WG {
            break;
        }
        var v = 0u;
        if lid.x >= offset {
            v = scan_buf[lid.x - offset];
        }
        workgroupBarrier();
        scan_buf[lid.x] += v;
        workgroupBarrier();
        offset = offset * 2u;
    }
    let inclusive_ones = scan_buf[lid.x];

    if !has_elem {
        return;
    }

    var local_pos: u32;
    if bit == 1u {
        local_pos = inclusive_ones - 1u;
    } else {
        local_pos = (lid.x + 1u - inclusive_ones) - 1u;
    }

    let base = block_offsets_c[wgid.x * 2u + bit];
    let dst = base + local_pos;
    dst_keys_c[dst] = key;
    dst_indices_c[dst] = src_indices_c[gid.x];
}

// ── Stage 4: cs_final_gather ─────────────────────────────────────────────────

@group(0) @binding(0) var<uniform> fu_fg: FrameUniform;
@group(0) @binding(1) var<storage, read> sorted_indices_fg: array<u32>;
@group(0) @binding(2) var<storage, read> static_objects_fg: array<StaticObjectRow>;

struct GpuInstanceDataOut {
    model: array<array<f32, 4>, 4>,
    normal_mat: array<array<f32, 4>, 3>,
    bounds: array<f32, 4>,
    prev_model: array<array<f32, 4>, 4>,
    mesh_id: u32,
    material_id: u32,
    flags: u32,
    lightmap_index: u32,
}
// Layout mirrors `GpuAabb` in indirect_dispatch.wgsl (32 bytes: min, pad, max, pad).
struct GpuInstanceAabbOut {
    min: array<f32, 3>,
    _pad0: f32,
    max: array<f32, 3>,
    _pad1: f32,
}
@group(0) @binding(3) var<storage, read_write> instances_out: array<GpuInstanceDataOut>;
@group(0) @binding(4) var<storage, read_write> aabbs_out: array<GpuInstanceAabbOut>;

@compute @workgroup_size(WG)
fn cs_final_gather(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= fu_fg.count {
        return;
    }
    let row = static_objects_fg[sorted_indices_fg[i]];
    instances_out[i] = GpuInstanceDataOut(
        row.transform,
        row.normal_mat,
        row.bounds,
        row.prev_transform,
        row.mesh_slot,
        row.material_slot,
        row.flags,
        0xFFFFFFFFu,
    );
    // Conservative world AABB: the bounding sphere's box.
    let r = row.bounds[3];
    aabbs_out[i] = GpuInstanceAabbOut(
        array<f32, 3>(row.bounds[0] - r, row.bounds[1] - r, row.bounds[2] - r),
        0.0,
        array<f32, 3>(row.bounds[0] + r, row.bounds[1] + r, row.bounds[2] + r),
        0.0,
    );
}

// ── Stage 5: group boundary detection (two-level scan) ──────────────────────
//
// Boundaries here MUST compare the real `(mesh_slot, material_slot)` pair
// (via `static_objects[sorted_indices[i]]`), never just `sorted_keys[i]`.
// The sort key is a 24-bit HASH (see `compute_sort_key`'s doc) -- two
// genuinely different `(mesh, material)` pairs can collide onto the same
// key, and the radix sort would then place them adjacently. Treating equal
// KEYS as equal GROUPS would silently merge those two different objects
// into one draw call, using only the first one's mesh/material for the
// whole group -- a real (if rare) rendering-correctness bug, not just a
// missed-optimization one. Comparing the real fields makes a collision cost
// only a slightly less-tidy sort order (two unrelated groups sharing a
// key-driven sort position), never a wrong merge.

@group(0) @binding(0) var<uniform> fu_gls: FrameUniform;
@group(0) @binding(1) var<storage, read> sorted_indices_gls: array<u32>;
@group(0) @binding(2) var<storage, read> static_objects_gls: array<StaticObjectRow>;
@group(0) @binding(3) var<storage, read_write> local_group_rank: array<u32>;
@group(0) @binding(4) var<storage, read_write> block_group_totals: array<u32>;

var<workgroup> group_scan_buf: array<u32, 256>;

fn same_draw_group(a: StaticObjectRow, b: StaticObjectRow) -> bool {
    return a.mesh_slot == b.mesh_slot && a.material_slot == b.material_slot;
}

@compute @workgroup_size(WG)
fn cs_group_local_scan(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let has_elem = gid.x < fu_gls.count;
    var is_start = 0u;
    if has_elem {
        if gid.x == 0u {
            is_start = 1u;
        } else {
            let cur = static_objects_gls[sorted_indices_gls[gid.x]];
            let prev = static_objects_gls[sorted_indices_gls[gid.x - 1u]];
            if !same_draw_group(cur, prev) {
                is_start = 1u;
            }
        }
    }
    group_scan_buf[lid.x] = is_start;
    workgroupBarrier();

    var offset = 1u;
    loop {
        if offset >= WG {
            break;
        }
        var v = 0u;
        if lid.x >= offset {
            v = group_scan_buf[lid.x - offset];
        }
        workgroupBarrier();
        group_scan_buf[lid.x] += v;
        workgroupBarrier();
        offset = offset * 2u;
    }

    if has_elem {
        local_group_rank[gid.x] = group_scan_buf[lid.x];
    }
    if lid.x == WG - 1u {
        block_group_totals[wgid.x] = group_scan_buf[lid.x];
    }
}

@group(0) @binding(0) var<uniform> fu_gbs: FrameUniform;
@group(0) @binding(1) var<storage, read_write> block_group_totals_s: array<u32>;
/// `[0]` becomes `group_count` after this runs -- turned into an indirect
/// dispatch size for stages 6/7 by `cs_prepare_groups` right below (fully
/// GPU-resident; also read back to the CPU afterward, small and bounded,
/// alongside the range tables -- see `src/lib.rs`).
@group(0) @binding(2) var<storage, read_write> group_count_out: array<u32>;

@compute @workgroup_size(1)
fn cs_group_block_scan() {
    var running = 0u;
    for (var blk = 0u; blk < fu_gbs.num_blocks; blk++) {
        let total = block_group_totals_s[blk];
        block_group_totals_s[blk] = running; // becomes each block's base
        running += total;
    }
    group_count_out[0] = running;
}

@group(0) @binding(0) var<uniform> fu_gw: FrameUniform;
@group(0) @binding(1) var<storage, read> sorted_indices_gw: array<u32>;
@group(0) @binding(2) var<storage, read> static_objects_gw: array<StaticObjectRow>;
@group(0) @binding(3) var<storage, read> local_group_rank_gw: array<u32>;
@group(0) @binding(4) var<storage, read> block_group_base_gw: array<u32>;
@group(0) @binding(5) var<storage, read_write> group_starts: array<u32>;

@compute @workgroup_size(WG)
fn cs_group_write(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let i = gid.x;
    if i >= fu_gw.count {
        return;
    }
    var is_start = i == 0u;
    if !is_start {
        let cur = static_objects_gw[sorted_indices_gw[i]];
        let prev = static_objects_gw[sorted_indices_gw[i - 1u]];
        is_start = !same_draw_group(cur, prev);
    }
    if !is_start {
        return;
    }
    let global_group_index = block_group_base_gw[wgid.x] + local_group_rank_gw[i] - 1u;
    group_starts[global_group_index] = i;
}

@group(0) @binding(0) var<storage, read> final_count: array<u32>;
@group(0) @binding(1) var<storage, read_write> group_starts_sentinel: array<u32>;
@group(0) @binding(2) var<storage, read> group_count_sentinel: array<u32>;

@compute @workgroup_size(1)
fn cs_group_write_sentinel() {
    let gc = group_count_sentinel[0];
    group_starts_sentinel[gc] = final_count[0];
}

/// Turns the just-computed `group_count` into an indirect dispatch size for
/// every group-count-indexed stage below (`cs_build_draw_calls`, `cs_range_
/// local_scan`, `cs_range_write`) -- same technique as `cs_prepare`, just
/// keyed off `group_count` instead of the gather's live-object count.
@group(0) @binding(0) var<storage, read> group_count_pg: array<u32>;
@group(0) @binding(1) var<storage, read_write> dispatch_args_groups: array<u32>;

@compute @workgroup_size(1)
fn cs_prepare_groups() {
    dispatch_args_groups[0] = max((group_count_pg[0] + WG - 1u) / WG, 1u);
    dispatch_args_groups[1] = 1u;
    dispatch_args_groups[2] = 1u;
}

// ── Stage 6: cs_build_draw_calls ────────────────────────────────────────────

struct GpuDrawCallOut {
    index_count: u32,
    first_index: u32,
    vertex_offset: i32,
    first_instance: u32,
    instance_count: u32,
}

/// Same five fields as `GpuDrawCallOut`, reordered to `wgpu`'s hardware
/// indirect-draw ABI (`index_count, instance_count, first_index,
/// base_vertex, first_instance`) -- `multi_draw_indexed_indirect` reads
/// this exact byte layout directly off the GPU, so it needs its own buffer
/// rather than reusing `draw_calls_out`. Matches `libhelio::
/// DrawIndexedIndirectArgs` field-for-field -- also reused by stage 8's
/// shadow-partition buffers below, which need the identical hardware ABI.
struct DrawIndexedIndirectArgsOut {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

/// Mirrors `libhelio::GpuMaterial` byte-for-byte (same struct every other
/// pass reading the shared `materials` buffer mirrors, e.g.
/// `forward_lit.wgsl`'s own `GpuMaterial`) -- only `.flags` is actually used
/// here, but the full struct must still match so its byte offset is right.
struct GpuMaterial {
    base_color: vec4<f32>,
    emissive: vec4<f32>,
    roughness_metallic: vec4<f32>,
    tex_base_color: u32,
    tex_normal: u32,
    tex_roughness: u32,
    tex_emissive: u32,
    tex_occlusion: u32,
    workflow: u32,
    flags: u32,
    material_class: u32,
    class_params: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> group_count_bdc: array<u32>;
@group(0) @binding(1) var<storage, read> group_starts_bdc: array<u32>;
@group(0) @binding(2) var<storage, read> sorted_indices_bdc: array<u32>;
@group(0) @binding(3) var<storage, read> static_objects_bdc: array<StaticObjectRow>;
@group(0) @binding(4) var<storage, read_write> draw_calls_out: array<GpuDrawCallOut>;
@group(0) @binding(5) var<storage, read_write> group_material_class: array<u32>;
@group(0) @binding(6) var<storage, read_write> group_graph_hash_lo: array<u32>;
@group(0) @binding(7) var<storage, read_write> group_graph_hash_hi: array<u32>;
@group(0) @binding(8) var<storage, read_write> group_shading: array<u32>; // bit0=transparent, bit1=forward
@group(0) @binding(9) var<storage, read> materials_bdc: array<GpuMaterial>;
@group(0) @binding(10) var<storage, read_write> indirect_out: array<DrawIndexedIndirectArgsOut>;

const MATERIAL_FLAG_TRANSPARENT_ONLY: u32 = 1u << 8u; // mirrors libhelio::material::FLAG_TRANSPARENT_ONLY
const MATERIAL_FLAG_FORWARD_SHADING: u32 = 1u << 9u;  // mirrors libhelio::material::FLAG_FORWARD_SHADING

@compute @workgroup_size(WG)
fn cs_build_draw_calls(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x;
    if g >= group_count_bdc[0] {
        return;
    }
    let start = group_starts_bdc[g];
    let end = group_starts_bdc[g + 1u];
    let row = static_objects_bdc[sorted_indices_bdc[start]];

    let instance_count = end - start;
    draw_calls_out[g] = GpuDrawCallOut(
        row.index_count,
        row.first_index,
        row.vertex_offset,
        start,
        instance_count,
    );
    indirect_out[g] = DrawIndexedIndirectArgsOut(
        row.index_count,
        instance_count,
        row.first_index,
        row.vertex_offset,
        start,
    );
    group_material_class[g] = row.material_class;
    group_graph_hash_lo[g] = row.graph_hash_lo;
    group_graph_hash_hi[g] = row.graph_hash_hi;

    // Shading category is a MATERIAL property (read straight from the
    // shared `materials` buffer by this group's first instance's material
    // slot) -- fully GPU-resident, no CPU round-trip needed between this
    // stage and stage 7's opaque/transparent/forward split.
    let mat_flags = materials_bdc[row.material_slot].flags;
    var shading = 0u;
    if (mat_flags & MATERIAL_FLAG_TRANSPARENT_ONLY) != 0u {
        shading |= 1u;
    }
    if (mat_flags & MATERIAL_FLAG_FORWARD_SHADING) != 0u {
        shading |= 2u;
    }
    group_shading[g] = shading;
}

// ── Stage 7: range boundary detection (two-level scan, over GROUPS) ────────

@group(0) @binding(0) var<storage, read> group_count_rls: array<u32>;
@group(0) @binding(1) var<storage, read> group_material_class_rls: array<u32>;
@group(0) @binding(2) var<storage, read> group_graph_hash_lo_rls: array<u32>;
@group(0) @binding(3) var<storage, read> group_graph_hash_hi_rls: array<u32>;
@group(0) @binding(4) var<storage, read_write> local_range_rank: array<u32>;
@group(0) @binding(5) var<storage, read_write> block_range_totals: array<u32>;
@group(0) @binding(6) var<storage, read> group_shading_rls: array<u32>;

var<workgroup> range_scan_buf: array<u32, 256>;

@compute @workgroup_size(WG)
fn cs_range_local_scan(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let count = group_count_rls[0];
    let has_elem = gid.x < count;
    var is_start = 0u;
    if has_elem {
        if gid.x == 0u {
            is_start = 1u;
        } else {
            let i = gid.x;
            if group_material_class_rls[i] != group_material_class_rls[i - 1u]
                || group_graph_hash_lo_rls[i] != group_graph_hash_lo_rls[i - 1u]
                || group_graph_hash_hi_rls[i] != group_graph_hash_hi_rls[i - 1u]
                || group_shading_rls[i] != group_shading_rls[i - 1u] {
                is_start = 1u;
            }
        }
    }
    range_scan_buf[lid.x] = is_start;
    workgroupBarrier();

    var offset = 1u;
    loop {
        if offset >= WG {
            break;
        }
        var v = 0u;
        if lid.x >= offset {
            v = range_scan_buf[lid.x - offset];
        }
        workgroupBarrier();
        range_scan_buf[lid.x] += v;
        workgroupBarrier();
        offset = offset * 2u;
    }

    if has_elem {
        local_range_rank[gid.x] = range_scan_buf[lid.x];
    }
    if lid.x == WG - 1u {
        block_range_totals[wgid.x] = range_scan_buf[lid.x];
    }
}

struct RangeBlockUniform {
    num_blocks: u32,
}
@group(0) @binding(0) var<uniform> rbu: RangeBlockUniform;
@group(0) @binding(1) var<storage, read_write> block_range_totals_s: array<u32>;
@group(0) @binding(2) var<storage, read_write> range_count_out: array<u32>;

@compute @workgroup_size(1)
fn cs_range_block_scan() {
    var running = 0u;
    for (var blk = 0u; blk < rbu.num_blocks; blk++) {
        let total = block_range_totals_s[blk];
        block_range_totals_s[blk] = running;
        running += total;
    }
    range_count_out[0] = running;
}

struct GpuRangeOut {
    material_class: u32,
    graph_hash_lo: u32,
    graph_hash_hi: u32,
    start: u32,
    count: u32,
}

@group(0) @binding(0) var<storage, read> group_count_rw: array<u32>;
@group(0) @binding(1) var<storage, read> group_material_class_rw: array<u32>;
@group(0) @binding(2) var<storage, read> group_graph_hash_lo_rw: array<u32>;
@group(0) @binding(3) var<storage, read> group_graph_hash_hi_rw: array<u32>;
@group(0) @binding(4) var<storage, read> group_shading_rw: array<u32>;
@group(0) @binding(5) var<storage, read> local_range_rank_rw: array<u32>;
@group(0) @binding(6) var<storage, read> block_range_base_rw: array<u32>;
@group(0) @binding(7) var<storage, read_write> opaque_ranges: array<GpuRangeOut>;
@group(0) @binding(8) var<storage, read_write> transparent_ranges: array<GpuRangeOut>;
@group(0) @binding(9) var<storage, read_write> forward_ranges: array<GpuRangeOut>;
@group(0) @binding(10) var<storage, read_write> range_bucket_counts: array<atomic<u32>>; // [opaque, transparent, forward]

@compute @workgroup_size(WG)
fn cs_range_write(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let count = group_count_rw[0];
    let g = gid.x;
    if g >= count {
        return;
    }
    var is_start = false;
    if g == 0u {
        is_start = true;
    } else if group_material_class_rw[g] != group_material_class_rw[g - 1u]
        || group_graph_hash_lo_rw[g] != group_graph_hash_lo_rw[g - 1u]
        || group_graph_hash_hi_rw[g] != group_graph_hash_hi_rw[g - 1u]
        || group_shading_rw[g] != group_shading_rw[g - 1u] {
        is_start = true;
    }
    if !is_start {
        return;
    }

    // Find this range's end: the next boundary, or `count` for the last one.
    // Cheap linear walk -- ranges are, by construction, contiguous runs of
    // groups sharing one (class, graph_hash); a real scene has tens to low
    // hundreds of groups, so this never approaches a hot loop.
    var end = g + 1u;
    loop {
        if end >= count {
            break;
        }
        if group_material_class_rw[end] != group_material_class_rw[g]
            || group_graph_hash_lo_rw[end] != group_graph_hash_lo_rw[g]
            || group_graph_hash_hi_rw[end] != group_graph_hash_hi_rw[g]
            || group_shading_rw[end] != group_shading_rw[g] {
            break;
        }
        end += 1u;
    }

    let range = GpuRangeOut(
        group_material_class_rw[g],
        group_graph_hash_lo_rw[g],
        group_graph_hash_hi_rw[g],
        g,
        end - g,
    );
    let shading = group_shading_rw[g];
    let is_transparent = (shading & 1u) != 0u;
    let is_forward = (shading & 2u) != 0u;
    if is_forward {
        let slot = atomicAdd(&range_bucket_counts[2], 1u);
        forward_ranges[slot] = range;
    } else if is_transparent {
        let slot = atomicAdd(&range_bucket_counts[1], 1u);
        transparent_ranges[slot] = range;
    } else {
        let slot = atomicAdd(&range_bucket_counts[0], 1u);
        opaque_ranges[slot] = range;
    }
}

// ── Stage 8: cs_shadow_partition ─────────────────────────────────────────────

const INSTANCE_FLAG_MOVABLE: u32 = 1u << 3u; // mirrors libhelio::INSTANCE_FLAG_MOVABLE

@group(0) @binding(0) var<uniform> fu_sp: FrameUniform;
@group(0) @binding(1) var<storage, read> sorted_indices_sp: array<u32>;
@group(0) @binding(2) var<storage, read> static_objects_sp: array<StaticObjectRow>;
@group(0) @binding(3) var<storage, read_write> shadow_static_indirect: array<DrawIndexedIndirectArgsOut>;
@group(0) @binding(4) var<storage, read_write> shadow_movable_indirect: array<DrawIndexedIndirectArgsOut>;
@group(0) @binding(5) var<storage, read_write> shadow_counts: array<atomic<u32>>; // [static, movable]

@compute @workgroup_size(WG)
fn cs_shadow_partition(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= fu_sp.count {
        return;
    }
    let row = static_objects_sp[sorted_indices_sp[i]];
    let entry = DrawIndexedIndirectArgsOut(
        row.index_count,
        1u,
        row.first_index,
        row.vertex_offset,
        i,
    );
    if (row.flags & INSTANCE_FLAG_MOVABLE) != 0u {
        let slot = atomicAdd(&shadow_counts[1], 1u);
        shadow_movable_indirect[slot] = entry;
    } else {
        let slot = atomicAdd(&shadow_counts[0], 1u);
        shadow_static_indirect[slot] = entry;
    }
}
