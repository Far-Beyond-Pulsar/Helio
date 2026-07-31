// SDF Classify — GPU-native brick classification shader.
//
// Replaces all CPU-side brick iteration, BVH traversal, edit-list building,
// dirty tracking, and scroll management.  Zero CPU loops.
//
// Dispatch: dispatch_workgroups(bricks_per_level / 64, level_count, 1)
//   = (64, 8, 1) for the default 128-grid / 8-brick / 8-level configuration.
//
// Each invocation handles one (brick, level) pair:
//   1. Early-out if the level is not dirty this frame.
//   2. Invert toroidal addressing to recover the brick's world coordinates.
//   3. Walk the flat GPU BVH to collect overlapping edit indices.
//   4. Check terrain intersection.
//   5. Hash the resulting edit list and compare with the stored per-brick hash.
//   6. If changed: write the new edit list, atomically append to the per-level
//      dirty list, and increment eval_indirect[level * 3 + 0].
//   7. Update all_brick_indices (direct atlas slot = brick_flat for active bricks).

// ── Shared struct definitions (must match sdf_scroll.wgsl and lib.rs exactly) ──

struct ClipConfig {
    level_count:          u32,
    grid_dim:             u32,
    brick_size:           u32,
    brick_grid_dim:       u32,   // = grid_dim / brick_size
    bricks_per_level:     u32,   // = brick_grid_dim^3
    atlas_bricks_per_axis: u32,
    base_voxel_size:      f32,
    edit_count:           u32,
    bvh_node_count:       u32,
    terrain_enabled:      u32,
    terrain_y_min:        f32,
    terrain_y_max:        f32,
    _pad0:                u32,
    _pad1:                u32,
    _pad2:                u32,
    _pad3:                u32,
    voxel_sizes_lo:       vec4<f32>,   // levels 0-3
    voxel_sizes_hi:       vec4<f32>,   // levels 4-7
};

struct ScrollState {
    snap_origins:  array<vec4<i32>, 8>,
    edit_gen:      u32,
    prev_edit_gen: u32,
    _pad0:         u32,
    _pad1:         u32,
};

struct GpuBvhNode {
    aabb_min:          vec3<f32>,
    left_or_edit_idx:  u32,   // leaf → edit index;  internal → left child
    aabb_max:          vec3<f32>,
    right_or_leaf:     u32,   // leaf → LEAF_SENTINEL;  internal → right child
};

// ── Constants ────────────────────────────────────────────────────────────────

const LEAF_SENTINEL:      u32 = 0xFFFFFFFFu;
const EMPTY_BRICK:        u32 = 0xFFFFFFFFu;
const MAX_EDITS_PER_BRICK: u32 = 64u;
const EDIT_LIST_STRIDE:   u32 = 65u;   // 1 (count) + 64 (indices)
const BVH_STACK_DEPTH:    u32 = 32u;
// FNV-1a constants
const FNV_OFFSET: u32 = 2166136261u;
const FNV_PRIME:  u32 = 16777619u;

// Per-level dirty-list stride: bricks_per_level entries (no count prefix; count
// comes from eval_indirect[level*3+0] which is also the indirect x dispatch size).
//
// Layout of dirty_bricks (global, all levels concatenated):
//   dirty_bricks[level * MAX_BRICKS_PER_LEVEL + slot] = packed_world_brick_coord
// where slot = the value returned by atomicAdd on eval_indirect[level*3+0].
const MAX_BRICKS_PER_LEVEL: u32 = 4096u; // must be >= bricks_per_level

// Pack signed world brick coordinates into a single u32 (10 bits each, bias 512).
fn pack_world_brick_coord(wx: i32, wy: i32, wz: i32) -> u32 {
    let bias = 512i;
    return (u32(wx + bias) & 0x3FFu)
         | ((u32(wy + bias) & 0x3FFu) << 10u)
         | ((u32(wz + bias) & 0x3FFu) << 20u);
}

// ── Bindings ─────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<uniform>            clip_config:          ClipConfig;
@group(0) @binding(1) var<storage, read>      scroll_state:         ScrollState;
@group(0) @binding(2) var<storage, read>      dirty_flags:          array<u32>;
@group(0) @binding(3) var<storage, read>      bvh_nodes:            array<GpuBvhNode>;
@group(0) @binding(4) var<storage, read_write> per_brick_hashes:    array<u32>;
@group(0) @binding(5) var<storage, read_write> per_brick_edit_lists: array<u32>;
@group(0) @binding(6) var<storage, read_write> all_brick_indices:   array<u32>;
@group(0) @binding(7) var<storage, read_write> dirty_bricks:        array<u32>;
@group(0) @binding(8) var<storage, read_write> eval_indirect:       array<atomic<u32>>;

// ── Helpers ──────────────────────────────────────────────────────────────────

fn level_voxel_size(level: u32) -> f32 {
    let lo = clip_config.voxel_sizes_lo;
    let hi = clip_config.voxel_sizes_hi;
    if level == 0u { return lo.x; }
    else if level == 1u { return lo.y; }
    else if level == 2u { return lo.z; }
    else if level == 3u { return lo.w; }
    else if level == 4u { return hi.x; }
    else if level == 5u { return hi.y; }
    else if level == 6u { return hi.z; }
    else { return hi.w; }
}

// ── Main kernel ──────────────────────────────────────────────────────────────
//
// Dispatch dimensions:
//   X: ceil(bricks_per_level / 64)  —  64 workgroups × 64 threads = 4096 bricks
//   Y: level_count                  —  one layer per clip level
//   Z: 1

@compute @workgroup_size(64, 1, 1)
fn cs_classify(
    @builtin(global_invocation_id) gid:  vec3<u32>,
    @builtin(workgroup_id)         wgid: vec3<u32>,
) {
    let level_idx  = gid.y;                        // 0..level_count-1
    let brick_flat = gid.x;                        // 0..bricks_per_level-1

    if level_idx  >= clip_config.level_count   { return; }
    if brick_flat >= clip_config.bricks_per_level { return; }

    // ── Early-out: level is clean this frame ─────────────────────────────
    if dirty_flags[level_idx] == 0u { return; }

    let bgd = clip_config.brick_grid_dim;

    // ── Flat index → 3D grid coordinates ────────────────────────────────
    let gz = brick_flat / (bgd * bgd);
    let gy = (brick_flat / bgd) % bgd;
    let gx = brick_flat % bgd;

    // ── Grid coordinates → world brick coordinates (toroidal inverse) ────
    //
    // The toroidal mapping is: grid_flat(wx) = ((wx % bgd) + bgd) % bgd
    // Inverse: given gx, the world x stored at grid position gx is the
    // unique wx in [wb_min_x, wb_min_x + bgd) s.t. (wx % bgd + bgd) % bgd == gx.
    //
    //   wb_min = snap_origin - bgd/2   (center-aligned volume)
    //   wx = wb_min_x + ((gx - ((wb_min_x % bgd) + bgd) % bgd + bgd) % bgd)
    //
    let snap = scroll_state.snap_origins[level_idx];
    let half_bgd = i32(bgd / 2u);
    let wb_min_x = snap.x - half_bgd;
    let wb_min_y = snap.y - half_bgd;
    let wb_min_z = snap.z - half_bgd;

    let mod_x = ((wb_min_x % i32(bgd)) + i32(bgd)) % i32(bgd);
    let mod_y = ((wb_min_y % i32(bgd)) + i32(bgd)) % i32(bgd);
    let mod_z = ((wb_min_z % i32(bgd)) + i32(bgd)) % i32(bgd);

    let wx = wb_min_x + ((i32(gx) - mod_x + i32(bgd)) % i32(bgd));
    let wy = wb_min_y + ((i32(gy) - mod_y + i32(bgd)) % i32(bgd));
    let wz = wb_min_z + ((i32(gz) - mod_z + i32(bgd)) % i32(bgd));

    // ── Brick world-space AABB ────────────────────────────────────────────
    let vs               = level_voxel_size(level_idx);
    let brick_world_size = vs * f32(clip_config.brick_size);
    let brick_min        = vec3<f32>(f32(wx), f32(wy), f32(wz)) * brick_world_size;
    let brick_max        = brick_min + vec3<f32>(brick_world_size);

    // ── GPU BVH traversal ─────────────────────────────────────────────────
    // Iterative stack-based AABB tree walk.  No CPU loops; all on GPU.
    var edit_indices: array<u32, 64>;
    var edit_count_local: u32 = 0u;

    if clip_config.bvh_node_count > 0u {
        var stack: array<u32, 32>;
        var sp:    u32 = 0u;
        stack[0] = 0u;   // root always at index 0
        sp = 1u;

        loop {
            if sp == 0u { break; }
            sp -= 1u;
            let ni   = stack[sp];
            let node = bvh_nodes[ni];

            // Slab AABB test.
            if any(brick_max < node.aabb_min) || any(brick_min > node.aabb_max) {
                continue;
            }

            if node.right_or_leaf == LEAF_SENTINEL {
                // Leaf — record the edit index.
                if edit_count_local < MAX_EDITS_PER_BRICK {
                    edit_indices[edit_count_local] = node.left_or_edit_idx;
                    edit_count_local += 1u;
                }
            } else {
                // Internal — push both children.
                if sp < BVH_STACK_DEPTH - 2u {
                    stack[sp]     = node.left_or_edit_idx;
                    stack[sp + 1u] = node.right_or_leaf;
                    sp += 2u;
                }
            }
        }
    }

    // ── Terrain intersection test ─────────────────────────────────────────
    let terrain_active = (clip_config.terrain_enabled != 0u)
        && (brick_max.y >= clip_config.terrain_y_min)
        && (brick_min.y <= clip_config.terrain_y_max);

    let is_active = (edit_count_local > 0u) || terrain_active;

    // ── Update all_brick_indices (read by ray-march shader) ───────────────
    let all_offset = level_idx * clip_config.bricks_per_level + brick_flat;
    all_brick_indices[all_offset] = select(EMPTY_BRICK, brick_flat, is_active);

    // ── FNV-1a hash of the edit list + terrain flag ───────────────────────
    var hash: u32 = FNV_OFFSET;
    for (var e = 0u; e < edit_count_local; e++) {
        hash = (hash ^ edit_indices[e]) * FNV_PRIME;
    }
    if terrain_active { hash = (hash ^ 0xDEADBEEFu) * FNV_PRIME; }
    if !is_active     { hash = 0u; }   // canonical sentinel for empty bricks

    // ── Change detection via stored hash ─────────────────────────────────
    let hash_offset  = level_idx * clip_config.bricks_per_level + brick_flat;
    let stored_hash  = per_brick_hashes[hash_offset];

    if hash == stored_hash { return; }   // nothing changed — skip this brick

    // ── Write updated hash ────────────────────────────────────────────────
    per_brick_hashes[hash_offset] = hash;

    if !is_active { return; }   // brick became empty — atlas slot will remain stale
                                // but the ray-march check (EMPTY_BRICK) guards it

    // ── Write per-brick edit list (flat global buffer, per-level stride) ──
    //   Layout: [level * bricks_per_level * EDIT_LIST_STRIDE
    //            + flat * EDIT_LIST_STRIDE]
    //               [+0]   = edit_count_local
    //               [+1..] = edit_indices[]
    let el_base = (level_idx * clip_config.bricks_per_level + brick_flat) * EDIT_LIST_STRIDE;
    per_brick_edit_lists[el_base] = edit_count_local;
    for (var e = 0u; e < edit_count_local; e++) {
        per_brick_edit_lists[el_base + 1u + e] = edit_indices[e];
    }

    // ── Append to per-level dirty brick list (atomically) ─────────────────
    //   eval_indirect[level*3+0] doubles as the dirty brick counter AND
    //   the x-dimension of the indirect evaluate dispatch.
    let dirty_slot = atomicAdd(&eval_indirect[level_idx * 3u], 1u);

    // Guard against overflow (should never happen with bricks_per_level slots).
    if dirty_slot < MAX_BRICKS_PER_LEVEL {
        dirty_bricks[level_idx * MAX_BRICKS_PER_LEVEL + dirty_slot] =
            pack_world_brick_coord(wx, wy, wz);
    }
}
