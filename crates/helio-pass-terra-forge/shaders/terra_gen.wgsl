// Terra Forge — Per-chunk GPU brick generation via SDF evaluation.
//
// Each workgroup processes one brick (8^3 voxels) within a single chunk.
// Dispatch: chunk_dim_bricks^3 workgroups (e.g. 32^3 = 32,768 per chunk).
//
// Writes to global brick_pool and voxel_pool at chunk-specific offsets.
// Conservative SDF classification skips empty/solid bricks.

struct GenUniforms {
    chunk_dim_bricks:   u32,
    brick_dim:          u32,
    voxel_size:         f32,
    planet_radius:      f32,
    chunk_world_origin: vec3<f32>,
    max_mixed_bricks:   u32,
    brick_pool_offset:  u32,
    voxel_pool_offset:  u32,
    edit_count:         u32,
    _pad1:              u32,
}

struct BrickMeta {
    data_offset: u32,
    occupancy:   u32,
}

@group(0) @binding(0) var<uniform> gen: GenUniforms;
@group(0) @binding(1) var<storage, read_write> brick_pool: array<BrickMeta>;
@group(0) @binding(2) var<storage, read_write> voxel_pool: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> alloc_counter: array<atomic<u32>>;

struct EditOp {
    shape_type: u32,   // 0=sphere, 1=box
    op_type:    u32,   // 0=add, 1=subtract
    material:   u32,
    blend_k:    f32,
    position:   vec3<f32>,
    _pad0:      f32,
    size:       vec3<f32>,
    _pad1:      f32,
}

@group(0) @binding(4) var<storage, read> edit_buffer: array<EditOp>;

const WORDS_PER_BRICK: u32 = 128u;
const BRICK_EMPTY: u32 = 0xFFFFFFFFu;
const BRICK_SOLID: u32 = 0xFFFFFFFEu;

// ── SDF + noise ──────────────────────────────────────────────────────────────

fn sdf_sphere(pos: vec3<f32>) -> f32 {
    return length(pos) - gen.planet_radius;
}

fn hash3(p: vec3<f32>) -> f32 {
    var q = fract(p * vec3<f32>(0.1031, 0.1030, 0.0973));
    q += dot(q, q.yxz + 33.33);
    return fract((q.x + q.y) * q.z);
}

fn noise3(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(hash3(i + vec3<f32>(0.0, 0.0, 0.0)),
                       hash3(i + vec3<f32>(1.0, 0.0, 0.0)), u.x),
                   mix(hash3(i + vec3<f32>(0.0, 1.0, 0.0)),
                       hash3(i + vec3<f32>(1.0, 1.0, 0.0)), u.x), u.y),
               mix(mix(hash3(i + vec3<f32>(0.0, 0.0, 1.0)),
                       hash3(i + vec3<f32>(1.0, 0.0, 1.0)), u.x),
                   mix(hash3(i + vec3<f32>(0.0, 1.0, 1.0)),
                       hash3(i + vec3<f32>(1.0, 1.0, 1.0)), u.x), u.y), u.z);
}

fn fbm(p: vec3<f32>, octaves: i32) -> f32 {
    var val = 0.0;
    var amp = 0.5;
    var freq_p = p;
    for (var i = 0; i < octaves; i++) {
        val += amp * (noise3(freq_p) - 0.5);
        freq_p *= 2.0;
        amp *= 0.5;
    }
    return val;
}

fn sdf_terrain(pos: vec3<f32>) -> f32 {
    let base = sdf_sphere(pos);
    let dir = normalize(pos + vec3<f32>(0.001));
    let terrain_noise = fbm(dir * 8.0, 4) * gen.planet_radius * 0.05;
    return base - terrain_noise;
}

fn material_at(pos: vec3<f32>, d: f32) -> u32 {
    if d > 0.0 { return 0u; }
    let depth = -d;
    let dir = normalize(pos + vec3<f32>(0.001));
    let slope_noise = fbm(dir * 16.0, 2);
    if depth < gen.voxel_size * 2.0 && slope_noise > -0.1 { return 2u; }
    if depth < gen.voxel_size * 8.0 { return 3u; }
    return 1u;
}

// ── SDF edit operations (Quilez smooth CSG) ──────────────────────────────────

fn sdf_edit_shape(pos: vec3<f32>, edit: EditOp) -> f32 {
    let p = pos - edit.position;
    if edit.shape_type == 0u {
        return length(p) - edit.size.x;
    }
    let q = abs(p) - edit.size;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn smooth_union(d1: f32, d2: f32, k: f32) -> f32 {
    let h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) - k * h * (1.0 - h);
}

fn smooth_subtraction(d1: f32, d2: f32, k: f32) -> f32 {
    let h = clamp(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0);
    return mix(d1, -d2, h) + k * h * (1.0 - h);
}

fn apply_edits(pos: vec3<f32>, base_d: f32) -> f32 {
    var d = base_d;
    for (var i = 0u; i < gen.edit_count; i++) {
        let edit = edit_buffer[i];
        let ed = sdf_edit_shape(pos, edit);
        if edit.op_type == 0u {
            d = smooth_union(d, ed, edit.blend_k);
        } else {
            d = smooth_subtraction(ed, d, edit.blend_k);
        }
    }
    return d;
}

fn material_at_edited(pos: vec3<f32>, base_d: f32, edited_d: f32) -> u32 {
    // Check if any edit created this voxel.
    for (var i = 0u; i < gen.edit_count; i++) {
        let edit = edit_buffer[i];
        let ed = sdf_edit_shape(pos, edit);
        if edit.op_type == 0u && ed < base_d && edited_d <= 0.0 {
            return edit.material;
        }
    }
    return material_at(pos, edited_d);
}

// ── Workgroup shared state ───────────────────────────────────────────────────

var<workgroup> wg_occupancy: atomic<u32>;
var<workgroup> wg_brick_slot: u32;
var<workgroup> wg_skip: u32;

@compute @workgroup_size(8, 8, 8)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {

    let cdim = gen.chunk_dim_bricks;
    let bd = gen.brick_dim;
    let vs = gen.voxel_size;

    let bx = wid.x;
    let by = wid.y;
    let bz = wid.z;

    if bx >= cdim || by >= cdim || bz >= cdim { return; }

    let local_brick_idx = bx + by * cdim + bz * cdim * cdim;
    let global_brick_idx = gen.brick_pool_offset + local_brick_idx;
    let local_idx = lid.x + lid.y * bd + lid.z * bd * bd;

    // Initialize workgroup state.
    if local_idx == 0u {
        atomicStore(&wg_occupancy, 0u);
        wg_skip = 0u;
        wg_brick_slot = 0u;
    }
    workgroupBarrier();

    // ── Conservative brick classification ────────────────────────────
    if local_idx == 0u {
        let brick_world_size = f32(bd) * vs;
        let brick_min = gen.chunk_world_origin + vec3<f32>(f32(bx), f32(by), f32(bz)) * brick_world_size;
        let brick_center = brick_min + vec3<f32>(brick_world_size * 0.5);
        let half_diag = 0.866 * brick_world_size;
        let d_base = sdf_terrain(brick_center);
        let d = apply_edits(brick_center, d_base);

        if d > half_diag {
            wg_skip = 1u;
        }
    }
    workgroupBarrier();

    if wg_skip == 1u {
        if local_idx == 0u {
            brick_pool[global_brick_idx] = BrickMeta(BRICK_EMPTY, 0u);
        }
        return;
    }

    // ── Per-voxel SDF evaluation ─────────────────────────────────────
    let brick_world_size = f32(bd) * vs;
    let brick_min = gen.chunk_world_origin + vec3<f32>(f32(bx), f32(by), f32(bz)) * brick_world_size;
    let voxel_center = brick_min + (vec3<f32>(f32(lid.x), f32(lid.y), f32(lid.z)) + 0.5) * vs;

    let base_d = sdf_terrain(voxel_center);
    let d = apply_edits(voxel_center, base_d);
    let mat_val = material_at_edited(voxel_center, base_d, d);

    if mat_val != 0u {
        atomicAdd(&wg_occupancy, 1u);
    }
    workgroupBarrier();

    // ── Allocate brick slot ──────────────────────────────────────────
    let occ = atomicLoad(&wg_occupancy);
    let bd3 = bd * bd * bd;

    if local_idx == 0u {
        if occ == 0u {
            brick_pool[global_brick_idx] = BrickMeta(BRICK_EMPTY, 0u);
            wg_skip = 2u;
        } else if occ == bd3 {
            brick_pool[global_brick_idx] = BrickMeta(BRICK_SOLID, bd3);
            wg_skip = 2u;
        } else {
            let local_slot = atomicAdd(&alloc_counter[0], 1u);
            if local_slot >= gen.max_mixed_bricks {
                brick_pool[global_brick_idx] = BrickMeta(BRICK_EMPTY, 0u);
                wg_skip = 2u;
            } else {
                let global_slot = gen.voxel_pool_offset + local_slot;
                wg_brick_slot = global_slot;
                brick_pool[global_brick_idx] = BrickMeta(global_slot, occ);
            }
        }
    }
    workgroupBarrier();

    if wg_skip == 2u { return; }

    // ── Write voxel data ─────────────────────────────────────────────
    let slot = wg_brick_slot;
    let word_base = slot * WORDS_PER_BRICK;
    let word_offset = local_idx >> 2u;
    let byte_shift = (local_idx & 3u) * 8u;
    let packed = mat_val << byte_shift;

    if packed != 0u {
        atomicOr(&voxel_pool[word_base + word_offset], packed);
    }
}
