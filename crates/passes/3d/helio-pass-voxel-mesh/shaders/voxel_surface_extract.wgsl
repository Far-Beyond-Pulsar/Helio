// GPU surface extraction for a single brick using Marching Cubes.
// One workgroup (64 threads) per brick.
// Indexed by workgroup_id.x into the dirty_bricks list.

// Must match MAX_SURFACE_VERTS_PER_BRICK/MAX_SURFACE_INDICES_PER_BRICK in
// helio_voxel_core::constants (vertex_buf/index_buf are sized from those).
const MAX_VERTS: u32 = 2048u;
const MAX_INDICES: u32 = 2048u;
// Each brick's voxel data is padded to 9x9x9 (one extra voxel of +X/+Y/+Z
// halo copied from the neighboring brick), so cells run 0..7 (8 per axis,
// corners 0..8) instead of 0..6 — the extra cell at each brick's +face reads
// the neighbor's first voxel via padding, which is what actually closes the
// seam between adjacent bricks. See VOXEL_MESH_BRICK_VOXEL_WORDS.
const CELLS_PER_DIM: u32 = 8u;
const TOTAL_CELLS: u32 = 512u; // 8×8×8
const PADDED_DIM: u32 = 9u;
const WG_SIZE: u32 = 64u;
const CELLS_PER_THREAD: u32 = (TOTAL_CELLS + WG_SIZE - 1u) / WG_SIZE;

struct GpuBrickMeta {
    data_offset: u32,
    occupancy: u32,
}

struct GpuBrickMeshlet {
    vertex_offset: u32,
    index_offset: u32,
    vertex_count: u32,
    index_count: u32,
    brick_index: u32,
    volume_id: u32,
    _pad: array<u32, 2>,
}

struct DrawIndexedIndirect {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

struct DirtyBrick {
    brick_slot: u32,
    volume_id: u32,
    origin_size: vec4<f32>, // xyz = world origin, w = voxel_size
}

@group(0) @binding(0) var<storage, read> brick_meta: array<GpuBrickMeta>;
@group(0) @binding(1) var<storage, read> voxel_data: array<u32>;
@group(0) @binding(2) var<storage, read_write> vertex_buf: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> index_buf: array<u32>;
@group(0) @binding(4) var<storage, read_write> descriptors: array<GpuBrickMeshlet>;
@group(0) @binding(5) var<storage, read_write> indirect_draws: array<DrawIndexedIndirect>;
@group(0) @binding(6) var<storage, read> dirty_bricks: array<DirtyBrick>;
@group(0) @binding(7) var<storage, read_write> normal_buf: array<vec4<f32>>;

var<workgroup> wg_vertex_count: atomic<u32>;
var<workgroup> wg_index_count: atomic<u32>;

// FXC lowers dynamically indexed shader constants into indexable temporary
// registers. The canonical 256 x 16 triangle table alone exceeds its 4096
// register limit, so Helio uploads the same packed table as read-only storage.
// Each case occupies two u32 words with one 4-bit edge index per nibble; 0xF
// is the end-of-row sentinel.
@group(0) @binding(8) var<storage, read> packed_tri_table: array<u32>;

fn triangle_edge(words: vec2<u32>, slot: u32) -> u32 {
    let word = select(words.x, words.y, slot >= 8u);
    return (word >> ((slot % 8u) * 4u)) & 0xFu;
}

fn edge_vertex(edge: u32) -> vec3<f32> {
    let edge_mid = array<vec3<f32>, 12>(
        vec3<f32>(0.5, 0.0, 0.0),
        vec3<f32>(1.0, 0.5, 0.0),
        vec3<f32>(0.5, 1.0, 0.0),
        vec3<f32>(0.0, 0.5, 0.0),
        vec3<f32>(0.5, 0.0, 1.0),
        vec3<f32>(1.0, 0.5, 1.0),
        vec3<f32>(0.5, 1.0, 1.0),
        vec3<f32>(0.0, 0.5, 1.0),
        vec3<f32>(0.0, 0.0, 0.5),
        vec3<f32>(1.0, 0.0, 0.5),
        vec3<f32>(1.0, 1.0, 0.5),
        vec3<f32>(0.0, 1.0, 0.5),
    );
    return edge_mid[edge];
}

// Brick data is a padded 9x9x9 block (indices 0..8 per axis) — see
// VOXEL_MESH_BRICK_VOXEL_WORDS / CELLS_PER_DIM.
fn read_voxel(data_offset: u32, x: u32, y: u32, z: u32) -> u32 {
    let linear = z * (PADDED_DIM * PADDED_DIM) + y * PADDED_DIM + x;
    let word_idx = data_offset + linear / 4u;
    let byte_in_word = linear % 4u;
    return (voxel_data[word_idx] >> (byte_in_word * 8u)) & 0xFFu;
}

fn read_voxel_f32(data_offset: u32, x: u32, y: u32, z: u32) -> f32 {
    return select(-1.0, 1.0, read_voxel(data_offset, x, y, z) > 0u);
}

// Clamps a central-difference sample coordinate to the padded brick's valid
// [0,8] range — without this, sampling at the extremes underflows/overflows
// the u32 coordinate and reads garbage (wrapped-around or out-of-brick) data.
fn clamped_voxel(v: i32) -> u32 {
    return u32(clamp(v, 0, i32(PADDED_DIM) - 1));
}

fn compute_normal(data_offset: u32, cx: u32, cy: u32, cz: u32) -> vec3<f32> {
    let icx = i32(cx);
    let icy = i32(cy);
    let icz = i32(cz);
    let sx = read_voxel_f32(data_offset, clamped_voxel(icx + 1), cy, cz) - read_voxel_f32(data_offset, clamped_voxel(icx - 1), cy, cz);
    let sy = read_voxel_f32(data_offset, cx, clamped_voxel(icy + 1), cz) - read_voxel_f32(data_offset, cx, clamped_voxel(icy - 1), cz);
    let sz = read_voxel_f32(data_offset, cx, cy, clamped_voxel(icz + 1)) - read_voxel_f32(data_offset, cx, cy, clamped_voxel(icz - 1));
    let n = vec3<f32>(sx, sy, sz);
    let magnitude_squared = dot(n, n);
    let inverse_length = inverseSqrt(max(magnitude_squared, 0.000001));
    return select(vec3<f32>(0.0, 1.0, 0.0), n * inverse_length, magnitude_squared >= 0.000001);
}

@compute @workgroup_size(WG_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let brick = dirty_bricks[wg_id.x];
    let brick_slot = brick.brick_slot;
    let volume_id = brick.volume_id;
    let origin = brick.origin_size.xyz;
    let vs = brick.origin_size.w;

    if lid.x == 0u {
        atomicStore(&wg_vertex_count, 0u);
        atomicStore(&wg_index_count, 0u);
    }
    workgroupBarrier();

    let brick_meta_entry = brick_meta[brick_slot];
    let data_offset = brick_meta_entry.data_offset;

    let thread_first = lid.x * CELLS_PER_THREAD;
    let thread_last = min(thread_first + CELLS_PER_THREAD, TOTAL_CELLS);

    for (var cell_linear = thread_first; cell_linear < thread_last; cell_linear++) {
        let cz = cell_linear / (CELLS_PER_DIM * CELLS_PER_DIM);
        let cy = (cell_linear / CELLS_PER_DIM) % CELLS_PER_DIM;
        let cx = cell_linear % CELLS_PER_DIM;

        var corner: array<u32, 8>;
        corner[0] = read_voxel(data_offset, cx,     cy,     cz);
        corner[1] = read_voxel(data_offset, cx + 1, cy,     cz);
        corner[2] = read_voxel(data_offset, cx + 1, cy + 1, cz);
        corner[3] = read_voxel(data_offset, cx,     cy + 1, cz);
        corner[4] = read_voxel(data_offset, cx,     cy,     cz + 1);
        corner[5] = read_voxel(data_offset, cx + 1, cy,     cz + 1);
        corner[6] = read_voxel(data_offset, cx + 1, cy + 1, cz + 1);
        corner[7] = read_voxel(data_offset, cx,     cy + 1, cz + 1);

        var cube_index: u32 = 0u;
        for (var i = 0u; i < 8u; i++) {
            if corner[i] != 0u {
                cube_index |= 1u << i;
            }
        }

        if cube_index == 0u || cube_index == 0xFFu {
            continue;
        }

        let table_offset = cube_index * 2u;
        let triangle_words = vec2<u32>(
            packed_tri_table[table_offset],
            packed_tri_table[table_offset + 1u],
        );
        var tri_count: u32 = 0u;
        for (var t = 0u; t < 15u; t++) {
            if triangle_edge(triangle_words, t) == 0xFu {
                break;
            }
            tri_count++;
        }

        let num_indices = tri_count;
        if num_indices == 0u {
            continue;
        }

        let vert_base = atomicAdd(&wg_vertex_count, num_indices);
        let index_base = atomicAdd(&wg_index_count, num_indices);

        if vert_base + num_indices > MAX_VERTS || index_base + num_indices > MAX_INDICES {
            continue;
        }

        let brick_vert_offset = brick_slot * MAX_VERTS;
        let brick_idx_offset = brick_slot * MAX_INDICES;
        let cell_world = vec3<f32>(f32(cx), f32(cy), f32(cz)) * vs + origin;

        var material: u32 = 0u;
        for (var i = 0u; i < 8u; i++) {
            if corner[i] != 0u {
                material = corner[i];
                break;
            }
        }

        for (var i = 0u; i < tri_count; i++) {
            let edge_idx = triangle_edge(triangle_words, i);
            if edge_idx >= 12u {
                continue;
            }
            let local_pos = edge_vertex(edge_idx);
            let world_pos = cell_world + local_pos * vs;
            let vi = vert_base + i;
            vertex_buf[brick_vert_offset + vi] = vec4<f32>(world_pos, f32(material));
            // Compute normal via central differences on the material field.
            // local_pos components are in {0.0, 0.5, 1.0} (edge_vertex midpoints);
            // round to the nearest corner voxel (0 or 1) — NOT scaled by the
            // brick size, which would read voxels far outside this brick.
            let nx = u32(round(local_pos.x));
            let ny = u32(round(local_pos.y));
            let nz = u32(round(local_pos.z));
            let n = compute_normal(data_offset, cx + nx, cy + ny, cz + nz);
            normal_buf[brick_vert_offset + vi] = vec4<f32>(n, 0.0);
        }

        for (var i = 0u; i < tri_count; i++) {
            let ii = index_base + i;
            index_buf[brick_idx_offset + ii] = vert_base + i;
        }
    }

    workgroupBarrier();

    if lid.x == 0u {
        let vc = min(atomicLoad(&wg_vertex_count), MAX_VERTS);
        let ic = min(atomicLoad(&wg_index_count), MAX_INDICES);

        let brick_vert_offset = brick_slot * MAX_VERTS;
        let brick_idx_offset = brick_slot * MAX_INDICES;

        descriptors[brick_slot] = GpuBrickMeshlet(
            brick_vert_offset,
            brick_idx_offset,
            vc,
            ic,
            brick_slot,
            volume_id,
            array<u32, 2>(0u, 0u),
        );

        let has_geom = select(0u, 1u, vc > 0u && ic > 0u);
        indirect_draws[brick_slot] = DrawIndexedIndirect(
            ic,
            has_geom,
            brick_idx_offset,
            i32(brick_vert_offset),
            0u,
        );
    }
}
