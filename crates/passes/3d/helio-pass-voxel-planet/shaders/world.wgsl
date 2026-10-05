// World constants and helpers shared by the engine and terrain programs:
// the grid mapping, layer quantization, coarse-level bounds and material ids.
const HEIGHT_ONE: i32 = 1000; // height units (mm) per metre

struct World {
    grid: vec4<i32>,  // reference cells, layer thickness (mm), grid cells, level offset
    scale: vec4<u32>, // domain scale (Q24), volume inv, volume shift, layer (Q16 of 0.1 m)
    bounds: array<vec4<i32>, 6>, // per-level finer-surface excess (level cells)
}

// Engine material ids (terrain::material).
const M_AIR: u32 = 0u;
const M_GRASS: u32 = 1u;
const M_DIRT: u32 = 2u;
const M_STONE: u32 = 3u;
const M_SAND: u32 = 4u;
const M_SNOW: u32 = 5u;
const M_WATER: u32 = 6u;
const M_GRAVEL: u32 = 7u;
const M_SANDSTONE: u32 = 8u;
const M_DARK_STONE: u32 = 9u;
const M_WOOD: u32 = 10u;
const M_LEAVES: u32 = 11u;
const M_CLAY: u32 = 12u;
const M_BRICK: u32 = 13u;
const M_PLANKS: u32 = 14u;
const M_COBBLE: u32 = 15u;
// Ground material flag: a single-voxel fleck that blends into grass once
// its cell is about a pixel wide.
const M_SPECK: u32 = 0x100u;
const M_ID: u32 = 0xffu;

// Face bases, matching grid::FACE_BASIS.
fn face_axis(face: u32, axis: u32) -> vec3<i32> {
    switch face * 3u + axis {
        case 0u: { return vec3<i32>(1, 0, 0); }
        case 1u: { return vec3<i32>(0, 0, -1); }
        case 2u: { return vec3<i32>(0, 1, 0); }
        case 3u: { return vec3<i32>(-1, 0, 0); }
        case 4u: { return vec3<i32>(0, 0, 1); }
        case 5u: { return vec3<i32>(0, 1, 0); }
        case 6u: { return vec3<i32>(0, 1, 0); }
        case 7u: { return vec3<i32>(1, 0, 0); }
        case 8u: { return vec3<i32>(0, 0, -1); }
        case 9u: { return vec3<i32>(0, -1, 0); }
        case 10u: { return vec3<i32>(1, 0, 0); }
        case 11u: { return vec3<i32>(0, 0, 1); }
        case 12u: { return vec3<i32>(0, 0, 1); }
        case 13u: { return vec3<i32>(1, 0, 0); }
        case 14u: { return vec3<i32>(0, 1, 0); }
        case 15u: { return vec3<i32>(0, 0, -1); }
        case 16u: { return vec3<i32>(-1, 0, 0); }
        default: { return vec3<i32>(0, 1, 0); }
    }
}

fn mul_q24(a: u32, r: u32) -> u32 {
    let a1 = a >> 16u;
    let a0 = a & 0xffffu;
    let r1 = r >> 16u;
    let r0 = r & 0xffffu;
    return ((a1 * r1) << 8u) + ((a1 * r0 + a0 * r1) >> 8u) + ((a0 * r0) >> 24u);
}

// Level cell centre in half cells of the reference grid (grid independent).
// Plane domain point in half reference cells relative to the world origin
// (`grid::plane_domain_point`): scaled by magnitude, symmetric about it.
fn plane_scaled(x: i32, level: u32) -> i32 {
    let origin = world.grid.z / 2;
    let v = (x << (level + 1u)) + (1 << level) - (origin << 1u);
    let m = i32(mul_q24(u32(abs(v)), world.scale.x));
    return select(m, -m, v < 0);
}

fn domain_point(face: u32, i: i32, j: i32, level: u32) -> vec3<i32> {
    if is_plane() {
        return vec3<i32>(plane_scaled(i, level), 0, -plane_scaled(j, level));
    }
    let reference = world.grid.x;
    let scale = world.scale.x;
    let half = 1u << level;
    let u = i32(mul_q24((bitcast<u32>(i) << (level + 1u)) + half, scale)) - reference;
    let v = i32(mul_q24((bitcast<u32>(j) << (level + 1u)) + half, scale)) - reference;
    return face_axis(face, 0u) * reference + face_axis(face, 1u) * u + face_axis(face, 2u) * v;
}

// `(a * b) >> s` of the exact 64-bit product, from 16-bit limbs
// (`grid::mul_shr`). The caller keeps the result within 32 bits.
fn mul_shr(a: u32, b: u32, s: u32) -> u32 {
    let a1 = a >> 16u;
    let a0 = a & 0xffffu;
    let b1 = b >> 16u;
    let b0 = b & 0xffffu;
    let m1 = a1 * b0;
    let m2 = a0 * b1;
    let mid = m1 + m2;
    let mid_carry = select(0u, 0x10000u, mid < m1);
    let lo0 = a0 * b0;
    let lo = lo0 + (mid << 16u);
    let lo_carry = select(0u, 1u, lo < lo0);
    let hi = a1 * b1 + (mid >> 16u) + mid_carry + lo_carry;
    if s == 0u { return lo; }
    if s >= 32u { return hi >> (s - 32u); }
    return (lo >> s) | (hi << (32u - s));
}

fn volume_component(p: i32, ratio: i32) -> i32 {
    let m = i32(mul_shr(u32(abs(p)), u32(abs(ratio)), 30u));
    return select(m, -m, (p < 0) != (ratio < 0));
}

// Seamless 3D domain point of a level cell centre (`grid::volume_point`):
// the column's domain point scaled by (R + h) / R on a sphere, the height
// as the vertical axis on a plane. Volumetric terrain samples 3D noise here.
fn volume_point(face: u32, i: i32, j: i32, k: i32, level: u32) -> vec3<i32> {
    let p = domain_point(face, i, j, level);
    let h = (k << (level + 1u)) + (1 << level);
    if is_plane() {
        let v = i32(mul_shr(u32(abs(h)), world.scale.w, 16u));
        return vec3<i32>(p.x, select(v, -v, h < 0), p.z);
    }
    let r = i32(mul_shr(u32(abs(h)), world.scale.y, world.scale.z));
    let ratio = select(r, -r, h < 0);
    return p + vec3<i32>(volume_component(p.x, ratio), volume_component(p.y, ratio), volume_component(p.z, ratio));
}

// Surface height of a level column: the terrain program at the column's
// footprint in reference cells.
fn field_height(face: u32, i: i32, j: i32, level: u32) -> i32 {
    return terrain_height(domain_point(face, i, j, level), level + u32(world.grid.w));
}

fn top_cells(height: i32, level: u32) -> i32 {
    return div_floor(height, world.grid.y) >> level;
}

fn terrain_kind(top: i32, k: i32) -> u32 {
    return select(0u, 1u, k < top);
}

// Ground slope in the 8x8 column block, eighths of a cell per cell
// (`terrain::block_slope`).
fn block_slope_of(t_x0: i32, t_x7: i32, t_y0: i32, t_y7: i32) -> i32 {
    return max(abs(t_x7 - t_x0), abs(t_y7 - t_y0)) * 8 / 7;
}

fn bound_margin(level: u32) -> i32 {
    return world.bounds[level >> 2u][level & 3u];
}
