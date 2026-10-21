//! CPU mirror of the procedural tuft geometry in `foliage_gbuffer.wgsl`.
//!
//! There is no vertex buffer and no index buffer anywhere in the grass path. Every vertex
//! is derived from `@builtin(vertex_index)` and a 16-byte record
//! ([`GpuBladeInstance`](helio_pass_foliage_place::GpuBladeInstance)), which is what makes
//! the world's foliage four `draw_indirect` calls. The cost of that is that the geometry
//! only exists inside a shader, where nothing can look at it — so its shapes are written
//! twice, here and in WGSL, and the tests in `tests/geometry.rs` check this copy's
//! properties: blocky blades are connected staircases that never share a cube, strips fit
//! their LOD's vertex count, smooth blades keep their length.
//!
//! A tuft's blades are a function of its seed, the blade's index and the type
//! ([`tuft_blade`]): they fan out from a tight base, lean out more at the rim, bend more
//! towards the tip, and some lie over.

use helio_pass_foliage_place::{hash_to_unit, GpuFoliageType, FOLIAGE_LOD_VERTEX_COUNTS};

/// Number of drawable foliage LOD levels.
pub const LOD_COUNT: usize = 4;

/// Segments up a smooth blade at each LOD: L0 and L1 strips, L2 cards (1), L3 a fan (1).
pub const LOD_SEGMENTS: [u32; LOD_COUNT] = [5, 3, 1, 1];

/// Vertices per instance at each LOD for smooth blades (the producer's
/// [`FOLIAGE_LOD_VERTEX_COUNTS`]; blocky blades draw `FOLIAGE_BLOCK_LOD_VERTEX_COUNTS`).
pub const LOD_VERTEX_COUNTS: [u32; LOD_COUNT] = FOLIAGE_LOD_VERTEX_COUNTS;

/// Cubes a blocky blade rises by at most (`FOLIAGE_STACK_CUBES`).
pub const STACK_CUBES: u32 = helio_pass_foliage_place::FOLIAGE_STACK_CUBES;
/// Cubes a blocky blade steps sideways by at most at L0 and at L1 (`FOLIAGE_L1_STEPS`).
pub const STACK_STEPS: u32 = 4;
pub const L1_STEPS: u32 = 2;
/// Boxes a blocky blade's strip holds at L0 and L1.
pub const L0_BOXES: u32 = 8;
pub const L1_BOXES: u32 = 5;
/// A smooth tuft's base: its blades root within this many metres of its centre.
pub const TUFT_BASE: f32 = 0.03;

/// `foliage_hash` (PCG output permutation).
pub fn hash(x: u32) -> u32 {
    helio_pass_foliage_place::foliage_hash(x)
}

/// One blade of a tuft (`Blade` in the shader).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Blade {
    pub rim: f32,
    pub angle: f32,
    pub compass: u32,
    pub lean: f32,
    pub curve: f32,
    pub fallen: bool,
    pub height: f32,
    pub width: f32,
}

/// Blade `j` of `n` of the tuft of `seed` (the record's seed half), `turn` (its aux byte
/// over 256) and `height_lerp` (its height byte over 255), of type `ty` (`tuft_blade`).
pub fn tuft_blade(ty: &GpuFoliageType, seed: u32, turn: f32, height_lerp: f32, j: u32, n: u32) -> Blade {
    let h = hash(seed.wrapping_mul(0x9e37_79b1).wrapping_add(j.wrapping_mul(0x85eb_ca6b)).wrapping_add(0x27d4_eb2f));
    let h2 = hash(h ^ 0x1656_67b1);
    let unit = hash_to_unit;
    let rim = if j == 0 { 0.0 } else { 0.35 + 0.65 * unit(h) };
    let angle = std::f32::consts::TAU * (turn + (j as f32 + 0.35 * unit(h2)) / n.max(1) as f32);
    let compass = ((turn * 8.0) as u32 + j * 3) % 8;
    let fallen = unit(hash(h ^ 0x9e37_79b9)) < ty.fall_chance && j != 0;
    let mut lean = (ty.lean_range[0] + (ty.lean_range[1] - ty.lean_range[0]) * rim) * (0.75 + 0.5 * unit(h2 >> 3));
    let curve = ty.curve * (0.6 + 0.8 * unit(h >> 5));
    let tuft_height = ty.height_range[0] + (ty.height_range[1] - ty.height_range[0]) * height_lerp;
    let mut height = tuft_height * (1.0 - 0.35 * rim) * (0.75 + 0.5 * unit(h >> 11));
    if fallen {
        lean = lean.max(1.0) * 2.5;
        height *= 0.6;
    }
    let width = ty.width_range[0] + (ty.width_range[1] - ty.width_range[0]) * unit(h2 >> 7);
    Blade { rim, angle, compass, lean, curve, fallen, height, width }
}

/// A grid compass step (`compass_step`): the eight neighbours of a cube, axis steps first.
pub fn compass_step(i: u32) -> [f32; 2] {
    [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]][(i & 7) as usize]
}

/// How many cubes a blocky blade rises and steps sideways, at most `steps` (`block_steps`).
pub fn block_steps(b: &Blade, cube: f32, scale: f32, steps: u32) -> [u32; 2] {
    let rise = ((b.height * scale / cube).round() as u32).min(STACK_CUBES);
    if rise == 0 {
        return [0, 0];
    }
    let side = ((rise as f32 * (b.lean + b.curve).max(0.0)).round() as u32).min(steps);
    [rise, side]
}

/// The sideways step the path has reached after rising to cube `v` (`block_target`).
pub fn block_target(b: &Blade, v: u32, rise: u32, side: u32) -> u32 {
    if rise <= 1 || side == 0 {
        return if rise <= 1 { side } else { 0 };
    }
    let t = v as f32 / (rise - 1) as f32;
    let g = (b.lean * t + b.curve * t * t) / (b.lean + b.curve).max(1.0e-4);
    ((side as f32 * g.clamp(0.0, 1.0)).round() as u32).min(side)
}

/// A box of a blocky shape: centre across the ground, half extents, bottom and top.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct BlockBox {
    pub centre: [f32; 2],
    pub half: [f32; 2],
    pub y0: f32,
    pub y1: f32,
}

/// The cubes of a blocky blade's staircase, in order, as `(sideways step, rise)`; blades
/// past the tuft's first leave out the shared root column (`block_path`'s walk).
pub fn block_cubes(b: &Blade, j: u32, cube: f32, scale: f32) -> Vec<[u32; 2]> {
    let [rise, side] = block_steps(b, cube, scale, STACK_STEPS);
    if rise == 0 {
        return Vec::new();
    }
    let side = if j > 0 { side.max(1) } else { side };
    let (mut h, mut v) = (0u32, 0u32);
    let mut cubes = vec![[0, 0]];
    loop {
        let way = if h < block_target(b, v, rise, side) {
            1
        } else if v + 1 < rise {
            0
        } else if h < side {
            1
        } else {
            break;
        };
        if way == 0 {
            v += 1;
        } else {
            h += 1;
        }
        cubes.push([h, v]);
    }
    if j > 0 {
        // The shared column (step 0) belongs to the tuft's first blade.
        cubes.retain(|c| c[0] > 0);
    }
    cubes
}

/// The boxes of a blocky blade stepping sideways at most `max_steps` (`block_path`, all of
/// them): runs up merged, axis runs sideways merged, diagonal steps a cube each.
pub fn block_boxes(b: &Blade, j: u32, cube: f32, scale: f32, leafy: bool, max_steps: u32) -> Vec<BlockBox> {
    let merge = STACK_CUBES;
    let [rise, side_steps] = block_steps(b, cube, scale, max_steps);
    let mut out = Vec::new();
    if rise == 0 {
        return out;
    }
    let side = if j > 0 { side_steps.max(1) } else { side_steps };
    let dir = compass_step(b.compass);
    let diagonal = b.compass >= 4;
    let size = cube * if j >= 8 { 0.86 } else { 1.0 };
    let height = cube * if leafy { 0.6 } else { 1.0 };
    let across = if leafy { b.width.max(size) } else { size };
    let (mut h, mut v) = (0u32, 0u32);
    let (mut run_h, mut run_v, mut run_len, mut run_way) = (0u32, 0u32, 1u32, 0u32);
    let mut started = j == 0;
    loop {
        let way = if h < block_target(b, v, rise, side) {
            1
        } else if v + 1 < rise {
            0
        } else if h < side {
            1
        } else {
            2
        };
        let last = way == 2;
        let breaks = last || way != run_way || (way == 0 && run_len >= merge) || (way == 1 && diagonal) || !started;
        if breaks && started {
            let first = [dir[0] * run_h as f32 * size, dir[1] * run_h as f32 * size];
            let extent = if run_way == 1 && !diagonal {
                [dir[0] * (run_len - 1) as f32 * size, dir[1] * (run_len - 1) as f32 * size]
            } else {
                [0.0, 0.0]
            };
            let centre = [first[0] + extent[0] * 0.5, first[1] + extent[1] * 0.5];
            let along = [dir[0].abs() * 0.5 * size + extent[0].abs() * 0.5, dir[1].abs() * 0.5 * size + extent[1].abs() * 0.5];
            let other = [dir[1].abs() * 0.5 * across, dir[0].abs() * 0.5 * across];
            let half = if diagonal { [0.5 * size; 2] } else { [along[0].max(other[0]), along[1].max(other[1])] };
            let up_len = if run_way == 0 { run_len } else { 1 };
            out.push(BlockBox { centre, half, y0: run_v as f32 * height, y1: (run_v + up_len) as f32 * height });
        }
        if last {
            break;
        }
        if way == 0 {
            v += 1;
        } else {
            h += 1;
        }
        if breaks {
            run_h = h;
            run_v = v;
            run_len = 1;
            run_way = way;
            started = started || way == 1;
        } else {
            run_len += 1;
        }
    }
    out
}

/// Strip position `vertex_index` over boxes of 11 vertices (`strip_slot`): the box and its
/// vertex, or `None` past the strip's end.
pub fn strip_slot(vertex_index: u32, boxes: u32) -> Option<(u32, u32)> {
    if boxes == 0 || vertex_index > 11 * (boxes - 1) + 9 {
        return None;
    }
    let (b, k) = (vertex_index / 11, vertex_index % 11);
    Some(match k {
        9 => (b, 8),
        10 => (b + 1, 0),
        _ => (b, k),
    })
}

/// A point of a smooth blade's spine at `s` in its own frame (+z its lean direction, +y
/// up), and its unit tangent (`blade_spine`).
pub fn blade_spine(b: &Blade, height: f32, s: f32) -> ([f32; 3], [f32; 3]) {
    let slope = b.lean + 2.0 * b.curve * s;
    let reach = height * (b.lean * s + b.curve * s * s);
    let lean_here = b.lean + b.curve * s;
    let rise = height * s / (1.0 + lean_here * lean_here).sqrt();
    let len = (1.0 + slope * slope).sqrt();
    ([0.0, rise, reach], [0.0, 1.0 / len, slope / len])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_fit_their_draws() {
        assert_eq!(strip_slot(0, 1), Some((0, 0)));
        assert_eq!(strip_slot(9, 1), Some((0, 8)));
        assert_eq!(strip_slot(10, 1), None);
        assert_eq!(strip_slot(10, 2), Some((1, 0)));
        assert_eq!(strip_slot(11 * L0_BOXES - 3, L0_BOXES), Some((L0_BOXES - 1, 8)));
        assert_eq!(11 * L0_BOXES - 2, helio_pass_foliage_place::FOLIAGE_BLOCK_LOD_VERTEX_COUNTS[0]);
        assert_eq!(11 * L1_BOXES - 2, helio_pass_foliage_place::FOLIAGE_BLOCK_LOD_VERTEX_COUNTS[1]);
    }
}
