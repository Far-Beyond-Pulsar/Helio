//! Broadleaf trees (oak, dark oak): recursive, tapering, forking branches
//! whose tips grow leaf clouds as random walks.
//!
//! A Rust port of `treegen_worker.py` of NGNT/treegen-pinegen (MIT, see
//! assets/trees/LICENSE-treegen-pinegen). The branch recursion, its length,
//! thickness, angle and fork probability laws and the leaf walks are the
//! original's; the grid is sparse, the vertical axis is +Y and the leaf walk
//! counts are parameters (the original scales them with a tree of 100+
//! voxels).

use super::rng::Rng;
use super::shapes::{clamp, normalize, Volume, CENTRE};

#[derive(Clone, Copy, Debug)]
pub struct OakParams {
    /// Branch generations (the original's `iterations`).
    pub iterations: u32,
    /// Overall size: 150 voxels * `size` is the tree's reach.
    pub size: f64,
    /// Trunk thickness relative to `size` (x6 for its radius in voxels).
    pub trunk_size: f64,
    /// 0..0.95: how much longer outer branches are than inner ones.
    pub wide: f64,
    /// How far forks part.
    pub spread: f64,
    /// Random bend of branches that do not fork.
    pub twisted: f64,
    /// Leaf clouds started at each branch tip, and the steps of each walk.
    pub leaf_walks: u32,
    pub leaf_steps: u32,
    /// -1 (leaves sink) .. 1 (leaves rise): bias of the vertical leaf steps.
    pub gravity: f64,
}

impl OakParams {
    /// A round, full oak of about a dozen blocks.
    pub fn oak() -> Self {
        Self { iterations: 5, size: 0.16, trunk_size: 1.45, wide: 0.55, spread: 0.55, twisted: 0.5, leaf_walks: 3, leaf_steps: 36, gravity: 0.1 }
    }

    /// A thick, low, broad crown.
    pub fn dark_oak() -> Self {
        Self { iterations: 5, size: 0.17, trunk_size: 2.1, wide: 0.7, spread: 0.7, twisted: 0.35, leaf_walks: 3, leaf_steps: 30, gravity: -0.05 }
    }
}

/// Wood and leaves of an oak grown from `seed`, base at `(CENTRE, 0, CENTRE)`.
pub fn generate(p: &OakParams, seed: u64) -> Volume {
    let mut rng = Rng::new(seed);
    let mut volume = Volume::default();
    let mut leaf_tips: Vec<[f64; 3]> = Vec::new();

    let iters = p.iterations.max(1) as f64;
    let size = 150.0 * p.size / iters;
    let trunk = p.trunk_size * p.size * 6.0;
    let wide = p.wide.min(0.95);
    let length0 = size * (1.0 - wide);
    let length1 = size * wide;
    let law = |i: u32| ((f64::from(i) - 1.0) / iters).sqrt();

    struct Branch<'a> {
        p: &'a OakParams,
        iters: u32,
        lengths: (f64, f64),
        trunk: f64,
        volume: &'a mut Volume,
        tips: &'a mut Vec<[f64; 3]>,
    }

    fn grow(b: &mut Branch, rng: &mut Rng, pos: [f64; 3], dir: [f64; 3], i: u32, law: &dyn Fn(u32) -> f64) {
        let length = b.lengths.0 + law(i) * (b.lengths.1 - b.lengths.0);
        let size = |i: u32| (1.0 - law(i)) * b.trunk;
        let s0 = size(i);
        let s1 = if i < b.iters { size(i + 1) } else { 0.0 };
        let end = [pos[0] + dir[0] * length, pos[1] + dir[1] * length, pos[2] + dir[2] * length];
        b.volume.capsule(pos, end, s0, s1, false);
        if i < b.iters {
            let mut forks = 1;
            let mut variance = f64::from(i) * 0.2 * b.p.twisted;
            if rng.f64() < law(i) {
                forks = 2;
                variance = 2.0 * b.p.spread * law(i);
            }
            for _ in 0..forks {
                let d = normalize(
                    [dir[0] + rng.uniform(-variance, variance), dir[1] + rng.uniform(-variance, variance), dir[2] + rng.uniform(-variance, variance)],
                    [0.0, 1.0, 0.0],
                );
                grow(b, rng, end, d, i + 1, law);
            }
        } else {
            b.tips.push(end);
            b.tips.push([(pos[0] + end[0]) / 2.0, (pos[1] + end[1]) / 2.0, (pos[2] + end[2]) / 2.0]);
        }
    }

    let mut branch = Branch { p, iters: p.iterations.max(1), lengths: (length0, length1), trunk, volume: &mut volume, tips: &mut leaf_tips };
    grow(&mut branch, &mut rng, [CENTRE, 0.0, CENTRE], [0.0, 1.0, 0.0], 1, &law);

    // Leaves: random walks from every tip, written where nothing is.
    let mut leaves: Vec<[i32; 3]> = Vec::new();
    {
        let mut seen = std::collections::HashSet::new();
        for tip in &leaf_tips {
            let start = [tip[0] as i32, tip[1] as i32, tip[2] as i32];
            for _ in 0..p.leaf_walks {
                let mut c = start;
                for _ in 0..p.leaf_steps {
                    if seen.insert(c) {
                        leaves.push(c);
                    }
                    match rng.int(1, 6) {
                        1 => c[0] -= 1,
                        2 => c[0] += 1,
                        3 | 4 => c[1] += if rng.uniform(-1.0, 1.0) < p.gravity { 1 } else { -1 },
                        5 => c[2] -= 1,
                        _ => c[2] += 1,
                    }
                }
            }
        }
    }
    for c in leaves {
        // Never under the ground the trunk stands on.
        if c[1] >= 2 {
            volume.set_leaf(c);
        }
    }
    let _ = clamp;
    volume
}
