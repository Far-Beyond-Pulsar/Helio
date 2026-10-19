//! Birch: a slim straight trunk with a trunk-first growth phase, then
//! whippy, upward-leaning branches carrying small leaf clusters.
//!
//! A Rust port of `birch_worker.py` of NGNT/treegen-pinegen (MIT, see
//! assets/trees/LICENSE-treegen-pinegen): the growth laws, the trunk phase,
//! the twist and bend, and the cluster walk with its trunk avoidance. The
//! original's UI-only parameters are fixed to its defaults; the vertical axis
//! is +Y.

use super::rng::Rng;
use super::shapes::{clamp, normalize, Volume, CENTRE};

#[derive(Clone, Copy, Debug)]
pub struct BirchParams {
    pub iterations: u32,
    pub size: f64,
    pub trunk_size: f64,
    pub trunk_height: f64,
    pub branch_density: f64,
    pub branch_length: f64,
    pub branch_dir: f64,
    pub twisted: f64,
    pub spread: f64,
    pub bend: f64,
    pub leaf_radius: f64,
    pub leaf_stretch: f64,
    pub leaf_gravity: f64,
    /// Clusters started at each qualifying branch tip, and walk steps each.
    pub clusters: u32,
    pub cluster_steps: u32,
}

impl BirchParams {
    pub fn birch() -> Self {
        Self {
            iterations: 12,
            size: 0.13,
            trunk_size: 1.0,
            trunk_height: 1.15,
            branch_density: 1.1,
            branch_length: 1.2,
            branch_dir: -0.5,
            twisted: 0.6,
            spread: 0.5,
            bend: 0.25,
            leaf_radius: 1.0,
            leaf_stretch: 1.6,
            leaf_gravity: 0.0,
            clusters: 2,
            cluster_steps: 6,
        }
    }
}

pub fn generate(p: &BirchParams, seed: u64) -> Volume {
    let mut rng = Rng::new(seed);
    let mut volume = Volume::default();
    let mut tips: Vec<[f64; 3]> = Vec::new();

    let iters = p.iterations.max(10);
    let fiters = f64::from(iters);
    let size = 150.0 * p.size / fiters;
    let trunk_size = (p.trunk_size * 0.7 + 0.3) * p.size * 3.0;
    let spread = clamp(p.spread, 0.3, 0.8);
    let length0 = size * (1.0 - spread * 0.5);
    let length1 = size * (spread * 0.5 + spread * 0.1);
    let trunk_phase = (fiters * 0.35) as u32;
    let branch_density = clamp(p.branch_density, 0.0, 3.0);

    let law = |i: u32| ((f64::from(i) - 1.0) / fiters).sqrt();
    let length = |i: u32| {
        let t = law(i);
        let base = length0 * (1.0 - t * 0.5) + length1 * t;
        if i <= trunk_phase {
            base * (0.5 + clamp(p.trunk_height, 0.0, 5.0) * 0.3)
        } else {
            base * clamp(p.branch_length, 0.0, 3.0)
        }
    };
    let thickness = |i: u32| (1.0 - law(i)) * trunk_size;
    let angle = |i: u32| {
        let t = f64::from(i.saturating_sub(trunk_phase)) / f64::from((iters - trunk_phase).max(1));
        clamp(0.15 + t * 0.25, 0.15, 0.4)
    };
    let probability = |i: u32| {
        if i <= trunk_phase {
            return 0.0;
        }
        let t = f64::from(i - trunk_phase) / f64::from((iters - trunk_phase).max(1));
        clamp(clamp(0.3 + t * 0.5, 0.0, 1.0) * branch_density, 0.0, 1.0)
    };

    struct Ctx<'a> {
        volume: &'a mut Volume,
        tips: &'a mut Vec<[f64; 3]>,
    }
    #[allow(clippy::too_many_arguments)]
    fn grow(
        c: &mut Ctx,
        rng: &mut Rng,
        p: &BirchParams,
        iters: u32,
        trunk_phase: u32,
        spread: f64,
        pos: [f64; 3],
        dir: [f64; 3],
        i: u32,
        f: &Laws,
    ) {
        let l = (f.length)(i);
        let s0 = (f.thickness)(i);
        let s1 = if i < iters { (f.thickness)(i + 1) } else { 0.0 };
        let end = [pos[0] + dir[0] * l, pos[1] + dir[1] * l, pos[2] + dir[2] * l];
        c.volume.capsule(pos, end, s0, s1, false);
        if i < iters {
            if i <= trunk_phase {
                grow(c, rng, p, iters, trunk_phase, spread, end, dir, i + 1, f);
                return;
            }
            let mut variance = f64::from(i) * 0.12 * (spread * 0.9);
            variance *= 1.0 + p.bend * 0.6;
            let mut forks = 1 + i32::from(rng.f64() < 0.4);
            if rng.f64() < (f.probability)(i) * 0.6 {
                forks = forks.max(2);
                variance = variance.max((f.angle)(i));
            }
            for _ in 0..forks {
                // The whole tree twists about the vertical as it grows.
                let turn = p.twisted * 0.15 * (f64::from(i) / f64::from(iters.max(1)));
                let (sin, cos) = turn.sin_cos();
                let (rx, rz) = (dir[0] * cos - dir[2] * sin, dir[0] * sin + dir[2] * cos);
                let d = normalize(
                    [
                        rx + rng.uniform(-variance, variance) + p.bend * 0.25 * rng.uniform(-1.0, 1.0) + p.branch_dir * 0.05,
                        dir[1] + rng.uniform(-variance, variance) + p.bend * 0.25 * rng.uniform(-1.0, 1.0),
                        rz + rng.uniform(-variance, variance * 0.6) - p.bend * 0.05 * rng.f64(),
                    ],
                    [0.0, 1.0, 0.0],
                );
                grow(c, rng, p, iters, trunk_phase, spread, end, d, i + 1, f);
            }
        } else {
            c.tips.push(end);
            c.tips.push([(pos[0] + end[0]) / 2.0, (pos[1] + end[1]) / 2.0, (pos[2] + end[2]) / 2.0]);
        }
    }
    struct Laws<'a> {
        length: &'a dyn Fn(u32) -> f64,
        thickness: &'a dyn Fn(u32) -> f64,
        angle: &'a dyn Fn(u32) -> f64,
        probability: &'a dyn Fn(u32) -> f64,
    }
    let laws = Laws { length: &length, thickness: &thickness, angle: &angle, probability: &probability };
    let mut ctx = Ctx { volume: &mut volume, tips: &mut tips };
    grow(&mut ctx, &mut rng, p, iters, trunk_phase, spread, [CENTRE, 0.0, CENTRE], [0.0, 1.0, 0.0], 1, &laws);

    // The trunk's height and base radius, from what grew.
    let trunk: Vec<[i32; 3]> = volume.wood().collect();
    let trunk_height = trunk.iter().map(|c| c[1]).max().unwrap_or(8);
    let trunk_radius = trunk
        .iter()
        .filter(|c| c[1] <= 2.min(trunk_height))
        .map(|c| (f64::from(c[0]) - CENTRE).hypot(f64::from(c[2]) - CENTRE))
        .fold(1.0f64, f64::max)
        .ceil();

    // Leaf clusters: random walks around tips, away from the trunk's column
    // and above the trunk's lowest quarter.
    let min_tip = f64::from(trunk_height) * 0.25;
    let radius = p.leaf_radius.round().max(1.0) as i32;
    let stretch = ((f64::from(radius)) * p.leaf_stretch) as i32;
    let gravity_up = 0.5 + p.leaf_gravity * 0.5;
    let mut leaves = Vec::new();
    for tip in &tips {
        let (x, y, z) = (tip[0].round() as i32, tip[1].round() as i32, tip[2].round() as i32);
        if f64::from(y) < min_tip {
            continue;
        }
        if (f64::from(x) - CENTRE).powi(2) + (f64::from(z) - CENTRE).powi(2) <= (trunk_radius + 1.0).powi(2) {
            continue;
        }
        for _ in 0..p.clusters.max(1) {
            let mut c = [x + rng.int(-radius, radius), y + rng.int(-stretch, stretch), z + rng.int(-radius, radius)];
            for _ in 0..p.cluster_steps.max(2) {
                if (f64::from(c[0]) - CENTRE).powi(2) + (f64::from(c[2]) - CENTRE).powi(2) > (trunk_radius + 0.5).powi(2) {
                    leaves.push(c);
                    // A small cluster, not a point: the walk's cell and its sides.
                    for d in [[1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1], [0, -1, 0]] {
                        if rng.f64() < 0.55 {
                            leaves.push([c[0] + d[0], c[1] + d[1], c[2] + d[2]]);
                        }
                    }
                }
                c[1] += if rng.f64() < gravity_up { -1 } else { 1 };
                c[0] += *rng.choice(&[-radius, 0, radius]);
                c[2] += *rng.choice(&[-radius, 0, radius]);
                c[1] += *rng.choice(&[-stretch, 0, stretch]);
            }
        }
    }
    for c in leaves {
        if c[1] >= 3 {
            volume.set_leaf(c);
        }
    }
    volume
}
