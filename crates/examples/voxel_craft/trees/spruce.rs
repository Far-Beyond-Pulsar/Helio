//! Spruce / pine: a tall slightly twisting leader with drooping side
//! branches in whorls, dressed in stretched leaf clusters that hang below the
//! branches (so the crown reads as tiered boughs).
//!
//! A Rust port of `pinegen_worker.py` of NGNT/treegen-pinegen (MIT, see
//! assets/trees/LICENSE-treegen-pinegen): the segment-wise leader, the branch
//! whorls (`generate_branches`, `branch`) and the clustered leaf spheres.

use super::rng::Rng;
use super::shapes::{clamp, normalize, Volume, CENTRE};

#[derive(Clone, Copy, Debug)]
pub struct SpruceParams {
    pub size: f64,
    pub twisted: f64,
    /// Bare trunk below the first branches (x10 blocks).
    pub trunk_height: f64,
    pub branch_density: f64,
    pub branch_length: f64,
    /// Branch tilt: negative droops.
    pub branch_dir: f64,
    pub leaves: f64,
    pub trunk_size: f64,
    pub leaf_radius: i32,
    pub leaf_stretch: f64,
    pub leaf_bias: f64,
}

impl SpruceParams {
    pub fn spruce() -> Self {
        Self {
            size: 0.26,
            twisted: 0.5,
            trunk_height: 0.35,
            branch_density: 0.26,
            branch_length: 0.9,
            branch_dir: -0.55,
            leaves: 1.3,
            trunk_size: 0.55,
            leaf_radius: 2,
            leaf_stretch: 1.6,
            leaf_bias: -0.3,
        }
    }

    /// A taller, narrower pine.
    pub fn pine() -> Self {
        Self { size: 0.34, branch_density: 0.2, branch_length: 0.7, branch_dir: -0.3, trunk_height: 0.9, leaf_radius: 2, ..Self::spruce() }
    }
}

pub fn generate(p: &SpruceParams, seed: u64) -> Volume {
    let mut rng = Rng::new(seed);
    let mut volume = Volume::default();
    let mut tips: Vec<[f64; 3]> = Vec::new();

    let size = clamp(p.size, 0.1, 3.0);
    let twist = clamp(p.twisted, 0.0, 4.0) / size.max(1e-4).sqrt();
    let trunk_height = p.trunk_height * 10.0;
    let density = clamp(p.branch_density, 0.0, 3.0) * 30.0;
    let branch_length = clamp(p.branch_length, 0.0, 3.0) * size * 20.0;
    let branch_dir = clamp(p.branch_dir, -5.0, 5.0);
    let leaves = clamp(p.leaves, 0.0, 2.0);
    let trunk_width = size * p.trunk_size * 1.1;
    let max_iter = (100.0 * size / 5.0).floor().max(2.0) as u32;
    const SEGMENT: f64 = 5.0;
    let fmax = f64::from(max_iter);

    struct Ctx<'a> {
        volume: &'a mut Volume,
        tips: &'a mut Vec<[f64; 3]>,
    }
    struct Law {
        trunk_width: f64,
        fmax: f64,
    }
    impl Law {
        fn thickness(&self, i: u32) -> f64 {
            let t = (f64::from(i) - 1.0) / self.fmax;
            (1.0 - t * t) * self.trunk_width
        }
    }

    // A side branch: wandering, rising slightly, leaving leaf anchors.
    fn side_branch(c: &mut Ctx, rng: &mut Rng, mut pos: [f64; 3], mut dir: [f64; 3], length: f64) {
        let steps = (length / 3.0).ceil().max(1.0) as i32;
        let l = length / f64::from(steps);
        for step in 0..steps {
            let end = [pos[0] + dir[0] * l, pos[1] + dir[1] * l, pos[2] + dir[2] * l];
            let k = 1.0 / f64::from(steps);
            dir = normalize(
                [dir[0] + rng.uniform(-k, k), dir[1] + rng.uniform(-k, k) + 0.4 * k, dir[2] + rng.uniform(-k, k)],
                [0.0, 1.0, 0.0],
            );
            // Only the bough's root is wood; the rest is needles.
            if step == 0 {
                c.volume.capsule(pos, end, 0.0, 0.0, false);
            } else {
                c.volume.set_leaf([end[0] as i32, end[1] as i32, end[2] as i32]);
            }
            c.tips.push(end);
            pos = end;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn leader(
        c: &mut Ctx,
        rng: &mut Rng,
        p: &SpruceParams,
        law: &Law,
        pos: [f64; 3],
        dir: [f64; 3],
        i: u32,
        max_iter: u32,
        twist: f64,
        trunk_height: f64,
        density: f64,
        branch_length: f64,
        branch_dir: f64,
    ) {
        let end = [pos[0] + dir[0] * SEGMENT, pos[1] + dir[1] * SEGMENT, pos[2] + dir[2] * SEGMENT];
        c.volume.capsule(pos, end, law.thickness(i), law.thickness(i), false);
        let progress = f64::from(i) / f64::from(max_iter);
        if end[1] > trunk_height {
            let count = ((1.0 - progress) * density + 1.0) as i32;
            for _ in 0..count {
                let a = rng.uniform(0.0, std::f64::consts::TAU);
                let d = normalize([a.cos(), rng.uniform(0.5, 1.0) * branch_dir, a.sin()], [1.0, 0.0, 0.0]);
                let length = (1.0 - progress) * branch_length * rng.uniform(0.5, 1.5);
                let t = rng.f64();
                let origin = [pos[0] + (end[0] - pos[0]) * t, pos[1] + (end[1] - pos[1]) * t, pos[2] + (end[2] - pos[2]) * t];
                side_branch(c, rng, origin, d, length + 3.0);
            }
        }
        if i < max_iter {
            let var = twist * 0.5 * progress * (1.0 - progress);
            let d = normalize(
                [dir[0] + rng.uniform(-var, var), dir[1] + rng.uniform(-var, var), dir[2] + rng.uniform(-var, var)],
                [0.0, 1.0, 0.0],
            );
            leader(c, rng, p, law, end, d, i + 1, max_iter, twist, trunk_height, density, branch_length, branch_dir);
        } else {
            c.tips.push(end);
            c.tips.push([(pos[0] + end[0]) / 2.0, (pos[1] + end[1]) / 2.0, (pos[2] + end[2]) / 2.0]);
        }
    }

    let law = Law { trunk_width, fmax };
    let mut ctx = Ctx { volume: &mut volume, tips: &mut tips };
    leader(&mut ctx, &mut rng, p, &law, [CENTRE, 0.0, CENTRE], [0.0, 1.0, 0.0], 1, max_iter, twist, trunk_height, density, branch_length, branch_dir);

    // Leaves: ellipsoid clusters at a share of the anchors, thinned, and
    // (with a negative bias) only at and below their anchor so boughs droop.
    let radius = p.leaf_radius.clamp(1, 4);
    let stretch = clamp(p.leaf_stretch, 0.1, 5.0);
    let bias = clamp(p.leaf_bias, -1.0, 1.0);
    let clusters = ((tips.len() as f64) * clamp(leaves, 0.1, 2.0)) as usize;
    let passes = ((4.0 * leaves) as i32).max(1);
    let sources = rng.sample(&tips, clusters.min(tips.len()));
    let mut cells = Vec::new();
    for s in sources {
        let (cx, cy, cz) = (s[0] as i32, s[1] as i32, s[2] as i32);
        for _ in 0..passes {
            for dx in -radius..=radius {
                for dy in -radius..=radius {
                    for dz in -radius..=radius {
                        let d = f64::from(dx * dx + dz * dz) + (f64::from(dy) * stretch).powi(2);
                        if d > f64::from(radius * radius) || (bias < 0.0 && dy > 0) || (bias > 0.0 && dy < 0) || rng.f64() < 0.3 {
                            continue;
                        }
                        cells.push([cx + dx, cy + dy, cz + dz]);
                    }
                }
            }
        }
    }
    for c in cells {
        if c[1] >= 3 {
            volume.set_leaf(c);
        }
    }
    volume
}
