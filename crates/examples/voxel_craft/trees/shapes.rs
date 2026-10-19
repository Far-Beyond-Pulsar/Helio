//! Voxel shapes the tree generators share: a sparse volume of wood and
//! leaves, round capsules and vector helpers. Coordinates are in blocks with
//! +Y up; generators work around [`CENTRE`] so that truncating to integers
//! behaves as on the original positive grids.

use std::collections::HashMap;

/// Where generators centre a tree in their working grid.
pub const CENTRE: f64 = 128.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    Wood,
    Leaf,
}

/// A sparse volume of tree parts.
#[derive(Default, Clone, Debug)]
pub struct Volume {
    pub cells: HashMap<[i32; 3], Part>,
}

impl Volume {
    pub fn get(&self, p: [i32; 3]) -> Option<Part> {
        self.cells.get(&p).copied()
    }

    pub fn set_wood(&mut self, p: [i32; 3]) {
        self.cells.insert(p, Part::Wood);
    }

    /// A leaf where there is nothing (wood is never replaced).
    pub fn set_leaf(&mut self, p: [i32; 3]) {
        self.cells.entry(p).or_insert(Part::Leaf);
    }

    pub fn wood(&self) -> impl Iterator<Item = [i32; 3]> + '_ {
        self.cells.iter().filter(|(_, p)| **p == Part::Wood).map(|(c, _)| *c)
    }

    /// Fill a round capsule from `a` to `b`, radius `r0` to `r1`, with wood
    /// (cells are the trunk of the position, as the originals' truncation
    /// does: `round` makes it the nearest cell instead).
    pub fn capsule(&mut self, a: [f64; 3], b: [f64; 3], r0: f64, r1: f64, round: bool) {
        let length = dist(a, b);
        let steps = ((length * 2.0) as i32).max(1);
        for i in 0..=steps {
            let t = f64::from(i) / f64::from(steps);
            let c = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])];
            let r = r0 + t * (r1 - r0);
            let reach = r.ceil() as i32;
            for dx in -reach..=reach {
                for dy in -reach..=reach {
                    for dz in -reach..=reach {
                        if f64::from(dx * dx + dy * dy + dz * dz) <= r * r {
                            let cell = if round {
                                [(c[0] + f64::from(dx) + 0.5).floor() as i32, (c[1] + f64::from(dy) + 0.5).floor() as i32, (c[2] + f64::from(dz) + 0.5).floor() as i32]
                            } else {
                                [(c[0] + f64::from(dx)) as i32, (c[1] + f64::from(dy)) as i32, (c[2] + f64::from(dz)) as i32]
                            };
                            self.set_wood(cell);
                        }
                    }
                }
            }
        }
    }
}

pub fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// `v` as a unit vector (`fallback` where it has no length).
pub fn normalize(v: [f64; 3], fallback: [f64; 3]) -> [f64; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if l > 1e-9 {
        [v[0] / l, v[1] / l, v[2] / l]
    } else {
        fallback
    }
}

pub fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    v.max(lo).min(hi)
}

/// Quintic smoothstep.
pub fn smooth5(x: f64) -> f64 {
    if x <= 0.0 {
        0.0
    } else if x >= 1.0 {
        1.0
    } else {
        x * x * x * (x * (x * 6.0 - 15.0) + 10.0)
    }
}
