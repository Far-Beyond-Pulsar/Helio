//! Procedural trees, bushes, cacti and boulders of small voxels, as the
//! voxel planet's stamps of shapes: trunks and branches are tapered
//! capsules, crowns sparse leaf balls with broken edges (light and sky
//! show through), conifers stacked cones. Sizes are in metres; a stamp's
//! offsets are cells (`VOXEL_M`), its anchor the first layer over the
//! ground with the trunk on the anchor column's centre.

mod rng;

pub use rng::Rng;

use crate::blocks::Block;
use crate::config::VOXEL_M;
use glam::DVec3;
use helio_pass_voxel_planet::stamp::{Part, Shape, Stamp};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Species {
    Oak,
    DarkOak,
    Birch,
    Spruce,
    Pine,
    Jungle,
}

impl Species {
    fn blocks(self) -> (Block, Block) {
        match self {
            Species::Oak | Species::DarkOak => (Block::OakLog, Block::OakLeaves),
            Species::Birch => (Block::BirchLog, Block::BirchLeaves),
            Species::Spruce | Species::Pine => (Block::SpruceLog, Block::SpruceLeaves),
            Species::Jungle => (Block::JungleLog, Block::JungleLeaves),
        }
    }
}

/// Shapes in metres from the anchor column's centre at ground level, into
/// a stamp in cells.
struct Builder {
    stamp: Stamp,
}

impl Builder {
    fn new() -> Self {
        Self { stamp: Stamp::new() }
    }

    /// A point (m) as the stamp's cells: the anchor column's centre is
    /// `(0.5, 0, 0.5)` cells, the ground's top at `y = 0`.
    fn at(p: DVec3) -> [f32; 3] {
        [(p.x / VOXEL_M + 0.5) as f32, (p.y / VOXEL_M) as f32, (p.z / VOXEL_M + 0.5) as f32]
    }

    fn cells(m: f64) -> f32 {
        (m / VOXEL_M) as f32
    }

    /// A limb from `a` to `b` (m), its radius tapering from `ra` to `rb`.
    fn limb(&mut self, a: DVec3, b: DVec3, ra: f64, rb: f64, block: Block) {
        self.stamp.capsule(Self::at(a), Self::at(b), [Self::cells(ra), Self::cells(rb)], block.material());
    }

    /// Foliage: a cone or ball (a limb of leaves) a share of whose cells
    /// is left out, its edge broken by `rough` metres and its outline into
    /// lumps up to `lump` metres deep, 40 cm across.
    fn foliage(&mut self, a: DVec3, b: DVec3, ra: f64, rb: f64, block: Block, sparse: f32, rough: f64, lump: f64) {
        let shape = Shape::Capsule { a: Self::at(a), b: Self::at(b), radius: [Self::cells(ra), Self::cells(rb)] };
        self.stamp.add(Part { sparse, rough: Self::cells(rough), lump: Self::cells(lump), lump_size: Self::cells(0.4).round().max(1.0) as u32, ..Part::solid(shape, block.material()) });
    }

    /// A flat-ended cone of foliage: a conifer's tier.
    fn tier(&mut self, a: DVec3, b: DVec3, ra: f64, rb: f64, block: Block, sparse: f32, rough: f64) {
        let shape = Shape::Cone { a: Self::at(a), b: Self::at(b), radius: [Self::cells(ra), Self::cells(rb)] };
        self.stamp.add(Part { sparse, rough: Self::cells(rough), ..Part::solid(shape, block.material()) });
    }

    /// A leaf mass: a lumpy ball of leaves.
    fn crown(&mut self, c: DVec3, r: f64, block: Block, sparse: f32) {
        self.foliage(c, c, r, r, block, sparse, r * 0.2, r * 0.4);
    }
}

/// A unit vector `azimuth` round the vertical and `elevation` above the
/// horizontal (radians).
fn direction(azimuth: f64, elevation: f64) -> DVec3 {
    DVec3::new(azimuth.cos() * elevation.cos(), elevation.sin(), azimuth.sin() * elevation.cos())
}

/// A trunk `height` metres to its fork, of `radius` at the ground, gently
/// curved, with roots flaring into the ground: the fork's point.
fn trunk(b: &mut Builder, rng: &mut Rng, height: f64, radius: f64, wood: Block) -> DVec3 {
    let lean = direction(rng.uniform(0.0, std::f64::consts::TAU), 0.0);
    let mid = DVec3::Y * height * 0.5 + lean * rng.uniform(0.0, 0.25);
    let fork = DVec3::Y * height + lean * rng.uniform(0.1, 0.5);
    b.limb(DVec3::Y * -0.5, mid, radius, radius * 0.85, wood);
    b.limb(mid, fork, radius * 0.85, radius * 0.72, wood);
    for n in 0..rng.int(3, 5) {
        let out = direction(f64::from(n) * 1.7 + rng.uniform(0.0, 0.6), 0.0);
        b.limb(DVec3::Y * radius * 1.4, out * radius * rng.uniform(2.2, 3.2) - DVec3::Y * 0.25, radius * 0.55, radius * 0.2, wood);
    }
    fork
}

/// How a species branches: its wood and leaves, its leaf masses' share
/// left out and radius range (m), how far a child limb turns from its
/// parent (radians) and how much it is drawn upward.
struct Limbs {
    wood: Block,
    leaves: Block,
    sparse: f32,
    cluster: [f64; 2],
    spread: f64,
    rise: f64,
}

impl Limbs {
    /// A limb from `from` along `dir` of `length` and base `radius`, then
    /// two or three children to `depth` more levels, each shorter and
    /// thinner; a leaf mass at every twig's end.
    fn grow(&self, b: &mut Builder, rng: &mut Rng, from: DVec3, dir: DVec3, length: f64, radius: f64, depth: u32) {
        let end = from + dir * length;
        b.limb(from, end, radius, radius * 0.7, self.wood);
        if depth == 0 {
            b.crown(end, rng.uniform(self.cluster[0], self.cluster[1]), self.leaves, self.sparse);
            return;
        }
        let side = dir.any_orthonormal_vector();
        for n in 0..rng.int(2, 3) {
            // Turned about the limb, then pulled outward and up.
            let around = glam::DQuat::from_axis_angle(dir, f64::from(n) * 2.4 + rng.uniform(0.0, 1.0));
            let tilt = glam::DQuat::from_axis_angle(around * side, self.spread * rng.uniform(0.6, 1.2));
            let outward = DVec3::new(end.x, 0.0, end.z).normalize_or_zero();
            let child = (tilt * dir + outward * 0.25 + DVec3::Y * self.rise * 0.5).normalize();
            let next = length * rng.uniform(0.7, 0.85);
            self.grow(b, rng, end, child, next, radius * 0.62, depth - 1);
        }
    }

    /// `count` leaf masses filling the crown round `centre`: a flattened
    /// ellipsoid `[radius, half height]` (m), closing the gaps between the
    /// twigs' masses.
    fn fill(&self, b: &mut Builder, rng: &mut Rng, centre: DVec3, extent: [f64; 2], count: u32) {
        for _ in 0..count {
            let at = centre + direction(rng.uniform(0.0, std::f64::consts::TAU), 0.0) * extent[0] * rng.uniform(0.0, 0.8) + DVec3::Y * extent[1] * rng.uniform(-0.6, 0.8);
            b.crown(at, rng.uniform(self.cluster[0], self.cluster[1]) * 1.25, self.leaves, self.sparse);
        }
    }
}

/// The tree of `species` grown from `seed`, as a stamp.
pub fn tree(species: Species, seed: u64) -> Stamp {
    // Seeds of nearby variants differ in a few bits: spread them.
    let mut rng = Rng::new(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(23) ^ 0x5851_f42d_4c95_7f2d);
    let (wood, leaves) = species.blocks();
    let mut b = Builder::new();
    match species {
        Species::Oak | Species::DarkOak => {
            let dark = species == Species::DarkOak;
            let height = if dark { rng.uniform(3.0, 4.0) } else { rng.uniform(3.2, 4.6) };
            let radius = if dark { rng.uniform(0.45, 0.6) } else { rng.uniform(0.38, 0.5) };
            let fork = trunk(&mut b, &mut rng, height, radius, wood);
            let limbs = Limbs { wood, leaves, sparse: if dark { 0.18 } else { 0.24 }, cluster: if dark { [1.5, 2.1] } else { [1.3, 1.9] }, spread: 0.55, rise: 0.35 };
            // Three or four main limbs out of the fork, each branching on.
            let main = rng.int(3, 4);
            let turn = rng.uniform(0.0, std::f64::consts::TAU);
            for n in 0..main {
                let azimuth = turn + (f64::from(n) + rng.uniform(-0.25, 0.25)) * std::f64::consts::TAU / f64::from(main);
                let dir = direction(azimuth, rng.uniform(0.45, 0.8));
                let length = rng.uniform(2.4, 3.0) * if dark { 1.1 } else { 1.0 };
                limbs.grow(&mut b, &mut rng, fork, dir, length, radius * 0.62, 2);
            }
            // The crown's top and its body, filled between the limbs' ends.
            limbs.fill(&mut b, &mut rng, fork + DVec3::Y * 2.8, [3.6, 2.2], 5);
        }
        Species::Birch => {
            let height = rng.uniform(4.5, 6.0);
            let fork = trunk(&mut b, &mut rng, height, 0.17, wood);
            let limbs = Limbs { wood, leaves, sparse: 0.32, cluster: [0.8, 1.2], spread: 0.35, rise: 0.75 };
            let leader = direction(rng.uniform(0.0, std::f64::consts::TAU), 1.35);
            let length = rng.uniform(2.4, 3.2);
            limbs.grow(&mut b, &mut rng, fork, leader, length, 0.12, 2);
            for n in 0..rng.int(2, 3) {
                let from = fork + DVec3::Y * rng.uniform(-1.2, 0.4);
                let dir = direction(f64::from(n) * 2.3 + rng.uniform(0.0, 0.8), rng.uniform(0.6, 0.9));
                let length = rng.uniform(1.6, 2.2);
                limbs.grow(&mut b, &mut rng, from, dir, length, 0.08, 2);
            }
            limbs.fill(&mut b, &mut rng, fork + DVec3::Y * 2.6, [1.8, 2.4], 3);
        }
        Species::Spruce | Species::Pine => {
            let pine = species == Species::Pine;
            let height = if pine { rng.uniform(12.0, 18.0) } else { rng.uniform(9.0, 14.0) };
            let top = DVec3::Y * height;
            b.limb(DVec3::Y * -0.4, top, if pine { 0.3 } else { 0.26 }, 0.04, wood);
            if pine {
                // A bare trunk under a crown of flat clusters.
                for _ in 0..rng.int(4, 6) {
                    let c = DVec3::Y * height * rng.uniform(0.72, 0.95) + direction(rng.uniform(0.0, std::f64::consts::TAU), 0.0) * rng.uniform(0.5, 1.6);
                    b.limb(DVec3::Y * c.y * 0.92, c, 0.08, 0.04, wood);
                    b.tier(c - DVec3::Y * 0.3, c + DVec3::Y * 0.5, rng.uniform(1.2, 1.7), 0.6, leaves, 0.25, 0.3);
                }
            } else {
                // Stacked cones, wide at the bottom.
                let tiers = rng.int(4, 6);
                let base = height * 0.18;
                for n in 0..tiers {
                    let t = f64::from(n) / f64::from(tiers);
                    let y0 = base + (height - base) * t * 0.92;
                    let reach = (2.4 - 1.9 * t) * height / 11.0;
                    b.tier(DVec3::Y * y0, DVec3::Y * (y0 + (height - base) / f64::from(tiers) * 1.6), reach, 0.15, leaves, 0.22, 0.3);
                }
            }
        }
        Species::Jungle => {
            let height = rng.uniform(15.0, 19.0);
            let radius = rng.uniform(0.5, 0.65);
            let top = DVec3::Y * height + direction(rng.uniform(0.0, std::f64::consts::TAU), 0.0) * rng.uniform(0.0, 0.8);
            b.limb(DVec3::Y * -0.5, top, radius, radius * 0.5, wood);
            // Buttress roots.
            for n in 0..rng.int(4, 6) {
                let out = direction(f64::from(n) * 1.3 + rng.uniform(0.0, 0.5), 0.0);
                b.limb(DVec3::Y * rng.uniform(1.2, 2.2) + out * radius * 0.5, out * rng.uniform(1.3, 2.0) - DVec3::Y * 0.3, radius * 0.45, 0.12, wood);
            }
            // Spreading limbs and a broad, layered canopy.
            let limbs = Limbs { wood, leaves, sparse: 0.22, cluster: [1.6, 2.4], spread: 0.6, rise: 0.25 };
            for n in 0..rng.int(4, 5) {
                let from = top - DVec3::Y * rng.uniform(1.0, 3.5);
                let dir = direction(f64::from(n) * 1.4 + rng.uniform(0.0, 0.6), rng.uniform(0.25, 0.5));
                let length = rng.uniform(2.8, 3.6);
                limbs.grow(&mut b, &mut rng, from, dir, length, radius * 0.4, 2);
            }
            limbs.fill(&mut b, &mut rng, top + DVec3::Y * 0.8, [5.0, 2.0], 6);
            // Vines' worth of leaves hanging down the trunk.
            for _ in 0..3 {
                let at = DVec3::Y * height * rng.uniform(0.3, 0.8) + direction(rng.uniform(0.0, std::f64::consts::TAU), 0.0) * radius;
                let down = at - DVec3::Y * rng.uniform(1.5, 3.0);
                b.foliage(at, down, 0.35, 0.2, leaves, 0.45, 0.15, 0.0);
            }
        }
    }
    b.stamp
}

/// A cactus grown from `seed`: a column and up to two arms.
pub fn cactus(seed: u64) -> Stamp {
    let mut rng = Rng::new(seed);
    let mut b = Builder::new();
    let height = rng.uniform(1.4, 3.2);
    b.limb(DVec3::Y * -0.3, DVec3::Y * height, 0.2, 0.17, Block::Cactus);
    for n in 0..rng.int(0, 2) {
        let out = direction(f64::from(n) * std::f64::consts::PI + rng.uniform(-0.4, 0.4), 0.0);
        let from = DVec3::Y * height * rng.uniform(0.4, 0.7);
        let elbow = from + out * rng.uniform(0.4, 0.6);
        b.limb(from, elbow, 0.13, 0.13, Block::Cactus);
        b.limb(elbow, elbow + DVec3::Y * rng.uniform(0.5, 1.0), 0.13, 0.11, Block::Cactus);
    }
    b.stamp
}

/// A bush of `leaves` grown from `seed`: a few low clusters.
pub fn bush(leaves: Block, seed: u64) -> Stamp {
    let mut rng = Rng::new(seed);
    let mut b = Builder::new();
    for _ in 0..rng.int(2, 4) {
        let c = DVec3::new(rng.uniform(-0.5, 0.5), rng.uniform(0.2, 0.5), rng.uniform(-0.5, 0.5));
        b.crown(c, rng.uniform(0.45, 0.8), leaves, 0.3);
    }
    b.stamp
}

/// A boulder of `block` grown from `seed`: rough lumps half sunk.
pub fn boulder(block: Block, seed: u64) -> Stamp {
    let mut rng = Rng::new(seed);
    let mut b = Builder::new();
    let size = rng.uniform(0.5, 1.3);
    for _ in 0..rng.int(1, 3) {
        let c = DVec3::new(rng.uniform(-0.4, 0.4), rng.uniform(-0.3, 0.1) * size, rng.uniform(-0.4, 0.4)) * size;
        let r = size * rng.uniform(0.6, 1.0);
        b.foliage(c, c, r, r, block, 0.0, r * 0.1, r * 0.3);
    }
    b.stamp
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Solid cells' box of a stamp: lowest and highest offsets, in metres.
    fn extent(s: &Stamp) -> (DVec3, DVec3) {
        let cells: Vec<[i32; 3]> = s.cells().into_iter().filter(|(_, m)| m.is_some()).map(|(p, _)| p).collect();
        let lo = cells.iter().fold([i32::MAX; 3], |a, p| [a[0].min(p[0]), a[1].min(p[1]), a[2].min(p[2])]);
        let hi = cells.iter().fold([i32::MIN; 3], |a, p| [a[0].max(p[0]), a[1].max(p[1]), a[2].max(p[2])]);
        let m = |c: [i32; 3]| DVec3::new(f64::from(c[0]), f64::from(c[1]), f64::from(c[2])) * VOXEL_M;
        (m(lo), m(hi))
    }

    /// Every species grows a tree of its size, its trunk standing on the
    /// anchor column, within a stamp's reach; a seed grows the same tree.
    #[test]
    fn every_species_grows_a_plausible_tree() {
        for (species, height) in [(Species::Oak, 7.0..13.0), (Species::DarkOak, 6.0..12.0), (Species::Birch, 8.0..16.0), (Species::Spruce, 9.0..18.0), (Species::Pine, 12.0..21.0), (Species::Jungle, 15.0..25.0)] {
            for seed in 0..3 {
                let s = tree(species, seed);
                let (lo, hi) = extent(&s);
                assert!(height.contains(&hi.y), "{species:?} {seed}: {}", hi.y);
                assert!(hi.x - lo.x > 1.0 && hi.x - lo.x < 24.0, "{species:?} {seed}: width {}", hi.x - lo.x);
                let (wood, _) = species.blocks();
                assert_eq!(s.get([0, 1, 0]), Some(Some(wood.material())), "{species:?} {seed}: trunk");
                s.compile().expect("within a stamp's reach");
                assert_eq!(s, tree(species, seed));
            }
            assert_ne!(tree(species, 1), tree(species, 2));
        }
    }

    #[test]
    fn small_growth_stands_on_its_anchor() {
        for seed in 0..4 {
            assert_eq!(cactus(seed).get([0, 2, 0]), Some(Some(Block::Cactus.material())));
            assert!(!bush(Block::OakLeaves, seed).cells().is_empty());
            assert!(boulder(Block::Stone, seed).get([0, -1, 0]).is_some());
        }
    }
}
