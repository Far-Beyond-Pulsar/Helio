//! Procedural blocky trees.
//!
//! The generators are Rust ports of NGNT/treegen-pinegen's tree workers (MIT,
//! `assets/trees/LICENSE-treegen-pinegen`): branching oaks, slim birches,
//! tiered spruces and pines, and buttressed jungle giants. Each makes a
//! [`Volume`] of wood and leaves from a seed; [`Tree`] re-bases it so the trunk
//! stands at the origin with +Y up and gives each part its species' block.

mod birch;
mod jungle;
mod oak;
mod rng;
mod shapes;
mod spruce;

pub use rng::Rng;

use crate::blocks::Block;
use shapes::{Part, Volume, CENTRE};

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
            Species::Oak => (Block::OakLog, Block::OakLeaves),
            Species::DarkOak => (Block::OakLog, Block::OakLeaves),
            Species::Birch => (Block::BirchLog, Block::BirchLeaves),
            Species::Spruce | Species::Pine => (Block::SpruceLog, Block::SpruceLeaves),
            Species::Jungle => (Block::JungleLog, Block::JungleLeaves),
        }
    }
}

/// A grown tree: blocks at offsets from its base (x, y up, z).
#[derive(Clone, Debug)]
pub struct Tree {
    pub species: Species,
    pub blocks: Vec<([i32; 3], Block)>,
}

impl Tree {
    /// The tree of `species` grown from `seed`.
    pub fn grow(species: Species, seed: u64) -> Tree {
        // Seeds of nearby trees differ in a few bits: spread them.
        let seed = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(23) ^ 0x5851_f42d_4c95_7f2d;
        let volume = match species {
            Species::Oak => oak::generate(&oak::OakParams::oak(), seed),
            Species::DarkOak => oak::generate(&oak::OakParams::dark_oak(), seed),
            Species::Birch => birch::generate(&birch::BirchParams::birch(), seed),
            Species::Spruce => spruce::generate(&spruce::SpruceParams::spruce(), seed),
            Species::Pine => spruce::generate(&spruce::SpruceParams::pine(), seed),
            Species::Jungle => jungle::generate(&jungle::JungleParams::jungle(), seed),
        };
        Self::from_volume(species, &volume)
    }

    fn from_volume(species: Species, volume: &Volume) -> Tree {
        let (log, leaf) = species.blocks();
        let centre = CENTRE as i32;
        let mut blocks: Vec<([i32; 3], Block)> = volume
            .cells
            .iter()
            .map(|(c, part)| ([c[0] - centre, c[1], c[2] - centre], if *part == Part::Wood { log } else { leaf }))
            .collect();
        // Deterministic order (the volume is a hash map).
        blocks.sort_by_key(|(c, _)| (c[1], c[0], c[2]));
        Tree { species, blocks }
    }

    /// Smallest and largest offsets of the tree's blocks.
    pub fn bounds(&self) -> ([i32; 3], [i32; 3]) {
        let mut lo = [i32::MAX; 3];
        let mut hi = [i32::MIN; 3];
        for (c, _) in &self.blocks {
            for k in 0..3 {
                lo[k] = lo[k].min(c[k]);
                hi[k] = hi[k].max(c[k]);
            }
        }
        (lo, hi)
    }

    /// Height of the tree in blocks.
    pub fn height(&self) -> i32 {
        let (lo, hi) = self.bounds();
        hi[1] - lo[1] + 1
    }

    pub fn log_count(&self) -> usize {
        let (log, _) = self.species.blocks();
        self.blocks.iter().filter(|(_, b)| *b == log).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats(species: Species) -> Vec<(i32, i32, usize, usize)> {
        (0..12u64)
            .map(|seed| {
                let t = Tree::grow(species, seed);
                let (lo, hi) = t.bounds();
                let width = (hi[0] - lo[0]).max(hi[2] - lo[2]) + 1;
                (t.height(), width, t.log_count(), t.blocks.len() - t.log_count())
            })
            .collect()
    }

    #[test]
    fn every_species_grows_a_plausible_tree() {
        for (species, height, width) in [
            (Species::Oak, 9..=26, 6..=26),
            (Species::DarkOak, 7..=22, 6..=26),
            (Species::Birch, 10..=30, 3..=20),
            (Species::Spruce, 12..=34, 5..=22),
            (Species::Pine, 14..=44, 4..=22),
            (Species::Jungle, 22..=42, 10..=36),
        ] {
            for (h, w, logs, leaves) in stats(species) {
                println!("{species:?}: height {h} width {w} logs {logs} leaves {leaves}");
                assert!(height.contains(&h), "{species:?} height {h}");
                assert!(width.contains(&w), "{species:?} width {w}");
                assert!(logs >= 6 && leaves >= 20, "{species:?} logs {logs} leaves {leaves}");
                assert!(logs + leaves < 4000, "{species:?} is too large: {}", logs + leaves);
            }
        }
    }

    #[test]
    fn trees_are_a_function_of_their_seed() {
        for species in [Species::Oak, Species::Birch, Species::Spruce, Species::Jungle] {
            assert_eq!(Tree::grow(species, 3).blocks, Tree::grow(species, 3).blocks);
            assert_ne!(Tree::grow(species, 3).blocks, Tree::grow(species, 4).blocks);
        }
    }

    #[test]
    fn trunks_stand_on_the_base() {
        for species in [Species::Oak, Species::Birch, Species::Spruce, Species::Jungle] {
            let t = Tree::grow(species, 9);
            let (log, _) = species.blocks();
            assert!(t.blocks.iter().any(|(c, b)| *b == log && c[1] == 0 && c[0].abs() <= 3 && c[2].abs() <= 3), "{species:?} trunk base");
        }
    }
}

#[cfg(test)]
mod preview {
    use super::*;

    /// `cargo test --bin voxel_craft print_trees -- --ignored --nocapture`:
    /// side silhouettes ('#' wood, '.' leaves) of each species.
    #[test]
    #[ignore]
    fn print_trees() {
        for species in [Species::Oak, Species::Birch, Species::Spruce, Species::Jungle] {
            let t = Tree::grow(species, 5);
            let (lo, hi) = t.bounds();
            println!("{species:?} {}x{}", hi[0] - lo[0] + 1, hi[1] - lo[1] + 1);
            for y in (lo[1]..=hi[1]).rev() {
                let row: String = (lo[0]..=hi[0])
                    .map(|x| {
                        let at = t.blocks.iter().filter(|(c, _)| c[0] == x && c[1] == y);
                        let mut s = ' ';
                        for (_, b) in at {
                            s = if b.surfaces()[1] != crate::blocks::Block::OakLeaves.surfaces()[1] && format!("{b:?}").contains("Log") { '#' } else if s == '#' { '#' } else { '.' };
                        }
                        s
                    })
                    .collect();
                println!("{row}");
            }
        }
    }
}
