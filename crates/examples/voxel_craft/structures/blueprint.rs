//! A structure as blocks around an anchor, built with box and line helpers.
//!
//! Offsets are `(x, y, z)` with +Y up and `y = 0` the floor level (the first
//! layer over the ground); the structure's footprint is centred on `x = z = 0`.

use crate::blocks::Block;
use std::collections::HashMap;

#[derive(Clone, Debug, Default)]
pub struct Blueprint {
    pub blocks: HashMap<[i32; 3], Option<Block>>,
    /// Half extents of the footprint (blocks from the centre column).
    pub half: [i32; 2],
    /// Layers the structure reaches above the floor.
    pub height: i32,
}

impl Blueprint {
    pub fn new(half_x: i32, half_z: i32) -> Self {
        Self { blocks: HashMap::new(), half: [half_x, half_z], height: 0 }
    }

    pub fn set(&mut self, at: [i32; 3], block: Block) {
        self.height = self.height.max(at[1] + 1);
        self.blocks.insert(at, Some(block));
    }

    /// Air where there may be terrain or leaves.
    pub fn clear(&mut self, at: [i32; 3]) {
        self.blocks.insert(at, None);
    }

    /// Fills the box between `a` and `b` (inclusive).
    pub fn fill(&mut self, a: [i32; 3], b: [i32; 3], block: Block) {
        for x in a[0].min(b[0])..=a[0].max(b[0]) {
            for y in a[1].min(b[1])..=a[1].max(b[1]) {
                for z in a[2].min(b[2])..=a[2].max(b[2]) {
                    self.set([x, y, z], block);
                }
            }
        }
    }

    /// Clears the box between `a` and `b` (inclusive).
    pub fn hollow(&mut self, a: [i32; 3], b: [i32; 3]) {
        for x in a[0].min(b[0])..=a[0].max(b[0]) {
            for y in a[1].min(b[1])..=a[1].max(b[1]) {
                for z in a[2].min(b[2])..=a[2].max(b[2]) {
                    self.clear([x, y, z]);
                }
            }
        }
    }

    /// A room's four walls (`a` to `b`, outer box) of `block`, between layers.
    pub fn walls(&mut self, a: [i32; 3], b: [i32; 3], block: Block) {
        let (x0, x1, z0, z1) = (a[0].min(b[0]), a[0].max(b[0]), a[2].min(b[2]), a[2].max(b[2]));
        for y in a[1].min(b[1])..=a[1].max(b[1]) {
            for x in x0..=x1 {
                self.set([x, y, z0], block);
                self.set([x, y, z1], block);
            }
            for z in z0..=z1 {
                self.set([x0, y, z], block);
                self.set([x1, y, z], block);
            }
        }
    }

    /// The blueprint turned by `quarter_turns` about the vertical, and mirrored
    /// in x when `mirror`.
    pub fn oriented(&self, quarter_turns: u32, mirror: bool) -> Blueprint {
        let turn = |p: [i32; 3]| {
            let [mut x, y, mut z] = p;
            if mirror {
                x = -x;
            }
            for _ in 0..quarter_turns % 4 {
                (x, z) = (-z, x);
            }
            [x, y, z]
        };
        let half = if quarter_turns % 2 == 1 { [self.half[1], self.half[0]] } else { self.half };
        Blueprint { blocks: self.blocks.iter().map(|(p, b)| (turn(*p), *b)).collect(), half, height: self.height }
    }
}
