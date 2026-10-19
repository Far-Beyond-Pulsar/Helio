//! A structure as blocks around an anchor: the voxel planet's [`Stamp`]
//! in this game's blocks, with the footprint and height a site needs.
//!
//! Offsets are `(x, y, z)` with +Y up and `y = 0` the floor level (the first
//! layer over the ground); the structure's footprint is centred on `x = z = 0`.

use crate::blocks::Block;
use helio_pass_voxel_planet::stamp::Stamp;

#[derive(Clone, Debug, Default)]
pub struct Blueprint {
    pub stamp: Stamp,
    /// Half extents of the footprint (blocks from the centre column).
    pub half: [i32; 2],
    /// Layers the structure reaches above the floor.
    pub height: i32,
}

impl Blueprint {
    pub fn new(half_x: i32, half_z: i32) -> Self {
        Self { stamp: Stamp::new(), half: [half_x, half_z], height: 0 }
    }

    pub fn set(&mut self, at: [i32; 3], block: Block) {
        self.height = self.height.max(at[1] + 1);
        self.stamp.set(at, block.material());
    }

    /// Air where there may be terrain or leaves.
    pub fn clear(&mut self, at: [i32; 3]) {
        self.stamp.clear(at);
    }

    /// Fills the box between `a` and `b` (inclusive).
    pub fn fill(&mut self, a: [i32; 3], b: [i32; 3], block: Block) {
        self.height = self.height.max(a[1].max(b[1]) + 1);
        self.stamp.fill(a, b, block.material());
    }

    /// Clears the box between `a` and `b` (inclusive).
    pub fn hollow(&mut self, a: [i32; 3], b: [i32; 3]) {
        self.stamp.hollow(a, b);
    }

    /// A room's four walls (`a` to `b`, outer box) of `block`, between layers.
    pub fn walls(&mut self, a: [i32; 3], b: [i32; 3], block: Block) {
        self.height = self.height.max(a[1].max(b[1]) + 1);
        self.stamp.walls(a, b, block.material());
    }

    /// The blueprint turned by `quarter_turns` about the vertical, and mirrored
    /// in x when `mirror`.
    pub fn oriented(&self, quarter_turns: u32, mirror: bool) -> Blueprint {
        let half = if quarter_turns % 2 == 1 { [self.half[1], self.half[0]] } else { self.half };
        Blueprint { stamp: self.stamp.oriented(quarter_turns, mirror), half, height: self.height }
    }
}
