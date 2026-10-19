//! The player's body against the blocks: a box that slides along walls, lands
//! on floors, steps up single blocks and is pulled to the planet's centre.
//!
//! The world is a cube sphere, so "down" is the direction to the centre and
//! the box's axes are the local up, forward and right. Collision samples
//! points over the box and asks which cells they are in.

use crate::config::{PLAYER_HEIGHT, PLAYER_WIDTH, VOXEL_M};
use crate::world::World;
use glam::DVec3;

/// The local frame at a point: up (away from the centre), and the tangent
/// axes.
#[derive(Clone, Copy, Debug)]
pub struct Frame {
    pub up: DVec3,
    pub forward: DVec3,
    pub right: DVec3,
}

impl Frame {
    /// The frame at `position` facing `heading` (projected onto the tangent
    /// plane).
    pub fn new(position: DVec3, heading: DVec3) -> Self {
        let up = position.normalize_or(DVec3::Y);
        let forward = (heading - up * heading.dot(up)).try_normalize().unwrap_or_else(|| up.any_orthonormal_vector());
        Frame { up, forward, right: forward.cross(up) }
    }
}

/// Offsets of the sample points: across the box at three heights.
fn samples() -> impl Iterator<Item = (f64, f64, f64)> {
    let r = PLAYER_WIDTH * 0.5 - 0.02;
    let heights = [0.002, 0.55, 1.1, 1.65, PLAYER_HEIGHT - 0.04];
    heights.into_iter().flat_map(move |h| [(0.0, 0.0), (r, r), (-r, r), (r, -r), (-r, -r), (r, 0.0), (-r, 0.0), (0.0, r), (0.0, -r)].into_iter().map(move |(x, z)| (h, x, z)))
}

/// Whether the box standing at `feet` (in `frame`) overlaps solid blocks.
pub fn collides(world: &World, feet: DVec3, frame: &Frame) -> bool {
    samples().any(|(h, x, z)| world.is_solid(world.cell_at(feet + frame.up * h + frame.right * x + frame.forward * z)))
}

/// Whether the cell is inside the box standing at `feet`.
pub fn occupies(world: &World, feet: DVec3, frame: &Frame, cell: helio_pass_voxel_planet::Cell) -> bool {
    samples().any(|(h, x, z)| world.cell_at(feet + frame.up * h + frame.right * x + frame.forward * z) == cell)
}

/// What a movement did.
#[derive(Clone, Copy, Debug, Default)]
pub struct Moved {
    pub landed: bool,
    pub hit_ceiling: bool,
    pub stepped: bool,
}

/// Moves the box from `feet` by `delta`, resolving vertical movement first and
/// then each horizontal axis, with step-up. Returns the new position.
pub fn slide(world: &World, mut feet: DVec3, delta: DVec3, frame: &Frame, can_step: bool) -> (DVec3, Moved) {
    let mut moved = Moved::default();
    let vertical = delta.dot(frame.up);
    let horizontal = [frame.right * delta.dot(frame.right), frame.forward * delta.dot(frame.forward)];

    // Vertical, in steps shorter than a block.
    let steps = (vertical.abs() / 0.2).ceil().max(1.0) as i32;
    for _ in 0..steps {
        let next = feet + frame.up * (vertical / f64::from(steps));
        if collides(world, next, frame) {
            if vertical < 0.0 {
                moved.landed = true;
                feet = rest_on_floor(world, feet, frame);
            } else {
                moved.hit_ceiling = true;
            }
            break;
        }
        feet = next;
    }

    // Horizontal axes.
    for part in horizontal {
        let length = part.length();
        if length < 1e-9 {
            continue;
        }
        let steps = (length / 0.2).ceil().max(1.0) as i32;
        for _ in 0..steps {
            let next = feet + part / f64::from(steps);
            if !collides(world, next, frame) {
                feet = next;
                continue;
            }
            // A block in the way: climb it if there is room over it.
            if can_step {
                let lifted = next + frame.up * (VOXEL_M + 0.02);
                if !collides(world, lifted, frame) && !collides(world, feet + frame.up * (VOXEL_M + 0.02), frame) {
                    feet = rest_on_floor(world, lifted, frame);
                    moved.stepped = true;
                    continue;
                }
            }
            break;
        }
    }
    (feet, moved)
}

/// `feet` lowered to rest on the floor under it (the top of the cell its sole
/// is in), when it is in one; otherwise unchanged.
pub fn rest_on_floor(world: &World, feet: DVec3, frame: &Frame) -> DVec3 {
    let grid = world.planet().grid();
    let mut down = feet;
    // Down to the first solid cell under the sole, within a block.
    for _ in 0..24 {
        let next = down - frame.up * 0.05;
        if world.is_solid(world.cell_at(next)) {
            let cell = world.cell_at(next);
            let top = grid.layer_radius(f64::from(cell.k) + 1.0);
            return grid.at_radial(next, top + 0.001);
        }
        down = next;
    }
    feet
}

/// Height above the nearest floor below, up to `limit` metres (None if
/// there is none within it).
pub fn floor_below(world: &World, feet: DVec3, frame: &Frame, limit: f64) -> Option<f64> {
    let mut d = 0.0;
    while d <= limit {
        if world.is_solid(world.cell_at(feet - frame.up * d)) {
            return Some(d);
        }
        d += 0.01;
    }
    None
}
