//! Breaking and placing blocks where the player looks.

use super::physics::{occupies, Frame};
use super::Player;
use crate::config::REACH;
use crate::input::Controls;
use crate::world::edits::BlockEdit;
use crate::world::World;
use glam::DVec3;
use helio_pass_voxel_planet::Cell;

/// The block under the crosshair.
#[derive(Clone, Copy, Debug)]
pub struct Target {
    pub cell: Cell,
    /// The empty cell in front of the face that was hit.
    pub front: Cell,
}

#[derive(Default)]
pub struct Interaction {
    pub target: Option<Target>,
    /// Seconds until the next block can be broken or placed.
    cooldown: f64,
}

/// Seconds between repeated breaks while the button is held.
const BREAK_INTERVAL: f64 = 0.17;

impl Interaction {
    /// Updates the target and applies this frame's clicks.
    pub fn update(&mut self, player: &mut Player, world: &mut World, controls: &Controls, dt: f64) {
        self.cooldown = (self.cooldown - dt).max(0.0);
        let eye = player.eye();
        let look = player.look();
        self.target = world.raycast(eye, look, REACH).map(|hit| Target { cell: hit.cell, front: hit.previous });
        let Some(target) = self.target else { return };

        if controls.mine && self.cooldown == 0.0 {
            println!("EDIT break {}:{}:{}:{}  (was {:?}, player at {:?})", target.cell.face, target.cell.i, target.cell.j, target.cell.k, world.block_at(target.cell).map(|b| b.name()), world.cell_at(player.position));
            world.edit(&[BlockEdit::clear(target.cell)]);
            self.cooldown = BREAK_INTERVAL;
        }
        if controls.pick {
            if let Some(block) = world.block_at(target.cell) {
                if let Some(slot) = crate::blocks::Block::HOTBAR.iter().position(|b| *b == block) {
                    player.hotbar = slot;
                }
            }
        }
        if controls.place && !world.is_solid(target.front) {
            let block = crate::blocks::Block::HOTBAR[player.hotbar];
            // Not into the player's own body.
            let frame: Frame = player.frame();
            if !occupies(world, player.position, &frame, target.front) {
                println!("EDIT place {}:{}:{}:{}:{}  (against {}:{}:{}:{} = {:?}, player at {:?})", target.front.face, target.front.i, target.front.j, target.front.k, block.name().replace(' ', "_"), target.cell.face, target.cell.i, target.cell.j, target.cell.k, world.block_at(target.cell).map(|b| b.name()), world.cell_at(player.position));
                world.edit(&[BlockEdit::set(target.front, block)]);
                self.cooldown = 0.12;
            }
        }
    }

    /// The twelve edges of the targeted cell, planet-centred.
    pub fn outline(&self, world: &World) -> Vec<(DVec3, DVec3)> {
        let Some(target) = self.target else { return Vec::new() };
        let grid = world.planet().grid();
        let c = target.cell;
        let corner = |a: i32, b: i32, k: i32| grid.position(c.face, [f64::from(c.i + a), f64::from(c.j + b), f64::from(c.k + k)]);
        let mut edges = Vec::with_capacity(12);
        for a in 0..2 {
            for b in 0..2 {
                edges.push((corner(a, b, 0), corner(a, b, 1)));
                edges.push((corner(a, 0, b), corner(a, 1, b)));
                edges.push((corner(0, a, b), corner(1, a, b)));
            }
        }
        edges
    }
}
