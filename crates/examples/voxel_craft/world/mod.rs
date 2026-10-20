//! The world the game sees: a voxel planet with edits applied off-thread
//! and block queries that include edits not yet in the latest snapshot. Its
//! decoration (trees, structures) is the planet's features (`decorate`),
//! part of its recipe.

pub mod decorate;
pub mod edits;
mod worker;

use crate::blocks::Block;
use crate::config::VOXEL_M;
use edits::Edit;
use glam::DVec3;
use helio_pass_voxel_planet::{Cell, Planet, RayHit};
use std::collections::HashMap;
use std::sync::Arc;
use worker::{Batch, Job, Worker};

pub struct World {
    snapshot: Arc<Planet>,
    worker: Worker,
    next_ticket: u64,
    /// Edits submitted but not yet in the snapshot, by cell: the ticket of the
    /// batch and what the cell is (`None`: air).
    pending: HashMap<Cell, (u64, Option<Block>)>,
    snapshot_ticket: u64,
}

impl World {
    pub fn new(planet: Planet) -> Self {
        let snapshot = Arc::new(planet.clone());
        Self { snapshot, worker: Worker::spawn(planet), next_ticket: 1, pending: HashMap::new(), snapshot_ticket: 0 }
    }

    /// The latest finished planet, for drawing and for queries.
    pub fn planet(&self) -> &Arc<Planet> {
        &self.snapshot
    }

    /// Takes finished snapshots from the worker. Call once a frame.
    pub fn update(&mut self) {
        if let Some(done) = self.worker.latest() {
            self.snapshot = done.planet;
            self.snapshot_ticket = done.ticket;
            self.pending.retain(|_, (ticket, _)| *ticket > done.ticket);
        }
    }

    /// Applies `edits` (off-thread); queries see them immediately.
    pub fn edit(&mut self, edits: &[Edit]) {
        if edits.is_empty() {
            return;
        }
        let ticket = self.next_ticket;
        self.next_ticket += 1;
        let grid = *self.snapshot.grid();
        let brushes = edits.iter().map(|e| e.brush(&grid)).collect();
        for e in edits {
            for (cell, block) in e.cells(&grid) {
                self.pending.insert(cell, (ticket, block));
            }
        }
        self.worker.submit(Job::Edit(Batch { ticket, brushes }));
    }

    pub fn is_solid(&self, cell: Cell) -> bool {
        match self.pending.get(&cell) {
            Some((_, block)) => block.is_some(),
            None => self.snapshot.solid(cell),
        }
    }

    /// The block at `cell` (`None`: air or ground the game has no block for).
    pub fn block_at(&self, cell: Cell) -> Option<Block> {
        match self.pending.get(&cell) {
            Some((_, block)) => *block,
            None if self.snapshot.solid(cell) => Block::from_material(self.snapshot.material(cell)),
            None => None,
        }
    }

    /// The cell containing a planet-centred point.
    pub fn cell_at(&self, p: DVec3) -> Cell {
        self.snapshot.grid().locate(p).0
    }

    pub fn cell_centre(&self, cell: Cell) -> DVec3 {
        self.snapshot.grid().cell_center(cell)
    }

    /// The first solid cell along a ray, as edited (pending edits included
    /// near the ray: re-checked against the cells they touch).
    pub fn raycast(&self, origin: DVec3, direction: DVec3, max: f64) -> Option<RayHit> {
        let hit = self.snapshot.raycast(origin, direction, max);
        // A pending edit may have removed the hit or added something nearer:
        // walk the ray in half-block steps over the pending cells.
        if self.pending.is_empty() {
            return hit;
        }
        let end = hit.as_ref().map_or(max, |h| h.distance);
        let step = VOXEL_M * 0.25;
        let mut t = 0.0;
        let mut previous = self.cell_at(origin);
        while t < end {
            let cell = self.cell_at(origin + direction * t);
            if cell != previous {
                if self.pending.get(&cell).is_some_and(|(_, b)| b.is_some()) {
                    return Some(RayHit { cell, previous, distance: t, normal: self.face_normal(previous, cell) });
                }
                previous = cell;
            }
            t += step;
        }
        match hit {
            Some(h) if matches!(self.pending.get(&h.cell), Some((_, None))) => self.raycast_past(origin, direction, max, h.distance),
            other => other,
        }
    }

    /// The ray's next hit after a removed cell: the planet's, from just past.
    fn raycast_past(&self, origin: DVec3, direction: DVec3, max: f64, from: f64) -> Option<RayHit> {
        let start = from + VOXEL_M * 0.6;
        if start >= max {
            return None;
        }
        self.snapshot.raycast(origin + direction * start, direction, max - start).map(|mut h| {
            h.distance += start;
            h
        })
    }

    fn face_normal(&self, from: Cell, to: Cell) -> DVec3 {
        (self.cell_centre(from) - self.cell_centre(to)).normalize_or_zero()
    }
}
