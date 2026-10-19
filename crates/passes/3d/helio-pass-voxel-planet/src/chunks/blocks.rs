//! Exact materials of edited cells, for the shader.
//!
//! A chunk's leaf holds two materials (a surface's and a subsurface's), which
//! is right for ground and wrong for what people build: a log against leaves,
//! a plank wall in a stone leaf. Edits, though, are stored exactly: the baked
//! store keeps every edited cell's material in 8x8x8 bricks. This table puts
//! the base level's bricks near the eye on the GPU (one byte a cell, in an
//! open-addressed hash keyed by brick) so the publish stage can show an edited
//! cell's own material wherever the leaf's pair would have it wrong.
//!
//! A world can hold far more edited bricks than the table does (a decorated
//! planet is hundreds of thousands), so the table is a cache around the eye:
//! columns of bricks within [`RADIUS`] brick columns are resident, those past
//! [`RADIUS`] + a margin are evicted as the eye moves, and what lies beyond
//! shows its leaf's pair (at that distance a block is a few pixels).
//!
//! Layout of the buffer, in words: `CAPACITY` table entries of four (face + 1,
//! brick i, j, k; face 0 is an empty entry, `u32::MAX` a freed one that probes
//! continue past) then `CAPACITY` bricks of 128 words (four cells' bytes each,
//! `x + 8 (y + 8 z)`), the brick of table entry `e` at
//! `CAPACITY * 4 + e * 128`.
//!
//! Changes queue and [`BlockMaterials::flush`] uploads a budget of them each
//! frame, small changes (an edit someone is looking at) first.

use crate::edit_store::{Brick, BrickKey, ColumnKey, EditStore};
use rustc_hash::FxHashMap;
use std::sync::Arc;

/// Bricks the table holds (a power of two).
pub const CAPACITY: usize = 1 << 17;
const BRICK_WORDS: usize = 128;
/// Probes before an insertion gives up (the shader probes as many).
pub const PROBES: u32 = 64;
/// Brick columns (8 cells) around the eye kept resident: 64 is 512 m.
pub const RADIUS: i32 = 64;
/// Columns this far past [`RADIUS`] stay before eviction (no thrashing).
const MARGIN: i32 = 8;
const FREED: i32 = -1;

/// The table's hash (the shader's `block_hash`).
pub fn hash(face: u32, bi: i32, bj: i32, bk: i32) -> u32 {
    let mut h = face.wrapping_mul(0x9e37_79b1) ^ (bi as u32).wrapping_mul(0x85eb_ca77) ^ (bj as u32).wrapping_mul(0xc2b2_ae3d) ^ (bk as u32).wrapping_mul(0x27d4_eb2f);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2c1b_3c6d);
    h ^= h >> 13;
    h
}

pub struct BlockMaterials {
    pub buffer: wgpu::Buffer,
    /// The table as the GPU has it.
    keys: Vec<[i32; 4]>,
    /// The store (a cheap shared clone): columns load from it as the eye nears them.
    store: EditStore,
    /// The columns on (or on their way to) the GPU, with their brick heights.
    resident: FxHashMap<ColumnKey, Vec<i32>>,
    /// Bricks to write: the latest state of each (`None`: now empty).
    pending: FxHashMap<BrickKey, Option<Arc<Brick>>>,
    /// Bricks of small changes: written before any of `pending`.
    urgent: FxHashMap<BrickKey, Option<Arc<Brick>>>,
    /// The eye's brick column the residency was last made for.
    centre: Option<(u8, i32, i32)>,
    warned: bool,
}

impl BlockMaterials {
    pub fn new(device: &wgpu::Device) -> Self {
        let words = CAPACITY * 4 + CAPACITY * BRICK_WORDS;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunk block materials"),
            size: (words * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            buffer,
            keys: vec![[0; 4]; CAPACITY],
            store: EditStore::default(),
            resident: FxHashMap::default(),
            pending: FxHashMap::default(),
            urgent: FxHashMap::default(),
            centre: None,
            warned: false,
        }
    }

    /// The table entry of a brick, adding it if it is new.
    fn entry(&mut self, key: &BrickKey) -> Option<(usize, bool)> {
        let wanted = [i32::from(key.face) + 1, key.bi, key.bj, key.bk];
        let start = hash(u32::from(key.face), key.bi, key.bj, key.bk) as usize;
        let mut free = None;
        for p in 0..PROBES as usize {
            let e = (start + p) & (CAPACITY - 1);
            if self.keys[e] == wanted {
                return Some((e, false));
            }
            if self.keys[e][0] == FREED && free.is_none() {
                free = Some(e);
            }
            if self.keys[e][0] == 0 {
                let e = free.unwrap_or(e);
                self.keys[e] = wanted;
                return Some((e, true));
            }
        }
        if let Some(e) = free {
            self.keys[e] = wanted;
            return Some((e, true));
        }
        None
    }

    /// Frees a brick's table entry.
    fn free(&mut self, queue: &wgpu::Queue, key: &BrickKey) {
        let wanted = [i32::from(key.face) + 1, key.bi, key.bj, key.bk];
        let start = hash(u32::from(key.face), key.bi, key.bj, key.bk) as usize;
        for p in 0..PROBES as usize {
            let e = (start + p) & (CAPACITY - 1);
            if self.keys[e] == wanted {
                self.keys[e] = [FREED, 0, 0, 0];
                queue.write_buffer(&self.buffer, (e * 16) as u64, bytemuck::cast_slice(&self.keys[e]));
                return;
            }
            if self.keys[e][0] == 0 {
                return;
            }
        }
    }

    fn write_brick(&mut self, queue: &wgpu::Queue, key: &BrickKey, brick: Option<&Brick>) {
        let Some((e, new)) = self.entry(key) else {
            if !self.warned {
                eprintln!("chunk block materials: table full, edited cells beyond it show their leaf material");
                self.warned = true;
            }
            return;
        };
        if new {
            queue.write_buffer(&self.buffer, (e * 16) as u64, bytemuck::cast_slice(&self.keys[e]));
        }
        let mut words = [0u32; BRICK_WORDS];
        if let Some(brick) = brick {
            for (c, cell) in brick.cells.iter().enumerate() {
                // Solid and painted cells show their material; air shows nothing.
                let material = if cell.state() >= 2 { cell.material() & 0xff } else { 0 };
                words[c / 4] |= material << ((c % 4) * 8);
            }
        }
        queue.write_buffer(&self.buffer, ((CAPACITY * 4 + e * BRICK_WORDS) * 4) as u64, bytemuck::cast_slice(&words));
    }

    /// Makes a column's bricks resident: queued at `urgent` or bulk priority.
    fn load(&mut self, column: ColumnKey, urgent: bool) {
        let mut heights = Vec::new();
        if let Some(now) = self.store.column(&column).cloned() {
            for (bk, brick) in now.iter() {
                let key = BrickKey { face: column.face, level: 0, bi: column.ci, bj: column.cj, bk: *bk };
                if urgent {
                    self.pending.remove(&key);
                    self.urgent.insert(key, Some(brick.clone()));
                } else {
                    self.urgent.remove(&key);
                    self.pending.insert(key, Some(brick.clone()));
                }
                heights.push(*bk);
            }
        }
        if !heights.is_empty() {
            self.resident.insert(column, heights);
        }
    }

    /// Everything the store holds at the base level.
    pub fn fill_all(&mut self, store: &EditStore) {
        self.store = store.clone();
        // Resident once the eye is known (`follow`).
        self.centre = None;
    }

    /// Follows `store`'s changes in `changed` columns. Resident columns are
    /// brought up to date (small changes first); others only update the
    /// record, and load when the eye comes near.
    pub fn sync(&mut self, store: &EditStore, changed: &[ColumnKey]) {
        self.store = store.clone();
        let mut done: rustc_hash::FxHashSet<ColumnKey> = Default::default();
        let small = changed.len() <= 256;
        for column in changed.iter().filter(|c| c.level == 0) {
            if !done.insert(*column) {
                continue;
            }
            let Some(before) = self.resident.remove(column) else {
                // New to the store: load it now if it is within reach of the eye.
                if let Some((face, ci, cj)) = self.centre {
                    if column.face == face && (column.ci - ci).abs() <= RADIUS && (column.cj - cj).abs() <= RADIUS {
                        self.load(*column, small);
                    }
                }
                continue;
            };
            self.load(*column, small);
            let now = self.resident.get(column).cloned().unwrap_or_default();
            for bk in before.into_iter().filter(|bk| !now.contains(bk)) {
                let key = BrickKey { face: column.face, level: 0, bi: column.ci, bj: column.cj, bk };
                self.pending.remove(&key);
                self.urgent.insert(key, None);
            }
            // A column that emptied stays known as resident while it has
            // queued clears: nothing more to track.
        }
    }

    /// Residency around the eye's brick column `(ci, cj)` of `face`: columns
    /// within [`RADIUS`] load, those past [`RADIUS`] + margin are evicted.
    pub fn follow(&mut self, queue: &wgpu::Queue, face: u8, ci: i32, cj: i32) {
        if let Some((f, i, j)) = self.centre {
            if f == face && (i - ci).abs() < 4 && (j - cj).abs() < 4 {
                return;
            }
        }
        self.centre = Some((face, ci, cj));
        let near = |c: &ColumnKey, r: i32| c.face == face && (c.ci - ci).abs() <= r && (c.cj - cj).abs() <= r;
        // Evict what the eye has left behind.
        let gone: Vec<ColumnKey> = self.resident.keys().filter(|c| !near(c, RADIUS + MARGIN)).copied().collect();
        for column in gone {
            for bk in self.resident.remove(&column).unwrap_or_default() {
                let key = BrickKey { face: column.face, level: 0, bi: column.ci, bj: column.cj, bk };
                self.pending.remove(&key);
                self.urgent.remove(&key);
                self.free(queue, &key);
            }
        }
        // Load the columns that came near, nearest first.
        let mut wanted: Vec<(i32, ColumnKey)> = Vec::new();
        {
            for di in -RADIUS..=RADIUS {
                for dj in -RADIUS..=RADIUS {
                    let c = ColumnKey { face, level: 0, ci: ci + di, cj: cj + dj };
                    if self.store.column(&c).is_some() && !self.resident.contains_key(&c) {
                        wanted.push((di.abs().max(dj.abs()), c));
                    }
                }
            }
        }
        wanted.sort_by_key(|(d, _)| *d);
        for (_, column) in wanted {
            self.load(column, false);
        }
    }

    /// Writes the urgent bricks, then queued ones until `budget` of them or
    /// about two milliseconds are spent; how many are left.
    pub fn flush(&mut self, queue: &wgpu::Queue, budget: usize) -> usize {
        let urgent: Vec<BrickKey> = self.urgent.keys().copied().collect();
        for key in urgent {
            if let Some(brick) = self.urgent.remove(&key) {
                self.write_brick(queue, &key, brick.as_deref());
            }
        }
        if self.pending.is_empty() {
            return 0;
        }
        let started = std::time::Instant::now();
        let keys: Vec<BrickKey> = self.pending.keys().take(budget).copied().collect();
        for key in keys {
            if let Some(brick) = self.pending.remove(&key) {
                self.write_brick(queue, &key, brick.as_deref());
            }
            if started.elapsed().as_micros() > 2000 {
                break;
            }
        }
        self.pending.len()
    }
}
