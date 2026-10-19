//! Exact materials of edited cells, for the shader.
//!
//! A chunk's leaf holds two materials (a surface's and a subsurface's), which
//! is right for ground and wrong for what people build: a log against leaves,
//! a plank wall in a stone leaf. Edits, though, are stored exactly: the baked
//! store keeps every edited cell's material in 8x8x8 bricks. This table puts
//! the base level's bricks on the GPU (one byte a cell, in an open-addressed
//! hash keyed by brick) so the publish stage can show an edited cell's own
//! material wherever the leaf's pair would have it wrong.
//!
//! Layout of the buffer, in words: `CAPACITY` table entries of four (face + 1,
//! brick i, j, k; face 0 is an empty entry) then `CAPACITY` bricks of 128
//! words (four cells' bytes each, `x + 8 (y + 8 z)`), the brick of table entry
//! `e` at `CAPACITY * 4 + e * 128`.

use crate::edit_store::{BrickKey, ColumnKey, EditStore};
use rustc_hash::FxHashMap;

/// Bricks the table holds (a power of two).
pub const CAPACITY: usize = 1 << 17;
const BRICK_WORDS: usize = 128;
/// Probes before an insertion gives up (the shader probes as many).
pub const PROBES: u32 = 64;

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
    /// The base level's bricks (their heights) the GPU has, by column.
    columns: FxHashMap<ColumnKey, Vec<i32>>,
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
        Self { buffer, keys: vec![[0; 4]; CAPACITY], columns: FxHashMap::default(), warned: false }
    }

    /// The table entry of a brick, adding it if it is new.
    fn entry(&mut self, key: &BrickKey) -> Option<(usize, bool)> {
        let wanted = [i32::from(key.face) + 1, key.bi, key.bj, key.bk];
        let start = hash(u32::from(key.face), key.bi, key.bj, key.bk) as usize;
        for p in 0..PROBES as usize {
            let e = (start + p) & (CAPACITY - 1);
            if self.keys[e] == wanted {
                return Some((e, false));
            }
            if self.keys[e][0] == 0 {
                self.keys[e] = wanted;
                return Some((e, true));
            }
        }
        None
    }

    fn write_brick(&mut self, queue: &wgpu::Queue, key: &BrickKey, brick: Option<&crate::edit_store::Brick>) {
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

    /// Everything the store holds at the base level.
    pub fn fill_all(&mut self, queue: &wgpu::Queue, store: &EditStore) {
        let mut seen: Vec<ColumnKey> = Vec::new();
        for (key, brick) in store.bricks().filter(|(k, _)| k.level == 0) {
            self.write_brick(queue, &key, Some(brick));
            self.columns.entry(key.column()).or_default().push(key.bk);
        }
    }

    /// Follows `store`'s changes in `changed` columns: their bricks are
    /// written again, and bricks they no longer hold are cleared.
    pub fn sync(&mut self, queue: &wgpu::Queue, store: &EditStore, changed: &[ColumnKey]) {
        let mut done: rustc_hash::FxHashSet<ColumnKey> = Default::default();
        for column in changed.iter().filter(|c| c.level == 0) {
            if !done.insert(*column) {
                continue;
            }
            let now = store.column(column);
            let before = self.columns.remove(column).unwrap_or_default();
            let mut heights: Vec<i32> = Vec::new();
            if let Some(now) = now {
                for (bk, brick) in now.iter() {
                    let key = BrickKey { face: column.face, level: 0, bi: column.ci, bj: column.cj, bk: *bk };
                    self.write_brick(queue, &key, Some(brick));
                    heights.push(*bk);
                }
            }
            for bk in before.into_iter().filter(|bk| !heights.contains(bk)) {
                let key = BrickKey { face: column.face, level: 0, bi: column.ci, bj: column.cj, bk };
                self.write_brick(queue, &key, None);
            }
            if !heights.is_empty() {
                self.columns.insert(*column, heights);
            }
        }
    }
}
