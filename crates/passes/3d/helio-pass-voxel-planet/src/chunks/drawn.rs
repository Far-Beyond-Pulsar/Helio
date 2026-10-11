//! The drawn chunks and their GPU chunk table, kept up to date by the
//! tiles that change: a frame costs what changed, not what is resident.
//!
//! The table is open addressing with linear probing over a fixed power of
//! two of entries (key, then slot; an empty key's x is -1), mirrored on
//! the CPU; deletion shifts the probe run back, so lookups never meet a
//! tombstone. Only changed entries are uploaded.

use super::TileKey;
use glam::{DMat3, DVec3};
use rustc_hash::FxHashMap;

/// A drawn chunk, as the draw and the chunk table need it.
#[derive(Clone, Copy, Debug)]
pub struct DrawnChunk {
    pub key: [i32; 4],
    pub slot: u32,
    pub origin: DVec3,
    pub inv: DMat3,
    /// Bounding sphere centre (planet-centred) and radius.
    pub centre: DVec3,
    pub radius: f64,
    pub z: (f64, f64),
}

const NONE: u32 = u32::MAX;
const EMPTY: [i32; 4] = [-1, 0, 0, 0];

pub fn key_hash(k: [i32; 4]) -> u32 {
    let mut h = (k[0] as u32).wrapping_mul(0x9E37_79B1);
    h = (h ^ k[1] as u32).wrapping_mul(0x85EB_CA77);
    h = (h ^ k[2] as u32).wrapping_mul(0xC2B2_AE3D);
    h = (h ^ k[3] as u32).wrapping_mul(0x27D4_EB2F);
    h ^ (h >> 15)
}

pub struct DrawnSet {
    pub list: Vec<DrawnChunk>,
    /// Index in `list` by slot.
    position: Vec<u32>,
    /// Drawn tiles and their slots.
    tiles: FxHashMap<TileKey, Vec<u32>>,
    /// Entries: key then (slot, 0, 0, 0).
    table: Vec<[i32; 4]>,
    mask: u32,
    /// Entries changed since the last upload.
    dirty: Vec<u32>,
}

impl DrawnSet {
    /// `slots` chunk slots, a table of `entries` (a power of two).
    pub fn new(slots: usize, entries: usize) -> Self {
        assert!(entries.is_power_of_two());
        Self {
            list: Vec::new(),
            position: vec![NONE; slots],
            tiles: FxHashMap::default(),
            table: vec![EMPTY; 2 * entries],
            mask: entries as u32 - 1,
            dirty: Vec::new(),
        }
    }

    pub fn mask(&self) -> u32 {
        self.mask
    }

    /// The whole table (its first upload).
    pub fn table(&self) -> &[[i32; 4]] {
        &self.table
    }

    /// Make `drawn` the drawn tiles: tiles that left or changed
    /// (`changed`) drop their chunks; tiles that came or changed add those
    /// `chunks_of` gives.
    pub fn set_tiles(&mut self, drawn: &[TileKey], changed: &rustc_hash::FxHashSet<TileKey>, mut chunks_of: impl FnMut(TileKey) -> Vec<DrawnChunk>) {
        let wanted: rustc_hash::FxHashSet<TileKey> = drawn.iter().copied().collect();
        let leaving: Vec<TileKey> = self.tiles.keys().filter(|k| !wanted.contains(k) || changed.contains(k)).copied().collect();
        for key in leaving {
            for slot in self.tiles.remove(&key).unwrap() {
                self.remove(slot);
            }
        }
        for key in drawn {
            if self.tiles.contains_key(key) {
                continue;
            }
            let chunks = chunks_of(*key);
            let slots = chunks.iter().map(|c| c.slot).collect();
            for c in chunks {
                self.insert(c);
            }
            self.tiles.insert(*key, slots);
        }
    }

    /// Between plans: the drawn tiles among `changed` take their chunks
    /// again.
    pub fn refresh(&mut self, changed: &rustc_hash::FxHashSet<TileKey>, mut chunks_of: impl FnMut(TileKey) -> Vec<DrawnChunk>) {
        for key in changed {
            let Some(slots) = self.tiles.remove(key) else { continue };
            for slot in slots {
                self.remove(slot);
            }
            let chunks = chunks_of(*key);
            let slots = chunks.iter().map(|c| c.slot).collect();
            for c in chunks {
                self.insert(c);
            }
            self.tiles.insert(*key, slots);
        }
    }

    fn insert(&mut self, c: DrawnChunk) {
        self.position[c.slot as usize] = self.list.len() as u32;
        let mut i = key_hash(c.key) & self.mask;
        while self.table[2 * i as usize][0] != -1 {
            i = (i + 1) & self.mask;
        }
        self.table[2 * i as usize] = c.key;
        self.table[2 * i as usize + 1] = [c.slot as i32, 0, 0, 0];
        self.dirty.push(i);
        self.list.push(c);
    }

    fn remove(&mut self, slot: u32) {
        let at = self.position[slot as usize];
        if at == NONE {
            return;
        }
        let c = self.list.swap_remove(at as usize);
        self.position[slot as usize] = NONE;
        if let Some(moved) = self.list.get(at as usize) {
            self.position[moved.slot as usize] = at;
        }
        // Find the entry, then shift the run after it back.
        let mut i = key_hash(c.key) & self.mask;
        while self.table[2 * i as usize] != c.key {
            i = (i + 1) & self.mask;
        }
        let mut j = i;
        loop {
            j = (j + 1) & self.mask;
            let k = self.table[2 * j as usize];
            if k[0] == -1 {
                break;
            }
            let home = key_hash(k) & self.mask;
            // The entry at j may move to i unless its home lies in (i, j].
            let stays = if i <= j { i < home && home <= j } else { i < home || home <= j };
            if !stays {
                self.table[2 * i as usize] = k;
                self.table[2 * i as usize + 1] = self.table[2 * j as usize + 1];
                self.dirty.push(i);
                i = j;
            }
        }
        self.table[2 * i as usize] = EMPTY;
        self.dirty.push(i);
    }

    pub fn contains(&self, slot: u32) -> bool {
        self.position[slot as usize] != NONE
    }

    /// Changed entries as contiguous runs: first entry and its words.
    pub fn take_changes(&mut self) -> Vec<(u32, Vec<[i32; 4]>)> {
        let mut dirty = std::mem::take(&mut self.dirty);
        dirty.sort_unstable();
        dirty.dedup();
        let mut runs: Vec<(u32, Vec<[i32; 4]>)> = Vec::new();
        for i in dirty {
            let entry = [self.table[2 * i as usize], self.table[2 * i as usize + 1]];
            match runs.last_mut() {
                Some((first, words)) if *first + (words.len() / 2) as u32 == i => words.extend(entry),
                _ => runs.push((i, entry.to_vec())),
            }
        }
        runs
    }

    /// The slot holding chunk `key` (as the GPU lookup finds it).
    #[cfg(test)]
    pub fn find(&self, key: [i32; 4]) -> Option<u32> {
        let mut i = key_hash(key) & self.mask;
        loop {
            let k = self.table[2 * i as usize];
            if k[0] == -1 {
                return None;
            }
            if k == key {
                return Some(self.table[2 * i as usize + 1][0] as u32);
            }
            i = (i + 1) & self.mask;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(slot: u32, key: [i32; 4]) -> DrawnChunk {
        DrawnChunk { key, slot, origin: DVec3::ZERO, inv: DMat3::IDENTITY, centre: DVec3::ZERO, radius: 1.0, z: (0.0, 64.0) }
    }

    #[test]
    fn the_table_finds_every_drawn_chunk_after_tiles_come_and_go() {
        // A small table so probe runs collide and wrap.
        let mut set = DrawnSet::new(4096, 1024);
        let tile = |n: i32| TileKey { face: 2, level: 0, cx: n, cy: n / 3 };
        let chunks = |k: TileKey| -> Vec<DrawnChunk> { (0..3).map(|z| chunk((k.cx * 3 + z) as u32, [2, k.cx, k.cy, z])).collect() };
        let mut rng = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for round in 0..200 {
            let drawn: Vec<TileKey> = (0..300).filter(|_| next() % 2 == 0).map(tile).collect();
            let changed: rustc_hash::FxHashSet<TileKey> = drawn.iter().filter(|_| next() % 7 == 0).copied().collect();
            set.set_tiles(&drawn, &changed, chunks);
            set.take_changes();
            assert_eq!(set.list.len(), drawn.len() * 3, "round {round}");
            for k in &drawn {
                for c in chunks(*k) {
                    assert_eq!(set.find(c.key), Some(c.slot), "round {round}");
                    assert_eq!(set.list[set.position[c.slot as usize] as usize].slot, c.slot);
                }
            }
            for n in 0..300 {
                if !drawn.contains(&tile(n)) {
                    assert_eq!(set.find([2, n, n / 3, 0]), None);
                }
            }
        }
    }
}
