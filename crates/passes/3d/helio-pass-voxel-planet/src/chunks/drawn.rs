//! The drawn chunks and their GPU chunk table, kept up to date by the
//! tiles that change: a frame costs what changed, not what is resident.
//!
//! The table holds the drawn chunks (key `(face | level << 8, cx, cy, cz)`,
//! value `(slot, 0, 0, 0)`) and the columns of the drawn cut (key
//! `(face | level << 8, cx, cy, COLUMN)`, value `(lo, hi, top, drawn)`):
//! a drawn tile's chunk layers [lo, hi), and for it and every ancestor the
//! top of its subtree's solid cells (`top`, in that level's cells, rounded
//! up): a max pyramid over the cut. Above `top` a column is air, so a ray
//! crosses it in one step; under a column the drawn tiles cover every point
//! once.
//!
//! The table is open addressing with linear probing over a fixed power of
//! two of entries (an empty key's x is -1), mirrored on the CPU; deletion
//! shifts the probe run back, so lookups never meet a tombstone. Only
//! changed entries are uploaded.

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
/// A column entry's layer key.
pub const COLUMN: i32 = i32::MIN;

/// A column of the drawn cut: its own chunk layers when drawn, and the
/// top of its subtree's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Column {
    drawn: Option<(i32, i32)>,
    top: i32,
}

/// No solid cell: a column's top.
const NO_TOP: i32 = -(1 << 26);

fn column_key(tile: TileKey) -> [i32; 4] {
    [i32::from(tile.face) | ((tile.level as i32) << 8), tile.cx, tile.cy, COLUMN]
}

fn parent(tile: TileKey) -> TileKey {
    TileKey { face: tile.face, level: tile.level + 1, cx: tile.cx >> 1, cy: tile.cy >> 1 }
}

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
    /// Drawn tiles' chunk layers [lo, hi) and top cell.
    layers: FxHashMap<TileKey, (i32, i32, i32)>,
    /// The table's columns: the drawn tiles' and their ancestors'.
    columns: FxHashMap<TileKey, Column>,
    /// Columns whose subtree changed, to settle by level.
    touched: Vec<TileKey>,
    /// The coarsest level.
    top_level: u32,
    /// Entries: key then value.
    table: Vec<[i32; 4]>,
    mask: u32,
    /// Entries changed since the last upload.
    dirty: Vec<u32>,
}

impl DrawnSet {
    /// `slots` chunk slots, a table of `entries` (a power of two), tiles up
    /// to `top_level`.
    pub fn new(slots: usize, entries: usize, top_level: u32) -> Self {
        assert!(entries.is_power_of_two());
        Self {
            list: Vec::new(),
            position: vec![NONE; slots],
            tiles: FxHashMap::default(),
            layers: FxHashMap::default(),
            columns: FxHashMap::default(),
            touched: Vec::new(),
            top_level,
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

    /// Tiles `removed` stop being drawn, drawn tiles that `changed` take
    /// their chunks again, tiles `added` are drawn with those `chunks_of`
    /// gives.
    pub fn update(&mut self, added: &[TileKey], removed: &[TileKey], changed: &rustc_hash::FxHashSet<TileKey>, mut chunks_of: impl FnMut(TileKey) -> Vec<DrawnChunk>) {
        for key in removed {
            if self.tiles.contains_key(key) {
                self.drop_tile(*key);
            }
        }
        for key in changed {
            if self.tiles.contains_key(key) {
                self.drop_tile(*key);
                self.add_tile(*key, chunks_of(*key));
            }
        }
        for key in added {
            if !self.tiles.contains_key(key) {
                self.add_tile(*key, chunks_of(*key));
            }
        }
        self.settle_columns();
    }

    /// Tiles drawn.
    pub fn tile_count(&self) -> usize {
        self.tiles.len()
    }

    fn drop_tile(&mut self, key: TileKey) {
        for slot in self.tiles.remove(&key).unwrap() {
            self.remove(slot);
        }
        self.layers.remove(&key);
        self.touched.push(key);
    }

    fn add_tile(&mut self, key: TileKey, chunks: Vec<DrawnChunk>) {
        let lo = chunks.iter().map(|c| c.key[3]).min();
        let hi = chunks.iter().map(|c| c.key[3] + 1).max();
        let top = chunks.iter().map(|c| c.key[3] * 64 + c.z.1.ceil() as i32).max().unwrap_or(NO_TOP);
        // No chunks: air (all its layers were empty).
        let (lo, hi) = lo.zip(hi).unwrap_or((NO_TOP >> 6, NO_TOP >> 6));
        let layers = (lo, hi, top);
        let slots = chunks.iter().map(|c| c.slot).collect();
        for c in chunks {
            self.insert(c);
        }
        self.tiles.insert(key, slots);
        self.layers.insert(key, layers);
        self.touched.push(key);
    }

    /// Recompute the touched columns, finest first, and their ancestors
    /// while they change; a column with nothing drawn in it leaves.
    fn settle_columns(&mut self) {
        let mut by_level: Vec<rustc_hash::FxHashSet<TileKey>> = vec![Default::default(); self.top_level as usize + 1];
        for key in std::mem::take(&mut self.touched) {
            by_level[key.level as usize].insert(key);
        }
        for level in 0..by_level.len() {
            for key in std::mem::take(&mut by_level[level]) {
                let old = self.columns.get(&key).copied();
                let drawn = self.layers.get(&key).copied();
                // The children's tops in this level's layers (rounded up).
                let below = if level == 0 {
                    None
                } else {
                    [(0, 0), (1, 0), (0, 1), (1, 1)]
                        .into_iter()
                        .filter_map(|(a, b)| self.columns.get(&TileKey { face: key.face, level: key.level - 1, cx: key.cx * 2 + a, cy: key.cy * 2 + b }))
                        .map(|c| (c.top + 1) >> 1)
                        .max()
                };
                let new = match (drawn, below) {
                    (None, None) => None,
                    (d, b) => Some(Column { drawn: d.map(|d| (d.0, d.1)), top: d.map(|d| d.2).into_iter().chain(b).max().unwrap() }),
                };
                if new == old {
                    continue;
                }
                match new {
                    Some(c) => {
                        self.columns.insert(key, c);
                        let (lo, hi) = c.drawn.unwrap_or((0, 0));
                        self.put(column_key(key), [lo, hi, c.top, i32::from(c.drawn.is_some())]);
                    }
                    None => {
                        self.columns.remove(&key);
                        self.take(column_key(key));
                    }
                }
                if key.level < self.top_level {
                    by_level[level + 1].insert(parent(key));
                }
            }
        }
    }

    fn insert(&mut self, c: DrawnChunk) {
        self.position[c.slot as usize] = self.list.len() as u32;
        self.put(c.key, [c.slot as i32, 0, 0, 0]);
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
        self.take(c.key);
    }

    /// Set entry `key`'s value, adding the entry if absent.
    fn put(&mut self, key: [i32; 4], value: [i32; 4]) {
        let mut i = key_hash(key) & self.mask;
        while self.table[2 * i as usize][0] != -1 && self.table[2 * i as usize] != key {
            i = (i + 1) & self.mask;
        }
        self.table[2 * i as usize] = key;
        self.table[2 * i as usize + 1] = value;
        self.dirty.push(i);
    }

    /// Remove entry `key` (present).
    fn take(&mut self, key: [i32; 4]) {
        // Find the entry, then shift the run after it back.
        let mut i = key_hash(key) & self.mask;
        while self.table[2 * i as usize] != key {
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

    /// The entries changed since the last call, each as its index, key
    /// and value (`chunk_table.wgsl`).
    pub fn take_changes(&mut self) -> Vec<[i32; 4]> {
        let mut dirty = std::mem::take(&mut self.dirty);
        dirty.sort_unstable();
        dirty.dedup();
        dirty.into_iter().flat_map(|i| [[i as i32, 0, 0, 0], self.table[2 * i as usize], self.table[2 * i as usize + 1]]).collect()
    }

    /// Entry `key`'s value (as the GPU lookup finds it).
    #[cfg(test)]
    fn find(&self, key: [i32; 4]) -> Option<[i32; 4]> {
        let mut i = key_hash(key) & self.mask;
        loop {
            let k = self.table[2 * i as usize];
            if k[0] == -1 {
                return None;
            }
            if k == key {
                return Some(self.table[2 * i as usize + 1]);
            }
            i = (i + 1) & self.mask;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiles of `a` not in `b`.
    fn diff(a: &[TileKey], b: &[TileKey]) -> Vec<TileKey> {
        a.iter().filter(|k| !b.contains(k)).copied().collect()
    }

    fn chunk(slot: u32, key: [i32; 4]) -> DrawnChunk {
        DrawnChunk { key, slot, origin: DVec3::ZERO, inv: DMat3::IDENTITY, centre: DVec3::ZERO, radius: 1.0, z: (0.0, 64.0) }
    }

    #[test]
    fn the_table_finds_every_drawn_chunk_after_tiles_come_and_go() {
        // A small table so probe runs collide and wrap.
        let mut set = DrawnSet::new(4096, 1024, 0);
        let tile = |n: i32| TileKey { face: 2, level: 0, cx: n, cy: n / 3 };
        let chunks = |k: TileKey| -> Vec<DrawnChunk> { (0..3).map(|z| chunk((k.cx * 3 + z) as u32, [2, k.cx, k.cy, z])).collect() };
        let mut rng = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut last: Vec<TileKey> = Vec::new();
        for round in 0..200 {
            let drawn: Vec<TileKey> = (0..300).filter(|_| next() % 2 == 0).map(tile).collect();
            let changed: rustc_hash::FxHashSet<TileKey> = drawn.iter().filter(|_| next() % 7 == 0).copied().collect();
            set.update(&diff(&drawn, &last), &diff(&last, &drawn), &changed, chunks);
            last = drawn.clone();
            set.take_changes();
            assert_eq!(set.list.len(), drawn.len() * 3, "round {round}");
            for k in &drawn {
                for c in chunks(*k) {
                    assert_eq!(set.find(c.key), Some([c.slot as i32, 0, 0, 0]), "round {round}");
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

    #[test]
    fn columns_hold_each_subtree_top_and_leave_with_it() {
        // Tiles of levels 0..3 in one level-3 column; a tile's chunks are
        // layers [0, n) for a random n.
        let mut set = DrawnSet::new(1 << 14, 1 << 12, 3);
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut slot = 0u32;
        let mut last: Vec<TileKey> = Vec::new();
        for round in 0..100 {
            // A random cut: refine each tile with probability 1/2.
            let mut drawn = Vec::new();
            let mut open = vec![TileKey { face: 1, level: 3, cx: 0, cy: 0 }];
            while let Some(k) = open.pop() {
                if k.level > 0 && next() % 2 == 0 {
                    open.extend([(0, 0), (1, 0), (0, 1), (1, 1)].map(|(a, b)| TileKey { face: 1, level: k.level - 1, cx: k.cx * 2 + a, cy: k.cy * 2 + b }));
                } else {
                    drawn.push(k);
                }
            }
            let layers: FxHashMap<TileKey, i32> = drawn.iter().map(|k| (*k, (next() % 5) as i32 + 1)).collect();
            let changed: rustc_hash::FxHashSet<TileKey> = drawn.iter().copied().collect();
            set.update(&diff(&drawn, &last), &diff(&last, &drawn), &changed, |k| {
                (0..layers[&k])
                    .map(|z| {
                        slot = (slot + 1) % (1 << 14);
                        chunk(slot, [1 | ((k.level as i32) << 8), k.cx, k.cy, z])
                    })
                    .collect()
            });
            last = drawn.clone();
            // Expected tops, finest first.
            let mut top: FxHashMap<TileKey, i32> = FxHashMap::default();
            for level in 0..=3u32 {
                for k in drawn.iter().filter(|k| k.level == level) {
                    top.insert(*k, layers[k] * 64);
                }
                let at: Vec<(TileKey, i32)> = top.iter().filter(|(k, _)| k.level == level).map(|(k, t)| (*k, *t)).collect();
                for (k, t) in at {
                    if level < 3 {
                        let p = parent(k);
                        let up = (t + 1) >> 1;
                        top.entry(p).and_modify(|v| *v = (*v).max(up)).or_insert(up);
                    }
                }
            }
            for (k, t) in &top {
                let entry = set.find(column_key(*k)).unwrap_or_else(|| panic!("round {round}: {k:?} missing"));
                let drawn_here = layers.contains_key(k);
                assert_eq!(entry[2], *t, "round {round}: {k:?}");
                assert_eq!(entry[3], i32::from(drawn_here), "round {round}: {k:?}");
                if drawn_here {
                    assert_eq!((entry[0], entry[1]), (0, layers[k]));
                }
            }
            // No column outside the cut's subtrees.
            assert_eq!(set.columns.len(), top.len(), "round {round}");
        }
    }
}
