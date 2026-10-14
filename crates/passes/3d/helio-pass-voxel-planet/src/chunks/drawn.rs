//! The drawn chunks and their GPU chunk table, kept up to date by the
//! tiles that change: a frame costs what changed, not what is resident.
//!
//! The table holds the drawn chunks (key `(face | level << 8, cx, cy, cz)`,
//! value `(slot, state, 0, 0)`) and the columns of the drawn cut (key
//! `(face | level << 8, cx, cy, COLUMN)`, value `(lo, hi, top, state)`):
//! a drawn tile's chunk layers [lo, hi), and for it and every ancestor the
//! top of its subtree's solid cells (`top`, in that level's cells, rounded
//! up): a max pyramid over the cut. Above `top` a column is air, so a ray
//! crosses it in one step; under a column the drawn tiles cover every point
//! once.
//!
//! A switch between levels fades: for `FADE_FRAMES` both levels are drawn
//! and each pixel shows one, by a dither `r` in [0, 1024) that moves each
//! frame (TSR blends them). A tile's `state` says which (`choice`): 1
//! drawn, 0 an ancestor; else fading since frame `start`,
//! `2 + (start << 2) + incoming + 2 coarse` (start modulo 2^20): an
//! incoming tile shows where r < q, an outgoing one where r >= q, with q
//! rising from 0 to 1024 over the fade, so each point is shown once; where
//! a tile does not show, the other level does (finer under the coarse
//! side, coarser over the fine side). The state holds for the whole fade:
//! the draw works out q from its frame.
//!
//! The table is open addressing with linear probing over a fixed power of
//! two of entries (an empty key's x is -1), mirrored on the CPU; deletion
//! shifts the probe run back, so lookups never meet a tombstone. Only
//! changed entries are uploaded.

use super::TileKey;
use rustc_hash::{FxHashMap, FxHashSet};

/// A drawn chunk, as the draw and the chunk table need it.
#[derive(Clone, Copy, Debug)]
pub struct DrawnChunk {
    pub key: [i32; 4],
    pub slot: u32,
    /// Occupied layers [lo, hi) (the whole chunk until read back).
    pub z: (f64, f64),
}

const NONE: u32 = u32::MAX;
const EMPTY: [i32; 4] = [-1, 0, 0, 0];
/// A column entry's layer key.
pub const COLUMN: i32 = i32::MIN;
/// A gap entry's layer key.
pub const GAP: i32 = i32::MIN + 1;

/// What a drawn tile holds where it has no chunks: chunk layers [lo, hi)
/// of solid rock between its chunks (none: lo >= hi), and whether it is
/// air, not rock, under its lowest chunk (a hollow). Elsewhere a missing
/// layer among its chunks is air and under them is rock.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Gap {
    pub solid: (i32, i32),
    pub below_air: bool,
}

/// A tile as drawn: its chunks and its gap.
pub struct TileDraw {
    pub chunks: Vec<DrawnChunk>,
    pub gap: Gap,
}

impl From<Vec<DrawnChunk>> for TileDraw {
    fn from(chunks: Vec<DrawnChunk>) -> Self {
        Self { chunks, gap: Gap::default() }
    }
}

/// Frames a switch between levels fades over.
pub const FADE_FRAMES: u64 = 16;


/// A column of the drawn cut: its own chunk layers when drawn, the top of
/// its subtree's, its state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Column {
    drawn: Option<(i32, i32)>,
    top: i32,
    state: i32,
}

/// A tile fading in or out from frame `start`, on the coarse side of its
/// switch or the fine side.
#[derive(Clone, Copy, Debug)]
struct Fade {
    start: u64,
    incoming: bool,
    coarse: bool,
}

/// Fade starts are kept modulo this.
const START_MASK: u64 = (1 << 20) - 1;

/// What a pixel with dither `r` does at a tile of `state` at `frame`: 1
/// shows it, 0 looks finer, 2 coarser (`chunk_draw.wgsl`'s `choice`).
#[cfg(test)]
fn choice(state: i32, r: i32, frame: u64) -> i32 {
    if state < 2 {
        return state;
    }
    let f = state - 2;
    let elapsed = frame.wrapping_sub((f >> 2) as u64) & START_MASK;
    let q = (elapsed * 1024 / FADE_FRAMES).min(1024) as i32;
    if (r < q) == (f & 1 != 0) {
        1
    } else if f & 2 != 0 {
        0
    } else {
        2
    }
}

/// No solid cell: a column's top.
const NO_TOP: i32 = -(1 << 26);

fn column_key(tile: TileKey) -> [i32; 4] {
    [i32::from(tile.face) | ((tile.level as i32) << 8), tile.cx, tile.cy, COLUMN]
}

fn gap_key(tile: TileKey) -> [i32; 4] {
    [i32::from(tile.face) | ((tile.level as i32) << 8), tile.cx, tile.cy, GAP]
}

/// The four tiles under `tile` (none at the finest level).
fn children(tile: TileKey) -> Vec<TileKey> {
    if tile.level == 0 {
        return Vec::new();
    }
    [(0, 0), (1, 0), (0, 1), (1, 1)].into_iter().map(|(a, b)| TileKey { face: tile.face, level: tile.level - 1, cx: tile.cx * 2 + a, cy: tile.cy * 2 + b }).collect()
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
    /// Drawn tiles with a gap entry.
    gaps: FxHashSet<TileKey>,
    /// The table's columns: the drawn tiles' and their ancestors'.
    columns: FxHashMap<TileKey, Column>,
    /// Columns whose subtree changed, to settle by level.
    touched: Vec<TileKey>,
    /// Drawn tiles fading in or out.
    fades: FxHashMap<TileKey, Fade>,
    /// Fading tiles strictly under each tile (only tiles with any): a
    /// switch finds the fades under it by descending where they are, not by
    /// scanning them all (a camera flown up and down fades 20 000 tiles).
    fade_below: FxHashMap<TileKey, u32>,
    /// The coarsest level.
    top_level: u32,
    /// Entries: key then value.
    table: Vec<[i32; 4]>,
    mask: u32,
    /// Entries changed since the last upload.
    dirty: Vec<u32>,
    /// Bumped whenever `list` changes (its slots are uploaded then).
    version: u64,
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
            gaps: FxHashSet::default(),
            columns: FxHashMap::default(),
            touched: Vec::new(),
            fades: FxHashMap::default(),
            fade_below: FxHashMap::default(),
            top_level,
            table: vec![EMPTY; 2 * entries],
            mask: entries as u32 - 1,
            dirty: Vec::new(),
            version: 0,
        }
    }

    pub fn mask(&self) -> u32 {
        self.mask
    }

    /// Changes whenever the drawn chunks do.
    pub fn version(&self) -> u64 {
        self.version
    }

    /// The whole table (its first upload).
    pub fn table(&self) -> &[[i32; 4]] {
        &self.table
    }

    /// At `frame`: tiles `removed` stop being drawn, drawn tiles that
    /// `changed` take their chunks again, tiles `added` are drawn with
    /// those `chunks_of` gives. A tile replaced by finer or coarser ones
    /// fades out as they fade in.
    pub fn update<D: Into<TileDraw>>(&mut self, added: &[TileKey], removed: &[TileKey], changed: &rustc_hash::FxHashSet<TileKey>, frame: u64, mut chunks_of: impl FnMut(TileKey) -> D) {
        let ended: Vec<TileKey> = self.fades.iter().filter(|(_, f)| frame >= f.start + FADE_FRAMES).map(|(k, _)| *k).collect();
        for key in ended {
            self.end_fade(key);
        }
        // A switch the plan changes again, or whose tiles change, ends at
        // once (with every switch it overlaps).
        if !self.fades.is_empty() {
            let touching: Vec<TileKey> = added.iter().chain(removed).chain(changed.iter().filter(|k| self.fades.contains_key(k))).copied().collect();
            for key in touching {
                self.snap(key);
            }
        }
        let gone: FxHashSet<TileKey> = removed.iter().filter(|k| self.tiles.contains_key(k)).copied().collect();
        let came: FxHashSet<TileKey> = added.iter().filter(|k| !self.tiles.contains_key(k)).copied().collect();
        let gone_above = self.ancestors_of(&gone);
        let came_above = self.ancestors_of(&came);
        for &key in &gone {
            // Finer tiles came under it, or a coarser one over it.
            let coarse = came_above.contains(&key);
            if coarse || self.ancestors(key).any(|a| came.contains(&a)) {
                self.start_fade(key, Fade { start: frame, incoming: false, coarse });
                self.touched.push(key);
            } else {
                self.drop_tile(key);
            }
        }
        for key in changed {
            if self.tiles.contains_key(key) {
                self.drop_tile(*key);
                self.add_tile(*key, chunks_of(*key).into());
            }
        }
        for &key in &came {
            let coarse = gone_above.contains(&key);
            if coarse || self.ancestors(key).any(|a| gone.contains(&a)) {
                self.start_fade(key, Fade { start: frame, incoming: true, coarse });
            }
            self.add_tile(key, chunks_of(key).into());
        }
        self.settle_columns();
    }

    /// The fade of `key` ends: an outgoing tile leaves.
    fn start_fade(&mut self, key: TileKey, fade: Fade) {
        if self.fades.insert(key, fade).is_none() {
            for a in self.ancestors(key) {
                *self.fade_below.entry(a).or_default() += 1;
            }
        }
    }

    /// `key`'s fade, gone from the fades and the counts over it.
    fn take_fade(&mut self, key: TileKey) -> Option<Fade> {
        let fade = self.fades.remove(&key)?;
        for a in self.ancestors(key) {
            if let Some(n) = self.fade_below.get_mut(&a) {
                *n -= 1;
                if *n == 0 {
                    self.fade_below.remove(&a);
                }
            }
        }
        Some(fade)
    }

    fn end_fade(&mut self, key: TileKey) {
        if self.take_fade(key).is_some_and(|f| !f.incoming) {
            self.drop_tile(key);
        } else {
            self.touched.push(key);
        }
    }

    /// End the fades overlapping `key` now, and those overlapping them.
    fn snap(&mut self, key: TileKey) {
        let mut work = vec![key];
        while let Some(t) = work.pop() {
            // It or its ancestors; its descendants, where there are fades.
            let mut hit: Vec<TileKey> = std::iter::once(t).chain(self.ancestors(t)).filter(|k| self.fades.contains_key(k)).collect();
            if self.fade_below.contains_key(&t) {
                let mut down = vec![t];
                while let Some(d) = down.pop() {
                    for c in children(d) {
                        if self.fades.contains_key(&c) {
                            hit.push(c);
                        }
                        if self.fade_below.contains_key(&c) {
                            down.push(c);
                        }
                    }
                }
            }
            for k in hit {
                self.end_fade(k);
                work.push(k);
            }
        }
    }

    fn ancestors(&self, key: TileKey) -> impl Iterator<Item = TileKey> + use<> {
        let top = self.top_level;
        std::iter::successors(Some(key), move |k| (k.level < top).then(|| parent(*k))).skip(1)
    }

    /// Every strict ancestor of `tiles`.
    fn ancestors_of(&self, tiles: &FxHashSet<TileKey>) -> FxHashSet<TileKey> {
        let mut out = FxHashSet::default();
        for &key in tiles {
            for a in self.ancestors(key) {
                if !out.insert(a) {
                    break;
                }
            }
        }
        out
    }

    /// A drawn tile's state.
    fn state_of(&self, key: TileKey) -> i32 {
        self.fades.get(&key).map_or(1, |f| 2 + (((f.start & START_MASK) as i32) << 2) + i32::from(f.incoming) + 2 * i32::from(f.coarse))
    }

    /// Whether `tile` is drawn.
    pub fn holds(&self, tile: TileKey) -> bool {
        self.tiles.contains_key(&tile)
    }

    /// `tile`'s drawn state (diagnostics): drawn, its column's state (1
    /// drawn, 0 an ancestor, else fading) and its fade (start, incoming,
    /// coarse side).
    pub fn describe(&self, tile: TileKey) -> (bool, Option<i32>, Option<(u64, bool, bool)>) {
        (self.tiles.contains_key(&tile), self.columns.get(&tile).map(|c| c.state), self.fades.get(&tile).map(|f| (f.start, f.incoming, f.coarse)))
    }

    /// Tiles drawn.
    pub fn tile_count(&self) -> usize {
        self.tiles.len()
    }

    /// Drawn tiles fading in or out.
    pub fn fading(&self) -> usize {
        self.fades.len()
    }

    fn drop_tile(&mut self, key: TileKey) {
        for slot in self.tiles.remove(&key).unwrap() {
            self.remove(slot);
        }
        self.layers.remove(&key);
        if self.gaps.remove(&key) {
            self.take(gap_key(key));
        }
        self.touched.push(key);
    }

    fn add_tile(&mut self, key: TileKey, draw: TileDraw) {
        let TileDraw { chunks, gap } = draw;
        if gap.below_air || gap.solid.0 < gap.solid.1 {
            self.put(gap_key(key), [gap.solid.0, gap.solid.1, i32::from(gap.below_air), 0]);
            self.gaps.insert(key);
        }
        let lo = chunks.iter().map(|c| c.key[3]).min();
        let hi = chunks.iter().map(|c| c.key[3] + 1).max();
        let top = chunks.iter().map(|c| c.key[3] * 64 + c.z.1.ceil() as i32).max().unwrap_or(NO_TOP);
        // No chunks: air (all its layers were empty).
        let (lo, hi) = lo.zip(hi).unwrap_or((NO_TOP >> 6, NO_TOP >> 6));
        let layers = (lo, hi, top);
        let slots = chunks.iter().map(|c| c.slot).collect();
        let state = self.state_of(key);
        for c in chunks {
            self.insert(c, state);
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
                    (d, b) => Some(Column {
                        drawn: d.map(|d| (d.0, d.1)),
                        top: d.map(|d| d.2).into_iter().chain(b).max().unwrap(),
                        state: if d.is_some() { self.state_of(key) } else { 0 },
                    }),
                };
                if new == old {
                    continue;
                }
                match new {
                    Some(c) => {
                        self.columns.insert(key, c);
                        let (lo, hi) = c.drawn.unwrap_or((0, 0));
                        self.put(column_key(key), [lo, hi, c.top, c.state]);
                        // Its chunks carry its state.
                        if c.drawn.is_some() && old.is_none_or(|o| o.state != c.state) {
                            for i in 0..self.tiles[&key].len() {
                                let slot = self.tiles[&key][i];
                                let chunk = self.list[self.position[slot as usize] as usize].key;
                                self.put(chunk, [slot as i32, c.state, 0, 0]);
                            }
                        }
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

    fn insert(&mut self, c: DrawnChunk, state: i32) {
        self.version += 1;
        self.position[c.slot as usize] = self.list.len() as u32;
        self.put(c.key, [c.slot as i32, state, 0, 0]);
        self.list.push(c);
    }

    fn remove(&mut self, slot: u32) {
        let at = self.position[slot as usize];
        if at == NONE {
            return;
        }
        self.version += 1;
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

    /// Random cuts of one level-3 tile, a new one every few frames: at every
    /// frame, for every dither, the tiles shown cover each level-0 column
    /// once, and each one's chunks carry its state.
    #[test]
    fn fades_show_every_column_once() {
        let mut set = DrawnSet::new(1 << 14, 1 << 12, 3);
        let mut rng = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let root = TileKey { face: 0, level: 3, cx: 0, cy: 0 };
        let mut last: Vec<TileKey> = Vec::new();
        let mut slot = 0u32;
        let mut frame = 0;
        for round in 0..300 {
            let mut drawn = Vec::new();
            let mut open = vec![root];
            while let Some(k) = open.pop() {
                if k.level > 0 && next() % 3 != 0 {
                    open.extend([(0, 0), (1, 0), (0, 1), (1, 1)].map(|(a, b)| TileKey { face: 0, level: k.level - 1, cx: k.cx * 2 + a, cy: k.cy * 2 + b }));
                } else {
                    drawn.push(k);
                }
            }
            set.update(&diff(&drawn, &last), &diff(&last, &drawn), &Default::default(), frame, |k| {
                slot = (slot + 1) % (1 << 14);
                vec![chunk(slot, [i32::from(k.face) | ((k.level as i32) << 8), k.cx, k.cy, 0])]
            });
            last = drawn;
            for _ in 0..(next() % 20) {
                frame += 1;
                set.update(&[], &[], &Default::default(), frame, |_| Vec::new());
                for r in [0, 1, 255, 511, 512, 900, 1023] {
                    let mut shown = [[0u8; 8]; 8];
                    for (k, _) in &set.tiles {
                        let state = set.find(column_key(*k)).unwrap()[3];
                        for &s in &set.tiles[k] {
                            let c = set.list[set.position[s as usize] as usize].key;
                            assert_eq!(set.find(c).unwrap()[1], state, "round {round}");
                        }
                        if choice(state, r, frame) == 1 {
                            let n = 1 << k.level;
                            for x in 0..n {
                                for y in 0..n {
                                    shown[(k.cx * n + x) as usize][(k.cy * n + y) as usize] += 1;
                                }
                            }
                        }
                    }
                    assert!(shown.iter().flatten().all(|&n| n == 1), "round {round} frame {frame} r {r}: {shown:?}");
                }
            }
        }
    }

    /// The draw's search for the tile a pixel of dither `r` shows over
    /// finest-level tile `(x, y)`, from a ray at `level` (`place_at`'s
    /// column search: coarser while a column says so, finer while it says
    /// so; down through undrawn ancestors only, it turns up once).
    fn search(set: &DrawnSet, x: i32, y: i32, level: u32, r: i32, frame: u64) -> Option<TileKey> {
        let mut diff = 0i32;
        let mut way = 0;
        let (mut turned, mut through_ancestors) = (false, true);
        for _ in 0..24 {
            let lv = (level as i32 + diff) as u32;
            let k = TileKey { face: 0, level: lv, cx: x >> lv, cy: y >> lv };
            let state = set.find(column_key(k)).map(|c| c[3]);
            let go = state.map_or(2, |s| choice(s, r, frame));
            if go == 1 {
                return Some(k);
            }
            if go == 2 || (go == 0 && way > 0) {
                if way < 0 {
                    if turned || !through_ancestors {
                        return None;
                    }
                    turned = true;
                    diff = 0;
                }
                if level as i32 + diff >= set.top_level as i32 {
                    return None;
                }
                way = 1;
                diff += 1;
            } else {
                if way > 0 || lv == 0 {
                    return None;
                }
                through_ancestors &= state == Some(0);
                way = -1;
                diff -= 1;
            }
        }
        None
    }

    #[test]
    fn every_ray_finds_the_tile_it_shows_through_jumps() {
        // Cuts of a level 8 root whose depth follows an eye that jumps:
        // tiles within `reach` (in finest tiles) of `eye` refine to `finest`.
        let top = 8u32;
        let cut = |eye: (i32, i32), finest: u32| -> Vec<TileKey> {
            let mut drawn = Vec::new();
            let mut open = vec![TileKey { face: 0, level: top, cx: 0, cy: 0 }];
            while let Some(k) = open.pop() {
                let n = 1i32 << k.level;
                let (x0, y0) = (k.cx * n, k.cy * n);
                let dx = (eye.0 - (x0 + n / 2)).abs().saturating_sub(n / 2);
                let dy = (eye.1 - (y0 + n / 2)).abs().saturating_sub(n / 2);
                let want = finest.max(((dx.max(dy) as f64 / 4.0).max(1.0).log2()) as u32);
                if k.level > want {
                    open.extend([(0, 0), (1, 0), (0, 1), (1, 1)].map(|(a, b)| TileKey { face: 0, level: k.level - 1, cx: k.cx * 2 + a, cy: k.cy * 2 + b }));
                } else {
                    drawn.push(k);
                }
            }
            drawn
        };
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut set = DrawnSet::new(1 << 16, 1 << 16, top);
        let mut slot = 0u32;
        let mut last: Vec<TileKey> = Vec::new();
        let mut frame = 0;
        let side = 1i32 << top;
        for round in 0..60 {
            // Mostly small moves; now and then a jump (finest level 0 to 5).
            let eye = ((next() % side as u64) as i32, (next() % side as u64) as i32);
            let finest = if next() % 4 == 0 { (next() % 6) as u32 } else { 0 };
            let drawn = cut(eye, finest);
            set.update(&diff(&drawn, &last), &diff(&last, &drawn), &Default::default(), frame, |k| {
                slot = (slot + 1) % (1 << 16);
                vec![chunk(slot, [i32::from(k.face) | ((k.level as i32) << 8), k.cx, k.cy, 0])]
            });
            last = drawn;
            for _ in 0..(1 + next() % 6) {
                frame += 1;
                set.update(&[], &[], &Default::default(), frame, |_| Vec::new());
                for r in [0, 300, 700, 1023] {
                    for _ in 0..400 {
                        let (x, y) = ((next() % side as u64) as i32, (next() % side as u64) as i32);
                        let shown: Vec<TileKey> = set.tiles.keys().filter(|k| k.cx == x >> k.level && k.cy == y >> k.level && choice(set.find(column_key(**k)).unwrap()[3], r, frame) == 1).copied().collect();
                        assert_eq!(shown.len(), 1, "round {round}: {shown:?}");
                        for level in 0..=top {
                            assert_eq!(search(&set, x, y, level, r, frame), Some(shown[0]), "round {round} frame {frame} r {r} at ({x}, {y}) from level {level}");
                        }
                    }
                }
            }
        }
    }

    /// Tiles of `a` not in `b`.
    fn diff(a: &[TileKey], b: &[TileKey]) -> Vec<TileKey> {
        a.iter().filter(|k| !b.contains(k)).copied().collect()
    }

    fn chunk(slot: u32, key: [i32; 4]) -> DrawnChunk {
        DrawnChunk { key, slot, z: (0.0, 64.0) }
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
            set.update(&diff(&drawn, &last), &diff(&last, &drawn), &changed, round, chunks);
            last = drawn.clone();
            set.take_changes();
            assert_eq!(set.list.len(), drawn.len() * 3, "round {round}");
            for k in &drawn {
                for c in chunks(*k) {
                    assert_eq!(set.find(c.key), Some([c.slot as i32, 1, 0, 0]), "round {round}");
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
            // Frames FADE_FRAMES apart: no fade under way.
            set.update(&diff(&drawn, &last), &diff(&last, &drawn), &changed, round * FADE_FRAMES, |k| {
                (0..layers[&k])
                    .map(|z| {
                        slot = (slot + 1) % (1 << 14);
                        chunk(slot, [1 | ((k.level as i32) << 8), k.cx, k.cy, z])
                    })
                    .collect::<Vec<_>>()
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
            set.update(&[], &[], &Default::default(), (round + 1) * FADE_FRAMES - 1, |_| Vec::new());
            set.update(&[], &[], &Default::default(), (round + 1) * FADE_FRAMES, |_| Vec::new());
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
