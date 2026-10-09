//! Baked edits: what brushes did to each cell, stored sparsely in bricks.
//!
//! A cell's whole edit history collapses exactly into one [`CellEdit`]:
//! unchanged, carved to air, filled with a material, or painted (recoloured
//! if solid). Applying a brush after any of these yields another of them
//! ([`CellEdit::then`]), so baking brushes in order loses nothing, and the
//! result is independent of the terrain under it: the terrain is only read
//! when a cell is sampled ([`CellEdit::apply`]).
//!
//! Base-level bricks (8x8x8 cells) hold the exact baked edits. Each coarser
//! level is built from the one below by majority of the eight child cells
//! ([`CellEdit::majority`]), so accumulated destruction (a hill removed by
//! thousands of small digs) shows at every distance, where a point sample
//! of each brush would omit brushes smaller than the level's cells.
//!
//! Bricks are stored by column (a renderer column's bricks are one lookup)
//! in shards shared copy-on-write: copying a store (what a renderer does to
//! publish an edited world) copies shard pointers, and an edit copies only
//! the shards, columns and bricks it touches. A recorded change keeps the
//! columns it replaced, which is how sealing is undone.
use crate::edits::{BrushOp, FaceBrush};
use crate::grid::{Grid, BRICK};
use rustc_hash::FxHashMap;
use std::sync::Arc;

/// A cell's baked edit (packed: state in bits 0..2, material in 8..16).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CellEdit(pub u16);

impl CellEdit {
    pub const UNCHANGED: Self = Self(0);
    pub const AIR: Self = Self(1);
    const SOLID: u16 = 2;
    const PAINT: u16 = 3;

    pub fn solid(material: u32) -> Self {
        Self(Self::SOLID | ((material as u16 & 0xff) << 8))
    }
    pub fn paint(material: u32) -> Self {
        Self(Self::PAINT | ((material as u16 & 0xff) << 8))
    }
    pub fn state(self) -> u16 {
        self.0 & 3
    }
    pub fn material(self) -> u32 {
        u32::from(self.0 >> 8)
    }
    pub fn is_unchanged(self) -> bool {
        self.state() == 0
    }

    /// This edit followed by a brush `op` with `material`.
    pub fn then(self, op: BrushOp, material: u32) -> Self {
        match op {
            BrushOp::Remove => Self::AIR,
            BrushOp::Add => Self::solid(material),
            BrushOp::Paint => match self.state() {
                0 | Self::PAINT => Self::paint(material),
                Self::SOLID => Self::solid(material),
                _ => Self::AIR,
            },
        }
    }

    /// This edit followed by `later` (composition of baked edits).
    pub fn then_edit(self, later: Self) -> Self {
        match later.state() {
            0 => self,
            1 => Self::AIR,
            Self::SOLID => later,
            _ => self.then(BrushOp::Paint, later.material()),
        }
    }

    /// The canonical `(kind, material)` of a cell after this edit, given the
    /// terrain's (material 0 on a solid cell: the terrain rule).
    pub fn apply(self, kind: u32, material: u32) -> (u32, u32) {
        match self.state() {
            0 => (kind, material),
            1 => (0, 0),
            Self::SOLID => (1, self.material()),
            _ if kind == 1 => (1, self.material()),
            _ => (0, 0),
        }
    }

    /// The edit a coarse cell shows for its eight children: unchanged when
    /// at least half of them are, else the most common state (air before
    /// solid before paint on ties) with its most common material.
    pub fn majority(children: [Self; 8]) -> Self {
        let mut counts = [0u8; 4];
        for c in children {
            counts[c.state() as usize] += 1;
        }
        if counts[0] >= 4 {
            return Self::UNCHANGED;
        }
        let state = (1..4u16).max_by_key(|&s| (counts[s as usize], 4 - s)).expect("three states");
        if state == 1 {
            return Self::AIR;
        }
        let mut best = (0u8, 0u32);
        for c in children.iter().filter(|c| c.state() == state) {
            let n = children.iter().filter(|o| **o == *c).count() as u8;
            if n > best.0 || (n == best.0 && c.material() < best.1) {
                best = (n, c.material());
            }
        }
        Self(state | ((best.1 as u16) << 8))
    }
}

/// Cells per brick.
pub const BRICK_CELLS: usize = (BRICK * BRICK * BRICK) as usize;

/// One 8x8x8 brick of a level's cells, indexed `x + 8 (y + 8 z)` with
/// `(x, y, z)` the cell's `(i, j, k)` modulo 8.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Brick {
    pub cells: [CellEdit; BRICK_CELLS],
}

impl Default for Brick {
    fn default() -> Self {
        Self { cells: [CellEdit::UNCHANGED; BRICK_CELLS] }
    }
}

impl Brick {
    pub fn index(i: i32, j: i32, k: i32) -> usize {
        (i.rem_euclid(BRICK) + BRICK * (j.rem_euclid(BRICK) + BRICK * k.rem_euclid(BRICK))) as usize
    }
    pub fn is_unchanged(&self) -> bool {
        self.cells.iter().all(|c| c.is_unchanged())
    }
    /// The edit every cell holds, if they all hold the same (the inside of
    /// a carved or filled region).
    pub fn uniform(&self) -> Option<CellEdit> {
        let first = self.cells[0];
        self.cells.iter().all(|c| *c == first).then_some(first)
    }
}

/// A brick: face, level and brick coordinates (cell index divided by 8).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BrickKey {
    pub face: u8,
    pub level: u8,
    pub bi: i32,
    pub bj: i32,
    pub bk: i32,
}

impl BrickKey {
    pub fn of_cell(face: u8, level: u32, i: i32, j: i32, k: i32) -> Self {
        Self { face, level: level as u8, bi: i.div_euclid(BRICK), bj: j.div_euclid(BRICK), bk: k.div_euclid(BRICK) }
    }
    pub fn parent(self) -> Self {
        Self { face: self.face, level: self.level + 1, bi: self.bi.div_euclid(2), bj: self.bj.div_euclid(2), bk: self.bk.div_euclid(2) }
    }
    /// The column holding this brick.
    pub fn column(self) -> ColumnKey {
        ColumnKey { face: self.face, level: self.level, ci: self.bi, cj: self.bj }
    }
}

/// A column of bricks: face, level and brick coordinates across the face.
/// At every level a renderer's column is exactly one brick column.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ColumnKey {
    pub face: u8,
    pub level: u8,
    pub ci: i32,
    pub cj: i32,
}

/// A column's stored bricks by height (`bk`), sorted.
pub type Column = Vec<(i32, Arc<Brick>)>;

const SHARDS: usize = 64;

fn shard_of(key: &ColumnKey) -> usize {
    // Neighbouring columns land in different shards.
    let h = (key.ci as u32).wrapping_mul(0x9e37_79b1) ^ (key.cj as u32).wrapping_mul(0x85eb_ca77) ^ (u32::from(key.level) << 3 | u32::from(key.face));
    (h.rotate_left(7) as usize) % SHARDS
}

type Shard = FxHashMap<ColumnKey, Arc<Column>>;

/// The columns a recorded change replaced and what they held before it
/// (`None`: absent). Restoring them undoes the change exactly.
#[derive(Clone, Default)]
pub struct StoreUndo {
    columns: Vec<(ColumnKey, Option<Arc<Column>>)>,
}

impl StoreUndo {
    pub fn columns(&self) -> impl Iterator<Item = ColumnKey> + '_ {
        self.columns.iter().map(|(key, _)| *key)
    }
}

/// Sparse baked edits of every level (see the module doc).
#[derive(Clone)]
pub struct EditStore {
    shards: Vec<Arc<Shard>>,
    /// Bricks stored per level.
    counts: Vec<usize>,
    /// While recording: each changed column's contents before its first
    /// change.
    recording: Option<FxHashMap<ColumnKey, Option<Arc<Column>>>>,
}

impl Default for EditStore {
    fn default() -> Self {
        Self { shards: (0..SHARDS).map(|_| Arc::new(Shard::default())).collect(), counts: Vec::new(), recording: None }
    }
}

impl EditStore {
    pub fn is_empty(&self) -> bool {
        self.counts.iter().all(|&n| n == 0)
    }
    /// Bricks stored at each level.
    pub fn brick_counts(&self) -> &[usize] {
        &self.counts
    }
    /// A column's stored bricks by height, if any.
    pub fn column(&self, key: &ColumnKey) -> Option<&Arc<Column>> {
        self.shards[shard_of(key)].get(key)
    }
    pub fn brick(&self, key: &BrickKey) -> Option<&Arc<Brick>> {
        let column = self.column(&key.column())?;
        column.binary_search_by_key(&key.bk, |(bk, _)| *bk).ok().map(|at| &column[at].1)
    }
    /// The baked edit of a level cell.
    pub fn cell(&self, face: u8, level: u32, i: i32, j: i32, k: i32) -> CellEdit {
        self.brick(&BrickKey::of_cell(face, level, i, j, k)).map_or(CellEdit::UNCHANGED, |b| b.cells[Brick::index(i, j, k)])
    }
    /// Every stored brick (any order).
    pub fn bricks(&self) -> impl Iterator<Item = (BrickKey, &Arc<Brick>)> + '_ {
        self.shards.iter().flat_map(|shard| shard.iter()).flat_map(|(c, column)| {
            column.iter().map(move |(bk, brick)| (BrickKey { face: c.face, level: c.level, bi: c.ci, bj: c.cj, bk: *bk }, brick))
        })
    }

    /// Store a brick (`None` or an unchanged brick removes it).
    pub fn set(&mut self, key: BrickKey, brick: Option<Brick>) {
        let column_key = key.column();
        let shard = Arc::make_mut(&mut self.shards[shard_of(&column_key)]);
        if let Some(recording) = &mut self.recording {
            recording.entry(column_key).or_insert_with(|| shard.get(&column_key).cloned());
        }
        let level = key.level as usize;
        if self.counts.len() <= level {
            self.counts.resize(level + 1, 0);
        }
        let brick = brick.filter(|b| !b.is_unchanged());
        let column = Arc::make_mut(shard.entry(column_key).or_default());
        match (column.binary_search_by_key(&key.bk, |(bk, _)| *bk), brick) {
            (Ok(at), Some(b)) => column[at].1 = Arc::new(b),
            (Err(at), Some(b)) => {
                column.insert(at, (key.bk, Arc::new(b)));
                self.counts[level] += 1;
            }
            (Ok(at), None) => {
                column.remove(at);
                self.counts[level] -= 1;
            }
            (Err(_), None) => {}
        }
        if column.is_empty() {
            shard.remove(&column_key);
        }
    }

    /// Record the columns every change replaces until [`Self::end_record`].
    pub fn begin_record(&mut self) {
        self.recording = Some(FxHashMap::default());
    }
    /// What changed since [`Self::begin_record`], to undo it with
    /// [`Self::restore`].
    pub fn end_record(&mut self) -> StoreUndo {
        let mut columns: Vec<_> = self.recording.take().unwrap_or_default().into_iter().collect();
        columns.sort_unstable_by_key(|(key, _)| *key);
        StoreUndo { columns }
    }
    /// Put back the columns a recorded change replaced.
    pub fn restore(&mut self, undo: StoreUndo) {
        for (key, before) in undo.columns {
            let level = key.level as usize;
            let shard = Arc::make_mut(&mut self.shards[shard_of(&key)]);
            let now = match &before {
                Some(column) => shard.insert(key, Arc::clone(column)),
                None => shard.remove(&key),
            };
            let (now, before) = (now.map_or(0, |c| c.len()), before.map_or(0, |c| c.len()));
            if self.counts.len() <= level {
                self.counts.resize(level + 1, 0);
            }
            self.counts[level] = self.counts[level] + before - now;
        }
    }

    /// Bake a brush resolved on one face into the base level, cell by cell
    /// with the exact containment test generation uses. Returns the base
    /// bricks it changed.
    pub fn bake(&mut self, grid: &Grid, fb: &FaceBrush, op: BrushOp) -> Vec<BrickKey> {
        let face = fb.face();
        let material = fb.material();
        let r = fb.extent_cells();
        let ci = i64::from(fb.center[0]) / 2;
        let cj = i64::from(fb.center[1]) / 2;
        let n = i64::from(grid.cells());
        let clamp = |lo: i64, hi: i64| if grid.is_plane() && grid.shape() == crate::grid::Shape::InfinitePlane { (lo, hi) } else { (lo.max(0), hi.min(n - 1)) };
        let (i0, i1) = clamp(ci - r, ci + r);
        let (j0, j1) = clamp(cj - r, cj + r);
        let (k0, k1) = (i64::from(fb.k_lo) / 2 - 1, i64::from(fb.k_hi) / 2 + 1);
        let mut touched: FxHashMap<BrickKey, Brick> = FxHashMap::default();
        for k in k0..=k1 {
            for j in j0..=j1 {
                for i in i0..=i1 {
                    let (i, j, k) = (i as i32, j as i32, k as i32);
                    let center = [crate::edits::center_half(i, 0), crate::edits::center_half(j, 0), crate::edits::center_half(k, 0)];
                    if !fb.contains(center, || grid.volume_point(face, i, j, k, 0)) {
                        continue;
                    }
                    let key = BrickKey::of_cell(face, 0, i, j, k);
                    let brick = touched.entry(key).or_insert_with(|| self.brick(&key).map_or_else(Brick::default, |b| (**b).clone()));
                    let cell = &mut brick.cells[Brick::index(i, j, k)];
                    *cell = cell.then(op, material);
                }
            }
        }
        let keys: Vec<BrickKey> = touched.keys().copied().collect();
        for (key, brick) in touched {
            self.set(key, Some(brick));
        }
        keys
    }

    /// Apply a brush to the cells already baked (a brush that stays analytic
    /// under the store: see [`crate::planet`]). Returns the bricks changed.
    pub fn apply_over(&mut self, grid: &Grid, fb: &FaceBrush, op: BrushOp) -> Vec<BrickKey> {
        let face = fb.face();
        let material = fb.material();
        let r = fb.extent_cells();
        let b = i64::from(BRICK);
        let ci = i64::from(fb.center[0]) / 2;
        let cj = i64::from(fb.center[1]) / 2;
        let (ci0, ci1) = ((ci - r).div_euclid(b), (ci + r).div_euclid(b));
        let (cj0, cj1) = ((cj - r).div_euclid(b), (cj + r).div_euclid(b));
        let (k0, k1) = (i64::from(fb.k_lo) / 2 - 1, i64::from(fb.k_hi) / 2 + 1);
        let in_range = |c: &ColumnKey| c.face == face && c.level == 0 && (ci0..=ci1).contains(&i64::from(c.ci)) && (cj0..=cj1).contains(&i64::from(c.cj));
        // The stored columns in the brush's rectangle: looked up when the
        // rectangle is smaller than the store, else found by a scan.
        let stored: usize = self.shards.iter().map(|s| s.len()).sum();
        let area = (ci1 - ci0 + 1).saturating_mul(cj1 - cj0 + 1);
        let columns: Vec<ColumnKey> = if (area as u64) < stored as u64 {
            (ci0..=ci1)
                .flat_map(|i| (cj0..=cj1).map(move |j| ColumnKey { face, level: 0, ci: i as i32, cj: j as i32 }))
                .filter(|c| self.column(c).is_some())
                .collect()
        } else {
            self.shards.iter().flat_map(|s| s.keys().copied()).filter(in_range).collect()
        };
        let mut changed = Vec::new();
        for column in columns {
            let bricks: Vec<(i32, Arc<Brick>)> = self
                .column(&column)
                .map(|c| c.iter().filter(|(bk, _)| i64::from(*bk) * b <= k1 && i64::from(*bk) * b + b > k0).cloned().collect())
                .unwrap_or_default();
            for (bk, stored) in bricks {
                let key = BrickKey { face, level: 0, bi: column.ci, bj: column.cj, bk };
                let mut brick = (*stored).clone();
                let mut any = false;
                for z in 0..BRICK {
                    for y in 0..BRICK {
                        for x in 0..BRICK {
                            let (i, j, k) = (key.bi * BRICK + x, key.bj * BRICK + y, key.bk * BRICK + z);
                            let center = [crate::edits::center_half(i, 0), crate::edits::center_half(j, 0), crate::edits::center_half(k, 0)];
                            if fb.contains(center, || grid.volume_point(face, i, j, k, 0)) {
                                let cell = &mut brick.cells[Brick::index(i, j, k)];
                                *cell = cell.then(op, material);
                                any = true;
                            }
                        }
                    }
                }
                if any {
                    self.set(key, Some(brick));
                    changed.push(key);
                }
            }
        }
        changed
    }

    /// Rebuild the coarser levels above changed base bricks, up to `levels`
    /// (exclusive). Returns every brick of every level that changed.
    pub fn rebuild_coarse(&mut self, changed: Vec<BrickKey>, levels: u32) -> Vec<BrickKey> {
        let mut all = changed.clone();
        let mut current = changed;
        for _ in 1..levels {
            let mut parents: Vec<BrickKey> = current.iter().map(|k| k.parent()).collect();
            parents.sort_unstable();
            parents.dedup();
            let mut next = Vec::new();
            for parent in parents {
                let brick = self.coarsen(parent);
                let old = self.brick(&parent).map(|b| (**b).clone()).unwrap_or_default();
                if brick != old {
                    self.set(parent, Some(brick));
                    next.push(parent);
                }
            }
            if next.is_empty() {
                break;
            }
            all.extend(next.iter().copied());
            current = next;
        }
        all
    }

    /// A coarse brick from the eight finer bricks under it.
    fn coarsen(&self, key: BrickKey) -> Brick {
        let mut out = Brick::default();
        let level = u32::from(key.level) - 1;
        for z in 0..BRICK {
            for y in 0..BRICK {
                for x in 0..BRICK {
                    let (i, j, k) = (key.bi * BRICK + x, key.bj * BRICK + y, key.bk * BRICK + z);
                    let mut children = [CellEdit::UNCHANGED; 8];
                    for (c, child) in children.iter_mut().enumerate() {
                        let (dx, dy, dz) = ((c & 1) as i32, ((c >> 1) & 1) as i32, (c >> 2) as i32);
                        *child = self.cell(key.face, level, i * 2 + dx, j * 2 + dy, k * 2 + dz);
                    }
                    out.cells[Brick::index(i, j, k)] = CellEdit::majority(children);
                }
            }
        }
        out
    }

    /// One above the highest baked solid cell of level column (i, j), if
    /// any.
    pub fn solid_top(&self, face: u8, level: u32, i: i32, j: i32) -> Option<i32> {
        let key = ColumnKey { face, level: level as u8, ci: i.div_euclid(BRICK), cj: j.div_euclid(BRICK) };
        self.column(&key)?.iter().rev().find_map(|(bk, brick)| {
            (0..BRICK).rev().find(|&z| brick.cells[Brick::index(i, j, z)].state() == CellEdit::SOLID).map(|z| bk * BRICK + z + 1)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_edits_compose_like_brushes_applied_in_order() {
        use BrushOp::*;
        // Every sequence of up to three brushes over air and solid terrain:
        // the baked edit applied to the terrain equals the brushes replayed.
        let ops = [(Remove, 0), (Add, 7), (Add, 9), (Paint, 4), (Paint, 5)];
        let replay = |seq: &[(BrushOp, u32)], kind: u32| {
            let (mut k, mut m) = (kind, 0);
            for &(op, mat) in seq {
                match op {
                    Remove => (k, m) = (0, 0),
                    Add => (k, m) = (1, mat),
                    Paint => {
                        if k == 1 {
                            m = mat
                        }
                    }
                }
            }
            (k, m)
        };
        let mut seqs: Vec<Vec<(BrushOp, u32)>> = vec![vec![]];
        for _ in 0..3 {
            let mut next = Vec::new();
            for s in &seqs {
                for &o in &ops {
                    let mut t = s.clone();
                    t.push(o);
                    next.push(t);
                }
            }
            seqs.extend(next);
        }
        for seq in &seqs {
            let edit = seq.iter().fold(CellEdit::UNCHANGED, |e, &(op, m)| e.then(op, m));
            for kind in [0, 1] {
                assert_eq!(edit.apply(kind, 0), replay(seq, kind), "{seq:?} over kind {kind}");
            }
            // Splitting the sequence anywhere and composing the halves agrees.
            for cut in 0..=seq.len() {
                let a = seq[..cut].iter().fold(CellEdit::UNCHANGED, |e, &(op, m)| e.then(op, m));
                let b = seq[cut..].iter().fold(CellEdit::UNCHANGED, |e, &(op, m)| e.then(op, m));
                assert_eq!(a.then_edit(b), edit, "{seq:?} cut at {cut}");
            }
        }
    }

    #[test]
    fn coarse_cells_take_the_majority() {
        let u = CellEdit::UNCHANGED;
        let a = CellEdit::AIR;
        let s = CellEdit::solid(3);
        assert_eq!(CellEdit::majority([a, a, a, a, u, u, u, u]), u, "half unchanged stays unchanged");
        assert_eq!(CellEdit::majority([a, a, a, a, a, u, u, u]), a);
        assert_eq!(CellEdit::majority([a, a, a, s, s, s, u, u]), a, "air wins ties");
        assert_eq!(CellEdit::majority([s, s, s, s, s, CellEdit::solid(4), u, u]), s);
    }
}
