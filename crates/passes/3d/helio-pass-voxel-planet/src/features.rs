//! Features: procedural content standing on the terrain (trees, rocks,
//! buildings), loaded around a viewer and dropped behind it.
//!
//! A host decides what stands in a region of the world (from the seed and
//! the terrain alone, so a region comes out the same whenever it is
//! decided) and hands the planet that region's brushes
//! ([`crate::Planet::load_features`]). They are not edits: the history,
//! undo and saves never hold them, and a region unloaded far from every
//! viewer gives its memory back. They read as part of the world, under
//! every edit: what a player digs or builds applies over them, whenever the
//! region was loaded.
//!
//! Each region bakes its brushes into a store of its own (base cells, as
//! edits bake), with the large analytic brushes applied over it as they
//! apply over the terrain. A column's features are the loaded regions'
//! bricks composed in key order; the planet composes them under its baked
//! edits (`planet::Edits`).

use crate::edit_store::{Brick, BrickKey, ColumnKey, EditStore};
use crate::edits::{Brush, EditLog};
use crate::grid::Grid;
use rustc_hash::FxHashMap;
use std::sync::Arc;

/// A square region of a face, `size` base columns a side, by index:
/// what a host decides features by ([`Region::key`] is the planet's key).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Region {
    pub face: u8,
    pub x: i32,
    pub y: i32,
}

impl Region {
    /// The region of `size` columns holding `cell`.
    pub fn of(cell: crate::Cell, size: i32) -> Self {
        Self { face: cell.face, x: cell.i.div_euclid(size), y: cell.j.div_euclid(size) }
    }
    /// Its key for [`crate::Planet::load_features`] (indices within 2^27).
    pub fn key(self) -> u64 {
        (u64::from(self.face) << 56) | (u64::from(self.x as u32 & 0x0fff_ffff) << 28) | u64::from(self.y as u32 & 0x0fff_ffff)
    }
    pub fn from_key(key: u64) -> Self {
        let index = |v: u64| ((v as u32) << 4) as i32 >> 4;
        Self { face: (key >> 56) as u8, x: index(key >> 28), y: index(key) }
    }
    /// The regions of its face within `radius` regions of it (by centre
    /// distance), nearest first, within the face's `regions` a side.
    pub fn around(self, radius: i32, regions: i32) -> Vec<Region> {
        let mut out: Vec<(i64, Region)> = Vec::new();
        for dx in -radius..=radius {
            for dy in -radius..=radius {
                let (x, y) = (self.x + dx, self.y + dy);
                let d = i64::from(dx * dx + dy * dy);
                if d <= i64::from(radius * radius) && (0..regions).contains(&x) && (0..regions).contains(&y) {
                    out.push((d, Region { face: self.face, x, y }));
                }
            }
        }
        out.sort_by_key(|(d, r)| (*d, r.x, r.y));
        out.into_iter().map(|(_, r)| r).collect()
    }
}

/// A loaded region: its brushes and what they baked.
#[derive(Clone)]
struct Loaded {
    brushes: Arc<[Brush]>,
    store: EditStore,
}

/// The loaded regions (see the module doc).
#[derive(Clone, Default)]
pub struct Features {
    regions: FxHashMap<u64, Loaded>,
    /// Each base column's regions, by key.
    columns: FxHashMap<ColumnKey, Vec<u64>>,
    /// Columns at every level over a region's base columns: how many base
    /// columns with features lie under each.
    covered: FxHashMap<ColumnKey, u32>,
    /// Bumped by every load and unload.
    version: u64,
}

impl Features {
    pub fn is_empty(&self) -> bool {
        self.regions.is_empty()
    }
    /// Regions loaded.
    pub fn len(&self) -> usize {
        self.regions.len()
    }
    /// The loaded regions' keys.
    pub fn keys(&self) -> impl Iterator<Item = u64> + '_ {
        self.regions.keys().copied()
    }
    pub fn contains(&self, key: u64) -> bool {
        self.regions.contains_key(&key)
    }
    /// Changes with every load and unload.
    pub fn version(&self) -> u64 {
        self.version
    }
    /// Base bricks loaded, over every region.
    pub fn bricks(&self) -> usize {
        self.regions.values().map(|r| r.store.brick_counts().first().copied().unwrap_or(0)).sum()
    }

    /// Whether any feature lies under column `key` (of any level).
    pub(crate) fn covers(&self, key: &ColumnKey) -> bool {
        self.covered.contains_key(key)
    }

    /// Base column `key`'s features: the regions' bricks composed in key
    /// order (`None`: no region holds it).
    pub(crate) fn column(&self, key: &ColumnKey) -> Option<Vec<(i32, Brick)>> {
        let regions = self.columns.get(key)?;
        let mut out: Vec<(i32, Brick)> = Vec::new();
        for id in regions {
            let Some(column) = self.regions[id].store.column(key) else { continue };
            for (bk, brick) in column.iter() {
                match out.binary_search_by_key(bk, |(b, _)| *b) {
                    Ok(at) => {
                        let under = &mut out[at].1;
                        for (cell, over) in under.cells.iter_mut().zip(brick.cells.iter()) {
                            *cell = cell.then_edit(*over);
                        }
                    }
                    Err(at) => out.insert(at, (*bk, (**brick).clone())),
                }
            }
        }
        Some(out)
    }

    /// Loads region `key` with `brushes` (replacing what it held), the large
    /// brushes `large` applied over them. The base columns it changed.
    pub(crate) fn load(&mut self, grid: &Grid, key: u64, brushes: Arc<[Brush]>, large: &EditLog) -> Result<Vec<ColumnKey>, String> {
        let mut store = EditStore::default();
        for brush in brushes.iter() {
            for fb in brush.resolve(grid)? {
                store.bake(grid, &fb, brush.op);
            }
        }
        for brush in large.brushes() {
            for fb in brush.resolve(grid)? {
                store.apply_over(grid, &fb, brush.op);
            }
        }
        let mut changed = self.unload(grid, key);
        let columns: Vec<ColumnKey> = store.columns().filter(|c| c.level == 0).collect();
        for column in &columns {
            let regions = self.columns.entry(*column).or_default();
            let at = regions.binary_search(&key).unwrap_or_else(|at| at);
            regions.insert(at, key);
            for up in ancestors(grid, *column) {
                *self.covered.entry(up).or_default() += 1;
            }
        }
        changed.extend(columns);
        changed.sort_unstable();
        changed.dedup();
        self.regions.insert(key, Loaded { brushes, store });
        self.version += 1;
        Ok(changed)
    }

    /// Unloads region `key`: the base columns it held.
    pub(crate) fn unload(&mut self, grid: &Grid, key: u64) -> Vec<ColumnKey> {
        let Some(region) = self.regions.remove(&key) else { return Vec::new() };
        let columns: Vec<ColumnKey> = region.store.columns().filter(|c| c.level == 0).collect();
        for column in &columns {
            if let Some(regions) = self.columns.get_mut(column) {
                regions.retain(|r| *r != key);
                if regions.is_empty() {
                    self.columns.remove(column);
                }
            }
            for up in ancestors(grid, *column) {
                if let Some(n) = self.covered.get_mut(&up) {
                    *n -= 1;
                    if *n == 0 {
                        self.covered.remove(&up);
                    }
                }
            }
        }
        self.version += 1;
        columns
    }

    /// A large brush sealed after the regions loaded: applied over them as
    /// over the terrain. The base columns it changed.
    pub(crate) fn apply_large(&mut self, grid: &Grid, brush: &Brush) -> Result<Vec<ColumnKey>, String> {
        let faces = brush.resolve(grid)?;
        let mut changed = Vec::new();
        for region in self.regions.values_mut() {
            for fb in &faces {
                changed.extend(region.store.apply_over(grid, fb, brush.op).into_iter().map(BrickKey::column));
            }
        }
        changed.sort_unstable();
        changed.dedup();
        Ok(changed)
    }

    /// Every region baked again under the large brushes `large` (after an
    /// undo changed them). The base columns they held before and after.
    pub(crate) fn rebake(&mut self, grid: &Grid, large: &EditLog) -> Result<Vec<ColumnKey>, String> {
        let regions: Vec<(u64, Arc<[Brush]>)> = self.regions.iter().map(|(k, r)| (*k, Arc::clone(&r.brushes))).collect();
        let mut changed = Vec::new();
        for (key, brushes) in regions {
            changed.extend(self.load(grid, key, brushes, large)?);
        }
        changed.sort_unstable();
        changed.dedup();
        Ok(changed)
    }
}

/// Base column `column` and the columns over it at every coarser level.
fn ancestors(grid: &Grid, column: ColumnKey) -> impl Iterator<Item = ColumnKey> {
    (0..grid.levels()).map(move |level| ColumnKey { face: column.face, level: level as u8, ci: column.ci >> level, cj: column.cj >> level })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn region_keys_round_trip() {
        for r in [Region { face: 5, x: 0, y: 0 }, Region { face: 2, x: 123_456, y: 7 }, Region { face: 0, x: (1 << 27) - 1, y: 1 << 26 }] {
            assert_eq!(Region::from_key(r.key()), r);
        }
        let around = Region { face: 1, x: 10, y: 10 }.around(3, 12);
        assert_eq!(around[0], Region { face: 1, x: 10, y: 10 });
        assert!(around.iter().all(|r| r.x < 12 && r.y < 12));
        assert_eq!(around.len(), around.iter().collect::<std::collections::HashSet<_>>().len());
    }
}
