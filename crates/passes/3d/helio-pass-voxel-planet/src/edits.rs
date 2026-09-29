//! Ordered destruction/construction brushes and their spatial index.
//!
//! A brush is authored in planet-centred metres. For evaluation it is
//! resolved per cube face into integer half-cell coordinates, so CPU queries
//! and GPU generation apply the exact same integer containment test. Later
//! brushes override earlier ones. At LOD level `L` a brush whose radius is
//! below half a level cell is omitted (it is smaller than the point sample).
use crate::grid::{face_axes, Grid};
use bytemuck::{Pod, Zeroable};
use glam::DVec3;
use rustc_hash::FxHashMap;
use std::sync::Arc;
use serde::{Deserialize, Serialize};

/// Largest brush radius in half cells; keeps the squared test within u32.
pub const MAX_RADIUS_HALF: u32 = 37_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BrushShape {
    Sphere,
    Cube,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BrushOp {
    /// Carve to air.
    Remove,
    /// Fill with a material.
    Add,
    /// Recolour existing solid cells.
    Paint,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Brush {
    pub center: [f64; 3],
    pub radius: f64,
    pub shape: BrushShape,
    pub op: BrushOp,
    #[serde(default)]
    pub material: u32,
}

/// A brush resolved into one face's integer index space (GPU layout).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct FaceBrush {
    /// Face in bits 0..3; op in 4..6; shape in 6..8; material in 8..16.
    pub flags: u32,
    pub radius_half: u32,
    pub pad: [u32; 2],
    /// Centre in half base cells (i, j, k).
    pub center: [i32; 4],
}

impl FaceBrush {
    pub fn face(&self) -> u8 {
        (self.flags & 7) as u8
    }
    pub fn op(&self) -> u32 {
        (self.flags >> 4) & 3
    }
    pub fn material(&self) -> u32 {
        (self.flags >> 8) & 0xff
    }
    /// Applies at `level` (radius at least half a level cell).
    pub fn active(&self, level: u32) -> bool {
        self.radius_half >= (1u32 << level)
    }
    /// Exact containment of a level cell centre given in half cells.
    pub fn contains(&self, center_half: [i32; 3]) -> bool {
        let r = self.radius_half;
        let mut sum = 0u32;
        for axis in 0..3 {
            let d = center_half[axis].wrapping_sub(self.center[axis]).unsigned_abs();
            if d > r {
                return false;
            }
            sum = sum.wrapping_add(d.wrapping_mul(d));
        }
        (self.flags >> 6) & 3 == 1 || sum <= r.wrapping_mul(r)
    }
}

/// Half-cell centre coordinate of level cell index `i`.
#[inline]
pub fn center_half(i: i32, level: u32) -> i32 {
    (i << (level + 1)).wrapping_add(1 << level)
}

impl Brush {
    /// Resolve into every face whose (extended) grid the brush touches.
    pub fn resolve(&self, grid: &Grid) -> Result<Vec<FaceBrush>, String> {
        if !(self.radius.is_finite() && self.radius > 0.0) {
            return Err("brush radius must be positive".into());
        }
        if self.material > 255 {
            return Err("brush material must fit in 8 bits".into());
        }
        let radius_half = (2.0 * self.radius / grid.voxel_size()).round();
        if radius_half > f64::from(MAX_RADIUS_HALF) {
            return Err(format!(
                "brush radius {} m exceeds {:.0} m on this grid",
                self.radius,
                f64::from(MAX_RADIUS_HALF) * grid.voxel_size() / 2.0
            ));
        }
        let radius_half = (radius_half as u32).max(1);
        let center = DVec3::from_array(self.center);
        let n = f64::from(grid.cells());
        let margin = f64::from(radius_half) / 2.0 + 2.0;
        let op = match self.op {
            BrushOp::Remove => 0u32,
            BrushOp::Add => 1,
            BrushOp::Paint => 2,
        };
        let shape = match self.shape {
            BrushShape::Sphere => 0u32,
            BrushShape::Cube => 1,
        };
        let mut out = Vec::new();
        for &face in grid.faces() {
            let [nrm, _, _] = face_axes(face);
            // On a sphere, only faces whose hemisphere clearly contains the
            // brush centre; extended coordinates degrade far from the face.
            if !grid.is_plane() && center.dot(nrm) <= 0.5 * center.length() {
                continue;
            }
            let Some(c) = grid.face_coords(face, center) else {
                continue;
            };
            if c[0] < -margin || c[0] > n + margin || c[1] < -margin || c[1] > n + margin {
                continue;
            }
            let half = c.map(|v| (v * 2.0).round());
            if half.iter().any(|v| v.abs() > 2.0e9) {
                return Err("brush centre is outside the planet grid".into());
            }
            out.push(FaceBrush {
                flags: u32::from(face) | (op << 4) | (shape << 6) | (self.material << 8),
                radius_half,
                pad: [0; 2],
                center: [half[0] as i32, half[1] as i32, half[2] as i32, 0],
            });
        }
        if out.is_empty() {
            return Err("brush does not intersect the voxel grid".into());
        }
        Ok(out)
    }
}

/// Index tile edge at bucket `g`, in base cells.
fn tile(g: u32) -> i64 {
    64i64 << g
}

/// Bucket whose tiles are at least one brush diameter wide.
fn bucket_of(radius_half: u32) -> u32 {
    let mut g = 0;
    while tile(g) < i64::from(radius_half) + 2 {
        g += 1;
    }
    g
}

/// An applied brush with stable id (its position in the ordered log).
#[derive(Clone, Debug)]
pub struct Resolved {
    pub brush: Brush,
    pub faces: Vec<FaceBrush>,
    /// Hash of the log up to and including this brush: equal prefix hashes
    /// mean equal logs up to here.
    pub prefix: u64,
}

/// Brushes per shared chunk.
const CHUNK: usize = 1024;
/// Index entries added before the recent tiles are sealed.
const SEAL: usize = 1024;

type TileKey = (u32, u8, i64, i64);

/// Brush ids per index tile: a sealed map shared between copies of the log
/// and the entries added since it was sealed. Copying is O(recent); sealing
/// merges into a fresh map every [`SEAL`] entries.
#[derive(Clone, Default)]
struct TileIndex {
    sealed: Arc<FxHashMap<TileKey, Arc<[u32]>>>,
    recent: FxHashMap<TileKey, Vec<u32>>,
    recent_ids: usize,
}

impl TileIndex {
    fn add(&mut self, key: TileKey, id: u32) {
        self.recent.entry(key).or_default().push(id);
        self.recent_ids += 1;
        if self.recent_ids >= SEAL {
            let sealed = Arc::make_mut(&mut self.sealed);
            for (key, ids) in self.recent.drain() {
                let merged: Arc<[u32]> = match sealed.get(&key) {
                    Some(old) => old.iter().copied().chain(ids).collect(),
                    None => ids.into(),
                };
                sealed.insert(key, merged);
            }
            self.recent_ids = 0;
        }
    }

    fn remove(&mut self, key: TileKey, id: u32) {
        if let Some(list) = self.recent.get_mut(&key) {
            let before = list.len();
            list.retain(|&b| b != id);
            self.recent_ids -= before - list.len();
            if list.is_empty() {
                self.recent.remove(&key);
            }
        }
        if self.sealed.get(&key).is_some_and(|list| list.contains(&id)) {
            let sealed = Arc::make_mut(&mut self.sealed);
            let kept: Arc<[u32]> = sealed[&key].iter().copied().filter(|&b| b != id).collect();
            if kept.is_empty() {
                sealed.remove(&key);
            } else {
                sealed.insert(key, kept);
            }
        }
    }

    fn get(&self, key: &TileKey) -> impl Iterator<Item = u32> + '_ {
        let sealed = self.sealed.get(key).into_iter().flat_map(|list| list.iter().copied());
        sealed.chain(self.recent.get(key).into_iter().flat_map(|list| list.iter().copied()))
    }

    fn iter(&self) -> impl Iterator<Item = (&TileKey, &[u32])> {
        let sealed = self.sealed.iter().map(|(key, list)| (key, &list[..]));
        sealed.chain(self.recent.iter().map(|(key, list)| (key, &list[..])))
    }
}

/// Ordered edit log plus a hierarchical tile index.
///
/// Copies share structure: brushes live in shared chunks of [`CHUNK`] and
/// the tile index keeps a sealed shared map, so copying a log with tens of
/// thousands of edits (what a renderer does to extend a published world)
/// costs about as much as copying a few thousand.
#[derive(Clone, Default)]
pub struct EditLog {
    chunks: Vec<Arc<Vec<Resolved>>>,
    len: usize,
    tiles: TileIndex,
    buckets: u32,
}

/// FNV-1a over a brush, continuing `seed`.
fn brush_hash(seed: u64, brush: &Brush) -> u64 {
    let mut h = seed ^ 0xcbf2_9ce4_8422_2325;
    let mut eat = |v: u64| {
        for byte in v.to_le_bytes() {
            h = (h ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3);
        }
    };
    for c in brush.center {
        eat(c.to_bits());
    }
    eat(brush.radius.to_bits());
    eat(brush.shape as u64);
    eat(brush.op as u64);
    eat(u64::from(brush.material));
    h
}

impl EditLog {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn brushes(&self) -> impl Iterator<Item = &Brush> {
        self.chunks.iter().flat_map(|chunk| chunk.iter()).map(|r| &r.brush)
    }
    pub fn resolved(&self, id: u32) -> &Resolved {
        let id = id as usize;
        &self.chunks[id / CHUNK][id % CHUNK]
    }
    /// Hash of the first `id + 1` brushes (see [`Resolved::prefix`]).
    pub fn prefix_hash(&self, id: u32) -> u64 {
        self.resolved(id).prefix
    }
    fn tiles_of(face_brush: &FaceBrush) -> (u32, i64, i64, i64, i64) {
        let g = bucket_of(face_brush.radius_half);
        let t = tile(g);
        let r = i64::from(face_brush.radius_half) / 2 + 1;
        let ci = i64::from(face_brush.center[0]) / 2;
        let cj = i64::from(face_brush.center[1]) / 2;
        (
            g,
            (ci - r).div_euclid(t),
            (ci + r).div_euclid(t),
            (cj - r).div_euclid(t),
            (cj + r).div_euclid(t),
        )
    }
    pub fn push(&mut self, grid: &Grid, brush: Brush) -> Result<u32, String> {
        let faces = brush.resolve(grid)?;
        let id = self.len as u32;
        for fb in &faces {
            let (g, i0, i1, j0, j1) = Self::tiles_of(fb);
            self.buckets = self.buckets.max(g + 1);
            for ti in i0..=i1 {
                for tj in j0..=j1 {
                    self.tiles.add((g, fb.face(), ti, tj), id);
                }
            }
        }
        let prefix = brush_hash(if id == 0 { 0 } else { self.prefix_hash(id - 1) }, &brush);
        if self.len % CHUNK == 0 {
            self.chunks.push(Arc::new(Vec::with_capacity(CHUNK)));
        }
        // Copies only this chunk when another log still shares it.
        Arc::make_mut(self.chunks.last_mut().expect("pushed above")).push(Resolved { brush, faces, prefix });
        self.len += 1;
        Ok(id)
    }
    /// Remove the most recent brush (undo).
    pub fn pop(&mut self) -> Option<Brush> {
        let chunk = self.chunks.last_mut()?;
        let last = Arc::make_mut(chunk).pop()?;
        if chunk.is_empty() {
            self.chunks.pop();
        }
        self.len -= 1;
        let id = self.len as u32;
        for fb in &last.faces {
            let (g, i0, i1, j0, j1) = Self::tiles_of(fb);
            for ti in i0..=i1 {
                for tj in j0..=j1 {
                    self.tiles.remove((g, fb.face(), ti, tj), id);
                }
            }
        }
        Some(last.brush)
    }
    /// Ordered face-brush references affecting the base-cell rectangle
    /// `[i0, i1] × [j0, j1]` of `face` at LOD `level`. Returned as
    /// `(brush id, index into that brush's faces)`.
    pub fn query(&self, face: u8, i0: i64, i1: i64, j0: i64, j1: i64, level: u32) -> Vec<(u32, u8)> {
        if self.len == 0 {
            return Vec::new();
        }
        let mut ids: Vec<u32> = Vec::new();
        for g in 0..self.buckets {
            // Every brush in bucket g has radius_half > tile(g-1) - 2; skip
            // buckets whose largest brush is below the level's threshold.
            if (tile(g) + 2) < (1i64 << level) {
                continue;
            }
            let t = tile(g);
            let (a0, a1, b0, b1) = (i0.div_euclid(t), i1.div_euclid(t), j0.div_euclid(t), j1.div_euclid(t));
            if (a1 - a0 + 1) * (b1 - b0 + 1) > 4096 {
                // A huge region at a fine bucket: scan the bucket instead.
                for ((bg, bf, ti, tj), list) in self.tiles.iter() {
                    if *bg == g && *bf == face && (a0..=a1).contains(ti) && (b0..=b1).contains(tj) {
                        ids.extend_from_slice(list);
                    }
                }
                continue;
            }
            for ti in a0..=a1 {
                for tj in b0..=b1 {
                    ids.extend(self.tiles.get(&(g, face, ti, tj)));
                }
            }
        }
        ids.sort_unstable();
        ids.dedup();
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            for (index, fb) in self.resolved(id).faces.iter().enumerate() {
                if fb.face() != face || !fb.active(level) {
                    continue;
                }
                let r = i64::from(fb.radius_half) / 2 + 1;
                let ci = i64::from(fb.center[0]) / 2;
                let cj = i64::from(fb.center[1]) / 2;
                if ci + r < i0 || ci - r > i1 || cj + r < j0 || cj - r > j1 {
                    continue;
                }
                out.push((id, index as u8));
            }
        }
        out
    }
}

/// Apply ordered brushes to a canonical terrain `(kind, material)` at a level
/// cell centre. Kind: 0 air, 1 solid.
pub fn apply(brushes: impl Iterator<Item = FaceBrush>, center: [i32; 3], mut kind: u32, mut material: u32) -> (u32, u32) {
    for b in brushes {
        if !b.contains(center) {
            continue;
        }
        match b.op() {
            0 => {
                kind = 0;
                material = 0;
            }
            1 => {
                kind = 1;
                material = b.material();
            }
            _ => {
                if kind == 1 {
                    material = b.material();
                }
            }
        }
    }
    (kind, material)
}

#[cfg(test)]
mod shared_log {
    use super::*;
    use crate::grid::Shape;

    fn brute(log: &EditLog, face: u8, i0: i64, i1: i64, j0: i64, j1: i64, level: u32) -> Vec<(u32, u8)> {
        let mut out = Vec::new();
        for id in 0..log.len() as u32 {
            for (index, fb) in log.resolved(id).faces.iter().enumerate() {
                let r = i64::from(fb.radius_half) / 2 + 1;
                let (ci, cj) = (i64::from(fb.center[0]) / 2, i64::from(fb.center[1]) / 2);
                if fb.face() == face && fb.active(level) && ci + r >= i0 && ci - r <= i1 && cj + r >= j0 && cj - r <= j1 {
                    out.push((id, index as u8));
                }
            }
        }
        out
    }

    #[test]
    fn copies_share_structure_and_queries_match_brute_force() {
        let grid = Grid::plane(Shape::Plane, 512.0, 0.1).unwrap();
        let brush = |k: usize| Brush {
            center: [((k * 37) % 400) as f64 - 200.0, 1.0, ((k * 91) % 400) as f64 - 200.0],
            radius: if k % 97 == 0 { 6.0 } else { 0.3 },
            shape: if k % 3 == 0 { BrushShape::Cube } else { BrushShape::Sphere },
            op: if k % 2 == 0 { BrushOp::Add } else { BrushOp::Remove },
            material: 13,
        };
        let mut a = EditLog::default();
        for k in 0..9_000 {
            a.push(&grid, brush(k)).unwrap();
        }
        let snapshot = a.clone();
        let mut b = a.clone();
        for k in 9_000..9_500 {
            b.push(&grid, brush(k)).unwrap();
        }
        for _ in 0..700 {
            b.pop().unwrap();
        }
        assert_eq!((snapshot.len(), a.len(), b.len()), (9_000, 9_000, 8_800));
        // Prefix hashes agree where the logs agree and differ after.
        assert_eq!(a.prefix_hash(8_799), b.prefix_hash(8_799));
        let face = crate::grid::PLANE_FACE;
        let c = grid.cells() as i64 / 2;
        for (log, name) in [(&snapshot, "snapshot"), (&a, "a"), (&b, "b")] {
            for &(lo, hi, level) in &[(c - 2000, c + 2000, 0u32), (c - 300, c - 100, 0), (0, grid.cells() as i64, 5)] {
                let mut got = log.query(face, lo, hi, lo, hi, level);
                got.sort_unstable();
                assert_eq!(got, brute(log, face, lo, hi, lo, hi, level), "{name} {lo}..{hi} level {level}");
            }
        }
        // Copying a large log is cheap: shared chunks, bounded recent tiles.
        let t = std::time::Instant::now();
        for _ in 0..100 {
            std::hint::black_box(a.clone());
        }
        assert!(t.elapsed().as_millis() < 200, "{:?} for 100 copies", t.elapsed());
    }
}
