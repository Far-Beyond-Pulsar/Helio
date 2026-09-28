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
}

/// Ordered edit log plus a hierarchical tile index.
#[derive(Clone, Default)]
pub struct EditLog {
    brushes: Vec<Resolved>,
    tiles: FxHashMap<(u32, u8, i64, i64), Vec<u32>>,
    buckets: u32,
}

impl EditLog {
    pub fn len(&self) -> usize {
        self.brushes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.brushes.is_empty()
    }
    pub fn brushes(&self) -> impl Iterator<Item = &Brush> {
        self.brushes.iter().map(|r| &r.brush)
    }
    pub fn resolved(&self, id: u32) -> &Resolved {
        &self.brushes[id as usize]
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
        let id = self.brushes.len() as u32;
        for fb in &faces {
            let (g, i0, i1, j0, j1) = Self::tiles_of(fb);
            self.buckets = self.buckets.max(g + 1);
            for ti in i0..=i1 {
                for tj in j0..=j1 {
                    self.tiles.entry((g, fb.face(), ti, tj)).or_default().push(id);
                }
            }
        }
        self.brushes.push(Resolved { brush, faces });
        Ok(id)
    }
    /// Remove the most recent brush (undo).
    pub fn pop(&mut self) -> Option<Brush> {
        let last = self.brushes.pop()?;
        let id = self.brushes.len() as u32;
        for fb in &last.faces {
            let (g, i0, i1, j0, j1) = Self::tiles_of(fb);
            for ti in i0..=i1 {
                for tj in j0..=j1 {
                    if let Some(list) = self.tiles.get_mut(&(g, fb.face(), ti, tj)) {
                        list.retain(|&b| b != id);
                        if list.is_empty() {
                            self.tiles.remove(&(g, fb.face(), ti, tj));
                        }
                    }
                }
            }
        }
        Some(last.brush)
    }
    /// Ordered face-brush references affecting the base-cell rectangle
    /// `[i0, i1] × [j0, j1]` of `face` at LOD `level`. Returned as
    /// `(brush id, index into that brush's faces)`.
    pub fn query(&self, face: u8, i0: i64, i1: i64, j0: i64, j1: i64, level: u32) -> Vec<(u32, u8)> {
        if self.brushes.is_empty() {
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
                for ((bg, bf, ti, tj), list) in &self.tiles {
                    if *bg == g && *bf == face && (a0..=a1).contains(ti) && (b0..=b1).contains(tj) {
                        ids.extend_from_slice(list);
                    }
                }
                continue;
            }
            for ti in a0..=a1 {
                for tj in b0..=b1 {
                    if let Some(list) = self.tiles.get(&(g, face, ti, tj)) {
                        ids.extend_from_slice(list);
                    }
                }
            }
        }
        ids.sort_unstable();
        ids.dedup();
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            for (index, fb) in self.brushes[id as usize].faces.iter().enumerate() {
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
