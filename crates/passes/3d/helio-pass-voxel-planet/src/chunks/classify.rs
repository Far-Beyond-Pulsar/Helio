//! What each node of the chunk octree holds (`docs/chunk-octree.md`): from
//! its footprint's measured (or estimated) ranges and the edits over it,
//! a node is Mixed (generated into a chunk), Buried (the field's volume
//! under its surface: rock until a ray reaches it), or uniform Empty or
//! Solid (drawn without data).

use super::CHUNK;
use crate::edit_store::EditStore;
use crate::edits::{center_half, FaceBrush};
use crate::grid::Grid;
use rustc_hash::FxHashSet;

/// Cells of its level a node's descendants' surface may lie past its
/// footprint's band (`EditView::classify`).
const DESCENT: i32 = 2;

/// The 64x64 columns of one level over a stack of nodes: the unit of
/// column generation and of classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Footprint {
    pub face: u8,
    pub level: u32,
    pub cx: i32,
    pub cy: i32,
}

impl Footprint {
    pub fn node(self, cz: i32) -> NodeKey {
        NodeKey { face: self.face, level: self.level, cx: self.cx, cy: self.cy, cz }
    }

    /// Its columns' half-cell bounds along i and j (inclusive).
    pub fn span_half(self) -> ((i64, i64), (i64, i64)) {
        let shift = self.level + 1;
        let span = |c: i32| (i64::from(c * CHUNK) << shift, (i64::from(c * CHUNK + CHUNK) << shift) - 1);
        (span(self.cx), span(self.cy))
    }
}

/// A node of the octree: 64^3 cells of `level`, layer `cz` over footprint
/// (`face`, `level`, `cx`, `cy`). Its key in the chunk table is
/// `(face | level << 8, cx, cy, cz)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeKey {
    pub face: u8,
    pub level: u32,
    pub cx: i32,
    pub cy: i32,
    pub cz: i32,
}

impl NodeKey {
    pub fn footprint(self) -> Footprint {
        Footprint { face: self.face, level: self.level, cx: self.cx, cy: self.cy }
    }

    pub fn table_key(self) -> [i32; 4] {
        [i32::from(self.face) | ((self.level as i32) << 8), self.cx, self.cy, self.cz]
    }

    pub fn from_table_key(k: [i32; 4]) -> Self {
        Self { face: (k[0] & 0xff) as u8, level: (k[0] >> 8) as u32, cx: k[1], cy: k[2], cz: k[3] }
    }

    pub fn parent(self) -> Self {
        Self { face: self.face, level: self.level + 1, cx: self.cx >> 1, cy: self.cy >> 1, cz: self.cz >> 1 }
    }

    /// The eight nodes under it (none at level 0); those past the face's
    /// edge (`tiles` footprints along it at the finer level) are left out.
    pub fn children(self, tiles: i32) -> impl Iterator<Item = NodeKey> {
        let level = self.level.checked_sub(1);
        (0..8).filter_map(move |n| {
            let level = level?;
            let c = Self { face: self.face, level, cx: self.cx * 2 + (n & 1), cy: self.cy * 2 + ((n >> 1) & 1), cz: self.cz * 2 + (n >> 2) };
            (c.cx < tiles && c.cy < tiles).then_some(c)
        })
    }

    /// Its layers [lo, hi) in cells of its level.
    pub fn layers(self) -> (i32, i32) {
        (self.cz * CHUNK, self.cz * CHUNK + CHUNK)
    }
}

/// What a node holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    /// A surface may cross it: generated into a chunk.
    Mixed,
    /// The field's volume (caves) under the surface band: drawn as rock
    /// until a ray hits it, then generated.
    Buried,
    Empty,
    Solid,
}

/// Layers (level cells, [lo, hi)) a footprint's field may change: its
/// volume (caves, overhangs) and the band around its surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ranges {
    pub volume: (i32, i32),
    pub surface: (i32, i32),
}

impl Ranges {
    /// A child footprint's ranges from its parent's: the same heights in
    /// cells half as tall, widened by a parent cell (a level stays within
    /// its parent's cell of the parent's surface).
    pub fn child(self) -> Self {
        let half = |(lo, hi): (i32, i32)| (lo.saturating_mul(2) - 2, hi.saturating_mul(2) + 2);
        Self { volume: half(self.volume), surface: half(self.surface) }
    }
}

/// A footprint's nodes by layer: Mixed in `mixed` (chunk layers,
/// inclusive), Buried in `buried`, else uniform (`EditView::uniform_solid`).
#[derive(Clone, Debug, PartialEq)]
pub struct Class {
    pub mixed: Vec<(i32, i32)>,
    pub buried: (i32, i32),
    /// The field's volume floor (level cells): rock under it, unless an
    /// edit says otherwise.
    volume_floor: i32,
    /// Brushes that add or remove over the footprint (indices in
    /// `EditView::faces`), in order.
    shapes: Vec<u32>,
}

/// The edits as classification reads them: brushes kept analytic,
/// resolved per face, and the baked store. Shared by the render side and
/// the planner's thread.
pub struct EditView {
    pub grid: Grid,
    pub faces: Vec<FaceBrush>,
    pub large_faces: usize,
    pub recent_counts: Vec<usize>,
    pub baked: EditStore,
    /// Where the baked bricks are (shared while the baked store is
    /// unchanged).
    pub bricks: std::sync::Arc<BrickIndex>,
    /// Bumped whenever the edits change.
    pub version: u64,
    /// Base layers under the surface where any level's cells may differ
    /// from the heightfield (`TerrainField::volume_bounds`): caves a coarse
    /// level cannot show still lie there, so its nodes are Buried, not
    /// Solid, and refine towards them.
    pub volume_depth: i32,
}

/// Where an edit store's bricks are, at every level they show and over
/// them: a node whose descendants hold baked edits is Mixed though its own
/// level's cells, built by majority, may not show them (a 4 m shaft in
/// 6.4 m cells), so the cut refines down to them.
#[derive(Default)]
pub struct BrickIndex {
    /// Footprints holding bricks of their level.
    pub footprints: FxHashSet<Footprint>,
    /// Each footprint's layers of nodes holding bricks or over them.
    pub nodes: rustc_hash::FxHashMap<Footprint, Vec<i32>>,
    /// Base layers the bricks span.
    pub layers: Option<(i32, i32)>,
}

impl BrickIndex {
    pub fn new(grid: &Grid, store: &EditStore) -> Self {
        let top = grid.levels() - 1;
        let mut index = Self::default();
        let mut marked: FxHashSet<NodeKey> = FxHashSet::default();
        for (k, _) in store.bricks() {
            let level = u32::from(k.level);
            index.footprints.insert(Footprint { face: k.face, level, cx: k.bi >> 3, cy: k.bj >> 3 });
            let (lo, hi) = ((k.bk * 8) << level, (k.bk * 8 + 8) << level);
            index.layers = Some(index.layers.map_or((lo, hi), |(a, b)| (a.min(lo), b.max(hi))));
            let mut node = NodeKey { face: k.face, level, cx: k.bi >> 3, cy: k.bj >> 3, cz: k.bk >> 3 };
            while marked.insert(node) && node.level < top {
                node = node.parent();
            }
        }
        for node in marked {
            index.nodes.entry(node.footprint()).or_default().push(node.cz);
        }
        index
    }
}

/// Whether `fb` can touch cells of `level` with half-cell bounds `i`, `j`,
/// `k` (inclusive).
pub fn brush_meets(fb: &FaceBrush, face: u8, level: u32, i: (i64, i64), j: (i64, i64), k: (i64, i64)) -> bool {
    if fb.face() != face || fb.radius_half < (1u32 << level) {
        return false;
    }
    let w = i64::from(fb.center[3]);
    let (ci, cj) = (i64::from(fb.center[0]), i64::from(fb.center[1]));
    ci + w >= i.0 && ci - w <= i.1 && cj + w >= j.0 && cj - w <= j.1 && i64::from(fb.k_hi) >= k.0 && i64::from(fb.k_lo) <= k.1
}

impl EditView {
    /// Brushes that add or remove and can touch `fp`'s columns, with their
    /// indices: at any size (one too small to change a cell of this level
    /// still makes the nodes it crosses Mixed, so the cut refines to it).
    fn shapes_over(&self, fp: Footprint) -> impl Iterator<Item = (usize, &FaceBrush)> {
        let (i, j) = fp.span_half();
        self.faces.iter().enumerate().filter(move |(_, fb)| fb.op() <= 1 && brush_meets(fb, fp.face, 0, i, j, (i64::MIN / 4, i64::MAX / 4)))
    }

    /// Layers [lo, hi) of `fp`'s level where brush `fb`'s surface can be
    /// over the footprint, with a cell of margin (none: lo >= hi). A box's
    /// heights; a ball's, the heights of its sphere's lower and upper caps
    /// over the footprint (one band where it reaches the sphere's
    /// silhouette; the air inside holds nothing). Along a direction at
    /// angle t from the ball's centre (d from the planet's, radius r) the
    /// caps lie at d cos t -+ sqrt(r^2 - d^2 sin^2 t): the lower rises with
    /// t, the upper falls, so the footprint's nearest and farthest angles
    /// bound them.
    pub fn surface_bands(&self, fb: &FaceBrush, fp: Footprint) -> [(i32, i32); 2] {
        let grid = &self.grid;
        let shift = fp.level + 1;
        let band = |k_lo: i32, k_hi: i32| ((k_lo.max(fb.k_lo) >> shift) - 1, (k_hi.min(fb.k_hi) >> shift) + 2);
        const NO_BAND: (i32, i32) = (0, 0);
        if fb.shape() != crate::edits::SHAPE_BALL || grid.is_plane() {
            return [band(fb.k_lo, fb.k_hi), NO_BAND];
        }
        let units = f64::from(grid.sphere_constants()[2]) / grid.radius();
        let centre = glam::IVec3::from_slice(&fb.ball[..3]).as_dvec3() / units;
        let (d, r) = (centre.length(), f64::from(fb.ball[3].unsigned_abs()) / units);
        // The footprint's directions in the brushes' (domain) space: its
        // middle, and the angle to its farthest corner cell (its edges are
        // great circles) and two base cells more.
        let size = CHUNK << fp.level;
        let (i0, j0) = (fp.cx * size, fp.cy * size);
        let towards = |i: i32, j: i32| grid.domain_point(fp.face, i, j, 0).as_dvec3().normalize();
        let middle = towards(i0 + size / 2, j0 + size / 2);
        let spread = [(0, 0), (1, 0), (0, 1), (1, 1)].iter().map(|&(a, b)| middle.angle_between(towards(i0 + a * (size - 1), j0 + b * (size - 1)))).fold(0.0, f64::max) * 1.01
            + 2.0 * grid.voxel_size() / grid.radius();
        let t = if d > 0.0 { middle.angle_between(centre / d) } else { 0.0 };
        let (near, far) = ((t - spread).max(0.0), (t + spread).min(std::f64::consts::PI));
        // Half base cells of a radius, two cells of margin.
        let k = |radius: f64, margin: f64| (((radius - grid.radius()) / grid.voxel_size() + margin) * 2.0).clamp(-2.0e9, 2.0e9) as i32;
        let cap = |t: f64, sign: f64| d * t.cos() + sign * (r * r - (d * t.sin()).powi(2)).max(0.0).sqrt();
        if d <= r {
            // The planet's centre is inside: the upper cap alone.
            return [band(k(cap(far, 1.0), -2.0), k(cap(near, 1.0), 2.0)), NO_BAND];
        }
        let silhouette = (r / d).asin();
        if near >= silhouette {
            return [NO_BAND; 2];
        }
        if far >= silhouette {
            return [band(k(cap(near, -1.0), -2.0), k(cap(near, 1.0), 2.0)), NO_BAND];
        }
        [band(k(cap(near, -1.0), -2.0), k(cap(far, -1.0), 2.0)), band(k(cap(far, 1.0), -2.0), k(cap(near, 1.0), 2.0))]
    }

    /// The layers [lo, hi) (level cells) of `fp`'s nodes holding baked
    /// bricks or over them.
    fn baked_bands(&self, fp: Footprint) -> impl Iterator<Item = (i32, i32)> + '_ {
        self.bricks.nodes.get(&fp).into_iter().flatten().map(|&cz| (cz * CHUNK, cz * CHUNK + CHUNK))
    }

    /// `fp`'s nodes from its `ranges`: the field's surface band and every
    /// edit's are Mixed (merged where fewer than two chunks apart), the
    /// rest of the field's volume Buried, the rest uniform. The field's
    /// band is widened by `DESCENT` cells: a node's descendants keep their
    /// surface within a cell of its own (coarse cells round their tops
    /// down), so a node beside the band may hold their ground.
    pub fn classify(&self, fp: Footprint, ranges: Ranges) -> Class {
        let mut bands: Vec<(i32, i32)> = Vec::new();
        let field = (ranges.surface.0 - DESCENT, ranges.volume.1 + DESCENT);
        if field.1 > field.0 {
            bands.push(field);
        }
        let mut shapes = Vec::new();
        for (n, fb) in self.shapes_over(fp) {
            let before = bands.len();
            bands.extend(self.surface_bands(fb, fp).into_iter().filter(|b| b.1 > b.0));
            if bands.len() > before {
                shapes.push(n as u32);
            }
        }
        bands.extend(self.baked_bands(fp));
        let mut layers: Vec<(i32, i32)> = bands.iter().map(|&(a, b)| (a.div_euclid(CHUNK), (b - 1).div_euclid(CHUNK))).collect();
        layers.sort_unstable();
        let mut mixed: Vec<(i32, i32)> = Vec::new();
        for (a, b) in layers {
            match mixed.last_mut() {
                Some(last) if a <= last.1 + 2 => last.1 = last.1.max(b),
                _ => mixed.push((a, b)),
            }
        }
        let floor = ranges.volume.0.min(ranges.surface.0 - (self.volume_depth >> fp.level) - DESCENT);
        let buried = (floor.div_euclid(CHUNK), (field.0 - 1).div_euclid(CHUNK));
        Class { mixed, buried, volume_floor: floor, shapes }
    }

    /// Node `cz` of `fp`, of class `class`. Outside every surface band a
    /// node lies wholly inside or outside each brush: at the cell under the
    /// footprint's centre in its middle layer, the last brush that adds or
    /// removes there decides; else the field's buried volume, rock under
    /// it and air over it.
    pub fn kind(&self, fp: Footprint, class: &Class, cz: i32) -> Kind {
        if class.mixed.iter().any(|&(a, b)| (a..=b).contains(&cz)) {
            return Kind::Mixed;
        }
        let (i, j, z) = (fp.cx * CHUNK + CHUNK / 2, fp.cy * CHUNK + CHUNK / 2, cz * CHUNK + CHUNK / 2);
        let centre = [center_half(i, fp.level), center_half(j, fp.level), center_half(z, fp.level)];
        let mut point = None;
        let brush = class.shapes.iter().rev().map(|&n| &self.faces[n as usize]).find(|b| b.contains(centre, || *point.get_or_insert_with(|| self.grid.volume_point(fp.face, i, j, z, fp.level))));
        match brush {
            Some(b) if b.op() == 1 => Kind::Solid,
            Some(_) => Kind::Empty,
            None if (class.buried.0..=class.buried.1).contains(&cz) => Kind::Buried,
            None if z < class.volume_floor => Kind::Solid,
            None => Kind::Empty,
        }
    }
}
