//! Voxel world grids: an equal-angle cube sphere (planets) or a flat plane,
//! finite or effectively infinite.
//!
//! On a sphere every cell is bounded by two planes through the planet centre
//! per horizontal axis and by two concentric spheres. Radial layers are always
//! aligned with gravity, so flat ground stays flat everywhere on the planet.
//! A plane uses the +Y face basis with axis-aligned cells and horizontal
//! layers. Either way a straight ray crosses each boundary family in closed
//! form, which lets the GPU traverse the exact canonical grid.
use glam::{DVec3, IVec3};
use serde::{Deserialize, Serialize};
use std::f64::consts::FRAC_PI_4;

/// Edge of one residency brick and one column footprint, in level cells.
pub const BRICK: i32 = 8;
/// Extra levels of divisibility so every level's columns tile a face exactly.
const COLUMN_LEVEL_BITS: u32 = 3;

/// Orthonormal cube face basis `(normal, a, b)` with `a × b = normal`.
pub const FACE_BASIS: [[[i32; 3]; 3]; 6] = [
    [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
    [[-1, 0, 0], [0, 0, 1], [0, 1, 0]],
    [[0, 1, 0], [1, 0, 0], [0, 0, -1]],
    [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
    [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
    [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
];

pub fn face_axes(face: u8) -> [DVec3; 3] {
    FACE_BASIS[face as usize].map(|v| IVec3::from_array(v).as_dvec3())
}

/// Face whose pyramid contains direction `p` (largest absolute component).
pub fn face_of(p: DVec3) -> u8 {
    let a = p.abs();
    if a.x >= a.y && a.x >= a.z {
        if p.x >= 0.0 {
            0
        } else {
            1
        }
    } else if a.y >= a.z {
        if p.y >= 0.0 {
            2
        } else {
            3
        }
    } else if p.z >= 0.0 {
        4
    } else {
        5
    }
}

/// World shape of a voxel grid.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Shape {
    /// A planet: an equal-angle cube sphere.
    #[default]
    Sphere,
    /// A square plane of a given edge length, centred on the origin.
    Plane,
    /// A plane without edges within reach: 2^27 reference cells (about
    /// 13 400 km) across, centred on the origin.
    InfinitePlane,
}

/// The one face of a plane grid (the +Y face basis: `i` along +X, `j`
/// along -Z, layers along +Y).
pub const PLANE_FACE: u8 = 2;
/// Width of an infinite plane in reference cells (keeps every domain
/// coordinate in 31 bits).
const INFINITE_PLANE_REFERENCE_CELLS: i64 = 1 << 27;

/// Base-resolution cell address. `k` is the signed radial layer; layer zero
/// starts at the datum (sea level). A plane grid uses face [`PLANE_FACE`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct Cell {
    pub face: u8,
    pub i: i32,
    pub j: i32,
    pub k: i32,
}

impl Cell {
    pub const fn new(face: u8, i: i32, j: i32, k: i32) -> Self {
        Self { face, i, j, k }
    }
    /// The containing cell at `level` (floor division on every axis).
    pub fn at_level(self, level: u32) -> Self {
        Self::new(self.face, self.i >> level, self.j >> level, self.k >> level)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Grid {
    shape: Shape,
    /// Planet radius (0 for planes).
    radius: f64,
    cells: i32,
    levels: u32,
    /// Exact radial layer thickness in millimetres (the authored size).
    layer_mm: u32,
    /// Cells per face of the 0.1 m reference grid that defines the noise domain.
    reference_cells: i32,
    /// `reference_cells / cells` in Q24.
    domain_scale: u32,
    /// `round(log2(reference_cells / cells))`.
    level_offset: u32,
}

/// Reference grid resolution: the terrain domain is identical for every
/// authored voxel size, so changing the grid never reshapes the planet.
pub const REFERENCE_VOXEL: f64 = 0.1;

fn face_cells(radius: f64, voxel_size: f64) -> (i64, u32) {
    edge_cells(radius * std::f64::consts::FRAC_PI_2, voxel_size)
}

/// Cells along a face edge of length `arc`, a multiple of every level's
/// column width, and the level count.
fn edge_cells(arc: f64, voxel_size: f64) -> (i64, u32) {
    // The coarsest level keeps roughly 1024 cells across a face.
    let mut levels = 1u32;
    while levels < 24 && arc / (voxel_size * f64::from(1u32 << (levels - 1))) > 1024.0 {
        levels += 1;
    }
    let unit = 1i64 << (levels - 1 + COLUMN_LEVEL_BITS);
    let blocks = (arc / voxel_size / unit as f64).round().max(1.0) as i64;
    (blocks * unit, levels)
}

impl Grid {
    /// A planet grid whose base cells are as close to `voxel_size` as the
    /// level hierarchy allows (within ~0.7 %). The planet radius is exact.
    pub fn new(radius: f64, voxel_size: f64) -> Result<Self, String> {
        if !(radius.is_finite() && radius >= 1_000.0 && radius <= 50_000_000.0) {
            return Err(format!("planet radius {radius} m is outside 1 km..50 000 km"));
        }
        if !(voxel_size.is_finite() && voxel_size >= 0.01 && voxel_size <= 64.0) {
            return Err(format!("voxel size {voxel_size} m is outside 0.01..64 m"));
        }
        let layer_mm = (voxel_size * 1000.0).round();
        if (layer_mm / 1000.0 - voxel_size).abs() > 1e-9 {
            return Err("voxel size must be a whole number of millimetres".into());
        }
        let (cells, levels) = face_cells(radius, voxel_size);
        let (reference, _) = face_cells(radius, REFERENCE_VOXEL);
        if cells >= (1i64 << 29) || reference >= (1i64 << 29) {
            return Err("voxel grid exceeds 2^29 cells per face".into());
        }
        let ratio = reference as f64 / cells as f64;
        if ratio > 15.0 {
            return Err("voxel size is too coarse for the reference terrain domain".into());
        }
        Ok(Self {
            shape: Shape::Sphere,
            radius,
            cells: cells as i32,
            levels,
            layer_mm: layer_mm as u32,
            reference_cells: reference as i32,
            domain_scale: (ratio * 16_777_216.0).round() as u32,
            level_offset: ratio.log2().round().max(0.0) as u32,
        })
    }
    /// A plane grid: `size` is the edge length for [`Shape::Plane`]
    /// (ignored for [`Shape::InfinitePlane`]). The layer thickness is exactly
    /// `voxel_size`, and so is the cell width up to the rounding of the edge.
    pub fn plane(shape: Shape, size: f64, voxel_size: f64) -> Result<Self, String> {
        if !(voxel_size.is_finite() && voxel_size >= 0.01 && voxel_size <= 64.0) {
            return Err(format!("voxel size {voxel_size} m is outside 0.01..64 m"));
        }
        let layer_mm = (voxel_size * 1000.0).round();
        if (layer_mm / 1000.0 - voxel_size).abs() > 1e-9 {
            return Err("voxel size must be a whole number of millimetres".into());
        }
        let reference_edge = INFINITE_PLANE_REFERENCE_CELLS as f64 * REFERENCE_VOXEL;
        let edge = match shape {
            Shape::Sphere => return Err("a sphere needs Grid::new".into()),
            Shape::Plane => {
                if !(size.is_finite() && size >= voxel_size * 64.0 && size <= reference_edge) {
                    return Err(format!("plane size {size} m is outside {:.1} m..{reference_edge:.0} m", voxel_size * 64.0));
                }
                size
            }
            Shape::InfinitePlane => reference_edge,
        };
        let (mut cells, mut levels) = edge_cells(edge, voxel_size);
        // Round the infinite plane down so its reference domain stays within 2^27 cells.
        while shape == Shape::InfinitePlane && (cells as f64 * voxel_size / REFERENCE_VOXEL) > INFINITE_PLANE_REFERENCE_CELLS as f64 {
            let unit = 1i64 << (levels - 1 + COLUMN_LEVEL_BITS);
            cells -= unit;
            if cells <= 0 {
                levels -= 1;
                cells = 1i64 << (levels - 1 + COLUMN_LEVEL_BITS);
            }
        }
        let ratio = voxel_size / REFERENCE_VOXEL;
        if ratio > 15.0 {
            return Err("voxel size is too coarse for the reference terrain domain".into());
        }
        Ok(Self {
            shape,
            radius: 0.0,
            cells: cells as i32,
            levels,
            layer_mm: layer_mm as u32,
            reference_cells: (cells as f64 * ratio).round() as i32,
            domain_scale: (ratio * 16_777_216.0).round() as u32,
            level_offset: ratio.log2().round().max(0.0) as u32,
        })
    }
    pub fn shape(&self) -> Shape {
        self.shape
    }
    pub fn is_plane(&self) -> bool {
        self.shape != Shape::Sphere
    }
    /// Faces holding cells: all six on a sphere, [`PLANE_FACE`] on a plane.
    pub fn faces(&self) -> &'static [u8] {
        if self.is_plane() {
            &[PLANE_FACE]
        } else {
            &[0, 1, 2, 3, 4, 5]
        }
    }
    /// Planet radius (0 for planes).
    pub fn radius(&self) -> f64 {
        self.radius
    }
    /// Plane cell index of the world origin on each horizontal axis (0 on a sphere).
    pub fn origin_index(&self) -> i32 {
        if self.is_plane() {
            self.cells / 2
        } else {
            0
        }
    }
    /// Radial coordinate of `p`: distance from the planet centre, or height
    /// above the plane. Layer `k` starts at radial `layer_radius(k)`.
    pub fn radial(&self, p: DVec3) -> f64 {
        if self.is_plane() {
            p.y
        } else {
            p.length()
        }
    }
    /// Height of `p` above the datum.
    pub fn height(&self, p: DVec3) -> f64 {
        self.radial(p) - self.radius
    }
    /// Local up at `p`.
    pub fn up(&self, p: DVec3) -> DVec3 {
        if self.is_plane() {
            DVec3::Y
        } else {
            p.normalize_or(DVec3::Y)
        }
    }
    /// `p` moved to radial coordinate `radial` along the local vertical.
    pub fn at_radial(&self, p: DVec3, radial: f64) -> DVec3 {
        if self.is_plane() {
            DVec3::new(p.x, radial, p.z)
        } else {
            p.normalize_or(DVec3::Y) * radial
        }
    }
    /// Distance along the datum between the ground points below `a` and
    /// `b`: great-circle distance on a sphere, horizontal on a plane.
    pub fn ground_distance(&self, a: DVec3, b: DVec3) -> f64 {
        if self.is_plane() {
            DVec3::new(a.x - b.x, 0.0, a.z - b.z).length()
        } else {
            a.angle_between(b) * self.radius
        }
    }
    /// Base cells along one face edge.
    pub fn cells(&self) -> i32 {
        self.cells
    }
    /// Number of LOD levels. Level `levels() - 1` covers a whole face with
    /// roughly a thousand cells.
    pub fn levels(&self) -> u32 {
        self.levels
    }
    /// Angular width of one base cell on a sphere; the cell width on a plane.
    pub fn delta(&self) -> f64 {
        if self.is_plane() {
            self.voxel_size()
        } else {
            std::f64::consts::FRAC_PI_2 / f64::from(self.cells)
        }
    }
    /// Radial layer thickness (the authored voxel size). The tangential cell
    /// width at the datum face centre differs by well under 1 %.
    pub fn voxel_size(&self) -> f64 {
        f64::from(self.layer_mm) / 1000.0
    }
    pub fn layer_mm(&self) -> u32 {
        self.layer_mm
    }
    pub fn tangential_size(&self) -> f64 {
        self.delta() * self.radius
    }
    pub fn reference_cells(&self) -> i32 {
        self.reference_cells
    }
    pub fn domain_scale(&self) -> u32 {
        self.domain_scale
    }
    pub fn level_offset(&self) -> u32 {
        self.level_offset
    }
    pub fn level_size(&self, level: u32) -> f64 {
        self.voxel_size() * f64::from(1u32 << level)
    }
    /// Face-plane angle at a continuous base index coordinate.
    pub fn angle(&self, index: f64) -> f64 {
        -FRAC_PI_4 + index * self.delta()
    }
    pub fn index_of_angle(&self, angle: f64) -> f64 {
        (angle + FRAC_PI_4) / self.delta()
    }
    pub fn layer_radius(&self, k: f64) -> f64 {
        self.radius + k * self.voxel_size()
    }
    /// Unit direction for continuous `(i, j)` index coordinates on `face`
    /// (the planet sphere only).
    pub fn direction(&self, face: u8, i: f64, j: f64) -> DVec3 {
        let [n, a, b] = face_axes(face);
        (n + a * self.angle(i).tan() + b * self.angle(j).tan()).normalize()
    }
    /// World position of continuous index coordinates (planet-centred on a
    /// sphere, origin-centred on a plane).
    pub fn position(&self, face: u8, index: [f64; 3]) -> DVec3 {
        if self.is_plane() {
            let [n, a, b] = face_axes(face);
            let c = f64::from(self.origin_index());
            let s = self.voxel_size();
            return a * ((index[0] - c) * s) + b * ((index[1] - c) * s) + n * (index[2] * s);
        }
        self.direction(face, index[0], index[1]) * self.layer_radius(index[2])
    }
    /// Point on the datum below continuous index coordinates `(i, j)`.
    pub fn ground_point(&self, face: u8, i: f64, j: f64) -> DVec3 {
        self.position(face, [i, j, 0.0])
    }
    pub fn cell_center(&self, cell: Cell) -> DVec3 {
        self.position(
            cell.face,
            [
                f64::from(cell.i) + 0.5,
                f64::from(cell.j) + 0.5,
                f64::from(cell.k) + 0.5,
            ],
        )
    }
    /// Continuous index coordinates of `p` in the (possibly extended) grid of
    /// `face`. `None` when `p` is not in that face's hemisphere.
    pub fn face_coords(&self, face: u8, p: DVec3) -> Option<[f64; 3]> {
        let [n, a, b] = face_axes(face);
        if self.is_plane() {
            if face != PLANE_FACE {
                return None;
            }
            let c = f64::from(self.origin_index());
            let s = self.voxel_size();
            return Some([p.dot(a) / s + c, p.dot(b) / s + c, p.dot(n) / s]);
        }
        let pn = p.dot(n);
        if pn <= 0.0 {
            return None;
        }
        let alpha = p.dot(a).atan2(pn);
        let beta = p.dot(b).atan2(pn);
        let k = (p.length() - self.radius) / self.voxel_size();
        Some([self.index_of_angle(alpha), self.index_of_angle(beta), k])
    }
    /// Canonical cell containing `p` and its continuous coordinates.
    pub fn locate(&self, p: DVec3) -> (Cell, [f64; 3]) {
        let face = if self.is_plane() { PLANE_FACE } else { face_of(p) };
        let c = self.face_coords(face, p).expect("point is in its face hemisphere");
        let last = self.cells - 1;
        let cell = Cell::new(
            face,
            (c[0].floor() as i64).clamp(0, i64::from(last)) as i32,
            (c[1].floor() as i64).clamp(0, i64::from(last)) as i32,
            c[2].floor().clamp(i32::MIN as f64, i32::MAX as f64) as i32,
        );
        (cell, c)
    }
    /// Neighbour across one cell face; `axis` 0/1 are the tangential index
    /// axes and 2 is radial. Crossing a cube edge resolves the neighbour's
    /// own face and indices through its centre point.
    pub fn neighbour(&self, cell: Cell, axis: usize, step: i32) -> Cell {
        let mut next = cell;
        match axis {
            0 => next.i += step,
            1 => next.j += step,
            _ => next.k += step,
        }
        if self.is_plane() || ((0..self.cells).contains(&next.i) && (0..self.cells).contains(&next.j)) {
            // A plane has no neighbouring face: indices past its edge are
            // outside the world.
            return next;
        }
        let centre = self.position(
            cell.face,
            [
                f64::from(next.i) + 0.5,
                f64::from(next.j) + 0.5,
                f64::from(next.k) + 0.5,
            ],
        );
        self.locate(centre).0
    }
    /// Integer point on the "index cube" used as the noise domain, in half
    /// base cells. Adjacent faces share their edge points, so procedural
    /// fields are continuous across cube edges.
    pub fn domain_point(&self, face: u8, i: i32, j: i32, level: u32) -> IVec3 {
        if self.is_plane() {
            return plane_domain_point(self.origin_index(), self.domain_scale, i, j, level);
        }
        domain_point(self.reference_cells, self.domain_scale, face, i, j, level)
    }

    /// Constants of [`volume_point`]: `(inv, shift, layer_q16)`. On a sphere
    /// `inv = floor(2^(30 + shift) / (2 R_layers))` with `inv` in
    /// `[2^23, 2^24)`; on a plane `layer_q16` converts half layers to half
    /// reference cells.
    pub fn volume_constants(&self) -> (u32, u32, u32) {
        let layer_q16 = ((u64::from(self.layer_mm) << 16) / 100) as u32;
        if self.is_plane() {
            return (0, 0, layer_q16);
        }
        let layers = ((self.radius * 1000.0 / f64::from(self.layer_mm)).round() as u64).max(1) * 2;
        let mut shift = 0u32;
        while (1u64 << (30 + shift)) / layers < (1 << 23) {
            shift += 1;
        }
        (((1u64 << (30 + shift)) / layers) as u32, shift, layer_q16)
    }

    /// Seamless 3D domain point of the level cell `(face, i, j, k)`.
    pub fn volume_point(&self, face: u8, i: i32, j: i32, k: i32, level: u32) -> IVec3 {
        let (inv, shift, layer_q16) = self.volume_constants();
        volume_point(self.domain_point(face, i, j, level), self.is_plane(), k, level, inv, shift, layer_q16)
    }
}

/// Approximately `(a * r) >> 24` with 16-bit limbs and only 32-bit integer
/// operations (mirrored bit-for-bit in WGSL). Valid while the result fits in
/// 31 bits; rounding is deterministic, not exact.
#[inline]
pub fn mul_q24(a: u32, r: u32) -> u32 {
    let (a1, a0) = (a >> 16, a & 0xffff);
    let (r1, r0) = (r >> 16, r & 0xffff);
    (a1.wrapping_mul(r1) << 8)
        .wrapping_add(a1.wrapping_mul(r0).wrapping_add(a0.wrapping_mul(r1)) >> 8)
        .wrapping_add(a0.wrapping_mul(r0) >> 24)
}

/// Domain point of a level cell centre (see [`Grid::domain_point`]), in half
/// cells of the reference grid.
pub fn domain_point(reference: i32, scale: u32, face: u8, i: i32, j: i32, level: u32) -> IVec3 {
    let [n, a, b] = FACE_BASIS[face as usize].map(IVec3::from_array);
    let half = 1u32 << level;
    let u = mul_q24(((i as u32) << (level + 1)).wrapping_add(half), scale) as i32 - reference;
    let v = mul_q24(((j as u32) << (level + 1)).wrapping_add(half), scale) as i32 - reference;
    n * reference + a * u + b * v
}

/// `(a * b) >> s` of the exact 64-bit product (`mul_shr` in WGSL computes
/// it with 16-bit limbs). The caller keeps the result within 32 bits.
#[inline]
pub fn mul_shr(a: u32, b: u32, s: u32) -> u32 {
    ((u64::from(a) * u64::from(b)) >> s) as u32
}

/// `p * r >> 30` for a signed component and a signed Q30 ratio.
#[inline]
fn scale_component(p: i32, ratio: i32) -> i32 {
    let m = mul_shr(p.unsigned_abs(), ratio.unsigned_abs(), 30) as i32;
    if (p < 0) != (ratio < 0) { -m } else { m }
}

/// Seamless 3D domain point of a level cell centre (half reference cells).
/// On a sphere the column's cube-surface domain point is scaled by
/// `(R + h) / R`, so points at equal height agree across cube edges; on a
/// plane the height is the vertical axis. Volumetric terrain (caves,
/// overhangs) samples 3D noise here. Mirrored by `volume_point` in WGSL.
#[allow(clippy::too_many_arguments)]
pub fn volume_point(p: IVec3, plane: bool, k: i32, level: u32, inv: u32, shift: u32, layer_q16: u32) -> IVec3 {
    // Cell centre height in half base layers.
    let h = (k << (level + 1)).wrapping_add(1 << level);
    if plane {
        let v = mul_shr(h.unsigned_abs(), layer_q16, 16) as i32;
        return IVec3::new(p.x, if h < 0 { -v } else { v }, p.z);
    }
    // Ratio h / (2 R_layers) in Q30: a Q24 ratio moved the point in ~0.4 m
    // steps at Earth radius (|p| ~ 2^27), stair-stepping caves.
    let r = mul_shr(h.unsigned_abs(), inv, shift) as i32;
    let ratio = if h < 0 { -r } else { r };
    p + IVec3::new(scale_component(p.x, ratio), scale_component(p.y, ratio), scale_component(p.z, ratio))
}

/// Domain point of a plane level cell centre, in half reference cells
/// relative to the world origin: `(u, 0, -v)` in the +Y face basis. The
/// scaling rounds by magnitude, so it is symmetric about the origin.
pub fn plane_domain_point(origin: i32, scale: u32, i: i32, j: i32, level: u32) -> IVec3 {
    let half = 1i32 << level;
    let scaled = |x: i32| {
        let x = x.wrapping_shl(level + 1).wrapping_add(half).wrapping_sub(origin << 1);
        let m = mul_q24(x.unsigned_abs(), scale) as i32;
        if x < 0 { -m } else { m }
    };
    IVec3::new(scaled(i), 0, -scaled(j))
}

/// Camera-relative description of one face's boundary families, computed in
/// `f64` and consumed in `f32` by the GPU and by precision tests.
///
/// For the α family (planes containing `b`), a plane at angle `α_c + δ` has
/// normal `m cos δ - q sin δ`; the camera lies on the δ = 0 plane at distance
/// `rho` from the plane axis. The β family is identical with `a` and `b`
/// exchanged.
#[derive(Clone, Copy, Debug, Default)]
pub struct FaceFrame {
    pub m: [DVec3; 2],
    pub q: [DVec3; 2],
    pub rho: [f64; 2],
    /// Camera base-cell index on each tangential axis (may be outside the face).
    pub index: [i64; 2],
    /// Fraction of the camera inside that index, in [0, 1).
    pub fraction: [f64; 2],
    pub valid: bool,
}

impl Grid {
    pub fn face_frame(&self, face: u8, eye: DVec3) -> FaceFrame {
        let [n, a, b] = face_axes(face);
        let mut frame = FaceFrame::default();
        if self.is_plane() {
            // Cell planes are parallel: `m` is their normal and the camera
            // index and fraction come straight from its coordinates.
            let c = self.face_coords(PLANE_FACE, eye).unwrap_or([0.0; 3]);
            for (axis, u) in [a, b].into_iter().enumerate() {
                frame.m[axis] = u;
                frame.q[axis] = n;
                let whole = c[axis].floor();
                frame.index[axis] = whole as i64;
                frame.fraction[axis] = c[axis] - whole;
            }
            frame.valid = face == PLANE_FACE;
            return frame;
        }
        for (axis, u) in [a, b].into_iter().enumerate() {
            // Angle of the eye around axis `w`, measured in the (u, n) plane.
            let pu = eye.dot(u);
            let pn = eye.dot(n);
            let rho = (pu * pu + pn * pn).sqrt();
            let angle = pu.atan2(pn);
            let (s, c) = angle.sin_cos();
            frame.m[axis] = u * c - n * s;
            frame.q[axis] = u * s + n * c;
            frame.rho[axis] = rho;
            let index = self.index_of_angle(angle);
            let whole = index.floor();
            frame.index[axis] = whole as i64;
            frame.fraction[axis] = index - whole;
        }
        // Extended coordinates stay meaningful for any eye that is not on the
        // opposite side of the planet; GPU traversal only uses a face frame
        // after locating a point in that face.
        frame.valid = eye.dot(n) > -0.5 * eye.length();
        frame
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn earth() -> Grid {
        Grid::new(6_371_000.0, 0.1).unwrap()
    }

    #[test]
    fn voxel_size_is_close_to_request_for_all_authored_sizes() {
        for step in 1..=10 {
            let size = f64::from(step) * 0.1;
            let grid = Grid::new(6_371_000.0, (size * 1000.0).round() / 1000.0).unwrap();
            let error = (grid.tangential_size() - size).abs() / size;
            assert!(error < 0.01, "{size}: {} ({error})", grid.tangential_size());
            // The same physical location maps to the same reference domain point.
            let reference = Grid::new(6_371_000.0, 0.1).unwrap();
            let dir = DVec3::new(0.3, 0.8, -0.2).normalize();
            let c = grid.locate(dir * 6_371_000.0).0;
            let r = reference.locate(grid.cell_center(c)).0;
            let d = grid.domain_point(c.face, c.i, c.j, 0) - reference.domain_point(r.face, r.i, r.j, 0);
            assert!(d.abs().max_element() <= 4 * step as i32 + 8, "{d}");
            assert_eq!(grid.cells() % (1 << (grid.levels() + 2)), 0);
        }
    }

    #[test]
    fn locate_round_trips_cell_centres_including_face_edges() {
        let grid = earth();
        let n = grid.cells();
        for face in 0..6u8 {
            for &(i, j) in &[(0, 0), (n - 1, 0), (0, n - 1), (n - 1, n - 1), (n / 2, n / 3), (7, n - 8)] {
                for &k in &[-100_000, -1, 0, 1, 90_000] {
                    let cell = Cell::new(face, i, j, k);
                    assert_eq!(grid.locate(grid.cell_center(cell)).0, cell);
                }
            }
        }
    }

    #[test]
    fn neighbours_across_cube_edges_are_adjacent_and_reciprocal() {
        let grid = earth();
        let n = grid.cells();
        for face in 0..6u8 {
            for (axis, step, cell) in [
                (0, 1, Cell::new(face, n - 1, n / 2, 3)),
                (0, -1, Cell::new(face, 0, n / 3, 3)),
                (1, 1, Cell::new(face, n / 4, n - 1, 3)),
                (1, -1, Cell::new(face, n / 5, 0, 3)),
            ] {
                let next = grid.neighbour(cell, axis, step);
                assert_ne!(next.face, face);
                let distance = grid.cell_center(cell).distance(grid.cell_center(next));
                assert!(distance < grid.voxel_size() * 1.5, "{distance}");
                // Some neighbour of `next` must return to `cell`.
                let back = (0..2)
                    .flat_map(|a| [-1, 1].map(|s| grid.neighbour(next, a, s)))
                    .any(|c| c == cell);
                assert!(back);
            }
        }
    }

    #[test]
    fn domain_points_are_continuous_across_edges() {
        let grid = earth();
        let n = grid.cells();
        let a = grid.domain_point(4, n - 1, n / 2, 0);
        let edge = grid.neighbour(Cell::new(4, n - 1, n / 2, 0), 0, 1);
        let b = grid.domain_point(edge.face, edge.i, edge.j, 0);
        assert!((a - b).abs().max_element() <= 2, "{a} {b}");
    }

    /// Volumetric noise samples one seamless 3D domain: cells at equal
    /// height across a cube edge get nearly equal points, a layer step moves
    /// the point by about one reference voxel per 0.1 m, and points stay in
    /// range from the core to far above the surface.
    #[test]
    fn volume_points_are_seamless_and_follow_height() {
        let grid = earth();
        let n = grid.cells();
        for k in [-1_000_000, -1_200, -1, 0, 37, 50_000] {
            let a = grid.volume_point(4, n - 1, n / 2, k, 0);
            let edge = grid.neighbour(Cell::new(4, n - 1, n / 2, k), 0, 1);
            let b = grid.volume_point(edge.face, edge.i, edge.j, edge.k, 0);
            assert!((a - b).abs().max_element() <= 4, "k {k}: {a} {b}");
        }
        // Radial step at a face centre: a 0.1 m layer scales the point by
        // 0.1 / R, and the cube half size is (pi/2) R / 0.1 m, so the point
        // moves pi/2 units (a horizontal 0.1 m cell moves it 2).
        let c = n / 2;
        let step = grid.volume_point(2, c, c, 1, 0) - grid.volume_point(2, c, c, 0, 0);
        let expected = std::f64::consts::FRAC_PI_2;
        assert!((f64::from(step.length_squared()).sqrt() - expected).abs() < 0.6, "{step}");
        let core = grid.volume_point(2, c, c, -(grid.radius() / grid.voxel_size()) as i32, 0);
        // inv carries 24 bits: |p| ~ 2^27 units lands within ~16 (0.8 m).
        assert!(core.abs().max_element() < 24, "the planet centre maps near the origin: {core}");
        let plane = Grid::plane(Shape::Plane, 4_000.0, 0.5).unwrap();
        let p = plane.volume_point(PLANE_FACE, 10, 10, 4, 0);
        assert_eq!(p.y, 45, "plane height in half reference cells: (4.5 * 0.5 m) / 0.05 m");
    }

    #[test]
    fn plane_grids_locate_cell_centres_and_keep_exact_layers() {
        for (shape, size, voxel) in [(Shape::Plane, 4_000.0, 0.1), (Shape::Plane, 900.0, 0.7), (Shape::InfinitePlane, 0.0, 0.1), (Shape::InfinitePlane, 0.0, 1.0)] {
            let grid = Grid::plane(shape, size, voxel).unwrap();
            assert_eq!(grid.faces(), &[PLANE_FACE]);
            assert!((grid.voxel_size() - voxel).abs() < 1e-12);
            assert_eq!(grid.cells() % (1 << (grid.levels() + 2)), 0);
            if shape == Shape::InfinitePlane {
                assert!(f64::from(grid.cells()) * voxel > 10_000_000.0, "{}", grid.cells());
                assert!(grid.reference_cells() <= 1 << 27);
            } else {
                assert!((f64::from(grid.cells()) * voxel - size).abs() <= voxel * f64::from(8u32 << grid.levels()));
            }
            let n = grid.cells();
            for &(i, j) in &[(0, 0), (n - 1, n - 1), (n / 2, n / 2 - 1), (7, n - 8)] {
                for &k in &[-3_000, -1, 0, 1, 40_000] {
                    let cell = Cell::new(PLANE_FACE, i, j, k);
                    assert_eq!(grid.locate(grid.cell_center(cell)).0, cell);
                    assert!((grid.height(grid.cell_center(cell)) - (f64::from(k) + 0.5) * voxel).abs() < 1e-6);
                }
            }
            // The origin sits on a cell corner; the domain is symmetric.
            let c = grid.origin_index();
            assert_eq!(grid.locate(DVec3::new(0.01, 0.01, -0.01)).0, Cell::new(PLANE_FACE, c, c, 0));
            let a = grid.domain_point(PLANE_FACE, c, c, 0);
            let b = grid.domain_point(PLANE_FACE, c - 1, c - 1, 0);
            assert_eq!(a, -b);
        }
    }

    #[test]
    fn plane_domain_matches_across_voxel_sizes() {
        let fine = Grid::plane(Shape::Plane, 5_000.0, 0.1).unwrap();
        let coarse = Grid::plane(Shape::Plane, 5_000.0, 0.4).unwrap();
        let p = DVec3::new(123.4, 0.0, -777.7);
        let a = fine.locate(p).0;
        let b = coarse.locate(p).0;
        let d = fine.domain_point(a.face, a.i, a.j, 0) - coarse.domain_point(b.face, b.i, b.j, 0);
        assert!(d.abs().max_element() <= 8, "{d}");
    }

    #[test]
    fn plane_face_frame_reproduces_camera_indices() {
        let grid = Grid::plane(Shape::InfinitePlane, 0.0, 0.1).unwrap();
        let eye = grid.position(PLANE_FACE, [12_345.25, 67_890.75, 18.5]);
        let frame = grid.face_frame(PLANE_FACE, eye);
        assert_eq!(frame.index, [12_345, 67_890]);
        assert!((frame.fraction[0] - 0.25).abs() < 1e-3);
        assert!((frame.fraction[1] - 0.75).abs() < 1e-3);
        assert!(frame.valid);
    }

    #[test]
    fn face_frame_reproduces_camera_indices() {
        let grid = earth();
        let eye = grid.position(4, [12_345.25, 67_890.75, 18.5]);
        let frame = grid.face_frame(4, eye);
        assert_eq!(frame.index, [12_345, 67_890]);
        assert!((frame.fraction[0] - 0.25).abs() < 1e-6);
        assert!((frame.fraction[1] - 0.75).abs() < 1e-6);
        assert!(frame.m[0].dot(eye).abs() < 1e-6);
        assert!(frame.m[1].dot(eye).abs() < 1e-6);
    }
}
