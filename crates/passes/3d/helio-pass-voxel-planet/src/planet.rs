//! Canonical editable voxel world (a planet or a plane): recipe, exact cell
//! queries and ray casts.
use crate::edits::{apply, center_half, Brush, EditLog, FaceBrush};
use crate::field::{self, FieldConstants, Landform, HEIGHT_ONE};
use crate::grid::{face_axes, Cell, Grid, Shape};
use glam::DVec3;
use serde::{Deserialize, Serialize};
use std::sync::Mutex;

pub const RECIPE_VERSION: u32 = 1;
pub const EARTH_RADIUS: f64 = 6_371_000.0;

/// Serialized authoring recipe. Edits are stored separately (see `journal`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PlanetRecipe {
    pub version: u32,
    /// Sphere (planet), finite plane or infinite plane.
    pub shape: Shape,
    /// Planet radius (spheres).
    pub radius_m: f64,
    /// Edge length of a finite plane, centred on the origin.
    pub plane_size_m: f64,
    pub voxel_size_m: f64,
    pub landform: Landform,
}

impl Default for PlanetRecipe {
    fn default() -> Self {
        Self {
            version: RECIPE_VERSION,
            shape: Shape::Sphere,
            radius_m: EARTH_RADIUS,
            plane_size_m: 4_096.0,
            voxel_size_m: 0.1,
            landform: Landform::default(),
        }
    }
}

impl PlanetRecipe {
    pub fn from_json(json: &str) -> Result<Self, String> {
        if json.trim().is_empty() {
            return Ok(Self::default());
        }
        let recipe: Self = serde_json::from_str(json).map_err(|e| e.to_string())?;
        if recipe.version != RECIPE_VERSION {
            return Err(format!("unsupported planet recipe version {}", recipe.version));
        }
        Ok(recipe)
    }
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("recipe serializes")
    }
}

/// Result of an exact ray cast on the base grid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHit {
    pub cell: Cell,
    /// Last empty cell before the hit (for building).
    pub previous: Cell,
    pub distance: f64,
    /// Outward normal of the entered cell face (planet-centred frame).
    pub normal: DVec3,
}

pub struct Planet {
    recipe: PlanetRecipe,
    grid: Grid,
    field: FieldConstants,
    edits: EditLog,
    revision: u64,
    /// Highest radius any add brush reaches.
    edit_top: f64,
    /// Lowest radius any remove brush reaches.
    edit_bottom: f64,
    heights: Mutex<rustc_hash::FxHashMap<(u8, i32, i32, u32), i32>>,
}

impl Clone for Planet {
    fn clone(&self) -> Self {
        Self {
            recipe: self.recipe.clone(),
            grid: self.grid,
            field: self.field,
            edits: self.edits.clone(),
            revision: self.revision,
            edit_top: self.edit_top,
            edit_bottom: self.edit_bottom,
            heights: Mutex::new(Default::default()),
        }
    }
}

impl Planet {
    pub fn new(recipe: PlanetRecipe) -> Result<Self, String> {
        let grid = match recipe.shape {
            Shape::Sphere => Grid::new(recipe.radius_m, recipe.voxel_size_m)?,
            shape => Grid::plane(shape, recipe.plane_size_m, recipe.voxel_size_m)?,
        };
        let field = FieldConstants::new(&grid, &recipe.landform);
        Ok(Self {
            recipe,
            grid,
            field,
            edits: EditLog::default(),
            revision: 0,
            edit_top: 0.0,
            edit_bottom: f64::INFINITY,
            heights: Mutex::new(Default::default()),
        })
    }
    pub fn recipe(&self) -> &PlanetRecipe {
        &self.recipe
    }
    pub fn grid(&self) -> &Grid {
        &self.grid
    }
    pub fn field(&self) -> &FieldConstants {
        &self.field
    }
    pub fn edits(&self) -> &EditLog {
        &self.edits
    }
    /// Increments with every applied or undone edit.
    pub fn revision(&self) -> u64 {
        self.revision
    }
    pub fn apply(&mut self, brush: Brush) -> Result<u32, String> {
        let id = self.edits.push(&self.grid, brush)?;
        // A cube brush reaches sqrt(3) radii from its centre.
        let reach = brush.radius * 1.7321 + self.grid.voxel_size();
        let centre = self.grid.radial(DVec3::from_array(brush.center));
        match brush.op {
            crate::edits::BrushOp::Add => self.edit_top = self.edit_top.max(centre + reach),
            crate::edits::BrushOp::Remove => self.edit_bottom = self.edit_bottom.min(centre - reach),
            _ => {}
        }
        self.revision += 1;
        Ok(id)
    }
    pub fn undo(&mut self) -> Option<Brush> {
        let brush = self.edits.pop()?;
        self.revision += 1;
        Some(brush)
    }
    /// Conservative bound on terrain surface height above the datum (m).
    pub fn max_terrain_height(&self) -> f64 {
        let k = &self.field;
        let mut sum = f64::from(k.levels[1].abs());
        for o in &k.octaves[crate::field::WARP_OCTAVES..k.header[1] as usize] {
            if o.kind >= 2 {
                sum += f64::from(o.amplitude.abs());
            }
        }
        sum / f64::from(HEIGHT_ONE) + 10.0
    }
    pub fn min_terrain_height(&self) -> f64 {
        -self.max_terrain_height() - f64::from(self.field.levels[0].abs()) / f64::from(HEIGHT_ONE)
    }
    /// Radius below which every cell is solid (terrain and removals).
    pub fn inner_radius(&self) -> f64 {
        (self.grid.radius() + self.min_terrain_height() - self.grid.voxel_size() * 4.0).min(self.edit_bottom)
    }
    /// Outer radius that bounds every solid cell (terrain and additions).
    pub fn outer_radius(&self) -> f64 {
        (self.grid.radius() + self.max_terrain_height() + self.grid.voxel_size() * 4.0).max(self.edit_top + self.grid.voxel_size())
    }
    /// Band-limited surface height of a level column, in height units.
    pub fn column_height(&self, face: u8, i: i32, j: i32, level: u32) -> i32 {
        let key = (face, i, j, level);
        if let Ok(cache) = self.heights.lock() {
            if let Some(&h) = cache.get(&key) {
                return h;
            }
        }
        let p = self.grid.domain_point(face, i, j, level);
        let h = field::height(&self.field, p, level);
        if let Ok(mut cache) = self.heights.lock() {
            if cache.len() > 1 << 20 {
                cache.clear();
            }
            cache.insert(key, h);
        }
        h
    }
    /// First air layer above the column (level cells).
    pub fn column_top(&self, face: u8, i: i32, j: i32, level: u32) -> i32 {
        field::top_cells(&self.field, self.column_height(face, i, j, level), level)
    }
    fn face_brushes(&self, face: u8, i: i32, j: i32, level: u32) -> Vec<FaceBrush> {
        let lo_i = i64::from(i) << level;
        let lo_j = i64::from(j) << level;
        let span = (1i64 << level) - 1;
        self.edits
            .query(face, lo_i, lo_i + span, lo_j, lo_j + span, level)
            .into_iter()
            .map(|(id, index)| self.edits.resolved(id).faces[index as usize])
            .collect()
    }
    /// Canonical `(kind, material)` of a level cell: kind 0 air, 1 solid.
    /// Material 0 on a solid cell means "terrain rule".
    pub fn sample_kind(&self, level: u32, face: u8, i: i32, j: i32, k: i32) -> (u32, u32) {
        let top = self.column_top(face, i, j, level);
        let kind = field::terrain_kind(top, k);
        let center = [center_half(i, level), center_half(j, level), center_half(k, level)];
        apply(self.face_brushes(face, i, j, level).into_iter(), center, kind, 0)
    }
    /// Canonical kind at a base cell.
    pub fn kind(&self, cell: Cell) -> u32 {
        self.sample_kind(0, cell.face, cell.i, cell.j, cell.k).0
    }
    pub fn solid(&self, cell: Cell) -> bool {
        self.kind(cell) == 1
    }
    /// Resolved material of a base cell (0 for air).
    pub fn material(&self, cell: Cell) -> u32 {
        let (kind, material) = self.sample_kind(0, cell.face, cell.i, cell.j, cell.k);
        match kind {
            0 => field::material::AIR,
            _ if material != 0 => material,
            _ => {
                let h = self.column_height(cell.face, cell.i, cell.j, 0);
                let top = field::top_cells(&self.field, h, 0);
                // Slope is measured inside the cell's 8x8 column block, which
                // is exactly what the GPU shading pass has resident.
                let (bi, bj) = (cell.i & !7, cell.j & !7);
                let slope = field::block_slope(|x, y| self.column_top(cell.face, bi + x, bj + y, 0), cell.i & 7, cell.j & 7);
                let p = self.grid.domain_point(cell.face, cell.i, cell.j, 0);
                let _ = h;
                field::ground_material(&self.field, p, top * self.field.header[2], top - 1 - cell.k, slope, cell.k)
            }
        }
    }
    /// Exact base-grid ray cast. Stops at the first cell for which `stop`
    /// returns true given its kind. `max_distance` may be infinite; the ray
    /// is clipped to the world's shell (or slab and edges on a plane).
    pub fn raycast_with(
        &self,
        origin: DVec3,
        direction: DVec3,
        max_distance: f64,
        stop: impl Fn(u32) -> bool,
    ) -> Option<RayHit> {
        if self.grid.is_plane() {
            return self.raycast_plane(origin, direction, max_distance, stop);
        }
        let d = direction.normalize();
        let grid = &self.grid;
        let outer = self.outer_radius();
        let b = origin.dot(d);
        let c = origin.length_squared() - outer * outer;
        let mut t = 0.0f64;
        if c > 0.0 {
            let disc = b * b - c;
            if b >= 0.0 || disc < 0.0 {
                return None;
            }
            t = -b - disc.sqrt();
        }
        let limit = max_distance.min(-b + (b * b - c).max(0.0).sqrt());
        let s = grid.voxel_size();
        let eps = s * 1e-6;
        let (mut cell, _) = grid.locate(origin + d * (t + eps));
        let mut previous = cell;
        let mut normal = -d;
        let n = grid.cells();
        for _ in 0..4_000_000 {
            if t > limit {
                return None;
            }
            if stop(self.kind(cell)) {
                return Some(RayHit {
                    cell,
                    previous,
                    distance: t,
                    normal,
                });
            }
            let [fn_, fa, fb] = face_axes(cell.face);
            let mut best = f64::INFINITY;
            let mut step = (0usize, 0i32, DVec3::ZERO);
            for (axis, (u, index)) in [(fa, cell.i), (fb, cell.j)].into_iter().enumerate() {
                for (offset, dir) in [(1, 1), (0, -1)] {
                    let angle = grid.angle(f64::from(index + offset));
                    let m = u * angle.cos() - fn_ * angle.sin();
                    let dm = d.dot(m);
                    // Leaving through the upper plane needs dm > 0, lower dm < 0.
                    if (dir == 1 && dm > 0.0) || (dir == -1 && dm < 0.0) {
                        let hit = -origin.dot(m) / dm;
                        if hit > t - eps && hit < best {
                            best = hit;
                            step = (axis, dir, -m * f64::from(dir));
                        }
                    }
                }
            }
            let r_lo = grid.layer_radius(f64::from(cell.k));
            let r_hi = r_lo + s;
            let ol = origin.length();
            let c_lo = (ol - r_lo) * (ol + r_lo);
            let disc_lo = b * b - c_lo;
            let mut radial = None;
            if disc_lo >= 0.0 {
                let root = disc_lo.sqrt();
                let enter = if b < 0.0 { c_lo / (-b + root) } else { -b - root };
                if enter > t - eps && b + enter < 0.0 {
                    radial = Some((enter, -1));
                }
            }
            if radial.is_none() {
                let c_hi = (ol - r_hi) * (ol + r_hi);
                let root = (b * b - c_hi).max(0.0).sqrt();
                let exit = if b > 0.0 { -c_hi / (b + root) } else { -b + root };
                radial = Some((exit, 1));
            }
            if let Some((hit, dir)) = radial {
                if hit < best {
                    best = hit;
                    let point = origin + d * hit;
                    step = (2, dir, point.normalize() * f64::from(dir) * -1.0);
                }
            }
            if !best.is_finite() {
                return None;
            }
            previous = cell;
            t = best.max(t);
            normal = step.2;
            let (axis, dir, _) = step;
            let mut next = cell;
            match axis {
                0 => next.i += dir,
                1 => next.j += dir,
                _ => next.k += dir,
            }
            if !(0..n).contains(&next.i) || !(0..n).contains(&next.j) {
                // Radial layers are shared by all faces; only (face, i, j) change.
                next = grid.locate(origin + d * (t + eps.max(t * 1e-12))).0;
                next.k = cell.k;
            }
            cell = next;
        }
        None
    }
    /// Cartesian cell walk on a plane grid, clipped to the slab between the
    /// lowest and highest solid layers and to the plane's edges.
    fn raycast_plane(&self, origin: DVec3, direction: DVec3, max_distance: f64, stop: impl Fn(u32) -> bool) -> Option<RayHit> {
        let grid = &self.grid;
        let d = direction.normalize();
        let [n_axis, a_axis, b_axis] = face_axes(crate::grid::PLANE_FACE);
        let o = grid.face_coords(crate::grid::PLANE_FACE, origin)?;
        let s = grid.voxel_size();
        let v = [d.dot(a_axis) / s, d.dot(b_axis) / s, d.dot(n_axis) / s];
        let n = f64::from(grid.cells());
        let lo = [0.0, 0.0, (self.inner_radius() / s).floor()];
        let hi = [n, n, (self.outer_radius() / s).ceil()];
        // Slab clipping in index space (t in metres).
        let (mut t0, mut t1) = (0.0f64, max_distance);
        for axis in 0..3 {
            if v[axis] == 0.0 {
                if o[axis] < lo[axis] || o[axis] >= hi[axis] {
                    return None;
                }
                continue;
            }
            let a = (lo[axis] - o[axis]) / v[axis];
            let b = (hi[axis] - o[axis]) / v[axis];
            t0 = t0.max(a.min(b));
            t1 = t1.min(a.max(b));
        }
        if t0 > t1 {
            return None;
        }
        let eps = 1e-9;
        let at = |t: f64| [o[0] + v[0] * t, o[1] + v[1] * t, o[2] + v[2] * t];
        let start = at(t0 + eps);
        let mut idx = [0i64; 3];
        for axis in 0..3 {
            idx[axis] = (start[axis].floor() as i64).clamp(lo[axis] as i64, hi[axis] as i64 - 1);
        }
        let step: [i64; 3] = std::array::from_fn(|axis| if v[axis] > 0.0 { 1 } else if v[axis] < 0.0 { -1 } else { 0 });
        let mut next: [f64; 3] = std::array::from_fn(|axis| {
            if step[axis] == 0 {
                f64::INFINITY
            } else {
                let boundary = idx[axis] as f64 + if step[axis] > 0 { 1.0 } else { 0.0 };
                (boundary - o[axis]) / v[axis]
            }
        });
        let delta: [f64; 3] = std::array::from_fn(|axis| if step[axis] == 0 { f64::INFINITY } else { 1.0 / v[axis].abs() });
        let axes = [a_axis, b_axis, n_axis];
        let cell_of = |idx: [i64; 3]| Cell::new(crate::grid::PLANE_FACE, idx[0] as i32, idx[1] as i32, idx[2] as i32);
        let mut t = t0;
        let mut previous = cell_of(idx);
        let mut normal = -d;
        for _ in 0..4_000_000 {
            if t > t1 {
                return None;
            }
            let cell = cell_of(idx);
            if stop(self.kind(cell)) {
                return Some(RayHit { cell, previous, distance: t, normal });
            }
            let axis = if next[0] <= next[1] && next[0] <= next[2] { 0 } else if next[1] <= next[2] { 1 } else { 2 };
            previous = cell;
            t = next[axis];
            next[axis] += delta[axis];
            idx[axis] += step[axis];
            normal = -axes[axis] * step[axis] as f64;
            if idx[axis] < lo[axis] as i64 || idx[axis] >= hi[axis] as i64 {
                return None;
            }
        }
        None
    }
    /// Ray cast that stops at solid cells (terrain and additions).
    pub fn raycast(&self, origin: DVec3, direction: DVec3, max_distance: f64) -> Option<RayHit> {
        self.raycast_with(origin, direction, max_distance, |kind| kind == 1)
    }
    /// A point `clearance` metres above the solid surface over `p` (a
    /// direction or any point above the ground point on a planet; any point
    /// on a plane).
    pub fn surface_point(&self, p: DVec3, clearance: f64) -> DVec3 {
        let up = self.grid.up(p);
        let top = self.grid.at_radial(p, self.outer_radius() + 1.0);
        match self.raycast(top, -up, f64::INFINITY) {
            Some(hit) => top - up * (hit.distance - clearance),
            None => self.grid.at_radial(p, self.grid.radius() + clearance),
        }
    }
    /// Distance from `eye` to the nearest possible solid cell, conservative.
    pub fn air_clearance(&self, eye: DVec3) -> f64 {
        let r = self.grid.radial(eye);
        let above_outer = r - self.outer_radius();
        if above_outer > 0.0 {
            return above_outer;
        }
        let (cell, _) = self.grid.locate(eye);
        let top = self.column_top(cell.face, cell.i, cell.j, 0);
        (f64::from(cell.k - top) * self.grid.voxel_size()).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edits::{BrushOp, BrushShape};

    fn planet() -> Planet {
        Planet::new(PlanetRecipe::default()).unwrap()
    }

    #[test]
    fn bound_margins_hold_for_sampled_cells() {
        for voxel in [0.1, 0.3, 1.0] {
            let p = Planet::new(PlanetRecipe { voxel_size_m: voxel, ..Default::default() }).unwrap();
            let g = *p.grid();
            let margins = p.field().bound_margins(&g);
            let mut rng = 0x2545_F491_4F6C_DD1Du64;
            let mut next = || {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                rng
            };
            let mut worst = vec![i32::MIN; g.levels() as usize];
            for _ in 0..6_000 {
                let level = 1 + (next() % u64::from(g.levels() - 1)) as u32;
                let cells = g.cells() >> level;
                let face = (next() % 6) as u8;
                let (i, j) = ((next() % cells as u64) as i32, (next() % cells as u64) as i32);
                let top = p.column_top(face, i, j, level);
                for _ in 0..6 {
                    let finer = (next() % u64::from(level)) as u32;
                    let shift = level - finer;
                    let fi = (i << shift) + (next() % (1u64 << shift)) as i32;
                    let fj = (j << shift) + (next() % (1u64 << shift)) as i32;
                    let fine = p.column_top(face, fi, fj, finer);
                    let excess = (((fine - 1) >> shift) + 1) - top;
                    worst[level as usize] = worst[level as usize].max(excess);
                    assert!(excess <= margins[level as usize], "voxel {voxel} level {level}: excess {excess} > {}", margins[level as usize]);
                }
            }
            eprintln!("voxel {voxel}: margins {:?}
worst {:?}", &margins[..g.levels() as usize], worst);
        }
    }

    fn plane(shape: Shape) -> Planet {
        Planet::new(PlanetRecipe { shape, plane_size_m: 3_000.0, ..Default::default() }).unwrap()
    }

    #[test]
    fn plane_raycast_down_hits_the_column_top_and_matches_edits() {
        for shape in [Shape::Plane, Shape::InfinitePlane] {
            let mut p = plane(shape);
            let g = *p.grid();
            let face = crate::grid::PLANE_FACE;
            for &(dx, dz) in &[(0.0, 0.0), (731.3, -412.9), (-1_100.0, 1_200.5)] {
                let ground = p.surface_point(DVec3::new(dx, 0.0, dz), 0.0);
                let (cell, _) = g.locate(ground + DVec3::Y * 0.01);
                let top = p.column_top(face, cell.i, cell.j, 0);
                assert_eq!(cell.k, top, "surface point sits on the first air layer");
                let eye = ground + DVec3::Y * 20.0;
                let hit = p.raycast(eye, -DVec3::Y, 1000.0).expect("hit");
                assert_eq!(hit.cell, Cell::new(face, cell.i, cell.j, top - 1));
                assert!((hit.distance - 20.0).abs() < g.voxel_size() * 1.01);
                assert_eq!(hit.normal, DVec3::Y);
            }
            // Oblique rays agree with a cell walk through a carved hole.
            let centre = p.surface_point(DVec3::new(50.0, 0.0, 60.0), -0.45);
            p.apply(Brush { center: centre.to_array(), radius: 0.45, shape: BrushShape::Sphere, op: BrushOp::Remove, material: 0 }).unwrap();
            let eye = centre + DVec3::new(0.3, 30.0, 0.0);
            let hit = p.raycast(eye, centre - eye, 100.0).expect("hit");
            assert!(p.solid(hit.cell));
            assert!(!p.solid(hit.previous));
            // Nothing past the edge of a finite plane.
            if shape == Shape::Plane {
                assert!(p.raycast(DVec3::new(5_000.0, 10.0, 0.0), -DVec3::Y, 1000.0).is_none());
            }
        }
    }

    #[test]
    fn raycast_down_hits_the_column_top() {
        let p = planet();
        let g = *p.grid();
        for &(face, fi, fj) in &[(4u8, 0.31, 0.62), (0, 0.9, 0.1), (3, 0.5, 0.5)] {
            let i = (f64::from(g.cells()) * fi) as i32;
            let j = (f64::from(g.cells()) * fj) as i32;
            let top = p.column_top(face, i, j, 0);
            let dir = g.direction(face, f64::from(i) + 0.5, f64::from(j) + 0.5);
            let eye = dir * g.layer_radius(f64::from(top) + 20.0);
            let hit = p.raycast(eye, -dir, 1000.0).expect("hit");
            assert_eq!(hit.cell, Cell::new(face, i, j, top - 1));
            assert!((hit.distance - 20.0 * g.voxel_size()).abs() < 1e-6);
        }
    }

    #[test]
    fn brushes_carve_and_fill_exactly() {
        let mut p = planet();
        let g = *p.grid();
        let (face, i, j) = (4u8, g.cells() / 3, g.cells() / 5);
        let top = p.column_top(face, i, j, 0);
        let centre = g.cell_center(Cell::new(face, i, j, top - 1));
        p.apply(Brush { center: centre.to_array(), radius: 1.0, shape: BrushShape::Sphere, op: BrushOp::Remove, material: 0 }).unwrap();
        assert!(!p.solid(Cell::new(face, i, j, top - 1)));
        assert!(!p.solid(Cell::new(face, i, j, top - 9)));
        assert!(p.solid(Cell::new(face, i, j, top - 12)));
        p.apply(Brush { center: centre.to_array(), radius: 0.25, shape: BrushShape::Cube, op: BrushOp::Add, material: field::material::BRICK }).unwrap();
        assert!(p.solid(Cell::new(face, i, j, top - 1)));
        assert_eq!(p.material(Cell::new(face, i, j, top - 1)), field::material::BRICK);
        assert!(p.undo().is_some());
        assert!(!p.solid(Cell::new(face, i, j, top - 1)));
    }

    #[test]
    fn oblique_raycast_matches_cell_walk_through_carved_hole() {
        let mut p = planet();
        let g = *p.grid();
        let (face, i, j) = (2u8, g.cells() / 2 + 17, g.cells() / 2 - 40);
        let top = p.column_top(face, i, j, 0);
        let centre = g.cell_center(Cell::new(face, i, j, top - 5));
        p.apply(Brush { center: centre.to_array(), radius: 0.45, shape: BrushShape::Sphere, op: BrushOp::Remove, material: 0 }).unwrap();
        let up = centre.normalize();
        let eye = centre + up * 30.0 + up.any_orthonormal_vector() * 0.3;
        let hit = p.raycast(eye, centre - eye, 100.0).expect("hit");
        assert!(p.solid(hit.cell));
        assert!(!p.solid(hit.previous));
        let n = g.voxel_size();
        assert!(g.cell_center(hit.cell).distance(eye + (centre - eye).normalize() * hit.distance) < n * 1.8);
    }
}
