//! Canonical editable voxel world (a planet or a plane): recipe, exact cell
//! queries and ray casts.
use crate::edits::{apply, center_half, Brush, EditLog, FaceBrush};
use crate::grid::{face_axes, Cell, Grid, Shape};
use crate::terrain::{self, material, TerrainField, TerrainSource, HEIGHT_ONE};

use glam::DVec3;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

pub const RECIPE_VERSION: u32 = 2;
pub const EARTH_RADIUS: f64 = 6_371_000.0;

/// Serialized authoring recipe: the world's form and its terrain generator.
/// Edits are stored separately (see `journal`).
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
    pub terrain: TerrainSource,
}

impl Default for PlanetRecipe {
    fn default() -> Self {
        Self {
            version: RECIPE_VERSION,
            shape: Shape::Sphere,
            radius_m: EARTH_RADIUS,
            plane_size_m: 4_096.0,
            voxel_size_m: 0.1,
            terrain: TerrainSource::default(),
        }
    }
}

impl PlanetRecipe {
    /// Fingerprint of the ground this recipe makes (form, voxel size,
    /// generator, seed and settings, the settings compared as JSON values):
    /// edits belong to it (`VoxelEditJournal::made_on`).
    pub fn fingerprint(&self) -> u64 {
        let settings: serde_json::Value = serde_json::from_str(&self.terrain.settings).unwrap_or(serde_json::Value::Null);
        // Only the size the shape uses.
        let (radius, plane) = match self.shape {
            Shape::Sphere => (self.radius_m, 0.0),
            _ => (0.0, self.plane_size_m),
        };
        let key = serde_json::json!([self.version, self.shape, radius, plane, self.voxel_size_m,
            self.terrain.generator, self.terrain.version, self.terrain.seed, settings]);
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for byte in key.to_string().bytes() {
            h = (h ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3);
        }
        // 0 means "no terrain yet".
        h.max(1)
    }
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
/// The base column a ray walk is in (`Planet::kind_in`).
struct RayColumn {
    key: (u8, i32, i32),
    top: i32,
    brushes: Vec<FaceBrush>,
}

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
    field: Arc<dyn TerrainField>,
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
            field: Arc::clone(&self.field),
            edits: self.edits.clone(),
            revision: self.revision,
            edit_top: self.edit_top,
            edit_bottom: self.edit_bottom,
            heights: Mutex::new(Default::default()),
        }
    }
}

impl Planet {
    pub fn new(mut recipe: PlanetRecipe) -> Result<Self, String> {
        // Name the registered generator version (0 accepts it), so saved
        // edits record the terrain they were made against.
        if recipe.terrain.version == 0 {
            recipe.terrain.version = terrain::find(&recipe.terrain.generator, 0)
                .ok_or_else(|| format!("unknown terrain generator {}", recipe.terrain.generator))?
                .info()
                .version;
        }
        let grid = match recipe.shape {
            Shape::Sphere => Grid::new(recipe.radius_m, recipe.voxel_size_m)?,
            shape => Grid::plane(shape, recipe.plane_size_m, recipe.voxel_size_m)?,
        };
        let field = terrain::build(&recipe.terrain, &grid)?;
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
    /// The terrain generator's field for this world's grid.
    pub fn field(&self) -> &dyn TerrainField {
        &*self.field
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
        f64::from(self.field.height_range().1) / f64::from(HEIGHT_ONE)
    }
    pub fn min_terrain_height(&self) -> f64 {
        f64::from(self.field.height_range().0) / f64::from(HEIGHT_ONE)
    }
    /// Radius below which every cell is solid (terrain and removals).
    pub fn inner_radius(&self) -> f64 {
        let caves = f64::from(self.field.volume_bounds().0) / f64::from(HEIGHT_ONE);
        (self.grid.radius() + self.min_terrain_height() - caves - self.grid.voxel_size() * 4.0).min(self.edit_bottom)
    }
    /// Outer radius that bounds every solid cell (terrain and additions).
    pub fn outer_radius(&self) -> f64 {
        (self.grid.radius() + self.max_terrain_height() + self.overhang_height() + self.grid.voxel_size() * 4.0).max(self.edit_top + self.grid.voxel_size())
    }
    /// Largest rise (m) of generated overhangs over the heightfield.
    pub fn overhang_height(&self) -> f64 {
        f64::from(self.field.volume_bounds().1) / f64::from(HEIGHT_ONE)
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
        let h = self.field.height(p, level + self.grid.level_offset());
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
        terrain::top_cells(&self.grid, self.column_height(face, i, j, level), level)
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
        let kind = terrain::generated_kind(&self.grid, &*self.field, face, i, j, k, level, top);
        let center = [center_half(i, level), center_half(j, level), center_half(k, level)];
        apply(self.face_brushes(face, i, j, level).into_iter(), center, || self.grid.volume_point(face, i, j, k, level), kind, 0)
    }
    /// [`Self::kind`] for cells walked by a ray: the column's top and
    /// brushes are looked up once per column, not per cell.
    fn kind_in(&self, column: &mut Option<RayColumn>, cell: Cell) -> u32 {
        let key = (cell.face, cell.i, cell.j);
        if column.as_ref().is_none_or(|c| c.key != key) {
            *column = Some(RayColumn {
                key,
                top: self.column_top(cell.face, cell.i, cell.j, 0),
                brushes: self.face_brushes(cell.face, cell.i, cell.j, 0),
            });
        }
        let c = column.as_ref().expect("filled above");
        let kind = terrain::generated_kind(&self.grid, &*self.field, cell.face, cell.i, cell.j, cell.k, 0, c.top);
        if c.brushes.is_empty() {
            return kind;
        }
        let center = [center_half(cell.i, 0), center_half(cell.j, 0), center_half(cell.k, 0)];
        apply(c.brushes.iter().copied(), center, || self.grid.volume_point(cell.face, cell.i, cell.j, cell.k, 0), kind, 0).0
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
            0 => material::AIR,
            _ if material != 0 => material,
            _ => {
                let top = self.column_top(cell.face, cell.i, cell.j, 0);
                // Slope is measured inside the cell's 8x8 column block, which
                // is exactly what the GPU shading pass has resident.
                let (bi, bj) = (cell.i & !7, cell.j & !7);
                let slope = terrain::block_slope(|x, y| self.column_top(cell.face, bi + x, bj + y, 0), cell.i & 7, cell.j & 7);
                let p = self.grid.domain_point(cell.face, cell.i, cell.j, 0);
                let top_height = self.column_height(cell.face, cell.i, cell.j, 0);
                // Depth counts from the generated top: overhangs and the
                // rock around caves lie below it.
                let generated = terrain::generated_top(&self.grid, &*self.field, cell.face, cell.i, cell.j, 0, top);
                let depth = (generated - 1 - cell.k).max(0);
                let surface = self.field.surface(p, self.grid.level_offset(), top_height) & 0xff;
                self.field.ground_material(p, surface, top_height, depth, slope, cell.k) & material::ID
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
        let mut column = None;
        let n = grid.cells();
        for _ in 0..4_000_000 {
            if t > limit {
                return None;
            }
            if stop(self.kind_in(&mut column, cell)) {
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
                    // That exit plane is never behind the ray: a crossing
                    // rounded behind `t` (grazing planes, 6e6 m origins) is
                    // taken now, not dropped (a dropped crossing left the
                    // index stale for the rest of the ray).
                    if (dir == 1 && dm > 0.0) || (dir == -1 && dm < 0.0) {
                        let hit = -origin.dot(m) / dm;
                        if hit < best {
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
                // Descending (before the perigee at -b): the lower layer
                // crossing is the exit, wherever rounding puts it.
                if b + t < 0.0 && b + enter < 0.0 {
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
        let mut column = None;
        for _ in 0..4_000_000 {
            if t > t1 {
                return None;
            }
            let cell = cell_of(idx);
            if stop(self.kind_in(&mut column, cell)) {
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
    ///
    /// Walks the base column under `p` down from the highest layer that can
    /// be solid there (its generated top, or the top of an Add brush over
    /// it), with the column's brushes queried once: a few cells, not a ray
    /// from the world's outer radius through the edit index cell by cell.
    pub fn surface_point(&self, p: DVec3, clearance: f64) -> DVec3 {
        let g = &self.grid;
        let (cell, _) = g.locate(g.at_radial(p, g.radius()));
        let (face, i, j) = (cell.face, cell.i, cell.j);
        let top = self.column_top(face, i, j, 0);
        let brushes = self.face_brushes(face, i, j, 0);
        let added = brushes.iter().filter(|b| b.op() == 1).map(|b| b.k_hi.div_euclid(2) + 1).max().unwrap_or(i32::MIN);
        let mut k = terrain::generated_top(g, &*self.field, face, i, j, 0, top).max(added);
        let floor = ((self.inner_radius() - g.radius()) / g.voxel_size()).floor() as i32;
        while k > floor {
            let below = k - 1;
            let kind = terrain::generated_kind(g, &*self.field, face, i, j, below, 0, top);
            let center = [center_half(i, 0), center_half(j, 0), center_half(below, 0)];
            if apply(brushes.iter().copied(), center, || g.volume_point(face, i, j, below, 0), kind, 0).0 == 1 {
                break;
            }
            k = below;
        }
        g.at_radial(p, g.layer_radius(f64::from(k)) + clearance)
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
        (f64::from(cell.k - top) * self.grid.voxel_size() - self.overhang_height()).max(0.0)
    }
    /// Height of `eye` above the generated ground directly below it (its
    /// column's top along the local vertical; edits are not considered).
    /// Unlike [`Self::air_clearance`] this is not a bound on the distance to
    /// all terrain: it is the altitude a camera or vehicle moves by.
    pub fn ground_height(&self, eye: DVec3) -> f64 {
        let (cell, _) = self.grid.locate(eye);
        // The generated top: overhang lips and cave mouths included.
        let top = terrain::generated_top(&self.grid, &*self.field, cell.face, cell.i, cell.j, 0, self.column_top(cell.face, cell.i, cell.j, 0));
        self.grid.height(eye) - f64::from(top) * self.grid.voxel_size()
    }
    /// Radial coordinate bounding the solid cells whose ground point lies
    /// within ground distance `radius` of the point below `eye`: generated
    /// terrain from column tops plus a margin, additions from the edit top.
    /// Never above [`Self::outer_radius`], which it falls back to where the
    /// region leaves the face.
    ///
    /// For level selection only, where an optimistic answer just means a
    /// coarser level draws that terrain: the margin is the field's certified
    /// one capped at 4 cells (sampled rises are at most 2 cells at every
    /// level, certified margins 13-28).
    ///
    /// Branch and bound: the region starts as a few coarse columns, and the
    /// column with the highest bound is split into its four children until
    /// that column is fine (level 3) or the split budget is spent.
    pub fn local_outer_radius(&self, eye: DVec3, radius: f64) -> f64 {
        const FINEST: u32 = 3;
        const SPLITS: usize = 192;
        let g = &self.grid;
        let global = self.outer_radius();
        // Smallest ground width of a base cell (equal-angle cube cells
        // shrink to 1/sqrt(2) of the centre width towards face edges).
        let base = if g.is_plane() { g.voxel_size() } else { g.delta() * g.radius() * 0.7 };
        // Coarsest level still resolving the region in a few cells. Field
        // queries exist at any level, beyond the world's resident ones (a
        // small plane has few): the region stays a few columns wide.
        let mut level = 0;
        while level + 1 < 24 && base * f64::from(1u32 << (level + 1)) * 3.0 < radius {
            level += 1;
        }
        let reach = (radius / (base * f64::from(1u32 << level))).ceil() as i32 + 1;
        let (cell, _) = g.locate(eye);
        let (ci, cj) = (cell.i >> level, cell.j >> level);
        let inside = |c: i32| c - reach >= 0 && ((i64::from(c + reach) + 1) << level) <= i64::from(g.cells());
        if !inside(ci) || !inside(cj) {
            return global;
        }
        let margins = self.field.bound_margins();
        // Highest surface (metres above the datum) any column inside a
        // level column can reach.
        let bound = |l: u32, a: i32, b: i32| {
            let margin = if l == 0 { 0 } else { margins[l as usize].min(4) };
            f64::from(self.column_top(cell.face, a, b, l) + margin) * g.level_size(l)
        };
        let mut heap = std::collections::BinaryHeap::new();
        for a in ci - reach..=ci + reach {
            for b in cj - reach..=cj + reach {
                heap.push((OrdF64(bound(level, a, b)), level, a, b));
            }
        }
        let mut splits = 0;
        let top = loop {
            let (OrdF64(top), l, a, b) = heap.pop().expect("region has columns");
            if l <= FINEST || splits == SPLITS {
                break top;
            }
            splits += 1;
            for (da, db) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                heap.push((OrdF64(bound(l - 1, a * 2 + da, b * 2 + db)), l - 1, a * 2 + da, b * 2 + db));
            }
        };
        let terrain = g.radius() + top + self.overhang_height() + g.voxel_size() * 4.0;
        terrain.max(self.edit_top + g.voxel_size()).min(global)
    }
}

/// Totally ordered f64 for heaps.
#[derive(Clone, Copy, PartialEq)]
struct OrdF64(f64);
impl Eq for OrdF64 {}
impl PartialOrd for OrdF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrdF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edits::{BrushOp, BrushShape};

    fn planet() -> Planet {
        Planet::new(PlanetRecipe::default()).unwrap()
    }

    /// Earth without caves and overhangs: a pure heightfield, for
    /// column-top invariants.
    fn heightfield(recipe: PlanetRecipe) -> Planet {
        let terrain = crate::layers::TerrainLayers::earth().heightfield().source(TerrainSource::default().seed);
        Planet::new(PlanetRecipe { terrain, ..recipe }).unwrap()
    }

    /// Diagnostic: surface material shares of a mountain flank as each
    /// level draws it (shading's coarse-cell rule), against level 0.
    #[test]
    #[ignore]
    fn material_shares_by_level() {
        use crate::terrain::{block_slope, material};
        let p = planet();
        let g = *p.grid();
        // Highest level-10 column near the harness spawn.
        let n = g.cells() >> 10;
        let mut best = (0, 0, i32::MIN);
        for a in 0..96 {
            for b in 0..96 {
                let u = (0.47 + 0.15 * (f64::from(a) / 95.0 * 2.0 - 1.0)).clamp(0.0, 0.999);
                let v = (0.53 + 0.15 * (f64::from(b) / 95.0 * 2.0 - 1.0)).clamp(0.0, 0.999);
                let (i, j) = ((u * f64::from(n)) as i32, (v * f64::from(n)) as i32);
                let h = p.column_top(2, i, j, 10);
                if h > best.2 {
                    best = (i, j, h);
                }
            }
        }
        let (ci, cj) = ((best.0 << 10) + 512, (best.1 << 10) + 512);
        let mut state = 0x1234_5678u64;
        let mut rand = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let class = |m: u32| match m & material::ID {
            material::STONE | material::DARK_STONE => 0,
            material::GRASS => 1,
            material::SNOW => 2,
            _ => 3,
        };
        let top_material = |level: u32, a: i32, b: i32, slope_at: Option<(i32, i32)>, point: Option<(i32, i32)>| {
            let top = p.column_top(2, a, b, level);
            let slope = match slope_at {
                Some((i, j)) => block_slope(|x, y| p.column_top(2, (i & !7) + x, (j & !7) + y, 0), i & 7, j & 7),
                None => block_slope(|x, y| p.column_top(2, (a & !7) + x, (b & !7) + y, level), a & 7, b & 7),
            };
            let q = match point {
                Some((i, j)) => g.domain_point(2, i, j, 0),
                None => g.domain_point(2, a, b, level),
            };
            p.field().ground_material(q, 0, (top << level) * g.layer_mm() as i32, 0, slope, (top - 1) << level)
        };
        let samples = 6000;
        let span = 40_000.0 / (g.delta() * g.radius());
        let points: Vec<(i32, i32)> = (0..samples)
            .map(|_| (ci + ((rand() - 0.5) * span) as i32, cj + ((rand() - 0.5) * span) as i32))
            .collect();
        eprintln!("rock grass snow other (fractions) over a 40 km square around the summit");
        for level in 0..10u32 {
            let mut shares = [[0usize; 4]; 4];
            for &(i, j) in &points {
                let (a, b) = (i >> level, j >> level);
                shares[0][class(top_material(level, a, b, None, None))] += 1;
                shares[1][class(top_material(level, a, b, Some((i, j)), None))] += 1;
                shares[2][class(top_material(level, a, b, None, Some((i, j))))] += 1;
                shares[3][class(top_material(level, a, b, Some((i, j)), Some((i, j))))] += 1;
            }
            let f = |s: [usize; 4]| format!("{:.3} {:.3} {:.3} {:.3}", s[0] as f64 / samples as f64, s[1] as f64 / samples as f64, s[2] as f64 / samples as f64, s[3] as f64 / samples as f64);
            eprintln!("L{level}: as-is {} | fine slope {} | fine point {} | both {}", f(shares[0]), f(shares[1]), f(shares[2]), f(shares[3]));
        }
    }

    /// The local terrain bound holds for every base column sampled in the
    /// region, and is far below the planet's peak over lowland.
    #[test]
    fn local_outer_radius_bounds_the_terrain_around_the_eye() {
        let p = planet();
        let g = *p.grid();
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut rand = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut below_peak = 0;
        for n in 0..24 {
            let dir = DVec3::new(rand() - 0.5, rand() - 0.5, rand() - 0.5).normalize();
            let radius = [150.0, 600.0, 2_500.0, 20_000.0][n % 4];
            let eye = p.surface_point(dir, 500.0);
            let bound = p.local_outer_radius(eye, radius);
            assert!(bound <= p.outer_radius());
            below_peak += usize::from(bound < p.outer_radius() - 500.0);
            let up = eye.normalize();
            let east = up.cross(DVec3::Y).try_normalize().unwrap_or(DVec3::X);
            let north = up.cross(east);
            for _ in 0..400 {
                let (a, r) = (rand() * std::f64::consts::TAU, radius * rand().sqrt());
                let ground = (up * g.radius() + (east * a.cos() + north * a.sin()) * r).normalize();
                let (cell, _) = g.locate(ground * g.radius());
                let top = g.radius() + f64::from(p.column_top(cell.face, cell.i, cell.j, 0)) * g.voxel_size();
                assert!(top <= bound, "{dir} r {radius}: column top {top} above bound {bound}");
            }
        }
        assert!(below_peak >= 12, "the bound should be local ({below_peak}/24 below the peak)");
    }

    #[test]
    fn ground_height_is_altitude_above_the_column_below() {
        let p = heightfield(PlanetRecipe::default());
        for dir in [DVec3::new(0.1, 1.0, 0.2), DVec3::new(0.9, 0.4, -0.3), DVec3::new(-0.2, -0.7, 0.8)] {
            for h in [0.5, 12.0, 3000.0] {
                let eye = p.surface_point(dir, h);
                let measured = p.ground_height(eye);
                assert!((measured - h).abs() <= p.grid().voxel_size() * 1.01, "{dir} {h}: {measured}");
                // A conservative clearance never exceeds it.
                assert!(p.air_clearance(eye) <= measured + p.grid().voxel_size());
            }
        }
    }

    #[test]
    fn terrain_bounds_hold_for_sampled_cells() {
        let flat = crate::layers::TerrainLayers::flat_at(3.3).source(7);
        for (terrain, voxel) in [(TerrainSource::default(), 0.1), (TerrainSource::default(), 0.3), (TerrainSource::default(), 1.0), (flat, 0.1)] {
            let p = Planet::new(PlanetRecipe { voxel_size_m: voxel, terrain: terrain.clone(), ..Default::default() }).unwrap();
            let worst = terrain::check_field(&p, 6_000).unwrap_or_else(|e| panic!("{} at {voxel} m: {e}", terrain.generator));
            eprintln!("{} {voxel}: margins {:?}\nworst {worst:?}", terrain.generator, &p.field().bound_margins()[..p.grid().levels() as usize]);
        }
    }

    fn plane(shape: Shape) -> Planet {
        heightfield(PlanetRecipe { shape, plane_size_m: 3_000.0, ..Default::default() })
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

    /// A point placed above the ground reads that height, wherever the
    /// ground is (overhang lips and cave mouths included).
    #[test]
    fn ground_height_is_height_above_the_generated_ground() {
        let p = planet();
        for dir in [DVec3::Y, DVec3::new(0.3, 1.0, -0.2), DVec3::new(-0.7, 0.2, 0.68), DVec3::new(0.1, -0.4, 0.9)] {
            let eye = p.surface_point(dir.normalize(), 2.0);
            let h = p.ground_height(eye);
            assert!((h - 2.0).abs() < 0.2, "{dir}: {h}");
        }
    }

    #[test]
    fn raycast_down_hits_the_column_top() {
        let p = heightfield(PlanetRecipe::default());
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
        p.apply(Brush { center: centre.to_array(), radius: 0.25, shape: BrushShape::Cube, op: BrushOp::Add, material: material::BRICK }).unwrap();
        assert!(p.solid(Cell::new(face, i, j, top - 1)));
        assert_eq!(p.material(Cell::new(face, i, j, top - 1)), material::BRICK);
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

#[cfg(test)]
mod scaling {
    use super::*;
    use crate::edits::{BrushOp, BrushShape};

    /// Cost of what sculpting does per stamp, as edits pile up in one area
    /// (editor strokes): copy the world, apply a stamp, cast the editor's
    /// aim ray, find the ground. Run with `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn sculpt_query_costs() {
        let mut planet = Planet::new(PlanetRecipe::default()).unwrap();
        let ground = planet.surface_point(DVec3::new(0.3, 1.0, 0.2), 0.0);
        let up = ground.normalize();
        let side = up.any_orthonormal_vector();
        let ahead = up.cross(side);
        let eye = ground + up * 8.0;
        let time = |f: &mut dyn FnMut()| {
            let t = std::time::Instant::now();
            for _ in 0..20 {
                f();
            }
            t.elapsed().as_secs_f64() * 1000.0 / 20.0
        };
        let mut stamp = 0usize;
        for target in [0usize, 100, 400, 1000, 2000] {
            while stamp < target {
                let angle = stamp as f64 * 0.6 / 6.0;
                let p = ground + (side * angle.cos() + ahead * angle.sin()) * 6.0;
                planet.apply(Brush { center: (p - up * 0.3).to_array(), radius: 1.0, shape: BrushShape::Sphere, op: BrushOp::Remove, material: 0 }).unwrap();
                stamp += 1;
            }
            let aim = (ground + side * 6.0 - eye).normalize();
            let clone = time(&mut || drop(std::hint::black_box(planet.clone())));
            let mut copy = planet.clone();
            let apply = time(&mut || {
                copy.apply(Brush { center: (ground - up * 0.3).to_array(), radius: 1.0, shape: BrushShape::Sphere, op: BrushOp::Remove, material: 0 }).unwrap();
            });
            let raycast = time(&mut || drop(std::hint::black_box(planet.raycast(eye, aim, 200.0))));
            let surface = time(&mut || drop(std::hint::black_box(planet.surface_point(ground + side * 6.0, 0.0))));
            let (cell, _) = planet.grid().locate(ground + side * 6.0 - up * 0.5);
            let solid = time(&mut || drop(std::hint::black_box(planet.solid(cell))));
            eprintln!("SCULPT_COST {stamp:5} edits: clone {clone:.3} ms, apply {apply:.3} ms, aim raycast {raycast:.3} ms, surface_point {surface:.3} ms, solid {solid:.4} ms");
        }
    }

    /// Cost of copying a world with many edits (what a shared world pays per
    /// appended edit). Run with `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn clone_cost_with_many_edits() {
        let mut planet = Planet::new(PlanetRecipe { shape: Shape::Plane, plane_size_m: 4096.0, ..Default::default() }).unwrap();
        for n in [1_000usize, 10_000, 50_000] {
            let appending = std::time::Instant::now();
            let before = planet.edits().len();
            while planet.edits().len() < n {
                let k = planet.edits().len() as f64;
                let p = DVec3::new((k * 7.31) % 900.0 - 450.0, 2.0, (k * 3.17) % 900.0 - 450.0);
                planet.apply(Brush { center: p.to_array(), radius: 0.05, shape: BrushShape::Cube, op: BrushOp::Add, material: 13 }).unwrap();
            }
            let per_edit = appending.elapsed().as_secs_f64() * 1e6 / (n - before) as f64;
            let t = std::time::Instant::now();
            let copies = 20;
            for _ in 0..copies {
                std::hint::black_box(planet.clone());
            }
            eprintln!("{n} edits: clone {:.3} ms, append {per_edit:.1} us/edit", t.elapsed().as_secs_f64() * 1000.0 / copies as f64);
        }
    }
}
