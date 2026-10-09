//! Level windows: which columns each clipmap level wants resident.
//!
//! Computing a window scans every candidate column of a level (hundreds of
//! thousands on a planet), so the planner runs on a background thread and
//! reports incremental add/remove diffs. The render thread only applies them.
use crate::grid::{face_axes, Grid, BRICK, PLANE_FACE};
use crate::residency::{key0, max_window_columns};
use glam::DVec3;
use rustc_hash::FxHashSet;
use std::sync::{mpsc, Mutex};

#[derive(Clone)]
pub struct WindowRequest {
    pub eye: DVec3,
    /// Level-0 range (metres).
    pub lod0: f64,
    /// Relative width of the traversal's stochastic level transition.
    pub lod_dither: f64,
    /// Radius bounding every solid cell.
    pub outer_radius: f64,
    /// The world, for local terrain bounds (none: the global bound only).
    pub planet: Option<std::sync::Arc<crate::planet::Planet>>,
    pub serial: u64,
}

#[derive(Default, Debug)]
pub struct LevelDiff {
    pub level: u32,
    pub active: bool,
    /// Window centre (the eye's ground point) and ground radius in metres:
    /// every column whose centre lies within the radius is wanted.
    pub center: DVec3,
    pub radius: f64,
    /// Newly wanted columns with their normalized distance (lower is sooner).
    pub adds: Vec<(f32, u64)>,
    pub removes: Vec<u64>,
}

#[derive(Default, Debug)]
pub struct WindowUpdate {
    pub serial: u64,
    pub levels: Vec<LevelDiff>,
    pub planning_ms: f64,
}

#[derive(Default)]
struct LevelState {
    active: bool,
    center: DVec3,
    radius: f64,
    wanted: FxHashSet<u64>,
    /// Last local terrain bound: where, over what ground radius, the bound,
    /// and the world's outer radius then (edits change it).
    bound: Option<(DVec3, f64, f64, f64)>,
}

pub struct WindowPlanner {
    grid: Grid,
    levels: Vec<LevelState>,
    lod_dither: Option<f64>,
}

fn pack(k0: u32, k1: u32) -> u64 {
    u64::from(k0) | (u64::from(k1) << 32)
}

/// The same finite selection range is used by tracing and window planning.
pub(crate) fn sanitize_lod_dither(value: f64) -> f64 {
    if value.is_finite() { value.clamp(0.0, 1.0) } else { 0.25 }
}

/// Whether the column nearest a plane window's centre inside block
/// `[x0, x1] x [y0, y1]` (clipped to the scan bounds) is within `limit`.
#[allow(clippy::too_many_arguments)]
fn plane_block_in_circle(ci: f64, cj: f64, x0: i32, x1: i32, y0: i32, y1: i32,
    lo_i: i32, hi_i: i32, lo_j: i32, hi_j: i32, limit: f64) -> bool {
    if x0.max(lo_i) > x1.min(hi_i) || y0.max(lo_j) > y1.min(hi_j) { return false; }
    let x = (ci.floor() as i32).clamp(x0.max(lo_i), x1.min(hi_i));
    let y = (cj.floor() as i32).clamp(y0.max(lo_j), y1.min(hi_j));
    (f64::from(x) + 0.5 - ci).hypot(f64::from(y) + 0.5 - cj) <= limit
}

fn column_in_circle(dn: f64, da: f64, db: f64, ta: f64, tb: f64, cos_limit: f64) -> bool {
    (dn + ta * da + tb * db) / (1.0 + ta * ta + tb * tb).sqrt() >= cos_limit
}

impl WindowPlanner {
    pub fn new(grid: Grid) -> Self {
        Self {
            grid,
            levels: (0..grid.levels()).map(|_| LevelState::default()).collect(),
            lod_dither: None,
        }
    }

    /// Columns of a plane level within `radius` of ground point `center`.
    fn scan_plane(&self, level: u32, center: DVec3, radius: f64) -> Vec<(f32, u64)> {
        let grid = self.grid;
        let col_cells = BRICK << level;
        let cols = grid.cells() / col_cells;
        let col = grid.level_size(level) * f64::from(BRICK);
        let c = grid.face_coords(PLANE_FACE, center).unwrap_or([0.0; 3]);
        let (ci, cj) = (c[0] / f64::from(col_cells), c[1] / f64::from(col_cells));
        let reach = (radius / col + 1.0).min(f64::from(cols));
        let lo_i = ((ci - reach).floor() as i32).max(0);
        let hi_i = ((ci + reach).ceil() as i32).min(cols - 1);
        let lo_j = ((cj - reach).floor() as i32).max(0);
        let hi_j = ((cj + reach).ceil() as i32).min(cols - 1);
        let limit = radius / col + 0.75;
        let mut out = Vec::new();
        if lo_i > hi_i || lo_j > hi_j { return out; }
        // Traversal admits complete tier-1 blocks. Expand only blocks that
        // intersect the original column-centre circle: partial boundary
        // blocks would otherwise stay unusable even after settling.
        for bj in lo_j / 4..=hi_j / 4 {
            let (y0, y1) = (bj * 4, (bj * 4 + 3).min(cols - 1));
            for bi in lo_i / 4..=hi_i / 4 {
                let (x0, x1) = (bi * 4, (bi * 4 + 3).min(cols - 1));
                if !plane_block_in_circle(ci, cj, x0, x1, y0, y1, lo_i, hi_i, lo_j, hi_j, limit) { continue; }
                // One priority per block also keeps admission grouped.
                let priority = ((f64::from(x0 + x1 + 1) * 0.5 - ci)
                    .hypot(f64::from(y0 + y1 + 1) * 0.5 - cj) / limit.max(1e-12)) as f32;
                for y in y0..=y1 {
                    for x in x0..=x1 {
                        out.push((priority, pack(key0(PLANE_FACE, level, x), y as u32)));
                    }
                }
            }
        }
        out
    }

    fn scan(&self, level: u32, dir: DVec3, radius: f64) -> Vec<(f32, u64)> {
        let grid = self.grid;
        if grid.is_plane() {
            return self.scan_plane(level, dir, radius);
        }
        let r0 = grid.radius();
        let col_cells = BRICK << level;
        let cols = grid.cells() / col_cells;
        let col_angle = grid.delta() * f64::from(col_cells);
        let theta = (radius / r0).min(std::f64::consts::PI);
        let cos_limit = (theta + col_angle * 0.75).min(std::f64::consts::PI).cos();
        let mut out = Vec::new();
        for face in 0..6u8 {
            let [n, a, b] = face_axes(face);
            let dn = dir.dot(n);
            if theta < 1.2 && dn < (theta + 1.0).min(std::f64::consts::PI).cos() {
                continue;
            }
            let (lo_i, hi_i, lo_j, hi_j) = if dn > 0.2 && theta < 0.9 {
                let ai = grid.index_of_angle(dir.dot(a).atan2(dn)) / f64::from(col_cells);
                let bj = grid.index_of_angle(dir.dot(b).atan2(dn)) / f64::from(col_cells);
                let reach = theta / col_angle / 0.7 + 2.0;
                (
                    ((ai - reach).floor() as i32).max(0),
                    ((ai + reach).ceil() as i32).min(cols - 1),
                    ((bj - reach).floor() as i32).max(0),
                    ((bj + reach).ceil() as i32).min(cols - 1),
                )
            } else {
                (0, cols - 1, 0, cols - 1)
            };
            if lo_i > hi_i || lo_j > hi_j {
                continue;
            }
            let block_lo_i = lo_i & !3;
            let block_hi_i = (hi_i | 3).min(cols - 1);
            let block_lo_j = lo_j & !3;
            let block_hi_j = (hi_j | 3).min(cols - 1);
            let tan_i: Vec<f64> = (block_lo_i..=block_hi_i)
                .map(|c| grid.angle((f64::from(c) + 0.5) * f64::from(col_cells)).tan())
                .collect();
            let tan_j: Vec<f64> = (block_lo_j..=block_hi_j)
                .map(|c| grid.angle((f64::from(c) + 0.5) * f64::from(col_cells)).tan())
                .collect();
            let (da, db) = (dir.dot(a), dir.dot(b));
            // Whole tier-1 blocks, as on a plane: every block with a column
            // centre inside the circle is wanted complete.
            for bj in lo_j / 4..=hi_j / 4 {
                let (y0, y1) = (bj * 4, (bj * 4 + 3).min(cols - 1));
                for bi in lo_i / 4..=hi_i / 4 {
                    let (x0, x1) = (bi * 4, (bi * 4 + 3).min(cols - 1));
                    let mut intersects = false;
                    'columns: for y in y0.max(lo_j)..=y1.min(hi_j) {
                        let tb = tan_j[(y - block_lo_j) as usize];
                        for x in x0.max(lo_i)..=x1.min(hi_i) {
                            let ta = tan_i[(x - block_lo_i) as usize];
                            if column_in_circle(dn, da, db, ta, tb, cos_limit) {
                                intersects = true;
                                break 'columns;
                            }
                        }
                    }
                    if !intersects { continue; }
                    let ta = grid.angle(f64::from(x0 + x1 + 1) * 0.5 * f64::from(col_cells)).tan();
                    let tb = grid.angle(f64::from(y0 + y1 + 1) * 0.5 * f64::from(col_cells)).tan();
                    let cos = (dn + ta * da + tb * db) / (1.0 + ta * ta + tb * tb).sqrt();
                    let priority = (cos.clamp(-1.0, 1.0).acos() / theta.max(1e-12)) as f32;
                    for y in y0..=y1 {
                        for x in x0..=x1 {
                            out.push((priority, pack(key0(face, level, x), y as u32)));
                        }
                    }
                }
            }
        }
        out
    }

    /// Radial bound of the terrain within a level's reach of the eye's
    /// ground point (see [`crate::planet::Planet::local_outer_radius`]).
    /// Computed only where it can matter: a level far above the highest
    /// terrain is off anyway, and one reaching far beyond the eye's height
    /// over the lowest terrain gets nearly the same window from the global
    /// bound. A bound over a 20 % larger region stays valid while the eye
    /// moves 20 % of the reach.
    fn local_outer(&mut self, request: &WindowRequest, level: u32, reach: f64) -> f64 {
        let grid = self.grid;
        let Some(planet) = &request.planet else { return request.outer_radius };
        let reach = reach + grid.level_size(level) * f64::from(BRICK) * 3.0;
        let height = grid.height(request.eye);
        if height - reach > planet.max_terrain_height()
            || reach > 4.0 * (height - planet.min_terrain_height()).max(0.0)
            || (!grid.is_plane() && reach > grid.radius() * 0.25)
        {
            return request.outer_radius;
        }
        let eye = request.eye;
        let state = &mut self.levels[level as usize];
        let stale = state.bound.is_none_or(|(at, radius, _, outer)| {
            grid.ground_distance(at, eye) > radius - reach || outer != request.outer_radius
        });
        if stale {
            let radius = reach * 1.2;
            state.bound = Some((eye, radius, planet.local_outer_radius(eye, radius), request.outer_radius));
        }
        state.bound.unwrap().2.min(request.outer_radius)
    }

    /// Diff every level against the request. Levels whose window has not
    /// moved enough are left untouched (hysteresis of three columns).
    pub fn update(&mut self, request: &WindowRequest) -> WindowUpdate {
        let started = std::time::Instant::now();
        let grid = self.grid;
        let r0 = grid.radius();
        let eye = request.eye;
        // Window centre: the eye direction on a sphere, its ground point on a plane.
        let dir = if grid.is_plane() { DVec3::new(eye.x, 0.0, eye.z) } else { eye.normalize() };
        let dither = sanitize_lod_dither(request.lod_dither);
        // A dither change moves every level's reach: rescan all of them.
        let dither_changed = self.lod_dither != Some(dither);
        self.lod_dither = Some(dither);
        // Below-datum terrain still has a horizon. Use the datum sphere as
        // the conservative radius, with nonnegative clearance and peak.
        let height = (grid.radial(eye) - r0.min(request.outer_radius)).max(0.0);
        let peak = (request.outer_radius - r0).max(0.0);
        // Farthest terrain that can rise above the horizon (none on a plane).
        let horizon = if grid.is_plane() {
            f64::INFINITY
        } else {
            (2.0 * r0 * height + height * height).sqrt() + (2.0 * r0 * peak + peak * peak).sqrt()
        };
        let top_level = grid.levels() - 1;
        let mut update = WindowUpdate {
            serial: request.serial,
            ..Default::default()
        };
        for level in 0..grid.levels() {
            let nominal = request.lod0 * f64::from(1u32 << level);
            // Traversal selects levels at t * (1 + d * (hash - 0.5)): a level
            // can be chosen out to nominal / (1 - d/2), and the next one can
            // start correspondingly early. A window ending at 1.05x nominal
            // left the dithered outer band to fall back to coarser levels.
            let selected_reach = nominal / (1.0 - dither * 0.5);
            let reach = (nominal * 1.05).max(selected_reach);
            let inner = if level == 0 { 0.0 } else {
                request.lod0 * f64::from(1u32 << (level - 1)) / (1.0 + dither * 0.5)
            };
            // Height over the highest terrain the level's window can hold:
            // over a meadow far below, fine levels are not needed at all.
            let altitude = grid.radial(eye) - self.local_outer(request, level, reach);
            let needed = level == top_level || (altitude < reach && inner < horizon);
            let state = &mut self.levels[level as usize];
            if !needed {
                if state.active {
                    update.levels.push(LevelDiff {
                        level,
                        active: false,
                        center: dir,
                        radius: 0.0,
                        adds: Vec::new(),
                        removes: state.wanted.drain().collect(),
                    });
                    state.active = false;
                }
                continue;
            }
            let col = grid.level_size(level) * f64::from(BRICK);
            // The direct-mapped summary tables bound the window diameter.
            let cap = col * f64::from(max_window_columns() / 2 - 2) * 0.8;
            // Recentring tolerates a three-column displacement (on a sphere a
            // chord: converted to a conservative arc). The scan admits centres
            // 0.75 columns further out and whole blocks; the slack covers the
            // rest of the centre hysteresis.
            let drift = if grid.is_plane() { col * 3.0 } else {
                2.0 * r0 * (col * 3.0 / (2.0 * r0)).min(1.0).asin()
            };
            let tangential_col = grid.delta() * f64::from(BRICK << level) * if grid.is_plane() { 1.0 } else { r0 };
            let slack = tangential_col * 0.25 + col * 0.25;
            let pad = (col * 2.0).max(drift + slack - (reach - selected_reach));
            let radius = if level == top_level {
                // The coarsest level covers the whole world.
                if grid.is_plane() { f64::from(grid.cells()) * grid.voxel_size() * 1.5 } else { r0 * 4.0 }
            } else {
                ((reach * reach - altitude.max(0.0).powi(2)).max(0.0).sqrt().min(horizon) + pad).min(cap)
            };
            let moved = if grid.is_plane() { state.center.distance(dir) } else { state.center.distance(dir) * r0 };
            if state.active && !dither_changed && moved <= col * 3.0 && (radius - state.radius).abs() <= state.radius * 0.08 + col {
                continue;
            }
            let scanned = self.scan(level, dir, radius);
            let state = &mut self.levels[level as usize];
            let mut next = FxHashSet::with_capacity_and_hasher(scanned.len(), Default::default());
            let mut adds = Vec::new();
            for (priority, key) in scanned {
                if !state.wanted.contains(&key) {
                    adds.push((priority, key));
                }
                next.insert(key);
            }
            let removes = state.wanted.iter().filter(|k| !next.contains(k)).copied().collect();
            state.wanted = next;
            state.active = true;
            state.center = dir;
            state.radius = radius;
            update.levels.push(LevelDiff {
                level,
                active: true,
                center: dir,
                radius,
                adds,
                removes,
            });
        }
        update.planning_ms = started.elapsed().as_secs_f64() * 1000.0;
        update
    }
}

/// Background planner: always works on the most recent request.
pub struct WindowWorker {
    requests: Option<mpsc::Sender<WindowRequest>>,
    updates: Mutex<mpsc::Receiver<WindowUpdate>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl WindowWorker {
    pub fn start(grid: Grid) -> Self {
        let (request_tx, request_rx) = mpsc::channel::<WindowRequest>();
        let (update_tx, update_rx) = mpsc::channel();
        let thread = std::thread::Builder::new()
            .name("voxel-planet-windows".into())
            .spawn(move || {
                let mut planner = WindowPlanner::new(grid);
                while let Ok(mut request) = request_rx.recv() {
                    while let Ok(newer) = request_rx.try_recv() {
                        request = newer;
                    }
                    if update_tx.send(planner.update(&request)).is_err() {
                        break;
                    }
                }
            })
            .expect("spawn window planner");
        Self {
            requests: Some(request_tx),
            updates: Mutex::new(update_rx),
            thread: Some(thread),
        }
    }
    pub fn request(&self, request: WindowRequest) {
        if let Some(tx) = &self.requests {
            let _ = tx.send(request);
        }
    }
    pub fn try_update(&self) -> Option<WindowUpdate> {
        self.updates.lock().ok()?.try_recv().ok()
    }
}

impl Drop for WindowWorker {
    fn drop(&mut self) {
        self.requests = None;
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planet::{Planet, PlanetRecipe};

    /// Traversal uses a level only inside complete 4x4-column blocks, so
    /// every wanted column brings its whole block (clipped to the face).
    fn assert_complete_blocks(grid: Grid, update: &WindowUpdate) {
        for diff in &update.levels {
            let wanted: FxHashSet<u64> = diff.adds.iter().map(|(_, key)| *key).collect();
            assert_eq!(wanted.len(), diff.adds.len(), "a column must be wanted once");
            for &key in &wanted {
                let k0 = key as u32;
                let (face, level, i, j) = (((k0 >> 24) & 7) as u8, k0 >> 27, (k0 & 0xff_ffff) as i32, (key >> 32) as i32);
                let cols = grid.cells() / (BRICK << level);
                for y in (j & !3)..=((j | 3).min(cols - 1)) {
                    for x in (i & !3)..=((i | 3).min(cols - 1)) {
                        assert!(wanted.contains(&pack(key0(face, level, x), y as u32)),
                            "L{level} column ({i},{j}) wanted without ({x},{y}) of its block");
                    }
                }
            }
        }
    }

    #[test]
    fn windows_want_complete_blocks_on_spheres_and_planes() {
        for recipe in [PlanetRecipe::default(), PlanetRecipe { shape: crate::grid::Shape::Plane, plane_size_m: 3_000.0, ..Default::default() }] {
            let planet = std::sync::Arc::new(Planet::new(recipe).unwrap());
            let grid = *planet.grid();
            let ground = planet.surface_point(grid.direction(2, 3e7, 4e7), 1.8);
            let ground = if grid.is_plane() { planet.surface_point(DVec3::new(37.3, 0.0, -81.9), 1.8) } else { ground };
            for (eye, lod0) in [(ground, 9.0), (ground + grid.up(ground) * 400.0, 31.0)] {
                let request = WindowRequest { eye, lod0, lod_dither: 0.25, outer_radius: planet.outer_radius(), planet: Some(planet.clone()), serial: 1 };
                let mut fresh = WindowPlanner::new(grid);
                assert_complete_blocks(grid, &fresh.update(&request));
            }
        }
    }
}