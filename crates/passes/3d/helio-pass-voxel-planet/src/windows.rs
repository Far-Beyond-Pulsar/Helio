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
    /// A bounded motion forecast. Coverage metadata remains centred on eye.
    pub prefetch_eye: Option<DVec3>,
    /// Level-0 range (metres).
    pub lod0: f64,
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
}

fn pack(k0: u32, k1: u32) -> u64 {
    u64::from(k0) | (u64::from(k1) << 32)
}

impl WindowPlanner {
    pub fn new(grid: Grid) -> Self {
        Self {
            grid,
            levels: (0..grid.levels()).map(|_| LevelState::default()).collect(),
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
        for y in lo_j..=hi_j {
            for x in lo_i..=hi_i {
                let d = (f64::from(x) + 0.5 - ci).hypot(f64::from(y) + 0.5 - cj);
                if d <= limit {
                    out.push(((d / limit.max(1e-12)) as f32, pack(key0(PLANE_FACE, level, x), y as u32)));
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
            let tan_i: Vec<f64> = (lo_i..=hi_i)
                .map(|c| grid.angle((f64::from(c) + 0.5) * f64::from(col_cells)).tan())
                .collect();
            let (da, db) = (dir.dot(a), dir.dot(b));
            for cj in lo_j..=hi_j {
                let tb = grid.angle((f64::from(cj) + 0.5) * f64::from(col_cells)).tan();
                for (x, &ta) in tan_i.iter().enumerate() {
                    let cos = (dn + ta * da + tb * db) / (1.0 + ta * ta + tb * tb).sqrt();
                    if cos < cos_limit {
                        continue;
                    }
                    let angle = cos.clamp(-1.0, 1.0).acos();
                    let key = pack(key0(face, level, lo_i + x as i32), cj as u32);
                    out.push(((angle / theta.max(1e-12)) as f32, key));
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
        let height = (grid.radial(eye) - r0).max(0.0);
        let peak = request.outer_radius - r0;
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
            let reach = request.lod0 * f64::from(1u32 << level) * 1.05;
            let inner = if level == 0 { 0.0 } else { request.lod0 * f64::from(1u32 << (level - 1)) };
            // Height over the highest terrain the level's window can hold:
            // over a meadow far below, fine levels are not needed at all.
            let altitude = grid.radial(eye) - self.local_outer(request, level, reach);
            let future = request.prefetch_eye.unwrap_or(eye);
            let future_altitude = grid.radial(future) - self.local_outer(request, level, reach);
            let needed = level == top_level || (altitude.min(future_altitude) < reach && inner < horizon);
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
            let radius = if level == top_level {
                // The coarsest level covers the whole world.
                if grid.is_plane() { f64::from(grid.cells()) * grid.voxel_size() * 1.5 } else { r0 * 4.0 }
            } else {
                let future_reach = (reach * reach - future_altitude.max(0.0).powi(2)).max(0.0).sqrt();
                let motion = grid.ground_distance(eye, future);
                ((reach * reach - altitude.max(0.0).powi(2)).max(0.0).sqrt()
                    .max(future_reach + motion.min(reach * 0.5)).min(horizon) + col * 2.0).min(cap)
            };
            let moved = if grid.is_plane() { state.center.distance(dir) } else { state.center.distance(dir) * r0 };
            if state.active && moved <= col * 3.0 && (radius - state.radius).abs() <= state.radius * 0.08 + col {
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
    use crate::{Planet, PlanetRecipe, TerrainSource};

    #[test]
    fn descending_forecast_admits_fine_windows_before_arrival_and_releases_them_at_stop() {
        let planet = std::sync::Arc::new(Planet::new(PlanetRecipe {
            shape: crate::grid::Shape::Plane,
            plane_size_m: 1000.0,
            terrain: TerrainSource { generator: crate::landform::FLAT_ID.into(), ..Default::default() },
            ..Default::default()
        }).unwrap());
        let mut planner = WindowPlanner::new(*planet.grid());
        let mut request = WindowRequest {
            eye: DVec3::Y * 1000.0,
            prefetch_eye: None,
            lod0: 100.0,
            outer_radius: planet.outer_radius(),
            planet: Some(planet), serial: 1,
        };
        assert!(!planner.update(&request).levels.iter().any(|l| l.level == 0 && l.active));
        request.prefetch_eye = Some(DVec3::Y * 40.0);
        request.serial += 1;
        let update = planner.update(&request);
        let fine = update.levels.iter().find(|l| l.level == 0).unwrap();
        assert!(fine.active && !fine.adds.is_empty());
        request.prefetch_eye = None;
        request.serial += 1;
        let update = planner.update(&request);
        let fine = update.levels.iter().find(|l| l.level == 0).unwrap();
        assert!(!fine.active && !fine.removes.is_empty());
    }
}
