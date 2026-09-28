//! Level windows: which columns each clipmap level wants resident.
//!
//! Computing a window scans every candidate column of a level (hundreds of
//! thousands on a planet), so the planner runs on a background thread and
//! reports incremental add/remove diffs. The render thread only applies them.
use crate::grid::{face_axes, Grid, BRICK};
use crate::residency::{key0, max_window_columns};
use glam::DVec3;
use rustc_hash::FxHashSet;
use std::sync::{mpsc, Mutex};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WindowRequest {
    pub eye: DVec3,
    /// Level-0 range (metres).
    pub lod0: f64,
    /// Radius bounding every solid cell.
    pub outer_radius: f64,
    pub serial: u64,
}

#[derive(Default, Debug)]
pub struct LevelDiff {
    pub level: u32,
    pub active: bool,
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

    fn scan(&self, level: u32, dir: DVec3, radius: f64) -> Vec<(f32, u64)> {
        let grid = self.grid;
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

    /// Diff every level against the request. Levels whose window has not
    /// moved enough are left untouched (hysteresis of three columns).
    pub fn update(&mut self, request: &WindowRequest) -> WindowUpdate {
        let started = std::time::Instant::now();
        let grid = self.grid;
        let r0 = grid.radius();
        let eye = request.eye;
        let dir = eye.normalize();
        let altitude = eye.length() - request.outer_radius;
        let height = (eye.length() - r0).max(0.0);
        let peak = request.outer_radius - r0;
        // Farthest terrain that can rise above the horizon.
        let horizon = (2.0 * r0 * height + height * height).sqrt() + (2.0 * r0 * peak + peak * peak).sqrt();
        let top_level = grid.levels() - 1;
        let mut update = WindowUpdate {
            serial: request.serial,
            ..Default::default()
        };
        for level in 0..grid.levels() {
            let reach = request.lod0 * f64::from(1u32 << level) * 1.05;
            let inner = if level == 0 { 0.0 } else { request.lod0 * f64::from(1u32 << (level - 1)) };
            let needed = level == top_level || (altitude < reach && inner < horizon);
            let state = &mut self.levels[level as usize];
            if !needed {
                if state.active {
                    update.levels.push(LevelDiff {
                        level,
                        active: false,
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
                r0 * 4.0
            } else {
                ((reach * reach - altitude.max(0.0).powi(2)).max(0.0).sqrt().min(horizon) + col * 2.0).min(cap)
            };
            let moved = state.center.distance(dir) * r0;
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
