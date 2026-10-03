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

/// The same finite selection range is used by tracing and window planning.
pub(crate) fn sanitize_lod_dither(value: f64) -> f64 {
    if value.is_finite() { value.clamp(0.0, 1.0) } else { 0.25 }
}

#[derive(Clone)]
pub struct WindowRequest {
    pub eye: DVec3,
    /// A bounded motion forecast. Coverage metadata remains centred on eye.
    pub prefetch_eye: Option<DVec3>,
    /// Priority-only anchors; neither changes the wanted window.
    pub priority_eye: Option<DVec3>,
    pub view_focus: Option<DVec3>,
    /// Level-0 range (metres).
    pub lod0: f64,
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
    /// Exact latest demand, shared with the planner without copying its keys.
    /// Lets bounded admission bypass obsolete diffs during continuous motion.
    pub wanted: Vec<(u32, std::sync::Arc<FxHashSet<u64>>)>,
    /// Worker updates replace full demand instead of retaining camera history.
    pub snapshot: bool,
}

#[derive(Default)]
struct LevelState {
    active: bool,
    center: DVec3,
    radius: f64,
    wanted: std::sync::Arc<FxHashSet<u64>>,
    /// Last local terrain bound: where, over what ground radius, the bound,
    /// and the world's outer radius then (edits change it).
    bound: Option<(DVec3, f64, f64, f64)>,
}

pub struct WindowPlanner {
    grid: Grid,
    levels: Vec<LevelState>,
    snapshot: bool,
    lod_dither: Option<f64>,
}

fn pack(k0: u32, k1: u32) -> u64 {
    u64::from(k0) | (u64::from(k1) << 32)
}

/// Retire whole summary owners before spreading removals across other
/// blocks. All three tiers must release their references before an incoming
/// block can reuse a toroidal slot. Membership and delta order stay intact.
pub(crate) fn order_removes_by_blocks(mut removes: Vec<u64>) -> Vec<u64> {
    removes.sort_by_key(|key| {
        let header = *key as u32 >> 24;
        let ci = (*key as u32 & 0xffffff) as i32;
        let cj = (*key >> 32) as i32;
        (header, ci >> 6, cj >> 6, ci >> 4, cj >> 4, ci >> 2, cj >> 2, ci, cj)
    });
    removes
}

/// The render thread applies only a bounded prefix of a diff per frame.
/// Admit nearby complete blocks before peripheral rows reach that prefix.
/// Bucket scanner-emitted blocks, then sort each bucket's block descriptors:
/// moving crescents often share the last bucket, despite differing urgency.
/// The background planner sorts blocks rather than individual columns.
fn order_adds_by_priority(adds: Vec<(f32, u64)>) -> Vec<(f32, u64)> {
    const BUCKETS: usize = 64;
    let bucket = |priority: f32| ((priority.max(0.0) * BUCKETS as f32) as usize).min(BUCKETS - 1);
    let block = |key: u64| (key as u32 & 0xff000000, (key as u32 & 0xffffff) >> 2, (key >> 34) as u32);
    let mut runs = Vec::new();
    let mut at = 0;
    while at < adds.len() {
        let start = at;
        let identity = block(adds[start].1);
        at += 1;
        while at < adds.len() && block(adds[at].1) == identity { at += 1; }
        runs.push((adds[start].0, start, at));
    }
    let mut offsets = [0usize; BUCKETS];
    for &(priority, _, _) in &runs { offsets[bucket(priority)] += 1; }
    let mut start = 0;
    for offset in &mut offsets {
        let count = *offset;
        *offset = start;
        start += count;
    }
    let mut grouped = vec![(0.0, 0, 0); runs.len()];
    let begins = offsets;
    for item in runs {
        let index = &mut offsets[bucket(item.0)];
        grouped[*index] = item;
        *index += 1;
    }
    for index in 0..BUCKETS {
        grouped[begins[index]..offsets[index]].sort_by(|a, b| a.0.total_cmp(&b.0));
    }
    let mut ordered = Vec::with_capacity(adds.len());
    for (_, start, end) in grouped { ordered.extend_from_slice(&adds[start..end]); }
    ordered
}

/// Rank only new blocks, after membership is fixed, before the render
/// thread's bounded diff prefix decides which columns can be generated.
fn prioritize_incoming_blocks(grid: Grid, request: &WindowRequest, level: u32, radius: f64, adds: &mut [(f32, u64)]) {
    let forecast = request.priority_eye.or(request.prefetch_eye);
    let focus = request.view_focus;
    if forecast.is_none() && focus.is_none() { return; }
    let forecast_penalty = forecast.map_or(0.0, |point| grid.ground_distance(request.eye, point) * 0.25);
    let focus_penalty = focus.map_or(0.0, |point| grid.ground_distance(request.eye, point) * 0.125);
    let cells = BRICK << level;
    let columns = grid.cells() / cells;
    let denominator = radius.max(grid.level_size(level));
    let mut start = 0;
    while start < adds.len() {
        let key = adds[start].1;
        let face = ((key as u32 >> 24) & 7) as u8;
        let bi = (key as u32 & 0xffffff) as i32 & !3;
        let bj = (key >> 32) as i32 & !3;
        let mut end = start + 1;
        while end < adds.len() {
            let other = adds[end].1;
            if ((other as u32 >> 24) & 7) as u8 != face
                || ((other as u32 & 0xffffff) as i32 & !3) != bi
                || ((other >> 32) as i32 & !3) != bj { break; }
            end += 1;
        }
        let i = f64::from(bi) + f64::from((columns - bi).min(4)) * 0.5;
        let j = f64::from(bj) + f64::from((columns - bj).min(4)) * 0.5;
        let point = grid.ground_point(face, i * f64::from(cells), j * f64::from(cells));
        let mut distance = grid.ground_distance(request.eye, point);
        if let Some(future) = forecast { distance = distance.min(grid.ground_distance(future, point) + forecast_penalty); }
        if let Some(focus) = focus { distance = distance.min(grid.ground_distance(focus, point) + focus_penalty); }
        let priority = (distance / denominator) as f32;
        for item in &mut adds[start..end] { item.0 = priority; }
        start = end;
    }
}

/// Keep altitude-scaled lookahead, with a horizontal floor for ground flight.
/// This forecast only prioritizes already wanted columns. The planner uses
/// the shorter clearance-clamped forecast to avoid growing windows sideways.
pub(crate) fn motion_forecast(grid: &Grid, eye: DVec3, previous: DVec3, dt: f64, lod0: f64, clearance: f64) -> Option<DVec3> {
    if !dt.is_finite() || dt <= 0.0 || dt > 0.25 || !eye.is_finite() || !previous.is_finite() {
        return None;
    }
    let step = eye - previous;
    if step.length_squared() <= 0.0001 { return None; }
    let forecast = step * (0.35 / dt.max(0.001)).min(24.0);
    let up = grid.up(eye);
    let radial = forecast.dot(up);
    let tangent = forecast - up * radial;
    let radial_limit = clearance.max(20.0) * 0.7;
    let tangent_limit = (lod0 * 0.5).max(radial_limit);
    Some(eye + tangent.clamp_length_max(tangent_limit) + up * radial.clamp(-radial_limit, radial_limit))
}

/// Preserve the original coverage forecast: horizontal lookahead enlarges
/// every side of a circular window, so it stays bounded by terrain clearance.
pub(crate) fn window_forecast(eye: DVec3, previous: DVec3, dt: f64, clearance: f64) -> Option<DVec3> {
    if !dt.is_finite() || dt <= 0.0 || dt > 0.25 || !eye.is_finite() || !previous.is_finite() {
        return None;
    }
    let step = eye - previous;
    if step.length_squared() <= 0.0001 { return None; }
    let forecast = step * (0.35 / dt.max(0.001)).min(24.0);
    Some(eye + forecast.clamp_length_max(clearance.max(20.0) * 0.7))
}

/// Approximate centre-view terrain focus for scheduling only. Intersect the
/// plane or the local ground-radius sphere; actual tracing still uses terrain.
pub(crate) fn visible_focus(grid: &Grid, eye: DVec3, forward: DVec3, ground_radial: f64, max_distance: f64) -> Option<DVec3> {
    if !eye.is_finite() || !ground_radial.is_finite() || !max_distance.is_finite() || max_distance <= 0.0 {
        return None;
    }
    let direction = forward.try_normalize()?;
    if direction.dot(grid.up(eye)) >= -1.0e-6 { return None; }
    let t = if grid.is_plane() {
        (ground_radial - eye.y) / direction.y
    } else {
        if ground_radial <= 0.0 { return None; }
        let radius = eye.length();
        let b = eye.dot(direction);
        // Avoid subtracting nearly equal squared planetary radii or roots.
        let c = (radius - ground_radial) * (radius + ground_radial);
        if c <= 0.0 { return None; }
        let discriminant = b * b - c;
        if discriminant < 0.0 { return None; }
        c / (-b + discriminant.sqrt())
    };
    if !t.is_finite() || t <= 0.0 || t > max_distance { return None; }
    let point = eye + direction * t;
    if !point.is_finite() { return None; }
    if grid.is_plane() && grid.shape() != crate::grid::Shape::InfinitePlane {
        let coords = grid.face_coords(PLANE_FACE, point)?;
        if coords[0] < 0.0 || coords[1] < 0.0 || coords[0] >= f64::from(grid.cells()) || coords[1] >= f64::from(grid.cells()) {
            return None;
        }
    }
    Some(point)
}

impl WindowPlanner {
    pub fn new(grid: Grid) -> Self {
        Self {
            grid,
            snapshot: false,
            lod_dither: None,
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
        if lo_i > hi_i || lo_j > hi_j { return out; }
        // Traversal admits complete tier-1 blocks. Expand only blocks that
        // intersect the original column-centre circle: partial boundary
        // blocks would otherwise stay unusable even after settling.
        for bj in lo_j / 4..=hi_j / 4 {
            let (y0, y1) = (bj * 4, (bj * 4 + 3).min(cols - 1));
            for bi in lo_i / 4..=hi_i / 4 {
                let (x0, x1) = (bi * 4, (bi * 4 + 3).min(cols - 1));
                let x = (ci.floor() as i32).clamp(x0.max(lo_i), x1.min(hi_i));
                let y = (cj.floor() as i32).clamp(y0.max(lo_j), y1.min(hi_j));
                let d = (f64::from(x) + 0.5 - ci).hypot(f64::from(y) + 0.5 - cj);
                if d > limit { continue; }
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
            for bj in lo_j / 4..=hi_j / 4 {
                let (y0, y1) = (bj * 4, (bj * 4 + 3).min(cols - 1));
                for bi in lo_i / 4..=hi_i / 4 {
                    let (x0, x1) = (bi * 4, (bi * 4 + 3).min(cols - 1));
                    let mut intersects = false;
                    'columns: for y in y0.max(lo_j)..=y1.min(hi_j) {
                        let tb = tan_j[(y - block_lo_j) as usize];
                        for x in x0.max(lo_i)..=x1.min(hi_i) {
                            let ta = tan_i[(x - block_lo_i) as usize];
                            let cos = (dn + ta * da + tb * db) / (1.0 + ta * ta + tb * tb).sqrt();
                            if cos >= cos_limit {
                                intersects = true;
                                break 'columns;
                            }
                        }
                    }
                    if !intersects { continue; }
                    // One priority per block also keeps admission grouped.
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
        let dither = sanitize_lod_dither(request.lod_dither);
        let dither_changed = self.lod_dither != Some(dither);
        self.lod_dither = Some(dither);
        // Window centre: the eye direction on a sphere, its ground point on a plane.
        let dir = if grid.is_plane() { DVec3::new(eye.x, 0.0, eye.z) } else { eye.normalize() };
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
            snapshot: self.snapshot,
            ..Default::default()
        };
        for level in 0..grid.levels() {
            let nominal = request.lod0 * f64::from(1u32 << level);
            // selected=t*(1+d*(hash-.5)); fine selection can extend to
            // nominal/(1-d/2), and the next level can start correspondingly early.
            let selected_reach = nominal / (1.0 - dither * 0.5);
            let reach = (nominal * 1.05).max(selected_reach);
            let inner = if level == 0 { 0.0 } else {
                request.lod0 * f64::from(1u32 << (level - 1)) / (1.0 + dither * 0.5)
            };
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
                        removes: if self.snapshot { Vec::new() }
                            else { order_removes_by_blocks(state.wanted.iter().copied().collect()) },
                    });
                    state.wanted = Default::default();
                    update.wanted.push((level, state.wanted.clone()));
                    state.active = false;
                }
                continue;
            }
            let col = grid.level_size(level) * f64::from(BRICK);
            // The direct-mapped summary tables bound the window diameter.
            let cap = col * f64::from(max_window_columns() / 2 - 2) * 0.8;
            // Recentring tolerates a three-column vector displacement. On a
            // sphere that is a chord: convert it to a conservative arc length.
            let drift = if grid.is_plane() { col * 3.0 } else {
                2.0 * r0 * (col * 3.0 / (2.0 * r0)).min(1.0).asin()
            };
            let tangential_col = grid.delta() * f64::from(BRICK << level)
                * if grid.is_plane() { 1.0 } else { r0 };
            // The scanner already admits centres an extra .75 angular column
            // outwards. A face's nearest centre is at most one angular column
            // away (its projection Jacobian norm is <=sqrt(2)). Complete blocks
            // are retained; this remaining slack covers centre hysteresis.
            let slack = tangential_col * 0.25 + col * 0.25;
            let pad = (col * 2.0).max(drift + slack - (reach - selected_reach));
            let radius = if level == top_level {
                // The coarsest level covers the whole world.
                if grid.is_plane() { f64::from(grid.cells()) * grid.voxel_size() * 1.5 } else { r0 * 4.0 }
            } else {
                let future_reach = (reach * reach - future_altitude.max(0.0).powi(2)).max(0.0).sqrt();
                let motion = grid.ground_distance(eye, future);
                ((reach * reach - altitude.max(0.0).powi(2)).max(0.0).sqrt()
                    .max(future_reach + motion.min(reach * 0.5)).min(horizon) + pad).min(cap)
            };
            let moved = if grid.is_plane() { state.center.distance(dir) } else { state.center.distance(dir) * r0 };
            let actual_drift = if grid.is_plane() { moved } else {
                2.0 * r0 * (moved / (2.0 * r0)).min(1.0).asin()
            };
            let needed_radius = ((selected_reach * selected_reach - altitude.max(0.0).powi(2)).max(0.0).sqrt()
                .max((selected_reach * selected_reach - future_altitude.max(0.0).powi(2)).max(0.0).sqrt()
                    + grid.ground_distance(eye, future).min(selected_reach * 0.5))
                .min(horizon) + actual_drift + slack).min(cap);
            if state.active && !dither_changed && moved <= col * 3.0
                && needed_radius <= state.radius
                && (radius - state.radius).abs() <= state.radius * 0.08 + col {
                continue;
            }
            let scanned = self.scan(level, dir, radius);
            let state = &mut self.levels[level as usize];
            let mut next = FxHashSet::with_capacity_and_hasher(scanned.len(), Default::default());
            let mut adds = Vec::new();
            for (priority, key) in scanned {
                if self.snapshot || !state.wanted.contains(&key) {
                    adds.push((priority, key));
                }
                next.insert(key);
            }
            let removes = if self.snapshot { Vec::new() }
                else { order_removes_by_blocks(state.wanted.iter().filter(|k| !next.contains(k)).copied().collect()) };
            prioritize_incoming_blocks(grid, request, level, radius, &mut adds);
            let adds = order_adds_by_priority(adds);
            state.wanted = std::sync::Arc::new(next);
            update.wanted.push((level, state.wanted.clone()));
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
    requests: Option<mpsc::Sender<WindowMessage>>,
    updates: Mutex<Option<mpsc::Receiver<WindowUpdate>>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

enum WindowMessage {
    Request(WindowRequest),
    RetireWanted(std::sync::Arc<FxHashSet<u64>>),
    RetireDiff(LevelDiff),
    RetirePayload(Box<dyn Send>),
}

impl WindowWorker {
    pub fn start(grid: Grid) -> Self {
        let (request_tx, request_rx) = mpsc::channel::<WindowMessage>();
        // Preserve every ordered delta without accumulating full demand
        // snapshots when the renderer is temporarily stalled.
        let (update_tx, update_rx) = mpsc::sync_channel(1);
        let thread = std::thread::Builder::new()
            .name("voxel-planet-windows".into())
            .spawn(move || {
                let mut planner = WindowPlanner::new(grid);
                planner.snapshot = true;
                while let Ok(message) = request_rx.recv() {
                    let mut request = match message {
                        WindowMessage::Request(request) => request,
                        WindowMessage::RetireWanted(wanted) => { drop(wanted); continue; }
                        WindowMessage::RetireDiff(diff) => { drop(diff); continue; }
                        WindowMessage::RetirePayload(payload) => { drop(payload); continue; }
                    };
                    while let Ok(message) = request_rx.try_recv() {
                        match message {
                            WindowMessage::Request(newer) => request = newer,
                            WindowMessage::RetireWanted(wanted) => drop(wanted),
                            WindowMessage::RetireDiff(diff) => drop(diff),
                            WindowMessage::RetirePayload(payload) => drop(payload),
                        }
                    }
                    if update_tx.send(planner.update(&request)).is_err() {
                        break;
                    }
                }
            })
            .expect("spawn window planner");
        Self {
            requests: Some(request_tx),
            updates: Mutex::new(Some(update_rx)),
            thread: Some(thread),
        }
    }
    pub fn request(&self, request: WindowRequest) {
        if let Some(tx) = &self.requests {
            let _ = tx.send(WindowMessage::Request(request));
        }
    }
    pub fn try_update(&self) -> Option<WindowUpdate> {
        self.updates.lock().ok()?.as_ref()?.try_recv().ok()
    }
    /// Free old snapshots and completed delta buffers off the render thread.
    pub(crate) fn retire_wanted(&self, wanted: std::sync::Arc<FxHashSet<u64>>) {
        if let Some(tx) = &self.requests { let _ = tx.send(WindowMessage::RetireWanted(wanted)); }
    }
    pub(crate) fn retire_diff(&self, diff: LevelDiff) {
        if let Some(tx) = &self.requests { let _ = tx.send(WindowMessage::RetireDiff(diff)); }
    }
    pub(crate) fn retire_payload(&self, payload: impl Send + 'static) {
        if let Some(tx) = &self.requests { let _ = tx.send(WindowMessage::RetirePayload(Box::new(payload))); }
    }
}

impl Drop for WindowWorker {
    fn drop(&mut self) {
        self.requests = None;
        // A bounded output may be blocked on send; disconnect it before
        // joining so shutdown does not depend on another render frame.
        if let Ok(updates) = self.updates.get_mut() { *updates = None; }
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
    fn bounded_worker_output_disconnects_before_shutdown_join() {
        let (requests, _rx) = mpsc::channel();
        let (updates, receiver) = mpsc::sync_channel(1);
        let (ready, ready_rx) = mpsc::channel();
        let worker = WindowWorker {
            requests: Some(requests), updates: Mutex::new(Some(receiver)),
            thread: Some(std::thread::spawn(move || {
                updates.send(WindowUpdate::default()).unwrap();
                ready.send(()).unwrap();
                assert!(updates.send(WindowUpdate::default()).is_err(),
                    "shutdown must disconnect an undrained, full output channel");
            })),
        };
        ready_rx.recv().unwrap();
        let (done, done_rx) = mpsc::channel();
        std::thread::spawn(move || { drop(worker); done.send(()).unwrap(); });
        done_rx.recv_timeout(std::time::Duration::from_secs(2)).expect("bounded worker shutdown deadlocked");
    }

    fn block_identity(key: u64) -> (u8, u32, i32, i32) {
        let (face, level, i, j) = column_identity(key);
        (face, level, i >> 2, j >> 2)
    }

    fn assert_block_runs_preserved(before: &[(f32, u64)], after: &[(f32, u64)]) {
        assert_eq!(before.len(), after.len());
        let keys = |list: &[(f32, u64)]| list.iter().map(|(_, key)| *key).collect::<FxHashSet<_>>();
        assert!(keys(before) == keys(after), "priority may not change wanted membership");
        let mut expected = rustc_hash::FxHashMap::<_, Vec<u64>>::default();
        for &(_, key) in before { expected.entry(block_identity(key)).or_default().push(key); }
        let mut seen = FxHashSet::default();
        let mut start = 0;
        while start < after.len() {
            let identity = block_identity(after[start].1);
            assert!(seen.insert(identity), "a complete block was split across the diff");
            let mut end = start + 1;
            while end < after.len() && block_identity(after[end].1) == identity { end += 1; }
            let actual: Vec<_> = after[start..end].iter().map(|(_, key)| *key).collect();
            assert_eq!(actual, expected[&identity], "stable ordering may not scramble a block's columns");
            start = end;
        }
    }

    #[test]
    fn retirement_order_preserves_membership_and_contiguous_owners_at_every_tier() {
        let original: Vec<_> = (0..2).flat_map(|face| (0..68).flat_map(move |j|
            (0..136).map(move |i| pack(key0(face, 0, i), j)))).collect();
        let ordered = order_removes_by_blocks(original.clone());
        assert_eq!(ordered.len(), original.len());
        assert_eq!(ordered.iter().copied().collect::<FxHashSet<_>>(), original.iter().copied().collect::<FxHashSet<_>>());
        let mut reversed = original;
        reversed.reverse();
        assert_eq!(order_removes_by_blocks(reversed), ordered, "hash iteration must not decide retirement order");
        for tier in 1..=3 {
            let mut seen = FxHashSet::default();
            let mut last = None;
            for &key in &ordered {
                let (face, level, i, j) = column_identity(key);
                let owner = (face, level, i >> (2 * tier), j >> (2 * tier));
                if last != Some(owner) {
                    assert!(seen.insert(owner), "tier{tier} owner is fragmented across bounded retirement rounds");
                    last = Some(owner);
                }
            }
        }
    }

    #[test]
    fn diff_priority_orders_equal_bucket_blocks_and_preserves_boosted_groups() {
        let mut original = Vec::new();
        // The first three all map to the last priority bucket. Negative
        // priorities also share a bucket, but stronger boosts must go first.
        for (index, priority) in [1.2, 1.01, 0.99, -0.1, -0.4, 0.99].into_iter().enumerate() {
            let members = if index == 3 { 8 } else { 16 };
            for member in 0..members {
                original.push((priority, pack(key0(PLANE_FACE, 0, index as i32 * 4 + member % 4), (member / 4) as u32)));
            }
        }
        let ordered = order_adds_by_priority(original.clone());
        assert_block_runs_preserved(&original, &ordered);
        assert!(ordered.windows(2).all(|pair| pair[0].0 <= pair[1].0));
        let tied: Vec<_> = ordered.iter().filter(|(priority, _)| *priority == 0.99).map(|(_, key)| *key).collect();
        let original_tied: Vec<_> = original.iter().filter(|(priority, _)| *priority == 0.99).map(|(_, key)| *key).collect();
        assert_eq!(tied, original_tied, "equal-priority blocks retain deterministic scanner order");
    }

    #[test]
    fn visible_forward_block_enters_diff_before_nearer_peripheral_rows_without_growing_window() {
        for shape in [crate::grid::Shape::Plane, crate::grid::Shape::Sphere] {
            let planet = Planet::new(PlanetRecipe {
                shape, radius_m: 10000.0, plane_size_m: 40000.0,
                terrain: TerrainSource { generator: crate::landform::FLAT_ID.into(), ..Default::default() },
                ..Default::default()
            }).unwrap();
            let grid = *planet.grid();
            let ground = grid.radius();
            let eye = DVec3::Y * (ground + 30.0);
            let request = WindowRequest { eye, prefetch_eye: None, priority_eye: None, view_focus: None,
                lod0: 160.0, lod_dither: 0.0, outer_radius: planet.outer_radius(), planet: None, serial: 1 };
            let mut baseline_planner = WindowPlanner::new(grid);
            let mut focused_planner = WindowPlanner::new(grid);
            baseline_planner.update(&request);
            focused_planner.update(&request);
            let eye = eye + DVec3::X * 50.0;
            let focus = DVec3::new(191.0, ground, 0.0);
            let request = WindowRequest { eye, serial: 2, ..request };
            let baseline = baseline_planner.update(&request);
            let focused = focused_planner.update(&WindowRequest {
                priority_eye: Some(DVec3::new(102.5, ground, 0.0)), view_focus: Some(focus), ..request
            });
            for (before, after) in baseline.levels.iter().zip(&focused.levels) {
                assert_eq!((before.level, before.active, before.center, before.radius),
                    (after.level, after.active, after.center, after.radius));
                assert_block_runs_preserved(&before.adds, &after.adds);
                // Both old and new demand contain whole blocks, so incoming
                // membership also consists of complete blocks.
                assert_complete_demand(grid, &after.adds);
            }
            let before = &baseline.levels.iter().find(|level| level.level == 0).unwrap().adds;
            let after = &focused.levels.iter().find(|level| level.level == 0).unwrap().adds;
            let find_block = |list: &[(f32, u64)], point| {
                let coords = grid.face_coords(PLANE_FACE, point).unwrap();
                let bi = (coords[0] / f64::from(BRICK)).floor() as i32 >> 2;
                let bj = (coords[1] / f64::from(BRICK)).floor() as i32 >> 2;
                list.iter().position(|(_, key)| block_identity(*key) == (PLANE_FACE, 0, bi, bj)).unwrap()
            };
            let peripheral = DVec3::new(170.0, ground, -30.0);
            assert!(find_block(before, peripheral) < find_block(before, focus),
                "fixture must expose distance-only ordering ahead of the visible focus");
            assert!(find_block(after, focus) < find_block(after, peripheral),
                "actual view focus must reach the bounded diff prefix first");
        }
    }

    fn column_identity(key: u64) -> (u8, u32, i32, i32) {
        let k0 = key as u32;
        (((k0 >> 24) & 7) as u8, k0 >> 27, (k0 & 0xff_ffff) as i32, (key >> 32) as i32)
    }
    fn assert_complete_demand(grid: Grid, scanned: &[(f32, u64)]) {
        let wanted: FxHashSet<_> = scanned.iter().map(|(_, key)| *key).collect();
        assert_eq!(wanted.len(), scanned.len(), "a column must be emitted only once");
        for &key in &wanted {
            let (face, level, i, j) = column_identity(key);
            let cols = grid.cells() / (BRICK << level);
            assert!((0..cols).contains(&i) && (0..cols).contains(&j));
            for y in (j & !3)..=((j | 3).min(cols - 1)) {
                for x in (i & !3)..=((i | 3).min(cols - 1)) {
                    assert!(wanted.contains(&pack(key0(face, level, x), y as u32)),
                        "column ({i},{j}) requires complete block, missing ({x},{y}) at L{level}");
                }
            }
        }
    }

    // The former centre-circle demand, independently checked against the
    // block closure so the fix neither drops coverage nor expands whole rings.
    fn circle_demand(grid: Grid, level: u32, center: DVec3, radius: f64) -> FxHashSet<u64> {
        let col_cells = BRICK << level;
        let cols = grid.cells() / col_cells;
        let mut wanted = FxHashSet::default();
        if grid.is_plane() {
            let coords = grid.face_coords(PLANE_FACE, center).unwrap();
            let (ci, cj) = (coords[0] / f64::from(col_cells), coords[1] / f64::from(col_cells));
            let reach = (radius / (grid.level_size(level) * f64::from(BRICK)) + 1.0).min(f64::from(cols));
            let limit = reach - 0.25;
            for j in ((cj - reach).floor() as i32).max(0)..=((cj + reach).ceil() as i32).min(cols - 1) {
                for i in ((ci - reach).floor() as i32).max(0)..=((ci + reach).ceil() as i32).min(cols - 1) {
                    if (f64::from(i) + 0.5 - ci).hypot(f64::from(j) + 0.5 - cj) <= limit {
                        wanted.insert(pack(key0(PLANE_FACE, level, i), j as u32));
                    }
                }
            }
        } else {
            let col_angle = grid.delta() * f64::from(col_cells);
            let theta = (radius / grid.radius()).min(std::f64::consts::PI);
            let cos_limit = (theta + col_angle * 0.75).min(std::f64::consts::PI).cos();
            for face in 0..6u8 {
                let [n, a, b] = face_axes(face);
                let dn = center.dot(n);
                if theta < 1.2 && dn < (theta + 1.0).min(std::f64::consts::PI).cos() { continue; }
                let (lo_i, hi_i, lo_j, hi_j) = if dn > 0.2 && theta < 0.9 {
                    let ci = grid.index_of_angle(center.dot(a).atan2(dn)) / f64::from(col_cells);
                    let cj = grid.index_of_angle(center.dot(b).atan2(dn)) / f64::from(col_cells);
                    let reach = theta / col_angle / 0.7 + 2.0;
                    (((ci - reach).floor() as i32).max(0), ((ci + reach).ceil() as i32).min(cols - 1),
                     ((cj - reach).floor() as i32).max(0), ((cj + reach).ceil() as i32).min(cols - 1))
                } else { (0, cols - 1, 0, cols - 1) };
                for j in lo_j..=hi_j {
                    for i in lo_i..=hi_i {
                        let direction = grid.direction(face, (i * col_cells) as f64 + f64::from(col_cells) * 0.5,
                            (j * col_cells) as f64 + f64::from(col_cells) * 0.5);
                        if center.dot(direction) >= cos_limit { wanted.insert(pack(key0(face, level, i), j as u32)); }
                    }
                }
            }
        }
        wanted
    }

    fn assert_only_block_closure(grid: Grid, scanned: &[(f32, u64)], original: &FxHashSet<u64>) {
        let mut expected = FxHashSet::default();
        for &key in original {
            let (face, level, i, j) = column_identity(key);
            let cols = grid.cells() / (BRICK << level);
            for y in (j & !3)..=((j | 3).min(cols - 1)) {
                for x in (i & !3)..=((i | 3).min(cols - 1)) {
                    expected.insert(pack(key0(face, level, x), y as u32));
                }
            }
        }
        let actual: FxHashSet<_> = scanned.iter().map(|(_, key)| *key).collect();
        assert!(actual == expected, "demand must be exactly the closure of intersected blocks: actual{} expected{}", actual.len(), expected.len());
        assert_complete_demand(grid, scanned);
    }

    #[test]
    fn tiny_window_at_block_corner_requests_usable_near_columns() {
        let planet = Planet::new(PlanetRecipe {
            shape: crate::grid::Shape::Plane, plane_size_m: 40000.0,
            terrain: TerrainSource { generator: crate::landform::FLAT_ID.into(), ..Default::default() },
            ..Default::default()
        }).unwrap();
        let grid = *planet.grid();
        let request = WindowRequest { eye: DVec3::Y * 0.5, prefetch_eye: None, priority_eye: None, view_focus: None,
            lod0: 1.405499947, lod_dither: 0.0, outer_radius: planet.outer_radius(), planet: None, serial: 1 };
        let update = WindowPlanner::new(grid).update(&request);
        let fine = update.levels.iter().find(|l| l.level == 0).unwrap();
        let old_reach = request.lod0 * 1.05;
        let old_altitude = (request.eye.y - request.outer_radius).max(0.0);
        let old_radius = (old_reach * old_reach - old_altitude.powi(2)).sqrt() + 1.6;
        let former = circle_demand(grid, 0, fine.center, old_radius);
        assert_eq!(former.len(), 68, "fixture must expose the former incomplete corner window");
        let original = circle_demand(grid, 0, fine.center, fine.radius);
        assert_only_block_closure(grid, &fine.adds, &original);
    }

    #[test]
    fn native_windows_add_only_boundary_block_padding_with_bounded_growth() {
        for shape in [crate::grid::Shape::Plane, crate::grid::Shape::Sphere] {
            let planet = Planet::new(PlanetRecipe { shape, radius_m: 1_000_000.0,
                plane_size_m: 40000.0, ..Default::default() }).unwrap();
            let grid = *planet.grid();
            let planner = WindowPlanner::new(grid);
            let col = grid.level_size(0) * f64::from(BRICK);
            for (radius_cols, growth) in [(18.0, 1.4), (80.0, 1.1), (200.0, 1.05)] {
                for center in [DVec3::Y, DVec3::new(1.0, 1.0, 0.0).normalize()] {
                    let center = if grid.is_plane() { DVec3::new(0.23, 0.0, 0.41) } else { center };
                    let radius = col * radius_cols;
                    let scanned = planner.scan(0, center, radius);
                    let original = circle_demand(grid, 0, center, radius);
                    assert_only_block_closure(grid, &scanned, &original);
                    assert!((scanned.len() as f64) <= original.len() as f64 * growth,
                        "boundary padding grew {:?} radius{radius_cols}: {} to {}", shape, original.len(), scanned.len());
                    if grid.is_plane() || center == DVec3::Y {
                        for face in 0..6 {
                            let indices: Vec<_> = scanned.iter().map(|(_, key)| column_identity(*key))
                                .filter(|(f, _, _, _)| *f == face).collect();
                            if indices.is_empty() { continue; }
                            for axis in [2, 3] {
                                let coordinate = |c: &(u8, u32, i32, i32)| if axis == 2 { c.2 } else { c.3 };
                                let span = indices.iter().map(coordinate).max().unwrap() - indices.iter().map(coordinate).min().unwrap() + 1;
                                assert!(span < max_window_columns() as i32, "block padding may not alias the summary torus");
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn finite_face_edge_demand_clips_complete_blocks_to_world_columns() {
        let planet = Planet::new(PlanetRecipe { shape: crate::grid::Shape::Plane,
            plane_size_m: 40500.0, ..Default::default() }).unwrap();
        let grid = *planet.grid();
        let level = grid.levels() - 2;
        let cols = grid.cells() / (BRICK << level);
        assert_eq!(cols % 4, 2, "fixture must have a partial final block");
        let center = DVec3::new(grid.cells() as f64 * grid.voxel_size() * 0.5 - 0.01, 0.0, 0.0);
        let radius = grid.level_size(level) * f64::from(BRICK) * 2.0;
        let scanned = WindowPlanner::new(grid).scan_plane(level, center, radius);
        assert_only_block_closure(grid, &scanned, &circle_demand(grid, level, center, radius));
        assert!(scanned.iter().any(|(_, key)| column_identity(*key).2 == cols - 1));
    }

    #[test]
    fn low_altitude_forecast_keeps_tangent_lookahead_without_unbounded_descent() {
        for shape in [crate::grid::Shape::Plane, crate::grid::Shape::Sphere] {
            let planet = Planet::new(PlanetRecipe { shape, ..Default::default() }).unwrap();
            let grid = planet.grid();
            let eye = DVec3::Y * if grid.is_plane() { 2.0 } else { grid.radius() + 2.0 };
            let previous = eye - DVec3::X * 2.0 + DVec3::Y * 2.0;
            let future = motion_forecast(grid, eye, previous, 0.016, 120.0, 2.0).unwrap();
            let offset = future - eye;
            let up = grid.up(eye);
            let radial = offset.dot(up);
            assert!(radial.abs() <= 14.000001);
            assert!((offset - up * radial).length() > 14.0,
                "horizontal prefetch must not inherit the small clearance clamp");
            assert!(offset.length() < 80.0);
            assert!(motion_forecast(grid, eye, previous, 0.3, 120.0, 2.0).is_none());
            let teleport = motion_forecast(grid, eye, eye - DVec3::X * 100_000.0, 0.016, 120.0, 2.0).unwrap();
            assert!((teleport - eye).length() <= 60.0, "teleport may not enqueue its entire path");
            let high_eye = grid.at_radial(eye, grid.radius() + 10_000.0);
            let high = motion_forecast(grid, high_eye, high_eye - DVec3::X * 1000.0, 0.016, 120.0, 10_000.0).unwrap();
            let lookahead = (high - high_eye).length();
            assert!(lookahead > 1000.0 && lookahead <= 7000.000001,
                "orbital lookahead must retain its altitude-scaled range: {lookahead}");
        }
    }

    #[test]
    fn centre_view_focus_is_stable_and_rejects_non_terrain_rays() {
        for shape in [crate::grid::Shape::Plane, crate::grid::Shape::Sphere] {
            let planet = Planet::new(PlanetRecipe { shape, ..Default::default() }).unwrap();
            let grid = planet.grid();
            let ground = if grid.is_plane() { 0.0 } else { grid.radius() };
            let eye = DVec3::Y * (ground + 30.0);
            let pitch = -0.45f64;
            let forward = DVec3::new(pitch.cos(), pitch.sin(), 0.0);
            let focus = visible_focus(grid, eye, forward, ground, 200.0).unwrap();
            assert!((focus.x - 30.0 / (-pitch).tan()).abs() < 0.01);
            assert!((grid.radial(focus) - ground).abs() < 1.0e-6);
            assert!(visible_focus(grid, eye, DVec3::Y, ground, 200.0).is_none());
            assert!(visible_focus(grid, eye, DVec3::X, ground, 200.0).is_none());
            assert!(visible_focus(grid, eye, forward, ground, 10.0).is_none());
        }
        let planet = Planet::new(PlanetRecipe { shape: crate::grid::Shape::Plane, plane_size_m: 100.0, ..Default::default() }).unwrap();
        assert!(visible_focus(planet.grid(), DVec3::new(49.0, 30.0, 0.0), DVec3::new(1.0, -0.5, 0.0), 0.0, 200.0).is_none());
    }

    #[test]
    fn shallow_view_focus_uses_camera_far_instead_of_fine_level_range() {
        for shape in [crate::grid::Shape::Plane, crate::grid::Shape::Sphere] {
            let planet = Planet::new(PlanetRecipe { shape, plane_size_m: 10000.0, ..Default::default() }).unwrap();
            let grid = planet.grid();
            let ground = if grid.is_plane() { 0.0 } else { grid.radius() };
            let eye = DVec3::Y * (ground + 30.0);
            let pitch = -12.0f64.to_radians();
            let forward = DVec3::new(pitch.cos(), pitch.sin(), 0.0);
            let lod0 = 15.5;
            assert!(visible_focus(grid, eye, forward, ground, lod0 * 4.0).is_none());
            let focus = visible_focus(grid, eye, forward, ground, 1000.0).unwrap();
            assert!((focus - eye).length() > 140.0 && (focus - eye).length() < 145.0);
            assert!((focus.x - 30.0 / (-pitch).tan()).abs() < 0.02);
            assert!((grid.radial(focus) - ground).abs() < 1.0e-6);
            assert!(visible_focus(grid, eye, forward, ground, 100.0).is_none(),
                "an intersection beyond the camera far plane must remain rejected");
        }
    }

    #[test]
    fn longer_priority_forecast_keeps_original_horizontal_window_count() {
        for shape in [crate::grid::Shape::Plane, crate::grid::Shape::Sphere] {
            let planet = Planet::new(PlanetRecipe {
                shape, radius_m: 1000.0, plane_size_m: 1000.0,
                terrain: TerrainSource { generator: crate::landform::FLAT_ID.into(), ..Default::default() },
                ..Default::default()
            }).unwrap();
            let grid = planet.grid();
            let ground = if grid.is_plane() { 0.0 } else { grid.radius() };
            let eye = DVec3::Y * (ground + 30.0);
            let previous = eye - DVec3::X * 2.4;
            let coverage = window_forecast(eye, previous, 0.016, 30.0).unwrap();
            let priority = motion_forecast(grid, eye, previous, 0.016, 120.0, 30.0).unwrap();
            assert!((coverage - eye).length() <= 21.000001);
            assert!((priority - eye).length() > 50.0);
            let old = eye + ((eye - previous) * (0.35f64 / 0.016).min(24.0)).clamp_length_max(21.0);
            let request = WindowRequest { eye, prefetch_eye: Some(coverage), priority_eye: None, view_focus: None, lod0: 120.0, lod_dither: 0.0,
                outer_radius: planet.outer_radius(), planet: None, serial: 1 };
            let candidate = WindowPlanner::new(*grid).update(&request);
            let baseline = WindowPlanner::new(*grid).update(&WindowRequest { prefetch_eye: Some(old), ..request.clone() });
            for (after, before) in candidate.levels.iter().zip(&baseline.levels) {
                assert_eq!(after.active, before.active);
                assert_eq!(after.radius, before.radius);
                let after: FxHashSet<_> = after.adds.iter().map(|(_, key)| *key).collect();
                let before: FxHashSet<_> = before.adds.iter().map(|(_, key)| *key).collect();
                assert_eq!(after, before, "priority-only motion may not grow the wanted window");
            }
            let expanded = WindowPlanner::new(*grid).update(&WindowRequest { prefetch_eye: Some(priority), ..request });
            let fine_count = |update: &WindowUpdate| update.levels.iter().find(|l| l.level == 0).unwrap().adds.len();
            assert!(fine_count(&expanded) > fine_count(&candidate), "fixture must expose the circular expansion regression");
        }
    }

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
            priority_eye: None, view_focus: None,
            lod0: 100.0, lod_dither: 0.0,
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

    #[test]
    fn below_datum_sphere_keeps_fine_windows_and_finite_horizon_metadata() {
        for voxel_size_m in [0.1, 0.3, 1.0] {
            for height in [-53.7, -600.0] {
                let planet = std::sync::Arc::new(Planet::new(PlanetRecipe {
                    shape: crate::grid::Shape::Sphere, radius_m: 1000.0, voxel_size_m,
                    terrain: TerrainSource {
                        generator: crate::landform::FLAT_ID.into(),
                        settings: format!("{{\"height_m\":{height}}}"),
                        ..Default::default()
                    }, ..Default::default()
                }).unwrap());
                assert!(planet.outer_radius() < planet.grid().radius());
                let eye = DVec3::Y * (planet.outer_radius() + 30.0);
                assert!(eye.length() < planet.grid().radius());
                let request = WindowRequest {
                    eye, prefetch_eye: None, priority_eye: None, view_focus: None, lod0: 120.0, lod_dither: 0.0,
                    outer_radius: planet.outer_radius(), planet: Some(planet.clone()), serial: 1,
                };
                let update = WindowPlanner::new(*planet.grid()).update(&request);
                assert!(update.levels.iter().all(|level| level.radius.is_finite() && level.center.is_finite()));
                for level in [0, 1] {
                    let fine = update.levels.iter().find(|diff| diff.level == level).unwrap();
                    assert!(fine.active && !fine.adds.is_empty(),
                        "negative terrain must retain fineL{level} coverage, height{height}, size{voxel_size_m}");
                }
            }
        }
    }

    fn flat_request(shape: crate::grid::Shape, height: f64, lod0: f64, dither: f64) -> (Grid, WindowRequest) {
        let planet = Planet::new(PlanetRecipe { shape, radius_m: 6_371_000.0, plane_size_m: 1000.0,
            terrain: TerrainSource { generator: crate::landform::FLAT_ID.into(), ..Default::default() },
            ..Default::default() }).unwrap();
        let grid = *planet.grid();
        let eye = DVec3::Y * (if grid.is_plane() { height } else { grid.radius() + height });
        (grid, WindowRequest { eye, prefetch_eye: None, priority_eye: None, view_focus: None,
            lod0, lod_dither: dither, outer_radius: if grid.is_plane() { 0.0 } else { grid.radius() }, planet: None, serial: 1 })
    }

    fn assert_surface_witness(grid: Grid, request: &WindowRequest, i: i32, j: i32, expose_old_gap: bool) {
        let point = grid.ground_point(PLANE_FACE, (f64::from(i) + 0.5) * 8.0, (f64::from(j) + 0.5) * 8.0);
        let delta = point - request.eye;
        let t = delta.length();
        // This is a real first intersection with the flat surface, not an
        // empty-space request. Inward sphere incidence proves the near root.
        if !grid.is_plane() { assert!(delta.dot(point) < 0.0); }
        let hd = f64::from(crate::noise::hash3(i >> 1, j >> 1, i32::from(PLANE_FACE) | 8, 0x2545f491) & 1023) / 1023.0;
        let selected = t * (1.0 + sanitize_lod_dither(request.lod_dither) * (hd - 0.5));
        assert!(selected < request.lod0, "surface ray must select L0: t{t} selected{selected}");
        let update = WindowPlanner::new(grid).update(request);
        let fine = update.levels.iter().find(|l| l.level == 0 && l.active).unwrap();
        let keys: FxHashSet<_> = fine.adds.iter().map(|(_, k)| *k).collect();
        for y in (j & !3)..=(j | 3) {
            for x in (i & !3)..=(i | 3) { assert!(keys.contains(&pack(key0(PLANE_FACE, 0, x), y as u32))); }
        }
        if expose_old_gap {
            let reach = request.lod0 * 1.05;
            let old_radius = (reach * reach - 0.3f64.powi(2)).sqrt() + 1.6;
            let old = WindowPlanner::new(grid).scan(0, fine.center, old_radius);
            assert!(!old.iter().any(|(_, k)| *k == pack(key0(PLANE_FACE, 0, i), j as u32)),
                "fixture must prove an actual omitted fine surface block");
        }
    }

    #[test]
    fn renderer_dither_known_surface_fringe_is_wanted_on_plane_and_sphere() {
        let lod0 = 0.1 / (2.0 * (std::f64::consts::FRAC_PI_8).tan() / 729.0);
        for (shape, i, j) in [(crate::grid::Shape::Plane, 500, 620),
            (crate::grid::Shape::Sphere, 6_225_799, 6_225_898)] {
            let (grid, request) = flat_request(shape, 0.3, lod0, 0.25);
            assert_surface_witness(grid, &request, i, j, true);
            let mut planner = WindowPlanner::new(grid);
            planner.update(&request);
            let point = grid.ground_point(PLANE_FACE, (f64::from(i) + 0.5) * 8.0, (f64::from(j) + 0.5) * 8.0);
            let moved = WindowRequest { eye: request.eye + DVec3::new(point.x, 0.0, point.z).normalize() * 2.39, serial: 2, ..request };
            planner.update(&moved);
            assert!(grid.ground_distance(moved.eye, point) < lod0 / 0.875);
            assert!(planner.levels[0].wanted.contains(&pack(key0(PLANE_FACE, 0, i), j as u32)));
        }
    }

    #[test]
    fn renderer_dither_zero_preserves_native_radius_and_max_has_real_surface_coverage() {
        let native_lod0 = 0.1 / (2.0 * (std::f64::consts::FRAC_PI_8).tan() / 729.0);
        let (grid, request) = flat_request(crate::grid::Shape::Plane, 0.3, native_lod0, 0.0);
        let update = WindowPlanner::new(grid).update(&request);
        let fine = update.levels.iter().find(|l| l.level == 0).unwrap();
        let reach = native_lod0 * 1.05;
        assert_eq!(fine.radius, (reach * reach - 0.3f64.powi(2)).sqrt() + 1.6);
        assert_surface_witness(grid, &request, 524, 620, false);
        let default = WindowPlanner::new(grid).update(&WindowRequest { lod_dither: 0.25, ..request.clone() });
        let default_fine = default.levels.iter().find(|l| l.level == 0).unwrap();
        assert!(default_fine.adds.len() <= fine.adds.len() * 5 / 4,
            "native fine-window closure growth must remain below25% in this fixture");
        eprintln!("native plane L0 demand: dither0 {} radius{:.6}; dither.25 {} radius{:.6}",
            fine.adds.len(), fine.radius, default_fine.adds.len(), default_fine.radius);
        let lod0 = 0.1 / (2.0 * (std::f64::consts::FRAC_PI_8).tan() / 300.0);
        let (grid, request) = flat_request(crate::grid::Shape::Plane, 0.3, lod0, 99.0);
        let (i, j) = (535..550).flat_map(|i| (620..628).map(move |j| (i, j))).find(|&(i, j)| {
            let t = (grid.ground_point(PLANE_FACE, (f64::from(i)+0.5)*8.0, (f64::from(j)+0.5)*8.0) - request.eye).length();
            let hd = f64::from(crate::noise::hash3(i>>1, j>>1, i32::from(PLANE_FACE)|8, 0x2545f491)&1023)/1023.0;
            t > lod0 * 1.5 && t*(0.5+hd) < lod0
        }).expect("max-dither witness beyond the nominal footprint");
        assert_surface_witness(grid, &request, i, j, true);
    }

    #[test]
    fn renderer_dither_retargets_same_view_activation_forecast_and_horizon() {
        let (grid, mut request) = flat_request(crate::grid::Shape::Plane, 110.0, 100.0, 0.0);
        let mut planner = WindowPlanner::new(grid);
        assert!(!planner.update(&request).levels.iter().any(|l| l.level == 0 && l.active));
        request.lod_dither = 0.25; request.serial += 1;
        assert!(planner.update(&request).levels.iter().any(|l| l.level == 0 && l.active));
        request.eye = DVec3::Y * 1000.0; request.prefetch_eye = Some(DVec3::Y * 110.0); request.serial += 1;
        assert!(WindowPlanner::new(grid).update(&request).levels.iter().any(|l| l.level == 0 && l.active));
        request.lod_dither = 0.0;
        assert!(!WindowPlanner::new(grid).update(&request).levels.iter().any(|l| l.level == 0 && l.active));
        let planet = Planet::new(PlanetRecipe { radius_m: 1000.0,
            terrain: TerrainSource { generator: crate::landform::FLAT_ID.into(), ..Default::default() },
            ..Default::default() }).unwrap();
        let grid = *planet.grid();
        let request = WindowRequest { eye: DVec3::Y * 1000.4, prefetch_eye: None,
            lod0: 30.0, outer_radius: grid.radius(), ..request };
        assert!(!WindowPlanner::new(grid).update(&request).levels.iter().any(|l| l.level == 1 && l.active));
        assert!(WindowPlanner::new(grid).update(&WindowRequest { lod_dither: 0.25, ..request }).levels.iter().any(|l| l.level == 1 && l.active));
    }

    #[test]
    fn renderer_dither_sanitization_and_torus_cap_are_shared_and_bounded() {
        for (raw, expected) in [(-1.0,0.0),(0.0,0.0),(0.25,0.25),(1.0,1.0),(2.0,1.0),
            (f64::NAN,0.25),(f64::INFINITY,0.25),(f64::NEG_INFINITY,0.25)] {
            assert_eq!(sanitize_lod_dither(raw), expected);
        }
        for shape in [crate::grid::Shape::Plane, crate::grid::Shape::Sphere] {
            let (grid, request) = flat_request(shape, 0.3, 300.0, 1.0);
            let update = WindowPlanner::new(grid).update(&request);
            for diff in update.levels.iter().filter(|l| l.level + 1 < grid.levels()) {
                let col = grid.level_size(diff.level) * f64::from(BRICK);
                assert!(diff.radius <= col * f64::from(max_window_columns()/2-2) * 0.8);
                assert_complete_demand(grid, &diff.adds);
            }
            assert!(update.levels.iter().any(|l| l.level + 1 == grid.levels() && l.active));
        }
    }

}
