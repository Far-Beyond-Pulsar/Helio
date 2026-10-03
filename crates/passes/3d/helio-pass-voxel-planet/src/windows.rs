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
    /// This message owns only the fine range; other levels arrive separately.
    /// Completion is determined by per-level serial authority, not this flag.
    pub partial: bool,
    /// Levels evaluated by this message, including unchanged/inactive levels.
    pub processed_levels: u32,
    /// Original enqueue time, shared by both independently planned ranges.
    pub issued_at: Option<std::time::Instant>,
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

/// Snapshot scanners retain one descriptor per intersecting summary block.
/// Membership and row-major column order are identical to the delta scanner.
#[derive(Clone, Copy)]
struct SnapshotBlock {
    priority: f32,
    first: u64,
    width: u32,
    height: u32,
}

fn snapshot_columns(grid: Grid, request: &WindowRequest, level: u32, radius: f64,
    mut blocks: Vec<SnapshotBlock>) -> (FxHashSet<u64>, Vec<(f32, u64)>) {
    let forecast = request.priority_eye.or(request.prefetch_eye);
    let focus = request.view_focus;
    if forecast.is_some() || focus.is_some() {
        let forecast_penalty = forecast.map_or(0.0, |point| grid.ground_distance(request.eye, point) * 0.25);
        let focus_penalty = focus.map_or(0.0, |point| grid.ground_distance(request.eye, point) * 0.125);
        let cells = BRICK << level;
        let columns = grid.cells() / cells;
        let denominator = radius.max(grid.level_size(level));
        for block in &mut blocks {
            let key = block.first;
            let face = ((key as u32 >> 24) & 7) as u8;
            let bi = (key as u32 & 0xffffff) as i32 & !3;
            let bj = (key >> 32) as i32 & !3;
            // Keep the original priority arithmetic, including clipped edges.
            let i = f64::from(bi) + f64::from((columns - bi).min(4)) * 0.5;
            let j = f64::from(bj) + f64::from((columns - bj).min(4)) * 0.5;
            let point = grid.ground_point(face, i * f64::from(cells), j * f64::from(cells));
            let mut distance = grid.ground_distance(request.eye, point);
            if let Some(future) = forecast { distance = distance.min(grid.ground_distance(future, point) + forecast_penalty); }
            if let Some(focus) = focus { distance = distance.min(grid.ground_distance(focus, point) + focus_penalty); }
            block.priority = (distance / denominator) as f32;
        }
    }
    const BUCKETS: usize = 64;
    let bucket = |priority: f32| ((priority.max(0.0) * BUCKETS as f32) as usize).min(BUCKETS - 1);
    let mut offsets = [0usize; BUCKETS];
    for block in &blocks { offsets[bucket(block.priority)] += 1; }
    let mut start = 0;
    for offset in &mut offsets {
        let count = *offset;
        *offset = start;
        start += count;
    }
    let begins = offsets;
    let mut grouped = vec![SnapshotBlock { priority: 0.0, first: 0, width: 0, height: 0 }; blocks.len()];
    for block in blocks.drain(..) {
        let index = &mut offsets[bucket(block.priority)];
        grouped[*index] = block;
        *index += 1;
    }
    drop(blocks);
    for index in 0..BUCKETS {
        grouped[begins[index]..offsets[index]].sort_by(|a, b| a.priority.total_cmp(&b.priority));
    }
    let count = grouped.iter().map(|block| (block.width * block.height) as usize).sum();
    let mut wanted = FxHashSet::with_capacity_and_hasher(count, Default::default());
    let mut adds = Vec::with_capacity(count);
    for block in grouped {
        for y in 0..block.height {
            for x in 0..block.width {
                let key = pack(block.first as u32 + x, (block.first >> 32) as u32 + y);
                wanted.insert(key);
                adds.push((block.priority, key));
            }
        }
    }
    (wanted, adds)
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


// A single-column spherical predicate shared with both complete-block scanners.
fn column_in_circle(dn: f64, da: f64, db: f64, ta: f64, tb: f64, cos_limit: f64) -> bool {
    (dn + ta * da + tb * db) / (1.0 + ta * ta + tb * tb).sqrt() >= cos_limit
}

/// Revalidate an already captured block against the CURRENT global-bound window.
/// This is a bounded conservative superset of local-bound planner demand, not a
/// frustum/occlusion test. No generator query or wanted-set expansion is performed.
pub(crate) fn current_block_wanted(grid: &Grid, request: &WindowRequest, key: u64) -> bool {
    let k0 = key as u32;
    let (face, level, i, j) = (((k0 >> 24) & 7) as u8, k0 >> 27,
        (k0 & 0xff_ffff) as i32, (key >> 32) as u32 as i32);
    if level >= grid.levels().min(3) || !grid.faces().contains(&face)
        || i < 0 || j < 0 || i & 3 != 0 || j & 3 != 0
        || !request.eye.is_finite() || !request.outer_radius.is_finite()
        || !request.lod0.is_finite() || request.lod0 <= 0.0 {
        return false;
    }
    let cells = BRICK << level;
    let cols = grid.cells() / cells;
    if i >= cols || j >= cols { return false; }
    let r0 = grid.radius();
    let dither = sanitize_lod_dither(request.lod_dither);
    let nominal = request.lod0 * f64::from(1u32 << level);
    let selected = nominal / (1.0 - dither * 0.5);
    let reach = (nominal * 1.05).max(selected);
    let inner = if level == 0 { 0.0 } else { nominal * 0.5 / (1.0 + dither * 0.5) };
    let future = request.prefetch_eye.unwrap_or(request.eye);
    if !future.is_finite() { return false; }
    let altitude = grid.radial(request.eye) - request.outer_radius;
    let future_altitude = grid.radial(future) - request.outer_radius;
    let height = (grid.radial(request.eye) - r0.min(request.outer_radius)).max(0.0);
    let peak = (request.outer_radius - r0).max(0.0);
    let horizon = if grid.is_plane() { f64::INFINITY } else {
        (2.0 * r0 * height + height * height).sqrt() + (2.0 * r0 * peak + peak * peak).sqrt()
    };
    if altitude.min(future_altitude) >= reach || inner >= horizon { return false; }
    let col = grid.level_size(level) * f64::from(BRICK);
    let cap = col * f64::from(max_window_columns() / 2 - 2) * 0.8;
    let drift = if grid.is_plane() { col * 3.0 } else {
        2.0 * r0 * (col * 3.0 / (2.0 * r0)).min(1.0).asin()
    };
    let tangential_col = grid.delta() * f64::from(cells) * if grid.is_plane() { 1.0 } else { r0 };
    let slack = tangential_col * 0.25 + col * 0.25;
    let pad = (col * 2.0).max(drift + slack - (reach - selected));
    let motion = grid.ground_distance(request.eye, future);
    let radius = ((reach * reach - altitude.max(0.0).powi(2)).max(0.0).sqrt()
        .max((reach * reach - future_altitude.max(0.0).powi(2)).max(0.0).sqrt()
            + motion.min(reach * 0.5)).min(horizon) + pad).min(cap);
    if !radius.is_finite() || radius < 0.0 { return false; }
    let (x1, y1) = ((i + 3).min(cols - 1), (j + 3).min(cols - 1));
    if grid.is_plane() {
        let c = grid.face_coords(PLANE_FACE, request.eye).unwrap_or([0.0; 3]);
        let (ci, cj) = (c[0] / f64::from(cells), c[1] / f64::from(cells));
        let bounds = (radius / col + 1.0).min(f64::from(cols));
        let (lo_i, hi_i) = (((ci - bounds).floor() as i32).max(0), ((ci + bounds).ceil() as i32).min(cols - 1));
        let (lo_j, hi_j) = (((cj - bounds).floor() as i32).max(0), ((cj + bounds).ceil() as i32).min(cols - 1));
        return plane_block_in_circle(ci, cj, i, x1, j, y1, lo_i, hi_i, lo_j, hi_j, radius / col + 0.75);
    }
    let dir = request.eye.normalize_or_zero();
    if dir.length_squared() == 0.0 { return false; }
    let [n, a, b] = face_axes(face);
    let (dn, da, db) = (dir.dot(n), dir.dot(a), dir.dot(b));
    let angle = grid.delta() * f64::from(cells);
    let theta = (radius / r0).min(std::f64::consts::PI);
    if theta < 1.2 && dn < (theta + 1.0).min(std::f64::consts::PI).cos() { return false; }
    let (lo_i, hi_i, lo_j, hi_j) = if dn > 0.2 && theta < 0.9 {
        let ai = grid.index_of_angle(da.atan2(dn)) / f64::from(cells);
        let bj = grid.index_of_angle(db.atan2(dn)) / f64::from(cells);
        let bounds = theta / angle / 0.7 + 2.0;
        (((ai - bounds).floor() as i32).max(0), ((ai + bounds).ceil() as i32).min(cols - 1),
         ((bj - bounds).floor() as i32).max(0), ((bj + bounds).ceil() as i32).min(cols - 1))
    } else { (0, cols - 1, 0, cols - 1) };
    let cos_limit = (theta + angle * 0.75).min(std::f64::consts::PI).cos();
    let mut ta = [0.0; 4];
    for x in i.max(lo_i)..=x1.min(hi_i) {
        ta[(x - i) as usize] = grid.angle((f64::from(x) + 0.5) * f64::from(cells)).tan();
    }
    for y in j.max(lo_j)..=y1.min(hi_j) {
        let tb = grid.angle((f64::from(y) + 0.5) * f64::from(cells)).tan();
        for x in i.max(lo_i)..=x1.min(hi_i) {
            if column_in_circle(dn, da, db, ta[(x - i) as usize], tb, cos_limit) { return true; }
        }
    }
    false
}

fn plane_block_in_circle(ci: f64, cj: f64, x0: i32, x1: i32, y0: i32, y1: i32,
    lo_i: i32, hi_i: i32, lo_j: i32, hi_j: i32, limit: f64) -> bool {
    if x0.max(lo_i) > x1.min(hi_i) || y0.max(lo_j) > y1.min(hi_j) { return false; }
    let x = (ci.floor() as i32).clamp(x0.max(lo_i), x1.min(hi_i));
    let y = (cj.floor() as i32).clamp(y0.max(lo_j), y1.min(hi_j));
    (f64::from(x) + 0.5 - ci).hypot(f64::from(y) + 0.5 - cj) <= limit
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
                if !plane_block_in_circle(ci, cj, x0, x1, y0, y1, lo_i, hi_i, lo_j, hi_j, limit) { continue; }
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
                            if column_in_circle(dn, da, db, ta, tb, cos_limit) {
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

    /// Columns of a plane level within `radius` of ground point `center`.
    fn scan_snapshot_plane(&self, level: u32, center: DVec3, radius: f64, base_priority: bool) -> Vec<SnapshotBlock> {
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
                let priority = if base_priority { ((f64::from(x0 + x1 + 1) * 0.5 - ci)
                    .hypot(f64::from(y0 + y1 + 1) * 0.5 - cj) / limit.max(1e-12)) as f32 } else { 0.0 };
                out.push(SnapshotBlock { priority, first: pack(key0(PLANE_FACE, level, x0), y0 as u32),
                    width: (x1 - x0 + 1) as u32, height: (y1 - y0 + 1) as u32 });
            }
        }
        out
    }

    fn scan_snapshot(&self, level: u32, dir: DVec3, radius: f64, base_priority: bool) -> Vec<SnapshotBlock> {
        let grid = self.grid;
        if grid.is_plane() {
            return self.scan_snapshot_plane(level, dir, radius, base_priority);
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
                            if column_in_circle(dn, da, db, ta, tb, cos_limit) {
                                intersects = true;
                                break 'columns;
                            }
                        }
                    }
                    if !intersects { continue; }
                    // Forecast/focus ranking replaces this value completely.
                    let priority = if base_priority {
                        let ta = grid.angle(f64::from(x0 + x1 + 1) * 0.5 * f64::from(col_cells)).tan();
                        let tb = grid.angle(f64::from(y0 + y1 + 1) * 0.5 * f64::from(col_cells)).tan();
                        let cos = (dn + ta * da + tb * db) / (1.0 + ta * ta + tb * tb).sqrt();
                        (cos.clamp(-1.0, 1.0).acos() / theta.max(1e-12)) as f32
                    } else { 0.0 };
                    out.push(SnapshotBlock { priority, first: pack(key0(face, level, x0), y0 as u32),
                        width: (x1 - x0 + 1) as u32, height: (y1 - y0 + 1) as u32 });
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
        self.update_range(request, 0..self.grid.levels())
    }

    /// Compute only the owned range. Independent workers never scan or retain
    /// wanted sets for the other's levels; both use the global coarsest level.
    fn update_range(&mut self, request: &WindowRequest, range: std::ops::Range<u32>) -> WindowUpdate {
        assert!(range.start < range.end && range.end <= self.grid.levels());
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
            partial: range.end < grid.levels(),
            processed_levels: ((1u32 << range.end) - 1) & !((1u32 << range.start) - 1),
            ..Default::default()
        };
        for level in range {
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
            let (next, adds, removes) = if self.snapshot {
                let base_priority = request.priority_eye.or(request.prefetch_eye).is_none() && request.view_focus.is_none();
                let blocks = self.scan_snapshot(level, dir, radius, base_priority);
                let (next, adds) = snapshot_columns(grid, request, level, radius, blocks);
                (next, adds, Vec::new())
            } else {
                // The original delta scanner remains independent of the
                // snapshot block path and its tests use it as the oracle.
                let scanned = self.scan(level, dir, radius);
                let state = &self.levels[level as usize];
                let mut next = FxHashSet::with_capacity_and_hasher(scanned.len(), Default::default());
                let mut adds = Vec::new();
                for (priority, key) in scanned {
                    if !state.wanted.contains(&key) { adds.push((priority, key)); }
                    next.insert(key);
                }
                let removes = order_removes_by_blocks(state.wanted.iter().filter(|k| !next.contains(k)).copied().collect());
                prioritize_incoming_blocks(grid, request, level, radius, &mut adds);
                (next, order_adds_by_priority(adds), removes)
            };
            let state = &mut self.levels[level as usize];
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
    #[cfg(test)]
    fn update_range_column_snapshot_oracle(&mut self, request: &WindowRequest, range: std::ops::Range<u32>) -> WindowUpdate {
        assert!(range.start < range.end && range.end <= self.grid.levels());
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
            partial: range.end < grid.levels(),
            processed_levels: ((1u32 << range.end) - 1) & !((1u32 << range.start) - 1),
            ..Default::default()
        };
        for level in range {
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

/// Merge only unpublished full snapshots from the same independently owned
/// range. A newer no-op certifies old membership, so retain its unconsumed
/// diff/wanted pair; a new changed or inactive level replaces both together.
fn coalesce_unpublished_snapshot(mut newer: WindowUpdate, older: WindowUpdate) -> WindowUpdate {
    assert!(newer.snapshot && older.snapshot);
    assert_ne!(newer.processed_levels, 0);
    assert_eq!(newer.processed_levels, older.processed_levels);
    if newer.serial < older.serial { return older; }
    let changed = newer.levels.iter().fold(0u32, |mask, diff| mask | (1u32 << diff.level))
        | newer.wanted.iter().fold(0u32, |mask, (level, _)| mask | (1u32 << level));
    // Superseded vectors and Arc values are destroyed by this worker, outside
    // the mailbox lock. Retained metadata remains paired with its own keys.
    for diff in older.levels {
        if changed & (1u32 << diff.level) == 0 { newer.levels.push(diff); }
    }
    for (level, wanted) in older.wanted {
        if changed & (1u32 << level) == 0 { newer.wanted.push((level, wanted)); }
    }
    newer.levels.sort_by_key(|diff| diff.level);
    newer.wanted.sort_by_key(|(level, _)| *level);
    newer
}

#[derive(Default)]
struct WindowMailboxState {
    closed: bool,
    pending: [Option<WindowUpdate>; 2],
    next_range: usize,
}

#[derive(Default)]
struct WindowMailbox {
    state: Mutex<WindowMailboxState>,
}

impl WindowMailbox {
    fn is_closed(&self) -> bool {
        self.state.lock().map_or(true, |state| state.closed)
    }

    /// Exactly one producer owns each slot. Take/merge/replace prevents a
    /// completed result from blocking the next request, with at most one
    /// pending fine and one pending far update. Large work is outside locks.
    fn publish(&self, range: usize, update: WindowUpdate) -> bool {
        let previous = {
            let Ok(mut state) = self.state.lock() else { return false };
            if state.closed { return false; }
            state.pending[range].take()
        };
        let update = match previous { Some(previous) => coalesce_unpublished_snapshot(update, previous), None => update };
        {
            let Ok(mut state) = self.state.lock() else { return false };
            if state.closed { return false; }
            debug_assert!(state.pending[range].is_none(), "one producer per owned range");
            state.pending[range] = Some(update);
        }
        true
    }

    fn take(&self) -> Option<WindowUpdate> {
        let mut state = self.state.lock().ok()?;
        if state.closed { return None; }
        for offset in 0..2 {
            let range = (state.next_range + offset) % 2;
            if let Some(update) = state.pending[range].take() {
                state.next_range = (range + 1) % 2;
                return Some(update);
            }
        }
        None
    }

    fn close(&self) -> [Option<WindowUpdate>; 2] {
        let mut state = self.state.lock().unwrap();
        state.closed = true;
        std::mem::take(&mut state.pending)
    }
}

/// Independent fine/far planners, each working on its most recent request.
pub struct WindowWorker {
    requests: Option<Vec<mpsc::Sender<WindowMessage>>>,
    updates: std::sync::Arc<WindowMailbox>,
    threads: Vec<std::thread::JoinHandle<()>>,
}

enum WindowMessage {
    Request(WindowRequest, std::time::Instant),
    RetireWanted(std::sync::Arc<FxHashSet<u64>>),
    RetireDiff(LevelDiff),
    RetirePayload(Box<dyn Send>),
}

impl WindowWorker {
    pub fn start(grid: Grid) -> Self {
        let updates = std::sync::Arc::new(WindowMailbox::default());
        let fine_end = grid.levels().min(3);
        let mut ranges = vec![0..fine_end];
        if fine_end < grid.levels() { ranges.push(fine_end..grid.levels()); }
        let mut requests = Vec::with_capacity(ranges.len());
        let mut threads = Vec::with_capacity(ranges.len());
        for (range_index, range) in ranges.into_iter().enumerate() {
            let (request_tx, request_rx) = mpsc::channel::<WindowMessage>();
            let updates = updates.clone();
            let name = if range.start == 0 { "voxel-planet-windows-fine" } else { "voxel-planet-windows-far" };
            let thread = std::thread::Builder::new()
            .name(name.into())
            .spawn(move || {
                let mut planner = WindowPlanner::new(grid);
                planner.snapshot = true;
                while let Ok(message) = request_rx.recv() {
                    if updates.is_closed() { break; }
                    let (mut request, mut issued_at) = match message {
                        WindowMessage::Request(request, issued_at) => (request, issued_at),
                        WindowMessage::RetireWanted(wanted) => { drop(wanted); continue; }
                        WindowMessage::RetireDiff(diff) => { drop(diff); continue; }
                        WindowMessage::RetirePayload(payload) => { drop(payload); continue; }
                    };
                    while let Ok(message) = request_rx.try_recv() {
                        match message {
                            WindowMessage::Request(newer, at) => { request = newer; issued_at = at; }
                            WindowMessage::RetireWanted(wanted) => drop(wanted),
                            WindowMessage::RetireDiff(diff) => drop(diff),
                            WindowMessage::RetirePayload(payload) => drop(payload),
                        }
                    }
                    if updates.is_closed() { break; }
                    let mut update = planner.update_range(&request, range.clone());
                    update.issued_at = Some(issued_at);
                    if !updates.publish(range_index, update) { break; }
                }
            })
            .expect("spawn window planner");
            requests.push(request_tx);
            threads.push(thread);
        }
        Self { requests: Some(requests), updates, threads }
    }
    pub fn request(&self, request: WindowRequest) {
        let issued_at = std::time::Instant::now();
        if let Some(channels) = &self.requests {
            for tx in channels {
                let _ = tx.send(WindowMessage::Request(request.clone(), issued_at));
            }
        }
    }
    pub fn try_update(&self) -> Option<WindowUpdate> { self.updates.take() }
    /// Free old snapshots and completed delta buffers off the render thread.
    pub(crate) fn retire_wanted(&self, wanted: std::sync::Arc<FxHashSet<u64>>) {
        if let Some(tx) = self.requests.as_ref().and_then(|channels| channels.last()) { let _ = tx.send(WindowMessage::RetireWanted(wanted)); }
    }
    pub(crate) fn retire_diff(&self, diff: LevelDiff) {
        if let Some(tx) = self.requests.as_ref().and_then(|channels| channels.last()) { let _ = tx.send(WindowMessage::RetireDiff(diff)); }
    }
    pub(crate) fn retire_payload(&self, payload: impl Send + 'static) {
        if let Some(tx) = self.requests.as_ref().and_then(|channels| channels.last()) { let _ = tx.send(WindowMessage::RetirePayload(Box::new(payload))); }
    }
}

impl Drop for WindowWorker {
    fn drop(&mut self) {
        // Close before joining; producers never need another render drain.
        // Move the at-most-two buffered snapshots to a shutdown-only thread:
        // their possibly large destructors run off the caller and off locks.
        let pending = self.updates.close();
        self.requests = None;
        let cleanup = pending.iter().any(Option::is_some).then(|| std::thread::spawn(move || drop(pending)));
        for thread in self.threads.drain(..) { let _ = thread.join(); }
        if let Some(cleanup) = cleanup { let _ = cleanup.join(); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Planet, PlanetRecipe, TerrainSource};

    fn assert_snapshot_column_parity(expected: &WindowUpdate, actual: &WindowUpdate) {
        assert_eq!(actual.serial, expected.serial);
        assert_eq!(actual.snapshot, expected.snapshot);
        assert_eq!(actual.partial, expected.partial);
        assert_eq!(actual.processed_levels, expected.processed_levels);
        assert_eq!(actual.levels.len(), expected.levels.len());
        for (actual, expected) in actual.levels.iter().zip(&expected.levels) {
            assert_eq!((actual.level, actual.active, actual.center, actual.radius),
                (expected.level, expected.active, expected.center, expected.radius));
            assert_eq!(actual.removes, expected.removes);
            let exact = |items: &[(f32, u64)]| items.iter().map(|(priority, key)| (priority.to_bits(), *key)).collect::<Vec<_>>();
            assert_eq!(exact(&actual.adds), exact(&expected.adds), "exact ordered priorities L{}", actual.level);
        }
        assert_eq!(actual.wanted.len(), expected.wanted.len());
        for ((level, wanted), (expected_level, expected_wanted)) in actual.wanted.iter().zip(&expected.wanted) {
            assert_eq!(level, expected_level);
            assert_eq!(wanted, expected_wanted, "exact wanted L{level}");
        }
    }

    #[test]
    fn snapshot_blocks_match_independent_column_scanner_at_faces_edges_and_priority_anchors() {
        for shape in [crate::grid::Shape::Plane, crate::grid::Shape::Sphere, crate::grid::Shape::InfinitePlane] {
            let planet = Planet::new(PlanetRecipe { shape, radius_m: 1000.0, plane_size_m: 40500.0,
                voxel_size_m: 0.1, ..Default::default() }).unwrap();
            let grid = *planet.grid();
            let planner = WindowPlanner::new(grid);
            let levels = [0, 1, 2, grid.levels().saturating_sub(2), grid.levels() - 1];
            for level in levels.into_iter().filter(|level| *level < grid.levels()) {
                let col = grid.level_size(level) * f64::from(BRICK);
                let centers = if grid.is_plane() {
                    let half = f64::from(grid.cells()) * grid.voxel_size() * 0.5;
                    vec![DVec3::new(0.23, 0.0, 0.41), DVec3::new(half - 0.01, 0.0, half - 0.01),
                        DVec3::new(-half + 0.01, 0.0, -half + 0.01)]
                } else {
                    vec![DVec3::Y, DVec3::new(1.0, 1.0, 0.0).normalize(),
                        DVec3::new(-1.0, 1.0, -1.0).normalize()]
                };
                for center in centers {
                    for radius in [col * 0.01, col * 2.0, col * 18.0, col * 80.0] {
                        for anchors in 0..4 {
                            let eye = if grid.is_plane() { center + DVec3::Y } else { center * (grid.radius() + 1.0) };
                            let request = WindowRequest { eye,
                                prefetch_eye: (anchors & 1 != 0).then_some(eye + DVec3::X * col),
                                priority_eye: (anchors == 3).then_some(eye + DVec3::Z * col * 3.0),
                                view_focus: (anchors & 2 != 0).then_some(eye + DVec3::Z * col * 2.0),
                                lod0: 100.0, lod_dither: 0.25, outer_radius: planet.outer_radius(), planet: None, serial: 1 };
                            let scanned = planner.scan(level, center, radius);
                            let expected_wanted: FxHashSet<_> = scanned.iter().map(|(_, key)| *key).collect();
                            let mut expected_adds = scanned;
                            prioritize_incoming_blocks(grid, &request, level, radius, &mut expected_adds);
                            let expected_adds = order_adds_by_priority(expected_adds);
                            let base_priority = request.priority_eye.or(request.prefetch_eye).is_none() && request.view_focus.is_none();
                            let blocks = planner.scan_snapshot(level, center, radius, base_priority);
                            let (wanted, adds) = snapshot_columns(grid, &request, level, radius, blocks);
                            assert_eq!(wanted, expected_wanted, "membership {shape:?} L{level} radius{radius} anchors{anchors}");
                            let exact = |items: &[(f32, u64)]| items.iter().map(|(priority, key)| (priority.to_bits(), *key)).collect::<Vec<_>>();
                            assert_eq!(exact(&adds), exact(&expected_adds), "order/priority {shape:?} L{level} radius{radius} anchors{anchors}");
                            assert_eq!(adds.len(), wanted.len(), "no duplicate columns");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn snapshot_descriptor_updates_match_legacy_full_snapshots_across_view_history() {
        for shape in [crate::grid::Shape::Plane, crate::grid::Shape::Sphere, crate::grid::Shape::InfinitePlane] {
            let planet = std::sync::Arc::new(Planet::new(PlanetRecipe { shape, radius_m: 1000.0,
                plane_size_m: 1024.0, voxel_size_m: 0.1, ..Default::default() }).unwrap());
            let grid = *planet.grid();
            let mut old = WindowPlanner::new(grid);
            old.snapshot = true;
            let mut new = WindowPlanner::new(grid);
            new.snapshot = true;
            for (step, (shift, height, dither)) in [(0.0, 2.0, 0.25), (0.0, 2.0, 0.25),
                (4.0, 2.0, 0.25), (20.0, 2.0, 0.25), (45.0, 300.0, 0.25),
                (45.0, 2.0, 0.25), (45.0, 2.0, 0.0), (45.0, 2.0, 1.0)].into_iter().enumerate() {
                let eye = if grid.is_plane() { DVec3::new(shift, height, -shift) }
                    else { DVec3::new(1.0 + shift / 1000.0, 1.0, 0.1).normalize() * (grid.radius() + height) };
                let request = WindowRequest { eye, prefetch_eye: Some(eye + DVec3::X * 3.0),
                    priority_eye: Some(eye + DVec3::Z * 20.0), view_focus: Some(eye + DVec3::X * 40.0),
                    lod0: 20.0, lod_dither: dither, outer_radius: planet.outer_radius(),
                    planet: Some(planet.clone()), serial: step as u64 + 1 };
                let expected = old.update_range_column_snapshot_oracle(&request, 0..grid.levels());
                let actual = new.update(&request);
                assert_snapshot_column_parity(&expected, &actual);
            }
        }
    }

    #[test]
    fn independent_ranges_preserve_full_planner_demand_and_order_across_view_history() {
        for shape in [crate::grid::Shape::Plane, crate::grid::Shape::Sphere, crate::grid::Shape::InfinitePlane] {
            let planet = Planet::new(PlanetRecipe { shape, radius_m: 1000.0, plane_size_m: 1024.0,
                voxel_size_m: 0.1, terrain: TerrainSource { generator: crate::landform::FLAT_ID.into(), ..Default::default() },
                ..Default::default() }).unwrap();
            let grid = *planet.grid();
            assert!(grid.levels() > 3);
            for snapshot in [false, true] {
                let mut full = WindowPlanner::new(grid);
                let mut fine = WindowPlanner::new(grid);
                let mut far = WindowPlanner::new(grid);
                full.snapshot = snapshot; fine.snapshot = snapshot; far.snapshot = snapshot;
                for (index, (horizontal, height, dither)) in [
                    (0.0, 0.3, 0.25), (1.0, 0.3, 0.25), (2.0, 0.3, 0.25),
                    (2.0, 0.3, 0.25), (2.0, 300.0, 0.25), (-1.0, 0.3, 1.0), (-1.0, 0.3, 1.0),
                ].into_iter().enumerate() {
                    let eye = if grid.is_plane() { DVec3::new(horizontal * 16.0, height, 3.0) }
                        else {
                            // Cross the +Y/+X seam, then stop and change altitude.
                            let direction = if horizontal == 0.0 { DVec3::Y }
                                else { DVec3::new(1.0 + (horizontal - 1.5) * 0.02, 1.0, 0.0).normalize() };
                            direction * (grid.radius() + height)
                        };
                    let request = WindowRequest { eye, prefetch_eye: None, priority_eye: None, view_focus: None,
                        lod0: 6.0, lod_dither: dither, outer_radius: grid.radius(), planet: None,
                        serial: index as u64 + 1 };
                    let expected = full.update(&request);
                    let near = fine.update_range(&request, 0..3);
                    let distant = far.update_range(&request, 3..grid.levels());
                    assert_eq!(near.processed_levels, 7);
                    assert!(near.partial && !distant.partial && !expected.partial);
                    assert_eq!(near.processed_levels & distant.processed_levels, 0);
                    assert_eq!(near.processed_levels | distant.processed_levels, expected.processed_levels);
                    assert_eq!((near.serial, distant.serial), (expected.serial, expected.serial));
                    assert_eq!((near.snapshot, distant.snapshot), (snapshot, snapshot));
                    assert!(near.issued_at.is_none() && distant.issued_at.is_none());
                    let diffs: Vec<_> = near.levels.iter().chain(&distant.levels).collect();
                    assert_eq!(diffs.len(), expected.levels.len());
                    for (actual, expected) in diffs.into_iter().zip(&expected.levels) {
                        assert_eq!((actual.level, actual.active, actual.center, actual.radius),
                            (expected.level, expected.active, expected.center, expected.radius));
                        assert_eq!(actual.adds, expected.adds, "stable complete-block priorities must agree");
                        assert_eq!(actual.removes, expected.removes, "retirement order must agree");
                    }
                    let wanted: Vec<_> = near.wanted.iter().chain(&distant.wanted).collect();
                    assert_eq!(wanted.len(), expected.wanted.len());
                    for (actual, expected) in wanted.into_iter().zip(&expected.wanted) {
                        assert_eq!(actual.0, expected.0);
                        assert_eq!(actual.1.as_ref(), expected.1.as_ref());
                    }
                    assert!(fine.levels[3..].iter().all(|l| !l.active && l.wanted.is_empty() && l.bound.is_none()));
                    assert!(far.levels[..3].iter().all(|l| !l.active && l.wanted.is_empty() && l.bound.is_none()));
                    if index == 3 || index == 6 {
                        assert!(expected.levels.is_empty(), "a stopped view is an authority-only update");
                        assert!(near.levels.is_empty() && distant.levels.is_empty());
                    }
                }
            }
        }
    }

    #[test]
    fn fine_worker_publishes_while_far_retirement_is_blocked() {
        struct BlockDrop { started: mpsc::Sender<()>, release: mpsc::Receiver<()> }
        impl Drop for BlockDrop {
            fn drop(&mut self) { let _ = self.started.send(()); let _ = self.release.recv(); }
        }
        let (grid, request) = flat_request(crate::grid::Shape::Plane, 0.3, 6.0, 0.25);
        let worker = WindowWorker::start(grid);
        assert_eq!(worker.threads.len(), 2);
        let (started, start_rx) = mpsc::channel();
        let (release, release_rx) = mpsc::channel();
        worker.retire_payload(BlockDrop { started, release: release_rx });
        let blocking = start_rx.recv_timeout(std::time::Duration::from_secs(2));
        worker.request(request.clone());
        let near = receive_window_update(&worker);
        // Always unblock before asserting, so a failed test cannot hang Drop.
        release.send(()).unwrap();
        blocking.expect("far retirement did not start");
        let near = near.expect("fine planner waited for far retirement");
        let far = receive_window_update(&worker).unwrap();
        assert_eq!(near.processed_levels, 7);
        assert!(near.partial && !far.partial);
        assert_eq!(near.processed_levels & far.processed_levels, 0);
        assert_eq!(near.processed_levels | far.processed_levels, (1u32 << grid.levels()) - 1);
        assert_eq!((near.serial, far.serial), (request.serial, request.serial));
        assert!(near.issued_at.is_some());
        assert_eq!(near.issued_at, far.issued_at, "both ranges retain one original enqueue stamp");
    }

    #[test]
    fn small_grid_uses_one_complete_worker_without_an_empty_authority_message() {
        let grid = Grid::plane(crate::grid::Shape::Plane, 6.4, 0.1).unwrap();
        assert!(grid.levels() <= 3);
        let worker = WindowWorker::start(grid);
        assert_eq!(worker.threads.len(), 1);
        let (_, mut request) = flat_request(crate::grid::Shape::Plane, 0.3, 2.0, 0.25);
        request.serial = 9;
        worker.request(request);
        let update = receive_window_update(&worker).unwrap();
        assert_eq!(update.processed_levels, (1u32 << grid.levels()) - 1);
        assert!(!update.partial);
        assert!(worker.try_update().is_none());
    }

    fn receive_window_update(worker: &WindowWorker) -> Result<WindowUpdate, &'static str> {
        let until = std::time::Instant::now() + std::time::Duration::from_secs(2);
        loop {
            if let Some(update) = worker.try_update() { return Ok(update); }
            if std::time::Instant::now() >= until { return Err("window mailbox timed out"); }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    fn mailbox_snapshot(serial: u64, mask: u32, changes: &[(u32, bool, u64)]) -> WindowUpdate {
        let mut update = WindowUpdate { serial, snapshot: true, partial: mask == 7, processed_levels: mask,
            issued_at: Some(std::time::Instant::now()), ..Default::default() };
        for &(level, active, key) in changes {
            let adds = if active { vec![(level as f32 + 0.125, key)] } else { Vec::new() };
            let wanted = std::sync::Arc::new(adds.iter().map(|(_, key)| *key).collect::<FxHashSet<_>>());
            update.wanted.push((level, wanted));
            update.levels.push(LevelDiff { level, active, center: DVec3::new(serial as f64, level as f64, 0.0),
                radius: serial as f64 + 0.5, adds, removes: Vec::new() });
        }
        update
    }

    #[test]
    fn latest_mailbox_changed_then_noop_retains_unpublished_pairs_and_new_authority() {
        let mailbox = WindowMailbox::default();
        let first = mailbox_snapshot(10, 7, &[(0, true, 100), (1, true, 101)]);
        let first_level1 = first.wanted[1].1.clone();
        assert!(mailbox.publish(0, first));
        let changed = mailbox_snapshot(11, 7, &[(0, true, 200)]);
        let changed_level0 = changed.wanted[0].1.clone();
        assert!(mailbox.publish(0, changed));
        let noop = mailbox_snapshot(12, 7, &[]);
        let newest_stamp = noop.issued_at;
        assert!(mailbox.publish(0, noop));
        let output = mailbox.take().unwrap();
        assert_eq!((output.serial, output.processed_levels, output.issued_at), (12, 7, newest_stamp));
        assert_eq!(output.levels.iter().map(|diff| diff.level).collect::<Vec<_>>(), [0, 1]);
        assert_eq!((output.levels[0].center.x, output.levels[0].radius), (11.0, 11.5));
        assert_eq!((output.levels[1].center.x, output.levels[1].radius), (10.0, 10.5));
        assert_eq!(output.levels[0].adds, [(0.125, 200)]);
        assert_eq!(output.levels[1].adds, [(1.125, 101)]);
        assert!(std::sync::Arc::ptr_eq(&output.wanted[0].1, &changed_level0));
        assert!(std::sync::Arc::ptr_eq(&output.wanted[1].1, &first_level1));
        assert!(mailbox.take().is_none());
    }

    #[test]
    fn latest_mailbox_inactive_replaces_active_and_stale_message_cannot_restore_it() {
        let mailbox = WindowMailbox::default();
        let active = mailbox_snapshot(10, 7, &[(0, true, 100)]);
        let old_wanted = active.wanted[0].1.clone();
        assert!(mailbox.publish(0, active));
        assert!(mailbox.publish(0, mailbox_snapshot(11, 7, &[(0, false, 0)])));
        assert_eq!(std::sync::Arc::strong_count(&old_wanted), 1, "superseded wanted was released");
        assert!(mailbox.publish(0, mailbox_snapshot(9, 7, &[(0, true, 999)])));
        assert!(mailbox.publish(0, mailbox_snapshot(12, 7, &[])));
        let output = mailbox.take().unwrap();
        assert_eq!(output.serial, 12);
        assert_eq!(output.levels.len(), 1);
        assert!(!output.levels[0].active);
        assert_eq!(output.levels[0].center.x, 11.0);
        assert!(output.levels[0].adds.is_empty());
        assert!(output.wanted[0].1.is_empty());
    }

    #[test]
    fn latest_mailbox_bounds_two_pending_outputs_and_does_not_mix_ranges() {
        let mailbox = WindowMailbox::default();
        for serial in 1..=100 {
            assert!(mailbox.publish(0, mailbox_snapshot(serial, 7, &[(0, true, serial)])));
            assert!(mailbox.publish(1, mailbox_snapshot(serial, 0x78, &[(3, true, serial + 1000)])));
            assert_eq!(mailbox.state.lock().unwrap().pending.iter().filter(|slot| slot.is_some()).count(), 2);
        }
        let fine = mailbox.take().unwrap();
        let far = mailbox.take().unwrap();
        assert_eq!((fine.serial, fine.processed_levels), (100, 7));
        assert_eq!((far.serial, far.processed_levels), (100, 0x78));
        assert_eq!(fine.levels[0].adds[0].1, 100);
        assert_eq!(far.levels[0].adds[0].1, 1100);
        assert!(mailbox.take().is_none());
        // Fair take order resumes at the other range when only fine was taken.
        assert!(mailbox.publish(0, mailbox_snapshot(101, 7, &[])));
        assert_eq!(mailbox.take().unwrap().processed_levels, 7);
        assert!(mailbox.publish(0, mailbox_snapshot(102, 7, &[])));
        assert!(mailbox.publish(1, mailbox_snapshot(102, 0x78, &[])));
        assert_eq!(mailbox.take().unwrap().processed_levels, 0x78);
    }

    #[test]
    fn fine_worker_keeps_computing_latest_request_with_both_outputs_undrained() {
        let (grid, mut request) = flat_request(crate::grid::Shape::Plane, 0.3, 6.0, 0.25);
        let worker = WindowWorker::start(grid);
        worker.request(request.clone());
        let wait_pending = |serial| {
            let until = std::time::Instant::now() + std::time::Duration::from_secs(2);
            loop {
                let matched = worker.updates.state.lock().unwrap().pending[0].as_ref().is_some_and(|update| update.serial == serial);
                if matched { return true; }
                if std::time::Instant::now() >= until { return false; }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        };
        assert!(wait_pending(request.serial));
        request.serial += 1;
        request.eye.x += 20.0;
        worker.request(request.clone());
        assert!(wait_pending(request.serial), "fine computation waited for undrained output");
        request.serial += 1;
        worker.request(request.clone());
        assert!(wait_pending(request.serial), "fine no-op authority waited for consumer");
        let output = worker.updates.state.lock().unwrap().pending[0].take().unwrap();
        assert_eq!(output.serial, request.serial);
        assert_eq!(output.processed_levels, 7);
        assert!(!output.wanted.is_empty(), "latest no-op lost unpublished full demand");
        assert!(output.levels.iter().any(|diff| diff.active && !diff.adds.is_empty()));
    }

    #[test]
    fn bounded_worker_output_closes_before_shutdown_join() {
        let updates = std::sync::Arc::new(WindowMailbox::default());
        let producer = updates.clone();
        let (ready, ready_rx) = mpsc::channel();
        let worker = WindowWorker { requests: None, updates,
            threads: vec![std::thread::spawn(move || {
                assert!(producer.publish(0, mailbox_snapshot(1, 7, &[])));
                ready.send(()).unwrap();
                let mut serial = 2;
                while producer.publish(0, mailbox_snapshot(serial, 7, &[])) { serial += 1; }
                assert!(producer.is_closed());
            })],
        };
        ready_rx.recv().unwrap();
        let (done, done_rx) = mpsc::channel();
        std::thread::spawn(move || { drop(worker); done.send(()).unwrap(); });
        done_rx.recv_timeout(std::time::Duration::from_secs(2)).expect("latest mailbox shutdown deadlocked");
    }

    #[test]
    fn bounded_worker_output_closes_before_both_shutdown_joins() {
        let updates = std::sync::Arc::new(WindowMailbox::default());
        let (ready, ready_rx) = mpsc::channel();
        let threads = (0..2).map(|range| {
            let producer = updates.clone();
            let ready = ready.clone();
            std::thread::spawn(move || {
                let mask = if range == 0 { 7 } else { 0x78 };
                assert!(producer.publish(range, mailbox_snapshot(1, mask, &[])));
                ready.send(()).unwrap();
                let mut serial = 2;
                while producer.publish(range, mailbox_snapshot(serial, mask, &[])) { serial += 1; }
                assert!(producer.is_closed());
            })
        }).collect();
        let worker = WindowWorker { requests: None, updates, threads };
        ready_rx.recv().unwrap(); ready_rx.recv().unwrap();
        let (done, done_rx) = mpsc::channel();
        std::thread::spawn(move || { drop(worker); done.send(()).unwrap(); });
        done_rx.recv_timeout(std::time::Duration::from_secs(2)).expect("two latest window producers failed to join");
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

    #[test]
    fn current_block_demand_matches_independent_global_column_oracle() {
        for shape in [crate::grid::Shape::Plane, crate::grid::Shape::Sphere, crate::grid::Shape::InfinitePlane] {
            let planet = crate::planet::Planet::new(crate::PlanetRecipe { shape,
                radius_m: 1000.0, plane_size_m: 1024.0, voxel_size_m: 0.1,
                terrain: crate::TerrainSource { generator: crate::landform::FLAT_ID.into(), ..Default::default() },
                ..Default::default() }).unwrap();
            let grid = *planet.grid();
            for dither in [0.0, 0.25, 1.0] {
                for direction in [DVec3::Y, DVec3::new(1.0, 1.0, 0.0).normalize(),
                    DVec3::new(1.0, 1.0, 1.0).normalize()] {
                    let eye = if grid.is_plane() { DVec3::new(2.1, 2.0, -1.7) }
                        else { direction * (grid.radius() + 2.0) };
                    let request = WindowRequest { eye, prefetch_eye: None, priority_eye: None,
                        view_focus: None, lod0: 8.0, lod_dither: dither, outer_radius: planet.outer_radius(),
                        planet: None, serial: 1 };
                    let mut oracle = WindowPlanner::new(grid);
                    oracle.snapshot = true;
                    let update = oracle.update_range_column_snapshot_oracle(&request, 0..grid.levels());
                    for level in 0..grid.levels().min(3) {
                        let wanted = update.wanted.iter().find(|(l, _)| *l == level).unwrap().1.clone();
                        let mut blocks: FxHashSet<_> = wanted.iter().map(|key| key & !(3u64 | (3u64 << 32))).collect();
                        for &face in grid.faces() {
                            let columns = grid.cells() / (BRICK << level);
                            // Include rejected remote/boundary/face keys, not just positive oracle keys.
                            // InfinitePlane L0 has exactly 2^24 columns: its one-past
                            // i cannot be passed to checked key0. The 32-bit j word
                            // can represent that boundary without corrupting face bits.
                            for (i, j) in [(0, 0), (0, columns), (columns - 4, columns - 4)] {
                                blocks.insert(pack(key0(face, level, i), j as u32));
                            }
                            if columns < 1 << 24 {
                                blocks.insert(pack(key0(face, level, columns), 0));
                            }
                            // Raw malformed keys test decoder rejection, without
                            // violating the production encoder's 24-bit invariant.
                            assert!(!current_block_wanted(&grid, &request,
                                pack((7u32 << 24) | (level << 27), 0)));
                            assert!(!current_block_wanted(&grid, &request,
                                pack(key0(face, level, 0), u32::MAX)));
                        }
                        for block in blocks {
                            let present = wanted.contains(&block);
                            assert_eq!(current_block_wanted(&grid, &request, block), present,
                                "shape {shape:?} level {level} dither {dither} block {block}");
                        }
                    }
                    let invalid = WindowRequest { lod0: f64::NAN, ..request.clone() };
                    assert!(!current_block_wanted(&grid, &invalid, 0));
                }
            }
        }
    }

    #[test]
    fn current_block_demand_rejects_altitude_invalid_and_remote_keys() {
        let grid = Grid::plane(crate::grid::Shape::Plane, 1024.0, 0.1).unwrap();
        let request = WindowRequest { eye: DVec3::Y * 2.0, prefetch_eye: None,
            priority_eye: None, view_focus: None, lod0: 8.0, lod_dither: 0.25,
            outer_radius: grid.radius(), planet: None, serial: 1 };
        let (cell, _) = grid.locate(DVec3::ZERO);
        let key = pack(key0(cell.face, 0, (cell.i >> 3) & !3), ((cell.j >> 3) & !3) as u32);
        assert!(current_block_wanted(&grid, &request, key));
        assert!(!current_block_wanted(&grid, &WindowRequest { eye: DVec3::Y * 100.0, ..request.clone() }, key));
        assert!(!current_block_wanted(&grid, &request, key + 1));
        assert!(!current_block_wanted(&grid, &request, pack(key0(cell.face, 3, 0), 0)));
        assert!(!current_block_wanted(&grid, &request, pack(key0(cell.face, 0, 0), 0)));
        // Clipped edge rectangle itself uses the same scanner predicate.
        assert!(plane_block_in_circle(2.1, 1.8, 0, 2, 0, 2, 0, 2, 0, 2, 1.0));
        assert!(!plane_block_in_circle(2.1, 1.8, 4, 6, 4, 6, 0, 2, 0, 2, 1.0));
    }

}
