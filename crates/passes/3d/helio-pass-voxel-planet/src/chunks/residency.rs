//! Chunk residency (docs/chunk-trees.md, phase 2): which tiles of the
//! quadtree are drawn, generated next and evicted, for a moving eye.
//!
//! A tile is refined while its cells would project larger than
//! `lod_pixels` pixels from the eye or from where the eye will be
//! `prefetch_seconds` ahead (and stays refined until they fall under
//! `coarsen` of that); out of view, `offscreen_pixels` times that. A refined tile keeps being drawn until all four of
//! its children are generated, so nothing is ever missing: detail arrives
//! as whole tiles replacing their parent. Children are requested once
//! their parent's layers are measured (their chunks then span only the
//! layers that can hold a surface), most wanted first. The cut covers the
//! whole world: roots are always resident, and tiles the eye cannot see
//! are drawn as they are, not refined (rays graze the horizon, and the
//! plan lags a climbing eye). Other tiles no frame has visited for
//! `keep_frames` are evicted.
//!
//! Pure CPU logic over tile keys and layer ranges, run on a [`Worker`]
//! thread; the GPU work lives in `chunks`.

use super::{tile_distance, Eye, TileKey, CHUNK};
use crate::grid::Grid;
use glam::DVec3;
use rustc_hash::{FxHashMap, FxHashSet};

/// Layers (level cells, [lo, hi)) a tile's chunks may hold: the field's
/// volume (caves) and the band around its surface; and the layers its
/// edits change (`NO_EDITS`: none), which only its extent includes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ranges {
    pub volume: (i32, i32),
    pub surface: (i32, i32),
    pub edits: (i32, i32),
}

/// `Ranges::edits` of a tile without edits.
pub const NO_EDITS: (i32, i32) = (i32::MAX, i32::MIN);

impl Ranges {
    /// A child's ranges from its parent's: the same heights in cells half
    /// as tall, widened by a parent cell (a level stays within its
    /// parent's cell of the parent's surface).
    fn child(self) -> Self {
        let half = |(lo, hi): (i32, i32)| (lo.saturating_mul(2) - 2, hi.saturating_mul(2) + 2);
        Self { volume: half(self.volume), surface: half(self.surface), edits: if self.edits.0 > self.edits.1 { NO_EDITS } else { half(self.edits) } }
    }

    /// Every layer the tile may hold something in: its volume and its
    /// edits (how far it is from an eye).
    pub fn extent(self) -> (i32, i32) {
        (self.volume.0.min(self.edits.0), self.volume.1.max(self.edits.1))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PlanConfig {
    /// Radians a pixel spans.
    pub pixel_angle: f64,
    /// Refine while a cell would cover more pixels than this.
    pub lod_pixels: f64,
    /// Out of view (past the view cone widened by `view_guard` radians),
    /// cells may cover this many times `lod_pixels`: a level coarser at 2,
    /// a quarter of the tiles there. A turn brings them in a level away.
    pub offscreen_pixels: f64,
    pub view_guard: f64,
    /// Coarsen again only under this share of `lod_pixels`.
    pub coarsen: f64,
    /// Planes: tiles farther than this are not drawn (spheres draw up to
    /// the horizon).
    pub max_distance: f64,
    /// Also refine for where the eye will be this far ahead.
    pub prefetch_seconds: f64,
    pub keep_frames: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct Tile {
    pub ranges: Ranges,
    /// Ranges measured on the GPU (else estimated from the parent's).
    pub exact: bool,
    /// Levels under the nearest ancestor whose ranges were measured (0:
    /// measured, or a root): an estimate's margin doubles at each.
    pub guess: u8,
    pub generated: bool,
    /// Its children were drawn in its place last time it was visited.
    pub refined: bool,
    pub used: u64,
    /// The plan that last drew it on screen (frame + 1; 0: never).
    pub drawn: u64,
}

/// Where the eye looks: tiles in view are generated first.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct View {
    pub forward: DVec3,
    /// Cosine of the half-angle of the cone around `forward` in view.
    pub cos_half: f64,
}

/// One frame's decisions.
#[derive(Clone, Debug, Default)]
pub struct Plan {
    /// Generated tiles to draw (they cover the view once each).
    pub drawn: Vec<TileKey>,
    /// Tiles to generate with their estimated ranges, most wanted first
    /// (at most `MAX_REQUESTS`).
    pub requests: Vec<(TileKey, Ranges)>,
    /// Tiles waiting to be generated in all.
    pub waiting: usize,
    /// Drawn tiles whose cells project larger than wanted, as a share of
    /// the drawn tiles (the refinement still to come).
    pub coarse_share: f64,
    /// Pixels the largest cell of a drawn tile in view spans from the eye
    /// (above the finest level).
    pub worst_px: f64,
    /// Tiles in view this plan switches between levels, and the pixels the
    /// largest cell of those spans from the eye (the coarser side of each
    /// switch): a switch shows when that is well over a pixel.
    pub switches: usize,
    pub switch_px: f64,
}

pub struct Planner {
    pub tiles: FxHashMap<TileKey, Tile>,
    roots: Vec<TileKey>,
    config: PlanConfig,
    /// Spheres: the radius of the lowest ground, whose horizon hides what
    /// lies beyond it (none when that ground reaches the centre).
    occluder: Option<f64>,
    /// The last plan's frame + 1 (0: none yet).
    last_plan: u64,
}

impl Planner {
    /// Roots: every tile of the coarsest level (whatever the eye's first
    /// place, it can go anywhere), with the planet's whole height range
    /// (`range`, base layers) as their estimate.
    pub fn new(grid: &Grid, range: (i32, i32), config: PlanConfig) -> Self {
        let level = grid.levels() - 1;
        let n = tiles_at(grid, level);
        let layers = ((range.0 >> level) - 1, (range.1 >> level) + 1);
        let ranges = Ranges { volume: layers, surface: layers, edits: NO_EDITS };
        let mut roots = Vec::new();
        let mut tiles = FxHashMap::default();
        for &face in grid.faces() {
            for cx in 0..n {
                for cy in 0..n {
                    let key = TileKey { face, level, cx, cy };
                    roots.push(key);
                    tiles.insert(key, Tile { ranges, exact: false, guess: 0, generated: false, refined: false, used: 0, drawn: 0 });
                }
            }
        }
        // A world whose lowest ground reaches its centre has no horizon.
        let occluder = (!grid.is_plane()).then(|| grid.radius() + f64::from(range.0) * f64::from(grid.layer_mm()) * 1e-3).filter(|r| *r > 0.0);
        Self { tiles, roots, config, occluder, last_plan: 0 }
    }

    /// Whether `tile` cannot be seen from `eye`: below its horizon over the
    /// lowest ground (spheres; an eye under that ground sees past it), or
    /// past `max_distance` (planes). The test is conservative: the tile's
    /// nearest point against the horizon distances of the eye and of the
    /// tile's highest layer.
    fn beyond(&self, grid: &Grid, key: TileKey, ranges: Ranges, eye: &Eye) -> bool {
        let distance = tile_distance(grid, key, ranges.extent(), eye);
        let Some(r) = self.occluder else { return distance > self.config.max_distance };
        let over = |radius: f64| (radius * radius - r * r).max(0.0).sqrt();
        let e = eye.position.length();
        if e <= r {
            return false;
        }
        let top = grid.radius() + f64::from(ranges.extent().1) * grid.level_size(key.level);
        distance > over(e) + over(top)
    }

    pub fn config(&self) -> &PlanConfig {
        &self.config
    }

    /// Cells of `tile` project this many times `lod_pixels` from the
    /// nearest of `eyes`.
    fn error(&self, grid: &Grid, key: TileKey, ranges: Ranges, eyes: &[Eye]) -> f64 {
        let distance = eyes.iter().map(|e| tile_distance(grid, key, ranges.extent(), e)).fold(f64::INFINITY, f64::min);
        grid.level_size(key.level) / distance.max(1e-3) / (self.config.pixel_angle * self.config.lod_pixels)
    }

    /// Visit the quadtree from the roots for an eye at `eye` moving at
    /// `velocity`.
    pub fn plan(&mut self, grid: &Grid, eye: DVec3, velocity: DVec3, view: Option<View>, frame: u64) -> Plan {
        let eyes = [Eye::new(grid, eye), Eye::new(grid, eye + velocity * self.config.prefetch_seconds)];
        // In view: the tile's nearest point within the cone (generous for
        // a tile's extent: anything beside it is near it); near it: within
        // the cone widened by the guard.
        let cone = |key: TileKey, ranges: Ranges, cos_half: f64| -> bool {
            let Some(v) = view else { return true };
            let offset = eyes[0].toward(grid, key, ranges.extent());
            let length = offset.length();
            let size = grid.level_size(key.level) * f64::from(CHUNK) * 1.5;
            length < size || offset.dot(v.forward) > cos_half * length - size
        };
        let guarded = view.map_or(-1.0, |v| (v.cos_half.clamp(-1.0, 1.0).acos() + self.config.view_guard).min(std::f64::consts::PI).cos());
        let in_view = |key: TileKey, ranges: Ranges| cone(key, ranges, view.map_or(-1.0, |v| v.cos_half));
        let mut drawn_in_view = 0usize;
        let mut plan = Plan::default();
        let mut wanted: Vec<(f64, TileKey)> = Vec::new();
        let mut coarse = 0usize;
        let mut stack: Vec<TileKey> = self.roots.clone();
        while let Some(key) = stack.pop() {
            let Some(tile) = self.tiles.get(&key).copied() else { continue };
            // The cut covers the whole world, whatever the eye sees: a ray
            // grazing the horizon crosses the air over tiles beyond it, and
            // the plan lags an eye that climbs. Roots are always resident
            // (only a root can be ungenerated here: children are visited
            // once all four are generated).
            if !tile.generated {
                wanted.push((f64::INFINITY, key));
                continue;
            }
            // Tiles that cannot be seen are drawn as they are, not refined.
            // Tiles the prefetch eye sees are ready when they come over the
            // horizon (a climb), not drawn coarse for a moment.
            let hidden = self.beyond(grid, key, tile.ranges, &eyes[0]);
            let beyond = hidden && self.beyond(grid, key, tile.ranges, &eyes[1]);
            if beyond {
                let t = self.tiles.get_mut(&key).unwrap();
                t.used = frame;
                t.refined = false;
                plan.drawn.push(key);
                continue;
            }
            let error = self.error(grid, key, tile.ranges, &eyes);
            let seen = in_view(key, tile.ranges);
            // On screen now: in view and over the eye's horizon (a tile only
            // the prefetch eye sees is drawn, unseen, until it is).
            let shows = seen && !hidden;
            // Pixels a cell spans from the eye now (not the prefetch eye).
            let now = self.error(grid, key, tile.ranges, &eyes[..1]);
            // Tiles in view first, by what they need now: the prefetch
            // eye's wants (a burst as the eye starts moving) come after
            // what is coarse on screen.
            let priority = if seen { now * 1e6 } else { now };
            // Out of the guarded view, coarser cells will do.
            let wanted_error = if seen || cone(key, tile.ranges, guarded) { error } else { error / self.config.offscreen_pixels };
            let finer = key.level > 0 && if tile.refined { wanted_error > self.config.coarsen } else { wanted_error > 1.0 };
            // Children wait for their parent's measured layers, unless it is
            // far too coarse (an eye that dropped from orbit) and a measured
            // ancestor is near: then they come with ranges estimated from it
            // (a readback deepens them).
            let deep = wanted_error >= DEEP_REFINE;
            let guess = if tile.exact { 0 } else { tile.guess };
            let children = if finer && (tile.exact || deep && guess < GUESS_LEVELS) { children(grid, key) } else { Vec::new() };
            let mut ready = !children.is_empty();
            for &child in &children {
                let entry = self.tiles.entry(child).or_insert(Tile { ranges: tile.ranges.child(), exact: false, guess: guess + 1, generated: false, refined: false, used: frame, drawn: 0 });
                entry.used = frame;
                // Until measured, a child's ranges follow its parent's (an
                // edit may have deepened them since).
                if !entry.exact {
                    entry.ranges = tile.ranges.child();
                    entry.guess = guess + 1;
                }
                if !entry.generated {
                    ready = false;
                    wanted.push((priority, child));
                }
            }
            // Far too coarse: the tiles under the eye down to the level it
            // wants are wanted now (each level's four, along the child
            // nearest it, up to `GUESS_LEVELS` under a measured tile), so
            // the cut refines through them as they come, not a measured
            // level at a time.
            if !children.is_empty() && deep {
                let mut at = key;
                let mut ranges = tile.ranges;
                let mut g = guess;
                loop {
                    ranges = ranges.child();
                    g += 1;
                    let near = |k: &TileKey| tile_distance(grid, *k, ranges.extent(), &eyes[0]);
                    let Some(next) = self::children(grid, at).into_iter().min_by(|a, b| near(a).total_cmp(&near(b))) else { break };
                    // Measured since: its own ranges.
                    if let Some(t) = self.tiles.get(&next).filter(|t| t.exact) {
                        ranges = t.ranges;
                        g = 0;
                    }
                    if next.level == 0 || g >= GUESS_LEVELS || self.error(grid, next, ranges, &eyes) <= 1.0 {
                        break;
                    }
                    for c in self::children(grid, next) {
                        let entry = self.tiles.entry(c).or_insert(Tile { ranges: ranges.child(), exact: false, guess: g + 1, generated: false, refined: false, used: frame, drawn: 0 });
                        entry.used = frame;
                        if !entry.generated {
                            wanted.push((priority, c));
                        }
                    }
                    at = next;
                }
            }

            // A switch shows where the last plan drew the side it leaves
            // (not where the tile came over the horizon).
            let shown = |k: &TileKey| self.last_plan != 0 && self.tiles.get(k).is_some_and(|t| t.drawn == self.last_plan);
            if shows && tile.refined != ready && if ready { shown(&key) } else { self::children(grid, key).iter().any(shown) } {
                plan.switches += 1;
                plan.switch_px = plan.switch_px.max(now);
            }
            if shows && !ready && key.level > 0 {
                plan.worst_px = plan.worst_px.max(now);
            }
            let t = self.tiles.get_mut(&key).unwrap();
            t.used = frame;
            t.refined = ready;
            if ready {
                stack.extend(children);
            } else {
                if shows {
                    t.drawn = frame + 1;
                }
                plan.drawn.push(key);
                if shows {
                    drawn_in_view += 1;
                    // The finest level cannot be finer.
                    if error > 1.5 && key.level > 0 {
                        coarse += 1;
                    }
                }
            }
        }
        plan.waiting = wanted.len();
        if wanted.len() > MAX_REQUESTS {
            wanted.select_nth_unstable_by(MAX_REQUESTS, |a, b| b.0.total_cmp(&a.0));
            wanted.truncate(MAX_REQUESTS);
        }
        wanted.sort_by(|a, b| b.0.total_cmp(&a.0));
        plan.requests = wanted.into_iter().map(|w| (w.1, self.tiles[&w.1].ranges)).collect();
        plan.coarse_share = coarse as f64 / drawn_in_view.max(1) as f64;
        self.last_plan = frame + 1;
        plan
    }

    pub fn set_generated(&mut self, key: TileKey) {
        if let Some(t) = self.tiles.get_mut(&key) {
            t.generated = true;
        }
    }

    /// `key`'s chunks are out of date until it is generated again.
    pub fn set_stale(&mut self, key: TileKey) {
        if let Some(t) = self.tiles.get_mut(&key) {
            t.generated = false;
        }
    }

    /// Ranges measured on the GPU for `key`.
    pub fn set_ranges(&mut self, key: TileKey, ranges: Ranges) {
        if let Some(t) = self.tiles.get_mut(&key) {
            t.ranges = ranges;
            t.exact = true;
            t.guess = 0;
        }
    }

    /// Tiles not visited for `keep_frames`, removed (roots never are: the
    /// cut always covers the world).
    pub fn evict(&mut self, frame: u64) -> Vec<TileKey> {
        let keep = self.config.keep_frames;
        let root = self.roots[0].level;
        let old: Vec<TileKey> = self.tiles.iter().filter(|(k, t)| t.used + keep < frame && k.level != root).map(|(k, _)| *k).collect();
        for k in &old {
            self.tiles.remove(k);
        }
        old
    }
}

/// What happened to tiles since the worker's last plan.
#[derive(Clone, Copy, Debug)]
pub enum Event {
    Generated(TileKey),
    Measured(TileKey, Ranges),
    /// Edited and not generated again yet: its chunks hold what was there
    /// before. It is not drawn (its nearest generated ancestor is) until
    /// it is `Generated` again.
    Stale(TileKey),
}

/// A plan the worker made, and the tiles it evicted.
pub struct Planned {
    pub plan: Plan,
    /// Tiles drawn now and not by the last plan sent, and the reverse.
    pub added: Vec<TileKey>,
    pub removed: Vec<TileKey>,
    pub evicted: Vec<TileKey>,
    /// What the plan was made for.
    pub made_for: PlannedFor,
}

/// What a plan was made for: events up to this number had reached the
/// planner; the eye and the view.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlannedFor {
    pub events_seen: u64,
    pub eye: DVec3,
    pub view: Option<View>,
}

struct Job {
    events_seen: u64,
    eye: DVec3,
    velocity: DVec3,
    view: Option<View>,
    frame: u64,
    events: Vec<Event>,
}

/// The planner on its own thread: a job at a time, the latest plan taken
/// without waiting.
pub struct Worker {
    jobs: std::sync::mpsc::Sender<Job>,
    results: std::sync::Mutex<std::sync::mpsc::Receiver<Planned>>,
    busy: bool,
    /// Events submitted so far.
    events_sent: u64,
}

impl Worker {
    pub fn spawn(grid: Grid, range: (i32, i32), config: PlanConfig) -> Self {
        let (jobs, inbox) = std::sync::mpsc::channel::<Job>();
        let (outbox, results) = std::sync::mpsc::channel();
        std::thread::Builder::new()
            .name("chunk residency".into())
            .spawn(move || {
                let mut planner = Planner::new(&grid, range, config);
                let mut drawn: FxHashSet<TileKey> = FxHashSet::default();
                while let Ok(job) = inbox.recv() {
                    for event in job.events {
                        match event {
                            Event::Generated(k) => planner.set_generated(k),
                            Event::Measured(k, r) => planner.set_ranges(k, r),
                            Event::Stale(k) => planner.set_stale(k),
                        }
                    }
                    let plan = planner.plan(&grid, job.eye, job.velocity, job.view, job.frame);
                    let evicted = planner.evict(job.frame);
                    let now: FxHashSet<TileKey> = plan.drawn.iter().copied().collect();
                    let added = plan.drawn.iter().filter(|k| !drawn.contains(k)).copied().collect();
                    let removed = drawn.iter().filter(|k| !now.contains(k)).copied().collect();
                    drawn = now;
                    if outbox.send(Planned { plan, added, removed, evicted, made_for: PlannedFor { events_seen: job.events_seen, eye: job.eye, view: job.view } }).is_err() {
                        break;
                    }
                }
            })
            .expect("chunk residency thread");
        Self { jobs, results: std::sync::Mutex::new(results), busy: false, events_sent: 0 }
    }

    /// Plan for this eye with these events unless a plan is under way
    /// (then the events stay with the caller).
    pub fn submit(&mut self, eye: DVec3, velocity: DVec3, view: Option<View>, frame: u64, events: &mut Vec<Event>) {
        if !self.busy {
            self.events_sent += events.len() as u64;
            self.busy = self.jobs.send(Job { events_seen: self.events_sent, eye, velocity, view, frame, events: std::mem::take(events) }).is_ok();
        }
    }

    /// The finished plan, if any.
    pub fn poll(&mut self) -> Option<Planned> {
        let planned = self.results.get_mut().ok()?.try_recv().ok()?;
        self.busy = false;
        Some(planned)
    }

    /// Events submitted so far.
    pub fn events_sent(&self) -> u64 {
        self.events_sent
    }
}

/// Requests a plan lists at most (more than a frame generates).
pub const MAX_REQUESTS: usize = 2048;

/// Cells this many times `lod_pixels` across are far too coarse: their
/// children and the tiles under the eye are wanted before measurements.
const DEEP_REFINE: f64 = 4.0;

/// Levels under a measured tile whose ranges may be estimated: a margin of
/// two of its cells is eight cells three levels down (deeper, chunk layers
/// would balloon).
const GUESS_LEVELS: u8 = 3;

/// Tiles along a face edge at `level`.
pub fn tiles_at(grid: &Grid, level: u32) -> i32 {
    ((grid.cells() >> level) + CHUNK - 1) / CHUNK
}

/// The (up to) four children of `key` inside its face.
pub fn children(grid: &Grid, key: TileKey) -> Vec<TileKey> {
    let n = tiles_at(grid, key.level - 1);
    [(0, 0), (1, 0), (0, 1), (1, 1)]
        .into_iter()
        .map(|(a, b)| TileKey { face: key.face, level: key.level - 1, cx: key.cx * 2 + a, cy: key.cy * 2 + b })
        .filter(|c| c.cx < n && c.cy < n)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (Grid, DVec3, Planner) {
        let grid = Grid::new(6_371_000.0, 0.1).unwrap();
        let eye = grid.position(2, [f64::from(grid.cells()) * 0.47, f64::from(grid.cells()) * 0.53, 20.0]);
        let config = PlanConfig { pixel_angle: 7.67e-4, lod_pixels: 1.0, offscreen_pixels: 2.0, view_guard: 0.35, coarsen: 0.8, max_distance: 400_000.0, prefetch_seconds: 0.5, keep_frames: 60 };
        let planner = Planner::new(&grid, (-100_000, 100_000), config);
        (grid, eye, planner)
    }

    /// Generate every request, as if the budget were unlimited.
    fn settle(grid: &Grid, eye: DVec3, planner: &mut Planner, frame: &mut u64) -> Plan {
        loop {
            *frame += 1;
            let plan = planner.plan(grid, eye, DVec3::ZERO, None, *frame);
            if plan.requests.is_empty() {
                return plan;
            }
            for (k, ranges) in plan.requests {
                planner.set_generated(k);
                planner.set_ranges(k, ranges);
            }
        }
    }

    #[test]
    fn a_settled_cut_reaches_level_zero_and_covers_each_column_once() {
        let (grid, eye, mut planner) = setup();
        let mut frame = 0;
        let plan = settle(&grid, eye, &mut planner, &mut frame);
        assert!(plan.drawn.iter().any(|k| k.level == 0));
        assert_eq!(plan.coarse_share, 0.0);
        // No drawn tile is an ancestor of another.
        let drawn: std::collections::HashSet<TileKey> = plan.drawn.iter().copied().collect();
        for k in &plan.drawn {
            let mut p = *k;
            while p.level + 1 < grid.levels() {
                p = TileKey { face: p.face, level: p.level + 1, cx: p.cx >> 1, cy: p.cy >> 1 };
                assert!(!drawn.contains(&p), "{k:?} under drawn {p:?}");
            }
        }
    }

    /// The root over the eye's column.
    fn root_of(grid: &Grid, mut k: TileKey) -> TileKey {
        while k.level + 1 < grid.levels() {
            k = TileKey { level: k.level + 1, cx: k.cx >> 1, cy: k.cy >> 1, ..k };
        }
        k
    }

    /// Every root's region is covered by drawn tiles, none under another,
    /// whatever the eye sees: rays graze the horizon and the plan lags an
    /// eye that climbs.
    fn assert_covers_the_world(grid: &Grid, planner: &Planner, plan: &Plan) {
        let drawn: std::collections::HashSet<TileKey> = plan.drawn.iter().copied().collect();
        let mut covered = FxHashSet::default();
        for k in &plan.drawn {
            covered.insert(root_of(grid, *k));
            let mut p = *k;
            while p.level + 1 < grid.levels() {
                p = TileKey { face: p.face, level: p.level + 1, cx: p.cx >> 1, cy: p.cy >> 1 };
                assert!(!drawn.contains(&p), "{k:?} under drawn {p:?}");
            }
        }
        for r in &planner.roots {
            assert!(covered.contains(r), "nothing drawn over {r:?}");
        }
        // A refined tile has all four children drawn or refined.
        for k in &plan.drawn {
            if k.level + 1 < grid.levels() {
                let parent = TileKey { face: k.face, level: k.level + 1, cx: k.cx >> 1, cy: k.cy >> 1 };
                for c in children(grid, parent) {
                    assert!(drawn.contains(&c) || planner.tiles.get(&c).is_some_and(|t| t.refined), "{c:?} beside drawn {k:?} is missing");
                }
            }
        }
    }

    #[test]
    fn the_cut_covers_the_world_wherever_the_eye_goes() {
        let (grid, eye, mut planner) = setup();
        let mut frame = 1;
        // From the far side of the planet the eye's ground is below the
        // horizon: drawn as its root, not refined.
        let far = -eye;
        let plan = settle(&grid, far, &mut planner, &mut frame);
        assert_covers_the_world(&grid, &planner, &plan);
        let here = settle(&grid, eye, &mut planner, &mut frame);
        assert_covers_the_world(&grid, &planner, &here);
        let root = root_of(&grid, *here.drawn.iter().find(|k| k.level == 0).unwrap());
        assert!(plan.drawn.contains(&root), "the far side refined the eye's ground");
        // Gone to the far side again, the eye's fine tiles age out; roots
        // never do.
        settle(&grid, far, &mut planner, &mut frame);
        frame += planner.config().keep_frames + 1;
        let plan = planner.plan(&grid, far, DVec3::ZERO, None, frame);
        let evicted = planner.evict(frame);
        assert!(!evicted.is_empty() && evicted.iter().all(|k| k.level + 1 < grid.levels()));
        assert!(plan.requests.is_empty());
        assert_covers_the_world(&grid, &planner, &plan);
    }

    /// A climb faster than the plan: from the ground straight to orbit in
    /// one plan, the cut still covers everything the new eye sees.
    #[test]
    fn a_sudden_climb_leaves_no_hole() {
        let (grid, eye, mut planner) = setup();
        let mut frame = 0;
        settle(&grid, eye, &mut planner, &mut frame);
        frame += 1;
        let plan = planner.plan(&grid, eye.normalize() * (grid.radius() + 400_000.0), DVec3::ZERO, None, frame);
        assert_covers_the_world(&grid, &planner, &plan);
    }

    /// From orbit the whole visible disc is drawn: tiles 2,000 km away,
    /// far past any fixed distance, are over the horizon.
    #[test]
    fn orbit_draws_up_to_the_horizon() {
        let (grid, eye, mut planner) = setup();
        let mut frame = 1;
        let orbit = eye.normalize() * (grid.radius() + 300_000.0);
        let plan = settle(&grid, orbit, &mut planner, &mut frame);
        let farthest = plan.drawn.iter().map(|k| tile_distance(&grid, *k, planner.tiles[k].ranges.extent(), &Eye::new(&grid, orbit))).fold(0.0, f64::max);
        assert!(farthest > 1_500_000.0, "drawn to {farthest:.0} m");
    }

    #[test]
    fn a_parent_stays_drawn_until_all_its_children_are_generated() {
        let (grid, eye, mut planner) = setup();
        let mut frame = 1;
        let plan = planner.plan(&grid, eye, DVec3::ZERO, None, frame);
        // Nothing generated: only roots requested, nothing drawn.
        assert!(plan.drawn.is_empty());
        for &(k, ranges) in &plan.requests {
            planner.set_generated(k);
            planner.set_ranges(k, ranges);
        }
        frame += 1;
        let plan = planner.plan(&grid, eye, DVec3::ZERO, None, frame);
        let parent = plan.drawn.iter().copied().find(|k| plan.requests.iter().any(|(r, _)| r.face == k.face && r.level + 1 == k.level && r.cx >> 1 == k.cx && r.cy >> 1 == k.cy)).unwrap();
        let kids = children(&grid, parent);
        // Three of four children generated: the parent is still drawn.
        for k in &kids[..3] {
            planner.set_generated(*k);
        }
        frame += 1;
        let plan = planner.plan(&grid, eye, DVec3::ZERO, None, frame);
        assert!(plan.drawn.contains(&parent));
        assert!(!plan.drawn.iter().any(|k| kids.contains(k)));
        planner.set_generated(kids[3]);
        frame += 1;
        let plan = planner.plan(&grid, eye, DVec3::ZERO, None, frame);
        assert!(!plan.drawn.contains(&parent));
    }

    #[test]
    fn moving_away_coarsens_and_evicts_the_fine_tiles() {
        let (grid, eye, mut planner) = setup();
        let mut frame = 0;
        settle(&grid, eye, &mut planner, &mut frame);
        let fine = planner.tiles.keys().filter(|k| k.level == 0).count();
        assert!(fine > 0);
        let far = eye + eye.normalize() * 200_000.0;
        settle(&grid, far, &mut planner, &mut frame);
        frame += 100;
        planner.plan(&grid, far, DVec3::ZERO, None, frame);
        let evicted = planner.evict(frame);
        assert!(evicted.iter().filter(|k| k.level == 0).count() == fine);
    }

    /// Out of view, a level coarser: the in-view cut is as fine, the whole
    /// cut much smaller.
    #[test]
    fn out_of_view_tiles_are_a_level_coarser() {
        let (grid, eye, _) = setup();
        let up = eye.normalize();
        let forward = (up.any_orthonormal_vector() - up * 0.3).normalize();
        let view = View { forward, cos_half: 0.4f64.cos() };
        let cut = |offscreen: f64| {
            let config = PlanConfig { pixel_angle: 7.67e-4, lod_pixels: 1.0, offscreen_pixels: offscreen, view_guard: 0.35, coarsen: 0.8, max_distance: 400_000.0, prefetch_seconds: 0.5, keep_frames: 60 };
            let mut planner = Planner::new(&grid, (-100_000, 100_000), config);
            let mut frame = 0;
            loop {
                frame += 1;
                let plan = planner.plan(&grid, eye, DVec3::ZERO, Some(view), frame);
                if plan.requests.is_empty() {
                    return plan;
                }
                for (k, ranges) in plan.requests {
                    planner.set_generated(k);
                    planner.set_ranges(k, ranges);
                }
            }
        };
        let (full, biased) = (cut(1.0), cut(2.0));
        assert_eq!(biased.coarse_share, 0.0);
        assert!(biased.worst_px <= full.worst_px + 1e-9, "{} > {}", biased.worst_px, full.worst_px);
        assert!(biased.drawn.len() * 2 < full.drawn.len(), "{} of {}", biased.drawn.len(), full.drawn.len());
    }

    #[test]
    fn prefetch_refines_ahead_of_a_moving_eye() {
        let (grid, eye, mut planner) = setup();
        let mut frame = 0;
        settle(&grid, eye, &mut planner, &mut frame);
        let ahead = (grid.position(2, [f64::from(grid.cells()) * 0.47 + 2000.0, f64::from(grid.cells()) * 0.53, 20.0]) - eye).normalize();
        frame += 1;
        let plan = planner.plan(&grid, eye, ahead * 400.0, None, frame);
        assert!(plan.requests.iter().any(|(k, _)| k.level == 0), "nothing requested ahead");
    }
}
