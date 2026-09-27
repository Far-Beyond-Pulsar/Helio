//! A view-selected tree of stored voxel bricks. Selection runs off the render
//! thread; publication retains complete coverage while GPU bricks are generated.
use crate::{
    landforms::RegionClass,
    world::{Edit, World},
    Params,
};
use glam::DVec3;
// Keys and slots are internal bounded integers, not untrusted input strings.
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::sync::Arc;
#[cfg(feature = "regional-publication-experiment")]
mod regional;
mod selection;
#[cfg(test)]
mod tests;

pub const BRICK_WORDS: usize = 2048; // 32^3 exact materials, or 9^3 densities + 8^3 pairs of bounds.
pub const BRICK_CAPACITY: usize = 65_536;
pub const NODE_CAPACITY: usize = 262_144;
pub const PUBLICATION_NODE_CAPACITY: usize = if cfg!(feature = "regional-publication-experiment") {
    NODE_CAPACITY * 2
} else {
    NODE_CAPACITY
};
pub const GENERATION_BATCH: usize = 256;
pub const AIR: u32 = 0xffff_ffff;
pub const SOLID: u32 = 0xffff_fffe;
const BRICK: u32 = 0x8000_0000;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Key {
    pub low: [i32; 3],
    pub level: u32,
}
impl Key {
    pub fn side(self) -> i32 {
        32 << self.level
    }
    fn overlaps(self, edit: Edit) -> bool {
        let r = i64::from(edit.radius_units().div_ceil(2)) + 9;
        (0..3).all(|a| {
            i64::from(self.low[a]) <= i64::from(edit.cell[a]) + r
                && i64::from(self.low[a]) + i64::from(self.side()) + i64::from(self.level > 0)
                    > i64::from(edit.cell[a]) - r
        })
    }
}
// CPU selection nodes; only the child links and root bounds are uploaded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Node {
    pub low: [i32; 3],
    pub level: u32,
    pub child: u32,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Job {
    pub low: [i32; 3],
    pub level: u32,
    pub slot: u32,
    pub pad: [u32; 3],
}
struct Plan {
    nodes: Vec<Node>,
    leaves: Vec<(usize, Key)>,
    world: Arc<World>,
    view: View,
    pixels: f64,
}
#[derive(Clone, Copy)]
struct View {
    eye: DVec3,
    forward: DVec3,
    up: DVec3,
    right: DVec3,
    tan: f64,
    aspect: f64,
    height: f64,
}
impl View {
    fn new(p: &Params) -> Self {
        Self {
            eye: DVec3::from_array(std::array::from_fn(|a| {
                (f64::from(p.origin[a]) + f64::from(p.fraction[a])) * 0.1
            })),
            forward: DVec3::from_array(std::array::from_fn(|a| f64::from(p.forward[a]))),
            up: DVec3::from_array(std::array::from_fn(|a| f64::from(p.up[a]))),
            right: DVec3::from_array(std::array::from_fn(|a| f64::from(p.right[a]))),
            tan: f64::from(p.up[3]).max(0.01),
            aspect: f64::from(p.right[3]).max(0.1),
            height: f64::from(p.screen[1]).max(1.0),
        }
    }
    fn changed(self, other: Self) -> bool {
        self.eye.distance(other.eye) > 1.6
            || self.forward.dot(other.forward) < 0.999
            || self.up.dot(other.up) < 0.999
            || self.tan != other.tan
            || self.height != other.height
            || self.aspect != other.aspect
    }
    fn pixel_size(self, key: Key, pixels: f64) -> f64 {
        let lo = DVec3::from_array(key.low.map(|v| f64::from(v) * 0.1));
        let size = f64::from(key.side()) * 0.1;
        let hi = lo + DVec3::splat(size);
        let distance = self.eye.clamp(lo, hi).distance(self.eye).max(0.01);
        if distance < 16.0 {
            return 0.0;
        } // gameplay radius always uses exact cells
        let relative = (lo + hi) * 0.5 - self.eye;
        let radius = size * 0.8660254038;
        let z = relative.dot(self.forward);
        let visible = z + radius > 0.0
            && relative.dot(self.right).abs() <= z.max(0.0) * self.tan * self.aspect + radius * 2.0
            && relative.dot(self.up).abs() <= z.max(0.0) * self.tan + radius * 2.0;
        let tolerance = if visible || distance < 16.0 {
            pixels
        } else {
            8.0
        };
        distance * 2.0 * self.tan / self.height * tolerance
    }
}

fn classify(world: &World, key: Key) -> u32 {
    let low = world.sample_cell(key.low);
    let high = world.sample_cell(key.low.map(|v| v + key.side() - 1));
    let closest = DVec3::from_array(std::array::from_fn(|a| {
        0i32.clamp(low[a], high[a]) as f64 * 0.1
    }));
    let mut class = if closest.length() > crate::world::procedural_outer_radius() + 0.1 {
        RegionClass::AllAir
    } else {
        world.classify_region(low, high)
    };
    for i in world.region_edits(low, high) {
        let e = world.edits[i];
        let target = if e.material == 0 {
            RegionClass::AllAir
        } else {
            RegionClass::AllSolid
        };
        let far = std::array::from_fn(|a| {
            if (i64::from(low[a]) - i64::from(e.cell[a])).abs()
                > (i64::from(high[a]) - i64::from(e.cell[a])).abs()
            {
                low[a]
            } else {
                high[a]
            }
        });
        if e.contains(far) {
            class = target;
        } else if class != target {
            class = RegionClass::Mixed;
        }
        // Added materials must be generated even if occupancy is uniform.
        if target == RegionClass::AllSolid && class == target {
            class = RegionClass::Mixed;
        }
    }
    match class {
        RegionClass::AllAir => AIR,
        RegionClass::AllSolid => SOLID,
        RegionClass::Mixed => 0,
    }
}
/// Classification is view independent. Keep the certified regions across
/// camera moves and capacity retries instead of resampling the same planet.
/// Edits invalidate intersecting keys, including undo and recipe replacement.
#[derive(Default)]
struct SelectionCache {
    // Internal, bounded integer keys: no adversarial string-hashing requirement.
    regions: rustc_hash::FxHashMap<Key, u32>,
    edits: Vec<Edit>,
    classified: usize,
    reused: usize,
    voxel_step: u32,
}
impl SelectionCache {
    #[cfg(test)]
    fn reconcile(&mut self, world: &World) {
        assert!(self.reconcile_cancellable(world, &|| false));
    }
    fn reconcile_cancellable(&mut self, world: &World, cancelled: &impl Fn() -> bool) -> bool {
        if self.voxel_step != world.voxel_step() {
            self.regions.clear();
            self.voxel_step = world.voxel_step();
        }
        self.classified = 0;
        self.reused = 0;
        let common = self
            .edits
            .iter()
            .zip(&world.edits)
            .take_while(|(a, b)| a == b)
            .count();
        if common != self.edits.len() || common != world.edits.len() {
            let changes: Vec<_> = self.edits[common..]
                .iter()
                .chain(&world.edits[common..])
                .copied()
                .collect();
            let mut interrupted = false;
            self.regions.retain(|key, _| {
                if interrupted {
                    return true;
                }
                for (i, edit) in changes.iter().enumerate() {
                    if i % 64 == 0 && cancelled() {
                        interrupted = true;
                        return true;
                    }
                    if key.overlaps(*edit) {
                        return false;
                    }
                }
                true
            });
            // Partial invalidation is harmless, but must retain the old edit
            // identity so a later request rechecks every remaining old entry.
            if interrupted {
                return false;
            }
            self.edits.clone_from(&world.edits);
        }
        // Bound CPU memory independently of travel distance. A cold cache
        // affects selection cost only; never occupancy or published geometry.
        if self.regions.len() >= NODE_CAPACITY * 2 {
            self.regions.retain(|key, _| key.level >= 10);
        }
        true
    }
    fn classify(&mut self, world: &World, key: Key) -> u32 {
        if let Some(&kind) = self.regions.get(&key) {
            self.reused += 1;
            return kind;
        }
        let kind = classify(world, key);
        // A single selection can retry at multiple pixel budgets. Bound the
        // cache during those retries, not only between requests.
        if self.regions.len() < NODE_CAPACITY * 2 {
            self.regions.insert(key, kind);
        }
        self.classified += 1;
        kind
    }
}

#[cfg(test)]
fn build(world: Arc<World>, view: View, max_leaves: usize, cache: &mut SelectionCache) -> Plan {
    build_cancellable(world, view, max_leaves, cache, || false).unwrap()
}
fn build_cancellable(
    world: Arc<World>,
    view: View,
    max_leaves: usize,
    cache: &mut SelectionCache,
    cancelled: impl Fn() -> bool,
) -> Option<Plan> {
    if cancelled() {
        return None;
    }
    if !cache.reconcile_cancellable(&world, &cancelled) {
        return None;
    }
    if cancelled() {
        return None;
    }
    // Retry selection at a coarser pixel budget if a pathological surface would
    // exceed physical storage. Report that budget; never silently omit leaves.
    let mut level = 22;
    if let Some((low, high)) = world.edit_index.bounds {
        let extent = low
            .into_iter()
            .chain(high)
            .map(i32::unsigned_abs)
            .max()
            .unwrap();
        while extent >= (16u32 << level) {
            level += 1;
        }
    }
    let root_low = [-(16i32 << level); 3];
    let mut pixels = 0.75;
    loop {
        let mut plan = Plan {
            nodes: Vec::new(),
            leaves: Vec::new(),
            world: world.clone(),
            view,
            pixels,
        };
        plan.nodes.push(Node {
            low: root_low,
            level,
            child: 0,
        });
        let mut pending = vec![0usize];
        while let Some(index) = pending.pop() {
            // Check before each classification: a single region is the largest
            // non-preemptible source operation, not an entire planetary plan.
            if cancelled() {
                return None;
            }
            let n = plan.nodes[index];
            let key = Key {
                low: n.low,
                level: n.level,
            };
            let kind = cache.classify(&world, key);
            if kind != 0 {
                plan.nodes[index].child = kind;
                continue;
            }
            if n.level == 0 || f64::from(1u32 << n.level) * 0.1 <= view.pixel_size(key, pixels) {
                plan.leaves.push((index, key));
                if plan.leaves.len() > max_leaves {
                    break;
                }
            } else {
                if plan.nodes.len() + 8 > NODE_CAPACITY {
                    break;
                }
                plan.nodes[index].child = plan.nodes.len() as u32;
                let half = key.side() / 2;
                for octant in 0..8 {
                    let low = std::array::from_fn(|a| n.low[a] + ((octant >> a) & 1) * half);
                    pending.push(plan.nodes.len());
                    plan.nodes.push(Node {
                        low,
                        level: n.level - 1,
                        child: 0,
                    });
                }
            }
        }
        if pending.is_empty()
            && plan.leaves.len() <= max_leaves
            && plan.nodes.len() + 8 <= NODE_CAPACITY
        {
            return Some(plan);
        }
        pixels *= 1.25;
    }
}

struct Entry {
    slot: usize,
    edits: Arc<Vec<Edit>>,
    touched: u64,
    voxel_step: u32,
}
struct Pending {
    plan: Plan,
    edits: Arc<Vec<Edit>>,
    // Ungenerated jobs have no slot. Allocate only when admitting a GPU batch.
    jobs: Vec<Job>,
    job_nodes: Vec<usize>,
    cursor: usize,
    #[cfg(feature = "regional-publication-experiment")]
    readiness: regional::Readiness,
}
impl Pending {
    fn new(plan: Plan, jobs: Vec<Job>, job_nodes: Vec<usize>, edits: Arc<Vec<Edit>>) -> Self {
        assert_eq!(jobs.len(), job_nodes.len());
        Self {
            #[cfg(feature = "regional-publication-experiment")]
            readiness: regional::Readiness::new(&plan.nodes, job_nodes.clone()),
            edits,
            plan,
            jobs,
            job_nodes,
            cursor: 0,
        }
    }
}
#[derive(Clone, Copy, Default, serde::Serialize)]
pub struct Stats {
    pub ready: bool,
    /// Published terrain exists, but the requested view or world is newer.
    pub refining: bool,
    pub nodes: usize,
    /// Leaf count of the last complete cut; a partial cut can reference both cuts.
    pub bricks: usize,
    pub pending: usize,
    pub generated: u64,
    pub reused: u64,
    /// Selection tolerance of the last complete cut, not an arrival-fidelity guarantee.
    pub pixel_budget: f64,
    pub planning: bool,
    /// Partial, revision-coherent publications; excludes complete-cut swaps.
    pub regional_publications: u64,
    /// Regions referencing a larger ancestor payload in the currently visible cut.
    pub fallback_regions: usize,
    /// Worker plans abandoned after a newer demand or shutdown.
    pub cancelled_plans: u64,
    /// Queued GPU jobs discarded before allocation/generation; not unique bricks.
    pub cancelled_jobs: u64,
    /// Current frame's render-thread demand admission time; excludes generation.
    pub update_cpu_ms: f64,
}

pub struct Residency {
    selector: selection::Worker,
    retargeting: bool,
    wanted: Option<(u64, Arc<World>, View)>,
    // Camera demand does not change the source. Reuse its edit identity so
    // ordinary travel needs neither full-cache reconciliation nor per-leaf
    // atomic retagging. Actual source changes still compare ordered edits.
    source_edits: Option<(Arc<World>, Arc<Vec<Edit>>)>,
    entries: HashMap<Key, Entry>,
    occupied: Vec<Option<Key>>,
    free: Vec<usize>,
    active_slots: HashSet<usize>,
    // Superseded versions still pinned by the last visible cut. Unlike cache
    // entries, these become free at the next complete publication.
    retired_slots: Vec<usize>,
    pending: Option<Pending>,
    active_world: Option<Arc<World>>,
    active_view: Option<View>,
    #[cfg(feature = "regional-publication-experiment")]
    complete_nodes: Vec<Node>,
    requested: bool,
    clock: u64,
    pub stats: Stats,
}
impl Residency {
    #[cfg(feature = "canonical-far-experiment")]
    pub(super) fn active_world(&self) -> Option<&Arc<World>> {
        self.active_world.as_ref()
    }
    pub fn active_voxel_step(&self) -> u32 {
        self.active_world
            .as_ref()
            .map_or(1, |world| world.voxel_step())
    }
    pub fn new(capacity: usize) -> Self {
        Self {
            selector: selection::Worker::new(capacity / 2 - 1024),
            // Whole-plan retargeting reduces arrival delay but still costs too
            // much render-thread admission work in flight. Keep it opt-in until
            // admission/publication become bounded region transactions.
            retargeting: std::env::var_os("HELIO_VOXEL_RETARGETING").is_some(),
            wanted: None,
            source_edits: None,
            entries: HashMap::default(),
            occupied: vec![None; capacity],
            free: (0..capacity).rev().collect(),
            active_slots: HashSet::default(),
            retired_slots: Vec::new(),
            pending: None,
            active_world: None,
            active_view: None,
            #[cfg(feature = "regional-publication-experiment")]
            complete_nodes: Vec::new(),
            requested: false,
            clock: 0,
            stats: Stats::default(),
        }
    }
    pub fn update(&mut self, world: &Arc<World>, params: &Params) {
        let start = std::time::Instant::now();
        self.clock += 1;
        let view = View::new(params);
        // Admit a completed snapshot before publishing the next camera demand.
        // Otherwise a camera moving every frame discards every completed plan
        // before it can generate even one brick. Source revisions cannot mix.
        let interruptible = self.retargeting && !cfg!(feature = "regional-publication-experiment");
        let mut available = if self.pending.is_none() || interruptible {
            self.selector.take()
        } else {
            None
        };
        if self.retargeting
            && available
                .as_ref()
                .is_some_and(|(_, p)| !Arc::ptr_eq(&p.world, world))
        {
            available = None;
        }
        // Bootstrap one complete cut while the camera moves; thereafter target
        // the current demand without waiting for obsolete generation to finish.
        let changed_demand = self.wanted.as_ref().is_none_or(|(_, w, v)| {
            !Arc::ptr_eq(w, world) || (self.stats.ready && v.changed(view))
        });
        if changed_demand && (self.retargeting || (self.pending.is_none() && !self.requested)) {
            // The older regional experiment can pin a union of both cuts. It
            // still finishes that bounded transaction before accepting another;
            // cancelling it requires a resident-region/coarsening contract.
            #[cfg(not(feature = "regional-publication-experiment"))]
            if self
                .pending
                .as_ref()
                .is_some_and(|p| !Arc::ptr_eq(&p.plan.world, world))
            {
                self.cancel_pending();
            }
            let serial = self.selector.submit(world.clone(), view);
            self.wanted = Some((serial, world.clone(), view));
            self.requested = true;
        }
        #[cfg(not(feature = "regional-publication-experiment"))]
        if available.is_some() && interruptible {
            self.cancel_pending();
        }
        if self.pending.is_none() {
            if let Some((serial, mut plan)) = available {
                if Some(serial) == self.wanted.as_ref().map(|(serial, _, _)| *serial) {
                    self.requested = false;
                }
                if self
                    .source_edits
                    .as_ref()
                    .is_none_or(|(w, _)| !Arc::ptr_eq(w, &plan.world))
                {
                    self.source_edits =
                        Some((plan.world.clone(), Arc::new(plan.world.edits.clone())));
                }
                let edits = self.source_edits.as_ref().unwrap().1.clone();
                let keys: HashSet<_> = plan.leaves.iter().map(|(_, k)| *k).collect();
                let mut changes = HashMap::<usize, Vec<Edit>>::default();
                let mut jobs = Vec::new();
                let mut job_nodes = Vec::new();
                for &(index, key) in &plan.leaves {
                    let valid = self.entries.get(&key).is_some_and(|e| {
                        e.voxel_step == plan.world.voxel_step()
                            && (Arc::ptr_eq(&e.edits, &edits)
                                || changes
                                    .entry(Arc::as_ptr(&e.edits) as usize)
                                    .or_insert_with(|| {
                                        let common = e
                                            .edits
                                            .iter()
                                            .zip(edits.iter())
                                            .take_while(|(a, b)| a == b)
                                            .count();
                                        e.edits[common..]
                                            .iter()
                                            .chain(edits[common..].iter())
                                            .copied()
                                            .collect()
                                    })
                                    .iter()
                                    .all(|e| !key.overlaps(*e)))
                    });
                    if valid {
                        let e = self.entries.get_mut(&key).unwrap();
                        e.touched = self.clock;
                        if !Arc::ptr_eq(&e.edits, &edits) {
                            e.edits = edits.clone();
                        }
                        self.stats.reused += 1;
                        plan.nodes[index].child = BRICK | e.slot as u32;
                    } else {
                        // Discard an invalid cached version unless it is still
                        // visible. Do not reserve replacement slots speculatively.
                        if self
                            .entries
                            .get(&key)
                            .is_some_and(|e| !self.active_slots.contains(&e.slot))
                        {
                            let old = self.entries.remove(&key).unwrap();
                            self.occupied[old.slot] = None;
                            self.free.push(old.slot);
                        }
                        jobs.push(Job {
                            low: key.low,
                            level: key.level,
                            slot: u32::MAX,
                            pad: [0, 0, plan.world.voxel_step()],
                        });
                        // Structural leaf marker only. Readiness excludes this
                        // node from publication until next_batch assigns a slot.
                        plan.nodes[index].child = BRICK;
                        job_nodes.push(index);
                    }
                }
                // Make room for only missing payloads. Reused target entries
                // and the complete visible cut remain pinned throughout this
                // transaction. Allocation itself stays bounded by next_batch.
                let needed = jobs.len().saturating_sub(self.free.len());
                if needed > 0 {
                    let mut victims: Vec<_> = self
                        .entries
                        .iter()
                        .filter(|(k, e)| !self.active_slots.contains(&e.slot) && !keys.contains(k))
                        .map(|(k, e)| (*k, e.slot, e.touched))
                        .collect();
                    victims.sort_unstable_by_key(|(_, _, t)| *t);
                    assert!(
                        victims.len() >= needed,
                        "two bounded voxel cuts fit the pool"
                    );
                    for (key, slot, _) in victims.into_iter().take(needed) {
                        self.entries.remove(&key);
                        self.occupied[slot] = None;
                        self.free.push(slot);
                    }
                }
                self.stats.pending = jobs.len();
                self.pending = Some(Pending::new(plan, jobs, job_nodes, edits));
            }
        }
        // Every published cut covers the complete world, including outside the
        // selected frustum. Keep rendering it during flight and teleports while
        // the replacement refines the arrival. Hiding it on each >128 m step
        // made sustained orbital descent display only the loading surface.
        // Picking/collision still use the authoritative CPU world; `ready`
        // means a complete visual cut exists, not that refinement has caught up.
        let changed = self
            .active_world
            .as_ref()
            .is_none_or(|w| !Arc::ptr_eq(w, world))
            || self.active_view.is_none_or(|v| v.changed(view));
        self.stats.refining = changed;
        self.stats.planning = self.requested;
        self.stats.cancelled_plans = self.selector.cancelled();
        self.stats.update_cpu_ms = start.elapsed().as_secs_f64() * 1000.0;
    }
    #[cfg(not(feature = "regional-publication-experiment"))]
    fn cancel_pending(&mut self) {
        let Some(pending) = self.pending.take() else {
            return;
        };
        self.stats.cancelled_jobs += (pending.jobs.len() - pending.cursor) as u64;
        // The unfinished suffix owns no slots. Generated entries survive as
        // reusable cache data; visible versions remain pinned independently.
        self.stats.pending = 0;
    }
    pub fn next_batch(&mut self) -> Option<(Vec<Job>, Vec<u32>, Arc<World>)> {
        let p = self.pending.as_mut()?;
        // A sampled distant brick evaluates 729 cells, versus 32768 for an
        // exact brick. Charging both one slot left most of the generation
        // budget unused in flight. Retain the previous worst-case sample
        // budget, but fill the dispatch when its bricks are cheaper.
        let sample_budget = if self.stats.ready {
            64
        } else {
            GENERATION_BATCH
        } * 32
            * 32
            * 32;
        let mut samples = 0;
        let mut batch = Vec::new();
        let mut references = Vec::new();
        while p.cursor < p.jobs.len() && batch.len() < GENERATION_BATCH {
            let mut job = p.jobs[p.cursor];
            let cost = if job.level == 0 {
                32 * 32 * 32
            } else {
                9 * 9 * 9
            };
            if samples + cost > sample_budget {
                break;
            }
            let side = 32i32 << job.level;
            let edits = p.plan.world.region_edits(
                job.low,
                job.low.map(|v| v + side - i32::from(job.level == 0)),
            );
            if references.len() + edits.len() > 262_144 {
                break;
            }
            job.pad[0] = references.len() as u32;
            job.pad[1] = edits.len() as u32;
            references.extend(edits.into_iter().map(|i| i as u32));
            let key = Key {
                low: job.low,
                level: job.level,
            };
            let slot = self.free.pop().expect("admitted voxel cut fits the pool");
            if let Some(old) = self.entries.insert(
                key,
                Entry {
                    slot,
                    edits: p.edits.clone(),
                    touched: self.clock,
                    voxel_step: p.plan.world.voxel_step(),
                },
            ) {
                if self.active_slots.contains(&old.slot) {
                    self.retired_slots.push(old.slot);
                } else {
                    self.occupied[old.slot] = None;
                    self.free.push(old.slot);
                }
            }
            self.occupied[slot] = Some(key);
            job.slot = slot as u32;
            p.jobs[p.cursor].slot = job.slot;
            p.plan.nodes[p.job_nodes[p.cursor]].child = BRICK | job.slot;
            batch.push(job);
            samples += cost;
            #[cfg(feature = "regional-publication-experiment")]
            p.readiness.generated(p.cursor);
            p.cursor += 1;
        }
        self.stats.generated += batch.len() as u64;
        self.stats.pending = p.jobs.len() - p.cursor;
        Some((batch, references, p.plan.world.clone()))
    }
    pub fn publish(&mut self) -> Option<Vec<Node>> {
        if !self
            .pending
            .as_ref()
            .is_some_and(|p| p.cursor == p.jobs.len())
        {
            #[cfg(feature = "regional-publication-experiment")]
            if let Some(pending) = self.pending.as_mut() {
                // Mixing source revisions or authored grids would expose stale
                // edits. This first candidate publishes regions only within one
                // immutable world; source changes retain atomic whole-cut swaps.
                let compatible = self
                    .active_world
                    .as_ref()
                    .is_some_and(|world| Arc::ptr_eq(world, &pending.plan.world))
                    && self
                        .complete_nodes
                        .first()
                        .zip(pending.plan.nodes.first())
                        .is_some_and(|(a, b)| a.low == b.low && a.level == b.level);
                if compatible && pending.readiness.changed {
                    pending.readiness.changed = false;
                    let nodes = regional::compose(
                        &self.complete_nodes,
                        &pending.plan.nodes,
                        &pending.readiness,
                    );
                    assert!(nodes.len() <= PUBLICATION_NODE_CAPACITY);
                    self.stats.regional_publications += 1;
                    self.stats.nodes = nodes.len();
                    self.stats.fallback_regions = nodes
                        .iter()
                        .filter(|n| {
                            n.child & BRICK != 0 && n.child < SOLID && n.child & 0x001f_0000 != 0
                        })
                        .count();
                    return Some(nodes);
                }
            }
            return None;
        }
        let pending = self.pending.take().unwrap();
        self.active_slots = pending
            .plan
            .nodes
            .iter()
            .filter(|n| n.child & BRICK != 0 && n.child < SOLID)
            .map(|n| (n.child & 0xffff) as usize)
            .collect();
        // Release superseded versions of a key now that no active node owns them.
        for slot in self.retired_slots.drain(..) {
            assert!(
                !self.active_slots.contains(&slot),
                "superseded version reused"
            );
            let key = self.occupied[slot]
                .take()
                .expect("retired slot freed twice");
            assert!(self.entries.get(&key).is_none_or(|e| e.slot != slot));
            self.free.push(slot);
        }
        self.stats.ready = true;
        self.stats.fallback_regions = 0;
        self.stats.nodes = pending.plan.nodes.len();
        self.stats.bricks = pending.plan.leaves.len();
        self.stats.pixel_budget = pending.plan.pixels;
        self.active_world = Some(pending.plan.world);
        self.active_view = Some(pending.plan.view);
        #[cfg(feature = "regional-publication-experiment")]
        self.complete_nodes.clone_from(&pending.plan.nodes);
        Some(pending.plan.nodes)
    }
}
