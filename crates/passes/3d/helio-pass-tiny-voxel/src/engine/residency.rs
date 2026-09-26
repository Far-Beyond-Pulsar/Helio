//! A view-selected tree of stored voxel bricks. Selection runs off the render
//! thread; a complete new tree is published only after its GPU bricks are ready.
use crate::{
    landforms::RegionClass,
    world::{Edit, World},
    Params,
};
use glam::DVec3;
use std::{
    collections::{HashMap, HashSet},
    sync::{mpsc, Arc, Mutex},
};
#[cfg(test)]
mod tests;

pub const BRICK_WORDS: usize = 2048; // 32^3 exact materials, or 9^3 densities + 8^3 pairs of bounds.
pub const BRICK_CAPACITY: usize = 65_536;
pub const NODE_CAPACITY: usize = 262_144;
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
#[derive(Clone, Copy)]
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
    fn reconcile(&mut self, world: &World) {
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
            self.regions
                .retain(|key, _| !changes.iter().any(|edit| key.overlaps(*edit)));
            self.edits.clone_from(&world.edits);
        }
        // Bound CPU memory independently of travel distance. A cold cache
        // affects selection cost only; never occupancy or published geometry.
        if self.regions.len() > NODE_CAPACITY * 2 {
            self.regions.retain(|key, _| key.level >= 10);
        }
    }
    fn classify(&mut self, world: &World, key: Key) -> u32 {
        if let Some(&kind) = self.regions.get(&key) {
            self.reused += 1;
            return kind;
        }
        let kind = classify(world, key);
        self.regions.insert(key, kind);
        self.classified += 1;
        kind
    }
}

fn build(world: Arc<World>, view: View, max_leaves: usize, cache: &mut SelectionCache) -> Plan {
    cache.reconcile(&world);
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
            return plan;
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
    jobs: Vec<Job>,
    cursor: usize,
}
#[derive(Clone, Copy, Default, serde::Serialize)]
pub struct Stats {
    pub ready: bool,
    /// Published terrain exists, but the requested view or world is newer.
    pub refining: bool,
    pub nodes: usize,
    pub bricks: usize,
    pub pending: usize,
    pub generated: u64,
    pub reused: u64,
    pub pixel_budget: f64,
    pub planning: bool,
}

pub struct Residency {
    requests: mpsc::SyncSender<(Arc<World>, View)>,
    results: Mutex<mpsc::Receiver<Plan>>,
    entries: HashMap<Key, Entry>,
    occupied: Vec<Option<Key>>,
    free: Vec<usize>,
    active: HashSet<Key>,
    pending: Option<Pending>,
    active_world: Option<Arc<World>>,
    active_view: Option<View>,
    requested: bool,
    clock: u64,
    pub stats: Stats,
}
impl Residency {
    pub fn active_voxel_step(&self) -> u32 {
        self.active_world
            .as_ref()
            .map_or(1, |world| world.voxel_step())
    }
    pub fn new(capacity: usize) -> Self {
        let (tx, rx) = mpsc::sync_channel::<(Arc<World>, View)>(1);
        let (done, result) = mpsc::sync_channel(1);
        std::thread::Builder::new()
            .name("voxel-selection".into())
            .spawn(move || {
                let mut cache = SelectionCache::default();
                while let Ok((world, view)) = rx.recv() {
                    let start = std::time::Instant::now();
                    let plan = build(world, view, capacity / 2 - 1024, &mut cache);
                    eprintln!(
                        "VOXEL_PLAN nodes={} bricks={} pixel_budget={:.3} selection_ms={:.2} classified={} cached={}",
                        plan.nodes.len(),
                        plan.leaves.len(),
                        plan.pixels,
                        start.elapsed().as_secs_f64() * 1000.0,
                        cache.classified,
                        cache.reused,
                    );
                    if done.send(plan).is_err() {
                        break;
                    }
                }
            })
            .expect("voxel selection worker");
        Self {
            requests: tx,
            results: Mutex::new(result),
            entries: HashMap::new(),
            occupied: vec![None; capacity],
            free: (0..capacity).rev().collect(),
            active: HashSet::new(),
            pending: None,
            active_world: None,
            active_view: None,
            requested: false,
            clock: 0,
            stats: Stats::default(),
        }
    }
    pub fn update(&mut self, world: &Arc<World>, params: &Params) {
        self.clock += 1;
        if self.pending.is_none() {
            if let Ok(mut plan) = self.results.get_mut().unwrap().try_recv() {
                self.requested = false;
                let edits = Arc::new(plan.world.edits.clone());
                let keys: HashSet<_> = plan.leaves.iter().map(|(_, k)| *k).collect();
                let mut changes = HashMap::<usize, Vec<Edit>>::new();
                for entry in self.entries.values() {
                    changes
                        .entry(Arc::as_ptr(&entry.edits) as usize)
                        .or_insert_with(|| {
                            let common = entry
                                .edits
                                .iter()
                                .zip(edits.iter())
                                .take_while(|(a, b)| a == b)
                                .count();
                            entry.edits[common..]
                                .iter()
                                .chain(edits[common..].iter())
                                .copied()
                                .collect()
                        });
                }
                // Reserve enough free slots once, before installing the cut.
                // Repeatedly searching a 65k-slot pool per brick is quadratic.
                let needed = keys.len().saturating_sub(self.free.len());
                if needed > 0 {
                    let mut victims: Vec<_> = self
                        .entries
                        .iter()
                        .filter(|(k, _)| !self.active.contains(k) && !keys.contains(k))
                        .map(|(k, e)| (*k, e.slot, e.touched))
                        .collect();
                    victims.sort_unstable_by_key(|(_, _, t)| *t);
                    for (key, slot, _) in victims.into_iter().take(needed) {
                        self.entries.remove(&key);
                        self.occupied[slot] = None;
                        self.free.push(slot);
                    }
                }
                let mut jobs = Vec::new();
                for &(index, key) in &plan.leaves {
                    let valid = self.entries.get(&key).is_some_and(|e| {
                        e.voxel_step == plan.world.voxel_step()
                            && changes[&(Arc::as_ptr(&e.edits) as usize)]
                                .iter()
                                .all(|e| !key.overlaps(*e))
                    });
                    let slot = if valid {
                        let e = self.entries.get_mut(&key).unwrap();
                        e.touched = self.clock;
                        e.edits = edits.clone();
                        self.stats.reused += 1;
                        e.slot
                    } else {
                        // A replacement never overwrites payloads still used by
                        // the active tree. Publication switches the complete cut.
                        let slot = self
                            .free
                            .pop()
                            .expect("two bounded voxel cuts fit the pool");
                        if let Some(old) = self.entries.insert(
                            key,
                            Entry {
                                slot,
                                edits: edits.clone(),
                                touched: self.clock,
                                voxel_step: plan.world.voxel_step(),
                            },
                        ) {
                            // The old slot stays pinned until tree publication.
                            if !self.active.contains(&key) {
                                self.occupied[old.slot] = None;
                                self.free.push(old.slot);
                            }
                        }
                        self.occupied[slot] = Some(key);
                        jobs.push(Job {
                            low: key.low,
                            level: key.level,
                            slot: slot as u32,
                            pad: [0, 0, plan.world.voxel_step()],
                        });
                        slot
                    };
                    plan.nodes[index].child = BRICK | slot as u32;
                }
                self.stats.pending = jobs.len();
                self.pending = Some(Pending {
                    plan,
                    jobs,
                    cursor: 0,
                });
            }
        }
        let view = View::new(params);
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
        if self.pending.is_none() && !self.requested && changed {
            self.requested = self.requests.try_send((world.clone(), view)).is_ok();
        }
        self.stats.planning = self.requested;
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
            batch.push(job);
            samples += cost;
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
            return None;
        }
        let pending = self.pending.take().unwrap();
        self.active = pending.plan.leaves.iter().map(|(_, k)| *k).collect();
        // Release superseded versions of a key now that no active node owns them.
        for (slot, key) in self.occupied.iter_mut().enumerate() {
            if key.is_some_and(|k| self.entries.get(&k).is_none_or(|e| e.slot != slot)) {
                *key = None;
                self.free.push(slot);
            }
        }
        self.stats.ready = true;
        self.stats.nodes = pending.plan.nodes.len();
        self.stats.bricks = pending.plan.leaves.len();
        self.stats.pixel_budget = pending.plan.pixels;
        self.active_world = Some(pending.plan.world);
        self.active_view = Some(pending.plan.view);
        Some(pending.plan.nodes)
    }
}
