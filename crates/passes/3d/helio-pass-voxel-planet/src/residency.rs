//! CPU side of the GPU clipmap: level windows, the column hash table mirror,
//! record and edit-list allocation, and prioritized generation jobs.
//!
//! The CPU decides *which* columns are resident; the GPU generates their
//! contents, allocates brick runs and publishes records in the same frame.
use crate::edits::FaceBrush;
use crate::grid::{Grid, BRICK};
use crate::windows::{WindowPlanner, WindowRequest, WindowUpdate, WindowWorker};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use crate::planet::Planet;
use bytemuck::{Pod, Zeroable};
use glam::DVec3;
use rustc_hash::FxHashMap;

pub const NONE: u32 = u32::MAX;
pub const TOMBSTONE: u32 = u32::MAX - 1;

#[derive(Clone, Copy, Debug)]
pub struct Capacity {
    pub table_bits: u32,
    pub records: u32,
    pub pool_units: u32,
    pub scratch_units: u32,
    pub edit_words: u32,
    pub max_jobs: u32,
    pub max_evictions: u32,
}

impl Default for Capacity {
    fn default() -> Self {
        Self {
            table_bits: 22,
            records: 3_000_000,
            pool_units: 4 << 20,
            scratch_units: 1 << 18,
            edit_words: 4 << 20,
            max_jobs: 16_384,
            max_evictions: 262_144,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct Job {
    pub key0: u32,
    pub key1: u32,
    pub record: u32,
    pub edits: u32,
    pub flags: u32,
    pub pad: [u32; 3],
}

pub fn key0(face: u8, level: u32, ci: i32) -> u32 {
    (ci as u32 & 0xff_ffff) | (u32::from(face) << 24) | (level << 27)
}

pub fn slot_hash(key0: u32, key1: u32) -> u32 {
    crate::field::hash3(key0 as i32, key1 as i32, 0x2f6b_1d3a, 0x9e37_79b9)
}

fn pack(k0: u32, k1: u32) -> u64 {
    u64::from(k0) | (u64::from(k1) << 32)
}

fn unpack(key: u64) -> (u8, u32, i32, i32) {
    let k0 = key as u32;
    let k1 = (key >> 32) as u32;
    (((k0 >> 24) & 7) as u8, k0 >> 27, ((k0 << 8) as i32) >> 8, k1 as i32)
}

/// Direct-mapped summary blocks: per (level, face) a toroidal table for each
/// tier of 4^tier x 4^tier columns. Entries are `[bi, bj, max top, count]`.
pub const BLOCK_TIERS: u32 = 3;
pub const BLOCK_LOG2: [u32; 3] = [7, 5, 3];

pub fn block_region() -> u32 {
    BLOCK_LOG2.iter().map(|l| 1u32 << (2 * l)).sum()
}

pub fn block_slot(level: u32, face: u8, tier: u32, bi: i32, bj: i32) -> u32 {
    let mut offset = (level * 6 + u32::from(face)) * block_region();
    for t in 1..tier {
        offset += 1 << (2 * BLOCK_LOG2[t as usize - 1]);
    }
    let l = BLOCK_LOG2[tier as usize - 1];
    let mask = (1i32 << l) - 1;
    offset + (((bj & mask) as u32) << l) + (bi & mask) as u32
}

/// Largest column window diameter the block tables support.
pub fn max_window_columns() -> i32 {
    (1 << BLOCK_LOG2[0]) * 4 - 8
}

#[derive(Clone, Copy, Debug)]
struct Block {
    slot: u32,
    refs: u32,
}

#[derive(Clone, Copy, Debug)]
struct Resident {
    record: u32,
    slot: u32,
    edit_block: Option<(u32, u32)>,
}

/// Priority-ordered pending key (lower priority value is issued first).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Pending(f32, u64);
impl Eq for Pending {}
impl PartialOrd for Pending {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Pending {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0).then(self.1.cmp(&other.1))
    }
}

#[derive(Default)]
struct Level {
    active: bool,
    /// Window centre and angular radius of the last applied diff.
    center: DVec3,
    radius_angle: f64,
    /// Wanted but not yet issued columns, and their priority heap (lazy).
    pending: rustc_hash::FxHashSet<u64>,
    heap: BinaryHeap<Reverse<Pending>>,
    keys: rustc_hash::FxHashSet<u64>,
}

enum Planner {
    Inline(WindowPlanner),
    Worker(WindowWorker),
}

/// Power-of-two block allocator for edit-reference lists.
#[derive(Default)]
struct EditHeap {
    top: u32,
    free: Vec<Vec<u32>>,
}

impl EditHeap {
    fn alloc(&mut self, words: u32, capacity: u32) -> Option<(u32, u32)> {
        let class = words.max(1).next_power_of_two().trailing_zeros();
        if self.free.len() <= class as usize {
            self.free.resize(class as usize + 1, Vec::new());
        }
        if let Some(base) = self.free[class as usize].pop() {
            return Some((base, class));
        }
        let size = 1u32 << class;
        (self.top + size <= capacity).then(|| {
            let base = self.top;
            self.top += size;
            (base, class)
        })
    }
    fn release(&mut self, block: (u32, u32)) {
        self.free[block.1 as usize].push(block.0);
    }
}

/// Work produced for one frame.
#[derive(Default)]
pub struct FrameWork {
    pub jobs: Vec<Job>,
    pub job_keys: Vec<u64>,
    pub evictions: Vec<u32>,
    pub table_writes: Vec<(u32, u32)>,
    pub edit_writes: Vec<(u32, Vec<u32>)>,
    pub brush_writes: Vec<(u32, FaceBrush)>,
    /// Summary block table writes `(slot, bi, bj)` in order; `bi = -1`
    /// releases a slot.
    pub block_inits: Vec<(u32, i32, i32)>,
    pub full_table: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub resident_columns: usize,
    pub pending_columns: usize,
    pub active_levels: u32,
    pub finest_level: u32,
    pub jobs: usize,
    pub evictions: usize,
    pub requeued: usize,
    pub window_rebuild_ms: f64,
    pub edit_words: u32,
    pub table_load: f32,
}

pub struct Residency {
    pub capacity: Capacity,
    grid: Grid,
    residents: FxHashMap<u64, Resident>,
    blocks: FxHashMap<(u32, u8, u32, i32, i32), Block>,
    block_owner: FxHashMap<u32, (u32, u8, u32, i32, i32)>,
    /// Dense list of live tier-1 block slots (GPU horizon build input) and
    /// each slot's position in it; `live_dirty` marks a pending upload.
    live_tier1: Vec<u32>,
    live_index: FxHashMap<u32, usize>,
    live_dirty: bool,
    table: Vec<u32>,
    used_slots: u32,
    tombstones: u32,
    free_records: Vec<u32>,
    next_record: u32,
    delayed_records: Vec<u32>,
    levels: Vec<Level>,
    edits: EditHeap,
    /// GPU face-brush index for each (brush id, face entry).
    brush_gpu: Vec<Vec<u32>>,
    synced: Vec<crate::edits::Brush>,
    next_brush: u32,
    urgent: Vec<u64>,
    pub stats: Stats,
    frame: u32,
    planner: Planner,
    last_request: Option<WindowRequest>,
    requested: u64,
    applied: u64,
    /// Columns the current windows no longer want.
    unwanted: rustc_hash::FxHashSet<u64>,
}

impl Residency {
    /// Residency whose windows are planned inline (deterministic; tests).
    pub fn new(grid: Grid, capacity: Capacity) -> Self {
        Self::with_planner(grid, capacity, Planner::Inline(WindowPlanner::new(grid)))
    }

    /// Residency whose windows are planned on a background thread.
    pub fn with_worker(grid: Grid, capacity: Capacity) -> Self {
        Self::with_planner(grid, capacity, Planner::Worker(WindowWorker::start(grid)))
    }

    fn with_planner(grid: Grid, capacity: Capacity, planner: Planner) -> Self {
        let levels = (0..grid.levels()).map(|_| Level::default()).collect();
        Self {
            capacity,
            grid,
            residents: FxHashMap::default(),
            blocks: FxHashMap::default(),
            block_owner: FxHashMap::default(),
            live_tier1: Vec::new(),
            live_index: FxHashMap::default(),
            live_dirty: false,
            table: vec![NONE; 1 << capacity.table_bits],
            used_slots: 0,
            tombstones: 0,
            free_records: Vec::new(),
            next_record: 0,
            delayed_records: Vec::new(),
            levels,
            edits: EditHeap::default(),
            brush_gpu: Vec::new(),
            synced: Vec::new(),
            next_brush: 0,
            urgent: Vec::new(),
            stats: Stats::default(),
            frame: 0,
            planner,
            last_request: None,
            requested: 0,
            applied: 0,
            unwanted: Default::default(),
        }
    }

    pub fn grid(&self) -> &Grid {
        &self.grid
    }

    pub fn table(&self) -> &[u32] {
        &self.table
    }

    fn mask(&self) -> u32 {
        (1u32 << self.capacity.table_bits) - 1
    }

    fn insert_slot(&mut self, k0: u32, k1: u32, record: u32) -> u32 {
        let mask = self.mask();
        let mut slot = slot_hash(k0, k1) & mask;
        loop {
            let v = self.table[slot as usize];
            if v == NONE || v == TOMBSTONE {
                if v == TOMBSTONE {
                    self.tombstones -= 1;
                }
                self.table[slot as usize] = record;
                self.used_slots += 1;
                return slot;
            }
            slot = (slot + 1) & mask;
        }
    }

    fn alloc_record(&mut self) -> Option<u32> {
        if let Some(r) = self.free_records.pop() {
            return Some(r);
        }
        (self.next_record < self.capacity.records).then(|| {
            self.next_record += 1;
            self.next_record - 1
        })
    }

    /// Mean level-0 visibility distance from the view: level cells project to
    /// `pixel` pixels at the start of their range.
    pub fn lod_distance(grid: &Grid, tan_half_fov: f64, height: u32, pixel: f64) -> f64 {
        let angle = 2.0 * tan_half_fov / f64::from(height.max(1));
        grid.voxel_size() * pixel / angle
    }

    /// Sync the edit log: upload new face brushes and schedule regeneration
    /// of resident columns touched by new or undone brushes.
    fn sync_edits(&mut self, planet: &Planet, work: &mut FrameWork) {
        let log = planet.edits();
        let current: Vec<_> = log.brushes().copied().collect();
        let common = self
            .synced
            .iter()
            .zip(&current)
            .take_while(|(a, b)| a == b)
            .count();
        let mut touched = Vec::new();
        for id in common..self.synced.len() {
            // Undone brushes: their old footprint must be regenerated.
            if let Ok(faces) = self.synced[id].resolve(&self.grid) {
                touched.extend(faces);
            }
        }
        self.brush_gpu.truncate(common);
        for id in common..current.len() {
            let resolved = log.resolved(id as u32);
            let mut indices = Vec::new();
            for fb in &resolved.faces {
                let index = self.next_brush;
                self.next_brush += 1;
                work.brush_writes.push((index, *fb));
                indices.push(index);
                touched.push(*fb);
            }
            self.brush_gpu.push(indices);
        }
        self.synced = current;
        for fb in touched {
            let r_cells = i64::from(fb.radius_half) / 2 + 1;
            let ci = i64::from(fb.center[0]) / 2;
            let cj = i64::from(fb.center[1]) / 2;
            for level in 0..self.grid.levels() {
                if !fb.active(level) {
                    continue;
                }
                let col = i64::from(BRICK) << level;
                let (i0, i1) = ((ci - r_cells).div_euclid(col), (ci + r_cells).div_euclid(col));
                let (j0, j1) = ((cj - r_cells).div_euclid(col), (cj + r_cells).div_euclid(col));
                if (i1 - i0 + 1) * (j1 - j0 + 1) > 1 << 16 {
                    continue;
                }
                for a in i0..=i1 {
                    for b in j0..=j1 {
                        let key = pack(key0(fb.face(), level, a as i32), b as i32 as u32);
                        if self.residents.contains_key(&key) {
                            self.urgent.push(key);
                        }
                    }
                }
            }
        }
    }

    fn edit_list(&mut self, planet: &Planet, key: u64, work: &mut FrameWork) -> Result<Option<(u32, u32)>, ()> {
        let (face, level, ci, cj) = unpack(key);
        let span = i64::from(BRICK) << level;
        let i0 = i64::from(ci) * span;
        let j0 = i64::from(cj) * span;
        let refs = planet.edits().query(face, i0, i0 + span - 1, j0, j0 + span - 1, level);
        if refs.is_empty() {
            return Ok(None);
        }
        let mut words = Vec::with_capacity(refs.len() + 1);
        words.push(refs.len() as u32);
        for (id, index) in refs {
            words.push(self.brush_gpu[id as usize][index as usize]);
        }
        let block = self
            .edits
            .alloc(words.len() as u32, self.capacity.edit_words)
            .ok_or(())?;
        work.edit_writes.push((block.0, words));
        Ok(Some(block))
    }

    fn acquire_blocks(&mut self, key: u64, work: &mut FrameWork) -> bool {
        let (face, level, ci, cj) = unpack(key);
        for tier in 1..=BLOCK_TIERS {
            let (bi, bj) = (ci >> (2 * tier), cj >> (2 * tier));
            let bkey = (level, face, tier, bi, bj);
            if let Some(b) = self.blocks.get_mut(&bkey) {
                b.refs += 1;
                continue;
            }
            let slot = block_slot(level, face, tier, bi, bj);
            if self.block_owner.get(&slot).is_some_and(|owner| *owner != bkey) {
                // Window larger than the table: this block never becomes complete.
                continue;
            }
            self.block_owner.insert(slot, bkey);
            work.block_inits.push((slot, bi, bj));
            self.blocks.insert(bkey, Block { slot, refs: 1 });
            if tier == 1 {
                self.live_index.insert(slot, self.live_tier1.len());
                self.live_tier1.push(slot);
                self.live_dirty = true;
            }
        }
        true
    }

    fn release_blocks(&mut self, key: u64, work: &mut FrameWork) {
        let (face, level, ci, cj) = unpack(key);
        for tier in 1..=BLOCK_TIERS {
            let bkey = (level, face, tier, ci >> (2 * tier), cj >> (2 * tier));
            let Some(b) = self.blocks.get_mut(&bkey) else { continue };
            b.refs -= 1;
            if b.refs == 0 {
                let b = self.blocks.remove(&bkey).unwrap();
                self.block_owner.remove(&b.slot);
                work.block_inits.push((b.slot, -1, -1));
                if tier == 1 {
                    if let Some(at) = self.live_index.remove(&b.slot) {
                        self.live_tier1.swap_remove(at);
                        if let Some(&moved) = self.live_tier1.get(at) {
                            self.live_index.insert(moved, at);
                        }
                        self.live_dirty = true;
                    }
                }
            }
        }
    }

    fn evict(&mut self, key: u64, work: &mut FrameWork) {
        if self.residents.contains_key(&key) {
            self.release_blocks(key, work);
        }
        if let Some(res) = self.residents.remove(&key) {
            self.table[res.slot as usize] = TOMBSTONE;
            self.tombstones += 1;
            self.used_slots -= 1;
            work.table_writes.push((res.slot, TOMBSTONE));
            work.evictions.push(res.record);
            self.delayed_records.push(res.record);
            if let Some(block) = res.edit_block {
                self.edits.release(block);
            }
            let (_, level, _, _) = unpack(key);
            self.levels[level as usize].keys.remove(&key);
        }
    }

    /// Apply a window diff: evict unwanted residents, queue new columns.
    fn apply(&mut self, update: WindowUpdate, work: &mut FrameWork) {
        for diff in update.levels {
            let level = diff.level as usize;
            self.levels[level].active = diff.active;
            self.levels[level].center = diff.center;
            self.levels[level].radius_angle = diff.radius_angle;
            for key in diff.removes {
                self.levels[level].pending.remove(&key);
                if self.residents.contains_key(&key) {
                    self.evict(key, work);
                }
            }
            if !diff.active {
                self.levels[level].heap.clear();
                self.levels[level].pending.clear();
            }
            for (priority, key) in diff.adds {
                if !self.residents.contains_key(&key) && self.levels[level].pending.insert(key) {
                    self.levels[level].heap.push(Reverse(Pending(priority, key)));
                }
            }
        }
        self.stats.window_rebuild_ms = update.planning_ms;
        self.applied = update.serial;
    }

    /// Plan one frame. `lod0` is the level-0 distance, `budget` the maximum
    /// number of column jobs.
    pub fn plan(&mut self, planet: &Planet, eye: DVec3, lod0: f64, budget: usize) -> FrameWork {
        self.frame = self.frame.wrapping_add(1);
        let mut work = FrameWork::default();
        // Records evicted last frame are safe to reuse now.
        let delayed = std::mem::take(&mut self.delayed_records);
        self.free_records.extend(delayed);
        self.sync_edits(planet, &mut work);
        // Ask the planner for new windows when the view changed, then apply
        // every diff that is ready (the worker always plans the latest view).
        let request = WindowRequest {
            eye,
            lod0,
            outer_radius: planet.outer_radius(),
            serial: self.requested + 1,
        };
        let changed = self.last_request.is_none_or(|last| {
            last.eye.distance(eye) > self.grid.voxel_size() * 2.0
                || (last.lod0 - lod0).abs() > lod0 * 0.01
                || last.outer_radius != request.outer_radius
        });
        if changed {
            self.requested = request.serial;
            self.last_request = Some(request);
            match &mut self.planner {
                Planner::Inline(planner) => {
                    let update = planner.update(&request);
                    self.apply(update, &mut work);
                }
                Planner::Worker(worker) => worker.request(request),
            }
        }
        if let Planner::Worker(worker) = &self.planner {
            let mut updates = Vec::new();
            while let Some(update) = worker.try_update() {
                updates.push(update);
            }
            for update in updates {
                self.apply(update, &mut work);
            }
        }
        // Rehash when tombstones dominate.
        let capacity = 1u32 << self.capacity.table_bits;
        if self.tombstones + self.used_slots > capacity / 2 {
            self.table.fill(NONE);
            self.used_slots = 0;
            self.tombstones = 0;
            let entries: Vec<(u64, u32)> = self.residents.iter().map(|(k, r)| (*k, r.record)).collect();
            for (key, record) in entries {
                let slot = self.insert_slot(key as u32, (key >> 32) as u32, record);
                self.residents.get_mut(&key).unwrap().slot = slot;
            }

            work.full_table = true;
        }
        // Urgent edit regenerations first.
        let mut urgent = std::mem::take(&mut self.urgent);
        urgent.sort_unstable();
        urgent.dedup();
        let mut deferred_urgent = Vec::new();
        for key in urgent {
            if work.jobs.len() >= budget {
                deferred_urgent.push(key);
                continue;
            }
            let Some(res) = self.residents.get(&key).copied() else { continue };
            let Ok(block) = self.edit_list(planet, key, &mut work) else {
                deferred_urgent.push(key);
                continue;
            };
            if let Some(old) = res.edit_block {
                self.edits.release(old);
            }
            self.residents.get_mut(&key).unwrap().edit_block = block;
            work.jobs.push(Job {
                key0: key as u32,
                key1: (key >> 32) as u32,
                record: res.record,
                edits: block.map_or(0, |b| b.0 + 1),
                flags: 1,
                pad: [0; 3],
            });
            work.job_keys.push(key);
        }
        self.urgent = deferred_urgent;
        // Merge pending windows by normalized distance; the coarsest level
        // (global coverage) always goes first.
        let top_level = self.grid.levels() - 1;
        while work.jobs.len() < budget {
            let mut best: Option<(f32, usize)> = None;
            for index in 0..self.levels.len() {
                let l = &mut self.levels[index];
                while let Some(Reverse(Pending(_, key))) = l.heap.peek() {
                    if l.pending.contains(key) {
                        break;
                    }
                    l.heap.pop();
                }
                if let Some(Reverse(Pending(priority, _))) = l.heap.peek() {
                    let p = if index as u32 == top_level { priority - 100.0 } else { *priority };
                    if best.is_none_or(|b| p < b.0) {
                        best = Some((p, index));
                    }
                }
            }
            let Some((_, index)) = best else { break };
            let Reverse(Pending(priority, key)) = self.levels[index].heap.pop().unwrap();
            self.levels[index].pending.remove(&key);
            if self.residents.contains_key(&key) {
                continue;
            }
            let requeue = |this: &mut Self| {
                this.levels[index].pending.insert(key);
                this.levels[index].heap.push(Reverse(Pending(priority, key)));
            };
            let Some(record) = self.alloc_record() else {
                requeue(self);
                break;
            };
            let Ok(block) = self.edit_list(planet, key, &mut work) else {
                self.free_records.push(record);
                requeue(self);
                break;
            };
            if !self.acquire_blocks(key, &mut work) {
                self.free_records.push(record);
                if let Some(b) = block {
                    self.edits.release(b);
                }
                requeue(self);
                break;
            }
            let slot = self.insert_slot(key as u32, (key >> 32) as u32, record);
            work.table_writes.push((slot, record));
            self.residents.insert(
                key,
                Resident {
                    record,
                    slot,
                    edit_block: block,
                },
            );
            self.levels[index].keys.insert(key);
            work.jobs.push(Job {
                key0: key as u32,
                key1: (key >> 32) as u32,
                record,
                edits: block.map_or(0, |b| b.0 + 1),
                flags: 0,
                pad: [0; 3],
            });
            work.job_keys.push(key);
        }
        let mut stats = self.stats;
        stats.resident_columns = self.residents.len();
        stats.pending_columns = self.levels.iter().map(|l| l.pending.len()).sum::<usize>() + self.urgent.len();
        let _ = &self.unwanted;
        stats.active_levels = self.levels.iter().filter(|l| l.active).count() as u32;
        stats.finest_level = self.levels.iter().position(|l| l.active).unwrap_or(0) as u32;
        stats.jobs = work.jobs.len();
        stats.evictions = work.evictions.len();
        stats.edit_words = self.edits.top;
        stats.table_load = (self.used_slots + self.tombstones) as f32 / capacity as f32;
        self.stats = stats;
        work
    }

    /// Re-queue columns whose jobs could not complete (scratch/pool pressure).
    pub fn requeue(&mut self, keys: impl IntoIterator<Item = (u64, u32)>) {
        let mut count = 0;
        for (key, status) in keys {
            if status == 1 {
                // Band overflow: stays unpublished; coarser levels cover it.
                continue;
            }
            if !self.residents.contains_key(&key) {
                continue;
            }
            self.urgent.push(key);
            count += 1;
        }
        self.stats.requeued += count;
    }

    /// Every pending window column has been issued.
    /// Live tier-1 summary block slots, when they changed since the last call.
    pub fn take_live_blocks(&mut self) -> Option<&[u32]> {
        std::mem::take(&mut self.live_dirty).then_some(self.live_tier1.as_slice())
    }
    pub fn live_block_count(&self) -> usize {
        self.live_tier1.len()
    }

    /// Per level, the angular distance from `eye_dir` within which every
    /// column the traversal can want is resident, so rays nearer than that
    /// never fall back to a coarser level: bounded by the (possibly lagging)
    /// window and by the nearest pending column. Inactive levels give 0.
    pub fn fallback_angles(&self, eye_dir: DVec3) -> Vec<f64> {
        let grid = self.grid;
        let urgent: rustc_hash::FxHashSet<u32> = self.urgent.iter().map(|k| unpack(*k).1).collect();
        self.levels
            .iter()
            .enumerate()
            .map(|(level, l)| {
                if !l.active || urgent.contains(&(level as u32)) {
                    return 0.0;
                }
                // Index-angle span of a column bounds its true angular size.
                let col = grid.delta() * f64::from(BRICK << level);
                let mut angle = l.radius_angle - col * 1.5 - l.center.angle_between(eye_dir);
                if l.pending.len() > 4096 {
                    return 0.0;
                }
                for key in &l.pending {
                    let (face, lv, ci, cj) = unpack(*key);
                    let size = f64::from(BRICK << lv);
                    let dir = grid.direction(face, (f64::from(ci) + 0.5) * size, (f64::from(cj) + 0.5) * size);
                    angle = angle.min(dir.angle_between(eye_dir) - col);
                }
                angle.max(0.0)
            })
            .collect()
    }

    pub fn idle(&self) -> bool {
        self.urgent.is_empty() && self.applied == self.requested && self.levels.iter().all(|l| l.pending.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planet::PlanetRecipe;

    #[test]
    fn windows_are_bounded_and_complete_on_the_ground_and_in_orbit() {
        let planet = Planet::new(PlanetRecipe::default()).unwrap();
        let grid = *planet.grid();
        let lod0 = Residency::lod_distance(&grid, (22.5f64).to_radians().tan(), 1080, 1.0);
        assert!(lod0 > 100.0 && lod0 < 200.0, "{lod0}");
        for eye in [
            planet.surface_point(grid.direction(4, 1e7, 3e7), 1.8),
            grid.direction(2, 5e7, 5e7) * (grid.radius() + 300_000.0),
        ] {
            let mut residency = Residency::new(grid, Capacity::default());
            let mut total = 0;
            for _ in 0..1000 {
                let work = residency.plan(&planet, eye, lod0, 100_000);
                total += work.jobs.len();
                if residency.idle() {
                    break;
                }
            }
            eprintln!("{:?} levels {:?}", residency.stats, residency.levels.iter().map(|l| (l.keys.len(), l.pending.len())).collect::<Vec<_>>());
            assert!(residency.idle());
            assert!(total < 1_900_000, "{total}");
            assert_eq!(total, residency.residents.len());
            // A repeated plan at the same pose issues no work.
            let work = residency.plan(&planet, eye, lod0, 100_000);
            assert!(work.jobs.is_empty() && work.evictions.is_empty());
            eprintln!("resident columns {}", residency.residents.len());
        }
    }

    #[test]
    fn table_lookup_matches_residents_after_moves() {
        let planet = Planet::new(PlanetRecipe::default()).unwrap();
        let grid = *planet.grid();
        let mut residency = Residency::new(grid, Capacity { table_bits: 20, ..Default::default() });
        let mut eye = planet.surface_point(grid.direction(0, 3e7, 4e7), 2.0);
        for step in 0..40 {
            let _ = residency.plan(&planet, eye, 120.0, 20_000);
            eye = planet.surface_point(eye + DVec3::new(0.0, 0.0, 70.0 * f64::from(step % 3)), 2.0);
        }
        for (key, res) in &residency.residents {
            let mut slot = slot_hash(*key as u32, (key >> 32) as u32) & residency.mask();
            loop {
                let v = residency.table[slot as usize];
                assert_ne!(v, NONE, "key not reachable");
                if v == res.record {
                    break;
                }
                slot = (slot + 1) & residency.mask();
            }
        }
    }
}
