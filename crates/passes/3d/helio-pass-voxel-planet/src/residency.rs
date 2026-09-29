//! CPU side of the GPU clipmap: level windows, the column hash table mirror,
//! record and edit-list allocation, and prioritized generation jobs.
//!
//! The CPU decides *which* columns are resident; the GPU generates their
//! contents, allocates brick runs and publishes records in the same frame.
use crate::column_index::{ColumnIndex, Resident};
use crate::edits::FaceBrush;
use crate::grid::{Grid, BRICK};
use crate::windows::{LevelDiff, WindowPlanner, WindowRequest, WindowUpdate, WindowWorker};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};
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

/// First key word of a level column: its (never negative) column index in
/// 24 bits, the face and the level. 2^24 columns cover a 0.1 m Earth face
/// (1.25e7 columns) and an infinite plane (2^24).
pub fn key0(face: u8, level: u32, ci: i32) -> u32 {
    debug_assert!((0..1 << 24).contains(&ci), "column index {ci} outside 24 bits");
    (ci as u32 & 0xff_ffff) | (u32::from(face) << 24) | (level << 27)
}

pub fn slot_hash(key0: u32, key1: u32) -> u32 {
    crate::noise::hash3(key0 as i32, key1 as i32, 0x2f6b_1d3a, 0x9e37_79b9)
}

fn pack(k0: u32, k1: u32) -> u64 {
    u64::from(k0) | (u64::from(k1) << 32)
}

fn unpack(key: u64) -> (u8, u32, i32, i32) {
    let k0 = key as u32;
    let k1 = (key >> 32) as u32;
    (((k0 >> 24) & 7) as u8, k0 >> 27, (k0 & 0xff_ffff) as i32, k1 as i32)
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
    /// Ground radius of the applied window (metres).
    radius: f64,
    /// Wanted but not yet issued columns, and their priority heap (lazy).
    pending: rustc_hash::FxHashSet<u64>,
    heap: BinaryHeap<Reverse<Pending>>,
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
    /// Resident columns and the GPU column table mirror.
    residents: ColumnIndex,
    blocks: FxHashMap<(u32, u8, u32, i32, i32), Block>,
    block_owner: FxHashMap<u32, (u32, u8, u32, i32, i32)>,
    /// Dense list of live tier-1 block slots (GPU horizon build input) and
    /// each slot's position in it; `live_dirty` marks a pending upload.
    live_tier1: Vec<u32>,
    live_index: FxHashMap<u32, usize>,
    live_dirty: bool,
    /// Resident columns without summary blocks (slot conflicts).
    block_conflicts: usize,
    free_records: Vec<u32>,
    next_record: u32,
    delayed_records: Vec<u32>,
    levels: Vec<Level>,
    edits: EditHeap,
    /// GPU face-brush index for each (brush id, face entry).
    brush_gpu: Vec<Vec<u32>>,
    synced: Vec<crate::edits::Brush>,
    /// Prefix hashes of `synced` (see `EditLog::prefix_hash`).
    synced_hash: Vec<u64>,
    next_brush: u32,
    urgent: Vec<u64>,
    pub stats: Stats,
    frame: u32,
    planner: Planner,
    last_request: Option<WindowRequest>,
    requested: u64,
    applied: u64,
    /// Window diffs not yet fully applied, oldest first. A diff can hold
    /// hundreds of thousands of columns (leaving the ground retires the fine
    /// levels at once); it is applied in order within a CPU budget per frame.
    diffs: VecDeque<QueuedDiff>,
    /// Per level, diffs still queued for it (its window is not yet exact).
    catching_up: Vec<u32>,
    /// CPU time per `plan` for applying diffs and admitting columns; `None`
    /// is unbounded (deterministic, for tests).
    cpu_budget: Option<std::time::Duration>,
}

/// A window diff being applied: removes first, then (for a level switched
/// off) clearing its queue, then adds, exactly as an immediate apply.
struct QueuedDiff {
    diff: LevelDiff,
    removed: usize,
    cleared: bool,
    added: usize,
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
            residents: ColumnIndex::new(capacity.table_bits),
            blocks: FxHashMap::default(),
            block_owner: FxHashMap::default(),
            live_tier1: Vec::new(),
            live_index: FxHashMap::default(),
            live_dirty: false,
            block_conflicts: 0,
            free_records: Vec::new(),
            next_record: 0,
            delayed_records: Vec::new(),
            levels,
            edits: EditHeap::default(),
            brush_gpu: Vec::new(),
            synced: Vec::new(),
            synced_hash: Vec::new(),
            next_brush: 0,
            urgent: Vec::new(),
            stats: Stats::default(),
            frame: 0,
            planner,
            last_request: None,
            requested: 0,
            applied: 0,
            diffs: VecDeque::new(),
            catching_up: vec![0; grid.levels() as usize],
            cpu_budget: None,
        }
    }

    pub fn grid(&self) -> &Grid {
        &self.grid
    }

    pub fn table(&self) -> &[u32] {
        self.residents.table()
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
        // Longest common prefix of the synced and current logs, found by
        // prefix hash in O(log n); an unchanged log costs O(1) per frame.
        let n = self.synced.len().min(log.len());
        let same = |k: usize| k == 0 || self.synced_hash[k - 1] == log.prefix_hash((k - 1) as u32);
        if n == self.synced.len() && n == log.len() && same(n) {
            return;
        }
        let common = if same(n) {
            n
        } else {
            let (mut lo, mut hi) = (0, n);
            while lo < hi {
                let mid = (lo + hi + 1) / 2;
                if same(mid) {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            lo
        };
        let mut touched = Vec::new();
        for id in common..self.synced.len() {
            // Undone brushes: their old footprint must be regenerated.
            if let Ok(faces) = self.synced[id].resolve(&self.grid) {
                touched.extend(faces);
            }
        }
        self.brush_gpu.truncate(common);
        self.synced.truncate(common);
        self.synced_hash.truncate(common);
        for id in common..log.len() {
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
            self.synced.push(resolved.brush);
            self.synced_hash.push(resolved.prefix);
        }
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
                        if self.residents.contains_key(key) {
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

    /// Reference every summary block of a column, or none when any tier's
    /// table slot belongs to another block (a window larger than the table).
    fn acquire_blocks(&mut self, key: u64, work: &mut FrameWork) -> bool {
        let (face, level, ci, cj) = unpack(key);
        let conflict = (1..=BLOCK_TIERS).any(|tier| {
            let bkey = (level, face, tier, ci >> (2 * tier), cj >> (2 * tier));
            !self.blocks.contains_key(&bkey)
                && self.block_owner.get(&block_slot(level, face, tier, bkey.3, bkey.4)).is_some_and(|owner| *owner != bkey)
        });
        if conflict {
            return false;
        }
        for tier in 1..=BLOCK_TIERS {
            let (bi, bj) = (ci >> (2 * tier), cj >> (2 * tier));
            let bkey = (level, face, tier, bi, bj);
            if let Some(b) = self.blocks.get_mut(&bkey) {
                b.refs += 1;
                continue;
            }
            let slot = block_slot(level, face, tier, bi, bj);
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
        match self.residents.get(key).map(|r| r.blocks) {
            Some(true) => self.release_blocks(key, work),
            Some(false) => self.block_conflicts -= 1,
            None => {}
        }
        if let Some(res) = self.residents.remove(key, &mut work.table_writes) {
            work.evictions.push(res.record);
            self.delayed_records.push(res.record);
            if let Some(block) = res.edit_block {
                self.edits.release(block);
            }
            let (_, level, _, _) = unpack(key);
        }
    }

    /// CPU time `plan` may spend per frame applying window diffs and
    /// admitting columns (`None`: unbounded). The rest carries over.
    pub fn set_cpu_budget(&mut self, budget: Option<std::time::Duration>) {
        self.cpu_budget = budget;
    }

    /// Queue a window diff; [`Self::apply_queued`] applies it in order.
    fn apply(&mut self, update: WindowUpdate) {
        for diff in update.levels {
            let level = diff.level as usize;
            // The window metadata changes at once; the level is marked as
            // catching up (no guaranteed coverage) until its ops are done.
            self.levels[level].active = diff.active;
            self.levels[level].center = diff.center;
            self.levels[level].radius = diff.radius;
            self.catching_up[level] += 1;
            self.diffs.push_back(QueuedDiff { diff, removed: 0, cleared: false, added: 0 });
        }
        self.stats.window_rebuild_ms = update.planning_ms;
        self.applied = update.serial;
    }

    /// Apply queued window diffs in order until done or out of time.
    fn apply_queued(&mut self, work: &mut FrameWork, out_of_time: &impl Fn() -> bool) {
        const CHUNK: usize = 2048;
        while let Some(mut queued) = self.diffs.pop_front() {
            let level = queued.diff.level as usize;
            while queued.removed < queued.diff.removes.len() {
                let end = (queued.removed + CHUNK).min(queued.diff.removes.len());
                for i in queued.removed..end {
                    let key = queued.diff.removes[i];
                    self.levels[level].pending.remove(&key);
                    if self.residents.contains_key(key) {
                        self.evict(key, work);
                    }
                }
                queued.removed = end;
                if out_of_time() {
                    self.diffs.push_front(queued);
                    return;
                }
            }
            if !queued.diff.active && !queued.cleared {
                self.levels[level].heap.clear();
                self.levels[level].pending.clear();
                queued.cleared = true;
            }
            while queued.added < queued.diff.adds.len() {
                let end = (queued.added + CHUNK).min(queued.diff.adds.len());
                for i in queued.added..end {
                    let (priority, key) = queued.diff.adds[i];
                    if !self.residents.contains_key(key) && self.levels[level].pending.insert(key) {
                        self.levels[level].heap.push(Reverse(Pending(priority, key)));
                    }
                }
                queued.added = end;
                if out_of_time() && queued.added < queued.diff.adds.len() {
                    self.diffs.push_front(queued);
                    return;
                }
            }
            self.catching_up[level] -= 1;
        }
    }

    /// Plan one frame. `lod0` is the level-0 distance, `budget` the maximum
    /// number of column jobs.
    pub fn plan(&mut self, planet: &Planet, eye: DVec3, lod0: f64, budget: usize) -> FrameWork {
        self.frame = self.frame.wrapping_add(1);
        let started = std::time::Instant::now();
        let budget_time = self.cpu_budget;
        let out_of_time = move || budget_time.is_some_and(|b| started.elapsed() >= b);
        let mut work = FrameWork::default();
        // Records evicted last frame are safe to reuse now.
        let delayed = std::mem::take(&mut self.delayed_records);
        self.free_records.extend(delayed);
        self.sync_edits(planet, &mut work);
        let t_edits = started.elapsed();
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
                    self.apply(update);
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
                self.apply(update);
            }
        }
        let t_drain = started.elapsed();
        self.apply_queued(&mut work, &out_of_time);
        let t_apply = started.elapsed();
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
            let Some(res) = self.residents.get(key) else { continue };
            let Ok(block) = self.edit_list(planet, key, &mut work) else {
                deferred_urgent.push(key);
                continue;
            };
            if let Some(old) = res.edit_block {
                self.edits.release(old);
            }
            self.residents.get_mut(key).unwrap().edit_block = block;
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
        // Admission costs ~2 us of CPU per column (edit query, summary
        // blocks, table), and discarding stale heap entries (columns a diff
        // removed while queued; a big window change leaves hundreds of
        // thousands) ~50 ns each: both are bounded by time as well as by the
        // GPU budget, and resume next frame.
        let mut steps = 0u32;
        'admit: while work.jobs.len() < budget {
            steps += 1;
            if steps % 64 == 0 && out_of_time() {
                break;
            }
            let mut best: Option<(f32, usize)> = None;
            for index in 0..self.levels.len() {
                let l = &mut self.levels[index];
                while let Some(Reverse(Pending(_, key))) = l.heap.peek() {
                    if l.pending.contains(key) {
                        break;
                    }
                    l.heap.pop();
                    steps += 1;
                    if steps % 1024 == 0 && out_of_time() {
                        break 'admit;
                    }
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
            if self.residents.contains_key(key) {
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
            // Columns always become resident; one whose summary blocks alias
            // another block's table slots simply has no summaries.
            let blocks = self.acquire_blocks(key, &mut work);
            if !blocks {
                self.block_conflicts += 1;
            }
            let slot = self.residents.insert(key, Resident { record, slot: 0, edit_block: block, blocks });
            work.table_writes.push((slot, record));
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
        if std::env::var_os("HELIO_VOXEL_PLAN_TRACE").is_some() && started.elapsed().as_secs_f64() > 0.01 {
            eprintln!(
                "PLAN_TRACE edits {:.2} drain {:.2} apply {:.2} admit {:.2} ms jobs {} evictions {} queued_diffs {}",
                t_edits.as_secs_f64() * 1e3,
                (t_drain - t_edits).as_secs_f64() * 1e3,
                (t_apply - t_drain).as_secs_f64() * 1e3,
                (started.elapsed() - t_apply).as_secs_f64() * 1e3,
                work.jobs.len(),
                work.evictions.len(),
                self.diffs.len()
            );
        }
        let mut stats = self.stats;
        stats.resident_columns = self.residents.len();
        stats.pending_columns = self.levels.iter().map(|l| l.pending.len()).sum::<usize>() + self.urgent.len();
        stats.active_levels = self.levels.iter().filter(|l| l.active).count() as u32;
        stats.finest_level = self.levels.iter().position(|l| l.active).unwrap_or(0) as u32;
        stats.jobs = work.jobs.len();
        stats.evictions = work.evictions.len();
        stats.edit_words = self.edits.top;
        stats.table_load = self.residents.load();
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
            if !self.residents.contains_key(key) {
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
    /// Every resident column owns its summary blocks, so a tier-1 block
    /// whose key does not match (or that has no published column) proves
    /// its columns absent.
    pub fn blocks_exact(&self) -> bool {
        self.block_conflicts == 0
    }
    pub fn live_block_count(&self) -> usize {
        self.live_tier1.len()
    }

    /// Per level, the ground distance (metres) from `eye` within which every
    /// column the traversal can want is resident, so rays nearer than that
    /// never fall back to a coarser level: bounded by the (possibly lagging)
    /// window and by the nearest pending column. Inactive levels give 0.
    pub fn fallback_distances(&self, eye: DVec3) -> Vec<f64> {
        let grid = self.grid;
        let ground = |p: DVec3| if grid.is_plane() { DVec3::new(p.x, 0.0, p.z) } else { p.normalize() };
        let urgent: rustc_hash::FxHashSet<u32> = self.urgent.iter().map(|k| unpack(*k).1).collect();
        self.levels
            .iter()
            .enumerate()
            .map(|(level, l)| {
                if !l.active || urgent.contains(&(level as u32)) || self.catching_up[level] > 0 {
                    return 0.0;
                }
                // A column's ground width (the index-angle span on a sphere
                // bounds its true size).
                let col = grid.delta() * f64::from(BRICK << level) * if grid.is_plane() { 1.0 } else { grid.radius() };
                let mut distance = l.radius - col * 1.5 - grid.ground_distance(l.center, ground(eye));
                if l.pending.len() > 4096 {
                    return 0.0;
                }
                for key in &l.pending {
                    let (face, lv, ci, cj) = unpack(*key);
                    let size = f64::from(BRICK << lv);
                    let p = grid.ground_point(face, (f64::from(ci) + 0.5) * size, (f64::from(cj) + 0.5) * size);
                    // Traversal uses only complete 4x4-column blocks while a
                    // level streams in: a pending column makes its whole
                    // block (within its diagonal, 5.7 columns) fall back.
                    distance = distance.min(grid.ground_distance(p, eye) - col * 6.0);
                }
                distance.max(0.0)
            })
            .collect()
    }

    pub fn idle(&self) -> bool {
        self.urgent.is_empty()
            && self.applied == self.requested
            && self.diffs.is_empty()
            && self.levels.iter().all(|l| l.pending.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planet::PlanetRecipe;

    /// The GPU hash table holds exactly the residents, each found by linear
    /// probing from its home slot before any empty slot (what `find_column`
    /// does), after evictions moved entries back.
    fn table_is_exact(r: &Residency) {
        let table = r.table();
        let mask = (table.len() - 1) as u32;
        assert_eq!(table.iter().filter(|&&v| v != NONE).count(), r.residents.len());
        for (key, res) in r.residents.iter() {
            assert_eq!(table[res.slot as usize], res.record);
            let mut slot = slot_hash(key as u32, (key >> 32) as u32) & mask;
            loop {
                let v = table[slot as usize];
                assert_ne!(v, NONE, "column {key:x} unreachable from its home slot");
                if v == res.record {
                    break;
                }
                slot = (slot + 1) & mask;
            }
        }
    }

    #[test]
    fn budgeted_planning_spreads_big_diffs_and_converges_to_the_same_residency() {
        let planet = Planet::new(PlanetRecipe::default()).unwrap();
        let grid = *planet.grid();
        let lod0 = Residency::lod_distance(&grid, (22.5f64).to_radians().tan(), 720, 1.0);
        let ground = planet.surface_point(grid.direction(2, 3e7, 4e7), 1.8);
        let high = ground.normalize() * (ground.length() + 8_000.0);
        let settle = |r: &mut Residency, eye: DVec3, worst: &mut f64| {
            for _ in 0..20_000 {
                let started = std::time::Instant::now();
                r.plan(&planet, eye, lod0, 100_000);
                *worst = worst.max(started.elapsed().as_secs_f64() * 1000.0);
                if r.idle() {
                    return;
                }
            }
            panic!("did not converge");
        };
        let mut plain = Residency::new(grid, Capacity::default());
        let mut budgeted = Residency::new(grid, Capacity::default());
        budgeted.set_cpu_budget(Some(std::time::Duration::from_millis(2)));
        let (mut plain_worst, mut budget_worst) = (0.0, 0.0);
        for eye in [ground, high, ground] {
            settle(&mut plain, eye, &mut plain_worst);
            settle(&mut budgeted, eye, &mut budget_worst);
            let keys = |r: &Residency| {
                let mut k: Vec<u64> = r.residents.iter().map(|(k, _)| k).collect();
                k.sort_unstable();
                k
            };
            assert_eq!(keys(&plain), keys(&budgeted));
            assert_eq!(plain.fallback_distances(eye), budgeted.fallback_distances(eye));
            for r in [&plain, &budgeted] {
                table_is_exact(r);
            }
        }
        // Frame cost stays near the budget (debug builds are slower). The
        // inline planner's own window diff is outside the budget.
        eprintln!("worst plan: unbounded {plain_worst:.1} ms, budgeted {budget_worst:.1} ms");
    }

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
            eprintln!("{:?} pending {:?}", residency.stats, residency.levels.iter().map(|l| l.pending.len()).collect::<Vec<_>>());
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
        table_is_exact(&residency);
    }
}
