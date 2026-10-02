//! CPU side of the GPU clipmap: level windows, the column hash table mirror,
//! record and edit-list allocation, and prioritized generation jobs.
//!
//! The CPU decides *which* columns are resident; the GPU generates their
//! contents, allocates brick runs and publishes records in the same frame.
use crate::column_index::{ColumnIndex, Resident};
use crate::edits::FaceBrush;
use crate::grid::{Grid, BRICK};
use crate::windows::{LevelDiff, WindowPlanner, WindowRequest, WindowUpdate, WindowWorker};
use std::collections::VecDeque;
use crate::planet::Planet;
use bytemuck::{Pod, Zeroable};
use glam::DVec3;
use rustc_hash::{FxHashMap, FxHashSet};

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

/// Priority buckets of the pending queue (normalized window distance).
const BUCKETS: usize = 64;

/// Wanted but not yet issued columns of one level, by priority bucket.
/// Removal is exact (swap-remove with a position index): a window moving
/// at speed retires most columns before they are issued, and a lazy heap
/// kept millions of stale entries that every admission had to pop.
#[derive(Default)]
struct PendingQueue {
    buckets: Vec<Vec<u64>>,
    at: FxHashMap<u64, (u8, u32)>,
    /// Lowest bucket that may be non-empty.
    lowest: usize,
}

impl PendingQueue {
    fn bucket(priority: f32) -> usize {
        ((priority.max(0.0) * BUCKETS as f32) as usize).min(BUCKETS - 1)
    }
    fn len(&self) -> usize {
        self.at.len()
    }
    fn is_empty(&self) -> bool {
        self.at.is_empty()
    }
    fn keys(&self) -> impl Iterator<Item = &u64> {
        self.at.keys()
    }
    /// Queue or reprioritize `key`; false when its bucket is unchanged.
    fn insert(&mut self, key: u64, bucket: usize) -> bool {
        if let Some(&(old, _)) = self.at.get(&key) {
            if old as usize == bucket {
                return false;
            }
            self.remove(key);
        }
        if self.buckets.is_empty() {
            self.buckets.resize(BUCKETS, Vec::new());
            self.lowest = BUCKETS;
        }
        let list = &mut self.buckets[bucket];
        self.at.insert(key, (bucket as u8, list.len() as u32));
        list.push(key);
        self.lowest = self.lowest.min(bucket);
        true
    }
    fn remove(&mut self, key: u64) -> bool {
        let Some((bucket, index)) = self.at.remove(&key) else { return false };
        let list = &mut self.buckets[bucket as usize];
        list.swap_remove(index as usize);
        if let Some(&moved) = list.get(index as usize) {
            self.at.get_mut(&moved).unwrap().1 = index;
        }
        true
    }
    /// Lowest non-empty bucket.
    fn best(&mut self) -> Option<usize> {
        while self.lowest < self.buckets.len() && self.buckets[self.lowest].is_empty() {
            self.lowest += 1;
        }
        (self.lowest < self.buckets.len()).then_some(self.lowest)
    }
    /// Take a column of the lowest bucket (the newest one queued there).
    fn pop(&mut self) -> Option<(u64, usize)> {
        let bucket = self.best()?;
        let key = self.buckets[bucket].pop()?;
        self.at.remove(&key);
        Some((key, bucket))
    }
    fn clear(&mut self) {
        self.buckets.iter_mut().for_each(Vec::clear);
        self.at.clear();
        self.lowest = self.buckets.len();
    }
}

#[derive(Default)]
struct Level {
    active: bool,
    /// Latest worker demand, independent of partially applied older diffs.
    wanted: Option<std::sync::Arc<FxHashSet<u64>>>,
    /// Window centre and angular radius of the last applied diff.
    center: DVec3,
    /// Ground radius of the applied window (metres).
    radius: f64,
    /// Wanted but not yet issued columns.
    pending: PendingQueue,
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

/// A submitted column owns both journals until its publication result is known.
struct EditPublication {
    record: u32,
    previous: Option<(u32, u32)>,
    next: Option<(u32, u32)>,
    evicted: bool,
    /// Initial allocation failures retry through normal visibility priorities.
    initial_bucket: Option<usize>,
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
    publishing: FxHashMap<u64, EditPublication>,
    initial_retries: FxHashSet<u64>,
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
    /// Window diffs in FIFO order within each level. A large fine-window
    /// retirement must not delay incoming demand at every coarser level.
    diffs: Vec<VecDeque<QueuedDiff>>,
    /// Next non-global level to receive a bounded diff round.
    diff_cursor: usize,
    /// Per level, diffs still queued for it (its window is not yet exact).
    catching_up: Vec<u32>,
    /// CPU time per `plan` for applying diffs and admitting columns; `None`
    /// is unbounded (deterministic, for tests).
    cpu_budget: Option<std::time::Duration>,
    prefetch_eye: Option<DVec3>,
    priority_eye: Option<DVec3>,
    view_focus: Option<DVec3>,
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
            publishing: FxHashMap::default(),
            initial_retries: FxHashSet::default(),
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
            diffs: (0..grid.levels()).map(|_| VecDeque::new()).collect(),
            diff_cursor: 0,
            catching_up: vec![0; grid.levels() as usize],
            cpu_budget: None,
            prefetch_eye: None,
            priority_eye: None,
            view_focus: None,
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
                if !fb.active(level) { continue; }
                let col = i64::from(BRICK) << level;
                // Extended brush coordinates overlap adjacent faces, but
                // column keys only encode valid indices on this face.
                let last = (i64::from(self.grid.cells()) - 1).div_euclid(col);
                let (i0, i1) = ((ci - r_cells).div_euclid(col).max(0), (ci + r_cells).div_euclid(col).min(last));
                let (j0, j1) = ((cj - r_cells).div_euclid(col).max(0), (cj + r_cells).div_euclid(col).min(last));
                if i0 > i1 || j0 > j1 { continue; }
                if (i1 - i0 + 1) * (j1 - j0 + 1) > 1 << 16 {
                    // A large valid brush can cover millions of possible
                    // columns, but only the resident subset needs rebuilding.
                    for (key, _) in self.residents.iter() {
                        let (face, lv, a, b) = unpack(key);
                        if face == fb.face() && lv == level
                            && (i0..=i1).contains(&i64::from(a))
                            && (j0..=j1).contains(&i64::from(b))
                        {
                            self.urgent.push(key);
                        }
                    }
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

    fn blocks_conflict(&self, key: u64) -> bool {
        let (face, level, ci, cj) = unpack(key);
        (1..=BLOCK_TIERS).any(|tier| {
            let bkey = (level, face, tier, ci >> (2 * tier), cj >> (2 * tier));
            !self.blocks.contains_key(&bkey)
                && self.block_owner.get(&block_slot(level, face, tier, bkey.3, bkey.4)).is_some_and(|owner| *owner != bkey)
        })
    }

    /// Columns in one aligned tier-1 block share every summary owner.
    /// Cache only the last block: an intervening admission may claim a slot
    /// that was free at an earlier check of another block.
    fn blocks_conflict_cached(&self, key: u64, last: &mut Option<(u64, bool)>) -> bool {
        let block = key & !(3u64 | (3u64 << 32));
        if let Some((previous, conflict)) = *last {
            if previous == block { return conflict; }
        }
        let conflict = self.blocks_conflict(key);
        *last = Some((block, conflict));
        conflict
    }

    /// Reference every summary block of a column, or none when any tier's
    /// table slot belongs to another block (a window larger than the table).
    #[cfg(test)]
    fn acquire_blocks(&mut self, key: u64, work: &mut FrameWork) -> bool {
        if self.blocks_conflict(key) {
            return false;
        }
        self.reference_blocks(key, work);
        true
    }

    /// Admission has checked ownership; no retirement occurs until the next
    /// plan, so referencing this same block cannot introduce an alias.
    fn reference_blocks(&mut self, key: u64, work: &mut FrameWork) {
        let (face, level, ci, cj) = unpack(key);
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
        self.initial_retries.remove(&key);
        match self.residents.get(key).map(|r| r.blocks) {
            Some(true) => self.release_blocks(key, work),
            Some(false) => self.block_conflicts -= 1,
            None => {}
        }
        if let Some(res) = self.residents.remove(key, &mut work.table_writes) {
            work.evictions.push(res.record);
            self.delayed_records.push(res.record);
            if let Some(publication) = self.publishing.get_mut(&key) {
                // The result can arrive after this record has been reused for
                // another key. Keep ownership until then and ignore that result.
                publication.evicted = true;
            } else if let Some(block) = res.edit_block {
                self.edits.release(block);
            }
        }
    }

    /// CPU time `plan` may spend per frame applying window diffs and
    /// admitting columns (`None`: unbounded). The rest carries over.
    pub fn set_cpu_budget(&mut self, budget: Option<std::time::Duration>) {
        self.cpu_budget = budget;
    }

    pub fn set_prefetch_eye(&mut self, eye: Option<DVec3>) {
        self.prefetch_eye = eye.filter(|eye| eye.is_finite());
    }

    /// Longer motion lookahead only reprioritizes existing pending columns.
    /// It does not enlarge the resident windows or their coverage metadata.
    pub fn set_priority_eye(&mut self, eye: Option<DVec3>) {
        self.priority_eye = eye.filter(|eye| eye.is_finite());
    }

    /// Priority-only focus: does not expand windows or claim resident data.
    pub fn set_view_focus(&mut self, focus: Option<DVec3>) {
        self.view_focus = focus.filter(|point| point.is_finite());
    }

    /// Publish latest demand before queueing bounded window operations.
    fn apply(&mut self, update: WindowUpdate) {
        for (level, wanted) in update.wanted {
            self.levels[level as usize].wanted = Some(wanted);
        }
        for diff in update.levels {
            let level = diff.level as usize;
            // The window metadata changes at once; the level is marked as
            // catching up (no guaranteed coverage) until its ops are done.
            self.levels[level].active = diff.active;
            self.levels[level].center = diff.center;
            self.levels[level].radius = diff.radius;
            self.catching_up[level] += 1;
            self.diffs[level].push_back(QueuedDiff { diff, removed: 0, cleared: false, added: 0 });
        }
        self.stats.window_rebuild_ms = update.planning_ms;
        self.applied = update.serial;
    }

    /// Global coverage wins; other levels share rounds, keeping their own FIFO.
    fn next_diff_level(&mut self) -> Option<usize> {
        let top = self.diffs.len() - 1;
        if !self.diffs[top].is_empty() { return Some(top); }
        for _ in 0..top {
            let level = self.diff_cursor;
            self.diff_cursor = (self.diff_cursor + 1) % top;
            if !self.diffs[level].is_empty() { return Some(level); }
        }
        None
    }

    /// Retire oldest owners while admitting the newest incoming demand.
    /// Latest membership makes out-of-order additions safe: an old removal
    /// cannot evict a returned column, and obsolete additions cannot run.
    fn apply_queued(&mut self, work: &mut FrameWork, out_of_time: &impl Fn() -> bool) {
        const CHUNK: usize = 256;
        while let Some(level) = self.next_diff_level() {
            let mut queued = self.diffs[level].pop_front().unwrap();
            let wanted = self.levels[level].wanted.clone();
            // Retiring a moving window must not consume every diff slice
            // before visible incoming columns reach the generation queue.
            // Keep the same operation bound, sharing active rounds equally.
            let remove_chunk = if queued.diff.active || wanted.is_some() { CHUNK / 2 } else { CHUNK };
            let end = (queued.removed + remove_chunk).min(queued.diff.removes.len());
            for i in queued.removed..end {
                let key = queued.diff.removes[i];
                if wanted.as_ref().is_some_and(|keys| keys.contains(&key)) { continue; }
                self.levels[level].pending.remove(key);
                if self.residents.contains_key(key) {
                    self.evict(key, work);
                }
            }
            queued.removed = end;
            if wanted.is_none() && !queued.diff.active && queued.removed == queued.diff.removes.len() && !queued.cleared {
                self.levels[level].pending.clear();
                queued.cleared = true;
            }
            if let Some(wanted) = &wanted {
                // A subsequent diff contains the frontier visible now. Do not
                // make it wait for an earlier camera's complete window.
                let newest = self.diffs[level].iter().rposition(|diff| diff.added < diff.diff.adds.len());
                let incoming = if let Some(at) = newest { &mut self.diffs[level][at] } else { &mut queued };
                let end = (incoming.added + CHUNK / 2).min(incoming.diff.adds.len());
                for i in incoming.added..end {
                    let (priority, key) = incoming.diff.adds[i];
                    if wanted.contains(&key) && !self.residents.contains_key(key) {
                        self.levels[level].pending.insert(key, PendingQueue::bucket(priority));
                    }
                }
                incoming.added = end;
            } else if queued.diff.active || queued.removed == queued.diff.removes.len() {
                let end = (queued.added + if queued.diff.active { CHUNK / 2 } else { CHUNK }).min(queued.diff.adds.len());
                for i in queued.added..end {
                    let (priority, key) = queued.diff.adds[i];
                    if !self.residents.contains_key(key) {
                        self.levels[level].pending.insert(key, PendingQueue::bucket(priority));
                    }
                }
                queued.added = end;
            }
            if queued.removed == queued.diff.removes.len() && queued.added == queued.diff.adds.len() {
                self.catching_up[level] -= 1;
            } else {
                self.diffs[level].push_front(queued);
            }
            // Newer additions may finish a diff before its FIFO turn.
            while self.diffs[level].back().is_some_and(|diff|
                diff.removed == diff.diff.removes.len() && diff.added == diff.diff.adds.len()) {
                self.diffs[level].pop_back();
                self.catching_up[level] -= 1;
            }
            if out_of_time() { return; }
        }
    }

    /// Refresh a fixed neighborhood of complete tier-1 blocks. Traversal
    /// only uses a streaming level after all sixteen columns of a block are
    /// published, so keep its pending columns together at one priority.
    /// Only already wanted columns are touched; incomplete window edges
    /// retain their original queue order rather than consuming this priority.
    fn refresh_near_pending(&mut self, eye: DVec3, deadline: Option<std::time::Instant>) {
        let out_of_time = || deadline.is_some_and(|at| std::time::Instant::now() >= at);
        if out_of_time() { return; }
        let grid = self.grid;
        let forecast = self.priority_eye.or(self.prefetch_eye);
        let forecast_penalty = forecast.map_or(0.0, |future| grid.ground_distance(eye, future) * 0.25);
        let focus = self.view_focus;
        let focus_penalty = focus.map_or(0.0, |point| grid.ground_distance(eye, point) * 0.125);
        let anchors = [Some(eye), forecast, focus];
        let mut coordinates = [[None; 3]; 6];
        for &face in grid.faces() {
            for (index, anchor) in anchors.iter().enumerate() {
                coordinates[face as usize][index] = anchor.and_then(|point| grid.face_coords(face, point));
            }
        }
        // At cube seams prioritize the actual camera's face before its
        // neighbors; the near levels are already visited before far levels.
        let primary_face = if grid.is_plane() { crate::grid::PLANE_FACE } else { crate::grid::face_of(eye) };
        let mut faces = [0u8; 6];
        faces[..grid.faces().len()].copy_from_slice(grid.faces());
        faces[..grid.faces().len()].sort_unstable_by_key(|face| *face != primary_face);
        for (level, state) in self.levels.iter_mut().enumerate() {
            if !state.active || state.pending.is_empty() {
                continue;
            }
            let cells = BRICK << level;
            let columns = grid.cells() / cells;
            let size = f64::from(cells);
            for &face in &faces[..grid.faces().len()] {
                // Check between fixed face groups, so refresh cannot consume
                // the generation-admission reserve when a backlog grows.
                if out_of_time() { return; }
                // Two aligned blocks per axis around each anchor: at most
                // 3 * 4 * 16 key probes per face/level, regardless of backlog.
                let mut blocks = [(0i32, 0i32, 0.0f64); 12];
                let mut count = 0usize;
                for coords in coordinates[face as usize].into_iter().flatten() {
                    let ci = (coords[0] / size).floor() as i32;
                    let cj = (coords[1] / size).floor() as i32;
                    let bi = ci.div_euclid(4);
                    let bj = cj.div_euclid(4);
                    let ni = bi + if ci.rem_euclid(4) < 2 { -1 } else { 1 };
                    let nj = bj + if cj.rem_euclid(4) < 2 { -1 } else { 1 };
                    for y in [bj, nj] {
                        for x in [bi, ni] {
                            if x < 0 || y < 0 || x >= columns / 4 || y >= columns / 4
                                || blocks[..count].iter().any(|&(i, j, _)| i == x && j == y) { continue; }
                            let point = grid.ground_point(face, f64::from(x * 4 + 2) * size, f64::from(y * 4 + 2) * size);
                            let current_distance = grid.ground_distance(point, eye);
                            let forecast_distance = forecast.map_or(current_distance, |future| {
                                current_distance.min(grid.ground_distance(point, future) + forecast_penalty)
                            });
                            let distance = focus.map_or(forecast_distance, |point_of_interest| {
                                forecast_distance.min(grid.ground_distance(point, point_of_interest) + focus_penalty)
                            });
                            blocks[count] = (x, y, distance);
                            count += 1;
                        }
                    }
                }
                // Buckets pop newest first: insert far blocks first, keeping
                // the actual-camera block first when bucket rounding ties.
                blocks[..count].sort_unstable_by(|a, b| b.2.total_cmp(&a.2));
                for &(bi, bj, distance) in &blocks[..count] {
                    // A CPU-resident full block has no pending admissions to
                    // group. Avoid sixteen hash probes every stopped frame.
                    if self.initial_retries.is_empty()
                        && self.blocks.get(&(level as u32, face, 1, bi, bj)).is_some_and(|block| block.refs == 16) {
                        continue;
                    }
                    let mut keys = [0u64; 16];
                    let mut complete = true;
                    for (index, key) in keys.iter_mut().enumerate() {
                        let i = bi * 4 + (index % 4) as i32;
                        let j = bj * 4 + (index / 4) as i32;
                        *key = pack(key0(face, level as u32, i), j as u32);
                        complete &= state.pending.at.contains_key(key) || self.residents.contains_key(*key);
                    }
                    if !complete { continue; }
                    let priority = distance / state.radius.max(grid.level_size(level as u32));
                    let bucket = PendingQueue::bucket(priority as f32);
                    for key in keys {
                        if state.pending.remove(key) {
                            state.pending.insert(key, bucket);
                        }
                    }
                }
            }
        }
    }

    /// Plan one frame. `lod0` is the level-0 distance, `budget` the maximum
    /// number of column jobs.
    pub fn plan(&mut self, planet: &std::sync::Arc<Planet>, eye: DVec3, lod0: f64, budget: usize) -> FrameWork {
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
            prefetch_eye: self.prefetch_eye,
            priority_eye: self.priority_eye,
            view_focus: self.view_focus,
            lod0,
            outer_radius: planet.outer_radius(),
            planet: Some(planet.clone()),
            serial: self.requested + 1,
        };
        let changed = self.last_request.as_ref().is_none_or(|last| {
            last.eye.distance(eye) > self.grid.voxel_size() * 2.0
                || (last.lod0 - lod0).abs() > lod0 * 0.01
                || last.outer_radius != request.outer_radius
                || last.prefetch_eye != request.prefetch_eye
        });
        if changed {
            self.requested = request.serial;
            self.last_request = Some(request.clone());
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
        // Diffs take at most 60 % of the time while columns wait, so a
        // window moving at speed cannot starve generation.
        let share = if self.levels.iter().all(|l| l.pending.is_empty()) { 1.0 } else { 0.6 };
        let apply_out_of_time = move || budget_time.is_some_and(|b| started.elapsed() >= b.mul_f64(share));
        self.apply_queued(&mut work, &apply_out_of_time);
        // Window diffs may spend 60% of the CPU budget. Refresh gets at most
        // the next 10%, leaving 30% for issuing generation jobs this frame.
        let refresh_deadline = budget_time.map(|budget| started + budget.mul_f64(0.7));
        self.refresh_near_pending(eye, refresh_deadline);
        let t_apply = started.elapsed();
        // Urgent edit regenerations first.
        let mut urgent = std::mem::take(&mut self.urgent);
        urgent.sort_unstable();
        urgent.dedup();
        let mut deferred_urgent = Vec::new();
        let mut urgent = urgent.into_iter();
        while let Some(key) = urgent.next() {
            if work.jobs.len() >= budget || out_of_time() {
                deferred_urgent.push(key);
                deferred_urgent.extend(urgent);
                break;
            }
            if self.publishing.contains_key(&key) {
                deferred_urgent.push(key);
                continue;
            }
            let Some(res) = self.residents.get(key) else { continue };
            let Ok(block) = self.edit_list(planet, key, &mut work) else {
                deferred_urgent.push(key);
                continue;
            };
            let initial = self.initial_retries.remove(&key);
            if initial {
                self.levels[unpack(key).1 as usize].pending.remove(key);
            }
            self.publishing.insert(key, EditPublication {
                record: res.record,
                previous: res.edit_block,
                next: block,
                evicted: false,
                initial_bucket: None,
            });
            work.jobs.push(Job {
                key0: key as u32,
                key1: (key >> 32) as u32,
                record: res.record,
                edits: block.map_or(0, |b| b.0 + 1),
                flags: u32::from(!initial),
                pad: [0; 3],
            });
            work.job_keys.push(key);
        }
        self.urgent = deferred_urgent;
        // Merge pending windows by normalized distance; the coarsest level
        // (global coverage) always goes first.
        let top_level = self.grid.levels() - 1;
        // Admission costs ~2 us of CPU per column (edit query, summary
        // blocks, table): bounded by time as well as by the GPU budget, and
        // resumes next frame.
        let mut steps = 0u32;
        let mut awaiting_publication = Vec::new();
        let mut last_summary_check = None;
        while work.jobs.len() < budget {
            steps += 1;
            if steps % 64 == 0 && out_of_time() {
                break;
            }
            let mut best: Option<(usize, usize)> = None;
            for index in 0..self.levels.len() {
                if let Some(bucket) = self.levels[index].pending.best() {
                    // Normalized distance, the global level before any other.
                    let p = if index as u32 == top_level { 0 } else { bucket + 1 };
                    if best.is_none_or(|b| p < b.0) {
                        best = Some((p, index));
                    }
                }
            }
            let Some((_, index)) = best else { break };
            let (key, bucket) = self.levels[index].pending.pop().unwrap();
            if self.levels[index].wanted.as_ref().is_some_and(|wanted| !wanted.contains(&key)) {
                continue;
            }
            let retry = self.initial_retries.contains(&key);
            if self.residents.contains_key(key) && !retry {
                continue;
            }
            let requeue = |this: &mut Self| {
                this.levels[index].pending.insert(key, bucket);
            };
            // A retired key may still have a publication result in flight.
            // Do not let that result acknowledge a different incarnation.
            if self.publishing.contains_key(&key) {
                // Keep this incarnation queued, but let unrelated columns
                // use the remaining budget while its GPU result is in flight.
                awaiting_publication.push((index, key, bucket));
                continue;
            }
            // Active diffs can now queue incoming columns before all outgoing
            // blocks retire. Wait for an alias owner instead of permanently
            // publishing a column without summaries during a large move.
            let conflict = if !retry || self.catching_up[index] > 0 {
                self.blocks_conflict_cached(key, &mut last_summary_check)
            } else { false };
            if self.catching_up[index] > 0 && conflict {
                awaiting_publication.push((index, key, bucket));
                continue;
            }
            let record = if retry { Some(self.residents.get(key).unwrap().record) } else { self.alloc_record() };
            let Some(record) = record else {
                requeue(self);
                break;
            };
            let Ok(block) = self.edit_list(planet, key, &mut work) else {
                if !retry { self.free_records.push(record); }
                requeue(self);
                break;
            };
            // Columns always become resident; one whose summary blocks alias
            // another block's table slots simply has no summaries.
            if !retry {
                let blocks = !conflict;
                if blocks { self.reference_blocks(key, &mut work); }
                if !blocks {
                    self.block_conflicts += 1;
                }
                let slot = self.residents.insert(key, Resident { record, slot: 0, edit_block: None, blocks });
                work.table_writes.push((slot, record));
            }
            self.initial_retries.remove(&key);
            self.publishing.insert(key, EditPublication {
                record,
                previous: None,
                next: block,
                evicted: false,
                initial_bucket: Some(bucket),
            });
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
        for (index, key, bucket) in awaiting_publication {
            self.levels[index].pending.insert(key, bucket);
        }
        let trace_ms: Option<f64> = std::env::var("HELIO_VOXEL_PLAN_TRACE").ok().map(|v| v.parse().unwrap_or(10.0));
        if trace_ms.is_some_and(|ms| started.elapsed().as_secs_f64() * 1e3 > ms) {
            eprintln!(
                "PLAN_TRACE edits {:.2} drain {:.2} apply {:.2} admit {:.2} ms jobs {} evictions {} queued_diffs {} steps {steps}",
                t_edits.as_secs_f64() * 1e3,
                (t_drain - t_edits).as_secs_f64() * 1e3,
                (t_apply - t_drain).as_secs_f64() * 1e3,
                (started.elapsed() - t_apply).as_secs_f64() * 1e3,
                work.jobs.len(),
                work.evictions.len(),
                self.diffs.iter().map(VecDeque::len).sum::<usize>()
            );
        }
        let mut stats = self.stats;
        stats.resident_columns = self.residents.len();
        stats.pending_columns = self.levels.iter().map(|l| l.pending.len()).sum::<usize>() + self.urgent.len() + self.publishing.len();
        stats.active_levels = self.levels.iter().filter(|l| l.active).count() as u32;
        stats.finest_level = self.levels.iter().position(|l| l.active).unwrap_or(0) as u32;
        stats.jobs = work.jobs.len();
        stats.evictions = work.evictions.len();
        stats.edit_words = self.edits.top;
        stats.table_load = self.residents.load();
        self.stats = stats;
        work
    }

    /// Acknowledge every submitted job, including successful publications.
    /// The engine must reserve a readback slot before issuing any jobs.
    pub fn complete_jobs(&mut self, results: impl IntoIterator<Item = (u64, u32)>) {
        let mut failed = Vec::new();
        for (key, status) in results {
            let Some(publication) = self.publishing.remove(&key) else { continue };
            if publication.evicted {
                if let Some(block) = publication.previous { self.edits.release(block); }
                if let Some(block) = publication.next { self.edits.release(block); }
                continue;
            }
            let resident = self.residents.get_mut(key).expect("live publication has a resident");
            assert_eq!(resident.record, publication.record);
            if status == 0 {
                resident.edit_block = publication.next;
                if let Some(block) = publication.previous { self.edits.release(block); }
            } else {
                // A failed replacement leaves the previous GPU column intact.
                if let Some(block) = publication.next { self.edits.release(block); }
                if let Some(bucket) = publication.initial_bucket {
                    if status != 1 {
                        self.initial_retries.insert(key);
                        self.levels[unpack(key).1 as usize].pending.insert(key, bucket);
                        self.stats.requeued += 1;
                    }
                } else {
                    failed.push((key, status));
                }
            }
        }
        self.requeue(failed);
        self.stats.pending_columns = self.levels.iter().map(|l| l.pending.len()).sum::<usize>()
            + self.urgent.len() + self.publishing.len();
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

    /// An edit journal or its publication is still changing GPU columns.
    /// Initial unedited column publications do not invalidate summaries.
    pub fn pending_edits(&self) -> bool {
        !self.urgent.is_empty()
            || self.publishing.values().any(|p| p.previous.is_some() || p.next.is_some())
    }

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
                for key in l.pending.keys() {
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
            && self.publishing.is_empty()
            && self.applied == self.requested
            && self.diffs.iter().all(VecDeque::is_empty)
            && self.levels.iter().all(|l| l.pending.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planet::PlanetRecipe;

    #[test]
    fn latest_frontier_bypasses_obsolete_diff_backlog_and_converges() {
        let planet = Planet::new(PlanetRecipe { shape: crate::grid::Shape::Plane, ..Default::default() }).unwrap();
        let mut r = Residency::new(*planet.grid(), Capacity { table_bits: 8, ..Default::default() });
        let keys = |start, len| (start..start + len).map(|i| pack(key0(crate::grid::PLANE_FACE, 0, i), 0)).collect::<Vec<_>>();
        let old = keys(1000, 1024);
        let latest = keys(3000, 16);
        for (serial, demand) in [(1, &old), (2, &latest)] {
            r.apply(WindowUpdate { serial,
                levels: vec![LevelDiff { level: 0, active: true,
                    adds: demand.iter().map(|&key| (0.1, key)).collect(),
                    removes: if serial == 2 { old.clone() } else { vec![] }, ..Default::default() }],
                wanted: vec![(0, std::sync::Arc::new(demand.iter().copied().collect()))],
                ..Default::default() });
        }
        let mut work = FrameWork::default();
        r.apply_queued(&mut work, &|| true);
        assert!(latest.iter().all(|key| r.levels[0].pending.at.contains_key(key)),
            "new visible demand must reach the first bounded slice");
        assert!(old.iter().all(|key| !r.levels[0].pending.at.contains_key(key)),
            "obsolete additions must never enter generation");
        assert!(r.catching_up[0] > 0, "partial retirement cannot claim complete coverage");
        r.apply_queued(&mut work, &|| false);
        assert_eq!(r.levels[0].pending.keys().copied().collect::<FxHashSet<_>>(), latest.into_iter().collect());
        assert_eq!(r.catching_up[0], 0);
    }

    #[test]
    fn returned_window_preserves_resident_and_pending_columns_during_old_retirement() {
        let (_, mut r, resident, _) = edit_fixture();
        let (_, level, i, j) = unpack(resident);
        let pending = pack(key0(crate::grid::PLANE_FACE, level, i + 1), j as u32);
        r.levels[level as usize].pending.insert(pending, 1);
        let wanted = std::sync::Arc::new([resident, pending].into_iter().collect::<FxHashSet<_>>());
        r.apply(WindowUpdate { serial: 1,
            levels: vec![LevelDiff { level, active: false, removes: vec![resident, pending], ..Default::default() }],
            wanted: vec![(level, Default::default())], ..Default::default() });
        r.apply(WindowUpdate { serial: 2,
            levels: vec![LevelDiff { level, active: true, adds: vec![(0.1, resident), (0.1, pending)], ..Default::default() }],
            wanted: vec![(level, wanted)], ..Default::default() });
        let mut work = FrameWork::default();
        r.apply_queued(&mut work, &|| false);
        assert!(r.residents.contains_key(resident), "old retirement cannot evict a returned resident");
        assert!(r.levels[level as usize].pending.at.contains_key(&pending), "old inactive clear cannot erase current demand");
        assert!(work.evictions.is_empty());
        assert_eq!(r.catching_up[level as usize], 0);
    }

    #[test]
    fn queued_column_moves_ahead_when_its_priority_changes() {
        let mut queue = PendingQueue::default();
        queue.insert(10, 40);
        queue.insert(20, 4);
        queue.insert(30, 40);
        assert!(queue.insert(10, 0));
        assert_eq!(queue.len(), 3);
        assert_eq!(queue.pop(), Some((10, 0)));
        assert_eq!(queue.pop(), Some((20, 4)));
        assert_eq!(queue.pop(), Some((30, 40)));
        assert!(queue.is_empty());
    }

    #[test]
    fn moving_camera_refreshes_surviving_pending_columns() {
        let planet = std::sync::Arc::new(Planet::new(PlanetRecipe {
            shape: crate::grid::Shape::Plane,
            plane_size_m: 1000.0,
            terrain: crate::TerrainSource { generator: crate::landform::FLAT_ID.into(), ..Default::default() },
            ..Default::default()
        }).unwrap());
        let mut residency = Residency::new(*planet.grid(), Capacity::default());
        let first_eye = DVec3::new(0.0, 2.0, 0.0);
        residency.plan(&planet, first_eye, 10.0, 0);
        let old = residency.levels[0].pending.at.clone();
        let moved_eye = first_eye + DVec3::X * 8.0;
        residency.plan(&planet, moved_eye, 10.0, 0);
        let promoted: Vec<_> = residency.levels[0].pending.at.iter()
            .filter(|(key, (bucket, _))| old.get(key).is_some_and(|(previous, _)| previous > bucket))
            .collect();
        assert!(!promoted.is_empty(), "overlapping jobs closer to the moving camera must be promoted");
        assert!(promoted.iter().all(|(key, _)| !residency.residents.contains_key(**key)),
            "reprioritization must preserve pending work without publishing unfinished columns");
    }

    #[test]
    fn retired_inflight_column_does_not_block_unrelated_admissions() {
        let (planet, mut residency, blocked, eye) = edit_fixture();
        residency.residents.remove(blocked, &mut Vec::new());
        residency.publishing.insert(blocked, EditPublication {
            record: 0, previous: None, next: None, evicted: true,
            initial_bucket: Some(0),
        });
        let (_, level, i, j) = unpack(blocked);
        let available = pack(key0(crate::grid::PLANE_FACE, level, i + 1), j as u32);
        residency.levels[level as usize].pending.insert(available, 1);
        residency.levels[level as usize].pending.insert(blocked, 0);
        let work = residency.plan(&planet, eye, 1.0, 2);
        assert_eq!(work.job_keys, vec![available]);
        assert!(residency.levels[level as usize].pending.at.contains_key(&blocked));
        assert_eq!(residency.publishing[&blocked].record, 0,
            "retired publication must retain its incarnation until completion");
    }

    #[test]
    fn active_window_hands_off_incoming_columns_before_retirement_finishes() {
        let planet = Planet::new(PlanetRecipe { shape: crate::grid::Shape::Plane, ..Default::default() }).unwrap();
        let mut residency = Residency::new(*planet.grid(), Capacity::default());
        let face = crate::grid::PLANE_FACE;
        let removed: Vec<_> = (1000..1512).map(|i| pack(key0(face, 0, i), 0)).collect();
        let added: Vec<_> = (2000..2512).map(|i| pack(key0(face, 0, i), 0)).collect();
        for &key in &removed { residency.levels[0].pending.insert(key, 40); }
        residency.apply(WindowUpdate { serial: 1, levels: vec![LevelDiff {
            level: 0, active: true, radius: 100.0,
            removes: removed.clone(), adds: added.iter().map(|key| (0.1, *key)).collect(),
            ..Default::default()
        }], ..Default::default() });
        let mut work = FrameWork::default();
        residency.apply_queued(&mut work, &|| true);
        let diff = residency.diffs[0].front().unwrap();
        assert_eq!((diff.removed, diff.added), (128, 128),
            "one bounded256-operation slice must expose incoming demand alongside retirement");
        assert!(added[..128].iter().all(|key| residency.levels[0].pending.at.contains_key(key)));
        assert!(removed[128..].iter().all(|key| residency.levels[0].pending.at.contains_key(key)));
        assert_eq!(residency.catching_up[0], 1, "partial handoff must not claim complete coverage");
        residency.apply_queued(&mut work, &|| false);
        let wanted: std::collections::HashSet<_> = residency.levels[0].pending.keys().copied().collect();
        assert_eq!(wanted, added.into_iter().collect());
        assert_eq!(residency.catching_up[0], 0);
        assert!(residency.diffs.iter().all(VecDeque::is_empty) && work.jobs.is_empty() && residency.publishing.is_empty());
    }

    #[test]
    fn diff_rounds_admit_coarser_blocks_without_overtaking_same_level_fifo() {
        let planet = Planet::new(PlanetRecipe { shape: crate::grid::Shape::Plane, ..Default::default() }).unwrap();
        let mut residency = Residency::new(*planet.grid(), Capacity::default());
        assert!(residency.grid.levels() > 2);
        let face = crate::grid::PLANE_FACE;
        let removed: Vec<_> = (1000..2024).map(|i| pack(key0(face, 0, i), 0)).collect();
        let added: Vec<_> = (3000..4024).map(|i| pack(key0(face, 0, i), 0)).collect();
        let coarser: Vec<_> = (8..12).flat_map(|j| (8..12).map(move |i| pack(key0(face, 1, i), j))).collect();
        let later = pack(key0(face, 0, 5000), 0);
        for &key in &removed { residency.levels[0].pending.insert(key, 40); }
        residency.apply(WindowUpdate { serial: 1, levels: vec![
            LevelDiff { level: 0, active: true, removes: removed, adds: added.iter().map(|key| (0.1, *key)).collect(), ..Default::default() },
            LevelDiff { level: 1, active: true, adds: coarser.iter().map(|key| (0.1, *key)).collect(), ..Default::default() },
        ], ..Default::default() });
        residency.apply(WindowUpdate { serial: 2, levels: vec![LevelDiff {
            level: 0, active: true, removes: vec![added[0]], adds: vec![(0.1, later)], ..Default::default()
        }], ..Default::default() });
        let rounds = std::cell::Cell::new(0);
        let mut work = FrameWork::default();
        residency.apply_queued(&mut work, &|| {
            rounds.set(rounds.get() + 1);
            rounds.get() == 2
        });
        assert_eq!(rounds.get(), 2);
        let first = residency.diffs[0].front().unwrap();
        assert_eq!((first.removed, first.added), (128, 128));
        assert!(coarser.iter().all(|key| residency.levels[1].pending.at.contains_key(key)),
            "a complete visible L1 block must reach generation before the large L0 delta drains");
        assert_eq!(residency.catching_up[1], 0);
        assert_eq!(residency.catching_up[0], 2);
        let second = &residency.diffs[0][1];
        assert_eq!((second.removed, second.added), (0, 0));
        assert!(residency.levels[0].pending.at.contains_key(&added[0]));
        assert!(!residency.levels[0].pending.at.contains_key(&later));
        residency.apply_queued(&mut work, &|| false);
        let expected: std::collections::HashSet<_> = added.into_iter().skip(1).chain([later]).collect();
        assert_eq!(residency.levels[0].pending.keys().copied().collect::<std::collections::HashSet<_>>(), expected);
        assert!(residency.catching_up.iter().all(|count| *count == 0));
        assert!(residency.diffs.iter().all(VecDeque::is_empty));
        assert!(work.jobs.is_empty() && residency.publishing.is_empty());
    }

    #[test]
    fn global_diff_priority_preserves_inactive_clear_and_later_activation() {
        let planet = Planet::new(PlanetRecipe { shape: crate::grid::Shape::Plane, ..Default::default() }).unwrap();
        let mut residency = Residency::new(*planet.grid(), Capacity::default());
        let top = residency.grid.levels() - 1;
        assert!(top > 1);
        let face = crate::grid::PLANE_FACE;
        let removed: Vec<_> = (1000..1512).map(|i| pack(key0(face, 0, i), 0)).collect();
        let stale = pack(key0(face, 0, 2000), 0);
        let block = |level| (8..12).flat_map(move |j| (8..12).map(move |i| pack(key0(face, level, i), j))).collect::<Vec<_>>();
        let global = block(top);
        let coarser = block(1);
        let incoming = block(0);
        for &key in &removed { residency.levels[0].pending.insert(key, 40); }
        residency.levels[0].pending.insert(stale, 40);
        residency.apply(WindowUpdate { serial: 1, levels: vec![
            LevelDiff { level: 0, active: false, removes: removed.clone(), ..Default::default() },
            LevelDiff { level: 1, active: true, adds: coarser.iter().map(|key| (0.1, *key)).collect(), ..Default::default() },
            LevelDiff { level: top, active: true, adds: global.iter().map(|key| (0.1, *key)).collect(), ..Default::default() },
        ], ..Default::default() });
        residency.apply(WindowUpdate { serial: 2, levels: vec![LevelDiff {
            level: 0, active: true, adds: incoming.iter().map(|key| (0.1, *key)).collect(), ..Default::default()
        }], ..Default::default() });
        let mut work = FrameWork::default();
        residency.apply_queued(&mut work, &|| true);
        assert!(global.iter().all(|key| residency.levels[top as usize].pending.at.contains_key(key)));
        assert_eq!(residency.catching_up[top as usize], 0);
        assert_eq!(residency.diffs[0].front().unwrap().removed, 0,
            "global coverage keeps priority over fine retirement");
        assert!(residency.levels[1].pending.at.is_empty());
        let rounds = std::cell::Cell::new(0);
        residency.apply_queued(&mut work, &|| {
            rounds.set(rounds.get() + 1);
            rounds.get() == 2
        });
        let retiring = residency.diffs[0].front().unwrap();
        assert_eq!(retiring.removed, 256);
        assert!(!retiring.cleared && residency.levels[0].pending.at.contains_key(&stale));
        assert!(removed[256..].iter().all(|key| residency.levels[0].pending.at.contains_key(key)));
        assert!(coarser.iter().all(|key| residency.levels[1].pending.at.contains_key(key)));
        assert_eq!(residency.diffs[0][1].added, 0,
            "a later activation may not overtake its level's unfinished retirement");
        residency.apply_queued(&mut work, &|| false);
        assert_eq!(residency.levels[0].pending.keys().copied().collect::<std::collections::HashSet<_>>(), incoming.into_iter().collect());
        assert_eq!(residency.levels[1].pending.keys().copied().collect::<std::collections::HashSet<_>>(), coarser.into_iter().collect());
        assert!(residency.catching_up.iter().all(|count| *count == 0));
        assert!(residency.diffs.iter().all(VecDeque::is_empty));
        assert!(work.jobs.is_empty() && residency.publishing.is_empty());
    }

    #[test]
    fn grouped_retirement_releases_alias_owners_before_the_level_delta_drains() {
        let planet = std::sync::Arc::new(Planet::new(PlanetRecipe {
            shape: crate::grid::Shape::Plane, ..Default::default()
        }).unwrap());
        let mut residency = Residency::new(*planet.grid(), Capacity::default());
        let face = crate::grid::PLANE_FACE;
        // Each wanted4x4 block is in a different tier-3 owner. This models
        // a clipped outgoing fringe: owners have only their resident refs.
        let mut removed = Vec::new();
        for bj in 0..8 {
            for bi in 0..8 {
                for j in bj * 64..bj * 64 + 4 {
                    for i in bi * 64..bi * 64 + 4 {
                        removed.push(pack(key0(face, 0, i), j as u32));
                    }
                }
            }
        }
        removed = crate::windows::order_removes_by_blocks(removed);
        let mut setup = FrameWork::default();
        for (record, &key) in removed.iter().enumerate() {
            assert!(residency.acquire_blocks(key, &mut setup));
            residency.residents.insert(key, Resident {
                record: record as u32, slot: 0, edit_block: None, blocks: true,
            });
        }
        residency.next_record = removed.len() as u32;
        let retired_owners: Vec<_> = removed[..128].chunks(16).map(|block| block[0]).collect();
        assert_eq!(retired_owners.len(), 8);
        let incoming: Vec<_> = removed[..16].iter().map(|&key| {
            let (face, level, i, j) = unpack(key);
            pack(key0(face, level, i + 512), j as u32)
        }).collect();
        assert!(incoming.iter().all(|&key| residency.blocks_conflict(key)));
        residency.apply(WindowUpdate { serial: 1, levels: vec![LevelDiff {
            level: 0, active: true, radius: 100.0,
            removes: removed.clone(), adds: incoming.iter().map(|key| (0.0, *key)).collect(),
            ..Default::default()
        }], ..Default::default() });
        let mut retirement = FrameWork::default();
        residency.apply_queued(&mut retirement, &|| true);
        assert_eq!(retirement.evictions.len(), 128);
        assert_eq!(residency.residents.len(), removed.len() - 128);
        for key in retired_owners {
            let (face, level, i, j) = unpack(key);
            for tier in 1..=BLOCK_TIERS {
                assert!(!residency.blocks.contains_key(&(level, face, tier, i >> (2 * tier), j >> (2 * tier))));
            }
        }
        assert!(incoming.iter().all(|&key| !residency.blocks_conflict(key)));
        let eye = DVec3::Y * 30.0;
        // Keep this isolated delta as the planner's current request, then
        // give admission its normal deadline and fixed16-job GPU allowance.
        residency.requested = 1;
        residency.last_request = Some(WindowRequest {
            eye, prefetch_eye: None, priority_eye: None, view_focus: None,
            lod0: 120.0, outer_radius: planet.outer_radius(), planet: Some(planet.clone()), serial: 1,
        });
        residency.set_cpu_budget(Some(std::time::Duration::ZERO));
        let work = residency.plan(&planet, eye, 120.0, 16);
        assert_eq!(work.job_keys.len(), incoming.len());
        assert_eq!(work.job_keys.iter().copied().collect::<std::collections::HashSet<_>>(), incoming.iter().copied().collect());
        assert!(work.job_keys.iter().all(|&key| residency.residents.get(key).unwrap().blocks),
            "incoming columns must acquire summaries before the remaining outgoing owners retire");
        assert_eq!(residency.catching_up[0], 1);
        assert!(!residency.diffs[0].is_empty() && residency.residents.len() > work.jobs.len());
        residency.complete_jobs(work.job_keys.iter().map(|&key| (key, 0)));
        residency.apply_queued(&mut FrameWork::default(), &|| false);
        assert_eq!(residency.catching_up[0], 0);
        assert_eq!(residency.residents.len(), incoming.len());
        assert!(residency.blocks_exact() && residency.publishing.is_empty());
    }

    #[test]
    fn early_incoming_summary_alias_waits_for_retirement_without_blocking_other_jobs() {
        let (planet, mut residency, old, eye) = edit_fixture();
        let mut work = FrameWork::default();
        assert!(residency.acquire_blocks(old, &mut work));
        residency.residents.get_mut(old).unwrap().blocks = true;
        let (face, level, i, j) = unpack(old);
        let incoming = pack(key0(face, level, i + 512), j as u32);
        let available = pack(key0(face, level, i + 80), j as u32);
        assert!(residency.blocks_conflict(incoming));
        assert!(!residency.blocks_conflict(available));
        residency.catching_up[level as usize] = 1;
        residency.levels[level as usize].pending.insert(incoming, 0);
        residency.levels[level as usize].pending.insert(available, 1);
        let work = residency.plan(&planet, eye, 1.0, 2);
        assert_eq!(work.job_keys, vec![available]);
        assert!(!residency.residents.contains_key(incoming));
        assert!(residency.levels[level as usize].pending.at.contains_key(&incoming));
        let mut retirement = FrameWork::default();
        residency.evict(old, &mut retirement);
        residency.catching_up[level as usize] = 0;
        let work = residency.plan(&planet, eye, 1.0, 1);
        assert_eq!(work.job_keys, vec![incoming]);
        assert!(residency.residents.get(incoming).unwrap().blocks,
            "early staging may not permanently publish a summaryless incoming column");
    }

    #[test]
    fn summary_check_revalidates_interleaved_full_keys_and_resets_after_retirement() {
        let (_, mut r, old, _) = edit_fixture();
        r.evict(old, &mut FrameWork::default());
        let (face, level, i, j) = unpack(old);
        let a = pack(key0(face, level, i & !3), (j & !3) as u32);
        let alias = pack(key0(face, level, (i & !3) + 512), (j & !3) as u32);
        let mut last = None;
        let mut work = FrameWork::default();
        // First inspect a free owner, then let its toroidal alias claim it.
        assert!(!r.blocks_conflict_cached(a, &mut last));
        assert!(!r.blocks_conflict_cached(alias, &mut last));
        r.reference_blocks(alias, &mut work);
        assert!(r.blocks_conflict_cached(a, &mut last), "a cached free slot cannot survive an intervening alias admission");
        // Clearing two low index bits must retain signed j, face and level.
        let different_level = pack(key0(face, level + 1, i & !3), (j & !3) as u32);
        assert!(!r.blocks_conflict_cached(different_level, &mut last));
        assert!(r.blocks_conflict_cached(a, &mut last));
        let negative = pack(key0(face, level + 1, i & !3), (-4i32) as u32);
        let positive_alias = pack(key0(face, level + 1, i & !3), 508);
        assert!(!r.blocks_conflict_cached(negative, &mut last));
        r.reference_blocks(negative, &mut work);
        assert!(r.blocks_conflict_cached(positive_alias, &mut last), "signed j and its positive toroidal alias must retain different identities");
        r.release_blocks(negative, &mut work);
        r.release_blocks(alias, &mut work);
        // Each plan owns a new cache, after retirement has completed.
        let mut next_plan = None;
        assert!(!r.blocks_conflict_cached(a, &mut next_plan));
        assert!(r.block_owner.is_empty() && r.blocks.is_empty());
    }

    #[test]
    fn cached_block_admission_defers_aliases_without_duplicating_summary_refs() {
        let (planet, mut r, old, eye) = edit_fixture();
        r.evict(old, &mut FrameWork::default());
        let (face, level, i, j) = unpack(old);
        let key = |offset| pack(key0(face, level, (i & !3) + offset), (j & !3) as u32);
        let a = key(0);
        let alias = key(512);
        let b = key(64);
        queue_complete_block(&mut r.levels[level as usize].pending, a, 0);
        queue_complete_block(&mut r.levels[level as usize].pending, alias, 1);
        queue_complete_block(&mut r.levels[level as usize].pending, b, 2);
        r.catching_up[level as usize] = 1;
        let work = r.plan(&planet, eye, 1.0, 48);
        assert_eq!(work.jobs.len(), 32);
        assert_eq!(r.levels[level as usize].pending.len(), 16, "only the blocked owner must carry over");
        assert!(work.job_keys.iter().all(|&key| r.residents.get(key).unwrap().blocks));
        for (&owner, block) in &r.blocks {
            let expected = work.job_keys.iter().filter(|&&key| {
                let (face, level, i, j) = unpack(key);
                owner == (level, face, owner.2, i >> (2 * owner.2), j >> (2 * owner.2))
            }).count() as u32;
            assert_eq!(block.refs, expected, "summary references must count admitted columns exactly");
            assert_eq!(r.block_owner[&block.slot], owner);
        }
        r.complete_jobs(work.job_keys.iter().map(|&key| (key, 0)));
        for &key in &work.job_keys {
            if block_identity(key) == block_identity(a) { r.evict(key, &mut FrameWork::default()); }
        }
        let next = r.plan(&planet, eye, 1.0, 16);
        assert_eq!(next.jobs.len(), 16, "retired owners must become admissible in the next plan");
        assert!(next.job_keys.iter().all(|&key| block_identity(key) == block_identity(alias)));
        assert!(next.job_keys.iter().all(|&key| r.residents.get(key).unwrap().blocks));
    }

    fn block_identity(key: u64) -> (u8, u32, i32, i32) {
        let (face, level, i, j) = unpack(key);
        (face, level, i >> 2, j >> 2)
    }

    fn queue_complete_block(pending: &mut PendingQueue, key: u64, bucket: usize) {
        let (face, level, bi, bj) = block_identity(key);
        for j in bj * 4..bj * 4 + 4 {
            for i in bi * 4..bi * 4 + 4 {
                pending.insert(pack(key0(face, level, i), j as u32), bucket);
            }
        }
    }

    #[test]
    fn current_camera_jobs_precede_equivalent_forecast_jobs() {
        let planet = Planet::new(PlanetRecipe { shape: crate::grid::Shape::Plane, ..Default::default() }).unwrap();
        let grid = *planet.grid();
        let eye = DVec3::Y * 2.0;
        let future = eye + DVec3::X * 20.0;
        let key_at = |point| {
            let (cell, _) = grid.locate(point);
            pack(key0(cell.face, 0, cell.i >> 3), (cell.j >> 3) as u32)
        };
        let current_key = key_at(eye);
        let forecast_key = key_at(future);
        let mut residency = Residency::new(grid, Capacity::default());
        residency.set_prefetch_eye(Some(future));
        let level = &mut residency.levels[0];
        level.active = true;
        level.radius = 100.0;
        queue_complete_block(&mut level.pending, current_key, 40);
        queue_complete_block(&mut level.pending, forecast_key, 40);
        residency.refresh_near_pending(eye, None);
        let pending = &mut residency.levels[0].pending;
        let current_priority = pending.at[&current_key].0;
        let forecast_priority = pending.at[&forecast_key].0;
        assert!(current_priority < forecast_priority && forecast_priority < 40);
        for key in [current_key, forecast_key] {
            for _ in 0..16 { assert_eq!(block_identity(pending.pop().unwrap().0), block_identity(key)); }
        }
    }

    #[test]
    fn priority_forecast_does_not_replan_or_expand_windows() {
        let planet = std::sync::Arc::new(Planet::new(PlanetRecipe {
            shape: crate::grid::Shape::Plane, plane_size_m: 1000.0,
            terrain: crate::TerrainSource { generator: crate::landform::FLAT_ID.into(), ..Default::default() },
            ..Default::default()
        }).unwrap());
        let mut residency = Residency::new(*planet.grid(), Capacity::default());
        let eye = DVec3::Y * 30.0;
        let coverage = eye + DVec3::X * 21.0;
        residency.set_prefetch_eye(Some(coverage));
        residency.plan(&planet, eye, 120.0, 0);
        let serial = residency.requested;
        let pending: Vec<_> = residency.levels.iter().map(|level| level.pending.keys().copied().collect::<std::collections::HashSet<_>>()).collect();
        residency.set_priority_eye(Some(eye + DVec3::X * 60.0));
        residency.plan(&planet, eye, 120.0, 0);
        assert_eq!(residency.requested, serial);
        assert_eq!(residency.last_request.as_ref().unwrap().prefetch_eye, Some(coverage));
        for (level, before) in residency.levels.iter().zip(pending) {
            assert_eq!(level.pending.keys().copied().collect::<std::collections::HashSet<_>>(), before);
        }
        assert!(residency.residents.len() == 0 && residency.publishing.is_empty());
        let wanted: Vec<_> = residency.levels.iter().map(|level| level.pending.keys().copied().collect::<std::collections::HashSet<_>>()).collect();
        let pitch = -12.0f64.to_radians();
        let focus = crate::windows::visible_focus(planet.grid(), eye, DVec3::new(pitch.cos(), pitch.sin(), 0.0), 0.0, 1000.0).unwrap();
        residency.set_view_focus(Some(focus));
        residency.plan(&planet, eye, 120.0, 0);
        assert_eq!(residency.requested, serial, "distant visible focus must not request a wider window");
        for (level, before) in residency.levels.iter().zip(wanted) {
            assert_eq!(level.pending.keys().copied().collect::<std::collections::HashSet<_>>(), before);
        }
        assert!(residency.residents.len() == 0 && residency.publishing.is_empty());
    }

    #[test]
    fn visible_forward_focus_promotes_existing_pending_work_without_publication() {
        let planet = Planet::new(PlanetRecipe { shape: crate::grid::Shape::Plane, ..Default::default() }).unwrap();
        let grid = *planet.grid();
        let eye = DVec3::Y * 30.0;
        let pitch = -0.45f64;
        let focus = crate::windows::visible_focus(&grid, eye, DVec3::new(pitch.cos(), pitch.sin(), 0.0), 0.0, 200.0).unwrap();
        let forecast = eye + DVec3::X * 43.0;
        let key_at = |point| {
            let (cell, _) = grid.locate(point);
            pack(key0(cell.face, 0, cell.i >> 3), (cell.j >> 3) as u32)
        };
        let current_key = key_at(eye);
        let focus_key = key_at(focus);
        let forecast_key = key_at(forecast);
        let behind_key = key_at(eye - DVec3::X * focus.x);
        let mut residency = Residency::new(grid, Capacity::default());
        residency.set_prefetch_eye(Some(forecast));
        residency.set_view_focus(Some(focus));
        let level = &mut residency.levels[0];
        level.active = true;
        level.radius = 100.0;
        for key in [current_key, focus_key, forecast_key, behind_key] { queue_complete_block(&mut level.pending, key, 50); }
        residency.refresh_near_pending(eye, None);
        assert_eq!(residency.residents.len(), 0);
        assert!(residency.publishing.is_empty());
        let pending = &mut residency.levels[0].pending;
        assert_eq!(pending.len(), 64, "focus must not enqueue data outside existing pending windows");
        for key in [current_key, focus_key, forecast_key, behind_key] {
            for _ in 0..16 { assert_eq!(block_identity(pending.pop().unwrap().0), block_identity(key)); }
        }
    }

    #[test]
    fn incomplete_block_is_not_promoted_and_complete_blocks_stay_together() {
        let planet = Planet::new(PlanetRecipe { shape: crate::grid::Shape::Plane, ..Default::default() }).unwrap();
        let grid = *planet.grid();
        let eye = DVec3::Y * 2.0;
        let (cell, _) = grid.locate(eye);
        let ci = cell.i >> 3;
        let cj = cell.j >> 3;
        let key = pack(key0(cell.face, 0, ci), cj as u32);
        let incomplete_key = pack(key0(cell.face, 0, (ci & !3) - 4), (cj & !3) as u32);
        let mut residency = Residency::new(grid, Capacity::default());
        let level = &mut residency.levels[0];
        level.active = true;
        level.radius = 100.0;
        queue_complete_block(&mut level.pending, key, 40);
        queue_complete_block(&mut level.pending, incomplete_key, 40);
        level.pending.remove(incomplete_key);
        let before: std::collections::HashSet<_> = level.pending.keys().copied().collect();
        residency.refresh_near_pending(eye, None);
        let pending = &mut residency.levels[0].pending;
        assert_eq!(before, pending.keys().copied().collect());
        for (&candidate, &(bucket, _)) in &pending.at {
            if block_identity(candidate) == block_identity(incomplete_key) { assert_eq!(bucket, 40); }
        }
        for _ in 0..16 { assert_eq!(block_identity(pending.pop().unwrap().0), block_identity(key)); }
        assert_eq!(pending.len(), 15);
        assert_eq!(residency.residents.len(), 0);
        assert!(residency.publishing.is_empty());
    }

    #[test]
    fn expired_refresh_reserve_keeps_pending_queue_and_admission_work_intact() {
        let planet = Planet::new(PlanetRecipe { shape: crate::grid::Shape::Plane, ..Default::default() }).unwrap();
        let grid = *planet.grid();
        let eye = DVec3::Y * 2.0;
        let (cell, _) = grid.locate(eye);
        let key = pack(key0(cell.face, 0, cell.i >> 3), (cell.j >> 3) as u32);
        let mut residency = Residency::new(grid, Capacity::default());
        let level = &mut residency.levels[0];
        level.active = true;
        level.radius = 100.0;
        queue_complete_block(&mut level.pending, key, 40);
        let before = level.pending.at.clone();
        residency.refresh_near_pending(eye, Some(std::time::Instant::now()));
        assert_eq!(residency.levels[0].pending.at, before,
            "elapsed refresh reserve must leave existing work ready for admission");
        assert!(residency.levels[0].pending.pop().is_some());
        assert!(residency.residents.len() == 0 && residency.publishing.is_empty());
    }

    fn edit_fixture() -> (std::sync::Arc<Planet>, Residency, u64, DVec3) {
        let planet = std::sync::Arc::new(Planet::new(PlanetRecipe {
            shape: crate::grid::Shape::Plane,
            ..Default::default()
        }).unwrap());
        let grid = *planet.grid();
        let eye = DVec3::new(0.0, 10.0, 0.0);
        let (cell, _) = grid.locate(DVec3::ZERO);
        let key = pack(key0(cell.face, 0, cell.i >> 3), (cell.j >> 3) as u32);
        let mut r = Residency::new(grid, Capacity { table_bits: 8, edit_words: 64, ..Default::default() });
        r.residents.insert(key, Resident { record: 0, slot: 0, edit_block: None, blocks: false });
        r.block_conflicts = 1;
        r.next_record = 1;
        r.last_request = Some(WindowRequest {
            eye, prefetch_eye: None, priority_eye: None, view_focus: None, lod0: 1.0,
            outer_radius: planet.outer_radius(), planet: Some(planet.clone()), serial: 1,
        });
        (planet, r, key, eye)
    }

    fn test_brush(radius: f64) -> crate::edits::Brush {
        crate::edits::Brush {
            center: [0.0, 0.0, 0.0], radius,
            shape: crate::edits::BrushShape::Sphere,
            op: crate::edits::BrushOp::Add, material: 2,
        }
    }

    #[test]
    fn large_brush_and_undo_invalidate_resident_fine_columns() {
        let (mut planet, mut r, key, _) = edit_fixture();
        std::sync::Arc::make_mut(&mut planet).apply(test_brush(1_000.0)).unwrap();
        let mut work = FrameWork::default();
        r.sync_edits(&planet, &mut work);
        assert!(r.urgent.contains(&key), "large footprint must not skip fine residents");
        r.urgent.clear();
        std::sync::Arc::make_mut(&mut planet).undo().unwrap();
        r.sync_edits(&planet, &mut work);
        assert!(r.urgent.contains(&key), "undo must rebuild the same resident footprint");
    }

    #[test]
    fn failed_edit_publication_preserves_refs_and_churn_reclaims_them() {
        let (mut planet, mut r, key, eye) = edit_fixture();
        assert!(!r.pending_edits());
        for round in 0..100 {
            std::sync::Arc::make_mut(&mut planet).apply(test_brush(0.2)).unwrap();
            // Avoid changing windows: only test edit publication ownership.
            r.last_request.as_mut().unwrap().outer_radius = planet.outer_radius();
            let work = r.plan(&planet, eye, 1.0, 1);
            assert_eq!(work.job_keys, vec![key]);
            let next = r.publishing[&key].next.unwrap();
            assert!(r.pending_edits());
            assert_eq!(r.residents.get(key).unwrap().edit_block, None);
            assert!(!r.edits.free[next.1 as usize].contains(&next.0));
            r.complete_jobs([(key, 2)]);
            assert!(r.pending_edits(), "failed edit must remain pending for retry");
            assert_eq!(r.residents.get(key).unwrap().edit_block, None);
            assert!(r.edits.free[next.1 as usize].contains(&next.0));
            let retry = r.plan(&planet, eye, 1.0, 1);
            assert_eq!(retry.job_keys, vec![key]);
            r.complete_jobs([(key, 0)]);
            let published = r.residents.get(key).unwrap().edit_block.unwrap();
            std::sync::Arc::make_mut(&mut planet).undo().unwrap();
            let undo = r.plan(&planet, eye, 1.0, 1);
            assert_eq!(undo.job_keys, vec![key]);
            assert!(r.pending_edits(), "undo must retain old edit ownership until publication");
            assert_eq!(r.residents.get(key).unwrap().edit_block, Some(published));
            assert!(!r.edits.free[published.1 as usize].contains(&published.0));
            // A repeated plan must not issue the same key before its result.
            let waiting = r.plan(&planet, eye, 1.0, 1);
            assert!(waiting.jobs.is_empty());
            r.complete_jobs([(key, 2)]);
            assert!(r.pending_edits(), "failed edit must remain pending for retry");
            assert_eq!(r.residents.get(key).unwrap().edit_block, Some(published));
            let retry = r.plan(&planet, eye, 1.0, 1);
            assert_eq!(retry.job_keys, vec![key]);
            r.complete_jobs([(key, 0)]);
            assert_eq!(r.residents.get(key).unwrap().edit_block, None);
            assert!(r.edits.free[published.1 as usize].contains(&published.0));
            assert!(r.publishing.is_empty());
            assert!(!r.pending_edits());
            assert_eq!(r.edits.top, 2, "edit refs leaked on round {round}");
        }
    }

    #[test]
    fn evicted_publication_reclaims_both_journals_without_touching_reused_record() {
        let (mut planet, mut r, key, eye) = edit_fixture();
        std::sync::Arc::make_mut(&mut planet).apply(test_brush(0.2)).unwrap();
        r.last_request.as_mut().unwrap().outer_radius = planet.outer_radius();
        r.plan(&planet, eye, 1.0, 1);
        r.complete_jobs([(key, 0)]);
        let previous = r.residents.get(key).unwrap().edit_block.unwrap();
        std::sync::Arc::make_mut(&mut planet).apply(test_brush(0.1)).unwrap();
        r.plan(&planet, eye, 1.0, 1);
        let next = r.publishing[&key].next.unwrap();
        let mut work = FrameWork::default();
        r.evict(key, &mut work);
        assert!(r.publishing[&key].evicted);
        assert!(r.pending_edits());
        assert!(!r.edits.free[previous.1 as usize].contains(&previous.0));
        assert!(!r.edits.free[next.1 as usize].contains(&next.0));
        let other = pack((key as u32) + 1, (key >> 32) as u32);
        let other_block = r.edits.alloc(2, r.capacity.edit_words).unwrap();
        r.residents.insert(other, Resident { record: 0, slot: 0, edit_block: Some(other_block), blocks: false });
        r.complete_jobs([(key, 0)]);
        assert_eq!(r.residents.get(other).unwrap().edit_block, Some(other_block));
        assert!(!r.edits.free[other_block.1 as usize].contains(&other_block.0));
        assert!(r.edits.free[previous.1 as usize].contains(&previous.0));
        assert!(r.edits.free[next.1 as usize].contains(&next.0));
        assert!(r.publishing.is_empty());
        assert!(!r.pending_edits());
    }

    #[test]
    fn unedited_publication_does_not_mark_edits_pending() {
        let (_, mut r, key, _) = edit_fixture();
        r.publishing.insert(key, EditPublication {
            record: 0, previous: None, next: None, evicted: false,
            initial_bucket: Some(0),
        });
        assert!(!r.pending_edits());
        r.complete_jobs([(key, 0)]);
        assert!(!r.pending_edits());
    }

    #[test]
    fn initial_pool_retry_keeps_visibility_priority_and_summary_ownership() {
        let (planet, mut r, key, eye) = edit_fixture();
        r.evict(key, &mut FrameWork::default());
        r.levels[0].pending.insert(key, BUCKETS - 1);
        let first = r.plan(&planet, eye, 1.0, 1);
        assert_eq!(first.job_keys, vec![key]);
        let record = first.jobs[0].record;
        let refs: Vec<_> = r.blocks.iter().map(|(key, block)| (*key, block.refs)).collect();
        r.complete_jobs([(key, 3)]);
        assert!(r.initial_retries.contains(&key));
        assert!(r.urgent.is_empty());
        assert!(!r.pending_edits(), "pool pressure is not an edit publication");
        assert_eq!(r.levels[0].pending.len(), 1);
        let retry = r.plan(&planet, eye, 1.0, 1);
        assert_eq!(retry.job_keys, vec![key]);
        assert_eq!(retry.jobs[0].record, record);
        assert_eq!(retry.jobs[0].flags, 0);
        assert!(retry.table_writes.is_empty(), "retry retains its table identity");
        assert!(retry.block_inits.is_empty(), "retry must not reacquire summary references");
        assert!(refs.iter().all(|(key, count)| r.blocks[key].refs == *count));
        r.complete_jobs([(key, 0)]);
        assert!(r.initial_retries.is_empty() && r.publishing.is_empty());
        assert!(!r.pending_edits());
    }

    #[test]
    fn urgent_retry_respects_cpu_deadline_and_preserves_edit_ownership() {
        let (mut planet, mut r, key, eye) = edit_fixture();
        std::sync::Arc::make_mut(&mut planet).apply(test_brush(0.2)).unwrap();
        r.last_request.as_mut().unwrap().outer_radius = planet.outer_radius();
        let first = r.plan(&planet, eye, 1.0, 1);
        assert_eq!(first.job_keys, vec![key]);
        r.complete_jobs([(key, 3)]);
        assert!(!r.initial_retries.contains(&key));
        assert!(r.urgent.contains(&key) && r.pending_edits());
        r.set_cpu_budget(Some(std::time::Duration::ZERO));
        let waiting = r.plan(&planet, eye, 1.0, 1);
        assert!(waiting.jobs.is_empty());
        assert!(r.urgent.contains(&key) && r.pending_edits());
        r.set_cpu_budget(None);
        let retry = r.plan(&planet, eye, 1.0, 1);
        assert_eq!(retry.job_keys, vec![key]);
        assert_eq!(retry.jobs[0].flags, 1);
        r.complete_jobs([(key, 0)]);
        assert!(!r.pending_edits());
    }

    #[test]
    fn face_edge_brush_invalidates_both_faces_without_outside_keys() {
        let mut planet = Planet::new(PlanetRecipe { radius_m: 1_000.0, ..Default::default() }).unwrap();
        let grid = *planet.grid();
        let center = grid.ground_point(2, 0.1, f64::from(grid.cells()) * 0.5);
        let brush = crate::edits::Brush { center: center.to_array(), ..test_brush(0.2) };
        let faces = brush.resolve(&grid).unwrap();
        assert!(faces.len() >= 2, "brush must straddle a face edge");
        let mut r = Residency::new(grid, Capacity { table_bits: 8, ..Default::default() });
        let mut expected = Vec::new();
        for (record, fb) in faces.iter().enumerate() {
            let i = (fb.center[0] / 2).clamp(0, grid.cells() - 1) >> 3;
            let j = (fb.center[1] / 2).clamp(0, grid.cells() - 1) >> 3;
            let key = pack(key0(fb.face(), 0, i), j as u32);
            expected.push(key);
            r.residents.insert(key, Resident { record: record as u32, ..Default::default() });
        }
        planet.apply(brush).unwrap();
        r.sync_edits(&planet, &mut FrameWork::default());
        for key in expected { assert!(r.urgent.contains(&key)); }
        assert!(r.urgent.iter().all(|&key| {
            let (_, level, i, j) = unpack(key);
            (0..grid.cells() >> (level + 3)).contains(&i)
                && (0..grid.cells() >> (level + 3)).contains(&j)
        }));
    }

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
        let planet = std::sync::Arc::new(Planet::new(PlanetRecipe::default()).unwrap());
        let grid = *planet.grid();
        let lod0 = Residency::lod_distance(&grid, (22.5f64).to_radians().tan(), 720, 1.0);
        let ground = planet.surface_point(grid.direction(2, 3e7, 4e7), 1.8);
        let high = ground.normalize() * (ground.length() + 8_000.0);
        let settle = |r: &mut Residency, eye: DVec3, worst: &mut f64| {
            for _ in 0..20_000 {
                let started = std::time::Instant::now();
                let work = r.plan(&planet, eye, lod0, 100_000);
                r.complete_jobs(work.job_keys.iter().map(|&key| (key, 0)));
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
        let planet = std::sync::Arc::new(Planet::new(PlanetRecipe::default()).unwrap());
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
                residency.complete_jobs(work.job_keys.iter().map(|&key| (key, 0)));
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
        let planet = std::sync::Arc::new(Planet::new(PlanetRecipe::default()).unwrap());
        let grid = *planet.grid();
        let mut residency = Residency::new(grid, Capacity { table_bits: 20, ..Default::default() });
        let mut eye = planet.surface_point(grid.direction(0, 3e7, 4e7), 2.0);
        for step in 0..40 {
            let work = residency.plan(&planet, eye, 120.0, 20_000);
            residency.complete_jobs(work.job_keys.iter().map(|&key| (key, 0)));
            eye = planet.surface_point(eye + DVec3::new(0.0, 0.0, 70.0 * f64::from(step % 3)), 2.0);
        }
        table_is_exact(&residency);
    }
}
