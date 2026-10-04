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
const VISIBLE_BLOCKS: usize = 64;
const CAMERA_CANDIDATES: usize = 3 * 8 * 8;
// Current footprint plus its departing publications awaiting GPU completion.
const CAMERA_LEASES: usize = 2 * CAMERA_CANDIDATES;
const TEMPORARY_LEASES: usize = VISIBLE_BLOCKS + CAMERA_LEASES;
const VISIBLE_ADMISSION_COLUMNS: usize = TEMPORARY_LEASES * 16;
const VISIBLE_LEASE_FRAMES: u32 = 32;
const VISIBLE_LEASE_TIME: std::time::Duration = std::time::Duration::from_millis(500);

#[derive(Clone, Copy)]
struct VisibleStamp {
    frame: u32,
    view: u32,
    at: std::time::Instant,
}

#[derive(Clone, Copy)]
enum LeaseOrigin {
    Captured(VisibleStamp),
    Camera,
}

#[derive(Clone, Copy)]
struct CameraView {
    forward: DVec3,
    up: DVec3,
    right: DVec3,
    tan_half: [f64; 2],
}

struct VisibleLease {
    origin: LeaseOrigin,
    /// First worker request whose result can supersede this temporary demand.
    serial: u64,
    retiring: bool,
    retired: usize,
    /// Current-frame geometric validation; never changes the captured stamp.
    current_demand_frame: Option<u32>,
}

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

/// Admission only removes pending keys until deferred keys are returned at
/// the end of a plan. While the selected bucket survives, no other level can
/// overtake it. Keep the cache local to that removal-only phase.
fn select_pending_level(levels: &mut [Level], top_level: u32,
    cached: &mut Option<(usize, usize)>) -> Option<(usize, usize)> {
    if let Some((index, bucket)) = *cached {
        if levels[index].pending.best() == Some(bucket) {
            return Some((index, bucket));
        }
    }
    let mut best: Option<(usize, usize, usize)> = None;
    for (index, level) in levels.iter_mut().enumerate() {
        if let Some(bucket) = level.pending.best() {
            let priority = if index as u32 == top_level { 0 } else { bucket + 1 };
            if best.is_none_or(|previous| priority < previous.0) {
                best = Some((priority, index, bucket));
            }
        }
    }
    *cached = best.map(|(_, index, bucket)| (index, bucket));
    *cached
}

/// Keep complete visible blocks in their camera-distance order across levels.
/// Global coverage still wins; an unavailable visible key is consumed once,
/// so it cannot prevent ordinary admission from making progress this plan.
fn pop_pending(levels: &mut [Level], top_level: u32,
    visible: &mut VecDeque<(usize, u64)>, cached: &mut Option<(usize, usize)>,
    deadline: Option<std::time::Instant>)
    -> Option<(usize, u64, usize)> {
    let top = top_level as usize;
    if let Some((key, bucket)) = levels[top].pending.pop() {
        return Some((top, key, bucket));
    }
    let mut skipped = 0usize;
    while !visible.is_empty() {
        if skipped % 64 == 0 && deadline.is_some_and(|at| std::time::Instant::now() >= at) {
            return None;
        }
        let (index, key) = visible.pop_front().unwrap();
        skipped += 1;
        let Some(level) = levels.get_mut(index) else { continue };
        let Some(&(bucket, _)) = level.pending.at.get(&key) else { continue };
        level.pending.remove(key);
        return Some((index, key, bucket as usize));
    }
    let (index, _) = select_pending_level(levels, top_level, cached)?;
    let (key, bucket) = levels[index].pending.pop()?;
    Some((index, key, bucket))
}

/// A full tier-1 block references its sixteen distinct resident columns.
/// Snapshot additions for that exact run cannot queue work unless a failed
/// initial publication is retrying. Partial or reordered runs use the usual
/// per-column gates, and publication in flight remains owned by its record.
fn resident_snapshot_run(adds: &[(f32, u64)], retrying: bool,
    refs: impl FnOnce((u32, u8, u32, i32, i32)) -> Option<u32>) -> usize {
    if retrying || adds.len() < 16 { return 0; }
    let (face, level, i, j) = unpack(adds[0].1);
    if i & 3 != 0 || j & 3 != 0 { return 0; }
    for (index, &(_, key)) in adds[..16].iter().enumerate() {
        let expected = pack(key0(face, level, i + (index % 4) as i32), (j + (index / 4) as i32) as u32);
        if key != expected { return 0; }
    }
    if refs((level, face, 1, i >> 2, j >> 2)) == Some(16) { 16 } else { 0 }
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
    pub requested_serial: u64,
    /// Minimum accepted authority over every grid level.
    pub applied_serial: u64,
    pub fine_applied_serial: u64,
    /// Unavailable when this grid has no levels beyond the fine range.
    pub far_applied_serial: Option<u64>,
    /// Original enqueue to apply latency of the last accepted range chunk.
    pub fine_apply_age_ms: Option<f64>,
    pub far_apply_age_ms: Option<f64>,
    /// Worker compute time of the last accepted chunk in each range.
    pub fine_planning_ms: Option<f64>,
    pub far_planning_ms: Option<f64>,
    /// Current eye's ground distance from the authoritative level-0 centre.
    /// Inactive level 0 has no meaningful centre.
    pub fine_window_lag_m: Option<f64>,
    pub edit_words: u32,
    pub table_load: f32,
    /// Allocated backing storage of queued add/remove delta vectors.
    pub queued_delta_bytes: usize,
    pub queued_delta_ops: usize,
    /// Capacity of latest shared membership sets, in keys.
    pub wanted_key_capacity: usize,
    /// Last plan's selected columns, including guarded deferrals.
    pub admission_attempts: usize,
    pub admission_alias_deferred: usize,
    pub admission_publication_deferred: usize,
    pub admission_batched_columns: usize,
    /// Last plan stage intervals; edit time includes record-return setup.
    /// Authority includes request submission and accepted update application;
    /// windows includes lease retirement and queued/snapshot work. Admission
    /// includes urgent edit jobs and deferred-column reinsertion.
    pub plan_edits_ms: f64,
    pub plan_authority_ms: f64,
    pub plan_windows_ms: f64,
    pub plan_near_ms: f64,
    pub plan_visible_ms: f64,
    pub plan_admission_ms: f64,
    /// Issued jobs by level range, including urgent regenerations; these are
    /// not publication completions or counts of visible coherent blocks.
    pub fine_jobs: usize,
    pub far_jobs: usize,
    /// Current selected tiles, live Camera leases and jobs for current
    /// selected tiles (any ownership), for levels 0–2. Jobs are not completions.
    pub camera_candidate_blocks: [u32; 3],
    pub camera_lease_blocks: [u32; 3],
    pub camera_jobs: [u32; 3],
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
    current_request: Option<WindowRequest>,
    ground_clearance: Option<f64>,
    camera_view: Option<CameraView>,
    requested: u64,
    applied: u64,
    /// Demand authority can arrive independently for fine and far levels.
    /// `applied` is their minimum, never a partial whole-world acknowledgment.
    applied_levels: Vec<u64>,
    /// Serial zero is a valid first inline update, distinct from no authority.
    applied_seen: u32,
    applied_issued_at: Vec<Option<std::time::Instant>>,
    /// Window diffs in FIFO order within each level. A large fine-window
    /// retirement must not delay incoming demand at every coarser level.
    diffs: Vec<VecDeque<QueuedDiff>>,
    queued_delta_bytes: usize,
    queued_delta_ops: usize,
    snapshot_mode: bool,
    retire_slot: usize,
    snapshot_epoch: u64,
    retire_started_epoch: u64,
    retire_finished_epoch: u64,
    obsolete_owners: VecDeque<OwnerRetirement>,
    /// Next non-global level to receive a bounded diff round.
    diff_cursor: usize,
    /// Per level, diffs still queued for it (its window is not yet exact).
    catching_up: Vec<u32>,
    /// CPU time per `plan` for applying diffs and admitting columns; `None`
    /// is unbounded (deterministic, for tests).
    cpu_budget: Option<std::time::Duration>,
    lod_dither: f64,
    prefetch_eye: Option<DVec3>,
    priority_eye: Option<DVec3>,
    view_focus: Option<DVec3>,
    /// Bounded asynchronous feedback from primary rays missing fine data.
    visible_blocks: Vec<u64>,
    /// At most 64 blocks' columns, retaining rank through partial budgets.
    visible_admission: VecDeque<(usize, u64)>,
    visible_rank_source: Option<VisibleStamp>,
    visible_source: Option<VisibleStamp>,
    visible_view: Option<VisibleStamp>,
    /// Full aligned block identities. Retiring entries still count toward
    /// the cap until their normal publication acknowledgments are finished.
    visible_leases: FxHashMap<u64, VisibleLease>,
}

/// A window diff being applied: removes first, then (for a level switched
/// off) clearing its queue, then adds, exactly as an immediate apply.
struct QueuedDiff {
    diff: LevelDiff,
    removed: usize,
    cleared: bool,
    added: usize,
    retirement_epoch: u64,
}

struct OwnerRetirement {
    owner: (u32, u8, u32, i32, i32),
    offset: u32,
}

fn delta_bytes(diff: &LevelDiff) -> usize {
    diff.adds.capacity() * std::mem::size_of::<(f32, u64)>()
        + diff.removes.capacity() * std::mem::size_of::<u64>()
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
            current_request: None,
            ground_clearance: None,
            camera_view: None,
            requested: 0,
            applied: 0,
            applied_levels: vec![0; grid.levels() as usize],
            applied_seen: 0,
            applied_issued_at: vec![None; grid.levels() as usize],
            diffs: (0..grid.levels()).map(|_| VecDeque::new()).collect(),
            queued_delta_bytes: 0,
            queued_delta_ops: 0,
            snapshot_mode: false,
            retire_slot: 0,
            snapshot_epoch: 0,
            retire_started_epoch: 0,
            retire_finished_epoch: 0,
            obsolete_owners: VecDeque::new(),
            diff_cursor: 0,
            catching_up: vec![0; grid.levels() as usize],
            cpu_budget: None,
            lod_dither: 0.0,
            prefetch_eye: None,
            priority_eye: None,
            view_focus: None,
            visible_blocks: Vec::new(),
            visible_admission: VecDeque::new(),
            visible_rank_source: None,
            visible_source: None,
            visible_view: None,
            visible_leases: FxHashMap::default(),
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
        self.reference_blocks_count(key, 1, work);
    }

    /// An aligned tier-1 block shares all three summary identities.
    fn reference_blocks_count(&mut self, key: u64, count: u32, work: &mut FrameWork) {
        let (face, level, ci, cj) = unpack(key);
        for tier in 1..=BLOCK_TIERS {
            let (bi, bj) = (ci >> (2 * tier), cj >> (2 * tier));
            let bkey = (level, face, tier, bi, bj);
            if let Some(b) = self.blocks.get_mut(&bkey) {
                b.refs += count;
                continue;
            }
            let slot = block_slot(level, face, tier, bi, bj);
            self.block_owner.insert(slot, bkey);
            work.block_inits.push((slot, bi, bj));
            self.blocks.insert(bkey, Block { slot, refs: count });
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

    pub(crate) fn set_lod_dither(&mut self, dither: f64) {
        self.lod_dither = crate::windows::sanitize_lod_dither(dither);
    }

    pub(crate) fn set_ground_clearance(&mut self, clearance: f64) {
        self.ground_clearance = clearance.is_finite().then_some(clearance);
    }

    pub(crate) fn set_camera_view(&mut self, forward: DVec3, up: DVec3, tan_half: [f64; 2]) {
        self.camera_view = forward.try_normalize().and_then(|forward|
            forward.cross(up).try_normalize().filter(|_|
                tan_half.into_iter().all(|value| value.is_finite() && value > 0.0))
                .map(|right| CameraView { forward, right, up: right.cross(forward), tan_half }));
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

    /// Queue at most64 block identities for the next bounded plan refresh.
    /// Feedback changes priority, never resident window coverage.
    pub fn prioritize_visible_blocks(&mut self, blocks: impl IntoIterator<Item = (u32, u32)>) {
        self.visible_admission.clear();
        self.visible_rank_source = None;
        self.visible_blocks.clear();
        self.visible_blocks.extend(blocks.into_iter().take(VISIBLE_BLOCKS).map(|(a, b)| pack(a, b)));
        self.visible_source = None;
    }

    pub(crate) fn set_visible_view(&mut self, frame: u32, view: u32, at: std::time::Instant) {
        self.visible_view = Some(VisibleStamp { frame, view, at });
    }

    /// Only the accepted GPU batch can create temporary demand. Preserve its
    /// original stamp so delayed or replayed feedback cannot extend a lease.
    pub(crate) fn prioritize_visible_blocks_from(&mut self, blocks: impl IntoIterator<Item = (u32, u32)>,
        frame: u32, view: u32, at: std::time::Instant) {
        self.prioritize_visible_blocks(blocks);
        self.visible_source = Some(VisibleStamp { frame, view, at });
        self.visible_rank_source = self.visible_source;
    }

    fn visible_columns(&self, key: u64) -> Option<([u64; 16], usize)> {
        let (face, level, i, j) = unpack(key);
        if level >= self.grid.levels() || !self.grid.faces().contains(&face) || i & 3 != 0 || j & 3 != 0 { return None; }
        let columns = self.grid.cells() / (BRICK << level);
        if !(0..columns).contains(&i) || !(0..columns).contains(&j) { return None; }
        let mut keys = [0; 16];
        let mut count = 0;
        for y in j..(j + 4).min(columns) {
            for x in i..(i + 4).min(columns) {
                keys[count] = pack(key0(face, level, x), y as u32);
                count += 1;
            }
        }
        Some((keys, count))
    }

    fn current_wanted(&self, key: u64) -> bool {
        self.levels[unpack(key).1 as usize].wanted.as_ref().is_some_and(|wanted| wanted.contains(&key))
    }

    fn transient_wanted(&self, key: u64) -> bool {
        self.visible_leases.get(&(key & !(3u64 | (3u64 << 32)))).is_some_and(|lease|
            !lease.retiring && (match lease.current_demand_frame {
                Some(frame) => frame == self.frame,
                None => self.applied_levels[unpack(key).1 as usize] < lease.serial,
            })
                && match lease.origin {
                    LeaseOrigin::Captured(source) => self.source_is_current(source, VISIBLE_LEASE_FRAMES),
                    LeaseOrigin::Camera => lease.current_demand_frame == Some(self.frame),
                })
    }

    fn current_captured_block(&self, key: u64) -> bool {
        let level = unpack(key).1 as usize;
        level < self.levels.len().min(3) && self.levels[level].active
            && self.current_request.as_ref().is_some_and(|request|
                crate::windows::current_block_wanted(&self.grid, request, key))
    }

    fn current_camera_block(&self, key: u64) -> bool {
        // Ground clearance bounds the column below the eye, not a nearby
        // raised slope. The current window predicate uses the terrain bound.
        self.current_captured_block(key)
    }

    fn protected_wanted(&self, key: u64) -> bool {
        self.current_wanted(key) || self.transient_wanted(key)
    }

    fn source_is_current(&self, source: VisibleStamp, frames: u32) -> bool {
        self.visible_view.is_some_and(|now| now.view == source.view
            && now.frame.wrapping_sub(source.frame) <= frames
            && now.at.checked_duration_since(source.at).is_some_and(|age| age <= VISIBLE_LEASE_TIME))
    }

    /// Expiry uses bounded normal eviction, including its record/journal
    /// quarantine. A newer wanted snapshot always protects a returned column.
    #[cfg(test)]
    fn retire_visible_leases(&mut self, work: &mut FrameWork, out_of_time: &impl Fn() -> bool) {
        let camera_blocks = self.current_request.as_ref()
            .map_or_else(Vec::new, |request| self.camera_blocks(request.eye, out_of_time));
        self.retire_visible_leases_current(&camera_blocks, work, out_of_time);
    }

    fn retire_visible_leases_current(&mut self, camera_blocks: &[u64], work: &mut FrameWork, out_of_time: &impl Fn() -> bool) {
        // Geometry was selected this frame before the bounded retirement walk.
        // Its unvisited tail must retain ownership during snapshot application.
        for block in camera_blocks {
            if let Some(lease) = self.visible_leases.get_mut(block) {
                if !lease.retiring && matches!(lease.origin, LeaseOrigin::Camera) {
                    lease.current_demand_frame = Some(self.frame);
                }
            }
        }
        let blocks: Vec<_> = self.visible_leases.keys().copied().collect();
        let mut steps = 0;
        let mut captured_steps = 0;
        let limit = if self.visible_leases.iter().any(|(&block, lease)|
            matches!(lease.origin, LeaseOrigin::Camera) && (lease.retiring || !camera_blocks.contains(&block))) {
            CAMERA_CANDIDATES * 16
        } else { 128 };
        for block in blocks {
            let lease = &self.visible_leases[&block];
            let source_live = match lease.origin {
                LeaseOrigin::Captured(source) => self.source_is_current(source, VISIBLE_LEASE_FRAMES),
                LeaseOrigin::Camera => camera_blocks.contains(&block),
            };
            let was_bounded = lease.current_demand_frame.is_some();
            let superseded = self.applied_levels[unpack(block).1 as usize] >= lease.serial;
            let ordinary = self.visible_columns(block).is_some_and(|(keys, count)|
                self.levels[unpack(block).1 as usize].active
                    && keys[..count].iter().all(|&key| self.current_wanted(key)));
            let bounded_current = !ordinary && !lease.retiring && source_live && match lease.origin {
                LeaseOrigin::Camera => true,
                LeaseOrigin::Captured(_) => (was_bounded || superseded)
                    && !out_of_time() && self.current_captured_block(block),
            };
            let expire = ordinary || !source_live || (was_bounded || superseded) && !bounded_current;
            let lease = self.visible_leases.get_mut(&block).unwrap();
            lease.current_demand_frame = bounded_current.then_some(self.frame);
            lease.retiring |= expire;
            if !self.visible_leases[&block].retiring { continue; }
            let (keys, count) = self.visible_columns(block).unwrap();
            let captured = matches!(self.visible_leases[&block].origin, LeaseOrigin::Captured(_));
            while self.visible_leases[&block].retired < count {
                if steps == limit || out_of_time() { return; }
                if captured && captured_steps == 128 { break; }
                let index = self.visible_leases[&block].retired;
                self.visible_leases.get_mut(&block).unwrap().retired += 1;
                steps += 1;
                captured_steps += usize::from(captured);
                let key = keys[index];
                if !self.current_wanted(key) {
                    self.levels[unpack(key).1 as usize].pending.remove(key);
                    self.initial_retries.remove(&key);
                    if self.residents.contains_key(key) { self.evict(key, work); }
                }
            }
            if self.visible_leases[&block].retired < count { continue; }
            if !keys[..count].iter().any(|key| self.publishing.contains_key(key)) {
                self.visible_leases.remove(&block);
            }
        }
    }

    fn refresh_visible_pending(&mut self, deadline: Option<std::time::Instant>) {
        self.refresh_visible_pending_until(|| deadline.is_some_and(|at| std::time::Instant::now() >= at));
    }

    fn refresh_visible_pending_until(&mut self, mut out_of_time: impl FnMut() -> bool) {
        let mut captured_leases = self.visible_leases.values()
            .filter(|lease| matches!(lease.origin, LeaseOrigin::Captured(_))).count();
        let rank_requests = self.visible_rank_source.is_none_or(|source| self.source_is_current(source, 8));
        if !rank_requests {
            self.visible_admission.clear();
            self.visible_rank_source = None;
            self.visible_blocks.clear();
        }
        let mut promoted = Vec::new();
        let mut pending = Vec::new();
        let mut blocks = std::mem::take(&mut self.visible_blocks);
        let requested_blocks = blocks.len();
        let mut ranked: FxHashSet<_> = if requested_blocks != 0 {
            self.visible_admission.iter().copied().collect()
        } else { FxHashSet::default() };
        let source = self.visible_source.take().filter(|source| self.source_is_current(*source, 8));
        // Reinsert leased demand after snapshot replacement cleared pending.
        for (&key, lease) in &self.visible_leases {
            if blocks.len() == VISIBLE_BLOCKS { break; }
            if matches!(lease.origin, LeaseOrigin::Captured(_)) && !blocks.contains(&key) { blocks.push(key); }
        }
        for (index, &key) in blocks.iter().enumerate() {
            if out_of_time() {
                // Resume only captured requests, retaining their original
                // age. Lease reinsertion is reconstructed on the next plan.
                if index < requested_blocks {
                    self.visible_blocks.extend_from_slice(&blocks[index..requested_blocks]);
                    self.visible_source = source;
                }
                break;
            }
            let Some((keys, count)) = self.visible_columns(key) else { continue };
            let level = unpack(key).1 as usize;
            let ordinary = self.levels[level].active && keys[..count].iter().all(|&key| self.current_wanted(key));
            if !ordinary && self.requested > self.applied_levels[unpack(key).1 as usize] && index < requested_blocks {
                if let Some(source) = source {
                    let after_authority = self.applied_issued_at[level].is_none_or(|issued| source.at >= issued);
                    let bounded_current = self.visible_leases.get(&key).is_some_and(|lease|
                        !lease.retiring && lease.current_demand_frame == Some(self.frame))
                        || self.current_request.is_some() && !out_of_time()
                            && self.current_captured_block(key);
                    if !after_authority && !bounded_current { continue; }
                    if let Some(lease) = self.visible_leases.get_mut(&key) {
                        if let LeaseOrigin::Captured(previous) = lease.origin {
                            let newer = source.frame.wrapping_sub(previous.frame);
                            if !lease.retiring && newer > 0 && newer < 0x8000_0000 {
                                lease.origin = LeaseOrigin::Captured(source);
                                lease.serial = self.requested;
                            }
                        }
                        if !lease.retiring && bounded_current {
                            lease.current_demand_frame = Some(self.frame);
                        }
                    } else if captured_leases < VISIBLE_BLOCKS && self.visible_leases.len() < TEMPORARY_LEASES {
                        self.visible_leases.insert(key, VisibleLease { origin: LeaseOrigin::Captured(source), serial: self.requested, retiring: false, retired: 0, current_demand_frame: bounded_current.then_some(self.frame) });
                        captured_leases += 1;
                    }
                }
            }
            if !ordinary && !self.transient_wanted(key) { continue; }
            promoted.push(key);
            for &key in &keys[..count] {
                if !self.publishing.contains_key(&key)
                    && (!self.residents.contains_key(key) || self.initial_retries.contains(&key)) {
                    pending.push((level, key));
                    // Leases requeue ordinary demand, but do not invent a
                    // new distance rank after the captured view expires.
                    if rank_requests && index < requested_blocks
                        && self.visible_admission.len() < VISIBLE_ADMISSION_COLUMNS
                        && ranked.insert((level, key)) {
                        self.visible_admission.push_back((level, key));
                    }
                }
            }
        }
        // Select nearest blocks first under the deadline, then commit that
        // bounded selection in reverse: buckets pop newest first. Remove all
        // selected keys before reinsertion so swap-removal cannot fragment
        // complete blocks, including keys already waiting in bucket zero.
        for &(level, key) in &pending { self.levels[level].pending.remove(key); }
        for &(level, key) in pending.iter().rev() { self.levels[level].pending.insert(key, 0); }
        for key in promoted { self.queue_obsolete_owners(key); }
    }

    fn camera_blocks(&self, eye: DVec3, out_of_time: &impl Fn() -> bool) -> Vec<u64> {
        let (Some(clearance), Some(view), Some(request)) =
            (self.ground_clearance, self.camera_view, self.current_request.as_ref()) else { return Vec::new() };
        let grid = self.grid;
        let ground_radial = grid.radial(eye) - clearance;
        let radial_up = grid.up(eye);
        let direction = (view.forward - radial_up * view.forward.dot(radial_up)).normalize_or_zero();
        let nearest = crate::windows::visible_focus(&grid, eye,
            view.forward - view.up * view.tan_half[1], ground_radial, request.lod0 * 8.0)
            .map_or(0.0, |lower| grid.ground_distance(eye, lower));
        let ground = eye - radial_up * clearance;
        let mut candidates = Vec::new();
        let mut inspected = FxHashSet::default();
        let planes = [view.forward,
            view.forward * view.tan_half[0] + view.right,
            view.forward * view.tan_half[0] - view.right,
            view.forward * view.tan_half[1] + view.up,
            view.forward * view.tan_half[1] - view.up];
        let mut covered_end = nearest;
        'levels: for level in 0..self.levels.len().min(3) {
            if !self.levels[level].active { continue; }
            let size = f64::from(BRICK << level);
            let width = grid.level_size(level as u32) * f64::from(BRICK * 4);
            let start = nearest.max(covered_end - width);
            let seed = ground + direction * (start + width * 3.0);
            covered_end = start + width * 6.0;
            let reach = request.lod0 * f64::from(1u32 << level) / (1.0 - request.lod_dither * 0.5);
            let radial_lo = request.planet.as_ref().map_or(ground_radial, |planet| planet.inner_radius())
                .max(grid.radial(eye) - reach);
            let radial_hi = request.outer_radius.min(grid.radial(eye) + reach);
            if radial_lo > radial_hi { continue; }
            // Keep the flat lower-edge strip and a small near strip: a raised
            // ridge can enter the view before that flat intersection.
            let near = ground + direction * (width * 2.0);
            for (mut seed, half_x, half_y) in [(seed, 4, 4), (near, 2, 4)] {
                if !grid.is_plane() { seed = seed.normalize_or_zero() * ground_radial; }
                let face = if grid.is_plane() { crate::grid::PLANE_FACE } else { crate::grid::face_of(seed) };
                let Some(coords) = grid.face_coords(face, seed) else { continue };
                let bi = (coords[0] / size).floor() as i32 / 4;
                let bj = (coords[1] / size).floor() as i32 / 4;
                for y in bj - half_y..bj + half_y {
                    for x in bi - half_x..bi + half_x {
                        if out_of_time() { break 'levels; }
                        if x < 0 || y < 0 { continue; }
                        let block = pack(key0(face, level as u32, x * 4), (y * 4) as u32);
                        if !inspected.insert(block) || !self.current_camera_block(block) { continue; }
                        let mut point = grid.ground_point(face, f64::from(x * 4 + 2) * size,
                            f64::from(y * 4 + 2) * size);
                        if grid.is_plane() { point.y = ground_radial; }
                        else { point = point.normalize_or_zero() * ground_radial; }
                        let relative = point - eye;
                        let radial_up = grid.up(point);
                        let midpoint = point + radial_up * ((radial_lo + radial_hi) * 0.5 - ground_radial);
                        let height = (radial_hi - radial_lo) * 0.5;
                        // Test the possible vertical interval, clipped to this
                        // level's reach. Scheduling never clips rendered terrain.
                        let bound = width * 1.75;
                        if planes.iter().any(|normal| normal.dot(midpoint - eye)
                            + height * normal.dot(radial_up).abs()
                            + bound * (*normal - radial_up * normal.dot(radial_up)).length() < 0.0) { continue; }
                        let distance2 = relative.length_squared().max(0.01);
                        let projected_cell = request.lod0 * f64::from(1u32 << level) / distance2.sqrt();
                        let oversized = if level == 0 { 1.0 } else { (projected_cell / 4.0).max(1.0) };
                        // Screen coverage per issued tile; oversized coarser cells
                        // cannot displace visibly resolved finer tiles.
                        let benefit = width * width / (distance2 * oversized.powi(4));
                        candidates.push((benefit, block));
                    }
                }
            }
        }
        candidates.sort_unstable_by(|a, b| unpack(a.1).1.cmp(&unpack(b.1).1)
            .then_with(|| b.0.total_cmp(&a.0)));
        let mut selected = [0; 3];
        candidates.retain(|&(_, key)| {
            let count = &mut selected[unpack(key).1 as usize];
            *count += 1;
            *count <= 8 * 8
        });
        candidates.truncate(CAMERA_CANDIDATES);
        candidates.into_iter().map(|(_, block)| block).collect()
    }

    /// Current camera tiles may arrive before the worker's wanted snapshot.
    /// Reuse its capped lease/publication path and FIFO; no forecast demand.
    fn refresh_camera_pending(&mut self, blocks: &[u64], deadline: Option<std::time::Instant>) {
        let out_of_time = || deadline.is_some_and(|at| std::time::Instant::now() >= at);
        if out_of_time() || self.ground_clearance.is_none() { return; }
        let mut candidates = Vec::new();
        let mut camera_leases = self.visible_leases.values()
            .filter(|lease| matches!(lease.origin, LeaseOrigin::Camera)).count();
        for &block in blocks {
            if out_of_time() { break; }
            let level = unpack(block).1 as usize;
            let Some((keys, count)) = self.visible_columns(block) else { continue };
            let ordinary = keys[..count].iter().all(|&key| self.current_wanted(key));
            if !ordinary {
                if let Some(lease) = self.visible_leases.get_mut(&block) {
                    // Captured stamps retain their original TTL and serial.
                    if matches!(lease.origin, LeaseOrigin::Camera) && !lease.retiring {
                        lease.current_demand_frame = Some(self.frame);
                    }
                } else if camera_leases < CAMERA_LEASES && self.visible_leases.len() < TEMPORARY_LEASES {
                    self.visible_leases.insert(block, VisibleLease {
                        origin: LeaseOrigin::Camera, serial: self.requested,
                        retiring: false, retired: 0, current_demand_frame: Some(self.frame),
                    });
                    camera_leases += 1;
                }
                if !self.transient_wanted(block) { continue; }
            }
            self.queue_obsolete_owners(block);
            for &key in &keys[..count] {
                if !self.publishing.contains_key(&key)
                    && (!self.residents.contains_key(key) || self.initial_retries.contains(&key)) {
                    self.levels[level].pending.insert(key, 0);
                    candidates.push((level, key));
                }
            }
        }
        if candidates.is_empty() { return; }
        // Current view ranks must replace the previous view's ordinary ranks,
        // which have no lease origin. They remain queued at their normal
        // bucket instead of masquerading as permanently captured fine work.
        let mut remaining = std::mem::take(&mut self.visible_admission);
        let current: FxHashSet<_> = candidates.iter().copied().collect();
        remaining.retain(|entry| !current.contains(entry)
            && self.visible_leases.get(&(entry.1 & !(3u64 | (3u64 << 32))))
                .is_some_and(|lease| matches!(lease.origin, LeaseOrigin::Captured(_))));
        let mut fine = VecDeque::new();
        for candidate in candidates {
            if fine.len() == VISIBLE_ADMISSION_COLUMNS { break; }
            fine.push_back(candidate);
        }
        // Captured clocks and pending ownership are unchanged. Their ranks
        // follow current demand, so old L2 work cannot hold up current L0.
        remaining.truncate((VISIBLE_ADMISSION_COLUMNS).saturating_sub(fine.len()));
        fine.append(&mut remaining);
        self.visible_admission = fine;
    }

    /// Publish latest demand before queueing bounded window operations.
    fn apply(&mut self, update: WindowUpdate) {
        // Legacy complete updates had no mask, acknowledged every level,
        // and could replace queued snapshot payloads at the same serial.
        // Ranged worker messages always carry an explicit authority mask,
        // including unchanged/inactive levels. Fine and far serials can arrive
        // out of order; only an older message for the SAME level is obsolete.
        let legacy = update.processed_levels == 0 && !update.partial;
        let processed = if update.processed_levels != 0 { update.processed_levels }
            else if !update.partial { u32::MAX }
            else { 0 };
        let mut accepted = 0u32;
        for (level, serial) in self.applied_levels.iter_mut().enumerate() {
            if processed & (1u32 << level) != 0 && (update.serial > *serial
                || self.applied_seen & (1u32 << level) == 0
                || legacy && update.serial == *serial) {
                accepted |= 1u32 << level;
                self.applied_seen |= 1u32 << level;
                *serial = update.serial;
                self.applied_issued_at[level] = update.issued_at;
            }
        }
        self.applied = self.applied_levels.iter().copied().min().unwrap_or(0);
        self.update_authority_stats();
        let fine_mask = (1u32 << self.grid.levels().min(3)) - 1;
        let age = update.issued_at.and_then(|issued|
            std::time::Instant::now().checked_duration_since(issued))
            .map(|age| age.as_secs_f64() * 1e3);
        if accepted & fine_mask != 0 {
            self.stats.fine_apply_age_ms = age;
            self.stats.fine_planning_ms = Some(update.planning_ms);
        }
        if accepted & !fine_mask != 0 {
            self.stats.far_apply_age_ms = age;
            self.stats.far_planning_ms = Some(update.planning_ms);
        }
        let accepts = |level: u32| level < 32 && accepted & (1u32 << level) != 0;
        // A second chunk at the same request serial can replace far demand
        // after the fine chunk's scan completed. Scan epochs describe actual
        // accepted snapshot payloads, not request/whole-world authority.
        if update.snapshot && (update.wanted.iter().any(|(level, _)| accepts(*level))
            || update.levels.iter().any(|diff| accepts(diff.level))) {
            self.snapshot_epoch += 1;
            if self.retire_slot == 0 { self.retire_started_epoch = self.snapshot_epoch; }
        }
        if accepted != 0 { self.snapshot_mode |= update.snapshot; }
        for (level, wanted) in update.wanted {
            if !accepts(level) {
                if let Planner::Worker(worker) = &self.planner { worker.retire_wanted(wanted); }
                continue;
            }
            if let Some(previous) = self.levels[level as usize].wanted.replace(wanted) {
                if let Planner::Worker(worker) = &self.planner { worker.retire_wanted(previous); }
            }
        }
        for diff in update.levels {
            if !accepts(diff.level) {
                if let Planner::Worker(worker) = &self.planner { worker.retire_diff(diff); }
                continue;
            }
            let level = diff.level as usize;
            if update.snapshot {
                // The worker supplies every current key, so old additions
                // need no historical replay. Actual residents retire below.
                for old in std::mem::take(&mut self.diffs[level]) {
                    self.queued_delta_bytes -= delta_bytes(&old.diff);
                    self.queued_delta_ops -= old.diff.removes.len() - old.removed + old.diff.adds.len() - old.added;
                    if let Planner::Worker(worker) = &self.planner { worker.retire_diff(old.diff); }
                }
                let pending = std::mem::take(&mut self.levels[level].pending);
                if let Planner::Worker(worker) = &self.planner { worker.retire_payload(pending); }
                self.catching_up[level] = 0;
            }
            // The window metadata changes at once; the level is marked as
            // catching up (no guaranteed coverage) until its ops are done.
            self.levels[level].active = diff.active;
            self.levels[level].center = diff.center;
            self.levels[level].radius = diff.radius;
            self.catching_up[level] += 1;
            self.queued_delta_bytes += delta_bytes(&diff);
            self.queued_delta_ops += diff.removes.len() + diff.adds.len();
            self.diffs[level].push_back(QueuedDiff { diff, removed: 0, cleared: false, added: 0,
                retirement_epoch: self.snapshot_epoch });
        }
        if accepted != 0 { self.stats.window_rebuild_ms = update.planning_ms; }
    }

    fn update_authority_stats(&mut self) {
        let fine = self.applied_levels.len().min(3);
        self.stats.requested_serial = self.requested;
        self.stats.applied_serial = self.applied;
        self.stats.fine_applied_serial = self.applied_levels[..fine].iter().copied().min().unwrap_or(0);
        self.stats.far_applied_serial = self.applied_levels[fine..].iter().copied().min();
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

    /// Prioritize obsolete alias owners without retaining old window keys.
    /// Retirement rechecks latest membership before every individual column.
    fn queue_obsolete_owners(&mut self, key: u64) {
        if !self.snapshot_mode { return; }
        let (face, level, i, j) = unpack(key);
        for tier in 1..=BLOCK_TIERS {
            if self.obsolete_owners.len() >= 64 { break; }
            let desired = (level, face, tier, i >> (2 * tier), j >> (2 * tier));
            let slot = block_slot(level, face, tier, desired.3, desired.4);
            if let Some(&owner) = self.block_owner.get(&slot) {
                if owner != desired && !self.obsolete_owners.iter().any(|task| task.owner == owner) {
                    self.obsolete_owners.push_back(OwnerRetirement { owner, offset: 0 });
                }
            }
        }
    }

    fn retire_obsolete_owner_step(&mut self, work: &mut FrameWork) -> bool {
        let Some(task) = self.obsolete_owners.front_mut() else { return false };
        if !self.blocks.contains_key(&task.owner) {
            self.obsolete_owners.pop_front();
            return true;
        }
        let (level, face, tier, bi, bj) = task.owner;
        let size = 1u32 << (2 * tier);
        // Walk complete tier-1 groups, then their sixteen columns. A large
        // outgoing owner is often only a clipped fringe; absent groups have
        // no resident summary references and need no sixteen scalar probes.
        let groups = size / 4;
        let group = task.offset / 16;
        let member = task.offset % 16;
        let i = bi * size as i32 + ((group % groups) * 4 + member % 4) as i32;
        let j = bj * size as i32 + ((group / groups) * 4 + member / 4) as i32;
        let absent = !self.blocks.contains_key(&(level, face, 1, i >> 2, j >> 2));
        task.offset += if absent { 16 - member } else { 1 };
        if task.offset == size * size { self.obsolete_owners.pop_front(); }
        if absent { return true; }
        let key = pack(key0(face, level, i), j as u32);
        if self.levels[level as usize].wanted.is_some() && !self.protected_wanted(key) {
            self.levels[level as usize].pending.remove(key);
            if self.residents.contains_key(key) { self.evict(key, work); }
        }
        true
    }

    /// Snapshot mode holds one full demand per level. Retire actual residents
    /// through a persistent table cursor, never millions of historical keys.
    fn apply_snapshot(&mut self, work: &mut FrameWork, out_of_time: &impl Fn() -> bool) {
        while (self.diffs.iter().any(|diffs| !diffs.is_empty())
            || self.retire_finished_epoch < self.snapshot_epoch) && !out_of_time() {
            for step in 0..128 {
                // Each eviction remains atomic; stop before the next key
                // once the window phase has consumed its time allowance.
                if out_of_time() { return; }
                // Protected alias owners can be requeued after every slice.
                // Reserve half the same bounded work for the normal cursor
                // until its epoch completes, then let owners use all of it.
                if (step < 64 || self.retire_finished_epoch >= self.snapshot_epoch)
                    && self.retire_obsolete_owner_step(work) { continue; }
                // Admission only inserts current demand. Once this snapshot's
                // pass is complete, remaining adds need no repeated scan.
                // Lease expiry retires separately, and aliases stay above.
                if self.retire_finished_epoch >= self.snapshot_epoch { break; }
                // Each step inspects one bitmap word at most. Empty slots
                // skip together, while the round/deadline bound is unchanged.
                let (slot, resident) = self.residents.retirement_step(self.retire_slot);
                self.retire_slot = slot;
                if let Some((key, _)) = resident {
                    let level = unpack(key).1 as usize;
                    if self.levels[level].wanted.is_some() && !self.protected_wanted(key) {
                        self.evict(key, work);
                        // Backshift may have moved another resident here.
                        continue;
                    }
                    self.retire_slot += 1;
                }
                if self.retire_slot == self.residents.table().len() {
                    self.retire_slot = 0;
                    self.retire_finished_epoch = self.retire_started_epoch;
                    self.retire_started_epoch = self.snapshot_epoch;
                }
            }
            let top = self.diffs.len() - 1;
            let has_adds = |level: usize| self.diffs[level].front().is_some_and(|q| q.added < q.diff.adds.len());
            let mut next = has_adds(top).then_some(top);
            if next.is_none() {
                for _ in 0..top {
                    let level = self.diff_cursor;
                    self.diff_cursor = (self.diff_cursor + 1) % top;
                    if has_adds(level) { next = Some(level); break; }
                }
            }
            if let Some(level) = next {
                let queued = self.diffs[level].front_mut().unwrap();
                let end = (queued.added + 128).min(queued.diff.adds.len());
                let state = &mut self.levels[level];
                let wanted = state.wanted.as_ref().unwrap();
                let mut at = queued.added;
                while at < end {
                    if out_of_time() { break; }
                    let skip = resident_snapshot_run(&queued.diff.adds[at..end], !self.initial_retries.is_empty(),
                        |identity| self.blocks.get(&identity).map(|block| block.refs));
                    if skip != 0 { at += skip; continue; }
                    let (priority, key) = queued.diff.adds[at];
                    if wanted.contains(&key) && !self.publishing.contains_key(&key)
                        && (!self.residents.contains_key(key) || self.initial_retries.contains(&key)) {
                        state.pending.insert(key, PendingQueue::bucket(priority));
                    }
                    at += 1;
                }
                self.queued_delta_ops -= at - queued.added;
                queued.added = at;
            }
            for level in 0..self.diffs.len() {
                if self.diffs[level].front().is_some_and(|q|
                    q.added == q.diff.adds.len() && self.retire_finished_epoch >= q.retirement_epoch) {
                    let completed = self.diffs[level].pop_front().unwrap();
                    self.queued_delta_bytes -= delta_bytes(&completed.diff);
                    self.catching_up[level] = 0;
                    if let Planner::Worker(worker) = &self.planner { worker.retire_diff(completed.diff); }
                }
            }
        }
    }

    /// Retire oldest owners while admitting the newest incoming demand.
    /// Latest membership makes out-of-order additions safe: an old removal
    /// cannot evict a returned column, and obsolete additions cannot run.
    fn apply_queued(&mut self, work: &mut FrameWork, out_of_time: &impl Fn() -> bool) {
        if self.snapshot_mode {
            self.apply_snapshot(work, out_of_time);
            return;
        }
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
                if self.protected_wanted(key) { continue; }
                self.levels[level].pending.remove(key);
                if self.residents.contains_key(key) {
                    self.evict(key, work);
                }
            }
            self.queued_delta_ops -= end - queued.removed;
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
                self.queued_delta_ops -= end - incoming.added;
                incoming.added = end;
            } else if queued.diff.active || queued.removed == queued.diff.removes.len() {
                let end = (queued.added + if queued.diff.active { CHUNK / 2 } else { CHUNK }).min(queued.diff.adds.len());
                for i in queued.added..end {
                    let (priority, key) = queued.diff.adds[i];
                    if !self.residents.contains_key(key) {
                        self.levels[level].pending.insert(key, PendingQueue::bucket(priority));
                    }
                }
                self.queued_delta_ops -= end - queued.added;
                queued.added = end;
            }
            if queued.removed == queued.diff.removes.len() && queued.added == queued.diff.adds.len() {
                self.catching_up[level] -= 1;
                self.queued_delta_bytes -= delta_bytes(&queued.diff);
                if let Planner::Worker(worker) = &self.planner { worker.retire_diff(queued.diff); }
            } else {
                self.diffs[level].push_front(queued);
            }
            // Newer additions may finish a diff before its FIFO turn.
            while self.diffs[level].back().is_some_and(|diff|
                diff.removed == diff.diff.removes.len() && diff.added == diff.diff.adds.len()) {
                let completed = self.diffs[level].pop_back().unwrap();
                self.catching_up[level] -= 1;
                self.queued_delta_bytes -= delta_bytes(&completed.diff);
                if let Planner::Worker(worker) = &self.planner { worker.retire_diff(completed.diff); }
            }
            if out_of_time() { return; }
        }
    }

    /// Refresh a fixed neighborhood of complete tier-1 blocks. Traversal
    /// only uses a streaming level after all sixteen columns of a block are
    /// published, so keep its pending columns together at one priority.
    /// Only complete current wanted blocks are touched. They may bypass an
    /// old diff's priority prefix; this does not expand window membership.
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
            if !state.active || (state.pending.is_empty()
                && (state.wanted.is_none() || self.catching_up[level] == 0)) {
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
                    if out_of_time() { return; }
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
                        complete &= state.wanted.as_ref().map_or_else(
                            || state.pending.at.contains_key(key) || self.residents.contains_key(*key),
                            |wanted| wanted.contains(key));
                    }
                    if !complete { continue; }
                    let priority = distance / state.radius.max(grid.level_size(level as u32));
                    let bucket = PendingQueue::bucket(priority as f32);
                    for key in keys {
                        let queued = state.pending.remove(key);
                        // A camera turn can expose wanted data before the
                        // worker's old priority prefix reaches pending. Bypass
                        // that ordering only for a fully current wanted block;
                        // admission still owns records, summaries and edits.
                        if queued || (state.wanted.is_some() && !self.residents.contains_key(key)
                            && !self.publishing.contains_key(&key)) {
                            state.pending.insert(key, bucket);
                        }
                    }
                }
            }
        }
    }

    /// Admit only a complete, unedited block already ranked by visible feedback.
    fn admit_visible_block(&mut self, planet: &Planet, index: usize, first: u64, bucket: usize,
        budget: usize, deadline: Option<std::time::Instant>, work: &mut FrameWork) -> bool {
        self.admit_pending_block(planet, index, first, bucket, budget, deadline, true, work)
    }

    /// Share the existing intact-block path with ordinary same-priority demand.
    fn admit_pending_block(&mut self, planet: &Planet, index: usize, selected: u64, bucket: usize,
        budget: usize, deadline: Option<std::time::Instant>, visible: bool, work: &mut FrameWork) -> bool {
        let expired = || deadline.is_some_and(|at| std::time::Instant::now() >= at);
        // Only intact blocks: never regroup partial work, overtake
        // global coverage, invent lease demand, or batch an edit publication.
        if expired() || budget.saturating_sub(work.jobs.len()) < 16
            || index == (self.grid.levels() - 1) as usize
            || !self.levels[(self.grid.levels() - 1) as usize].pending.is_empty() {
            return false;
        }
        let first = selected & !(3u64 | (3u64 << 32));
        let Some((keys, 16)) = self.visible_columns(first) else { return false };
        if unpack(first).1 as usize != index { return false; }
        if visible && (selected != first || self.visible_admission.len() < 15
            || !self.visible_admission.iter().take(15).zip(&keys[1..])
                .all(|(&(level, key), &expected)| level == index && key == expected))
            || !visible && !self.visible_admission.is_empty() {
            return false;
        }
        let available = self.free_records.len()
            + self.capacity.records.saturating_sub(self.next_record) as usize;
        let table_limit = (1usize << self.capacity.table_bits) - 1;
        if available < 16 || self.residents.len().saturating_add(16) > table_limit {
            return false;
        }
        if self.blocks_conflict(first) { return false; }
        // Visible keys may share an existing captured lease. Ordinary keys
        // must all belong to current wanted demand at the selected bucket.
        // Match scalar admission without creating or refreshing any demand.
        let transient = visible && self.transient_wanted(first);
        for &key in &keys {
            if (!self.current_wanted(key) && !transient) || self.residents.contains_key(key)
                || self.initial_retries.contains(&key) || self.publishing.contains_key(&key)
                || (key != selected && self.levels[index].pending.at.get(&key)
                    .is_none_or(|entry| !visible && entry.0 as usize != bucket)) {
                return false;
            }
        }
        if !planet.edits().is_empty() {
            // Match edit_list's base-cell query units and level threshold.
            // A brush anywhere in this conservative union keeps the whole
            // block on the scalar path; unrelated world edits do not disable
            // unedited block admission throughout the planet.
            let (face, level, i, j) = unpack(first);
            let span = i64::from(BRICK) << level;
            let i0 = i64::from(i) * span;
            let j0 = i64::from(j) * span;
            if !planet.edits().query(face, i0, i0 + 4 * span - 1,
                j0, j0 + 4 * span - 1, level).is_empty() {
                return false;
            }
        }
        if expired() { return false; }
        // All fallible guards ran before mutation. The preflight reserved
        // sixteen individual record identities; shared refs change only once.
        self.reference_blocks_count(first, 16, work);
        for &key in &keys {
            let column_bucket = if key == selected { bucket } else {
                if visible { self.visible_admission.pop_front(); }
                let bucket = self.levels[index].pending.at[&key].0 as usize;
                self.levels[index].pending.remove(key);
                bucket
            };
            let record = self.alloc_record().expect("complete block record preflight");
            let slot = self.residents.insert(key, Resident { record, slot: 0, edit_block: None, blocks: true });
            work.table_writes.push((slot, record));
            self.publishing.insert(key, EditPublication {
                record, previous: None, next: None, evicted: false, initial_bucket: Some(column_bucket),
            });
            work.jobs.push(Job {
                key0: key as u32, key1: (key >> 32) as u32, record, edits: 0, flags: 0, pad: [0; 3],
            });
            work.job_keys.push(key);
        }
        true
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
            lod_dither: self.lod_dither,
            outer_radius: planet.outer_radius(),
            planet: Some(planet.clone()),
            serial: self.requested + 1,
        };
        self.current_request = Some(request.clone());
        let changed = self.last_request.as_ref().is_none_or(|last| {
            last.eye.distance(eye) > self.grid.voxel_size() * 2.0
                || (last.lod0 - lod0).abs() > lod0 * 0.01
                || last.lod_dither != request.lod_dither
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
        // New demand may not have reached pending yet. Keep admission's
        // share even when queues were empty before applying the first diff.
        let apply_out_of_time = move || budget_time.is_some_and(|b| started.elapsed() >= b.mul_f64(0.6));
        let camera_blocks = self.camera_blocks(eye, &apply_out_of_time);
        self.retire_visible_leases_current(&camera_blocks, &mut work, &apply_out_of_time);
        if !apply_out_of_time() { self.apply_queued(&mut work, &apply_out_of_time); }
        let t_windows = started.elapsed();
        // Window diffs may spend 60% of the CPU budget. Refresh gets at most
        // the next 10%, leaving 30% for issuing generation jobs this frame.
        let refresh_deadline = budget_time.map(|budget| started + budget.mul_f64(0.7));
        let visible_deadline = budget_time.map(|budget| started + budget.mul_f64(0.65));
        self.refresh_visible_pending(visible_deadline);
        let t_visible = started.elapsed();
        self.refresh_camera_pending(&camera_blocks, refresh_deadline);
        self.refresh_near_pending(eye, refresh_deadline);
        let t_apply = started.elapsed();
        let t_near = t_windows + (t_apply - t_visible);
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
        let mut pending_selection = None;
        let admission_deadline = budget_time.map(|budget| started + budget);
        let mut admission_attempts = 0;
        let mut alias_deferred = 0;
        let mut publication_deferred = 0;
        let mut batched_columns = 0;
        let mut batch_attempts = 0;
        let mut last_batch_block = None;
        while work.jobs.len() < budget {
            steps += 1;
            if (steps == 1 || steps % 64 == 0) && out_of_time() {
                break;
            }
            let Some((index, key, bucket)) = pop_pending(&mut self.levels, top_level,
                &mut self.visible_admission, &mut pending_selection, admission_deadline) else { break };
            admission_attempts += 1;
            let block_key = key & !(3u64 | (3u64 << 32));
            let visible_batch = key == block_key
                && self.visible_admission.front().is_some_and(|&(level, next)|
                    level == index && next & !(3u64 | (3u64 << 32)) == key);
            if index != top_level as usize && (visible_batch && batch_attempts < TEMPORARY_LEASES
                || self.visible_admission.is_empty() && last_batch_block != Some(block_key)) {
                if visible_batch { batch_attempts += 1; }
                last_batch_block = Some(block_key);
                let admitted = if visible_batch {
                    self.admit_visible_block(planet, index, key, bucket, budget, admission_deadline, &mut work)
                } else {
                    self.admit_pending_block(planet, index, key, bucket, budget, admission_deadline, false, &mut work)
                };
                if admitted {
                    admission_attempts += 15;
                    batched_columns += 16;
                    last_summary_check = None;
                    if out_of_time() { break; }
                    continue;
                }
                if out_of_time() {
                    self.levels[index].pending.insert(key, bucket);
                    if visible_batch { self.visible_admission.push_front((index, key)); }
                    break;
                }
            }
            let transient = self.transient_wanted(key);
            let leased = self.visible_leases.contains_key(&(key & !(3u64 | (3u64 << 32))));
            if (self.levels[index].wanted.as_ref().is_some_and(|wanted| !wanted.contains(&key))
                || leased && !self.current_wanted(key)) && !transient {
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
                publication_deferred += 1;
                // Keep this incarnation queued, but let unrelated columns
                // use the remaining budget while its GPU result is in flight.
                awaiting_publication.push((index, key, bucket));
                continue;
            }
            // Active diffs can now queue incoming columns before all outgoing
            // blocks retire. Wait for an alias owner instead of permanently
            // publishing a column without summaries during a large move.
            let conflict = if !retry || self.catching_up[index] > 0 || transient {
                self.blocks_conflict_cached(key, &mut last_summary_check)
            } else { false };
            if (self.catching_up[index] > 0 || transient) && conflict {
                alias_deferred += 1;
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
            self.queue_obsolete_owners(key);
            self.levels[index].pending.insert(key, bucket);
        }
        let t_admission = started.elapsed();
        let trace_ms: Option<f64> = std::env::var("HELIO_VOXEL_PLAN_TRACE").ok().map(|v| v.parse().unwrap_or(10.0));
        if trace_ms.is_some_and(|ms| started.elapsed().as_secs_f64() * 1e3 > ms) {
            eprintln!(
                "PLAN_TRACE edits {:.2} drain {:.2} apply {:.2} admit {:.2} ms jobs {} evictions {} queued_diffs {} steps {steps} queued_bytes {} queued_ops {} wanted_capacity {} admission_attempts {admission_attempts} alias_deferred {alias_deferred} publication_deferred {publication_deferred} batched_columns {batched_columns}",
                t_edits.as_secs_f64() * 1e3,
                (t_drain - t_edits).as_secs_f64() * 1e3,
                (t_apply - t_drain).as_secs_f64() * 1e3,
                (started.elapsed() - t_apply).as_secs_f64() * 1e3,
                work.jobs.len(),
                work.evictions.len(),
                self.diffs.iter().map(VecDeque::len).sum::<usize>(),
                self.queued_delta_bytes,
                self.queued_delta_ops,
                self.levels.iter().filter_map(|level| level.wanted.as_ref()).map(|wanted| wanted.capacity()).sum::<usize>()
            );
        }
        self.update_authority_stats();
        let mut stats = self.stats;
        stats.camera_candidate_blocks = [0; 3];
        stats.camera_lease_blocks = [0; 3];
        stats.camera_jobs = [0; 3];
        for &block in &camera_blocks {
            stats.camera_candidate_blocks[unpack(block).1 as usize] += 1;
        }
        for (&block, lease) in &self.visible_leases {
            if !lease.retiring && lease.current_demand_frame == Some(self.frame)
                && matches!(lease.origin, LeaseOrigin::Camera) {
                stats.camera_lease_blocks[unpack(block).1 as usize] += 1;
            }
        }
        let selected_camera: FxHashSet<_> = camera_blocks.iter().copied().collect();
        for &key in &work.job_keys {
            if selected_camera.contains(&(key & !(3u64 | (3u64 << 32)))) {
                stats.camera_jobs[unpack(key).1 as usize] += 1;
            }
        }
        stats.plan_edits_ms = t_edits.as_secs_f64() * 1e3;
        stats.plan_authority_ms = (t_drain - t_edits).as_secs_f64() * 1e3;
        stats.plan_windows_ms = (t_windows - t_drain).as_secs_f64() * 1e3;
        stats.plan_near_ms = (t_near - t_windows).as_secs_f64() * 1e3;
        stats.plan_visible_ms = (t_apply - t_near).as_secs_f64() * 1e3;
        stats.plan_admission_ms = (t_admission - t_apply).as_secs_f64() * 1e3;
        (stats.fine_jobs, stats.far_jobs) = work.jobs.iter().fold((0, 0), |(fine, far), job|
            if job.key0 >> 27 < 3 { (fine + 1, far) } else { (fine, far + 1) });
        stats.fine_window_lag_m = self.levels[0].active.then(||
            self.grid.ground_distance(self.levels[0].center, eye));
        stats.resident_columns = self.residents.len();
        stats.pending_columns = self.levels.iter().map(|l| l.pending.len()).sum::<usize>() + self.urgent.len() + self.publishing.len();
        stats.active_levels = self.levels.iter().filter(|l| l.active).count() as u32;
        stats.finest_level = self.levels.iter().position(|l| l.active).unwrap_or(0) as u32;
        stats.jobs = work.jobs.len();
        stats.evictions = work.evictions.len();
        stats.edit_words = self.edits.top;
        stats.table_load = self.residents.load();
        stats.queued_delta_bytes = self.queued_delta_bytes;
        stats.queued_delta_ops = self.queued_delta_ops;
        stats.wanted_key_capacity = self.levels.iter().filter_map(|level| level.wanted.as_ref()).map(|wanted| wanted.capacity()).sum();
        stats.admission_attempts = admission_attempts;
        stats.admission_alias_deferred = alias_deferred;
        stats.admission_publication_deferred = publication_deferred;
        stats.admission_batched_columns = batched_columns;
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
            && (self.applied_seen == 0 || self.applied_seen == (1u32 << self.grid.levels()) - 1)
            && (!self.snapshot_mode || self.retire_finished_epoch >= self.snapshot_epoch)
            && self.diffs.iter().all(VecDeque::is_empty)
            && self.levels.iter().all(|l| l.pending.is_empty())
            && self.visible_leases.is_empty()
    }
}

#[cfg(test)]
mod tests {
    impl VisibleLease {
        fn captured_source(&self) -> VisibleStamp {
            match self.origin { LeaseOrigin::Captured(source) => source, LeaseOrigin::Camera => panic!("expected capture") }
        }
    }
    use super::*;
    use crate::planet::PlanetRecipe;

    #[test]
    fn resident_snapshot_run_requires_exact_complete_owner() {
        let adds: Vec<_> = (0..16).map(|member| (0.1, pack(key0(2, 3, 8 + member % 4), (12 + member / 4) as u32))).collect();
        let lookups = std::cell::Cell::new(0);
        assert_eq!(resident_snapshot_run(&adds, false, |identity| {
            assert_eq!(identity, (3, 2, 1, 2, 3));
            lookups.set(lookups.get() + 1);
            Some(16)
        }), 16);
        assert_eq!(lookups.get(), 1, "a complete resident run needs one summary lookup");
        for refs in [None, Some(15), Some(17)] {
            assert_eq!(resident_snapshot_run(&adds, false, |_| refs), 0);
        }
        assert_eq!(resident_snapshot_run(&adds, false, |identity|
            (identity == (3, 2, 1, 130, 3)).then_some(16)), 0,
            "a toroidal alias is not the exact summary owner");
    }

    #[test]
    fn resident_snapshot_run_preserves_retries_partial_and_reordered_demand() {
        let adds: Vec<_> = (0..16).map(|member| (0.1, pack(key0(2, 3, 8 + member % 4), (12 + member / 4) as u32))).collect();
        let never_lookup = |_| -> Option<u32> { panic!("unproven runs must use per-column gates") };
        assert_eq!(resident_snapshot_run(&adds, true, never_lookup), 0);
        assert_eq!(resident_snapshot_run(&adds[..15], false, never_lookup), 0);
        let mut changed = adds.clone();
        changed.swap(6, 7);
        assert_eq!(resident_snapshot_run(&changed, false, never_lookup), 0);
        changed = adds.clone();
        changed[6] = changed[5];
        assert_eq!(resident_snapshot_run(&changed, false, never_lookup), 0);
        changed = adds.clone();
        changed[15].1 = pack(key0(2, 3, 12), 15);
        assert_eq!(resident_snapshot_run(&changed, false, never_lookup), 0);
        changed = adds.clone();
        changed.rotate_left(1);
        assert_eq!(resident_snapshot_run(&changed, false, never_lookup), 0);
    }

    #[test]
    fn resident_snapshot_skip_preserves_publication_retirement_and_deadline() {
        let planet = Planet::new(PlanetRecipe { shape: crate::grid::Shape::Plane, ..Default::default() }).unwrap();
        let mut r = Residency::new(*planet.grid(), Capacity { table_bits: 8, ..Default::default() });
        let face = crate::grid::PLANE_FACE;
        let run = |i0| (12..16).flat_map(move |j| (i0..i0 + 4).map(move |i|
            (0.1, pack(key0(face, 0, i), j as u32)))).collect::<Vec<_>>();
        let resident = run(8);
        let missing = run(12);
        let mut work = FrameWork::default();
        for (record, &(_, key)) in resident.iter().enumerate() {
            assert!(r.acquire_blocks(key, &mut work));
            r.residents.insert(key, Resident { record: record as u32, slot: 0, edit_block: None, blocks: true });
            r.publishing.insert(key, EditPublication { record: record as u32,
                previous: None, next: None, evicted: false, initial_bucket: Some(6) });
        }
        let obsolete = pack(key0(face, 0, 20), 20);
        assert!(r.acquire_blocks(obsolete, &mut work));
        r.residents.insert(obsolete, Resident { record: 16, slot: 0, edit_block: None, blocks: true });
        let adds: Vec<_> = resident.iter().chain(&missing).copied().collect();
        let update = |serial| WindowUpdate { serial, snapshot: true,
            wanted: vec![(0, std::sync::Arc::new(adds.iter().map(|&(_, key)| key).collect()))],
            levels: vec![LevelDiff { level: 0, active: true, adds: adds.clone(), ..Default::default() }],
            ..Default::default() };
        r.apply(update(1));
        r.apply_snapshot(&mut work, &|| true);
        assert_eq!(r.diffs[0][0].added, 0, "an expired slice cannot consume a resident run");
        r.apply_snapshot(&mut work, &|| false);
        assert_eq!(r.levels[0].pending.keys().copied().collect::<FxHashSet<_>>(),
            missing.iter().map(|&(_, key)| key).collect());
        assert_eq!(r.blocks[&(0, face, 1, 2, 3)].refs, 16);
        assert!(resident.iter().all(|&(_, key)| r.publishing.contains_key(&key)));
        assert!(!r.residents.contains_key(obsolete) && work.evictions.contains(&16));
        assert!(r.diffs[0].is_empty() && r.catching_up[0] == 0 && r.queued_delta_ops == 0);
        // A failed initial publication retains its summary but must requeue.
        let retry = resident[6].1;
        r.publishing.remove(&retry);
        r.initial_retries.insert(retry);
        r.apply(update(2));
        r.apply_snapshot(&mut work, &|| false);
        assert!(r.levels[0].pending.at.contains_key(&retry));
        assert_eq!(r.levels[0].pending.len(), missing.len() + 1);
    }

    #[test]
    fn cached_pending_selection_matches_full_scan_across_plan_boundaries() {
        fn fixture() -> Vec<Level> {
            let mut levels: Vec<_> = (0..6).map(|_| Level::default()).collect();
            for (level, key, bucket) in [(0,100,4), (0,101,4), (0,102,20),
                (1,200,4), (1,201,0), (2,300,0), (2,301,63),
                (4,400,3), (5,500,50), (5,501,1)] {
                levels[level].pending.insert(key, bucket);
            }
            levels
        }
        fn oracle_pop(levels: &mut [Level]) -> Option<(usize, u64, usize)> {
            // Independent full scan: global coverage wins, ties retain the
            // lowest level, and each bucket retains its normal LIFO order.
            let mut choices = Vec::new();
            for (index, level) in levels.iter_mut().enumerate() {
                if let Some(bucket) = level.pending.best() {
                    choices.push((if index == 5 { 0 } else { bucket + 1 }, index));
                }
            }
            let (_, index) = choices.into_iter().min()?;
            let (key, bucket) = levels[index].pending.pop().unwrap();
            Some((index, key, bucket))
        }
        let mut cached_levels = fixture();
        let mut oracle_levels = fixture();
        let mut observed = Vec::new();
        for (phase, budget) in [7, 2, usize::MAX].into_iter().enumerate() {
            // A new plan must see new urgent priorities and previously
            // deferred publications rather than retaining its old winner.
            let mut selection = None;
            for _ in 0..budget {
                let expected = oracle_pop(&mut oracle_levels);
                let actual = select_pending_level(&mut cached_levels, 5, &mut selection)
                    .map(|(index, _)| {
                        let (key, bucket) = cached_levels[index].pending.pop().unwrap();
                        (index, key, bucket)
                    });
                assert_eq!(actual, expected);
                let Some((_, key, _)) = actual else { break };
                observed.push(key);
            }
            let incoming: &[(usize, u64, usize)] = match phase {
                0 => &[(5,501,1), (2,300,0), (3,350,0)],
                1 => &[(5,502,63), (0,103,0)],
                _ => &[],
            };
            for &(level, key, bucket) in incoming {
                cached_levels[level].pending.insert(key, bucket);
                oracle_levels[level].pending.insert(key, bucket);
            }
        }
        assert_eq!(observed, [501,500,201,300,400,101,100,
            501,300,502,103,350,200,102,301]);
        assert!(cached_levels.iter().all(|level| level.pending.is_empty()));
    }

    #[test]
    fn visible_rank_crosses_levels_and_yields_to_global_coverage() {
        let mut levels: Vec<Level> = (0..6).map(|_| Level::default()).collect();
        // The visible L3 block is nearer than the L0 block. Ordinary bucket
        // ties used to select L0 first, losing the engine's distance order.
        for key in 300..316 { levels[3].pending.insert(key, 0); }
        for key in 100..116 { levels[0].pending.insert(key, 0); }
        levels[5].pending.insert(500, 63);
        let mut visible: VecDeque<_> = (300..316).map(|key| (3, key))
            .chain((100..116).map(|key| (0, key))).collect();
        let mut cached = None;
        assert_eq!(pop_pending(&mut levels, 5, &mut visible, &mut cached, None), Some((5, 500, 63)));
        // A partial frame stops after eight jobs without changing rank.
        for key in 300..308 {
            assert_eq!(pop_pending(&mut levels, 5, &mut visible, &mut cached, None), Some((3, key, 0)));
        }
        assert_eq!(visible.front(), Some(&(3, 308)));
        for key in 308..316 {
            assert_eq!(pop_pending(&mut levels, 5, &mut visible, &mut cached, None), Some((3, key, 0)));
        }
        assert_eq!(pop_pending(&mut levels, 5, &mut visible, &mut cached, None), Some((0, 100, 0)));
    }

    #[test]
    fn unavailable_visible_rank_does_not_starve_ordinary_admission() {
        let mut levels: Vec<Level> = (0..6).map(|_| Level::default()).collect();
        levels[0].pending.insert(100, 3);
        levels[3].pending.insert(300, 0);
        let mut visible = VecDeque::from([(3, 999), (3, 300)]);
        let mut cached = None;
        assert_eq!(pop_pending(&mut levels, 5, &mut visible, &mut cached, None), Some((3, 300, 0)));
        // Production may defer this key for an in-flight publication/alias.
        // It returns to its ordinary bucket, not the visible head this plan.
        levels[3].pending.insert(300, 4);
        assert_eq!(pop_pending(&mut levels, 5, &mut visible, &mut cached, None), Some((0, 100, 3)));
        assert_eq!(pop_pending(&mut levels, 5, &mut visible, &mut cached, None), Some((3, 300, 4)));
        assert_eq!(pop_pending(&mut levels, 5, &mut visible, &mut cached, None), None);
    }

    #[test]
    fn expired_visible_admission_deadline_keeps_remaining_rank() {
        let mut levels: Vec<Level> = (0..6).map(|_| Level::default()).collect();
        levels[0].pending.insert(100, 0);
        let mut visible = VecDeque::from([(3, 999), (0, 100)]);
        let mut cached = None;
        assert_eq!(pop_pending(&mut levels, 5, &mut visible, &mut cached,
            Some(std::time::Instant::now())), None);
        assert_eq!(visible.len(), 2);
        assert_eq!(pop_pending(&mut levels, 5, &mut visible, &mut cached, None), Some((0, 100, 0)));
    }

    #[test]
    fn visible_rank_replacement_and_source_expiry_discard_old_camera_order() {
        let (_, mut r, _, _) = edit_fixture();
        r.visible_admission.push_back((0, 123));
        r.prioritize_visible_blocks(std::iter::empty());
        assert!(r.visible_admission.is_empty());
        let at = std::time::Instant::now();
        r.prioritize_visible_blocks_from(std::iter::empty(), 10, 7, at);
        r.visible_admission.push_back((0, 123));
        r.set_visible_view(19, 7, at);
        r.refresh_visible_pending(None);
        assert!(r.visible_admission.is_empty());
        assert!(r.visible_rank_source.is_none());
        let (_, mut r, _, at) = visible_bridge_fixture();
        let keys = lease_block(&mut r, 1000, at);
        assert_eq!(r.visible_admission.len(), keys.len());
        r.set_visible_view(19, 7, at);
        r.refresh_visible_pending(None);
        assert!(r.transient_wanted(keys[0]), "the 32-frame ownership lease remains valid");
        assert!(keys.iter().all(|key| r.levels[0].pending.at.contains_key(key)));
        assert!(r.visible_admission.is_empty(), "lease reinsertion must not recreate expired 8-frame camera rank");
    }

    fn snapshot_update(serial: u64, keys: &[u64]) -> WindowUpdate {
        WindowUpdate { serial, snapshot: true,
            wanted: vec![(0, std::sync::Arc::new(keys.iter().copied().collect()))],
            levels: vec![LevelDiff { level: 0, active: true, radius: 100.0,
                adds: keys.iter().map(|&key| (0.0, key)).collect(), ..Default::default() }],
            ..Default::default() }
    }

    #[test]
    fn early_unchanged_authority_supersedes_only_its_levels_leases() {
        let (_, mut r, _, at) = visible_bridge_fixture();
        assert!(r.grid.levels() > 3);
        let fine = pack(key0(crate::grid::PLANE_FACE, 0, 1000), 1000);
        let far = pack(key0(crate::grid::PLANE_FACE, 3, 128), 128);
        r.levels[3].wanted = Some(Default::default());
        r.prioritize_visible_blocks_from([fine, far].map(|key| (key as u32, (key >> 32) as u32)), 10, 7, at);
        r.refresh_visible_pending(None);
        assert!(r.transient_wanted(fine) && r.transient_wanted(far));
        r.apply(WindowUpdate { serial: 2, partial: true, processed_levels: 0b111,
            snapshot: true, ..Default::default() });
        assert_eq!(&r.applied_levels[..3], &[2, 2, 2], "unchanged levels still acknowledge authority");
        assert_eq!(r.applied, 1, "unpublished far levels prevent whole-world readiness");
        assert!(!r.idle());
        assert!(!r.transient_wanted(fine) && r.transient_wanted(far));
        let mut work = FrameWork::default();
        r.retire_visible_leases(&mut work, &|| false);
        assert!(!r.visible_leases.contains_key(&fine));
        assert!(r.visible_leases.contains_key(&far) && r.transient_wanted(far));
        assert_eq!(r.snapshot_epoch, 0, "authority-only no-op does not rescan unchanged membership");
        r.apply(WindowUpdate { serial: 2, processed_levels: ((1 << r.grid.levels()) - 1) & !0b111,
            snapshot: true, ..Default::default() });
        assert_eq!(r.applied, 2);
        assert!(!r.transient_wanted(far));
        r.retire_visible_leases(&mut work, &|| false);
        assert!(r.idle());
    }

    #[test]
    fn same_serial_far_snapshot_restarts_retirement_after_fine_scan() {
        for far_diff in [false, true] {
            let (planet, mut r, fine, _) = edit_fixture();
            let far = pack(key0(crate::grid::PLANE_FACE, 3, 128), 128);
            r.residents.insert(far, Resident { record: 1, blocks: false, ..Default::default() });
            r.block_conflicts += 1;
            r.next_record = 2;
            r.levels[3].wanted = Some(std::sync::Arc::new([far].into_iter().collect()));
            r.requested = 2;
            r.apply(WindowUpdate { serial: 2, partial: true, processed_levels: 0b111, snapshot: true,
                wanted: vec![(0, Default::default())],
                levels: vec![LevelDiff { level: 0, ..Default::default() }], ..Default::default() });
            let mut work = FrameWork::default();
            r.apply_snapshot(&mut work, &|| false);
            assert_eq!(r.retire_finished_epoch, 1);
            assert!(!r.residents.contains_key(fine) && r.residents.contains_key(far));
            assert!(r.diffs.iter().all(VecDeque::is_empty));
            assert!(!r.idle(), "the completed near scan cannot acknowledge far authority");
            r.apply(WindowUpdate { serial: 2, processed_levels: ((1 << r.grid.levels()) - 1) & !0b111,
                snapshot: true, wanted: vec![(3, Default::default())],
                levels: if far_diff { vec![LevelDiff { level: 3, ..Default::default() }] } else { vec![] },
                ..Default::default() });
            assert_eq!(r.snapshot_epoch, 2);
            assert!(!r.idle(), "wanted-only authority still owns an unfinished retirement scan");
            r.apply_snapshot(&mut work, &|| false);
            assert!(!r.residents.contains_key(far), "far owners must retire at the same request serial");
            assert_eq!(r.retire_finished_epoch, 2);
            assert!(r.diffs.iter().all(VecDeque::is_empty));
            assert!(r.catching_up.iter().all(|&count| count == 0));
            assert_eq!((r.queued_delta_bytes, r.queued_delta_ops), (0, 0));
            assert_eq!(work.evictions, vec![0, 1]);
            assert!(r.idle());
            table_is_exact(&r);
            drop(planet);
        }
    }

    #[test]
    fn ranged_authority_accepts_old_far_without_rolling_back_new_fine() {
        let (_, mut r, _, _) = edit_fixture();
        let fine = pack(key0(crate::grid::PLANE_FACE, 0, 1000), 1000);
        let obsolete = pack(key0(crate::grid::PLANE_FACE, 0, 2000), 1000);
        let far = pack(key0(crate::grid::PLANE_FACE, 3, 128), 128);
        r.requested = 14;
        let mut near = snapshot_update(14, &[fine]);
        near.partial = true;
        near.processed_levels = 0b111;
        r.apply(near);
        r.apply(WindowUpdate { serial: 10, processed_levels: ((1 << r.grid.levels()) - 1) & !0b111,
            snapshot: true, wanted: vec![(3, std::sync::Arc::new([far].into_iter().collect()))],
            levels: vec![LevelDiff { level: 3, active: true, adds: vec![(0.0, far)], ..Default::default() }],
            ..Default::default() });
        assert_eq!(r.applied, 10);
        assert_eq!((r.stats.fine_applied_serial, r.stats.far_applied_serial), (14, Some(10)));
        assert!(r.current_wanted(fine) && r.current_wanted(far));
        let before = (r.snapshot_epoch, r.queued_delta_bytes, r.queued_delta_ops);
        let mut stale = snapshot_update(13, &[obsolete]);
        stale.partial = true;
        stale.processed_levels = 0b111;
        r.apply(stale);
        assert_eq!(before, (r.snapshot_epoch, r.queued_delta_bytes, r.queued_delta_ops));
        assert!(r.current_wanted(fine) && !r.current_wanted(obsolete));
        assert_eq!(r.diffs[0][0].diff.adds, vec![(0.0, fine)]);
        assert!(!r.idle());
        r.apply(WindowUpdate { serial: 14, processed_levels: ((1 << r.grid.levels()) - 1) & !0b111,
            snapshot: true, ..Default::default() });
        assert_eq!(r.applied, 14);
        // A duplicate chunk cannot replace the installed membership either.
        let mut duplicate = snapshot_update(14, &[obsolete]);
        duplicate.partial = true;
        duplicate.processed_levels = 0b111;
        r.apply(duplicate);
        assert!(r.current_wanted(fine) && !r.current_wanted(obsolete));
    }

    #[test]
    fn initial_explicit_zero_serial_installs_demand_without_allowing_duplicate_rollback() {
        let (_, mut r, _, _) = edit_fixture();
        let current = pack(key0(crate::grid::PLANE_FACE, 0, 1000), 1000);
        let stale = pack(key0(crate::grid::PLANE_FACE, 0, 2000), 1000);
        let mut first = snapshot_update(0, &[current]);
        first.processed_levels = 1;
        r.apply(first);
        assert!(r.current_wanted(current));
        assert_eq!(r.queued_delta_ops, 1);
        let mut duplicate = snapshot_update(0, &[stale]);
        duplicate.processed_levels = 1;
        r.apply(duplicate);
        assert!(r.current_wanted(current) && !r.current_wanted(stale));
        assert_eq!(r.queued_delta_ops, 1);
        assert_eq!(r.snapshot_epoch, 1);
    }

    #[test]
    fn installed_enqueue_authority_rejects_old_capture_but_allows_new_bridge() {
        let (_, mut r, _, at) = visible_bridge_fixture();
        let block = pack(key0(crate::grid::PLANE_FACE, 0, 1000), 1000);
        let issued = at + std::time::Duration::from_millis(1);
        r.apply(WindowUpdate { serial: 2, partial: true, processed_levels: 0b111,
            issued_at: Some(issued), ..Default::default() });
        r.requested = 3;
        r.prioritize_visible_blocks_from([(block as u32, (block >> 32) as u32)], 10, 7, at);
        r.refresh_visible_pending(None);
        assert!(r.visible_leases.is_empty() && r.levels[0].pending.is_empty(),
            "the perpetually open next-request gap cannot revive pre-authority GPU demand");
        r.set_visible_view(11, 7, issued);
        r.prioritize_visible_blocks_from([(block as u32, (block >> 32) as u32)], 11, 7, issued);
        r.refresh_visible_pending(None);
        assert!(r.transient_wanted(block), "capture after installed authority can bridge request 3");
        assert_eq!(r.visible_leases[&block].serial, 3);
        // A newer far enqueue stamp must neither supersede this fine lease
        // nor prevent prioritizing complete current fine wanted membership.
        r.apply(WindowUpdate { serial: 3, processed_levels: 1 << 3,
            issued_at: Some(issued + std::time::Duration::from_millis(1)), ..Default::default() });
        assert!(r.transient_wanted(block));
        let (keys, count) = r.visible_columns(block).unwrap();
        r.apply(WindowUpdate { serial: 3, partial: true, processed_levels: 0b111,
            issued_at: Some(issued + std::time::Duration::from_millis(2)),
            wanted: vec![(0, std::sync::Arc::new(keys[..count].iter().copied().collect()))],
            levels: vec![LevelDiff { level: 0, active: true, ..Default::default() }], ..Default::default() });
        r.retire_visible_leases(&mut FrameWork::default(), &|| false);
        r.prioritize_visible_blocks_from([(block as u32, (block >> 32) as u32)], 11, 7, issued);
        r.refresh_visible_pending(None);
        assert!(r.visible_leases.is_empty());
        assert!(keys[..count].iter().all(|key| r.levels[0].pending.at.contains_key(key)),
            "ordinary current wanted priority does not require a newer capture");
    }

    #[test]
    fn range_latency_and_window_lag_keep_unavailable_distinct_from_zero() {
        let (planet, mut r, _, eye) = edit_fixture();
        let issued = std::time::Instant::now() - std::time::Duration::from_millis(100);
        r.apply(WindowUpdate { serial: 1, partial: true, processed_levels: 0b111,
            issued_at: Some(issued), ..Default::default() });
        assert!(r.stats.fine_apply_age_ms.is_some_and(|age| age >= 100.0));
        assert_eq!(r.stats.far_apply_age_ms, None);
        let fine_age = r.stats.fine_apply_age_ms;
        r.apply(WindowUpdate { serial: 1, processed_levels: 1 << 3,
            issued_at: Some(std::time::Instant::now() + std::time::Duration::from_secs(10)), ..Default::default() });
        assert_eq!(r.stats.fine_apply_age_ms, fine_age);
        assert_eq!(r.stats.far_apply_age_ms, None, "future clocks cannot report a fake zero latency");
        r.plan(&planet, eye, 1.0, 0);
        assert_eq!(r.stats.fine_window_lag_m, None);
        r.levels[0].active = true;
        r.levels[0].center = DVec3::new(3.0, 0.0, 4.0);
        r.plan(&planet, eye, 1.0, 0);
        assert_eq!(r.stats.fine_window_lag_m, Some(5.0));
    }

    #[test]
    fn moving_snapshots_bound_history_and_converge_without_generating_obsolete_keys() {
        let (planet, mut r, old, eye) = edit_fixture();
        let face = crate::grid::PLANE_FACE;
        let mut latest = Vec::new();
        for serial in 1..=500 {
            let i = 1000 + (serial % 50) as i32 * 4;
            latest = (1000..1004).flat_map(|j| (i..i + 4).map(move |x| pack(key0(face, 0, x), j))).collect();
            r.apply(snapshot_update(serial, &latest));
            assert_eq!(r.diffs.iter().map(VecDeque::len).sum::<usize>(), 1);
            assert_eq!(r.queued_delta_ops, 16);
            assert_eq!(r.queued_delta_bytes, 16 * std::mem::size_of::<(f32, u64)>());
            assert!(r.levels[0].pending.is_empty(), "superseded pending history must not accumulate");
        }
        r.requested = r.applied;
        let work = r.plan(&planet, eye, 1.0, 16);
        assert_eq!(work.job_keys.iter().copied().collect::<FxHashSet<_>>(), latest.iter().copied().collect());
        assert_eq!(work.evictions, vec![0]);
        assert!(!r.residents.contains_key(old));
        r.complete_jobs(work.job_keys.iter().map(|&key| (key, 0)));
        assert!(r.idle());
        assert_eq!((r.queued_delta_bytes, r.queued_delta_ops), (0, 0));
        assert!(latest.iter().all(|&key| r.residents.get(key).unwrap().blocks));
    }

    #[test]
    fn snapshot_retirement_rechecks_backshifted_slots_and_quarantines_publications() {
        let (planet, _, _, _) = edit_fixture();
        let mut r = Residency::new(*planet.grid(), Capacity { table_bits: 8, ..Default::default() });
        let keys: Vec<_> = (1000..5000).map(|i| pack(key0(crate::grid::PLANE_FACE, 0, i), 0))
            .filter(|key| slot_hash(*key as u32, (*key >> 32) as u32) & 255 == 10).take(3).collect();
        assert_eq!(keys.len(), 3);
        for (record, &key) in keys.iter().enumerate() {
            r.residents.insert(key, Resident { record: record as u32, ..Default::default() });
            r.block_conflicts += 1;
        }
        r.publishing.insert(keys[0], EditPublication {
            record: 0, previous: None, next: None, evicted: false, initial_bucket: Some(0),
        });
        r.apply(snapshot_update(1, &keys[1..2]));
        r.retire_slot = 10;
        r.retire_started_epoch = 1;
        let mut work = FrameWork::default();
        r.apply_snapshot(&mut work, &|| false);
        assert_eq!(work.evictions.len(), 2);
        assert_eq!(r.residents.iter().map(|(key, _)| key).collect::<Vec<_>>(), vec![keys[1]]);
        assert!(r.publishing[&keys[0]].evicted);
        r.complete_jobs([(keys[0], 3)]);
        assert!(r.publishing.is_empty() && r.initial_retries.is_empty());
        assert_eq!(r.catching_up[0], 0);
    }

    #[test]
    fn snapshot_deadline_slices_retirement_and_additions_without_skipping_work() {
        let (planet, _, _, _) = edit_fixture();
        let face = crate::grid::PLANE_FACE;
        let mut r = Residency::new(*planet.grid(), Capacity { table_bits: 8, ..Default::default() });
        let old: Vec<_> = (1000..5000).map(|i| pack(key0(face, 0, i), 0))
            .filter(|key| slot_hash(*key as u32, (*key >> 32) as u32) & 255 == 10).take(3).collect();
        assert_eq!(old.len(), 3);
        for (record, &key) in old.iter().enumerate() {
            r.residents.insert(key, Resident { record: record as u32, ..Default::default() });
            r.block_conflicts += 1;
        }
        r.publishing.insert(old[0], EditPublication {
            record: 0, previous: None, next: None, evicted: false, initial_bucket: Some(0),
        });
        r.apply(snapshot_update(1, &old[1..2]));
        r.retire_slot = 10;
        let calls = std::cell::Cell::new(0);
        let mut work = FrameWork::default();
        r.apply_snapshot(&mut work, &|| { calls.set(calls.get() + 1); calls.get() > 2 });
        assert_eq!(work.evictions, vec![0]);
        assert_eq!(r.retire_slot, 10, "resume at the backshifted slot after the last atomic eviction");
        assert_eq!(r.diffs[0].front().unwrap().added, 0);
        assert_eq!(r.queued_delta_ops, 1, "expired retirement must not consume the additions batch");
        assert!(r.publishing[&old[0]].evicted);
        r.apply_snapshot(&mut work, &|| false);
        assert_eq!(work.evictions, vec![0, 2]);
        assert!(r.residents.contains_key(old[1]));
        r.complete_jobs([(old[0], 3)]);
        assert!(r.publishing.is_empty() && r.initial_retries.is_empty());
        table_is_exact(&r);

        let mut r = Residency::new(*planet.grid(), Capacity { table_bits: 10, ..Default::default() });
        let wanted: Vec<_> = (1000..1256).map(|i| pack(key0(face, 0, i), 0)).collect();
        r.residents.insert(wanted[2], Resident { record: 0, ..Default::default() });
        r.block_conflicts += 1;
        r.publishing.insert(wanted[2], EditPublication {
            record: 0, previous: None, next: None, evicted: false, initial_bucket: Some(0),
        });
        r.apply(snapshot_update(1, &wanted));
        r.retire_finished_epoch = r.snapshot_epoch;
        let calls = std::cell::Cell::new(0);
        r.apply_snapshot(&mut FrameWork::default(), &|| { calls.set(calls.get() + 1); calls.get() > 7 });
        assert_eq!(r.diffs[0].front().unwrap().added, 5);
        assert_eq!(r.queued_delta_ops, wanted.len() - 5, "only inspected additions advance the cursor");
        assert_eq!(r.levels[0].pending.keys().copied().collect::<FxHashSet<_>>(),
            wanted[..5].iter().copied().filter(|key| *key != wanted[2]).collect());
        assert!(!r.publishing[&wanted[2]].evicted);
        r.apply_snapshot(&mut FrameWork::default(), &|| false);
        assert!(r.diffs[0].is_empty() && r.catching_up[0] == 0);
        assert_eq!((r.queued_delta_ops, r.queued_delta_bytes), (0, 0));
        assert_eq!(r.levels[0].pending.keys().copied().collect::<FxHashSet<_>>(),
            wanted.iter().copied().filter(|key| *key != wanted[2]).collect());
        assert!(r.residents.contains_key(wanted[2]) && !r.publishing[&wanted[2]].evicted);
        table_is_exact(&r);
    }

    #[test]
    fn snapshot_sparse_retirement_finishes_once_while_adds_continue_and_restarts_on_new_demand() {
        let (planet, _, _, _) = edit_fixture();
        let mut r = Residency::new(*planet.grid(), Capacity { table_bits: 10, ..Default::default() });
        let face = crate::grid::PLANE_FACE;
        let keys: Vec<_> = (1000..2024).map(|i| pack(key0(face, 0, i), 0)).collect();
        r.residents.insert(keys[0], Resident { record: 0, ..Default::default() });
        r.block_conflicts += 1;
        r.apply(snapshot_update(1, &keys));
        let round = |r: &mut Residency, work: &mut FrameWork| {
            let calls = std::cell::Cell::new(0);
            r.apply_snapshot(work, &|| { calls.set(calls.get() + 1); calls.get() > 257 });
        };
        let mut work = FrameWork::default();
        round(&mut r, &mut work);
        assert_eq!(r.retire_finished_epoch, 1,
            "one resident plus16 bitmap words must finish within128 bounded retirement steps");
        let remaining = r.queued_delta_ops;
        assert!(remaining > 0 && remaining < keys.len(), "the long fullwanted list remains partially admitted");
        assert_eq!(r.retire_slot, 0);
        round(&mut r, &mut work);
        assert_eq!(r.retire_slot, 0, "unchanged residents must not be rescanned while adds continue");
        assert_eq!(r.retire_finished_epoch, 1);
        assert!(r.queued_delta_ops < remaining);
        assert!(work.evictions.is_empty());

        // A subsequent serial drops the resident; completed-pass state must
        // not prevent this new retirement, even while the old adds were long.
        r.apply(snapshot_update(2, &keys[1..]));
        round(&mut r, &mut work);
        assert_eq!(r.retire_finished_epoch, 2);
        assert_eq!(work.evictions, vec![0]);
        assert!(!r.residents.contains_key(keys[0]));
    }

    #[test]
    fn visible_alias_retirement_releases_obsolete_owners_before_full_table_scan() {
        let (planet, _, _, _) = edit_fixture();
        let mut r = Residency::new(*planet.grid(), Capacity { table_bits: 14, ..Default::default() });
        let face = crate::grid::PLANE_FACE;
        let old: Vec<_> = (1000..1004).flat_map(|j| (1000..1004).map(move |i| pack(key0(face, 0, i), j))).collect();
        let current: Vec<_> = old.iter().map(|&key| {
            let (_, _, i, j) = unpack(key);
            pack(key0(face, 0, i + 512), j as u32)
        }).collect();
        for (record, &key) in old.iter().enumerate() {
            assert!(r.acquire_blocks(key, &mut FrameWork::default()));
            r.residents.insert(key, Resident { record: record as u32, blocks: true, ..Default::default() });
        }
        r.apply(snapshot_update(1, &current));
        r.prioritize_visible_blocks([(current[0] as u32, (current[0] >> 32) as u32)]);
        r.refresh_visible_pending(None);
        let rounds = std::cell::Cell::new(0);
        let mut work = FrameWork::default();
        r.apply_snapshot(&mut work, &|| { rounds.set(rounds.get() + 1); rounds.get() > 257 });
        assert_eq!(work.evictions.len(), 16, "the obsolete4x4 owner must retire in the first bounded round");
        assert!(r.retire_finished_epoch < 1, "fixture must not rely on a whole-table scan");
        assert!(current.iter().all(|&key| !r.blocks_conflict(key)));
        assert!(current.iter().all(|key| r.levels[0].pending.at.contains_key(key)));
        assert!(r.catching_up[0] > 0);
    }

    #[test]
    fn tier3_alias_retirement_keeps_its_cursor_and_protects_returned_columns() {
        let (planet, _, _, _) = edit_fixture();
        let mut r = Residency::new(*planet.grid(), Capacity { table_bits: 12, ..Default::default() });
        let face = crate::grid::PLANE_FACE;
        let old = pack(key0(face, 0, 960), 960);
        let returned = pack(key0(face, 0, 1000), 1000);
        for (record, key) in [old, returned].into_iter().enumerate() {
            assert!(r.acquire_blocks(key, &mut FrameWork::default()));
            r.residents.insert(key, Resident { record: record as u32, blocks: true, ..Default::default() });
        }
        let current: Vec<_> = (960..964).flat_map(|j| (1472..1476).map(move |i| pack(key0(face, 0, i), j))).collect();
        let mut wanted = current.clone();
        wanted.push(returned);
        r.apply(snapshot_update(1, &wanted));
        r.queue_obsolete_owners(current[0]);
        let round = |r: &mut Residency, work: &mut FrameWork| {
            for _ in 0..16 { r.retire_obsolete_owner_step(work); }
        };
        let mut work = FrameWork::default();
        round(&mut r, &mut work);
        let first = r.obsolete_owners.front().unwrap();
        assert_eq!(first.owner.2, 3);
        let offset = first.offset;
        assert!(offset > 0 && offset < 4096);
        round(&mut r, &mut work);
        assert!(r.obsolete_owners.front().unwrap().offset > offset,
            "deadline slices must continue the tier3 scan instead of restarting its first row");
        r.apply_snapshot(&mut work, &|| false);
        assert!(r.residents.contains_key(returned), "latest wanted membership protects returned terrain through every owner tier");
        assert!(!r.residents.contains_key(old));
        assert!(r.blocks_conflict(current[0]), "a still wanted old owner cannot be silently reassigned");
        r.apply(snapshot_update(2, &current));
        r.queue_obsolete_owners(current[0]);
        r.apply_snapshot(&mut work, &|| false);
        assert!(!r.residents.contains_key(returned));
        assert!(!r.blocks_conflict(current[0]));
        assert_eq!(r.catching_up[0], 0);
    }

    #[test]
    fn sparse_tier3_owner_skips_absent_groups_before_the_alias_can_publish() {
        let (planet, _, _, _) = edit_fixture();
        let mut r = Residency::new(*planet.grid(), Capacity::default());
        let face = crate::grid::PLANE_FACE;
        // Last column of a 64x64 owner: scalar retirement would require all
        // 4096 probes even though only this last tier-1 group is resident.
        let old = pack(key0(face, 0, 1023), 1023);
        let incoming = pack(key0(face, 0, 1535), 1023);
        assert!(r.acquire_blocks(old, &mut FrameWork::default()));
        r.residents.insert(old, Resident { record: 0, blocks: true, ..Default::default() });
        r.levels[0].wanted = Some(std::sync::Arc::new([incoming].into_iter().collect()));
        r.obsolete_owners.push_back(OwnerRetirement { owner: (0, face, 3, 15, 15), offset: 0 });
        assert!(r.blocks_conflict(incoming));
        let mut work = FrameWork::default();
        let mut steps = 0;
        while !r.obsolete_owners.is_empty() {
            assert!(r.retire_obsolete_owner_step(&mut work));
            steps += 1;
            assert!(steps <= 272, "sparse owner must skip absent tier-1 groups");
        }
        assert_eq!(steps, 271);
        eprintln!("sparse tier-3 retirement: {steps} probes, scalar traversal: 4096");
        assert_eq!(work.evictions, vec![0]);
        assert!(!r.residents.contains_key(old));
        assert!(!r.blocks_conflict(incoming));
        assert!(r.blocks.is_empty() && r.block_owner.is_empty());
    }

    #[test]
    fn protected_alias_requeues_cannot_starve_bounded_snapshot_retirement() {
        let (planet, _, _, _) = edit_fixture();
        let mut r = Residency::new(*planet.grid(), Capacity { table_bits: 14, ..Default::default() });
        let face = crate::grid::PLANE_FACE;
        let protected = pack(key0(face, 0, 1000), 1000);
        assert!(r.acquire_blocks(protected, &mut FrameWork::default()));
        r.residents.insert(protected, Resident { record: 0, blocks: true, ..Default::default() });
        let current: Vec<_> = (960..964).flat_map(|j| (1472..1476).map(move |i| pack(key0(face, 0, i), j))).collect();
        let mut wanted = current.clone();
        wanted.push(protected);
        r.apply(snapshot_update(1, &wanted));
        r.queue_obsolete_owners(current[0]);
        assert_eq!(r.obsolete_owners.len(), 1);
        assert_eq!(r.obsolete_owners.front().unwrap().owner.2, 3);
        let owner = r.obsolete_owners.front().unwrap().owner;
        let slot = r.blocks[&owner].slot;
        let mut work = FrameWork::default();
        for _ in 0..128 {
            let calls = std::cell::Cell::new(0);
            // Grant at most one full128-step retirement/addition pair,
            // with refresh between budget slices.
            r.apply_snapshot(&mut work, &|| { calls.set(calls.get() + 1); calls.get() > 257 });
            r.queue_obsolete_owners(current[0]);
        }
        // Under owner-only retirement, refresh repeatedly requeues the same
        // protected task. The normal cursor never gets a step and this
        // serial cannot complete.
        assert_eq!(r.retire_finished_epoch, 1);
        assert!(r.diffs[0].is_empty() && r.catching_up[0] == 0);
        assert_eq!((r.queued_delta_ops, r.queued_delta_bytes), (0, 0));
        assert!(work.evictions.is_empty());
        assert!(r.residents.contains_key(protected));
        assert_eq!(r.blocks[&owner].refs, 1);
        assert_eq!(r.block_owner[&slot], owner);
        assert!(r.blocks_conflict(current[0]), "a protected owner must never be reassigned");
    }

    #[test]
    fn completed_snapshot_serial_returns_the_full_retirement_slice_to_owners() {
        let (planet, _, _, _) = edit_fixture();
        let mut r = Residency::new(*planet.grid(), Capacity { table_bits: 14, ..Default::default() });
        let face = crate::grid::PLANE_FACE;
        let protected = pack(key0(face, 0, 1000), 1000);
        assert!(r.acquire_blocks(protected, &mut FrameWork::default()));
        r.residents.insert(protected, Resident { record: 0, blocks: true, ..Default::default() });
        let conflict = pack(key0(face, 0, 1472), 960);
        let mut wanted: Vec<_> = (1000..2024).map(|i| pack(key0(face, 0, i), 0)).collect();
        wanted.extend([protected, conflict]);
        r.apply(snapshot_update(1, &wanted));
        r.retire_finished_epoch = r.snapshot_epoch;
        r.queue_obsolete_owners(conflict);
        assert_eq!(r.obsolete_owners.len(), 1);
        assert_eq!(r.obsolete_owners.front().unwrap().owner.2, 3);
        let calls = std::cell::Cell::new(0);
        let mut work = FrameWork::default();
        r.apply_snapshot(&mut work, &|| { calls.set(calls.get() + 1); calls.get() > 257 });
        assert!(r.obsolete_owners.front().unwrap().offset >= 128,
            "the same slice may skip absent groups but must still advance");
        assert_eq!(r.retire_finished_epoch, 1);
        assert_eq!(r.retire_slot, 0, "a completed serial needs no normal rescan");
        assert_eq!(r.queued_delta_ops, wanted.len() - 128);
        assert!(work.evictions.is_empty() && r.residents.contains_key(protected));
    }

    #[test]
    fn visible_feedback_promotes_only_complete_current_blocks_without_ownership_changes() {
        let (_, mut r, resident, _) = edit_fixture();
        let face = crate::grid::PLANE_FACE;
        let block = |i| (0..4).flat_map(move |j| (i..i + 4).map(move |x| pack(key0(face, 0, x), j))).collect::<Vec<_>>();
        let visible = block(1000);
        let incomplete = block(2000);
        let stale = block(3000);
        let wanted = visible.iter().chain(&incomplete[..15]).copied().collect::<FxHashSet<_>>();
        r.levels[0].active = true;
        r.levels[0].wanted = Some(std::sync::Arc::new(wanted.clone()));
        r.levels[0].pending.insert(visible[0], 50);
        let requests = [visible[0], incomplete[0], stale[0]];
        r.prioritize_visible_blocks(requests.map(|key| (key as u32, (key >> 32) as u32)));
        r.refresh_visible_pending(None);
        assert_eq!(r.levels[0].pending.keys().copied().collect::<FxHashSet<_>>(), visible.iter().copied().collect());
        assert!(visible.iter().all(|key| r.levels[0].pending.at[key].0 == 0));
        assert_eq!(r.levels[0].wanted.as_deref(), Some(&wanted));
        assert_eq!(r.residents.len(), 1);
        assert!(r.residents.contains_key(resident) && r.publishing.is_empty() && r.blocks.is_empty());
    }

    #[test]
    fn visible_feedback_obeys_fixed_work_cap_and_elapsed_deadline() {
        let (_, mut r, _, _) = edit_fixture();
        let face = crate::grid::PLANE_FACE;
        let blocks: Vec<_> = (0..65).map(|b| pack(key0(face, 0, 1000 + b * 4), 0)).collect();
        let wanted = blocks.iter().flat_map(|&key| {
            let (_, _, i, j) = unpack(key);
            (j..j + 4).flat_map(move |y| (i..i + 4).map(move |x| pack(key0(face, 0, x), y as u32)))
        }).collect::<FxHashSet<_>>();
        r.levels[0].active = true;
        r.levels[0].wanted = Some(std::sync::Arc::new(wanted));
        let requests = || blocks.iter().map(|&key| (key as u32, (key >> 32) as u32));
        r.prioritize_visible_blocks(requests());
        r.refresh_visible_pending(Some(std::time::Instant::now()));
        assert!(r.levels[0].pending.is_empty(), "feedback cannot spend admission's elapsed reserve");
        r.prioritize_visible_blocks(requests());
        r.refresh_visible_pending(None);
        assert_eq!(r.levels[0].pending.len(), 64 * 16);
        assert!(!r.levels[0].pending.at.contains_key(&blocks[64]));
        r.levels[0].active = false;
        r.prioritize_visible_blocks([(blocks[64] as u32, (blocks[64] >> 32) as u32)]);
        r.refresh_visible_pending(None);
        assert_eq!(r.levels[0].pending.len(), 64 * 16, "inactive feedback cannot resurrect a level");
    }

    fn visible_order_fixture() -> (std::sync::Arc<Planet>, Residency, DVec3, Vec<u64>, Vec<u64>) {
        let (planet, mut r, old, eye) = edit_fixture();
        let (face, _, i, j) = unpack(old);
        r.evict(old, &mut FrameWork::default());
        let block = |offset| {
            let key = pack(key0(face, 0, (i & !3) + offset), (j & !3) as u32);
            let (keys, count) = r.visible_columns(key).unwrap();
            keys[..count].to_vec()
        };
        // Both blocks are outside the camera-anchor refresh neighborhood.
        let near = block(40);
        let far = block(80);
        let point = |key| {
            let (face, _, i, j) = unpack(key);
            planet.grid().ground_point(face, f64::from((i + 2) * BRICK), f64::from((j + 2) * BRICK))
        };
        assert!(planet.grid().ground_distance(eye, point(near[0])) < planet.grid().ground_distance(eye, point(far[0])));
        r.levels[0].wanted = Some(std::sync::Arc::new(near.iter().chain(&far).copied().collect()));
        r.levels[0].active = true;
        for &key in near.iter().chain(&far) { r.levels[0].pending.insert(key, 0); }
        (planet, r, eye, near, far)
    }

    fn batch_admission_fixture() -> (std::sync::Arc<Planet>, Residency, Vec<u64>, Vec<u64>) {
        let (planet, mut r, _, near, far) = visible_order_fixture();
        r.levels[0].pending.remove(near[0]); // pop_pending has selected the first key.
        r.visible_admission.extend(near[1..].iter().chain(&far).map(|&key| (0, key)));
        (planet, r, near, far)
    }

    #[test]
    fn complete_ordinary_batch_uses_selected_bucket_without_visible_feedback() {
        let (planet, mut r, eye, near, far) = visible_order_fixture();
        let work = r.plan(&planet, eye, 1.0, 16);
        assert_eq!(work.job_keys, far, "same-bucket ordinary LIFO selection remains authoritative");
        assert_eq!(r.stats.admission_batched_columns, 16);
        assert_eq!(r.stats.admission_attempts, 16);
        assert!(r.visible_admission.is_empty() && r.visible_leases.is_empty());
        assert!(near.iter().all(|&key| !r.residents.contains_key(key)));
        assert!(work.jobs.iter().all(|job| job.edits == 0 && job.flags == 0));
        assert_eq!(work.jobs.iter().map(|job| job.record).collect::<FxHashSet<_>>().len(), 16);
        assert!(r.blocks.values().all(|block| block.refs == 16));
        r.complete_jobs(work.job_keys.iter().map(|&key| (key, 0)));
        for key in work.job_keys { r.evict(key, &mut FrameWork::default()); }
        assert!(r.blocks.is_empty() && r.block_owner.is_empty());
        table_is_exact(&r);
    }

    #[test]
    fn ordinary_batch_exceeds_visible_capacity_and_obeys_partial_job_budget() {
        for budget in [1536, 1540] {
            let (planet, fixture, old, eye) = edit_fixture();
            let mut r = Residency::new(*planet.grid(), Capacity { table_bits: 12, ..Default::default() });
            r.last_request = fixture.last_request.clone();
            let (face, _, i, j) = unpack(old);
            let mut wanted = FxHashSet::default();
            for block in 0..100 {
                let first = pack(key0(face, 0, (i & !3) + 40 + block * 4), (j & !3) as u32);
                let (keys, count) = r.visible_columns(first).unwrap();
                assert_eq!(count, 16);
                for key in keys { wanted.insert(key); r.levels[0].pending.insert(key, 0); }
            }
            r.levels[0].active = true;
            r.levels[0].wanted = Some(std::sync::Arc::new(wanted.clone()));
            let work = r.plan(&planet, eye, 1.0, budget);
            assert_eq!(work.jobs.len(), budget);
            assert_eq!(r.stats.admission_batched_columns, 1536,
                "ordinary generation must not stop batching at visible feedback's64-block cap");
            assert_eq!(r.levels[0].pending.len(), wanted.len() - budget);
            assert!(work.job_keys.iter().all(|key| wanted.contains(key)));
            assert_eq!(work.job_keys.iter().copied().collect::<FxHashSet<_>>().len(), budget);
            assert!(r.visible_admission.is_empty() && r.visible_leases.is_empty());
            table_is_exact(&r);
            r.complete_jobs(work.job_keys.iter().map(|&key| (key, 0)));
            for key in work.job_keys { r.evict(key, &mut FrameWork::default()); }
            assert!(r.blocks.is_empty() && r.block_owner.is_empty());
        }
    }

    #[test]
    fn ordinary_batch_rejects_partial_mixed_priority_owned_or_edited_demand() {
        for case in 0..10 {
            let (mut planet, mut r, _, near, _) = visible_order_fixture();
            let selected = near[15];
            r.levels[0].pending.remove(selected);
            let mut budget = 16;
            let mut deadline = None;
            match case {
                0 => { r.levels[0].pending.remove(near[7]); }
                1 => { r.levels[0].pending.insert(near[7], 1); }
                2 => { std::sync::Arc::make_mut(r.levels[0].wanted.as_mut().unwrap()).remove(&near[7]); }
                3 => { r.residents.insert(near[7], Resident { record: 0, ..Default::default() }); }
                4 => { r.publishing.insert(near[7], EditPublication {
                    record: 0, previous: None, next: None, evicted: false, initial_bucket: Some(0),
                }); }
                5 => { r.visible_admission.push_back((0, near[7])); }
                6 => {
                    let top = r.grid.levels() - 1;
                    r.levels[top as usize].pending.insert(pack(key0(0, top, 0), 0), 0);
                }
                7 => budget = 15,
                8 => deadline = Some(std::time::Instant::now()),
                9 => {
                    let (face, _, i, j) = unpack(near[0]);
                    let center = planet.grid().ground_point(face, f64::from(i * BRICK) + 0.5, f64::from(j * BRICK) + 0.5);
                    std::sync::Arc::make_mut(&mut planet).apply(crate::edits::Brush {
                        center: center.to_array(), ..test_brush(0.2)
                    }).unwrap();
                }
                _ => unreachable!(),
            }
            let before = (r.residents.len(), r.publishing.len(), r.levels[0].pending.len(),
                r.next_record, r.free_records.clone(), r.visible_admission.clone());
            let mut work = FrameWork::default();
            assert!(!r.admit_pending_block(&planet, 0, selected, 0, budget, deadline, false, &mut work), "case {case}");
            assert!(work.jobs.is_empty() && work.block_inits.is_empty() && work.table_writes.is_empty());
            assert_eq!(before, (r.residents.len(), r.publishing.len(), r.levels[0].pending.len(),
                r.next_record, r.free_records.clone(), r.visible_admission.clone()), "case {case}");
        }
    }

    #[test]
    fn complete_visible_batch_preserves_individual_publications_and_reference_release() {
        let (planet, mut r, near, far) = batch_admission_fixture();
        let mut work = FrameWork::default();
        assert!(r.admit_visible_block(&planet, 0, near[0], 0, 16, None, &mut work));
        assert_eq!(work.job_keys, near);
        assert_eq!(r.visible_admission.iter().map(|&(_, key)| key).collect::<Vec<_>>(), far);
        assert_eq!(work.jobs.iter().map(|job| job.record).collect::<FxHashSet<_>>().len(), 16);
        assert_eq!(work.table_writes.len(), 16);
        assert_eq!(r.blocks.len(), 3);
        assert!(r.blocks.values().all(|block| block.refs == 16));
        assert!(work.jobs.iter().all(|job| job.edits == 0 && job.flags == 0));
        table_is_exact(&r);
        // One failed GPU publication retries its own record without acquiring
        // references again; successful columns remain independently resident.
        let record = r.residents.get(near[0]).unwrap().record;
        r.complete_jobs(near.iter().enumerate().map(|(i, &key)| (key, if i == 0 { 3 } else { 0 })));
        assert_eq!(r.initial_retries.iter().copied().collect::<Vec<_>>(), vec![near[0]]);
        assert_eq!(r.residents.get(near[0]).unwrap().record, record);
        assert!(r.blocks.values().all(|block| block.refs == 16));
        for &key in &near { r.evict(key, &mut FrameWork::default()); }
        assert!(r.blocks.is_empty() && r.block_owner.is_empty() && r.live_tier1.is_empty());
        assert_eq!(r.delayed_records.len(), 17); // includes the original fixture's evicted record.
        table_is_exact(&r);
    }

    #[test]
    fn complete_visible_batch_rejects_every_partial_or_owned_case_without_mutation() {
        for case in 0..13 {
            let (mut planet, mut r, near, _) = batch_admission_fixture();
            let mut budget = 16;
            let mut deadline = None;
            match case {
                0 => budget = 15,
                1 => deadline = Some(std::time::Instant::now()),
                2 => {
                    let (face, _, i, j) = unpack(near[0]);
                    let center = planet.grid().ground_point(face, f64::from(i * BRICK) + 0.5, f64::from(j * BRICK) + 0.5);
                    std::sync::Arc::make_mut(&mut planet).apply(crate::edits::Brush {
                        center: center.to_array(), ..test_brush(0.2)
                    }).unwrap();
                }
                3 => r.capacity.records = 15,
                4 => { r.levels[0].pending.remove(near[7]); }
                5 => { r.visible_admission.remove(7); }
                6 => { r.residents.insert(near[7], Resident { record: 1, ..Default::default() }); }
                7 => { r.initial_retries.insert(near[7]); }
                8 => {
                    r.publishing.insert(near[7], EditPublication {
                        record: 1, previous: None, next: None, evicted: false, initial_bucket: Some(0),
                    });
                }
                9 => { std::sync::Arc::make_mut(r.levels[0].wanted.as_mut().unwrap()).remove(&near[7]); }
                10 => {
                    let (face, level, i, j) = unpack(near[0]);
                    let slot = block_slot(level, face, 3, i >> 6, j >> 6);
                    r.block_owner.insert(slot, (level, face, 3, (i >> 6) + 8, j >> 6));
                    assert!(r.blocks_conflict(near[0]));
                }
                11 => r.visible_admission[0].0 = 1,
                12 => r.visible_admission.swap(0, 1),
                _ => unreachable!(),
            }
            let before = (r.residents.len(), r.publishing.len(), r.next_record,
                r.free_records.clone(), r.visible_admission.clone(), r.levels[0].pending.len(), r.block_owner.clone());
            let mut work = FrameWork::default();
            assert!(!r.admit_visible_block(&planet, 0, near[0], 0, budget, deadline, &mut work), "case {case}");
            assert!(work.jobs.is_empty() && work.job_keys.is_empty() && work.table_writes.is_empty() && work.block_inits.is_empty());
            assert_eq!(before, (r.residents.len(), r.publishing.len(), r.next_record,
                r.free_records.clone(), r.visible_admission.clone(), r.levels[0].pending.len(), r.block_owner.clone()), "case {case}");
            assert!(r.blocks.is_empty());
        }
    }

    #[test]
    fn complete_visible_batch_cannot_overtake_global_coverage_or_invent_lease_demand() {
        let (planet, mut r, near, _) = batch_admission_fixture();
        let mut work = FrameWork::default();
        let top = r.grid.levels() - 1;
        let global = pack(key0(0, top, 0), 0);
        r.levels[top as usize].pending.insert(global, 0);
        assert!(!r.admit_visible_block(&planet, 0, near[0], 0, 16, None, &mut work));
        r.levels[top as usize].pending.remove(global);
        std::sync::Arc::make_mut(r.levels[0].wanted.as_mut().unwrap()).remove(&near[7]);
        assert!(!r.transient_wanted(near[7]));
        assert!(!r.admit_visible_block(&planet, 0, near[0], 0, 16, None, &mut work));
        assert!(work.jobs.is_empty() && r.blocks.is_empty() && r.visible_leases.is_empty());
    }

    #[test]
    fn complete_visible_batch_accumulates_existing_parent_refs_without_reinitializing_them() {
        let (planet, mut r, near, _) = batch_admission_fixture();
        let mut work = FrameWork::default();
        assert!(r.admit_visible_block(&planet, 0, near[0], 0, 32, None, &mut work));
        let second = r.visible_columns(near[0] + 4).unwrap().0;
        std::sync::Arc::make_mut(r.levels[0].wanted.as_mut().unwrap()).extend(second);
        r.visible_admission.clear();
        for &key in &second[1..] {
            r.levels[0].pending.insert(key, 0);
            r.visible_admission.push_back((0, key));
        }
        assert!(r.admit_visible_block(&planet, 0, second[0], 0, 32, None, &mut work));
        assert_eq!(work.block_inits.len(), 4, "second tier-1 block shares both existing parents");
        for (&owner, block) in &r.blocks {
            let expected = work.job_keys.iter().filter(|&&key| {
                let (face, level, i, j) = unpack(key);
                owner == (level, face, owner.2, i >> (2 * owner.2), j >> (2 * owner.2))
            }).count() as u32;
            assert_eq!(block.refs, expected);
            assert_eq!(r.block_owner[&block.slot], owner);
        }
        r.complete_jobs(work.job_keys.iter().map(|&key| (key, 0)));
        for &key in &near { r.evict(key, &mut FrameWork::default()); }
        assert!(r.blocks.values().all(|block| block.refs == 16));
        assert!(second.iter().all(|&key| r.residents.get(key).unwrap().blocks));
        table_is_exact(&r);
    }

    #[test]
    fn complete_visible_batch_plan_uses_existing_rank_and_reports_last_plan_counters() {
        let (planet, mut r, eye, near, far) = visible_order_fixture();
        r.prioritize_visible_blocks([near[0], far[0]].map(|key| (key as u32, (key >> 32) as u32)));
        let work = r.plan(&planet, eye, 1.0, 16);
        assert_eq!(work.job_keys, near);
        assert_eq!(r.stats.admission_batched_columns, 16);
        assert_eq!(r.stats.admission_attempts, 16);
        assert_eq!(r.stats.admission_alias_deferred, 0);
        assert_eq!(r.stats.admission_publication_deferred, 0);
        assert!(far.iter().all(|&key| !r.residents.contains_key(key)));
        r.complete_jobs(work.job_keys.iter().map(|&key| (key, 0)));
        r.set_cpu_budget(Some(std::time::Duration::ZERO));
        assert!(r.plan(&planet, eye, 1.0, 16).jobs.is_empty());
        assert_eq!(r.stats.admission_attempts, 0);
        assert_eq!(r.stats.admission_batched_columns, 0);
    }

    #[test]
    fn complete_visible_batch_accepts_distant_persisted_edit_without_dropping_it() {
        let (mut planet, mut r, near, _) = batch_admission_fixture();
        let brush = crate::edits::Brush { op: crate::edits::BrushOp::Remove, ..test_brush(1.5) };
        std::sync::Arc::make_mut(&mut planet).apply(brush).unwrap();
        assert_eq!(planet.edits().len(), 1);
        let mut work = FrameWork::default();
        r.sync_edits(&planet, &mut work);
        assert!(near.iter().all(|&key| r.edit_list(&planet, key, &mut work).unwrap().is_none()));
        assert!(r.admit_visible_block(&planet, 0, near[0], 0, 16, None, &mut work));
        assert_eq!(work.job_keys, near);
        assert!(work.jobs.iter().all(|job| job.edits == 0));
        assert_eq!(planet.edits().brushes().copied().collect::<Vec<_>>(), vec![brush]);
    }

    #[test]
    fn complete_visible_batch_keeps_intersecting_and_tangent_edits_on_scalar_path_at_each_level() {
        use crate::edits::{Brush, BrushOp};
        for level in [0, 3] {
            for op in [BrushOp::Add, BrushOp::Remove, BrushOp::Paint] {
                for tangent in [false, true] {
                    let (mut planet, mut r, near, _) = batch_admission_fixture();
                    let (face, _, i, j) = unpack(near[0]);
                    let first = pack(key0(face, level, (i >> level) & !3), ((j >> level) & !3) as u32);
                    let keys = r.visible_columns(first).unwrap().0;
                    let index = level as usize;
                    r.levels[index].wanted = Some(std::sync::Arc::new(keys.into_iter().collect()));
                    r.visible_admission.clear();
                    r.levels[index].pending.clear();
                    for &key in &keys[1..] {
                        r.levels[index].pending.insert(key, 0);
                        r.visible_admission.push_back((index, key));
                    }
                    let (_, _, i, j) = unpack(first);
                    let span = i64::from(BRICK) << level;
                    let i0 = i64::from(i) * span;
                    let j0 = i64::from(j) * span;
                    let i1 = i0 + 4 * span - 1;
                    let radius = planet.grid().voxel_size() * 5.0;
                    // Query containment includes one conservative base-cell
                    // margin. Test equality at that margin, including L3.
                    let x = if tangent { i1 + 6 } else { i0 };
                    let center = planet.grid().ground_point(face, x as f64, (j0 + 2 * span) as f64);
                    let brush = Brush { center: center.to_array(), radius, op, material: 3, ..test_brush(radius) };
                    let resolved = brush.resolve(planet.grid()).unwrap();
                    if tangent {
                        let face_brush = resolved.iter().find(|b| b.face() == face).unwrap();
                        assert_eq!(i64::from(face_brush.center[0]) / 2 - i64::from(face_brush.radius_half) / 2 - 1, i1);
                        assert!(face_brush.active(level));
                    }
                    std::sync::Arc::make_mut(&mut planet).apply(brush).unwrap();
                    let mut work = FrameWork::default();
                    r.sync_edits(&planet, &mut work);
                    let before = (r.next_record, r.visible_admission.clone(), r.levels[index].pending.len(), work.table_writes.len());
                    assert!(!r.admit_visible_block(&planet, index, first, 0, 16, None, &mut work), "{op:?} tangent={tangent} level={level}");
                    assert!(work.jobs.is_empty() && r.blocks.is_empty());
                    assert_eq!(before, (r.next_record, r.visible_admission.clone(), r.levels[index].pending.len(), work.table_writes.len()));
                    // The unchanged scalar edit-list path must still attach
                    // the real journal brush to at least one touched column.
                    assert!(keys.iter().any(|&key| r.edit_list(&planet, key, &mut work).unwrap().is_some()), "{op:?} tangent={tangent} level={level}");
                    assert_eq!(planet.edits().len(), 1);
                }
            }
        }
    }

    #[test]
    fn visible_nearest_complete_block_is_admitted_before_farther_with_partial_budget() {
        let (planet, mut r, eye, near, far) = visible_order_fixture();
        let requests = || [near[0], far[0]].map(|key| (key as u32, (key >> 32) as u32));
        let mut admitted = FxHashSet::default();
        for _ in 0..2 {
            r.prioritize_visible_blocks(requests());
            let work = r.plan(&planet, eye, 1.0, 8);
            assert_eq!(work.jobs.len(), 8);
            assert!(work.job_keys.iter().all(|key| near.contains(key)), "nearest visible geometry must consume the partial budget first");
            admitted.extend(work.job_keys.iter().copied());
            r.complete_jobs(work.job_keys.iter().map(|key| (*key, 0)));
        }
        assert_eq!(admitted, near.iter().copied().collect());
        assert!(far.iter().all(|key| !r.residents.contains_key(*key)));
        let work = r.plan(&planet, eye, 1.0, 16);
        assert_eq!(work.job_keys.iter().copied().collect::<FxHashSet<_>>(), far.iter().copied().collect());
    }

    #[test]
    fn visible_changed_nearest_order_reprioritizes_existing_bucket_zero_blocks() {
        let (planet, mut r, _, near, far) = visible_order_fixture();
        r.prioritize_visible_blocks([near[0], far[0]].map(|key| (key as u32, (key >> 32) as u32)));
        r.refresh_visible_pending(None);
        assert!(near.iter().chain(&far).all(|key| r.levels[0].pending.at[key].0 == 0));
        let (face, _, i, j) = unpack(far[0]);
        let mut eye = planet.grid().ground_point(face, f64::from((i + 42) * BRICK), f64::from((j + 2) * BRICK));
        eye.y = 10.0;
        let distance = |key| {
            let (face, _, i, j) = unpack(key);
            planet.grid().ground_distance(eye, planet.grid().ground_point(face,
                f64::from((i + 2) * BRICK), f64::from((j + 2) * BRICK)))
        };
        assert!(distance(far[0]) < distance(near[0]), "the camera moved beyond both blocks, reversing their physical distance order");
        // Keep this test's authored demand fixed, independently of the planner.
        r.last_request.as_mut().unwrap().eye = eye;
        r.prioritize_visible_blocks([far[0], near[0]].map(|key| (key as u32, (key >> 32) as u32)));
        let work = r.plan(&planet, eye, 1.0, 16);
        assert_eq!(work.job_keys.iter().copied().collect::<FxHashSet<_>>(), far.iter().copied().collect());
        assert!(near.iter().all(|key| !r.residents.contains_key(*key)));
    }

    #[test]
    fn elapsed_plan_deadline_preserves_diffs_and_pending_without_admitting_work() {
        let (planet, mut r, _, eye) = edit_fixture();
        let key = pack(key0(crate::grid::PLANE_FACE, 0, 1000), 0);
        r.levels[0].pending.insert(key, 1);
        r.apply(WindowUpdate { levels: vec![LevelDiff {
            level: 0, active: true, adds: vec![(0.0, key)], ..Default::default()
        }], ..Default::default() });
        r.set_cpu_budget(Some(std::time::Duration::ZERO));
        let work = r.plan(&planet, eye, 1.0, 16);
        assert!(work.jobs.is_empty() && work.evictions.is_empty());
        assert_eq!(r.diffs[0][0].added, 0, "expired budget must not apply a mandatory first batch");
        assert!(r.levels[0].pending.at.contains_key(&key));
        assert_eq!(r.catching_up[0], 1);
        r.set_cpu_budget(None);
        let work = r.plan(&planet, eye, 1.0, 16);
        assert_eq!(work.job_keys, vec![key], "deferred demand must remain admissible next frame");
    }

    fn visible_bridge_fixture() -> (std::sync::Arc<Planet>, Residency, DVec3, std::time::Instant) {
        let (planet, mut r, old, eye) = edit_fixture();
        r.evict(old, &mut FrameWork::default());
        r.levels[0].wanted = Some(Default::default());
        r.levels[0].active = false;
        r.snapshot_mode = true;
        r.requested = 2;
        r.applied = 1;
        r.applied_levels.fill(1);
        r.applied_seen = (1u32 << r.grid.levels()) - 1;
        let at = std::time::Instant::now();
        r.set_visible_view(10, 7, at);
        (planet, r, eye, at)
    }

    fn lease_block(r: &mut Residency, i: i32, at: std::time::Instant) -> Vec<u64> {
        let key = pack(key0(crate::grid::PLANE_FACE, 0, i), 1000);
        r.prioritize_visible_blocks_from([(key as u32, (key >> 32) as u32)], 10, 7, at);
        r.refresh_visible_pending(None);
        let (keys, count) = r.visible_columns(key).unwrap();
        keys[..count].to_vec()
    }

    #[test]
    fn complete_visible_batch_accepts_captured_lease_and_preserves_expiry_ownership() {
        let (planet, mut r, _, at) = visible_bridge_fixture();
        let keys = lease_block(&mut r, 1000, at);
        let first = keys[0];
        assert!(keys.iter().all(|&key| !r.current_wanted(key) && r.transient_wanted(key)));
        assert_eq!(r.visible_admission.pop_front(), Some((0, first)));
        r.levels[0].pending.remove(first);
        let lease = &r.visible_leases[&first];
        let original = (lease.captured_source().frame, lease.captured_source().view, lease.captured_source().at, lease.serial, lease.retiring, lease.retired);
        let mut work = FrameWork::default();
        assert!(r.admit_visible_block(&planet, 0, first, 0, 16, None, &mut work));
        assert_eq!(work.job_keys, keys);
        assert_eq!(r.visible_leases.len(), 1);
        let lease = &r.visible_leases[&first];
        assert_eq!(original, (lease.captured_source().frame, lease.captured_source().view, lease.captured_source().at, lease.serial, lease.retiring, lease.retired));
        assert!(r.blocks.values().all(|block| block.refs == 16));
        assert_eq!(work.jobs.iter().map(|job| job.record).collect::<FxHashSet<_>>().len(), 16);
        table_is_exact(&r);
        // Expiry releases each ordinary reference and quarantines in-flight
        // publications; batching cannot extend the captured source lifetime.
        r.set_visible_view(11, 7, at + std::time::Duration::from_millis(501));
        let mut expired = FrameWork::default();
        r.retire_visible_leases(&mut expired, &|| false);
        assert_eq!(expired.evictions.len(), 16);
        assert!(r.blocks.is_empty() && r.block_owner.is_empty());
        assert!(keys.iter().all(|key| r.publishing[key].evicted));
        assert_eq!(r.visible_leases.len(), 1, "in-flight ownership still counts toward the lease cap");
        r.complete_jobs(keys.iter().map(|&key| (key, 0)));
        r.retire_visible_leases(&mut expired, &|| false);
        assert!(r.publishing.is_empty() && r.visible_leases.is_empty());
        table_is_exact(&r);
    }

    #[test]
    fn complete_visible_batch_rejects_invalid_captured_leases_without_mutation() {
        for cause in 0..8 {
            let (planet, mut r, _, at) = visible_bridge_fixture();
            let keys = lease_block(&mut r, 1000, at);
            let first = keys[0];
            assert_eq!(r.visible_admission.pop_front(), Some((0, first)));
            r.levels[0].pending.remove(first);
            match cause {
                0 => r.set_visible_view(43, 7, at + std::time::Duration::from_millis(1)),
                1 => r.set_visible_view(11, 7, at + std::time::Duration::from_millis(501)),
                2 => r.set_visible_view(11, 8, at + std::time::Duration::from_millis(1)),
                3 => r.visible_leases.get_mut(&first).unwrap().retiring = true,
                4 => r.applied_levels[0] = r.visible_leases[&first].serial,
                5 => r.applied_levels[0] = r.visible_leases[&first].serial + 1,
                6 => { r.visible_leases.remove(&first); }
                7 => r.set_visible_view(11, 7, at - std::time::Duration::from_millis(1)),
                _ => unreachable!(),
            }
            assert!(!r.transient_wanted(first), "cause {cause}");
            let lease_stamp = |r: &Residency| r.visible_leases.get(&first).map(|lease|
                (lease.captured_source().frame, lease.captured_source().view, lease.captured_source().at, lease.serial, lease.retiring, lease.retired));
            let before = (r.next_record, r.free_records.clone(), r.delayed_records.clone(),
                r.visible_admission.clone(), r.levels[0].pending.len(), r.visible_leases.len(), lease_stamp(&r));
            let mut work = FrameWork::default();
            assert!(!r.admit_visible_block(&planet, 0, first, 0, 16, None, &mut work), "cause {cause}");
            assert_eq!(before, (r.next_record, r.free_records.clone(), r.delayed_records.clone(),
                r.visible_admission.clone(), r.levels[0].pending.len(), r.visible_leases.len(), lease_stamp(&r)));
            assert!(work.jobs.is_empty() && work.table_writes.is_empty() && work.block_inits.is_empty());
            assert!(r.blocks.is_empty() && r.block_owner.is_empty() && r.publishing.is_empty());
        }
    }

    #[test]
    fn visible_refresh_deadline_resumes_requested_tail_with_original_source() {
        let (_, mut r, _, at) = visible_bridge_fixture();
        let old_lease = lease_block(&mut r, 1008, at);
        let blocks = [1000, 1004].map(|i| pack(key0(crate::grid::PLANE_FACE, 0, i), 1000));
        r.prioritize_visible_blocks_from(blocks.map(|key| (key as u32, (key >> 32) as u32)), 10, 7, at);
        let mut checks = 0;
        r.refresh_visible_pending_until(|| { checks += 1; checks > 1 });
        let (first, count) = r.visible_columns(blocks[0]).unwrap();
        assert_eq!(count, 16);
        assert!(first.iter().all(|key| r.levels[0].pending.at.contains_key(key)));
        assert_eq!(r.visible_admission.len(), 16, "the processed block's entire rank must commit together");
        assert_eq!(r.visible_blocks, vec![blocks[1]], "retain requests, not appended lease reinsertion");
        let source = r.visible_source.unwrap();
        assert_eq!((source.frame, source.view, source.at), (10, 7, at));
        assert!(old_lease.iter().all(|key| r.levels[0].pending.at.contains_key(key)));
        r.set_visible_view(11, 7, at + std::time::Duration::from_millis(1));
        r.refresh_visible_pending_until(|| false);
        let (second, count) = r.visible_columns(blocks[1]).unwrap();
        assert_eq!(count, 16);
        assert!(second.iter().all(|key| r.levels[0].pending.at.contains_key(key)));
        assert_eq!(r.visible_admission.len(), 32);
        let lease = &r.visible_leases[&blocks[1]];
        assert_eq!((lease.captured_source().frame, lease.captured_source().view, lease.captured_source().at), (10, 7, at),
            "resumption cannot refresh the captured source or extend its lease");
        assert!(r.visible_blocks.is_empty() && r.visible_source.is_none());
    }

    #[test]
    fn visible_refresh_deferred_source_expiry_discards_requests() {
        for cause in 0..3 {
            let (_, mut r, _, at) = visible_bridge_fixture();
            let block = pack(key0(crate::grid::PLANE_FACE, 0, 1000), 1000);
            r.prioritize_visible_blocks_from([(block as u32, (block >> 32) as u32)], 10, 7, at);
            r.refresh_visible_pending_until(|| true);
            assert_eq!(r.visible_blocks, vec![block]);
            match cause {
                0 => r.set_visible_view(19, 7, at),
                1 => r.set_visible_view(11, 8, at),
                _ => r.set_visible_view(11, 7, at + VISIBLE_LEASE_TIME + std::time::Duration::from_nanos(1)),
            }
            r.refresh_visible_pending_until(|| false);
            assert!(r.visible_blocks.is_empty() && r.visible_source.is_none());
            assert!(r.visible_admission.is_empty() && r.visible_rank_source.is_none());
            assert!(r.visible_leases.is_empty() && r.levels[0].pending.is_empty(),
                "expired captured requests cannot create demand or leases");
        }
    }

    #[test]
    fn visible_refresh_fresh_batch_replaces_unprocessed_tail() {
        let (_, mut r, _, at) = visible_bridge_fixture();
        let block = |i| pack(key0(crate::grid::PLANE_FACE, 0, i), 1000);
        let stale = block(1000);
        r.prioritize_visible_blocks_from([(stale as u32, (stale >> 32) as u32)], 10, 7, at);
        r.refresh_visible_pending_until(|| true);
        let fresh = block(1004);
        let captured = at + std::time::Duration::from_millis(1);
        r.set_visible_view(11, 7, captured);
        r.prioritize_visible_blocks_from([(fresh as u32, (fresh >> 32) as u32)], 11, 7, captured);
        assert_eq!(r.visible_blocks, vec![fresh]);
        r.refresh_visible_pending_until(|| false);
        let (fresh_keys, count) = r.visible_columns(fresh).unwrap();
        assert_eq!(count, 16);
        assert!(fresh_keys.iter().all(|key| r.levels[0].pending.at.contains_key(key)));
        assert!(!r.visible_leases.contains_key(&stale));
        let lease = &r.visible_leases[&fresh];
        assert_eq!((lease.captured_source().frame, lease.captured_source().view, lease.captured_source().at), (11, 7, captured));
        assert_eq!(r.visible_admission.len(), 16);
    }

    #[test]
    fn visible_bridge_rejects_stale_sources_and_requires_worker_lag() {
        let (_, mut r, _, at) = visible_bridge_fixture();
        let key = pack(key0(crate::grid::PLANE_FACE, 0, 1000), 1000);
        for (frame, view, time) in [(1, 7, at), (10, 8, at), (10, 7, at - VISIBLE_LEASE_TIME * 2)] {
            r.prioritize_visible_blocks_from([(key as u32, (key >> 32) as u32)], frame, view, time);
            r.refresh_visible_pending(None);
            assert!(r.visible_leases.is_empty() && r.levels[0].pending.is_empty());
        }
        r.requested = r.applied;
        lease_block(&mut r, 1000, at);
        assert!(r.visible_leases.is_empty(), "ordinary snapshots remain authoritative when caught up");
        r.requested += 1;
        lease_block(&mut r, 1000, at);
        assert_eq!(r.visible_leases.len(), 1);
        r.requested += 1;
        lease_block(&mut r, 1000, at);
        assert_eq!(r.visible_leases[&key].serial, 2, "replaying the same GPU frame cannot renew its request epoch");
        r.set_visible_view(11, 8, at);
        r.retire_visible_leases(&mut FrameWork::default(), &|| false);
        assert!(r.visible_leases.is_empty() && r.levels[0].pending.is_empty());
    }

    #[test]
    fn visible_bridge_survives_stale_snapshot_queue_replacement_then_transfers() {
        let (planet, mut r, eye, at) = visible_bridge_fixture();
        let keys = lease_block(&mut r, 1000, at);
        r.apply(snapshot_update(1, &[]));
        assert!(r.levels[0].pending.is_empty());
        let work = r.plan(&planet, eye, 1.0, 16);
        assert_eq!(work.job_keys.iter().copied().collect::<FxHashSet<_>>(), keys.iter().copied().collect());
        assert!(keys.iter().all(|key| r.residents.get(*key).unwrap().blocks));
        r.complete_jobs(work.job_keys.iter().map(|key| (*key, 0)));
        r.apply(snapshot_update(2, &keys));
        let work = r.plan(&planet, eye, 1.0, 16);
        assert!(work.evictions.is_empty() && work.jobs.is_empty());
        assert!(r.visible_leases.is_empty() && r.idle());
        assert!(keys.iter().all(|key| r.residents.contains_key(*key)), "the authoritative wanted set protects transferred columns");
    }

    #[test]
    fn visible_bridge_does_not_steal_live_summary_owners_when_level_is_settled() {
        let (planet, mut r, eye, at) = visible_bridge_fixture();
        let keys = lease_block(&mut r, 1000, at);
        let old: Vec<_> = keys.iter().map(|key| {
            let (face, level, i, j) = unpack(*key);
            pack(key0(face, level, i - 512), j as u32)
        }).collect();
        r.levels[0].wanted = Some(std::sync::Arc::new(old.iter().copied().collect()));
        for (record, key) in old.iter().enumerate() {
            assert!(r.acquire_blocks(*key, &mut FrameWork::default()));
            r.residents.insert(*key, Resident { record: record as u32, blocks: true, ..Default::default() });
        }
        r.next_record = old.len() as u32;
        r.delayed_records.clear();
        assert_eq!(r.catching_up[0], 0);
        let work = r.plan(&planet, eye, 1.0, 16);
        assert!(work.jobs.is_empty() && work.evictions.is_empty() && r.publishing.is_empty());
        assert_eq!(r.block_conflicts, 0);
        assert!(old.iter().all(|key| r.residents.get(*key).unwrap().blocks));
        assert!(keys.iter().all(|key| r.levels[0].pending.at.contains_key(key)));
    }

    #[test]
    fn visible_bridge_expiry_quarantines_inflight_edits_and_cannot_retry_old_keys() {
        let (mut planet, mut r, eye, at) = visible_bridge_fixture();
        let (cell, _) = planet.grid().locate(DVec3::ZERO);
        let i = (cell.i >> 3) & !3;
        let j = (cell.j >> 3) & !3;
        let key = pack(key0(cell.face, 0, i), j as u32);
        std::sync::Arc::make_mut(&mut planet).apply(test_brush(0.2)).unwrap();
        r.last_request.as_mut().unwrap().outer_radius = planet.outer_radius();
        r.prioritize_visible_blocks_from([(key as u32, (key >> 32) as u32)], 10, 7, at);
        r.refresh_visible_pending(None);
        let first = r.plan(&planet, eye, 1.0, 16);
        assert_eq!(first.jobs.len(), 16);
        let journals: Vec<_> = r.publishing.values().filter_map(|p| p.next).collect();
        assert!(!journals.is_empty(), "temporary columns still query the canonical edit journal");
        r.set_visible_view(10 + VISIBLE_LEASE_FRAMES + 1, 7, at);
        let expired = r.plan(&planet, eye, 1.0, 16);
        assert!(expired.jobs.is_empty());
        assert_eq!(expired.evictions.len(), 16);
        assert_eq!(r.visible_leases.len(), 1, "inflight acknowledgments retain their capped lease slot");
        assert!(r.publishing.values().all(|p| p.evicted));
        r.complete_jobs(first.job_keys.iter().map(|key| (*key, 3)));
        assert!(r.initial_retries.is_empty() && r.publishing.is_empty());
        r.retire_visible_leases(&mut FrameWork::default(), &|| false);
        assert!(r.visible_leases.is_empty());
        assert!(journals.iter().all(|block| r.edits.free[block.1 as usize].contains(&block.0)));
    }

    #[test]
    fn visible_bridge_expiry_removes_failed_initial_publication_retry() {
        let (planet, mut r, eye, at) = visible_bridge_fixture();
        let keys = lease_block(&mut r, 1000, at);
        let first = r.plan(&planet, eye, 1.0, 1);
        r.complete_jobs([(first.job_keys[0], 3)]);
        assert_eq!(r.initial_retries.len(), 1);
        r.set_visible_view(10, 7, at + VISIBLE_LEASE_TIME + std::time::Duration::from_millis(1));
        let expired = r.plan(&planet, eye, 1.0, 16);
        assert!(expired.jobs.is_empty() && r.initial_retries.is_empty());
        assert!(keys.iter().all(|key| !r.residents.contains_key(*key)));
        assert!(r.visible_leases.is_empty() && r.levels[0].pending.is_empty());
    }

    #[test]
    fn visible_bridge_caps_churn_and_retires_in_bounded_slices() {
        let (_, mut r, _, at) = visible_bridge_fixture();
        let face = crate::grid::PLANE_FACE;
        let blocks: Vec<_> = (0..VISIBLE_BLOCKS as i32).map(|n| pack(key0(face, 0, 1000 + 4 * n), 1000)).collect();
        r.prioritize_visible_blocks_from(blocks.iter().map(|key| (*key as u32, (*key >> 32) as u32)), 10, 7, at);
        r.refresh_visible_pending(None);
        assert_eq!(r.visible_leases.len(), VISIBLE_BLOCKS);
        assert_eq!(r.levels[0].pending.len(), 1024);
        assert_eq!(r.visible_admission.len(), VISIBLE_BLOCKS * 16);
        for i in (3000..3400).step_by(4) {
            lease_block(&mut r, i, at);
            assert_eq!(r.visible_leases.len(), VISIBLE_BLOCKS);
            assert_eq!(r.levels[0].pending.len(), 1024);
            assert!(r.visible_admission.len() <= VISIBLE_BLOCKS * 16);
        }
        r.set_visible_view(43, 7, at);
        r.retire_visible_leases(&mut FrameWork::default(), &|| false);
        assert_eq!(r.visible_leases.len(), VISIBLE_BLOCKS - 8);
        assert_eq!(r.levels[0].pending.len(), 1024 - 128);
        r.set_visible_view(10, 7, at);
        lease_block(&mut r, 3000, at);
        assert_eq!(r.visible_leases.len(), VISIBLE_BLOCKS - 7);
        assert!(r.visible_leases.values().filter(|lease| lease.retiring).count() <= VISIBLE_BLOCKS);
        r.set_visible_view(43, 7, at);
        for _ in 0..8 { r.retire_visible_leases(&mut FrameWork::default(), &|| false); }
        assert!(r.visible_leases.is_empty() && r.levels[0].pending.is_empty());
    }

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
        assert_eq!(r.queued_delta_ops, r.diffs.iter().flatten().map(|q|
            q.diff.removes.len() - q.removed + q.diff.adds.len() - q.added).sum::<usize>());
        assert_eq!(r.queued_delta_bytes, r.diffs.iter().flatten().map(|q| delta_bytes(&q.diff)).sum::<usize>());
        r.apply_queued(&mut work, &|| false);
        assert_eq!(r.levels[0].pending.keys().copied().collect::<FxHashSet<_>>(), latest.into_iter().collect());
        assert_eq!(r.catching_up[0], 0);
        assert_eq!((r.queued_delta_ops, r.queued_delta_bytes), (0, 0));
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
            lod0: 120.0, lod_dither: 0.0, outer_radius: planet.outer_radius(), planet: Some(planet.clone()), serial: 1,
        });
        residency.set_cpu_budget(None);
        let work = residency.plan(&planet, eye, 120.0, 16);
        assert_eq!(work.job_keys.len(), incoming.len());
        assert_eq!(work.job_keys.iter().copied().collect::<std::collections::HashSet<_>>(), incoming.iter().copied().collect());
        assert!(work.job_keys.iter().all(|&key| residency.residents.get(key).unwrap().blocks),
            "incoming columns must acquire summaries before the remaining outgoing owners retire");
        assert_eq!(residency.catching_up[0], 0);
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
            eye, prefetch_eye: None, priority_eye: None, view_focus: None, lod0: 1.0, lod_dither: 0.0,
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

    #[test]
    fn renderer_dither_setting_replans_without_camera_motion() {
        let planet = std::sync::Arc::new(Planet::new(crate::planet::PlanetRecipe {
            shape: crate::grid::Shape::Plane, plane_size_m: 1000.0,
            terrain: crate::terrain::TerrainSource { generator: crate::landform::FLAT_ID.into(), ..Default::default() },
            ..Default::default() }).unwrap());
        let mut residency = Residency::new(*planet.grid(), Capacity::default());
        let eye = DVec3::Y * 110.0;
        residency.plan(&planet, eye, 100.0, 0);
        let serial = residency.requested;
        residency.set_lod_dither(0.25);
        residency.plan(&planet, eye, 100.0, 0);
        assert_eq!(residency.requested, serial + 1);
        assert_eq!(residency.last_request.as_ref().unwrap().lod_dither, 0.25);
        assert!(residency.levels[0].active);
        residency.set_lod_dither(f64::NAN);
        residency.plan(&planet, eye, 100.0, 0);
        assert_eq!(residency.requested, serial + 1, "equivalent sanitized settings must not restart planning");
    }


    fn focus_prefix_fixture() -> (std::sync::Arc<Planet>, Residency, DVec3, Vec<u64>) {
        let planet = std::sync::Arc::new(Planet::new(PlanetRecipe { shape: crate::grid::Shape::Plane,
            plane_size_m: 1000.0, ..Default::default() }).unwrap());
        let grid = *planet.grid();
        let eye = DVec3::Y * 30.0;
        let (cell, _) = grid.locate(eye);
        let ci = (cell.i >> 3) & !3;
        let cj = (cell.j >> 3) & !3;
        let focus_keys: Vec<_> = (cj..cj+4).flat_map(|j|
            (ci+24..ci+28).map(move |i| pack(key0(crate::grid::PLANE_FACE,0,i),j as u32))).collect();
        let focus = grid.ground_point(crate::grid::PLANE_FACE,
            f64::from(ci+26)*8.0,f64::from(cj+2)*8.0);
        let old: Vec<_> = (cj..cj+16).flat_map(|j|
            (ci-80..ci-64).map(move |i| pack(key0(crate::grid::PLANE_FACE,0,i),j as u32))).collect();
        let keys: Vec<_> = old.into_iter().chain(focus_keys.iter().copied()).collect();
        let mut r = Residency::new(grid,Capacity::default());
        let mut update = snapshot_update(1,&keys);
        for (priority,_) in &mut update.levels[0].adds { *priority=0.9; }
        r.apply(update);
        r.requested=1;
        r.last_request=Some(WindowRequest {eye,prefetch_eye:None,priority_eye:None,view_focus:None,
            lod0:120.0,lod_dither:0.0,outer_radius:planet.outer_radius(),planet:Some(planet.clone()),serial:1});
        r.set_view_focus(Some(focus));
        (planet,r,eye,focus_keys)
    }

    #[test]
    fn wanted_focus_bypasses_old_unadmitted_prefix_with_normal_publication() {
        let (planet,mut r,eye,focus) = focus_prefix_fixture();
        assert!(r.levels[0].pending.is_empty());
        assert_eq!(r.diffs[0][0].added,0);
        r.refresh_near_pending(eye,None);
        assert_eq!(r.levels[0].pending.len(),16);
        assert!(focus.iter().all(|k|r.levels[0].pending.at.contains_key(k)));
        assert_eq!(r.diffs[0][0].added,0,"priority refresh must not consume or replace the worker prefix");
        assert!(r.residents.is_empty() && r.publishing.is_empty() && r.blocks.is_empty());
        r.refresh_near_pending(eye,None);
        assert_eq!(r.levels[0].pending.len(),16,"repeated refresh must not duplicate wanted keys");
        let work = r.plan(&planet,eye,120.0,16);
        assert_eq!(r.requested,1,"camera-only focus must preserve window membership and serial");
        assert_eq!(work.job_keys.iter().copied().collect::<FxHashSet<_>>(),focus.iter().copied().collect());
        assert_eq!(work.jobs.len(),16);
        for key in &focus {
            let resident=r.residents.get(*key).unwrap();
            assert!(resident.blocks);
            assert_eq!(r.publishing[key].record,resident.record);
        }
        r.complete_jobs(work.job_keys.iter().map(|k|(*k,0)));
        assert!(r.publishing.is_empty());
    }

    #[test]
    fn wanted_focus_rejects_mixed_retired_and_expired_demand() {
        for retired in [false,true] {
            let (_,mut r,eye,focus) = focus_prefix_fixture();
            // All16 still sit in old pending, but15 current wanted members
            // cannot authorize a complete block assembled from mixed epochs.
            for &key in &focus {r.levels[0].pending.insert(key,50);}
            let mut wanted = r.levels[0].wanted.as_ref().unwrap().as_ref().clone();
            wanted.remove(&focus[0]);
            r.levels[0].wanted=Some(std::sync::Arc::new(wanted));
            let stamp=VisibleStamp {frame:10,view:7,at:std::time::Instant::now()};
            r.visible_view=Some(stamp);
            r.visible_leases.insert(focus[0] & !(3u64 | (3u64<<32)),
                VisibleLease {origin:LeaseOrigin::Captured(stamp),serial:2,retiring:false,retired:0,current_demand_frame:None});
            assert!(r.transient_wanted(focus[0]),"a valid GPU lease deliberately supplies the missing member");
            r.levels[0].active=!retired;
            let before=r.levels[0].pending.at.clone();
            r.refresh_near_pending(eye,None);
            assert_eq!(r.levels[0].pending.at,before);
            assert!(r.residents.is_empty() && r.publishing.is_empty());
        }
        let (_,mut r,eye,_) = focus_prefix_fixture();
        r.refresh_near_pending(eye,Some(std::time::Instant::now()));
        assert!(r.levels[0].pending.is_empty(),"an elapsed refresh deadline admits no partial block");
    }

    #[test]
    fn wanted_focus_preserves_partial_job_budget_and_coarser_priority() {
        let (planet,mut r,eye,focus) = focus_prefix_fixture();
        r.refresh_near_pending(eye,None);
        let coarse=pack(key0(crate::grid::PLANE_FACE,1,1),1);
        r.levels[1].active=true;
        r.levels[1].pending.insert(coarse,0);
        let work=r.plan(&planet,eye,120.0,7);
        assert_eq!(work.jobs.len(),7,"promotion must not round a GPU allowance up to a complete block");
        assert_eq!(work.job_keys[0],coarse,"a closer coarser pending key must retain its priority");
        assert!(work.job_keys[1..].iter().all(|key|focus.contains(key)));
        assert_eq!(focus.iter().filter(|key|r.levels[0].pending.at.contains_key(key)).count(),10);
    }

    fn current_bridge_fixture() -> (std::sync::Arc<Planet>, Residency, u64, std::time::Instant) {
        let (planet, mut r, eye, at) = visible_bridge_fixture();
        let (cell, _) = r.grid.locate(eye);
        let key = pack(key0(cell.face, 0, (cell.i >> 3) & !3), ((cell.j >> 3) & !3) as u32);
        let request = WindowRequest { eye, prefetch_eye: None, priority_eye: None,
            view_focus: None, lod0: 160.0, lod_dither: 0.25,
            outer_radius: planet.outer_radius(), planet: Some(planet.clone()), serial: 3 };
        // Keep plan's renderer settings identical to the outstanding request.
        // Otherwise its default dither of 0 installs a new inline wanted snapshot.
        r.set_lod_dither(request.lod_dither);
        r.set_camera_view(DVec3::new(1.0, -0.17, 0.0), DVec3::Y, [0.65, 0.414]);
        r.current_request = Some(request.clone());
        r.last_request = Some(request);
        r.requested = 3;
        r.levels[0].active = true;
        r.applied_levels[0] = 2;
        r.applied_issued_at[0] = Some(at + std::time::Duration::from_millis(1));
        (planet, r, key, at)
    }

    #[test]
    fn current_revalidation_holds_preauthority_demand_across_ack_without_renewing_capture() {
        let (_, mut r, block, at) = current_bridge_fixture();
        r.prioritize_visible_blocks_from([(block as u32, (block >> 32) as u32)], 10, 7, at);
        r.refresh_visible_pending(None);
        assert!(r.transient_wanted(block));
        assert_eq!(r.visible_admission.len(), 16);
        let original = (r.visible_leases[&block].captured_source().frame, r.visible_leases[&block].captured_source().at);
        for serial in 3..7 {
            r.frame += 1;
            r.requested = serial + 1;
            r.apply(WindowUpdate { serial, partial: true, processed_levels: 1,
                issued_at: Some(at + std::time::Duration::from_millis(serial)), ..Default::default() });
            r.retire_visible_leases(&mut FrameWork::default(), &|| false);
            assert!(r.transient_wanted(block), "current geometry remains useful after authority acknowledgment");
            assert_eq!(original, (r.visible_leases[&block].captured_source().frame, r.visible_leases[&block].captured_source().at));
            assert_eq!(r.visible_leases[&block].serial, 3, "geometric revalidation never renews source epoch");
        }
        // Replaying the same source cannot refresh either clock.
        r.prioritize_visible_blocks_from([(block as u32, (block >> 32) as u32)], 10, 7, at);
        r.refresh_visible_pending(None);
        assert_eq!(r.visible_leases[&block].serial, 3);
        r.set_visible_view(10 + VISIBLE_LEASE_FRAMES + 1, 7, at);
        r.retire_visible_leases(&mut FrameWork::default(), &|| false);
        assert!(r.visible_leases.is_empty());
        r.refresh_visible_pending(None);
        assert!(r.visible_leases.is_empty());
    }

    #[test]
    fn camera_arrival_issues_fine_before_old_far_without_worker_membership() {
        let (planet, mut r, _, _) = current_bridge_fixture();
        let eye = r.current_request.as_ref().unwrap().eye;
        r.set_ground_clearance(10.0);
        let far = pack(key0(crate::grid::PLANE_FACE, 3, 128), 128);
        let (keys, count) = r.visible_columns(far).unwrap();
        r.levels[3].active = true;
        r.levels[3].wanted = Some(std::sync::Arc::new(keys[..count].iter().copied().collect()));
        for &key in &keys[..count] {
            r.levels[3].pending.insert(key, 0);
            r.visible_admission.push_back((3, key));
        }
        assert!(r.levels[0].wanted.as_ref().unwrap().is_empty());
        let work = r.plan(&planet, eye, 160.0, 16);
        assert_eq!(work.jobs.len(), 16);
        assert!(work.job_keys.iter().all(|&key| unpack(key).1 == 0 && !r.current_wanted(key)));
        assert!(work.job_keys.iter().all(|&key| r.transient_wanted(key)));
        assert!(r.visible_leases.values().all(|lease| matches!(lease.origin, LeaseOrigin::Camera)));
        assert_eq!(r.stats.admission_batched_columns, 16);
        assert_eq!(r.stats.camera_jobs, [16, 0, 0]);
        r.complete_jobs(work.job_keys.iter().map(|&key| (key, 0)));
        table_is_exact(&r);
        assert!(keys.iter().all(|&key| !r.residents.contains_key(key)));
    }

    #[test]
    fn camera_arrival_preserves_captured_capacity_while_admitting_visible_camera_tiles() {
        let (_, mut r, _, at) = current_bridge_fixture();
        let eye = r.current_request.as_ref().unwrap().eye;
        r.set_ground_clearance(10.0);
        r.levels[1].active = true;
        r.levels[2].active = true;
        let blocks: Vec<_> = (0..VISIBLE_BLOCKS).map(|n|
            pack(key0(crate::grid::PLANE_FACE, 3, 128 + n as i32 * 4), 128)).collect();
        for &block in &blocks {
            r.visible_leases.insert(block, VisibleLease {
                origin: LeaseOrigin::Captured(VisibleStamp { frame: 10, view: 7, at }),
                serial: 3, retiring: false, retired: 0, current_demand_frame: None,
            });
        }
        let current = r.camera_blocks(eye, &|| false);
        assert!(current.len() > VISIBLE_BLOCKS && current.len() <= CAMERA_CANDIDATES);
        r.set_visible_view(11, 7, at);
        r.prioritize_visible_blocks_from(blocks.iter().map(|&key| (key as u32, (key >> 32) as u32)), 11, 7, at);
        r.refresh_visible_pending(None);
        r.refresh_camera_pending(&current, None);
        assert_eq!(r.visible_leases.values().filter(|lease| matches!(lease.origin, LeaseOrigin::Captured(_))).count(), VISIBLE_BLOCKS);
        assert_eq!(r.visible_leases.values().filter(|lease| matches!(lease.origin, LeaseOrigin::Camera)).count(), current.len());
        assert!(r.visible_leases.len() <= TEMPORARY_LEASES && r.visible_admission.len() <= VISIBLE_ADMISSION_COLUMNS);
        assert_eq!(r.visible_admission.front().unwrap().0, 0);
    }

    #[test]
    fn camera_arrival_new_ordinary_view_replaces_old_ranks_and_precedes_captured_coarse() {
        let (planet, mut r, _, at) = current_bridge_fixture();
        let eye = r.current_request.as_ref().unwrap().eye;
        r.set_ground_clearance(10.0);
        let blocks = r.camera_blocks(eye, &|| false);
        let (old, old_count) = r.visible_columns(blocks[0]).unwrap();
        let (new, new_count) = r.visible_columns(blocks[1]).unwrap();
        let wanted: FxHashSet<_> = old[..old_count].iter().chain(&new[..new_count]).copied().collect();
        r.levels[0].wanted = Some(std::sync::Arc::new(wanted));
        r.refresh_camera_pending(&blocks[..1], None);
        assert_eq!(r.visible_admission.len(), old_count);
        assert!(r.visible_leases.is_empty(), "ordinary Camera ranks have no lease origin");

        let captured = pack(key0(crate::grid::PLANE_FACE, 2, 128), 128);
        let (captured_keys, captured_count) = r.visible_columns(captured).unwrap();
        let source = VisibleStamp { frame: 10, view: 7, at };
        r.visible_leases.insert(captured, VisibleLease { origin: LeaseOrigin::Captured(source),
            serial: 3, retiring: false, retired: 0, current_demand_frame: Some(r.frame) });
        for &key in &captured_keys[..captured_count] {
            r.levels[2].pending.insert(key, 0);
            r.visible_admission.push_back((2, key));
        }
        r.refresh_camera_pending(&blocks[1..2], None);
        let ranks: Vec<_> = r.visible_admission.iter().copied().collect();
        assert_eq!(&ranks[..new_count], &new[..new_count].iter().map(|&key| (0, key)).collect::<Vec<_>>());
        assert_eq!(&ranks[new_count..], &captured_keys[..captured_count].iter().map(|&key| (2, key)).collect::<Vec<_>>());
        assert!(old[..old_count].iter().all(|key| r.levels[0].pending.at.contains_key(key)));
        assert_eq!(r.visible_leases[&captured].captured_source().at, source.at);
        assert_eq!(r.visible_leases[&captured].captured_source().frame, source.frame);
        assert!(new[..new_count].iter().all(|&key| r.current_wanted(key)));
        assert!(!r.visible_leases.contains_key(&blocks[1]));
        let work = r.plan(&planet, eye, 160.0, 16);
        assert_eq!(work.jobs.len(), 16);
        assert!(work.job_keys.iter().all(|&key| r.current_wanted(key)));
        assert_eq!(r.stats.camera_jobs, [16, 0, 0], "current ordinary jobs count despite having no Camera lease");
    }

    #[test]
    fn camera_arrival_expiry_preserves_edited_inflight_publication() {
        let (mut planet, mut r, _, _) = current_bridge_fixture();
        let eye = r.current_request.as_ref().unwrap().eye;
        r.set_ground_clearance(10.0);
        let first = r.camera_blocks(eye, &|| false)[0];
        let (face, level, i, j) = unpack(first);
        let center = r.grid.ground_point(face, f64::from((i + 2) * (BRICK << level)),
            f64::from((j + 2) * (BRICK << level)));
        std::sync::Arc::make_mut(&mut planet).apply(crate::edits::Brush {
            center: center.to_array(), ..test_brush(1.5)
        }).unwrap();
        r.last_request.as_mut().unwrap().outer_radius = planet.outer_radius();
        let work = r.plan(&planet, eye, 160.0, 64);
        assert!(!work.jobs.is_empty());
        assert!(work.jobs.iter().any(|job| job.edits != 0));
        let departed = eye + DVec3::new(10_000.0, 0.0, 0.0);
        r.current_request.as_mut().unwrap().eye = departed;
        r.frame += 1;
        assert!(work.job_keys.iter().all(|&key| !r.transient_wanted(key)));
        let mut retired = FrameWork::default();
        for _ in 0..8 { r.retire_visible_leases(&mut retired, &|| false); }
        assert_eq!(retired.evictions.len(), work.jobs.len());
        assert!(work.job_keys.iter().all(|key| r.publishing[key].evicted));
        assert!(!r.visible_leases.is_empty(), "in-flight ownership keeps its capacity slot");
        r.complete_jobs(work.job_keys.iter().map(|&key| (key, 0)));
        r.retire_visible_leases(&mut retired, &|| false);
        assert!(r.visible_leases.is_empty() && r.publishing.is_empty());
        assert_eq!(planet.edits().len(), 1);
        table_is_exact(&r);
    }

    #[test]
    fn camera_arrival_origin_caps_and_mixed_retirement_are_bounded() {
        let (_, mut r, _, at) = current_bridge_fixture();
        let eye = r.current_request.as_ref().unwrap().eye;
        r.set_ground_clearance(10.0);
        let current = r.camera_blocks(eye, &|| false);
        for (origin, count, level) in [
            (LeaseOrigin::Captured(VisibleStamp { frame: 10, view: 7, at }), VISIBLE_BLOCKS, 3),
            (LeaseOrigin::Camera, CAMERA_LEASES, 0),
        ] {
            for n in 0..count {
                let base = if level == 3 { 128 } else { 1000 };
                let block = pack(key0(crate::grid::PLANE_FACE, level, base + n as i32 * 4), base as u32);
                assert!(r.visible_columns(block).is_some());
                r.visible_leases.insert(block, VisibleLease { origin, serial: 3,
                    retiring: false, retired: 0, current_demand_frame: Some(r.frame) });
            }
        }
        assert_eq!(r.visible_leases.len(), TEMPORARY_LEASES);
        r.refresh_camera_pending(&current, None);
        assert!(current.iter().all(|block| !r.visible_leases.contains_key(block)));
        let extra = pack(key0(crate::grid::PLANE_FACE, 0, 3000), 1000);
        r.prioritize_visible_blocks_from([(extra as u32, (extra >> 32) as u32)], 10, 7, at);
        r.refresh_visible_pending(None);
        assert!(!r.visible_leases.contains_key(&extra));
        assert_eq!(r.visible_leases.values().filter(|lease| matches!(lease.origin, LeaseOrigin::Camera)).count(), CAMERA_LEASES);
        assert_eq!(r.visible_leases.values().filter(|lease| matches!(lease.origin, LeaseOrigin::Captured(_))).count(), VISIBLE_BLOCKS);
        for lease in r.visible_leases.values_mut() { lease.retiring = true; }
        r.retire_visible_leases_current(&[], &mut FrameWork::default(), &|| false);
        assert_eq!(r.visible_leases.len(), TEMPORARY_LEASES - CAMERA_CANDIDATES);
        let captured_left = r.visible_leases.values().filter(|lease| matches!(lease.origin, LeaseOrigin::Captured(_))).count();
        assert!(captured_left >= VISIBLE_BLOCKS - 8, "mixed Camera work cannot increase captured retirement beyond 128 columns");
        assert!(r.visible_leases.values().all(|lease| lease.retired == 0));
    }

    #[test]
    fn camera_arrival_grazing_strip_preserves_captured_fine_and_view_guards() {
        let (_, mut r, _, at) = current_bridge_fixture();
        let eye = r.current_request.as_ref().unwrap().eye + DVec3::Y * 40.0;
        r.current_request.as_mut().unwrap().eye = eye;
        r.set_ground_clearance(50.0);
        let view = r.camera_view.unwrap();
        let focus = crate::windows::visible_focus(&r.grid, eye, view.forward, 0.0, 1000.0).unwrap();
        assert!(focus.distance(eye) > r.current_request.as_ref().unwrap().lod0);
        let before = r.camera_blocks(eye, &|| false);
        assert!(before.len() > 4 && before.len() <= CAMERA_CANDIDATES);
        assert!(before.iter().all(|&key| {
            let (face, level, i, j) = unpack(key);
            let point = r.grid.ground_point(face, f64::from((i + 2) * (BRICK << level)),
                f64::from((j + 2) * (BRICK << level)));
            (point - eye).dot(view.forward) > 0.0
        }), "behind-eye/offscreen under-camera tiles do not consume the cap");
        let block = *before.last().unwrap();
        r.visible_admission.push_back((3, pack(key0(crate::grid::PLANE_FACE, 3, 128), 128)));
        r.prioritize_visible_blocks_from([(block as u32, (block >> 32) as u32)], 10, 7, at);
        r.refresh_visible_pending(None);
        let stamp = r.visible_leases[&block].captured_source();
        r.refresh_camera_pending(&before, None);
        let first = r.visible_admission.front().unwrap().1;
        assert_eq!(first & !(3u64 | (3u64 << 32)), before[0]);
        assert_eq!(r.visible_leases[&block].captured_source().at, stamp.at);
        assert_eq!(r.visible_leases[&block].captured_source().frame, stamp.frame);
        let high = eye + DVec3::Y * (r.current_request.as_ref().unwrap().outer_radius + 1_000.0);
        r.current_request.as_mut().unwrap().eye = high;
        r.set_ground_clearance(1_000.0);
        assert!(r.camera_blocks(high, &|| false).is_empty());
        r.current_request.as_mut().unwrap().eye = eye;
        r.set_ground_clearance(50.0);
        r.levels[0].active = false;
        assert!(r.camera_blocks(eye, &|| false).is_empty());
    }

    #[test]
    fn camera_arrival_selection_departure_releases_lease_without_expanding_cap() {
        let (_, mut r, _, _) = current_bridge_fixture();
        let eye = r.current_request.as_ref().unwrap().eye;
        r.set_ground_clearance(10.0);
        let current = r.camera_blocks(eye, &|| false);
        r.refresh_camera_pending(&current, None);
        let old = current[0];
        assert!(r.current_camera_block(old));
        r.set_camera_view(DVec3::new(-1.0, -0.17, 0.0), DVec3::Y, [0.65, 0.414]);
        let turned = r.camera_blocks(eye, &|| false);
        assert!(!turned.contains(&old) && r.current_camera_block(old),
            "a turned-away tile may remain inside the radial window");
        r.frame += 1;
        r.retire_visible_leases_current(&turned, &mut FrameWork::default(), &|| false);
        assert!(!r.transient_wanted(old));
        r.refresh_camera_pending(&turned, None);
        assert!(r.visible_leases.len() <= TEMPORARY_LEASES);
    }

    #[test]
    fn camera_arrival_radial_down_view_requests_ground_and_up_view_does_not() {
        let (_, mut r, _, _) = current_bridge_fixture();
        let eye = r.current_request.as_ref().unwrap().eye;
        r.set_ground_clearance(10.0);
        r.set_camera_view(-DVec3::Y, DVec3::Z, [0.65, 0.414]);
        let down = r.camera_blocks(eye, &|| false);
        assert!(!down.is_empty() && down.len() <= CAMERA_CANDIDATES);
        assert!(down.iter().all(|&key| r.current_camera_block(key)));
        r.refresh_camera_pending(&down, None);
        assert!(r.visible_leases.values().all(|lease| matches!(lease.origin, LeaseOrigin::Camera)));
        r.set_camera_view(DVec3::Y, DVec3::Z, [0.65, 0.414]);
        r.current_request.as_mut().unwrap().outer_radius = 0.0;
        assert!(r.camera_blocks(eye, &|| false).is_empty());
    }

    #[test]
    fn camera_arrival_raised_visible_tile_survives_flat_proxy_and_missed_ground() {
        for rising_view in [false, true] {
            let (mut planet, mut r, _, _) = current_bridge_fixture();
            let eye = DVec3::new(0.0, 50.0, 0.0);
            let actual = DVec3::new(8.0, if rising_view { 56.0 } else { 46.0 }, 0.0);
            std::sync::Arc::make_mut(&mut planet).apply(crate::edits::Brush {
                center: actual.to_array(), ..test_brush(2.0)
            }).unwrap();
            let request = r.current_request.as_mut().unwrap();
            request.eye = eye;
            request.lod0 = 30.0;
            request.outer_radius = planet.outer_radius();
            request.planet = Some(planet.clone());
            r.set_ground_clearance(50.0);
            r.set_camera_view(DVec3::new(1.0, if rising_view { 0.8 } else { -0.17 }, 0.0), DVec3::Y, [0.65, 0.414]);
            let view = r.camera_view.unwrap();
            let relative = actual - eye;
            assert!(relative.dot(view.forward) > 0.0
                && relative.dot(view.up).abs() < relative.dot(view.forward) * view.tan_half[1]);
            let proxy = DVec3::new(actual.x, 0.0, actual.z) - eye;
            assert!(proxy.dot(view.up).abs() > proxy.dot(view.forward) * view.tan_half[1] + 3.2 * 1.75);
            if rising_view {
                assert!(crate::windows::visible_focus(&r.grid, eye,
                    view.forward - view.up * view.tan_half[1], 0.0, 240.0).is_none());
            }
            let (cell, _) = r.grid.locate(actual);
            let block = pack(key0(cell.face, 0, (cell.i >> 3) & !3), ((cell.j >> 3) & !3) as u32);
            let current = r.camera_blocks(eye, &|| false);
            assert!(current.len() <= CAMERA_CANDIDATES && current.contains(&block));
            assert!(r.current_camera_block(block), "under-eye clearance is not a distance bound on a raised ridge");
        }
    }

    #[test]
    fn camera_arrival_selected_deadline_tail_retains_current_frame_ownership() {
        let (_, mut r, _, _) = current_bridge_fixture();
        let eye = r.current_request.as_ref().unwrap().eye;
        r.set_ground_clearance(10.0);
        let current = r.camera_blocks(eye, &|| false);
        r.refresh_camera_pending(&current, None);
        assert!(r.visible_leases.len() > 8);
        let tail = *r.visible_leases.keys().last().unwrap();
        for (&key, lease) in &mut r.visible_leases {
            if key != tail { lease.retiring = true; }
        }
        let old_frame = r.frame;
        r.frame += 1;
        assert_eq!(r.visible_leases[&tail].current_demand_frame, Some(old_frame));
        let checks = std::cell::Cell::new(0);
        r.retire_visible_leases_current(&[tail], &mut FrameWork::default(), &|| {
            checks.set(checks.get() + 1);
            checks.get() > 128
        });
        assert_eq!(r.visible_leases[&tail].current_demand_frame, Some(r.frame));
        assert!(r.transient_wanted(tail) && r.protected_wanted(tail));
        assert!(r.visible_leases.len() > 1, "deadline leaves an unvisited ownership tail");
        assert!(matches!(r.visible_leases[&tail].origin, LeaseOrigin::Camera));
    }

    #[test]
    fn camera_arrival_overlapping_bands_cover_the_middle_grazing_strip() {
        let (_, mut r, _, _) = current_bridge_fixture();
        let eye = r.current_request.as_ref().unwrap().eye;
        r.current_request.as_mut().unwrap().lod0 = 88.0;
        r.set_ground_clearance(10.0);
        r.levels[1].active = true;
        r.levels[2].active = true;
        let current = r.camera_blocks(eye, &|| false);
        assert!(current.len() <= CAMERA_CANDIDATES);
        assert!(current.windows(2).all(|pair| unpack(pair[0]).1 <= unpack(pair[1]).1),
            "current fine demand must precede a larger coarser tile");
        assert!(current.iter().any(|&key| unpack(key).1 == 0));
        assert!(current.iter().any(|&key| unpack(key).1 > 0), "projected coverage cannot spend every slot on L0");
        let view = r.camera_view.unwrap();
        for ahead in [40.0, 50.0, 60.0] {
            let depth = (DVec3::new(ahead, 0.0, eye.z) - eye).dot(view.forward);
            for fraction in [-1.0, -0.5, 0.0, 0.5, 1.0] {
                let lateral = depth * view.tan_half[0] * fraction;
                let covered = current.iter().any(|&key| {
                    let (face, level, i, j) = unpack(key);
                    let cells = BRICK << level;
                    let a = r.grid.ground_point(face, f64::from(i * cells), f64::from(j * cells));
                    let b = r.grid.ground_point(face, f64::from((i + 4) * cells), f64::from((j + 4) * cells));
                    a.x.min(b.x) <= ahead && a.x.max(b.x) >= ahead
                        && a.z.min(b.z) <= eye.z + lateral && a.z.max(b.z) >= eye.z + lateral
                });
                assert!(covered, "frustum band has missing demand at ahead {ahead}, lateral {lateral}");
            }
        }
        for lateral in [-16.0, -8.0, 0.0, 8.0, 16.0] {
            let mut intervals = Vec::new();
            for &key in &current {
                let (face, level, i, j) = unpack(key);
                let cells = BRICK << level;
                let a = r.grid.ground_point(face, f64::from(i * cells), f64::from(j * cells));
                let b = r.grid.ground_point(face, f64::from((i + 4) * cells), f64::from((j + 4) * cells));
                let strip = eye.z + lateral;
                if a.z.min(b.z) <= strip && a.z.max(b.z) >= strip {
                    intervals.push((a.x.min(b.x), a.x.max(b.x)));
                }
            }
            intervals.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
            let mut reached = 40.0;
            for (start, end) in intervals {
                if start <= reached + 1.0e-6 { reached = reached.max(end); }
            }
            assert!(reached >= 60.0, "visible strip at lateral {lateral} leaves 40–60m without current demand; reached {reached}");
        }
    }

    #[test]
    fn current_revalidation_rejects_remote_inactive_stale_and_deadline() {
        for cause in 0..6 {
            let (_, mut r, block, at) = current_bridge_fixture();
            match cause {
                0 => r.current_request.as_mut().unwrap().eye += DVec3::X * 1000.0,
                1 => r.levels[0].active = false,
                2 => r.set_visible_view(19, 7, at),
                3 => r.set_visible_view(10, 8, at),
                4 => r.set_visible_view(10, 7, at + VISIBLE_LEASE_TIME * 2),
                _ => {}
            }
            r.prioritize_visible_blocks_from([(block as u32, (block >> 32) as u32)], 10, 7, at);
            r.refresh_visible_pending_until(|| cause == 5);
            assert!(r.visible_leases.is_empty() && r.visible_admission.is_empty() && r.levels[0].pending.is_empty());
        }
    }

    #[test]
    fn current_revalidation_keeps_edits_and_quarantines_expired_publication() {
        let (mut planet, mut r, block, at) = current_bridge_fixture();
        std::sync::Arc::make_mut(&mut planet).apply(test_brush(1.5)).unwrap();
        // Keep the worker outstanding: plan must not install an inline wanted set here.
        r.last_request.as_mut().unwrap().outer_radius = planet.outer_radius();
        r.prioritize_visible_blocks_from([(block as u32, (block >> 32) as u32)], 10, 7, at);
        r.refresh_visible_pending(None);
        let request = r.current_request.as_ref().unwrap().clone();
        let first = r.plan(&planet, request.eye, request.lod0, 16);
        assert_eq!(first.jobs.len(), 16);
        assert_eq!(r.requested, 3, "the fixture must leave worker authority outstanding");
        assert!(first.job_keys.iter().all(|key| unpack(*key).1 == 0 && !r.current_wanted(*key)),
            "the admitted jobs must be the leased fine block, not global inline coverage");
        assert!(r.publishing.values().any(|p| p.next.is_some()), "canonical edit lists remain scalar and intact");
        r.applied_levels[0] = r.requested;
        r.frame += 1;
        let mut hold = FrameWork::default();
        r.retire_visible_leases(&mut hold, &|| false);
        assert!(hold.evictions.is_empty());
        assert!(first.job_keys.iter().all(|key| r.protected_wanted(*key)));
        assert!(r.publishing.values().all(|p| !p.evicted));
        r.complete_jobs(first.job_keys.iter().map(|key| (*key, 0)));
        assert!(r.publishing.is_empty() && first.job_keys.iter().all(|key| r.residents.contains_key(*key)));
        // A departure invalidates the per-frame exception even before original TTL.
        r.current_request.as_mut().unwrap().eye += DVec3::X * 1000.0;
        r.frame += 1;
        let mut expired = FrameWork::default();
        r.retire_visible_leases(&mut expired, &|| false);
        assert_eq!(expired.evictions.len(), 16);
        assert!(r.visible_leases.is_empty() && r.blocks.is_empty());
        table_is_exact(&r);
    }

    #[test]
    fn current_revalidation_transfers_to_authority_and_expires_inflight_without_refresh() {
        let (planet, mut r, block, at) = current_bridge_fixture();
        r.prioritize_visible_blocks_from([(block as u32, (block >> 32) as u32)], 10, 7, at);
        r.refresh_visible_pending(None);
        let request = r.current_request.as_ref().unwrap().clone();
        let work = r.plan(&planet, request.eye, request.lod0, 16);
        assert_eq!(work.jobs.len(), 16);
        assert_eq!(r.requested, 3, "the fixture must leave worker authority outstanding");
        assert!(work.job_keys.iter().all(|key| unpack(*key).1 == 0 && !r.current_wanted(*key)),
            "the admitted jobs must be the leased fine block, not global inline coverage");
        let keys = work.job_keys.clone();
        // A live publication survives repeated acknowledgment while independently current.
        r.applied_levels[0] = r.requested;
        r.frame += 1;
        r.retire_visible_leases(&mut FrameWork::default(), &|| false);
        assert!(r.publishing.values().all(|p| !p.evicted));
        // Original wall-clock TTL still quarantines in-flight records.
        r.set_visible_view(11, 7, at + VISIBLE_LEASE_TIME + std::time::Duration::from_millis(1));
        let mut expired = FrameWork::default();
        r.retire_visible_leases(&mut expired, &|| false);
        assert_eq!(expired.evictions.len(), 16);
        assert!(r.publishing.values().all(|p| p.evicted));
        assert_eq!(r.visible_leases.len(), 1);
        r.complete_jobs(keys.iter().map(|&key| (key, 0)));
        r.retire_visible_leases(&mut expired, &|| false);
        assert!(r.visible_leases.is_empty() && r.publishing.is_empty());
        table_is_exact(&r);

        let (_, mut r, block, at) = current_bridge_fixture();
        r.prioritize_visible_blocks_from([(block as u32, (block >> 32) as u32)], 10, 7, at);
        r.refresh_visible_pending(None);
        let (keys, count) = r.visible_columns(block).unwrap();
        r.levels[0].wanted = Some(std::sync::Arc::new(keys[..count].iter().copied().collect()));
        r.applied_levels[0] = 3;
        let mut handoff = FrameWork::default();
        r.retire_visible_leases(&mut handoff, &|| false);
        assert!(r.visible_leases.is_empty() && handoff.evictions.is_empty());
        assert!(keys.iter().all(|&key| r.current_wanted(key)));
    }


    /// Capture hash iteration order first, then modify only values. This
    /// deterministically puts a current bounded lease behind retirement's
    /// exact 128-step limit (or behind an already elapsed deadline), without
    /// assuming any particular hash order.
    fn current_revalidation_retirement_tail(expired_blocks: usize, elapsed_deadline: bool) {
        let (planet, mut r, base, at) = current_bridge_fixture();
        let (face, level, i, j) = unpack(base);
        r.set_visible_view(10, 7, at + std::time::Duration::from_millis(3));
        for offset in 0..expired_blocks + 2 {
            let block = pack(key0(face, level, i + offset as i32 * 4), j as u32);
            assert!(r.current_captured_block(block));
            assert_eq!(r.visible_columns(block).unwrap().1, 16);
            r.visible_leases.insert(block, VisibleLease {
                origin: LeaseOrigin::Captured(VisibleStamp { frame: 10, view: 7, at }),
                serial: 3, retiring: false, retired: 0,
                current_demand_frame: Some(r.frame),
            });
        }
        let ordered: Vec<_> = r.visible_leases.keys().copied().collect();
        for &block in &ordered[..expired_blocks] {
            r.visible_leases.get_mut(&block).unwrap().retiring = true;
        }
        let bounded = ordered[expired_blocks];
        let legacy = ordered[expired_blocks + 1];
        let lease = r.visible_leases.get_mut(&legacy).unwrap();
        lease.current_demand_frame = None;
        if let LeaseOrigin::Captured(source) = &mut lease.origin { source.at = at + std::time::Duration::from_millis(2); }
        let (keys, count) = r.visible_columns(bounded).unwrap();
        let wanted = keys[1];
        r.levels[0].wanted = Some(std::sync::Arc::new([wanted].into_iter().collect()));
        for &key in &keys[..count] { r.levels[0].pending.insert(key, 0); }
        r.visible_admission = keys[1..count].iter().map(|&key| (0, key)).collect();
        let original = (r.visible_leases[&bounded].captured_source().frame,
            r.visible_leases[&bounded].captured_source().at, r.visible_leases[&bounded].serial);
        let old_frame = r.frame;
        r.frame += 1;
        let mut retired = FrameWork::default();
        r.retire_visible_leases(&mut retired, &|| elapsed_deadline);
        // The final retiring block causes the early return before visiting
        // either tail lease. Both remain live and the bounded marker is old.
        assert_eq!(r.visible_leases[&ordered[expired_blocks - 1]].retired, 0);
        assert_eq!(r.visible_leases[&bounded].current_demand_frame, Some(old_frame));
        assert!(!r.visible_leases[&bounded].retiring);
        assert!(r.applied_levels[0] < r.visible_leases[&bounded].serial);
        assert!(r.source_is_current(r.visible_leases[&bounded].captured_source(), VISIBLE_LEASE_FRAMES));
        assert!(!r.transient_wanted(bounded), "unvisited current-bound demand must fail closed");
        assert!(!r.protected_wanted(bounded));
        assert!(r.protected_wanted(wanted), "ordinary authority membership remains protected");
        assert!(r.transient_wanted(legacy), "an unacknowledged original bridge is unchanged");
        assert!(!r.admit_visible_block(&planet, 0, bounded, 0, 16, None, &mut FrameWork::default()),
            "a complete ranked block cannot use an unvalidated tail lease");
        // A later plan may revalidate the same original capture; it must
        // restore only geometric permission, never its serial or TTL.
        r.frame += 1;
        r.retire_visible_leases(&mut retired, &|| false);
        assert_eq!(r.visible_leases[&bounded].current_demand_frame, Some(r.frame));
        assert!(r.transient_wanted(bounded) && r.protected_wanted(bounded));
        assert!(r.transient_wanted(legacy));
        assert_eq!(original, (r.visible_leases[&bounded].captured_source().frame,
            r.visible_leases[&bounded].captured_source().at, r.visible_leases[&bounded].serial));
    }

    #[test]
    fn current_revalidation_unvisited_128_step_tail_fails_closed_then_recovers() {
        current_revalidation_retirement_tail(9, false);
    }

    #[test]
    fn current_revalidation_unvisited_deadline_tail_fails_closed_then_recovers() {
        current_revalidation_retirement_tail(1, true);
    }

}
