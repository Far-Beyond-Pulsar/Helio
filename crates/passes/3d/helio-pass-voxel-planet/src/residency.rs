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
const CAMERA_CANDIDATES: usize = 3 * 128;
// Existing bounded storage for current demand and departing publications.
const CAMERA_LEASES: usize = 384;
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
        self.take(key).is_some()
    }
    fn take(&mut self, key: u64) -> Option<usize> {
        let (bucket, index) = self.at.remove(&key)?;
        let list = &mut self.buckets[bucket as usize];
        list.swap_remove(index as usize);
        if let Some(&moved) = list.get(index as usize) {
            self.at.get_mut(&moved).unwrap().1 = index;
        }
        Some(bucket as usize)
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
        let Some(bucket) = level.pending.take(key) else { continue };
        return Some((index, key, bucket));
    }
    let (index, _) = select_pending_level(levels, top_level, cached)?;
    let (key, bucket) = levels[index].pending.pop()?;
    Some((index, key, bucket))
}

fn current_geometry_allows(grid: &Grid, request: Option<&WindowRequest>, key: u64) -> bool {
    unpack(key).1 >= 3 || request.is_none_or(|request|
        crate::windows::current_block_wanted(grid, request, key & !(3u64 | (3u64 << 32))))
}

type SelectedOwners = FxHashMap<u32, (u32, u8, u32, i32, i32)>;

fn selected_owner_allows(key: u64, selected: &SelectedOwners) -> bool {
    let (face, level, i, j) = unpack(key);
    if selected.is_empty() { return true; }
    (1..=BLOCK_TIERS).all(|tier| {
        let owner = (level, face, tier, i >> (2 * tier), j >> (2 * tier));
        selected.get(&block_slot(level, face, tier, owner.3, owner.4)).is_none_or(|wanted| *wanted == owner)
    })
}

fn select_camera_owners(blocks: &mut Vec<u64>) -> SelectedOwners {
    let mut selected = SelectedOwners::default();
    blocks.retain(|&key| {
        if !selected_owner_allows(key, &selected) { return false; }
        let (face, level, i, j) = unpack(key);
        for tier in 1..=BLOCK_TIERS {
            let owner = (level, face, tier, i >> (2 * tier), j >> (2 * tier));
            selected.insert(block_slot(level, face, tier, owner.3, owner.4), owner);
        }
        true
    });
    selected
}

fn cached_current_geometry(grid: &Grid, request: Option<&WindowRequest>, key: u64,
    last: &mut Option<(u64, bool)>, selected: &SelectedOwners) -> bool {
    let block = key & !(3u64 | (3u64 << 32));
    if let Some((_, allowed)) = last.filter(|&(previous, _)| previous == block) { return allowed; }
    let allowed = selected_owner_allows(key, selected) && current_geometry_allows(grid, request, key);
    *last = Some((block, allowed));
    allowed
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
    /// selected tiles (any ownership), for camera_base_level and its next
    /// two levels. Jobs are not completions.
    pub camera_base_level: u32,
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
    /// Last acknowledged job batch hit the GPU brick pool limit.
    pool_pressure: bool,
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
    /// Resume an interrupted intact-tile preflight through scalar admission.
    timed_out_batch: Option<u64>,
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
    selected: bool,
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
            pool_pressure: false,
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
            timed_out_batch: None,
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
            self.block_owner.get(&block_slot(level, face, tier, bkey.3, bkey.4)).is_some_and(|owner| *owner != bkey)
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
                self.attach_summary_owner(bkey, work);
                continue;
            }
            let slot = block_slot(level, face, tier, bi, bj);
            self.block_owner.insert(slot, bkey);
            self.dirty_summary_chain(bkey, work);
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
                self.blocks.remove(&bkey);
                self.detach_summary_owner(bkey, work);
            }
        }
        // A retained detached child's later eviction may still decrement an
        // attached parent. Recount that parent's actual attached children.
        self.dirty_summary_chain((level, face, 1, ci >> 2, cj >> 2), work);
    }

    fn dirty_summary_chain(&self, owner: (u32, u8, u32, i32, i32), work: &mut FrameWork) {
        let (level, face, tier, i, j) = owner;
        for ancestor in tier..=BLOCK_TIERS {
            let shift = 2 * (ancestor - tier);
            let wanted = (level, face, ancestor, i >> shift, j >> shift);
            let slot = block_slot(level, face, ancestor, wanted.3, wanted.4);
            // Never reset a replacement occupying the same physical slot.
            if self.block_owner.get(&slot) == Some(&wanted) {
                work.block_inits.push((slot, wanted.3, wanted.4));
            }
        }
    }

    fn dirty_detached_publications(&self, work: &mut FrameWork) {
        let mut previous = None;
        for index in 0..work.job_keys.len() {
            let key = work.job_keys[index] & !(3u64 | (3u64 << 32));
            if previous == Some(key) { continue; }
            previous = Some(key);
            let (face, level, i, j) = unpack(key);
            let child = (level, face, 1, i >> 2, j >> 2);
            let slot = block_slot(level, face, 1, child.3, child.4);
            if self.block_owner.get(&slot) != Some(&child) {
                // Urgent regeneration and scalar initial retries can publish a
                // retained detached record. Parents must exclude its absent
                // child even if publication increments their matching identity.
                self.dirty_summary_chain(child, work);
            }
        }
    }

    fn detach_summary_owner(&mut self, owner: (u32, u8, u32, i32, i32), work: &mut FrameWork) {
        let (level, face, tier, i, j) = owner;
        let slot = block_slot(level, face, tier, i, j);
        if self.block_owner.get(&slot) != Some(&owner) { return; }
        self.block_owner.remove(&slot);
        work.block_inits.push((slot, -1, -1));
        self.dirty_summary_chain(owner, work);
        if tier == 1 {
            if let Some(at) = self.live_index.remove(&slot) {
                self.live_tier1.swap_remove(at);
                if let Some(&moved) = self.live_tier1.get(at) { self.live_index.insert(moved, at); }
                self.live_dirty = true;
            }
        }
    }

    fn attach_summary_owner(&mut self, owner: (u32, u8, u32, i32, i32), work: &mut FrameWork) {
        let Some(block) = self.blocks.get(&owner) else { return };
        let slot = block.slot;
        if self.block_owner.get(&slot) == Some(&owner) { return; }
        debug_assert!(!self.block_owner.contains_key(&slot));
        self.block_owner.insert(slot, owner);
        self.dirty_summary_chain(owner, work);
        if owner.2 == 1 {
            self.live_index.insert(slot, self.live_tier1.len());
            self.live_tier1.push(slot);
            self.live_dirty = true;
        }
    }

    fn handoff_camera_owners(&mut self, selected: &SelectedOwners, work: &mut FrameWork,
        out_of_time: &impl Fn() -> bool) {
        for (&slot, &wanted) in selected {
            if out_of_time() { break; }
            if let Some(&owner) = self.block_owner.get(&slot) {
                if owner != wanted { self.detach_summary_owner(owner, work); }
            }
            // Only referenced blocks can attach: missing demand acquires
            // summaries during normal admission, never as zero-ref ghosts.
            self.attach_summary_owner(wanted, work);
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

    fn camera_base_level(&self) -> usize {
        let (Some(clearance), Some(request)) = (self.ground_clearance, self.current_request.as_ref())
            else { return 0 };
        (0..self.levels.len().saturating_sub(2)).find(|&level|
            clearance.abs() <= request.lod0 * f64::from(1u32 << level) * 0.5)
            .unwrap_or_else(|| self.levels.len().saturating_sub(3))
    }

    fn current_camera_block(&self, key: u64) -> bool {
        let (face, level, i, j) = unpack(key);
        let base = self.camera_base_level();
        if !(base..(base + 3).min(self.levels.len())).contains(&(level as usize))
            || self.visible_columns(key).is_none() { return false; }
        self.current_request.as_ref().is_some_and(|request| {
            if !request.eye.is_finite() || !request.lod0.is_finite() || request.lod0 <= 0.0
                || !request.outer_radius.is_finite() { return false; }
            let size = f64::from(BRICK << level);
            let point = self.grid.ground_point(face, f64::from(i + 2) * size, f64::from(j + 2) * size);
            let reach = request.lod0 * f64::from(1u32 << level);
            self.grid.radial(request.eye) - request.outer_radius <= reach
                && self.grid.ground_distance(request.eye, point) <= reach + self.grid.level_size(level) * 32.0
        })
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
        // Stamping selected demand is necessary for the later admission pass.
        // Expiry scans must not consume its reserved time after preparation ends.
        if out_of_time() { return; }
        // Snapshot Camera demand is a current-frame block lease, not an
        // eviction list. Keep exact records and journals for background cleanup.
        if self.snapshot_mode {
            let mut expired_camera = FxHashSet::default();
            self.visible_leases.retain(|&block, lease| {
                let keep = !matches!(lease.origin, LeaseOrigin::Camera)
                    || !lease.retiring && camera_blocks.contains(&block);
                if !keep { expired_camera.insert(block); }
                keep
            });
            if !expired_camera.is_empty() {
                self.visible_admission.retain(|&(_, key)| !expired_camera.contains(&(key & !(3u64 | (3u64 << 32)))));
                // A view-only turn does not request new worker membership.
                // Wake its existing cursor without restarting an active scan.
                self.snapshot_epoch += 1;
                if self.retire_slot == 0 { self.retire_started_epoch = self.snapshot_epoch; }
            }
        }
        let blocks: Vec<_> = self.visible_leases.keys().copied().collect();
        let mut steps = 0;
        let mut captured_steps = 0;
        // Legacy diffs retain their original bounded expiry walk: their
        // explicit removals cannot be consumed as snapshot retirement.
        let limit = if !self.snapshot_mode && self.visible_leases.iter().any(|(&block, lease)|
            matches!(lease.origin, LeaseOrigin::Camera) && (lease.retiring || !camera_blocks.contains(&block))) {
            CAMERA_CANDIDATES * 16
        } else { 128 };
        for block in blocks {
            let lease = &self.visible_leases[&block];
            if self.snapshot_mode && matches!(lease.origin, LeaseOrigin::Camera) { continue; }
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
        if let Some(deadline) = deadline {
            self.refresh_visible_pending_cohort_until(8, || std::time::Instant::now() >= deadline);
        } else {
            self.refresh_visible_pending_until(|| false);
        }
    }

    /// Ordinary columns can publish without summary references while a slot
    /// aliases. A complete selected resident tile must acquire those missing
    /// references after handoff, even though it needs no generation job.
    fn promote_camera_residents(&mut self, blocks: &[u64], selected: &SelectedOwners,
        work: &mut FrameWork, out_of_time: &impl Fn() -> bool) {
        'tiles: for &block in blocks {
            if out_of_time() { break; }
            if !selected_owner_allows(block, selected) || self.blocks_conflict(block) { continue; }
            let Some((keys, count)) = self.visible_columns(block) else { continue };
            let (face, level, i, j) = unpack(block);
            let owner = (level, face, 1, i >> 2, j >> 2);
            if self.blocks.get(&owner).is_some_and(|state| state.refs == count as u32) { continue; }
            let mut missing = [false; 16];
            let mut references = 0;
            for (index, &key) in keys[..count].iter().enumerate() {
                if out_of_time() { break 'tiles; }
                let Some(resident) = self.residents.get(key) else { continue 'tiles };
                if self.publishing.get(&key).is_some_and(|publication|
                    publication.evicted || publication.record != resident.record) { continue 'tiles; }
                missing[index] = !resident.blocks;
                references += u32::from(!resident.blocks);
            }
            if references == 0 { continue; }
            if out_of_time() { break; }
            // The complete preflight owns exact record identities. No records,
            // edits, publication acknowledgments or pending keys are replaced.
            self.reference_blocks_count(block, references, work);
            for (index, &key) in keys[..count].iter().enumerate() {
                if missing[index] { self.residents.get_mut(key).unwrap().blocks = true; }
            }
            self.block_conflicts -= references as usize;
            self.dirty_summary_chain(owner, work);
        }
    }

    fn refresh_visible_pending_until(&mut self, out_of_time: impl FnMut() -> bool) {
        self.refresh_visible_pending_cohort_until(VISIBLE_BLOCKS, out_of_time);
    }

    fn refresh_visible_pending_cohort_until(&mut self, cohort_limit: usize, mut out_of_time: impl FnMut() -> bool) {
        // Do not scan or rebuild the existing FIFO after this phase expired.
        // Captured requests keep their original source for the next frame.
        if out_of_time() { return; }
        let mut captured_leases = self.visible_leases.values()
            .filter(|lease| matches!(lease.origin, LeaseOrigin::Captured(_))).count();
        let rank_requests = self.visible_rank_source.is_none_or(|source| self.source_is_current(source, 8));
        if !rank_requests {
            self.visible_admission.clear();
            self.visible_rank_source = None;
            self.visible_blocks.clear();
        }
        if out_of_time() { return; }
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
        let mut staged = Vec::new();
        'tiles: for (index, &key) in blocks.iter().enumerate() {
            if staged.len() == cohort_limit || out_of_time() {
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
            let mut pending = Vec::with_capacity(count);
            for &column in &keys[..count] {
                if out_of_time() {
                    if index < requested_blocks {
                        self.visible_blocks.extend_from_slice(&blocks[index..requested_blocks]);
                        self.visible_source = source;
                    }
                    break 'tiles;
                }
                if !self.publishing.contains_key(&column)
                    && (self.initial_retries.contains(&column) || !self.residents.contains_key(column)) {
                    pending.push(column);
                }
            }
            staged.push((key, level, pending, index < requested_blocks));
        }
        // Remove the complete cohort before reverse insertion: swap removal
        // cannot fragment an earlier tile, and nearest tiles pop first even
        // when captured distance ranks expired. Budgeted cohorts contain at
        // most 128 columns, rather than an unchecked 1024-column final commit.
        for (_, level, pending, _) in &staged {
            for &column in pending { self.levels[*level].pending.remove(column); }
        }
        for (_, level, pending, _) in staged.iter().rev() {
            for &column in pending.iter().rev() { self.levels[*level].pending.insert(column, 0); }
        }
        for (key, level, pending, requested) in staged {
            for column in pending {
                // Lease reinsertion never renews expired distance ranks.
                if rank_requests && requested
                    && self.visible_admission.len() < VISIBLE_ADMISSION_COLUMNS
                    && ranked.insert((level, column)) {
                    self.visible_admission.push_back((level, column));
                }
            }
            self.queue_obsolete_owners(key);
        }
    }

    fn camera_blocks(&self, eye: DVec3, out_of_time: &impl Fn() -> bool) -> Vec<u64> {
        let (Some(clearance), Some(view), Some(request)) =
            (self.ground_clearance, self.camera_view, self.current_request.as_ref()) else { return Vec::new() };
        fn clip(poly: &mut Vec<[f64; 2]>, normal: [f64; 2], offset: f64) {
            let input = std::mem::take(poly);
            let Some(mut previous) = input.last().copied() else { return };
            let signed = |p: [f64; 2]| p[0] * normal[0] + p[1] * normal[1] + offset;
            let mut before = signed(previous);
            for point in input {
                let after = signed(point);
                if (before >= 0.0) != (after >= 0.0) {
                    let t = before / (before - after);
                    poly.push([previous[0] + (point[0] - previous[0]) * t,
                        previous[1] + (point[1] - previous[1]) * t]);
                }
                if after >= 0.0 { poly.push(point); }
                previous = point;
                before = after;
            }
        }
        let grid = self.grid;
        let ground_radial = grid.radial(eye) - clearance;
        let radial_up = grid.up(eye);
        let direction = (view.forward - radial_up * view.forward.dot(radial_up)).try_normalize()
            .unwrap_or_else(|| view.up - radial_up * view.up.dot(radial_up));
        let side = direction.cross(radial_up).normalize_or_zero();
        let ground = eye - radial_up * clearance;
        let bottom = view.forward - view.up * view.tan_half[1];
        let downward = -bottom.dot(radial_up);
        let entry_reach = (downward > 0.0).then(|| [-1.0, 0.0, 1.0].into_iter().map(|x| {
            let ray = bottom + view.right * (view.tan_half[0] * x);
            clearance.max(0.0) * (ray - radial_up * ray.dot(radial_up)).length()
                / (-ray.dot(radial_up)).max(downward * 0.25)
        }).fold(0.0, f64::max));
        let planes = [view.forward,
            view.forward * view.tan_half[0] + view.right,
            view.forward * view.tan_half[0] - view.right,
            view.forward * view.tan_half[1] + view.up,
            view.forward * view.tan_half[1] - view.up];
        let mut candidates = Vec::new();
        let mut inspected = FxHashSet::default();
        let base = self.camera_base_level();
        'levels: for level in base..(base + 3).min(self.levels.len()) {
            let size = f64::from(BRICK << level);
            let width = grid.level_size(level as u32) * f64::from(BRICK * 4);
            let tile_radius = width * std::f64::consts::FRAC_1_SQRT_2
                * if grid.is_plane() { 1.0 } else { (ground_radial / grid.radius()).max(1.0) };
            // This band needs level L wherever L+1 would exceed four pixels.
            // Whole tiles overlap its edges rather than moving the band to fit
            // an arbitrary square centered near the lower screen edge.
            let far = request.lod0 * f64::from(1u32 << level) * 0.5;
            let near = if level == 0 { 0.0 } else { far * 0.5 };
            let reach = request.lod0 * f64::from(1u32 << level) / (1.0 - request.lod_dither * 0.5);
            let radial_lo = request.planet.as_ref().map_or(ground_radial, |planet| planet.inner_radius())
                .max(grid.radial(eye) - reach);
            let radial_hi = request.outer_radius.min(grid.radial(eye) + reach);
            if radial_lo > radial_hi { continue; }
            let mut blocks = Vec::new();
            // A ray starts in air columns before its first ground hit. Discover
            // that small swept corridor first, so the fixed 512-probe ceiling
            // cannot omit the traversal dependencies behind the ground wedge.
            for corridor in [true, false] {
                if corridor && entry_reach.is_none() { continue; }
                if clearance.abs() > far + tile_radius { continue; }
                let radius = if corridor { entry_reach.unwrap().min(far) + tile_radius }
                    else { ((far + tile_radius).powi(2) - clearance.powi(2)).max(0.0).sqrt() };
                let circumscribed = radius / (std::f64::consts::PI / 8.0).cos();
                let mut polygon: Vec<_> = (0..8).map(|i| {
                    let a = (f64::from(i) + 0.5) * std::f64::consts::FRAC_PI_4;
                    [a.cos() * circumscribed, a.sin() * circumscribed]
                }).collect();
                let curvature = if grid.is_plane() { 0.0 } else { radius * radius / ground_radial.max(1.0) };
                for normal in planes {
                    clip(&mut polygon, [normal.dot(direction), normal.dot(side)],
                        normal.dot(ground - eye) + tile_radius * normal.length()
                            + curvature * normal.dot(radial_up).abs()
                            + if corridor { normal.dot(radial_up).max(0.0) * clearance.max(0.0) } else { 0.0 });
                }
                for &face in grid.faces() {
                    if polygon.is_empty() { break; }
                    let Some(mut poly) = polygon.iter().map(|p| grid.face_coords(face,
                        ground + direction * p[0] + side * p[1])
                        .map(|c| [c[0] / (size * 4.0), c[1] / (size * 4.0)]))
                        .collect::<Option<Vec<_>>>() else { continue };
                    let edge = f64::from(grid.cells()) / (size * 4.0);
                    for (normal, offset) in [([1.0, 0.0], 0.0), ([-1.0, 0.0], edge),
                        ([0.0, 1.0], 0.0), ([0.0, -1.0], edge)] { clip(&mut poly, normal, offset); }
                    if poly.is_empty() { continue; }
                    let lo = poly.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min).floor().max(0.0) as i32;
                    let hi = poly.iter().map(|p| p[1]).fold(f64::NEG_INFINITY, f64::max).ceil().min(edge) as i32;
                    for y in lo..hi {
                        let row = f64::from(y) + 0.5;
                        let mut xs = Vec::new();
                        for (a, b) in poly.iter().zip(poly.iter().cycle().skip(1)).take(poly.len()) {
                            if (a[1] <= row && b[1] > row) || (b[1] <= row && a[1] > row) {
                                xs.push(a[0] + (b[0] - a[0]) * (row - a[1]) / (b[1] - a[1]));
                            }
                        }
                        if xs.len() < 2 { continue; }
                        let left = xs.iter().copied().fold(f64::INFINITY, f64::min).floor().max(0.0) as i32;
                        let right = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max).ceil().min(edge) as i32;
                        for x in left..right {
                            if out_of_time() { break 'levels; }
                            if blocks.len() == 512 { break; }
                            blocks.push(pack(key0(face, level as u32, x * 4), (y * 4) as u32));
                        }
                    }
                }
            }
            // Raised terrain can enter before the flat ground wedge. Keep a
            // bounded near neighborhood, ranked after real ground coverage.
            let seed = ground + direction * (width * 2.0);
            let face = if grid.is_plane() { crate::grid::PLANE_FACE } else { crate::grid::face_of(seed) };
            if let Some(coords) = grid.face_coords(face, seed) {
                let bi = (coords[0] / size).floor() as i32 / 4;
                let bj = (coords[1] / size).floor() as i32 / 4;
                let edge = grid.cells() / ((BRICK << level) * 4) as i32;
                for y in bj - 4..bj + 4 {
                    for x in bi - 2..bi + 2 {
                        if (0..edge).contains(&x) && (0..edge).contains(&y) {
                            blocks.push(pack(key0(face, level as u32, x * 4), (y * 4) as u32));
                        }
                    }
                }
            }
            for block in blocks {
                if out_of_time() { break 'levels; }
                if !inspected.insert(block) || !self.current_camera_block(block) { continue; }
                let (face, _, i, j) = unpack(block);
                let mut point = grid.ground_point(face, f64::from(i + 2) * size, f64::from(j + 2) * size);
                if grid.is_plane() { point.y = ground_radial; }
                else { point = point.normalize_or_zero() * ground_radial; }
                let relative = point - eye;
                let up = grid.up(point);
                let distance = relative.length();
                let ground_visible = distance - tile_radius <= far && distance + tile_radius >= near
                    && planes.iter().all(|normal| normal.dot(relative)
                        + tile_radius * (*normal - up * normal.dot(up)).length() >= 0.0);
                let horizontal = (point - ground - radial_up * (point - ground).dot(radial_up)).length();
                let entry = entry_reach.is_some_and(|reach| horizontal - tile_radius <= reach.min(far))
                    && planes.iter().all(|normal| normal.dot(relative)
                        + normal.dot(up).max(0.0) * clearance.max(0.0)
                        + tile_radius * (*normal - up * normal.dot(up)).length() >= 0.0);
                let midpoint = point + up * ((radial_lo + radial_hi) * 0.5 - ground_radial);
                let height = (radial_hi - radial_lo) * 0.5;
                if !entry && !ground_visible && planes.iter().any(|normal| normal.dot(midpoint - eye)
                    + height * normal.dot(up).abs() + width * 1.75
                        * (*normal - up * normal.dot(up)).length() < 0.0) { continue; }
                let benefit = width * width / distance.powi(2).max(0.01);
                candidates.push((if entry { 2u8 } else { u8::from(ground_visible) }, benefit, block));
            }
        }
        candidates.sort_unstable_by(|a, b| unpack(a.2).1.cmp(&unpack(b.2).1)
            .then_with(|| b.0.cmp(&a.0)).then_with(|| b.1.total_cmp(&a.1)));
        let mut ground: [Vec<u64>; 3] = std::array::from_fn(|_| Vec::new());
        let mut raised: [Vec<u64>; 3] = std::array::from_fn(|_| Vec::new());
        let mut entry: [Vec<u64>; 3] = std::array::from_fn(|_| Vec::new());
        for (class, _, key) in candidates {
            let band = unpack(key).1 as usize - base;
            let (list, cap) = match class { 2 => (&mut entry[band], 64),
                1 => (&mut ground[band], 160), _ => (&mut raised[band], 8) };
            if list.len() < cap { list.push(key); }
        }
        let mut selected = Vec::with_capacity(CAMERA_CANDIDATES);
        for bands in [&entry, &ground, &raised] {
            for row in 0..bands.iter().map(Vec::len).max().unwrap_or(0) {
                for band in bands {
                    if selected.len() == CAMERA_CANDIDATES { break; }
                    if let Some(&key) = band.get(row) { selected.push(key); }
                }
            }
        }
        // Fair shared-cap selection keeps each band's benefit order. Plan
        // interleaves these coherent whole tiles before bounded admission.
        selected.sort_by_key(|&key| unpack(key).1);
        selected
    }

    // A failed initial publication matters only to its exact 4x4 level tile.
    // Full resident tiles elsewhere can retain their preparation fast path.
    fn tile_has_initial_retry(retries: &FxHashSet<u64>, face: u8, level: u32,
        bi: i32, bj: i32) -> bool {
        if retries.is_empty() { return false; }
        (0..16).any(|index| retries.contains(&pack(
            key0(face, level, bi * 4 + index % 4), (bj * 4 + index / 4) as u32)))
    }

    /// Current camera tiles may arrive before the worker's wanted snapshot.
    /// Reuse its capped lease/publication path and FIFO; no forecast demand.
    fn refresh_camera_pending(&mut self, blocks: &[u64], deadline: Option<std::time::Instant>) {
        self.refresh_camera_pending_until(blocks, || deadline.is_some_and(|at| std::time::Instant::now() >= at));
    }

    fn refresh_camera_pending_until(&mut self, blocks: &[u64], mut out_of_time: impl FnMut() -> bool) {
        if out_of_time() || self.ground_clearance.is_none() { return; }
        let mut candidates = Vec::new();
        let mut refreshed = FxHashSet::default();
        let mut finished_tiles = FxHashSet::default();
        let mut camera_leases = self.visible_leases.values()
            .filter(|lease| matches!(lease.origin, LeaseOrigin::Camera)).count();
        'tiles: for &block in blocks {
            if out_of_time() { break; }
            let level = unpack(block).1 as usize;
            let Some((keys, count)) = self.visible_columns(block) else { continue };
            // Current tile authorization already proves demand for every
            // member. Captured clocks remain unchanged; fresh Camera demand
            // may refresh its existing lease without sixteen wanted probes.
            if let Some(lease) = self.visible_leases.get_mut(&block) {
                if matches!(lease.origin, LeaseOrigin::Camera) && !lease.retiring {
                    lease.current_demand_frame = Some(self.frame);
                }
            }
            let leased = self.transient_wanted(block);
            let ordinary = !leased && keys[..count].iter().all(|&key| self.current_wanted(key));
            if !leased && !ordinary {
                // An existing expired/retiring lease cannot be renewed here.
                if self.visible_leases.contains_key(&block) || camera_leases >= CAMERA_LEASES
                    || self.visible_leases.len() >= TEMPORARY_LEASES { continue; }
                self.visible_leases.insert(block, VisibleLease {
                    origin: LeaseOrigin::Camera, serial: self.requested,
                    retiring: false, retired: 0, current_demand_frame: Some(self.frame),
                });
                camera_leases += 1;
            }
            self.queue_obsolete_owners(block);
            let (face, _, i, j) = unpack(block);
            // Tier-1 refs count exact resident columns, including in-flight
            // publications. Preserve the lease above, but avoid probing all
            // sixteen records again when none can need an initial retry.
            if count == 16
                && self.blocks.get(&(level as u32, face, 1, i >> 2, j >> 2))
                    .is_some_and(|owner| owner.refs == 16)
                && !Self::tile_has_initial_retry(&self.initial_retries, face, level as u32, i >> 2, j >> 2) {
                finished_tiles.insert(block);
                continue;
            }
            let mut pending = Vec::with_capacity(count);
            for &key in &keys[..count] {
                if out_of_time() { break 'tiles; }
                if !self.publishing.contains_key(&key)
                    && (self.initial_retries.contains(&key) || !self.residents.contains_key(key)) {
                    pending.push(key);
                }
            }
            if out_of_time() { break; }
            if !pending.is_empty() {
                for key in pending {
                    self.levels[level].pending.insert(key, 0);
                    candidates.push((level, key));
                }
                refreshed.insert(block);
            }
        }
        if candidates.is_empty() && self.visible_admission.is_empty() { return; }
        // Current view ranks must replace the previous view's ordinary ranks,
        // which have no lease origin. They remain queued at their normal
        // bucket instead of masquerading as permanently captured fine work.
        let current: FxHashSet<_> = blocks.iter().copied().collect();
        let mut remaining = std::mem::take(&mut self.visible_admission);
        let mut last_finished = None;
        remaining.retain(|entry| {
            let block = entry.1 & !(3u64 | (3u64 << 32));
            // Intake can stop before reaching this tile. Check its coherent
            // FIFO run once too, without probing any resident columns.
            if last_finished.is_none_or(|previous| previous != block) {
                let (face, level, i, j) = unpack(block);
                if self.blocks.get(&(level, face, 1, i >> 2, j >> 2)).is_some_and(|owner| owner.refs == 16)
                    && !Self::tile_has_initial_retry(&self.initial_retries, face, level, i >> 2, j >> 2) {
                    finished_tiles.insert(block);
                }
                last_finished = Some(block);
            }
            !finished_tiles.contains(&block) && !refreshed.contains(&block) && (current.contains(&block)
                || self.visible_leases.get(&block)
                    .is_some_and(|lease| matches!(lease.origin, LeaseOrigin::Captured(_))))
        });
        // Move the selected column allocation directly into the FIFO. Ranking
        // replacement needs tile identities, not a second per-column hash set.
        let mut fine: VecDeque<_> = candidates.into();
        fine.truncate(VISIBLE_ADMISSION_COLUMNS);
        // A captured ridge can require a finer level than the under-eye ground
        // bands. Keep that accepted whole-tile prefix and its original clocks.
        let base = self.camera_base_level();
        let mut finer = VecDeque::new();
        remaining.retain(|&entry| {
            if entry.0 < base { finer.push_back(entry); false } else { true }
        });
        fine.truncate(VISIBLE_ADMISSION_COLUMNS.saturating_sub(finer.len()));
        finer.append(&mut fine);
        remaining.truncate(VISIBLE_ADMISSION_COLUMNS.saturating_sub(finer.len()));
        finer.append(&mut remaining);
        self.visible_admission = finer;
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
                    self.obsolete_owners.push_back(OwnerRetirement { owner, offset: 0, selected: false });
                }
            }
        }
    }

    fn prioritize_camera_owners(&mut self, blocks: &[u64], selected: &SelectedOwners) {
        // Clear withdrawn priority even when the deadline leaves a task
        // unvisited. A returned view may regenerate its earlier prefix.
        for task in &mut self.obsolete_owners {
            let (level, face, tier, i, j) = task.owner;
            task.selected &= selected.get(&block_slot(level, face, tier, i, j))
                .is_some_and(|wanted| *wanted != task.owner);
        }
        // A live captured stamp can refer to terrain hundreds of metres
        // behind the current view. Only conflicting selected slots supersede
        // it; its journals and records still retire through normal eviction.
        for (&block, lease) in &mut self.visible_leases {
            if !selected_owner_allows(block, selected) { lease.retiring = true; }
        }
        if !self.snapshot_mode { return; }
        let mut owners = Vec::new();
        for &key in blocks {
            let (face, level, i, j) = unpack(key);
            for tier in 1..=BLOCK_TIERS {
                let desired = (level, face, tier, i >> (2 * tier), j >> (2 * tier));
                let slot = block_slot(level, face, tier, desired.3, desired.4);
                if let Some(&owner) = self.block_owner.get(&slot) {
                    if owner != desired && !owners.contains(&owner) { owners.push(owner); }
                }
                if owners.len() == 64 { break; }
            }
            if owners.len() == 64 { break; }
        }
        // Preserve every existing walk cursor, but process current view
        // aliases before old admission requests in the same bounded FIFO.
        for owner in owners.into_iter().rev() {
            let mut task = self.obsolete_owners.iter().position(|task| task.owner == owner)
                .and_then(|at| self.obsolete_owners.remove(at))
                .unwrap_or(OwnerRetirement { owner, offset: 0, selected: false });
            // Previously visited columns may have been protected by captured
            // demand. Revisit that prefix once when priority changes, then
            // preserve progress throughout subsequent selected slices.
            if !task.selected { task.offset = 0; }
            task.selected = true;
            if self.obsolete_owners.len() == 64 { self.obsolete_owners.pop_back(); }
            self.obsolete_owners.push_front(task);
        }
    }

    #[cfg(test)]
    fn retire_obsolete_owner_step(&mut self, work: &mut FrameWork) -> bool {
        self.retire_obsolete_owner_step_current(work, &mut None, &SelectedOwners::default())
    }

    fn retire_obsolete_owner_step_current(&mut self, work: &mut FrameWork,
        last_demand: &mut Option<(u64, bool)>, selected: &SelectedOwners) -> bool {
        let Some(task) = self.obsolete_owners.front_mut() else { return false };
        let (level, face, tier, i, j) = task.owner;
        if self.block_owner.get(&block_slot(level, face, tier, i, j)) != Some(&task.owner) {
            self.obsolete_owners.pop_front();
            return true;
        }
        let (level, face, tier, bi, bj) = task.owner;
        let selected_conflict = selected.get(&block_slot(level, face, tier, bi, bj))
            .is_some_and(|wanted| *wanted != task.owner);
        task.selected &= selected_conflict;
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
        let wanted_now = self.current_wanted(key)
            && cached_current_geometry(&self.grid, self.current_request.as_ref(), key, last_demand, selected);
        if self.levels[level as usize].wanted.is_some() && !wanted_now
            && (selected_conflict || !self.transient_wanted(key)) {
            self.levels[level as usize].pending.remove(key);
            if self.residents.contains_key(key) { self.evict(key, work); }
        }
        true
    }

    /// Snapshot mode holds one full demand per level. Retire actual residents
    /// through a persistent table cursor, never millions of historical keys.
    fn apply_snapshot(&mut self, work: &mut FrameWork, out_of_time: &impl Fn() -> bool) {
        self.apply_snapshot_selected(work, out_of_time, &SelectedOwners::default());
    }

    fn apply_snapshot_selected(&mut self, work: &mut FrameWork, out_of_time: &impl Fn() -> bool,
        selected: &SelectedOwners) {
        let mut last_demand = None;
        while (self.diffs.iter().any(|diffs| !diffs.is_empty())
            || self.retire_finished_epoch < self.snapshot_epoch
            || !self.obsolete_owners.is_empty()) && !out_of_time() {
            for step in 0..128 {
                // Each eviction remains atomic; stop before the next key
                // once the window phase has consumed its time allowance.
                if out_of_time() { return; }
                // Protected alias owners can be requeued after every slice.
                // Reserve half the same bounded work for the normal cursor
                // until its epoch completes, then let owners use all of it.
                if (step < 64 || self.retire_finished_epoch >= self.snapshot_epoch)
                    && self.retire_obsolete_owner_step_current(work, &mut last_demand, selected) { continue; }
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
                        let (face, _, i, j) = unpack(key);
                        let owner = (level as u32, face, 1, i >> 2, j >> 2);
                        // A reselected resident can precede its new Camera lease.
                        let selected_current = selected.get(&block_slot(level as u32, face, 1, owner.3, owner.4)) == Some(&owner);
                        if !selected_current {
                            self.evict(key, work);
                            // Backshift may have moved another resident here.
                            continue;
                        }
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
                    let allowed = cached_current_geometry(&self.grid, self.current_request.as_ref(), key, &mut last_demand, selected);
                    if allowed && wanted.contains(&key) && !self.publishing.contains_key(&key)
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
                    if self.blocks.get(&(level as u32, face, 1, bi, bj)).is_some_and(|block| block.refs == 16)
                        && !Self::tile_has_initial_retry(&self.initial_retries, face, level as u32, bi, bj) {
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
        let (face, level, i, j) = unpack(first);
        if self.blocks.get(&(level, face, 1, i >> 2, j >> 2)).is_some_and(|block| block.refs > 0)
            || self.blocks_conflict(first) { return false; }
        // Visible keys may share an existing captured lease. Ordinary keys
        // must all belong to current wanted demand at the selected bucket.
        // Match scalar admission without creating or refreshing any demand.
        let transient = visible && self.transient_wanted(first);
        if !transient && !current_geometry_allows(&self.grid, self.current_request.as_ref(), first) {
            return false;
        }
        let mut column_buckets = [bucket; 16];
        for (member, &key) in keys.iter().enumerate() {
            if self.initial_retries.contains(&key) || self.publishing.contains_key(&key)
                || (!transient && !self.current_wanted(key)) || self.residents.contains_key(key) {
                return false;
            }
            if key != selected {
                let Some(&(pending_bucket, _)) = self.levels[index].pending.at.get(&key) else { return false };
                if !visible && pending_bucket as usize != bucket { return false; }
                column_buckets[member] = pending_bucket as usize;
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
        for (member, &key) in keys.iter().enumerate() {
            if key != selected {
                if visible { self.visible_admission.pop_front(); }
                self.levels[index].pending.remove(key);
            }
            let column_bucket = column_buckets[member];
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

    fn retire_pool_pressure(&mut self, work: &mut FrameWork, selected: &SelectedOwners,
        frame_deadline: Option<std::time::Instant>) {
        if !self.pool_pressure { return; }
        let mut deadline = std::time::Instant::now() + std::time::Duration::from_millis(1);
        if let Some(frame_deadline) = frame_deadline { deadline = deadline.min(frame_deadline); }
        let expired = || std::time::Instant::now() >= deadline;
        if expired() { return; }
        // Pool-full retries cannot make progress without releasing old data.
        // Advance the existing protected retirement cursor before preparation
        // can consume the whole frame budget; keep its epoch and quarantine.
        if self.snapshot_mode { self.apply_snapshot_selected(work, &expired, selected); }
        else { self.apply_queued(work, &expired); }
        self.clear_resolved_pool_pressure();
    }

    fn clear_resolved_pool_pressure(&mut self) {
        if self.initial_retries.is_empty() && self.urgent.is_empty()
            && (!self.snapshot_mode || self.retire_finished_epoch >= self.snapshot_epoch) {
            self.pool_pressure = false;
        }
    }

    /// Bootstrap global coverage without first scanning obsolete fine residents.
    /// Latest wanted membership still rejects old queued additions.
    fn queue_global_adds(&mut self, out_of_time: &impl Fn() -> bool) {
        let top = self.levels.len() - 1;
        let wanted = self.levels[top].wanted.clone();
        for queued in &mut self.diffs[top] {
            let start = queued.added;
            while queued.added < queued.diff.adds.len() && !out_of_time() {
                let (priority, key) = queued.diff.adds[queued.added];
                if wanted.as_ref().is_some_and(|wanted| wanted.contains(&key))
                    && !self.publishing.contains_key(&key)
                    && (!self.residents.contains_key(key) || self.initial_retries.contains(&key)) {
                    self.levels[top].pending.insert(key, PendingQueue::bucket(priority));
                }
                queued.added += 1;
            }
            self.queued_delta_ops -= queued.added - start;
            if out_of_time() { break; }
        }
    }

    /// Plan one frame. `lod0` is the level-0 distance, `budget` the maximum
    /// number of column jobs.
    pub fn plan(&mut self, planet: &std::sync::Arc<Planet>, eye: DVec3, lod0: f64, budget: usize) -> FrameWork {
        self.frame = self.frame.wrapping_add(1);
        let started = std::time::Instant::now();
        let trace_ms: Option<f64> = std::env::var("HELIO_VOXEL_PLAN_TRACE").ok().map(|v| v.parse().unwrap_or(10.0));
        let trace_admission = trace_ms.is_some() && self.frame % 30 == 0;
        let budget_time = self.cpu_budget;
        let out_of_time = move || budget_time.is_some_and(|b| started.elapsed() >= b);
        // Discovery and ownership preparation share the intake deadline.
        // Repeating this prefix must leave time for already queued columns.
        let refresh_deadline = budget_time.map(|budget|
            started + budget - (budget / 4).min(std::time::Duration::from_millis(1)));
        let preparation_out_of_time = move || refresh_deadline
            .is_some_and(|deadline| std::time::Instant::now() >= deadline);
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
        self.queue_global_adds(&preparation_out_of_time);
        // Current view work owns admission before obsolete window retirement.
        // The existing total deadline bounds every phase of this plan.
        let mut camera_blocks = self.camera_blocks(eye, &preparation_out_of_time);
        // Keep each tile's sixteen columns together, advancing all view bands
        // before a longer finest-band prefix can consume the deadline.
        let mut band_rank = [0u32; 32];
        camera_blocks.sort_by_cached_key(|&key| {
            let level = unpack(key).1 as usize;
            let rank = band_rank[level];
            band_rank[level] += 1;
            (rank, level)
        });
        let selected_owners = select_camera_owners(&mut camera_blocks);
        self.handoff_camera_owners(&selected_owners, &mut work, &preparation_out_of_time);
        self.promote_camera_residents(&camera_blocks, &selected_owners, &mut work, &preparation_out_of_time);
        self.retire_visible_leases_current(&camera_blocks, &mut work, &preparation_out_of_time);
        self.retire_pool_pressure(&mut work, &selected_owners, refresh_deadline);
        let t_windows = started.elapsed();
        // Preparation must leave time to submit the columns it selects.
        // Captured feedback keeps its clock/FIFO validation before Camera
        // intake, but cannot consume the entire preparation allowance.
        let feedback_deadline = refresh_deadline.map(|deadline| {
            let now = std::time::Instant::now();
            now + deadline.saturating_duration_since(now) / 2
        });
        // Validate Captured clocks before building the new Camera FIFO; stale
        // rank cleanup cannot erase current demand. Geometry still gates leases.
        if !out_of_time() { self.refresh_visible_pending(feedback_deadline); }
        self.refresh_camera_pending(&camera_blocks, refresh_deadline);
        let t_visible = started.elapsed();
        // Ordinary near demand cannot delay already queued visible work.
        if self.visible_admission.is_empty() && !out_of_time() {
            self.refresh_near_pending(eye, refresh_deadline);
        }
        let t_apply = started.elapsed();
        let t_near = t_windows + (t_apply - t_visible);
        // Urgent edit regenerations first.
        let urgent_count = self.urgent.len();
        let urgent_started = trace_admission.then(std::time::Instant::now);
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
        let urgent_time = urgent_started.map_or(std::time::Duration::ZERO, |at| at.elapsed());
        // Merge pending windows by normalized distance; the coarsest level
        // (global coverage) always goes first.
        let top_level = self.grid.levels() - 1;
        // Admission costs ~2 us of CPU per column (edit query, summary
        // blocks, table): bounded by time as well as by the GPU budget, and
        // resumes next frame.
        let mut steps = 0u32;
        let mut awaiting_publication = Vec::new();
        let mut last_summary_check = None;
        let mut last_demand = None;
        let mut pending_selection = None;
        let admission_deadline = budget_time.map(|budget| started + budget);
        let mut admission_attempts = 0;
        let mut alias_deferred = 0;
        let mut publication_deferred = 0;
        let mut batched_columns = 0;
        let mut batch_attempts = 0;
        let mut last_batch_block = None;
        let mut pop_time = std::time::Duration::ZERO;
        let mut batch_time = std::time::Duration::ZERO;
        let mut stop_reason = "queue_empty";
        let mut stop_key = None;
        while work.jobs.len() < budget {
            steps += 1;
            if (steps == 1 || steps % 64 == 0) && out_of_time() {
                stop_reason = "frame_deadline";
                break;
            }
            let pop_started = trace_admission.then(std::time::Instant::now);
            let next = pop_pending(&mut self.levels, top_level,
                &mut self.visible_admission, &mut pending_selection, admission_deadline);
            pop_time += pop_started.map_or(std::time::Duration::ZERO, |at| at.elapsed());
            let Some((index, key, bucket)) = next else {
                if out_of_time() { stop_reason = "pop_deadline"; }
                break;
            };
            stop_key = Some(key);
            admission_attempts += 1;
            let block_key = key & !(3u64 | (3u64 << 32));
            let visible_batch = key == block_key
                && self.visible_admission.front().is_some_and(|&(level, next)|
                    level == index && next & !(3u64 | (3u64 << 32)) == key);
            let preferred = selected_owner_allows(block_key, &selected_owners);
            // Retried/in-flight heads cannot be sixteen new records. Reuse
            // their exact scalar path before an impossible batch preflight.
            if preferred && self.timed_out_batch != Some(block_key)
                && !self.initial_retries.contains(&key) && !self.publishing.contains_key(&key)
                && index != top_level as usize && (visible_batch && batch_attempts < TEMPORARY_LEASES
                || self.visible_admission.is_empty() && last_batch_block != Some(block_key)) {
                if visible_batch { batch_attempts += 1; }
                last_batch_block = Some(block_key);
                let batch_started = trace_admission.then(std::time::Instant::now);
                let admitted = if visible_batch {
                    self.admit_visible_block(planet, index, key, bucket, budget, admission_deadline, &mut work)
                } else {
                    self.admit_pending_block(planet, index, key, bucket, budget, admission_deadline, false, &mut work)
                };
                batch_time += batch_started.map_or(std::time::Duration::ZERO, |at| at.elapsed());
                if admitted {
                    admission_attempts += 15;
                    batched_columns += 16;
                    last_summary_check = None;
                    if out_of_time() { stop_reason = "batch_commit_deadline"; break; }
                    continue;
                }
                if out_of_time() {
                    self.levels[index].pending.insert(key, bucket);
                    if visible_batch { self.visible_admission.push_front((index, key)); }
                    self.timed_out_batch = Some(block_key);
                    stop_reason = "batch_preflight_deadline";
                    break;
                }
            }
            let transient = self.transient_wanted(key);
            if !preferred {
                if self.timed_out_batch == Some(block_key) { self.timed_out_batch = None; }
                continue;
            }
            let allowed = cached_current_geometry(&self.grid, self.current_request.as_ref(), key, &mut last_demand, &selected_owners);
            if !allowed && !transient { continue; }
            let leased = self.visible_leases.contains_key(&(key & !(3u64 | (3u64 << 32))));
            if (self.levels[index].wanted.as_ref().is_some_and(|wanted| !wanted.contains(&key))
                || leased && !self.current_wanted(key)) && !transient {
                continue;
            }
            let retry = self.initial_retries.contains(&key);
            if !retry && self.residents.contains_key(key) {
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
                stop_reason = "record_capacity";
                break;
            };
            let Ok(block) = self.edit_list(planet, key, &mut work) else {
                if !retry { self.free_records.push(record); }
                requeue(self);
                stop_reason = "edit_capacity";
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
            if self.timed_out_batch == Some(block_key) && self.residents.get(key).is_some_and(|resident| resident.blocks) {
                // The exact partial-tile guard now makes batching impossible;
                // the next frame can cheaply finish its remaining columns.
                self.timed_out_batch = None;
            }
        }
        for (index, key, bucket) in awaiting_publication {
            self.queue_obsolete_owners(key);
            self.levels[index].pending.insert(key, bucket);
        }
        if work.jobs.len() >= budget { stop_reason = "job_budget"; }
        self.dirty_detached_publications(&mut work);
        let t_admission = started.elapsed();
        // Capacity-blocked admission also reaches this cleanup. Evicted record
        // identities retain their existing next-frame quarantine.
        if !out_of_time() {
            if self.snapshot_mode { self.apply_snapshot_selected(&mut work, &out_of_time, &selected_owners); }
            else { self.apply_queued(&mut work, &out_of_time); }
        }
        let background = started.elapsed() - t_admission;
        if trace_admission && trace_ms.is_some_and(|ms| started.elapsed().as_secs_f64() * 1e3 > ms) {
            let key = stop_key.unwrap_or(0);
            let mut attached = [0usize; 3];
            let mut missing = [0usize; 3];
            for (&slot, &owner) in &selected_owners {
                let tier = owner.2 as usize - 1;
                if self.block_owner.get(&slot) == Some(&owner) { attached[tier] += 1; }
                else { missing[tier] += 1; }
            }
            let detached_full = camera_blocks.iter().filter(|&&key| {
                let (face, level, i, j) = unpack(key);
                let owner = (level, face, 1, i >> 2, j >> 2);
                self.blocks.get(&owner).is_some_and(|block| block.refs == 16)
                    && self.block_owner.get(&block_slot(level, face, 1, i >> 2, j >> 2)) != Some(&owner)
            }).count();
            let (entry_cell, _) = self.grid.locate(eye);
            let entry_base = self.camera_base_level() as u32;
            let mut entry_refs = [0u32; 3];
            let mut entry_attached = [false; 3];
            let mut entry_selected = [false; 3];
            for band in 0..3 {
                let level = entry_base + band as u32;
                let (i, j) = ((entry_cell.i >> (3 + level)) & !3, (entry_cell.j >> (3 + level)) & !3);
                let block = pack(key0(entry_cell.face, level, i), j as u32);
                let owner = (level, entry_cell.face, 1, i >> 2, j >> 2);
                entry_refs[band] = self.blocks.get(&owner).map_or(0, |state| state.refs);
                entry_attached[band] = self.block_owner.get(&block_slot(level, entry_cell.face, 1, i >> 2, j >> 2)) == Some(&owner);
                entry_selected[band] = camera_blocks.contains(&block);
            }
            eprintln!("PLAN_ADMISSION_TRACE urgent_count {urgent_count} urgent_remaining {} urgent_ms {:.3} pop_ms {:.3} batch_ms {:.3} fifo_remaining {} stop {stop_reason} level {} key {key:#018x} initial_retry {} resident {} publishing {} free_records {} next_record {} record_capacity {} selected_attached {attached:?} selected_missing {missing:?} detached_full_camera {detached_full} entry_selected {entry_selected:?} entry_refs {entry_refs:?} entry_attached {entry_attached:?}",
                self.urgent.len(), urgent_time.as_secs_f64() * 1e3, pop_time.as_secs_f64() * 1e3,
                batch_time.as_secs_f64() * 1e3, self.visible_admission.len(), unpack(key).1,
                self.initial_retries.contains(&key), self.residents.contains_key(key), self.publishing.contains_key(&key),
                self.free_records.len(), self.next_record, self.capacity.records);
        }
        if trace_ms.is_some_and(|ms| started.elapsed().as_secs_f64() * 1e3 > ms) {
            eprintln!(
                "PLAN_TRACE edits {:.2} drain {:.2} apply {:.2} admit {:.2} ms jobs {} evictions {} queued_diffs {} steps {steps} queued_bytes {} queued_ops {} wanted_capacity {} admission_attempts {admission_attempts} alias_deferred {alias_deferred} publication_deferred {publication_deferred} batched_columns {batched_columns}",
                t_edits.as_secs_f64() * 1e3,
                (t_drain - t_edits).as_secs_f64() * 1e3,
                (t_apply - t_drain + background).as_secs_f64() * 1e3,
                (t_admission - t_apply).as_secs_f64() * 1e3,
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
        stats.camera_base_level = self.camera_base_level() as u32;
        stats.camera_candidate_blocks = [0; 3];
        stats.camera_lease_blocks = [0; 3];
        stats.camera_jobs = [0; 3];
        for &block in &camera_blocks {
            if let Some(index) = unpack(block).1.checked_sub(stats.camera_base_level).filter(|&i| i < 3) {
                stats.camera_candidate_blocks[index as usize] += 1;
            }
        }
        for (&block, lease) in &self.visible_leases {
            if !lease.retiring && lease.current_demand_frame == Some(self.frame)
                && matches!(lease.origin, LeaseOrigin::Camera) {
                if let Some(index) = unpack(block).1.checked_sub(stats.camera_base_level).filter(|&i| i < 3) {
                    stats.camera_lease_blocks[index as usize] += 1;
                }
            }
        }
        let selected_camera: FxHashSet<_> = camera_blocks.iter().copied().collect();
        for &key in &work.job_keys {
            if selected_camera.contains(&(key & !(3u64 | (3u64 << 32)))) {
                if let Some(index) = unpack(key).1.checked_sub(stats.camera_base_level).filter(|&i| i < 3) {
                    stats.camera_jobs[index as usize] += 1;
                }
            }
        }
        stats.plan_edits_ms = t_edits.as_secs_f64() * 1e3;
        stats.plan_authority_ms = (t_drain - t_edits).as_secs_f64() * 1e3;
        stats.plan_windows_ms = (t_windows - t_drain + background).as_secs_f64() * 1e3;
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
        let mut completed = false;
        let mut pool_full = false;
        for (key, status) in results {
            let Some(publication) = self.publishing.remove(&key) else { continue };
            if publication.evicted {
                if let Some(block) = publication.previous { self.edits.release(block); }
                if let Some(block) = publication.next { self.edits.release(block); }
                continue;
            }
            let resident = self.residents.get_mut(key).expect("live publication has a resident");
            assert_eq!(resident.record, publication.record);
            completed = true;
            pool_full |= status == 3;
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
        if completed {
            if pool_full && !self.pool_pressure && self.snapshot_mode
                && self.retire_finished_epoch >= self.snapshot_epoch {
                self.snapshot_epoch += 1;
                if self.retire_slot == 0 { self.retire_started_epoch = self.snapshot_epoch; }
            }
            self.pool_pressure |= pool_full;
        }
        self.requeue(failed);
        // An unrelated success cannot abandon older pool-backed retries or
        // their reclaim pass. Replacement failures enter urgent via requeue.
        if completed && !pool_full { self.clear_resolved_pool_pressure(); }
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
        let (planet, mut r, old, _) = edit_fixture();
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
        let eye = set_fixture_request(&mut r, latest[0], 120.0);
        let work = r.plan(&planet, eye, 120.0, 16);
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
        r.obsolete_owners.push_back(OwnerRetirement { owner: (0, face, 3, 15, 15), offset: 0, selected: false });
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
        // These authored current blocks are32–64m away, so their fine
        // window must encompass them rather than use the edit fixture's1m.
        r.last_request.as_mut().unwrap().lod0 = 120.0;
        (planet, r, eye, near, far)
    }

    fn batch_admission_fixture() -> (std::sync::Arc<Planet>, Residency, Vec<u64>, Vec<u64>) {
        let (planet, mut r, _, near, far) = visible_order_fixture();
        r.levels[0].pending.remove(near[0]); // pop_pending has selected the first key.
        r.visible_admission.extend(near[1..].iter().chain(&far).map(|&key| (0, key)));
        (planet, r, near, far)
    }

    #[test]
    fn timed_out_batch_resumes_scalar_and_finishes_exact_partial_tile() {
        let (planet, mut r, eye, near, far) = visible_order_fixture();
        r.prioritize_visible_blocks([near[0], far[0]].map(|key| (key as u32, (key >> 32) as u32)));
        // State retained when the previous frame restored an expired preflight.
        r.timed_out_batch = Some(near[0]);
        let first = r.plan(&planet, eye, 120.0, 16);
        assert_eq!(first.job_keys, near);
        assert_eq!(r.stats.admission_batched_columns, 0, "the stalled tile must advance through scalar admission");
        assert_eq!(r.timed_out_batch, None);
        assert_eq!(first.jobs.iter().map(|job| job.record).collect::<FxHashSet<_>>().len(), 16);
        assert!(far.iter().all(|&key| !r.residents.contains_key(key)));
        r.complete_jobs(first.job_keys.iter().map(|&key| (key, 0)));
        let next = r.plan(&planet, eye, 120.0, 16);
        assert_eq!(next.job_keys, far);
        assert_eq!(r.stats.admission_batched_columns, 16, "other intact tiles retain the fast path");
        table_is_exact(&r);
    }

    #[test]
    fn complete_ordinary_batch_uses_selected_bucket_without_visible_feedback() {
        let (planet, mut r, eye, near, far) = visible_order_fixture();
        let work = r.plan(&planet, eye, 120.0, 16);
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
            r.last_request.as_mut().unwrap().lod0 = 120.0;
            let (face, _, i, j) = unpack(old);
            let mut wanted = FxHashSet::default();
            for block in 0..100 {
                let first = pack(key0(face, 0, (i & !3) + 40 + (block % 10) * 4), ((j & !3) + (block / 10) * 4) as u32);
                let (keys, count) = r.visible_columns(first).unwrap();
                assert_eq!(count, 16);
                for key in keys { wanted.insert(key); r.levels[0].pending.insert(key, 0); }
            }
            r.levels[0].active = true;
            r.levels[0].wanted = Some(std::sync::Arc::new(wanted.clone()));
            let work = r.plan(&planet, eye, 120.0, budget);
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
    fn complete_visible_batch_preserves_preflight_buckets_on_failed_publication() {
        let (planet, mut r, keys, _) = batch_admission_fixture();
        for (member, &key) in keys.iter().enumerate().skip(1) {
            r.levels[0].pending.insert(key, member % BUCKETS);
        }
        let mut work = FrameWork::default();
        assert!(r.admit_visible_block(&planet, 0, keys[0], 0, 16, None, &mut work));
        assert_eq!(work.job_keys, keys);
        for (member, &key) in keys.iter().enumerate() {
            assert_eq!(r.publishing[&key].initial_bucket, Some(member % BUCKETS));
        }
        r.complete_jobs(keys.iter().map(|&key| (key, 2)));
        for (member, &key) in keys.iter().enumerate() {
            assert!(r.initial_retries.contains(&key));
            assert_eq!(r.levels[0].pending.at[&key].0 as usize, member % BUCKETS,
                "failed individual publication returns to its original priority bucket");
        }
        assert!(keys.iter().all(|&key| r.residents.get(key).is_some_and(|resident| resident.blocks)));
        table_is_exact(&r);
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
    fn complete_visible_batch_accumulates_existing_parent_refs_and_recounts_dirty_ancestors() {
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
        assert_eq!(work.block_inits.iter().map(|init| init.0).collect::<FxHashSet<_>>().len(), 4,
            "second tier-1 block shares both existing parent allocations");
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
        let work = r.plan(&planet, eye, 120.0, 16);
        assert_eq!(work.job_keys, near);
        assert_eq!(r.stats.admission_batched_columns, 16);
        assert_eq!(r.stats.admission_attempts, 16);
        assert_eq!(r.stats.admission_alias_deferred, 0);
        assert_eq!(r.stats.admission_publication_deferred, 0);
        assert!(far.iter().all(|&key| !r.residents.contains_key(key)));
        r.complete_jobs(work.job_keys.iter().map(|&key| (key, 0)));
        r.set_cpu_budget(Some(std::time::Duration::ZERO));
        assert!(r.plan(&planet, eye, 120.0, 16).jobs.is_empty());
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
            let work = r.plan(&planet, eye, 120.0, 8);
            assert_eq!(work.jobs.len(), 8);
            assert!(work.job_keys.iter().all(|key| near.contains(key)), "nearest visible geometry must consume the partial budget first");
            admitted.extend(work.job_keys.iter().copied());
            r.complete_jobs(work.job_keys.iter().map(|key| (*key, 0)));
        }
        assert_eq!(admitted, near.iter().copied().collect());
        assert!(far.iter().all(|key| !r.residents.contains_key(*key)));
        let work = r.plan(&planet, eye, 120.0, 16);
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
        let work = r.plan(&planet, eye, 120.0, 16);
        assert_eq!(work.job_keys.iter().copied().collect::<FxHashSet<_>>(), far.iter().copied().collect());
        assert!(near.iter().all(|key| !r.residents.contains_key(*key)));
    }

    #[test]
    fn elapsed_plan_deadline_preserves_diffs_and_pending_without_admitting_work() {
        let (planet, mut r, _, eye) = edit_fixture();
        let (cell, _) = r.grid.locate(eye);
        let key = pack(key0(cell.face, 0, (cell.i >> 3) + 1), (cell.j >> 3) as u32);
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
        r.refresh_visible_pending_until(|| { checks += 1; checks > 19 });
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
    fn visible_refresh_expired_entry_and_partial_tile_preserve_requests_and_ranks() {
        let (_, mut r, _, at) = visible_bridge_fixture();
        let blocks = [1000, 1004].map(|i| pack(key0(crate::grid::PLANE_FACE, 0, i), 1000));
        r.prioritize_visible_blocks_from(blocks.map(|key| (key as u32, (key >> 32) as u32)), 10, 7, at);
        r.visible_admission.push_back((0, 123));
        let before = (r.visible_blocks.clone(), r.visible_admission.clone(),
            r.visible_source.map(|source| (source.frame, source.view, source.at)));
        r.refresh_visible_pending_cohort_until(8, || true);
        assert_eq!(before, (r.visible_blocks.clone(), r.visible_admission.clone(),
            r.visible_source.map(|source| (source.frame, source.view, source.at))));
        assert!(r.visible_leases.is_empty() && r.levels[0].pending.is_empty());
        let mut checks = 0;
        r.refresh_visible_pending_cohort_until(8, || { checks += 1; checks > 7 });
        assert_eq!(r.visible_blocks, blocks);
        assert_eq!(r.visible_admission, before.1, "an incomplete tile never appends partial distance ranks");
        assert!(r.levels[0].pending.is_empty() && r.obsolete_owners.is_empty());
        let stamp = r.visible_source.unwrap();
        assert_eq!((stamp.frame, stamp.view, stamp.at), (10, 7, at));
        r.visible_admission.clear();
        r.refresh_visible_pending_cohort_until(8, || false);
        assert!(r.visible_blocks.is_empty());
        assert_eq!(r.visible_admission.len(), 32);
        for (index, block) in blocks.into_iter().enumerate() {
            let (keys, count) = r.visible_columns(block).unwrap();
            assert_eq!(count, 16);
            assert_eq!(r.visible_admission.iter().skip(index * 16).take(16).copied().collect::<Vec<_>>(),
                keys.map(|key| (0, key)));
            let source = r.visible_leases[&block].captured_source();
            assert_eq!((source.frame, source.view, source.at), (10, 7, at));
        }
    }

    #[test]
    fn visible_refresh_bounded_cohort_keeps_complete_nearest_bucket_order_without_ranks() {
        let (_, mut r, _, at) = visible_bridge_fixture();
        lease_block(&mut r, 1000, at);
        lease_block(&mut r, 1004, at);
        let order: Vec<_> = r.visible_leases.keys().copied().collect();
        // Eight-frame distance priority expires while the original 32-frame
        // ownership lease remains live. Lease reinsertion must use buckets.
        r.set_visible_view(19, 7, at + std::time::Duration::from_millis(1));
        r.refresh_visible_pending_cohort_until(8, || false);
        assert!(r.visible_admission.is_empty());
        for block in order {
            let (keys, count) = r.visible_columns(block).unwrap();
            assert_eq!(count, 16);
            for key in keys {
                assert_eq!(r.levels[0].pending.pop(), Some((key, 0)),
                    "whole-cohort reverse insertion keeps the first selected tile above later tiles");
            }
        }
        assert!(r.levels[0].pending.is_empty());
    }

    #[test]
    fn visible_refresh_bounded_cohort_retains_original_requested_tail() {
        let (_, mut r, _, at) = visible_bridge_fixture();
        let blocks: Vec<_> = (0..10).map(|n| pack(key0(crate::grid::PLANE_FACE, 0, 1000 + n * 4), 1000)).collect();
        r.prioritize_visible_blocks_from(blocks.iter().map(|&key| (key as u32, (key >> 32) as u32)), 10, 7, at);
        r.refresh_visible_pending_cohort_until(8, || false);
        assert_eq!(r.visible_admission.len(), 128);
        assert_eq!(r.visible_blocks, blocks[8..]);
        let source = r.visible_source.unwrap();
        assert_eq!((source.frame, source.view, source.at), (10, 7, at));
        r.refresh_visible_pending_cohort_until(8, || false);
        assert!(r.visible_blocks.is_empty() && r.visible_source.is_none());
        assert_eq!(r.visible_admission.len(), 160, "requested tail precedes reconstructed lease demand");
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
        let (face, level, i, j) = unpack(incoming[0]);
        let eye = residency.grid.ground_point(face, f64::from((i + 2) * (BRICK << level)),
            f64::from((j + 2) * (BRICK << level))) + DVec3::Y * 30.0;
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
        let (planet, mut residency, old, _) = edit_fixture();
        let mut work = FrameWork::default();
        assert!(residency.acquire_blocks(old, &mut work));
        residency.residents.get_mut(old).unwrap().blocks = true;
        let (face, level, i, j) = unpack(old);
        let incoming = pack(key0(face, level, i + 512), j as u32);
        let available = pack(key0(face, level, i + 480), j as u32);
        let eye = set_fixture_request(&mut residency, incoming, 120.0);
        assert!(residency.blocks_conflict(incoming));
        assert!(!residency.blocks_conflict(available));
        residency.catching_up[level as usize] = 1;
        residency.levels[level as usize].pending.insert(incoming, 0);
        residency.levels[level as usize].pending.insert(available, 1);
        let work = residency.plan(&planet, eye, 120.0, 2);
        assert_eq!(work.job_keys, vec![available]);
        assert!(!residency.residents.contains_key(incoming));
        assert!(residency.levels[level as usize].pending.at.contains_key(&incoming));
        let mut retirement = FrameWork::default();
        residency.evict(old, &mut retirement);
        residency.catching_up[level as usize] = 0;
        let work = residency.plan(&planet, eye, 120.0, 1);
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
        let (face, _, _, _) = unpack(old);
        // Two toroidal aliases cannot simultaneously lie in a fine current
        // window. Exercise their shared admission/reference guards atL3.
        let level = 3;
        let (i, j) = (0, 0);
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
        let (cell, _) = r.grid.locate(eye);
        let coarse=pack(key0(cell.face,1,cell.i >> 4),(cell.j >> 4) as u32);
        r.levels[1].active=true;
        r.levels[1].pending.insert(coarse,0);
        let work=r.plan(&planet,eye,120.0,7);
        assert_eq!(work.jobs.len(),7,"promotion must not round a GPU allowance up to a complete block");
        assert_eq!(work.job_keys[0],coarse,"a closer coarser pending key must retain its priority");
        assert!(work.job_keys[1..].iter().all(|key|focus.contains(key)));
        assert_eq!(focus.iter().filter(|key|r.levels[0].pending.at.contains_key(key)).count(),10);
    }

    fn set_fixture_request(r: &mut Residency, key: u64, lod0: f64) -> DVec3 {
        let (face, level, i, j) = unpack(key);
        let eye = r.grid.ground_point(face, f64::from((i + 2) * (BRICK << level)),
            f64::from((j + 2) * (BRICK << level))) + DVec3::Y * 10.0;
        let request = r.last_request.as_mut().unwrap();
        request.eye = eye;
        request.lod0 = lod0;
        eye
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

    fn current_alias_fixture() -> (std::sync::Arc<Planet>, Residency, Vec<u64>, Vec<u64>) {
        let (planet, mut r, _, _) = current_bridge_fixture();
        let eye = r.current_request.as_ref().unwrap().eye;
        r.set_ground_clearance(10.0);
        let incoming = r.camera_blocks(eye, &|| false)[0];
        let (face, level, i, j) = unpack(incoming);
        let outgoing = pack(key0(face, level, i - 512), j as u32);
        let columns = |r: &Residency, block| {
            let (keys, count) = r.visible_columns(block).unwrap();
            keys[..count].to_vec()
        };
        let old = columns(&r, outgoing);
        let new = columns(&r, incoming);
        r.levels[0].wanted = Some(std::sync::Arc::new(old.iter().chain(&new).copied().collect()));
        for &key in &old {
            let record = r.alloc_record().unwrap();
            assert!(r.acquire_blocks(key, &mut FrameWork::default()));
            r.residents.insert(key, Resident { record, blocks: true, ..Default::default() });
        }
        assert!(r.current_wanted(old[0]), "accepted worker demand intentionally still wants the old owner");
        assert!(!current_geometry_allows(&r.grid, r.current_request.as_ref(), old[0]));
        assert!(current_geometry_allows(&r.grid, r.current_request.as_ref(), new[0]));
        assert!(r.blocks_conflict(new[0]));
        assert!(r.diffs.iter().all(|diff| diff.is_empty()));
        assert_eq!(r.retire_finished_epoch, r.snapshot_epoch);
        (planet, r, old, new)
    }

    #[test]
    fn summary_promotion_selected_ordinary_conflict_records_need_no_regeneration() {
        let (planet, mut r, old, current) = current_alias_fixture();
        let eye = r.current_request.as_ref().unwrap().eye;
        let view = r.camera_view.take();
        let clearance = r.ground_clearance.take();
        for &key in &current { r.levels[0].pending.insert(key, 0); }
        let generated = r.plan(&planet, eye, 160.0, 16);
        assert_eq!(generated.jobs.len(), 16);
        assert!(generated.job_keys.iter().all(|key| current.contains(key)));
        assert!(current.iter().all(|&key| !r.residents.get(key).unwrap().blocks));
        assert_eq!(r.block_conflicts, 16);
        r.complete_jobs(generated.job_keys.iter().map(|&key| (key, 0)));
        let records: Vec<_> = old.iter().chain(&current).map(|&key| r.residents.get(key).unwrap().record).collect();
        let previous = r.edits.alloc(2, r.capacity.edit_words).unwrap();
        let next = r.edits.alloc(2, r.capacity.edit_words).unwrap();
        r.residents.get_mut(current[0]).unwrap().edit_block = Some(previous);
        r.publishing.insert(current[0], EditPublication { record: r.residents.get(current[0]).unwrap().record,
            previous: Some(previous), next: Some(next), evicted: false, initial_bucket: None });
        r.camera_view = view;
        r.ground_clearance = clearance;
        let attached = r.plan(&planet, eye, 160.0, 0);
        assert!(attached.jobs.is_empty() && attached.evictions.is_empty());
        assert!(current.iter().all(|&key| r.residents.get(key).unwrap().blocks));
        assert_eq!(r.block_conflicts, 0);
        let (face, level, i, j) = unpack(current[0]);
        for tier in 1..=3 {
            let owner = (level, face, tier, i >> (2 * tier), j >> (2 * tier));
            assert_eq!(r.blocks[&owner].refs, 16);
            assert_eq!(r.block_owner[&r.blocks[&owner].slot], owner);
            assert!(attached.block_inits.contains(&(r.blocks[&owner].slot, owner.3, owner.4)));
        }
        assert_eq!(old.iter().chain(&current).map(|&key| r.residents.get(key).unwrap().record).collect::<Vec<_>>(), records);
        assert_eq!(r.residents.get(current[0]).unwrap().edit_block, Some(previous));
        assert_eq!(r.publishing[&current[0]].next, Some(next));
        assert!(!r.edits.free[previous.1 as usize].contains(&previous.0));
        assert!(!r.edits.free[next.1 as usize].contains(&next.0));
        let repeated = r.plan(&planet, eye, 160.0, 0);
        assert!(repeated.block_inits.is_empty(), "already referenced tiles must not recount every unchanged frame");
        table_is_exact(&r);
    }

    #[test]
    fn summary_promotion_complete_tile_guards_preserve_partial_deadline_and_alias_ownership() {
        for case in 0..4 {
            let (_, mut r, _, current) = current_alias_fixture();
            let mut blocks = vec![current[0]];
            let selected = select_camera_owners(&mut blocks);
            if case != 0 { r.handoff_camera_owners(&selected, &mut FrameWork::default(), &|| false); }
            for (index, &key) in current.iter().enumerate() {
                if case == 1 && index == 15 { continue; }
                let record = r.alloc_record().unwrap();
                r.residents.insert(key, Resident { record, blocks: false, ..Default::default() });
                r.block_conflicts += 1;
            }
            if case == 3 {
                r.publishing.insert(current[0], EditPublication { record: r.residents.get(current[0]).unwrap().record + 1,
                    previous: None, next: None, evicted: true, initial_bucket: None });
            }
            let owners = r.block_owner.clone();
            let conflicts = r.block_conflicts;
            let checks = std::cell::Cell::new(0usize);
            let mut work = FrameWork::default();
            r.promote_camera_residents(&blocks, &selected, &mut work, &|| {
                checks.set(checks.get() + 1);
                case == 2 && checks.get() > 8
            });
            assert_eq!(r.block_owner, owners, "case {case} cannot steal or partially attach a tile");
            assert_eq!(r.block_conflicts, conflicts);
            assert!(current.iter().filter_map(|&key| r.residents.get(key)).all(|resident| !resident.blocks));
            assert!(work.block_inits.is_empty() && work.jobs.is_empty() && work.evictions.is_empty());
        }
    }

    #[test]
    fn summary_handoff_unchanged_frame_does_not_reset_attached_summaries() {
        let (_, mut r, old, _) = current_alias_fixture();
        let mut current = vec![old[0]];
        let selected = select_camera_owners(&mut current);
        let owners = r.block_owner.clone();
        let mut work = FrameWork::default();
        r.handoff_camera_owners(&selected, &mut work, &|| false);
        assert!(work.block_inits.is_empty() && work.jobs.is_empty() && work.evictions.is_empty());
        assert_eq!(r.block_owner, owners);
        assert!(r.blocks.values().all(|block| block.refs == 16));
    }

    #[test]
    fn summary_handoff_dirty_child_recounts_unselected_parents_and_retained_siblings() {
        let (_, mut r, old, _) = current_alias_fixture();
        let (face, level, i, j) = unpack(old[0]);
        let sibling = pack(key0(face, level, i ^ 4), j as u32);
        let (siblings, count) = r.visible_columns(sibling).unwrap();
        for &key in &siblings[..count] {
            let record = r.alloc_record().unwrap();
            assert!(r.acquire_blocks(key, &mut FrameWork::default()));
            r.residents.insert(key, Resident { record, blocks: true, ..Default::default() });
        }
        let child = (level, face, 1, i >> 2, j >> 2);
        let parent = (level, face, 2, i >> 4, j >> 4);
        let grandparent = (level, face, 3, i >> 6, j >> 6);
        let records: Vec<_> = old.iter().chain(&siblings[..count])
            .map(|&key| r.residents.get(key).unwrap().record).collect();
        let mut work = FrameWork::default();
        r.detach_summary_owner(child, &mut work);
        for owner in [parent, grandparent] {
            assert!(work.block_inits.contains(&(r.blocks[&owner].slot, owner.3, owner.4)),
                "unselected attached ancestors must drop the detached child's count");
        }
        work = FrameWork::default();
        work.job_keys.extend([old[0], old[1], siblings[0]]);
        r.dirty_detached_publications(&mut work);
        assert_eq!(work.block_inits.len(), 2,
            "detached urgent/scalar publication dirties only its attached parents once per tile");
        for owner in [parent, grandparent] {
            assert!(work.block_inits.contains(&(r.blocks[&owner].slot, owner.3, owner.4)));
        }
        work = FrameWork::default();
        r.attach_summary_owner(child, &mut work);
        for owner in [child, parent, grandparent] {
            assert!(work.block_inits.contains(&(r.blocks[&owner].slot, owner.3, owner.4)),
                "ordered rebuild must include returning child and all existing parents");
        }
        assert_eq!(r.blocks[&child].refs, 16);
        assert_eq!(r.blocks[&(level, face, 1, (i ^ 4) >> 2, j >> 2)].refs, 16);
        assert_eq!(r.blocks[&parent].refs, 32);
        assert_eq!(r.blocks[&grandparent].refs, 32);
        assert_eq!(old.iter().chain(&siblings[..count])
            .map(|&key| r.residents.get(key).unwrap().record).collect::<Vec<_>>(), records);
        assert!(work.jobs.is_empty() && work.evictions.is_empty());
        // A later partial eviction must dirty the still-attached ancestors too.
        work = FrameWork::default();
        r.evict(old[0], &mut work);
        assert_eq!(r.blocks[&child].refs, 15);
        assert_eq!(r.blocks[&parent].refs, 31);
        for owner in [child, parent, grandparent] {
            assert!(work.block_inits.contains(&(r.blocks[&owner].slot, owner.3, owner.4)));
        }
        table_is_exact(&r);
    }

    #[test]
    fn summary_handoff_retains_exact_records_and_detached_release_cannot_clear_replacement() {
        let (_, mut r, old, new) = current_alias_fixture();
        let old_records: Vec<_> = old.iter().map(|&key| r.residents.get(key).unwrap().record).collect();
        let mut blocks = vec![new[0]];
        let selected = select_camera_owners(&mut blocks);
        r.queue_obsolete_owners(new[0]);
        let mut work = FrameWork::default();
        let attached_before = r.block_owner.clone();
        r.handoff_camera_owners(&selected, &mut work, &|| true);
        assert_eq!(r.block_owner, attached_before);
        assert!(work.block_inits.is_empty(), "expired budget cannot detach any owner");
        r.handoff_camera_owners(&selected, &mut work, &|| false);
        assert!(work.evictions.is_empty() && work.jobs.is_empty());
        assert_eq!(r.blocks.len(), 3, "absent incoming demand cannot create ghost summaries");
        assert!(r.blocks.values().all(|block| block.refs == 16));
        assert!(r.block_owner.is_empty() && r.live_tier1.is_empty());
        assert_eq!(old.iter().map(|&key| r.residents.get(key).unwrap().record).collect::<Vec<_>>(), old_records);
        while r.retire_obsolete_owner_step(&mut work) {}
        assert!(work.evictions.is_empty(), "detached summary walks stop without retiring exact records");
        for &key in &new {
            let record = r.alloc_record().unwrap();
            assert!(r.acquire_blocks(key, &mut work));
            r.residents.insert(key, Resident { record, blocks: true, ..Default::default() });
        }
        let attached = r.block_owner.clone();
        let live = r.live_tier1.clone();
        work.block_inits.clear();
        for &key in &old { r.evict(key, &mut work); }
        assert_eq!(r.block_owner, attached);
        assert_eq!(r.live_tier1, live);
        assert!(work.block_inits.is_empty(), "old detached refs cannot reset replacement GPU slots");
        assert!(new.iter().all(|&key| r.residents.get(key).unwrap().blocks));
        table_is_exact(&r);
    }

    #[test]
    fn summary_handoff_return_through_sibling_reattaches_parents_without_regeneration() {
        let (_, mut r, old, new) = current_alias_fixture();
        let (face, level, i, j) = unpack(old[0]);
        let sibling = pack(key0(face, level, i ^ 4), j as u32);
        let (siblings, count) = r.visible_columns(sibling).unwrap();
        for &key in &siblings[..count] {
            let record = r.alloc_record().unwrap();
            assert!(r.acquire_blocks(key, &mut FrameWork::default()));
            r.residents.insert(key, Resident { record, blocks: true, ..Default::default() });
        }
        let mut incoming = vec![new[0]];
        let selected = select_camera_owners(&mut incoming);
        let mut work = FrameWork::default();
        r.handoff_camera_owners(&selected, &mut work, &|| false);
        for &key in &new {
            let record = r.alloc_record().unwrap();
            assert!(r.acquire_blocks(key, &mut work));
            r.residents.insert(key, Resident { record, blocks: true, ..Default::default() });
        }
        let records: Vec<_> = siblings[..count].iter().map(|&key| r.residents.get(key).unwrap().record).collect();
        let mut returning = vec![sibling];
        let selected = select_camera_owners(&mut returning);
        work = FrameWork::default();
        r.handoff_camera_owners(&selected, &mut work, &|| false);
        assert!(!r.blocks_conflict(sibling));
        for (&slot, &owner) in &selected {
            assert_eq!(r.block_owner.get(&slot), Some(&owner));
            if owner.2 > 1 {
                assert!(work.block_inits.contains(&(slot, owner.3, owner.4)), "reattached parent must rebuild existing exact children");
            }
        }
        assert_eq!(siblings[..count].iter().map(|&key| r.residents.get(key).unwrap().record).collect::<Vec<_>>(), records);
        assert!(work.jobs.is_empty() && work.evictions.is_empty());
        assert_eq!(r.live_tier1.iter().copied().collect::<FxHashSet<_>>().len(), r.live_tier1.len());
        assert!(old.iter().chain(&new).all(|&key| r.residents.contains_key(key)));
        table_is_exact(&r);
    }

    #[test]
    fn summary_handoff_preserves_inflight_edit_journal_until_normal_retirement_ack() {
        let (_, mut r, old, new) = current_alias_fixture();
        let edit = r.edits.alloc(2, r.capacity.edit_words).unwrap();
        let record = r.residents.get(old[0]).unwrap().record;
        r.residents.get_mut(old[0]).unwrap().edit_block = Some(edit);
        r.publishing.insert(old[0], EditPublication { record, previous: Some(edit), next: Some(edit), evicted: false, initial_bucket: None });
        let mut incoming = vec![new[0]];
        let selected = select_camera_owners(&mut incoming);
        let mut work = FrameWork::default();
        r.handoff_camera_owners(&selected, &mut work, &|| false);
        assert_eq!(r.residents.get(old[0]).unwrap().edit_block, Some(edit));
        assert!(!r.publishing[&old[0]].evicted);
        assert!(!r.edits.free[edit.1 as usize].contains(&edit.0));
        assert!(!r.free_records.contains(&record));
        r.evict(old[0], &mut work);
        assert!(r.publishing[&old[0]].evicted);
        assert!(!r.edits.free[edit.1 as usize].contains(&edit.0));
        r.complete_jobs([(old[0], 0)]);
        assert!(r.edits.free[edit.1 as usize].contains(&edit.0));
        assert!(!r.publishing.contains_key(&old[0]));
    }

    #[test]
    fn current_alias_settled_demand_retires_stale_wanted_and_publishes_current_block() {
        let (planet, mut r, old, new) = current_alias_fixture();
        let epoch = r.snapshot_epoch;
        r.queue_obsolete_owners(new[0]);
        let mut retired = FrameWork::default();
        r.apply_snapshot(&mut retired, &|| false);
        assert_eq!(retired.evictions.len(), old.len());
        assert!(old.iter().all(|&key| !r.residents.contains_key(key) && r.current_wanted(key)));
        assert!(new.iter().all(|&key| !r.blocks_conflict(key)));
        assert!(r.obsolete_owners.is_empty());
        assert_eq!(r.snapshot_epoch, epoch, "a stationary turn needs no new worker epoch to retire aliases");

        let eye = r.current_request.as_ref().unwrap().eye;
        let work = r.plan(&planet, eye, 160.0, 16);
        assert_eq!(work.job_keys, new);
        r.complete_jobs(work.job_keys.iter().map(|&key| (key, 0)));
        table_is_exact(&r);
        assert!(new.iter().all(|&key| r.residents.contains_key(key)));
    }

    #[test]
    fn current_alias_stale_adds_and_admission_preserve_evicted_edit_publication() {
        let (mut planet, mut r, old, new) = current_alias_fixture();
        let (face, level, i, j) = unpack(old[0]);
        let center = r.grid.ground_point(face, f64::from((i + 2) * (BRICK << level)),
            f64::from((j + 2) * (BRICK << level)));
        std::sync::Arc::make_mut(&mut planet).apply(crate::edits::Brush {
            center: center.to_array(), ..test_brush(1.5)
        }).unwrap();
        r.sync_edits(&planet, &mut FrameWork::default());
        r.urgent.clear();
        for request in [&mut r.current_request, &mut r.last_request] {
            let request = request.as_mut().unwrap();
            request.outer_radius = planet.outer_radius();
            request.planet = Some(planet.clone());
        }
        let previous = r.edits.alloc(2, r.capacity.edit_words).unwrap();
        let next = r.edits.alloc(2, r.capacity.edit_words).unwrap();
        let record = r.residents.get(old[0]).unwrap().record;
        r.residents.get_mut(old[0]).unwrap().edit_block = Some(previous);
        r.publishing.insert(old[0], EditPublication { record, previous: Some(previous),
            next: Some(next), evicted: false, initial_bucket: None });
        r.queue_obsolete_owners(new[0]);
        r.apply_snapshot(&mut FrameWork::default(), &|| false);
        assert!(r.publishing[&old[0]].evicted);
        for journal in [previous, next] {
            assert!(!r.edits.free[journal.1 as usize].contains(&journal.0));
        }

        let wanted: Vec<_> = old.iter().chain(&new).copied().collect();
        r.apply(snapshot_update(3, &wanted));
        r.apply_snapshot(&mut FrameWork::default(), &|| false);
        assert!(old.iter().all(|key| !r.levels[0].pending.at.contains_key(key)), "stale accepted additions cannot reclaim an alias");
        assert!(new.iter().all(|key| r.levels[0].pending.at.contains_key(key)));
        r.complete_jobs([(old[0], 2)]);
        assert!(!r.publishing.contains_key(&old[0]) && !r.initial_retries.contains(&old[0]));
        for journal in [previous, next] {
            assert!(r.edits.free[journal.1 as usize].contains(&journal.0));
        }
        assert_eq!(planet.edits().len(), 1, "retiring GPU ownership retains the canonical edit");

        // Even an already queued stale block must fail both the batch and
        // scalar gates, instead of recapturing the just released owner slot.
        r.levels[0].pending.clear();
        for &key in &old[1..] { r.levels[0].pending.insert(key, 0); }
        let mut rejected = FrameWork::default();
        assert!(!r.admit_pending_block(&planet, 0, old[0], 0, 16, None, false, &mut rejected));
        assert!(rejected.jobs.is_empty());
        r.levels[0].pending.insert(old[0], 0);
        r.ground_clearance = None;
        let eye = r.current_request.as_ref().unwrap().eye;
        let work = r.plan(&planet, eye, 160.0, 1);
        assert!(work.job_keys.iter().all(|key| !old.contains(key)));
        assert!(old.iter().all(|&key| !r.residents.contains_key(key)));
        table_is_exact(&r);
    }

    #[test]
    fn current_alias_selected_replaces_live_captured_owner_and_quarantines_edit_ack() {
        let (mut planet, mut r, old, new) = current_alias_fixture();
        let source = r.visible_view.unwrap();
        r.visible_leases.insert(old[0], VisibleLease {
            origin: LeaseOrigin::Captured(source), serial: r.requested,
            retiring: false, retired: 0, current_demand_frame: None,
        });
        assert!(r.source_is_current(source, VISIBLE_LEASE_FRAMES) && r.transient_wanted(old[0]));
        assert!(r.applied_levels[0] < r.visible_leases[&old[0]].serial,
            "the live captured owner still awaits its worker acknowledgment");
        let (face, level, i, j) = unpack(old[0]);
        let center = r.grid.ground_point(face, f64::from((i + 2) * (BRICK << level)),
            f64::from((j + 2) * (BRICK << level)));
        std::sync::Arc::make_mut(&mut planet).apply(crate::edits::Brush {
            center: center.to_array(), ..test_brush(1.5)
        }).unwrap();
        r.sync_edits(&planet, &mut FrameWork::default());
        r.urgent.clear();
        for request in [&mut r.current_request, &mut r.last_request] {
            let request = request.as_mut().unwrap();
            request.outer_radius = planet.outer_radius();
            request.planet = Some(planet.clone());
        }
        let previous = r.edits.alloc(2, r.capacity.edit_words).unwrap();
        let next = r.edits.alloc(2, r.capacity.edit_words).unwrap();
        let record = r.residents.get(old[0]).unwrap().record;
        r.residents.get_mut(old[0]).unwrap().edit_block = Some(previous);
        r.publishing.insert(old[0], EditPublication { record, previous: Some(previous),
            next: Some(next), evicted: false, initial_bucket: None });
        // Put a partially walked older request before the current alias.
        let oldest = (0, face, 3, -8, -8);
        r.obsolete_owners.push_back(OwnerRetirement { owner: oldest, offset: 256, selected: false });
        let mut blocks = vec![new[0]];
        let selected = select_camera_owners(&mut blocks);
        r.prioritize_camera_owners(&blocks, &selected);
        assert_ne!(r.obsolete_owners.front().unwrap().owner, oldest);
        assert!(r.obsolete_owners.iter().any(|task| task.owner == oldest && task.offset == 256));
        assert_eq!(r.visible_leases[&old[0]].captured_source().frame, source.frame);
        assert_eq!(r.visible_leases[&old[0]].captured_source().at, source.at);
        assert!(r.visible_leases[&old[0]].retiring && !r.transient_wanted(old[0]));
        let mut retired = FrameWork::default();
        r.apply_snapshot_selected(&mut retired, &|| false, &selected);
        assert_eq!(retired.evictions.len(), old.len());
        assert!(new.iter().all(|&key| !r.blocks_conflict(key)));
        assert!(r.publishing[&old[0]].evicted);
        for journal in [previous, next] {
            assert!(!r.edits.free[journal.1 as usize].contains(&journal.0));
        }
        assert!(r.delayed_records.contains(&record), "retired records stay quarantined until a later frame");

        // Fresh feedback for the same losing owner cannot refresh its stamp
        // or reissue work before the current selected block is published.
        r.prioritize_visible_blocks_from([(old[0] as u32, (old[0] >> 32) as u32)],
            source.frame + 1, source.view, source.at + std::time::Duration::from_millis(1));
        r.refresh_visible_pending(None);
        assert!(r.visible_leases[&old[0]].retiring && !r.transient_wanted(old[0]));
        for &key in &old { r.levels[0].pending.insert(key, 0); }
        let eye = r.current_request.as_ref().unwrap().eye;
        let work = r.plan(&planet, eye, 160.0, 16);
        assert_eq!(work.job_keys, new);
        assert!(old.iter().all(|&key| !r.residents.contains_key(key)));
        r.complete_jobs([(old[0], 2)]);
        for journal in [previous, next] {
            assert!(r.edits.free[journal.1 as usize].contains(&journal.0));
        }
        r.complete_jobs(work.job_keys.iter().map(|&key| (key, 0)));
        table_is_exact(&r);
        assert_eq!(planet.edits().len(), 1);
    }

    #[test]
    fn current_alias_selected_promotion_revisits_protected_prefix_once_then_keeps_progress() {
        let (_, mut r, old, new) = current_alias_fixture();
        r.queue_obsolete_owners(new[0]);
        for task in &mut r.obsolete_owners { task.offset = 8; }
        assert!(old.iter().all(|&key| r.residents.contains_key(key)));
        let mut blocks = vec![new[0]];
        let selected = select_camera_owners(&mut blocks);
        r.prioritize_camera_owners(&blocks, &selected);
        assert!(r.obsolete_owners.iter().all(|task| task.offset == 0 && task.selected));
        let mut work = FrameWork::default();
        let mut last = None;
        for _ in 0..8 {
            assert!(r.retire_obsolete_owner_step_current(&mut work, &mut last, &selected));
        }
        assert_eq!(work.evictions.len(), 8, "the earlier protected prefix retires in this bounded slice");
        assert_eq!(r.obsolete_owners.front().unwrap().offset, 8);
        r.prioritize_camera_owners(&blocks, &selected);
        assert_eq!(r.obsolete_owners.front().unwrap().offset, 8,
            "continuing selected priority must not restart a partial walk every frame");
        r.prioritize_camera_owners(&[], &SelectedOwners::default());
        assert!(r.obsolete_owners.iter().all(|task| !task.selected),
            "withdrawal clears priority without requiring a deadline-limited walker visit");
        let record = r.alloc_record().unwrap();
        assert!(r.acquire_blocks(old[0], &mut work));
        r.residents.insert(old[0], Resident { record, blocks: true, ..Default::default() });
        r.prioritize_camera_owners(&blocks, &selected);
        assert_eq!(r.obsolete_owners.front().unwrap().offset, 0,
            "a later conflicting view must revisit regenerated prefix columns");
        r.apply_snapshot_selected(&mut work, &|| false, &selected);
        assert_eq!(work.evictions.len(), 17);
        assert!(new.iter().all(|&key| !r.blocks_conflict(key)));
    }

    #[test]
    fn current_alias_selected_winner_is_coherent_and_nonconflicting_capture_stays_live() {
        let (_, mut r, old, new) = current_alias_fixture();
        let source = r.visible_view.unwrap();
        let safe = new[0] + 4;
        r.visible_leases.insert(safe, VisibleLease { origin: LeaseOrigin::Captured(source),
            serial: r.requested, retiring: false, retired: 0, current_demand_frame: None });
        let mut blocks = vec![new[0], old[0], safe];
        let selected = select_camera_owners(&mut blocks);
        assert_eq!(blocks, vec![new[0], safe], "one ranked winner owns all three slots; the losing alias cannot ping-pong");
        assert!(!selected_owner_allows(old[0], &selected));
        r.prioritize_camera_owners(&blocks, &selected);
        assert!(!r.visible_leases[&safe].retiring && r.transient_wanted(safe));
        let mut repeated = vec![new[0], old[0], safe];
        assert_eq!(select_camera_owners(&mut repeated), selected);
        assert_eq!(repeated, blocks);
        assert!(selected.len() <= CAMERA_CANDIDATES * BLOCK_TIERS as usize);
        assert!(r.obsolete_owners.len() <= 64);
    }

    #[test]
    fn current_alias_returned_and_live_transient_owners_remain_protected() {
        for transient in [false, true] {
            let (_, mut r, old, new) = current_alias_fixture();
            if transient {
                let source = r.visible_view.unwrap();
                r.visible_leases.insert(old[0], VisibleLease {
                    origin: LeaseOrigin::Captured(source), serial: r.requested,
                    retiring: false, retired: 0, current_demand_frame: Some(r.frame),
                });
                assert!(r.transient_wanted(old[0]));
            } else {
                let (face, level, i, j) = unpack(old[0]);
                let eye = r.grid.ground_point(face, f64::from((i + 2) * (BRICK << level)),
                    f64::from((j + 2) * (BRICK << level))) + DVec3::Y * 10.0;
                r.current_request.as_mut().unwrap().eye = eye;
                assert!(current_geometry_allows(&r.grid, r.current_request.as_ref(), old[0]));
            }
            r.queue_obsolete_owners(new[0]);
            let mut retired = FrameWork::default();
            r.apply_snapshot(&mut retired, &|| false);
            assert!(retired.evictions.is_empty());
            assert!(old.iter().all(|&key| r.residents.contains_key(key)));
            assert!(r.blocks_conflict(new[0]), "current or transient owner cannot be silently replaced");
        }
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
    fn plan_view_first_camera_leases_protect_residents_from_old_removals() {
        for snapshot in [false, true] {
            let (planet, mut r, _, _) = current_bridge_fixture();
            let eye = r.current_request.as_ref().unwrap().eye;
            r.set_ground_clearance(10.0);
            let block = r.camera_blocks(eye, &|| false)[0];
            let (keys, count) = r.visible_columns(block).unwrap();
            for &key in &keys[..count] {
                let record = r.alloc_record().unwrap();
                assert!(r.acquire_blocks(key, &mut FrameWork::default()));
                r.residents.insert(key, Resident { record, blocks: true, ..Default::default() });
            }
            let level = unpack(block).1;
            r.snapshot_mode = snapshot;
            r.apply(WindowUpdate {
                serial: 4, snapshot, partial: true, processed_levels: 1 << level,
                wanted: vec![(level, Default::default())],
                levels: vec![LevelDiff { level, active: true,
                    removes: if snapshot { Vec::new() } else { keys[..count].to_vec() },
                    ..Default::default() }], ..Default::default()
            });
            let work = r.plan(&planet, eye, 160.0, 0);
            assert!(r.transient_wanted(block), "current camera owns its records before background retirement");
            assert!(work.evictions.is_empty(), "old {} retirement must not remove visible records", if snapshot { "snapshot" } else { "diff" });
            assert!(keys[..count].iter().all(|&key| r.residents.contains_key(key)));
        }
    }

    #[test]
    fn plan_preparation_expiry_stamps_current_leases_without_unbudgeted_retirement() {
        let (_, mut r, _, _) = current_bridge_fixture();
        let eye = r.current_request.as_ref().unwrap().eye;
        r.set_ground_clearance(10.0);
        let blocks = r.camera_blocks(eye, &|| false);
        r.refresh_camera_pending(&blocks[..2], None);
        assert_eq!(r.visible_leases.len(), 2);
        let prior_frame = r.frame;
        let ranks = r.visible_admission.clone();
        r.frame += 1;
        let mut work = FrameWork::default();
        r.retire_visible_leases_current(&blocks[..1], &mut work, &|| true);
        assert_eq!(r.visible_leases.len(), 2, "expired preparation cannot scan and retire the omitted lease");
        assert_eq!(r.visible_leases[&blocks[0]].current_demand_frame, Some(r.frame));
        assert_eq!(r.visible_leases[&blocks[1]].current_demand_frame, Some(prior_frame));
        assert!(r.transient_wanted(blocks[0]));
        assert!(!r.transient_wanted(blocks[1]), "unvisited old demand is not renewed");
        assert_eq!(r.visible_admission, ranks);
        assert!(work.evictions.is_empty() && work.block_inits.is_empty());
    }

    #[test]
    fn plan_preparation_backlog_leaves_time_for_existing_pending_publication() {
        let (planet, mut r, _, _) = current_bridge_fixture();
        let eye = r.current_request.as_ref().unwrap().eye;
        r.set_ground_clearance(10.0);
        let top = r.grid.levels() - 1;
        let (cell, _) = r.grid.locate(eye);
        let global = pack(key0(cell.face, top, cell.i >> (top + 3)), (cell.j >> (top + 3)) as u32);
        // Replay a large valid addition backlog without inventing extra wanted
        // tiles. The already pending column must not wait for that scan to end.
        r.apply(WindowUpdate {
            serial: 4, snapshot: true, partial: true, processed_levels: 1 << top,
            wanted: vec![(top, std::sync::Arc::new([global].into_iter().collect()))],
            levels: vec![LevelDiff { level: top, active: true,
                adds: vec![(0.0, global); 500_000], ..Default::default() }],
            ..Default::default()
        });
        r.levels[top as usize].pending.insert(global, 0);
        r.set_cpu_budget(Some(std::time::Duration::from_millis(4)));
        // A preempted test process can miss one frame's allowance; subsequent
        // bounded plans must make progress, without a wall-clock assertion.
        let mut submitted = None;
        for _ in 0..4 {
            let work = r.plan(&planet, eye, 160.0, 1);
            if !work.jobs.is_empty() { submitted = Some(work); break; }
        }
        let work = submitted.expect("window preparation must leave admission time");
        assert_eq!(work.job_keys, vec![global]);
        assert!(r.queued_delta_ops > 0, "publication precedes completion of the addition backlog");
        assert!(r.publishing.contains_key(&global));
        table_is_exact(&r);
    }

    #[test]
    fn plan_view_first_global_bootstrap_and_urgent_edit_keep_priority() {
        for urgent in [false, true] {
            let (planet, mut r, _, _) = current_bridge_fixture();
            let eye = r.current_request.as_ref().unwrap().eye;
            r.set_ground_clearance(10.0);
            let top = r.grid.levels() - 1;
            let (cell, _) = r.grid.locate(eye);
            let global = pack(key0(cell.face, top, cell.i >> (top + 3)), (cell.j >> (top + 3)) as u32);
            let fine = r.camera_blocks(eye, &|| false)[0];
            r.apply(WindowUpdate {
                serial: 4, snapshot: true, partial: true, processed_levels: (1 << top) | 1,
                wanted: vec![(top, std::sync::Arc::new([global].into_iter().collect())), (0, Default::default())],
                levels: vec![LevelDiff { level: top, active: true, adds: vec![(0.0, global)], ..Default::default() },
                    LevelDiff { level: 0, active: true, adds: vec![(0.0, fine)], ..Default::default() }],
                ..Default::default()
            });
            let cursor = r.retire_slot;
            r.queue_global_adds(&|| false);
            assert_eq!(r.retire_slot, cursor, "bootstrap does not scan residents");
            assert_eq!(r.diffs[0][0].added, 0, "fine snapshot stays background work");
            assert!(!r.levels[top as usize].pending.is_empty());
            if urgent {
                let record = r.alloc_record().unwrap();
                assert!(r.acquire_blocks(fine, &mut FrameWork::default()));
                r.residents.insert(fine, Resident { record, blocks: true, ..Default::default() });
                r.urgent.push(fine);
            }
            let work = r.plan(&planet, eye, 160.0, 1);
            assert_eq!(work.job_keys, vec![if urgent { fine } else { global }]);
            if urgent { assert_eq!(work.jobs[0].flags, 1, "urgent published record remains a replacement"); }
        }
    }

    #[test]
    fn plan_view_first_capacity_cleanup_reclaims_only_after_admission() {
        let (planet, mut r, old, new) = current_alias_fixture();
        let eye = r.current_request.as_ref().unwrap().eye;
        r.levels[0].wanted = Some(Default::default());
        r.capacity.records = r.next_record;
        r.delayed_records.clear();
        r.free_records.clear();
        r.snapshot_mode = true;
        r.snapshot_epoch += 1;
        r.retire_started_epoch = r.snapshot_epoch;
        r.retire_finished_epoch = 0;
        let old_records: Vec<_> = old.iter().map(|&key| r.residents.get(key).unwrap().record).collect();
        let blocked = r.plan(&planet, eye, 160.0, 16);
        assert!(blocked.jobs.is_empty(), "same-frame evictions retain their record quarantine");
        assert_eq!(blocked.evictions.len(), old.len(), "capacity failure still reaches background cleanup");
        assert!(old_records.iter().all(|record| blocked.evictions.contains(record)));
        assert!(r.free_records.is_empty());
        let next = r.plan(&planet, eye, 160.0, 16);
        assert_eq!(next.jobs.len(), 16);
        assert_eq!(next.job_keys, new, "the queued current tile uses safely reclaimed records next frame");
    }

    #[test]
    fn pool_pressure_reclaims_before_admission_preserving_selected_edits_and_deadline() {
        let (_, mut r, old, current) = current_alias_fixture();
        let record = r.alloc_record().unwrap();
        let previous = r.edits.alloc(2, r.capacity.edit_words).unwrap();
        let next = r.edits.alloc(2, r.capacity.edit_words).unwrap();
        r.residents.insert(current[0], Resident { record, edit_block: Some(previous), ..Default::default() });
        r.publishing.insert(current[0], EditPublication { record, previous: Some(previous),
            next: Some(next), evicted: false, initial_bucket: None });
        r.levels[0].wanted = Some(Default::default());
        let epoch = r.snapshot_epoch;
        assert_eq!(r.retire_finished_epoch, epoch);
        r.complete_jobs([(current[0], 3)]);
        assert!(r.pool_pressure && r.urgent.contains(&current[0]), "replacement pool failure must activate reclamation");
        assert_eq!(r.snapshot_epoch, epoch + 1, "pool pressure wakes a settled cursor once");
        r.complete_jobs(std::iter::empty());
        r.complete_jobs([(u64::MAX, 0)]);
        assert!(r.pool_pressure, "empty or stale acknowledgements cannot clear pressure");
        r.publishing.insert(old[0], EditPublication {
            record: r.residents.get(old[0]).unwrap().record, previous: None,
            next: None, evicted: false, initial_bucket: Some(0),
        });
        let mut blocks = vec![current[0]];
        let selected = select_camera_owners(&mut blocks);
        let mut work = FrameWork::default();
        let quarantined = r.delayed_records.len();
        r.retire_pool_pressure(&mut work, &selected, Some(std::time::Instant::now()));
        assert!(work.evictions.is_empty(), "expired total frame budget admits no reclamation slice");
        r.retire_pool_pressure(&mut work, &selected, None);
        assert_eq!(work.evictions.len(), old.len(), "obsolete pool-backed data is reclaimed before view preparation");
        assert!(old.iter().all(|&key| !r.residents.contains_key(key)));
        assert!(r.residents.contains_key(current[0]), "reselected records need protection before Camera lease creation");
        assert_eq!(r.residents.get(current[0]).unwrap().edit_block, Some(previous));
        assert!(!r.edits.free[previous.1 as usize].contains(&previous.0));
        assert!(r.edits.free[next.1 as usize].contains(&next.0));
        assert!(r.free_records.is_empty(), "retired records retain their next-frame quarantine");
        assert_eq!(r.delayed_records.len(), quarantined + old.len());
        r.refresh_camera_pending(&blocks, Some(std::time::Instant::now()));
        assert!(work.jobs.is_empty(), "reclamation already progressed even when admission has no remaining time");
        let woke = r.snapshot_epoch;
        r.complete_jobs([(old[0], 0)]);
        assert!(r.pool_pressure, "evicted publication acknowledgements do not clear live pool pressure");
        let other_record = r.alloc_record().unwrap();
        r.residents.insert(current[1], Resident { record: other_record, ..Default::default() });
        r.publishing.insert(current[1], EditPublication { record: other_record, previous: None,
            next: None, evicted: false, initial_bucket: Some(0) });
        r.publishing.insert(current[0], EditPublication { record, previous: Some(previous),
            next: None, evicted: false, initial_bucket: None });
        r.complete_jobs([(current[0], 3), (current[1], 0)]);
        assert!(r.pool_pressure, "a mixed live batch containing pool failure retains pressure");
        assert_eq!(r.snapshot_epoch, woke, "mixed acknowledgements do not repeatedly restart reclamation");
        r.publishing.insert(current[0], EditPublication { record, previous: Some(previous),
            next: None, evicted: false, initial_bucket: None });
        r.complete_jobs([(current[0], 0)]);
        assert!(r.pool_pressure, "a success cannot abandon queued replacement retry ownership");
        // The production urgent loop consumes a retry before submitting it.
        // Resolve that queue and finish the actual protected reclaim cursor.
        r.urgent.clear();
        r.apply_snapshot_selected(&mut FrameWork::default(), &|| false, &selected);
        r.retire_pool_pressure(&mut FrameWork::default(), &selected, None);
        assert!(!r.pool_pressure, "resolved retries and completed retirement return to normal admission");
        r.complete_jobs([(u64::MAX, 3)]);
        assert!(!r.pool_pressure, "stale pool failures cannot re-enter pressure");
        assert!(r.edits.free[previous.1 as usize].contains(&previous.0));
    }

    #[test]
    fn admission_retry_head_reuses_exact_record_before_large_ordinary_queue() {
        let (planet, mut r, _, _) = current_bridge_fixture();
        let eye = r.current_request.as_ref().unwrap().eye;
        r.set_ground_clearance(10.0);
        let block = r.camera_blocks(eye, &|| false)[0];
        let (keys, count) = r.visible_columns(block).unwrap();
        assert_eq!(count, 16);
        r.levels[0].wanted = Some(std::sync::Arc::new(keys.into_iter().collect()));
        for &key in &keys {
            let record = r.alloc_record().unwrap();
            assert!(r.acquire_blocks(key, &mut FrameWork::default()));
            r.residents.insert(key, Resident { record, blocks: true, ..Default::default() });
        }
        let record = r.residents.get(keys[0]).unwrap().record;
        r.initial_retries.insert(keys[0]);
        r.levels[0].pending.insert(keys[0], 0);
        r.visible_admission.extend(keys.map(|key| (0, key)));
        for j in 0..200 {
            for i in 0..200 {
                r.levels[3].pending.insert(pack(key0(crate::grid::PLANE_FACE, 3, i), j as u32), BUCKETS - 1);
            }
        }
        let work = r.plan(&planet, eye, 160.0, 16);
        assert_eq!(work.job_keys.first(), Some(&keys[0]), "a known retry head must reach its scalar record reuse before ordinary work");
        assert_eq!(work.jobs[0].record, record);
        assert_eq!(r.publishing[&keys[0]].record, record);
        assert!(!r.initial_retries.contains(&keys[0]));
        assert!(work.jobs.len() <= 16);
        assert_eq!(r.levels[3].pending.len(), 40_000, "visible retry priority does not consume the ordinary backlog");
        table_is_exact(&r);
    }

    #[test]
    fn pool_pressure_success_keeps_older_initial_retry_until_it_resolves() {
        let (planet, mut r, _, current) = current_alias_fixture();
        let eye = r.current_request.as_ref().unwrap().eye;
        r.levels[0].wanted = Some(std::sync::Arc::new(current.iter().copied().collect()));
        let selected = select_camera_owners(&mut vec![current[0]]);
        r.handoff_camera_owners(&selected, &mut FrameWork::default(), &|| false);
        for &key in &current[..2] {
            let record = r.alloc_record().unwrap();
            assert!(r.acquire_blocks(key, &mut FrameWork::default()));
            r.residents.insert(key, Resident { record, blocks: true, ..Default::default() });
            r.publishing.insert(key, EditPublication { record, previous: None, next: None,
                evicted: false, initial_bucket: Some(0) });
        }
        r.complete_jobs([(current[0], 3)]);
        assert!(r.pool_pressure && r.initial_retries.contains(&current[0]));
        r.complete_jobs([(current[1], 0)]);
        assert!(r.pool_pressure && r.initial_retries.contains(&current[0]),
            "unrelated successful initial publication cannot abandon an older failed initial retry");
        let work = r.plan(&planet, eye, 160.0, 1);
        assert_eq!(work.job_keys, vec![current[0]], "the failed record is reused through normal admission");
        assert!(!r.initial_retries.contains(&current[0]));
        r.complete_jobs([(current[0], 0)]);
        assert_eq!(r.retire_finished_epoch, r.snapshot_epoch);
        assert!(!r.pool_pressure, "resolved retry and finished pass release the pressure slice");
        table_is_exact(&r);
    }

    #[test]
    fn pool_pressure_completed_retirement_clears_without_another_acknowledgment() {
        let (_, mut r, _, _) = current_alias_fixture();
        r.pool_pressure = true;
        r.snapshot_epoch += 1;
        r.retire_started_epoch = r.snapshot_epoch;
        assert!(r.retire_finished_epoch < r.snapshot_epoch);
        r.clear_resolved_pool_pressure();
        assert!(r.pool_pressure, "unfinished retirement remains live even after retry queues resolve");
        r.apply_snapshot(&mut FrameWork::default(), &|| false);
        r.retire_pool_pressure(&mut FrameWork::default(), &SelectedOwners::default(), None);
        assert_eq!(r.retire_finished_epoch, r.snapshot_epoch);
        assert!(!r.pool_pressure, "retirement completion need not wait for a nonexistent next job acknowledgment");
    }

    #[test]
    fn pool_pressure_expired_diff_deadline_preserves_explicit_removals() {
        let (_, mut r, old, _) = current_alias_fixture();
        r.snapshot_mode = false;
        r.apply(WindowUpdate { serial: 4, partial: true, processed_levels: 1,
            wanted: vec![(0, Default::default())],
            levels: vec![LevelDiff { level: 0, active: true,
                removes: old.clone(), ..Default::default() }], ..Default::default() });
        let key = old[0];
        r.publishing.insert(key, EditPublication { record: r.residents.get(key).unwrap().record,
            previous: None, next: None, evicted: false, initial_bucket: Some(0) });
        r.complete_jobs([(key, 3)]);
        assert!(r.pool_pressure);
        let mut work = FrameWork::default();
        r.retire_pool_pressure(&mut work, &Default::default(), Some(std::time::Instant::now()));
        assert!(work.evictions.is_empty());
        assert_eq!(r.diffs[0][0].removed, 0);
        assert!(old.iter().all(|&key| r.residents.contains_key(key)));
        r.retire_pool_pressure(&mut work, &Default::default(), None);
        assert_eq!(work.evictions.len(), old.len());
    }

    #[test]
    fn pool_pressure_initial_retry_and_clear_ack_preserve_normal_budget_and_ownership() {
        let (planet, mut r, _, _) = current_bridge_fixture();
        let eye = r.current_request.as_ref().unwrap().eye;
        r.set_ground_clearance(10.0);
        let current = r.camera_blocks(eye, &|| false)[0];
        r.refresh_camera_pending(&[current], None);
        let first = r.plan(&planet, eye, 160.0, 16);
        assert_eq!(first.jobs.len(), 16);
        let key = first.job_keys[0];
        r.complete_jobs(first.job_keys.iter().map(|&key| (key, 3)));
        assert!(r.pool_pressure && r.initial_retries.contains(&key));
        let record = r.residents.get(key).unwrap().record;
        r.set_cpu_budget(Some(std::time::Duration::ZERO));
        let deferred = r.plan(&planet, eye, 160.0, 16);
        assert!(deferred.jobs.is_empty() && deferred.evictions.is_empty());
        assert_eq!(r.residents.get(key).unwrap().record, record);
        r.set_cpu_budget(None);
        let retry = r.plan(&planet, eye, 160.0, 1);
        assert_eq!(retry.job_keys, vec![key]);
        assert_eq!(retry.jobs[0].record, record);
        r.complete_jobs([(key, 0)]);
        assert!(r.pool_pressure, "the other fifteen failed initial publications still need reclamation");
        assert_eq!(r.initial_retries.len(), 15);
        let remaining = r.plan(&planet, eye, 160.0, 15);
        assert_eq!(remaining.jobs.len(), 15);
        assert!(remaining.job_keys.iter().all(|key| first.job_keys.contains(key) && *key != first.job_keys[0]));
        r.complete_jobs(remaining.job_keys.iter().map(|&key| (key, 0)));
        r.apply_snapshot(&mut FrameWork::default(), &|| false);
        r.retire_pool_pressure(&mut FrameWork::default(), &Default::default(), None);
        assert!(r.initial_retries.is_empty());
        assert_eq!(r.retire_finished_epoch, r.snapshot_epoch);
        assert!(!r.pool_pressure);
    }

    #[test]
    fn plan_view_first_settled_full_pool_turn_wakes_background_retirement() {
        let (planet, mut r, old, new) = current_alias_fixture();
        let eye = r.current_request.as_ref().unwrap().eye;
        r.levels[0].wanted = Some(Default::default());
        r.capacity.records = r.next_record;
        r.delayed_records.clear();
        r.free_records.clear();
        assert!(r.snapshot_mode && r.diffs.iter().all(VecDeque::is_empty));
        assert_eq!(r.retire_finished_epoch, r.snapshot_epoch, "start with a completed retirement scan");
        let block = old[0] & !(3u64 | (3u64 << 32));
        r.visible_leases.insert(block, VisibleLease { origin: LeaseOrigin::Camera,
            serial: r.requested, retiring: false, retired: 0,
            current_demand_frame: Some(r.frame) });
        let serial = r.requested;
        let first = r.plan(&planet, eye, 160.0, 16);
        assert_eq!(r.requested, serial, "stationary view-basis turn does not request a new worker window");
        assert!(first.jobs.is_empty(), "full pool retains same-frame quarantine");
        assert_eq!(first.evictions.len(), old.len(), "lease departure wakes completed normal retirement");
        assert!(!r.visible_leases.contains_key(&block));
        let second = r.plan(&planet, eye, 160.0, 16);
        assert_eq!(second.job_keys, new, "next frame reuses safely retired records for the current tile");
    }

    #[test]
    fn plan_view_first_whole_tiles_advance_all_bands_with_stale_fine_snapshot() {
        let (planet, mut r, _, _) = current_bridge_fixture();
        let eye = r.current_request.as_ref().unwrap().eye;
        r.set_ground_clearance(10.0);
        let bands = r.camera_blocks(eye, &|| false);
        assert!([0, 1, 2].iter().all(|level| bands.iter().any(|&key| unpack(key).1 == *level)));
        let stale = pack(key0(crate::grid::PLANE_FACE, 0, 0), 0);
        r.apply(WindowUpdate {
            serial: 4, snapshot: true, partial: true, processed_levels: 1,
            wanted: vec![(0, Default::default())],
            levels: vec![LevelDiff { level: 0, active: true,
                adds: vec![(0.0, stale); 128], ..Default::default() }], ..Default::default()
        });
        let work = r.plan(&planet, eye, 160.0, 48);
        assert_eq!(work.jobs.len(), 48);
        for (level, keys) in work.job_keys.chunks_exact(16).enumerate() {
            assert!(keys.iter().all(|&key| unpack(key).1 == level as u32));
            let block = keys[0] & !(3u64 | (3u64 << 32));
            let (expected, count) = r.visible_columns(block).unwrap();
            assert_eq!(count, 16);
            assert_eq!(keys, expected, "FIFO interleaves complete tiles, never individual columns");
        }
        assert!(!work.job_keys.contains(&stale));
    }

    #[test]
    fn plan_view_first_full_old_camera_cap_drops_authorization_without_record_walk() {
        let (planet, mut r, _, _) = current_bridge_fixture();
        let eye = r.current_request.as_ref().unwrap().eye;
        r.set_ground_clearance(10.0);
        let incoming = r.camera_blocks(eye, &|| false)[0];
        let (face, level, i, j) = unpack(incoming);
        r.capacity.table_bits = 14;
        r.residents = ColumnIndex::new(14);
        let mut old = Vec::new();
        for index in 0..CAMERA_LEASES {
            let block = pack(key0(face, level, i - 512 - index as i32 * 4), j as u32);
            let (keys, count) = r.visible_columns(block).unwrap();
            for &key in &keys[..count] {
                let record = r.alloc_record().unwrap();
                let blocks = r.acquire_blocks(key, &mut FrameWork::default());
                r.residents.insert(key, Resident { record, blocks, ..Default::default() });
                r.visible_admission.push_back((level as usize, key));
                old.push(key);
            }
            r.visible_leases.insert(block, VisibleLease { origin: LeaseOrigin::Camera,
                serial: r.requested, retiring: false, retired: 0,
                current_demand_frame: Some(r.frame) });
        }
        assert_eq!(r.visible_leases.len(), 384);
        assert_eq!(old.len(), 6144);
        let previous = r.edits.alloc(2, r.capacity.edit_words).unwrap();
        let next = r.edits.alloc(2, r.capacity.edit_words).unwrap();
        let record = r.residents.get(old[0]).unwrap().record;
        r.residents.get_mut(old[0]).unwrap().edit_block = Some(previous);
        r.publishing.insert(old[0], EditPublication { record, previous: Some(previous),
            next: Some(next), evicted: false, initial_bucket: None });
        let refs: Vec<_> = r.blocks.iter().map(|(&key, block)| (key, block.refs)).collect();
        let mut retired = FrameWork::default();
        r.retire_slot = 37;
        let started_epoch = r.retire_started_epoch;
        let epoch = r.snapshot_epoch;
        r.retire_visible_leases_current(&[incoming], &mut retired, &|| false);
        assert_eq!(r.retire_slot, 37, "moving demand never restarts an active retirement cursor");
        assert_eq!(r.retire_started_epoch, started_epoch);
        assert_eq!(r.snapshot_epoch, epoch + 1);
        assert!(r.visible_leases.is_empty() && r.visible_admission.is_empty());
        assert!(retired.evictions.is_empty() && retired.block_inits.is_empty());
        assert_eq!(r.residents.len(), 6144, "authorization removal does not scan or evict old records");
        assert!(refs.iter().all(|(key, count)| r.blocks[key].refs == *count));
        assert!(!r.publishing[&old[0]].evicted);
        assert!(!r.edits.free[previous.1 as usize].contains(&previous.0));
        assert!(!r.edits.free[next.1 as usize].contains(&next.0));
        r.levels[0].wanted = Some(Default::default());
        r.retire_finished_epoch = r.snapshot_epoch;
        let work = r.plan(&planet, eye, 160.0, 16);
        let (expected, count) = r.visible_columns(incoming).unwrap();
        assert_eq!(count, 16);
        assert_eq!(work.job_keys, expected, "new current tile is admitted despite the full previous lease cap");
        let selected: Vec<_> = (1..=BLOCK_TIERS).map(|tier| {
            let owner = (level, face, tier, i >> (2 * tier), j >> (2 * tier));
            (block_slot(level, face, tier, owner.3, owner.4), owner)
        }).collect();
        let mut late = FrameWork::default();
        for &key in &old[..16] {
            if r.residents.contains_key(key) { r.evict(key, &mut late); }
        }
        assert!(r.publishing[&old[0]].evicted);
        r.complete_jobs([(old[0], 0)]);
        assert!(selected.iter().all(|(slot, owner)| r.block_owner.get(slot) == Some(owner)),
            "late old release and publication cannot clear a current summary owner");
        assert!(r.edits.free[previous.1 as usize].contains(&previous.0));
        assert!(r.edits.free[next.1 as usize].contains(&next.0));
        assert!(expected.iter().all(|&key| r.residents.get(key).is_some_and(|r| r.blocks)));
    }

    #[test]
    fn plan_view_first_fresh_captured_ridge_precedes_nonempty_ground_bands() {
        let (mut planet, mut r, _, at) = current_bridge_fixture();
        let eye = DVec3::new(0.0, 50.0, 0.0);
        let ridge = DVec3::new(8.0, 46.0, 0.0);
        std::sync::Arc::make_mut(&mut planet).apply(crate::edits::Brush {
            center: ridge.to_array(), ..test_brush(2.0)
        }).unwrap();
        r.set_ground_clearance(50.0);
        r.set_camera_view(DVec3::new(1.0, -0.17, 0.0), DVec3::Y, [0.65, 0.414]);
        for request in [&mut r.current_request, &mut r.last_request] {
            let request = request.as_mut().unwrap();
            request.eye = eye;
            request.lod0 = 30.0;
            request.outer_radius = planet.outer_radius();
            request.planet = Some(planet.clone());
        }
        let base = r.camera_base_level();
        assert!(base > 0);
        assert!(!r.camera_blocks(eye, &|| false).is_empty());
        let (cell, _) = r.grid.locate(ridge);
        let block = pack(key0(cell.face, 0, (cell.i >> 3) & !3), ((cell.j >> 3) & !3) as u32);
        r.prioritize_visible_blocks_from([(block as u32, (block >> 32) as u32)], 10, 7, at);
        let work = r.plan(&planet, eye, 30.0, 32);
        assert_eq!(work.jobs.len(), 32);
        let (ridge_keys, count) = r.visible_columns(block).unwrap();
        assert_eq!(count, 16);
        assert_eq!(&work.job_keys[..16], &ridge_keys,
            "fresh fine captured terrain must not starve behind ongoing ground demand");
        assert!(work.job_keys[16..].iter().all(|&key| unpack(key).1 as usize >= base));
        let LeaseOrigin::Captured(source) = r.visible_leases[&block].origin else { panic!("captured origin replaced") };
        assert_eq!(source.frame, 10);
        assert_eq!(source.view, 7);
        assert_eq!(source.at, at, "intake never renews a captured source clock");
    }

    #[test]
    fn camera_refresh_live_tile_authorization_preserves_expiry_and_partial_members() {
        for kind in 0..4 {
            let (_, mut r, _, _) = current_bridge_fixture();
            let eye = r.current_request.as_ref().unwrap().eye;
            r.set_ground_clearance(10.0);
            let block = r.camera_blocks(eye, &|| false)[0];
            let (keys, _) = r.visible_columns(block).unwrap();
            let mut source = r.visible_view.unwrap();
            if kind == 3 { source.frame = source.frame.wrapping_sub(VISIBLE_LEASE_FRAMES + 1); }
            let origin = if kind < 2 { LeaseOrigin::Camera } else { LeaseOrigin::Captured(source) };
            r.visible_leases.insert(block, VisibleLease { origin, serial: r.requested,
                retiring: kind == 1, retired: 0, current_demand_frame: Some(r.frame) });
            for &key in &keys[..5] {
                let record = r.alloc_record().unwrap();
                assert!(r.acquire_blocks(key, &mut FrameWork::default()));
                r.residents.insert(key, Resident { record, blocks: true, ..Default::default() });
            }
            let record = r.residents.get(keys[4]).unwrap().record;
            r.publishing.insert(keys[4], EditPublication { record, previous: None,
                next: None, evicted: false, initial_bucket: Some(0) });
            r.refresh_camera_pending(&[block], None);
            let expected: Vec<_> = if kind == 0 || kind == 2 {
                keys[5..].iter().map(|&key| (0, key)).collect()
            } else { Vec::new() };
            assert_eq!(r.visible_admission.iter().copied().collect::<Vec<_>>(), expected, "case {kind}");
            assert!(!r.publishing[&keys[4]].evicted);
            if let LeaseOrigin::Captured(retained) = r.visible_leases[&block].origin {
                assert_eq!((retained.frame, retained.view, retained.at), (source.frame, source.view, source.at));
            }
        }
    }

    #[test]
    fn camera_refresh_completed_tile_preserves_lease_edits_and_captured_fifo() {
        let (_, mut r, _, _) = current_bridge_fixture();
        let eye = r.current_request.as_ref().unwrap().eye;
        r.set_ground_clearance(10.0);
        let block = r.camera_blocks(eye, &|| false)[0];
        r.refresh_camera_pending(&[block], None);
        let (keys, count) = r.visible_columns(block).unwrap();
        assert_eq!(count, 16);
        for &key in &keys {
            let record = r.alloc_record().unwrap();
            assert!(r.acquire_blocks(key, &mut FrameWork::default()));
            r.residents.insert(key, Resident { record, blocks: true, ..Default::default() });
        }
        let previous = r.edits.alloc(2, r.capacity.edit_words).unwrap();
        let next = r.edits.alloc(2, r.capacity.edit_words).unwrap();
        let record = r.residents.get(keys[0]).unwrap().record;
        r.residents.get_mut(keys[0]).unwrap().edit_block = Some(previous);
        r.publishing.insert(keys[0], EditPublication { record, previous: Some(previous),
            next: Some(next), evicted: false, initial_bucket: None });
        r.urgent.push(keys[0]);
        let (face, level, i, j) = unpack(block);
        let captured = pack(key0(face, level, i + 4), j as u32);
        r.initial_retries.insert(captured);
        assert!(!Residency::tile_has_initial_retry(&r.initial_retries, face, level, i >> 2, j >> 2));
        assert!(Residency::tile_has_initial_retry(&r.initial_retries, face, level, (i + 4) >> 2, j >> 2));
        let source = r.visible_view.unwrap();
        r.visible_leases.insert(captured, VisibleLease { origin: LeaseOrigin::Captured(source),
            serial: r.requested, retiring: false, retired: 0,
            current_demand_frame: Some(r.frame) });
        let (captured_keys, _) = r.visible_columns(captured).unwrap();
        r.visible_admission.extend(captured_keys.map(|key| (level as usize, key)));
        let before = r.levels[level as usize].pending.at.clone();
        r.frame += 1;
        r.refresh_camera_pending(&[block], None);
        assert_eq!(r.levels[level as usize].pending.at, before,
            "completed resident tiles do not repeat per-column pending work");
        assert_eq!(r.visible_admission.iter().copied().collect::<Vec<_>>(),
            captured_keys.map(|key| (level as usize, key)),
            "completed Camera ranks clear without discarding Captured demand");
        assert_eq!(r.visible_leases[&block].current_demand_frame, Some(r.frame));
        let LeaseOrigin::Captured(retained) = r.visible_leases[&captured].origin else { panic!("captured clock replaced") };
        assert_eq!((retained.frame, retained.view, retained.at), (source.frame, source.view, source.at));
        assert_eq!(r.residents.get(keys[0]).unwrap().edit_block, Some(previous));
        assert_eq!(r.publishing[&keys[0]].next, Some(next));
        assert_eq!(r.urgent, vec![keys[0]]);
        assert!(!r.edits.free[previous.1 as usize].contains(&previous.0));
        assert!(!r.edits.free[next.1 as usize].contains(&next.0));
    }

    #[test]
    fn near_refresh_complete_tile_ignores_unrelated_initial_retry() {
        let planet = Planet::new(PlanetRecipe { shape: crate::grid::Shape::Plane, ..Default::default() }).unwrap();
        let grid = *planet.grid();
        let eye = DVec3::Y * 2.0;
        let (cell, _) = grid.locate(eye);
        let i = (cell.i >> 3) & !3;
        let j = (cell.j >> 3) & !3;
        let block = pack(key0(cell.face, 0, i), j as u32);
        let mut r = Residency::new(grid, Capacity { table_bits: 8, ..Default::default() });
        let (keys, count) = r.visible_columns(block).unwrap();
        assert_eq!(count, 16);
        for &key in &keys {
            let record = r.alloc_record().unwrap();
            assert!(r.acquire_blocks(key, &mut FrameWork::default()));
            r.residents.insert(key, Resident { record, blocks: true, ..Default::default() });
            r.levels[0].pending.insert(key, 40);
        }
        r.levels[0].active = true;
        r.levels[0].radius = 100.0;
        r.levels[0].wanted = Some(std::sync::Arc::new(keys.into_iter().collect()));
        let unrelated = pack(key0(cell.face, 0, i + 8), j as u32);
        r.initial_retries.insert(unrelated);
        let before = r.levels[0].pending.at.clone();
        r.refresh_near_pending(eye, None);
        assert_eq!(r.levels[0].pending.at, before,
            "an unrelated failed tile must not reprioritize a complete resident tile");
        r.initial_retries.insert(keys[2]);
        r.refresh_near_pending(eye, None);
        assert_ne!(r.levels[0].pending.at[&keys[2]].0, 40,
            "a full tile containing a failed initial publication must still refresh its retry");
        assert!(r.initial_retries.contains(&keys[2]) && r.initial_retries.contains(&unrelated));
        assert_eq!(r.residents.len(), 16);
        assert!(r.publishing.is_empty());
    }

    #[test]
    fn camera_refresh_partial_and_complete_retry_keep_exact_issuable_fifo() {
        for resident_count in [9, 16] {
            let (_, mut r, _, _) = current_bridge_fixture();
            let eye = r.current_request.as_ref().unwrap().eye;
            r.set_ground_clearance(10.0);
            let block = r.camera_blocks(eye, &|| false)[0];
            let source = r.visible_view.unwrap();
            r.visible_leases.insert(block, VisibleLease { origin: LeaseOrigin::Captured(source),
                serial: r.requested, retiring: false, retired: 0,
                current_demand_frame: Some(r.frame) });
            r.refresh_camera_pending(&[block], None);
            let (keys, _) = r.visible_columns(block).unwrap();
            for &key in &keys[..resident_count] {
                let record = r.alloc_record().unwrap();
                assert!(r.acquire_blocks(key, &mut FrameWork::default()));
                r.residents.insert(key, Resident { record, blocks: true, ..Default::default() });
            }
            let record = r.residents.get(keys[8]).unwrap().record;
            r.publishing.insert(keys[8], EditPublication { record, previous: None,
                next: None, evicted: false, initial_bucket: Some(0) });
            r.initial_retries.insert(keys[2]);
            r.refresh_camera_pending(&[block], None);
            let expected: Vec<_> = keys.iter().enumerate()
                .filter(|&(index, _)| index == 2 || index >= resident_count)
                .map(|(_, &key)| (0, key)).collect();
            assert_eq!(r.visible_admission.iter().copied().collect::<Vec<_>>(), expected,
                "tile replacement retains every missing column and retry exactly once");
            assert!(r.initial_retries.contains(&keys[2]));
            assert!(!r.publishing[&keys[8]].evicted);
            assert!(matches!(r.visible_leases[&block].origin, LeaseOrigin::Captured(_)));
        }
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
    fn camera_arrival_descends_before_worker_activation_without_reviving_capture() {
        let (planet, mut r, _, at) = current_bridge_fixture();
        let eye = r.current_request.as_ref().unwrap().eye;
        r.set_ground_clearance(10.0);
        r.levels[0].active = false;
        let current = r.camera_blocks(eye, &|| false);
        assert!(current.iter().any(|&key| unpack(key).1 == 0));
        let first = current[0];
        assert!(r.current_camera_block(first) && !r.current_captured_block(first));
        r.prioritize_visible_blocks_from([(first as u32, (first >> 32) as u32)], 10, 7, at);
        r.refresh_visible_pending(None);
        assert!(r.visible_leases.is_empty() && r.levels[0].pending.is_empty(),
            "inactive captured feedback remains rejected");
        let work = r.plan(&planet, eye, 160.0, 16);
        assert_eq!(work.jobs.len(), 16);
        assert!(work.job_keys.iter().all(|&key| unpack(key).1 == 0 && r.transient_wanted(key)));
        assert!(!r.levels[0].active && r.levels[0].wanted.as_ref().unwrap().is_empty(),
            "Camera leases do not forge worker activation or authority");
        assert_eq!(r.stats.camera_jobs, [16, 0, 0]);
        r.complete_jobs(work.job_keys.iter().map(|&key| (key, 0)));
        let mut inactive = snapshot_update(3, &[]);
        inactive.levels[0].active = false;
        r.apply(inactive);
        let waiting = r.plan(&planet, eye, 160.0, 0);
        assert!(waiting.evictions.is_empty());
        assert!(work.job_keys.iter().all(|&key| r.residents.contains_key(key) && r.transient_wanted(key)),
            "a delayed inactive acknowledgment cannot evict current Camera terrain");
        table_is_exact(&r);
        let high = eye + DVec3::Y * (planet.outer_radius() + 1_000.0);
        r.current_request.as_mut().unwrap().eye = high;
        r.set_ground_clearance(1_000.0);
        assert!(r.camera_base_level() > 0);
        assert!(r.camera_blocks(high, &|| false).iter().all(|&key| unpack(key).1 > 0));
        r.frame += 1;
        let mut retired = FrameWork::default();
        r.retire_visible_leases(&mut retired, &|| false);
        assert!(retired.evictions.is_empty());
        assert!(work.job_keys.iter().all(|&key| r.residents.contains_key(key) && !r.transient_wanted(key)),
            "departed Camera authorization does not discard exact records");
        assert!(!r.visible_admission.iter().any(|&(_, key)| work.job_keys.contains(&key)));
        for &key in &work.job_keys { r.evict(key, &mut retired); }
        assert_eq!(retired.evictions.len(), work.jobs.len());
        assert!(work.job_keys.iter().all(|&key| !r.residents.contains_key(key)));
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
    fn camera_refresh_finished_fifo_tail_drops_full_tiles_but_keeps_exact_retries() {
        let (planet, mut r, _, at) = current_bridge_fixture();
        let eye = r.current_request.as_ref().unwrap().eye;
        r.set_ground_clearance(10.0);
        let blocks = r.camera_blocks(eye, &|| false)[..3].to_vec();
        r.refresh_camera_pending(&blocks, None);
        let mut work = FrameWork::default();
        for &block in &blocks {
            let level = unpack(block).1 as usize;
            assert_eq!(r.visible_admission.pop_front(), Some((level, block)));
            r.levels[level].pending.remove(block);
            assert!(r.admit_visible_block(&planet, level, block, 0, 48, None, &mut work));
        }
        assert_eq!(work.job_keys.len(), 48);
        r.complete_jobs(work.job_keys.iter().enumerate().map(|(index, &key)| (key, if index < 32 { 0 } else { 3 })));
        assert_eq!(r.initial_retries.len(), 16);
        let source = VisibleStamp { frame: 10, view: 7, at };
        r.visible_leases.get_mut(&blocks[2]).unwrap().origin = LeaseOrigin::Captured(source);
        r.visible_admission = work.job_keys.iter().map(|&key| (unpack(key).1 as usize, key)).collect();
        // Intake expires before any tile is reached. Its existing FIFO still
        // proves full tiles by exact refs, independently of per-column probes.
        let mut checks = 0;
        r.refresh_camera_pending_until(&blocks, || { checks += 1; checks > 1 });
        assert_eq!(r.visible_admission.iter().map(|entry| entry.1).collect::<Vec<_>>(), work.job_keys[32..]);
        assert_eq!(r.initial_retries.len(), 16, "finished filtering never consumes a failed record retry");
        r.refresh_camera_pending_until(&blocks, || false);
        assert_eq!(r.visible_admission.len(), 16);
        let retained = r.visible_leases[&blocks[2]].captured_source();
        assert_eq!((retained.frame, retained.view, retained.at), (source.frame, source.view, source.at));
        table_is_exact(&r);
    }

    #[test]
    fn camera_arrival_partial_refresh_preserves_current_fifo_and_resumes_whole_tile() {
        let (planet, mut r, _, _) = current_bridge_fixture();
        let eye = r.current_request.as_ref().unwrap().eye;
        r.set_ground_clearance(10.0);
        let blocks = r.camera_blocks(eye, &|| false);
        let selected = [blocks[1], blocks[0]];
        r.refresh_camera_pending(&selected[1..], None);
        let before = r.visible_admission.clone();
        let (new, count) = r.visible_columns(selected[0]).unwrap();
        assert_eq!(count, 16);
        let mut checks = 0;
        r.refresh_camera_pending_until(&selected, || { checks += 1; checks > 7 });
        assert_eq!(r.visible_admission, before, "interrupted staging preserves already queued current tiles");
        assert!(new.iter().all(|key| !r.levels[0].pending.at.contains_key(key)),
            "a partial tile cannot consume pending/FIFO priority");
        assert!(r.transient_wanted(selected[0]), "selected camera authorization stays valid for resumption");
        r.refresh_camera_pending_until(&selected, || false);
        assert_eq!(r.visible_admission.len(), 32);
        assert_eq!(r.visible_admission.iter().take(16).copied().collect::<Vec<_>>(), new.map(|key| (0, key)));
        let work = r.plan(&planet, eye, 160.0, 32);
        assert_eq!(work.jobs.len(), 32, "resumed current demand reaches normal admission");
        assert_eq!(r.publishing.len(), 32);
        table_is_exact(&r);
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
        assert!(retired.evictions.is_empty() && r.visible_leases.is_empty());
        assert!(work.job_keys.iter().all(|key| r.residents.contains_key(*key) && !r.publishing[key].evicted),
            "lease expiry retains exact records and edit publication ownership");
        for &key in &work.job_keys { r.evict(key, &mut retired); }
        assert_eq!(retired.evictions.len(), work.jobs.len());
        assert!(work.job_keys.iter().all(|key| r.publishing[key].evicted));
        assert!(work.job_keys.iter().all(|key| r.publishing.contains_key(key)),
            "normal eviction quarantines in-flight ownership until acknowledgment");
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
        assert!(r.visible_leases.values().all(|lease| matches!(lease.origin, LeaseOrigin::Captured(_))),
            "all expired Camera authorizations release their bounded capacity");
        let captured_left = r.visible_leases.values().filter(|lease| matches!(lease.origin, LeaseOrigin::Captured(_))).count();
        assert_eq!(captured_left, VISIBLE_BLOCKS - 8);
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
            let width = r.grid.level_size(level) * f64::from(BRICK * 4);
            (point - eye).dot(view.forward) + width * std::f64::consts::FRAC_1_SQRT_2
                + view.forward.dot(r.grid.up(eye)).max(0.0) * 50.0 >= 0.0
        }), "unrelated behind-eye tiles stay outside the swept entry corridor");
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
        let high_base = r.camera_base_level();
        assert!(high_base > 0);
        assert!(r.camera_blocks(high, &|| false).iter().all(|&key| unpack(key).1 as usize >= high_base),
            "high-altitude demand uses adaptive bands rather than near-ground ghosts");
        r.current_request.as_mut().unwrap().eye = eye;
        r.set_ground_clearance(50.0);
        r.levels[0].active = false;
        let current = r.camera_blocks(eye, &|| false);
        assert!(!current.is_empty(), "current geometry can precede worker activation");
        assert!(current.iter().all(|&key| !r.current_captured_block(key)),
            "captured feedback still rejects inactive levels");
    }

    #[test]
    fn camera_arrival_selection_departure_releases_lease_without_expanding_cap() {
        let (_, mut r, _, _) = current_bridge_fixture();
        let eye = r.current_request.as_ref().unwrap().eye;
        r.set_ground_clearance(10.0);
        let current = r.camera_blocks(eye, &|| false);
        r.refresh_camera_pending(&current, None);
        // Entry air columns stay needed after a turn. Use a forward ground
        // tile beyond that corridor to exercise real selection departure.
        let old = *current.iter().find(|&&key| {
            let (face, level, i, j) = unpack(key);
            level == 0 && r.grid.ground_point(face, f64::from((i + 2) * BRICK),
                f64::from((j + 2) * BRICK)).x - eye.x > 30.0
        }).expect("forward ground tile outside the entry corridor");
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
            let (mut planet, mut r, _, at) = current_bridge_fixture();
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
            assert!(current.len() <= CAMERA_CANDIDATES);
            assert!(current.iter().all(|&key| unpack(key).1 >= r.camera_base_level() as u32));
            // Current ground bands do not prove that a nearby raised ridge
            // cannot need finer data. Actual captured hits retain that path.
            assert!(r.current_captured_block(block));
            r.prioritize_visible_blocks_from([(block as u32, (block >> 32) as u32)], 10, 7, at);
            r.refresh_visible_pending(None);
            assert!(r.transient_wanted(block));
            assert!(matches!(r.visible_leases[&block].origin, LeaseOrigin::Captured(_)));
            assert!(r.levels[0].pending.len() >= 16);
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
        assert_eq!(r.visible_leases.len(), 1, "old Camera authorization is removed without a column retirement walk");
        assert!(matches!(r.visible_leases[&tail].origin, LeaseOrigin::Camera));
    }

    fn camera_ground_band_fixture(height: f64) -> (std::sync::Arc<Planet>, Residency, DVec3) {
        // Keep three projected bands available throughout the native 1km descent.
        let planet = std::sync::Arc::new(Planet::new(PlanetRecipe {
            shape: crate::grid::Shape::Plane, plane_size_m: 32_768.0, ..Default::default()
        }).unwrap());
        let mut r = Residency::new(*planet.grid(), Capacity { table_bits: 10, ..Default::default() });
        let eye = DVec3::new(0.0, height, 0.0);
        r.current_request = Some(WindowRequest { eye, prefetch_eye: None, priority_eye: None,
            view_focus: None, lod0: 88.0, lod_dither: 0.25, outer_radius: planet.outer_radius(),
            planet: Some(planet.clone()), serial: 3 });
        r.set_lod_dither(0.25);
        r.snapshot_mode = true;
        r.requested = 3;
        r.applied = 1;
        r.set_ground_clearance(height);
        let pitch = -9.6f64.to_radians();
        let tan_y = (std::f64::consts::PI / 8.0).tan();
        r.set_camera_view(DVec3::new(pitch.cos(), pitch.sin(), 0.0), DVec3::Y,
            [tan_y * 1196.0 / 729.0, tan_y]);
        (planet, r, eye)
    }

    #[test]
    fn camera_entry_corridor_contains_native_air_ray_tiles_before_ground() {
        let (_, r, eye) = camera_ground_band_fixture(9.6);
        let blocks = r.camera_blocks(eye, &|| false);
        let view = r.camera_view.unwrap();
        let bottom = view.forward - view.up * view.tan_half[1];
        let first_ground = 9.6 / -bottom.y;
        assert!((14.0..20.0).contains(&(first_ground * bottom.x)));
        let tile = |point: DVec3| {
            let (cell, _) = r.grid.locate(point);
            pack(key0(cell.face, 0, (cell.i >> 3) & !3), ((cell.j >> 3) & !3) as u32)
        };
        let origin = tile(eye);
        assert!(blocks.contains(&origin), "the fast trace's initial fine column is a dependency even when its ground proxy is offscreen");
        assert!(blocks.iter().position(|&key| key == origin).unwrap() < 32);
        for ix in 0..9 {
            for iy in 0..5 {
                let (x, y) = (f64::from(ix) / 4.0 - 1.0, f64::from(iy) / 2.0 - 1.0);
                let ray = view.forward + view.right * (x * view.tan_half[0]) + view.up * (y * view.tan_half[1]);
                for step in 0..32 {
                    let fraction = f64::from(step) / 32.0;
                    let point = eye + ray * (first_ground * fraction);
                    assert!(point.y > 0.0);
                    let key = tile(point);
                    assert!(blocks.contains(&key), "missing entry air column x{x}/y{y}/fraction{fraction}");
                    assert!(r.current_camera_block(key));
                }
            }
        }
        assert!(!blocks.contains(&tile(eye - DVec3::X * 20.0)), "unrelated backward columns remain excluded");
        assert!(blocks.len() <= CAMERA_CANDIDATES);
    }

    #[test]
    fn camera_ground_bands_cover_native_center_and_required_lateral_tiles() {
        let (_, r, eye) = camera_ground_band_fixture(9.6);
        let blocks = r.camera_blocks(eye, &|| false);
        assert!(blocks.len() <= CAMERA_CANDIDATES && CAMERA_LEASES == 384);
        let view = r.camera_view.unwrap();
        let mut per_band = [0usize; 3];
        for &key in &blocks { per_band[unpack(key).1 as usize] += 1; }
        assert!(per_band.iter().all(|&count| count > 64 && count <= 168));
        for (ahead, lateral) in [(25.0, 10.0), (40.0, 14.0), (57.0, 0.0),
            (57.0, 30.0), (110.0, 60.0), (150.0, 80.0)] {
            let point = DVec3::new(ahead, 0.0, lateral);
            let relative = point - eye;
            let depth = relative.dot(view.forward);
            assert!(depth > 0.0 && relative.dot(view.up).abs() < depth * view.tan_half[1]
                && relative.dot(view.right).abs() < depth * view.tan_half[0]);
            let level = (0..3).find(|&l| relative.length() <= 44.0 * f64::from(1u32 << l)).unwrap();
            assert!(88.0 * f64::from(1u32 << (level + 1)) / relative.length() > 4.0);
            let (cell, _) = r.grid.locate(point);
            let key = pack(key0(cell.face, level, (cell.i >> (3 + level)) & !3),
                ((cell.j >> (3 + level)) & !3) as u32);
            assert!(blocks.contains(&key), "missing actual ground tile at {ahead}/{lateral}, level{level}");
        }
        assert!(r.camera_blocks(eye, &|| true).is_empty());
    }

    #[test]
    fn camera_ground_bands_follow_high_descent_without_fine_ghosts() {
        for height in [1000.0, 390.0, 100.0, 50.0, 9.6] {
            let (planet, mut r, eye) = camera_ground_band_fixture(height);
            let base = r.camera_base_level();
            let blocks = r.camera_blocks(eye, &|| false);
            assert!(!blocks.is_empty() && blocks.len() <= CAMERA_CANDIDATES);
            assert!(blocks.iter().all(|&key| (base..base + 3).contains(&(unpack(key).1 as usize))));
            let far = 44.0 * f64::from(1u32 << base);
            assert!(height <= far && (base == 0 || height > far * 0.5));
            r.refresh_camera_pending(&blocks, None);
            assert!(r.visible_leases.len() <= CAMERA_LEASES);
            assert!(r.visible_admission.iter().any(|&(level, _)| (base..base + 3).contains(&level)));
            // Selected identities and stats use absolute levels safely during
            // descent; a source snapshot need not have activated them yet.
            r.last_request = r.current_request.clone();
            let work = r.plan(&planet, eye, 88.0, 16);
            assert_eq!(r.stats.camera_base_level, base as u32);
            assert_eq!(r.stats.camera_jobs.iter().sum::<u32>(), work.jobs.len() as u32);
            assert!(work.job_keys.iter().all(|&key| (base..base + 3).contains(&(unpack(key).1 as usize))));
        }
    }

    #[test]
    fn camera_ground_bands_cross_cube_face_and_stay_in_world_bounds() {
        let planet = std::sync::Arc::new(Planet::new(PlanetRecipe { radius_m: 1000.0, ..Default::default() }).unwrap());
        let mut r = Residency::new(*planet.grid(), Capacity { table_bits: 10, ..Default::default() });
        // The seam is 20m ahead, between the lower-edge and centre ground hits.
        let ground = DVec3::new(1.0, 0.96, 0.0).normalize() * 1000.0;
        let up = ground.normalize();
        let tangent = DVec3::new(-up.y, up.x, 0.0);
        let eye = ground + up * 9.6;
        r.current_request = Some(WindowRequest { eye, prefetch_eye: None, priority_eye: None,
            view_focus: None, lod0: 88.0, lod_dither: 0.25, outer_radius: planet.outer_radius(),
            planet: Some(planet), serial: 3 });
        r.set_ground_clearance(9.6);
        r.set_camera_view(tangent - up * 0.17, up, [0.68, 0.414]);
        let blocks = r.camera_blocks(eye, &|| false);
        assert!(!blocks.is_empty() && blocks.len() <= CAMERA_CANDIDATES);
        assert!(blocks.iter().any(|&key| unpack(key).0 == 0));
        assert!(blocks.iter().any(|&key| unpack(key).0 == 2));
        assert!(blocks.iter().all(|&key| r.visible_columns(key).is_some()));
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
