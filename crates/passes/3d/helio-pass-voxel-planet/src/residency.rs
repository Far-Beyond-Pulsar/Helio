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
use std::sync::mpsc;
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
            // 8M slots: probe runs stay far below the GPU's 64-slot limit at
            // the record cap (at 4M slots and 2.5M+ columns they exceeded it,
            // hiding columns from the traversal).
            table_bits: 23,
            records: 3_000_000,
            // 512 MB. A 2560x1440 ground view used 99% of 4M units without
            // caves (1.65M columns; coarse relief columns take 3 units), and
            // cave walls are mixed bricks.
            pool_units: 8 << 20,
            // Every band brick of a frame's jobs before compaction: a cave
            // column holds ~150 bricks at level 0 (a heightfield column 2-4).
            scratch_units: 1 << 20,
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
    /// Queue `key` in `bucket`; false if already queued.
    fn insert(&mut self, key: u64, bucket: usize) -> bool {
        if self.at.contains_key(&key) {
            return false;
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
    /// Move a queued column to `bucket` (no-op if it is not queued).
    fn set_bucket(&mut self, key: u64, bucket: usize) {
        if self.at.get(&key).is_some_and(|(b, _)| usize::from(*b) != bucket) {
            self.remove(key);
            self.insert(key, bucket);
        }
    }
    /// Every queued column, nearest bucket first.
    fn snapshot(&self) -> Vec<u64> {
        self.buckets.iter().flatten().copied().collect()
    }
}

#[derive(Default)]
struct Level {
    active: bool,
    /// Window centre and angular radius of the last applied diff.
    center: DVec3,
    /// Ground radius of the applied window (metres).
    radius: f64,
    /// Wanted but not yet issued columns.
    pending: PendingQueue,
    /// Eye ground point the pending priorities are ranked against, and the
    /// queued columns still to be re-ranked against it.
    ranked_at: Option<DVec3>,
    rerank: Vec<u64>,
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
    pub evictions: Vec<u32>,
    /// Column table patches `(slot, record)`. The GPU applies them in
    /// parallel, so `plan` returns each slot once with its final value
    /// (backward-shift deletion rewrites a slot several times per frame; an
    /// earlier value winning left an empty slot inside a probe run).
    pub table_writes: Vec<(u32, u32)>,
    pub edit_writes: Vec<(u32, Vec<u32>)>,
    pub brush_writes: Vec<(u32, FaceBrush)>,
    /// Summary block table writes `(slot, bi, bj)`, each slot once with its
    /// final state; `bi = -1` releases a slot.
    pub block_inits: Vec<(u32, i32, i32)>,
}

impl FrameWork {
    /// Reduce the patch lists to one final write per slot (see the fields).
    fn finish(&mut self, table: &[u32]) {
        let mut sent = rustc_hash::FxHashSet::default();
        self.table_writes.retain(|(slot, _)| sent.insert(*slot));
        for (slot, value) in &mut self.table_writes {
            *value = table[*slot as usize];
        }
        let mut last = FxHashMap::default();
        for (index, (slot, _, _)) in self.block_inits.iter().enumerate() {
            last.insert(*slot, index);
        }
        let mut index = 0;
        self.block_inits.retain(|(slot, _, _)| {
            index += 1;
            last[slot] == index - 1
        });
    }
}

/// What [`Residency::fallback_distances`] needs, detached from the residency
/// so the render thread can evaluate it for any eye while the residency
/// worker plans the next frame.
#[derive(Clone)]
pub struct Coverage {
    grid: Grid,
    levels: Vec<LevelCoverage>,
}

#[derive(Clone, Default)]
struct LevelCoverage {
    /// No guaranteed coverage (inactive, catching up, urgent regenerations
    /// or too many pending columns to list).
    none: bool,
    center: DVec3,
    /// Window radius less 1.5 columns (metres).
    reach: f64,
    /// Column ground width (metres).
    col: f64,
    /// Ground points of the pending columns.
    pending: Vec<DVec3>,
}

impl Coverage {
    /// No level has guaranteed coverage (before the first plan).
    pub fn none(grid: Grid) -> Self {
        let level = LevelCoverage { none: true, ..Default::default() };
        Self { grid, levels: vec![level; grid.levels() as usize] }
    }

    /// See [`Residency::fallback_distances`].
    pub fn fallback_distances(&self, eye: DVec3) -> Vec<f64> {
        let grid = self.grid;
        let ground = if grid.is_plane() { DVec3::new(eye.x, 0.0, eye.z) } else { eye.normalize() };
        self.levels
            .iter()
            .map(|l| {
                if l.none {
                    return 0.0;
                }
                let mut distance = l.reach - grid.ground_distance(l.center, ground);
                for p in &l.pending {
                    // Traversal uses only complete 4x4-column blocks while a
                    // level streams in: a pending column makes its whole
                    // block (within its diagonal, 5.7 columns) fall back.
                    distance = distance.min(grid.ground_distance(*p, eye) - l.col * 6.0);
                }
                distance.max(0.0)
            })
            .collect()
    }
}

/// Ground point of a column's centre.
fn column_ground(grid: &Grid, key: u64) -> DVec3 {
    let (face, level, ci, cj) = unpack(key);
    let size = f64::from(BRICK << level);
    grid.ground_point(face, (f64::from(ci) + 0.5) * size, (f64::from(cj) + 0.5) * size)
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
    /// Columns skipped because the GPU lookup could not reach their slot.
    pub table_refused: usize,
    /// Pending columns re-ranked against a moved eye (cumulative).
    pub reranked: usize,
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
    /// Traversal level-transition dither (see `set_lod_dither`).
    lod_dither: f64,
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
            lod_dither: 0.25,
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
        }
    }

    /// CPU time `plan` may spend per frame applying window diffs and
    /// admitting columns (`None`: unbounded). The rest carries over.
    pub fn set_cpu_budget(&mut self, budget: Option<std::time::Duration>) {
        self.cpu_budget = budget;
    }

    /// Width of the traversal's stochastic level transition; windows cover
    /// the whole band in which a level can be selected.
    pub fn set_lod_dither(&mut self, dither: f64) {
        self.lod_dither = dither;
    }

    /// Queue a window diff; [`Self::apply_queued`] applies it in order.
    fn apply(&mut self, update: WindowUpdate) {
        for diff in update.levels {
            let level = diff.level as usize;
            // The window metadata changes at once; the level is marked as
            // catching up (no guaranteed coverage) until its ops are done.
            let l = &mut self.levels[level];
            l.active = diff.active;
            l.center = diff.center;
            l.radius = diff.radius;
            // The planner ranks adds by distance from the window centre (the
            // eye's ground point when it planned).
            if l.pending.is_empty() && l.rerank.is_empty() {
                l.ranked_at = Some(diff.center);
            }
            self.catching_up[level] += 1;
            self.diffs.push_back(QueuedDiff { diff, removed: 0, cleared: false, added: 0 });
        }
        self.stats.window_rebuild_ms = update.planning_ms;
        self.applied = update.serial;
    }

    /// Apply queued window diffs in order until done or out of time.
    fn apply_queued(&mut self, work: &mut FrameWork, out_of_time: &impl Fn() -> bool) {
        const CHUNK: usize = 256;
        while let Some(mut queued) = self.diffs.pop_front() {
            let level = queued.diff.level as usize;
            while queued.removed < queued.diff.removes.len() {
                let end = (queued.removed + CHUNK).min(queued.diff.removes.len());
                for i in queued.removed..end {
                    let key = queued.diff.removes[i];
                    self.levels[level].pending.remove(key);
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
                let l = &mut self.levels[level];
                l.pending.clear();
                l.rerank.clear();
                l.ranked_at = None;
                queued.cleared = true;
            }
            while queued.added < queued.diff.adds.len() {
                let end = (queued.added + CHUNK).min(queued.diff.adds.len());
                for i in queued.added..end {
                    let (priority, key) = queued.diff.adds[i];
                    if !self.residents.contains_key(key) {
                        self.levels[level].pending.insert(key, PendingQueue::bucket(priority));
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

    /// Re-rank pending columns against the current eye. Priorities are set
    /// when a window is planned; once admission lags (fast flight at high
    /// resolution) the eye moves on and columns it now flies over still wait
    /// behind ones it has left. When the eye has moved 1/16 of a level's
    /// radius from where its queue was ranked, every queued column is
    /// re-bucketed by its distance from the eye, a bounded number per frame
    /// (farthest-ranked first: the leading edge of the window).
    fn rerank(&mut self, eye: DVec3, out_of_time: &impl Fn() -> bool) {
        const PER_FRAME: usize = 32_768;
        let grid = self.grid;
        let ground = if grid.is_plane() { DVec3::new(eye.x, 0.0, eye.z) } else { eye.normalize() };
        let mut done = 0;
        for l in &mut self.levels {
            if !l.active || l.radius <= 0.0 {
                continue;
            }
            // Normalized like the planner's priorities (the angular radius
            // is capped at pi on a sphere).
            let radius = if grid.is_plane() { l.radius } else { l.radius.min(std::f64::consts::PI * grid.radius()) };
            if l.rerank.is_empty() {
                let drift = l.ranked_at.map_or(f64::INFINITY, |at| grid.ground_distance(at, ground));
                if l.pending.len() < 64 || drift < radius / 16.0 {
                    continue;
                }
                l.rerank = l.pending.snapshot();
                l.ranked_at = Some(ground);
            }
            let ranked_at = l.ranked_at.unwrap_or(ground);
            while let Some(key) = l.rerank.pop() {
                let distance = grid.ground_distance(column_ground(&grid, key), ranked_at);
                l.pending.set_bucket(key, PendingQueue::bucket((distance / radius) as f32));
                done += 1;
                if done % 256 == 0 && (done >= PER_FRAME || out_of_time()) {
                    self.stats.reranked += done;
                    return;
                }
            }
        }
        self.stats.reranked += done;
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
            lod0,
            lod_dither: self.lod_dither,
            outer_radius: planet.outer_radius(),
            planet: Some(planet.clone()),
            serial: self.requested + 1,
        };
        let changed = self.last_request.as_ref().is_none_or(|last| {
            last.eye.distance(eye) > self.grid.voxel_size() * 2.0
                || (last.lod0 - lod0).abs() > lod0 * 0.01
                || last.lod_dither != request.lod_dither
                || last.outer_radius != request.outer_radius
        });
        // Coalesce: no new plan while the last one is outstanding or its
        // diffs are still being applied. The planner diffs against the last
        // window it sent, so the next diff spans all motion since then. A
        // request per moved frame queued every intermediate window instead;
        // at speed (and at high resolution, with ~4x larger diffs) the queue
        // grew without bound (1000+ level diffs, evictions and admission
        // lagging further every frame: refinement stopped).
        let ready = self.applied == self.requested && self.diffs.len() < self.levels.len();
        if changed && ready {
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
        self.rerank(eye, &apply_out_of_time);
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
        }
        self.urgent = deferred_urgent;
        // Merge pending windows by normalized distance; the coarsest level
        // (global coverage) always goes first.
        let top_level = self.grid.levels() - 1;
        // Admission costs ~2 us of CPU per column (edit query, summary
        // blocks, table): bounded by time as well as by the GPU budget, and
        // resumes next frame.
        let mut steps = 0u32;
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
            if self.residents.contains_key(key) {
                continue;
            }
            if !self.residents.can_insert(key) {
                // Unreachable for the GPU lookup: leave it to coarser levels.
                self.stats.table_refused += 1;
                continue;
            }
            let requeue = |this: &mut Self| {
                this.levels[index].pending.insert(key, bucket);
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
        work.finish(self.residents.table());
        work
    }

    /// Plan one background frame (see [`ResidencyWorker`]).
    fn plan_request(&mut self, request: PlanRequest) -> PlanResult {
        let started = std::time::Instant::now();
        self.requeue(request.failed);
        self.set_cpu_budget(Some(request.cpu_budget));
        self.set_lod_dither(request.lod_dither);
        let work = self.plan(&request.planet, request.eye, request.lod0, request.budget);
        PlanResult {
            work,
            stats: self.stats,
            coverage: self.coverage(),
            idle: self.idle(),
            blocks_exact: self.blocks_exact(),
            live_blocks: self.take_live_blocks().map(<[u32]>::to_vec),
            live_block_count: self.live_block_count(),
            queued_diffs: self.queued_diffs(),
            queued_ops: self.queued_ops(),
            queued_adds: self.queued_adds(),
            table: request.table.then(|| self.table().to_vec()),
            probe: request.probe.then(|| self.table_probe_stats(crate::column_index::MAX_PROBES)),
            plan_ms: started.elapsed().as_secs_f64() * 1000.0,
        }
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
    pub fn table_probe_stats(&self, limit: u32) -> (u32, usize) {
        self.residents.probe_stats(limit)
    }
    pub fn queued_diffs(&self) -> usize {
        self.diffs.len()
    }
    /// Window diff operations (removes and adds) not applied yet.
    pub fn queued_ops(&self) -> usize {
        self.diffs.iter().map(|q| q.diff.removes.len() - q.removed + q.diff.adds.len() - q.added).sum()
    }
    /// Columns queued diffs will add (wanted, not pending yet).
    pub fn queued_adds(&self) -> usize {
        self.diffs.iter().map(|q| q.diff.adds.len() - q.added).sum()
    }
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
        self.coverage().fallback_distances(eye)
    }

    /// The state [`Self::fallback_distances`] depends on.
    pub fn coverage(&self) -> Coverage {
        let grid = self.grid;
        let urgent: rustc_hash::FxHashSet<u32> = self.urgent.iter().map(|k| unpack(*k).1).collect();
        let levels = self
            .levels
            .iter()
            .enumerate()
            .map(|(level, l)| {
                if !l.active || urgent.contains(&(level as u32)) || self.catching_up[level] > 0 || l.pending.len() > 4096 {
                    return LevelCoverage { none: true, ..Default::default() };
                }
                // A column's ground width (the index-angle span on a sphere
                // bounds its true size).
                let col = grid.delta() * f64::from(BRICK << level) * if grid.is_plane() { 1.0 } else { grid.radius() };
                LevelCoverage {
                    none: false,
                    center: l.center,
                    reach: l.radius - col * 1.5,
                    col,
                    pending: l.pending.keys().map(|key| column_ground(&grid, *key)).collect(),
                }
            })
            .collect();
        Coverage { grid, levels }
    }

    pub fn idle(&self) -> bool {
        self.urgent.is_empty()
            && self.applied == self.requested
            && self.diffs.is_empty()
            && self.levels.iter().all(|l| l.pending.is_empty())
    }
}

/// Input of one background plan.
pub struct PlanRequest {
    pub planet: std::sync::Arc<Planet>,
    pub eye: DVec3,
    pub lod0: f64,
    /// Traversal level-transition dither (`Residency::set_lod_dither`).
    pub lod_dither: f64,
    /// Maximum column jobs.
    pub budget: usize,
    /// CPU time for diffs, re-ranking and admission.
    pub cpu_budget: std::time::Duration,
    /// Job outcomes read back since the last request (see [`Residency::requeue`]).
    pub failed: Vec<(u64, u32)>,
    /// Diagnostics: also return a copy of the column table.
    pub table: bool,
    /// Diagnostics: also scan the table's probe runs.
    pub probe: bool,
}

/// One plan's work and the residency state right after it, which is what
/// the GPU holds once `work` is uploaded.
pub struct PlanResult {
    pub work: FrameWork,
    pub stats: Stats,
    pub coverage: Coverage,
    pub idle: bool,
    pub blocks_exact: bool,
    /// Live tier-1 summary block slots, when they changed.
    pub live_blocks: Option<Vec<u32>>,
    pub live_block_count: usize,
    pub queued_diffs: usize,
    pub queued_ops: usize,
    pub queued_adds: usize,
    pub table: Option<Vec<u32>>,
    /// Longest probe run and entries beyond the GPU probe limit.
    pub probe: Option<(u32, usize)>,
    /// CPU time of the plan on the worker (ms).
    pub plan_ms: f64,
}

impl PlanResult {
    /// The state before the first plan: nothing resident, nothing covered.
    pub fn initial(grid: Grid) -> Self {
        Self {
            work: FrameWork::default(),
            stats: Stats::default(),
            coverage: Coverage::none(grid),
            idle: false,
            blocks_exact: true,
            live_blocks: None,
            live_block_count: 0,
            queued_diffs: 0,
            queued_ops: 0,
            queued_adds: 0,
            table: None,
            probe: None,
            plan_ms: 0.0,
        }
    }
}

/// Residency on its own thread. Admission costs ~2 us of CPU per column
/// (hash table, summary blocks, edit query: mostly cache misses), and fast
/// flight at high resolution wants 100k+ new columns per second; on the
/// render thread it took 1.5-4 ms of every frame and still lagged. The
/// render thread submits one request per frame and uploads each result in
/// the frame after: at most one plan is in flight, and results are uploaded
/// in order, exactly once, as `Residency::plan` would have been called.
pub struct ResidencyWorker {
    requests: Option<mpsc::Sender<PlanRequest>>,
    results: std::sync::Mutex<mpsc::Receiver<PlanResult>>,
    thread: Option<std::thread::JoinHandle<()>>,
    in_flight: bool,
}

impl ResidencyWorker {
    pub fn start(grid: Grid, capacity: Capacity) -> Self {
        let (request_tx, request_rx) = mpsc::channel::<PlanRequest>();
        let (result_tx, result_rx) = mpsc::channel();
        let thread = std::thread::Builder::new()
            .name("voxel-planet-residency".into())
            .spawn(move || {
                let mut residency = Residency::with_worker(grid, capacity);
                while let Ok(request) = request_rx.recv() {
                    if result_tx.send(residency.plan_request(request)).is_err() {
                        break;
                    }
                }
            })
            .expect("spawn residency worker");
        Self { requests: Some(request_tx), results: std::sync::Mutex::new(result_rx), thread: Some(thread), in_flight: false }
    }

    /// Start a plan; none may be in flight.
    pub fn submit(&mut self, request: PlanRequest) {
        debug_assert!(!self.in_flight, "one plan at a time");
        if let Some(tx) = &self.requests {
            self.in_flight = tx.send(request).is_ok();
        }
    }

    /// The in-flight plan's result, if it is done.
    pub fn try_take(&mut self) -> Option<PlanResult> {
        if !self.in_flight {
            return None;
        }
        let results = self.results.get_mut().unwrap_or_else(std::sync::PoisonError::into_inner);
        match results.try_recv() {
            Ok(result) => {
                self.in_flight = false;
                Some(result)
            }
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => {
                eprintln!("voxel planet residency worker stopped; terrain no longer streams");
                self.requests = None;
                self.in_flight = false;
                None
            }
        }
    }

    pub fn in_flight(&self) -> bool {
        self.in_flight
    }
}

impl Drop for ResidencyWorker {
    fn drop(&mut self) {
        self.requests = None;
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
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
        let planet = std::sync::Arc::new(Planet::new(PlanetRecipe::default()).unwrap());
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

    /// When admission lags and the eye moves on, the next column issued is
    /// one under the eye, not one near where the window was planned.
    #[test]
    fn pending_columns_are_reranked_against_the_moved_eye() {
        let planet = std::sync::Arc::new(Planet::new(PlanetRecipe::default()).unwrap());
        let grid = *planet.grid();
        let lod0 = Residency::lod_distance(&grid, (22.5f64).to_radians().tan(), 1080, 1.0);
        let ground = planet.surface_point(grid.direction(2, 3e7, 4e7), 1.8);
        let mut r = Residency::new(grid, Capacity::default());
        // Nothing is issued (budget 0): every wanted column stays pending.
        r.plan(&planet, ground, lod0, 0);
        let radius = r.levels[0].radius;
        assert!(r.levels[0].pending.len() > 10_000, "{}", r.levels[0].pending.len());
        let east = ground.normalize().any_orthonormal_vector();
        let eye = planet.surface_point(ground + east * radius * 0.4, 1.8);
        for _ in 0..64 {
            r.plan(&planet, eye, lod0, 0);
        }
        assert!(r.levels[0].rerank.is_empty() && r.stats.reranked > 0);
        let (key, _) = r.levels[0].pending.pop().unwrap();
        let distance = grid.ground_distance(column_ground(&grid, key), eye.normalize());
        assert!(distance < radius / 32.0, "next column {distance:.1} m from the eye (radius {radius:.1} m)");
    }

    #[test]
    fn table_lookup_matches_residents_after_moves() {
        let planet = std::sync::Arc::new(Planet::new(PlanetRecipe::default()).unwrap());
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
