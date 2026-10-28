//! The sun's occlusion pages (`helio_pass_shadow_matrix::sun_occlusion`):
//! which pages of the virtual clipmap are resident, which to trace next and
//! where each sits in the atlas, as Unreal's virtual shadow maps keep a
//! directional light's pages, planned on the CPU from the pages the frame's
//! pixels asked for (read back a frame or two later; meanwhile a reader
//! takes the nearest coarser level resident).
//!
//! An epoch is the sun the pages were traced for: their grid's axes (right
//! and down across the sun, forward from it) and anchor (planet-centred;
//! texels counted from it, so pages stay put as the eye moves and never
//! re-snap). When the sun has turned [`TURN`] from the current epoch's, a
//! new epoch starts and asks for its own pages; readers take the last
//! epoch's until the new one has every page asked for, then the last is
//! dropped. A page's texels are kept until the terrain under them changes
//! (`invalidate`: generated or dug chunks' boxes and moved volumes,
//! projected along the sun: traced again at once), the cut it was traced
//! through refines or was still settling (traced again in the background
//! once the world settles: a still view's breathing cut never churns), or
//! nothing asks for it and its physical page is needed.

use glam::DVec3;
use helio_pass_shadow_matrix::{
    sun_occlusion_entry, sun_occlusion_entry_physical, sun_occlusion_table_index,
    SunOcclusionEpoch, SunOcclusionFrame, SunOcclusionLevel, SUN_OCCLUSION_ATLAS_PAGES,
    SUN_OCCLUSION_LEVELS, SUN_OCCLUSION_PAGE, SUN_OCCLUSION_PHYSICAL, SUN_OCCLUSION_TABLE_WORDS,
    SUN_OCCLUSION_WINDOW,
};
use rustc_hash::{FxHashMap, FxHashSet};

pub(super) const LEVELS: usize = SUN_OCCLUSION_LEVELS;
const PAGE: i32 = SUN_OCCLUSION_PAGE as i32;
const PAGE_SHIFT: u32 = SUN_OCCLUSION_PAGE.trailing_zeros();
const WINDOW: i32 = SUN_OCCLUSION_WINDOW as i32;
/// Pages traced a frame at most: 48 of 64^2 texels, fewer rays than the one
/// 512^2 level a frame the clipmap cost before it was paged (pixels whose
/// pages are not there yet walk their own sun meanwhile).
pub(super) const TRACE_PAGES: usize = 48;
/// A trace view's side in pages (views are aligned to these blocks).
pub(super) const VIEW_PAGES: i32 = 8;
const VIEW_SHIFT: u32 = VIEW_PAGES.trailing_zeros();
/// Trace views a frame at most (each culls the drawn chunks and seeds its
/// rays: the per-view cost), and of those, views of background work (pages
/// next to fresh requests, pages of a coarser cut).
pub(super) const MAX_VIEWS: usize = 4;
const BACKGROUND_VIEWS: usize = 1;
/// Frames a page stays asked for after the last request naming it.
const ASKED_FRAMES: u64 = 30;
/// Frames since a page was last asked for before its physical page may go
/// to another.
const KEEP_FRAMES: u64 = 8;
/// The sun's turn (rad) that starts a new epoch once the last is done
/// (shadows that much off their direction, as a level of the old clipmap
/// was let lag), and the turn past which the last is dropped at once.
const TURN: f64 = 0.002;
const JUMP: f64 = 0.05;
/// Frames a new epoch may take to replace the last before it is dropped
/// anyway (pages the atlas cannot hold never complete it).
const HANDOVER_FRAMES: u64 = 120;
/// Levels whose pages are this wide (m) or wider keep the 2x2 pages nearest
/// the eye resident whether asked or not: what readers without requests
/// (GI hits, fog) fall back on around the eye.
const BASE_PAGE_M: f64 = 128.0;

/// Why a resident page is traced again: the terrain under it changed (at
/// once), or it was traced through a coarser or unsettled cut than now (in
/// the background, once the world has settled).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Stale {
    No,
    Coarse,
    Changed,
}

/// A page of a level: its coordinates in the level's pages from the
/// epoch's anchor.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(super) struct PageKey {
    pub level: u8,
    pub page: [i32; 2],
}

/// An epoch: the sun its pages were traced for (`sun`, toward it), their
/// grid's axes (right, down, forward from the sun) and anchor; its table
/// slot; its resident pages and the pages asked for (the last frame each
/// was asked in).
pub(super) struct Epoch {
    pub id: u64,
    pub sun: DVec3,
    pub axes: [DVec3; 3],
    pub anchor: DVec3,
    pub slot: usize,
    created: u64,
    resident: FxHashMap<PageKey, u16>,
    asked: FxHashMap<PageKey, u64>,
    /// The latest frame whose requests were read for it.
    heard: Option<u64>,
}

impl Epoch {
    fn new(id: u64, sun: DVec3, anchor: DVec3, slot: usize, frame: u64) -> Self {
        let forward = -sun;
        let hint = if forward.y.abs() < 0.9 {
            DVec3::Y
        } else {
            DVec3::X
        };
        let right = forward.cross(hint).normalize();
        let up = right.cross(forward);
        Self {
            id,
            sun,
            axes: [right, -up, forward],
            anchor,
            slot,
            created: frame,
            resident: FxHashMap::default(),
            asked: FxHashMap::default(),
            heard: None,
        }
    }

    /// Planet-centred `p` along right and down from the anchor (m).
    pub fn across(&self, p: DVec3) -> [f64; 2] {
        let d = p - self.anchor;
        [d.dot(self.axes[0]), d.dot(self.axes[1])]
    }

    /// The point of the plane through the anchor across the sun at `at`
    /// (m along right and down).
    pub fn point(&self, at: [f64; 2]) -> DVec3 {
        self.anchor + self.axes[0] * at[0] + self.axes[1] * at[1]
    }

    /// `p`'s texel in a level of texels `texel` m wide: whole, fraction.
    fn texel_of(&self, p: DVec3, texel: f64) -> ([i64; 2], [f64; 2]) {
        let c = self.across(p).map(|c| c / texel);
        let whole = c.map(f64::floor);
        (
            [whole[0] as i64, whole[1] as i64],
            [c[0] - whole[0], c[1] - whole[1]],
        )
    }

    /// Whether every page asked for is resident and its terrain unchanged.
    fn complete(&self, physical: &[Physical]) -> bool {
        self.heard.is_some()
            && self.asked.keys().all(|k| {
                self.resident
                    .get(k)
                    .is_some_and(|&p| physical[p as usize].stale != Stale::Changed)
            })
    }
}

/// A physical page: the epoch and page it holds, its near plane (planet-
/// centred distance along its epoch's sun), the last frame it was asked
/// for, and whether it is traced again ([`Stale`]).
#[derive(Clone, Copy)]
struct Physical {
    owner: Option<(u64, PageKey)>,
    near: f64,
    used: u64,
    stale: Stale,
}

/// A trace view of the current epoch: a level, its top-left page (a block
/// of [`VIEW_PAGES`] pages aligned to it), and the pages to trace in it
/// with the physical pages they go to.
pub(super) struct Window {
    pub level: u8,
    pub origin: [i32; 2],
    pub pages: Vec<(PageKey, u16)>,
}

/// Pages asked for in a frame, as the shadow pass marked them: the epoch
/// and the eye's page in each level (the window's centre) that frame.
#[derive(Clone, Copy)]
pub(super) struct Asking {
    pub epoch: u64,
    pub frame: u64,
    pub eye_pages: [[i32; 2]; LEVELS],
}

/// What the planner did this frame (diagnostics).
#[derive(Clone, Copy, Debug, Default)]
pub struct SunPageStats {
    pub resident: usize,
    pub asked: usize,
    /// Pages asked for and missing, whose terrain changed, traced through a
    /// coarser cut (counted once the world settled), next to fresh
    /// requests.
    pub missing: usize,
    pub stale: usize,
    pub coarse: usize,
    pub prefetch: usize,
    pub traced: usize,
    pub views: usize,
    pub epochs: usize,
    pub out_of_pages: bool,
}

pub(super) struct SunPages {
    /// The finest texel (m).
    pub finest: f64,
    /// The current epoch, then the last one (until replaced).
    pub epochs: Vec<Epoch>,
    physical: Vec<Physical>,
    /// The page tables as uploaded (`sun_occlusion_table_index`), and the
    /// rows changed since the last upload.
    pub table: Vec<u32>,
    pub dirty_rows: FxHashSet<usize>,
    next_epoch: u64,
    pub stats: SunPageStats,
}

impl SunPages {
    /// Pages for a world whose finest voxel is `voxel` m: the finest texel a
    /// power of two near half of it.
    pub fn new(voxel: f64) -> Self {
        Self {
            finest: (voxel * 0.5).log2().floor().exp2(),
            epochs: Vec::new(),
            physical: vec![
                Physical {
                    owner: None,
                    near: 0.0,
                    used: 0,
                    stale: Stale::No,
                };
                SUN_OCCLUSION_PHYSICAL as usize
            ],
            table: vec![0; SUN_OCCLUSION_TABLE_WORDS as usize],
            dirty_rows: FxHashSet::default(),
            next_epoch: 0,
            stats: SunPageStats::default(),
        }
    }

    /// Level `level`'s texel (m).
    pub fn texel(&self, level: usize) -> f64 {
        self.finest * (level as f64).exp2()
    }

    /// The current epoch.
    pub fn current(&self) -> Option<&Epoch> {
        self.epochs.first()
    }

    /// The eye's page in each level of the current epoch.
    pub fn eye_pages(&self, eye: DVec3) -> [[i32; 2]; LEVELS] {
        let mut out = [[0; 2]; LEVELS];
        if let Some(e) = self.current() {
            for (level, o) in out.iter_mut().enumerate() {
                let (t, _) = e.texel_of(eye, self.texel(level));
                *o = t.map(|t| (t >> PAGE_SHIFT) as i32);
            }
        }
        out
    }

    /// The frame's epochs for the sun toward `sun` and the eye at `eye`
    /// (planet-centred): a new one when the sun turned past [`TURN`] (and
    /// the last is done) or [`JUMP`], or the eye left the anchor's texel
    /// range; the last dropped once the current has every page asked for.
    /// Then the pages kept around the eye asked for.
    pub fn begin(&mut self, frame: u64, eye: DVec3, sun: DVec3) {
        let sun = sun.normalize();
        let reach = self.finest * f64::from(1u32 << 30);
        let state = self.current().map(|c| {
            (
                c.sun.dot(sun).clamp(-1.0, 1.0).acos(),
                (eye - c.anchor).length() > reach,
            )
        });
        let start = match state {
            None => true,
            Some((turn, far)) => {
                if turn > JUMP || far {
                    self.drop_last();
                    true
                } else {
                    turn > TURN && self.epochs.len() == 1
                }
            }
        };
        if self.epochs.len() == 2
            && (self.epochs[0].complete(&self.physical)
                || frame > self.epochs[0].created + HANDOVER_FRAMES)
        {
            self.drop_last();
        }
        if start {
            let slot = self.epochs.first().map_or(0, |c| 1 - c.slot);
            let id = self.next_epoch;
            self.next_epoch += 1;
            self.epochs.insert(0, Epoch::new(id, sun, eye, slot, frame));
        }
        // The 2x2 pages nearest the eye in the coarse levels.
        let finest = self.finest;
        let e = &mut self.epochs[0];
        for level in 0..LEVELS {
            let texel = finest * (level as f64).exp2();
            if texel * f64::from(PAGE) < BASE_PAGE_M {
                continue;
            }
            let (t, _) = e.texel_of(eye, texel);
            let page = t.map(|t| (t >> PAGE_SHIFT) as i32);
            let side = t.map(|t| {
                if (t & (PAGE as i64 - 1)) < PAGE as i64 / 2 {
                    -1
                } else {
                    1
                }
            });
            for (dx, dy) in [(0, 0), (side[0], 0), (0, side[1]), (side[0], side[1])] {
                e.asked.insert(
                    PageKey {
                        level: level as u8,
                        page: [page[0] + dx, page[1] + dy],
                    },
                    frame,
                );
            }
        }
    }

    /// Drops the last epoch: its pages freed, their table entries cleared.
    fn drop_last(&mut self) {
        if self.epochs.len() < 2 {
            return;
        }
        let last = self.epochs.pop().unwrap();
        for (key, &p) in &last.resident {
            self.clear_entry(last.slot, *key, p);
            self.physical[p as usize].owner = None;
        }
    }

    fn clear_entry(&mut self, slot: usize, key: PageKey, physical: u16) {
        let i = sun_occlusion_table_index(slot, key.level as usize, key.page);
        if sun_occlusion_entry_physical(self.table[i], key.page) == Some(u32::from(physical)) {
            self.table[i] = 0;
            self.dirty_rows.insert(i / SUN_OCCLUSION_WINDOW as usize);
        }
    }

    /// The pages a frame's shadow pass asked for (`asking`; `words`, its
    /// bits: a level's window after another, rows of the window).
    pub fn asked(&mut self, asking: &Asking, words: &[u32]) {
        let Some(e) = self.epochs.iter_mut().find(|e| e.id == asking.epoch) else {
            return;
        };
        let per_level = (WINDOW * WINDOW) as usize;
        e.heard = Some(e.heard.map_or(asking.frame, |h| h.max(asking.frame)));
        for (w, &word) in words.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let b = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let index = w * 32 + b;
                let level = index / per_level;
                if level >= LEVELS {
                    continue;
                }
                let at = index % per_level;
                let eye = asking.eye_pages[level];
                let key = PageKey {
                    level: level as u8,
                    page: [
                        eye[0] + (at as i32 % WINDOW) - WINDOW / 2,
                        eye[1] + (at as i32 / WINDOW) - WINDOW / 2,
                    ],
                };
                let f = e.asked.entry(key).or_insert(asking.frame);
                *f = (*f).max(asking.frame);
                if let Some(&p) = e.resident.get(&key) {
                    let used = &mut self.physical[p as usize].used;
                    *used = (*used).max(asking.frame);
                }
            }
        }
    }

    /// Pages whose rays cross any of `boxes` (world) traced again: before
    /// others where the terrain changed in them, in the background where
    /// the cut refined (`why`). Each box marks the pages over it
    /// in the levels where it spans at most 2x2 pages and every coarser
    /// one; a finer page is stale when one of its ancestors is marked (the
    /// region marked is at most a page larger than the box at that level).
    /// Hash lookups only: a settle's thousands of boxes a frame against
    /// thousands of pages stay well under a millisecond.
    pub fn invalidate(&mut self, boxes: &[[DVec3; 2]], why: Stale) {
        if boxes.is_empty() {
            return;
        }
        let finest = self.finest;
        let page_m = |level: usize| finest * (level as f64).exp2() * f64::from(PAGE);
        for e in &self.epochs {
            if e.resident.is_empty() {
                continue;
            }
            let mut marked: FxHashSet<PageKey> = FxHashSet::default();
            for [lo, hi] in boxes {
                // The box's extent across the sun (m from the anchor).
                let mut r = [
                    f64::INFINITY,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    f64::NEG_INFINITY,
                ];
                for c in 0..8 {
                    let a = e.across(DVec3::new(
                        if c & 1 == 0 { lo.x } else { hi.x },
                        if c & 2 == 0 { lo.y } else { hi.y },
                        if c & 4 == 0 { lo.z } else { hi.z },
                    ));
                    r = [r[0].min(a[0]), r[1].min(a[1]), r[2].max(a[0]), r[3].max(a[1])];
                }
                let span = (r[2] - r[0]).max(r[3] - r[1]).max(0.0);
                let first = (0..LEVELS).find(|&l| span <= page_m(l)).unwrap_or(LEVELS - 1);
                for level in first..LEVELS {
                    let m = page_m(level);
                    let (x0, y0) = ((r[0] / m).floor() as i32, (r[1] / m).floor() as i32);
                    let (x1, y1) = ((r[2] / m).floor() as i32, (r[3] / m).floor() as i32);
                    for y in y0..=y1.min(y0 + 1) {
                        for x in x0..=x1.min(x0 + 1) {
                            marked.insert(PageKey {
                                level: level as u8,
                                page: [x, y],
                            });
                        }
                    }
                }
            }
            for (key, &p) in &e.resident {
                let hit = (usize::from(key.level)..LEVELS).any(|l| {
                    let up = (l - usize::from(key.level)) as u32;
                    marked.contains(&PageKey {
                        level: l as u8,
                        page: key.page.map(|c| c >> up),
                    })
                });
                if hit {
                    let stale = &mut self.physical[p as usize].stale;
                    if why == Stale::Changed || *stale == Stale::No {
                        *stale = why;
                    }
                }
            }
        }
    }

    /// The pages to trace this frame, in views, with their physical pages
    /// (allocated, their table entries set): pages asked for and missing
    /// (coarse first: they back the finer ones' readers), pages whose
    /// terrain changed; then in the background ([`BACKGROUND_VIEWS`]) pages
    /// next to fresh requests (the view moving onto them) and, once the
    /// world has `settled`, pages of a coarser cut; at most
    /// [`TRACE_PAGES`] in [`MAX_VIEWS`] views. Physical pages come free, or
    /// from pages not asked for in [`KEEP_FRAMES`] (the last epoch's first,
    /// then the longest unused).
    pub fn plan(&mut self, frame: u64, settled: bool) -> Vec<Window> {
        let mut stats = SunPageStats {
            epochs: self.epochs.len(),
            ..Default::default()
        };
        let Some(e) = self.epochs.first_mut() else {
            self.stats = stats;
            return Vec::new();
        };
        e.asked.retain(|_, f| *f + ASKED_FRAMES >= frame);
        let physical = &self.physical;
        // (class, coarser first, key): 0 missing, 1 changed, 2 next to a
        // fresh request, 3 of a coarser cut.
        let mut wanted: Vec<(u8, std::cmp::Reverse<u8>, PageKey)> = Vec::new();
        let fresh = e.heard;
        let mut near: FxHashSet<PageKey> = FxHashSet::default();
        for (&key, &f) in &e.asked {
            match e.resident.get(&key).map(|&p| physical[p as usize]) {
                None => {
                    stats.missing += 1;
                    wanted.push((0, std::cmp::Reverse(key.level), key));
                }
                Some(p) if p.stale == Stale::Changed => {
                    stats.stale += 1;
                    wanted.push((1, std::cmp::Reverse(key.level), key));
                }
                Some(p) if p.stale == Stale::Coarse && settled => {
                    stats.coarse += 1;
                    wanted.push((3, std::cmp::Reverse(key.level), key));
                }
                _ => {}
            }
            if Some(f) == fresh {
                for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                    let n = PageKey {
                        level: key.level,
                        page: [key.page[0] + dx, key.page[1] + dy],
                    };
                    if !e.asked.contains_key(&n) && !e.resident.contains_key(&n) {
                        near.insert(n);
                    }
                }
            }
        }
        stats.prefetch = near.len();
        wanted.extend(near.into_iter().map(|k| (2, std::cmp::Reverse(k.level), k)));
        wanted.sort_unstable_by_key(|&(class, coarse, k)| (class, coarse, k.page[1], k.page[0]));
        stats.asked = e.asked.len();
        // Views: a block of pages a view, in the order their best page
        // came.
        let mut views: Vec<Window> = Vec::new();
        let (mut taken, mut background) = (0, 0);
        for &(class, _, key) in &wanted {
            if taken == TRACE_PAGES {
                break;
            }
            let origin = key.page.map(|p| (p >> VIEW_SHIFT) << VIEW_SHIFT);
            let view = match views
                .iter_mut()
                .position(|v| v.level == key.level && v.origin == origin)
            {
                Some(v) => v,
                None if class >= 2 && background == BACKGROUND_VIEWS => continue,
                None if views.len() < MAX_VIEWS => {
                    background += usize::from(class >= 2);
                    views.push(Window {
                        level: key.level,
                        origin,
                        pages: Vec::new(),
                    });
                    views.len() - 1
                }
                None => continue,
            };
            views[view].pages.push((key, u16::MAX));
            taken += 1;
        }
        // Physical pages: those the pages hold, then free ones, then the
        // last epoch's, then the longest unused of the current's.
        let current = e.id;
        let mut spare: Vec<u16> = (0..SUN_OCCLUSION_PHYSICAL as u16)
            .filter(|&p| {
                let ph = &physical[p as usize];
                ph.owner.is_none_or(|(id, _)| id != current) || ph.used + KEEP_FRAMES < frame
            })
            .collect();
        // Popped from the back: the best last.
        spare.sort_unstable_by_key(|&p| {
            let ph = &physical[p as usize];
            let rank = match ph.owner {
                None => 2,
                Some((id, _)) if id != current => 1,
                _ => 0,
            };
            (rank, std::cmp::Reverse(ph.used))
        });
        let last_slot = self.epochs.get(1).map(|l| (l.id, l.slot));
        let mut evicted: Vec<(Option<usize>, PageKey, u16)> = Vec::new();
        let e = &mut self.epochs[0];
        // Pages traced again in place keep theirs.
        let held: FxHashSet<u16> = views
            .iter()
            .flat_map(|v| &v.pages)
            .filter_map(|(key, _)| e.resident.get(key).copied())
            .collect();
        for v in &mut views {
            v.pages.retain_mut(|(key, p)| {
                if let Some(&own) = e.resident.get(key) {
                    *p = own;
                    return true;
                }
                while let Some(s) = spare.pop() {
                    if held.contains(&s) {
                        continue;
                    }
                    *p = s;
                    return true;
                }
                stats.out_of_pages = true;
                false
            });
        }
        views.retain(|v| !v.pages.is_empty());
        // Take the physical pages from their owners.
        for v in &views {
            for &(key, p) in &v.pages {
                if let Some((id, old)) = self.physical[p as usize].owner {
                    if (id, old) == (current, key) {
                        continue;
                    }
                    let slot = if id == current {
                        e.resident.remove(&old);
                        Some(e.slot)
                    } else {
                        last_slot.filter(|(l, _)| *l == id).map(|(_, s)| s)
                    };
                    evicted.push((slot, old, p));
                }
            }
        }
        for (slot, old, p) in evicted {
            if let Some(slot) = slot {
                self.clear_entry(slot, old, p);
            }
            if let Some(last) = self.epochs.get_mut(1) {
                if self.physical[p as usize]
                    .owner
                    .is_some_and(|(id, _)| id == last.id)
                {
                    last.resident.remove(&old);
                }
            }
            self.physical[p as usize].owner = None;
        }
        stats.traced = views.iter().map(|v| v.pages.len()).sum();
        stats.views = views.len();
        stats.resident = self.epochs.iter().map(|e| e.resident.len()).sum();
        self.stats = stats;
        views
    }

    /// The pages of `window` traced this frame through a near plane `near`
    /// (world distance along the current epoch's sun), the world `settled`
    /// or not (traced again in the background once it is): resident, their
    /// table entries set.
    pub fn traced(&mut self, window: &Window, near: f64, frame: u64, settled: bool) {
        let (id, slot) = (self.epochs[0].id, self.epochs[0].slot);
        for &(key, p) in &window.pages {
            self.epochs[0].resident.insert(key, p);
            // Pages traced ahead of a request are kept as if asked now.
            let used = self.epochs[0]
                .asked
                .get(&key)
                .copied()
                .unwrap_or(0)
                .max(frame);
            self.physical[p as usize] = Physical {
                owner: Some((id, key)),
                near,
                used,
                stale: if settled { Stale::No } else { Stale::Coarse },
            };
            let i = sun_occlusion_table_index(slot, key.level as usize, key.page);
            self.table[i] = sun_occlusion_entry(key.page, u32::from(p));
            self.dirty_rows.insert(i / SUN_OCCLUSION_WINDOW as usize);
        }
    }

    /// The frame for the eye at `eye` (planet-centred): the epochs, the eye
    /// in each level; and each physical page's near plane relative to the
    /// eye (along its epoch's forward: a point `p` lies `dot(p, forward) -
    /// near` past it).
    pub fn frame(&self, eye: DVec3) -> (SunOcclusionFrame, Vec<f32>) {
        let mut frame = SunOcclusionFrame {
            info: [
                self.finest as f32,
                LEVELS as f32,
                SUN_OCCLUSION_PAGE as f32,
                SUN_OCCLUSION_WINDOW as f32,
            ],
            atlas: [SUN_OCCLUSION_ATLAS_PAGES, SUN_OCCLUSION_TABLE_WORDS, 0, 0],
            ..Default::default()
        };
        for (out, e) in frame.epochs.iter_mut().zip(&self.epochs) {
            let [right, down, forward] = e.axes.map(|a| a.as_vec3());
            *out = SunOcclusionEpoch {
                right: right.extend((e.slot + 1) as f32).to_array(),
                down: down.extend(0.0).to_array(),
                forward: forward.extend(0.0).to_array(),
                levels: std::array::from_fn(|level| {
                    let (t, f) = e.texel_of(eye, self.texel(level));
                    SunOcclusionLevel {
                        texel: t.map(|t| t as i32),
                        frac: f.map(|f| f as f32),
                    }
                }),
            };
        }
        let near = self
            .physical
            .iter()
            .map(|p| {
                p.owner
                    .and_then(|(id, _)| self.epochs.iter().find(|e| e.id == id))
                    .map_or(0.0, |e| (eye.dot(e.sun) - p.near) as f32)
            })
            .collect();
        (frame, near)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pages_asked_are_traced_coarse_first_and_kept() {
        let mut pages = SunPages::new(0.1);
        let eye = DVec3::new(0.0, 1000.0, 0.0);
        let sun = DVec3::new(0.3, 0.8, 0.4).normalize();
        pages.begin(0, eye, sun);
        // Base coverage only: coarse levels, four pages each.
        let views = pages.plan(0, true);
        let traced: usize = views.iter().map(|v| v.pages.len()).sum();
        assert!(traced > 0 && traced <= TRACE_PAGES);
        assert!(views
            .iter()
            .all(|v| pages.texel(v.level as usize) * f64::from(PAGE) >= BASE_PAGE_M));
        for v in &views {
            pages.traced(v, 0.0, 0, true);
        }
        // Everything asked traced within a few frames, then nothing.
        let mut frame = 1;
        loop {
            pages.begin(frame, eye, sun);
            let views = pages.plan(frame, true);
            if views.is_empty() {
                break;
            }
            for v in &views {
                pages.traced(v, 0.0, frame, true);
            }
            frame += 1;
            assert!(frame < 20);
        }
        let asking = Asking {
            epoch: pages.current().unwrap().id,
            frame,
            eye_pages: pages.eye_pages(eye),
        };
        // A request: level 3, the window's centre page.
        let mut words = vec![0u32; LEVELS * (WINDOW * WINDOW) as usize / 32];
        let bit = 3 * (WINDOW * WINDOW) as usize + (WINDOW / 2 * WINDOW + WINDOW / 2) as usize;
        words[bit / 32] |= 1 << (bit % 32);
        pages.asked(&asking, &words);
        let views = pages.plan(frame, true);
        assert_eq!(views[0].level, 3);
        assert_eq!(views[0].pages[0].0.page, asking.eye_pages[3]);
    }

    #[test]
    fn changed_boxes_stale_the_pages_over_them() {
        let mut pages = SunPages::new(0.1);
        let eye = DVec3::new(0.0, 1000.0, 0.0);
        let sun = DVec3::Y;
        pages.begin(0, eye, sun);
        loop {
            let views = pages.plan(0, true);
            if views.is_empty() {
                break;
            }
            for v in &views {
                pages.traced(v, 0.0, 0, true);
            }
        }
        let e = pages.current().unwrap();
        // A 1 m box at the eye's column: the coarse pages over it, nothing
        // far away.
        let near: Vec<PageKey> = e.resident.keys().copied().collect();
        pages.invalidate(
            &[[eye - DVec3::splat(0.5), eye + DVec3::splat(0.5)]],
            Stale::Changed,
        );
        let e = pages.current().unwrap();
        let stale = |k: &PageKey| pages.physical[e.resident[k] as usize].stale == Stale::Changed;
        let over: Vec<&PageKey> = near
            .iter()
            .filter(|k| {
                let m = pages.texel(usize::from(k.level)) * f64::from(PAGE);
                let a = e.across(eye);
                (0..2).all(|i| (a[i] / m).floor() as i32 == k.page[i])
            })
            .collect();
        assert!(!over.is_empty() && over.iter().all(|k| stale(k)));
        assert!(near.iter().filter(|k| stale(k)).count() <= over.len() * 4);
    }

    #[test]
    fn a_turning_sun_hands_over_epochs() {
        let mut pages = SunPages::new(0.1);
        let eye = DVec3::new(0.0, 1000.0, 0.0);
        let mut sun = DVec3::new(0.3, 0.8, 0.4).normalize();
        for frame in 0..400 {
            pages.begin(frame, eye, sun);
            for v in pages.plan(frame, true) {
                pages.traced(&v, 0.0, frame, true);
            }
            let asking = Asking {
                epoch: pages.current().unwrap().id,
                frame,
                eye_pages: pages.eye_pages(eye),
            };
            pages.asked(&asking, &[]);
            sun = glam::DQuat::from_rotation_y(0.0005).mul_vec3(sun);
            assert!(pages.epochs.len() <= 2);
        }
        // The sun turned 0.2 rad: epochs followed it within a few turns.
        let c = pages.current().unwrap();
        assert!(
            c.sun.dot(sun).acos() < 4.0 * TURN,
            "{}",
            c.sun.dot(sun).acos()
        );
    }
}
