//! Bounded, revision-aware transient slot ownership for the voxel mesh pass.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::{VoxelChunkKey, VOXEL_PADDED_WORDS};

/// SceneDB entity bits include its generation; kind distinguishes object/terrain rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VoxelEntryId {
    pub entity_bits: u64,
    pub kind: u8,
}

#[derive(Clone, Debug)]
pub struct VoxelPreparedBrick {
    pub key: VoxelChunkKey,
    pub words: [u32; VOXEL_PADDED_WORDS],
    pub origin: [f32; 3],
    pub voxel_size: f32,
    pub mode: u32,
    /// One-based local slots resolve to these SceneDB material record IDs.
    pub material_ids: Vec<u32>,
}

impl VoxelPreparedBrick {
    pub const UPLOAD_BYTES: usize = VOXEL_PADDED_WORDS * 4;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VoxelFrameBudget {
    pub max_bricks: usize,
    pub max_upload_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VoxelResidencyError {
    InvalidBudget,
    InvalidBrick,
    TooManyMaterials(usize),
    UnsupportedMaterialId(u32),
    TooManyBricks { requested: usize, capacity: usize },
    Capacity { requested: usize, available: usize },
    StaleRevision,
}

#[derive(Clone, Debug)]
pub struct VoxelUpload {
    pub entry: VoxelEntryId,
    pub slot: u32,
    pub brick: VoxelPreparedBrick,
}

#[derive(Clone, Debug)]
pub struct VoxelPromotion {
    pub entry: VoxelEntryId,
    pub generation: u64,
    pub revision: u64,
    pub old_slots: Vec<u32>,
    pub new_slots: Vec<u32>,
}

#[derive(Default)]
pub struct VoxelFrameWork {
    pub uploads: Vec<VoxelUpload>,
    pub promotions: Vec<VoxelPromotion>,
    pub deferred_bricks: usize,
}

struct Pending {
    generation: u64,
    revision: u64,
    all_slots: Vec<u32>,
    bricks: VecDeque<(u32, VoxelPreparedBrick)>,
}

#[derive(Default)]
struct Entry {
    active_slots: Vec<u32>,
    active_tag: Option<(u64, u64)>,
    desired_tag: Option<(u64, u64)>,
    pending: Option<Pending>,
    last_use: u64,
    needs_rebuild: bool,
}

/// Slot count is a device budget, never an authored-entry cap. Canonical bytes
/// remain in SceneDB, so evicted entries can be prepared again.
pub struct VoxelResidency {
    owners: Vec<Option<VoxelEntryId>>,
    external_slots: HashSet<u32>,
    entries: HashMap<VoxelEntryId, Entry>,
    clock: u64,
    pub evictions: u64,
    pub rebuilds: u64,
    pub stale_results: u64,
}

impl VoxelResidency {
    pub fn new(capacity: usize) -> Self {
        Self {
            owners: vec![None; capacity],
            external_slots: HashSet::new(),
            entries: HashMap::new(),
            clock: 0,
            evictions: 0,
            rebuilds: 0,
            stale_results: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.owners.len()
    }
    /// Legacy explicit-slot callers and the SceneDB allocator cannot own the
    /// same slot. Reservation is retained until clear_brick_slot releases it.
    pub fn reserve_external(&mut self, slot: u32) -> bool {
        let Some(owner) = self.owners.get(slot as usize) else {
            return false;
        };
        if owner.is_some() {
            return false;
        }
        self.external_slots.insert(slot);
        true
    }
    pub fn release_external(&mut self, slot: u32) {
        self.external_slots.remove(&slot);
    }
    pub fn resident_bricks(&self) -> usize {
        self.owners.iter().filter(|slot| slot.is_some()).count()
    }
    pub fn staging_bricks(&self) -> usize {
        self.entries
            .values()
            .map(|e| e.pending.as_ref().map_or(0, |p| p.all_slots.len()))
            .sum()
    }
    pub fn needs_rebuild(&self, id: VoxelEntryId) -> bool {
        self.entries
            .get(&id)
            .is_some_and(|entry| entry.needs_rebuild)
    }
    pub fn active_tag(&self, id: VoxelEntryId) -> Option<(u64, u64)> {
        self.entries.get(&id).and_then(|entry| entry.active_tag)
    }

    /// Queue a complete prepared result. All slots are reserved up front so
    /// capacity failure leaves the previous complete result untouched.
    /// Returns slots from evicted entries for the caller to clear on the GPU.
    pub fn queue_revision(
        &mut self,
        id: VoxelEntryId,
        generation: u64,
        revision: u64,
        bricks: Vec<VoxelPreparedBrick>,
    ) -> Result<Vec<u32>, VoxelResidencyError> {
        if bricks.len() > self.capacity() {
            return Err(VoxelResidencyError::TooManyBricks {
                requested: bricks.len(),
                capacity: self.capacity(),
            });
        }
        for brick in &bricks {
            if brick.material_ids.len() > 255 {
                return Err(VoxelResidencyError::TooManyMaterials(
                    brick.material_ids.len(),
                ));
            }
            if let Some(&id) = brick.material_ids.iter().find(|&&id| id >= 0x8000_0000) {
                return Err(VoxelResidencyError::UnsupportedMaterialId(id));
            }
            if !brick.voxel_size.is_finite()
                || brick.voxel_size <= 0.0
                || brick.origin.iter().any(|v| !v.is_finite())
            {
                return Err(VoxelResidencyError::InvalidBrick);
            }
        }
        let entry = self.entries.entry(id).or_default();
        if let Some((old_generation, old_revision)) = entry.desired_tag {
            if (generation, revision) < (old_generation, old_revision)
                || ((generation, revision) == (old_generation, old_revision)
                    && !entry.needs_rebuild)
            {
                self.stale_results += 1;
                return Err(VoxelResidencyError::StaleRevision);
            }
        }
        let reclaimable_pending = entry.pending.as_ref().map_or(0, |p| p.all_slots.len());
        let reclaimable_other: usize = self
            .entries
            .iter()
            .filter(|(candidate, entry)| **candidate != id && entry.pending.is_none())
            .map(|(_, entry)| entry.active_slots.len())
            .sum();
        let possible = self.free_slots() + reclaimable_pending + reclaimable_other;
        if possible < bricks.len() {
            return Err(VoxelResidencyError::Capacity {
                requested: bricks.len(),
                available: possible,
            });
        }
        if let Some(old) = self
            .entries
            .get_mut(&id)
            .and_then(|entry| entry.pending.take())
        {
            for slot in old.all_slots {
                self.owners[slot as usize] = None;
            }
        }
        let mut cleared = Vec::new();
        let needed = bricks.len();
        while self.free_slots() < needed {
            let victim = self
                .entries
                .iter()
                .filter(|(candidate, entry)| {
                    **candidate != id && entry.pending.is_none() && !entry.active_slots.is_empty()
                })
                .min_by_key(|(_, entry)| entry.last_use)
                .map(|(candidate, _)| *candidate);
            let Some(victim) = victim else {
                return Err(VoxelResidencyError::Capacity {
                    requested: needed,
                    available: self.free_slots(),
                });
            };
            let victim_entry = self
                .entries
                .get_mut(&victim)
                .expect("selected entry exists");
            for slot in victim_entry.active_slots.drain(..) {
                self.owners[slot as usize] = None;
                cleared.push(slot);
            }
            victim_entry.active_tag = None;
            victim_entry.needs_rebuild = true;
            self.evictions += 1;
        }
        let mut all_slots = Vec::with_capacity(needed);
        let mut pending_bricks = VecDeque::with_capacity(needed);
        let mut free = self
            .owners
            .iter()
            .enumerate()
            .filter(|(slot, owner)| {
                owner.is_none() && !self.external_slots.contains(&(*slot as u32))
            })
            .map(|(slot, _)| slot as u32)
            .collect::<Vec<_>>()
            .into_iter();
        for brick in bricks {
            let slot = free.next().expect("capacity checked");
            all_slots.push(slot);
            pending_bricks.push_back((slot, brick));
        }
        for &slot in &all_slots {
            self.owners[slot as usize] = Some(id);
        }
        let entry = self.entries.get_mut(&id).expect("entry inserted");
        if entry.needs_rebuild {
            self.rebuilds += 1;
        }
        entry.needs_rebuild = false;
        entry.desired_tag = Some((generation, revision));
        entry.pending = Some(Pending {
            generation,
            revision,
            all_slots,
            bricks: pending_bricks,
        });
        self.clock = self.clock.saturating_add(1);
        entry.last_use = self.clock;
        Ok(cleared)
    }

    fn free_slots(&self) -> usize {
        self.owners
            .iter()
            .enumerate()
            .filter(|(slot, owner)| {
                owner.is_none() && !self.external_slots.contains(&(*slot as u32))
            })
            .count()
    }

    /// A frame receives at most its brick and byte budgets. Promotion is
    /// emitted only with the final upload of one entry's complete result.
    pub fn take_frame(
        &mut self,
        budget: VoxelFrameBudget,
    ) -> Result<VoxelFrameWork, VoxelResidencyError> {
        if budget.max_bricks == 0 || budget.max_upload_bytes < VoxelPreparedBrick::UPLOAD_BYTES {
            return Err(VoxelResidencyError::InvalidBudget);
        }
        let mut work = VoxelFrameWork::default();
        let max_by_bytes = budget.max_upload_bytes / VoxelPreparedBrick::UPLOAD_BYTES;
        let max = budget.max_bricks.min(max_by_bytes);
        let mut ids: Vec<_> = self.entries.keys().copied().collect();
        ids.sort_by_key(|id| self.entries.get(id).map_or(u64::MAX, |e| e.last_use));
        for id in ids {
            let entry = self.entries.get_mut(&id).expect("listed entry exists");
            let Some(pending) = entry.pending.as_mut() else {
                continue;
            };
            while work.uploads.len() < max {
                let Some((slot, brick)) = pending.bricks.pop_front() else {
                    break;
                };
                work.uploads.push(VoxelUpload {
                    entry: id,
                    slot,
                    brick,
                });
            }
            if pending.bricks.is_empty() {
                let complete = entry.pending.take().expect("pending exists");
                let old_slots =
                    std::mem::replace(&mut entry.active_slots, complete.all_slots.clone());
                entry.active_tag = Some((complete.generation, complete.revision));
                for &slot in &old_slots {
                    self.owners[slot as usize] = None;
                }
                work.promotions.push(VoxelPromotion {
                    entry: id,
                    generation: complete.generation,
                    revision: complete.revision,
                    old_slots,
                    new_slots: complete.all_slots,
                });
            }
            if work.uploads.len() == max {
                break;
            }
        }
        work.deferred_bricks = self
            .entries
            .values()
            .map(|e| e.pending.as_ref().map_or(0, |p| p.bricks.len()))
            .sum();
        Ok(work)
    }

    /// Removal/replacement releases transient slots. The caller clears these
    /// draw commands; it does not mutate the canonical SceneDB payload store.
    pub fn remove(&mut self, id: VoxelEntryId) -> Vec<u32> {
        let Some(entry) = self.entries.remove(&id) else {
            return Vec::new();
        };
        let mut slots = entry.active_slots;
        if let Some(pending) = entry.pending {
            slots.extend(pending.all_slots);
        }
        for &slot in &slots {
            self.owners[slot as usize] = None;
        }
        slots
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brick(n: i64) -> VoxelPreparedBrick {
        VoxelPreparedBrick {
            key: VoxelChunkKey::new(n, 0, 0, 0),
            words: [0; VOXEL_PADDED_WORDS],
            origin: [0.0; 3],
            voxel_size: 1.0,
            mode: 0,
            material_ids: vec![5],
        }
    }
    fn id(n: u64) -> VoxelEntryId {
        VoxelEntryId {
            entity_bits: n,
            kind: 1,
        }
    }
    fn budget(n: usize) -> VoxelFrameBudget {
        VoxelFrameBudget {
            max_bricks: n,
            max_upload_bytes: n * VoxelPreparedBrick::UPLOAD_BYTES,
        }
    }

    #[test]
    fn old_complete_result_remains_until_last_budgeted_upload() {
        let mut cache = VoxelResidency::new(4);
        cache.queue_revision(id(1), 1, 1, vec![brick(0)]).unwrap();
        assert_eq!(cache.take_frame(budget(1)).unwrap().promotions.len(), 1);
        cache
            .queue_revision(id(1), 1, 2, vec![brick(0), brick(1)])
            .unwrap();
        let first = cache.take_frame(budget(1)).unwrap();
        assert!(first.promotions.is_empty());
        assert_eq!(cache.active_tag(id(1)), Some((1, 1)));
        assert_eq!(first.deferred_bricks, 1);
        let final_frame = cache.take_frame(budget(1)).unwrap();
        assert_eq!(final_frame.promotions[0].old_slots.len(), 1);
        assert_eq!(final_frame.promotions[0].new_slots.len(), 2);
        assert_eq!(cache.active_tag(id(1)), Some((1, 2)));
    }

    #[test]
    fn capacity_evicts_other_entry_but_preserves_canonical_rebuild_signal() {
        let mut cache = VoxelResidency::new(3);
        cache.queue_revision(id(1), 1, 1, vec![brick(0)]).unwrap();
        cache.take_frame(budget(1)).unwrap();
        cache.queue_revision(id(2), 1, 1, vec![brick(1)]).unwrap();
        cache.take_frame(budget(1)).unwrap();
        let cleared = cache
            .queue_revision(id(3), 1, 1, vec![brick(2), brick(3)])
            .unwrap();
        assert_eq!(cleared.len(), 1);
        assert!(cache.needs_rebuild(id(1)));
        assert_eq!(cache.evictions, 1);
    }

    #[test]
    fn stale_generation_and_removal_never_replace_newer_output() {
        let mut cache = VoxelResidency::new(3);
        cache.queue_revision(id(1), 2, 1, vec![brick(0)]).unwrap();
        assert_eq!(
            cache
                .queue_revision(id(1), 1, 99, vec![brick(1)])
                .unwrap_err(),
            VoxelResidencyError::StaleRevision
        );
        assert_eq!(cache.remove(id(1)).len(), 1);
        assert_eq!(cache.resident_bricks(), 0);
    }
}
