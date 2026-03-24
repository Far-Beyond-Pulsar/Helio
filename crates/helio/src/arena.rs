use std::marker::PhantomData;

use crate::handles::Handle;

#[derive(Clone, Copy, Debug)]
struct DenseSlotMeta {
    generation: u32,
    dense_index: u32,
    occupied: bool,
}

pub struct DenseRemove<T, H> {
    pub removed: T,
    pub dense_index: usize,
    pub moved: Option<(H, usize)>,
}

pub struct DenseArena<T, H> {
    slots: Vec<DenseSlotMeta>,
    dense: Vec<T>,
    dense_to_slot: Vec<u32>,
    free_list: Vec<u32>,
    marker: PhantomData<H>,
}

impl<T, H: Handle> DenseArena<T, H> {
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            dense: Vec::new(),
            dense_to_slot: Vec::new(),
            free_list: Vec::new(),
            marker: PhantomData,
        }
    }

    /// Number of live items in the arena.
    pub fn dense_len(&self) -> usize {
        self.dense.len()
    }

    /// Access item by dense-array index (0..dense_len). Used for bulk rebuilds.
    pub fn get_dense(&self, index: usize) -> Option<&T> {
        self.dense.get(index)
    }

    /// Mutable access by dense-array index. Used to patch GPU slot bookkeeping after rebuild.
    pub fn get_dense_mut(&mut self, index: usize) -> Option<&mut T> {
        self.dense.get_mut(index)
    }

    pub fn insert(&mut self, value: T) -> (H, usize) {
        let dense_index = self.dense.len();
        let slot_index = if let Some(slot) = self.free_list.pop() {
            let meta = &mut self.slots[slot as usize];
            meta.occupied = true;
            meta.dense_index = dense_index as u32;
            slot
        } else {
            let slot = self.slots.len() as u32;
            self.slots.push(DenseSlotMeta {
                generation: 1,
                dense_index: dense_index as u32,
                occupied: true,
            });
            slot
        };

        self.dense.push(value);
        self.dense_to_slot.push(slot_index);
        let generation = self.slots[slot_index as usize].generation;
        (H::from_parts(slot_index, generation), dense_index)
    }

    pub fn get_mut_with_index(&mut self, handle: H) -> Option<(usize, &mut T)> {
        let meta = *self.slots.get(handle.slot() as usize)?;
        if !meta.occupied || meta.generation != handle.generation() {
            return None;
        }
        let dense_index = meta.dense_index as usize;
        self.dense
            .get_mut(dense_index)
            .map(|value| (dense_index, value))
    }

    pub fn get_with_index(&self, handle: H) -> Option<(usize, &T)> {
        let meta = *self.slots.get(handle.slot() as usize)?;
        if !meta.occupied || meta.generation != handle.generation() {
            return None;
        }
        let dense_index = meta.dense_index as usize;
        self.dense
            .get(dense_index)
            .map(|value| (dense_index, value))
    }

    pub fn remove(&mut self, handle: H) -> Option<DenseRemove<T, H>> {
        let slot_index = handle.slot() as usize;
        let meta = self.slots.get(slot_index).copied()?;
        if !meta.occupied || meta.generation != handle.generation() {
            return None;
        }

        let dense_index = meta.dense_index as usize;
        let removed = self.dense.swap_remove(dense_index);
        self.dense_to_slot.swap_remove(dense_index);

        let moved = if dense_index < self.dense.len() {
            let moved_slot = self.dense_to_slot[dense_index] as usize;
            self.slots[moved_slot].dense_index = dense_index as u32;
            Some((
                H::from_parts(
                    self.dense_to_slot[dense_index],
                    self.slots[moved_slot].generation,
                ),
                dense_index,
            ))
        } else {
            None
        };

        let slot = &mut self.slots[slot_index];
        slot.occupied = false;
        slot.generation = slot.generation.wrapping_add(1).max(1);
        self.free_list.push(slot_index as u32);

        Some(DenseRemove {
            removed,
            dense_index,
            moved,
        })
    }
}

#[derive(Debug)]
struct SparseSlot<T> {
    generation: u32,
    value: Option<T>,
}

pub struct SparsePool<T, H> {
    slots: Vec<SparseSlot<T>>,
    free_list: Vec<u32>,
    live_count: usize,
    marker: PhantomData<H>,
}

impl<T, H: Handle> SparsePool<T, H> {
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_list: Vec::new(),
            live_count: 0,
            marker: PhantomData,
        }
    }

    pub fn insert(&mut self, value: T) -> (H, usize, bool) {
        if let Some(slot) = self.free_list.pop() {
            let entry = &mut self.slots[slot as usize];
            entry.value = Some(value);
            self.live_count += 1;
            return (H::from_parts(slot, entry.generation), slot as usize, false);
        }

        let slot = self.slots.len() as u32;
        self.slots.push(SparseSlot {
            generation: 1,
            value: Some(value),
        });
        self.live_count += 1;
        (H::from_parts(slot, 1), slot as usize, true)
    }

    pub fn get(&self, handle: H) -> Option<&T> {
        let slot = self.slots.get(handle.slot() as usize)?;
        if slot.generation != handle.generation() {
            return None;
        }
        slot.value.as_ref()
    }

    pub fn get_by_slot(&self, slot_index: usize) -> Option<&T> {
        self.slots.get(slot_index)?.value.as_ref()
    }

    pub fn get_mut_by_slot(&mut self, slot_index: usize) -> Option<&mut T> {
        self.slots.get_mut(slot_index)?.value.as_mut()
    }

    pub fn get_mut_with_slot(&mut self, handle: H) -> Option<(usize, &mut T)> {
        let slot_index = handle.slot() as usize;
        let slot = self.slots.get_mut(slot_index)?;
        if slot.generation != handle.generation() {
            return None;
        }
        slot.value.as_mut().map(|value| (slot_index, value))
    }

    pub fn remove(&mut self, handle: H) -> Option<(usize, T)> {
        let slot_index = handle.slot() as usize;
        let slot = self.slots.get_mut(slot_index)?;
        if slot.generation != handle.generation() {
            return None;
        }
        let value = slot.value.take()?;
        slot.generation = slot.generation.wrapping_add(1).max(1);
        self.free_list.push(slot_index as u32);
        self.live_count = self.live_count.saturating_sub(1);
        Some((slot_index, value))
    }

    pub fn live_len(&self) -> usize {
        self.live_count
    }

    pub fn slot_len(&self) -> usize {
        self.slots.len()
    }

    pub fn has_free_slot(&self) -> bool {
        !self.free_list.is_empty()
    }
}

