//! Resident column index: the CPU mirror of the GPU column hash table plus
//! per-record column data.
//!
//! The GPU finds a column by linear probing from `slot_hash(key)` until an
//! empty slot, comparing the key stored in each probed record. The CPU uses
//! the very same table for its own lookups, and keeps each column's data in
//! fixed-size chunks indexed by record, so neither side ever rehashes or
//! reallocates in bulk: a general hash map of ~1M residents doubled its
//! capacity in one frame (10-70 ms stalls while leaving the ground).
//!
//! Deletion is backward-shift (Knuth's algorithm R) instead of tombstones, so
//! probe runs never degrade and the table never needs a rebuild. Every slot
//! write is returned to the caller for the frame's GPU table patch.
use crate::residency::{slot_hash, NONE};

/// Data of a resident column.
#[derive(Clone, Copy, Debug, Default)]
pub struct Resident {
    /// GPU record (column slot in the record/brick pools).
    pub record: u32,
    /// Hash table slot holding `record`.
    pub slot: u32,
    /// Edit-reference list block (heap block, size class), if any.
    pub edit_block: Option<(u32, u32)>,
    /// Holds references on its summary blocks (false: a slot conflict).
    pub blocks: bool,
}

#[derive(Clone, Copy, Default)]
struct Entry {
    key: u64,
    resident: Resident,
}

const CHUNK_BITS: u32 = 16;

pub struct ColumnIndex {
    table: Vec<u32>,
    /// One bit per occupied table slot, for bounded sparse retirement.
    occupied: Vec<u64>,
    mask: u32,
    /// Entries by record, in chunks of 2^16 that are never moved.
    chunks: Vec<Box<[Entry]>>,
    len: usize,
}

impl ColumnIndex {
    pub fn new(table_bits: u32) -> Self {
        let slots = 1usize << table_bits;
        Self { table: vec![NONE; slots], occupied: vec![0; slots.div_ceil(64)],
            mask: (1u32 << table_bits) - 1, chunks: Vec::new(), len: 0 }
    }

    /// The table as the GPU sees it (record per slot, `NONE` when empty).
    pub fn table(&self) -> &[u32] {
        &self.table
    }

    /// Inspect exactly one occupancy word from `start`. Return its next
    /// resident slot, or the first slot of the next word when none remain.
    /// Removal backshifts later entries, so recheck a removed resident's slot.
    pub(crate) fn retirement_step(&self, start: usize) -> (usize, Option<(u64, Resident)>) {
        debug_assert!(start < self.table.len());
        let word = start / 64;
        let remaining = self.occupied[word] & (u64::MAX << (start % 64));
        if remaining == 0 { return (((word + 1) * 64).min(self.table.len()), None); }
        let slot = word * 64 + remaining.trailing_zeros() as usize;
        let record = self.table[slot];
        let entry = self.entry(record);
        (slot, Some((entry.key, entry.resident)))
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn load(&self) -> f32 {
        self.len as f32 / self.table.len() as f32
    }

    fn home(&self, key: u64) -> u32 {
        slot_hash(key as u32, (key >> 32) as u32) & self.mask
    }

    fn entry(&self, record: u32) -> &Entry {
        &self.chunks[(record >> CHUNK_BITS) as usize][(record & ((1 << CHUNK_BITS) - 1)) as usize]
    }

    fn entry_mut(&mut self, record: u32) -> &mut Entry {
        let chunk = (record >> CHUNK_BITS) as usize;
        while self.chunks.len() <= chunk {
            self.chunks.push(vec![Entry::default(); 1 << CHUNK_BITS].into_boxed_slice());
        }
        &mut self.chunks[chunk][(record & ((1 << CHUNK_BITS) - 1)) as usize]
    }

    /// Slot holding `key`, found exactly as the GPU does.
    fn find(&self, key: u64) -> Option<u32> {
        let mut slot = self.home(key);
        loop {
            let record = self.table[slot as usize];
            if record == NONE {
                return None;
            }
            if self.entry(record).key == key {
                return Some(slot);
            }
            slot = (slot + 1) & self.mask;
        }
    }

    pub fn contains_key(&self, key: u64) -> bool {
        self.find(key).is_some()
    }

    pub fn get(&self, key: u64) -> Option<Resident> {
        self.find(key).map(|slot| self.entry(self.table[slot as usize]).resident)
    }

    pub fn get_mut(&mut self, key: u64) -> Option<&mut Resident> {
        let slot = self.find(key)?;
        let record = self.table[slot as usize];
        Some(&mut self.entry_mut(record).resident)
    }

    /// Insert a column that is not resident; returns its slot (the caller
    /// writes `(slot, record)` to the GPU table).
    pub fn insert(&mut self, key: u64, mut resident: Resident) -> u32 {
        debug_assert!(!self.contains_key(key));
        assert!((self.len as u64) < u64::from(self.mask), "column table full");
        let mut slot = self.home(key);
        while self.table[slot as usize] != NONE {
            slot = (slot + 1) & self.mask;
        }
        self.table[slot as usize] = resident.record;
        self.occupied[slot as usize / 64] |= 1u64 << (slot % 64);
        resident.slot = slot;
        *self.entry_mut(resident.record) = Entry { key, resident };
        self.len += 1;
        slot
    }

    /// Remove a column; every changed slot is appended to `writes`.
    pub fn remove(&mut self, key: u64, writes: &mut Vec<(u32, u32)>) -> Option<Resident> {
        let slot = self.find(key)?;
        let removed = self.entry(self.table[slot as usize]).resident;
        // Backward shift: later entries of the probe run whose home does not
        // lie cyclically in (gap, next] move back into the gap.
        let mut gap = slot;
        let mut next = (slot + 1) & self.mask;
        loop {
            let record = self.table[next as usize];
            if record == NONE {
                break;
            }
            let home = self.home(self.entry(record).key);
            let stays = if gap <= next { gap < home && home <= next } else { gap < home || home <= next };
            if !stays {
                self.table[gap as usize] = record;
                writes.push((gap, record));
                self.entry_mut(record).resident.slot = gap;
                gap = next;
            }
            next = (next + 1) & self.mask;
        }
        self.table[gap as usize] = NONE;
        // Backshift keeps every intermediate gap occupied; only the final
        // gap becomes empty, so no moved record needs an occupancy update.
        self.occupied[gap as usize / 64] &= !(1u64 << (gap % 64));
        writes.push((gap, NONE));
        self.len -= 1;
        Some(removed)
    }

    /// Resident columns (scans the table; for tests and diagnostics).
    pub fn iter(&self) -> impl Iterator<Item = (u64, Resident)> + '_ {
        self.table.iter().filter(|&&r| r != NONE).map(|&r| {
            let e = self.entry(r);
            (e.key, e.resident)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Random inserts and removes in a tiny table (long probe runs and
    /// wrap-around) always match a reference map, and every slot write keeps
    /// a GPU mirror identical.
    #[test]
    fn matches_a_map_under_churn() {
        let mut index = ColumnIndex::new(10);
        let mut mirror = vec![NONE; 1 << 10];
        let mut reference = std::collections::HashMap::new();
        let mut free: Vec<u32> = (0..900).rev().collect();
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut rand = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..200_000 {
            let key = rand() % 1500;
            let mut writes = Vec::new();
            if reference.contains_key(&key) {
                let removed = index.remove(key, &mut writes).unwrap();
                assert_eq!(Some(removed.record), reference.remove(&key));
                free.push(removed.record);
            } else if let Some(record) = free.pop() {
                let slot = index.insert(key, Resident { record, ..Default::default() });
                writes.push((slot, record));
                reference.insert(key, record);
            }
            for (slot, value) in writes {
                mirror[slot as usize] = value;
                assert_eq!(index.occupied[slot as usize / 64] & (1u64 << (slot % 64)) != 0,
                    index.table[slot as usize] != NONE, "occupancy must follow every shifted/freed slot");
            }
            assert_eq!(index.occupied.iter().map(|word| word.count_ones() as usize).sum::<usize>(), index.len());
        }
        assert_eq!(mirror, index.table);
        assert_eq!(index.len(), reference.len());
        for (key, record) in &reference {
            let r = index.get(*key).unwrap();
            assert_eq!(r.record, *record);
            assert_eq!(index.table[r.slot as usize], *record);
        }
        for key in 0..1500u64 {
            assert_eq!(index.contains_key(key), reference.contains_key(&key));
        }
    }

    #[test]
    fn sparse_retirement_rechecks_word_edges_after_wrapped_backshift_and_refill() {
        let mut index = ColumnIndex::new(8);
        // Collisions cross both a bitmap-word boundary and the table wrap.
        let mut keys = Vec::new();
        for home in [62, 255] {
            keys.extend((0..100_000u64).filter(|&key| index.home(key) == home).take(4));
        }
        assert_eq!(keys.len(), 8);
        for (record, &key) in keys.iter().enumerate() {
            index.insert(key, Resident { record: record as u32, ..Default::default() });
        }
        let keep: std::collections::HashSet<_> = [keys[2], keys[6]].into_iter().collect();
        let mut visited = std::collections::HashSet::new();
        let mut slot = 0;
        let mut steps = 0;
        while slot < index.table.len() {
            steps += 1;
            let (next, resident) = index.retirement_step(slot);
            assert!(next >= slot && next <= ((slot / 64 + 1) * 64), "each step inspects one word only");
            slot = next;
            if let Some((key, _)) = resident {
                visited.insert(key);
                if !keep.contains(&key) {
                    index.remove(key, &mut Vec::new());
                    continue;
                }
                slot += 1;
            }
        }
        assert!(steps <= keys.len() + 4 + keys.len(), "empty slots must skip wordwise");
        assert_eq!(index.iter().map(|(key, _)| key).collect::<std::collections::HashSet<_>>(), keep);
        assert!(keys.iter().all(|key| visited.contains(key)));

        // Reusing a formerly empty slot must restore its bit, including a
        // record reused after retirement. Compare enumeration to the table.
        index.insert(keys[0], Resident { record: 0, ..Default::default() });
        let mut seen = std::collections::HashSet::new();
        let mut slot = 0;
        while slot < index.table.len() {
            let (next, resident) = index.retirement_step(slot);
            slot = next;
            if let Some((key, _)) = resident { seen.insert(key); slot += 1; }
        }
        assert_eq!(seen, index.iter().map(|(key, _)| key).collect());
        assert!(seen.contains(&keys[0]));
    }
}
