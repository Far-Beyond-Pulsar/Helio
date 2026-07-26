use bytemuck::{Pod, Zeroable};
use helio_planet_voxel_core::{AddressError, PageKey, PlanetId, PlanetPageKey};

pub const PAGE_TABLE_EMPTY: u32 = 0;
pub const PAGE_TABLE_OCCUPIED: u32 = 1;
pub const PAGE_TABLE_TOMBSTONE: u32 = 2;

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuPageTableEntry {
    pub planet_id: [u32; 4],
    pub relative_lod0_cell_min: [i32; 3],
    pub lod: u32,
    pub slot: u32,
    pub generation_low: u32,
    pub generation_high: u32,
    pub state: u32,
}

impl GpuPageTableEntry {
    pub fn occupied(key: GpuLookupKey, slot: u32, generation: u64) -> GpuPageTableEntry {
        Self {
            planet_id: key.planet_id,
            relative_lod0_cell_min: key.relative_lod0_cell_min,
            lod: key.lod,
            slot,
            generation_low: generation as u32,
            generation_high: (generation >> 32) as u32,
            state: PAGE_TABLE_OCCUPIED,
        }
    }

    pub const fn tombstone() -> Self {
        Self {
            planet_id: [0; 4],
            relative_lod0_cell_min: [0; 3],
            lod: 0,
            slot: 0,
            generation_low: 0,
            generation_high: 0,
            state: PAGE_TABLE_TOMBSTONE,
        }
    }

    pub const fn generation(self) -> u64 {
        self.generation_low as u64 | ((self.generation_high as u64) << 32)
    }

    pub const fn key(self) -> GpuLookupKey {
        GpuLookupKey {
            planet_id: self.planet_id,
            relative_lod0_cell_min: self.relative_lod0_cell_min,
            lod: self.lod,
        }
    }

    pub const fn is_occupied(self) -> bool {
        self.state == PAGE_TABLE_OCCUPIED
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuResidencyUniform {
    pub table_mask: u32,
    pub max_probe: u32,
    pub resident_pages: u32,
    pub _pad: u32,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuResidencyCounters {
    pub resident_pages: u32,
    pub resident_cell_bytes_low: u32,
    pub resident_cell_bytes_high: u32,
    pub table_occupied: u32,
    pub table_tombstones: u32,
    pub uploads_published: u32,
    pub evictions_published: u32,
    pub stale_rejections: u32,
    pub generation_conflicts: u32,
    pub backpressure_events: u32,
    pub table_saturation_events: u32,
    pub batches_submitted: u32,
    pub cell_bytes_uploaded_low: u32,
    pub cell_bytes_uploaded_high: u32,
    pub peak_resident_pages: u32,
    pub peak_resident_cell_bytes_low: u32,
    pub peak_resident_cell_bytes_high: u32,
    pub allocated_gpu_bytes_low: u32,
    pub allocated_gpu_bytes_high: u32,
    pub resource_buffers: u32,
    pub atlas_shards: u32,
    pub device_rebuilds: u32,
    pub _pad: [u32; 2],
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuLookupQuery {
    pub planet_id: [u32; 4],
    pub relative_lod0_cell_min: [i32; 3],
    pub lod: u32,
}

impl From<GpuLookupKey> for GpuLookupQuery {
    fn from(key: GpuLookupKey) -> Self {
        Self {
            planet_id: key.planet_id,
            relative_lod0_cell_min: key.relative_lod0_cell_min,
            lod: key.lod,
        }
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuLookupResult {
    pub slot: u32,
    pub generation_low: u32,
    pub generation_high: u32,
    pub probes_and_found: u32,
}

impl GpuLookupResult {
    pub const fn found(self) -> bool {
        self.probes_and_found & 0x8000_0000 != 0
    }

    pub const fn probes(self) -> u32 {
        self.probes_and_found & 0x7fff_ffff
    }

    pub const fn generation(self) -> u64 {
        self.generation_low as u64 | ((self.generation_high as u64) << 32)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GpuLookupKey {
    pub planet_id: [u32; 4],
    pub relative_lod0_cell_min: [i32; 3],
    pub lod: u32,
}

impl GpuLookupKey {
    pub fn from_planet_page(
        key: PlanetPageKey,
        frame_origin_lod0_cell: [i64; 3],
    ) -> Result<Self, AddressError> {
        Ok(Self {
            planet_id: planet_words(key.planet),
            relative_lod0_cell_min: key.page.relative_lod0_cell_min(frame_origin_lod0_cell)?,
            lod: u32::from(key.page.lod),
        })
    }

    pub fn from_parts(
        planet: PlanetId,
        page: PageKey,
        frame_origin_lod0_cell: [i64; 3],
    ) -> Result<Self, AddressError> {
        Self::from_planet_page(PlanetPageKey::new(planet, page), frame_origin_lod0_cell)
    }

    pub fn hash(self) -> u32 {
        let mut hash = 0x811c_9dc5;
        for value in self
            .planet_id
            .into_iter()
            .chain(self.relative_lod0_cell_min.map(|value| value as u32))
            .chain([self.lod])
        {
            hash = mix_hash(hash, value);
        }
        hash
    }
}

pub const fn mix_hash(hash: u32, value: u32) -> u32 {
    let mut mixed = hash ^ value;
    mixed = mixed.wrapping_mul(0x045d_9f3b);
    mixed ^ (mixed >> 16)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PageTable {
    entries: Vec<GpuPageTableEntry>,
    max_probe: u32,
    occupied: u32,
    tombstones: u32,
}

impl PageTable {
    pub fn new(capacity: u32, max_probe: u32) -> Result<Self, PageTableError> {
        if !capacity.is_power_of_two() {
            return Err(PageTableError::CapacityNotPowerOfTwo(capacity));
        }
        if max_probe == 0 || max_probe > capacity {
            return Err(PageTableError::InvalidMaxProbe {
                max_probe,
                capacity,
            });
        }
        Ok(Self {
            entries: vec![GpuPageTableEntry::default(); capacity as usize],
            max_probe,
            occupied: 0,
            tombstones: 0,
        })
    }

    pub fn entries(&self) -> &[GpuPageTableEntry] {
        &self.entries
    }

    pub fn capacity(&self) -> u32 {
        self.entries.len() as u32
    }

    pub fn max_probe(&self) -> u32 {
        self.max_probe
    }

    pub fn occupied(&self) -> u32 {
        self.occupied
    }

    pub fn tombstones(&self) -> u32 {
        self.tombstones
    }

    pub fn can_insert(&self, key: GpuLookupKey) -> bool {
        matches!(
            self.find_slot(key),
            ProbeResult::Found(_) | ProbeResult::Vacant(_)
        )
    }

    pub fn insert(&mut self, entry: GpuPageTableEntry) -> Result<u32, PageTableError> {
        if !entry.is_occupied() {
            return Err(PageTableError::EntryNotOccupied(entry.state));
        }
        match self.find_slot(entry.key()) {
            ProbeResult::Found(index) => {
                self.entries[index as usize] = entry;
                Ok(index)
            }
            ProbeResult::Vacant(index) => {
                if self.entries[index as usize].state == PAGE_TABLE_TOMBSTONE {
                    self.tombstones -= 1;
                }
                self.entries[index as usize] = entry;
                self.occupied += 1;
                Ok(index)
            }
            ProbeResult::Saturated => Err(PageTableError::ProbeSaturated {
                hash: entry.key().hash(),
                max_probe: self.max_probe,
            }),
        }
    }

    pub fn remove(&mut self, key: GpuLookupKey) -> Option<GpuPageTableEntry> {
        let ProbeResult::Found(index) = self.find_slot(key) else {
            return None;
        };
        let removed = self.entries[index as usize];
        self.entries[index as usize] = GpuPageTableEntry::tombstone();
        self.occupied -= 1;
        self.tombstones += 1;
        Some(removed)
    }

    pub fn lookup(&self, key: GpuLookupKey) -> Option<(u32, GpuPageTableEntry)> {
        let ProbeResult::Found(index) = self.find_slot(key) else {
            return None;
        };
        Some((index, self.entries[index as usize]))
    }

    pub fn compact(&mut self) -> Result<(), PageTableError> {
        let occupied: Vec<_> = self
            .entries
            .iter()
            .copied()
            .filter(|entry| entry.is_occupied())
            .collect();
        let mut rebuilt = Self::new(self.capacity(), self.max_probe)?;
        for entry in occupied {
            rebuilt.insert(entry)?;
        }
        *self = rebuilt;
        Ok(())
    }

    fn find_slot(&self, key: GpuLookupKey) -> ProbeResult {
        let mask = self.capacity() - 1;
        let start = key.hash() & mask;
        let mut first_tombstone = None;
        for probe in 0..self.max_probe {
            let index = start.wrapping_add(probe) & mask;
            let entry = self.entries[index as usize];
            match entry.state {
                PAGE_TABLE_EMPTY => {
                    return ProbeResult::Vacant(first_tombstone.unwrap_or(index));
                }
                PAGE_TABLE_TOMBSTONE if first_tombstone.is_none() => {
                    first_tombstone = Some(index);
                }
                PAGE_TABLE_OCCUPIED if entry.key() == key => {
                    return ProbeResult::Found(index);
                }
                _ => {}
            }
        }
        first_tombstone.map_or(ProbeResult::Saturated, ProbeResult::Vacant)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProbeResult {
    Found(u32),
    Vacant(u32),
    Saturated,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PageTableError {
    #[error("page table capacity {0} must be a non-zero power of two")]
    CapacityNotPowerOfTwo(u32),
    #[error("maximum probe count {max_probe} must be within 1..={capacity}")]
    InvalidMaxProbe { max_probe: u32, capacity: u32 },
    #[error("page table insertion requires occupied state, got {0}")]
    EntryNotOccupied(u32),
    #[error("page table probe saturated for hash {hash:#010x} after {max_probe} probes")]
    ProbeSaturated { hash: u32, max_probe: u32 },
}

fn planet_words(planet: PlanetId) -> [u32; 4] {
    let mut words = [0_u32; 4];
    for (word, bytes) in words.iter_mut().zip(planet.0.chunks_exact(4)) {
        *word = u32::from_le_bytes(bytes.try_into().expect("four-byte chunk"));
    }
    words
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(index: i32) -> GpuLookupKey {
        GpuLookupKey {
            planet_id: [1, 2, 3, 4],
            relative_lod0_cell_min: [index, -index, index.wrapping_mul(17)],
            lod: index.unsigned_abs() % 8,
        }
    }

    #[test]
    fn signed_multi_planet_keys_do_not_false_hit() {
        let mut table = PageTable::new(32, 32).unwrap();
        for index in -8..8 {
            let lookup = key(index);
            table
                .insert(GpuPageTableEntry::occupied(
                    lookup,
                    (index + 8) as u32,
                    index.unsigned_abs() as u64 + 10,
                ))
                .unwrap();
        }
        for index in -8..8 {
            let (_, entry) = table.lookup(key(index)).unwrap();
            assert_eq!(entry.slot, (index + 8) as u32);
        }
        let mut other_planet = key(2);
        other_planet.planet_id[3] ^= 1;
        assert_eq!(table.lookup(other_planet), None);
    }

    #[test]
    fn tombstones_keep_probe_chains_and_are_reused() {
        let mut collision = None;
        'outer: for first in -200..200 {
            for second in first + 1..200 {
                if key(first).hash() & 7 == key(second).hash() & 7 {
                    collision = Some((key(first), key(second)));
                    break 'outer;
                }
            }
        }
        let (first, second) = collision.unwrap();
        let mut table = PageTable::new(8, 8).unwrap();
        table
            .insert(GpuPageTableEntry::occupied(first, 1, 1))
            .unwrap();
        table
            .insert(GpuPageTableEntry::occupied(second, 2, 2))
            .unwrap();
        table.remove(first).unwrap();
        assert_eq!(table.lookup(second).unwrap().1.slot, 2);
        table
            .insert(GpuPageTableEntry::occupied(first, 3, 3))
            .unwrap();
        assert_eq!(table.tombstones(), 0);
        assert_eq!(table.lookup(first).unwrap().1.slot, 3);
    }

    #[test]
    fn bounded_probe_reports_saturation_without_mutation() {
        let mut same_bucket = Vec::new();
        for index in -10_000..10_000 {
            let candidate = key(index);
            if candidate.hash() & 15 == 0 {
                same_bucket.push(candidate);
                if same_bucket.len() == 3 {
                    break;
                }
            }
        }
        let mut table = PageTable::new(16, 2).unwrap();
        for (slot, lookup) in same_bucket.iter().take(2).enumerate() {
            table
                .insert(GpuPageTableEntry::occupied(*lookup, slot as u32, 1))
                .unwrap();
        }
        assert!(!table.can_insert(same_bucket[2]));
        let before = table.clone();
        assert!(matches!(
            table.insert(GpuPageTableEntry::occupied(same_bucket[2], 9, 1)),
            Err(PageTableError::ProbeSaturated { .. })
        ));
        assert_eq!(table, before);
    }

    #[test]
    fn compaction_removes_tombstones_without_losing_entries() {
        let mut table = PageTable::new(32, 32).unwrap();
        for index in 0..12 {
            table
                .insert(GpuPageTableEntry::occupied(key(index), index as u32, 4))
                .unwrap();
        }
        for index in [1, 3, 5, 7] {
            table.remove(key(index)).unwrap();
        }
        assert_eq!(table.tombstones(), 4);
        table.compact().unwrap();
        assert_eq!(table.tombstones(), 0);
        for index in [0, 2, 4, 6, 8, 9, 10, 11] {
            assert!(table.lookup(key(index)).is_some());
        }
    }
}
