use crate::{
    CellWord, ContractError, PageEvict, PageUpload, PlanetPageKey, SourceGeneration,
    VisiblePageSet, PAGE_CELL_BYTES,
};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidencyConfig {
    pub max_resident_pages: usize,
    pub max_cell_bytes: usize,
    pub max_eviction_watermarks: usize,
}

impl ResidencyConfig {
    pub fn new(
        max_resident_pages: usize,
        max_cell_bytes: usize,
        max_eviction_watermarks: usize,
    ) -> Result<Self, ResidencyConfigError> {
        if max_resident_pages == 0 {
            return Err(ResidencyConfigError::ZeroResidentPages);
        }
        if u32::try_from(max_resident_pages).is_err() {
            return Err(ResidencyConfigError::ResidentPageSlotsExceedGpuIndex);
        }
        if max_cell_bytes == 0 {
            return Err(ResidencyConfigError::ZeroCellBytes);
        }
        if max_eviction_watermarks == 0 {
            return Err(ResidencyConfigError::ZeroEvictionWatermarks);
        }
        Ok(Self {
            max_resident_pages,
            max_cell_bytes,
            max_eviction_watermarks,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ResidencyConfigError {
    #[error("planetary residency must provide at least one page slot")]
    ZeroResidentPages,
    #[error("planetary residency page slots exceed the GPU u32 slot address space")]
    ResidentPageSlotsExceedGpuIndex,
    #[error("planetary residency cell-byte budget must be non-zero")]
    ZeroCellBytes,
    #[error("planetary residency must provide at least one eviction watermark")]
    ZeroEvictionWatermarks,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentPage {
    pub slot: u32,
    pub generation: SourceGeneration,
    /// Monotonic token local to this cache lifetime. GPU work uses this token
    /// instead of attempting to pack the two authoritative source generations.
    pub publication_generation: u64,
    pub cells: Box<[CellWord]>,
    last_access: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EvictedPage {
    pub key: PlanetPageKey,
    pub slot: u32,
    pub generation: SourceGeneration,
    pub publication_generation: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UploadOutcome {
    Inserted {
        slot: u32,
        evicted: Vec<EvictedPage>,
    },
    Replaced {
        slot: u32,
        previous_generation: SourceGeneration,
    },
    Duplicate {
        slot: u32,
    },
    Stale {
        newest_generation: SourceGeneration,
    },
    GenerationConflict {
        slot: u32,
    },
    Backpressure(BackpressureReason),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackpressureReason {
    PageExceedsCellByteBudget,
    AllEvictionCandidatesVisible,
    EvictionWatermarkCapacity,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvictOutcome {
    Recorded { removed: Option<EvictedPage> },
    Stale { newest_generation: SourceGeneration },
    Backpressure(BackpressureReason),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VisibilityOutcome {
    Applied {
        resident: usize,
        missing: usize,
        generation_mismatches: usize,
    },
    Duplicate,
    Stale {
        newest_frame: u64,
    },
    FrameConflict,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResidencyCounters {
    pub uploads_inserted: u64,
    pub uploads_replaced: u64,
    pub uploads_duplicate: u64,
    pub uploads_stale: u64,
    pub generation_conflicts: u64,
    pub local_evictions: u64,
    pub authoritative_evictions: u64,
    pub backpressure_events: u64,
    pub invalid_messages: u64,
    pub resident_pages: usize,
    pub resident_cell_bytes: usize,
    pub peak_resident_pages: usize,
    pub peak_resident_cell_bytes: usize,
    pub eviction_watermarks: usize,
}

pub struct ResidentPageCache {
    config: ResidencyConfig,
    occupied_slots: BTreeMap<u32, PlanetPageKey>,
    pages: BTreeMap<PlanetPageKey, ResidentPage>,
    eviction_watermarks: BTreeMap<PlanetPageKey, SourceGeneration>,
    visible: BTreeMap<PlanetPageKey, (SourceGeneration, u8)>,
    last_visible_frame: Option<u64>,
    access_clock: u64,
    publication_clock: u64,
    counters: ResidencyCounters,
}

impl ResidentPageCache {
    pub fn new(config: ResidencyConfig) -> Self {
        Self {
            config,
            occupied_slots: BTreeMap::new(),
            pages: BTreeMap::new(),
            eviction_watermarks: BTreeMap::new(),
            visible: BTreeMap::new(),
            last_visible_frame: None,
            access_clock: 0,
            publication_clock: 0,
            counters: ResidencyCounters::default(),
        }
    }

    pub fn config(&self) -> ResidencyConfig {
        self.config
    }

    pub fn counters(&self) -> ResidencyCounters {
        self.counters
    }

    pub fn resident(&self, key: PlanetPageKey) -> Option<&ResidentPage> {
        self.pages.get(&key)
    }

    pub fn resident_pages(
        &self,
    ) -> impl ExactSizeIterator<Item = (PlanetPageKey, &ResidentPage)> + '_ {
        self.pages.iter().map(|(key, page)| (*key, page))
    }

    pub fn eviction_watermark(&self, key: PlanetPageKey) -> Option<SourceGeneration> {
        self.eviction_watermarks.get(&key).copied()
    }

    pub fn apply_upload(&mut self, upload: PageUpload) -> Result<UploadOutcome, ContractError> {
        if let Err(error) = upload.validate() {
            self.counters.invalid_messages += 1;
            return Err(error);
        }

        if let Some(watermark) = self.eviction_watermarks.get(&upload.key).copied() {
            if upload.generation <= watermark {
                self.counters.uploads_stale += 1;
                return Ok(UploadOutcome::Stale {
                    newest_generation: watermark,
                });
            }
        }

        if let Some(existing) = self.pages.get(&upload.key) {
            if upload.generation < existing.generation {
                let newest_generation = existing.generation;
                self.counters.uploads_stale += 1;
                return Ok(UploadOutcome::Stale { newest_generation });
            }
            if upload.generation == existing.generation {
                if upload.cells == existing.cells {
                    let slot = existing.slot;
                    self.counters.uploads_duplicate += 1;
                    return Ok(UploadOutcome::Duplicate { slot });
                }
                let slot = existing.slot;
                self.counters.generation_conflicts += 1;
                return Ok(UploadOutcome::GenerationConflict { slot });
            }

            let publication_generation = self.next_publication_generation()?;
            let access = self.next_access();
            let existing = self
                .pages
                .get_mut(&upload.key)
                .expect("resident page was just found");
            let previous_generation = existing.generation;
            let slot = existing.slot;
            existing.generation = upload.generation;
            existing.publication_generation = publication_generation;
            existing.cells = upload.cells;
            existing.last_access = access;
            self.counters.uploads_replaced += 1;
            return Ok(UploadOutcome::Replaced {
                slot,
                previous_generation,
            });
        }

        if PAGE_CELL_BYTES > self.config.max_cell_bytes {
            self.counters.backpressure_events += 1;
            return Ok(UploadOutcome::Backpressure(
                BackpressureReason::PageExceedsCellByteBudget,
            ));
        }

        let pages_for_slot = self
            .pages
            .len()
            .saturating_add(1)
            .saturating_sub(self.config.max_resident_pages);
        let bytes_after = self
            .pages
            .len()
            .saturating_add(1)
            .saturating_mul(PAGE_CELL_BYTES);
        let bytes_over = bytes_after.saturating_sub(self.config.max_cell_bytes);
        let pages_for_bytes = bytes_over.div_ceil(PAGE_CELL_BYTES);
        let eviction_count = pages_for_slot.max(pages_for_bytes);

        let mut candidates: Vec<_> = self
            .pages
            .iter()
            .filter(|(key, page)| !self.is_visible(**key, page.generation))
            .map(|(key, page)| (page.last_access, *key))
            .collect();
        candidates.sort_unstable();
        if candidates.len() < eviction_count {
            self.counters.backpressure_events += 1;
            return Ok(UploadOutcome::Backpressure(
                BackpressureReason::AllEvictionCandidatesVisible,
            ));
        }

        let publication_generation = self.next_publication_generation()?;
        let mut evicted = Vec::with_capacity(eviction_count);
        for (_, key) in candidates.into_iter().take(eviction_count) {
            evicted.push(self.remove_resident(key));
            self.counters.local_evictions += 1;
        }

        let slot = (0..self.config.max_resident_pages as u32)
            .find(|slot| !self.occupied_slots.contains_key(slot))
            .expect("budget planning guarantees a free page slot");
        self.occupied_slots.insert(slot, upload.key);
        let access = self.next_access();
        self.pages.insert(
            upload.key,
            ResidentPage {
                slot,
                generation: upload.generation,
                publication_generation,
                cells: upload.cells,
                last_access: access,
            },
        );
        self.counters.uploads_inserted += 1;
        self.refresh_resident_counters();
        Ok(UploadOutcome::Inserted { slot, evicted })
    }

    pub fn apply_evict(&mut self, evict: PageEvict) -> Result<EvictOutcome, ContractError> {
        if let Err(error) = evict.validate() {
            self.counters.invalid_messages += 1;
            return Err(error);
        }

        if let Some(watermark) = self.eviction_watermarks.get(&evict.key).copied() {
            if evict.generation <= watermark {
                self.counters.uploads_stale += 1;
                return Ok(EvictOutcome::Stale {
                    newest_generation: watermark,
                });
            }
        } else if self.eviction_watermarks.len() == self.config.max_eviction_watermarks {
            self.counters.backpressure_events += 1;
            return Ok(EvictOutcome::Backpressure(
                BackpressureReason::EvictionWatermarkCapacity,
            ));
        }

        self.eviction_watermarks.insert(evict.key, evict.generation);
        let should_remove = self
            .pages
            .get(&evict.key)
            .is_some_and(|page| page.generation <= evict.generation);
        let removed = should_remove.then(|| self.remove_resident(evict.key));
        self.counters.authoritative_evictions += 1;
        self.refresh_resident_counters();
        Ok(EvictOutcome::Recorded { removed })
    }

    /// Retires a generation watermark only after the producer has established
    /// that no upload or eviction at or below `through_generation` can still
    /// arrive (normally by draining the relevant queue/fence).
    pub fn retire_eviction_watermark(
        &mut self,
        key: PlanetPageKey,
        through_generation: SourceGeneration,
    ) -> bool {
        let can_retire = self
            .eviction_watermarks
            .get(&key)
            .is_some_and(|watermark| *watermark <= through_generation);
        if can_retire {
            self.eviction_watermarks.remove(&key);
            self.refresh_resident_counters();
        }
        can_retire
    }

    pub fn apply_visible_set(
        &mut self,
        set: VisiblePageSet,
    ) -> Result<VisibilityOutcome, ContractError> {
        if let Err(error) = set.validate(self.config.max_resident_pages) {
            self.counters.invalid_messages += 1;
            return Err(error);
        }
        let canonical: BTreeMap<_, _> = set
            .pages
            .iter()
            .map(|page| (page.key, (page.generation, page.transition_mask)))
            .collect();

        if let Some(newest_frame) = self.last_visible_frame {
            if set.frame_index < newest_frame {
                return Ok(VisibilityOutcome::Stale { newest_frame });
            }
            if set.frame_index == newest_frame {
                return Ok(if canonical == self.visible {
                    VisibilityOutcome::Duplicate
                } else {
                    VisibilityOutcome::FrameConflict
                });
            }
        }

        self.visible = canonical;
        self.last_visible_frame = Some(set.frame_index);
        let access = self.next_access();
        let mut resident = 0;
        let mut missing = 0;
        let mut generation_mismatches = 0;
        for (key, (generation, _)) in &self.visible {
            match self.pages.get_mut(key) {
                Some(page) if page.generation == *generation => {
                    page.last_access = access;
                    resident += 1;
                }
                Some(_) => generation_mismatches += 1,
                None => missing += 1,
            }
        }
        Ok(VisibilityOutcome::Applied {
            resident,
            missing,
            generation_mismatches,
        })
    }

    fn is_visible(&self, key: PlanetPageKey, generation: SourceGeneration) -> bool {
        self.visible
            .get(&key)
            .is_some_and(|(visible_generation, _)| *visible_generation == generation)
    }

    fn next_access(&mut self) -> u64 {
        self.access_clock = self.access_clock.saturating_add(1);
        self.access_clock
    }

    fn next_publication_generation(&mut self) -> Result<u64, ContractError> {
        self.publication_clock = self
            .publication_clock
            .checked_add(1)
            .ok_or(ContractError::PublicationGenerationOverflow)?;
        Ok(self.publication_clock)
    }

    fn remove_resident(&mut self, key: PlanetPageKey) -> EvictedPage {
        let page = self
            .pages
            .remove(&key)
            .expect("evicted page must be resident");
        let removed_slot = self.occupied_slots.remove(&page.slot);
        debug_assert_eq!(removed_slot, Some(key));
        EvictedPage {
            key,
            slot: page.slot,
            generation: page.generation,
            publication_generation: page.publication_generation,
        }
    }

    fn refresh_resident_counters(&mut self) {
        self.counters.resident_pages = self.pages.len();
        self.counters.resident_cell_bytes = self.pages.len().saturating_mul(PAGE_CELL_BYTES);
        self.counters.peak_resident_pages = self
            .counters
            .peak_resident_pages
            .max(self.counters.resident_pages);
        self.counters.peak_resident_cell_bytes = self
            .counters
            .peak_resident_cell_bytes
            .max(self.counters.resident_cell_bytes);
        self.counters.eviction_watermarks = self.eviction_watermarks.len();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PageKey, PlanetId, VisiblePage};

    fn key(index: i64) -> PlanetPageKey {
        PlanetPageKey::new(PlanetId([1; 16]), PageKey::new(0, [index, 0, 0]))
    }

    const fn generation(page: u64) -> SourceGeneration {
        SourceGeneration::new(1, page)
    }

    fn upload(index: i64, generation: u64, cell: CellWord) -> PageUpload {
        PageUpload::new(
            key(index),
            self::generation(generation),
            vec![cell; crate::PAGE_CELL_COUNT],
        )
        .unwrap()
    }

    fn cache(pages: usize, byte_pages: usize, watermarks: usize) -> ResidentPageCache {
        ResidentPageCache::new(
            ResidencyConfig::new(pages, byte_pages * PAGE_CELL_BYTES, watermarks).unwrap(),
        )
    }

    #[test]
    fn configuration_rejects_slots_that_cannot_be_gpu_indexed() {
        let too_many = usize::try_from(u64::from(u32::MAX) + 1);
        if let Ok(too_many) = too_many {
            assert_eq!(
                ResidencyConfig::new(too_many, PAGE_CELL_BYTES, 1),
                Err(ResidencyConfigError::ResidentPageSlotsExceedGpuIndex)
            );
        }
    }

    #[test]
    fn lowest_slot_is_reused_after_deterministic_lru_eviction() {
        let mut cache = cache(2, 2, 2);
        assert!(matches!(
            cache.apply_upload(upload(1, 1, CellWord::AIR)).unwrap(),
            UploadOutcome::Inserted { slot: 0, .. }
        ));
        assert!(matches!(
            cache.apply_upload(upload(2, 1, CellWord::AIR)).unwrap(),
            UploadOutcome::Inserted { slot: 1, .. }
        ));
        let outcome = cache
            .apply_upload(upload(3, 1, CellWord::new(-1, 1, 0)))
            .unwrap();
        assert_eq!(
            outcome,
            UploadOutcome::Inserted {
                slot: 0,
                evicted: vec![EvictedPage {
                    key: key(1),
                    slot: 0,
                    generation: generation(1),
                    publication_generation: 1,
                }],
            }
        );
        assert_eq!(cache.counters().resident_pages, 2);
        assert_eq!(cache.counters().resident_cell_bytes, 2 * PAGE_CELL_BYTES);
    }

    #[test]
    fn visible_pages_are_never_evicted_to_hide_capacity_failure() {
        let mut cache = cache(1, 1, 1);
        cache.apply_upload(upload(1, 4, CellWord::AIR)).unwrap();
        cache
            .apply_visible_set(VisiblePageSet {
                frame_index: 1,
                pages: vec![VisiblePage {
                    key: key(1),
                    generation: generation(4),
                    transition_mask: 0,
                }],
            })
            .unwrap();
        assert_eq!(
            cache.apply_upload(upload(2, 1, CellWord::AIR)).unwrap(),
            UploadOutcome::Backpressure(BackpressureReason::AllEvictionCandidatesVisible)
        );
        assert!(cache.resident(key(1)).is_some());
    }

    #[test]
    fn pending_visibility_protects_a_matching_late_upload() {
        let mut cache = cache(1, 1, 1);
        assert_eq!(
            cache
                .apply_visible_set(VisiblePageSet {
                    frame_index: 1,
                    pages: vec![VisiblePage {
                        key: key(1),
                        generation: generation(4),
                        transition_mask: 0,
                    }],
                })
                .unwrap(),
            VisibilityOutcome::Applied {
                resident: 0,
                missing: 1,
                generation_mismatches: 0,
            }
        );
        cache.apply_upload(upload(1, 4, CellWord::AIR)).unwrap();
        assert_eq!(
            cache.apply_upload(upload(2, 1, CellWord::AIR)).unwrap(),
            UploadOutcome::Backpressure(BackpressureReason::AllEvictionCandidatesVisible)
        );
    }

    #[test]
    fn byte_budget_can_reject_a_page_without_allocating_or_panicking() {
        let config = ResidencyConfig::new(8, PAGE_CELL_BYTES - 1, 1).unwrap();
        let mut cache = ResidentPageCache::new(config);
        assert_eq!(
            cache.apply_upload(upload(0, 0, CellWord::AIR)).unwrap(),
            UploadOutcome::Backpressure(BackpressureReason::PageExceedsCellByteBudget)
        );
        assert_eq!(cache.counters().resident_pages, 0);
    }

    #[test]
    fn same_generation_with_different_data_is_a_conflict() {
        let mut cache = cache(1, 1, 1);
        cache.apply_upload(upload(0, 3, CellWord::AIR)).unwrap();
        assert!(matches!(
            cache
                .apply_upload(upload(0, 3, CellWord::new(-1, 1, 0)))
                .unwrap(),
            UploadOutcome::GenerationConflict { slot: 0 }
        ));
        assert_eq!(cache.resident(key(0)).unwrap().cells[0], CellWord::AIR);
    }

    #[test]
    fn replacement_planet_generation_dominates_retired_page_counters() {
        let mut cache = cache(1, 1, 2);
        cache
            .apply_upload(
                PageUpload::new(
                    key(0),
                    SourceGeneration::new(1, u64::MAX),
                    vec![CellWord::AIR; crate::PAGE_CELL_COUNT],
                )
                .unwrap(),
            )
            .unwrap();
        assert_eq!(cache.resident(key(0)).unwrap().publication_generation, 1);

        assert!(matches!(
            cache
                .apply_upload(
                    PageUpload::new(
                        key(0),
                        SourceGeneration::new(2, 0),
                        vec![CellWord::new(-1, 7, 0); crate::PAGE_CELL_COUNT],
                    )
                    .unwrap()
                )
                .unwrap(),
            UploadOutcome::Replaced { .. }
        ));
        let replacement = cache.resident(key(0)).unwrap();
        assert_eq!(replacement.generation, SourceGeneration::new(2, 0));
        assert_eq!(replacement.publication_generation, 2);

        assert_eq!(
            cache
                .apply_upload(
                    PageUpload::new(
                        key(0),
                        SourceGeneration::new(1, u64::MAX),
                        vec![CellWord::AIR; crate::PAGE_CELL_COUNT],
                    )
                    .unwrap()
                )
                .unwrap(),
            UploadOutcome::Stale {
                newest_generation: SourceGeneration::new(2, 0),
            }
        );
        assert_eq!(
            cache
                .apply_evict(PageEvict {
                    key: key(0),
                    generation: SourceGeneration::new(1, u64::MAX),
                })
                .unwrap(),
            EvictOutcome::Recorded { removed: None }
        );
        assert_eq!(
            cache.resident(key(0)).unwrap().generation,
            SourceGeneration::new(2, 0)
        );
    }

    #[test]
    fn duplicates_reuse_and_changed_sources_advance_publication_generation() {
        let mut cache = cache(1, 1, 1);
        cache.apply_upload(upload(0, 1, CellWord::AIR)).unwrap();
        let first = cache.resident(key(0)).unwrap().publication_generation;
        assert!(matches!(
            cache.apply_upload(upload(0, 1, CellWord::AIR)).unwrap(),
            UploadOutcome::Duplicate { .. }
        ));
        assert_eq!(
            cache.resident(key(0)).unwrap().publication_generation,
            first
        );
        cache
            .apply_upload(upload(0, 2, CellWord::new(-1, 1, 0)))
            .unwrap();
        assert_eq!(
            cache.resident(key(0)).unwrap().publication_generation,
            first + 1
        );
    }

    #[test]
    fn eviction_before_upload_blocks_late_generation() {
        let mut cache = cache(1, 1, 1);
        assert_eq!(
            cache
                .apply_evict(PageEvict {
                    key: key(7),
                    generation: generation(9),
                })
                .unwrap(),
            EvictOutcome::Recorded { removed: None }
        );
        assert_eq!(
            cache.apply_upload(upload(7, 9, CellWord::AIR)).unwrap(),
            UploadOutcome::Stale {
                newest_generation: generation(9),
            }
        );
        assert!(matches!(
            cache.apply_upload(upload(7, 10, CellWord::AIR)).unwrap(),
            UploadOutcome::Inserted { slot: 0, .. }
        ));
    }

    #[test]
    fn authoritative_eviction_only_removes_covered_generations() {
        let mut cache = cache(1, 1, 2);
        cache.apply_upload(upload(1, 10, CellWord::AIR)).unwrap();
        assert_eq!(
            cache
                .apply_evict(PageEvict {
                    key: key(1),
                    generation: generation(9),
                })
                .unwrap(),
            EvictOutcome::Recorded { removed: None }
        );
        assert_eq!(cache.resident(key(1)).unwrap().generation, generation(10));
        assert!(matches!(
            cache
                .apply_evict(PageEvict {
                    key: key(1),
                    generation: generation(10),
                })
                .unwrap(),
            EvictOutcome::Recorded {
                removed: Some(EvictedPage { generation: value, .. })
            } if value == generation(10)
        ));
        assert!(cache.resident(key(1)).is_none());
    }

    #[test]
    fn local_budget_eviction_allows_same_generation_rebuild() {
        let mut cache = cache(1, 1, 1);
        cache.apply_upload(upload(1, 3, CellWord::AIR)).unwrap();
        cache.apply_upload(upload(2, 1, CellWord::AIR)).unwrap();
        assert!(matches!(
            cache.apply_upload(upload(1, 3, CellWord::AIR)).unwrap(),
            UploadOutcome::Inserted { .. }
        ));
        assert_eq!(cache.eviction_watermark(key(1)), None);
    }

    #[test]
    fn eviction_watermarks_are_bounded_and_explicitly_retired() {
        let mut cache = cache(1, 1, 1);
        cache
            .apply_evict(PageEvict {
                key: key(1),
                generation: generation(2),
            })
            .unwrap();
        assert_eq!(
            cache
                .apply_evict(PageEvict {
                    key: key(2),
                    generation: generation(1),
                })
                .unwrap(),
            EvictOutcome::Backpressure(BackpressureReason::EvictionWatermarkCapacity)
        );
        assert!(!cache.retire_eviction_watermark(key(1), generation(1)));
        assert!(cache.retire_eviction_watermark(key(1), generation(2)));
        assert!(matches!(
            cache
                .apply_evict(PageEvict {
                    key: key(2),
                    generation: generation(1),
                })
                .unwrap(),
            EvictOutcome::Recorded { .. }
        ));
        assert_eq!(cache.counters().eviction_watermarks, 1);
    }

    #[test]
    fn visibility_frames_cannot_roll_back_or_change_in_place() {
        let mut cache = cache(2, 2, 1);
        let page = VisiblePage {
            key: key(0),
            generation: generation(1),
            transition_mask: 0,
        };
        let set = VisiblePageSet {
            frame_index: 5,
            pages: vec![page],
        };
        assert!(matches!(
            cache.apply_visible_set(set.clone()).unwrap(),
            VisibilityOutcome::Applied { .. }
        ));
        assert_eq!(
            cache.apply_visible_set(set).unwrap(),
            VisibilityOutcome::Duplicate
        );
        assert_eq!(
            cache
                .apply_visible_set(VisiblePageSet {
                    frame_index: 5,
                    pages: Vec::new(),
                })
                .unwrap(),
            VisibilityOutcome::FrameConflict
        );
        assert_eq!(
            cache
                .apply_visible_set(VisiblePageSet {
                    frame_index: 4,
                    pages: Vec::new(),
                })
                .unwrap(),
            VisibilityOutcome::Stale { newest_frame: 5 }
        );
    }
}
