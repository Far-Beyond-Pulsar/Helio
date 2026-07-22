use crate::{
    GpuAllocationPlan, GpuConfigError, GpuLookupKey, GpuPageTableEntry, GpuResidencyCounters,
    GpuResidencyUniform, PageTable, PageTableError, PlanetaryVoxelGpuConfig,
};
use helio_planet_voxel_core::{
    AddressError, CellWord, ContractError, EvictOutcome, GpuPageMeta, GpuPageMetaError,
    PAGE_CELL_BYTES, PAGE_CELL_COUNT, PageEvict, PageUpload, PlanetFrameUniform, PlanetId,
    PlanetPageKey, ResidentPageCache, SourceGeneration, UploadOutcome, VisibilityOutcome,
    VisiblePageSet,
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameUpdateOutcome {
    Applied { previous_frame: Option<u64> },
    Duplicate,
    Stale { newest_frame: u64 },
    FrameConflict,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GpuUploadOutcome {
    Residency(UploadOutcome),
    PageTableBackpressure,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuResourceStats {
    pub buffers: u32,
    pub atlas_shards: u32,
    pub allocated_bytes: u64,
}

pub struct PlanetaryVoxelResidency {
    config: PlanetaryVoxelGpuConfig,
    plan: GpuAllocationPlan,
    cache: ResidentPageCache,
    frames: BTreeMap<PlanetId, PlanetFrameUniform>,
    visible: BTreeMap<PlanetPageKey, (SourceGeneration, u8)>,
    table: PageTable,
    published_table: Vec<GpuPageTableEntry>,
    published_metadata: Vec<GpuPageMeta>,
    resources: GpuResidencyResources,
    counters: GpuResidencyCounters,
    cell_bytes_uploaded: u64,
}

impl PlanetaryVoxelResidency {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: PlanetaryVoxelGpuConfig,
    ) -> Result<Self, GpuResidencyError> {
        let plan = config.allocation_plan(&device.limits())?;
        let cache = ResidentPageCache::new(config.residency_config()?);
        let table = PageTable::new(config.table_capacity, config.max_probe)?;
        let resources = GpuResidencyResources::new(device, &plan);
        let mut residency = Self {
            config,
            plan,
            cache,
            frames: BTreeMap::new(),
            visible: BTreeMap::new(),
            table,
            published_table: vec![GpuPageTableEntry::default(); config.table_capacity as usize],
            published_metadata: vec![GpuPageMeta::default(); config.max_resident_pages as usize],
            resources,
            counters: GpuResidencyCounters::default(),
            cell_bytes_uploaded: 0,
        };
        residency.publish_state(queue, true)?;
        Ok(residency)
    }

    pub fn config(&self) -> PlanetaryVoxelGpuConfig {
        self.config
    }

    pub fn allocation_plan(&self) -> &GpuAllocationPlan {
        &self.plan
    }

    pub fn cache(&self) -> &ResidentPageCache {
        &self.cache
    }

    pub fn counters(&self) -> GpuResidencyCounters {
        self.counters
    }

    pub fn page_table(&self) -> &PageTable {
        &self.table
    }

    pub fn planet_frame_count(&self) -> usize {
        self.frames.len()
    }

    pub fn planet_frame(&self, planet: PlanetId) -> Option<PlanetFrameUniform> {
        self.frames.get(&planet).copied()
    }

    pub fn resource_stats(&self) -> GpuResourceStats {
        GpuResourceStats {
            buffers: self.resources.atlas_shards.len() as u32 + 5,
            atlas_shards: self.resources.atlas_shards.len() as u32,
            allocated_bytes: self.plan.total_bytes,
        }
    }

    pub fn atlas_buffers(&self) -> impl ExactSizeIterator<Item = &wgpu::Buffer> {
        self.resources.atlas_shards.iter()
    }

    pub fn metadata_buffer(&self) -> &wgpu::Buffer {
        &self.resources.metadata
    }

    pub fn page_table_buffer(&self) -> &wgpu::Buffer {
        &self.resources.page_table
    }

    pub fn residency_uniform_buffer(&self) -> &wgpu::Buffer {
        &self.resources.uniform
    }

    pub fn counters_buffer(&self) -> &wgpu::Buffer {
        &self.resources.counters
    }

    /// Residency has no size-dependent resources. A surface resize therefore
    /// deliberately leaves every allocation and generation untouched.
    pub fn resize(&mut self, _width: u32, _height: u32) {}

    pub fn set_planet_frame(
        &mut self,
        queue: &wgpu::Queue,
        frame: PlanetFrameUniform,
    ) -> Result<FrameUpdateOutcome, GpuResidencyError> {
        let planet = frame.planet_id();
        let frame_number = frame.frame_number();
        if !self.frames.contains_key(&planet)
            && self.frames.len() == self.config.max_resident_pages as usize
        {
            return Err(GpuResidencyError::PlanetFrameCapacity {
                maximum: self.config.max_resident_pages,
            });
        }
        if let Some(current) = self.frames.get(&planet).copied() {
            let current_number = current.frame_number();
            if frame_number < current_number {
                return Ok(FrameUpdateOutcome::Stale {
                    newest_frame: current_number,
                });
            }
            if frame_number == current_number {
                return Ok(if current == frame {
                    FrameUpdateOutcome::Duplicate
                } else {
                    FrameUpdateOutcome::FrameConflict
                });
            }
        }

        let previous_frame = self.frames.get(&planet).map(|value| value.frame_number());
        let mut candidate_frames = self.frames.clone();
        candidate_frames.insert(planet, frame);
        let candidate_table = self.build_table(&candidate_frames)?;
        let candidate_metadata = self.build_metadata(&candidate_frames)?;
        self.frames = candidate_frames;
        self.table = candidate_table;
        self.publish_metadata(queue, &candidate_metadata, false);
        self.publish_table(queue, false);
        self.refresh_and_publish_counters(queue);
        Ok(FrameUpdateOutcome::Applied { previous_frame })
    }

    /// Releases a frame only when no resident or visible page can still refer
    /// to it. This keeps multi-planet churn bounded without invalidating GPU
    /// addresses that are already published.
    pub fn remove_planet_frame(&mut self, planet: PlanetId) -> Result<bool, GpuResidencyError> {
        if self
            .cache
            .resident_pages()
            .any(|(key, _)| key.planet == planet)
            || self.visible.keys().any(|key| key.planet == planet)
        {
            return Err(GpuResidencyError::PlanetFrameInUse(planet));
        }
        Ok(self.frames.remove(&planet).is_some())
    }

    pub fn apply_upload_batch(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        uploads: Vec<PageUpload>,
    ) -> Result<Vec<GpuUploadOutcome>, GpuResidencyError> {
        self.check_batch_capacity(uploads.len())?;
        let mut lookup_keys = Vec::with_capacity(uploads.len());
        for upload in &uploads {
            upload.validate()?;
            lookup_keys.push(self.lookup_key(upload.key)?);
        }

        let mut dirty_slots = BTreeSet::new();
        let mut outcomes = Vec::with_capacity(uploads.len());
        for (upload, lookup_key) in uploads.into_iter().zip(lookup_keys) {
            let page_key = upload.key;
            let mut candidate_table = self.table.clone();
            // Capacity probing does not publish this placeholder. The real
            // entry below receives the cache-assigned renderer-local token.
            let placeholder = GpuPageTableEntry::occupied(lookup_key, 0, 0);
            if candidate_table.insert(placeholder).is_err() {
                candidate_table.compact()?;
                if candidate_table.insert(placeholder).is_err() {
                    self.counters.table_saturation_events =
                        self.counters.table_saturation_events.saturating_add(1);
                    self.counters.backpressure_events =
                        self.counters.backpressure_events.saturating_add(1);
                    outcomes.push(GpuUploadOutcome::PageTableBackpressure);
                    continue;
                }
            }

            let outcome = self.cache.apply_upload(upload)?;
            match &outcome {
                UploadOutcome::Inserted { slot, evicted } => {
                    for removed in evicted {
                        candidate_table.remove(self.lookup_key(removed.key)?);
                        dirty_slots.insert(removed.slot);
                    }
                    candidate_table.insert(GpuPageTableEntry::occupied(
                        lookup_key,
                        *slot,
                        self.cache
                            .resident(page_key)
                            .map(|page| page.publication_generation)
                            .ok_or(GpuResidencyError::ResidentPageMissing)?,
                    ))?;
                    dirty_slots.insert(*slot);
                    self.table = candidate_table;
                    self.counters.uploads_published =
                        self.counters.uploads_published.saturating_add(1);
                }
                UploadOutcome::Replaced { slot, .. } => {
                    let publication_generation = self
                        .cache
                        .resident(page_key)
                        .map(|page| page.publication_generation)
                        .ok_or(GpuResidencyError::ResidentPageMissing)?;
                    candidate_table.insert(GpuPageTableEntry::occupied(
                        lookup_key,
                        *slot,
                        publication_generation,
                    ))?;
                    dirty_slots.insert(*slot);
                    self.table = candidate_table;
                    self.counters.uploads_published =
                        self.counters.uploads_published.saturating_add(1);
                }
                UploadOutcome::Stale { .. } => {
                    self.counters.stale_rejections =
                        self.counters.stale_rejections.saturating_add(1);
                }
                UploadOutcome::GenerationConflict { .. } => {
                    self.counters.generation_conflicts =
                        self.counters.generation_conflicts.saturating_add(1);
                }
                UploadOutcome::Backpressure(_) => {
                    self.counters.backpressure_events =
                        self.counters.backpressure_events.saturating_add(1);
                }
                UploadOutcome::Duplicate { .. } => {}
            }
            outcomes.push(GpuUploadOutcome::Residency(outcome));
        }

        // The cell-copy submission is placed on the queue before any table or
        // metadata writes below. A later consumer submission therefore cannot
        // observe a generation whose complete cell page is not already ahead
        // of it on the same queue timeline.
        self.publish_dirty_slots(device, queue, &dirty_slots)?;
        let metadata = self.build_metadata(&self.frames)?;
        self.publish_metadata(queue, &metadata, false);
        self.publish_table(queue, false);
        self.refresh_and_publish_counters(queue);
        Ok(outcomes)
    }

    pub fn apply_evict_batch(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        evictions: Vec<PageEvict>,
    ) -> Result<Vec<EvictOutcome>, GpuResidencyError> {
        self.check_batch_capacity(evictions.len())?;
        for eviction in &evictions {
            eviction.validate()?;
        }
        let mut dirty_slots = BTreeSet::new();
        let mut outcomes = Vec::with_capacity(evictions.len());
        for eviction in evictions {
            let outcome = self.cache.apply_evict(eviction)?;
            match outcome {
                EvictOutcome::Recorded { removed } => {
                    if let Some(removed) = removed {
                        self.table.remove(self.lookup_key(removed.key)?);
                        dirty_slots.insert(removed.slot);
                        self.counters.evictions_published =
                            self.counters.evictions_published.saturating_add(1);
                    }
                }
                EvictOutcome::Stale { .. } => {
                    self.counters.stale_rejections =
                        self.counters.stale_rejections.saturating_add(1);
                }
                EvictOutcome::Backpressure(_) => {
                    self.counters.backpressure_events =
                        self.counters.backpressure_events.saturating_add(1);
                }
            }
            outcomes.push(outcome);
        }
        if self.table.tombstones() > self.table.occupied()
            || self.table.tombstones() > self.table.capacity() / 4
        {
            self.table.compact()?;
        }
        // Poisoning removed slots precedes removing their discoverable table
        // entries on the queue timeline, matching the upload publication rule.
        self.publish_dirty_slots(device, queue, &dirty_slots)?;
        let metadata = self.build_metadata(&self.frames)?;
        self.publish_metadata(queue, &metadata, false);
        self.publish_table(queue, false);
        self.refresh_and_publish_counters(queue);
        Ok(outcomes)
    }

    pub fn apply_visible_set(
        &mut self,
        queue: &wgpu::Queue,
        set: VisiblePageSet,
    ) -> Result<VisibilityOutcome, GpuResidencyError> {
        set.validate(self.cache.config().max_resident_pages)?;
        let canonical: BTreeMap<_, _> = set
            .pages
            .iter()
            .map(|page| (page.key, (page.generation, page.transition_mask)))
            .collect();
        let outcome = self.cache.apply_visible_set(set)?;
        if matches!(outcome, VisibilityOutcome::Applied { .. }) {
            self.visible = canonical;
            let metadata = self.build_metadata(&self.frames)?;
            self.publish_metadata(queue, &metadata, false);
        }
        self.refresh_and_publish_counters(queue);
        Ok(outcome)
    }

    pub fn retire_eviction_watermark(
        &mut self,
        key: PlanetPageKey,
        through_generation: SourceGeneration,
    ) -> bool {
        self.cache
            .retire_eviction_watermark(key, through_generation)
    }

    pub fn compact_page_table(&mut self, queue: &wgpu::Queue) -> Result<(), GpuResidencyError> {
        self.table.compact()?;
        self.publish_table(queue, false);
        self.refresh_and_publish_counters(queue);
        Ok(())
    }

    pub fn recreate_gpu_resources(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<(), GpuResidencyError> {
        let plan = self.config.allocation_plan(&device.limits())?;
        let resources = GpuResidencyResources::new(device, &plan);
        self.plan = plan;
        self.resources = resources;
        self.published_table.fill(GpuPageTableEntry::default());
        self.published_metadata.fill(GpuPageMeta::default());

        let slots: Vec<_> = self
            .cache
            .resident_pages()
            .map(|(_, page)| page.slot)
            .collect();
        for chunk in slots.chunks(self.config.max_batch_pages as usize) {
            let dirty_slots = chunk.iter().copied().collect();
            self.publish_dirty_slots(device, queue, &dirty_slots)?;
        }
        self.counters.device_rebuilds = self.counters.device_rebuilds.saturating_add(1);
        let metadata = self.build_metadata(&self.frames)?;
        self.publish_metadata(queue, &metadata, true);
        self.publish_table(queue, true);
        self.refresh_and_publish_counters(queue);
        Ok(())
    }

    fn check_batch_capacity(&self, actual: usize) -> Result<(), GpuResidencyError> {
        if actual > self.config.max_batch_pages as usize {
            return Err(GpuResidencyError::BatchCapacity {
                actual,
                maximum: self.config.max_batch_pages,
            });
        }
        Ok(())
    }

    fn lookup_key(&self, key: PlanetPageKey) -> Result<GpuLookupKey, GpuResidencyError> {
        let frame = self
            .frames
            .get(&key.planet)
            .ok_or(GpuResidencyError::MissingPlanetFrame(key.planet))?;
        Ok(GpuLookupKey::from_planet_page(
            key,
            frame.frame_origin_lod0_cell(),
        )?)
    }

    fn build_table(
        &self,
        frames: &BTreeMap<PlanetId, PlanetFrameUniform>,
    ) -> Result<PageTable, GpuResidencyError> {
        let mut table = PageTable::new(self.config.table_capacity, self.config.max_probe)?;
        for (key, page) in self.cache.resident_pages() {
            let frame = frames
                .get(&key.planet)
                .ok_or(GpuResidencyError::MissingPlanetFrame(key.planet))?;
            let lookup = GpuLookupKey::from_planet_page(key, frame.frame_origin_lod0_cell())?;
            table.insert(GpuPageTableEntry::occupied(
                lookup,
                page.slot,
                page.publication_generation,
            ))?;
        }
        Ok(table)
    }

    fn build_metadata(
        &self,
        frames: &BTreeMap<PlanetId, PlanetFrameUniform>,
    ) -> Result<Vec<GpuPageMeta>, GpuResidencyError> {
        let mut metadata = vec![GpuPageMeta::default(); self.config.max_resident_pages as usize];
        for (key, page) in self.cache.resident_pages() {
            let frame = frames
                .get(&key.planet)
                .ok_or(GpuResidencyError::MissingPlanetFrame(key.planet))?;
            let transition_mask = self
                .visible
                .get(&key)
                .filter(|(generation, _)| *generation == page.generation)
                .map_or(0, |(_, mask)| *mask);
            metadata[page.slot as usize] = GpuPageMeta::new(
                key.page,
                frame.frame_origin_lod0_cell(),
                page.slot,
                page.publication_generation,
                transition_mask,
            )?;
        }
        Ok(metadata)
    }

    fn publish_dirty_slots(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        dirty_slots: &BTreeSet<u32>,
    ) -> Result<(), GpuResidencyError> {
        if dirty_slots.is_empty() {
            return Ok(());
        }
        if dirty_slots.len() > self.config.max_batch_pages as usize {
            return Err(GpuResidencyError::DirtySlotCapacity {
                actual: dirty_slots.len(),
                maximum: self.config.max_batch_pages,
            });
        }

        let mut cells = Vec::with_capacity(dirty_slots.len() * PAGE_CELL_COUNT);
        let pages_by_slot: BTreeMap<_, _> = self
            .cache
            .resident_pages()
            .map(|(_, page)| (page.slot, page.cells.as_ref()))
            .collect();
        for slot in dirty_slots {
            if let Some(page_cells) = pages_by_slot.get(slot) {
                cells.extend_from_slice(page_cells);
            } else {
                cells.resize(cells.len() + PAGE_CELL_COUNT, CellWord::AIR);
            }
        }
        queue.write_buffer(&self.resources.staging, 0, bytemuck::cast_slice(&cells));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Planetary Voxel Cell Publish"),
        });
        for (staging_page, slot) in dirty_slots.iter().enumerate() {
            let (shard, destination_offset) = self
                .plan
                .shard_for_slot(*slot)
                .ok_or(GpuResidencyError::InvalidSlot(*slot))?;
            encoder.copy_buffer_to_buffer(
                &self.resources.staging,
                staging_page as u64 * PAGE_CELL_BYTES as u64,
                &self.resources.atlas_shards[shard],
                destination_offset,
                PAGE_CELL_BYTES as u64,
            );
        }
        queue.submit([encoder.finish()]);
        self.counters.batches_submitted = self.counters.batches_submitted.saturating_add(1);
        self.cell_bytes_uploaded = self
            .cell_bytes_uploaded
            .saturating_add((dirty_slots.len() as u64).saturating_mul(PAGE_CELL_BYTES as u64));
        Ok(())
    }

    fn publish_state(
        &mut self,
        queue: &wgpu::Queue,
        force_all: bool,
    ) -> Result<(), GpuResidencyError> {
        let metadata = self.build_metadata(&self.frames)?;
        self.publish_metadata(queue, &metadata, force_all);
        self.publish_table(queue, force_all);
        self.refresh_and_publish_counters(queue);
        Ok(())
    }

    fn publish_metadata(&mut self, queue: &wgpu::Queue, new: &[GpuPageMeta], force_all: bool) {
        write_changed_ranges(
            queue,
            &self.resources.metadata,
            &mut self.published_metadata,
            new,
            force_all,
        );
    }

    fn publish_table(&mut self, queue: &wgpu::Queue, force_all: bool) {
        write_changed_ranges(
            queue,
            &self.resources.page_table,
            &mut self.published_table,
            self.table.entries(),
            force_all,
        );
    }

    fn refresh_and_publish_counters(&mut self, queue: &wgpu::Queue) {
        let cache = self.cache.counters();
        self.counters.resident_pages = saturating_u32(cache.resident_pages as u64);
        let resident_bytes = cache.resident_cell_bytes as u64;
        self.counters.resident_cell_bytes_low = resident_bytes as u32;
        self.counters.resident_cell_bytes_high = (resident_bytes >> 32) as u32;
        self.counters.table_occupied = self.table.occupied();
        self.counters.table_tombstones = self.table.tombstones();
        self.counters.cell_bytes_uploaded_low = self.cell_bytes_uploaded as u32;
        self.counters.cell_bytes_uploaded_high = (self.cell_bytes_uploaded >> 32) as u32;
        self.counters.peak_resident_pages = saturating_u32(cache.peak_resident_pages as u64);
        let peak_bytes = cache.peak_resident_cell_bytes as u64;
        self.counters.peak_resident_cell_bytes_low = peak_bytes as u32;
        self.counters.peak_resident_cell_bytes_high = (peak_bytes >> 32) as u32;
        self.counters.allocated_gpu_bytes_low = self.plan.total_bytes as u32;
        self.counters.allocated_gpu_bytes_high = (self.plan.total_bytes >> 32) as u32;
        self.counters.resource_buffers = self.resources.atlas_shards.len() as u32 + 5;
        self.counters.atlas_shards = self.resources.atlas_shards.len() as u32;

        let uniform = GpuResidencyUniform {
            table_mask: self.table.capacity() - 1,
            max_probe: self.table.max_probe(),
            resident_pages: self.counters.resident_pages,
            _pad: 0,
        };
        queue.write_buffer(&self.resources.uniform, 0, bytemuck::bytes_of(&uniform));
        queue.write_buffer(
            &self.resources.counters,
            0,
            bytemuck::bytes_of(&self.counters),
        );
    }
}

struct GpuResidencyResources {
    atlas_shards: Vec<wgpu::Buffer>,
    metadata: wgpu::Buffer,
    page_table: wgpu::Buffer,
    staging: wgpu::Buffer,
    uniform: wgpu::Buffer,
    counters: wgpu::Buffer,
}

impl GpuResidencyResources {
    fn new(device: &wgpu::Device, plan: &GpuAllocationPlan) -> Self {
        let atlas_shards = plan
            .shards
            .iter()
            .enumerate()
            .map(|(index, shard)| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("Planetary Voxel Cell Atlas Shard {index}")),
                    size: shard.size_bytes,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })
            })
            .collect();
        Self {
            atlas_shards,
            metadata: create_buffer(
                device,
                "Planetary Voxel Page Metadata",
                plan.metadata_bytes,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            ),
            page_table: create_buffer(
                device,
                "Planetary Voxel Page Table",
                plan.page_table_bytes,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            ),
            staging: create_buffer(
                device,
                "Planetary Voxel Upload Staging",
                plan.staging_bytes,
                wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            ),
            uniform: create_buffer(
                device,
                "Planetary Voxel Residency Uniform",
                core::mem::size_of::<GpuResidencyUniform>() as u64,
                wgpu::BufferUsages::UNIFORM
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            ),
            counters: create_buffer(
                device,
                "Planetary Voxel Residency Counters",
                core::mem::size_of::<GpuResidencyCounters>() as u64,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            ),
        }
    }
}

fn create_buffer(
    device: &wgpu::Device,
    label: &'static str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}

fn write_changed_ranges<T: bytemuck::Pod + PartialEq + Copy>(
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    published: &mut [T],
    current: &[T],
    force_all: bool,
) {
    debug_assert_eq!(published.len(), current.len());
    if current.is_empty() {
        return;
    }
    if force_all {
        queue.write_buffer(buffer, 0, bytemuck::cast_slice(current));
        published.copy_from_slice(current);
        return;
    }

    let element_bytes = core::mem::size_of::<T>() as u64;
    let mut start = 0;
    while start < current.len() {
        while start < current.len() && published[start] == current[start] {
            start += 1;
        }
        if start == current.len() {
            break;
        }
        let mut end = start + 1;
        while end < current.len() && published[end] != current[end] {
            end += 1;
        }
        queue.write_buffer(
            buffer,
            start as u64 * element_bytes,
            bytemuck::cast_slice(&current[start..end]),
        );
        published[start..end].copy_from_slice(&current[start..end]);
        start = end;
    }
}

fn saturating_u32(value: u64) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

#[derive(Debug, thiserror::Error)]
pub enum GpuResidencyError {
    #[error(transparent)]
    Config(#[from] GpuConfigError),
    #[error(transparent)]
    PageTable(#[from] PageTableError),
    #[error(transparent)]
    Contract(#[from] ContractError),
    #[error(transparent)]
    Address(#[from] AddressError),
    #[error(transparent)]
    Metadata(#[from] GpuPageMetaError),
    #[error("planet {0:?} has no registered camera-local frame")]
    MissingPlanetFrame(PlanetId),
    #[error("planet-frame registry reached its bounded capacity of {maximum}")]
    PlanetFrameCapacity { maximum: u32 },
    #[error("planet frame {0:?} is still referenced by resident or visible pages")]
    PlanetFrameInUse(PlanetId),
    #[error("batch has {actual} pages; staging capacity is {maximum}")]
    BatchCapacity { actual: usize, maximum: u32 },
    #[error("batch dirtied {actual} slots; staging capacity is {maximum}")]
    DirtySlotCapacity { actual: usize, maximum: u32 },
    #[error("page slot {0} is outside the configured atlas")]
    InvalidSlot(u32),
    #[error("resident page disappeared while publishing a validated update")]
    ResidentPageMissing,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_narrowing_saturates() {
        assert_eq!(saturating_u32(u64::MAX), u32::MAX);
        assert_eq!(saturating_u32(7), 7);
    }

    #[test]
    fn upload_outcome_keeps_core_backpressure_reason_typed() {
        let outcome = GpuUploadOutcome::Residency(UploadOutcome::Backpressure(
            helio_planet_voxel_core::BackpressureReason::AllEvictionCandidatesVisible,
        ));
        assert!(matches!(
            outcome,
            GpuUploadOutcome::Residency(UploadOutcome::Backpressure(
                helio_planet_voxel_core::BackpressureReason::AllEvictionCandidatesVisible
            ))
        ));
    }
}
