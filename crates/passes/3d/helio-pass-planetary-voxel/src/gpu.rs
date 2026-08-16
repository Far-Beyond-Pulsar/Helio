use crate::{
    GpuAllocationPlan, GpuConfigError, GpuLookupKey, GpuPageTableEntry, GpuResidencyCounters,
    GpuResidencyUniform, PageTable, PageTableError, PlanetaryVoxelGpuConfig,
};
use helio_planet_voxel_core::{
    AddressError, ContractError, EvictOutcome, GpuPageMeta, GpuPageMetaError, PageEvict,
    PageUpload, PlanetFrameUniform, PlanetId, PlanetPageKey, ResidentPageCache, SourceGeneration,
    UploadOutcome, VisibilityOutcome, VisiblePageSet, PAGE_CELL_BYTES,
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
    pub textures: u32,
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
    publication_epoch: u64,
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
            publication_epoch: 1,
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

    pub const fn publication_epoch(&self) -> u64 {
        self.publication_epoch
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
            buffers: 4,
            textures: 1,
            allocated_bytes: self.plan.total_bytes,
        }
    }

    pub fn atlas_texture(&self) -> &wgpu::Texture {
        &self.resources.atlas
    }

    pub fn atlas_view(&self) -> &wgpu::TextureView {
        &self.resources.atlas_view
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
        _queue: &wgpu::Queue,
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
        self.frames.insert(planet, frame);
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
        _device: &wgpu::Device,
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

        // Texture writes are queued before table and metadata publication. A
        // consumer therefore cannot discover a generation whose complete page
        // tile is not already ahead of it on the same queue timeline.
        self.publish_dirty_slots(queue, &dirty_slots)?;
        if !dirty_slots.is_empty() {
            self.advance_publication_epoch();
        }
        let metadata = self.build_metadata()?;
        self.publish_metadata(queue, &metadata, false);
        self.publish_table(queue, false);
        self.refresh_and_publish_counters(queue);
        Ok(outcomes)
    }

    pub fn apply_evict_batch(
        &mut self,
        _device: &wgpu::Device,
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
        // Removed slots are no longer discoverable through the table. A later
        // reuse overwrites the complete texture tile before publishing its
        // replacement entry.
        self.publish_dirty_slots(queue, &dirty_slots)?;
        if !dirty_slots.is_empty() {
            self.advance_publication_epoch();
        }
        let metadata = self.build_metadata()?;
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
            let metadata = self.build_metadata()?;
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
        self.advance_publication_epoch();
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
        self.advance_publication_epoch();
        self.published_table.fill(GpuPageTableEntry::default());
        self.published_metadata.fill(GpuPageMeta::default());

        let slots: Vec<_> = self
            .cache
            .resident_pages()
            .map(|(_, page)| page.slot)
            .collect();
        for chunk in slots.chunks(self.config.max_batch_pages as usize) {
            let dirty_slots = chunk.iter().copied().collect();
            self.publish_dirty_slots(queue, &dirty_slots)?;
        }
        self.counters.device_rebuilds = self.counters.device_rebuilds.saturating_add(1);
        let metadata = self.build_metadata()?;
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
        Ok(GpuLookupKey::from_planet_page(key)?)
    }

    fn build_metadata(&self) -> Result<Vec<GpuPageMeta>, GpuResidencyError> {
        let mut metadata = vec![GpuPageMeta::default(); self.config.max_resident_pages as usize];
        for (key, page) in self.cache.resident_pages() {
            let transition_mask = self
                .visible
                .get(&key)
                .filter(|(generation, _)| *generation == page.generation)
                .map_or(0, |(_, mask)| *mask);
            metadata[page.slot as usize] = GpuPageMeta::new(
                key.page,
                page.slot,
                page.publication_generation,
                transition_mask,
            )?;
        }
        Ok(metadata)
    }

    fn publish_dirty_slots(
        &mut self,
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

        let pages_by_slot: BTreeMap<_, _> = self
            .cache
            .resident_pages()
            .map(|(_, page)| (page.slot, page.cells.as_ref()))
            .collect();
        for slot in dirty_slots {
            let Some(page_cells) = pages_by_slot.get(slot) else {
                // An evicted slot is no longer discoverable through the page
                // table. Its texels may remain until a complete replacement
                // page is queued into the same slot.
                continue;
            };
            let [x, y, z] = self
                .plan
                .atlas
                .origin_for_slot(*slot)
                .ok_or(GpuResidencyError::InvalidSlot(*slot))?;
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.resources.atlas,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x, y, z },
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(page_cells),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some((helio_planet_voxel_core::PAGE_EDGE * 4) as u32),
                    rows_per_image: Some(helio_planet_voxel_core::PAGE_EDGE as u32),
                },
                wgpu::Extent3d {
                    width: helio_planet_voxel_core::PAGE_EDGE as u32,
                    height: helio_planet_voxel_core::PAGE_EDGE as u32,
                    depth_or_array_layers: helio_planet_voxel_core::PAGE_EDGE as u32,
                },
            );
            self.cell_bytes_uploaded = self
                .cell_bytes_uploaded
                .saturating_add(PAGE_CELL_BYTES as u64);
        }
        self.counters.batches_submitted = self.counters.batches_submitted.saturating_add(1);
        Ok(())
    }

    fn publish_state(
        &mut self,
        queue: &wgpu::Queue,
        force_all: bool,
    ) -> Result<(), GpuResidencyError> {
        let metadata = self.build_metadata()?;
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
        self.counters.resource_buffers = 4;
        self.counters.resource_textures = 1;
        self.counters.atlas_capacity_pages = self.plan.atlas.capacity_pages;

        let uniform = GpuResidencyUniform {
            table_mask: self.table.capacity() - 1,
            max_probe: self.table.max_probe(),
            resident_pages: self.counters.resident_pages,
            atlas_tiles_x: self.plan.atlas.tile_count[0],
            atlas_tiles_y: self.plan.atlas.tile_count[1],
            atlas_tiles_z: self.plan.atlas.tile_count[2],
            publication_epoch_low: self.publication_epoch as u32,
            publication_epoch_high: (self.publication_epoch >> 32) as u32,
        };
        queue.write_buffer(&self.resources.uniform, 0, bytemuck::bytes_of(&uniform));
        queue.write_buffer(
            &self.resources.counters,
            0,
            bytemuck::bytes_of(&self.counters),
        );
    }

    fn advance_publication_epoch(&mut self) {
        self.publication_epoch = self.publication_epoch.wrapping_add(1).max(1);
    }
}

struct GpuResidencyResources {
    atlas: wgpu::Texture,
    atlas_view: wgpu::TextureView,
    metadata: wgpu::Buffer,
    page_table: wgpu::Buffer,
    uniform: wgpu::Buffer,
    counters: wgpu::Buffer,
}

impl GpuResidencyResources {
    fn new(device: &wgpu::Device, plan: &GpuAllocationPlan) -> Self {
        let atlas = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Planetary Voxel Cell Atlas"),
            size: wgpu::Extent3d {
                width: plan.atlas.extent[0],
                height: plan.atlas.extent[1],
                depth_or_array_layers: plan.atlas.extent[2],
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let atlas_view = atlas.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Planetary Voxel Cell Atlas View"),
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        });
        Self {
            atlas,
            atlas_view,
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
