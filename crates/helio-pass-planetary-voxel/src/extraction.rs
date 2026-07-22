use bytemuck::{Pod, Zeroable};
use helio_planet_voxel_core::{PlanetPageKey, TRANSITION_FACE_MASK};
use std::collections::BTreeMap;

pub const TERRAIN_MESHLET_MAX_VERTICES: u32 = 64;
pub const TERRAIN_MESHLET_MAX_TRIANGLES: u32 = 96;

/// One bounded GPU extraction request. The page slot resolves through the
/// residency metadata; generations make late extraction dispatches harmless.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuExtractionRequest {
    pub page_slot: u32,
    pub generation_low: u32,
    pub generation_high: u32,
    pub transition_mask: u32,
    pub dirty_microbricks_low: u32,
    pub dirty_microbricks_high: u32,
    pub _pad: [u32; 2],
}

impl GpuExtractionRequest {
    pub fn new(
        page_slot: u32,
        generation: u64,
        transition_mask: u8,
        dirty_microbricks: u64,
    ) -> Result<Self, ExtractionError> {
        if transition_mask & !TRANSITION_FACE_MASK != 0 {
            return Err(ExtractionError::TransitionMask(transition_mask));
        }
        Ok(Self {
            page_slot,
            generation_low: generation as u32,
            generation_high: (generation >> 32) as u32,
            transition_mask: u32::from(transition_mask),
            dirty_microbricks_low: dirty_microbricks as u32,
            dirty_microbricks_high: (dirty_microbricks >> 32) as u32,
            _pad: [0; 2],
        })
    }

    pub const fn generation(self) -> u64 {
        (self.generation_low as u64) | ((self.generation_high as u64) << 32)
    }

    pub const fn dirty_microbricks(self) -> u64 {
        (self.dirty_microbricks_low as u64) | ((self.dirty_microbricks_high as u64) << 32)
    }
}

/// Generation-tagged ranges published atomically for one page.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuExtractionRange {
    pub first_vertex: u32,
    pub vertex_count: u32,
    pub first_index: u32,
    pub index_count: u32,
    pub first_meshlet: u32,
    pub meshlet_count: u32,
    pub generation_low: u32,
    pub generation_high: u32,
}

impl GpuExtractionRange {
    pub const fn generation(self) -> u64 {
        (self.generation_low as u64) | ((self.generation_high as u64) << 32)
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct GpuTerrainVertex {
    pub position: [f32; 3],
    pub material: u32,
    pub normal: [f32; 3],
    pub flags: u32,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuTerrainMeshlet {
    pub first_index: u32,
    pub index_count: u32,
    pub first_vertex: u32,
    pub vertex_count: u32,
    pub bounds_offset: u32,
    pub generation_low: u32,
    pub generation_high: u32,
    pub _pad: u32,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuExtractionCounters {
    pub requests: u32,
    pub active_cells: u32,
    pub vertices: u32,
    pub indices: u32,
    pub meshlets: u32,
    pub completed: u32,
    pub stale_rejected: u32,
    pub overflowed: u32,
    pub vertex_overflow: u32,
    pub index_overflow: u32,
    pub meshlet_overflow: u32,
    pub _pad: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExtractionLimits {
    pub max_page_slots: u32,
    pub max_pending_pages: u32,
    pub max_vertices: u32,
    pub max_indices: u32,
    pub max_meshlets: u32,
}

impl ExtractionLimits {
    pub fn new(
        max_page_slots: u32,
        max_pending_pages: u32,
        max_vertices: u32,
        max_indices: u32,
        max_meshlets: u32,
    ) -> Result<Self, ExtractionError> {
        if max_page_slots == 0
            || max_pending_pages == 0
            || max_pending_pages > max_page_slots
            || max_vertices == 0
            || max_indices == 0
            || max_meshlets == 0
        {
            return Err(ExtractionError::InvalidLimits);
        }
        let limits = Self {
            max_page_slots,
            max_pending_pages,
            max_vertices,
            max_indices,
            max_meshlets,
        };
        limits.allocation_plan()?;
        Ok(limits)
    }

    pub fn allocation_plan(self) -> Result<ExtractionAllocationPlan, ExtractionError> {
        let request_bytes = bytes_for::<GpuExtractionRequest>(self.max_pending_pages)?;
        let page_range_bytes = bytes_for::<GpuExtractionRange>(self.max_page_slots)?;
        let vertex_bytes = bytes_for::<GpuTerrainVertex>(self.max_vertices)?;
        let index_bytes = bytes_for::<u32>(self.max_indices)?;
        let meshlet_bytes = bytes_for::<GpuTerrainMeshlet>(self.max_meshlets)?;
        let counter_bytes = core::mem::size_of::<GpuExtractionCounters>() as u64;
        let total_bytes = [
            request_bytes,
            page_range_bytes,
            vertex_bytes,
            index_bytes,
            meshlet_bytes,
            counter_bytes,
        ]
        .into_iter()
        .try_fold(0_u64, |total, bytes| {
            total
                .checked_add(bytes)
                .ok_or(ExtractionError::ArithmeticOverflow)
        })?;
        Ok(ExtractionAllocationPlan {
            request_bytes,
            page_range_bytes,
            vertex_bytes,
            index_bytes,
            meshlet_bytes,
            counter_bytes,
            total_bytes,
        })
    }

    pub fn validate_device(self, limits: &wgpu::Limits) -> Result<(), ExtractionError> {
        let plan = self.allocation_plan()?;
        for (name, requested) in [
            ("extraction requests", plan.request_bytes),
            ("page extraction ranges", plan.page_range_bytes),
            ("terrain vertices", plan.vertex_bytes),
            ("terrain indices", plan.index_bytes),
            ("terrain meshlets", plan.meshlet_bytes),
            ("extraction counters", plan.counter_bytes),
        ] {
            if requested > limits.max_buffer_size
                || requested > limits.max_storage_buffer_binding_size
            {
                return Err(ExtractionError::DeviceBufferLimit {
                    name,
                    requested,
                    max_buffer_bytes: limits.max_buffer_size,
                    max_storage_bytes: limits.max_storage_buffer_binding_size,
                });
            }
        }
        Ok(())
    }
}

impl Default for ExtractionLimits {
    fn default() -> Self {
        Self::new(256, 32, 1_048_576, 3_145_728, 32_768)
            .expect("default extraction budgets are valid")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExtractionAllocationPlan {
    pub request_bytes: u64,
    pub page_range_bytes: u64,
    pub vertex_bytes: u64,
    pub index_bytes: u64,
    pub meshlet_bytes: u64,
    pub counter_bytes: u64,
    pub total_bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SurfaceCounts {
    pub vertices: u32,
    pub indices: u32,
    pub meshlets: u32,
}

impl SurfaceCounts {
    fn validate(self) -> Result<(), ExtractionError> {
        if !self.indices.is_multiple_of(3) {
            return Err(ExtractionError::NonTriangleIndexCount(self.indices));
        }
        if (self.vertices == 0 || self.indices == 0) && self != Self::default() {
            return Err(ExtractionError::IncompleteSurfaceCounts);
        }
        if self.meshlets == 0 && self.indices != 0 {
            return Err(ExtractionError::IncompleteSurfaceCounts);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ArenaSlice {
    pub first: u32,
    pub count: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SurfaceAllocation {
    pub vertices: ArenaSlice,
    pub indices: ArenaSlice,
    pub meshlets: ArenaSlice,
}

impl SurfaceAllocation {
    pub const fn counts(self) -> SurfaceCounts {
        SurfaceCounts {
            vertices: self.vertices.count,
            indices: self.indices.count,
            meshlets: self.meshlets.count,
        }
    }

    pub const fn gpu_range(self, generation: u64) -> GpuExtractionRange {
        GpuExtractionRange {
            first_vertex: self.vertices.first,
            vertex_count: self.vertices.count,
            first_index: self.indices.first,
            index_count: self.indices.count,
            first_meshlet: self.meshlets.first,
            meshlet_count: self.meshlets.count,
            generation_low: generation as u32,
            generation_high: (generation >> 32) as u32,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExtractionReservation {
    pub key: PlanetPageKey,
    pub generation: u64,
    pub allocation: SurfaceAllocation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PublishedSurface {
    pub generation: u64,
    pub allocation: SurfaceAllocation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReservationOutcome {
    Reserved(ExtractionReservation),
    Current(PublishedSurface),
    DuplicatePending(ExtractionReservation),
    Stale { newest_generation: u64 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PublicationOutcome {
    Published {
        current: PublishedSurface,
        replaced: Option<PublishedSurface>,
    },
    Stale {
        newest_generation: u64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExtractionEvictOutcome {
    Evicted,
    Missing,
    Stale { newest_generation: u64 },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExtractionPublisherCounters {
    pub current_pages: usize,
    pub pending_pages: usize,
    pub used_vertices: u32,
    pub used_indices: u32,
    pub used_meshlets: u32,
    pub pending_high_water: usize,
    pub vertex_high_water: u32,
    pub index_high_water: u32,
    pub meshlet_high_water: u32,
    pub reservations: u64,
    pub publications: u64,
    pub replacements: u64,
    pub cancellations: u64,
    pub evictions: u64,
    pub stale_rejected: u64,
    pub backpressured: u64,
}

#[derive(Clone, Copy, Debug, Default)]
struct PagePublicationState {
    current: Option<PublishedSurface>,
    pending: Option<ExtractionReservation>,
}

/// CPU reference for the generation-safe bounded GPU publication contract.
/// A replacement reserves new ranges without freeing the current surface;
/// publication swaps generations atomically and only then recycles the old
/// ranges.
pub struct BoundedExtractionPublisher {
    limits: ExtractionLimits,
    vertices: RangeAllocator,
    indices: RangeAllocator,
    meshlets: RangeAllocator,
    pages: BTreeMap<PlanetPageKey, PagePublicationState>,
    counters: ExtractionPublisherCounters,
}

impl BoundedExtractionPublisher {
    pub fn new(limits: ExtractionLimits) -> Self {
        Self {
            vertices: RangeAllocator::new(limits.max_vertices),
            indices: RangeAllocator::new(limits.max_indices),
            meshlets: RangeAllocator::new(limits.max_meshlets),
            pages: BTreeMap::new(),
            counters: ExtractionPublisherCounters::default(),
            limits,
        }
    }

    pub const fn limits(&self) -> ExtractionLimits {
        self.limits
    }

    pub fn current(&self, key: PlanetPageKey) -> Option<PublishedSurface> {
        self.pages.get(&key).and_then(|state| state.current)
    }

    pub fn pending(&self, key: PlanetPageKey) -> Option<ExtractionReservation> {
        self.pages.get(&key).and_then(|state| state.pending)
    }

    pub fn counters(&self) -> ExtractionPublisherCounters {
        let mut counters = self.counters;
        counters.current_pages = self
            .pages
            .values()
            .filter(|state| state.current.is_some())
            .count();
        counters.pending_pages = self
            .pages
            .values()
            .filter(|state| state.pending.is_some())
            .count();
        counters.used_vertices = self.vertices.used();
        counters.used_indices = self.indices.used();
        counters.used_meshlets = self.meshlets.used();
        counters
    }

    pub fn reserve(
        &mut self,
        key: PlanetPageKey,
        generation: u64,
        counts: SurfaceCounts,
    ) -> Result<ReservationOutcome, ExtractionError> {
        counts.validate()?;
        if let Some(current) = self.current(key) {
            if generation < current.generation {
                self.counters.stale_rejected = self.counters.stale_rejected.saturating_add(1);
                return Ok(ReservationOutcome::Stale {
                    newest_generation: current.generation,
                });
            }
            if generation == current.generation {
                return Ok(ReservationOutcome::Current(current));
            }
        }
        if let Some(pending) = self.pending(key) {
            if generation < pending.generation {
                self.counters.stale_rejected = self.counters.stale_rejected.saturating_add(1);
                return Ok(ReservationOutcome::Stale {
                    newest_generation: pending.generation,
                });
            }
            if generation == pending.generation {
                if pending.allocation.counts() != counts {
                    return Err(ExtractionError::GenerationConflict { key, generation });
                }
                return Ok(ReservationOutcome::DuplicatePending(pending));
            }
            self.cancel_pending(key, pending.generation)?;
        }

        let pending_pages = self
            .pages
            .values()
            .filter(|state| state.pending.is_some())
            .count();
        if pending_pages >= self.limits.max_pending_pages as usize {
            self.counters.backpressured = self.counters.backpressured.saturating_add(1);
            return Err(ExtractionError::PendingCapacity {
                maximum: self.limits.max_pending_pages,
            });
        }

        let vertices = match self.vertices.reserve(counts.vertices) {
            Some(slice) => slice,
            None => return self.capacity_error(ExtractionCapacity::Vertices),
        };
        let indices = match self.indices.reserve(counts.indices) {
            Some(slice) => slice,
            None => {
                self.vertices.release(vertices);
                return self.capacity_error(ExtractionCapacity::Indices);
            }
        };
        let meshlets = match self.meshlets.reserve(counts.meshlets) {
            Some(slice) => slice,
            None => {
                self.indices.release(indices);
                self.vertices.release(vertices);
                return self.capacity_error(ExtractionCapacity::Meshlets);
            }
        };
        let reservation = ExtractionReservation {
            key,
            generation,
            allocation: SurfaceAllocation {
                vertices,
                indices,
                meshlets,
            },
        };
        self.pages.entry(key).or_default().pending = Some(reservation);
        self.counters.reservations = self.counters.reservations.saturating_add(1);
        self.refresh_high_water();
        Ok(ReservationOutcome::Reserved(reservation))
    }

    pub fn publish(
        &mut self,
        reservation: ExtractionReservation,
    ) -> Result<PublicationOutcome, ExtractionError> {
        let Some(state) = self.pages.get_mut(&reservation.key) else {
            return Err(ExtractionError::ReservationMissing(reservation.key));
        };
        let newest_generation = state
            .current
            .map(|current| current.generation)
            .into_iter()
            .chain(state.pending.map(|pending| pending.generation))
            .max();
        if let Some(newest_generation) =
            newest_generation.filter(|newest| reservation.generation < *newest)
        {
            self.counters.stale_rejected = self.counters.stale_rejected.saturating_add(1);
            return Ok(PublicationOutcome::Stale { newest_generation });
        }
        if state.pending != Some(reservation) {
            return Err(ExtractionError::ReservationMismatch {
                key: reservation.key,
                generation: reservation.generation,
            });
        }
        state.pending = None;
        let current = PublishedSurface {
            generation: reservation.generation,
            allocation: reservation.allocation,
        };
        let replaced = state.current.replace(current);
        if let Some(old) = replaced {
            self.release(old.allocation);
            self.counters.replacements = self.counters.replacements.saturating_add(1);
        }
        self.counters.publications = self.counters.publications.saturating_add(1);
        Ok(PublicationOutcome::Published { current, replaced })
    }

    pub fn cancel_pending(
        &mut self,
        key: PlanetPageKey,
        generation: u64,
    ) -> Result<bool, ExtractionError> {
        let Some(state) = self.pages.get_mut(&key) else {
            return Ok(false);
        };
        let Some(pending) = state.pending else {
            return Ok(false);
        };
        if pending.generation != generation {
            return Err(ExtractionError::ReservationMismatch { key, generation });
        }
        state.pending = None;
        self.release(pending.allocation);
        self.counters.cancellations = self.counters.cancellations.saturating_add(1);
        self.remove_empty_state(key);
        Ok(true)
    }

    pub fn evict(&mut self, key: PlanetPageKey, generation: u64) -> ExtractionEvictOutcome {
        let Some(state) = self.pages.get(&key).copied() else {
            return ExtractionEvictOutcome::Missing;
        };
        let newest_generation = state
            .current
            .map(|surface| surface.generation)
            .into_iter()
            .chain(state.pending.map(|pending| pending.generation))
            .max()
            .unwrap_or(0);
        if generation < newest_generation {
            self.counters.stale_rejected = self.counters.stale_rejected.saturating_add(1);
            return ExtractionEvictOutcome::Stale { newest_generation };
        }
        self.pages.remove(&key);
        if let Some(current) = state.current {
            self.release(current.allocation);
        }
        if let Some(pending) = state.pending {
            self.release(pending.allocation);
        }
        self.counters.evictions = self.counters.evictions.saturating_add(1);
        ExtractionEvictOutcome::Evicted
    }

    fn capacity_error<T>(&mut self, capacity: ExtractionCapacity) -> Result<T, ExtractionError> {
        self.counters.backpressured = self.counters.backpressured.saturating_add(1);
        Err(ExtractionError::ArenaCapacity { capacity })
    }

    fn release(&mut self, allocation: SurfaceAllocation) {
        self.vertices.release(allocation.vertices);
        self.indices.release(allocation.indices);
        self.meshlets.release(allocation.meshlets);
    }

    fn remove_empty_state(&mut self, key: PlanetPageKey) {
        if self
            .pages
            .get(&key)
            .is_some_and(|state| state.current.is_none() && state.pending.is_none())
        {
            self.pages.remove(&key);
        }
    }

    fn refresh_high_water(&mut self) {
        let counters = self.counters();
        self.counters.pending_high_water =
            self.counters.pending_high_water.max(counters.pending_pages);
        self.counters.vertex_high_water =
            self.counters.vertex_high_water.max(counters.used_vertices);
        self.counters.index_high_water = self.counters.index_high_water.max(counters.used_indices);
        self.counters.meshlet_high_water =
            self.counters.meshlet_high_water.max(counters.used_meshlets);
    }
}

#[derive(Clone, Debug)]
struct RangeAllocator {
    capacity: u32,
    free: Vec<ArenaSlice>,
}

impl RangeAllocator {
    fn new(capacity: u32) -> Self {
        Self {
            capacity,
            free: vec![ArenaSlice {
                first: 0,
                count: capacity,
            }],
        }
    }

    fn reserve(&mut self, count: u32) -> Option<ArenaSlice> {
        if count == 0 {
            return Some(ArenaSlice::default());
        }
        let index = self.free.iter().position(|range| range.count >= count)?;
        let allocation = ArenaSlice {
            first: self.free[index].first,
            count,
        };
        self.free[index].first += count;
        self.free[index].count -= count;
        if self.free[index].count == 0 {
            self.free.remove(index);
        }
        Some(allocation)
    }

    fn release(&mut self, released: ArenaSlice) {
        if released.count == 0 {
            return;
        }
        debug_assert!(released.first.saturating_add(released.count) <= self.capacity);
        let insert_at = self
            .free
            .partition_point(|range| range.first < released.first);
        self.free.insert(insert_at, released);
        let mut index = insert_at.saturating_sub(1);
        while index + 1 < self.free.len() {
            let left_end = self.free[index].first + self.free[index].count;
            if left_end < self.free[index + 1].first {
                index += 1;
                continue;
            }
            debug_assert_eq!(left_end, self.free[index + 1].first);
            let right = self.free.remove(index + 1);
            self.free[index].count += right.count;
        }
    }

    fn used(&self) -> u32 {
        self.capacity - self.free.iter().map(|range| range.count).sum::<u32>()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExtractionCapacity {
    Vertices,
    Indices,
    Meshlets,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ExtractionError {
    #[error("extraction limits must be non-zero and pending pages cannot exceed page slots")]
    InvalidLimits,
    #[error("extraction byte arithmetic overflowed")]
    ArithmeticOverflow,
    #[error("transition mask {0:#010b} uses bits outside the six page faces")]
    TransitionMask(u8),
    #[error("surface index count {0} is not divisible by three")]
    NonTriangleIndexCount(u32),
    #[error("non-empty surfaces require vertices, triangle indices, and meshlets")]
    IncompleteSurfaceCounts,
    #[error("pending extraction capacity {maximum} is exhausted")]
    PendingCapacity { maximum: u32 },
    #[error("the bounded {capacity:?} extraction arena is exhausted")]
    ArenaCapacity { capacity: ExtractionCapacity },
    #[error("page {key:?} generation {generation} conflicts with an existing reservation")]
    GenerationConflict { key: PlanetPageKey, generation: u64 },
    #[error("page {0:?} has no pending extraction reservation")]
    ReservationMissing(PlanetPageKey),
    #[error("page {key:?} generation {generation} does not match its pending reservation")]
    ReservationMismatch { key: PlanetPageKey, generation: u64 },
    #[error(
        "{name} buffer requests {requested} bytes (buffer limit {max_buffer_bytes}, storage binding limit {max_storage_bytes})"
    )]
    DeviceBufferLimit {
        name: &'static str,
        requested: u64,
        max_buffer_bytes: u64,
        max_storage_bytes: u64,
    },
}

fn bytes_for<T>(count: u32) -> Result<u64, ExtractionError> {
    u64::from(count)
        .checked_mul(core::mem::size_of::<T>() as u64)
        .ok_or(ExtractionError::ArithmeticOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;
    use helio_planet_voxel_core::{PageKey, PlanetId};

    fn key(index: i64) -> PlanetPageKey {
        PlanetPageKey::new(PlanetId([1; 16]), PageKey::new(0, [index, 0, 0]))
    }

    fn counts(vertices: u32, triangles: u32, meshlets: u32) -> SurfaceCounts {
        SurfaceCounts {
            vertices,
            indices: triangles * 3,
            meshlets,
        }
    }

    fn reservation(outcome: ReservationOutcome) -> ExtractionReservation {
        match outcome {
            ReservationOutcome::Reserved(reservation) => reservation,
            other => panic!("expected a new reservation, got {other:?}"),
        }
    }

    #[test]
    fn gpu_requests_preserve_full_generations_and_dirty_masks() {
        assert_eq!(crate::PRODUCTION_EXTRACTION_ALGORITHM, "gpu_transvoxel");
        let request = GpuExtractionRequest::new(
            17,
            0xfedc_ba98_7654_3210,
            TRANSITION_FACE_MASK,
            0x8000_0000_0000_0001,
        )
        .unwrap();
        assert_eq!(request.generation(), 0xfedc_ba98_7654_3210);
        assert_eq!(request.dirty_microbricks(), 0x8000_0000_0000_0001);
        assert_eq!(
            GpuExtractionRequest::new(0, 0, 0x80, 0),
            Err(ExtractionError::TransitionMask(0x80))
        );
    }

    #[test]
    fn byte_plan_is_exact_and_checked_against_device_limits() {
        let limits = ExtractionLimits::new(4, 2, 100, 300, 10).unwrap();
        let plan = limits.allocation_plan().unwrap();
        assert_eq!(plan.request_bytes, 2 * 32);
        assert_eq!(plan.page_range_bytes, 4 * 32);
        assert_eq!(plan.vertex_bytes, 100 * 32);
        assert_eq!(plan.index_bytes, 300 * 4);
        assert_eq!(plan.meshlet_bytes, 10 * 32);
        assert_eq!(plan.counter_bytes, 48);

        let device_limits = wgpu::Limits {
            max_buffer_size: 3_199,
            max_storage_buffer_binding_size: 3_199,
            ..wgpu::Limits::downlevel_defaults()
        };
        assert!(matches!(
            limits.validate_device(&device_limits),
            Err(ExtractionError::DeviceBufferLimit {
                name: "terrain vertices",
                ..
            })
        ));
    }

    #[test]
    fn replacement_keeps_current_ranges_until_atomic_publication() {
        let limits = ExtractionLimits::new(2, 1, 20, 60, 4).unwrap();
        let mut publisher = BoundedExtractionPublisher::new(limits);
        let first = reservation(publisher.reserve(key(0), 1, counts(8, 8, 1)).unwrap());
        publisher.publish(first).unwrap();
        let before = publisher.current(key(0)).unwrap();

        let replacement = reservation(publisher.reserve(key(0), 2, counts(10, 10, 1)).unwrap());
        assert_eq!(publisher.current(key(0)), Some(before));
        assert_eq!(publisher.pending(key(0)), Some(replacement));
        assert_eq!(publisher.counters().used_vertices, 18);

        let outcome = publisher.publish(replacement).unwrap();
        assert!(matches!(
            outcome,
            PublicationOutcome::Published {
                replaced: Some(surface),
                ..
            } if surface == before
        ));
        assert_eq!(publisher.current(key(0)).unwrap().generation, 2);
        assert_eq!(publisher.counters().used_vertices, 10);
    }

    #[test]
    fn stale_publications_and_evictions_cannot_replace_newer_surfaces() {
        let mut publisher =
            BoundedExtractionPublisher::new(ExtractionLimits::new(2, 2, 32, 96, 4).unwrap());
        let first = reservation(publisher.reserve(key(-1), 4, counts(8, 4, 1)).unwrap());
        publisher.publish(first).unwrap();
        assert_eq!(
            publisher.reserve(key(-1), 3, counts(8, 4, 1)).unwrap(),
            ReservationOutcome::Stale {
                newest_generation: 4
            }
        );
        assert_eq!(
            publisher.evict(key(-1), 3),
            ExtractionEvictOutcome::Stale {
                newest_generation: 4
            }
        );
        assert_eq!(publisher.current(key(-1)).unwrap().generation, 4);

        let fifth = reservation(publisher.reserve(key(-1), 5, counts(8, 4, 1)).unwrap());
        let sixth = reservation(publisher.reserve(key(-1), 6, counts(8, 4, 1)).unwrap());
        assert_eq!(
            publisher.publish(fifth).unwrap(),
            PublicationOutcome::Stale {
                newest_generation: 6
            }
        );
        publisher.publish(sixth).unwrap();
        assert_eq!(publisher.current(key(-1)).unwrap().generation, 6);
    }

    #[test]
    fn capacity_failure_rolls_back_partial_ranges_and_reuses_freed_space() {
        let mut publisher =
            BoundedExtractionPublisher::new(ExtractionLimits::new(2, 2, 10, 12, 1).unwrap());
        assert_eq!(
            publisher.reserve(key(0), 1, counts(8, 5, 1)),
            Err(ExtractionError::ArenaCapacity {
                capacity: ExtractionCapacity::Indices
            })
        );
        assert_eq!(publisher.counters().used_vertices, 0);

        let valid = reservation(publisher.reserve(key(0), 1, counts(8, 4, 1)).unwrap());
        publisher.publish(valid).unwrap();
        assert_eq!(publisher.evict(key(0), 1), ExtractionEvictOutcome::Evicted);
        assert_eq!(publisher.counters().used_vertices, 0);
        assert_eq!(publisher.counters().used_indices, 0);
        assert_eq!(publisher.counters().used_meshlets, 0);

        let reused = reservation(publisher.reserve(key(1), 1, counts(10, 4, 1)).unwrap());
        assert_eq!(reused.allocation.vertices.first, 0);
    }

    #[test]
    fn pending_capacity_and_duplicate_generations_are_explicit() {
        let mut publisher =
            BoundedExtractionPublisher::new(ExtractionLimits::new(2, 1, 32, 96, 4).unwrap());
        let first = reservation(publisher.reserve(key(0), 1, counts(8, 4, 1)).unwrap());
        assert_eq!(
            publisher.reserve(key(0), 1, counts(8, 4, 1)).unwrap(),
            ReservationOutcome::DuplicatePending(first)
        );
        assert!(matches!(
            publisher.reserve(key(1), 1, counts(8, 4, 1)),
            Err(ExtractionError::PendingCapacity { maximum: 1 })
        ));
    }
}
