//! Acceleration structures: the BLASes and TLASes every ray-traced kind
//! builds through, with the build policy shipping ray tracers use.
//!
//! Helio knows no object kind: a kind's projection (meshes, virtual-geometry
//! proxies, ...) keys its BLASes by ids it chooses, states what each one's
//! geometry does at runtime ([`Movability`]) and hands the managers complete
//! instance lists. The managers decide how each structure is built:
//!
//! - **Static geometry** (anything that does not deform) is built
//!   `PREFER_FAST_TRACE | ALLOW_COMPACTION` and compacted once the device
//!   says the build is done ([`BlasManager::maintain`]): no wait, a byte
//!   budget a frame ([`BlasPolicy::compaction_bytes_per_frame`]), the
//!   compacted handle replacing the built one in the TLAS.
//! - **Deforming geometry** ([`Movability::Dynamic`]: skinning, cloth) is
//!   built `PREFER_FAST_BUILD | ALLOW_UPDATE` and refit when its positions
//!   change, with a full rebuild every [`BlasPolicy::refits_per_rebuild`]
//!   refits, staggered across BLASes so rebuilds do not land on one frame.
//! - **The TLAS** is built `PREFER_FAST_TRACE | ALLOW_UPDATE`: refit when
//!   only transforms or masks changed, rebuilt when its membership changed or
//!   when the motion since its last build ([`TlasPolicy`]) bounds the quality
//!   a refit loses.
//!
//! Memory is accounted from the sizes the backend gave each structure
//! ([`BlasMemory`], [`TlasManager::memory`]), and both managers report their
//! frame into the renderer's named stats ([`BlasManager::report`],
//! [`TlasManager::report`]) under the names in [`stat`].
//!
//! Ordering: submit the encoders the managers record into in the order they
//! were recorded. A refit recorded before a different build of the same
//! structure but submitted after it is rejected at submit.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use crate::movability::Movability;
use crate::stats::FrameStats;

/// The frame stats the managers report ([`BlasManager::report`],
/// [`TlasManager::report`]). Every manager adds into the same names, so the
/// frame's totals cover every kind's structures.
pub mod stat {
    /// Bytes of the resident BLASes (compacted sizes where compacted).
    pub const BLAS_BYTES: &str = "rays.blas_bytes";
    /// Bytes the resident BLASes took as built, before any compaction:
    /// `BLAS_BYTES` subtracted from it is what compaction saved.
    pub const BLAS_BYTES_BEFORE_COMPACTION: &str = "rays.blas_bytes_before_compaction";
    /// Bytes of the TLASes: structures and their instance buffers.
    pub const TLAS_BYTES: &str = "rays.tlas_bytes";
    /// Scratch bytes the frame's BLAS and TLAS builds and refits needed.
    pub const SCRATCH_BYTES: &str = "rays.as_scratch_bytes";
    /// BLAS full builds recorded in the frame (new structures and rebuilds).
    pub const BLAS_BUILDS: &str = "rays.blas_builds";
    /// BLAS refits recorded in the frame.
    pub const BLAS_REFITS: &str = "rays.blas_refits";
    /// BLASes compacted in the frame.
    pub const BLAS_COMPACTIONS: &str = "rays.blas_compactions";
    /// TLAS full builds recorded in the frame.
    pub const TLAS_BUILDS: &str = "rays.tlas_builds";
    /// TLAS refits recorded in the frame.
    pub const TLAS_REFITS: &str = "rays.tlas_refits";
}

/// Failures detected before recording an acceleration-structure build.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum AccelerationError {
    #[error(
        "ray queries and nonzero acceleration-structure limits must be enabled on this device"
    )]
    Unsupported,
    #[error("invalid BLAS geometry: {0}")]
    InvalidGeometry(&'static str),
    #[error("TLAS references missing BLAS for geometry {0}")]
    MissingBlas(u64),
    #[error("TLAS instance count exceeds the device or host index limit")]
    TooManyInstances,
    #[error("TLAS instance transform contains a non-finite value")]
    InvalidTransform,
}

/// Opaque triangle geometry in an existing GPU allocation. Positions are Float32x3
/// at the start of each vertex; optional indices are geometry-local Uint32 values.
///
/// The two revisions are how a caller says what it wrote in place, without
/// the manager reading vertex data; buffers, offsets and counts are part of
/// the cache key independently of them.
pub struct BlasGeometry<'a> {
    /// Increment after writing new vertex positions in place, the triangles
    /// unchanged. A deforming BLAS refits; a static one is built again.
    pub positions_revision: u64,
    /// Increment after writing new indices in place, or anything else that
    /// changes which vertices form the triangles. Always a full build.
    pub topology_revision: u64,
    pub vertices: &'a wgpu::Buffer,
    pub first_vertex: u32,
    pub vertex_count: u32,
    pub vertex_stride: u64,
    pub indices: Option<&'a wgpu::Buffer>,
    pub first_index: u32,
    pub index_count: u32,
}

/// What a refit must keep: wgpu's update rule (same size descriptors, index
/// buffer and first index) plus the caller's topology revision.
#[derive(PartialEq, Eq)]
struct ShapeKey {
    opaque: bool,
    topology_revision: u64,
    vertex_count: u32,
    indices: Option<wgpu::Buffer>,
    first_index: u32,
    index_count: u32,
}

/// What a refit may change.
#[derive(PartialEq, Eq)]
struct PositionsKey {
    positions_revision: u64,
    vertices: wgpu::Buffer,
    first_vertex: u32,
    vertex_stride: u64,
}

impl BlasGeometry<'_> {
    fn validate(&self) -> Result<(), AccelerationError> {
        use AccelerationError::InvalidGeometry as Invalid;
        if self.vertex_count == 0
            || self.vertex_stride < 12
            || !self.vertex_stride.is_multiple_of(4)
        {
            return Err(Invalid(
                "positions require vertices and a stride >= 12 divisible by 4",
            ));
        }
        if !self
            .vertices
            .usage()
            .intersects(wgpu::BufferUsages::BLAS_INPUT | wgpu::BufferUsages::COPY_SRC)
        {
            return Err(Invalid(
                "vertex buffer requires BLAS_INPUT or COPY_SRC usage",
            ));
        }
        let count = u64::from(self.first_vertex) + u64::from(self.vertex_count);
        let end = count.checked_mul(self.vertex_stride);
        if end.is_none_or(|end| end > self.vertices.size()) {
            return Err(Invalid("vertex range exceeds its buffer"));
        }
        if let Some(indices) = self.indices {
            if !indices
                .usage()
                .intersects(wgpu::BufferUsages::BLAS_INPUT | wgpu::BufferUsages::COPY_SRC)
            {
                return Err(Invalid(
                    "index buffer requires BLAS_INPUT or COPY_SRC usage",
                ));
            }
            if self.index_count == 0 || !self.index_count.is_multiple_of(3) {
                return Err(Invalid(
                    "indexed triangles require a positive multiple of three indices",
                ));
            }
            if (u64::from(self.first_index) + u64::from(self.index_count)) * 4 > indices.size() {
                return Err(Invalid("index range exceeds its buffer"));
            }
        } else if self.first_index != 0
            || self.index_count != 0
            || !self.vertex_count.is_multiple_of(3)
        {
            return Err(Invalid(
                "non-indexed triangles require a multiple of three vertices and no index range",
            ));
        }
        Ok(())
    }

    fn shape(&self, opaque: bool) -> ShapeKey {
        ShapeKey {
            opaque,
            topology_revision: self.topology_revision,
            vertex_count: self.vertex_count,
            indices: self.indices.cloned(),
            first_index: self.first_index,
            index_count: self.index_count,
        }
    }

    fn positions(&self) -> PositionsKey {
        PositionsKey {
            positions_revision: self.positions_revision,
            vertices: self.vertices.clone(),
            first_vertex: self.first_vertex,
            vertex_stride: self.vertex_stride,
        }
    }

    fn triangles(&self) -> u64 {
        u64::from(
            if self.indices.is_some() {
                self.index_count
            } else {
                self.vertex_count
            } / 3,
        )
    }
}

/// One geometry of a BLAS and its triangles' opacity. A ray query reports a
/// hit's segment as its `geometry_index`, in segment order.
pub struct BlasSegment<'a> {
    pub geometry: BlasGeometry<'a>,
    pub opaque: bool,
}

/// What [`BlasManager::build`] recorded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlasWork {
    /// The BLAS already holds these inputs; nothing was recorded.
    Unchanged,
    /// A full build (a new structure, or a deforming one's rebuild).
    Built,
    /// A refit of a deforming BLAS to its new positions.
    Refit,
}

impl BlasWork {
    /// Whether a build or refit was recorded into the encoder.
    pub fn recorded(self) -> bool {
        self != Self::Unchanged
    }
}

/// How a [`BlasManager`] builds and compacts. The defaults are the
/// reasoning in the field docs; an owner with other needs sets its own
/// ([`BlasManager::set_policy`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlasPolicy {
    /// Refits of a deforming BLAS between its full rebuilds; 0 rebuilds on
    /// every change. A refit keeps the tree of the last build and only moves
    /// its boxes, so trace cost grows as the shape drifts from that pose;
    /// a rebuild costs several refits. 16 bounds the drift to a quarter of a
    /// second of animation at 60 Hz for an amortized cost of about one
    /// extra refit a frame. Each BLAS's first rebuild comes after a
    /// different number of refits, so rebuilds spread over frames.
    /// [`BlasManager::request_rebuild`] rebuilds at once (a large
    /// deformation the caller knows of: a ragdoll, a destruction).
    pub refits_per_rebuild: u32,
    /// Bytes of built BLASes compacted per [`BlasManager::maintain`] (at
    /// least one BLAS each call). A compaction is a GPU copy of the compacted
    /// size, and the built structure stays alive until the frames using it
    /// are done, so this bounds both the copy time a frame takes and the
    /// transient memory compaction adds. 32 MiB is well under a millisecond
    /// of copy at the bandwidth of hardware with ray queries.
    pub compaction_bytes_per_frame: u64,
}

impl Default for BlasPolicy {
    fn default() -> Self {
        Self {
            refits_per_rebuild: 16,
            compaction_bytes_per_frame: 32 << 20,
        }
    }
}

/// A [`BlasManager`]'s memory, from the sizes the backend gave each
/// structure.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BlasMemory {
    /// Resident BLASes.
    pub count: u64,
    /// Their bytes (compacted sizes where compacted).
    pub bytes: u64,
    /// Their bytes as built, before compaction.
    pub bytes_before_compaction: u64,
}

/// What a [`BlasManager`] has recorded since it was created.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BlasCounters {
    /// Full builds: new structures and rebuilds of deforming ones.
    pub builds: u64,
    /// Refits of deforming BLASes.
    pub refits: u64,
    /// The full builds [`BlasPolicy::refits_per_rebuild`] made (part of
    /// `builds`).
    pub policy_rebuilds: u64,
    /// Static BLASes compacted.
    pub compactions: u64,
    /// Scratch bytes the recorded builds and refits needed.
    pub scratch_bytes: u64,
}

/// A deforming BLAS's refit state.
struct Deforming {
    /// Refits before the policy rebuilds it.
    refits_left: u32,
    rebuild_requested: bool,
    /// Copies of COPY_SRC-only index pools, kept for refits: a refit must
    /// build from the same index buffer.
    index_copies: Vec<Option<wgpu::Buffer>>,
}

struct Entry {
    blas: wgpu::Blas,
    /// Changes whenever `blas` is replaced or built again in place (a
    /// refit), so a TLAS rebuilds or refits for exactly the instances whose
    /// structure changed.
    handle: u64,
    shape: Vec<ShapeKey>,
    positions: Vec<PositionsKey>,
    deforming: Option<Deforming>,
    /// Current bytes, and bytes as built.
    bytes: u64,
    built_bytes: u64,
}

/// A static BLAS waiting for compaction: its id and handle (a replaced
/// handle drops out), and the [`BlasManager::maintain`] epoch it was built in.
struct Compaction {
    id: u64,
    handle: u64,
    epoch: u64,
}

/// Bottom-level acceleration structures, keyed by an id their owner chooses;
/// see the module docs for the policy.
pub struct BlasManager {
    entries: HashMap<u64, Entry>,
    device: Arc<wgpu::Device>,
    rt_available: bool,
    policy: BlasPolicy,
    /// Changes whenever a BLAS is built (refits included), replaced or
    /// evicted. TLAS reuse must include this generation because a geometry
    /// id can retain its numeric identity while pointing at a newly-built
    /// BLAS, and a TLAS must be built after every BLAS it holds.
    revision: u64,
    next_handle: u64,
    /// Deforming BLASes created, to stagger their first rebuilds.
    next_phase: u32,
    /// Built static BLASes to prepare for compaction, oldest first.
    to_prepare: VecDeque<Compaction>,
    /// Prepared ones, in preparation order (the order they become ready).
    preparing: VecDeque<Compaction>,
    epoch: u64,
    memory: BlasMemory,
    counters: BlasCounters,
    reported: BlasCounters,
}

impl BlasManager {
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        let rt_available = device
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
            && device.limits().max_blas_geometry_count > 0
            && device.limits().max_blas_primitive_count > 0;
        Self {
            entries: HashMap::new(),
            device,
            rt_available,
            policy: BlasPolicy::default(),
            revision: 0,
            next_handle: 0,
            next_phase: 0,
            to_prepare: VecDeque::new(),
            preparing: VecDeque::new(),
            epoch: 0,
            memory: BlasMemory::default(),
            counters: BlasCounters::default(),
            reported: BlasCounters::default(),
        }
    }

    pub fn is_rt_available(&self) -> bool {
        self.rt_available
    }

    pub fn policy(&self) -> BlasPolicy {
        self.policy
    }

    /// Takes effect for the refits and compactions after it.
    pub fn set_policy(&mut self, policy: BlasPolicy) {
        self.policy = policy;
    }

    /// Monotonic-ish cache generation for consumers that retain TLAS instances.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn memory(&self) -> BlasMemory {
        self.memory
    }

    pub fn counters(&self) -> BlasCounters {
        self.counters
    }

    /// Record a build of BLAS `geometry_id` from `segments` (a mesh's
    /// sections, each with its own opacity), without CPU copies, submission
    /// or host waits, if its inputs changed. `movability` says what its
    /// geometry does: [`Movability::can_deform`] builds it for refits,
    /// anything else for tracing and compaction (see the module docs).
    ///
    /// BLAS_INPUT allocations are used directly; COPY_SRC-only pools are
    /// copied on the GPU into build inputs. Submit this encoder before any
    /// TLAS/query consuming the result. If the encoder is abandoned, remove
    /// the affected geometry or clear the manager before reusing it.
    /// GPU-written indices must be within vertex_count; their contents cannot
    /// be validated here without a readback. An invalid input removes the
    /// BLAS, so the old shape is never traced.
    pub fn build(
        &mut self,
        geometry_id: u64,
        encoder: &mut wgpu::CommandEncoder,
        segments: Vec<BlasSegment<'_>>,
        movability: Movability,
    ) -> Result<BlasWork, AccelerationError> {
        if !self.rt_available {
            return Err(AccelerationError::Unsupported);
        }
        if let Err(error) = self.validate(&segments) {
            // An invalid replacement must not leave the old shape queryable.
            self.remove_blas(geometry_id);
            return Err(error);
        }
        let deforming = movability.can_deform();
        let shape: Vec<ShapeKey> = segments
            .iter()
            .map(|segment| segment.geometry.shape(segment.opaque))
            .collect();
        let positions: Vec<PositionsKey> = segments
            .iter()
            .map(|segment| segment.geometry.positions())
            .collect();

        // A deforming BLAS of the same shape: refit, or the policy's rebuild.
        if let Some(entry) = self.entries.get_mut(&geometry_id) {
            if entry.shape == shape && entry.deforming.is_some() == deforming {
                let requested = entry
                    .deforming
                    .as_ref()
                    .is_some_and(|d| d.rebuild_requested);
                if entry.positions == positions && !requested {
                    return Ok(BlasWork::Unchanged);
                }
                if let Some(state) = entry.deforming.as_mut() {
                    let rebuild = requested || state.refits_left == 0;
                    let refits_per_rebuild = self.policy.refits_per_rebuild;
                    let sizes = entry.blas.sizes();
                    let work = if rebuild {
                        entry.blas.request_rebuild();
                        state.refits_left = refits_per_rebuild;
                        state.rebuild_requested = false;
                        self.counters.builds += 1;
                        if !requested {
                            self.counters.policy_rebuilds += 1;
                        }
                        self.counters.scratch_bytes += sizes.build_scratch;
                        BlasWork::Built
                    } else {
                        state.refits_left -= 1;
                        self.counters.refits += 1;
                        self.counters.scratch_bytes += sizes.update_scratch;
                        BlasWork::Refit
                    };
                    record(
                        &self.device,
                        encoder,
                        &entry.blas,
                        &segments,
                        Some(&state.index_copies),
                    );
                    entry.positions = positions;
                    // Its bounds changed: the TLAS holding it must be built
                    // after it (wgpu rejects a TLAS older than its BLASes).
                    entry.handle = self.next_handle;
                    self.next_handle += 1;
                    self.revision = self.revision.wrapping_add(1);
                    return Ok(work);
                }
            }
        }

        // A new structure: first build, new shape, or static geometry that
        // changed (a compacted BLAS cannot be built again).
        let sizes: Vec<_> = segments
            .iter()
            .map(|segment| size_descriptor(&segment.geometry, segment.opaque))
            .collect();
        let (flags, update_mode) = if deforming {
            (
                wgpu::AccelerationStructureFlags::PREFER_FAST_BUILD
                    | wgpu::AccelerationStructureFlags::ALLOW_UPDATE,
                wgpu::AccelerationStructureUpdateMode::PreferUpdate,
            )
        } else {
            (
                wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE
                    | wgpu::AccelerationStructureFlags::ALLOW_COMPACTION,
                wgpu::AccelerationStructureUpdateMode::Build,
            )
        };
        let blas = self.device.create_blas(
            &wgpu::CreateBlasDescriptor {
                label: Some(if deforming {
                    "deforming_geometry_blas"
                } else {
                    "static_geometry_blas"
                }),
                flags,
                update_mode,
            },
            wgpu::BlasGeometrySizeDescriptors::Triangles { descriptors: sizes },
        );
        let index_copies = record(&self.device, encoder, &blas, &segments, None);
        let built = blas.sizes();
        self.counters.builds += 1;
        self.counters.scratch_bytes += built.build_scratch;
        let deforming = deforming.then(|| {
            // Staggered over the N + 1 frames of a cycle (N refits, one
            // rebuild): the k-th deforming BLAS first rebuilds after
            // N - k % (N + 1) refits, then after every N.
            let n = self.policy.refits_per_rebuild;
            let phase = self.next_phase;
            self.next_phase = self.next_phase.wrapping_add(1);
            Deforming {
                refits_left: n - phase % n.saturating_add(1),
                rebuild_requested: false,
                index_copies,
            }
        });
        let handle = self.next_handle;
        self.next_handle += 1;
        if deforming.is_none() {
            // Replaced or removed BLASes leave stale items, dropped when
            // maintained; an owner that never maintains must not grow this.
            if self.to_prepare.len() > 2 * self.entries.len() + 64 {
                let entries = &self.entries;
                self.to_prepare
                    .retain(|c| entries.get(&c.id).is_some_and(|e| e.handle == c.handle));
            }
            self.to_prepare.push_back(Compaction {
                id: geometry_id,
                handle,
                epoch: self.epoch,
            });
        }
        let entry = Entry {
            blas,
            handle,
            shape,
            positions,
            deforming,
            bytes: built.structure,
            built_bytes: built.structure,
        };
        if let Some(old) = self.entries.insert(geometry_id, entry) {
            self.forget(&old);
        }
        self.memory.count += 1;
        self.memory.bytes += built.structure;
        self.memory.bytes_before_compaction += built.structure;
        self.revision = self.revision.wrapping_add(1);
        Ok(BlasWork::Built)
    }

    fn validate(&self, segments: &[BlasSegment<'_>]) -> Result<(), AccelerationError> {
        let limits = self.device.limits();
        if segments.is_empty() {
            return Err(AccelerationError::InvalidGeometry("a BLAS needs a segment"));
        }
        if segments.len() as u64 > u64::from(limits.max_blas_geometry_count) {
            return Err(AccelerationError::InvalidGeometry(
                "segment count exceeds the device limit",
            ));
        }
        let mut primitives = 0u64;
        for segment in segments {
            segment.geometry.validate()?;
            primitives += segment.geometry.triangles();
        }
        if primitives > u64::from(limits.max_blas_primitive_count) {
            return Err(AccelerationError::InvalidGeometry(
                "triangle count exceeds the device limit",
            ));
        }
        Ok(())
    }

    /// Make the next change of deforming BLAS `geometry_id` a full rebuild
    /// (a deformation the caller knows is large). The policy's count starts
    /// again from it. No effect on static BLASes, which every change rebuilds.
    pub fn request_rebuild(&mut self, geometry_id: u64) {
        if let Some(state) = self
            .entries
            .get_mut(&geometry_id)
            .and_then(|entry| entry.deforming.as_mut())
        {
            state.rebuild_requested = true;
        }
    }

    /// Compaction, once a frame, before recording the frame's builds: every
    /// encoder recorded into before the previous call must have been
    /// submitted by now (static BLASes are prepared one call after their
    /// build, so a frame's encoder may be submitted at any point of that
    /// frame).
    ///
    /// Static BLASes built two calls ago are prepared for compaction; those
    /// whose readiness the device reported (on its own submits and polls: this
    /// never waits or polls) are compacted, up to
    /// [`BlasPolicy::compaction_bytes_per_frame`]. A compacted handle replaces
    /// the built one, which the TLAS instances recorded before keep alive
    /// until their frames are done; [`Self::revision`] changes, so a TLAS
    /// built after this refers to the compacted structure. Work is the BLASes
    /// built or compacted, never every BLAS.
    pub fn maintain(&mut self, queue: &wgpu::Queue) {
        self.epoch += 1;
        while let Some(next) = self.to_prepare.front() {
            if next.epoch + 2 > self.epoch {
                break;
            }
            let next = self.to_prepare.pop_front().expect("front");
            let Some(entry) = self.current(&next) else {
                continue;
            };
            entry.blas.prepare_compaction_async(|_| {});
            self.preparing.push_back(next);
        }
        // Builds finish in submission order, and so do their readiness
        // reports: the first one not ready ends the scan.
        let mut budget = self.policy.compaction_bytes_per_frame;
        let mut compacted_any = false;
        while let Some(next) = self.preparing.front() {
            let Some(entry) = self.current(next) else {
                self.preparing.pop_front();
                continue;
            };
            if !entry.blas.ready_for_compaction() || (compacted_any && entry.bytes > budget) {
                break;
            }
            let next = self.preparing.pop_front().expect("front");
            let handle = self.next_handle;
            self.next_handle += 1;
            let entry = self.entries.get_mut(&next.id).expect("current");
            let compacted = queue.compact_blas(&entry.blas);
            let bytes = compacted.sizes().structure;
            budget = budget.saturating_sub(entry.bytes);
            self.memory.bytes = self.memory.bytes - entry.bytes + bytes;
            entry.blas = compacted;
            entry.handle = handle;
            entry.bytes = bytes;
            self.counters.compactions += 1;
            self.revision = self.revision.wrapping_add(1);
            compacted_any = true;
        }
    }

    fn current(&self, compaction: &Compaction) -> Option<&Entry> {
        self.entries
            .get(&compaction.id)
            .filter(|entry| entry.handle == compaction.handle)
    }

    /// Add this frame's acceleration memory and the work since the last
    /// report to `stats` (the names in [`stat`]). Call once a frame.
    pub fn report(&mut self, stats: &FrameStats) {
        let now = self.counters;
        let last = std::mem::replace(&mut self.reported, now);
        stats.add_named(stat::BLAS_BYTES, self.memory.bytes);
        stats.add_named(
            stat::BLAS_BYTES_BEFORE_COMPACTION,
            self.memory.bytes_before_compaction,
        );
        stats.add_named(stat::SCRATCH_BYTES, now.scratch_bytes - last.scratch_bytes);
        stats.add_named(stat::BLAS_BUILDS, now.builds - last.builds);
        stats.add_named(stat::BLAS_REFITS, now.refits - last.refits);
        stats.add_named(stat::BLAS_COMPACTIONS, now.compactions - last.compactions);
    }

    pub fn get_blas(&self, geometry_id: u64) -> Option<&wgpu::Blas> {
        self.entries.get(&geometry_id).map(|entry| &entry.blas)
    }

    /// The BLAS and its generation, which changes whenever the BLAS is
    /// replaced (a new structure, a compaction) or built in place (a refit,
    /// a deforming BLAS's rebuild).
    fn handle(&self, geometry_id: u64) -> Option<(&wgpu::Blas, u64)> {
        self.entries
            .get(&geometry_id)
            .map(|entry| (&entry.blas, entry.handle))
    }

    fn forget(&mut self, entry: &Entry) {
        self.memory.count -= 1;
        self.memory.bytes -= entry.bytes;
        self.memory.bytes_before_compaction -= entry.built_bytes;
    }

    pub fn remove_blas(&mut self, geometry_id: u64) {
        if let Some(entry) = self.entries.remove(&geometry_id) {
            self.forget(&entry);
            self.revision = self.revision.wrapping_add(1);
        }
    }

    pub fn clear(&mut self) {
        if !self.entries.is_empty() {
            self.entries.clear();
            self.memory = BlasMemory::default();
            self.revision = self.revision.wrapping_add(1);
        }
        self.to_prepare.clear();
        self.preparing.clear();
    }

    /// Evict geometry no longer referenced by the authoritative caster set.
    pub fn retain(&mut self, mut live: impl FnMut(u64) -> bool) {
        let mut removed = BlasMemory::default();
        self.entries.retain(|id, entry| {
            let keep = live(*id);
            if !keep {
                removed.count += 1;
                removed.bytes += entry.bytes;
                removed.bytes_before_compaction += entry.built_bytes;
            }
            keep
        });
        if removed.count > 0 {
            self.memory.count -= removed.count;
            self.memory.bytes -= removed.bytes;
            self.memory.bytes_before_compaction -= removed.bytes_before_compaction;
            self.revision = self.revision.wrapping_add(1);
        }
    }
}

fn size_descriptor(
    geometry: &BlasGeometry<'_>,
    opaque: bool,
) -> wgpu::BlasTriangleGeometrySizeDescriptor {
    wgpu::BlasTriangleGeometrySizeDescriptor {
        vertex_format: wgpu::VertexFormat::Float32x3,
        vertex_count: geometry.vertex_count,
        index_format: geometry.indices.map(|_| wgpu::IndexFormat::Uint32),
        index_count: geometry.indices.map(|_| geometry.index_count),
        flags: if opaque {
            wgpu::AccelerationStructureGeometryFlags::OPAQUE
        } else {
            wgpu::AccelerationStructureGeometryFlags::empty()
        },
    }
}

/// Record a build of `blas` from `segments`. Pools without BLAS_INPUT are
/// copied on the GPU into build inputs; `kept_indices` are earlier index
/// copies to build from instead (a refit must keep its index buffer).
/// Returns the index copies, for a deforming BLAS to keep.
fn record(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    blas: &wgpu::Blas,
    segments: &[BlasSegment<'_>],
    kept_indices: Option<&[Option<wgpu::Buffer>]>,
) -> Vec<Option<wgpu::Buffer>> {
    let mut copy_input = |source: &wgpu::Buffer, offset: u64, size: u64| {
        if source.usage().contains(wgpu::BufferUsages::BLAS_INPUT) {
            return None;
        }
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("BLAS pooled geometry build input"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::BLAS_INPUT,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(source, offset, &buffer, 0, size);
        Some(buffer)
    };
    let vertex_copies: Vec<_> = segments
        .iter()
        .map(|segment| {
            let geometry = &segment.geometry;
            copy_input(
                geometry.vertices,
                u64::from(geometry.first_vertex) * geometry.vertex_stride,
                u64::from(geometry.vertex_count) * geometry.vertex_stride,
            )
        })
        .collect();
    let index_copies: Vec<_> = match kept_indices {
        Some(kept) => kept.to_vec(),
        None => segments
            .iter()
            .map(|segment| {
                let geometry = &segment.geometry;
                geometry.indices.and_then(|indices| {
                    copy_input(
                        indices,
                        u64::from(geometry.first_index) * 4,
                        u64::from(geometry.index_count) * 4,
                    )
                })
            })
            .collect(),
    };
    let sizes: Vec<_> = segments
        .iter()
        .map(|segment| size_descriptor(&segment.geometry, segment.opaque))
        .collect();
    let triangles = segments
        .iter()
        .zip(vertex_copies.iter().zip(&index_copies))
        .zip(&sizes)
        .map(|((segment, (vertex_copy, index_copy)), size)| {
            let geometry = &segment.geometry;
            wgpu::BlasTriangleGeometry {
                size,
                vertex_buffer: vertex_copy.as_ref().unwrap_or(geometry.vertices),
                first_vertex: if vertex_copy.is_some() {
                    0
                } else {
                    geometry.first_vertex
                },
                vertex_stride: geometry.vertex_stride,
                index_buffer: index_copy.as_ref().or(geometry.indices),
                first_index: geometry.indices.map(|_| {
                    if index_copy.is_some() {
                        0
                    } else {
                        geometry.first_index
                    }
                }),
                transform_buffer: None,
                transform_buffer_offset: None,
            }
        })
        .collect();
    let entry = wgpu::BlasBuildEntry {
        blas,
        geometry: wgpu::BlasGeometries::TriangleGeometries(triangles),
    };
    encoder
        .build_acceleration_structures(std::iter::once(&entry), std::iter::empty::<&wgpu::Tlas>());
    index_copies
}

/// When a [`TlasManager`] rebuilds instead of refitting. The defaults are the
/// reasoning in the field docs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TlasPolicy {
    /// Refits between full builds. Every ray of the frame starts in the TLAS,
    /// so its quality matters more than any one BLAS's, and a build is cheap
    /// (linear in instances); 8 bounds a mover's drift to a few frames while
    /// sparing most builds in a scene where a few things move.
    pub max_refits: u32,
    /// Rebuild once the instances moved since the last full build (each
    /// counted once per frame it moved) reach this fraction of the instance
    /// count. A refit leaves every moved instance's ancestors grown, so the
    /// loss follows how much of the scene moved: a frame in which half of a
    /// scene moves rebuilds, as a refit would cost about as much in trace
    /// time as it saves in build time.
    pub moved_fraction: f32,
}

impl Default for TlasPolicy {
    fn default() -> Self {
        Self {
            max_refits: 8,
            moved_fraction: 0.5,
        }
    }
}

/// What a [`TlasManager`] has recorded since it was created.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TlasCounters {
    /// Full builds.
    pub builds: u64,
    /// Refits.
    pub refits: u64,
    /// The full builds [`TlasPolicy`] made (part of `builds`): the others
    /// were a new structure, a changed instance count or membership, or
    /// [`TlasManager::request_rebuild`].
    pub policy_rebuilds: u64,
    /// Scratch bytes the recorded builds and refits needed.
    pub scratch_bytes: u64,
}

/// The frame's Top-Level Acceleration Structure (TLAS); see the module docs
/// for the policy.
pub struct TlasManager {
    tlas: Option<wgpu::Tlas>,
    device: Arc<wgpu::Device>,
    max_instances: u32,
    populated_slots: usize,
    rt_available: bool,
    policy: TlasPolicy,
    last_instances: Vec<TlasInstanceInput>,
    last_masks: Vec<u8>,
    /// Each slot's BLAS handle generation, as last written.
    last_handles: Vec<u64>,
    last_blas_revision: Option<u64>,
    refits: u32,
    moved: u64,
    rebuild_requested: bool,
    counters: TlasCounters,
    reported: TlasCounters,
}

impl TlasManager {
    pub fn new(device: Arc<wgpu::Device>, max_instances: u32) -> Self {
        let rt_available = device
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
            && device.limits().max_tlas_instance_count > 0;
        let max_instances = max_instances
            .max(1)
            .min(device.limits().max_tlas_instance_count.max(1));
        Self {
            tlas: None,
            device,
            max_instances: max_instances.max(1),
            populated_slots: 0,
            rt_available,
            policy: TlasPolicy::default(),
            last_instances: Vec::new(),
            last_masks: Vec::new(),
            last_handles: Vec::new(),
            last_blas_revision: None,
            refits: 0,
            moved: 0,
            rebuild_requested: false,
            counters: TlasCounters::default(),
            reported: TlasCounters::default(),
        }
    }

    pub fn is_rt_available(&self) -> bool {
        self.rt_available
    }

    pub fn policy(&self) -> TlasPolicy {
        self.policy
    }

    pub fn set_policy(&mut self, policy: TlasPolicy) {
        self.policy = policy;
    }

    pub fn counters(&self) -> TlasCounters {
        self.counters
    }

    /// Bytes of the TLAS: its structure and instance buffer.
    pub fn memory(&self) -> u64 {
        self.tlas.as_ref().map_or(0, |tlas| {
            let sizes = tlas.sizes();
            sizes.structure + sizes.instances
        })
    }

    /// Add the TLAS's bytes and its builds since the last report to `stats`
    /// (the names in [`stat`]). Call once a frame.
    pub fn report(&mut self, stats: &FrameStats) {
        let now = self.counters;
        let last = std::mem::replace(&mut self.reported, now);
        stats.add_named(stat::TLAS_BYTES, self.memory());
        stats.add_named(stat::SCRATCH_BYTES, now.scratch_bytes - last.scratch_bytes);
        stats.add_named(stat::TLAS_BUILDS, now.builds - last.builds);
        stats.add_named(stat::TLAS_REFITS, now.refits - last.refits);
    }

    pub fn tlas(&self) -> Option<&wgpu::Tlas> {
        self.tlas.as_ref()
    }

    pub fn as_binding(&self) -> Option<wgpu::BindingResource<'_>> {
        self.tlas.as_ref().map(|t| t.as_binding())
    }

    /// Make the next build a full one: for a caller whose slots changed
    /// meaning in a way the manager cannot see (a slot reused for another
    /// instance of the same BLAS with the same custom data).
    pub fn request_rebuild(&mut self) {
        self.rebuild_requested = true;
    }

    pub fn invalidate(&mut self) {
        self.tlas = None;
        self.populated_slots = 0;
        self.last_instances.clear();
        self.last_handles.clear();
        self.last_blas_revision = None;
    }

    /// Build the TLAS from a complete list of BLAS + transform pairs. Capacity
    /// grows instead of dropping casters. Invalid input clears readiness so a
    /// caller cannot accidentally obtain the previous frame's TLAS via `tlas()`.
    /// The caller must record/submit geometry builds before this build and queries.
    pub fn build(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        instances: &[TlasInstanceInput],
        blas_manager: &BlasManager,
    ) -> Result<(), AccelerationError> {
        self.build_masked(encoder, instances, blas_manager, &[])
    }

    /// Optional per-instance visibility masks. Missing entries retain all bits.
    ///
    /// Only the slots whose instance, mask or BLAS changed are rewritten.
    /// The build is a refit when every slot keeps its BLAS id and custom data
    /// and the count is unchanged, unless [`TlasPolicy`] says the motion
    /// since the last full build calls for one; otherwise a full build.
    pub fn build_masked(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        instances: &[TlasInstanceInput],
        blas_manager: &BlasManager,
        masks: &[u8],
    ) -> Result<(), AccelerationError> {
        if !self.rt_available {
            return Err(AccelerationError::Unsupported);
        }

        let validation = (|| {
            let count =
                u32::try_from(instances.len()).map_err(|_| AccelerationError::TooManyInstances)?;
            if count > self.device.limits().max_tlas_instance_count {
                return Err(AccelerationError::TooManyInstances);
            }
            for input in instances {
                if blas_manager.get_blas(input.geometry_id).is_none() {
                    return Err(AccelerationError::MissingBlas(input.geometry_id));
                }
                if input.transform.iter().any(|value| !value.is_finite()) {
                    return Err(AccelerationError::InvalidTransform);
                }
            }
            Ok(count)
        })();
        let count = match validation {
            Ok(count) => count,
            Err(error) => {
                self.invalidate();
                return Err(error);
            }
        };
        if count > self.max_instances {
            self.max_instances = count
                .checked_next_power_of_two()
                .unwrap_or(count)
                .min(self.device.limits().max_tlas_instance_count);
            self.invalidate();
        }

        // Static scenes commonly call this once per render frame. Reusing the
        // existing TLAS avoids re-recording and rebuilding every instance when
        // neither transforms nor the referenced BLAS handles changed.
        if self.tlas.is_some()
            && self.last_blas_revision == Some(blas_manager.revision())
            && self.last_instances == instances
            && self.last_masks == masks
        {
            return Ok(());
        }

        let fresh = self.tlas.is_none();
        let tlas = self.tlas.get_or_insert_with(|| {
            self.device.create_tlas(&wgpu::CreateTlasDescriptor {
                label: Some("frame_tlas"),
                max_instances: self.max_instances,
                flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE
                    | wgpu::AccelerationStructureFlags::ALLOW_UPDATE,
                update_mode: wgpu::AccelerationStructureUpdateMode::PreferUpdate,
            })
        });

        // Rewrite only the slots whose instance, mask or BLAS handle changed:
        // a moving object costs its own slot, not the whole scene's.
        let mask = |masks: &[u8], i: usize| masks.get(i).copied().unwrap_or(0xFF);
        let mut membership = fresh || count as usize != self.populated_slots;
        let mut moved = 0u64;
        let mut rewritten = false;
        self.last_handles.resize(instances.len(), u64::MAX);
        for (i, input) in instances.iter().enumerate() {
            let (blas, handle) = blas_manager
                .handle(input.geometry_id)
                .expect("validated above");
            let last = self.last_instances.get(i);
            let unchanged = last == Some(input)
                && self.last_handles[i] == handle
                && mask(&self.last_masks, i) == mask(masks, i);
            if unchanged {
                continue;
            }
            match last {
                Some(last)
                    if last.geometry_id == input.geometry_id
                        && last.custom_data == input.custom_data =>
                {
                    // Moved, or its BLAS was rebuilt or compacted.
                    if last.transform != input.transform || self.last_handles[i] != handle {
                        moved += 1;
                    }
                }
                // Another instance in this slot.
                _ => membership = true,
            }
            tlas[i] = Some(wgpu::TlasInstance::new(
                blas,
                input.transform,
                input.custom_data,
                mask(masks, i),
            ));
            self.last_handles[i] = handle;
            rewritten = true;
        }
        for i in count as usize..self.populated_slots {
            tlas[i] = None;
        }
        self.populated_slots = count as usize;
        self.last_blas_revision = Some(blas_manager.revision());
        if !(rewritten || membership || self.rebuild_requested) {
            // Only BLASes this TLAS does not hold changed.
            self.last_masks.clear();
            self.last_masks.extend_from_slice(masks);
            return Ok(());
        }

        let by_policy = !membership
            && !self.rebuild_requested
            && (self.refits >= self.policy.max_refits
                || (self.moved + moved) as f64
                    >= f64::from(self.policy.moved_fraction) * f64::from(count.max(1)));
        let sizes = tlas.sizes();
        if membership || self.rebuild_requested || by_policy {
            // A changed count is a full build in wgpu anyway; asking keeps the
            // intent explicit for a membership change at an equal count.
            tlas.request_rebuild();
            self.rebuild_requested = false;
            self.refits = 0;
            self.moved = 0;
            self.counters.builds += 1;
            self.counters.policy_rebuilds += u64::from(by_policy);
            self.counters.scratch_bytes += sizes.build_scratch;
        } else {
            self.refits += 1;
            self.moved += moved;
            self.counters.refits += 1;
            self.counters.scratch_bytes += sizes.update_scratch;
        }

        let tlas_ref: &wgpu::Tlas = &*tlas;
        encoder.build_acceleration_structures(
            std::iter::empty::<&wgpu::BlasBuildEntry<'_>>(),
            std::iter::once(tlas_ref),
        );
        self.last_instances.clear();
        self.last_instances.extend_from_slice(instances);
        self.last_masks.clear();
        self.last_masks.extend_from_slice(masks);
        Ok(())
    }
}

/// Input for one TLAS instance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TlasInstanceInput {
    pub geometry_id: u64,
    pub transform: [f32; 12],
    /// The instance's 24-bit custom data (a ray query's
    /// `instance_custom_data`): by convention the first row of the
    /// instance's per-segment table, a hit's row being this plus its
    /// `geometry_index`.
    pub custom_data: u32,
}
