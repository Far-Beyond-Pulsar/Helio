use std::sync::Arc;
use crate::{mesh::GpuMesh, material::GpuMaterial};

/// Stable, opaque batch key. Stored as a plain u32 in hot paths — no indirection,
/// no pointer hashing, no Arc<T> addresses. The registry is the single source of truth.
///
/// FNV-1a hashes used in bundle caches operate on `handle.0` directly.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct HismHandle(pub u32);

/// Per-handle entry. Includes optional LOD chain. Trivially indexable.
pub struct HismEntry {
    /// Base mesh/material (LOD 0 — highest quality, also used when no LODs registered).
    pub mesh:     Arc<GpuMesh>,
    pub material: Arc<GpuMaterial>,
    /// Optional coarser LOD levels in order (lowest screen coverage threshold = highest LOD number).
    /// Empty for meshes that do not use LODs.
    pub lod_levels: Vec<LodLevel>,
}

/// One LOD step: replace `base_mesh` when `screen_coverage < threshold`.
pub struct LodLevel {
    pub mesh:                     Arc<GpuMesh>,
    /// From 0.0 (always visible) to 1.0 (only visible when filling the screen).
    /// Evaluated per-instance by the batch builder.
    pub screen_coverage_threshold: f32,
}

impl HismEntry {
    /// Select the appropriate LOD mesh for a given screen coverage ratio.
    pub fn mesh_for_coverage(&self, coverage: f32) -> &Arc<GpuMesh> {
        for lod in self.lod_levels.iter().rev() {
            if coverage < lod.screen_coverage_threshold {
                return &lod.mesh;
            }
        }
        &self.mesh
    }
}

/// Global registry.  Append-only once the frame loop starts.
/// All handles are valid for the lifetime of the registry.
pub struct HismRegistry {
    entries: Vec<HismEntry>,
}

impl HismRegistry {
    pub fn new() -> Self { Self { entries: Vec::new() } }

    /// Register a mesh+material, returns its stable handle.
    pub fn register(
        &mut self,
        mesh:     Arc<GpuMesh>,
        material: Arc<GpuMaterial>,
    ) -> HismHandle {
        let h = HismHandle(self.entries.len() as u32);
        self.entries.push(HismEntry { mesh, material, lod_levels: Vec::new() });
        h
    }

    /// Register with LOD chain. `lod_levels` should be ordered from finest to coarsest.
    pub fn register_with_lods(
        &mut self,
        mesh:       Arc<GpuMesh>,
        material:   Arc<GpuMaterial>,
        lod_levels: Vec<LodLevel>,
    ) -> HismHandle {
        let h = HismHandle(self.entries.len() as u32);
        self.entries.push(HismEntry { mesh, material, lod_levels });
        h
    }

    /// O(1) lookup — panics on out-of-bounds (programmer error, not runtime error).
    #[inline(always)]
    pub fn get(&self, handle: HismHandle) -> &HismEntry {
        &self.entries[handle.0 as usize]
    }

    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }
}

/// Accumulator used by the batch builder: one per unique HismHandle, per frame.
pub struct HismBatch {
    pub handle:   HismHandle,
    pub transforms: Vec<glam::Mat4>,   // flattened to mat4×4 before upload
    pub lod_selected_mesh: Option<u32>, // LOD mesh index or 0 (base)
}

/// FNV-1a hash seed.
const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME:  u64 = 0x100000001b3;

/// Build a structural hash over a draw list for RenderBundle cache invalidation.
/// Does NOT use pointer addresses — uses `(hism_handle u32, instance_count u32)` pairs.
/// Result is deterministic across frames, across reallocs.
pub fn draw_list_hash(draws: &[crate::mesh::DrawCall]) -> u64 {
    let mut h = FNV_OFFSET;
    for d in draws {
        h ^= d.hism_handle.0 as u64;
        h = h.wrapping_mul(FNV_PRIME);
        h ^= d.instance_count as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Structural hash of a shadow draw list for shadow bundle invalidation.
pub fn shadow_draw_list_hash(draws: &[crate::mesh::DrawCall]) -> u64 {
    draw_list_hash(draws)
}
