//! MeshDrawCommand — pre-built, cached draw commands.
//!
//! Equivalent to Unreal Engine's `FMeshDrawCommand`.  Each command is built
//! once when an object is registered (or its material changes) and cached in
//! a sorted list.  Passes iterate the lists directly without any per-frame
//! HashMap scan or batch-sort work.
//!
//! # Equivalents
//!
//! | Helio                    | Unreal                     |
//! |--------------------------|----------------------------|
//! | `MeshDrawCommand`        | `FMeshDrawCommand`         |
//! | `DrawCommandCache`       | `FMeshPassProcessor` cache |
//! | `DrawCommandCache::insert` | `AddMeshBatch`           |
//! | `DrawCommandCache::remove` | `RemovePrimitive`        |

use std::sync::Arc;

/// A pre-built draw command for one primitive in one render pass.
///
/// Storing `vertex_buf`, `index_buf`, `material_bg` as `Arc` means commands
/// can be cheaply cloned into multiple pass lists without heap allocations.
/// The `sort_key` is precomputed so list insertions maintain sorted order
/// without an explicit comparison loop.
#[derive(Clone)]
pub struct MeshDrawCommand {
    pub vertex_buf:   Arc<wgpu::Buffer>,
    pub index_buf:    Arc<wgpu::Buffer>,
    pub index_count:  u32,
    /// Offset (in indices) of this mesh's data in the index buffer.
    pub first_index:  u32,
    /// Group 1 material bind group for this draw call.
    pub material_bg:  Arc<wgpu::BindGroup>,
    /// Slot in the GPU Scene buffer.
    /// Used as `first_instance` in draw calls so the vertex shader can
    /// read the model transform via `@builtin(instance_index)`.
    pub primitive_id: u32,
    /// Precomputed sort key (material pointer in high 32 bits, primitive_id in
    /// low 32 bits).  Sorting by this key minimises GPU state changes.
    pub sort_key:     u64,
    pub transparent:  bool,
    pub cast_shadow:  bool,
}

impl MeshDrawCommand {
    /// Build a sort key: material pointer in high 32 bits, primitive_id in low.
    pub fn make_sort_key(material_bg: &Arc<wgpu::BindGroup>, primitive_id: u32) -> u64 {
        ((Arc::as_ptr(material_bg) as u64) << 32) | (primitive_id as u64)
    }
}

/// Sorted, cached draw command lists for all render passes.
///
/// Lists are rebuilt only when the scene changes structure (object added,
/// removed, or its material changes).  Static scenes cost zero per-frame
/// rebuild at steady state.
#[derive(Default)]
pub struct DrawCommandCache {
    /// Opaque objects sorted by `sort_key` (PSO then material) for minimal
    /// render state changes.
    pub opaque:      Vec<MeshDrawCommand>,
    /// Transparent objects.  These are depth-sorted per-frame in
    /// `TransparentPass`; the cache stores them in insertion order.
    pub transparent: Vec<MeshDrawCommand>,
    /// All shadow-casting objects (superset of opaque+transparent).
    pub shadow:      Vec<MeshDrawCommand>,
    /// Monotonically increasing counter, bumped on every structural mutation.
    /// Passes can cache their own generation and skip processing when unchanged.
    pub generation:  u64,
}

impl DrawCommandCache {
    pub fn new() -> Self { Self::default() }

    /// Insert a command into the relevant list(s) and maintain sort order.
    pub fn insert(&mut self, cmd: MeshDrawCommand) {
        if cmd.transparent {
            self.transparent.push(cmd.clone());
            // Transparent list is depth-sorted in the pass, not here.
        } else {
            // Maintain sorted order for opaque list (binary search insertion).
            let pos = self.opaque.partition_point(|c| c.sort_key <= cmd.sort_key);
            self.opaque.insert(pos, cmd.clone());
        }

        if cmd.cast_shadow {
            let pos = self.shadow.partition_point(|c| c.sort_key <= cmd.sort_key);
            self.shadow.insert(pos, cmd.clone());
        }

        self.generation = self.generation.wrapping_add(1);
    }

    /// Remove all commands for the given primitive_id from every list.
    pub fn remove(&mut self, primitive_id: u32) {
        self.opaque.retain(|c| c.primitive_id != primitive_id);
        self.transparent.retain(|c| c.primitive_id != primitive_id);
        self.shadow.retain(|c| c.primitive_id != primitive_id);
        self.generation = self.generation.wrapping_add(1);
    }

    /// Replace all commands for a primitive (e.g. material changed).
    pub fn replace(&mut self, primitive_id: u32, new_cmd: MeshDrawCommand) {
        self.remove(primitive_id);
        self.insert(new_cmd);
    }

    /// Total draw commands across all lists.
    pub fn total_commands(&self) -> usize {
        self.opaque.len() + self.transparent.len()
    }

    pub fn is_empty(&self) -> bool {
        self.opaque.is_empty() && self.transparent.is_empty()
    }
}

/// CPU-side metadata kept alongside each GPU Scene slot.
///
/// This is the lean replacement for the old `RegisteredProxy` — it holds only
/// what can't be derived from the GPU Scene buffer itself.
pub struct ProxyMeta {
    /// Slot in the GPU Scene (used for all GPU Scene updates).
    pub slot:           crate::gpu_scene::PrimitiveSlot,
    /// Cached CPU-side transform hash (skip GPU updates for unchanged matrices).
    pub transform_hash: u64,
    /// Whether this object contributes to rendering.
    pub enabled:        bool,
    /// Material bind group (needed to build new MDC entries on material change).
    pub material_bg:    Arc<wgpu::BindGroup>,
    /// Mesh info needed to rebuild the MDC entry.
    pub vertex_buf:     Arc<wgpu::Buffer>,
    pub index_buf:      Arc<wgpu::Buffer>,
    pub index_count:    u32,
    pub transparent:    bool,
    pub cast_shadow:    bool,
}
