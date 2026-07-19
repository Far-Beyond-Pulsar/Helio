//! `helio-scenedb`: the Helio <-> SceneDB binding seam (M3-a T9, design
//! Rev 2 S5). This is the ONE crate in the Helio workspace allowed to depend
//! on `pulsar_scenedb` -- see `Cargo.toml`'s dependency comment for why this
//! path-dep resolves only when Helio is vendored at `crates/renderer/helio`
//! inside Pulsar-Native (CONTRACTS C0: SceneDB never depends on Helio; this
//! is the inverse edge, and it lives entirely on the Helio side).
//!
//! [`SceneDbBinding`] binds every SceneDB-owned GPU buffer relevant to
//! rendering (instance transforms, instance info, the slot/generation
//! mirrors, the mesh/cluster/meshlet asset tables, and per-cell metadata)
//! into TWO read-only bind groups (M3-b T4, contract #47 -- see the struct
//! doc below) that Helio's render/compute passes consume directly -- no
//! data ever crosses through a CPU-side copy. [`wgsl::SCENE_BINDINGS_WGSL`]
//! is the renderer-side source of truth for the struct layouts and the
//! `@group`/`@binding` indices; `tests/seam_smoke.rs` is the proof that the
//! whole chain -- SceneDB write API, frame boundary, GPU buffer, these bind
//! groups, a real shader, a Helio-owned output buffer, readback -- works
//! byte-exact on the actual vendored wgpu-30 stack.

pub mod wgsl;

use pulsar_scenedb::gpu::{
    ClusterBuffer, MaterialRegistry, MeshRegistry, MeshletBuffer, SceneGpuStore,
};

/// The Helio-side handle to every SceneDB-owned GPU buffer, split across two
/// read-only bind groups ([`wgsl::SCENE_BINDINGS_WGSL`]) -- M3-b T4, closing
/// perf-validation report contract #47 (the seam's original single 9-entry
/// group was ONE OVER the WebGPU default per-stage storage-buffer budget,
/// `Limits::default().max_storage_buffers_per_shader_stage == 8`).
///
/// ## `@group(0)` -- the cull-read set (`cull_layout` / `cull_bind_group`)
///
/// | binding | contents            | source accessor                        |
/// |---------|----------------------|-----------------------------------------|
/// | 0       | instance transforms  | `SceneGpuStore::transform_buffer`       |
/// | 1       | instance info        | `SceneGpuStore::instance_info_buffer`   |
/// | 2       | slot mirror          | `SceneGpuStore::slot_mirror_buffer`     |
/// | 3       | generations          | `SceneGpuStore::generation_buffer`      |
/// | 4       | mesh configurator    | `MeshRegistry::buffer`                  |
///
/// ## `@group(1)` -- the draw/material set (`draw_layout` / `draw_bind_group`)
///
/// | binding | contents            | source accessor                        |
/// |---------|----------------------|-----------------------------------------|
/// | 0       | cluster DAG          | `ClusterBuffer::buffer`                 |
/// | 1       | meshlets             | `MeshletBuffer::buffer`                 |
/// | 2       | cell metadata        | `SceneGpuStore::cell_metadata_buffer`   |
/// | 3       | material registry    | `MaterialRegistry::buffer`              |
///
/// `@group(2)` is RESERVED for M3-b T5's per-view `ViewTokenBuffers`
/// (`tokens`, `expected_gens`) -- NOT added by this task; T5 claims that
/// index without renumbering anything here.
///
/// Geometry (vertex/index) buffers bind at draw time, not here (they vary
/// per draw call); textures bind through Helio's own bindless array. Every
/// entry above is a read-only storage buffer -- SceneDB owns the write path
/// (`write_transform`/`write_instance_info`/the frame-boundary drivers,
/// `MeshRegistry::register`, `MaterialRegistry::register`,
/// `StreamingGrid::write_cell_metadata`); Helio only ever reads.
///
/// ## Visibility split and the per-stage arithmetic (M3-b T4)
///
/// The split is not just "cull entries vs. draw entries" -- within
/// `@group(0)`, TWO of the five entries (`instances`, `instance_info`) also
/// need to be readable from the vertex/fragment stages, because the β draw
/// executor (M3-b T7, design's minimal indirect-draw path) fetches the
/// instance transform (via `visible_instance_ids` + instance index) and
/// reads `InstanceInfo.mesh_index` to pick a flat output color -- it never
/// touches `slot_mirror`/`generations`/`mesh_meta`, which are cull-internal
/// bookkeeping (generation validation and the mesh-bounds/AABB check, design
/// §4's β term list). Giving the WHOLE of `@group(0)` blanket
/// `VERTEX_FRAGMENT | COMPUTE` visibility (the group-level reading of the
/// plan's instruction) would put `@group(0)`'s 5 entries and `@group(1)`'s 4
/// draw entries in the SAME vertex/fragment budget -- 9, busting the very
/// 8-entry default limit this task exists to fix, just on a different stage
/// than #47's original compute-stage MISS. Per-ENTRY visibility avoids that
/// while still giving the cull compute pass everything it needs:
///
/// ```text
/// @group(0) (cull-read set, 5 entries)
///   binding 0 instances       -- VERTEX_FRAGMENT | COMPUTE (cull transform fetch + draw vertex fetch)
///   binding 1 instance_info   -- VERTEX_FRAGMENT | COMPUTE (cull mesh_index bounds check + draw flat-color pick)
///   binding 2 slot_mirror     -- COMPUTE only (cull generation-validation indexing)
///   binding 3 generations     -- COMPUTE only (cull generation-validation)
///   binding 4 mesh_meta       -- COMPUTE only (cull mesh-bounds check + local AABB)
///
/// @group(1) (draw/material set, 4 entries) -- all VERTEX_FRAGMENT only,
/// cull never binds this group:
///   binding 0 clusters, binding 1 meshlets, binding 2 cell_meta, binding 3 materials
/// ```
///
/// **Per-stage totals (arithmetic, current M3-b T4 state, before T5's
/// `@group(2)`):**
///
/// - COMPUTE: group0's COMPUTE-visible entries (5) + group1's COMPUTE-visible
///   entries (0, none of group1 is COMPUTE-visible) = **5 / 8** (3 headroom).
///   T5 adds `ViewTokenBuffers` (`tokens` + `expected_gens`, COMPUTE-only,
///   `@group(2)`, 2 entries) on top of this: 5 + 2 = **7 / 8** (1 headroom
///   left for whatever the cull pass's own output buffers turn out to need
///   in this same pipeline layout -- likely their own group, per the T4
///   smoke test's own out-buffer pattern below, not a claim on this
///   remaining 1).
/// - VERTEX: group0's VERTEX_FRAGMENT-visible entries (2: `instances`,
///   `instance_info`) + group1's VERTEX_FRAGMENT-visible entries (4, all of
///   group1) = **6 / 8** (2 headroom).
/// - FRAGMENT: identical to VERTEX (same two groups, same visibility bits)
///   = **6 / 8** (2 headroom).
///
/// This IS the budget-relevant choice the plan calls out: it is what lets
/// `tests/seam_smoke.rs` drop `adapter.limits()` for `wgpu::Limits::
/// default()` and still pass (its own compute pipeline only ever binds
/// `@group(0)` for the SceneDB side -- 5 COMPUTE-visible entries -- plus its
/// own 2 output buffers at `@group(2)`, `5 + 2 = 7 ≤ 8`; `@group(1)` is
/// still constructed and bound at pipeline-layout index 1 so the shader
/// module's declarations of it type-check and the pipeline layout is
/// positionally valid, but none of its entries are COMPUTE-visible so it
/// contributes 0 to that stage's count).
pub struct SceneDbBinding {
    pub cull_layout: wgpu::BindGroupLayout,
    pub cull_bind_group: wgpu::BindGroup,
    pub draw_layout: wgpu::BindGroupLayout,
    pub draw_bind_group: wgpu::BindGroup,
}

impl SceneDbBinding {
    /// Rebuilt at renderer construction -- Test 13's mechanism (M3-b): the
    /// bind groups are torn down and rebuilt across a renderer drop/rebind
    /// window, never mutated in place, so this constructor is the ONLY way
    /// a `SceneDbBinding` comes into existence.
    ///
    /// ## Signature note (checked against the plan's sketch, no deviation)
    ///
    /// The M3-a task spec speculated that binding 7 (cell metadata) might
    /// need an extra `&StreamingGrid` parameter, since `StreamingGrid` is
    /// what classifies domains and computes alpha. It does not: cell
    /// metadata's GPU buffer is allocated by, and its accessor lives on,
    /// `SceneGpuStore` itself (`cell_metadata_buffer()`, `gpu/scene_store.rs`
    /// line ~796) -- `StreamingGrid::write_cell_metadata(queue, buf)` only
    /// *writes into* that buffer (obtained by the caller from `store`), it
    /// does not own or expose it. Binding (read-only, at renderer
    /// construction) needs only the buffer handle, which `store` already
    /// provides. So this signature matches the plan's sketch exactly, with
    /// no extra parameter -- verified by reading `gpu/scene_store.rs` and
    /// `gpu/grid.rs` (`grep pub fn.*buffer` finds `cell_metadata_buffer` on
    /// `SceneGpuStore`; `grid.rs` exposes no buffer accessor of its own).
    /// Unchanged by the M3-b T4 group split (same five source stores feed
    /// both groups; only the bind-group/layout construction below changed).
    pub fn new(
        device: &wgpu::Device,
        store: &SceneGpuStore,
        meshes: &MeshRegistry,
        clusters: &ClusterBuffer,
        meshlets: &MeshletBuffer,
        materials: &MaterialRegistry,
    ) -> Self {
        // -- @group(0): the cull-read set. Per-entry visibility (see this
        // struct's doc comment for the full arithmetic): `instances` and
        // `instance_info` are also read by the draw pass's vertex/fragment
        // stages (M3-b T7), everything else here is cull-compute-only.
        let cull_shared_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT | wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let cull_only_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let cull_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scenedb-cull-binding-layout"),
            entries: &[
                cull_shared_entry(0), // instances
                cull_shared_entry(1), // instance_info
                cull_only_entry(2),   // slot_mirror
                cull_only_entry(3),   // generations
                cull_only_entry(4),   // mesh_meta
            ],
        });
        let cull_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scenedb-cull-binding"),
            layout: &cull_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: store.transform_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: store.instance_info_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: store.slot_mirror_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: store.generation_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: meshes.buffer().as_entire_binding(),
                },
            ],
        });

        // -- @group(1): the draw/material set. All entries VERTEX_FRAGMENT
        // only -- the cull compute pass never binds this group.
        let draw_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let draw_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scenedb-draw-binding-layout"),
            entries: &[
                draw_entry(0), // clusters
                draw_entry(1), // meshlets
                draw_entry(2), // cell_meta
                draw_entry(3), // materials
            ],
        });
        let draw_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scenedb-draw-binding"),
            layout: &draw_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: clusters.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: meshlets.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: store.cell_metadata_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: materials.buffer().as_entire_binding(),
                },
            ],
        });

        Self { cull_layout, cull_bind_group, draw_layout, draw_bind_group }
    }
}
