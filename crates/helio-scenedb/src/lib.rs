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
//! into a single read-only bind group that Helio's render/compute passes
//! consume directly -- no data ever crosses through a CPU-side copy.
//! [`wgsl::SCENE_BINDINGS_WGSL`] is the renderer-side source of truth for
//! the struct layouts; `tests/seam_smoke.rs` is the proof that the whole
//! chain -- SceneDB write API, frame boundary, GPU buffer, this bind group,
//! a real shader, a Helio-owned output buffer, readback -- works byte-exact
//! on the actual vendored wgpu-30 stack.

pub mod wgsl;

use pulsar_scenedb::gpu::{ClusterBuffer, MeshRegistry, MeshletBuffer, SceneGpuStore};

/// The Helio-side handle to every SceneDB-owned GPU buffer, bound read-only
/// into one bind group at bindings 0-7 ([`wgsl::SCENE_BINDINGS_WGSL`]):
///
/// | binding | contents            | source accessor                                          |
/// |---------|----------------------|-----------------------------------------------------------|
/// | 0       | instance transforms  | `SceneGpuStore::transform_buffer`                          |
/// | 1       | instance info        | `SceneGpuStore::instance_info_buffer`                      |
/// | 2       | slot mirror          | `SceneGpuStore::slot_mirror_buffer`                        |
/// | 3       | generations          | `SceneGpuStore::generation_buffer`                         |
/// | 4       | mesh configurator    | `MeshRegistry::buffer`                                     |
/// | 5       | cluster DAG          | `ClusterBuffer::buffer`                                    |
/// | 6       | meshlets             | `MeshletBuffer::buffer`                                    |
/// | 7       | cell metadata        | `SceneGpuStore::cell_metadata_buffer`                      |
///
/// Geometry (vertex/index) buffers bind at draw time, not here (they vary
/// per draw call); textures bind through Helio's own bindless array. Every
/// entry above is a read-only storage buffer -- SceneDB owns the write path
/// (`write_transform`/`write_instance_info`/the frame-boundary drivers,
/// `MeshRegistry::register`, `StreamingGrid::write_cell_metadata`); Helio
/// only ever reads.
pub struct SceneDbBinding {
    pub layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
}

impl SceneDbBinding {
    /// Rebuilt at renderer construction -- Test 13's mechanism (M3-b): the
    /// bind group is torn down and rebuilt across a renderer drop/rebind
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
    pub fn new(
        device: &wgpu::Device,
        store: &SceneGpuStore,
        meshes: &MeshRegistry,
        clusters: &ClusterBuffer,
        meshlets: &MeshletBuffer,
    ) -> Self {
        let entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            // Visible to every stage: cull/draw (M3-b) reads these from
            // compute and vertex/fragment stages alike; there is no
            // per-binding reason to narrow visibility at the seam.
            visibility: wgpu::ShaderStages::all(),
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scenedb-binding-layout"),
            entries: &[
                entry(0),
                entry(1),
                entry(2),
                entry(3),
                entry(4),
                entry(5),
                entry(6),
                entry(7),
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scenedb-binding"),
            layout: &layout,
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
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: clusters.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: meshlets.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: store.cell_metadata_buffer().as_entire_binding(),
                },
            ],
        });
        Self { layout, bind_group }
    }
}
