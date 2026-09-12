//! SceneDB-owned visibility and coordinate-space metadata.
//!
//! The gbuffer/cull passes project these rows into transient instance and
//! coordinate-space inputs. No renderer-side group registry is authoritative.
use pulsar_scenedb::gpu::{BufferHandle, BufferKey, GpuMirrorHandle};
use pulsar_scenedb_derive::SceneStore;
use std::marker::PhantomData;

/// Stable group membership. `group_mask == 0` means always visible.
#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "render_groups")]
pub struct RenderGroupComponent {
    #[gpu]
    pub group_mask: u64,
}

/// A movable SceneDB sublevel. The matrix is copied into the transient
/// coordinate-space projection by the render bridge; it is not a renderer
/// scene record.
#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "sublevels")]
pub struct SublevelComponent {
    #[gpu]
    pub group_mask: u64,
    #[gpu]
    pub placement: [[f32; 4]; 4],
}

/// The renderer consumes sectioned-object placement as a transient set of
/// draw records, while the asset and authored section list remain on the
/// owning SceneDB entity (`SectionedMeshComponent`/`MeshObjectComponent`).
/// This small packed row is the only persistent GPU-facing state needed by
/// the sectioned-object pass.
#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "sectioned_objects")]
pub struct SectionedObjectComponent {
    #[gpu]
    pub bounds: [f32; 4],
    #[gpu]
    pub group_mask: u64,
    #[gpu]
    pub flags: u32,
    #[gpu]
    pub movable: u32,
}

/// A single-mesh static-object placement: mesh + material asset references,
/// a world transform, a culling bound, and the resolved static draw
/// parameters a GPU-driven instancing/culling pipeline needs to read this
/// row directly with no per-frame CPU involvement.
///
/// `SectionedObjectComponent` above covers the multi-material case, which
/// additionally needs the owning `MultiMeshId`/per-section materials tracked
/// elsewhere.
///
/// # Why the draw parameters are stored, not derived per frame
///
/// `mesh_id` (= the mesh asset's pool slot), `index_count`/`first_index`/
/// `vertex_offset` (its vertex/index range), and `material_class`/
/// `graph_hash` (the owning material's pipeline-selection key) are all
/// **static properties of the mesh/material assets**, immutable for the
/// life of those assets. Re-deriving them every frame would require a
/// per-frame CPU query; instead the frontend resolves them ONCE, via
/// `Renderer::mesh_slice`/`material_batch_key` (read-only asset queries,
/// not scene authoring), at the moment it spawns this component — see
/// [`StaticObjectComponent::new`].
///
/// `mesh`/`material` handles themselves are stored as raw `(slot,
/// generation)` pairs rather than `helio::MeshId`/`MaterialId` directly:
/// those handle types aren't `bytemuck::Pod`, so this row keeps their exact
/// bit pattern and reconstructs the typed handle with
/// `MeshId::from_raw`/`MaterialId::from_raw` on read (see `mesh()`/
/// `material()` below) — needed only if the frontend wants to re-resolve
/// the asset later (e.g. after a hot-reload); the draw pipeline itself never
/// needs the typed handle, only the plain `u32`s below.
///
/// # What's NOT solved by this component alone
///
/// Building the *grouped* `draw_calls`/`instances`/`aabbs` arrays a
/// GPU-driven culling pass reads (batching same-mesh-same-material
/// instances into one indirect draw call, partitioned into contiguous
/// opaque/transparent/forward ranges by `material_class`) requires sorting
/// every live `StaticObjectComponent` row by `(material_class, graph_hash,
/// mesh_id, material_id)` — a cross-entity operation, not a per-row one.
/// That GPU sort/group/dispatch pipeline is specified but not yet
/// implemented; see the Helio issue tracking it. Until it lands, a
/// consumer must still assemble those arrays itself from a `World` query
/// over this component (a per-frame CPU cost, same category as the
/// pre-existing shadow-atlas scoring — tracked, not silently hidden).
#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "static_objects")]
pub struct StaticObjectComponent {
    #[gpu]
    pub mesh_slot: u32,
    #[gpu]
    pub mesh_generation: u32,
    #[gpu]
    pub material_slot: u32,
    #[gpu]
    pub material_generation: u32,
    #[gpu]
    pub transform: [[f32; 4]; 4],
    #[gpu]
    pub normal_mat: [[f32; 4]; 3],
    #[gpu]
    pub bounds: [f32; 4],
    /// Resolved once at spawn time from `Renderer::mesh_slice` -- see the
    /// struct doc's "Why the draw parameters are stored" section.
    #[gpu]
    pub index_count: u32,
    #[gpu]
    pub first_index: u32,
    #[gpu]
    pub vertex_offset: i32,
    /// Resolved once at spawn time from `Renderer::material_batch_key`.
    #[gpu]
    pub material_class: u32,
    #[gpu]
    pub graph_hash_lo: u32,
    #[gpu]
    pub graph_hash_hi: u32,
    /// See `libhelio::GpuInstanceData::flags`.
    #[gpu]
    pub flags: u32,
}

impl StaticObjectComponent {
    /// `renderer` supplies the one-time `mesh_slice`/`material_batch_key`
    /// asset queries this needs — see the struct doc.
    pub fn new(
        renderer: &helio::Renderer,
        mesh: helio::MeshId,
        material: helio::MaterialId,
        transform: glam::Mat4,
        bounds: [f32; 4],
        flags: u32,
    ) -> Option<Self> {
        let slice = renderer.mesh_slice(mesh)?;
        let (material_class, graph_hash) = renderer.material_batch_key(material)?;
        let normal = normal_matrix_cols(transform);
        Some(Self {
            mesh_slot: mesh.slot(),
            mesh_generation: mesh.generation(),
            material_slot: material.slot(),
            material_generation: material.generation(),
            transform: transform.to_cols_array_2d(),
            normal_mat: normal,
            bounds,
            index_count: slice.index_count,
            first_index: slice.first_index,
            vertex_offset: slice.first_vertex as i32,
            material_class,
            graph_hash_lo: graph_hash as u32,
            graph_hash_hi: (graph_hash >> 32) as u32,
            flags,
        })
    }

    pub fn mesh(&self) -> helio::MeshId {
        helio::MeshId::from_raw(self.mesh_slot, self.mesh_generation)
    }

    pub fn material(&self) -> helio::MaterialId {
        helio::MaterialId::from_raw(self.material_slot, self.material_generation)
    }

    pub fn transform(&self) -> glam::Mat4 {
        glam::Mat4::from_cols_array_2d(&self.transform)
    }

    pub fn graph_hash(&self) -> u64 {
        (self.graph_hash_hi as u64) << 32 | self.graph_hash_lo as u64
    }

    /// Re-place this row at a new transform/bounds, recomputing `normal_mat`
    /// to match (it's derived from `transform`, so the two must never drift
    /// apart). `mesh`/`material` and their resolved draw parameters are
    /// static asset properties and are carried over unchanged -- moving an
    /// object never needs to re-resolve them.
    pub fn with_transform(mut self, transform: glam::Mat4, bounds: [f32; 4]) -> Self {
        self.transform = transform.to_cols_array_2d();
        self.normal_mat = normal_matrix_cols(transform);
        self.bounds = bounds;
        self
    }
}

/// Inverse-transpose of `m`'s upper-left 3x3, as three padded columns
/// (matches `libhelio::GpuInstanceData::normal_mat`'s layout).
fn normal_matrix_cols(m: glam::Mat4) -> [[f32; 4]; 3] {
    let m3 = glam::Mat3::from_mat4(m);
    let inv_t = m3.inverse().transpose();
    let c0 = inv_t.x_axis;
    let c1 = inv_t.y_axis;
    let c2 = inv_t.z_axis;
    [
        [c0.x, c0.y, c0.z, 0.0],
        [c1.x, c1.y, c1.z, 0.0],
        [c2.x, c2.y, c2.z, 0.0],
    ]
}

macro_rules! binding {
    ($name:ident, $ty:ty, $key:literal) => {
        #[derive(Clone)]
        pub struct $name {
            handle: BufferHandle,
            _marker: PhantomData<$ty>,
        }
        impl $name {
            pub fn resolve(mirror: &GpuMirrorHandle) -> Option<Self> {
                mirror
                    .store()
                    .resolve_buffer_handle(BufferKey::of($key))
                    .map(|handle| Self {
                        handle,
                        _marker: PhantomData,
                    })
            }
            pub fn buffer(&self) -> &wgpu::Buffer {
                &self.handle.buffer
            }
            pub fn epoch(&self) -> u64 {
                self.handle.epoch
            }
        }
    };
}

binding!(
    RenderGroupSceneBinding,
    RenderGroupComponent,
    "render_groups"
);
binding!(SublevelSceneBinding, SublevelComponent, "sublevels");
binding!(
    SectionedObjectSceneBinding,
    SectionedObjectComponent,
    "sectioned_objects"
);

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn component_layouts_are_stable() {
        assert_eq!(std::mem::size_of::<RenderGroupComponent>(), 8);
        assert_eq!(std::mem::size_of::<SublevelComponent>(), 72);
        assert_eq!(std::mem::size_of::<SectionedObjectComponent>(), 32);
        assert_eq!(std::mem::align_of::<SublevelComponent>(), 8);
    }
}
#[cfg(test)]
mod lifecycle_tests {
    use super::{RenderGroupComponent, SublevelComponent};
    #[test]
    fn group_and_sublevel_rows_have_independent_lifetimes() {
        let mut world = pulsar_scenedb::World::new();
        let group = world.spawn();
        let level = world.spawn();
        let g = RenderGroupComponent { group_mask: 1 };
        let s = SublevelComponent {
            group_mask: 1,
            placement: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [3.0, 4.0, 5.0, 1.0],
            ],
        };
        world.insert(group, g);
        world.insert(level, s);
        assert_eq!(world.remove::<RenderGroupComponent>(group), Some(g));
        assert!(world.get::<SublevelComponent>(level).is_some());
        assert_eq!(world.remove::<SublevelComponent>(level), Some(s));
    }
}
