//! SceneDB-owned visibility and coordinate-space metadata.
//!
//! The gbuffer/cull passes project these rows into transient instance and
//! coordinate-space inputs. No renderer-side group registry is authoritative.
#![allow(deprecated)]
use pulsar_scenedb::gpu::{BufferHandle, BufferKey, GpuMirrorHandle};
use pulsar_scenedb_derive::SceneStore;
use std::marker::PhantomData;

/// Mesh payload authored as a SceneDB component.
///
/// The two variable-length fields share SceneDB's keyed geometry pools. Their
/// generated `*_gpu_handle` accessors expose the allocated ranges needed by
/// `StaticObjectComponent`; the renderer never owns or resolves a mesh asset.
#[derive(SceneStore, Clone, Debug)]
pub struct MeshComponent {
    #[gpu(buffer = "builtin_mesh_vertex", mirror = Once)]
    pub vertices: Vec<helio_core::PackedVertex>,
    #[gpu(buffer = "builtin_mesh_index", mirror = Once)]
    pub indices: Vec<u32>,
}

/// A material authored as a SceneDB component.
///
/// The row deliberately keeps the existing G-buffer shader ABI for this
/// migration step, but its lifetime and updates are now owned by the World.
/// Render passes resolve the packed `"materials"` buffer by key; no renderer
/// material table publication is required for the row itself.
#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "materials")]
pub struct MaterialComponent {
    #[gpu]
    pub base_color: [f32; 4],
    #[gpu]
    pub emissive: [f32; 4],
    #[gpu]
    pub roughness_metallic: [f32; 4],
    #[gpu]
    pub tex_base_color: u32,
    #[gpu]
    pub tex_normal: u32,
    #[gpu]
    pub tex_roughness: u32,
    #[gpu]
    pub tex_emissive: u32,
    #[gpu]
    pub tex_occlusion: u32,
    #[gpu]
    pub workflow: u32,
    #[gpu]
    pub flags: u32,
    #[gpu]
    pub material_class: u32,
    #[gpu]
    pub class_params: [f32; 4],
}

impl From<helio_mats::GpuMaterial> for MaterialComponent {
    fn from(value: helio_mats::GpuMaterial) -> Self {
        bytemuck::cast(value)
    }
}

impl From<MaterialComponent> for helio_mats::GpuMaterial {
    fn from(value: MaterialComponent) -> Self {
        bytemuck::cast(value)
    }
}

impl MaterialComponent {
    /// Construct a material row without involving a renderer or an asset
    /// registry. Texture values are SceneDB texture-store slot indices.
    pub fn new(
        base_color: [f32; 4],
        roughness: f32,
        metallic: f32,
        emissive: [f32; 3],
        emissive_strength: f32,
    ) -> Self {
        let missing = helio_mats::GpuMaterial::NO_TEXTURE;
        Self {
            base_color,
            emissive: [emissive[0], emissive[1], emissive[2], emissive_strength],
            roughness_metallic: [roughness, metallic, 1.5, 0.5],
            tex_base_color: missing,
            tex_normal: missing,
            tex_roughness: missing,
            tex_emissive: missing,
            tex_occlusion: missing,
            workflow: 0,
            flags: 0,
            material_class: 0,
            class_params: [0.0; 4],
        }
    }
}

/// Stable group membership. `group_mask == 0` means always visible.
#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "render_groups")]
pub struct RenderGroupComponent {
    #[gpu]
    pub group_mask: u64,
}

/// Stable content-space index for an authored sublevel.
///
/// Sublevels are indexed content spaces rather than SceneDB entities. Index
/// zero is the default level and always denotes identity/world space. The
/// index is intentionally a plain scalar here: resolving an index to content
/// and projecting it into a transient GPU coordinate-space slot belongs to a
/// later scene-runtime phase.
pub type SubLevelIndex = u32;

/// The default level's authored sublevel index.
pub const DEFAULT_SUBLEVEL_INDEX: SubLevelIndex = 0;

/// SceneDB-authored placement of an indexed sublevel inside its owning level.
///
/// This is the direct-instancing primitive for reusable content (for example,
/// an aircraft or room). It is deliberately separate from portal data:
/// portals display a peer portal's context through an aperture and do not
/// become sublevel actors. The row is registered and mirrored now so the
/// authoritative SceneDB schema exists without changing the current render
/// passes; a later resolver will project it into coordinate-space slots.
#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "sublevel_actors")]
pub struct SubLevelActorComponent {
    #[gpu]
    pub sublevel_index: SubLevelIndex,
    #[gpu]
    pub flags: u32,
    #[gpu]
    pub transform: [[f32; 4]; 4],
}

impl SubLevelActorComponent {
    /// Bit in [`Self::flags`] that enables this actor instance.
    pub const FLAG_ENABLED: u32 = 1 << 0;

    pub fn new(sublevel_index: SubLevelIndex, transform: glam::Mat4) -> Self {
        Self {
            sublevel_index,
            flags: Self::FLAG_ENABLED,
            transform: transform.to_cols_array_2d(),
        }
    }

    pub fn transform(&self) -> glam::Mat4 {
        glam::Mat4::from_cols_array_2d(&self.transform)
    }

    pub fn is_enabled(&self) -> bool {
        self.flags & Self::FLAG_ENABLED != 0
    }
}

/// A movable SceneDB sublevel. The matrix is copied into the transient
/// coordinate-space projection by the render bridge; it is not a renderer
/// scene record.
#[deprecated(
    note = "legacy movable sublevel projection; use SubLevelIndex and SubLevelActorComponent"
)]
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

/// A single-mesh static-object placement: SceneDB-owned mesh/material row
/// indices, a world transform, a culling bound, and the static draw
/// parameters a GPU-driven instancing/culling pipeline needs to read this
/// row directly with no per-frame CPU involvement.
///
/// `SectionedObjectComponent` above covers the multi-material case, which
/// additionally needs the owning `MultiMeshId`/per-section materials tracked
/// elsewhere.
///
/// # Why the draw parameters are stored, not derived per frame
///
/// `index_count`/`first_index`/`vertex_offset` (the mesh's vertex/index
/// range), and `material_class`/`graph_hash` (the material's
/// pipeline-selection key) are authored alongside the row. They are not
/// resolved through a renderer API; the SceneDB asset/ingestion layer owns
/// that resolution before inserting this component.
///
/// `mesh_slot` and `material_slot` are indices into the keyed SceneDB asset
/// buffers. The draw pipeline consumes those plain indices; no renderer
/// handle type is stored or reconstructed here.
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
    /// Previous frame's `transform`, for TAA/TSR motion vectors. Equal to
    /// `transform` (zero apparent velocity) the frame this row is first
    /// inserted or whenever a caller uses [`Self::with_transform`] without
    /// tracking real per-frame motion -- a quality simplification (a
    /// stationary-looking first frame), never a correctness one.
    #[gpu]
    pub prev_transform: [[f32; 4]; 4],
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
    /// See `helio_pass_object_batch::GpuInstanceData::flags`.
    #[gpu]
    pub flags: u32,
}

impl StaticObjectComponent {
    /// Construct an object row from already-resolved SceneDB asset metadata.
    ///
    /// `mesh_slot` and `material_slot` are plain SceneDB row indices. The
    /// renderer is intentionally absent: resolving those indices and the
    /// associated draw range belongs to the asset/component authoring side.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mesh_slot: u32,
        mesh_generation: u32,
        material_slot: u32,
        material_generation: u32,
        transform: glam::Mat4,
        bounds: [f32; 4],
        index_count: u32,
        first_index: u32,
        vertex_offset: i32,
        material_class: u32,
        graph_hash: u64,
        flags: u32,
    ) -> Self {
        Self {
            mesh_slot,
            mesh_generation,
            material_slot,
            material_generation,
            transform: transform.to_cols_array_2d(),
            prev_transform: transform.to_cols_array_2d(),
            normal_mat: normal_matrix_cols(transform),
            bounds,
            index_count,
            first_index,
            vertex_offset,
            material_class,
            graph_hash_lo: graph_hash as u32,
            graph_hash_hi: (graph_hash >> 32) as u32,
            flags,
        }
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
        self.prev_transform = self.transform;
        self.transform = transform.to_cols_array_2d();
        self.normal_mat = normal_matrix_cols(transform);
        self.bounds = bounds;
        self
    }
}

/// Inverse-transpose of `m`'s upper-left 3x3, as three padded columns
/// (matches `helio_pass_object_batch::GpuInstanceData::normal_mat`'s layout).
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
binding!(
    SubLevelActorSceneBinding,
    SubLevelActorComponent,
    "sublevel_actors"
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
        assert_eq!(std::mem::size_of::<MaterialComponent>(), 96);
        assert_eq!(std::mem::size_of::<RenderGroupComponent>(), 8);
        assert_eq!(std::mem::size_of::<SubLevelActorComponent>(), 72);
        assert_eq!(std::mem::size_of::<SublevelComponent>(), 72);
        assert_eq!(std::mem::size_of::<SectionedObjectComponent>(), 32);
        assert_eq!(std::mem::align_of::<SubLevelActorComponent>(), 4);
        assert_eq!(std::mem::align_of::<SublevelComponent>(), 8);
    }

    #[test]
    fn sublevel_actor_round_trips_index_and_transform() {
        let actor =
            SubLevelActorComponent::new(7, glam::Mat4::from_translation(glam::vec3(3.0, 4.0, 5.0)));
        assert_eq!(actor.sublevel_index, 7);
        assert_eq!(actor.flags, SubLevelActorComponent::FLAG_ENABLED);
        assert_eq!(actor.transform().w_axis, glam::vec4(3.0, 4.0, 5.0, 1.0));
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
