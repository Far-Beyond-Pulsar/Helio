//! SceneDB components for Helio's standalone hosts: examples, demos, the
//! wasm runner, snapshots and tests.
//!
//! Helio defines no components (docs/architecture/layered-rewrite.md,
//! phase 1): its passes read buffers under [`helio_core::BufferContract`]s,
//! and whoever produces the data registers the components that fill them.
//! Pulsar registers its own; this crate is the same thing for Helio's own
//! hosts. Nothing below the host layer may depend on it.
//!
//! Row components ([`rows`]) carry their contract row's exact fields, so
//! struct literals read like the row; `same_layout!` fails the build if they
//! drift apart. Insert a bare row with [`WorldRows::insert_row`]: SceneDB
//! accepts any `Send + Sync` value as a component, so `world.insert(e, row)`
//! compiles but is never mirrored to the GPU.

use pulsar_scenedb::{Entity, World};
use pulsar_scenedb_derive::SceneStore;

pub mod ray_proxies;
pub mod ray_tracing;
pub mod rows;
pub use rows::*;

/// A contract row with the component that stores it.
pub trait RowComponent: Sized {
    type Component: From<Self> + Send + Sync + 'static;
}

/// Insert contract rows as their components.
pub trait WorldRows {
    /// Insert `row` as its [`RowComponent::Component`], so SceneDB mirrors it
    /// into the contract's buffer.
    fn insert_row<R: RowComponent>(&mut self, entity: Entity, row: R);
}

impl WorldRows for World {
    fn insert_row<R: RowComponent>(&mut self, entity: Entity, row: R) {
        self.insert(entity, R::Component::from(row));
    }
}

/// Ties a component to its contract row: same size, every field at the same
/// offset, stored under the contract's key. Adds the conversions. (Removal
/// needs nothing: `#[derive(SceneStore)]` already zeroes a component's GPU
/// row when it leaves an entity.)
#[macro_export]
macro_rules! same_layout {
    ($component:ident, $row:path, $contract:path, $key:literal; $($field:ident),* $(,)?) => {
        const _: () = {
            assert!(
                ::core::mem::size_of::<$component>() == ::core::mem::size_of::<$row>(),
                concat!(stringify!($component), " and its row differ in size")
            );
            assert!(
                ::core::mem::align_of::<$component>() == ::core::mem::align_of::<$row>(),
                concat!(stringify!($component), " and its row differ in alignment")
            );
            $(
                assert!(
                    ::core::mem::offset_of!($component, $field)
                        == ::core::mem::offset_of!($row, $field),
                    concat!(stringify!($component), ".", stringify!($field), " moved")
                );
            )*
            $contract.assert_producer::<$component>($key);
        };

        impl ::core::convert::From<$row> for $component {
            fn from(row: $row) -> Self {
                ::bytemuck::cast(row)
            }
        }

        impl ::core::convert::From<$component> for $row {
            fn from(component: $component) -> Self {
                ::bytemuck::cast(component)
            }
        }

        impl $component {
            /// This component as its contract row.
            pub fn row(&self) -> $row {
                ::bytemuck::cast(*self)
            }
        }

        // Same fields, same types, same `repr(C)`: same size *and* alignment,
        // so the component reads as its row in place.
        impl ::core::ops::Deref for $component {
            type Target = $row;
            fn deref(&self) -> &$row {
                ::bytemuck::cast_ref(self)
            }
        }

        impl ::core::ops::DerefMut for $component {
            fn deref_mut(&mut self) -> &mut $row {
                ::bytemuck::cast_mut(self)
            }
        }

        impl $crate::RowComponent for $row {
            type Component = $component;
        }
    };
}

/// Mesh payload: fills the [`helio_pass_gbuffer::MESH_VERTICES`] and
/// [`helio_pass_gbuffer::MESH_INDICES`] pools.
///
/// The two variable-length fields share SceneDB's keyed geometry pools. Their
/// generated `*_gpu_handle` accessors expose the allocated ranges needed by
/// `StaticObjectRow`; the renderer never owns or resolves a mesh asset.
#[derive(SceneStore, Clone, Debug)]
pub struct MeshComponent {
    #[gpu(buffer = "builtin_mesh_vertex", mirror = Once)]
    pub vertices: Vec<helio_core::PackedVertex>,
    #[gpu(buffer = "builtin_mesh_index", mirror = Once)]
    pub indices: Vec<u32>,
}

/// One derived portal chain ([`helio_pass_portal_cull::PortalChain`]),
/// stored in the portal-chain pool the portal passes bind.
///
/// The chain is deliberately a variable-length SceneDB field. SceneDB mirrors
/// `portals` through one growable `u32` payload pool and one growable
/// `VarLenHandle` table; the renderer binds both buffers and uses the handle's
/// `count` as the runtime recursion depth. This is the single representation
/// for every depth, rather than a family of fixed `[u32; N]` component types.
#[derive(SceneStore, Clone, Debug, PartialEq)]
pub struct PortalChainComponent {
    #[gpu(buffer = "PortalChainComponent::portals")]
    pub portals: Vec<u32>,
}

/// Spawn a SceneDB material row for every material in a loaded
/// [`helio_asset_compat::ConvertedScene`], returning one [`pulsar_scenedb::Entity`] per
/// `scene.materials[i]`, in the same order.
///
/// This is the direct SceneDB replacement for the old `Scene`-based
/// `upload_scene_materials`: since a material is now just a
/// [`MaterialComponent`] row, spawning one *is* the whole
/// job. Its old job also included uploading each material's textures into
/// the (now-deleted) `Scene`'s bindless texture table and patching the
/// resulting indices into the `GpuMaterial` before inserting it -- bindless
/// texture upload has no SceneDB-authored home yet, so `ConvertedMaterial::
/// textures` is not consulted here. `convert_material`'s own `GpuMaterial`
/// already carries `NO_TEXTURE` in every `tex_*` slot (texture info is
/// returned separately, in `textures`), so every spawned material carries
/// its real base color/roughness/metallic/emissive factors but none of its
/// texture maps.
pub fn upload_scene_materials(
    world: &mut pulsar_scenedb::World,
    scene: &helio_asset_compat::ConvertedScene,
) -> Vec<pulsar_scenedb::Entity> {
    scene
        .materials
        .iter()
        .map(|converted| {
            let entity = world.spawn();
            world.insert(
                entity,
                MaterialComponent::from(helio_pass_gbuffer::MaterialRow::from(converted.gpu)),
            );
            entity
        })
        .collect()
}

impl From<helio_pass_portal_cull::PortalChain> for PortalChainComponent {
    fn from(chain: helio_pass_portal_cull::PortalChain) -> Self {
        Self {
            portals: chain.portals,
        }
    }
}

/// Publish a portal projection's derived rows into `world`.
///
/// The portal GPU ABI is entity-indexed. Callers reserve a contiguous
/// projection entity range whose entity indices match the dense row IDs, and
/// keep those entities separate from authored portal entities. This does not
/// spawn or repurpose entities: their lifetime stays with the host.
pub fn publish_portal_projection(
    world: &mut World,
    frame: &helio_pass_portal_cull::PortalProjectionFrame,
    view_entities: &[Entity],
    chain_entities: &[Entity],
    counts_entity: Entity,
) -> Result<(), helio_pass_portal_cull::ProjectionError> {
    use helio_pass_portal_cull::ProjectionError;
    for (kind, expected, entities) in [
        ("portal view", frame.portal_views.len(), view_entities),
        ("portal chain", frame.portal_chains.len(), chain_entities),
    ] {
        if entities.len() != expected {
            return Err(ProjectionError::SceneRowCount {
                kind,
                expected,
                actual: entities.len(),
            });
        }
        for (dense_index, entity) in entities.iter().enumerate() {
            if entity.index() as usize != dense_index {
                return Err(ProjectionError::NonDenseSceneRow {
                    kind,
                    dense_index,
                    entity_index: entity.index(),
                });
            }
        }
    }
    for (entity, row) in view_entities.iter().zip(&frame.portal_views) {
        world.insert_row(*entity, *row);
    }
    for (entity, chain) in chain_entities.iter().zip(&frame.portal_chains) {
        world.insert(*entity, PortalChainComponent::from(chain.clone()));
    }
    world.insert_row(counts_entity, frame.counts);
    Ok(())
}

/// Project a SceneDB world into the [`helio::SceneGeometry`] a bake reads
/// (Helio#256): every drawn object (`StaticObjectRow` and the
/// `MeshComponent` it references, in world space) and every live light
/// (`LightRow`). This is the explicit CPU projection
/// [`helio::Renderer::set_bake_scene`] takes; the renderer itself never traverses
/// scene entities.
///
/// Call it after the scene is populated and before
/// [`helio::Renderer::auto_bake`]. Objects whose mesh row is gone (a stale
/// generation) are skipped.
#[cfg(feature = "bake")]
pub fn bake_scene_from_world(world: &pulsar_scenedb::World) -> helio::SceneGeometry {
    use helio::{GpuLight, LightSource, LightSourceKind, LightType, MeshUpload};

    let meshes: std::collections::HashMap<u32, (u32, &MeshComponent)> = world
        .query::<&MeshComponent>()
        .map(|(entity, mesh)| (entity.index(), (entity.generation(), mesh)))
        .collect();
    let mut scene = helio::SceneGeometry::new();
    for (_, object) in world.query::<&StaticObjectComponent>() {
        let Some(&(generation, mesh)) = meshes.get(&object.mesh_slot) else {
            continue;
        };
        // Object rows store `generation + 1` so zero means "never written".
        if generation.wrapping_add(1) != object.mesh_generation {
            continue;
        }
        let upload = MeshUpload {
            vertices: mesh.vertices.clone(),
            indices: mesh.indices.clone(),
        };
        let transform = glam::Mat4::from_cols_array_2d(&object.transform);
        scene.add_mesh(helio::mesh_upload_to_bake(
            &upload,
            transform,
            Some(object.mesh_slot),
        ));
    }
    for (_, light) in world.query::<&LightComponent>() {
        let light = GpuLight::from(helio_pass_forward_lit::LightRow::from(*light));
        if light.color_intensity[3] <= 0.0 {
            continue;
        }
        let [x, y, z, range] = light.position_range;
        let direction = [
            light.direction_outer[0],
            light.direction_outer[1],
            light.direction_outer[2],
        ];
        let kind = match light.light_type {
            t if t == LightType::Directional as u32 => LightSourceKind::Directional { direction },
            // GpuLight stores cosines; Nebula takes radians.
            t if t == LightType::Spot as u32 => LightSourceKind::Spot {
                position: [x, y, z],
                direction,
                range,
                inner_angle: light.inner_angle.clamp(-1.0, 1.0).acos(),
                outer_angle: light.direction_outer[3].clamp(-1.0, 1.0).acos(),
            },
            _ => LightSourceKind::Point {
                position: [x, y, z],
                range,
            },
        };
        scene.add_light(LightSource {
            kind,
            color: [
                light.color_intensity[0],
                light.color_intensity[1],
                light.color_intensity[2],
            ],
            intensity: light.color_intensity[3],
            bake_enabled: true,
            casts_shadows: light.shadow_index != u32::MAX,
        });
    }
    scene
}

#[cfg(all(test, feature = "bake"))]
mod bake_scene_tests {
    use super::*;
    use helio::{GpuLight, LightSourceKind, LightType, PackedVertex};
    use helio_pass_forward_lit::LightRow;
    use helio_pass_gbuffer::StaticObjectRow;

    fn vertex(position: [f32; 3]) -> PackedVertex {
        PackedVertex {
            position,
            ..Default::default()
        }
    }

    /// Helio#256: the bake input comes from SceneDB rows, in world space,
    /// and skips what is not really there.
    #[test]
    fn projects_objects_and_live_lights_from_the_world() {
        let mut world = pulsar_scenedb::World::new();
        let mesh = world.spawn();
        world.insert(
            mesh,
            MeshComponent {
                vertices: vec![
                    vertex([0.0, 0.0, 0.0]),
                    vertex([1.0, 0.0, 0.0]),
                    vertex([0.0, 1.0, 0.0]),
                ],
                indices: vec![0, 1, 2],
            },
        );
        let object = |mesh_generation: u32, offset: f32| {
            StaticObjectRow::new(
                mesh.index(),
                mesh_generation,
                0,
                1,
                glam::Mat4::from_translation(glam::Vec3::new(offset, 0.0, 0.0)),
                [offset, 0.0, 0.0, 1.0],
                3,
                0,
                0,
                0,
                0,
                0,
            )
        };
        let live = world.spawn();
        world.insert_row(live, object(mesh.generation().wrapping_add(1), 10.0));
        // A row pointing at a mesh generation that no longer exists.
        let stale = world.spawn();
        world.insert_row(stale, object(mesh.generation().wrapping_add(2), 20.0));

        let mut sun = GpuLight::default();
        sun.light_type = LightType::Directional as u32;
        sun.direction_outer = [0.0, -1.0, 0.0, 0.0];
        sun.color_intensity = [1.0, 0.9, 0.8, 3.0];
        let sun_entity = world.spawn();
        world.insert_row(sun_entity, LightRow::from(sun));
        let vacant = world.spawn();
        world.insert_row(
            vacant,
            LightRow::from(GpuLight {
                color_intensity: [0.0; 4],
                ..GpuLight::default()
            }),
        );

        let scene = bake_scene_from_world(&world);
        assert_eq!(scene.meshes.len(), 1, "stale object rows are skipped");
        assert_eq!(
            scene.meshes[0].positions[1],
            [11.0, 0.0, 0.0],
            "positions are world space"
        );
        assert_eq!(scene.meshes[0].indices, vec![0, 1, 2]);
        assert_eq!(scene.lights.len(), 1, "dark rows are not lights");
        assert!(
            matches!(scene.lights[0].kind, LightSourceKind::Directional { direction } if direction == [0.0, -1.0, 0.0])
        );
        assert_eq!(scene.lights[0].intensity, 3.0);
        assert!(
            !scene.lights[0].casts_shadows,
            "GpuLight::default() requests no shadow"
        );
    }
}
