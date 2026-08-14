//! [`HelioRenderSubsystem`]: drives `helio::Scene`'s existing, unmodified
//! public authoring API from SceneDB [`crate::components`] writes, so a
//! gameplay/editor caller's "upload" is just a plain `World::insert`/
//! `get_mut` call.
//!
//! ## Why this is two calls, not one
//!
//! `pulsar_scenedb::Subsystem: Any + Send + Sync`. `helio::Scene` is not
//! `Send`/`Sync` (it owns `Box<dyn SceneActorTrait>`/`Box<dyn ComponentVec>`
//! trait objects with no such bound declared) -- confirmed by the compiler,
//! not assumed. So `HelioRenderSubsystem` cannot own a `Scene` directly and
//! still implement `Subsystem`. Instead:
//!
//! - [`HelioRenderSubsystem::simulate_b`] (the actual `Subsystem` hook,
//!   invoked by `SceneDb::step()`) only touches `World` and its own
//!   `Send + Sync` bookkeeping -- it turns this frame's `Delta` into a
//!   queue of [`PendingOp`]s, plain owned data, no `Scene` reference
//!   anywhere near it.
//! - [`HelioRenderSubsystem::apply_to`] drains that queue against a real
//!   `&mut Scene` the embedding app passes in explicitly, once per frame,
//!   immediately after `scene_db.step()` and immediately before
//!   `scene.flush()` (see the crate-level integration plan's
//!   frame-loop-placement section) -- this is a plain method call, not a
//!   `Subsystem` hook, so it has no `Send + Sync` obligation to satisfy.
//!
//! ## Per-component-kind dispatch, not one generic object shape
//!
//! Each purpose-built component in [`crate::components`] routes through a
//! DIFFERENT Helio API family, because Helio itself doesn't have one
//! generic "object" shape either:
//!
//! - [`crate::StaticMeshComponent`] -> `insert_object`/`update_object_transform`/
//!   `update_object_bounds`/`remove_object` (plain, single-material objects).
//! - [`crate::MultiMaterialStaticMeshComponent`] -> `insert_sectioned_object`/
//!   `update_sectioned_object_transform`/`remove_sectioned_object` (Helio's
//!   multi-section mesh path -- a materially different API, no bounds
//!   update or flags/groups support at all: `insert_sectioned_object`
//!   hardcodes `flags = 0b11, groups = NONE` internally).
//! - [`crate::LightComponent`] -> `insert_light_with_movability`/`update_light`/
//!   `remove_light`, tagged the same way objects are (`Scene::light_by_tag`
//!   is the reverse lookup -- no separate map needed on the SceneDB side).
//! - [`crate::MaterialSlot`]/[`crate::StaticMeshComponent`]'s own inline
//!   `material` field -> `insert_material`/`update_material`/`remove_material`
//!   (see the scope note below for why this is still a real Helio-pool
//!   `MaterialId`, not literal buffer aliasing, today).
//!
//! ## Scope note: materials are a translation layer, not (yet) buffer aliasing
//!
//! Every `#[gpu(buffer = "builtin_material")]`-tagged field (`MaterialSlot::data`,
//! `StaticMeshComponent::material`) is a real, independently provable
//! zero-copy shared buffer with whatever else registers under that key
//! (see `tests/material_buffer_identity.rs`). But `helio::Scene::insert_object`
//! validates its `material: MaterialId` argument against Helio's OWN
//! `SparsePool<MaterialRecord, MaterialId>` -- an independent
//! slot/generation-checked pool, decoupled from whatever buffer backs the
//! GPU-visible bytes. Wiring `insert_object` to accept a SceneDB-entity-
//! index-derived handle directly would mean either bypassing that
//! validation or replacing the pool's own slot assignment -- real surgery
//! on `helio::Scene`'s source, out of scope for this pass. So today,
//! [`HelioRenderSubsystem::apply_to`] ALSO mints a real, pool-registered
//! `MaterialId` per material-bearing entity (via `insert_material`/
//! `update_material`) purely so `insert_object`/`insert_sectioned_object`
//! have something valid to reference -- the GPU bytes live in two places
//! at once (Helio's own material buffer AND the SceneDB-owned
//! `builtin_material` buffer) until a follow-up repoints `insert_object`'s
//! validation/storage at the SceneDB buffer directly.

use std::any::Any;
use std::collections::{HashMap, HashSet};

use glam::Mat4;
use helio::{GroupMask, MaterialId, MeshId, MultiMeshId, Movability, ObjectDescriptor, Scene, SectionedInstanceId};
use libhelio::material::GpuMaterial;
use libhelio::GpuLight;
use pulsar_scenedb::gpu::{HarvestPhase, RetiredPhase, SceneGpuStore, SimulateA, SimulateB};
use pulsar_scenedb::{component_id, ComponentId, Entity, Subsystem, World};

use crate::components::{
    LightComponent, MaterialSlot, MultiMaterialStaticMeshComponent, RenderBounds, RenderFlags, RenderTransform, StaticMeshComponent,
};

struct StaticMeshSnapshot {
    mesh: MeshId,
    transform: Mat4,
    bounds: [f32; 4],
    flags: u32,
    movability: Option<Movability>,
    groups: GroupMask,
    /// Which of {`StaticMeshComponent`, `RenderTransform`, `RenderBounds`,
    /// `RenderFlags`} changed this frame for this entity. Component-level
    /// granularity, not per-field -- so ANY `StaticMeshComponent` delta is
    /// conservatively treated as "the mesh might have changed" (there is
    /// no cheap `update_object_mesh`), even though a material-only edit
    /// (already handled by the material upsert step, see the module doc)
    /// didn't strictly need a rebuild. Correct either way; a known,
    /// documented place a future finer-grained (per-field) change signal
    /// could trim redundant rebuilds.
    changed: HashSet<ComponentId>,
}

struct MultiMeshSnapshot {
    mesh: MultiMeshId,
    materials: smallvec::SmallVec<[Entity; 4]>,
    transform: Mat4,
    bounds: [f32; 4],
    changed: HashSet<ComponentId>,
}

enum PendingOp {
    UpsertMaterial(Entity, GpuMaterial),
    RemoveMaterial(Entity),
    UpsertLight(Entity, GpuLight, Option<Movability>),
    RemoveLight(Entity),
    UpsertStaticMesh(Entity, StaticMeshSnapshot),
    RemoveObject(Entity),
    UpsertMultiMesh(Entity, MultiMeshSnapshot),
    RemoveSectionedObject(Entity),
}

/// Registers as a [`Subsystem`] on a [`pulsar_scenedb::SceneDb`] whose
/// `World` has an attached [`pulsar_scenedb::SharedChangeTracker`]
/// (`World::attach_change_tracker`) -- with that in place, gameplay/editor
/// code authors renderable state with plain `World` calls:
///
/// ```ignore
/// let object = world.spawn_bundle((
///     StaticMeshComponent::new(mesh_id, gpu_material),
///     RenderTransform(transform),
///     RenderBounds(bounds),
///     RenderFlags::default(),
/// ));
/// // ... later, every frame something moves:
/// world.get_mut::<RenderTransform>(object).unwrap().0 = new_transform;
/// ```
///
/// Per frame:
/// ```ignore
/// scene_db.step();                         // runs HelioRenderSubsystem::simulate_b
/// helio_render.apply_to(&mut helio_scene);  // applies this frame's queued ops
/// helio_scene.flush();
/// renderer.render(&helio_scene, target)?;
/// ```
#[derive(Default)]
pub struct HelioRenderSubsystem {
    /// Keyed by whichever entity supplied the `GpuMaterial` -- either a
    /// bare [`MaterialSlot`] entity, or a [`StaticMeshComponent`]-bearing
    /// entity's own id (inline material, no separate material entity).
    /// Lights don't need an equivalent map -- they're tagged the same way
    /// objects are and resolved via `Scene::light_by_tag`.
    material_ids: HashMap<Entity, MaterialId>,
    sectioned_ids: HashMap<Entity, SectionedInstanceId>,
    pending: Vec<PendingOp>,
}

impl HelioRenderSubsystem {
    pub fn new() -> Self {
        Self::default()
    }

    /// Packs an [`Entity`] into Helio's `user_tag`. `+1`, not identity:
    /// `Entity::new(0, 0).bits() == 0` for the very first entity ever
    /// spawned in a fresh `World`, and `ObjectDescriptor::user_tag`
    /// documents `0` as its "untagged" sentinel -- an unoffset tag would
    /// make that one specific entity's object permanently unreachable via
    /// `Scene::object_by_tag`. Only plain (`StaticMeshComponent`) objects
    /// are tagged this way -- `insert_sectioned_object` has no `user_tag`
    /// parameter at all (hardcodes none internally), so sectioned objects
    /// are tracked via `Self::sectioned_ids` instead.
    #[inline]
    pub fn tag_for(entity: Entity) -> u64 {
        entity.bits() + 1
    }

    /// Applies every op queued by [`Self::simulate_b`] since the last call,
    /// against a real `Scene`. Call once per frame -- see [`Self`]'s doc
    /// for where this sits relative to `scene_db.step()`/`scene.flush()`.
    pub fn apply_to(&mut self, scene: &mut Scene) {
        for op in self.pending.drain(..) {
            match op {
                PendingOp::UpsertMaterial(entity, gpu) => {
                    if let Some(&id) = self.material_ids.get(&entity) {
                        let _ = scene.update_material(id, gpu);
                    } else {
                        let id = scene.insert_material(gpu);
                        self.material_ids.insert(entity, id);
                    }
                }
                PendingOp::RemoveMaterial(entity) => {
                    if let Some(id) = self.material_ids.remove(&entity) {
                        let _ = scene.remove_material(id);
                    }
                }
                PendingOp::UpsertLight(entity, gpu, movability) => {
                    let tag = Self::tag_for(entity);
                    match scene.light_by_tag(tag) {
                        Some(id) => {
                            let _ = scene.update_light(id, gpu);
                        }
                        None => {
                            scene.insert_light_with_movability(gpu, movability, tag);
                        }
                    }
                }
                PendingOp::RemoveLight(entity) => {
                    if let Some(id) = scene.light_by_tag(Self::tag_for(entity)) {
                        let _ = scene.remove_light(id);
                    }
                }
                PendingOp::RemoveObject(entity) => {
                    if let Some(id) = scene.object_by_tag(Self::tag_for(entity)) {
                        let _ = scene.remove_object(id);
                    }
                }
                PendingOp::UpsertStaticMesh(entity, snap) => {
                    let Some(&material) = self.material_ids.get(&entity) else {
                        // This entity's own material upsert hasn't run yet
                        // within this apply_to call -- can't happen since
                        // materials are queued before objects in
                        // simulate_b, see that method's doc.
                        continue;
                    };
                    let desc = ObjectDescriptor {
                        mesh: snap.mesh,
                        material,
                        transform: snap.transform,
                        bounds: snap.bounds,
                        flags: snap.flags,
                        groups: snap.groups,
                        movability: snap.movability,
                        user_tag: Self::tag_for(entity),
                    };
                    let rebuild = snap.changed.contains(&component_id::<StaticMeshComponent>())
                        || snap.changed.contains(&component_id::<RenderFlags>());
                    match scene.object_by_tag(Self::tag_for(entity)) {
                        None => {
                            let _ = scene.insert_object(desc);
                        }
                        Some(id) if rebuild => {
                            let _ = scene.remove_object(id);
                            let _ = scene.insert_object(desc);
                        }
                        Some(id) => {
                            if snap.changed.contains(&component_id::<RenderTransform>()) {
                                let _ = scene.update_object_transform(id, desc.transform);
                            }
                            if snap.changed.contains(&component_id::<RenderBounds>()) {
                                let _ = scene.update_object_bounds(id, desc.bounds);
                            }
                        }
                    }
                }
                PendingOp::RemoveSectionedObject(entity) => {
                    if let Some(id) = self.sectioned_ids.remove(&entity) {
                        let _ = scene.remove_sectioned_object(id);
                    }
                }
                PendingOp::UpsertMultiMesh(entity, snap) => {
                    let Some(materials): Option<Vec<MaterialId>> =
                        snap.materials.iter().map(|e| self.material_ids.get(e).copied()).collect()
                    else {
                        // At least one referenced MaterialSlot hasn't been
                        // upserted yet this frame -- skip; the next change
                        // to either side re-queues this.
                        continue;
                    };

                    let rebuild = snap.changed.contains(&component_id::<MultiMaterialStaticMeshComponent>());
                    match self.sectioned_ids.get(&entity).copied() {
                        None => {
                            if let Ok(id) = scene.insert_sectioned_object(snap.mesh, &materials, snap.transform, snap.bounds, None) {
                                self.sectioned_ids.insert(entity, id);
                            }
                        }
                        Some(id) if rebuild => {
                            let _ = scene.remove_sectioned_object(id);
                            self.sectioned_ids.remove(&entity);
                            if let Ok(new_id) = scene.insert_sectioned_object(snap.mesh, &materials, snap.transform, snap.bounds, None) {
                                self.sectioned_ids.insert(entity, new_id);
                            }
                        }
                        Some(id) => {
                            // No per-field granularity (see StaticMeshSnapshot's
                            // doc) -- transform is the only cheap update
                            // Helio exposes for sectioned instances at all,
                            // so just always re-push it when this entity
                            // was touched and isn't a full rebuild.
                            let _ = scene.update_sectioned_object_transform(id, snap.transform);
                        }
                    }
                }
            }
        }
    }
}

impl Subsystem for HelioRenderSubsystem {
    fn name(&self) -> &'static str {
        "helio_render"
    }

    fn simulate_a(&mut self, _world: &mut World, _witness: &SimulateA) {}

    /// Drains this frame's `Delta` from `world`'s attached change tracker
    /// (a no-op if none is attached -- see [`Self`]'s doc) and queues
    /// [`PendingOp`]s for [`Self::apply_to`]: materials and lights first
    /// (so an object referencing a material added in the same frame
    /// resolves correctly once `apply_to` runs), then removals for
    /// despawned entities, then static-mesh and multi-material object
    /// upserts/updates.
    fn simulate_b(&mut self, world: &mut World, _witness: &SimulateB) {
        let Some(tracker) = world.change_tracker().cloned() else {
            return;
        };
        let delta = tracker.drain_with_world(world);

        let material_slot_cid = component_id::<MaterialSlot>();
        let static_mesh_cid = component_id::<StaticMeshComponent>();
        let multi_mesh_cid = component_id::<MultiMaterialStaticMeshComponent>();
        let light_cid = component_id::<LightComponent>();
        let shared_cids: [ComponentId; 3] =
            [component_id::<RenderTransform>(), component_id::<RenderBounds>(), component_id::<RenderFlags>()];

        // -- Materials: both bare MaterialSlot entities and StaticMeshComponent's
        // own inline material (keyed by the mesh entity itself).
        for cd in &delta.component_deltas {
            if cd.component_type == material_slot_cid {
                match world.get::<MaterialSlot>(cd.entity) {
                    Some(slot) => self.pending.push(PendingOp::UpsertMaterial(cd.entity, slot.data.0)),
                    None => self.pending.push(PendingOp::RemoveMaterial(cd.entity)),
                }
            }
            if cd.component_type == static_mesh_cid {
                match world.get::<StaticMeshComponent>(cd.entity) {
                    Some(sm) => self.pending.push(PendingOp::UpsertMaterial(cd.entity, sm.material.0)),
                    None => self.pending.push(PendingOp::RemoveMaterial(cd.entity)),
                }
            }
            if cd.component_type == light_cid {
                match world.get::<LightComponent>(cd.entity) {
                    Some(light) => self.pending.push(PendingOp::UpsertLight(cd.entity, light.light, light.movability)),
                    None => self.pending.push(PendingOp::RemoveLight(cd.entity)),
                }
            }
        }

        // -- Despawned entities: drop whatever they owned, across every kind.
        for &entity in &delta.despawned {
            self.pending.push(PendingOp::RemoveMaterial(entity));
            self.pending.push(PendingOp::RemoveLight(entity));
            self.pending.push(PendingOp::RemoveObject(entity));
            self.pending.push(PendingOp::RemoveSectionedObject(entity));
        }

        // -- Object entities touched this frame (StaticMeshComponent or
        // MultiMaterialStaticMeshComponent, plus the shared RenderTransform/
        // RenderBounds/RenderFlags), with which component types changed.
        let mut touched: HashMap<Entity, HashSet<ComponentId>> = HashMap::new();
        for cd in &delta.component_deltas {
            if cd.component_type == static_mesh_cid || cd.component_type == multi_mesh_cid || shared_cids.contains(&cd.component_type) {
                touched.entry(cd.entity).or_default().insert(cd.component_type);
            }
        }
        for &(entity, _) in &delta.spawned {
            touched.entry(entity).or_default();
        }

        for (entity, changed) in touched {
            if !world.is_alive(entity) {
                continue;
            }
            if let Some(sm) = world.get::<StaticMeshComponent>(entity) {
                let RenderFlags { flags, movability, groups } = world.get::<RenderFlags>(entity).copied().unwrap_or_default();
                let transform = world.get::<RenderTransform>(entity).map(|c| c.0).unwrap_or(Mat4::IDENTITY);
                let bounds = world.get::<RenderBounds>(entity).map(|c| c.0).unwrap_or([0.0, 0.0, 0.0, 1.0]);
                self.pending.push(PendingOp::UpsertStaticMesh(
                    entity,
                    StaticMeshSnapshot { mesh: sm.mesh.0, transform, bounds, flags, movability, groups, changed },
                ));
            } else if let Some(mm) = world.get::<MultiMaterialStaticMeshComponent>(entity) {
                let transform = world.get::<RenderTransform>(entity).map(|c| c.0).unwrap_or(Mat4::IDENTITY);
                let bounds = world.get::<RenderBounds>(entity).map(|c| c.0).unwrap_or([0.0, 0.0, 0.0, 1.0]);
                self.pending.push(PendingOp::UpsertMultiMesh(
                    entity,
                    MultiMeshSnapshot { mesh: mm.mesh, materials: mm.materials.clone(), transform, bounds, changed },
                ));
            } else {
                // Neither kind present anymore -- a removal (best-effort
                // against both possible Helio-side representations).
                self.pending.push(PendingOp::RemoveObject(entity));
                self.pending.push(PendingOp::RemoveSectionedObject(entity));
            }
        }
    }

    fn harvest(&mut self, _store: &SceneGpuStore, _phase: &HarvestPhase) {}
    fn boundary(&mut self, _phase: &RetiredPhase) {}

    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
