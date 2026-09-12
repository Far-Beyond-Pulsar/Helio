//! Reflection capture component (Phase D, Pulsar-Native#558) — the first of
//! the "no purpose-built component exists yet" primitives from Helio's own
//! inventory. Places a parallax-corrected reflection probe influence volume
//! in the scene.
//!
//! Helio already has full native support for this (`Scene::
//! insert_reflection_capture`/`update_reflection_capture`/
//! `remove_reflection_capture`, `ReflectionCaptureDescriptor`) — the gap
//! this closes is purely the author-facing `#[engine_class]` wrapper, same
//! as every component migrated in Phase B4/B5.
//!
//! `libhelio::ReflectionCaptureShape`/`ReflectionCaptureMobility` aren't
//! re-exported from `helio`'s own crate root and aren't reflection-friendly
//! (no `Serialize`/`Deserialize`) even if they were, so this module defines
//! its own mirrored enums rather than reusing them directly or editing the
//! Helio submodule (already in an unrelated dirty state locally; not worth
//! a cross-repo change for a two-variant enum).
//!
//! Unlike `StaticMeshComponent`/`LightComponent`/`PortalComponent`, this
//! goes through `Scene::insert_reflection_capture`/etc. directly rather than
//! `SceneEntity::reflection_capture(..)` + `Scene::insert_entity`: those
//! components need the `SceneEntity`/tag machinery specifically for
//! click-to-select picking, which reflection captures don't have wired up
//! yet (deferred, along with gizmo interaction, for a follow-up — this pass
//! is the data-sync mechanism, not full editor interaction). The direct
//! `Scene` methods hand back the typed `ReflectionCaptureId` this
//! component's own cache needs anyway.

use engine_class_derive::{engine_class, register_runtime_behavior, register_world_component};
use glam::{EulerRot, Mat4, Quat, Vec3};
use helio::{ReflectionCaptureDescriptor, Renderer};
use libhelio::{
    ReflectionCaptureMobility as HelioReflectionCaptureMobility,
    ReflectionCaptureShape as HelioReflectionCaptureShape,
};
use pulsar_reflection::{
    get_subsystem, ComponentRuntimeBehavior, ComponentRuntimeContext, Reflectable,
    RuntimeComponentOwner,
};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use pulsar_scenedb::gpu::{BufferHandle, BufferKey, GpuMirrorHandle};
use pulsar_scenedb_derive::SceneStore;

use crate::subsystems::ReflectionCaptureCache;

pub const REFLECTION_CAPTURE_CLASS_NAME: &str = "ReflectionCaptureComponent";

/// Packed SceneDB projection for a reflection probe.
///
/// `ReflectionCaptureComponent` remains the author-facing value.  This
/// companion is the GPU-facing row and deliberately contains no Helio arena
/// handle or renderer-owned lifetime.  Probe cubemap residency is frame
/// derived; `cubemap_index == -1` means that no resident layer is available.
#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "reflection_captures")]
pub struct ReflectionCaptureGpuComponent {
    #[gpu] pub position_radius: [f32; 4],
    #[gpu] pub extents_transition: [f32; 4],
    #[gpu] pub world_to_local: [[f32; 4]; 4],
    #[gpu] pub cubemap_index: i32,
    #[gpu] pub shape: u32,
    #[gpu] pub mobility: u32,
    #[gpu] pub brightness: f32,
}

impl From<libhelio::GpuReflectionCapture> for ReflectionCaptureGpuComponent {
    fn from(value: libhelio::GpuReflectionCapture) -> Self { bytemuck::cast(value) }
}

impl From<ReflectionCaptureGpuComponent> for libhelio::GpuReflectionCapture {
    fn from(value: ReflectionCaptureGpuComponent) -> Self { bytemuck::cast(value) }
}

#[derive(Clone)]
pub struct ReflectionCaptureSceneBinding {
    handle: BufferHandle,
    _record: PhantomData<ReflectionCaptureGpuComponent>,
}

impl ReflectionCaptureSceneBinding {
    pub fn resolve(mirror: &GpuMirrorHandle) -> Option<Self> {
        mirror.store().resolve_buffer_handle(BufferKey::of("reflection_captures"))
            .map(|handle| Self { handle, _record: PhantomData })
    }
    pub fn buffer(&self) -> &wgpu::Buffer { &self.handle.buffer }
    pub fn epoch(&self) -> u64 { self.handle.epoch }
}

/// Influence-volume shape. Mirrors `libhelio::ReflectionCaptureShape`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Reflectable)]
pub enum ReflectionCaptureShape {
    /// Radial influence, faded over the outer 10% of `influence_radius`.
    Sphere,
    /// Oriented box influence (takes its rotation from the owning object),
    /// faded over `transition_distance` from each face.
    Box,
}

impl Default for ReflectionCaptureShape {
    fn default() -> Self {
        Self::Sphere
    }
}

/// How the capture's cubemap pixels are produced. Mirrors
/// `libhelio::ReflectionCaptureMobility`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Reflectable)]
pub enum ReflectionCaptureMobility {
    /// Pre-filtered offline by the probe baker. The only mode that
    /// contributes anything today.
    Static,
    /// Re-rendered at runtime rather than baked. **Not implemented in Helio
    /// yet** — a `Dynamic` capture is inert (never assigned a cubemap
    /// layer), see `libhelio::ReflectionCaptureMobility::Dynamic`'s own doc.
    /// Exposed now for forward compatibility, not because it does anything.
    Dynamic,
}

impl Default for ReflectionCaptureMobility {
    fn default() -> Self {
        Self::Static
    }
}

/// Places a reflection probe influence volume in the scene.
#[engine_class(category = "Rendering", clone, debug, serialize, deserialize)]
pub struct ReflectionCaptureComponent {
    #[property]
    pub enabled: bool,
    #[property]
    pub shape: ReflectionCaptureShape,
    #[property]
    pub mobility: ReflectionCaptureMobility,
    /// Sphere influence radius, in world units. Ignored for `Box` shape.
    #[property(min = 0.1, max = 10000.0, step = 0.5)]
    pub influence_radius: f32,
    /// Box half-extents, in capture-local space. Ignored for `Sphere` shape.
    #[property]
    pub extents: [f32; 3],
    /// Distance over which a box capture fades out at its faces. Sphere
    /// captures always fade over the outer 10% of `influence_radius`
    /// instead, regardless of this value.
    #[property(min = 0.0, max = 100.0, step = 0.1)]
    pub transition_distance: f32,
    /// Linear multiplier on the sampled radiance.
    #[property(min = 0.0, max = 10.0, step = 0.05)]
    pub brightness: f32,
}

impl Default for ReflectionCaptureComponent {
    fn default() -> Self {
        Self {
            enabled: true,
            shape: ReflectionCaptureShape::Sphere,
            mobility: ReflectionCaptureMobility::Static,
            influence_radius: 10.0,
            extents: [5.0; 3],
            transition_distance: 1.0,
            brightness: 1.0,
        }
    }
}

#[register_world_component]
#[register_runtime_behavior]
impl ComponentRuntimeBehavior for ReflectionCaptureComponent {
    const CLASS_NAME: &'static str = REFLECTION_CAPTURE_CLASS_NAME;

    fn sync_component(
        owner: &RuntimeComponentOwner,
        _component_index: usize,
        component: &Self,
        context: &mut dyn ComponentRuntimeContext,
    ) {
        // Phase 1: consult the cache and release its borrow immediately --
        // a `ComponentRuntimeContext` can't hand out two simultaneous
        // mutable subsystem borrows (same constraint `PortalComponent`
        // documents for the same reason).
        let cached_id = get_subsystem!(context, ReflectionCaptureCache).get(owner.scene_object_id);

        if !component.enabled {
            if let Some(id) = cached_id {
                let removed = get_subsystem!(context, Renderer)
                    .remove_reflection_capture(id);
                if removed {
                    get_subsystem!(context, ReflectionCaptureCache).remove(owner.scene_object_id);
                }
            }
            return;
        }

        // Box captures take their rotation from `transform`; sphere captures
        // use only the translation (`ReflectionCaptureDescriptor`'s own
        // doc). Scale is deliberately not applied here -- `extents`/
        // `influence_radius` are this component's own authored size fields,
        // independent of the owning object's scale, matching how
        // `PortalComponent`'s width/height aren't scaled by owner.scale
        // either.
        let rotation = Quat::from_euler(
            EulerRot::YXZ,
            owner.rotation[1].to_radians(),
            owner.rotation[0].to_radians(),
            owner.rotation[2].to_radians(),
        );
        let transform = Mat4::from_rotation_translation(rotation, Vec3::from_array(owner.position));

        let descriptor = ReflectionCaptureDescriptor {
            shape: match component.shape {
                ReflectionCaptureShape::Sphere => HelioReflectionCaptureShape::Sphere,
                ReflectionCaptureShape::Box => HelioReflectionCaptureShape::Box,
            },
            mobility: match component.mobility {
                ReflectionCaptureMobility::Static => HelioReflectionCaptureMobility::Static,
                ReflectionCaptureMobility::Dynamic => HelioReflectionCaptureMobility::Dynamic,
            },
            transform,
            influence_radius: component.influence_radius,
            extents: component.extents,
            transition_distance: component.transition_distance,
            brightness: component.brightness,
        };

        match cached_id {
            Some(id) => {
                let _ = get_subsystem!(context, Renderer)
                    .update_reflection_capture(id, &descriptor);
            }
            None => {
                let inserted = get_subsystem!(context, Renderer)
                    .insert_reflection_capture(descriptor);
                if let Ok(id) = inserted {
                    get_subsystem!(context, ReflectionCaptureCache)
                        .insert(owner.scene_object_id.to_string(), id);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_subsystems::{Subsystem, SubsystemContext};
    use pulsar_reflection::{apply_runtime_behavior_for_class, Subsystems};
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};

    struct TestRuntimeContext {
        project_root: PathBuf,
        subsystems: Subsystems,
        errors: Vec<String>,
    }

    impl ComponentRuntimeContext for TestRuntimeContext {
        fn subsystems_mut(&mut self) -> &mut Subsystems {
            &mut self.subsystems
        }
        fn project_root(&self) -> &Path {
            &self.project_root
        }
        fn report_error(&mut self, message: String) {
            self.errors.push(message);
        }
    }

    fn owner<'a>(props: &'a HashMap<String, serde_json::Value>) -> RuntimeComponentOwner<'a> {
        RuntimeComponentOwner {
            scene_object_id: "probe",
            position: [1.0, 2.0, 3.0],
            rotation: [0.0; 3],
            scale: [1.0; 3],
            props,
        }
    }

    #[test]
    fn runtime_behavior_has_the_reflected_component_name() {
        assert_eq!(
            <ReflectionCaptureComponent as ComponentRuntimeBehavior>::CLASS_NAME,
            REFLECTION_CAPTURE_CLASS_NAME
        );
    }

    #[test]
    fn disabling_a_never_inserted_capture_is_a_quiet_no_op() {
        let mut subsystems = Subsystems::new();
        subsystems.register(ReflectionCaptureCache::new());
        let mut context = TestRuntimeContext {
            project_root: PathBuf::from("."),
            subsystems,
            errors: Vec::new(),
        };
        let props = HashMap::new();
        let disabled = ReflectionCaptureComponent { enabled: false, ..Default::default() };

        assert!(apply_runtime_behavior_for_class(
            REFLECTION_CAPTURE_CLASS_NAME,
            &owner(&props),
            0,
            &serde_json::to_value(disabled).unwrap(),
            &mut context,
        ));
        assert!(context.errors.is_empty());
    }
}
