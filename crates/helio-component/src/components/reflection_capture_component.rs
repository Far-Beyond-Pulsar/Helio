//! Reflection capture component (Phase D, Pulsar-Native#558): places a
//! parallax-corrected reflection probe influence volume in the scene.
//!
//! **Not rendered by this engine.** Deferred lighting samples a capture only
//! once a probe bake has assigned it a cubemap layer, and the engine runs no
//! probe baker, so a capture would never contribute. The component keeps
//! its authored settings, writes no scene rows and is reported unsupported
//! (Pulsar-Native#1035, Phase 4). `helio_pass_deferred_light::
//! ReflectionCaptureComponent` remains the only schema of the
//! `"reflection_captures"` buffer.

use engine_class_derive::{engine_class, register_runtime_behavior, register_world_component};
use pulsar_reflection::{
    ComponentRuntimeBehavior, ComponentRuntimeContext, Reflectable, RuntimeComponentOwner,
};
use serde::{Deserialize, Serialize};

pub const REFLECTION_CAPTURE_CLASS_NAME: &str = "ReflectionCaptureComponent";

/// Influence-volume shape. Mirrors `helio_pass_deferred_light::ReflectionCaptureShape`.
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
/// `helio_pass_deferred_light::ReflectionCaptureMobility`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Reflectable)]
pub enum ReflectionCaptureMobility {
    /// Pre-filtered offline by the probe baker. The only mode that
    /// contributes anything today.
    Static,
    /// Re-rendered at runtime rather than baked. **Not implemented in Helio
    /// yet** — a `Dynamic` capture is inert (never assigned a cubemap
    /// layer), see `helio_pass_deferred_light::ReflectionCaptureMobility::Dynamic`'s own doc.
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
        _owner: &RuntimeComponentOwner,
        _component_index: usize,
        _component: &Self,
        _context: &mut dyn ComponentRuntimeContext,
    ) {
        // Not rendered (see the module doc): nothing to sync.
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
        let mut context = TestRuntimeContext {
            project_root: PathBuf::from("."),
            subsystems,
            errors: Vec::new(),
        };
        let props = HashMap::new();
        let disabled = ReflectionCaptureComponent {
            enabled: false,
            ..Default::default()
        };

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
