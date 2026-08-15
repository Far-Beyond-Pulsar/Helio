//! Material override component for customizing object appearance

use engine_class_derive::{engine_class, register_runtime_behavior, register_world_component};
use pulsar_reflection::{ComponentRuntimeBehavior, ComponentRuntimeContext, RuntimeComponentOwner};

/// Material override component for per-object material customization
///
/// Allows overriding material properties on a per-instance basis without
/// creating new material assets.
#[engine_class(category = "Rendering", default, clone, debug, serialize, deserialize)]
pub struct MaterialOverrideComponent {
    /// Base color tint
    #[property]
    pub base_color: [f32; 4],

    /// Metallic factor (0 = dielectric, 1 = metal)
    #[property(min = 0.0, max = 1.0, step = 0.01)]
    pub metallic: f32,

    /// Roughness factor (0 = smooth/glossy, 1 = rough/matte)
    #[property(min = 0.0, max = 1.0, step = 0.01)]
    pub roughness: f32,

    /// Emissive color (RGB, HDR values allowed)
    #[property]
    pub emissive_color: [f32; 3],

    /// Emissive intensity multiplier
    #[property(min = 0.0, max = 100.0, step = 0.1)]
    pub emissive_intensity: f32,

    /// Alpha/opacity (0 = fully transparent, 1 = fully opaque)
    #[property(min = 0.0, max = 1.0, step = 0.01)]
    pub alpha: f32,

    /// UV scale for tiling textures
    #[property(min = 0.01, max = 100.0, step = 0.1)]
    pub uv_scale_x: f32,

    #[property(min = 0.01, max = 100.0, step = 0.1)]
    pub uv_scale_y: f32,

    /// UV offset for scrolling textures
    #[property(min = -10.0, max = 10.0, step = 0.01)]
    pub uv_offset_x: f32,

    #[property(min = -10.0, max = 10.0, step = 0.01)]
    pub uv_offset_y: f32,
}

// Pulsar-Native#561 (editor-side "everything is a real World component, no
// ComponentDb-only flat JSON left"). `sync_component` is a no-op for now --
// actually applying the override to the owning object's Helio material is
// real, separate, not-yet-built plumbing (needs to look up the target
// StaticMeshComponent's bound material and push these fields onto it via
// Helio's material API; tracked separately, see the spawned
// "Fix MaterialSection's mismatched class name" investigation, which found
// the properties-panel side of this component was *also* completely
// disconnected -- wrong class-name string, so it never even reached
// storage). This change only makes the component itself a real, live,
// individually-editable `World` value like every other component -- the
// same free-win shape as `RigidbodyComponent`/`LODComponent`.
#[register_world_component]
#[register_runtime_behavior]
impl ComponentRuntimeBehavior for MaterialOverrideComponent {
    const CLASS_NAME: &'static str = "MaterialOverrideComponent";

    fn sync_component(
        _owner: &RuntimeComponentOwner,
        _component_index: usize,
        _component: &Self,
        _context: &mut dyn ComponentRuntimeContext,
    ) {
        // TODO(Pulsar-Native#561): apply these fields to the owning object's
        // bound Helio material once that plumbing exists. Not yet built --
        // this is deliberately a no-op, not a broken implementation.
    }
}
