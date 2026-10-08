//! Portal authoring component.
//!
//! **Not rendered by this engine.** Helio's portal passes draw linked
//! portal pairs: each portal names its peer, and a projection bridge turns
//! the pairs into view and chain rows at reserved, dense entity slots. This
//! component authors no peer, and the editor world cannot provide those
//! slots, so it writes no scene rows and is reported unsupported
//! (Pulsar-Native#1035, Phase 4).

use engine_class_derive::{engine_class, register_runtime_behavior, register_world_component};
use pulsar_reflection::{ComponentRuntimeBehavior, ComponentRuntimeContext, RuntimeComponentOwner};

pub const PORTAL_CLASS_NAME: &str = "PortalComponent";

#[engine_class(category = "Rendering", clone, debug, serialize, deserialize)]
#[category("General", category_color = "#6EC5FF")]
pub struct PortalComponent {
    #[property]
    pub enabled: bool,
    #[property]
    pub portal_id: i32,
    #[property(min = 0.1, max = 1000.0, step = 0.1, category = "General")]
    pub width: f32,
    #[property(min = 0.1, max = 1000.0, step = 0.1, category = "General")]
    pub height: f32,
}

impl Default for PortalComponent {
    fn default() -> Self {
        Self {
            enabled: true,
            portal_id: 0,
            width: 2.0,
            height: 3.0,
        }
    }
}

#[register_world_component]
#[register_runtime_behavior]
impl ComponentRuntimeBehavior for PortalComponent {
    const CLASS_NAME: &'static str = PORTAL_CLASS_NAME;

    fn sync_component(
        _owner: &RuntimeComponentOwner,
        _component_index: usize,
        _component: &Self,
        _context: &mut dyn ComponentRuntimeContext,
    ) {
        // Not rendered (see the module doc): nothing to sync.
    }
}
