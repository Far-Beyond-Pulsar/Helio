//! SceneDB World registration for voxel authoring components.
//!
//! These behaviors intentionally do not invoke rendering or generation. A
//! future voxel backend can observe the typed SceneDB rows and consume
//! revisioned external updates without coupling component hydration to a pass.

use engine_class_derive::{register_component_runtime, register_world_component};

use super::{TerrainEventsEventWriterExt as _, VoxelComponent, VoxelFlatTerrainComponent, VoxelLandformComponent, VoxelTerrainComponent};

// Component callbacks borrow the authoritative terrain row from SceneDB.
// The pending event list is transient transport state populated only after a
// successful block edit; the EventHub writer queues delivery after the
// component borrow is released.
#[register_component_runtime(class = "VoxelTerrainComponent", enabled = enabled)]
impl VoxelTerrainComponent {
    /// Native Rust counterpart to Blueprint listeners. The generated
    /// adapter receives this event through the host-owned inbox on the next
    /// component phase, after Gamma delivery and outside the emitter borrow.
    #[bp_handler("block_broken")]
    fn on_block_broken(
        &mut self,
        _context: &mut pulsar_world_registry::ComponentContext<'_>,
        _block: super::BlockData,
    ) {
        // Terrain-specific Rust systems can add per-instance reactions here.
    }

    #[bp_handler("block_placed")]
    fn on_block_placed(
        &mut self,
        _context: &mut pulsar_world_registry::ComponentContext<'_>,
        _block: super::BlockData,
    ) {
        // Terrain-specific Rust systems can add per-instance reactions here.
    }

    #[bp_handler("block_material_changed")]
    fn on_block_material_changed(
        &mut self,
        _context: &mut pulsar_world_registry::ComponentContext<'_>,
        _change: super::BlockMaterialChange,
    ) {
        // Terrain-specific Rust systems can add per-instance reactions here.
    }

    fn tick(
        &mut self,
        context: &mut pulsar_world_registry::ComponentContext<'_>,
        _delta_seconds: f32,
    ) {
        macro_rules! flush_events {
            ($field:ident, $writer:ident, $event_name:literal) => {{
                let mut events = std::mem::take(&mut self.$field).into_iter();
                while let Some(event) = events.next() {
                    if let Err(error) = context.events.$writer(event.clone()) {
                        tracing::warn!(entity = ?context.entity, %error, event = $event_name, "could not queue terrain event; retaining it");
                        self.$field.push(event);
                        self.$field.extend(events);
                        break;
                    }
                }
            }};
        }
        flush_events!(pending_block_broken, block_broken, "block_broken");
        flush_events!(pending_block_placed, block_placed, "block_placed");
        flush_events!(pending_block_material_changed, block_material_changed, "block_material_changed");
    }
}

#[register_world_component]
impl VoxelComponent {}

#[register_world_component]
impl VoxelTerrainComponent {}

#[register_world_component]
impl VoxelLandformComponent {}

#[register_world_component]
impl VoxelFlatTerrainComponent {}

// Reported unfinished (Pulsar-Native#1035, Phase 4; tracked in #1056): the
// properties card shows the reason and issue, and attaching one logs them once.
pulsar_world_registry::declare_unfinished_component!(
    "VoxelComponent",
    "no voxel renderer draws a free-standing voxel component; use a voxel terrain",
    "https://github.com/Far-Beyond-Pulsar/Pulsar-Native/issues/1056",
);
