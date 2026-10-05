//! SceneDB World registration for voxel authoring components.
//!
//! These behaviors intentionally do not invoke rendering or generation. A
//! future voxel backend can observe the typed SceneDB rows and consume
//! revisioned external updates without coupling component hydration to a pass.

use engine_class_derive::{register_component_runtime, register_runtime_behavior, register_world_component};
use pulsar_reflection::{ComponentRuntimeBehavior, ComponentRuntimeContext, RuntimeComponentOwner};

use super::{TerrainEventsEventWriterExt as _, VoxelComponent, VoxelFlatTerrainComponent, VoxelLandformComponent, VoxelTerrainComponent};

// Component callbacks borrow the authoritative terrain row from SceneDB.
// The pending event list is transient transport state populated only after a
// successful block removal; the EventHub writer queues delivery after the
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

    fn tick(
        &mut self,
        context: &mut pulsar_world_registry::ComponentContext<'_>,
        _delta_seconds: f32,
    ) {
        let mut pending = std::mem::take(&mut self.pending_block_broken);
        let mut pending = pending.into_iter();
        while let Some(block) = pending.next() {
            if let Err(error) = context.events.block_broken(block.clone()) {
                tracing::warn!(entity = ?context.entity, %error, "could not queue terrain block_broken event; retaining it");
                self.pending_block_broken.push(block);
                self.pending_block_broken.extend(pending);
                break;
            }
        }
    }
}

#[register_world_component]
#[register_runtime_behavior]
impl ComponentRuntimeBehavior for VoxelComponent {
    const CLASS_NAME: &'static str = "VoxelComponent";

    fn sync_component(
        _owner: &RuntimeComponentOwner,
        _component_index: usize,
        _component: &Self,
        _context: &mut dyn ComponentRuntimeContext,
    ) {
        // Authoring data is already present as a typed SceneDB World row.
    }
}

#[register_world_component]
#[register_runtime_behavior]
impl ComponentRuntimeBehavior for VoxelTerrainComponent {
    const CLASS_NAME: &'static str = "VoxelTerrainComponent";

    fn sync_component(
        _owner: &RuntimeComponentOwner,
        _component_index: usize,
        _component: &Self,
        _context: &mut dyn ComponentRuntimeContext,
    ) {
        // A future voxel backend will consume terrain configuration and
        // external revisioned data batches independently of scene hydration.
    }
}

#[register_world_component]
#[register_runtime_behavior]
impl ComponentRuntimeBehavior for VoxelLandformComponent {
    const CLASS_NAME: &'static str = "VoxelLandformComponent";

    fn sync_component(
        _owner: &RuntimeComponentOwner,
        _component_index: usize,
        _component: &Self,
        _context: &mut dyn ComponentRuntimeContext,
    ) {
        // Settings of the terrain on the same entity, read when it is projected.
    }
}

#[register_world_component]
#[register_runtime_behavior]
impl ComponentRuntimeBehavior for VoxelFlatTerrainComponent {
    const CLASS_NAME: &'static str = "VoxelFlatTerrainComponent";

    fn sync_component(
        _owner: &RuntimeComponentOwner,
        _component_index: usize,
        _component: &Self,
        _context: &mut dyn ComponentRuntimeContext,
    ) {
        // Settings of the terrain on the same entity, read when it is projected.
    }
}
