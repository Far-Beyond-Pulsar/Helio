//! SceneDB World registration for voxel authoring components.
//!
//! These behaviors intentionally do not invoke rendering or generation. A
//! future voxel backend can observe the typed SceneDB rows and consume
//! revisioned external updates without coupling component hydration to a pass.

use engine_class_derive::{register_runtime_behavior, register_world_component};
use pulsar_reflection::{ComponentRuntimeBehavior, ComponentRuntimeContext, RuntimeComponentOwner};

use super::{VoxelComponent, VoxelFlatTerrainComponent, VoxelLandformComponent, VoxelMoonComponent, VoxelTerrainComponent};

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

#[register_world_component]
#[register_runtime_behavior]
impl ComponentRuntimeBehavior for VoxelMoonComponent {
    const CLASS_NAME: &'static str = "VoxelMoonComponent";

    fn sync_component(
        _owner: &RuntimeComponentOwner,
        _component_index: usize,
        _component: &Self,
        _context: &mut dyn ComponentRuntimeContext,
    ) {
        // Settings of the terrain on the same entity, read when it is projected.
    }
}
