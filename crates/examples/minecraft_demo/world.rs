//! SceneDB-facing world assembly for the Minecraft demo.

use super::generation::{encode_component_settings, TerrainSettings};
use helio_pass_voxel_mesh::{VoxelComponent, VoxelTerrain};
use pulsar_scenedb::{Entity, World};

pub fn spawn_world(world: &mut World, settings: &TerrainSettings) -> Entity {
    let mut terrain = VoxelTerrain::with_dimensions(settings.dimensions);
    terrain.generate_at(settings.seed, [0, 0]);
    let entity = world.spawn();
    let mut component = VoxelComponent::new(terrain, 0.75, 0);
    encode_component_settings(settings, &mut component);
    component.set_chunk_coord(0, 0);
    world.insert(entity, component);
    entity
}

pub fn spawn_default_world(world: &mut World) -> Entity {
    spawn_world(world, &TerrainSettings::default())
}
