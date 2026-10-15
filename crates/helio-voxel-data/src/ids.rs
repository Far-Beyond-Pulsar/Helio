//! Names shared by the hosts and the renderers: Helio's terrain generator
//! and the two voxel renderers. A terrain generator itself
//! (`helio_pass_voxel_planet::terrain`) is a field evaluated identically on
//! CPU and GPU, so worlds of any size need no stored chunks.

/// Helio's built-in terrain generator: ordered layer stacks (planets,
/// moons, flat worlds) on planets, planes and infinite planes. Its settings
/// component is `VoxelTerrainLayersComponent`.
pub const VOXEL_TERRAIN_GENERATOR: &str = "helio.terrain";
pub const VOXEL_TERRAIN_GENERATOR_VERSION: u32 = 1;
/// The streamed voxel terrain renderer, which draws every registered
/// terrain generator.
pub const VOXEL_TERRAIN_RENDERER: &str = "helio.voxel-terrain";
/// The voxel object renderer: greedy-meshes a free-standing volume's chunks
/// into an ordinary mesh draw placed by its owner.
pub const VOXEL_MESH_RENDERER: &str = "helio.voxel-mesh";
