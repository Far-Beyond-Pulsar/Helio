//! Water's shoreline from Helio's voxel terrain.
//!
//! Water bakes its shore map from any [`WaterTerrainProvider`] and names no
//! terrain; the voxel terrain names no water. This is where the two meet: a
//! provider answering water's elevation queries with the voxel world's
//! canonical ground-column query.

use glam::DVec3;
use helio_pass_voxel_planet::{Matter, Planet, VolumeTransform};
use helio_pass_water_sim::{WaterShoreMap, WaterTerrainProvider};

/// [`WaterTerrainProvider`] over a voxel planet. On sphere-shaped planets the
/// query follows the voxel world's radial terrain column under each
/// ocean-plane sample; on voxel planes it is the exact vertical column.
/// Procedural terrain and edits are included.
pub struct VoxelShoreTerrain<'a> {
    pub planet: &'a Planet,
    pub transform: VolumeTransform,
    /// Absolute-world height of the ocean rest plane.
    pub rest_y: f64,
}

impl WaterTerrainProvider for VoxelShoreTerrain<'_> {
    fn elevation_y(&self, world_x: f64, world_z: f64) -> Option<f64> {
        let planet = self.planet;
        let transform = self.transform;
        let radius = planet.outer_radius();
        if !planet.grid().is_plane()
            && (world_x - transform.position.x).hypot(world_z - transform.position.z) > radius
        {
            return None;
        }
        // `surface_point_of` is the voxel world's canonical direct column
        // query: for a plane it is exact, for a sphere only the small
        // projection residual differs.
        let sample = DVec3::new(world_x, self.rest_y, world_z);
        let local = transform.to_local(sample);
        let ground = planet.surface_point_of(local, 0.0, Matter::Ground);
        Some(transform.to_world(ground).y)
    }
}

/// Bake a shore map (`center_xz`, `extent_xz` in metres) from a voxel
/// planet's terrain with the ocean resting at `rest_y`. A one-time CPU bake:
/// call after authoring or editing the terrain, not in the render loop.
pub fn shore_map_from_voxel_planet(
    center_xz: [f64; 2],
    extent_xz: [f32; 2],
    rest_y: f64,
    planet: &Planet,
    transform: VolumeTransform,
) -> Result<WaterShoreMap, &'static str> {
    WaterShoreMap::from_terrain(
        center_xz,
        extent_xz,
        rest_y,
        &VoxelShoreTerrain {
            planet,
            transform,
            rest_y,
        },
    )
}
