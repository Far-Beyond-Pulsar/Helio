//! The water feature's public contract.
//!
//! Water bakes its shoreline from a terrain's elevations
//! (`helio_pass_water_sim::WaterShoreMap::from_terrain`) and names no
//! terrain: a terrain feature answers [`WaterTerrainProvider`].

/// Supplies actual terrain elevations in absolute world coordinates. `None`
/// means the requested location has no terrain data; baking then fails rather
/// than silently treating missing data as ocean floor.
pub trait WaterTerrainProvider: Sync {
    fn elevation_y(&self, world_x: f64, world_z: f64) -> Option<f64>;
}
