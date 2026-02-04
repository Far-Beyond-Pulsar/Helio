pub struct TerrainTessellation {
    pub enabled: bool,
    pub max_tessellation_factor: f32,
    pub distance_scale: f32,
    pub displacement_scale: f32,
}

impl Default for TerrainTessellation {
    fn default() -> Self {
        Self {
            enabled: true,
            max_tessellation_factor: 64.0,
            distance_scale: 100.0,
            displacement_scale: 1.0,
        }
    }
}
