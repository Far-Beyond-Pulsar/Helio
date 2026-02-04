pub struct TerrainLayer {
    pub albedo_texture: Option<u32>,
    pub normal_texture: Option<u32>,
    pub height_texture: Option<u32>,
    pub metallic_roughness: Option<u32>,
    pub tiling: f32,
}

pub struct TerrainMaterial {
    pub layers: Vec<TerrainLayer>,
    pub blend_map: Option<u32>,
    pub displacement_strength: f32,
    pub tessellation_factor: f32,
}

impl TerrainMaterial {
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            blend_map: None,
            displacement_strength: 1.0,
            tessellation_factor: 1.0,
        }
    }
}

impl Default for TerrainMaterial {
    fn default() -> Self {
        Self::new()
    }
}
