use glam::Vec3;

pub struct TerrainClipmapLevel {
    pub scale: f32,
    pub resolution: u32,
    pub center: Vec3,
}

pub struct TerrainClipmap {
    pub levels: Vec<TerrainClipmapLevel>,
    pub level_count: u32,
    pub resolution_per_level: u32,
    pub scale_multiplier: f32,
}

impl TerrainClipmap {
    pub fn new(level_count: u32, resolution_per_level: u32) -> Self {
        let mut levels = Vec::new();
        
        for i in 0..level_count {
            let scale = (1 << i) as f32;
            levels.push(TerrainClipmapLevel {
                scale,
                resolution: resolution_per_level,
                center: Vec3::ZERO,
            });
        }
        
        Self {
            levels,
            level_count,
            resolution_per_level,
            scale_multiplier: 2.0,
        }
    }
    
    pub fn update(&mut self, camera_position: Vec3) {
        for level in &mut self.levels {
            level.center = Vec3::new(
                (camera_position.x / level.scale).floor() * level.scale,
                camera_position.y,
                (camera_position.z / level.scale).floor() * level.scale,
            );
        }
    }
}
