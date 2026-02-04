use glam::{Vec3, Vec4};

pub struct ClusteredLighting {
    pub tile_size: u32,
    pub cluster_depth_slices: u32,
    pub max_lights_per_cluster: u32,
}

impl ClusteredLighting {
    pub fn new() -> Self {
        Self {
            tile_size: 16,
            cluster_depth_slices: 16,
            max_lights_per_cluster: 256,
        }
    }
    
    pub fn compute_cluster_index(&self, screen_pos: Vec3, screen_size: (u32, u32)) -> u32 {
        let tile_x = (screen_pos.x / self.tile_size as f32) as u32;
        let tile_y = (screen_pos.y / self.tile_size as f32) as u32;
        let depth_slice = (screen_pos.z * self.cluster_depth_slices as f32) as u32;
        
        let tiles_x = (screen_size.0 + self.tile_size - 1) / self.tile_size;
        let tiles_y = (screen_size.1 + self.tile_size - 1) / self.tile_size;
        
        tile_x + tile_y * tiles_x + depth_slice * tiles_x * tiles_y
    }
}

impl Default for ClusteredLighting {
    fn default() -> Self {
        Self::new()
    }
}

pub struct TiledLighting {
    pub tile_size: u32,
    pub max_lights_per_tile: u32,
}

impl TiledLighting {
    pub fn new() -> Self {
        Self {
            tile_size: 16,
            max_lights_per_tile: 256,
        }
    }
}

impl Default for TiledLighting {
    fn default() -> Self {
        Self::new()
    }
}
