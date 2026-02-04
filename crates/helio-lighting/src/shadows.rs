use glam::{Mat4, Vec3, Vec4};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShadowTechnique {
    Basic,
    PCF,
    PCSS,
    VSM,
    ESM,
    MSM,
    RayTraced,
}

#[derive(Debug, Clone, Copy)]
pub struct ShadowCascade {
    pub view_projection: Mat4,
    pub split_distance: f32,
    pub bounds: (Vec3, Vec3),
}

pub struct CascadedShadowMaps {
    pub cascades: Vec<ShadowCascade>,
    pub cascade_count: u32,
    pub resolution: u32,
    pub technique: ShadowTechnique,
    pub bias: f32,
    pub normal_bias: f32,
    pub fade_distance: f32,
}

impl CascadedShadowMaps {
    pub fn new(cascade_count: u32, resolution: u32) -> Self {
        Self {
            cascades: Vec::with_capacity(cascade_count as usize),
            cascade_count,
            resolution,
            technique: ShadowTechnique::PCSS,
            bias: 0.0005,
            normal_bias: 0.001,
            fade_distance: 1.0,
        }
    }
    
    pub fn update_cascades(&mut self, camera_view: Mat4, camera_proj: Mat4, light_dir: Vec3) {
        self.cascades.clear();
        
        let split_lambda = 0.75;
        let near = 0.1;
        let far = 100.0;
        
        for i in 0..self.cascade_count {
            let split_near = if i == 0 {
                near
            } else {
                Self::cascade_split(i, self.cascade_count, near, far, split_lambda)
            };
            
            let split_far = Self::cascade_split(i + 1, self.cascade_count, near, far, split_lambda);
            
            // Calculate cascade frustum and light view-projection
            // Simplified version - would compute proper frustum bounds
            let view_proj = Mat4::IDENTITY;
            
            self.cascades.push(ShadowCascade {
                view_projection: view_proj,
                split_distance: split_far,
                bounds: (Vec3::ZERO, Vec3::ONE),
            });
        }
    }
    
    fn cascade_split(index: u32, count: u32, near: f32, far: f32, lambda: f32) -> f32 {
        let i = index as f32;
        let n = count as f32;
        
        let log = near * (far / near).powf(i / n);
        let uniform = near + (far - near) * (i / n);
        
        lambda * log + (1.0 - lambda) * uniform
    }
}

pub struct ShadowAtlas {
    pub resolution: u32,
    pub slots: Vec<ShadowSlot>,
}

#[derive(Debug, Clone)]
pub struct ShadowSlot {
    pub offset: (u32, u32),
    pub size: u32,
    pub occupied: bool,
}

impl ShadowAtlas {
    pub fn new(resolution: u32) -> Self {
        Self {
            resolution,
            slots: Vec::new(),
        }
    }
    
    pub fn allocate(&mut self, size: u32) -> Option<usize> {
        for (i, slot) in self.slots.iter_mut().enumerate() {
            if !slot.occupied && slot.size >= size {
                slot.occupied = true;
                return Some(i);
            }
        }
        None
    }
    
    pub fn free(&mut self, index: usize) {
        if let Some(slot) = self.slots.get_mut(index) {
            slot.occupied = false;
        }
    }
}

pub struct VirtualShadowMaps {
    pub resolution: u32,
    pub page_size: u32,
    pub cache_size: u32,
    pub enable_clipmap: bool,
}

impl VirtualShadowMaps {
    pub fn new() -> Self {
        Self {
            resolution: 16384,
            page_size: 128,
            cache_size: 8192,
            enable_clipmap: true,
        }
    }
}

impl Default for VirtualShadowMaps {
    fn default() -> Self {
        Self::new()
    }
}
