use glam::{Mat4, Vec3};

pub struct Decal {
    pub transform: Mat4,
    pub albedo_texture: Option<u32>,
    pub normal_texture: Option<u32>,
    pub opacity: f32,
    pub blend_mode: DecalBlendMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecalBlendMode {
    Translucent,
    Additive,
    Multiply,
    Normal,
}

pub struct DecalSystem {
    pub decals: Vec<Decal>,
    pub max_decals: usize,
}

impl DecalSystem {
    pub fn new(max_decals: usize) -> Self {
        Self {
            decals: Vec::with_capacity(max_decals),
            max_decals,
        }
    }
    
    pub fn add_decal(&mut self, decal: Decal) {
        if self.decals.len() < self.max_decals {
            self.decals.push(decal);
        }
    }
}
