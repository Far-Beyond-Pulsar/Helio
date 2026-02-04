use glam::{Vec3, Vec4};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmitterShape {
    Point,
    Sphere,
    Box,
    Cone,
    Mesh,
}

pub struct ParticleEmitter {
    pub position: Vec3,
    pub shape: EmitterShape,
    pub emission_rate: f32,
    pub burst_count: u32,
    pub initial_velocity: Vec3,
    pub velocity_randomness: f32,
    pub lifetime: f32,
    pub lifetime_randomness: f32,
    pub initial_size: Vec3,
    pub size_randomness: f32,
    pub initial_color: Vec4,
    pub color_randomness: f32,
    pub gravity: Vec3,
}

impl Default for ParticleEmitter {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            shape: EmitterShape::Point,
            emission_rate: 10.0,
            burst_count: 0,
            initial_velocity: Vec3::Y,
            velocity_randomness: 0.1,
            lifetime: 5.0,
            lifetime_randomness: 0.5,
            initial_size: Vec3::ONE,
            size_randomness: 0.1,
            initial_color: Vec4::ONE,
            color_randomness: 0.0,
            gravity: Vec3::new(0.0, -9.8, 0.0),
        }
    }
}
