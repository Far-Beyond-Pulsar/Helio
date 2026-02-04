use glam::{Vec3, Vec4};
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Particle {
    pub position: Vec3,
    pub age: f32,
    pub velocity: Vec3,
    pub lifetime: f32,
    pub size: Vec3,
    pub rotation: f32,
    pub color: Vec4,
}

unsafe impl Pod for Particle {}
unsafe impl Zeroable for Particle {}

pub struct ParticleSystem {
    pub particles: Vec<Particle>,
    pub max_particles: usize,
    pub emission_rate: f32,
    pub gpu_simulation: bool,
}

impl ParticleSystem {
    pub fn new(max_particles: usize) -> Self {
        Self {
            particles: Vec::with_capacity(max_particles),
            max_particles,
            emission_rate: 10.0,
            gpu_simulation: true,
        }
    }
    
    pub fn update(&mut self, dt: f32) {
        self.particles.retain_mut(|p| {
            p.age += dt;
            p.position += p.velocity * dt;
            p.age < p.lifetime
        });
    }
    
    pub fn emit(&mut self, position: Vec3, velocity: Vec3, lifetime: f32) {
        if self.particles.len() < self.max_particles {
            self.particles.push(Particle {
                position,
                age: 0.0,
                velocity,
                lifetime,
                size: Vec3::ONE,
                rotation: 0.0,
                color: Vec4::ONE,
            });
        }
    }
}
