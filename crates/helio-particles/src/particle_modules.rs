use glam::{Vec3, Vec4};

pub trait ParticleModule {
    fn update(&self, position: &mut Vec3, velocity: &mut Vec3, color: &mut Vec4, size: &mut Vec3, age: f32, lifetime: f32, dt: f32);
}

pub struct ColorOverLifetime {
    pub start_color: Vec4,
    pub end_color: Vec4,
}

impl ParticleModule for ColorOverLifetime {
    fn update(&self, _position: &mut Vec3, _velocity: &mut Vec3, color: &mut Vec4, _size: &mut Vec3, age: f32, lifetime: f32, _dt: f32) {
        let t = age / lifetime;
        *color = self.start_color.lerp(self.end_color, t);
    }
}

pub struct SizeOverLifetime {
    pub start_size: Vec3,
    pub end_size: Vec3,
}

impl ParticleModule for SizeOverLifetime {
    fn update(&self, _position: &mut Vec3, _velocity: &mut Vec3, _color: &mut Vec4, size: &mut Vec3, age: f32, lifetime: f32, _dt: f32) {
        let t = age / lifetime;
        *size = self.start_size.lerp(self.end_size, t);
    }
}

pub struct VelocityOverLifetime {
    pub linear: Vec3,
    pub orbital: Vec3,
}

impl ParticleModule for VelocityOverLifetime {
    fn update(&self, _position: &mut Vec3, velocity: &mut Vec3, _color: &mut Vec4, _size: &mut Vec3, _age: f32, _lifetime: f32, dt: f32) {
        *velocity += self.linear * dt;
    }
}

pub struct ForceOverLifetime {
    pub force: Vec3,
}

impl ParticleModule for ForceOverLifetime {
    fn update(&self, _position: &mut Vec3, velocity: &mut Vec3, _color: &mut Vec4, _size: &mut Vec3, _age: f32, _lifetime: f32, dt: f32) {
        *velocity += self.force * dt;
    }
}
