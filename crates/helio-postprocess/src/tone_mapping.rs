use glam::Vec3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToneMappingOperator {
    None,
    Reinhard,
    ReinhardExtended,
    Filmic,
    Uncharted2,
    ACES,
    AGX,
}

pub struct ToneMapping {
    pub operator: ToneMappingOperator,
    pub exposure: f32,
    pub white_point: f32,
}

impl Default for ToneMapping {
    fn default() -> Self {
        Self {
            operator: ToneMappingOperator::ACES,
            exposure: 1.0,
            white_point: 11.2,
        }
    }
}

pub fn aces_tonemap(color: Vec3) -> Vec3 {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    
    ((color * (color * a + b)) / (color * (color * c + d) + e)).clamp(Vec3::ZERO, Vec3::ONE)
}

pub fn reinhard_tonemap(color: Vec3) -> Vec3 {
    color / (Vec3::ONE + color)
}

pub fn uncharted2_tonemap(color: Vec3) -> Vec3 {
    let a = 0.15;
    let b = 0.50;
    let c = 0.10;
    let d = 0.20;
    let e = 0.02;
    let f = 0.30;
    
    ((color * (color * a + c * b) + d * e) / (color * (color * a + b) + d * f)) - e / f
}
