use glam::{Vec3, Vec4};

pub struct PbrMaterial {
    pub albedo: Vec4,
    pub metallic: f32,
    pub roughness: f32,
    pub reflectance: f32,
    pub normal: Vec3,
    pub emissive: Vec3,
    pub occlusion: f32,
}

impl Default for PbrMaterial {
    fn default() -> Self {
        Self {
            albedo: Vec4::ONE,
            metallic: 0.0,
            roughness: 0.5,
            reflectance: 0.5,
            normal: Vec3::new(0.0, 0.0, 1.0),
            emissive: Vec3::ZERO,
            occlusion: 1.0,
        }
    }
}

pub fn disney_diffuse(n_dot_v: f32, n_dot_l: f32, l_dot_h: f32, roughness: f32) -> f32 {
    let fd90 = 0.5 + 2.0 * l_dot_h * l_dot_h * roughness;
    let light_scatter = (1.0 + (fd90 - 1.0) * (1.0 - n_dot_l).powi(5));
    let view_scatter = (1.0 + (fd90 - 1.0) * (1.0 - n_dot_v).powi(5));
    light_scatter * view_scatter
}

pub fn ggx_distribution(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    a2 / (std::f32::consts::PI * d * d)
}

pub fn schlick_fresnel(v_dot_h: f32, f0: Vec3) -> Vec3 {
    f0 + (Vec3::ONE - f0) * (1.0 - v_dot_h).powi(5)
}

pub fn smith_ggx_geometry(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    
    let ggx1 = n_dot_v / (n_dot_v * (1.0 - k) + k);
    let ggx2 = n_dot_l / (n_dot_l * (1.0 - k) + k);
    
    ggx1 * ggx2
}
