use glam::{Vec3, Vec4, Mat4};
use bytemuck::{Pod, Zeroable};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightType {
    Directional,
    Point,
    Spot,
    Area,
    Sky,
}

#[derive(Debug, Clone)]
pub struct DirectionalLight {
    pub direction: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub cast_shadows: bool,
    pub shadow_cascade_count: u32,
    pub shadow_distance: f32,
    pub shadow_bias: f32,
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self {
            direction: Vec3::new(0.3, -0.7, 0.2).normalize(),
            color: Vec3::ONE,
            intensity: 100000.0,
            cast_shadows: true,
            shadow_cascade_count: 4,
            shadow_distance: 100.0,
            shadow_bias: 0.0005,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PointLight {
    pub position: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub radius: f32,
    pub cast_shadows: bool,
    pub shadow_resolution: u32,
}

impl Default for PointLight {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            color: Vec3::ONE,
            intensity: 1000.0,
            radius: 10.0,
            cast_shadows: true,
            shadow_resolution: 512,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpotLight {
    pub position: Vec3,
    pub direction: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub radius: f32,
    pub inner_cone_angle: f32,
    pub outer_cone_angle: f32,
    pub cast_shadows: bool,
    pub shadow_resolution: u32,
}

impl Default for SpotLight {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            direction: Vec3::NEG_Y,
            color: Vec3::ONE,
            intensity: 1000.0,
            radius: 10.0,
            inner_cone_angle: 0.785, // 45 degrees
            outer_cone_angle: 1.047, // 60 degrees
            cast_shadows: true,
            shadow_resolution: 512,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AreaLight {
    pub position: Vec3,
    pub rotation: glam::Quat,
    pub color: Vec3,
    pub intensity: f32,
    pub width: f32,
    pub height: f32,
    pub shape: AreaLightShape,
    pub cast_shadows: bool,
    pub two_sided: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AreaLightShape {
    Rectangle,
    Disk,
    Sphere,
    Tube,
}

impl Default for AreaLight {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: glam::Quat::IDENTITY,
            color: Vec3::ONE,
            intensity: 100.0,
            width: 1.0,
            height: 1.0,
            shape: AreaLightShape::Rectangle,
            cast_shadows: false,
            two_sided: false,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LightUniform {
    pub position_radius: Vec4,
    pub color_intensity: Vec4,
    pub direction_type: Vec4,
    pub spot_angles: Vec4,
}

unsafe impl Pod for LightUniform {}
unsafe impl Zeroable for LightUniform {}

impl From<&PointLight> for LightUniform {
    fn from(light: &PointLight) -> Self {
        Self {
            position_radius: Vec4::new(
                light.position.x,
                light.position.y,
                light.position.z,
                light.radius,
            ),
            color_intensity: Vec4::new(
                light.color.x,
                light.color.y,
                light.color.z,
                light.intensity,
            ),
            direction_type: Vec4::new(0.0, 0.0, 0.0, 1.0), // Type 1 = Point
            spot_angles: Vec4::ZERO,
        }
    }
}

impl From<&DirectionalLight> for LightUniform {
    fn from(light: &DirectionalLight) -> Self {
        Self {
            position_radius: Vec4::ZERO,
            color_intensity: Vec4::new(
                light.color.x,
                light.color.y,
                light.color.z,
                light.intensity,
            ),
            direction_type: Vec4::new(
                light.direction.x,
                light.direction.y,
                light.direction.z,
                0.0, // Type 0 = Directional
            ),
            spot_angles: Vec4::ZERO,
        }
    }
}

impl From<&SpotLight> for LightUniform {
    fn from(light: &SpotLight) -> Self {
        Self {
            position_radius: Vec4::new(
                light.position.x,
                light.position.y,
                light.position.z,
                light.radius,
            ),
            color_intensity: Vec4::new(
                light.color.x,
                light.color.y,
                light.color.z,
                light.intensity,
            ),
            direction_type: Vec4::new(
                light.direction.x,
                light.direction.y,
                light.direction.z,
                2.0, // Type 2 = Spot
            ),
            spot_angles: Vec4::new(
                light.inner_cone_angle.cos(),
                light.outer_cone_angle.cos(),
                0.0,
                0.0,
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightingMode {
    Forward,
    Deferred,
    Clustered,
    Tiled,
}

pub struct LightingSystem {
    pub directional_lights: Vec<DirectionalLight>,
    pub point_lights: Vec<PointLight>,
    pub spot_lights: Vec<SpotLight>,
    pub area_lights: Vec<AreaLight>,
    pub mode: LightingMode,
}

impl LightingSystem {
    pub fn new(mode: LightingMode) -> Self {
        Self {
            directional_lights: Vec::new(),
            point_lights: Vec::new(),
            spot_lights: Vec::new(),
            area_lights: Vec::new(),
            mode,
        }
    }
    
    pub fn add_directional_light(&mut self, light: DirectionalLight) {
        self.directional_lights.push(light);
    }
    
    pub fn add_point_light(&mut self, light: PointLight) {
        self.point_lights.push(light);
    }
    
    pub fn add_spot_light(&mut self, light: SpotLight) {
        self.spot_lights.push(light);
    }
    
    pub fn add_area_light(&mut self, light: AreaLight) {
        self.area_lights.push(light);
    }
    
    pub fn total_light_count(&self) -> usize {
        self.directional_lights.len()
            + self.point_lights.len()
            + self.spot_lights.len()
            + self.area_lights.len()
    }
}

impl Default for LightingSystem {
    fn default() -> Self {
        Self::new(LightingMode::Deferred)
    }
}
