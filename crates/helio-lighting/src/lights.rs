use glam::{Mat4, Vec3};
use bytemuck::{Pod, Zeroable};

#[derive(Clone, Copy, Debug)]
pub struct DirectionalLight {
    pub direction: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub cast_shadows: bool,
    pub shadow_distance: f32,
    pub cascade_count: u32,
}

impl DirectionalLight {
    pub fn new(direction: Vec3, color: Vec3, intensity: f32) -> Self {
        Self {
            direction: direction.normalize(),
            color,
            intensity,
            cast_shadows: true,
            shadow_distance: 100.0,
            cascade_count: 4,
        }
    }

    pub fn get_view_matrix(&self, focus_point: Vec3) -> Mat4 {
        Mat4::look_at_rh(focus_point - self.direction * 50.0, focus_point, Vec3::Y)
    }
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::ONE, 1.0)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DirectionalLightGpu {
    pub direction: [f32; 3],
    pub intensity: f32,
    pub color: [f32; 3],
    pub cast_shadows: u32,
    pub cascade_splits: [f32; 4],
    pub _padding: [u32; 4],
}

impl From<&DirectionalLight> for DirectionalLightGpu {
    fn from(light: &DirectionalLight) -> Self {
        Self {
            direction: light.direction.to_array(),
            intensity: light.intensity,
            color: light.color.to_array(),
            cast_shadows: light.cast_shadows as u32,
            cascade_splits: [0.0; 4],
            _padding: [0; 4],
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PointLight {
    pub position: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub radius: f32,
    pub cast_shadows: bool,
}

impl PointLight {
    pub fn new(position: Vec3, color: Vec3, intensity: f32, radius: f32) -> Self {
        Self {
            position,
            color,
            intensity,
            radius,
            cast_shadows: true,
        }
    }

    pub fn attenuation(&self, distance: f32) -> f32 {
        let d = distance / self.radius;
        let d2 = d * d;
        let d4 = d2 * d2;
        (1.0 - d4).max(0.0) / (1.0 + d2)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct PointLightGpu {
    pub position: [f32; 3],
    pub radius: f32,
    pub color: [f32; 3],
    pub intensity: f32,
    pub cast_shadows: u32,
    pub _padding: [u32; 3],
}

impl From<&PointLight> for PointLightGpu {
    fn from(light: &PointLight) -> Self {
        Self {
            position: light.position.to_array(),
            radius: light.radius,
            color: light.color.to_array(),
            intensity: light.intensity,
            cast_shadows: light.cast_shadows as u32,
            _padding: [0; 3],
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SpotLight {
    pub position: Vec3,
    pub direction: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub radius: f32,
    pub inner_cone_angle: f32,
    pub outer_cone_angle: f32,
    pub cast_shadows: bool,
}

impl SpotLight {
    pub fn new(
        position: Vec3,
        direction: Vec3,
        color: Vec3,
        intensity: f32,
        radius: f32,
        inner_angle: f32,
        outer_angle: f32,
    ) -> Self {
        Self {
            position,
            direction: direction.normalize(),
            color,
            intensity,
            radius,
            inner_cone_angle: inner_angle,
            outer_cone_angle: outer_angle,
            cast_shadows: true,
        }
    }

    pub fn get_view_projection_matrix(&self) -> Mat4 {
        let view = Mat4::look_at_rh(self.position, self.position + self.direction, Vec3::Y);
        let proj = Mat4::perspective_rh(
            self.outer_cone_angle * 2.0,
            1.0,
            0.1,
            self.radius,
        );
        proj * view
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SpotLightGpu {
    pub position: [f32; 3],
    pub radius: f32,
    pub direction: [f32; 3],
    pub intensity: f32,
    pub color: [f32; 3],
    pub inner_cone_cos: f32,
    pub outer_cone_cos: f32,
    pub cast_shadows: u32,
    pub _padding: [u32; 2],
}

impl From<&SpotLight> for SpotLightGpu {
    fn from(light: &SpotLight) -> Self {
        Self {
            position: light.position.to_array(),
            radius: light.radius,
            direction: light.direction.to_array(),
            intensity: light.intensity,
            color: light.color.to_array(),
            inner_cone_cos: light.inner_cone_angle.cos(),
            outer_cone_cos: light.outer_cone_angle.cos(),
            cast_shadows: light.cast_shadows as u32,
            _padding: [0; 2],
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum AreaLightShape {
    Rectangle,
    Disk,
    Sphere,
    Tube,
}

#[derive(Clone, Copy, Debug)]
pub struct AreaLight {
    pub position: Vec3,
    pub direction: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub width: f32,
    pub height: f32,
    pub shape: AreaLightShape,
    pub two_sided: bool,
}

impl AreaLight {
    pub fn new_rectangle(
        position: Vec3,
        direction: Vec3,
        color: Vec3,
        intensity: f32,
        width: f32,
        height: f32,
    ) -> Self {
        Self {
            position,
            direction: direction.normalize(),
            color,
            intensity,
            width,
            height,
            shape: AreaLightShape::Rectangle,
            two_sided: false,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct AreaLightGpu {
    pub position: [f32; 3],
    pub shape: u32,
    pub direction: [f32; 3],
    pub intensity: f32,
    pub color: [f32; 3],
    pub width: f32,
    pub height: f32,
    pub two_sided: u32,
    pub _padding: [u32; 2],
}

impl From<&AreaLight> for AreaLightGpu {
    fn from(light: &AreaLight) -> Self {
        Self {
            position: light.position.to_array(),
            shape: light.shape as u32,
            direction: light.direction.to_array(),
            intensity: light.intensity,
            color: light.color.to_array(),
            width: light.width,
            height: light.height,
            two_sided: light.two_sided as u32,
            _padding: [0; 2],
        }
    }
}
