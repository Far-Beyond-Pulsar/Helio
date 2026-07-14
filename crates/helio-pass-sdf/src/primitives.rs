//! SDF shape primitives and parameters.

/// SDF shape type discriminant matching the WGSL shader switch cases.
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SdfShapeType {
    Sphere = 0,
    Cube = 1,
    Capsule = 2,
    Torus = 3,
    Cylinder = 4,
}

/// Shape-specific parameters packed into 4 floats for GPU layout.
///
/// | Shape    | param0         | param1       | param2  | param3   |
/// |----------|----------------|--------------|---------|----------|
/// | Sphere   | radius         | —            | —       | —        |
/// | Cube     | half_x         | half_y       | half_z  | —        |
/// | Capsule  | radius         | half_height  | —       | —        |
/// | Torus    | major_radius   | minor_radius | —       | —        |
/// | Cylinder | radius         | half_height  | —       | —        |
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SdfShapeParams {
    pub param0: f32,
    pub param1: f32,
    pub param2: f32,
    pub param3: f32,
}

impl Default for SdfShapeParams {
    fn default() -> Self {
        Self {
            param0: 0.0,
            param1: 0.0,
            param2: 0.0,
            param3: 0.0,
        }
    }
}

impl SdfShapeParams {
    pub fn sphere(radius: f32) -> Self {
        Self {
            param0: radius,
            ..Default::default()
        }
    }

    pub fn cube(half_x: f32, half_y: f32, half_z: f32) -> Self {
        Self {
            param0: half_x,
            param1: half_y,
            param2: half_z,
            param3: 0.0,
        }
    }

    pub fn capsule(radius: f32, half_height: f32) -> Self {
        Self {
            param0: radius,
            param1: half_height,
            ..Default::default()
        }
    }

    pub fn torus(major_radius: f32, minor_radius: f32) -> Self {
        Self {
            param0: major_radius,
            param1: minor_radius,
            ..Default::default()
        }
    }

    pub fn cylinder(radius: f32, half_height: f32) -> Self {
        Self {
            param0: radius,
            param1: half_height,
            ..Default::default()
        }
    }
}
