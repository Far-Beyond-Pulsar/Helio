//! SDF shape types and parameters.

use bytemuck::{Pod, Zeroable};

/// Supported SDF primitive shapes.
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SdfShapeType {
    Sphere   = 0,
    Cube     = 1,
    Capsule  = 2,
    Torus    = 3,
    Cylinder = 4,
}

/// Shape-specific parameters packed into 4 floats (maps to `vec4<f32>` in WGSL).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SdfShapeParams {
    pub param0: f32,
    pub param1: f32,
    pub param2: f32,
    pub param3: f32,
}

impl SdfShapeParams {
    pub fn sphere(radius: f32) -> Self {
        Self { param0: radius, param1: 0.0, param2: 0.0, param3: 0.0 }
    }
    pub fn cube(half_x: f32, half_y: f32, half_z: f32) -> Self {
        Self { param0: half_x, param1: half_y, param2: half_z, param3: 0.0 }
    }
    pub fn capsule(radius: f32, half_height: f32) -> Self {
        Self { param0: radius, param1: half_height, param2: 0.0, param3: 0.0 }
    }
    pub fn torus(major_r: f32, minor_r: f32) -> Self {
        Self { param0: major_r, param1: minor_r, param2: 0.0, param3: 0.0 }
    }
    pub fn cylinder(radius: f32, half_height: f32) -> Self {
        Self { param0: radius, param1: half_height, param2: 0.0, param3: 0.0 }
    }
}
