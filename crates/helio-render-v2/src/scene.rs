//! Scene database – the authoritative source for all rendered content

use crate::features::{BillboardInstance, LightType};
use crate::mesh::GpuMesh;

/// A single renderable object in the scene
#[derive(Clone)]
pub struct SceneObject {
    pub mesh: GpuMesh,
}

impl SceneObject {
    pub fn new(mesh: GpuMesh) -> Self {
        Self { mesh }
    }
}

/// A light source in the scene
#[derive(Clone, Debug)]
pub struct SceneLight {
    pub light_type: LightType,
    pub position: [f32; 3],
    pub direction: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
    pub range: f32,
}

impl SceneLight {
    pub fn directional(direction: [f32; 3], color: [f32; 3], intensity: f32) -> Self {
        Self {
            light_type: LightType::Directional,
            position: [0.0; 3],
            direction,
            color,
            intensity,
            range: 1000.0,
        }
    }

    pub fn point(position: [f32; 3], color: [f32; 3], intensity: f32, range: f32) -> Self {
        Self {
            light_type: LightType::Point,
            position,
            direction: [0.0, -1.0, 0.0],
            color,
            intensity,
            range,
        }
    }
}

/// The scene database – defines all rendered content
///
/// This is the single authoritative source for everything the renderer draws.
/// No content is hardcoded in the renderer itself.
pub struct Scene {
    pub objects: Vec<SceneObject>,
    pub lights: Vec<SceneLight>,
    pub ambient_color: [f32; 3],
    pub ambient_intensity: f32,
    pub billboards: Vec<BillboardInstance>,
    /// Background/sky clear color. Default is black.
    pub sky_color: [f32; 3],
}

impl Scene {
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
            lights: Vec::new(),
            ambient_color: [0.0, 0.0, 0.0],
            ambient_intensity: 0.0,
            billboards: Vec::new(),
            sky_color: [0.0, 0.0, 0.0],
        }
    }

    pub fn with_sky(mut self, color: [f32; 3]) -> Self {
        self.sky_color = color;
        self
    }

    pub fn add_object(mut self, mesh: GpuMesh) -> Self {
        self.objects.push(SceneObject::new(mesh));
        self
    }

    pub fn add_light(mut self, light: SceneLight) -> Self {
        self.lights.push(light);
        self
    }

    pub fn add_billboard(mut self, billboard: BillboardInstance) -> Self {
        self.billboards.push(billboard);
        self
    }

    pub fn with_ambient(mut self, color: [f32; 3], intensity: f32) -> Self {
        self.ambient_color = color;
        self.ambient_intensity = intensity;
        self
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}
