use glam::Vec3;

/// Opaque handle to a persistent scene instance.
///
/// Returned by [`Renderer::add_instance`]. Pass to [`Renderer::remove_instance`]
/// or [`Renderer::set_instance_transform`] to mutate the live scene.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct InstanceId(pub(crate) u32);

/// Opaque handle to a persistent scene light.
///
/// Returned by [`Renderer::add_light`]. Pass to [`Renderer::remove_light`]
/// or [`Renderer::update_light`] to mutate the live scene.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct LightId(pub(crate) u32);

/// A single light in the scene.
#[derive(Clone, Debug)]
pub struct SceneLight {
    pub light_type:  LightType,
    pub position:    Vec3,
    pub direction:   Vec3,
    pub color:       Vec3,
    pub intensity:   f32,
    pub range:       f32,
    pub inner_angle: f32,   // radians, spot only
    pub outer_angle: f32,   // radians, spot only
    pub cast_shadows: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LightType {
    Directional,
    Point,
    Spot,
}

impl SceneLight {
    pub fn directional(direction: Vec3, color: Vec3, intensity: f32) -> Self {
        Self {
            light_type:   LightType::Directional,
            position:     Vec3::ZERO,
            direction:    direction.normalize(),
            color,
            intensity,
            range:        0.0,
            inner_angle:  0.0,
            outer_angle:  0.0,
            cast_shadows: false,
        }
    }

    pub fn point(position: Vec3, color: Vec3, intensity: f32, range: f32) -> Self {
        Self {
            light_type:   LightType::Point,
            position,
            direction:    -Vec3::Y,
            color,
            intensity,
            range,
            inner_angle:  0.0,
            outer_angle:  0.0,
            cast_shadows: false,
        }
    }

    pub fn spot(position: Vec3, direction: Vec3, color: Vec3, intensity: f32, range: f32, inner_angle: f32, outer_angle: f32) -> Self {
        Self {
            light_type:   LightType::Spot,
            position,
            direction:    direction.normalize(),
            color,
            intensity,
            range,
            inner_angle,
            outer_angle,
            cast_shadows: false,
        }
    }

    pub fn with_shadows(mut self) -> Self { self.cast_shadows = true; self }
}

/// Atmospheric sky parameters.
#[derive(Clone, Debug)]
pub struct SkyAtmosphere {
    pub sun_direction:   Vec3,
    pub sun_intensity:   f32,
    pub rayleigh_scale:  f32,
    pub mie_scale:       f32,
    pub planet_radius:   f32,
    pub atmosphere_radius: f32,
}

impl Default for SkyAtmosphere {
    fn default() -> Self {
        SkyAtmosphere {
            sun_direction:    Vec3::new(0.0, 1.0, 0.3).normalize(),
            sun_intensity:    22.0,
            rayleigh_scale:   1.0,
            mie_scale:        1.0,
            planet_radius:    6371000.0,
            atmosphere_radius: 6471000.0,
        }
    }
}

/// Simple IBL skylight (read from a pre-baked env cube).
#[derive(Clone, Debug)]
pub struct Skylight {
    pub intensity:  f32,
    pub env_cube:   std::sync::Arc<wgpu::Texture>,
}

/// Packed GPU light for STORAGE binding (64 bytes).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Debug)]
pub struct GpuLight {
    pub position_type:     [f32; 4],   // xyz=position, w=type (0=dir,1=point,2=spot)
    pub direction_range:   [f32; 4],   // xyz=direction, w=range
    pub color_intensity:   [f32; 4],   // xyz=color, w=intensity
    pub cos_angles_shadow: [f32; 4],   // x=cos(inner), y=cos(outer), z=cast_shadows (0/1), w=pad
}

impl GpuLight {
    pub fn from_scene_light(l: &SceneLight) -> Self {
        let type_id = match l.light_type {
            LightType::Directional => 0.0,
            LightType::Point       => 1.0,
            LightType::Spot        => 2.0,
        };
        GpuLight {
            position_type:     [l.position.x, l.position.y, l.position.z, type_id],
            direction_range:   [l.direction.x, l.direction.y, l.direction.z, l.range],
            color_intensity:   [l.color.x, l.color.y, l.color.z, l.intensity],
            cos_angles_shadow: [
                l.inner_angle.cos(),
                l.outer_angle.cos(),
                l.cast_shadows as u32 as f32,
                0.0,
            ],
        }
    }
}
