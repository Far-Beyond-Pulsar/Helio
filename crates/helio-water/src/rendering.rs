pub struct WaterRendering {
    pub reflection_quality: ReflectionQuality,
    pub refraction_enabled: bool,
    pub subsurface_scattering: bool,
    pub foam_texture: Option<u32>,
    pub normal_map: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReflectionQuality {
    Planar,
    ScreenSpace,
    RayTraced,
}

impl Default for WaterRendering {
    fn default() -> Self {
        Self {
            reflection_quality: ReflectionQuality::ScreenSpace,
            refraction_enabled: true,
            subsurface_scattering: true,
            foam_texture: None,
            normal_map: None,
        }
    }
}
