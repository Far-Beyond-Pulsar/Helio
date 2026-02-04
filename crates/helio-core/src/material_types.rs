use glam::{Vec3, Vec4};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ShadingModel {
    Standard,
    Subsurface,
    ClearCoat,
    Cloth,
    Hair,
    Eye,
    Unlit,
}

impl Default for ShadingModel {
    fn default() -> Self {
        Self::Standard
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AlphaMode {
    Opaque,
    Mask,
    Blend,
}

impl Default for AlphaMode {
    fn default() -> Self {
        Self::Opaque
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaterialProperties {
    pub base_color: Vec4,
    pub emissive: Vec3,
    pub metallic: f32,
    pub roughness: f32,
    pub reflectance: f32,
    pub anisotropy: f32,
    pub anisotropy_rotation: f32,
    pub clear_coat: f32,
    pub clear_coat_roughness: f32,
    pub sheen_color: Vec3,
    pub sheen_roughness: f32,
    pub subsurface_color: Vec3,
    pub subsurface_power: f32,
    pub transmission: f32,
    pub thickness: f32,
    pub ior: f32,
    pub alpha_cutoff: f32,
    pub shading_model: ShadingModel,
    pub alpha_mode: AlphaMode,
    pub double_sided: bool,
}

impl Default for MaterialProperties {
    fn default() -> Self {
        Self {
            base_color: Vec4::ONE,
            emissive: Vec3::ZERO,
            metallic: 0.0,
            roughness: 0.5,
            reflectance: 0.5,
            anisotropy: 0.0,
            anisotropy_rotation: 0.0,
            clear_coat: 0.0,
            clear_coat_roughness: 0.0,
            sheen_color: Vec3::ZERO,
            sheen_roughness: 0.0,
            subsurface_color: Vec3::ZERO,
            subsurface_power: 12.234,
            transmission: 0.0,
            thickness: 0.5,
            ior: 1.5,
            alpha_cutoff: 0.5,
            shading_model: ShadingModel::Standard,
            alpha_mode: AlphaMode::Opaque,
            double_sided: false,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MaterialGpuData {
    pub base_color: [f32; 4],
    pub emissive: [f32; 3],
    pub metallic: f32,
    pub roughness: f32,
    pub reflectance: f32,
    pub anisotropy: f32,
    pub anisotropy_rotation: f32,
    pub clear_coat: f32,
    pub clear_coat_roughness: f32,
    pub sheen_color: [f32; 3],
    pub sheen_roughness: f32,
    pub subsurface_color: [f32; 3],
    pub subsurface_power: f32,
    pub transmission: f32,
    pub thickness: f32,
    pub ior: f32,
    pub alpha_cutoff: f32,
    pub shading_model: u32,
    pub alpha_mode: u32,
    pub double_sided: u32,
    pub _padding: u32,
}

impl From<MaterialProperties> for MaterialGpuData {
    fn from(props: MaterialProperties) -> Self {
        Self {
            base_color: props.base_color.to_array(),
            emissive: props.emissive.to_array(),
            metallic: props.metallic,
            roughness: props.roughness,
            reflectance: props.reflectance,
            anisotropy: props.anisotropy,
            anisotropy_rotation: props.anisotropy_rotation,
            clear_coat: props.clear_coat,
            clear_coat_roughness: props.clear_coat_roughness,
            sheen_color: props.sheen_color.to_array(),
            sheen_roughness: props.sheen_roughness,
            subsurface_color: props.subsurface_color.to_array(),
            subsurface_power: props.subsurface_power,
            transmission: props.transmission,
            thickness: props.thickness,
            ior: props.ior,
            alpha_cutoff: props.alpha_cutoff,
            shading_model: props.shading_model as u32,
            alpha_mode: props.alpha_mode as u32,
            double_sided: props.double_sided as u32,
            _padding: 0,
        }
    }
}
