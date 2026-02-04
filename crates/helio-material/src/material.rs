use glam::{Vec2, Vec3, Vec4};
use std::collections::HashMap;
use bitflags::bitflags;

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct MaterialFlags: u32 {
        const TWO_SIDED = 1 << 0;
        const ALPHA_BLEND = 1 << 1;
        const ALPHA_TEST = 1 << 2;
        const UNLIT = 1 << 3;
        const SUBSURFACE = 1 << 4;
        const CLEAR_COAT = 1 << 5;
        const CLOTH = 1 << 6;
        const SPECULAR = 1 << 7;
        const ANISOTROPIC = 1 << 8;
        const SHEEN = 1 << 9;
        const TRANSMISSION = 1 << 10;
        const VOLUME = 1 << 11;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShadingModel {
    Unlit,
    DefaultLit,
    Subsurface,
    PreintegratedSkin,
    ClearCoat,
    Cloth,
    Hair,
    Eye,
    Foliage,
    TwoSidedFoliage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    Opaque,
    Masked,
    Translucent,
    Additive,
    Modulate,
}

#[derive(Debug, Clone)]
pub struct Material {
    pub name: String,
    pub shading_model: ShadingModel,
    pub blend_mode: BlendMode,
    pub flags: MaterialFlags,
    
    // PBR parameters
    pub base_color: Vec4,
    pub metallic: f32,
    pub roughness: f32,
    pub specular: f32,
    pub emissive: Vec3,
    pub emissive_strength: f32,
    pub normal_scale: f32,
    pub occlusion_strength: f32,
    
    // Advanced parameters
    pub anisotropy: f32,
    pub anisotropy_rotation: f32,
    pub subsurface_color: Vec3,
    pub subsurface_radius: f32,
    pub transmission: f32,
    pub thickness: f32,
    pub ior: f32,
    
    // Clear coat
    pub clear_coat: f32,
    pub clear_coat_roughness: f32,
    pub clear_coat_normal: Vec3,
    
    // Sheen (cloth)
    pub sheen_color: Vec3,
    pub sheen_roughness: f32,
    
    // Textures (indices into texture array)
    pub base_color_texture: Option<u32>,
    pub metallic_roughness_texture: Option<u32>,
    pub normal_texture: Option<u32>,
    pub occlusion_texture: Option<u32>,
    pub emissive_texture: Option<u32>,
    pub opacity_texture: Option<u32>,
    
    // UV transforms
    pub uv_offset: Vec2,
    pub uv_scale: Vec2,
    pub uv_rotation: f32,
    
    // Rendering parameters
    pub alpha_cutoff: f32,
    pub shadow_bias: f32,
    pub shadow_slope_bias: f32,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            name: String::from("Default"),
            shading_model: ShadingModel::DefaultLit,
            blend_mode: BlendMode::Opaque,
            flags: MaterialFlags::empty(),
            base_color: Vec4::ONE,
            metallic: 0.0,
            roughness: 0.5,
            specular: 0.5,
            emissive: Vec3::ZERO,
            emissive_strength: 1.0,
            normal_scale: 1.0,
            occlusion_strength: 1.0,
            anisotropy: 0.0,
            anisotropy_rotation: 0.0,
            subsurface_color: Vec3::ONE,
            subsurface_radius: 1.0,
            transmission: 0.0,
            thickness: 0.0,
            ior: 1.5,
            clear_coat: 0.0,
            clear_coat_roughness: 0.0,
            clear_coat_normal: Vec3::ZERO,
            sheen_color: Vec3::ZERO,
            sheen_roughness: 0.0,
            base_color_texture: None,
            metallic_roughness_texture: None,
            normal_texture: None,
            occlusion_texture: None,
            emissive_texture: None,
            opacity_texture: None,
            uv_offset: Vec2::ZERO,
            uv_scale: Vec2::ONE,
            uv_rotation: 0.0,
            alpha_cutoff: 0.5,
            shadow_bias: 0.0005,
            shadow_slope_bias: 0.001,
        }
    }
}

impl Material {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }
    
    pub fn with_base_color(mut self, color: Vec4) -> Self {
        self.base_color = color;
        self
    }
    
    pub fn with_metallic_roughness(mut self, metallic: f32, roughness: f32) -> Self {
        self.metallic = metallic;
        self.roughness = roughness;
        self
    }
    
    pub fn with_emissive(mut self, color: Vec3, strength: f32) -> Self {
        self.emissive = color;
        self.emissive_strength = strength;
        self
    }
    
    pub fn with_shading_model(mut self, model: ShadingModel) -> Self {
        self.shading_model = model;
        self
    }
    
    pub fn with_blend_mode(mut self, mode: BlendMode) -> Self {
        self.blend_mode = mode;
        self
    }
    
    pub fn enable_two_sided(mut self) -> Self {
        self.flags |= MaterialFlags::TWO_SIDED;
        self
    }
    
    pub fn enable_alpha_blend(mut self) -> Self {
        self.flags |= MaterialFlags::ALPHA_BLEND;
        self.blend_mode = BlendMode::Translucent;
        self
    }
}

pub struct MaterialLibrary {
    materials: HashMap<String, Material>,
}

impl MaterialLibrary {
    pub fn new() -> Self {
        let mut library = Self {
            materials: HashMap::new(),
        };
        
        // Add default materials
        library.add(Material::new("Default"));
        library.add(Material::new("Metal")
            .with_metallic_roughness(1.0, 0.2));
        library.add(Material::new("Plastic")
            .with_metallic_roughness(0.0, 0.5));
        library.add(Material::new("Glass")
            .with_metallic_roughness(0.0, 0.0)
            .with_blend_mode(BlendMode::Translucent));
        
        library
    }
    
    pub fn add(&mut self, material: Material) {
        self.materials.insert(material.name.clone(), material);
    }
    
    pub fn get(&self, name: &str) -> Option<&Material> {
        self.materials.get(name)
    }
    
    pub fn get_mut(&mut self, name: &str) -> Option<&mut Material> {
        self.materials.get_mut(name)
    }
}

impl Default for MaterialLibrary {
    fn default() -> Self {
        Self::new()
    }
}
