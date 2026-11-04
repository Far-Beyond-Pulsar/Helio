//! The material row engines author ([`MaterialRow`], under [`MATERIALS`]):
//! the GPU material ABI ([`crate::GpuMaterial`]) as a SceneDB row. Every
//! surface a kind draws reads it by the material index its own rows carry.

/// A material authored as a SceneDB component.
///
/// The row deliberately keeps the existing G-buffer shader ABI for this
/// migration step, but its lifetime and updates are now owned by the World.
/// Render passes resolve the packed `"materials"` buffer by key; no renderer
/// material table publication is required for the row itself.
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct MaterialRow {
    pub base_color: [f32; 4],
    pub emissive: [f32; 4],
    pub roughness_metallic: [f32; 4],
    pub tex_base_color: u32,
    pub tex_normal: u32,
    pub tex_roughness: u32,
    pub tex_emissive: u32,
    pub tex_occlusion: u32,
    pub workflow: u32,
    pub flags: u32,
    pub material_class: u32,
    pub class_params: [f32; 4],
}

// SAFETY: a `#[repr(C)]` bytemuck `Pod` row: no padding bytes, any bit
// pattern valid. Lets an engine's SceneDB component embed it.
unsafe impl pulsar_scenedb::Pod for MaterialRow {}

/// The buffer the renderer reads [`MaterialRow`]s from. Engines produce it
/// (a SceneDB component stored under this key, or a scene derivation).
pub const MATERIALS: helio_core::BufferContract<MaterialRow> =
    helio_core::BufferContract::new("materials");

impl From<crate::GpuMaterial> for MaterialRow {
    fn from(value: crate::GpuMaterial) -> Self {
        bytemuck::cast(value)
    }
}

impl From<MaterialRow> for crate::GpuMaterial {
    fn from(value: MaterialRow) -> Self {
        bytemuck::cast(value)
    }
}

impl MaterialRow {
    /// Construct a material row without involving a renderer or an asset
    /// registry. Texture values are SceneDB texture-store slot indices.
    pub fn new(
        base_color: [f32; 4],
        roughness: f32,
        metallic: f32,
        emissive: [f32; 3],
        emissive_strength: f32,
    ) -> Self {
        let missing = crate::GpuMaterial::NO_TEXTURE;
        Self {
            base_color,
            emissive: [emissive[0], emissive[1], emissive[2], emissive_strength],
            roughness_metallic: [roughness, metallic, 1.5, 0.5],
            tex_base_color: missing,
            tex_normal: missing,
            tex_roughness: missing,
            tex_emissive: missing,
            tex_occlusion: missing,
            workflow: 0,
            flags: 0,
            material_class: 0,
            class_params: [0.0; 4],
        }
    }

    /// A surface material with an opacity. `alpha < 1` selects the blended,
    /// transparent-only path (glass); the alpha rides in `base_color.w`.
    pub fn from_surface(
        base_color: [f32; 3],
        alpha: f32,
        roughness: f32,
        metallic: f32,
        emissive: [f32; 3],
        emissive_strength: f32,
    ) -> Self {
        let alpha = alpha.clamp(0.0, 1.0);
        let mut material = Self::new(
            [base_color[0], base_color[1], base_color[2], alpha],
            roughness,
            metallic,
            emissive,
            emissive_strength,
        );
        if alpha < 1.0 {
            material.flags |= crate::FLAG_ALPHA_BLEND | crate::FLAG_TRANSPARENT_ONLY;
        }
        material
    }
}


#[cfg(test)]
mod tests {
    #[test]
    fn the_row_is_the_gpu_material() {
        assert_eq!(std::mem::size_of::<super::MaterialRow>(), 96);
        assert_eq!(
            std::mem::size_of::<super::MaterialRow>(),
            std::mem::size_of::<crate::GpuMaterial>()
        );
    }
}
