//! Material resource management for the scene.
//!
//! Materials define surface appearance (color, roughness, metallic, etc.) and can
//! reference textures. Materials are stored in a sparse pool with reference counting.
//! Multiple objects can reference the same material.

use bytemuck::Zeroable;
use helio_v3::GpuMaterial;

use crate::handles::MaterialId;
use crate::material::{MaterialAsset, MaterialTextures};

use super::super::errors::{invalid, Result, SceneError};
use super::super::helpers::{each_material_texture_ref, gpu_material_textures};
use super::super::types::MaterialRecord;

/// Tombstone material used when a material slot is freed (preserves slot stability).
fn tombstone_material() -> GpuMaterial {
    GpuMaterial {
        tex_base_color: GpuMaterial::NO_TEXTURE,
        tex_normal: GpuMaterial::NO_TEXTURE,
        tex_roughness: GpuMaterial::NO_TEXTURE,
        tex_emissive: GpuMaterial::NO_TEXTURE,
        tex_occlusion: GpuMaterial::NO_TEXTURE,
        ..GpuMaterial::zeroed()
    }
}

/// Tombstone material textures used when a material slot is freed.
fn tombstone_material_textures() -> crate::material::GpuMaterialTextures {
    crate::material::GpuMaterialTextures::missing()
}

impl super::super::Scene {
    /// Insert a material into the scene's material pool.
    ///
    /// This is a convenience method that converts a [`GpuMaterial`] into a [`MaterialAsset`]
    /// and calls [`insert_material_asset`](Self::insert_material_asset).
    ///
    /// # Parameters
    /// - `material`: GPU material parameters (base color, roughness, metallic, etc.)
    ///
    /// # Returns
    /// A [`MaterialId`] handle that can be used with [`insert_object`](crate::Scene::insert_object).
    ///
    /// # Panics
    /// Never panics (plain GPU materials have no texture references to validate).
    ///
    /// # Example
    /// ```ignore
    /// let material_id = scene.insert_material(GpuMaterial {
    ///     base_color: [1.0, 0.0, 0.0, 1.0], // Red
    ///     roughness: 0.5,
    ///     metallic: 0.0,
    ///     ..Default::default()
    /// });
    /// ```
    pub fn insert_material(&mut self, material: GpuMaterial) -> MaterialId {
        self.insert_material_asset(material.into())
            .expect("plain GPU materials must insert without texture validation failures")
    }

    /// Insert a material asset (with texture references) into the scene's material pool.
    ///
    /// Validates all texture references, increments texture reference counts, and uploads
    /// material data to GPU storage buffers.
    ///
    /// # Parameters
    /// - `material`: Material asset containing:
    ///   - GPU material parameters (colors, roughness, metallic, etc.)
    ///   - Texture references (base color, normal, roughness/metallic, etc.)
    ///
    /// # Errors
    /// - [`SceneError::InvalidHandle`] if any texture reference is invalid
    ///
    /// # Returns
    /// A [`MaterialId`] handle that can be used with objects.
    ///
    /// # Performance
    /// - CPU cost: O(1) handle allocation + O(textures) validation
    /// - GPU cost: Uploads material data to storage buffers (or reuses existing slot)
    /// - Memory: Materials are stored in GPU-local storage buffers
    ///
    /// # Slot Reuse
    /// If a material with the same parameters was previously inserted and removed,
    /// the slot may be reused to maintain slot stability for GPU indexing.
    ///
    /// # Example
    /// ```ignore
    /// let material_id = scene.insert_material_asset(MaterialAsset {
    ///     gpu: GpuMaterial {
    ///         base_color: [1.0, 1.0, 1.0, 1.0],
    ///         roughness: 0.8,
    ///         metallic: 0.0,
    ///         ..Default::default()
    ///     },
    ///     textures: MaterialTextures {
    ///         base_color: Some(MaterialTextureRef {
    ///             texture: albedo_texture_id,
    ///             uv_channel: 0,
    ///             transform: TextureTransform::default(),
    ///         }),
    ///         ..Default::default()
    ///     },
    /// })?;
    /// ```
    pub fn insert_material_asset(&mut self, material: MaterialAsset) -> Result<MaterialId> {
        self.validate_material_textures(&material.textures)?;
        self.bump_texture_refs(&material.textures, 1)?;

        let gpu_textures = gpu_material_textures(&material.textures);
        let (id, slot, is_new) = self.materials.insert(MaterialRecord {
            gpu: material.gpu,
            textures: material.textures,
            ref_count: 0,
        });
        if is_new {
            let pushed = self.gpu_scene.materials.push(material.gpu);
            debug_assert_eq!(pushed, slot);
            let pushed = self.material_textures.push(gpu_textures);
            debug_assert_eq!(pushed, slot);
        } else {
            let updated_material = self.gpu_scene.materials.update(slot, material.gpu);
            let updated_textures = self.material_textures.update(slot, gpu_textures);
            debug_assert!(updated_material && updated_textures);
        }
        Ok(id)
    }

    /// Update a material's GPU parameters (without changing texture references).
    ///
    /// This is useful for animating material properties (e.g., emissive color) without
    /// triggering texture reference count changes.
    ///
    /// # Parameters
    /// - `id`: Material handle
    /// - `material`: New GPU material parameters
    ///
    /// # Errors
    /// - [`SceneError::InvalidHandle`] if the material ID is invalid
    ///
    /// # Returns
    /// `Ok(())` if the material was successfully updated.
    ///
    /// # Performance
    /// - CPU cost: O(1)
    /// - GPU cost: Updates material storage buffer slot
    ///
    /// # Example
    /// ```ignore
    /// // Animate emissive color
    /// let mut material = scene.get_material(material_id)?;
    /// material.emissive = [1.0, 0.5, 0.0, 1.0]; // Orange glow
    /// scene.update_material(material_id, material)?;
    /// ```
    pub fn update_material(&mut self, id: MaterialId, material: GpuMaterial) -> Result<()> {
        let Some((slot, record)) = self.materials.get_mut_with_slot(id) else {
            return Err(invalid("material"));
        };
        record.gpu = material;
        let updated = self.gpu_scene.materials.update(slot, material);
        debug_assert!(updated);
        Ok(())
    }

    /// Update a material asset (including texture references).
    ///
    /// Validates new texture references, updates reference counts for both old and new
    /// textures, and uploads updated data to GPU storage buffers.
    ///
    /// # Parameters
    /// - `id`: Material handle
    /// - `material`: New material asset with updated GPU parameters and/or texture references
    ///
    /// # Errors
    /// - [`SceneError::InvalidHandle`] if the material ID or any texture reference is invalid
    ///
    /// # Returns
    /// `Ok(())` if the material was successfully updated.
    ///
    /// # Performance
    /// - CPU cost: O(1) + O(textures) validation and ref count updates
    /// - GPU cost: Updates material and texture storage buffer slots
    ///
    /// # Example
    /// ```ignore
    /// // Swap base color texture
    /// let mut material = scene.get_material_asset(material_id)?;
    /// material.textures.base_color = Some(MaterialTextureRef {
    ///     texture: new_albedo_texture_id,
    ///     ..material.textures.base_color.unwrap()
    /// });
    /// scene.update_material_asset(material_id, material)?;
    /// ```
    pub fn update_material_asset(&mut self, id: MaterialId, material: MaterialAsset) -> Result<()> {
        self.validate_material_textures(&material.textures)?;
        let Some(old_textures) = self.materials.get(id).map(|record| record.textures.clone())
        else {
            return Err(invalid("material"));
        };
        self.bump_texture_refs(&material.textures, 1)?;
        self.bump_texture_refs(&old_textures, -1)?;

        let Some((slot, record)) = self.materials.get_mut_with_slot(id) else {
            return Err(invalid("material"));
        };
        record.gpu = material.gpu;
        record.textures = material.textures.clone();

        let updated_material = self.gpu_scene.materials.update(slot, material.gpu);
        let updated_textures = self
            .material_textures
            .update(slot, gpu_material_textures(&material.textures));
        debug_assert!(updated_material && updated_textures);
        Ok(())
    }

    /// Remove a material from the scene's material pool.
    ///
    /// Decrements reference counts for all referenced textures and writes a tombstone
    /// value to the material's GPU slot to preserve slot stability.
    ///
    /// # Errors
    /// - [`SceneError::InvalidHandle`] if the material ID is invalid
    /// - [`SceneError::ResourceInUse`] if any objects are still using this material
    ///
    /// # Returns
    /// `Ok(())` if the material was successfully removed.
    ///
    /// # Example
    /// ```ignore
    /// // Remove all objects using the material first
    /// for obj_id in objects_using_material {
    ///     scene.remove_object(obj_id)?;
    /// }
    ///
    /// // Now the material can be removed
    /// scene.remove_material(material_id)?;
    /// ```
    pub fn remove_material(&mut self, id: MaterialId) -> Result<()> {
        let Some(record) = self.materials.get(id) else {
            return Err(invalid("material"));
        };
        if record.ref_count != 0 {
            return Err(SceneError::ResourceInUse {
                resource: "material",
            });
        }
        let (slot, removed) = self
            .materials
            .remove(id)
            .ok_or_else(|| invalid("material"))?;
        self.bump_texture_refs(&removed.textures, -1)?;
        let updated_material = self.gpu_scene.materials.update(slot, tombstone_material());
        let updated_textures = self
            .material_textures
            .update(slot, tombstone_material_textures());
        debug_assert!(updated_material && updated_textures);
        Ok(())
    }

    /// Get read-only access to the material texture storage buffer.
    ///
    /// Returns the GPU buffer containing material texture descriptors (texture indices,
    /// UV channels, transforms). Used by the renderer to bind the material texture buffer.
    ///
    /// # Returns
    /// A reference to the GPU buffer containing [`GpuMaterialTextures`] structs.
    pub fn material_texture_buffer(&self) -> &wgpu::Buffer {
        self.material_textures.buffer()
    }

    // ── Internal helper methods ────────────────────────────────────────────────

    /// Validate that all texture references in a material exist.
    ///
    /// Returns an error if any texture ID is invalid.
    pub(in crate::scene) fn validate_material_textures(
        &self,
        textures: &MaterialTextures,
    ) -> Result<()> {
        let mut validation = Ok(());
        each_material_texture_ref(textures, |texture| {
            if validation.is_err() {
                return;
            }
            if self.textures.get(texture.texture).is_none() {
                validation = Err(invalid("texture"));
            }
        });
        validation
    }

    /// Increment or decrement reference counts for all textures in a material.
    ///
    /// Used when materials are inserted, updated, or removed to track texture usage.
    pub(in crate::scene) fn bump_texture_refs(
        &mut self,
        textures: &MaterialTextures,
        delta: i32,
    ) -> Result<()> {
        for texture in [
            textures.base_color,
            textures.normal,
            textures.roughness_metallic,
            textures.emissive,
            textures.occlusion,
            textures.specular_color,
            textures.specular_weight,
        ]
        .into_iter()
        .flatten()
        {
            let (_, record) = self
                .textures
                .get_mut_with_slot(texture.texture)
                .ok_or_else(|| invalid("texture"))?;
            if delta >= 0 {
                record.ref_count = record.ref_count.saturating_add(delta as u32);
            } else {
                record.ref_count = record.ref_count.saturating_sub((-delta) as u32);
            }
        }
        Ok(())
    }
}
