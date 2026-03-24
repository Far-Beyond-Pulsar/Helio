//! Texture resource management for the scene.
//!
//! Textures are stored in a sparse pool with reference counting. Multiple materials
//! can reference the same texture. Textures cannot be removed while materials are using them.
//!
//! # Capacity Limits
//!
//! The scene supports a maximum of [`MAX_TEXTURES`](crate::material::MAX_TEXTURES) (16384)
//! concurrent textures due to bindless array limits.

use std::sync::Arc;
use wgpu::util::DeviceExt;

use crate::handles::TextureId;
use crate::material::{TextureUpload, MAX_TEXTURES};

use super::super::errors::{invalid, Result, SceneError};
use super::super::types::TextureRecord;

impl super::super::Scene {
    /// Insert a texture into the scene's texture pool.
    ///
    /// Uploads texture data to GPU memory and creates a texture view and sampler.
    /// Returns a handle that can be referenced by materials.
    ///
    /// # Parameters
    /// - `texture`: Texture upload data containing:
    ///   - Image data (raw RGBA bytes)
    ///   - Width and height
    ///   - Format (RGBA8, SRGBA8, etc.)
    ///   - Sampler settings (filter modes, address modes)
    ///
    /// # Errors
    /// - [`SceneError::TextureCapacityExceeded`] if the texture pool is at capacity (16384 textures)
    ///
    /// # Returns
    /// A [`TextureId`] handle that can be used with material texture slots.
    ///
    /// # Performance
    /// - CPU cost: O(1) handle allocation
    /// - GPU cost: Uploads texture data, creates view and sampler
    /// - Memory: Texture data is stored in GPU-local memory
    ///
    /// # Example
    /// ```ignore
    /// let texture_id = scene.insert_texture(TextureUpload {
    ///     label: Some("Albedo Map".into()),
    ///     width: 1024,
    ///     height: 1024,
    ///     format: wgpu::TextureFormat::Rgba8UnormSrgb,
    ///     data: image_bytes,
    ///     sampler: SamplerDescriptor {
    ///         mag_filter: wgpu::FilterMode::Linear,
    ///         min_filter: wgpu::FilterMode::Linear,
    ///         mipmap_filter: wgpu::FilterMode::Linear,
    ///         ..Default::default()
    ///     },
    /// })?;
    /// ```
    pub fn insert_texture(&mut self, texture: TextureUpload) -> Result<TextureId> {
        if !self.textures.has_free_slot() && self.textures.slot_len() >= MAX_TEXTURES {
            return Err(SceneError::TextureCapacityExceeded);
        }

        helio_v3::upload::record_upload_bytes(texture.data.len() as u64);
        let gpu_texture = self.gpu_scene.device.create_texture_with_data(
            &self.gpu_scene.queue,
            &wgpu::TextureDescriptor {
                label: texture.label.as_deref(),
                size: wgpu::Extent3d {
                    width: texture.width,
                    height: texture.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: texture.format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &texture.data,
        );
        let view = gpu_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = self
            .gpu_scene
            .device
            .create_sampler(&wgpu::SamplerDescriptor {
                label: texture.label.as_deref(),
                address_mode_u: texture.sampler.address_mode_u,
                address_mode_v: texture.sampler.address_mode_v,
                address_mode_w: texture.sampler.address_mode_w,
                mag_filter: texture.sampler.mag_filter,
                min_filter: texture.sampler.min_filter,
                mipmap_filter: texture.sampler.mipmap_filter,
                ..Default::default()
            });
        let (id, _, _) = self.textures.insert(TextureRecord {
            _texture: gpu_texture,
            view,
            sampler,
            ref_count: 0,
        });
        self.texture_binding_version = self.texture_binding_version.wrapping_add(1);
        Ok(id)
    }

    /// Remove a texture from the scene's texture pool.
    ///
    /// # Errors
    /// - [`SceneError::InvalidHandle`] if the texture ID is invalid
    /// - [`SceneError::ResourceInUse`] if any materials are still using this texture
    ///
    /// # Returns
    /// `Ok(())` if the texture was successfully removed.
    ///
    /// # Side Effects
    /// Increments the texture binding version, which signals the renderer to rebuild
    /// the bindless texture array.
    ///
    /// # Example
    /// ```ignore
    /// // Remove all materials using the texture first
    /// scene.remove_material(material_id)?;
    ///
    /// // Now the texture can be removed
    /// scene.remove_texture(texture_id)?;
    /// ```
    pub fn remove_texture(&mut self, id: TextureId) -> Result<()> {
        let Some(texture) = self.textures.get(id) else {
            return Err(invalid("texture"));
        };
        if texture.ref_count != 0 {
            return Err(SceneError::ResourceInUse {
                resource: "texture",
            });
        }
        self.textures.remove(id).ok_or_else(|| invalid("texture"))?;
        self.texture_binding_version = self.texture_binding_version.wrapping_add(1);
        Ok(())
    }

    /// Get the current texture binding version.
    ///
    /// This version number increments whenever textures are added or removed.
    /// The renderer uses it to detect when the bindless texture array needs
    /// to be rebuilt.
    ///
    /// # Returns
    /// A monotonically increasing version number (wraps on overflow).
    pub fn texture_binding_version(&self) -> u64 {
        self.texture_binding_version
    }

    /// Get the texture view for a given slot index.
    ///
    /// Returns the placeholder white texture view if the slot is invalid or empty.
    /// Used internally by the renderer to build bindless texture arrays.
    ///
    /// # Parameters
    /// - `slot`: Texture slot index (not a TextureId)
    ///
    /// # Returns
    /// A reference to the texture view, or the placeholder view if slot is invalid.
    pub fn texture_view_for_slot(&self, slot: usize) -> &wgpu::TextureView {
        self.textures
            .get_by_slot(slot)
            .map(|texture| &texture.view)
            .unwrap_or(&self.placeholder_view)
    }

    /// Get the sampler for a given slot index.
    ///
    /// Returns the placeholder sampler if the slot is invalid or empty.
    /// Used internally by the renderer to build bindless sampler arrays.
    ///
    /// # Parameters
    /// - `slot`: Texture slot index (not a TextureId)
    ///
    /// # Returns
    /// A reference to the sampler, or the placeholder sampler if slot is invalid.
    pub fn texture_sampler_for_slot(&self, slot: usize) -> &wgpu::Sampler {
        self.textures
            .get_by_slot(slot)
            .map(|texture| &texture.sampler)
            .unwrap_or(&self.placeholder_sampler)
    }
}

