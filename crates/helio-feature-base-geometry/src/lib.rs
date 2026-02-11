use helio_core::{TextureId, TextureManager};
use helio_features::{Feature, FeatureContext, ShaderInjection};
use std::sync::Arc;

/// Blending mode for billboard rendering
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    /// No blending, opaque rendering
    Opaque,
    /// Alpha blending for transparency
    Transparent,
}

/// Single billboard instance in the scene
#[derive(Clone)]
pub struct BillboardData {
    /// World space position
    pub position: [f32; 3],
    /// Scale (width, height) in world units
    pub scale: [f32; 2],
    /// Texture to display
    pub texture: TextureId,
    /// Blending mode
    pub blend_mode: BlendMode,
}

impl BillboardData {
    pub fn new(position: [f32; 3], scale: [f32; 2], texture: TextureId) -> Self {
        Self {
            position,
            scale,
            texture,
            blend_mode: BlendMode::Transparent,
        }
    }
    
    pub fn with_blend_mode(mut self, mode: BlendMode) -> Self {
        self.blend_mode = mode;
        self
    }
}

/// Base geometry feature providing fundamental geometry rendering.
///
/// This feature provides the base shader template that other features
/// inject into. It handles vertex transformation, normal calculation,
/// and basic output setup. Also includes billboard rendering for gizmos.
pub struct BaseGeometry {
    enabled: bool,
    shader_template: &'static str,
    billboard_shader: &'static str,
    texture_manager: Option<Arc<TextureManager>>,
}

impl BaseGeometry {
    /// Create a new base geometry feature.
    pub fn new() -> Self {
        Self {
            enabled: true,
            shader_template: include_str!("../shaders/base_geometry.wgsl"),
            billboard_shader: include_str!("../shaders/billboard.wgsl"),
            texture_manager: None,
        }
    }

    /// Get the shader template for use with the feature renderer.
    ///
    /// This template contains injection markers where other features
    /// can insert their shader code.
    pub fn shader_template(&self) -> &str {
        self.shader_template
    }
    
    /// Get the billboard shader code.
    pub fn billboard_shader(&self) -> &str {
        self.billboard_shader
    }
    
    /// Set the texture manager for loading billboard textures
    pub fn set_texture_manager(&mut self, manager: Arc<TextureManager>) {
        self.texture_manager = Some(manager);
    }
    
    /// Get a reference to the texture manager
    pub fn texture_manager(&self) -> Option<&Arc<TextureManager>> {
        self.texture_manager.as_ref()
    }
}

impl Default for BaseGeometry {
    fn default() -> Self {
        Self::new()
    }
}

impl Feature for BaseGeometry {
    fn name(&self) -> &str {
        "base_geometry"
    }

    fn init(&mut self, _context: &FeatureContext) {
        log::debug!("Base geometry feature initialized");
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn shader_injections(&self) -> Vec<ShaderInjection> {
        // Base geometry provides the template, not injections
        Vec::new()
    }
    
    fn cleanup(&mut self, _context: &FeatureContext) {
        // No GPU resources to clean up
    }
}
