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

/// Billboard rendering feature
/// 
/// This feature adds support for camera-facing billboards with PNG textures.
/// Billboards can be used for standalone scene objects or editor gizmos (lights, etc.)
pub struct BillboardFeature {
    enabled: bool,
    texture_manager: Option<Arc<TextureManager>>,
}

impl BillboardFeature {
    pub fn new() -> Self {
        Self {
            enabled: true,
            texture_manager: None,
        }
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

impl Feature for BillboardFeature {
    fn name(&self) -> &str {
        "billboards"
    }
    
    fn init(&mut self, _context: &FeatureContext) {
        // Billboards use their own separate render pipeline
        // No shader injection needed - they're rendered separately
    }
    
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    
    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
    
    fn shader_injections(&self) -> Vec<ShaderInjection> {
        // Billboards don't inject into the main shader pipeline
        // They use their own separate pipeline
        Vec::new()
    }
}

impl Default for BillboardFeature {
    fn default() -> Self {
        Self::new()
    }
}

