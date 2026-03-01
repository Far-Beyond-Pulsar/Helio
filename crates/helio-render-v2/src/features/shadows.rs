//! Shadow mapping feature

use super::{FeatureContext, PrepareContext};
use crate::features::{Feature, ShaderDefine};
use crate::passes::ShadowPass;
use crate::Result;
use std::collections::HashMap;
use std::sync::Arc;

/// Default shadow atlas resolution per shadow map tile
pub const DEFAULT_SHADOW_ATLAS_SIZE: u32 = 2048;
/// Default maximum number of shadow-casting lights
pub const DEFAULT_MAX_SHADOW_LIGHTS: u32 = 8;

/// Shadow mapping feature
///
/// Creates a 2D-array depth texture (shadow atlas) with one layer per
/// shadow-casting light, then registers a `ShadowPass` that fills it each
/// frame.  The geometry shader samples the atlas for PCF shadow lookups when
/// `ENABLE_SHADOWS` is set.
pub struct ShadowsFeature {
    enabled: bool,
    atlas_size: u32,
    max_shadow_lights: u32,
    shadow_atlas: Option<wgpu::Texture>,
    shadow_atlas_view: Option<Arc<wgpu::TextureView>>,
    shadow_sampler: Option<Arc<wgpu::Sampler>>,
}

impl ShadowsFeature {
    pub fn new() -> Self {
        Self {
            enabled: true,
            atlas_size: DEFAULT_SHADOW_ATLAS_SIZE,
            max_shadow_lights: DEFAULT_MAX_SHADOW_LIGHTS,
            shadow_atlas: None,
            shadow_atlas_view: None,
            shadow_sampler: None,
        }
    }

    pub fn with_atlas_size(mut self, size: u32) -> Self {
        self.atlas_size = size;
        self
    }

    pub fn with_max_lights(mut self, max: u32) -> Self {
        self.max_shadow_lights = max;
        self
    }

    /// Full-atlas texture view (D2Array) – bind to lighting group binding 1
    pub fn shadow_atlas_view(&self) -> Option<&Arc<wgpu::TextureView>> {
        self.shadow_atlas_view.as_ref()
    }

    /// Comparison sampler for PCF – bind to lighting group binding 2
    pub fn shadow_sampler(&self) -> Option<&Arc<wgpu::Sampler>> {
        self.shadow_sampler.as_ref()
    }
}

impl Default for ShadowsFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl Feature for ShadowsFeature {
    fn name(&self) -> &str {
        "shadows"
    }

    fn register(&mut self, ctx: &mut FeatureContext) -> Result<()> {
        // Shadow atlas: one layer per shadow-casting light
        let atlas = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Shadow Atlas"),
            size: wgpu::Extent3d {
                width: self.atlas_size,
                height: self.atlas_size,
                depth_or_array_layers: self.max_shadow_lights,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        // Full-atlas view used by the geometry shader
        let atlas_view = atlas.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Shadow Atlas View"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        // PCF comparison sampler
        let sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Shadow Comparison Sampler"),
            compare: Some(wgpu::CompareFunction::LessEqual),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Per-layer views so the shadow pass can render into individual layers
        let layer_views: Vec<Arc<wgpu::TextureView>> = (0..self.max_shadow_lights)
            .map(|layer| {
                Arc::new(atlas.create_view(&wgpu::TextureViewDescriptor {
                    label: Some(&format!("Shadow Atlas Layer {layer}")),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                }))
            })
            .collect();

        self.shadow_atlas_view = Some(Arc::new(atlas_view));
        self.shadow_sampler = Some(Arc::new(sampler));
        self.shadow_atlas = Some(atlas);

        ctx.shadow_atlas_view = self.shadow_atlas_view.clone();
        ctx.shadow_sampler = self.shadow_sampler.clone();

        ctx.graph.add_pass(ShadowPass::new(layer_views));

        log::info!(
            "Shadows feature registered: {}×{} atlas, {} max lights",
            self.atlas_size,
            self.atlas_size,
            self.max_shadow_lights
        );
        Ok(())
    }

    fn prepare(&mut self, _ctx: &PrepareContext) -> Result<()> {
        Ok(())
    }

    fn shader_defines(&self) -> HashMap<String, ShaderDefine> {
        let mut defines = HashMap::new();
        defines.insert("ENABLE_SHADOWS".into(), ShaderDefine::Bool(self.enabled));
        defines.insert(
            "MAX_SHADOW_LIGHTS".into(),
            ShaderDefine::U32(if self.enabled { self.max_shadow_lights } else { 0 }),
        );
        defines
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn cleanup(&mut self, _device: &wgpu::Device) {
        self.shadow_atlas = None;
        self.shadow_atlas_view = None;
        self.shadow_sampler = None;
    }
}
