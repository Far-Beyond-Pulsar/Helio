//! Feature registry for managing features

use super::{Feature, FeatureContext, PrepareContext, ShaderDefine};
use crate::{Error, Result};
use std::collections::HashMap;
use bitflags::bitflags;

bitflags! {
    /// Feature flags for pipeline variants
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct FeatureFlags: u32 {
        const SHADOWS = 1 << 0;
        const IBL = 1 << 1;
        const BLOOM = 1 << 2;
        const VOLUMETRIC_CLOUDS = 1 << 3;
        const PROCEDURAL_SKY = 1 << 4;
        const BILLBOARDS = 1 << 5;
    }
}

/// Registry for managing features
pub struct FeatureRegistry {
    features: HashMap<String, Box<dyn Feature>>,
    active_flags: FeatureFlags,
}

impl FeatureRegistry {
    pub fn new() -> Self {
        Self {
            features: HashMap::new(),
            active_flags: FeatureFlags::empty(),
        }
    }

    /// Create a builder for fluent API
    pub fn builder() -> FeatureRegistryBuilder {
        FeatureRegistryBuilder::new()
    }

    /// Register a feature
    pub fn register(&mut self, feature: Box<dyn Feature>) {
        let name = feature.name().to_string();
        self.features.insert(name, feature);
    }

    /// Enable a feature
    pub fn enable(&mut self, name: &str) -> Result<()> {
        let feature = self
            .features
            .get_mut(name)
            .ok_or_else(|| Error::Feature(format!("Feature '{}' not found", name)))?;

        feature.set_enabled(true);

        // Update active flags (simplified - in real implementation would map name to flag)
        // For now, just mark as changed

        Ok(())
    }

    /// Disable a feature
    pub fn disable(&mut self, name: &str) -> Result<()> {
        let feature = self
            .features
            .get_mut(name)
            .ok_or_else(|| Error::Feature(format!("Feature '{}' not found", name)))?;

        feature.set_enabled(false);

        Ok(())
    }

    /// Get active feature flags
    pub fn active_flags(&self) -> FeatureFlags {
        self.active_flags
    }

    /// Get shader defines from all features (enabled or not)
    /// Disabled features contribute their default (off) values so the shader
    /// always has every override constant it needs.
    pub fn collect_shader_defines(&self) -> HashMap<String, ShaderDefine> {
        let mut defines = HashMap::new();

        for feature in self.features.values() {
            defines.extend(feature.shader_defines());
        }

        defines
    }

    /// Initialize all features by calling `register()` on each
    pub fn register_all(&mut self, ctx: &mut FeatureContext) -> Result<()> {
        for feature in self.features.values_mut() {
            feature.register(ctx)?;
        }
        Ok(())
    }

    /// Prepare all enabled features
    pub fn prepare_all(&mut self, ctx: &PrepareContext) -> Result<()> {
        for feature in self.features.values_mut() {
            if feature.is_enabled() {
                feature.prepare(ctx)?;
            }
        }
        Ok(())
    }

    /// Get a feature by name
    pub fn get(&self, name: &str) -> Option<&dyn Feature> {
        self.features.get(name).map(|f| &**f)
    }

    /// Get a mutable reference to a specific feature by type
    pub fn get_typed_mut<T: Feature + 'static>(&mut self, name: &str) -> Option<&mut T> {
        self.features.get_mut(name)
            .and_then(|f| f.as_any_mut().downcast_mut::<T>())
    }

    // TODO: Fix lifetime issues with these methods
    // /// Get a mutable feature by name
    // pub fn get_mut<'a>(&'a mut self, name: &str) -> Option<&'a mut (dyn Feature + 'a)> {
    //     self.features.get_mut(name).map(|f| &mut **f)
    // }

    // /// Iterate over all features
    // pub fn iter(&self) -> impl Iterator<Item = &dyn Feature> {
    //     self.features.values().map(|f| &**f)
    // }

    // /// Iterate over all features mutably
    // pub fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut (dyn Feature + 'a)> + 'a {
    //     self.features.values_mut().map(|f| &mut **f)
    // }
}

/// Builder for FeatureRegistry
pub struct FeatureRegistryBuilder {
    features: Vec<Box<dyn Feature>>,
}

impl FeatureRegistryBuilder {
    pub fn new() -> Self {
        Self {
            features: Vec::new(),
        }
    }

    /// Add a feature to the registry
    pub fn with_feature(mut self, feature: impl Feature + 'static) -> Self {
        self.features.push(Box::new(feature));
        self
    }

    /// Build the registry
    pub fn build(self) -> FeatureRegistry {
        let mut registry = FeatureRegistry::new();

        for feature in self.features {
            registry.register(feature);
        }

        registry
    }
}

impl Default for FeatureRegistry {
    fn default() -> Self {
        Self::new()
    }
}
