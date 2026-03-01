//! Feature trait definition
//!
//! This is the core interface that all rendering features implement.

use super::{FeatureContext, PrepareContext};
use crate::Result;
use std::collections::HashMap;

/// Shader specialization constant value
#[derive(Clone, Debug)]
pub enum ShaderDefine {
    Bool(bool),
    U32(u32),
    F32(f32),
}

/// Path to a shader module
pub type ShaderModulePath = String;

/// Feature trait - implemented by all rendering features
///
/// **Key differences from old system:**
/// - ❌ OLD: `shader_injections()` returns string fragments
/// - ✅ NEW: `shader_defines()` returns specialization constants
/// - ✅ NEW: `register()` adds passes to render graph
///
/// **Lifecycle:**
/// 1. `register()` - Called once during renderer initialization
/// 2. `prepare()` - Called every frame before rendering
/// 3. `on_state_change()` - Called when feature is enabled/disabled
/// 4. `cleanup()` - Called when renderer is destroyed
pub trait Feature: Send + Sync + AsAny {
    /// Unique name for this feature (lowercase snake_case)
    fn name(&self) -> &str;

    /// Register passes and resources with the render graph
    ///
    /// Called once during renderer initialization. Features should:
    /// - Create persistent GPU resources (textures, buffers)
    /// - Register render passes with the graph
    /// - Store resource IDs for later use
    ///
    /// # Example
    /// ```ignore
    /// fn register(&mut self, ctx: &mut FeatureContext) -> Result<()> {
    ///     // Create shadow atlas
    ///     self.shadow_atlas = Some(ctx.resources.create_texture("shadow_atlas", &desc)?);
    ///
    ///     // Register shadow pass
    ///     self.pass_id = Some(ctx.graph.add_pass(ShadowPass::new()));
    ///
    ///     Ok(())
    /// }
    /// ```
    fn register(&mut self, ctx: &mut FeatureContext) -> Result<()>;

    /// Update per-frame data
    ///
    /// Called every frame before rendering. Features should:
    /// - Update uniform buffers
    /// - Contribute to shared bind groups
    /// - Update animation state
    ///
    /// # Example
    /// ```ignore
    /// fn prepare(&mut self, ctx: &PrepareContext) -> Result<()> {
    ///     // Update light buffer
    ///     ctx.queue.write_buffer(&self.light_buffer, 0, &light_data);
    ///
    ///     // Contribute shadow atlas to lighting bind group
    ///     ctx.resources.contribute_to_lighting_group(
    ///         Some(&self.shadow_atlas_view),
    ///         None,
    ///     );
    ///
    ///     Ok(())
    /// }
    /// ```
    fn prepare(&mut self, ctx: &PrepareContext) -> Result<()>;

    /// Called when feature is enabled or disabled
    ///
    /// This allows features to respond to runtime toggling.
    /// The pipeline will automatically swap to a variant with/without
    /// this feature's specialization constants.
    fn on_state_change(&mut self, enabled: bool, ctx: &mut FeatureContext) -> Result<()> {
        let _ = (enabled, ctx);
        Ok(())
    }

    /// Get shader specialization constants
    ///
    /// Returns constants that are used to enable/disable code paths in shaders.
    /// These are applied at pipeline creation time, not shader compilation time.
    ///
    /// # Example
    /// ```ignore
    /// fn shader_defines(&self) -> HashMap<String, ShaderDefine> {
    ///     let mut defines = HashMap::new();
    ///     defines.insert("ENABLE_SHADOWS".into(), ShaderDefine::Bool(true));
    ///     defines.insert("MAX_LIGHTS".into(), ShaderDefine::U32(8));
    ///     defines
    /// }
    /// ```
    ///
    /// In the shader:
    /// ```wgsl
    /// override ENABLE_SHADOWS: bool = false;
    /// override MAX_LIGHTS: u32 = 8;
    ///
    /// if (ENABLE_SHADOWS) {
    ///     // Shadow sampling code
    /// }
    /// ```
    fn shader_defines(&self) -> HashMap<String, ShaderDefine> {
        HashMap::new()
    }

    /// Get shader modules needed by this feature
    ///
    /// Returns paths to shader files that should be available when
    /// compiling pipelines for this feature.
    fn shader_modules(&self) -> Vec<ShaderModulePath> {
        Vec::new()
    }

    /// Cleanup GPU resources
    ///
    /// Called when the feature is destroyed. Features should release
    /// all GPU resources they created.
    fn cleanup(&mut self, device: &wgpu::Device) {
        let _ = device;
    }

    /// Check if feature is currently enabled
    fn is_enabled(&self) -> bool {
        true
    }

    /// Enable or disable this feature
    fn set_enabled(&mut self, enabled: bool) {
        let _ = enabled;
    }
}

/// Helper trait for downcasting feature trait objects
pub trait AsAny {
    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

impl<T: Feature + 'static> AsAny for T {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
