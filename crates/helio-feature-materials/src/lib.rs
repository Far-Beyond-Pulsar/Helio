use helio_features::{Feature, FeatureContext, ShaderInjection, ShaderInjectionPoint};

/// Material data structure for PBR-like materials.
/// Optimized memory layout: 32 bytes total (2 vec4 uniforms)
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct MaterialData {
    pub base_color: [f32; 4],      // 16 bytes (vec4)
    pub metallic: f32,              // 4 bytes
    pub roughness: f32,             // 4 bytes
    pub emissive_strength: f32,     // 4 bytes (repurposed from padding)
    pub ao: f32,                    // 4 bytes (ambient occlusion, repurposed from padding)
}

impl Default for MaterialData {
    fn default() -> Self {
        Self {
            base_color: [1.0, 1.0, 1.0, 1.0],
            metallic: 0.0,
            roughness: 0.5,
            emissive_strength: 0.0,
            ao: 1.0,
        }
    }
}

/// Basic materials feature with procedural texture generation.
///
/// Provides checkerboard pattern materials for visual interest.
/// In a real application, this would handle texture sampling and
/// material property management.
pub struct BasicMaterials {
    enabled: bool,
}

impl BasicMaterials {
    /// Create a new basic materials feature.
    pub fn new() -> Self {
        Self {
            enabled: true,
        }
    }
}

impl Default for BasicMaterials {
    fn default() -> Self {
        Self::new()
    }
}

impl Feature for BasicMaterials {
    fn name(&self) -> &str {
        "basic_materials"
    }

    fn init(&mut self, _context: &FeatureContext) {
        log::debug!("Basic materials feature initialized");
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn shader_injections(&self) -> Vec<ShaderInjection> {
        vec![
            // Material bindings (low priority - injected early)
            ShaderInjection::with_priority(
                ShaderInjectionPoint::FragmentPreamble,
                include_str!("../shaders/material_bindings.wgsl"),
                -10,
            ),
            // Material functions
            ShaderInjection::with_priority(
                ShaderInjectionPoint::FragmentPreamble,
                include_str!("../shaders/material_functions.wgsl"),
                -5,
            ),
            // Apply material early in fragment processing (now with camera position)
            ShaderInjection::with_priority(
                ShaderInjectionPoint::FragmentMain,
                "    final_color = apply_material_color(final_color, input.tex_coords, input.world_position, camera.position);",
                -10,
            ),
        ]
    }
    
    fn cleanup(&mut self, _context: &FeatureContext) {
        // No GPU resources to clean up
    }
}
