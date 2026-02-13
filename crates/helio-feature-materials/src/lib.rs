use helio_features::{Feature, FeatureContext, ShaderInjection, ShaderInjectionPoint};
use helio_features::{ExportedData, FeatureData};
use std::collections::HashMap;
use std::sync::Arc;

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

/// Exportable material properties for other features to consume.
///
/// This is the CPU-side representation of material data that can be
/// shared with other features (especially lighting) for PBR calculations.
#[derive(Debug, Clone)]
pub struct MaterialProperties {
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub emissive_strength: f32,
    pub ao: f32,
}

impl MaterialProperties {
    pub fn from_material_data(data: &MaterialData) -> Self {
        Self {
            base_color: data.base_color,
            metallic: data.metallic,
            roughness: data.roughness,
            emissive_strength: data.emissive_strength,
            ao: data.ao,
        }
    }
}

impl FeatureData for MaterialProperties {
    fn description(&self) -> &str {
        "Material PBR properties"
    }
}

/// Basic materials feature with procedural texture generation.
///
/// Provides checkerboard pattern materials for visual interest.
/// In a real application, this would handle texture sampling and
/// material property management.
pub struct BasicMaterials {
    enabled: bool,
    /// Current material properties (for export to other features)
    properties: MaterialData,
}

impl BasicMaterials {
    /// Create a new basic materials feature.
    pub fn new() -> Self {
        Self {
            enabled: true,
            properties: MaterialData::default(),
        }
    }

    /// Update material properties (for runtime adjustments).
    pub fn set_properties(&mut self, properties: MaterialData) {
        self.properties = properties;
    }

    /// Get current material properties.
    pub fn get_properties(&self) -> &MaterialData {
        &self.properties
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
            // Declare emissive variable early (so lighting can access it)
            ShaderInjection::with_priority(
                ShaderInjectionPoint::FragmentMain,
                "    var material_emissive = vec3<f32>(0.0, 0.0, 0.0);",
                -15,
            ),
            // Apply material early in fragment processing
            ShaderInjection::with_priority(
                ShaderInjectionPoint::FragmentMain,
                "    final_color = apply_material_color(final_color, input.tex_coords);",
                -10,
            ),
            // Calculate emissive based on world position
            ShaderInjection::with_priority(
                ShaderInjectionPoint::FragmentMain,
                "    material_emissive = get_emissive_color(input.world_position);",
                -9,
            ),
            // Add emissive AFTER all lighting/shadows (high priority)
            ShaderInjection::with_priority(
                ShaderInjectionPoint::FragmentColorCalculation,
                "    final_color = final_color + material_emissive; // Self-illumination",
                15, // Very high priority - run after shadows
            ),
        ]
    }
    
    fn cleanup(&mut self, _context: &FeatureContext) {
        // No GPU resources to clean up
    }

    fn export_data(&self) -> HashMap<String, ExportedData> {
        let mut exports = HashMap::new();
        
        // Export material properties as CPU data (zero-copy via Arc)
        let props = Arc::new(MaterialProperties::from_material_data(&self.properties));
        exports.insert("properties".to_string(), ExportedData::CpuData(props));
        
        exports
    }
}
