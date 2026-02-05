use helio_features::{Feature, FeatureContext, ShaderInjection, ShaderInjectionPoint};

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct MaterialData {
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub _padding: [f32; 2],
}

impl Default for MaterialData {
    fn default() -> Self {
        Self {
            base_color: [1.0, 1.0, 1.0, 1.0],
            metallic: 0.0,
            roughness: 0.5,
            _padding: [0.0; 2],
        }
    }
}

pub struct BasicMaterials {
    enabled: bool,
    material_bindings: String,
    material_functions: String,
}

impl BasicMaterials {
    pub fn new() -> Self {
        let material_bindings = include_str!("../shaders/material_bindings.wgsl").to_string();
        let material_functions = include_str!("../shaders/material_functions.wgsl").to_string();

        Self {
            enabled: true,
            material_bindings,
            material_functions,
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
        // Materials will be bound per-object by the application
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn shader_injections(&self) -> Vec<ShaderInjection> {
        vec![
            // Inject material bindings
            ShaderInjection {
                point: ShaderInjectionPoint::FragmentPreamble,
                code: self.material_bindings.clone(),
                priority: -10, // Lower priority so it goes before other fragment code
            },
            // Inject material functions
            ShaderInjection {
                point: ShaderInjectionPoint::FragmentPreamble,
                code: self.material_functions.clone(),
                priority: -5,
            },
            // Apply material color with texture coordinates
            ShaderInjection {
                point: ShaderInjectionPoint::FragmentMain,
                code: "    final_color = apply_material_color(final_color, input.tex_coords);".to_string(),
                priority: -10,
            },
        ]
    }
}
