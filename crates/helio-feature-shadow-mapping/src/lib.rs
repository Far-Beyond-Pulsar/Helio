use blade_graphics as gpu;
use helio_features::{Feature, FeatureContext, ShaderInjection, ShaderInjectionPoint};
use std::sync::Arc;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadowUniforms {
    pub light_view_proj: [[f32; 4]; 4],
    pub light_direction: [f32; 3],
    pub shadow_bias: f32,
}

pub struct ShadowMapping {
    enabled: bool,
    shadow_map: Option<gpu::Texture>,
    shadow_map_view: Option<gpu::TextureView>,
    shadow_map_size: u32,
    light_direction: glam::Vec3,
    context: Option<Arc<gpu::Context>>,
}

impl ShadowMapping {
    pub fn new() -> Self {
        Self {
            enabled: true,
            shadow_map: None,
            shadow_map_view: None,
            shadow_map_size: 2048,
            light_direction: glam::Vec3::new(0.5, -1.0, 0.3).normalize(),
            context: None,
        }
    }

    pub fn with_size(mut self, size: u32) -> Self {
        self.shadow_map_size = size;
        self
    }

    pub fn get_light_view_proj(&self) -> glam::Mat4 {
        // Orthographic projection for directional light
        let light_pos = -self.light_direction * 20.0;
        let view = glam::Mat4::look_at_rh(
            light_pos,
            glam::Vec3::ZERO,
            glam::Vec3::Y,
        );

        // Orthographic bounds to cover the scene
        let projection = glam::Mat4::orthographic_rh(
            -15.0, 15.0,  // left, right
            -15.0, 15.0,  // bottom, top
            0.1, 50.0,    // near, far
        );

        projection * view
    }
}

impl Default for ShadowMapping {
    fn default() -> Self {
        Self::new()
    }
}

impl Feature for ShadowMapping {
    fn name(&self) -> &str {
        "shadow_mapping"
    }

    fn init(&mut self, context: &FeatureContext) {
        self.context = Some(context.gpu.clone());

        // Create shadow map texture
        let shadow_map = context.gpu.create_texture(gpu::TextureDesc {
            name: "shadow_map",
            format: gpu::TextureFormat::Depth32Float,
            size: gpu::Extent {
                width: self.shadow_map_size,
                height: self.shadow_map_size,
                depth: 1,
            },
            dimension: gpu::TextureDimension::D2,
            array_layer_count: 1,
            mip_level_count: 1,
            usage: gpu::TextureUsage::TARGET | gpu::TextureUsage::RESOURCE,
            sample_count: 1,
            external: None,
        });

        let shadow_map_view = context.gpu.create_texture_view(
            shadow_map,
            gpu::TextureViewDesc {
                name: "shadow_map_view",
                format: gpu::TextureFormat::Depth32Float,
                dimension: gpu::ViewDimension::D2,
                subresources: &Default::default(),
            },
        );

        self.shadow_map = Some(shadow_map);
        self.shadow_map_view = Some(shadow_map_view);

        log::info!("Shadow mapping initialized with {}x{} shadow map",
                   self.shadow_map_size, self.shadow_map_size);
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn shader_injections(&self) -> Vec<ShaderInjection> {
        let shadow_functions = include_str!("../shaders/shadow_functions.wgsl").to_string();

        vec![
            ShaderInjection {
                point: ShaderInjectionPoint::FragmentPreamble,
                code: shadow_functions,
                priority: 5,
            },
            ShaderInjection {
                point: ShaderInjectionPoint::FragmentColorCalculation,
                code: "    final_color = apply_shadow(final_color, input.world_position, input.world_normal);".to_string(),
                priority: 10,
            },
        ]
    }

    fn pre_render_pass(&mut self, encoder: &mut gpu::CommandEncoder, context: &FeatureContext) {
        // Shadow pass will be implemented here
        // For now, we'll just clear the shadow map
        if let Some(shadow_view) = self.shadow_map_view {
            if let mut _pass = encoder.render(
                "shadow_pass",
                gpu::RenderTargetSet {
                    colors: &[],
                    depth_stencil: Some(gpu::RenderTarget {
                        view: shadow_view,
                        init_op: gpu::InitOp::Clear(gpu::TextureColor::White),
                        finish_op: gpu::FinishOp::Store,
                    }),
                },
            ) {
                // Shadow rendering will be implemented in the next iteration
                // This requires access to the scene geometry
            }
        }
    }
}
