/// Pipeline cache with variant system for hot-swapping features

use super::{PipelineVariant, PipelineDescriptor};
use crate::features::{FeatureFlags, ShaderDefine};
use crate::resources::BindGroupLayouts;
use crate::{Result, Error};
use std::collections::HashMap;
use std::sync::Arc;

/// Key for pipeline cache lookup
#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub struct PipelineKey {
    pub shader_id: String,
    pub features: FeatureFlags,
    pub variant: PipelineVariant,
}

/// Pipeline cache manages compiled pipelines with instant feature hot-swapping
pub struct PipelineCache {
    device: Arc<wgpu::Device>,
    cache: HashMap<PipelineKey, Arc<wgpu::RenderPipeline>>,
    shader_modules: HashMap<String, Arc<wgpu::ShaderModule>>,
    active_features: FeatureFlags,
    bind_group_layouts: Arc<BindGroupLayouts>,
    surface_format: wgpu::TextureFormat,
}

impl PipelineCache {
    pub fn new(device: Arc<wgpu::Device>, layouts: Arc<BindGroupLayouts>, surface_format: wgpu::TextureFormat) -> Self {
        Self {
            device,
            cache: HashMap::new(),
            shader_modules: HashMap::new(),
            active_features: FeatureFlags::empty(),
            bind_group_layouts: layouts,
            surface_format,
        }
    }

    /// Get or create a pipeline variant
    pub fn get_or_create(
        &mut self,
        shader_source: &str,
        shader_id: String,
        defines: &HashMap<String, ShaderDefine>,
        variant: PipelineVariant,
    ) -> Result<Arc<wgpu::RenderPipeline>> {
        let key = PipelineKey {
            shader_id: shader_id.clone(),
            features: self.active_features,
            variant,
        };

        // Return cached pipeline if it exists
        if let Some(pipeline) = self.cache.get(&key) {
            log::trace!("Using cached pipeline: {:?}", key);
            return Ok(pipeline.clone());
        }

        log::info!("Creating new pipeline variant: {:?}", key);

        // Apply shader defines to source
        let processed_source = self.apply_defines(shader_source, defines);

        // Create shader module
        let shader_module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&shader_id),
            source: wgpu::ShaderSource::Wgsl(processed_source.into()),
        });

        let shader_module = Arc::new(shader_module);
        self.shader_modules.insert(shader_id.clone(), shader_module.clone());

        // Create pipeline layout
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{}_layout", shader_id)),
            bind_group_layouts: &[
                &self.bind_group_layouts.global,
                &self.bind_group_layouts.material,
                &self.bind_group_layouts.lighting,
            ],
            push_constant_ranges: &[],
        });

        // Create pipeline based on variant
        let pipeline = match variant {
            PipelineVariant::Forward => {
                self.create_forward_pipeline(&pipeline_layout, &shader_module)?
            }
            PipelineVariant::DepthOnly => {
                self.create_depth_only_pipeline(&pipeline_layout, &shader_module)?
            }
            PipelineVariant::Deferred => {
                return Err(Error::Pipeline("Deferred rendering not implemented yet".into()));
            }
        };

        let pipeline = Arc::new(pipeline);
        self.cache.insert(key, pipeline.clone());

        Ok(pipeline)
    }

    fn create_forward_pipeline(
        &self,
        layout: &wgpu::PipelineLayout,
        shader: &wgpu::ShaderModule,
    ) -> Result<wgpu::RenderPipeline> {
        Ok(self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Forward Pipeline"),
            layout: Some(layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: "vs_main",
                buffers: &[
                    // Vertex buffer layout matching PackedVertex from helio-core
                    wgpu::VertexBufferLayout {
                        array_stride: 32, // 3*4 + 4 + 2*4 + 4 + 4 = 32 bytes
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            // position: vec3<f32>
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 0,
                                shader_location: 0,
                            },
                            // bitangent_sign: f32
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32,
                                offset: 12,
                                shader_location: 1,
                            },
                            // tex_coords: vec2<f32>
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 16,
                                shader_location: 2,
                            },
                            // normal: u32 (packed)
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Uint32,
                                offset: 24,
                                shader_location: 3,
                            },
                            // tangent: u32 (packed)
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Uint32,
                                offset: 28,
                                shader_location: 4,
                            },
                        ],
                    },
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        }))
    }

    fn create_depth_only_pipeline(
        &self,
        layout: &wgpu::PipelineLayout,
        shader: &wgpu::ShaderModule,
    ) -> Result<wgpu::RenderPipeline> {
        Ok(self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Depth Only Pipeline"),
            layout: Some(layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: "vs_main",
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: 32,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 0,
                                shader_location: 0,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32,
                                offset: 12,
                                shader_location: 1,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 16,
                                shader_location: 2,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Uint32,
                                offset: 24,
                                shader_location: 3,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Uint32,
                                offset: 28,
                                shader_location: 4,
                            },
                        ],
                    },
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: None, // Depth-only, no fragment shader
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        }))
    }

    /// Apply shader defines to source code
    fn apply_defines(&self, source: &str, defines: &HashMap<String, ShaderDefine>) -> String {
        let mut result = String::new();

        // Add override declarations at the top
        for (name, value) in defines {
            match value {
                ShaderDefine::Bool(b) => {
                    result.push_str(&format!("override {}: bool = {};\n", name, b));
                }
                ShaderDefine::U32(u) => {
                    result.push_str(&format!("override {}: u32 = {}u;\n", name, u));
                }
                ShaderDefine::F32(f) => {
                    result.push_str(&format!("override {}: f32 = {};\n", name, f));
                }
            }
        }

        result.push_str(source);
        result
    }

    /// Set active features (hot-swap pipelines)
    pub fn set_active_features(&mut self, features: FeatureFlags) {
        if self.active_features != features {
            log::info!("Active features changed: {:?} -> {:?}", self.active_features, features);
            self.active_features = features;
            // Pipelines will be lazily recreated on next get_or_create()
        }
    }

    /// Get current active features
    pub fn active_features(&self) -> FeatureFlags {
        self.active_features
    }

    /// Pre-compile common variants to avoid first-frame hitches
    pub fn precompile_variants(&mut self, variants: &[(String, PipelineVariant)]) {
        for (shader_id, variant) in variants {
            log::debug!("Precompiling pipeline variant: {} {:?}", shader_id, variant);
            // TODO: Precompile with different feature combinations
        }
    }

    /// Clear the pipeline cache
    pub fn clear(&mut self) {
        log::info!("Clearing pipeline cache ({} pipelines)", self.cache.len());
        self.cache.clear();
        self.shader_modules.clear();
    }
}
