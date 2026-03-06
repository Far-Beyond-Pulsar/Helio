/// Pipeline cache with variant system for hot-swapping features

use super::PipelineVariant;
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

        // Build pipeline layout — varies by variant
        let pipeline_layout = match variant {
            PipelineVariant::GBufferWrite => {
                // G-buffer writer: group 0 camera/globals, group 1 material.
                // No lighting bind group — this pass doesn't evaluate any lights.
                self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(&format!("{}_layout", shader_id)),
                    bind_group_layouts: &[
                        Some(&self.bind_group_layouts.global  as &wgpu::BindGroupLayout),
                        Some(&self.bind_group_layouts.material as &wgpu::BindGroupLayout),
                    ],
                    immediate_size: 0,
                })
            }
            PipelineVariant::DeferredLighting => {
                // Deferred lighting: group 0 camera/globals, group 1 gbuffer_read,
                // group 2 lights/shadows/env (same as Forward's group 2).
                self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(&format!("{}_layout", shader_id)),
                    bind_group_layouts: &[
                        Some(&self.bind_group_layouts.global       as &wgpu::BindGroupLayout),
                        Some(&self.bind_group_layouts.gbuffer_read as &wgpu::BindGroupLayout),
                        Some(&self.bind_group_layouts.lighting     as &wgpu::BindGroupLayout),
                    ],
                    immediate_size: 0,
                })
            }
            _ => {
                // Forward / DepthOnly / Deferred-alias: classic three-group layout
                self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(&format!("{}_layout", shader_id)),
                    bind_group_layouts: &[
                        Some(&self.bind_group_layouts.global   as &wgpu::BindGroupLayout),
                        Some(&self.bind_group_layouts.material as &wgpu::BindGroupLayout),
                        Some(&self.bind_group_layouts.lighting as &wgpu::BindGroupLayout),
                    ],
                    immediate_size: 0,
                })
            }
        };

        // Create pipeline based on variant
        let pipeline = match variant {
            PipelineVariant::Forward => {
                self.create_forward_pipeline(&pipeline_layout, &shader_module)?
            }
            PipelineVariant::TransparentForward => {
                self.create_transparent_forward_pipeline(&pipeline_layout, &shader_module)?
            }
            PipelineVariant::DepthOnly => {
                self.create_depth_only_pipeline(&pipeline_layout, &shader_module)?
            }
            PipelineVariant::GBufferWrite => {
                self.create_gbuffer_write_pipeline(&pipeline_layout, &shader_module)?
            }
            PipelineVariant::DeferredLighting => {
                self.create_deferred_lighting_pipeline(&pipeline_layout, &shader_module)?
            }
            PipelineVariant::Deferred => {
                return Err(Error::Pipeline("Use PipelineVariant::DeferredLighting".into()));
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
                entry_point: Some("vs_main"),
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
                entry_point: Some("fs_main"),
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
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
        }))
    }

    fn create_transparent_forward_pipeline(
        &self,
        layout: &wgpu::PipelineLayout,
        shader: &wgpu::ShaderModule,
    ) -> Result<wgpu::RenderPipeline> {
        Ok(self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Transparent Forward Pipeline"),
            layout: Some(layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: 32,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
                            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32, offset: 12, shader_location: 1 },
                            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 16, shader_location: 2 },
                            wgpu::VertexAttribute { format: wgpu::VertexFormat::Uint32, offset: 24, shader_location: 3 },
                            wgpu::VertexAttribute { format: wgpu::VertexFormat::Uint32, offset: 28, shader_location: 4 },
                        ],
                    },
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.surface_format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
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
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
        }))
    }

    /// Standard 32-byte packed vertex layout shared by all geometry-writing pipelines.
    fn geometry_vertex_buffers() -> [wgpu::VertexBufferLayout<'static>; 1] {
        [wgpu::VertexBufferLayout {
            array_stride: 32,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset:  0, shader_location: 0 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32,   offset: 12, shader_location: 1 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 16, shader_location: 2 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Uint32,    offset: 24, shader_location: 3 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Uint32,    offset: 28, shader_location: 4 },
            ],
        }]
    }

    /// G-buffer write: 4 colour targets, depth-write, no lighting uniforms.
    fn create_gbuffer_write_pipeline(
        &self,
        layout: &wgpu::PipelineLayout,
        shader: &wgpu::ShaderModule,
    ) -> Result<wgpu::RenderPipeline> {
        let vbufs = Self::geometry_vertex_buffers();
        Ok(self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("GBuffer Write Pipeline"),
            layout: Some(layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                buffers: &vbufs,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                targets: &[
                    // target 0: albedo (Rgba8Unorm)
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    // target 1: normals (Rgba16Float)
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    // target 2: ORM (Rgba8Unorm)
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    // target 3: emissive (Rgba16Float)
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
        }))
    }

    /// Deferred lighting: fullscreen triangle (no vertex buffers), reads G-buffer group 1,
    /// runs PBR+shadows, outputs to the swapchain format.  No depth write.
    fn create_deferred_lighting_pipeline(
        &self,
        layout: &wgpu::PipelineLayout,
        shader: &wgpu::ShaderModule,
    ) -> Result<wgpu::RenderPipeline> {
        Ok(self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Deferred Lighting Pipeline"),
            layout: Some(layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                buffers: &[],   // fullscreen triangle — no vertex buffer
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.surface_format,
                    blend: None,        // opaque; sky pixels are discarded in shader
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,    // reads depth from G-buffer; never writes depth
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
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
                entry_point: Some("vs_main"),
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
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
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
