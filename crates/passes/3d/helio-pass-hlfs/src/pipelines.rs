use crate::HlfsMode;

pub(crate) const COMMON: &str = include_str!("../shaders/common.wgsl");
const LIGHTING: &str = include_str!("../shaders/lighting.wgsl");
const SHADOWS: &str = include_str!("../shaders/shadows.wgsl");
pub(crate) fn shader_source(stage: &str) -> String {
    match stage {
        "depth" => include_str!("../shaders/depth_pyramid.wgsl").into(),
        "grid" => [COMMON, include_str!("../shaders/light_grid.wgsl")].concat(),
        "screen_space" => [
            COMMON,
            LIGHTING,
            SHADOWS,
            include_str!("../shaders/sample.wgsl"),
        ]
        .concat(),
        "spatial" => [COMMON, LIGHTING, include_str!("../shaders/spatial.wgsl")].concat(),
        "temporal" => [COMMON, include_str!("../shaders/temporal.wgsl")].concat(),
        "composite" => [
            COMMON,
            LIGHTING,
            SHADOWS,
            include_str!("../shaders/composite.wgsl"),
        ]
        .concat(),
        "ray_query_prototype" => {
            let sample = include_str!("../shaders/sample.wgsl").replace(
                "return shadow_factor(id,surface.position,surface.normal,vec2<f32>(pixel)+0.5,globals.frame);",
                "return rt_shadow(lights[id],surface.position,surface.normal);");
            [
                "enable wgpu_ray_query;\n",
                COMMON,
                LIGHTING,
                SHADOWS,
                include_str!("../shaders/ray_shadow.wgsl"),
                &sample,
            ]
            .concat()
        }
        _ => panic!("unknown HLFS shader stage"),
    }
}

fn entry(
    binding: u32,
    ty: wgpu::BindingType,
    visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty,
        count: None,
    }
}
fn storage(read_only: bool) -> wgpu::BindingType {
    wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only },
        has_dynamic_offset: false,
        min_binding_size: None,
    }
}
fn uniform() -> wgpu::BindingType {
    wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Uniform,
        has_dynamic_offset: false,
        min_binding_size: None,
    }
}
fn texture(dimension: wgpu::TextureViewDimension, depth: bool) -> wgpu::BindingType {
    wgpu::BindingType::Texture {
        sample_type: if depth {
            wgpu::TextureSampleType::Depth
        } else {
            wgpu::TextureSampleType::Float { filterable: false }
        },
        view_dimension: dimension,
        multisampled: false,
    }
}
fn uint_texture() -> wgpu::BindingType {
    wgpu::BindingType::Texture {
        sample_type: wgpu::TextureSampleType::Uint,
        view_dimension: wgpu::TextureViewDimension::D2,
        multisampled: false,
    }
}
fn storage_texture(format: wgpu::TextureFormat) -> wgpu::BindingType {
    wgpu::BindingType::StorageTexture {
        access: wgpu::StorageTextureAccess::WriteOnly,
        format,
        view_dimension: wgpu::TextureViewDimension::D2,
    }
}
fn bgl(
    device: &wgpu::Device,
    label: &str,
    entries: &[wgpu::BindGroupLayoutEntry],
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries,
    })
}

fn compute(
    device: &wgpu::Device,
    label: &str,
    shader: &wgpu::ShaderModule,
    entry: &str,
    layout: &wgpu::PipelineLayout,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        module: shader,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// The visibility backend owns only its sampling pipelines. Grid construction,
/// reservoir bindings and denoising remain in the shared pass.
pub(crate) enum VisibilityPipelines {
    ScreenSpace {
        regular: wgpu::ComputePipeline,
        small: wgpu::ComputePipeline,
    },
}
impl VisibilityPipelines {
    fn new(
        device: &wgpu::Device,
        mode: HlfsMode,
        common: &wgpu::BindGroupLayout,
        gbuffer: &wgpu::BindGroupLayout,
        reservoirs: &wgpu::BindGroupLayout,
    ) -> Self {
        match mode {
            HlfsMode::ScreenSpace => {
                let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("HLFS ScreenSpace visibility"),
                    source: wgpu::ShaderSource::Wgsl(shader_source("screen_space").into()),
                });
                let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("HLFS ScreenSpace visibility"),
                    bind_group_layouts: &[Some(common), Some(gbuffer), Some(reservoirs)],
                    immediate_size: 0,
                });
                Self::ScreenSpace {
                    regular: compute(
                        device,
                        "HLFS ScreenSpace visibility",
                        &shader,
                        "sample_lights",
                        &layout,
                    ),
                    small: compute(
                        device,
                        "HLFS ScreenSpace small population",
                        &shader,
                        "sample_small",
                        &layout,
                    ),
                }
            }
        }
    }

    pub fn pipeline(&self, small_population: bool) -> &wgpu::ComputePipeline {
        match self {
            Self::ScreenSpace { regular, small } => {
                if small_population {
                    small
                } else {
                    regular
                }
            }
        }
    }
}

pub(crate) struct Pipelines {
    pub common_bgl: wgpu::BindGroupLayout,
    pub gbuffer_bgl: wgpu::BindGroupLayout,
    pub depth_bgl: wgpu::BindGroupLayout,
    pub grid_bgl: wgpu::BindGroupLayout,
    pub sample_bgl: wgpu::BindGroupLayout,
    pub temporal_bgl: wgpu::BindGroupLayout,
    pub spatial_bgl: wgpu::BindGroupLayout,
    pub composite_bgl: wgpu::BindGroupLayout,
    pub depth_reduce: wgpu::ComputePipeline,
    pub coarse: wgpu::ComputePipeline,
    pub fine: wgpu::ComputePipeline,
    pub visibility: VisibilityPipelines,
    pub temporal: wgpu::ComputePipeline,
    pub spatial: wgpu::ComputePipeline,
    pub composite: wgpu::RenderPipeline,
}
impl Pipelines {
    pub fn set_mode(&mut self, device: &wgpu::Device, mode: HlfsMode) {
        self.visibility = VisibilityPipelines::new(
            device,
            mode,
            &self.common_bgl,
            &self.gbuffer_bgl,
            &self.sample_bgl,
        );
    }

    pub fn new(device: &wgpu::Device, output_format: wgpu::TextureFormat, mode: HlfsMode) -> Self {
        use wgpu::{ShaderStages as S, TextureFormat as F, TextureViewDimension as D};
        let all = S::COMPUTE | S::FRAGMENT;
        let common_bgl = bgl(
            device,
            "HLFS common layout",
            &[
                entry(0, uniform(), all),
                entry(1, storage(true), all),
                entry(2, storage(true), all),
                entry(3, uniform(), all),
                entry(4, texture(D::D2Array, true), all),
                entry(
                    5,
                    wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    all,
                ),
                entry(6, storage(true), all),
                entry(7, texture(D::D2Array, false), all),
            ],
        );
        let mut gbuffer_entries: Vec<_> = (0..10)
            .map(|i| entry(i, texture(D::D2, i == 4), all))
            .collect();
        gbuffer_entries[6].ty = wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: D::D2,
            multisampled: false,
        };
        gbuffer_entries[7].ty = wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering);
        let gbuffer_bgl = bgl(device, "HLFS GBuffer layout", &gbuffer_entries);
        let depth_bgl = bgl(
            device,
            "HLFS depth layout",
            &[
                entry(0, texture(D::D2, false), S::COMPUTE),
                entry(1, storage_texture(F::R32Float), S::COMPUTE),
            ],
        );
        let grid_bgl = bgl(
            device,
            "HLFS light grid layout",
            &[
                entry(0, storage(false), S::COMPUTE),
                entry(1, storage(false), S::COMPUTE),
                entry(2, storage_texture(F::R32Float), S::COMPUTE),
            ],
        );
        let sample_bgl = bgl(
            device,
            "HLFS sampling layout",
            &[
                entry(0, storage(true), S::COMPUTE),
                entry(1, storage(true), S::COMPUTE),
                entry(2, storage(false), S::COMPUTE),
                entry(3, storage_texture(F::Rg32Uint), S::COMPUTE),
                entry(4, uint_texture(), S::COMPUTE),
                entry(5, texture(D::D2, false), S::COMPUTE),
            ],
        );
        let temporal_bgl = bgl(
            device,
            "HLFS temporal layout",
            &[
                entry(0, uint_texture(), S::COMPUTE),
                entry(1, uint_texture(), S::COMPUTE),
                entry(2, uint_texture(), S::COMPUTE),
                entry(3, storage_texture(F::Rg32Uint), S::COMPUTE),
                entry(4, storage_texture(F::Rg32Uint), S::COMPUTE),
                entry(5, storage(true), S::COMPUTE),
                entry(6, storage(true), S::COMPUTE),
            ],
        );
        let spatial_bgl = bgl(
            device,
            "HLFS spatial layout",
            &[
                entry(0, uint_texture(), S::COMPUTE),
                entry(1, uint_texture(), S::COMPUTE),
                entry(2, storage_texture(F::Rg32Uint), S::COMPUTE),
            ],
        );
        let composite_bgl = bgl(
            device,
            "HLFS composite layout",
            &(0..5)
                .map(|i| {
                    entry(
                        i,
                        if i == 2 {
                            texture(D::D2, false)
                        } else {
                            uint_texture()
                        },
                        S::FRAGMENT,
                    )
                })
                .collect::<Vec<_>>(),
        );
        let module = |name| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(shader_source(name).into()),
            })
        };
        let grid_shader = module("grid");
        let temporal_shader = module("temporal");
        let composite_shader = module("composite");
        let layout =
            |label, third: &wgpu::BindGroupLayout, fourth: Option<&wgpu::BindGroupLayout>| {
                let mut layouts = vec![Some(&common_bgl), Some(&gbuffer_bgl), Some(third)];
                if let Some(fourth) = fourth {
                    layouts.push(Some(fourth));
                }
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(label),
                    bind_group_layouts: &layouts,
                    immediate_size: 0,
                })
            };
        let depth_reduce = compute(
            device,
            "HLFS depth reduction",
            &module("depth"),
            "reduce_depth",
            &layout("HLFS depth", &depth_bgl, None),
        );
        let grid_layout = layout("HLFS grid", &grid_bgl, None);
        let coarse = compute(
            device,
            "HLFS coarse culling",
            &grid_shader,
            "coarse",
            &grid_layout,
        );
        let fine = compute(
            device,
            "HLFS fine depth culling",
            &grid_shader,
            "fine",
            &grid_layout,
        );
        let visibility =
            VisibilityPipelines::new(device, mode, &common_bgl, &gbuffer_bgl, &sample_bgl);
        let temporal = compute(
            device,
            "HLFS temporal denoising",
            &temporal_shader,
            "temporal",
            &layout("HLFS temporal", &temporal_bgl, None),
        );
        let spatial = compute(
            device,
            "HLFS spatial denoising",
            &module("spatial"),
            "spatial",
            &layout("HLFS spatial", &spatial_bgl, None),
        );
        let composite = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("HLFS denoise and composite"),
            layout: Some(&layout("HLFS composite", &composite_bgl, None)),
            vertex: wgpu::VertexState {
                module: &composite_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &composite_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: output_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        Self {
            common_bgl,
            gbuffer_bgl,
            depth_bgl,
            grid_bgl,
            sample_bgl,
            temporal_bgl,
            spatial_bgl,
            composite_bgl,
            depth_reduce,
            coarse,
            fine,
            visibility,
            temporal,
            spatial,
            composite,
        }
    }
}
