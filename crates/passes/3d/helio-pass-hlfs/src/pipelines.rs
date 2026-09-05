pub(crate) const COMMON: &str = include_str!("../shaders/common.wgsl");
const LIGHTING: &str = include_str!("../shaders/lighting.wgsl");
const SHADOWS: &str = include_str!("../shaders/shadows.wgsl");
pub(crate) fn shader_source(stage: &str) -> String {
    match stage {
        "grid" => [COMMON, include_str!("../shaders/light_grid.wgsl")].concat(),
        "sample" => [
            COMMON,
            LIGHTING,
            SHADOWS,
            include_str!("../shaders/sample.wgsl"),
        ]
        .concat(),
        "temporal" => [COMMON, include_str!("../shaders/temporal.wgsl")].concat(),
        "composite" => [
            COMMON,
            LIGHTING,
            SHADOWS,
            include_str!("../shaders/composite.wgsl"),
        ]
        .concat(),
        "sample_rt" => {
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

pub(crate) struct Pipelines {
    pub common_bgl: wgpu::BindGroupLayout,
    pub gbuffer_bgl: wgpu::BindGroupLayout,
    pub grid_bgl: wgpu::BindGroupLayout,
    pub sample_bgl: wgpu::BindGroupLayout,
    pub temporal_bgl: wgpu::BindGroupLayout,
    pub composite_bgl: wgpu::BindGroupLayout,
    pub rt_bgl: Option<wgpu::BindGroupLayout>,
    pub coarse: wgpu::ComputePipeline,
    pub fine: wgpu::ComputePipeline,
    pub sample: wgpu::ComputePipeline,
    pub sample_rt: Option<wgpu::ComputePipeline>,
    pub temporal: wgpu::ComputePipeline,
    pub composite: wgpu::RenderPipeline,
}
impl Pipelines {
    pub fn new(device: &wgpu::Device, output_format: wgpu::TextureFormat) -> Self {
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
                entry(3, storage_texture(F::R32Uint), S::COMPUTE),
                entry(4, storage_texture(F::R32Uint), S::COMPUTE),
                entry(5, uint_texture(), S::COMPUTE),
                entry(6, texture(D::D2, false), S::COMPUTE),
            ],
        );
        let mut temporal_entries: Vec<_> = (0..5)
            .map(|i| entry(i, uint_texture(), S::COMPUTE))
            .collect();
        temporal_entries.extend([
            entry(5, storage_texture(F::R32Uint), S::COMPUTE),
            entry(6, storage_texture(F::R32Uint), S::COMPUTE),
            entry(7, storage_texture(F::Rg32Uint), S::COMPUTE),
            entry(8, storage(true), S::COMPUTE),
        ]);
        let temporal_bgl = bgl(device, "HLFS temporal layout", &temporal_entries);
        let composite_bgl = bgl(
            device,
            "HLFS composite layout",
            &(0..4)
                .map(|i| {
                    entry(
                        i,
                        if i < 3 {
                            uint_texture()
                        } else {
                            texture(D::D2, false)
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
        let sample_shader = module("sample");
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
        let compute = |label, shader: &wgpu::ShaderModule, entry, layout: &wgpu::PipelineLayout| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                module: shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let grid_layout = layout("HLFS grid", &grid_bgl, None);
        let coarse = compute("HLFS coarse culling", &grid_shader, "coarse", &grid_layout);
        let fine = compute(
            "HLFS fine depth culling",
            &grid_shader,
            "fine",
            &grid_layout,
        );
        let sample = compute(
            "HLFS stochastic lighting",
            &sample_shader,
            "sample_lights",
            &layout("HLFS sample", &sample_bgl, None),
        );
        let temporal = compute(
            "HLFS temporal denoising",
            &temporal_shader,
            "temporal",
            &layout("HLFS temporal", &temporal_bgl, None),
        );
        let rt_bgl = device
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
            .then(|| {
                bgl(
                    device,
                    "HLFS ray query layout",
                    &[entry(
                        0,
                        wgpu::BindingType::AccelerationStructure {
                            vertex_return: false,
                        },
                        S::COMPUTE,
                    )],
                )
            });
        let sample_rt = rt_bgl.as_ref().map(|rt| {
            compute(
                "HLFS stochastic ray query lighting",
                &module("sample_rt"),
                "sample_lights",
                &layout("HLFS sample RT", &sample_bgl, Some(rt)),
            )
        });
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
            grid_bgl,
            sample_bgl,
            temporal_bgl,
            composite_bgl,
            rt_bgl,
            coarse,
            fine,
            sample,
            sample_rt,
            temporal,
            composite,
        }
    }
}
