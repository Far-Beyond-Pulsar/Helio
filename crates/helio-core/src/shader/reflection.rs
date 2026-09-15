//! Runtime WGSL reflection used by opt-in automated pass construction.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindingKind {
    UniformBuffer,
    StorageBuffer,
    SampledTexture,
    DepthTexture,
    StorageTexture,
    Sampler,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReflectedLayout {
    UniformBuffer,
    StorageBuffer {
        read_only: bool,
    },
    Sampler {
        comparison: bool,
    },
    SampledTexture {
        dimension: naga::ImageDimension,
        arrayed: bool,
        multisampled: bool,
        kind: naga::ScalarKind,
    },
    DepthTexture {
        dimension: naga::ImageDimension,
        arrayed: bool,
        multisampled: bool,
    },
    StorageTexture {
        dimension: naga::ImageDimension,
        arrayed: bool,
        format: naga::StorageFormat,
        access: naga::StorageAccess,
    },
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReflectedBinding {
    pub name: String,
    pub group: u32,
    pub binding: u32,
    pub kind: BindingKind,
    pub layout: ReflectedLayout,
}

impl ReflectedLayout {
    /// Converts reflected WGSL resource metadata into a wgpu layout entry.
    /// Visibility is intentionally supplied by the caller because a global
    /// can be referenced from more than one entry point and naga's global
    /// declaration alone does not encode the host's desired visibility mask.
    pub fn to_layout_entry(
        self,
        binding: u32,
        visibility: wgpu::ShaderStages,
    ) -> Option<wgpu::BindGroupLayoutEntry> {
        use wgpu::{
            BindGroupLayoutEntry, BindingType, BufferBindingType, SamplerBindingType,
            StorageTextureAccess, TextureSampleType, TextureViewDimension,
        };
        let view_dimension =
            |dimension: naga::ImageDimension, arrayed: bool| match (dimension, arrayed) {
                (naga::ImageDimension::D1, false) => Some(TextureViewDimension::D1),
                (naga::ImageDimension::D1, true) => None,
                (naga::ImageDimension::D2, false) => Some(TextureViewDimension::D2),
                (naga::ImageDimension::D2, true) => Some(TextureViewDimension::D2Array),
                (naga::ImageDimension::D3, _) => Some(TextureViewDimension::D3),
                (naga::ImageDimension::Cube, false) => Some(TextureViewDimension::Cube),
                (naga::ImageDimension::Cube, true) => Some(TextureViewDimension::CubeArray),
            };
        let storage_format = |format: naga::StorageFormat| match format {
            naga::StorageFormat::R8Unorm => Some(wgpu::TextureFormat::R8Unorm),
            naga::StorageFormat::R8Snorm => Some(wgpu::TextureFormat::R8Snorm),
            naga::StorageFormat::R8Uint => Some(wgpu::TextureFormat::R8Uint),
            naga::StorageFormat::R8Sint => Some(wgpu::TextureFormat::R8Sint),
            naga::StorageFormat::R16Uint => Some(wgpu::TextureFormat::R16Uint),
            naga::StorageFormat::R16Sint => Some(wgpu::TextureFormat::R16Sint),
            naga::StorageFormat::R16Float => Some(wgpu::TextureFormat::R16Float),
            naga::StorageFormat::Rg8Unorm => Some(wgpu::TextureFormat::Rg8Unorm),
            naga::StorageFormat::Rg8Snorm => Some(wgpu::TextureFormat::Rg8Snorm),
            naga::StorageFormat::Rg8Uint => Some(wgpu::TextureFormat::Rg8Uint),
            naga::StorageFormat::Rg8Sint => Some(wgpu::TextureFormat::Rg8Sint),
            naga::StorageFormat::R32Uint => Some(wgpu::TextureFormat::R32Uint),
            naga::StorageFormat::R32Sint => Some(wgpu::TextureFormat::R32Sint),
            naga::StorageFormat::R32Float => Some(wgpu::TextureFormat::R32Float),
            naga::StorageFormat::Rg16Uint => Some(wgpu::TextureFormat::Rg16Uint),
            naga::StorageFormat::Rg16Sint => Some(wgpu::TextureFormat::Rg16Sint),
            naga::StorageFormat::Rg16Float => Some(wgpu::TextureFormat::Rg16Float),
            naga::StorageFormat::Rgba8Unorm => Some(wgpu::TextureFormat::Rgba8Unorm),
            naga::StorageFormat::Rgba8Snorm => Some(wgpu::TextureFormat::Rgba8Snorm),
            naga::StorageFormat::Rgba8Uint => Some(wgpu::TextureFormat::Rgba8Uint),
            naga::StorageFormat::Rgba8Sint => Some(wgpu::TextureFormat::Rgba8Sint),
            naga::StorageFormat::Bgra8Unorm => Some(wgpu::TextureFormat::Bgra8Unorm),
            naga::StorageFormat::Rg32Uint => Some(wgpu::TextureFormat::Rg32Uint),
            naga::StorageFormat::Rg32Sint => Some(wgpu::TextureFormat::Rg32Sint),
            naga::StorageFormat::Rg32Float => Some(wgpu::TextureFormat::Rg32Float),
            naga::StorageFormat::Rgba16Uint => Some(wgpu::TextureFormat::Rgba16Uint),
            naga::StorageFormat::Rgba16Sint => Some(wgpu::TextureFormat::Rgba16Sint),
            naga::StorageFormat::Rgba16Float => Some(wgpu::TextureFormat::Rgba16Float),
            naga::StorageFormat::Rgba32Uint => Some(wgpu::TextureFormat::Rgba32Uint),
            naga::StorageFormat::Rgba32Sint => Some(wgpu::TextureFormat::Rgba32Sint),
            naga::StorageFormat::Rgba32Float => Some(wgpu::TextureFormat::Rgba32Float),
            // Do not manufacture a compatible-looking layout for a format
            // that WebGPU cannot represent. The caller turns this into the
            // typed UnsupportedBinding error.
            _ => return None,
        };
        let entry = match self {
            Self::UniformBuffer => BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            Self::StorageBuffer { read_only } => BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            Self::Sampler { comparison } => BindingType::Sampler(if comparison {
                SamplerBindingType::Comparison
            } else {
                SamplerBindingType::Filtering
            }),
            Self::SampledTexture {
                dimension,
                arrayed,
                multisampled,
                kind,
            } => BindingType::Texture {
                sample_type: match kind {
                    naga::ScalarKind::Sint => TextureSampleType::Sint,
                    naga::ScalarKind::Uint => TextureSampleType::Uint,
                    _ => TextureSampleType::Float { filterable: true },
                },
                view_dimension: view_dimension(dimension, arrayed)?,
                multisampled,
            },
            Self::DepthTexture {
                dimension,
                arrayed,
                multisampled,
            } => BindingType::Texture {
                sample_type: TextureSampleType::Depth,
                view_dimension: view_dimension(dimension, arrayed)?,
                multisampled,
            },
            Self::StorageTexture {
                dimension,
                arrayed,
                format,
                access,
            } => BindingType::StorageTexture {
                access: if access.contains(naga::StorageAccess::STORE)
                    && access.contains(naga::StorageAccess::LOAD)
                {
                    StorageTextureAccess::ReadWrite
                } else if access.contains(naga::StorageAccess::STORE) {
                    StorageTextureAccess::WriteOnly
                } else {
                    StorageTextureAccess::ReadOnly
                },
                format: storage_format(format)?,
                view_dimension: view_dimension(dimension, arrayed)?,
            },
            Self::Other => return None,
        };
        Some(BindGroupLayoutEntry {
            binding,
            visibility,
            ty: entry,
            count: None,
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ReflectionError {
    #[error("WGSL parse failed: {0}")]
    Parse(String),
    #[error("WGSL validation failed: {0}")]
    Validation(String),
    #[error("reflected binding '{0}' has no published resource")]
    MissingResource(String),
    #[error("reflected binding '{0}' uses an unsupported resource type")]
    UnsupportedBinding(String),
}

/// Builds bind-group entries by matching reflected shader globals to the
/// generic per-frame binding projection.
pub fn populate_bind_group_entries<'a>(
    bindings: &[ReflectedBinding],
    group: u32,
    overrides: &crate::graph::BindingOverrideBuilder,
    resources: &libhelio::ResourceRegistry<'a>,
) -> Result<Vec<wgpu::BindGroupEntry<'a>>, ReflectionError> {
    bindings
        .iter()
        .filter(|binding| binding.group == group)
        .map(|binding| {
            if binding.layout == ReflectedLayout::Other
                || binding
                    .layout
                    .to_layout_entry(binding.binding, wgpu::ShaderStages::all())
                    .is_none()
            {
                return Err(ReflectionError::UnsupportedBinding(binding.name.clone()));
            }
            let resource_name = overrides.resolve(&binding.name);
            let resource = resources
                .binding(resource_name.as_ref())
                .ok_or_else(|| ReflectionError::MissingResource(resource_name.into_owned()))?;
            Ok(wgpu::BindGroupEntry {
                binding: binding.binding,
                resource,
            })
        })
        .collect()
}

/// A reflected shader declaration used by the executor's opt-in generic path.
/// The source is kept by the pass (normally an `include_str!`), while the
/// executor owns the derived layouts and per-frame bind groups.
#[derive(Debug, Clone, Copy)]
pub struct ReflectedShader<'a> {
    pub source: &'a str,
    pub visibility: wgpu::ShaderStages,
}

/// Executor-owned reflection products for one opt-in pass. Bind-group layouts
/// and the pipeline layout are created once when the graph is locked and stay
/// alive for every bind group and pipeline that uses them.
pub struct ReflectedPipeline {
    pub bindings: Vec<ReflectedBinding>,
    pub layouts: Vec<wgpu::BindGroupLayout>,
    pub pipeline_layout: wgpu::PipelineLayout,
    pub directives: crate::shader::PipelineDirectives,
    pub overrides: crate::graph::BindingOverrideBuilder,
}

impl ReflectedPipeline {
    pub fn primitive_state(&self) -> wgpu::PrimitiveState {
        self.directives.primitive_state()
    }

    pub fn blend_state(&self) -> Option<wgpu::BlendState> {
        self.directives.blend_state()
    }

    pub fn depth_stencil_state(
        &self,
        format: wgpu::TextureFormat,
    ) -> Option<wgpu::DepthStencilState> {
        self.directives.depth_stencil_state(format)
    }
}

pub fn create_reflected_pipeline(
    device: &wgpu::Device,
    label: &str,
    shader: ReflectedShader<'_>,
) -> Result<ReflectedPipeline, ReflectionError> {
    let source = crate::shader::resolve(shader.source);
    let bindings = reflect(&source)?;
    let directives = crate::shader::parse_directives(&source)
        .map_err(|error| ReflectionError::Parse(error.to_string()))?;
    let layouts = create_bind_group_layouts(device, label, &bindings, shader.visibility);
    let layout_refs: Vec<&wgpu::BindGroupLayout> = layouts.iter().collect();
    let pipeline_layout = create_pipeline_layout(device, label, &layout_refs);
    Ok(ReflectedPipeline {
        bindings,
        layouts,
        pipeline_layout,
        directives,
        overrides: crate::graph::BindingOverrideBuilder::new(),
    })
}

/// Creates bind groups using layouts prepared at graph-lock time.
pub fn create_reflected_bind_groups_with_layouts<'a>(
    label: &str,
    bindings: &[ReflectedBinding],
    layouts: &[wgpu::BindGroupLayout],
    overrides: &crate::graph::BindingOverrideBuilder,
    resources: &libhelio::ResourceRegistry<'a>,
    device: &wgpu::Device,
) -> Result<Vec<wgpu::BindGroup>, ReflectionError> {
    layouts
        .iter()
        .enumerate()
        .map(|(group, layout)| {
            let entries =
                populate_bind_group_entries(bindings, group as u32, overrides, resources)?;
            Ok(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("{label} Group {group}")),
                layout,
                entries: &entries,
            }))
        })
        .collect()
}

/// Creates all bind groups for a reflected shader from the generic resource
/// projection. Layouts and groups are returned together so callers can keep
/// the layouts alive for pipelines and reuse the groups until their resource
/// generation changes.
pub fn create_reflected_bind_groups<'a>(
    device: &wgpu::Device,
    label: &str,
    bindings: &[ReflectedBinding],
    overrides: &crate::graph::BindingOverrideBuilder,
    resources: &libhelio::ResourceRegistry<'a>,
    visibility: wgpu::ShaderStages,
) -> Result<(Vec<wgpu::BindGroupLayout>, Vec<wgpu::BindGroup>), ReflectionError> {
    let layouts = create_bind_group_layouts(device, label, bindings, visibility);
    let groups = layouts
        .iter()
        .enumerate()
        .map(|(group, layout)| {
            let entries =
                populate_bind_group_entries(bindings, group as u32, overrides, resources)?;
            Ok(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("{label} Group {group}")),
                layout,
                entries: &entries,
            }))
        })
        .collect::<Result<Vec<_>, ReflectionError>>()?;
    Ok((layouts, groups))
}

/// Reflects resource globals from a WGSL module after Helio include expansion.
/// Binding names are retained for the default name-matching contract; callers
/// can apply `BindingOverrideBuilder` for exceptional mappings.
pub fn reflect(source: &str) -> Result<Vec<ReflectedBinding>, ReflectionError> {
    let module = naga::front::wgsl::parse_str(source)
        .map_err(|error| ReflectionError::Parse(error.emit_to_string(source)))?;
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|error| ReflectionError::Validation(format!("{error:?}")))?;

    let mut bindings = Vec::new();
    for (_, global) in module.global_variables.iter() {
        let Some(resource) = global.binding else {
            continue;
        };
        let (kind, layout) = match global.space {
            naga::AddressSpace::Uniform => {
                (BindingKind::UniformBuffer, ReflectedLayout::UniformBuffer)
            }
            naga::AddressSpace::Storage { access } => (
                BindingKind::StorageBuffer,
                ReflectedLayout::StorageBuffer {
                    read_only: !access.contains(naga::StorageAccess::STORE),
                },
            ),
            naga::AddressSpace::Handle => match &module.types[global.ty].inner {
                naga::TypeInner::Sampler { comparison } => (
                    BindingKind::Sampler,
                    ReflectedLayout::Sampler {
                        comparison: *comparison,
                    },
                ),
                naga::TypeInner::Image { class, .. } => match class {
                    naga::ImageClass::Storage { format, access } => {
                        if let naga::TypeInner::Image { dim, arrayed, .. } =
                            &module.types[global.ty].inner
                        {
                            (
                                BindingKind::StorageTexture,
                                ReflectedLayout::StorageTexture {
                                    dimension: *dim,
                                    arrayed: *arrayed,
                                    format: *format,
                                    access: *access,
                                },
                            )
                        } else {
                            unreachable!()
                        }
                    }
                    naga::ImageClass::Sampled { kind, multi } => {
                        if let naga::TypeInner::Image { dim, arrayed, .. } =
                            &module.types[global.ty].inner
                        {
                            (
                                BindingKind::SampledTexture,
                                ReflectedLayout::SampledTexture {
                                    dimension: *dim,
                                    arrayed: *arrayed,
                                    multisampled: *multi,
                                    kind: *kind,
                                },
                            )
                        } else {
                            unreachable!()
                        }
                    }
                    naga::ImageClass::Depth { multi } => {
                        if let naga::TypeInner::Image { dim, arrayed, .. } =
                            &module.types[global.ty].inner
                        {
                            (
                                BindingKind::DepthTexture,
                                ReflectedLayout::DepthTexture {
                                    dimension: *dim,
                                    arrayed: *arrayed,
                                    multisampled: *multi,
                                },
                            )
                        } else {
                            unreachable!()
                        }
                    }
                    naga::ImageClass::External => {
                        (BindingKind::SampledTexture, ReflectedLayout::Other)
                    }
                },
                _ => (BindingKind::Other, ReflectedLayout::Other),
            },
            _ => (BindingKind::Other, ReflectedLayout::Other),
        };
        bindings.push(ReflectedBinding {
            name: global
                .name
                .clone()
                .unwrap_or_else(|| format!("group{}_binding{}", resource.group, resource.binding)),
            group: resource.group,
            binding: resource.binding,
            kind,
            layout,
        });
    }
    bindings.sort_by_key(|binding| (binding.group, binding.binding));
    Ok(bindings)
}

/// Builds deterministic bind-group layout entries for one reflected group.
/// Unsupported resource classes are omitted instead of silently receiving an
/// incompatible layout; callers can then report the shader variable name.
pub fn layout_entries(
    bindings: &[ReflectedBinding],
    group: u32,
    visibility: wgpu::ShaderStages,
) -> Vec<wgpu::BindGroupLayoutEntry> {
    bindings
        .iter()
        .filter(|binding| binding.group == group)
        .filter_map(|binding| binding.layout.to_layout_entry(binding.binding, visibility))
        .collect()
}

/// Creates one bind-group layout for every reflected group, preserving group
/// numbering by inserting an empty layout for groups with no supported
/// resources. This is the executor-facing half of runtime reflection: shader
/// authors do not need to duplicate the binding layout in Rust just to keep a
/// generated WGSL variant valid.
pub fn create_bind_group_layouts(
    device: &wgpu::Device,
    label: &str,
    bindings: &[ReflectedBinding],
    visibility: wgpu::ShaderStages,
) -> Vec<wgpu::BindGroupLayout> {
    let group_count = bindings
        .iter()
        .map(|binding| binding.group)
        .max()
        .map_or(0, |group| group + 1);
    (0..group_count)
        .map(|group| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(&format!("{label} Group {group}")),
                entries: &layout_entries(bindings, group, visibility),
            })
        })
        .collect()
}

/// Creates a pipeline layout from layouts synthesized by
/// [`create_bind_group_layouts`]. The caller owns the returned layouts and must
/// keep them alive for the lifetime of pipelines created from this layout.
pub fn create_pipeline_layout(
    device: &wgpu::Device,
    label: &str,
    layouts: &[&wgpu::BindGroupLayout],
) -> wgpu::PipelineLayout {
    let layouts: Vec<Option<&wgpu::BindGroupLayout>> = layouts.iter().copied().map(Some).collect();
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &layouts,
        immediate_size: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::BindingOverrideBuilder;
    #[test]
    fn reflects_named_resources_in_binding_order() {
        let bindings = reflect(r#"
            @group(1) @binding(2) var tex: texture_2d<f32>;
            @group(0) @binding(0) var<uniform> globals: vec4<f32>;
            @group(1) @binding(0) var samp: sampler;
            @fragment fn main() -> @location(0) vec4<f32> { return textureSample(tex, samp, vec2(0.0)); }
        "#).unwrap();
        assert_eq!(bindings[0].name, "globals");
        assert_eq!(bindings[1].name, "samp");
        assert_eq!(bindings[2].name, "tex");
        assert_eq!(bindings[2].kind, BindingKind::SampledTexture);
        assert!(bindings[0]
            .layout
            .to_layout_entry(0, wgpu::ShaderStages::FRAGMENT)
            .is_some());
        assert_eq!(
            layout_entries(&bindings, 0, wgpu::ShaderStages::VERTEX).len(),
            1
        );
        assert!(bindings[2]
            .layout
            .to_layout_entry(2, wgpu::ShaderStages::FRAGMENT)
            .is_some());
    }

    #[test]
    fn reflected_entries_report_missing_resources() {
        let bindings = reflect(
            r#"
            @group(0) @binding(0) var t_color: texture_2d<f32>;
            @fragment fn main() -> @location(0) vec4<f32> {
                return textureLoad(t_color, vec2<i32>(0), 0);
            }
        "#,
        )
        .unwrap();
        let resources = libhelio::ResourceRegistry::empty();
        let result =
            populate_bind_group_entries(&bindings, 0, &BindingOverrideBuilder::new(), &resources);
        assert!(matches!(result, Err(ReflectionError::MissingResource(name)) if name == "color"));
    }

    #[test]
    fn unsupported_reflected_bindings_are_typed_errors() {
        let bindings = [ReflectedBinding {
            name: "external".into(),
            group: 0,
            binding: 0,
            kind: BindingKind::Other,
            layout: ReflectedLayout::Other,
        }];
        let result = populate_bind_group_entries(
            &bindings,
            0,
            &BindingOverrideBuilder::new(),
            &libhelio::ResourceRegistry::empty(),
        );
        assert!(
            matches!(result, Err(ReflectionError::UnsupportedBinding(name)) if name == "external")
        );
    }

    #[test]
    fn bindless_array_shape_remains_an_explicit_unsupported_binding() {
        let bindings = [ReflectedBinding {
            name: "textures".into(),
            group: 0,
            binding: 0,
            kind: BindingKind::Other,
            layout: ReflectedLayout::Other,
        }];
        let result = populate_bind_group_entries(
            &bindings,
            0,
            &BindingOverrideBuilder::new(),
            &libhelio::ResourceRegistry::empty(),
        );
        assert!(
            matches!(result, Err(ReflectionError::UnsupportedBinding(name)) if name == "textures")
        );
    }

    #[test]
    fn directive_state_is_available_as_pipeline_state() {
        let directives = crate::shader::parse_directives(
            "//!blend alpha\n//!depth less_equal no_write\n//!cull none\n//!topology line",
        )
        .unwrap();
        assert_eq!(
            directives.primitive_state().topology,
            wgpu::PrimitiveTopology::LineList
        );
        assert_eq!(directives.primitive_state().cull_mode, None);
        assert!(directives.blend_state().is_some());
        let depth = directives
            .depth_stencil_state(wgpu::TextureFormat::Depth32Float)
            .unwrap();
        assert_eq!(depth.depth_write_enabled, Some(false));
        assert_eq!(depth.depth_compare, Some(wgpu::CompareFunction::LessEqual));
    }
}
