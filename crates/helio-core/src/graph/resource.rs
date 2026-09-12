use std::collections::HashMap;

use super::executor::format_bpp;

// ── Resource Declaration API (used by RenderPass::declare_resources) ──────

/// Texture format specification for transient resources.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceFormat {
    Rgba16Float,
    Rg11b10Ufloat,
    Rgba8UnormSrgb,
    Bgra8Unorm,
    Bgra8UnormSrgb,
    R16Float,
    R32Float,
    R8Unorm,
    Rgba8Unorm,
    Rg16Float,
    Depth32Float,
    R32Uint,
}

impl ResourceFormat {
    pub fn to_wgpu(self) -> wgpu::TextureFormat {
        match self {
            Self::Rg11b10Ufloat => wgpu::TextureFormat::Rg11b10Ufloat,
            Self::Rgba16Float => wgpu::TextureFormat::Rgba16Float,
            Self::Rgba8UnormSrgb => wgpu::TextureFormat::Rgba8UnormSrgb,
            Self::Bgra8Unorm => wgpu::TextureFormat::Bgra8Unorm,
            Self::Bgra8UnormSrgb => wgpu::TextureFormat::Bgra8UnormSrgb,
            Self::R16Float => wgpu::TextureFormat::R16Float,
            Self::R32Float => wgpu::TextureFormat::R32Float,
            Self::R8Unorm => wgpu::TextureFormat::R8Unorm,
            Self::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
            Self::Rg16Float => wgpu::TextureFormat::Rg16Float,
            Self::Depth32Float => wgpu::TextureFormat::Depth32Float,
            Self::R32Uint => wgpu::TextureFormat::R32Uint,
        }
    }
}

impl From<wgpu::TextureFormat> for ResourceFormat {
    fn from(f: wgpu::TextureFormat) -> Self {
        match f {
            wgpu::TextureFormat::Rg11b10Ufloat => Self::Rg11b10Ufloat,
            wgpu::TextureFormat::Rgba16Float => Self::Rgba16Float,
            wgpu::TextureFormat::Rgba8UnormSrgb => Self::Rgba8UnormSrgb,
            wgpu::TextureFormat::Bgra8Unorm => Self::Bgra8Unorm,
            wgpu::TextureFormat::Bgra8UnormSrgb => Self::Bgra8UnormSrgb,
            wgpu::TextureFormat::R16Float => Self::R16Float,
            wgpu::TextureFormat::R32Float => Self::R32Float,
            wgpu::TextureFormat::R8Unorm => Self::R8Unorm,
            wgpu::TextureFormat::Rgba8Unorm => Self::Rgba8Unorm,
            wgpu::TextureFormat::Rg16Float => Self::Rg16Float,
            wgpu::TextureFormat::Depth32Float => Self::Depth32Float,
            wgpu::TextureFormat::R32Uint => Self::R32Uint,
            // Any other format asked for via `write_color_raw` that this
            // transient-resource enum doesn't (yet) have a dedicated variant
            // for would previously have silently aliased to Rgba16Float here
            // — surfacing as a confusing "wrong sample type" bind-group
            // validation error far from the actual mistake. Fail loudly
            // instead: add the missing variant above rather than guessing.
            other => panic!(
                "ResourceFormat::from({other:?}): no transient-resource variant for this \
                 wgpu::TextureFormat — add one instead of letting it silently alias to \
                 another format"
            ),
        }
    }
}

/// Size specification for transient resources.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceSize {
    /// Match the internal render resolution (render_scale × output)
    MatchSurface,
    /// Match the full output/display resolution
    Output,
    Absolute {
        width: u32,
        height: u32,
    },
    /// Output resolution divided by `divisor`.
    ///
    /// Note this divides the *output* resolution, not the internal one. An effect
    /// buffer that gets sampled alongside `depth` or the gbuffer almost certainly
    /// wants [`ScaledInternal`](Self::ScaledInternal) instead — those live at the
    /// internal resolution, which differs from output whenever render_scale != 1.
    Scaled {
        divisor: u32,
    },
    /// Internal render resolution divided by `divisor`.
    ///
    /// The right choice for reduced-resolution effect buffers that pair with
    /// internal-resolution inputs (depth, gbuffer), since it scales with them.
    ScaledInternal {
        divisor: u32,
    },
}

/// Resource access mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceAccess {
    Read,
    Write,
}

/// Resource declaration — describes a resource a pass reads or writes.
#[derive(Debug, Clone, PartialEq)]
pub struct ResourceDecl {
    pub name: &'static str,
    pub format: Option<ResourceFormat>,
    pub size: Option<ResourceSize>,
    pub access: ResourceAccess,
    /// Number of array layers (1 for 2D, N for 2D array).
    pub layers: u32,
    /// Extra texture usage flags beyond RENDER_ATTACHMENT | TEXTURE_BINDING.
    pub extra_usage: wgpu::TextureUsages,
    /// Compound-resource tag set by [`ResourceBuilder::write_group`]. Two
    /// write declarations from the same pass sharing a `group` are resolved
    /// together into one `PrePassAction::Group` by the allocator, generically
    /// over arity — see `docs/helio_3_0_spec.md` §5. `None` for every
    /// ordinary single-view declaration.
    pub group: Option<&'static str>,
}

/// Resource dependency builder — used in `RenderPass::declare_resources()`.
pub struct ResourceBuilder {
    declarations: Vec<ResourceDecl>,
}

impl ResourceBuilder {
    pub fn new() -> Self {
        Self {
            declarations: Vec::with_capacity(8),
        }
    }

    /// Read a resource written by an earlier pass.
    pub fn read(&mut self, name: &'static str) {
        self.declarations.push(ResourceDecl {
            name,
            format: None,
            size: None,
            access: ResourceAccess::Read,
            layers: 1,
            extra_usage: wgpu::TextureUsages::empty(),
            group: None,
        });
    }

    /// Shared push path for every single-view write declaration — reused by
    /// `write_color` and `write_group` so neither duplicates `ResourceDecl`
    /// construction.
    fn push_write(
        &mut self,
        name: &'static str,
        format: ResourceFormat,
        size: ResourceSize,
        group: Option<&'static str>,
    ) {
        self.declarations.push(ResourceDecl {
            name,
            format: Some(format),
            size: Some(size),
            access: ResourceAccess::Write,
            layers: 1,
            extra_usage: wgpu::TextureUsages::empty(),
            group,
        });
    }

    /// Write a color texture. The graph creates and owns this texture.
    pub fn write_color(&mut self, name: &'static str, format: ResourceFormat, size: ResourceSize) {
        self.push_write(name, format, size, None);
    }

    /// Declares a named group of `N` color views produced and consumed as one
    /// unit (e.g. GBuffer's albedo/normal/orm/emissive bundle). The allocator
    /// groups these by declaration — not by pattern-matching exact string
    /// suffixes — so any future compound resource gets the same handling with
    /// no new code in `resource_lifetime.rs`. See `docs/helio_3_0_spec.md` §5.
    pub fn write_group<const N: usize>(
        &mut self,
        group_name: &'static str,
        members: [(&'static str, ResourceFormat); N],
        size: ResourceSize,
    ) {
        for (name, format) in members {
            self.push_write(name, format, size, Some(group_name));
        }
    }

    /// Add extra usage flags to every declaration tagged with `group_name`
    /// (i.e. every member pushed by a prior [`write_group`](Self::write_group)
    /// call for that name). Generic counterpart to
    /// [`with_extra_usage`](Self::with_extra_usage), which only touches the
    /// single most-recently-added declaration.
    pub fn with_group_extra_usage(
        &mut self,
        group_name: &'static str,
        usage: wgpu::TextureUsages,
    ) -> &mut Self {
        for decl in self.declarations.iter_mut() {
            if decl.group == Some(group_name) {
                decl.extra_usage = usage;
            }
        }
        self
    }

    /// Write a depth texture.
    pub fn write_depth(&mut self, name: &'static str, size: ResourceSize) {
        self.write_color(name, ResourceFormat::Depth32Float, size);
    }

    /// Write a color texture using a raw `wgpu::TextureFormat`.
    pub fn write_color_raw(
        &mut self,
        name: &'static str,
        format: wgpu::TextureFormat,
        size: ResourceSize,
    ) {
        self.write_color(name, ResourceFormat::from(format), size);
    }

    /// Set array layers on the most recently added declaration (for array textures).
    pub fn with_layers(&mut self, layers: u32) -> &mut Self {
        if let Some(decl) = self.declarations.last_mut() {
            decl.layers = layers;
        }
        self
    }

    /// Add extra usage flags to the most recently added declaration.
    pub fn with_extra_usage(&mut self, usage: wgpu::TextureUsages) -> &mut Self {
        if let Some(decl) = self.declarations.last_mut() {
            decl.extra_usage = usage;
        }
        self
    }

    /// Declare a storage-buffer resource written by this pass.
    /// The graph tracks it for dependency ordering; the pass allocates it manually.
    pub fn write_buffer(&mut self, name: &'static str) {
        self.declarations.push(ResourceDecl {
            name,
            format: None,
            size: Some(ResourceSize::MatchSurface),
            access: ResourceAccess::Write,
            layers: 1,
            extra_usage: wgpu::TextureUsages::empty(),
            group: None,
        });
    }

    pub fn declarations(&self) -> &[ResourceDecl] {
        &self.declarations
    }
}

/// Resource lifetime handle (placeholder for future ref-counting).
pub struct ResourceHandle;

impl ResourceHandle {
    pub fn named(_name: &str) -> Self {
        Self
    }
}

// ── Graph Texture Pool (owns and aliases inter-pass textures) ──────────────

/// Size reference for graph-allocated textures.
#[derive(Debug, Clone, Copy)]
pub enum ResSize {
    Internal,
    Output,
    Absolute(u32, u32),
}

/// Descriptor for creating a graph-managed texture.
#[derive(Debug, Clone)]
pub struct TextureDescriptor {
    pub name: String,
    pub format: wgpu::TextureFormat,
    pub width: u32,
    pub height: u32,
    pub depth_or_array_layers: u32,
    pub mip_level_count: u32,
    pub sample_count: u32,
    pub usage: wgpu::TextureUsages,
    /// Same alias_group → same backing allocation (lifetime must not overlap).
    pub alias_group: Option<String>,
}

/// A texture allocation owned by the graph.
pub struct GraphTexture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub desc: TextureDescriptor,
    allocation_id: usize,
}

/// Pool of graph-owned textures with lifetime-based aliasing.
///
/// Non-overlapping textures in the same alias group share a single
/// `wgpu::Texture` allocation, dramatically reducing peak VRAM.
pub struct GraphTexturePool {
    textures: Vec<GraphTexture>,
    name_map: HashMap<String, usize>,
    alias_refs: HashMap<String, u32>,
    physical_allocations: usize,
    xr_active: bool,
}

impl GraphTexturePool {
    pub fn new() -> Self {
        Self {
            textures: Vec::new(),
            name_map: HashMap::new(),
            alias_refs: HashMap::new(),
            physical_allocations: 0,
            xr_active: false,
        }
    }

    pub fn set_xr_mode(&mut self, active: bool) {
        self.xr_active = active;
    }

    /// Allocate a texture. If `alias_group` matches a released compatible
    /// texture, reuses the existing underlying GPU allocation.
    pub fn allocate(&mut self, device: &wgpu::Device, desc: TextureDescriptor) -> &GraphTexture {
        let array_layers = if self.xr_active {
            desc.depth_or_array_layers.max(1).max(2)
        } else {
            desc.depth_or_array_layers.max(1)
        };

        if let Some(group) = desc.alias_group.as_deref() {
            if self.alias_refs.get(group).copied().unwrap_or(0) == 0 {
                if let Some(source_index) = self.textures.iter().position(|candidate| {
                    candidate.desc.alias_group.as_deref() == Some(group)
                        && candidate.desc.format == desc.format
                        && candidate.desc.width >= desc.width.max(1)
                        && candidate.desc.height >= desc.height.max(1)
                        && candidate.desc.depth_or_array_layers.max(if self.xr_active {
                            2
                        } else {
                            1
                        }) >= array_layers
                        && candidate.desc.mip_level_count >= desc.mip_level_count.max(1)
                        && candidate.desc.sample_count == desc.sample_count.max(1)
                        && candidate.desc.usage.contains(desc.usage)
                }) {
                    let source_texture = self.textures[source_index].texture.clone();
                    let view = if self.xr_active {
                        source_texture.create_view(&wgpu::TextureViewDescriptor {
                            label: Some(&desc.name),
                            dimension: Some(wgpu::TextureViewDimension::D2Array),
                            array_layer_count: Some(2),
                            ..Default::default()
                        })
                    } else {
                        source_texture.create_view(&wgpu::TextureViewDescriptor {
                            label: Some(&desc.name),
                            ..Default::default()
                        })
                    };
                    let idx = self.textures.len();
                    self.textures.push(GraphTexture {
                        texture: source_texture,
                        view,
                        desc: desc.clone(),
                        allocation_id: self.textures[source_index].allocation_id,
                    });
                    self.name_map.insert(desc.name.clone(), idx);
                    *self.alias_refs.entry(group.to_owned()).or_insert(0) += 1;
                    return &self.textures[idx];
                }
            }
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&desc.name),
            size: wgpu::Extent3d {
                width: desc.width.max(1),
                height: desc.height.max(1),
                depth_or_array_layers: array_layers,
            },
            mip_level_count: desc.mip_level_count.max(1),
            sample_count: desc.sample_count.max(1),
            dimension: wgpu::TextureDimension::D2,
            format: desc.format,
            usage: desc.usage,
            view_formats: &[],
        });
        let view = if self.xr_active {
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&desc.name),
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                array_layer_count: Some(2),
                ..Default::default()
            })
        } else {
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&desc.name),
                ..Default::default()
            })
        };

        let idx = self.textures.len();
        let allocation_id = self.physical_allocations;
        self.physical_allocations += 1;
        self.textures.push(GraphTexture {
            texture,
            view,
            desc: desc.clone(),
            allocation_id,
        });
        self.name_map.insert(desc.name.clone(), idx);

        if let Some(group) = &desc.alias_group {
            *self.alias_refs.entry(group.clone()).or_insert(0) += 1;
        }

        &self.textures[idx]
    }

    pub fn get_view(&self, name: &str) -> Option<&wgpu::TextureView> {
        self.name_map.get(name).map(|&idx| &self.textures[idx].view)
    }

    pub fn get_texture(&self, name: &str) -> Option<&wgpu::Texture> {
        self.name_map
            .get(name)
            .map(|&idx| &self.textures[idx].texture)
    }

    /// Number of logical graph resources currently mapped in the pool.
    pub fn resource_count(&self) -> usize {
        self.textures.len()
    }

    /// Number of physical `wgpu::Texture` objects created by this pool.
    /// Aliased logical resources share one physical allocation.
    pub fn physical_allocation_count(&self) -> usize {
        self.physical_allocations
    }

    /// Estimated bytes reserved by physical textures. Aliased logical views
    /// are counted once, using the largest descriptor that owns an allocation.
    pub fn physical_vram_bytes(&self) -> u64 {
        let mut by_allocation = HashMap::<usize, u64>::new();
        for texture in &self.textures {
            let bytes = texture.desc.width.max(1) as u64
                * texture.desc.height.max(1) as u64
                * texture.desc.depth_or_array_layers.max(1) as u64
                * texture.desc.sample_count.max(1) as u64
                * format_bpp(texture.desc.format) as u64
                / 8;
            by_allocation
                .entry(texture.allocation_id)
                .and_modify(|current| *current = (*current).max(bytes))
                .or_insert(bytes);
        }
        by_allocation.values().sum()
    }

    /// Returns the physical allocation identity for a logical resource.
    /// Intended for diagnostics and aliasing contract tests.
    pub fn allocation_id(&self, name: &str) -> Option<usize> {
        self.name_map
            .get(name)
            .map(|&idx| self.textures[idx].allocation_id)
    }

    /// Release a texture in an alias group, decrementing its ref count.
    pub fn release(&mut self, name: &str) {
        if let Some(&idx) = self.name_map.get(name) {
            if let Some(ref group) = self.textures[idx].desc.alias_group {
                if let Some(count) = self.alias_refs.get_mut(group.as_str()) {
                    *count = count.saturating_sub(1);
                }
            }
        }
    }

    pub fn clear(&mut self) {
        self.textures.clear();
        self.name_map.clear();
        self.alias_refs.clear();
        self.physical_allocations = 0;
    }
}

/// Allocates graph textures at a specific resolution.
pub struct ResourceAllocator {
    pub pool: GraphTexturePool,
    pub internal_w: u32,
    pub internal_h: u32,
    pub output_w: u32,
    pub output_h: u32,
}

impl ResourceAllocator {
    pub fn new(internal_w: u32, internal_h: u32, output_w: u32, output_h: u32) -> Self {
        Self {
            pool: GraphTexturePool::new(),
            internal_w,
            internal_h,
            output_w,
            output_h,
        }
    }

    pub fn allocate(&mut self, device: &wgpu::Device, desc: TextureDescriptor) -> &GraphTexture {
        self.pool.allocate(device, desc)
    }

    pub fn allocate_color(
        &mut self,
        device: &wgpu::Device,
        name: &'static str,
        format: wgpu::TextureFormat,
        size: ResSize,
        alias_group: Option<&'static str>,
    ) -> &GraphTexture {
        let (w, h) = self.resolve_size(size);
        self.allocate(
            device,
            TextureDescriptor {
                name: name.to_string(),
                format,
                width: w,
                height: h,
                depth_or_array_layers: 1,
                mip_level_count: 1,
                sample_count: 1,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                alias_group: alias_group.map(|s| s.to_string()),
            },
        )
    }

    pub fn allocate_depth(
        &mut self,
        device: &wgpu::Device,
        name: &'static str,
        size: ResSize,
        alias_group: Option<&'static str>,
    ) -> &GraphTexture {
        let (w, h) = self.resolve_size(size);
        self.allocate(
            device,
            TextureDescriptor {
                name: name.to_string(),
                format: wgpu::TextureFormat::Depth32Float,
                width: w,
                height: h,
                depth_or_array_layers: 1,
                mip_level_count: 1,
                sample_count: 1,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                alias_group: alias_group.map(|s| s.to_string()),
            },
        )
    }

    fn resolve_size(&self, size: ResSize) -> (u32, u32) {
        match size {
            ResSize::Internal => (self.internal_w, self.internal_h),
            ResSize::Output => (self.output_w, self.output_h),
            ResSize::Absolute(width, height) => (width, height),
        }
    }
}
