use std::collections::HashMap;

// ── Resource Declaration API (used by RenderPass::declare_resources) ──────

/// Texture format specification for transient resources.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceFormat {
    Rgba16Float,
    Rgba8UnormSrgb,
    Bgra8UnormSrgb,
    R16Float,
    R32Float,
    R8Unorm,
    Rgba8Unorm,
    Rg16Float,
    Depth32Float,
}

impl ResourceFormat {
    pub fn to_wgpu(self) -> wgpu::TextureFormat {
        match self {
            Self::Rgba16Float => wgpu::TextureFormat::Rgba16Float,
            Self::Rgba8UnormSrgb => wgpu::TextureFormat::Rgba8UnormSrgb,
            Self::Bgra8UnormSrgb => wgpu::TextureFormat::Bgra8UnormSrgb,
            Self::R16Float => wgpu::TextureFormat::R16Float,
            Self::R32Float => wgpu::TextureFormat::R32Float,
            Self::R8Unorm => wgpu::TextureFormat::R8Unorm,
            Self::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
            Self::Rg16Float => wgpu::TextureFormat::Rg16Float,
            Self::Depth32Float => wgpu::TextureFormat::Depth32Float,
        }
    }
}

impl From<wgpu::TextureFormat> for ResourceFormat {
    fn from(f: wgpu::TextureFormat) -> Self {
        match f {
            wgpu::TextureFormat::Rgba16Float => Self::Rgba16Float,
            wgpu::TextureFormat::Rgba8UnormSrgb => Self::Rgba8UnormSrgb,
            wgpu::TextureFormat::Bgra8UnormSrgb => Self::Bgra8UnormSrgb,
            wgpu::TextureFormat::R16Float => Self::R16Float,
            wgpu::TextureFormat::R32Float => Self::R32Float,
            wgpu::TextureFormat::R8Unorm => Self::R8Unorm,
            wgpu::TextureFormat::Rgba8Unorm => Self::Rgba8Unorm,
            wgpu::TextureFormat::Rg16Float => Self::Rg16Float,
            wgpu::TextureFormat::Depth32Float => Self::Depth32Float,
            _ => Self::Rgba16Float,
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
    Absolute { width: u32, height: u32 },
    Scaled { divisor: u32 },
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
        });
    }

    /// Write a color texture. The graph creates and owns this texture.
    pub fn write_color(&mut self, name: &'static str, format: ResourceFormat, size: ResourceSize) {
        self.declarations.push(ResourceDecl {
            name,
            format: Some(format),
            size: Some(size),
            access: ResourceAccess::Write,
            layers: 1,
            extra_usage: wgpu::TextureUsages::empty(),
        });
    }

    /// Write a depth texture.
    pub fn write_depth(&mut self, name: &'static str, size: ResourceSize) {
        self.write_color(name, ResourceFormat::Depth32Float, size);
    }

    /// Write a color texture using a raw `wgpu::TextureFormat`.
    pub fn write_color_raw(&mut self, name: &'static str, format: wgpu::TextureFormat, size: ResourceSize) {
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

    /// Legacy fallback: declare a write with inferred format/size from ResourceSlot name.
    pub fn write_legacy(&mut self, name: &'static str) {
        self.declarations.push(ResourceDecl {
            name,
            format: None,
            size: Some(ResourceSize::MatchSurface),
            access: ResourceAccess::Write,
            layers: 1,
            extra_usage: wgpu::TextureUsages::empty(),
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
}

/// Pool of graph-owned textures with lifetime-based aliasing.
///
/// Non-overlapping textures in the same alias group share a single
/// `wgpu::Texture` allocation, dramatically reducing peak VRAM.
pub struct GraphTexturePool {
    textures: Vec<GraphTexture>,
    name_map: HashMap<String, usize>,
    alias_refs: HashMap<String, u32>,
}

impl GraphTexturePool {
    pub fn new() -> Self {
        Self {
            textures: Vec::new(),
            name_map: HashMap::new(),
            alias_refs: HashMap::new(),
        }
    }

    /// Allocate a texture. If `alias_group` matches a released texture, reuses it.
    pub fn allocate(
        &mut self,
        device: &wgpu::Device,
        desc: TextureDescriptor,
    ) -> &GraphTexture {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&desc.name),
            size: wgpu::Extent3d {
                width: desc.width.max(1),
                height: desc.height.max(1),
                depth_or_array_layers: desc.depth_or_array_layers.max(1),
            },
            mip_level_count: desc.mip_level_count.max(1),
            sample_count: desc.sample_count.max(1),
            dimension: wgpu::TextureDimension::D2,
            format: desc.format,
            usage: desc.usage,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some(&desc.name),
            ..Default::default()
        });

        let idx = self.textures.len();
        self.textures.push(GraphTexture { texture, view, desc: desc.clone() });
        self.name_map.insert(desc.name.clone(), idx);

        if let Some(group) = &desc.alias_group {
            self.alias_refs.insert(group.clone(), 1);
        }

        &self.textures[idx]
    }

    pub fn get_view(&self, name: &str) -> Option<&wgpu::TextureView> {
        self.name_map.get(name).map(|&idx| &self.textures[idx].view)
    }

    pub fn get_texture(&self, name: &str) -> Option<&wgpu::Texture> {
        self.name_map.get(name).map(|&idx| &self.textures[idx].texture)
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
        Self { pool: GraphTexturePool::new(), internal_w, internal_h, output_w, output_h }
    }

    pub fn allocate(&mut self, device: &wgpu::Device, desc: TextureDescriptor) -> &GraphTexture {
        self.pool.allocate(device, desc)
    }

    pub fn allocate_color(
        &mut self, device: &wgpu::Device, name: &'static str,
        format: wgpu::TextureFormat, size: ResSize, alias_group: Option<&'static str>,
    ) -> &GraphTexture {
        let (w, h) = self.resolve_size(size);
        self.allocate(device, TextureDescriptor {
            name: name.to_string(), format, width: w, height: h,
            depth_or_array_layers: 1, mip_level_count: 1, sample_count: 1,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            alias_group: alias_group.map(|s| s.to_string()),
        })
    }

    pub fn allocate_depth(
        &mut self, device: &wgpu::Device, name: &'static str,
        size: ResSize, alias_group: Option<&'static str>,
    ) -> &GraphTexture {
        let (w, h) = self.resolve_size(size);
        self.allocate(device, TextureDescriptor {
            name: name.to_string(), format: wgpu::TextureFormat::Depth32Float, width: w, height: h,
            depth_or_array_layers: 1, mip_level_count: 1, sample_count: 1,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            alias_group: alias_group.map(|s| s.to_string()),
        })
    }

    fn resolve_size(&self, size: ResSize) -> (u32, u32) {
        match size {
            ResSize::Internal => (self.internal_w, self.internal_h),
            ResSize::Output => (self.output_w, self.output_h),
            ResSize::Absolute(width, height) => (width, height),
        }
    }
}
