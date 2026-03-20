//! Resource declaration and lifetime management.
//!
//! This module provides types for declaring resource dependencies between passes,
//! enabling automatic texture allocation and data flow routing.
//!
//! # Design Pattern: Automatic Resource Management
//!
//! Passes declare resource dependencies via `declare_resources()`:
//!
//! ```text
//! GBufferPass:
//!   writes: ["gbuffer_albedo", "gbuffer_normal", "gbuffer_depth"]
//!
//! DeferredLightPass:
//!   reads: ["gbuffer_albedo", "gbuffer_normal", "gbuffer_depth"]
//!   writes: ["hdr_main"]
//!
//! BloomPass:
//!   reads: ["hdr_main"]
//!   writes: ["bloom_result"]
//!
//! FxaaPass:
//!   reads: ["bloom_result"]
//!   writes: "final" (special name for swapchain)
//! ```
//!
//! The graph automatically:
//! 1. **Creates transient textures** based on write declarations
//! 2. **Routes data flow** through FrameResources (write → read)
//! 3. **Validates dependencies** at graph construction time
//! 4. **Optimizes lifetimes** (future: reuse texture memory)
//!
//! # Performance Guarantees
//!
//! - **Zero-copy**: Resources borrowed via references, never cloned
//! - **Zero allocations** in render loop (all pre-allocated during graph construction)
//! - **Validation at build time**: Missing dependencies caught early
//!
//! # Example
//!
//! ```rust,no_run
//! use helio_v3::{RenderPass, PassContext, PrepareContext, Result};
//! use helio_v3::graph::{ResourceBuilder, ResourceFormat, ResourceSize};
//!
//! struct DeferredLightPass;
//!
//! impl RenderPass for DeferredLightPass {
//!     fn name(&self) -> &'static str {
//!         "DeferredLightPass"
//!     }
//!
//!     fn declare_resources(&self, builder: &mut ResourceBuilder) {
//!         // Read GBuffer outputs
//!         builder.read("gbuffer_albedo");
//!         builder.read("gbuffer_normal");
//!         builder.read("gbuffer_depth");
//!
//!         // Write HDR color buffer
//!         builder.write_color(
//!             "hdr_main",
//!             ResourceFormat::Rgba16Float,
//!             ResourceSize::MatchSurface,
//!         );
//!     }
//!
//!     fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
//!         // Access "hdr_main" via ctx.get_color_target()
//!         Ok(())
//!     }
//! }
//! ```

/// Texture format specification for transient resources.
///
/// Defines the pixel format of a graph-managed texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceFormat {
    /// Rgba16Float — HDR color buffer (standard for post-processing chains)
    Rgba16Float,
    /// Rgba8UnormSrgb — LDR color with sRGB gamma
    Rgba8UnormSrgb,
    /// Bgra8UnormSrgb — Surface format (common swapchain format)
    Bgra8UnormSrgb,
    /// R16Float — Single-channel HDR (e.g., SSAO output)
    R16Float,
    /// R8Unorm — Single-channel LDR (e.g., shadow mask)
    R8Unorm,
    /// Depth32Float — Depth buffer
    Depth32Float,
}

impl ResourceFormat {
    /// Converts to wgpu::TextureFormat.
    pub fn to_wgpu(self) -> wgpu::TextureFormat {
        match self {
            Self::Rgba16Float => wgpu::TextureFormat::Rgba16Float,
            Self::Rgba8UnormSrgb => wgpu::TextureFormat::Rgba8UnormSrgb,
            Self::Bgra8UnormSrgb => wgpu::TextureFormat::Bgra8UnormSrgb,
            Self::R16Float => wgpu::TextureFormat::R16Float,
            Self::R8Unorm => wgpu::TextureFormat::R8Unorm,
            Self::Depth32Float => wgpu::TextureFormat::Depth32Float,
        }
    }
}

/// Size specification for transient resources.
///
/// Determines the dimensions of a graph-managed texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceSize {
    /// Match the surface/swapchain resolution (default for fullscreen passes)
    MatchSurface,
    /// Absolute size in pixels
    Absolute { width: u32, height: u32 },
    /// Scaled relative to surface (e.g., half-res for bloom downsampling)
    Scaled { divisor: u32 },
}

/// Resource declaration — describes a texture that a pass reads or writes.
///
/// Used by ResourceBuilder to track dependencies and create transient textures.
///
/// # Performance
///
/// - **Zero-copy**: Stored inline in Vec, no heap allocations per declaration
/// - **Small footprint**: ~48 bytes per declaration
#[derive(Debug, Clone, PartialEq)]
pub struct ResourceDecl {
    /// Resource name (e.g., "hdr_main", "gbuffer_albedo")
    ///
    /// Names are matched across passes to route data flow.
    /// Use `'static str` to avoid allocations.
    pub name: &'static str,

    /// Texture format (only for writes; reads infer from writer)
    pub format: Option<ResourceFormat>,

    /// Texture size (only for writes; reads infer from writer)
    pub size: Option<ResourceSize>,

    /// Access mode
    pub access: ResourceAccess,
}

/// Resource access mode — read or write.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceAccess {
    /// Pass reads from this resource (must be written by an earlier pass)
    Read,
    /// Pass writes to this resource (creates a new output)
    Write,
}

/// Resource dependency builder.
///
/// Used by passes to declare resource dependencies via `declare_resources()`.
/// The graph uses this to create transient textures and route data flow.
///
/// # Design
///
/// Passes declare read/write dependencies:
/// - `read(name)`: Pass reads from this resource (must be written earlier)
/// - `write_color(name, format, size)`: Pass writes a color texture
/// - `write_depth(name, size)`: Pass writes a depth texture
///
/// The graph analyzes declarations to:
/// 1. Create transient textures for all writes
/// 2. Validate that all reads have a matching write
/// 3. Populate FrameResources with texture views
///
/// # Example
///
/// ```rust,no_run
/// # use helio_v3::graph::{ResourceBuilder, ResourceFormat, ResourceSize};
/// # let mut builder = ResourceBuilder::new();
/// builder.read("gbuffer_albedo");
/// builder.read("gbuffer_normal");
/// builder.write_color("hdr_main", ResourceFormat::Rgba16Float, ResourceSize::MatchSurface);
/// ```
///
/// # Performance
///
/// - **Pre-allocated**: Vec capacity set during graph construction
/// - **Zero-copy**: Stores `&'static str` names, no allocations
pub struct ResourceBuilder {
    /// All declared resource dependencies (reads + writes)
    ///
    /// Collected during `declare_resources()` calls, then analyzed by RenderGraph.
    declarations: Vec<ResourceDecl>,
}

impl ResourceBuilder {
    /// Creates a new resource builder with pre-allocated capacity.
    ///
    /// # Performance
    ///
    /// - Pre-allocates space for ~8 declarations (typical pass has 2-4)
    /// - Zero heap allocations for small passes
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            declarations: Vec::with_capacity(8),
        }
    }

    /// Declares that the pass reads from a resource.
    ///
    /// The resource must be written by an earlier pass in the graph.
    /// Format and size are inferred from the writer.
    ///
    /// # Parameters
    ///
    /// - `name`: Resource name (e.g., "gbuffer_albedo", "hdr_main")
    ///
    /// # Panics
    ///
    /// Graph construction will panic if no pass writes this resource.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use helio_v3::graph::ResourceBuilder;
    /// # let mut builder = ResourceBuilder::new();
    /// builder.read("gbuffer_albedo");
    /// builder.read("gbuffer_normal");
    /// ```
    ///
    /// # Performance
    ///
    /// - **O(1)**: Pushes to pre-allocated Vec
    /// - **Zero allocations**: Uses `&'static str`
    pub fn read(&mut self, name: &'static str) {
        self.declarations.push(ResourceDecl {
            name,
            format: None,
            size: None,
            access: ResourceAccess::Read,
        });
    }

    /// Declares that the pass writes a color texture.
    ///
    /// The graph will create a transient texture with the specified format and size.
    /// Other passes can read this resource by name.
    ///
    /// # Parameters
    ///
    /// - `name`: Resource name (e.g., "hdr_main", "bloom_result")
    /// - `format`: Pixel format (e.g., Rgba16Float for HDR)
    /// - `size`: Texture dimensions (e.g., MatchSurface for fullscreen)
    ///
    /// # Special Names
    ///
    /// - `"final"`: Writes to the swapchain (use surface format)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use helio_v3::graph::{ResourceBuilder, ResourceFormat, ResourceSize};
    /// # let mut builder = ResourceBuilder::new();
    /// builder.write_color("hdr_main", ResourceFormat::Rgba16Float, ResourceSize::MatchSurface);
    /// builder.write_color("bloom_half", ResourceFormat::Rgba16Float, ResourceSize::Scaled { divisor: 2 });
    /// ```
    ///
    /// # Performance
    ///
    /// - **O(1)**: Pushes to pre-allocated Vec
    /// - **Zero allocations**: Uses `&'static str`
    pub fn write_color(
        &mut self,
        name: &'static str,
        format: ResourceFormat,
        size: ResourceSize,
    ) {
        self.declarations.push(ResourceDecl {
            name,
            format: Some(format),
            size: Some(size),
            access: ResourceAccess::Write,
        });
    }

    /// Declares that the pass writes a depth texture.
    ///
    /// Convenience wrapper for `write_color` with Depth32Float format.
    ///
    /// # Parameters
    ///
    /// - `name`: Resource name (e.g., "depth_prepass")
    /// - `size`: Texture dimensions (typically MatchSurface)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use helio_v3::graph::{ResourceBuilder, ResourceSize};
    /// # let mut builder = ResourceBuilder::new();
    /// builder.write_depth("depth_prepass", ResourceSize::MatchSurface);
    /// ```
    pub fn write_depth(&mut self, name: &'static str, size: ResourceSize) {
        self.write_color(name, ResourceFormat::Depth32Float, size);
    }

    /// Returns all declared resources (consumed by RenderGraph).
    ///
    /// # Performance
    ///
    /// - **O(1)**: Moves Vec, no cloning
    pub(crate) fn finish(self) -> Vec<ResourceDecl> {
        self.declarations
    }

    /// Returns references to all declarations (for inspection).
    ///
    /// # Performance
    ///
    /// - **O(1)**: Returns slice reference, zero-copy
    pub fn declarations(&self) -> &[ResourceDecl] {
        &self.declarations
    }
}

/// Resource lifetime handle (future feature).
///
/// Represents a reference to a graph-managed resource. The graph will allocate/deallocate
/// resources based on lifetime analysis.
///
/// # Design (Future)
///
/// Resources are reference-counted by the graph:
/// - Created when first written
/// - Kept alive while any pass reads from it
/// - Freed when no longer needed
///
/// # Example (Future API)
///
/// ```rust,no_run
/// use helio_v3::graph::ResourceHandle;
///
/// let gbuffer_albedo = ResourceHandle::named("gbuffer_albedo");
/// let gbuffer_normal = ResourceHandle::named("gbuffer_normal");
/// ```
pub struct ResourceHandle {
    // Future: Resource lifetime tracking
    // name: String,
    // refcount: Arc<AtomicUsize>,
}

impl ResourceHandle {
    /// Creates a named resource handle.
    ///
    /// # Parameters
    ///
    /// - `name`: Resource name (e.g., "gbuffer_albedo")
    ///
    /// # Example (Future API)
    ///
    /// ```rust,no_run
    /// use helio_v3::graph::ResourceHandle;
    ///
    /// let handle = ResourceHandle::named("gbuffer_albedo");
    /// ```
    pub fn named(_name: &str) -> Self {
        Self {
            // Future: Initialize resource tracking
        }
    }
}
