//! Resource declaration and lifetime management.
//!
//! This module provides types for declaring resource dependencies between passes.
//! This is a **future feature** for automatic resource lifetime management and pass reordering.
//!
//! # Design Pattern: Resource Dependencies (Future)
//!
//! Passes will declare resource dependencies via `declare_resources()`:
//!
//! ```text
//! GBufferPass:
//!   writes: ["gbuffer_albedo", "gbuffer_normal", "gbuffer_depth"]
//!
//! DeferredLightPass:
//!   reads: ["gbuffer_albedo", "gbuffer_normal", "gbuffer_depth"]
//!   writes: ["final_color"]
//!
//! BloomPass:
//!   reads: ["final_color"]
//!   writes: ["bloom_blur"]
//! ```
//!
//! The graph will use this information to:
//! 1. **Reorder passes**: Automatically determine execution order based on dependencies
//! 2. **Manage lifetimes**: Allocate/deallocate resources only when needed
//! 3. **Parallelize**: Execute independent passes in parallel (GPU queue parallelism)
//!
//! # Performance (Future)
//!
//! - **Automatic parallelism**: Independent passes run on separate GPU queues
//! - **Memory optimization**: Resources allocated/freed based on lifetime analysis
//! - **Validation**: Detect missing dependencies at graph construction time
//!
//! # Example (Future API)
//!
//! ```rust,no_run
//! use helio_v3::{RenderPass, PassContext, PrepareContext, Result};
//! use helio_v3::graph::ResourceBuilder;
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
//!         // Write final color
//!         builder.write("final_color");
//!     }
//!
//!     fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
//!         // ... render code ...
//!         Ok(())
//!     }
//! }
//! ```

/// Resource dependency builder (future feature).
///
/// Used by passes to declare resource dependencies via `declare_resources()`.
/// The graph will use this information for automatic pass reordering and resource lifetime management.
///
/// # Design (Future)
///
/// Passes declare read/write dependencies:
/// - `read(name)`: Pass reads from this resource
/// - `write(name)`: Pass writes to this resource
///
/// The graph builds a dependency DAG and determines execution order.
///
/// # Example (Future API)
///
/// ```rust,no_run
/// # use helio_v3::graph::ResourceBuilder;
/// # let mut builder = ResourceBuilder::new();
/// builder.read("gbuffer_albedo");
/// builder.read("gbuffer_normal");
/// builder.write("final_color");
/// ```
pub struct ResourceBuilder {
    // Future: Track read/write dependencies
    // reads: Vec<String>,
    // writes: Vec<String>,
}

impl ResourceBuilder {
    /// Creates a new resource builder.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {}
    }

    /// Declares that the pass reads from a resource.
    ///
    /// # Parameters
    ///
    /// - `name`: Resource name (e.g., "gbuffer_albedo")
    ///
    /// # Example (Future API)
    ///
    /// ```rust,no_run
    /// # use helio_v3::graph::ResourceBuilder;
    /// # let mut builder = ResourceBuilder::new();
    /// builder.read("gbuffer_albedo");
    /// builder.read("gbuffer_normal");
    /// ```
    pub fn read(&mut self, _name: &str) {
        // Future: Track read dependency
    }

    /// Declares that the pass writes to a resource.
    ///
    /// # Parameters
    ///
    /// - `name`: Resource name (e.g., "final_color")
    ///
    /// # Example (Future API)
    ///
    /// ```rust,no_run
    /// # use helio_v3::graph::ResourceBuilder;
    /// # let mut builder = ResourceBuilder::new();
    /// builder.write("final_color");
    /// ```
    pub fn write(&mut self, _name: &str) {
        // Future: Track write dependency
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
