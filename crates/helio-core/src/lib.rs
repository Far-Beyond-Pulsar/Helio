//! Helio's gpu and graph layer (`architecture/layers.toml`, layer `core`):
//! what every pass, feature and host builds on, with no object kind in it.
//!
//! - The render graph ([`graph`], [`RenderGraph`]): passes ([`RenderPass`],
//!   [`traits`]), named resources and their lifetimes, render-pass fusion,
//!   scene membership (a pass schedules only while the scene holds its
//!   content, `RenderPass::is_member`) and resizes in place.
//! - Pass contexts ([`PassContext`], [`PrepareContext`]) and command recording
//!   ([`cmd`]).
//! - Views ([`view`]): the primary view and the captures features ask for.
//! - Scene input ([`scene_input`]): [`BufferContract`], [`PoolContract`] and
//!   [`PayloadContract`], the contracts components publish under and passes
//!   read through [`SceneBufferProjection`]; scene derivations
//!   ([`scene_derivation`]) that turn authored rows into the rows passes read.
//! - Uploads and compiles: a frame's buffer writes staged through one ring
//!   ([`upload::FrameUploads`]), pipeline compiles off the frame
//!   ([`compile::Compiling`]), GPU objects kept across frames ([`gpu_cache`]).
//! - Shaders ([`shader`]): [`include_wgsl!`], the shared prelude and the
//!   `//!use` snippet system, through which a pass includes another owner's
//!   WGSL instead of copying it.
//! - Profiling ([`profiling`], [`cpu_scope!`]): GPU timestamps per pass
//!   ([`RenderTimingSnapshot`]) and CPU scopes into the Pulsar profiler.
//! - Frame facts every pass shares: the depth convention ([`depth`]: reversed
//!   Z, infinite far plane), camera uniforms ([`camera`]), temporal sampling
//!   ([`temporal`]) and named frame stats ([`stats`]).
//!
//! The repository README tours the layers above this one; how to write a
//! pass is `docs/custom-render-pass.md`.

/// Whether reflection features (screen-space, planar, and cubemap-capture
/// reflections) are available on this target.
///
/// Apple targets render them incorrectly, so the whole reflection group is
/// compiled out there: `SsrPass` and `PlanarReflectionPass` are left out of the
/// graph, and a lighting model reading this skips both composites and the
/// reflection-capture cube array. Everything else it lights (direct light,
/// ambient, indirect diffuse) is untouched.
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub const REFLECTIONS_SUPPORTED: bool = false;
/// See the macOS/iOS variant above.
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub const REFLECTIONS_SUPPORTED: bool = true;

pub mod acceleration;
pub mod camera;
pub mod cmd;
pub mod compile;
pub mod context;
pub mod depth;
pub mod entity;
pub mod error;
pub mod frame_demands;
pub mod frame_input_publisher;
mod frame_inputs;
pub mod frame_storage;
pub mod gpu_cache;
pub mod graph;
pub mod instance;
pub mod movability;
pub mod profiling;
pub mod radix_sort;
pub mod registry;
pub mod render_environment;
pub mod resource_keys;
pub mod scene_derivation;
pub mod scene_input;
pub mod scene_liveness;
pub mod shader;
pub mod stats;
pub mod temporal;
pub mod texture_mips;
pub mod traits;
pub mod upload;
pub mod view;

/// RAII CPU span that lands in the engine flamegraph's profiler, on the thread
/// that creates it. A no-op unless the `profiling` feature is enabled (and always
/// on wasm, where the profiler is not built). The feature check lives here, not
/// in the [`cpu_scope!`] expansion, so callers need no matching cfg.
pub struct CpuScope {
    #[cfg(all(not(target_arch = "wasm32"), feature = "profiling"))]
    _scope: ::profiling::ProfileScope,
}

impl CpuScope {
    #[inline]
    pub fn new(name: &'static str) -> Self {
        #[cfg(all(not(target_arch = "wasm32"), feature = "profiling"))]
        {
            Self {
                _scope: ::profiling::ProfileScope::new_static(name),
            }
        }
        #[cfg(not(all(not(target_arch = "wasm32"), feature = "profiling")))]
        {
            let _ = name;
            Self {}
        }
    }
}

/// Open a [`CpuScope`] named `$name` that ends with the enclosing block.
#[macro_export]
macro_rules! cpu_scope {
    ($name:expr) => {
        let _helio_cpu_scope = $crate::CpuScope::new($name);
    };
}

// Generic types owned directly by helio-core: graph scheduling, the open
// resource registry, and render primitives (camera, mobility) that name no
// specific pass or scene-object type. Pass-specific GPU shapes (lights,
// materials, geometry instances, ...) live in their owning `helio-pass-*`
// crate, never here.
pub use camera::GpuCameraUniforms;
pub use cmd::{CommandRecorder, ComputeCmds, RenderCmds};
pub use movability::Movability;
pub use registry::{ResourceKey, ResourceRegistry, Tracked, ViewGroup};
pub use render_environment::RenderEnvironment;
pub use stats::{FrameStats, StatSlot, StatsSnapshot};
pub use view::{
    ViewBudget, ViewBuild, ViewContext, ViewDescription, ViewGraphSource, ViewKind, ViewRequest,
};

// Re-export managers
pub use crate::acceleration::{
    AccelerationError, BlasGeometry, BlasManager, BlasSegment, TlasInstanceInput, TlasManager,
};
// Re-export core types
pub use context::{PassContext, PrepareContext};
pub use entity::Entity;
pub use error::{Error, Result};
pub use frame_demands::{is_demanded, FrameDemands, FRAME_DEMANDS};
pub use frame_input_publisher::{FrameInputContext, FrameInputPublisher};
pub use frame_inputs::{CoordinateSpaceAllocator, CoordinateSpacesFrameData, RenderFrameInputs};
pub use frame_storage::RenderFrameStorage;
pub use graph::{
    check_routing, BindingOverrideBuilder, DebugPassInfo, DebugResourceInfo, DeclaredKind,
    DeclaredResource, FinishSegment, FrameDebugData, GraphTimelineData, GraphTimelinePass,
    PassDeclaration, PassIdentity, PipelineFormatCache, PipelineFormatKey, RenderGraph, RenderSize,
    RoutingIssue, RoutingKind, RoutingRead, RoutingSeverity, RoutingUnit, SceneContent, SwapPlan,
    SwapPolicy,
};
pub use instance::wgpu_instance_descriptor;

/// Descriptor-indexed texture arrays: a table of textures a shader indexes
/// per pixel (a kind's bindless table: the materials'). The pair of features
/// is one capability; a device with only one of them uses none.
pub const TEXTURE_BINDING_ARRAYS: wgpu::Features = wgpu::Features::TEXTURE_BINDING_ARRAY
    .union(wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING);

/// The most textures a texture table bound as binding arrays holds, whatever
/// the device offers; the device's own binding-array limits can lower it (see
/// [`texture_table_capacity`]).
///
/// The table is sized once (each reader compiles its binding array at this
/// size and binds every slot), so its per-reader cost is a bind group of
/// this many texture descriptors, rebuilt when a texture is added: the cap
/// bounds that cost while holding a large level's textures (thousands).
/// Mobile GPUs (Apple, Android/Adreno) and the browser keep the
/// WebGPU baseline of 16: their shader compilers and descriptor limits choke
/// on wide binding arrays.
#[cfg(not(any(
    target_arch = "wasm32",
    target_os = "macos",
    target_os = "ios",
    target_os = "android"
)))]
pub const TEXTURE_TABLE_LIMIT: usize = 4096;
#[cfg(any(
    target_arch = "wasm32",
    target_os = "macos",
    target_os = "ios",
    target_os = "android"
))]
pub const TEXTURE_TABLE_LIMIT: usize = 16;

/// The most sampled textures (and samplers) per shader stage a host asks a
/// device for: the size cap of a texture table bound one binding per texture
/// (no binding arrays). WebGPU's baseline guarantees 16; native Vulkan/D3D12
/// offer 256. Binding arrays are counted against the device's binding-array
/// limits instead, which a host takes as the adapter offers them.
#[cfg(not(any(
    target_arch = "wasm32",
    target_os = "macos",
    target_os = "ios",
    target_os = "android"
)))]
pub const TEXTURE_TABLE_BINDINGS_LIMIT: usize = 256;
#[cfg(any(
    target_arch = "wasm32",
    target_os = "macos",
    target_os = "ios",
    target_os = "android"
))]
pub const TEXTURE_TABLE_BINDINGS_LIMIT: usize = 16;

/// Sampled-texture slots a stage that binds the texture table one binding per
/// texture (no binding arrays) keeps for its own inputs: the most any reader
/// needs, a G-buffer reader's five (depth and four G-buffer targets).
pub const TEXTURE_TABLE_RESERVE: usize = 5;

/// Whether a device binds the texture table as one indexed binding array
/// ([`TEXTURE_BINDING_ARRAYS`]); otherwise one binding per texture. WebGPU
/// in a browser never does.
pub fn texture_table_uses_binding_arrays(features: wgpu::Features) -> bool {
    !cfg!(target_arch = "wasm32") && features.contains(TEXTURE_BINDING_ARRAYS)
}

/// How many textures the texture table holds on a device with `features`
/// and `limits`: every stage that binds it can address all of them. The
/// scene's texture store is this big, and each reader's bindings are.
///
/// The table is read through one sampler, an ordinary binding beside it (a
/// kind's table owns which), never a sampler per texture: samplers are a
/// handful of filter states, not per-texture resources, so the table's size
/// never counts against a device's sampler limits (D3D12's shader-visible
/// sampler heap is 2048).
///
/// With binding arrays the table is one array of textures, counted against
/// `max_binding_array_elements_per_shader_stage` and capped at
/// [`TEXTURE_TABLE_LIMIT`]. Without, each texture takes a binding beside the
/// reader's own sampled inputs ([`TEXTURE_TABLE_RESERVE`]), in one bind group
/// with the sampler and two bindings of the reader's, capped at
/// [`TEXTURE_TABLE_BINDINGS_LIMIT`].
pub fn texture_table_capacity(features: wgpu::Features, limits: &wgpu::Limits) -> usize {
    let capacity = if texture_table_uses_binding_arrays(features) {
        (limits.max_binding_array_elements_per_shader_stage as usize).min(TEXTURE_TABLE_LIMIT)
    } else {
        (limits.max_sampled_textures_per_shader_stage as usize)
            .saturating_sub(TEXTURE_TABLE_RESERVE)
            .min((limits.max_bindings_per_bind_group as usize).saturating_sub(3))
            .min(TEXTURE_TABLE_BINDINGS_LIMIT)
    };
    assert!(
        capacity > 0,
        "Helio requires at least one sampled texture binding"
    );
    capacity
}
pub use profiling::{GpuTimingAvailability, Profiler, RenderPassTiming, RenderTimingSnapshot};
pub use scene_derivation::{
    run_scene_derivations, SceneDerivation, SceneDerivationContext, SceneDerivationFactory,
    SceneDerivationOutput,
};
pub use scene_input::{
    BufferContract, BufferHandle, BufferKey, ContractError, FlushedRows, PayloadContract,
    PoolContract, SceneBufferProjection, SceneInput, ScenePayloads, CAMERA_BUFFER_KEY,
    ENTITY_GENERATIONS_KEY,
};
pub use scene_liveness::SceneBufferLiveness;
pub use shader::{populate_bind_group_entries, CachedBindGroup, ReflectedShader};
pub use traits::{
    AsAny, DebugViewDescriptor, DebugViewLegend, DebugViewLegendStop, MaybeSend, MaybeSync,
    RenderPass,
};
