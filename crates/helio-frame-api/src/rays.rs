//! The scene as rays see it: the frame's top-level acceleration structure
//! and the buffers that follow its instance order.
//!
//! Whatever builds the TLAS (a meshes plugin's ray-tracing projection)
//! publishes it as the [`SCENE_RAYS`] payload before a frame; the renderer
//! hands the TLAS to that frame's passes (`RenderEnvironment::tlas`) and each
//! buffer as a graph resource under its key, then drops the payload: a frame
//! nobody published for has no TLAS rather than a stale one. Readers trace
//! the TLAS (ray-traced shadows, reflections, GI) and bind the buffers they
//! know by key; [`RAY_TRANSMISSION`] is the frame's, every instance's
//! thin-sheet transmission. The rows a kind keys by instance for its own
//! readers (the meshes' instance rows and proxy pool) are its api's.
//!
//! How others do it: Unreal's `FRayTracingScene` and Bevy Solari's
//! `RaytracingSceneBindings` are built from the scene's primitives inside the
//! renderer, which knows every primitive type. Here the renderer knows none:
//! the builder is the kind's, the TLAS is the frame's, so a second kind of
//! traced content joins the same TLAS without a renderer setter.

/// The frame's TLAS and the buffers that follow it, for one frame.
pub const SCENE_RAYS: helio_core::PayloadContract<SceneRays> =
    helio_core::PayloadContract::new("scene_rays");

/// Graph key of every TLAS instance's thin-sheet transmission. Storage
/// starts with a `[u32; 4]` header: flags, row count, zero, zero. Flag bit 0
/// means BLAS triangle opacity matches the rows (zero RGB opaque, all other
/// rows non-opaque); with flags 0 the shader overrides BLAS opacity. Rows
/// are `[f32; 4]`: finite linear RGB in [0, 1], then zero padding. Opaque
/// instances use RGB zero. Absent: every hit is opaque.
pub const RAY_TRANSMISSION: &str = "ray_transmission";

/// Marker opting a shader into [`RAY_TRANSMISSION_SNIPPET`].
pub const RAY_TRANSMISSION_MARKER: &str = "//!use frame_ray_transmission";
/// [`RAY_TRANSMISSION`]'s WGSL: `RayTransmissionData`, the header flag
/// `RAY_TRANSMISSION_OPACITY_MATCHES`, `ray_transmission_count()` and
/// `ray_transmission_of(instance)`, for a shader binding the buffer as
/// `ray_transmission`.
pub const RAY_TRANSMISSION_SNIPPET: helio_core::shader::ShaderSnippet =
    helio_core::wgsl_snippet!(RAY_TRANSMISSION_MARKER, "shaders/ray_transmission.wgsl");

/// One frame's ray-traced scene ([`SCENE_RAYS`]).
#[derive(Clone)]
pub struct SceneRays {
    pub tlas: wgpu::Tlas,
    /// Buffers indexed by the TLAS's instance order (or addressed by its
    /// geometry), each published under its graph key for the frame:
    /// [`RAY_TRANSMISSION`], and whatever a kind's readers take by its own
    /// keys.
    pub buffers: Vec<(&'static str, wgpu::Buffer)>,
}

impl SceneRays {
    pub fn new(tlas: wgpu::Tlas) -> Self {
        Self {
            tlas,
            buffers: Vec::new(),
        }
    }

    /// Publishes `buffer` under `key` beside the TLAS.
    pub fn with(mut self, key: &'static str, buffer: wgpu::Buffer) -> Self {
        self.buffers.push((key, buffer));
        self
    }

    /// The buffer published under `key`.
    pub fn buffer(&self, key: &str) -> Option<&wgpu::Buffer> {
        self.buffers
            .iter()
            .find(|(candidate, _)| *candidate == key)
            .map(|(_, buffer)| buffer)
    }
}
