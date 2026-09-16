//! Generic per-frame environment parameters shared by many shading passes.
//!
//! Published by the `Renderer` under the well-known `"render_environment"`
//! [`crate::ResourceKey`]. Deliberately holds only values that describe the
//! frame in general (clear color, ambient fallback, an optional hardware
//! ray-tracing acceleration structure) — nothing here names a specific pass
//! or scene-object type. A technique-specific environment parameter (for
//! example a radiance-cascades GI volume extent) is published as its own
//! resource by the pass crate that owns that technique, not folded in here.

/// Backend material-texture bindings used by passes that sample material rows.
///
/// The material rows themselves are SceneDB component data. This value only
/// describes the backend descriptor bindings needed to sample any referenced
/// textures; it is not a scene container or an ownership model for materials.
#[derive(Clone, Copy)]
pub struct RenderEnvironment<'a> {
    pub clear_color: [f32; 4],
    pub ambient_color: [f32; 3],
    pub ambient_intensity: f32,
    /// Hardware ray tracing TLAS, if available. None on non-RT hardware or WASM.
    pub tlas: Option<&'a wgpu::Tlas>,
}
