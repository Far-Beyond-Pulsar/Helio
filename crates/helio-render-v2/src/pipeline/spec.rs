//! Pipeline specialization constants

/// Pipeline variant type
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub enum PipelineVariant {
    /// Forward rendering (legacy, kept for compatibility/debug)
    Forward,
    /// Depth-only (shadow pass)
    DepthOnly,
    /// Write geometry data to 4 G-buffer textures — no lighting.
    GBufferWrite,
    /// Deferred fullscreen lighting pass — reads G-buffer, runs PBR.
    DeferredLighting,
    /// Deferred rendering (unused alias)
    Deferred,
}
