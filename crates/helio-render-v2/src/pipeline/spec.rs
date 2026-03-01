//! Pipeline specialization constants

/// Pipeline variant type
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub enum PipelineVariant {
    /// Forward rendering
    Forward,
    /// Depth-only (shadow pass)
    DepthOnly,
    /// Deferred rendering (future)
    Deferred,
}
