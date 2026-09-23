//! Generic per-frame inputs forwarded to render passes by the graph.
//!
//! The renderer facade owns the authored/projection data, while passes own the
//! GPU resources and pass-specific interpretation.  Keeping this small input
//! contract in the core graph avoids making the facade downcast into concrete
//! passes merely to forward portal/sublevel state.

/// Host-provided projection inputs that may affect pass-owned GPU state.
///
/// `coordinate_spaces` uses slot zero for world space.  `projection_counts`
/// is an optional pair of live counts for a producer/consumer projection;
/// the meaning of each lane is intentionally left to the pass family that
/// consumes it.  The graph only broadcasts the data and has no pass-specific
/// knowledge.
#[derive(Clone, Copy, Debug)]
pub struct RenderFrameInputs<'a> {
    pub coordinate_spaces: &'a [glam::Mat4],
    pub projection_counts: Option<[u32; 2]>,
}

/// GPU buffers containing the current and previous coordinate-space tables.
///
/// The owning pass publishes this view before execution; consumers only see a
/// generic borrowed buffer pair and do not need to know which pass allocated
/// it.
#[derive(Clone, Copy)]
pub struct CoordinateSpacesFrameData<'a> {
    pub coordinate_spaces: &'a wgpu::Buffer,
    pub coordinate_spaces_prev: &'a wgpu::Buffer,
}
