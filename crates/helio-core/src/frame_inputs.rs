//! Generic per-frame inputs forwarded to render passes by the graph.
//!
//! The renderer facade owns the authored/projection data, while passes own the
//! GPU resources and pass-specific interpretation.  Keeping this small input
//! contract in the core graph avoids making the facade downcast into concrete
//! passes merely to forward coordinate-space state. What a feature's own
//! passes need from the host each frame (live counts of what it projects)
//! is the feature's payload ([`crate::PayloadContract`]), not a field here.

/// Host-provided projection inputs that may affect pass-owned GPU state.
///
/// `coordinate_spaces` uses slot zero for world space. The graph only
/// broadcasts the data and has no pass-specific knowledge.
#[derive(Clone, Copy, Debug)]
pub struct RenderFrameInputs<'a> {
    pub coordinate_spaces: &'a [glam::Mat4],
    /// The host is rendering a view for an XR headset. Passes may use this to
    /// select stereo-friendly quality paths without depending on an XR crate.
    pub xr_mode: bool,
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

/// Hands out slots in the coordinate-space table.
///
/// Several producers put spaces in one table: the scene's rigid frames,
/// placed sublevels, and what features project (the matrix mapping what a
/// view shows into where it shows it). They must not number the table independently, so
/// whoever owns the table (the host) implements this and each producer asks it
/// for its slots.
pub trait CoordinateSpaceAllocator {
    /// The slot of `matrix` (a space-to-world transform; slot 0 is the world and
    /// is never handed out), reusing the slot of an equal matrix already given
    /// out. `None` when the table is full.
    fn space_for(&mut self, matrix: glam::Mat4) -> Option<u32>;
}
