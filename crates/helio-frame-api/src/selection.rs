//! The editor's selection, as the frame sees it: which objects are selected
//! (a row per object, written by the host) and the mask every kind draws its
//! selected geometry into ([`crate::Phase::SelectionMask`]), which the
//! selection outline composites over the final image.
//!
//! How others do it: Unreal draws selected primitives into custom depth (a
//! second depth target holding only them) and its editor composite detects
//! the mask's edges, dimming the parts the scene's depth hides. The mask here
//! is that custom depth, filled through a draw phase so the frame names no
//! kind: each kind that can be selected draws its own selected geometry, as
//! it draws its shadow casters.

use helio_core::BufferContract;

/// One row per object entity (its index): non-zero while the object is
/// selected. A kind joins its drawn rows to their owning object to read it;
/// rows past the end, and objects never selected, read as not selected.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ObjectSelectedRow {
    pub selected: u32,
}

/// The host's selection rows ([`ObjectSelectedRow`]), keyed by object entity.
pub const OBJECT_SELECTED: BufferContract<ObjectSelectedRow> =
    BufferContract::new("object_selected");

/// The selected geometry's depth (`Depth32Float`, the internal resolution,
/// reversed Z like the scene's, cleared to [`helio_core::depth::CLEAR`]):
/// [`crate::Phase::SelectionMask`]'s attachment. Drawers write it with the
/// view's projection **without** the temporal jitter (`jitter_frame.xy`), so
/// the outline drawn from it holds still while the scene resolves
/// temporally.
pub const MASK_DEPTH: &str = "selection_depth";
