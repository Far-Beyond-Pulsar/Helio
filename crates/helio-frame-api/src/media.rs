//! Participating media: what something drawn after the media slot reads to
//! put the medium between the eye and itself.
//!
//! A media feature integrates the medium along each view ray at
//! [`crate::Slot::Media`] and publishes its result with [`publish`] when its
//! node builds; a reader that composites at its own depth (transparency)
//! asks [`current`] when it builds and binds a stand-in when nothing
//! published. The publisher names its resources; neither side knows the
//! other.
//!
//! The contract ([`MediaResources`]):
//! - `grid`: a 3D texture over the view frustum (`Rgba16Float`): `rgb` the
//!   light scattered toward the eye between the eye and the slice, `a` the
//!   transmittance over that distance. Its X and Y are the unjittered
//!   screen; its Z slices are logarithmic in view depth from 0.5 m to the
//!   far distance (a slice's centre at `(slice + 0.5) / slices`).
//! - `parameters`: a uniform buffer whose second `vec4`'s `y` is that far
//!   distance; 0.5 or less means the grid holds no medium this frame.
//!
//! A surface at depth `d` and colour `c` is seen as `c * a + rgb`, sampled
//! at `d`.
//!
//! The grid's depth mapping and that composite are [`SNIPPET`]'s WGSL
//! (`MEDIA_NEAR`, `media_slice_from_view_depth`, `media_view_depth_from_slice`,
//! `media_grid_w`, `media_seen`): the publisher writes slices by it and every
//! reader samples by it, so none keeps its own copy.

use helio_feature_api::BuildEnv;

/// Marker opting a shader into [`SNIPPET`].
pub const MARKER: &str = "//!use frame_media";
/// The grid's depth mapping and the composite (see the module).
pub const SNIPPET: helio_core::shader::ShaderSnippet =
    helio_core::wgsl_snippet!(MARKER, "shaders/media.wgsl");

/// The resources a media feature published (see the module).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MediaResources {
    /// The integrated medium (a pass-published 3D texture view).
    pub grid: &'static str,
    /// Its parameters (a buffer).
    pub parameters: &'static str,
}

struct Published(Option<MediaResources>);

/// Publishes the frame's integrated medium for the nodes built after this
/// one.
pub fn publish(env: &mut BuildEnv<'_>, media: MediaResources) {
    env.shared(|| Published(None)).0 = Some(media);
}

/// The integrated medium nodes built so far published, if any.
pub fn current(env: &mut BuildEnv<'_>) -> Option<MediaResources> {
    env.shared(|| Published(None)).0
}
