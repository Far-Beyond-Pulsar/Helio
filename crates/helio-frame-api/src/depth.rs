//! The frame's depth resources besides the G-buffer's depth
//! ([`crate::gbuffer::DEPTH`]): what the renderer and the spine publish for
//! every kind to read by these names.

pub use crate::gbuffer::DEPTH;

/// The scene depth at output resolution when the frame renders at a lower
/// internal resolution (written by the renderer; absent otherwise, when
/// readers fall back to [`DEPTH`]). Overlays drawn after the upscale test
/// against it.
pub const FULL_RES_DEPTH: &str = "full_res_depth";

/// The depth pyramid (Hi-Z) the spine builds at [`crate::Slot::Visibility`]
/// from the previous frame's [`DEPTH`]: each texel the farthest depth in its
/// footprint (`helio_hiz_occludes` in the prelude tests against it; the
/// spine's `hiz_trace` snippet marches it).
pub const HIZ: &str = "hiz";
/// The same pyramid holding the nearest depth below each texel.
pub const HIZ_NEAREST: &str = "hiz_nearest";
/// The sampler the pyramid is read with.
pub const HIZ_SAMPLER: &str = "hiz_sampler";
