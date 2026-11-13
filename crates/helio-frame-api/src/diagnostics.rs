//! The frame's diagnostics: the performance overlay's modes, and what a kind
//! owns that the diagnostics read or draw.
//!
//! The overlay is the frame's; what it measures is not: a lighting model's
//! per-tile light loops, a material's shading cost. A kind offers a count
//! while a graph builds (a feature's `prepare` step) and draws its own view
//! for the overlay mode that names it ([`PerfOverlayMode::ShaderComplexity`])
//! or as a debug view, with the frame's [`HEATMAP_SNIPPET`] colours. So the
//! overlay names no kind's buffer and models no kind's shading; without the
//! kind there is no such view, and nothing stands in for it.

use helio_core::shader::ShaderSnippet;
use helio_feature_api::BuildEnv;

/// What the performance overlay shows. [`ShaderComplexity`] is drawn by the
/// kind that owns shading cost, not by the overlay.
///
/// [`ShaderComplexity`]: PerfOverlayMode::ShaderComplexity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum PerfOverlayMode {
    #[default]
    Disabled = 0,
    PassOverdraw = 1,
    ShaderComplexity = 2,
    TileLightCount = 3,
    PassOutput = 4,
}

/// Marker opting a shader into [`HEATMAP_SNIPPET`].
pub const HEATMAP_MARKER: &str = "//!use frame_heatmap";
/// `heatmap_color(value)` (blue to red over 0..1) and `heatmap_scale_bar`
/// (the legend strip along the display's bottom edge): the ramp every
/// diagnostic view draws with.
pub const HEATMAP_SNIPPET: ShaderSnippet =
    helio_core::wgsl_snippet!(HEATMAP_MARKER, "shaders/heatmap.wgsl");

/// A per-screen-tile count the overlay's tile view and a kind's shading-cost
/// view read (a lighting model's light-loop count, say).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileCounter {
    /// A buffer resource: one `u32` per `tile_px`-square screen tile of the
    /// internal resolution, row-major.
    pub resource: &'static str,
    pub tile_px: u32,
}

#[derive(Default)]
struct Offers {
    counter: Option<TileCounter>,
}

/// Offers the tile counter the diagnostics read (the last offer wins).
pub fn offer_tile_counter(env: &mut BuildEnv<'_>, counter: TileCounter) {
    env.shared(Offers::default).counter = Some(counter);
}

/// The tile counter a kind offered, if any.
pub fn tile_counter(env: &mut BuildEnv<'_>) -> Option<TileCounter> {
    env.shared(Offers::default).counter.clone()
}
