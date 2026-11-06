//! What the frame's diagnostics (the performance overlay) measure that a kind
//! owns. The overlay is the frame's; what it measures is not: a lighting
//! model's per-tile light loops, a material's shading cost. A kind offers each
//! while a graph builds (a feature's `prepare` step), and the overlay reads
//! what was offered, so it names no kind's buffer and copies no kind's
//! shading. Without an offer the overlay shows what it measures itself.

use helio_core::shader::ShaderSnippet;
use helio_feature_api::BuildEnv;

/// A per-screen-tile count the overlay's shading-cost and tile views read
/// (a lighting model's light-loop count, say).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileCounter {
    /// A buffer resource: one `u32` per `tile_px`-square screen tile of the
    /// internal resolution, row-major.
    pub resource: &'static str,
    pub tile_px: u32,
    /// The kind of the node that writes it: the overlay measures the shading
    /// cost after that node.
    pub after_node: String,
}

/// Marker of a [`SurfaceProbe`]'s snippet.
pub const SURFACE_PROBE_MARKER: &str = "//!use frame_surface_probe";

/// A surface's shading as the overlay's cost probe times it: a snippet under
/// [`SURFACE_PROBE_MARKER`] defining
/// `fn surface_probe(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, base_color: vec3<f32>, roughness: f32, metallic: f32) -> vec3<f32>`
/// (the reflected radiance per unit light), then the snippets it uses.
#[derive(Clone)]
pub struct SurfaceProbe {
    pub snippets: Vec<ShaderSnippet>,
}

#[derive(Default)]
struct Offers {
    counter: Option<TileCounter>,
    probe: Option<SurfaceProbe>,
}

/// Offers the tile counter the overlay reads (the last offer wins).
pub fn offer_tile_counter(env: &mut BuildEnv<'_>, counter: TileCounter) {
    env.shared(Offers::default).counter = Some(counter);
}

/// The tile counter a kind offered, if any.
pub fn tile_counter(env: &mut BuildEnv<'_>) -> Option<TileCounter> {
    env.shared(Offers::default).counter.clone()
}

/// Offers the surface the overlay's cost probe times (the last offer wins).
pub fn offer_surface_probe(env: &mut BuildEnv<'_>, probe: SurfaceProbe) {
    env.shared(Offers::default).probe = Some(probe);
}

/// The surface probe a kind offered, if any.
pub fn surface_probe(env: &mut BuildEnv<'_>) -> Option<SurfaceProbe> {
    env.shared(Offers::default).probe.clone()
}
