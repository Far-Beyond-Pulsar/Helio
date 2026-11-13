//! The scene colour chain: scene-linear -> media -> anti-aliasing -> lens ->
//! tone map -> output.
//!
//! Post features never name each other's outputs. A node that works on the
//! scene colour reads [`current`] when it builds and, if it writes a new
//! texture, calls [`advance`] with its name; the next node reads that. Nodes
//! build in execution order, so the chain is the order the preset placed
//! them in. Names stay `&'static str` (graph resources are), and each pass
//! still owns its output texture.
//!
//! **Alpha.** The chain's alpha is the scene's coverage and nothing else:
//! 1 wherever anything is drawn, the background colour's alpha elsewhere
//! (opaque unless a host clears to a transparent background on purpose). So
//! the frame's output is opaque by default, and a host that composites it by
//! alpha (the editor's viewport) shows exactly the frame. A pass on the chain
//! forwards the alpha it read or writes 1; it never puts a mask or its own
//! bookkeeping there (TSR keeps its transparency coverage in its history,
//! not in its output). A pass that put a mask in alpha once made every frame
//! transparent: the editor's viewport showed the UI behind it.

use helio_feature_api::BuildEnv;

/// The lit scene-linear colour: cleared by the background, written by the
/// lighting model, drawn over by overlays.
pub const SCENE_COLOR: &str = "pre_aa";

/// The scene colour nodes built so far left as current, and the earlier
/// link it may stand in for (see [`advance_aliasing`]).
struct ColorChain(&'static str, Option<&'static str>);

fn chain<'a>(env: &'a mut BuildEnv<'_>) -> &'a mut ColorChain {
    env.shared(|| ColorChain(SCENE_COLOR, None))
}

/// The scene colour a node building now reads (and draws over, when it
/// draws in place).
pub fn current(env: &mut BuildEnv<'_>) -> &'static str {
    chain(env).0
}

/// Makes `written` the scene colour the nodes built after this one read.
pub fn advance(env: &mut BuildEnv<'_>, written: &'static str) {
    *chain(env) = ColorChain(written, None);
}

/// Like [`advance`], for a node that may publish `aliased`'s view as
/// `written` instead of writing a new image (when its effect is certainly
/// a no-op that frame). Readers of `written` must also read [`alias`], so
/// the graph's pool keeps that image alive while they may see it.
pub fn advance_aliasing(env: &mut BuildEnv<'_>, written: &'static str, aliased: &'static str) {
    *chain(env) = ColorChain(written, Some(aliased));
}

/// The image the current scene colour may be a view of (see
/// [`advance_aliasing`]); a node reading [`current`] reads this too.
pub fn alias(env: &mut BuildEnv<'_>) -> Option<&'static str> {
    chain(env).1
}

/// Light an optical effect adds to the image inside the tone map (a lens's
/// flares, ghosts, halo and streaks): a scene-linear `Rgba16Float` texture
/// at any resolution, composited after exposure is metered and with the
/// scene's exposure, so it never brightens its own metering. A post feature
/// publishes its image by its own name with [`publish_glare`] when its node
/// builds; the tone map reads [`glare`] when it builds and composites black
/// when nothing published. Neither side knows the other.
struct Glare(Option<&'static str>);

/// Publishes `name` as the frame's glare for the nodes built after this one.
pub fn publish_glare(env: &mut BuildEnv<'_>, name: &'static str) {
    env.shared(|| Glare(None)).0 = Some(name);
}

/// The glare a node built so far published, if any.
pub fn glare(env: &mut BuildEnv<'_>) -> Option<&'static str> {
    env.shared(|| Glare(None)).0
}

/// Set by the frame setup when the frame resolves temporally: what draws
/// over the scene colour with no motion vectors of its own (water's animated
/// surface) tells the resolve where, through [`REACTIVE_MASK`].
#[derive(Clone, Copy, Debug)]
pub struct TemporalResolve;

/// Set by the frame setup when the lighting model leaves transparency to the
/// temporal resolve (a light-field model): transparent drawers write their
/// coverage into [`REACTIVE_MASK`] every frame. Implies [`TemporalResolve`].
#[derive(Clone, Copy, Debug)]
pub struct ReactiveMask;

/// The reactive mask's graph key: per-pixel coverage (`R8Unorm`, 1 = do not
/// accumulate) that what draws over the scene colour after lighting writes
/// and the temporal resolve reads.
///
/// A frame has it only while one of its writers is a member: transparent
/// drawers under [`ReactiveMask`], water while the scene holds water. The
/// first writer of a frame allocates (`write_color_raw`, `R8Unorm`,
/// `MatchSurface`) and clears it, later ones load it; the resolve reads it
/// optionally and does no coverage work in frames without it.
pub const REACTIVE_MASK: &str = "transparency_reactivity";
