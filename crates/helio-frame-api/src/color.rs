//! The scene colour chain: scene-linear -> media -> anti-aliasing -> lens ->
//! tone map -> output.
//!
//! Post features never name each other's outputs. A node that works on the
//! scene colour reads [`current`] when it builds and, if it writes a new
//! texture, calls [`advance`] with its name; the next node reads that. Nodes
//! build in execution order, so the chain is the order the preset placed
//! them in. Names stay `&'static str` (graph resources are), and each pass
//! still owns its output texture.

use helio_feature_api::BuildEnv;

/// The lit scene-linear colour: cleared by the background, written by the
/// lighting model, drawn over by overlays.
pub const SCENE_COLOR: &str = "pre_aa";

/// The scene colour nodes built so far left as current.
struct ColorChain(&'static str);

/// The scene colour a node building now reads (and draws over, when it
/// draws in place).
pub fn current(env: &mut BuildEnv<'_>) -> &'static str {
    env.shared(|| ColorChain(SCENE_COLOR)).0
}

/// Makes `written` the scene colour the nodes built after this one read.
pub fn advance(env: &mut BuildEnv<'_>, written: &'static str) {
    env.shared(|| ColorChain(SCENE_COLOR)).0 = written;
}

/// Set by a preset whose anti-aliasing wants a reactive mask from what draws
/// over the scene colour after lighting (transparency under HLFS and TSR).
#[derive(Clone, Copy, Debug)]
pub struct ReactiveMask;
