//! The views a frame renders, as a node sees them when it builds
//! (`helio_core::view`).
//!
//! A graph renders one view: the frame's camera or a capture (a planar
//! reflection, into its requester's target). The graph's builder supplies
//! the [`ViewDescription`] as a setting; a drawer reads it here and bakes the
//! view's facts into what it builds:
//!
//! - its winding ([`front_face`]): a mirrored view flips front faces;
//! - its clip ([`clipped`]): a pipeline built for a clipped view discards
//!   what the camera's clip plane cuts (the prelude's `helio_view_clipped`;
//!   [`CLIP_CONSTANT`] is the override that compiles it in), so the primary
//!   view keeps early depth.

use helio_feature_api::BuildEnv;

pub use helio_core::view::{
    ViewBudget, ViewContext, ViewDescription, ViewKind, ViewRequest, VIEW_CLIPPED, VIEW_MIRRORED,
};

/// The view the graph being built renders: the primary view unless its
/// builder says otherwise.
pub fn view_of(env: &BuildEnv<'_>) -> ViewDescription {
    env.setting::<ViewDescription>()
        .copied()
        .unwrap_or(ViewDescription::PRIMARY)
}

/// The front face a drawer's pipelines use in the graph being built.
pub fn front_face(env: &BuildEnv<'_>) -> wgpu::FrontFace {
    view_of(env).front_face()
}

/// Whether the graph being built renders a clipped view: its drawers test
/// the camera's clip plane.
pub fn clipped(env: &BuildEnv<'_>) -> bool {
    view_of(env).clipped
}

/// The WGSL override constant a drawer's shader gates its clip test on:
/// `override HELIO_VIEW_CLIP: bool = false;` and
/// `if HELIO_VIEW_CLIP && helio_view_clipped(camera, world_pos) { discard; }`.
pub const CLIP_CONSTANT: &str = "HELIO_VIEW_CLIP";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_mirrored_view_flips_winding_and_the_primary_does_not() {
        assert_eq!(ViewDescription::PRIMARY.front_face(), wgpu::FrontFace::Ccw);
        let mirrored = ViewDescription {
            kind: ViewKind::Capture,
            mirrored: true,
            clipped: true,
            depth: 1,
        };
        assert_eq!(mirrored.front_face(), wgpu::FrontFace::Cw);
        assert_eq!(mirrored.flags(), VIEW_MIRRORED | VIEW_CLIPPED);
    }
}
