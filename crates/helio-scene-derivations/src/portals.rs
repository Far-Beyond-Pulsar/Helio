//! The portal feature's host API: what a host tells the portal passes about a
//! frame. The renderer knows no portal; these reach the passes through it.

use helio::Renderer;
pub use helio_pass_portal_instances::PortalDebugText;
use helio_pass_portal_instances::PortalEditorOverlayPass;

/// Set the world-anchored pixel text shown by the editor's portal overlay.
/// `false` when the graph has no overlay pass.
pub fn set_debug_texts(renderer: &mut Renderer, texts: Vec<PortalDebugText>) -> bool {
    renderer
        .find_pass_mut::<PortalEditorOverlayPass>()
        .is_some_and(|pass| pass.set_debug_texts(texts))
}

/// Set transient portal-through geometry for the editor diagnostics view.
pub fn set_debug_geometry(renderer: &mut Renderer, vertices: Vec<[f64; 3]>) -> bool {
    renderer
        .find_pass_mut::<PortalEditorOverlayPass>()
        .is_some_and(|pass| pass.set_ghost_vertices(vertices))
}

/// Upload a resolver-generated portal projection frame built with its own
/// table (`PortalProjectionBridge::build`): its spaces become the table. A
/// host that shares the table with other producers builds with `build_into`
/// and calls `Renderer::set_coordinate_spaces` and
/// `Renderer::set_portal_projection_counts` itself.
pub fn set_projection_frame(
    renderer: &mut Renderer,
    frame: &helio_pass_portal_cull::PortalProjectionFrame,
) {
    renderer.set_coordinate_spaces(frame.renderer_coordinate_spaces());
    renderer.set_portal_projection_counts(
        frame.counts.portal_view_count,
        frame.counts.portal_chain_count,
    );
}
