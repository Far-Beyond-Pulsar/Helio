use helio_render_v2::Renderer;

pub fn enable_live_dashboard(renderer: &mut Renderer) {
    match renderer.start_live_portal_default() {
        Ok(url) => {
            log::info!("Live dashboard: {url}");
        }
        Err(e) => {
            log::warn!("Failed to start live dashboard: {e}");
        }
    }
}
