// Support both helio::Renderer and helio_render_v2::Renderer

#[cfg(feature = "live-portal")]
pub fn enable_live_dashboard(renderer: &mut helio::Renderer) {
    match renderer.start_live_portal_default() {
        Ok(url) => {
            log::info!("🌐 Live performance dashboard: {url}");
        }
        Err(e) => {
            log::warn!("Failed to start live dashboard: {e}");
        }
    }
}

#[cfg(not(feature = "live-portal"))]
pub fn enable_live_dashboard(_renderer: &mut helio::Renderer) {
    log::warn!("Live portal not enabled. Rebuild with --features live-portal");
}
