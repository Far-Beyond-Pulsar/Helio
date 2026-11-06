//! Virtual geometry in a renderer config (the passes are `helio_pass_vg`'s).

use helio::RendererConfig;

/// `config` building the default deferred graph's virtual-geometry passes for
/// `runtime` (when the device supports them). `VG_DEBUG_VIEW` selects a
/// virtual-geometry debug view at creation (tests, headless captures); an
/// explicit `debug_mode` is left alone.
pub fn with_virtual_geometry(
    mut config: RendererConfig,
    runtime: &helio_vg_stream::VgRuntime,
) -> RendererConfig {
    config
        .features
        .insert(helio_vg_stream::VgSettings::for_runtime(runtime));
    if config.debug_mode == 0 {
        if let Some(view) = helio_vg_format::debug_view::from_env() {
            config.debug_mode = view.id();
        }
    }
    config
}
