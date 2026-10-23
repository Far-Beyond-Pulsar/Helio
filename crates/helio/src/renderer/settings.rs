//! The renderer's host settings ([`helio_settings::RENDERER`]) read into a
//! [`RendererConfig`].

use helio_settings::Resolved;

use super::{PerfOverlayMode, RenderMode, RendererConfig};

/// The temporal super resolution a host asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TsrSetting {
    /// The scene's preference (a voxel world's preset), off without one.
    Auto,
    Off,
    Preset(helio_pass_tsr::TsrQuality),
}

impl TsrSetting {
    pub fn from_settings(settings: &Resolved) -> Self {
        use helio_pass_tsr::TsrQuality;
        match settings.text("tsr_quality") {
            "off" => Self::Off,
            "performance" => Self::Preset(TsrQuality::Performance),
            "balanced" => Self::Preset(TsrQuality::Balanced),
            "quality" => Self::Preset(TsrQuality::Quality),
            "native" => Self::Preset(TsrQuality::Native),
            _ => Self::Auto,
        }
    }

    /// The preset to render with, given the scene's preference.
    pub fn resolve(self, scene: Option<helio_pass_tsr::TsrQuality>) -> Option<helio_pass_tsr::TsrQuality> {
        match self {
            Self::Auto => scene,
            Self::Off => None,
            Self::Preset(quality) => Some(quality),
        }
    }
}

fn tier<T>(text: &str, [low, medium, high, ultra]: [T; 4]) -> T {
    match text {
        "low" => low,
        "high" => high,
        "ultra" => ultra,
        _ => medium,
    }
}

impl RendererConfig {
    /// The config with a host's [`helio_settings::RENDERER`] values. The
    /// temporal preset is the host's to resolve ([`TsrSetting`]): this sets
    /// the render scale it uses while TSR is off.
    pub fn with_settings(mut self, settings: &Resolved) -> Self {
        use helio_pass_foliage_place::FoliageQuality;
        use helio_pass_postprocess::HdrOutputMode;
        use helio_pass_shadow_matrix::ShadowQuality;
        debug_assert!(std::ptr::eq(settings.group(), &helio_settings::RENDERER));
        let f = |key| settings.number(key) as f32;
        self.render_scale = f("render_scale").clamp(0.25, 1.0);
        self.render_mode = match settings.text("render_mode") {
            "forward_opaque" => RenderMode::ForwardOpaque,
            "forward_only" => RenderMode::ForwardOnly,
            _ => RenderMode::Deferred,
        };
        self.shadow_quality = tier(
            settings.text("shadow_quality"),
            [ShadowQuality::Low, ShadowQuality::Medium, ShadowQuality::High, ShadowQuality::Ultra],
        );
        let budget = &mut self.shadow_budget;
        budget.memory_bytes = (settings.number("shadow_memory_mb") * 1024.0 * 1024.0) as u64;
        budget.max_resolution = settings.text("shadow_max_resolution").parse().unwrap_or(budget.max_resolution);
        budget.updates_per_frame = settings.number("shadow_updates_per_frame") as u32;
        budget.update_texels_per_frame = (settings.number("shadow_update_megatexels") * 1024.0 * 1024.0) as u32;
        budget.max_distance = f("shadow_max_distance");
        budget.fade_frames = settings.number("shadow_fade_frames") as u32;
        budget.hysteresis = f("shadow_hysteresis");
        budget.page_invalidation = settings.bool("shadow_page_invalidation");
        self.enable_environment_reflections = settings.bool("environment_reflections");
        self.enable_ssr = settings.bool("screen_space_reflections");
        self.enable_planar_reflections = settings.bool("planar_reflections");
        self.gi_probe_spacing = settings.optional("gi_probe_spacing").map(|m| m as f32);
        self.enable_foliage = settings.bool("foliage");
        self.foliage_quality = tier(
            settings.text("foliage_quality"),
            [FoliageQuality::Low, FoliageQuality::Medium, FoliageQuality::High, FoliageQuality::Ultra],
        );
        self.foliage_range = settings.optional("foliage_range").map(|m| m as f32);
        self.foliage_tufts_per_m2 = settings.optional("foliage_tufts_per_m2").map(|n| n as f32);
        self.foliage_blades_per_tuft = settings.optional("foliage_blades_per_tuft").map(|n| n as f32);
        self.hdr_output_mode = match settings.text("hdr_output_mode") {
            "hdr10" => HdrOutputMode::Hdr10,
            "scrgb" => HdrOutputMode::ScRgb,
            "passthrough" => HdrOutputMode::Passthrough,
            _ => HdrOutputMode::Ldr,
        };
        self.background_color = settings.color("background_color");
        self.enable_portals = settings.bool("portals");
        self.perf_overlay_mode = match settings.text("perf_overlay") {
            "pass_overdraw" => PerfOverlayMode::PassOverdraw,
            "shader_complexity" => PerfOverlayMode::ShaderComplexity,
            "tile_light_count" => PerfOverlayMode::TileLightCount,
            "pass_output" => PerfOverlayMode::PassOutput,
            _ => PerfOverlayMode::Disabled,
        };
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_settings_are_the_default_config() {
        let config = || RendererConfig::new(640, 360, wgpu::TextureFormat::Rgba8Unorm);
        let configured = config().with_settings(&helio_settings::RENDERER.defaults());
        assert_eq!(format!("{configured:?}"), format!("{:?}", config()));
        assert_eq!(TsrSetting::from_settings(&helio_settings::RENDERER.defaults()), TsrSetting::Auto);
    }

    #[test]
    fn every_choice_maps() {
        for setting in helio_settings::RENDERER.settings {
            let helio_settings::Kind::Choice(options) = setting.kind else { continue };
            for option in options {
                let settings = helio_settings::RENDERER
                    .resolve(|key| (key == setting.key).then(|| helio_settings::Value::Text(option.value.into())));
                assert_eq!(settings.text(setting.key), option.value);
                // Only the default option may land on the default config.
                let configured = RendererConfig::new(8, 8, wgpu::TextureFormat::Rgba8Unorm).with_settings(&settings);
                let default = RendererConfig::new(8, 8, wgpu::TextureFormat::Rgba8Unorm);
                let is_default = setting.default == helio_settings::Value::Text(option.value.into());
                let unchanged = format!("{configured:?}") == format!("{default:?}");
                // `tsr_quality` is resolved by the host, not the config.
                if setting.key != "tsr_quality" {
                    assert_eq!(unchanged, is_default, "{} = {}", setting.key, option.value);
                }
            }
        }
    }
}
