//! The foliage feature's settings: what a renderer config carries for it
//! (`RendererConfig::features`) and the node builds read.

use crate::FoliageQuality;

/// How much foliage to place, and how far.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FoliageSettings {
    /// Build the foliage passes. A scene with no `foliage_types` rows pays
    /// nothing for this being on.
    pub enabled: bool,
    /// Densest ground in tufts per square metre, or `None` for the default.
    pub tufts_per_m2: Option<f32>,
    /// Blades per tuft, or `None` for the default.
    pub blades_per_tuft: Option<f32>,
    /// The quality preset: default range, density multiplier, near LOD
    /// distance scale and cull granularity.
    pub quality: FoliageQuality,
    /// Metres foliage reaches, or `None` for the quality preset's. Bands of
    /// tiles (`plan_bands`) carry it out there.
    pub range: Option<f32>,
}

impl Default for FoliageSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            tufts_per_m2: None,
            blades_per_tuft: None,
            quality: FoliageQuality::Medium,
            range: None,
        }
    }
}

impl FoliageSettings {
    /// The settings with a host's values (keys of
    /// [`helio_settings::RENDERER`]).
    pub fn with_host_settings(mut self, settings: &helio_settings::Resolved) -> Self {
        self.enabled = settings.bool("foliage");
        self.quality = match settings.text("foliage_quality") {
            "low" => FoliageQuality::Low,
            "high" => FoliageQuality::High,
            "ultra" => FoliageQuality::Ultra,
            _ => FoliageQuality::Medium,
        };
        self.range = settings.optional("foliage_range").map(|m| m as f32);
        self.tufts_per_m2 = settings.optional("foliage_tufts_per_m2").map(|n| n as f32);
        self.blades_per_tuft = settings
            .optional("foliage_blades_per_tuft")
            .map(|n| n as f32);
        self
    }

    /// Metres foliage reaches: the setting, or the quality's.
    pub fn reach(&self) -> f32 {
        self.range.unwrap_or(self.quality.range())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_defaults_are_the_default_settings() {
        let settings = helio_settings::RENDERER.defaults();
        assert_eq!(
            FoliageSettings::default().with_host_settings(&settings),
            FoliageSettings::default()
        );
    }

    #[test]
    fn every_quality_choice_maps() {
        let setting = helio_settings::RENDERER
            .settings
            .iter()
            .find(|s| s.key == "foliage_quality")
            .expect("the renderer settings describe foliage quality");
        let helio_settings::Kind::Choice(options) = setting.kind else {
            panic!("foliage_quality is a choice");
        };
        for option in options {
            let settings = helio_settings::RENDERER.resolve(|key| {
                (key == setting.key).then(|| helio_settings::Value::Text(option.value.into()))
            });
            let mapped = FoliageSettings::default().with_host_settings(&settings);
            let is_default = setting.default == helio_settings::Value::Text(option.value.into());
            assert_eq!(
                mapped == FoliageSettings::default(),
                is_default,
                "{}",
                option.value
            );
        }
    }
}
