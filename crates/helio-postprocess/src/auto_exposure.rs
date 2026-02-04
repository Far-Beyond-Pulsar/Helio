pub struct AutoExposure {
    pub enabled: bool,
    pub min_luminance: f32,
    pub max_luminance: f32,
    pub speed_up: f32,
    pub speed_down: f32,
    pub exposure_compensation: f32,
    pub metering_mode: MeteringMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeteringMode {
    Average,
    SpotCenter,
    CenterWeighted,
}

impl Default for AutoExposure {
    fn default() -> Self {
        Self {
            enabled: true,
            min_luminance: 0.03,
            max_luminance: 2.0,
            speed_up: 2.0,
            speed_down: 1.0,
            exposure_compensation: 0.0,
            metering_mode: MeteringMode::Average,
        }
    }
}
