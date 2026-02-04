use glam::Vec2;

pub struct FFTWaveSimulation {
    pub resolution: u32,
    pub size: f32,
    pub wind_speed: f32,
    pub wind_direction: Vec2,
    pub wave_amplitude: f32,
    pub choppiness: f32,
    pub time_scale: f32,
}

impl Default for FFTWaveSimulation {
    fn default() -> Self {
        Self {
            resolution: 512,
            size: 1000.0,
            wind_speed: 15.0,
            wind_direction: Vec2::new(1.0, 0.0).normalize(),
            wave_amplitude: 1.0,
            choppiness: 1.5,
            time_scale: 1.0,
        }
    }
}

pub struct GerstnerWaves {
    pub wave_count: u32,
    pub amplitude: f32,
    pub wavelength: f32,
    pub speed: f32,
    pub steepness: f32,
}

impl Default for GerstnerWaves {
    fn default() -> Self {
        Self {
            wave_count: 4,
            amplitude: 1.0,
            wavelength: 10.0,
            speed: 2.0,
            steepness: 0.5,
        }
    }
}
