use serde::{Deserialize, Serialize};
use nebula_core::traits::BakeInput;

/// Number of frequency bands modelled in the acoustic simulation.
pub const FREQ_BAND_COUNT: usize = 8;

/// Centre frequencies (Hz) of the 8 octave bands.
pub const FREQ_BAND_CENTRES: [f32; FREQ_BAND_COUNT] = [
    62.5, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0,
];

/// A single listener position at which RIRs and reverb parameters are baked.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListenerPoint {
    pub position: [f32; 3],
    /// Optional label for debugging / editor display.
    pub label: Option<String>,
}

/// Configuration for acoustic / room-impulse-response baking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcousticConfig {
    /// Listener positions at which full RIRs are computed.
    pub listener_points: Vec<ListenerPoint>,
    /// Maximum number of specular image-source order reflections.
    pub max_order: u32,
    /// Number of stochastic (diffuse) rays fired per listener per frequency band.
    pub diffuse_rays: u32,
    /// Maximum ray path length (seconds of travel at speed-of-sound = 343 m/s).
    pub max_duration_secs: f32,
    /// Temporal resolution of the impulse response (seconds per sample).
    pub time_resolution_secs: f32,
    /// Air absorption coefficient per metre per frequency band (ISO 9613-1 typical values).
    pub air_absorption: [f32; FREQ_BAND_COUNT],
    /// Whether to pre-mix all listener RIRs into a single reverb zone estimate.
    pub emit_reverb_zone: bool,
    /// Voxel cell size used for sound-occlusion / visibility queries.
    pub occlusion_cell_size: f32,
}

impl Default for AcousticConfig {
    fn default() -> Self {
        Self {
            listener_points: Vec::new(),
            max_order: 2,
            diffuse_rays: 512,
            max_duration_secs: 2.0,
            time_resolution_secs: 1.0 / 44100.0,
            air_absorption: [0.0002, 0.0004, 0.0006, 0.001, 0.002, 0.004, 0.008, 0.016],
            emit_reverb_zone: true,
            occlusion_cell_size: 0.5,
        }
    }
}

impl AcousticConfig {
    /// Fast low-quality preview preset (few rays, low order, short tail).
    pub fn fast() -> Self {
        Self { max_order: 1, diffuse_rays: 64, max_duration_secs: 0.5, ..Default::default() }
    }

    /// High-quality production preset.
    pub fn ultra() -> Self {
        Self { max_order: 5, diffuse_rays: 8192, max_duration_secs: 5.0, ..Default::default() }
    }
}

impl BakeInput for AcousticConfig {}
