use serde::{Deserialize, Serialize};
use nebula_core::{error::NebulaError, traits::BakeOutput};
use nebula_serialize::chunk::ChunkTag;
use crate::config::FREQ_BAND_COUNT;

/// Chunk tag for baked acoustic impulse-response data.
pub const CHUNK_TAG: ChunkTag = ChunkTag::from_bytes(*b"AUIR");

// ── Impulse Response ──────────────────────────────────────────────────────────

/// A single-channel, per-frequency-band Room Impulse Response (RIR).
///
/// Each band stores samples at the configured time resolution.
/// Bands are ordered by increasing centre frequency (see [`FREQ_BAND_CENTRES`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpulseResponse {
    pub listener_position: [f32; 3],
    /// Samples per second.
    pub sample_rate: u32,
    /// `FREQ_BAND_COUNT` channels of time-domain samples.
    pub bands: [Vec<f32>; FREQ_BAND_COUNT],
    /// Pre-computed T60 reverberation time per band (seconds).
    pub t60_per_band: [f32; FREQ_BAND_COUNT],
    /// RT60 estimated from Sabine's formula (broadband, seconds).
    pub broadband_t60: f32,
    /// Early-reflections/late-reverb transition time (seconds).
    pub early_late_split_secs: f32,
}

// ── Reverb Zone ───────────────────────────────────────────────────────────────

/// Spatial reverb zone parameters derived from acoustic simulation.
///
/// Engines can use these as game-audio reverb effect parameters without
/// needing to convolve the full impulse response at runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReverbZone {
    /// AABB min corner of the zone.
    pub aabb_min: [f32; 3],
    /// AABB max corner of the zone.
    pub aabb_max: [f32; 3],
    /// Average broadband T60 (seconds).
    pub t60: f32,
    /// Early-decay time (seconds).
    pub edt: f32,
    /// Clarity metric C80 (dB).
    pub c80: f32,
    /// Definition metric D50.
    pub d50: f32,
    /// Room gain (dB) — perceived loudness boost from early reflections.
    pub room_gain_db: f32,
    /// Broadband direct-to-reverb ratio (dB).
    pub drr_db: f32,
    /// Per-band absorption coefficient estimate [0, 1].
    pub absorption: [f32; FREQ_BAND_COUNT],
}

// ── AcousticOutput ────────────────────────────────────────────────────────────

/// Full baked acoustic output for a scene.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcousticOutput {
    /// One RIR per listener point in the same order as [`AcousticConfig::listener_points`].
    pub impulse_responses: Vec<ImpulseResponse>,
    /// Optional spatial reverb zones covering the scene.
    pub reverb_zones: Vec<ReverbZone>,
    /// JSON-serialised [`AcousticConfig`] used to produce this output.
    pub config_json: String,
}

impl BakeOutput for AcousticOutput {
    fn kind_name() -> &'static str { "acoustic" }
}

impl AcousticOutput {
    pub fn serialize_to_bytes(&self) -> Result<Vec<u8>, NebulaError> {
        bincode::serde::encode_to_vec(self, bincode::config::standard())
            .map_err(|e| NebulaError::Serialize(e.to_string()))
    }

    pub fn deserialize_from_bytes(bytes: &[u8]) -> Result<Self, NebulaError> {
        let (v, _) = bincode::serde::decode_from_slice(bytes, bincode::config::standard())
            .map_err(|e| NebulaError::Deserialize(e.to_string()))?;
        Ok(v)
    }
}
