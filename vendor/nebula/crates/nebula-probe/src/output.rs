use serde::{Deserialize, Serialize};
use nebula_core::{error::NebulaError, traits::BakeOutput};
use nebula_serialize::chunk::ChunkTag;

/// Chunk tag for baked reflection cubemap data.
pub const REFLECTION_CHUNK_TAG: ChunkTag = ChunkTag::from_bytes(*b"RPRO");
/// Chunk tag for baked irradiance spherical-harmonic data.
pub const IRRADIANCE_CHUNK_TAG: ChunkTag = ChunkTag::from_bytes(*b"IRSH");

// ── Reflection output ────────────────────────────────────────────────────────

/// Baked output produced by [`ProbeConfig`] for one reflection probe position.
///
/// Contains 6 cubemap faces (ordered +X, −X, +Y, −Y, +Z, −Z).
/// Each face is `face_resolution × face_resolution` RGBA f32 (or RGBE u8).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflectionOutput {
    /// Cubemap face size in pixels.
    pub face_resolution: u32,
    /// Number of specular mip levels stored consecutively in `face_data`.
    pub mip_levels: u32,
    /// Whether pixels are RGBE-encoded (4 bytes) rather than RGBA32F (16 bytes).
    pub is_rgbe: bool,
    /// Raw face texel data.  Layout: [mip][face] → contiguous row-major pixels.
    pub face_data: Vec<u8>,
    /// JSON-serialised [`ProbeConfig`] used to produce this output.
    pub config_json: String,
}

impl BakeOutput for ReflectionOutput {
    fn kind_name() -> &'static str { "reflection_probe" }
}

impl ReflectionOutput {
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

// ── Irradiance SH output ─────────────────────────────────────────────────────

/// A single RGB spherical-harmonic coefficient.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ShCoeff { pub r: f32, pub g: f32, pub b: f32 }

/// Baked irradiance described as spherical-harmonic coefficients.
///
/// Coefficients are ordered by band-major SH index (l=0,m=0 first).
/// Order 3 (9 coefficients) covers all DC + first two AC bands which is
/// sufficient for smooth low-frequency diffuse irradiance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrradianceOutput {
    /// SH order used (1, 2, or 3).
    pub sh_order: u32,
    /// `(sh_order + 1)²` RGB coefficients.
    pub coefficients: Vec<ShCoeff>,
    /// JSON-serialised [`ProbeConfig`] used to produce this output.
    pub config_json: String,
}

impl BakeOutput for IrradianceOutput {
    fn kind_name() -> &'static str { "irradiance_probe" }
}

impl IrradianceOutput {
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
