use serde::{Deserialize, Serialize};
use nebula_core::traits::BakeOutput;
use nebula_serialize::ChunkTag;

/// The chunk tag this baker writes — declared here, never in nebula-serialize.
pub const CHUNK_TAG: ChunkTag = ChunkTag::from_bytes(*b"LMAP");

/// The final result of a lightmap bake for a single scene.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LightmapOutput {
    /// Lightmap atlas width (px).
    pub width:  u32,
    /// Lightmap atlas height (px).
    pub height: u32,
    /// Number of components per texel (4 = RGBA).
    pub channels: u32,
    /// Whether data is stored as f32 (`true`) or f16 (`false`).
    pub is_f32: bool,
    /// Raw texel bytes: RGBA32F or RGBA16F, row-major, top-left origin.
    pub texels: Vec<u8>,
    /// Per-mesh UV-to-atlas offset/scale so the runtime can sample correctly.
    pub atlas_regions: Vec<AtlasRegion>,
    /// The config used to produce this bake (for reproducibility).
    pub config_json: String,
}

/// Maps one mesh's lightmap UVs to a region inside the atlas.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct AtlasRegion {
    pub mesh_id:    uuid::Uuid,
    /// Top-left corner in [0,1] atlas space.
    pub uv_offset:  [f32; 2],
    /// Width/height in [0,1] atlas space.
    pub uv_scale:   [f32; 2],
}

impl BakeOutput for LightmapOutput {
    fn kind_name() -> &'static str { "lightmap" }
}
