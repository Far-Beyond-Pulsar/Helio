use serde::{Deserialize, Serialize};
use nebula_core::{error::NebulaError, traits::BakeOutput};
use nebula_serialize::chunk::ChunkTag;

/// Chunk tag for baked navigation mesh data.
pub const CHUNK_TAG: ChunkTag = ChunkTag::from_bytes(*b"NAVM");

/// A single vertex in the navigation mesh.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct NavVertex {
    pub position: [f32; 3],
}

/// A convex polygon in the navigation mesh.
///
/// Indices reference the `vertices` array in [`NavOutput`].  All polygons
/// are convex and have 3–6 vertices (Recast-style constraint).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NavPolygon {
    /// Vertex indices (3 to 6, counter-clockwise winding viewed from above).
    pub vertex_indices: Vec<u32>,
    /// Indices of adjacent polygons (parallel to `vertex_indices`).
    /// `u32::MAX` means no neighbour on that edge.
    pub neighbour_indices: Vec<u32>,
    /// Optional per-polygon area flags usable by the game's pathfinding.
    pub area_flags: u32,
}

/// Baked navigation mesh output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NavOutput {
    /// All mesh vertices (shared across all polygons).
    pub vertices: Vec<NavVertex>,
    /// Convex polygon list.
    pub polygons: Vec<NavPolygon>,
    /// Scene-space bounding box of the nav mesh.
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    /// Total walkable area in square world units.
    pub walkable_area: f32,
    /// JSON-serialised [`NavConfig`] used to produce this output.
    pub config_json: String,
}

impl BakeOutput for NavOutput {
    fn kind_name() -> &'static str { "navmesh" }
}

impl NavOutput {
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
