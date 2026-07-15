use serde::{Deserialize, Serialize};
use nebula_core::{error::NebulaError, traits::BakeOutput};
use nebula_serialize::chunk::ChunkTag;

/// Chunk tag for baked Potentially Visible Set data.
pub const CHUNK_TAG: ChunkTag = ChunkTag::from_bytes(*b"PVSS");

/// A 3-D grid cell index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CellIndex { pub x: i32, pub y: i32, pub z: i32 }

/// Baked PVS output.
///
/// Each source cell stores a bitfield of visible target cells.  The bitfield
/// is stored as a flat `Vec<u64>` in row-major (x-fastest) grid order.
///
/// To test visibility from cell A (flat index `a`) to cell B (flat index `b`):
/// ```rust,ignore
/// let word = a * words_per_cell + b / 64;
/// let bit  = b % 64;
/// let visible = (pvs.bits[word] >> bit) & 1 == 1;
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PvsOutput {
    /// World-space minimum corner of the PVS grid.
    pub world_min: [f32; 3],
    /// World-space maximum corner of the PVS grid.
    pub world_max: [f32; 3],
    /// Cell dimensions in X, Y, Z.
    pub grid_dims: [u32; 3],
    /// Voxel cell side length (same in all three axes).
    pub cell_size: f32,
    /// Total number of cells = `grid_dims[0] * grid_dims[1] * grid_dims[2]`.
    pub cell_count: u32,
    /// Number of `u64` words per source cell = `ceil(cell_count / 64)`.
    pub words_per_cell: u32,
    /// Flat packed bitfield.  Length = `cell_count * words_per_cell`.
    pub bits: Vec<u64>,
    /// JSON-serialised [`PvsConfig`] used to produce this output.
    pub config_json: String,
}

impl PvsOutput {
    /// Returns `true` if cell at flat index `from_cell` can see `to_cell`.
    #[inline]
    pub fn is_visible(&self, from_cell: usize, to_cell: usize) -> bool {
        let word = from_cell * self.words_per_cell as usize + to_cell / 64;
        let bit  = to_cell % 64;
        self.bits.get(word).map_or(false, |w| (w >> bit) & 1 == 1)
    }

    /// Returns the flat cell index for world-space position `p`, or `None` if
    /// the position is outside the grid.
    pub fn cell_at(&self, p: [f32; 3]) -> Option<usize> {
        let dx = ((p[0] - self.world_min[0]) / self.cell_size).floor() as i32;
        let dy = ((p[1] - self.world_min[1]) / self.cell_size).floor() as i32;
        let dz = ((p[2] - self.world_min[2]) / self.cell_size).floor() as i32;
        let [gx, gy, gz] = self.grid_dims.map(|d| d as i32);
        if dx < 0 || dy < 0 || dz < 0 || dx >= gx || dy >= gy || dz >= gz { return None; }
        Some((dz as usize * gy as usize + dy as usize) * gx as usize + dx as usize)
    }
}

impl BakeOutput for PvsOutput {
    fn kind_name() -> &'static str { "pvs" }
}

impl PvsOutput {
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
