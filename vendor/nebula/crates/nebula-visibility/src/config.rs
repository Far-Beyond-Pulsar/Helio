use serde::{Deserialize, Serialize};
use nebula_core::traits::BakeInput;

/// Configuration for Potentially Visible Set (PVS) baking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PvsConfig {
    /// Voxel cell side length in world units.
    ///
    /// Smaller cells give finer-grained visibility data at the cost of more
    /// GPU memory and longer bake times.  A value of 2–4 m works well for
    /// typical game levels.
    pub cell_size: f32,

    /// Number of random rays cast from each vis-cell centre per frame.
    ///
    /// Higher values reduce false negatives (missing visibility) but increase
    /// cost O(n_cells × ray_budget).
    pub ray_budget: u32,

    /// Whether to apply conservative dilation after initial ray casting.
    ///
    /// Conservative dilation inflates the visible set by one cell in each
    /// axis, trading a slightly larger PVS for zero false-negative risk
    /// (no geometry is incorrectly culled).
    pub conservative: bool,

    /// Minimum number of rays that must hit a cell for it to be considered
    /// visible from the source cell.
    pub visibility_threshold: u32,

    /// Maximum ray distance in world units.
    pub max_ray_distance: f32,
}

impl Default for PvsConfig {
    fn default() -> Self {
        Self {
            cell_size: 3.0,
            ray_budget: 256,
            conservative: true,
            visibility_threshold: 1,
            max_ray_distance: 500.0,
        }
    }
}

impl PvsConfig {
    /// Coarse fast-preview preset (large cells, few rays).
    pub fn fast() -> Self {
        Self { cell_size: 8.0, ray_budget: 32, conservative: false, ..Default::default() }
    }

    /// High-precision production preset.
    pub fn ultra() -> Self {
        Self { cell_size: 1.5, ray_budget: 2048, conservative: true, ..Default::default() }
    }
}

impl BakeInput for PvsConfig {}
