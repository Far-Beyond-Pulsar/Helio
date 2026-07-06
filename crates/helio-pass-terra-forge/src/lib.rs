pub mod biome;
pub mod gpu_types;
pub mod rendering;
pub mod terrain;

pub use biome::default_palette;
pub use terrain::{generate_sphere_brickmap, BrickMapData};

// ═══════════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════════

/// Default voxel size in world units (0.1 m = 10 cm).
pub const VOXEL_SIZE: f32 = 0.1;

/// Default planet radius in world units.
pub const DEFAULT_PLANET_RADIUS: f32 = 1000.0;

/// Max chunks to generate per frame (streaming budget).
pub(crate) const CHUNKS_PER_FRAME: usize = 8;

/// Halton base-2/base-3 jitter sequence — matches TaaPass exactly so that
/// the ray march applies the same subpixel offset TaaPass expects.
pub(crate) const HALTON_JITTER: [[f32; 2]; 16] = [
    [0.500000, 0.333333],
    [0.250000, 0.666667],
    [0.750000, 0.111111],
    [0.125000, 0.444444],
    [0.625000, 0.777778],
    [0.375000, 0.222222],
    [0.875000, 0.555556],
    [0.062500, 0.888889],
    [0.562500, 0.037037],
    [0.312500, 0.370370],
    [0.812500, 0.703704],
    [0.187500, 0.148148],
    [0.687500, 0.481481],
    [0.437500, 0.814815],
    [0.937500, 0.259259],
    [0.031250, 0.592593],
];

// ═══════════════════════════════════════════════════════════════════════════════
// TerraForgePass — main voxel pass struct
// ═══════════════════════════════════════════════════════════════════════════════

use crate::gpu_types::*;
use crate::terrain::ChunkSlot;

pub struct TerraForgePass {
    pub(crate) uniform_buf: wgpu::Buffer,
    pub(crate) camera_buf: wgpu::Buffer,
    pub(crate) chunk_table_buf: wgpu::Buffer,
    pub(crate) indir_grid_buf: wgpu::Buffer,
    pub(crate) brick_pool_buf: wgpu::Buffer,
    pub(crate) voxel_pool_buf: wgpu::Buffer,
    pub(crate) palette_buf: wgpu::Buffer,
    pub(crate) edit_buf: wgpu::Buffer,
    pub(crate) mat_tex: wgpu::Texture,
    pub(crate) mat_view: wgpu::TextureView,
    pub(crate) mat_tex_half: wgpu::Texture,
    pub(crate) mat_view_half: wgpu::TextureView,
    pub(crate) norm_tex: wgpu::Texture,
    pub(crate) norm_view: wgpu::TextureView,
    pub(crate) norm_tex_half: wgpu::Texture,
    pub(crate) norm_view_half: wgpu::TextureView,
    pub(crate) ray_march_pipeline: wgpu::ComputePipeline,
    pub(crate) ray_march_bgl: wgpu::BindGroupLayout,
    pub(crate) ray_march_bind_group: wgpu::BindGroup,
    pub(crate) shade_pipeline: wgpu::RenderPipeline,
    pub(crate) shade_bgl: wgpu::BindGroupLayout,
    pub(crate) shade_bind_group: wgpu::BindGroup,
    pub(crate) gen_pipeline: wgpu::ComputePipeline,
    pub(crate) gen_bgl: wgpu::BindGroupLayout,
    pub(crate) gen_bg: wgpu::BindGroup,
    pub(crate) gen_uniform_buf: wgpu::Buffer,
    pub(crate) alloc_counter_buf: wgpu::Buffer,
    pub(crate) chunk_slots: Vec<ChunkSlot>,
    pub(crate) chunk_table_cpu: Vec<ChunkInfo>,
    pub(crate) indir_grid_cpu: Vec<u32>,
    pub(crate) initialized: bool,
    pub(crate) edits: Vec<EditOp>,
    pub(crate) edits_dirty: bool,
    pub(crate) surface_format: wgpu::TextureFormat,
    pub(crate) voxel_size: f32,
    pub(crate) planet_radius: f32,
    pub(crate) effective_max_mixed: u32,
    pub(crate) chunk_world_size: f32,
    pub(crate) indir_origin: [i32; 3],
    pub(crate) ray_w: u32,
    pub(crate) ray_h: u32,
    pub(crate) ray_w_half: u32,
    pub(crate) ray_h_half: u32,
}

impl TerraForgePass {
    pub fn planet_radius(&self) -> f32 {
        self.planet_radius
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_types::WORDS_PER_BRICK;

    #[test]
    fn brickmap_center_is_solid() {
        let data = generate_sphere_brickmap(8, 8, 28.0);
        let idx = 4 + 4 * 8 + 4 * 8 * 8;
        assert_ne!(data.brick_grid[idx].data_offset, BRICK_EMPTY);
        assert!(data.brick_grid[idx].occupancy > 0);
    }

    #[test]
    fn brickmap_corner_is_empty() {
        let data = generate_sphere_brickmap(8, 8, 20.0);
        assert_eq!(data.brick_grid[0].data_offset, BRICK_EMPTY);
        assert_eq!(data.brick_grid[0].occupancy, 0);
    }

    #[test]
    fn brickmap_allocated_bricks_reasonable() {
        let data = generate_sphere_brickmap(16, 8, 56.0);
        let total = 16u32 * 16 * 16;
        assert!(data.allocated_bricks > 0);
        assert!(data.allocated_bricks < total);
        assert_eq!(
            data.voxel_pool.len(),
            data.allocated_bricks as usize * WORDS_PER_BRICK as usize
        );
    }

    #[test]
    fn brickmap_occupancy_counts_correct() {
        let data = generate_sphere_brickmap(4, 8, 14.0);
        let total_occ: u32 = data.brick_grid.iter().map(|b| b.occupancy).sum();
        let mut actual = 0u32;
        for brick in &data.brick_grid {
            if brick.data_offset == BRICK_EMPTY {
                continue;
            }
            let base = brick.data_offset as usize * WORDS_PER_BRICK as usize;
            for w in 0..WORDS_PER_BRICK as usize {
                let word = data.voxel_pool[base + w];
                for b in 0..4u32 {
                    if (word >> (b * 8)) & 0xFF != 0 {
                        actual += 1;
                    }
                }
            }
        }
        assert_eq!(total_occ, actual);
    }

    #[test]
    fn uniforms_size() {
        assert_eq!(std::mem::size_of::<GpuUniforms>(), 80);
        assert_eq!(std::mem::size_of::<GenUniforms>(), 48);
        assert_eq!(std::mem::size_of::<ChunkInfo>(), 32);
    }

    #[test]
    fn palette_has_256_entries() {
        let p = default_palette();
        assert_eq!(p.len(), 256);
    }

    #[test]
    fn find_planet_chunks_count() {
        let chunks = TerraForgePass::find_planet_chunks(40.0, 25.6);
        assert!(
            chunks.len() > 20,
            "Expected >20 chunks, got {}",
            chunks.len()
        );
        assert!(
            chunks.len() <= MAX_LOADED_CHUNKS as usize,
            "Too many chunks: {}",
            chunks.len()
        );
    }
}
