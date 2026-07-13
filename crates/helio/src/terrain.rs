//! CPU-side voxel terrain component and brick baker.
//!
//! Both passes read a dense per-volume brick grid, but from different GPU
//! buffers with different layouts:
//! - `VoxelMeshPass` owns its own `brick_meta_buf`/`voxel_data_buf`, storing
//!   each brick as a **padded 9x9x9** block (`upload_*_mesh`) so its
//!   marching-cubes extract pass can read one voxel of +X/+Y/+Z halo and
//!   close the seam between adjacent bricks (see `voxel_surface_extract.wgsl`
//!   /`CELLS_PER_DIM`). It re-extracts real triangles on `mark_dirty()`.
//! - `VoxelRayMarchPass` reads the *shared* `GpuScene::voxel_brick_pool`/
//!   `voxel_data_pool` directly (`upload_*_raymarch`), storing each brick as
//!   the raw **8x8x8** the DDA marcher indexes every frame — nothing in the
//!   engine bakes edits into that pool on its own, so this component owns that
//!   data on the CPU and uploads it directly via `queue.write_buffer`.
//!
//! The grid is always a dense 64^3 voxel volume (8 bricks per axis of 8
//! voxels each — fixed GPU-side by the engine's `BRICK_SIZE` constant).

pub const BRICK_DIM: u32 = 8;
pub const BRICKS_PER_AXIS: u32 = 8;
pub const VOXEL_TERRAIN_GRID_DIM: u32 = BRICKS_PER_AXIS * BRICK_DIM; // 64
                                                                     // VoxelMeshPass's extract shader reads a padded 9x9x9 block per brick (one
                                                                     // extra voxel of +X/+Y/+Z halo from the neighbor brick) so marching cubes can
                                                                     // cover the boundary cell between adjacent bricks — without it every brick
                                                                     // edge has a visible seam/gap. See voxel_surface_extract.wgsl::CELLS_PER_DIM.
pub const PADDED_DIM: u32 = BRICK_DIM + 1; // 9
pub const PADDED_VOXELS_PER_BRICK: usize = (PADDED_DIM * PADDED_DIM * PADDED_DIM) as usize; // 729
pub const WORDS_PER_BRICK: usize = PADDED_VOXELS_PER_BRICK.div_ceil(4); // 183

// VoxelRayMarchPass indexes the raw (unpadded) 8x8x8 brick directly —
// see voxel_raymarch.wgsl::read_voxel.
pub const RAYMARCH_VOXELS_PER_BRICK: usize = (BRICK_DIM * BRICK_DIM * BRICK_DIM) as usize; // 512
pub const RAYMARCH_WORDS_PER_BRICK: usize = RAYMARCH_VOXELS_PER_BRICK / 4; // 128

// ── materials ───────────────────────────────────────────────────────────────

pub const MAT_AIR: u8 = 0;
pub const MAT_GRASS: u8 = 1;
pub const MAT_DIRT: u8 = 2;
pub const MAT_STONE: u8 = 3;
pub const MAT_ORE: u8 = 4;

// ── cheap deterministic value noise (no external crate needed) ─────────────

fn hash(x: i32, y: i32, z: i32, seed: u32) -> f32 {
    let mut h = (x as u32)
        .wrapping_mul(374761393)
        .wrapping_add((y as u32).wrapping_mul(668265263))
        .wrapping_add((z as u32).wrapping_mul(2654435761))
        .wrapping_add(seed.wrapping_mul(2246822519));
    h = (h ^ (h >> 15)).wrapping_mul(2246822519);
    h = (h ^ (h >> 13)).wrapping_mul(3266489917);
    h ^= h >> 16;
    (h as f32 / u32::MAX as f32) * 2.0 - 1.0
}

fn smoothstep(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn value_noise2(x: f32, z: f32, seed: u32) -> f32 {
    let x0 = x.floor() as i32;
    let z0 = z.floor() as i32;
    let sx = smoothstep(x - x0 as f32);
    let sz = smoothstep(z - z0 as f32);
    let n00 = hash(x0, 0, z0, seed);
    let n10 = hash(x0 + 1, 0, z0, seed);
    let n01 = hash(x0, 0, z0 + 1, seed);
    let n11 = hash(x0 + 1, 0, z0 + 1, seed);
    lerp(lerp(n00, n10, sx), lerp(n01, n11, sx), sz)
}

fn fbm2(x: f32, z: f32, seed: u32, octaves: u32) -> f32 {
    let mut amp = 0.5;
    let mut freq = 1.0;
    let mut sum = 0.0;
    let mut norm = 0.0;
    for i in 0..octaves {
        sum += value_noise2(x * freq, z * freq, seed.wrapping_add(i * 101)) * amp;
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    sum / norm
}

// ── world ────────────────────────────────────────────────────────────────────

/// Dense 64^3 voxel material grid, baked into GPU bricks on demand.
pub struct VoxelTerrain {
    materials: Vec<u8>,
}

impl VoxelTerrain {
    pub fn empty() -> Self {
        Self {
            materials: vec![
                MAT_AIR;
                (VOXEL_TERRAIN_GRID_DIM * VOXEL_TERRAIN_GRID_DIM * VOXEL_TERRAIN_GRID_DIM)
                    as usize
            ],
        }
    }

    fn idx(x: u32, y: u32, z: u32) -> usize {
        (x + y * VOXEL_TERRAIN_GRID_DIM + z * VOXEL_TERRAIN_GRID_DIM * VOXEL_TERRAIN_GRID_DIM)
            as usize
    }

    fn in_bounds(x: i32, y: i32, z: i32) -> bool {
        x >= 0
            && y >= 0
            && z >= 0
            && (x as u32) < VOXEL_TERRAIN_GRID_DIM
            && (y as u32) < VOXEL_TERRAIN_GRID_DIM
            && (z as u32) < VOXEL_TERRAIN_GRID_DIM
    }

    /// Fills the grid with procedurally generated hills, dirt/stone layers, caves and ore.
    pub fn generate(&mut self, seed: u32) {
        let base_height = VOXEL_TERRAIN_GRID_DIM as f32 * 0.45;
        let amplitude = VOXEL_TERRAIN_GRID_DIM as f32 * 0.22;
        let freq = 1.0 / 18.0;

        for x in 0..VOXEL_TERRAIN_GRID_DIM {
            for z in 0..VOXEL_TERRAIN_GRID_DIM {
                let h = fbm2(x as f32 * freq, z as f32 * freq, seed, 4);
                let terrain_height = base_height + h * amplitude;

                for y in 0..VOXEL_TERRAIN_GRID_DIM {
                    let yf = y as f32;
                    if yf > terrain_height {
                        self.materials[Self::idx(x, y, z)] = MAT_AIR;
                        continue;
                    }

                    let depth = terrain_height - yf;
                    let mut mat = if depth < 1.0 {
                        MAT_GRASS
                    } else if depth < 4.0 {
                        MAT_DIRT
                    } else {
                        MAT_STONE
                    };

                    // No cave carving here: VoxelMeshPass caps each brick at
                    // MAX_SURFACE_VERTS_PER_BRICK/MAX_SURFACE_INDICES_PER_BRICK
                    // (256/768) — a cave-riddled brick's internal surface area
                    // blows well past that budget and geometry gets silently
                    // truncated mid-brick. A plain heightfield keeps each
                    // brick's surface to roughly one layer of cells.
                    if mat == MAT_STONE
                        && hash(x as i32, y as i32, z as i32, seed ^ 0x1234_5678) > 0.985
                    {
                        mat = MAT_ORE;
                    }

                    self.materials[Self::idx(x, y, z)] = mat;
                }
            }
        }
    }

    /// Applies a sphere edit (add fills with `material`, subtract clears to air) in
    /// voxel-grid coordinates. Returns the touched region's brick range for partial rebaking.
    pub fn paint_sphere(
        &mut self,
        center: [f32; 3],
        radius: f32,
        material: u8,
        add: bool,
    ) -> Option<BrickRange> {
        let r = radius.ceil() as i32;
        let cx = center[0].floor() as i32;
        let cy = center[1].floor() as i32;
        let cz = center[2].floor() as i32;
        let r2 = radius * radius;

        let mut touched = false;
        let mut min = [VOXEL_TERRAIN_GRID_DIM as i32; 3];
        let mut max = [-1i32; 3];

        for dz in -r..=r {
            for dy in -r..=r {
                for dx in -r..=r {
                    let d2 = (dx * dx + dy * dy + dz * dz) as f32;
                    if d2 > r2 {
                        continue;
                    }
                    let (x, y, z) = (cx + dx, cy + dy, cz + dz);
                    if !Self::in_bounds(x, y, z) {
                        continue;
                    }
                    self.materials[Self::idx(x as u32, y as u32, z as u32)] =
                        if add { material } else { MAT_AIR };
                    touched = true;
                    min[0] = min[0].min(x);
                    min[1] = min[1].min(y);
                    min[2] = min[2].min(z);
                    max[0] = max[0].max(x);
                    max[1] = max[1].max(y);
                    max[2] = max[2].max(z);
                }
            }
        }

        if !touched {
            return None;
        }
        Some(BrickRange {
            min: [
                (min[0] as u32) / BRICK_DIM,
                (min[1] as u32) / BRICK_DIM,
                (min[2] as u32) / BRICK_DIM,
            ],
            max: [
                (max[0] as u32) / BRICK_DIM,
                (max[1] as u32) / BRICK_DIM,
                (max[2] as u32) / BRICK_DIM,
            ],
        })
    }

    /// Bakes a brick's padded 9x9x9 voxel block (see `PADDED_DIM`): local
    /// indices 0..=7 are this brick's own voxels, index 8 on each axis reads
    /// one voxel of halo from the +X/+Y/+Z neighbor brick (or air, past the
    /// world edge) — matches voxel_surface_extract.wgsl::read_voxel exactly.
    fn bake_brick(&self, bx: u32, by: u32, bz: u32, data_out: &mut [u32; WORDS_PER_BRICK]) -> bool {
        let mut occupied = false;
        for lz in 0..PADDED_DIM {
            for ly in 0..PADDED_DIM {
                for lx in 0..PADDED_DIM {
                    let gx = bx * BRICK_DIM + lx;
                    let gy = by * BRICK_DIM + ly;
                    let gz = bz * BRICK_DIM + lz;
                    let mat = if gx < VOXEL_TERRAIN_GRID_DIM
                        && gy < VOXEL_TERRAIN_GRID_DIM
                        && gz < VOXEL_TERRAIN_GRID_DIM
                    {
                        self.materials[Self::idx(gx, gy, gz)]
                    } else {
                        MAT_AIR
                    };
                    if mat != MAT_AIR {
                        occupied = true;
                    }
                    let linear = (lz * PADDED_DIM * PADDED_DIM + ly * PADDED_DIM + lx) as usize;
                    let word = linear / 4;
                    let byte_in_word = linear % 4;
                    data_out[word] |= (mat as u32) << (byte_in_word * 8);
                }
            }
        }
        occupied
    }

    /// World-space origin of a brick's local (0,0,0) voxel corner — the value
    /// `VoxelMeshPass::mark_dirty` needs so its extract shader can place
    /// generated vertices in world space (see `voxel_surface_extract.wgsl`).
    fn brick_origin(bx: u32, by: u32, bz: u32, voxel_size: f32) -> [f32; 3] {
        let half = VOXEL_TERRAIN_GRID_DIM as f32 / 2.0;
        let gx = (bx * BRICK_DIM) as f32 - half;
        let gy = (by * BRICK_DIM) as f32 - half;
        let gz = (bz * BRICK_DIM) as f32 - half;
        [gx * voxel_size, gy * voxel_size, gz * voxel_size]
    }

    /// Re-bakes and uploads the bricks touched by a `BrickRange` into
    /// `VoxelMeshPass`'s buffers. Returns `(brick_idx, origin)` for every
    /// touched brick (occupied or not) — the caller must `mark_dirty()` each
    /// one so the extract pass re-runs (an emptied brick needs to re-extract
    /// to zero triangles too, not just a newly-filled one).
    pub fn upload_range_mesh(
        &self,
        queue: &wgpu::Queue,
        brick_meta_buf: &wgpu::Buffer,
        voxel_data_buf: &wgpu::Buffer,
        voxel_size: f32,
        range: BrickRange,
    ) -> Vec<(u32, [f32; 3])> {
        let mut touched = Vec::new();
        for bz in range.min[2]..=range.max[2] {
            for by in range.min[1]..=range.max[1] {
                for bx in range.min[0]..=range.max[0] {
                    let brick_idx =
                        bz * BRICKS_PER_AXIS * BRICKS_PER_AXIS + by * BRICKS_PER_AXIS + bx;
                    let mut brick_words = [0u32; WORDS_PER_BRICK];
                    let occupied = self.bake_brick(bx, by, bz, &mut brick_words);

                    let data_offset = brick_idx * WORDS_PER_BRICK as u32;
                    // VoxelMeshPass's GpuBrickMeta is two plain u32 fields
                    // (data_offset, occupancy) — unlike VoxelRayMarchPass's
                    // packed single word, see voxel_surface_extract.wgsl.
                    let meta = [data_offset, occupied as u32];

                    queue.write_buffer(
                        brick_meta_buf,
                        (brick_idx as u64) * 8,
                        bytemuck::cast_slice(&meta),
                    );
                    queue.write_buffer(
                        voxel_data_buf,
                        (data_offset as u64) * 4,
                        bytemuck::cast_slice(&brick_words),
                    );

                    touched.push((brick_idx, Self::brick_origin(bx, by, bz, voxel_size)));
                }
            }
        }
        touched
    }

    /// Bakes and uploads every brick in the volume. See `upload_range_mesh`.
    pub fn upload_all_mesh(
        &self,
        queue: &wgpu::Queue,
        brick_meta_buf: &wgpu::Buffer,
        voxel_data_buf: &wgpu::Buffer,
        voxel_size: f32,
    ) -> Vec<(u32, [f32; 3])> {
        self.upload_range_mesh(
            queue,
            brick_meta_buf,
            voxel_data_buf,
            voxel_size,
            BrickRange {
                min: [0, 0, 0],
                max: [
                    BRICKS_PER_AXIS - 1,
                    BRICKS_PER_AXIS - 1,
                    BRICKS_PER_AXIS - 1,
                ],
            },
        )
    }

    /// Bakes a brick's raw (unpadded) 8x8x8 voxel block for VoxelRayMarchPass.
    /// Matches voxel_raymarch.wgsl::read_voxel's `linear = z*64 + y*8 + x`.
    fn bake_brick_raymarch(
        &self,
        bx: u32,
        by: u32,
        bz: u32,
        data_out: &mut [u32; RAYMARCH_WORDS_PER_BRICK],
    ) -> bool {
        let mut occupied = false;
        for lz in 0..BRICK_DIM {
            for ly in 0..BRICK_DIM {
                for lx in 0..BRICK_DIM {
                    let gx = bx * BRICK_DIM + lx;
                    let gy = by * BRICK_DIM + ly;
                    let gz = bz * BRICK_DIM + lz;
                    let mat = self.materials[Self::idx(gx, gy, gz)];
                    if mat != MAT_AIR {
                        occupied = true;
                    }
                    let linear = (lz * BRICK_DIM * BRICK_DIM + ly * BRICK_DIM + lx) as usize;
                    let word = linear / 4;
                    let byte_in_word = linear % 4;
                    data_out[word] |= (mat as u32) << (byte_in_word * 8);
                }
            }
        }
        occupied
    }

    /// Re-bakes and uploads the bricks touched by a `BrickRange` into the
    /// scene's shared `voxel_brick_pool`/`voxel_data_pool` (VoxelRayMarchPass).
    /// GpuBrickMeta here is a single packed word: occupancy in the top byte,
    /// data_offset in the low 24 bits — see voxel_raymarch.wgsl's meta mask.
    pub fn upload_range_raymarch(
        &self,
        queue: &wgpu::Queue,
        brick_pool: &wgpu::Buffer,
        data_pool: &wgpu::Buffer,
        range: BrickRange,
    ) {
        for bz in range.min[2]..=range.max[2] {
            for by in range.min[1]..=range.max[1] {
                for bx in range.min[0]..=range.max[0] {
                    let brick_idx =
                        bz * BRICKS_PER_AXIS * BRICKS_PER_AXIS + by * BRICKS_PER_AXIS + bx;
                    let mut brick_words = [0u32; RAYMARCH_WORDS_PER_BRICK];
                    let occupied = self.bake_brick_raymarch(bx, by, bz, &mut brick_words);

                    let data_offset = brick_idx * RAYMARCH_WORDS_PER_BRICK as u32;
                    let meta_word = if occupied {
                        (1u32 << 24) | data_offset
                    } else {
                        0
                    };

                    queue.write_buffer(
                        brick_pool,
                        (brick_idx as u64) * 2 * 4,
                        bytemuck::bytes_of(&meta_word),
                    );
                    queue.write_buffer(
                        data_pool,
                        (data_offset as u64) * 4,
                        bytemuck::cast_slice(&brick_words),
                    );
                }
            }
        }
    }

    /// Uploads a full bake to the shared GPU voxel pools. See `upload_range_raymarch`.
    pub fn upload_all_raymarch(
        &self,
        queue: &wgpu::Queue,
        brick_pool: &wgpu::Buffer,
        data_pool: &wgpu::Buffer,
    ) {
        self.upload_range_raymarch(
            queue,
            brick_pool,
            data_pool,
            BrickRange {
                min: [0, 0, 0],
                max: [
                    BRICKS_PER_AXIS - 1,
                    BRICKS_PER_AXIS - 1,
                    BRICKS_PER_AXIS - 1,
                ],
            },
        )
    }
}

#[derive(Clone, Copy)]
pub struct BrickRange {
    min: [u32; 3],
    max: [u32; 3],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_terrain_contains_air_and_solid_voxels() {
        let mut terrain = VoxelTerrain::empty();
        terrain.generate(1);

        assert!(terrain.materials.contains(&MAT_AIR));
        assert!(terrain.materials.iter().any(|&material| material != MAT_AIR));
    }

    #[test]
    fn sphere_edits_update_the_dense_material_grid() {
        let mut terrain = VoxelTerrain::empty();
        let center = [32.0, 32.0, 32.0];
        let center_index = VoxelTerrain::idx(32, 32, 32);

        assert!(terrain.paint_sphere(center, 2.0, MAT_ORE, true).is_some());
        assert_eq!(terrain.materials[center_index], MAT_ORE);

        assert!(terrain.paint_sphere(center, 2.0, MAT_ORE, false).is_some());
        assert_eq!(terrain.materials[center_index], MAT_AIR);
    }
}
