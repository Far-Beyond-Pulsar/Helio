//! CPU-side voxel terrain component and brick baker.
//!
//! Both passes read a dense per-volume brick grid, but from different GPU
//! buffers with different layouts:
//! - `VoxelMeshPass` owns its own `brick_meta_buf`/`voxel_data_buf`, storing
//!   each brick as a **padded 9x9x9** block (`upload_*_mesh`) so its
//!   marching-cubes extract pass can read one voxel of +X/+Y/+Z halo and
//!   close the seam between adjacent bricks (see `voxel_surface_extract.wgsl`
//!   /`CELLS_PER_DIM`). It re-extracts real triangles on `mark_dirty()`.
//! - `VoxelRayMarchPass` exposes its own bounded brick/data buffers and accepts
//!   explicit uploads (`upload_*_raymarch`), storing each brick as the raw
//!   **8x8x8** the DDA marcher indexes every frame.
//!
//! The grid is always a dense 64^3 voxel volume (8 bricks per axis of 8
//! voxels each — fixed GPU-side by the engine's `BRICK_SIZE` constant).

pub const BRICK_DIM: u32 = 8;
pub const BRICKS_PER_AXIS: u32 = 8;
pub const VOXEL_TERRAIN_GRID_DIM: u32 = BRICKS_PER_AXIS * BRICK_DIM; // 64
/// Surface extraction modes exposed to host applications.
pub const VOXEL_MODE_SURFACE: u32 = 0;
pub const VOXEL_MODE_CUBES: u32 = 1;
// Both-sided halo matches voxel_surface_extract.wgsl and the SceneDB codec.
pub const PADDED_DIM: u32 = BRICK_DIM + 2; // 10
pub const PADDED_VOXELS_PER_BRICK: usize = (PADDED_DIM * PADDED_DIM * PADDED_DIM) as usize; // 1000
pub const WORDS_PER_BRICK: usize = PADDED_VOXELS_PER_BRICK.div_ceil(4); // 250

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
pub const MAT_SAND: u8 = 5;
pub const MAT_WATER: u8 = 6;
pub const MAT_BEDROCK: u8 = 7;
pub const MAT_COAL_ORE: u8 = 8;
pub const MAT_IRON_ORE: u8 = 9;
pub const MAT_GOLD_ORE: u8 = 10;
pub const MAT_LOG: u8 = 11;
pub const MAT_LEAVES: u8 = 12;

/// One caller-authored voxel mutation. Coordinates are local to a component.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VoxelBlockUpdate {
    pub position: [u32; 3],
    pub material: u8,
}

/// Inclusive rectangular area mutation. Coordinates are local to a component.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VoxelAreaUpdate {
    pub min: [u32; 3],
    pub max: [u32; 3],
    pub material: u8,
}

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
#[derive(Clone)]
pub struct VoxelTerrain {
    materials: Vec<u8>,
    dimensions: [u32; 3],
}

/// SceneDB-owned voxel volume instance.
///
/// The dense voxel payload is a SceneDB variable-length GPU field, while the
/// transform and identity are ordinary SceneDB GPU columns. Every entity may
/// carry one of these; there is deliberately no singleton world assumption.
#[derive(pulsar_scenedb_derive::SceneStore, Clone, Debug)]
pub struct VoxelComponent {
    #[gpu(buffer = "voxel_materials", mirror = DirtyTracked)]
    pub materials: Vec<u8>,
    #[gpu(buffer = "voxel_transforms")]
    pub local_to_world: [f32; 16],
    #[gpu(buffer = "voxel_volume_params")]
    pub volume_params: [f32; 4],
    #[gpu(buffer = "voxel_volume_ids")]
    pub volume_id_seed: [u32; 3],
    #[gpu(buffer = "voxel_dimensions")]
    pub dimensions: [u32; 3],
    #[gpu(buffer = "voxel_generation")]
    pub generation: [f32; 16],
    #[gpu(buffer = "voxel_generation_flags")]
    pub generation_flags: [u32; 3],
    #[gpu(buffer = "voxel_streaming")]
    pub streaming: [u32; 3],
    #[gpu(buffer = "voxel_editing")]
    pub editing: [f32; 12],
    #[gpu(buffer = "voxel_palette", mirror = DirtyTracked)]
    pub palette: Vec<[f32; 4]>,
    /// `[mode, flags, reserved]`; mode selects surface or blocky
    /// cube extraction for this volume.
    #[gpu(buffer = "voxel_render_settings")]
    pub render_settings: [u32; 3],
}

impl VoxelComponent {
    pub fn new(terrain: VoxelTerrain, voxel_size: f32, volume_id: u32) -> Self {
        let dimensions = terrain.dimensions;
        Self {
            materials: terrain.materials,
            local_to_world: glam::Mat4::IDENTITY.to_cols_array(),
            volume_params: [voxel_size, dimensions[0] as f32, 0.0, 0.0],
            volume_id_seed: [volume_id, 0, 0],
            dimensions,
            generation: [0.0; 16],
            generation_flags: [0; 3],
            streaming: [0; 3],
            editing: [0.0; 12],
            palette: Vec::new(),
            render_settings: [VOXEL_MODE_SURFACE, 0, 0],
        }
    }

    pub fn with_palette(
        terrain: VoxelTerrain,
        voxel_size: f32,
        volume_id: u32,
        palette: Vec<[f32; 4]>,
    ) -> Self {
        let mut component = Self::new(terrain, voxel_size, volume_id);
        component.palette = palette;
        component
    }

    pub fn dimensions(&self) -> [u32; 3] {
        self.dimensions
    }

    pub fn set_render_mode(&mut self, mode: u32) {
        self.render_settings[0] = mode;
    }

    pub fn render_mode(&self) -> u32 {
        self.render_settings[0]
    }

    pub fn voxel_size(&self) -> f32 {
        self.volume_params[0]
    }

    pub fn volume_id(&self) -> u32 {
        self.volume_id_seed[0]
    }

    pub fn seed(&self) -> u32 {
        self.volume_id_seed[1]
    }

    pub fn set_seed(&mut self, seed: u32) {
        self.volume_id_seed[1] = seed;
    }

    fn terrain_snapshot(&self) -> VoxelTerrain {
        VoxelTerrain {
            materials: self.materials.clone(),
            dimensions: self.dimensions,
        }
    }

    pub fn generate(&mut self, seed: u32) {
        let mut terrain = self.terrain_snapshot();
        let (chunk_x, chunk_z) = self.chunk_coord();
        terrain.generate_at(seed, [chunk_x, chunk_z]);
        self.materials = terrain.materials;
        self.set_seed(seed);
    }

    /// Generate this component as a deterministic infinite-world chunk.
    /// Chunk coordinates are signed and encoded in the SceneDB streaming row.
    pub fn generate_chunk(&mut self, seed: u32, chunk_x: i32, chunk_z: i32) {
        self.set_chunk_coord(chunk_x, chunk_z);
        self.generate(seed);
    }

    pub fn set_chunk_coord(&mut self, chunk_x: i32, chunk_z: i32) {
        self.streaming[0] = chunk_x as u32;
        self.streaming[2] = chunk_z as u32;
        let world_x = chunk_x as f32 * self.dimensions[0] as f32 * self.voxel_size();
        let world_z = chunk_z as f32 * self.dimensions[2] as f32 * self.voxel_size();
        self.local_to_world =
            glam::Mat4::from_translation(glam::vec3(world_x, 0.0, world_z)).to_cols_array();
    }

    pub fn chunk_coord(&self) -> (i32, i32) {
        (self.streaming[0] as i32, self.streaming[2] as i32)
    }

    pub fn paint_sphere(
        &mut self,
        center: [f32; 3],
        radius: f32,
        material: u8,
        add: bool,
    ) -> Option<BrickRange> {
        let mut terrain = self.terrain_snapshot();
        let range = terrain.paint_sphere(center, radius, material, add);
        self.materials = terrain.materials;
        range
    }

    /// Apply N arbitrary block edits and return the combined dirty-brick range.
    pub fn apply_block_updates(&mut self, updates: &[VoxelBlockUpdate]) -> Option<BrickRange> {
        let mut terrain = self.terrain_snapshot();
        let range = terrain.apply_block_updates(updates);
        self.materials = terrain.materials;
        range
    }

    /// Apply N inclusive rectangular area edits and return the combined dirty
    /// range. Overlapping updates are applied in input order.
    pub fn apply_area_updates(&mut self, updates: &[VoxelAreaUpdate]) -> Option<BrickRange> {
        let mut terrain = self.terrain_snapshot();
        let range = terrain.apply_area_updates(updates);
        self.materials = terrain.materials;
        range
    }

    pub fn set_block(&mut self, position: [u32; 3], material: u8) -> Option<BrickRange> {
        self.apply_block_updates(&[VoxelBlockUpdate { position, material }])
    }

    pub fn upload_all_mesh(
        &self,
        queue: &wgpu::Queue,
        brick_meta_buf: &wgpu::Buffer,
        voxel_data_buf: &wgpu::Buffer,
    ) -> Vec<(u32, [f32; 3], bool)> {
        self.terrain_snapshot().upload_all_mesh(
            queue,
            brick_meta_buf,
            voxel_data_buf,
            self.voxel_size(),
        )
    }

    pub fn upload_range_mesh(
        &self,
        queue: &wgpu::Queue,
        brick_meta_buf: &wgpu::Buffer,
        voxel_data_buf: &wgpu::Buffer,
        range: BrickRange,
    ) -> Vec<(u32, [f32; 3], bool)> {
        self.terrain_snapshot().upload_range_mesh(
            queue,
            brick_meta_buf,
            voxel_data_buf,
            self.voxel_size(),
            range,
        )
    }
}

impl VoxelTerrain {
    pub fn empty() -> Self {
        Self::with_dimensions([VOXEL_TERRAIN_GRID_DIM; 3])
    }

    pub fn with_dimensions(dimensions: [u32; 3]) -> Self {
        Self {
            materials: vec![MAT_AIR; (dimensions[0] * dimensions[1] * dimensions[2]) as usize],
            dimensions,
        }
    }

    fn idx(&self, x: u32, y: u32, z: u32) -> usize {
        (x + y * self.dimensions[0] + z * self.dimensions[0] * self.dimensions[1]) as usize
    }

    fn set_material(&mut self, x: u32, y: u32, z: u32, material: u8) {
        let index = self.idx(x, y, z);
        self.materials[index] = material;
    }

    fn in_bounds(&self, x: i32, y: i32, z: i32) -> bool {
        x >= 0
            && y >= 0
            && z >= 0
            && (x as u32) < self.dimensions[0]
            && (y as u32) < self.dimensions[1]
            && (z as u32) < self.dimensions[2]
    }

    /// Fills the grid with a Minecraft-style block world: bedrock, stone,
    /// dirt/grass strata, beaches, sea water, caves, ores, and trees.
    pub fn generate(&mut self, seed: u32) {
        self.generate_at(seed, [0, 0]);
    }

    fn merge_dirty_range(dirty: &mut Option<BrickRange>, position: [u32; 3]) {
        let brick = [
            position[0] / BRICK_DIM,
            position[1] / BRICK_DIM,
            position[2] / BRICK_DIM,
        ];
        if let Some(range) = dirty {
            for axis in 0..3 {
                range.min[axis] = range.min[axis].min(brick[axis]);
                range.max[axis] = range.max[axis].max(brick[axis]);
            }
        } else {
            *dirty = Some(BrickRange {
                min: brick,
                max: brick,
            });
        }
    }

    pub fn apply_block_updates(&mut self, updates: &[VoxelBlockUpdate]) -> Option<BrickRange> {
        let mut dirty = None;
        for update in updates {
            let [x, y, z] = update.position;
            if x >= self.dimensions[0] || y >= self.dimensions[1] || z >= self.dimensions[2] {
                continue;
            }
            self.set_material(x, y, z, update.material);
            Self::merge_dirty_range(&mut dirty, update.position);
        }
        dirty
    }

    pub fn apply_area_updates(&mut self, updates: &[VoxelAreaUpdate]) -> Option<BrickRange> {
        let mut dirty = None;
        for update in updates {
            let min = [
                update.min[0].min(self.dimensions[0] - 1),
                update.min[1].min(self.dimensions[1] - 1),
                update.min[2].min(self.dimensions[2] - 1),
            ];
            let max = [
                update.max[0].min(self.dimensions[0] - 1),
                update.max[1].min(self.dimensions[1] - 1),
                update.max[2].min(self.dimensions[2] - 1),
            ];
            if min.iter().zip(max.iter()).any(|(lo, hi)| lo > hi) {
                continue;
            }
            for z in min[2]..=max[2] {
                for y in min[1]..=max[1] {
                    for x in min[0]..=max[0] {
                        self.set_material(x, y, z, update.material);
                    }
                }
            }
            let area_max = [max[0], max[1], max[2]];
            Self::merge_dirty_range(&mut dirty, min);
            Self::merge_dirty_range(&mut dirty, area_max);
        }
        dirty
    }

    /// Generate a chunk using global voxel coordinates. This makes adjacent
    /// component instances agree at chunk boundaries without a world-sized
    /// allocation or a singleton terrain assumption.
    pub fn generate_at(&mut self, seed: u32, chunk: [i32; 2]) {
        let sea_level = 25.0;
        let base_height = 29.0;
        let amplitude = 13.0;
        let freq = 0.055;

        for x in 0..self.dimensions[0] {
            for z in 0..self.dimensions[2] {
                let wx = x as i32 + chunk[0] * self.dimensions[0] as i32;
                let wz = z as i32 + chunk[1] * self.dimensions[2] as i32;
                let h = fbm2(wx as f32 * freq, wz as f32 * freq, seed, 4);
                let terrain_height = base_height + h * amplitude;

                for y in 0..self.dimensions[1] {
                    let yf = y as f32;
                    if y == 0 {
                        self.set_material(x, y, z, MAT_BEDROCK);
                    } else if yf > terrain_height {
                        self.set_material(
                            x,
                            y,
                            z,
                            if yf <= sea_level { MAT_WATER } else { MAT_AIR },
                        );
                        continue;
                    }

                    let depth = terrain_height - yf;
                    let near_beach = terrain_height <= sea_level + 1.5;
                    let mut mat = if near_beach && depth < 3.0 {
                        MAT_SAND
                    } else if depth < 1.0 {
                        MAT_GRASS
                    } else if depth < 4.0 {
                        MAT_DIRT
                    } else {
                        MAT_STONE
                    };

                    if mat == MAT_STONE {
                        let cave = fbm2(
                            wx as f32 * 0.11,
                            wz as f32 * 0.11 + y as f32 * 0.07,
                            seed ^ 0x51_7A_9E,
                            3,
                        );
                        if y > 5 && y < terrain_height as u32 - 2 && cave > 0.78 {
                            mat = MAT_AIR;
                        } else {
                            let ore = hash(wx, y as i32, wz, seed ^ 0x1234_5678);
                            mat = if y < 12 && ore > 0.965 {
                                MAT_GOLD_ORE
                            } else if y < 28 && ore > 0.94 {
                                MAT_IRON_ORE
                            } else if ore > 0.90 {
                                MAT_COAL_ORE
                            } else {
                                mat
                            };
                        }
                    }

                    self.set_material(x, y, z, mat);
                }
            }
        }

        // Sparse, deterministic trees make the result read as a block world
        // without overwhelming the surface extraction budget.
        for x in 3..self.dimensions[0].saturating_sub(3) {
            for z in 3..self.dimensions[2].saturating_sub(3) {
                let wx = x as i32 + chunk[0] * self.dimensions[0] as i32;
                let wz = z as i32 + chunk[1] * self.dimensions[2] as i32;
                if hash(wx, 91, wz, seed ^ 0x7E_2A) < 0.93 {
                    continue;
                }
                let h = (base_height
                    + fbm2(wx as f32 * freq, wz as f32 * freq, seed, 4) * amplitude)
                    as i32;
                if !(1..(self.dimensions[1] as i32 - 7)).contains(&h) {
                    continue;
                }
                for trunk_y in h..h + 4 {
                    self.set_material(x, trunk_y as u32, z, MAT_LOG);
                }
                for dz in -2i32..=2 {
                    for dx in -2i32..=2 {
                        if dx.abs() + dz.abs() <= 3 {
                            let lx = (x as i32 + dx) as u32;
                            let lz = (z as i32 + dz) as u32;
                            self.set_material(lx, (h + 4) as u32, lz, MAT_LEAVES);
                        }
                    }
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
        let mut min = [
            self.dimensions[0] as i32,
            self.dimensions[1] as i32,
            self.dimensions[2] as i32,
        ];
        let mut max = [-1i32; 3];

        for dz in -r..=r {
            for dy in -r..=r {
                for dx in -r..=r {
                    let d2 = (dx * dx + dy * dy + dz * dz) as f32;
                    if d2 > r2 {
                        continue;
                    }
                    let (x, y, z) = (cx + dx, cy + dy, cz + dz);
                    if !self.in_bounds(x, y, z) {
                        continue;
                    }
                    self.set_material(
                        x as u32,
                        y as u32,
                        z as u32,
                        if add { material } else { MAT_AIR },
                    );
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

    /// Bakes a 10³ brick input with local sample coordinates -1..=8.
    fn bake_brick(&self, bx: u32, by: u32, bz: u32, data_out: &mut [u32; WORDS_PER_BRICK]) -> bool {
        let mut occupied = false;
        for lz in 0..PADDED_DIM {
            for ly in 0..PADDED_DIM {
                for lx in 0..PADDED_DIM {
                    let gx = (bx * BRICK_DIM) as i64 + lx as i64 - 1;
                    let gy = (by * BRICK_DIM) as i64 + ly as i64 - 1;
                    let gz = (bz * BRICK_DIM) as i64 + lz as i64 - 1;
                    let mat = if gx >= 0
                        && gy >= 0
                        && gz >= 0
                        && gx < i64::from(self.dimensions[0])
                        && gy < i64::from(self.dimensions[1])
                        && gz < i64::from(self.dimensions[2])
                    {
                        self.materials[self.idx(gx as u32, gy as u32, gz as u32)]
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
    fn brick_origin(&self, bx: u32, by: u32, bz: u32, voxel_size: f32) -> [f32; 3] {
        let half = [
            self.dimensions[0] as f32,
            self.dimensions[1] as f32,
            self.dimensions[2] as f32,
        ];
        let gx = (bx * BRICK_DIM) as f32 - half[0] * 0.5;
        let gy = (by * BRICK_DIM) as f32 - half[1] * 0.5;
        let gz = (bz * BRICK_DIM) as f32 - half[2] * 0.5;
        [gx * voxel_size, gy * voxel_size, gz * voxel_size]
    }

    fn brick_dims(&self) -> [u32; 3] {
        [
            self.dimensions[0].div_ceil(BRICK_DIM),
            self.dimensions[1].div_ceil(BRICK_DIM),
            self.dimensions[2].div_ceil(BRICK_DIM),
        ]
    }

    /// Re-bakes and uploads the bricks touched by a `BrickRange` into
    /// `VoxelMeshPass`'s buffers. Returns `(brick_idx, origin, occupied)` for
    /// every touched brick — the caller must `mark_dirty()` each
    /// one so the extract pass re-runs (an emptied brick needs to re-extract
    /// to zero triangles too, not just a newly-filled one).
    pub fn upload_range_mesh(
        &self,
        queue: &wgpu::Queue,
        brick_meta_buf: &wgpu::Buffer,
        voxel_data_buf: &wgpu::Buffer,
        voxel_size: f32,
        range: BrickRange,
    ) -> Vec<(u32, [f32; 3], bool)> {
        let mut touched = Vec::new();
        for bz in range.min[2]..=range.max[2] {
            for by in range.min[1]..=range.max[1] {
                for bx in range.min[0]..=range.max[0] {
                    let brick_dims = self.brick_dims();
                    let brick_idx = bz * brick_dims[0] * brick_dims[1] + by * brick_dims[0] + bx;
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

                    touched.push((
                        brick_idx,
                        self.brick_origin(bx, by, bz, voxel_size),
                        occupied,
                    ));
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
    ) -> Vec<(u32, [f32; 3], bool)> {
        self.upload_range_mesh(
            queue,
            brick_meta_buf,
            voxel_data_buf,
            voxel_size,
            BrickRange {
                min: [0, 0, 0],
                max: [
                    self.brick_dims()[0] - 1,
                    self.brick_dims()[1] - 1,
                    self.brick_dims()[2] - 1,
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
                    let mat = self.materials[self.idx(gx, gy, gz)];
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
    /// explicitly supplied `VoxelRayMarchPass` buffers.
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
                    let brick_dims = self.brick_dims();
                    let brick_idx = bz * brick_dims[0] * brick_dims[1] + by * brick_dims[0] + bx;
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

    /// Uploads a full bake to explicitly supplied raymarch GPU pools. See
    /// `upload_range_raymarch`.
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
                    self.brick_dims()[0] - 1,
                    self.brick_dims()[1] - 1,
                    self.brick_dims()[2] - 1,
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
        assert!(terrain
            .materials
            .iter()
            .any(|&material| material != MAT_AIR));
    }

    #[test]
    fn sphere_edits_update_the_dense_material_grid() {
        let mut terrain = VoxelTerrain::empty();
        let center = [32.0, 32.0, 32.0];
        let center_index = terrain.idx(32, 32, 32);

        assert!(terrain.paint_sphere(center, 2.0, MAT_ORE, true).is_some());
        assert_eq!(terrain.materials[center_index], MAT_ORE);

        assert!(terrain.paint_sphere(center, 2.0, MAT_ORE, false).is_some());
        assert_eq!(terrain.materials[center_index], MAT_AIR);
    }
}
