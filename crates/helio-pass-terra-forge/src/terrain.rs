use crate::gpu_types::{
    BrickMeta, ChunkInfo, EditOp, GenUniforms, BRICK_DIM, BRICK_EMPTY, BRICKS_PER_CHUNK,
    CHUNK_DIM_BRICKS, INDIR_EMPTY, INDIR_GRID_DIM, MAX_EDITS, MAX_LOADED_CHUNKS,
    WORDS_PER_BRICK,
};
use crate::TerraForgePass;
use bytemuck::Zeroable;

// ── CPU brick-map generation (tests) ───────────────────────────────────────────

pub struct BrickMapData {
    pub brick_grid: Vec<BrickMeta>,
    pub voxel_pool: Vec<u32>,
    pub allocated_bricks: u32,
}

pub fn generate_sphere_brickmap(
    grid_dim_bricks: u32,
    brick_dim: u32,
    radius_voxels: f32,
) -> BrickMapData {
    let total_voxels_per_axis = grid_dim_bricks * brick_dim;
    let center = total_voxels_per_axis as f32 * 0.5;
    let r2 = radius_voxels * radius_voxels;
    let words_per_brick = (brick_dim * brick_dim * brick_dim / 4) as usize;

    let total_bricks = (grid_dim_bricks * grid_dim_bricks * grid_dim_bricks) as usize;
    let mut brick_grid = vec![
        BrickMeta {
            data_offset: BRICK_EMPTY,
            occupancy: 0,
        };
        total_bricks
    ];

    let mut voxel_pool: Vec<u32> = Vec::new();
    let mut next_slot = 0u32;

    for bz in 0..grid_dim_bricks {
        for by in 0..grid_dim_bricks {
            for bx in 0..grid_dim_bricks {
                let brick_voxel_min_x = bx * brick_dim;
                let brick_voxel_min_y = by * brick_dim;
                let brick_voxel_min_z = bz * brick_dim;
                let brick_voxel_max_x = brick_voxel_min_x + brick_dim;
                let brick_voxel_max_y = brick_voxel_min_y + brick_dim;
                let brick_voxel_max_z = brick_voxel_min_z + brick_dim;

                let cx = (center).clamp(brick_voxel_min_x as f32, brick_voxel_max_x as f32);
                let cy = (center).clamp(brick_voxel_min_y as f32, brick_voxel_max_y as f32);
                let cz = (center).clamp(brick_voxel_min_z as f32, brick_voxel_max_z as f32);
                let dx = cx - center;
                let dy = cy - center;
                let dz = cz - center;
                if dx * dx + dy * dy + dz * dz > r2 {
                    continue;
                }

                let mut brick_words = vec![0u32; words_per_brick];
                let mut occ = 0u32;

                for lz in 0..brick_dim {
                    let gz = brick_voxel_min_z + lz;
                    let ddz = gz as f32 + 0.5 - center;
                    for ly in 0..brick_dim {
                        let gy = brick_voxel_min_y + ly;
                        let ddy = gy as f32 + 0.5 - center;
                        for lx in 0..brick_dim {
                            let gx = brick_voxel_min_x + lx;
                            let ddx = gx as f32 + 0.5 - center;
                            let dist2 = ddx * ddx + ddy * ddy + ddz * ddz;
                            if dist2 <= r2 {
                                let local_idx =
                                    (lx + ly * brick_dim + lz * brick_dim * brick_dim) as usize;
                                let word = local_idx / 4;
                                let byte_shift = (local_idx % 4) * 8;
                                brick_words[word] |= 1u32 << byte_shift;
                                occ += 1;
                            }
                        }
                    }
                }

                if occ > 0 {
                    let brick_idx =
                        (bx + by * grid_dim_bricks + bz * grid_dim_bricks * grid_dim_bricks)
                            as usize;
                    brick_grid[brick_idx] = BrickMeta {
                        data_offset: next_slot,
                        occupancy: occ,
                    };
                    voxel_pool.extend_from_slice(&brick_words);
                    next_slot += 1;
                }
            }
        }
    }

    BrickMapData {
        brick_grid,
        voxel_pool,
        allocated_bricks: next_slot,
    }
}

// ── Chunk slot tracking ──────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub(crate) struct ChunkSlot {
    pub pos: [i32; 3],
    pub loaded: bool,
    pub last_used_frame: u64,
}

impl Default for ChunkSlot {
    fn default() -> Self {
        Self {
            pos: [0; 3],
            loaded: false,
            last_used_frame: 0,
        }
    }
}

// ── Public edit API ─────────────────────────────────────────────────────────

impl TerraForgePass {
    pub fn add_edit(&mut self, edit: EditOp) {
        if self.edits.len() >= MAX_EDITS as usize {
            log::warn!("Terra Forge: edit buffer full ({} max)", MAX_EDITS);
            return;
        }
        let edit_r = edit.size[0].max(edit.size[1]).max(edit.size[2]) + edit.blend_k;
        for slot in &mut self.chunk_slots {
            if !slot.loaded {
                continue;
            }
            let chunk_min = [
                slot.pos[0] as f32 * self.chunk_world_size,
                slot.pos[1] as f32 * self.chunk_world_size,
                slot.pos[2] as f32 * self.chunk_world_size,
            ];
            let chunk_max = [
                chunk_min[0] + self.chunk_world_size,
                chunk_min[1] + self.chunk_world_size,
                chunk_min[2] + self.chunk_world_size,
            ];
            if sphere_aabb_test(edit.position, edit_r, chunk_min, chunk_max) {
                slot.loaded = false;
            }
        }
        self.edits.push(edit);
        self.edits_dirty = true;
        log::info!("Terra Forge: edit added ({} total)", self.edits.len());
    }
}

// ── Chunk streaming ──────────────────────────────────────────────────────────

impl TerraForgePass {
    pub(crate) fn find_surface_chunks_near(
        cam_pos: [f32; 3],
        planet_radius: f32,
        chunk_world_size: f32,
    ) -> Vec<[i32; 3]> {
        let cam_chunk = [
            (cam_pos[0] / chunk_world_size).floor() as i32,
            (cam_pos[1] / chunk_world_size).floor() as i32,
            (cam_pos[2] / chunk_world_size).floor() as i32,
        ];
        let half_grid = (INDIR_GRID_DIM / 2) as i32;

        let mut chunks: Vec<([i32; 3], f32)> = Vec::new();

        for dz in -half_grid..half_grid {
            for dy in -half_grid..half_grid {
                for dx in -half_grid..half_grid {
                    let cx = cam_chunk[0] + dx;
                    let cy = cam_chunk[1] + dy;
                    let cz = cam_chunk[2] + dz;

                    let cmin = [
                        cx as f32 * chunk_world_size,
                        cy as f32 * chunk_world_size,
                        cz as f32 * chunk_world_size,
                    ];
                    let cmax = [
                        (cx + 1) as f32 * chunk_world_size,
                        (cy + 1) as f32 * chunk_world_size,
                        (cz + 1) as f32 * chunk_world_size,
                    ];

                    let nx = 0.0f32.clamp(cmin[0], cmax[0]);
                    let ny = 0.0f32.clamp(cmin[1], cmax[1]);
                    let nz = 0.0f32.clamp(cmin[2], cmax[2]);
                    let near_dist2 = nx * nx + ny * ny + nz * nz;

                    let fx = if cmin[0].abs() > cmax[0].abs() {
                        cmin[0]
                    } else {
                        cmax[0]
                    };
                    let fy = if cmin[1].abs() > cmax[1].abs() {
                        cmin[1]
                    } else {
                        cmax[1]
                    };
                    let fz = if cmin[2].abs() > cmax[2].abs() {
                        cmin[2]
                    } else {
                        cmax[2]
                    };
                    let far_dist2 = fx * fx + fy * fy + fz * fz;

                    let inner_r = planet_radius * 0.90;
                    let outer_r = planet_radius * 1.10;

                    if far_dist2 >= inner_r * inner_r && near_dist2 <= outer_r * outer_r {
                        let center = [
                            (cx as f32 + 0.5) * chunk_world_size,
                            (cy as f32 + 0.5) * chunk_world_size,
                            (cz as f32 + 0.5) * chunk_world_size,
                        ];
                        let cam_dist2 = (center[0] - cam_pos[0]).powi(2)
                            + (center[1] - cam_pos[1]).powi(2)
                            + (center[2] - cam_pos[2]).powi(2);
                        chunks.push(([cx, cy, cz], cam_dist2));
                    }
                }
            }
        }

        chunks.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        chunks.into_iter().map(|(pos, _)| pos).collect()
    }

    pub(crate) fn find_free_slot(&self) -> Option<usize> {
        self.chunk_slots.iter().position(|s| !s.loaded)
    }

    pub(crate) fn evict_lru_chunk(&mut self) -> Option<usize> {
        self.chunk_slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.loaded)
            .min_by_key(|(_, s)| s.last_used_frame)
            .map(|(i, _)| i)
    }

    pub(crate) fn clear_chunk_gpu_data(&self, slot_idx: usize, queue: &wgpu::Queue) {
        let bp_off = slot_idx as u64 * BRICKS_PER_CHUNK as u64;
        let empty_bricks = vec![
            BrickMeta {
                data_offset: BRICK_EMPTY,
                occupancy: 0,
            };
            BRICKS_PER_CHUNK as usize
        ];
        queue.write_buffer(
            &self.brick_pool_buf,
            bp_off * std::mem::size_of::<BrickMeta>() as u64,
            bytemuck::cast_slice(&empty_bricks),
        );

        let vp_off = slot_idx as u64 * self.effective_max_mixed as u64;
        let vp_words = self.effective_max_mixed as u64 * WORDS_PER_BRICK as u64;
        let vp_bytes = vp_words * 4;
        const CLEAR_CHUNK: u64 = 1024 * 1024;
        let mut offset = 0u64;
        let zeros = vec![0u8; CLEAR_CHUNK as usize];
        while offset < vp_bytes {
            let len = (vp_bytes - offset).min(CLEAR_CHUNK);
            queue.write_buffer(
                &self.voxel_pool_buf,
                vp_off * WORDS_PER_BRICK as u64 * 4 + offset,
                &zeros[..len as usize],
            );
            offset += len;
        }
    }

    pub(crate) fn rebuild_indir_grid(&mut self, queue: &wgpu::Queue) {
        let dim = INDIR_GRID_DIM as i32;
        self.indir_grid_cpu.fill(INDIR_EMPTY);

        for (slot_idx, slot) in self.chunk_slots.iter().enumerate() {
            if !slot.loaded {
                continue;
            }
            let ox = slot.pos[0] - self.indir_origin[0];
            let oy = slot.pos[1] - self.indir_origin[1];
            let oz = slot.pos[2] - self.indir_origin[2];
            if ox < 0 || ox >= dim || oy < 0 || oy >= dim || oz < 0 || oz >= dim {
                continue;
            }
            let ix = ((slot.pos[0] % dim) + dim) % dim;
            let iy = ((slot.pos[1] % dim) + dim) % dim;
            let iz = ((slot.pos[2] % dim) + dim) % dim;
            let flat = (ix + iy * dim + iz * dim * dim) as usize;
            self.indir_grid_cpu[flat] = slot_idx as u32;
        }

        queue.write_buffer(
            &self.indir_grid_buf,
            0,
            bytemuck::cast_slice(&self.indir_grid_cpu),
        );
    }

    pub(crate) fn upload_edits(&mut self, queue: &wgpu::Queue) {
        if !self.edits.is_empty() {
            queue.write_buffer(&self.edit_buf, 0, bytemuck::cast_slice(&self.edits));
        }
        self.edits_dirty = false;
    }

    pub(crate) fn generate_chunks(
        &mut self,
        positions: &[[i32; 3]],
        queue: &wgpu::Queue,
        device: &wgpu::Device,
        frame: u64,
    ) {
        if positions.is_empty() {
            return;
        }

        for &chunk_pos in positions {
            let slot_idx = match self.find_free_slot() {
                Some(i) => i,
                None => match self.evict_lru_chunk() {
                    Some(i) => {
                        self.chunk_slots[i].loaded = false;
                        self.chunk_table_cpu[i] = ChunkInfo::zeroed();
                        i
                    }
                    None => continue,
                },
            };

            self.clear_chunk_gpu_data(slot_idx, queue);

            let bp_off = slot_idx as u32 * BRICKS_PER_CHUNK;
            let vp_off = slot_idx as u32 * self.effective_max_mixed;

            let gen_uniforms = GenUniforms {
                chunk_dim_bricks: CHUNK_DIM_BRICKS,
                brick_dim: BRICK_DIM,
                voxel_size: self.voxel_size,
                planet_radius: self.planet_radius,
                chunk_world_origin: [
                    chunk_pos[0] as f32 * self.chunk_world_size,
                    chunk_pos[1] as f32 * self.chunk_world_size,
                    chunk_pos[2] as f32 * self.chunk_world_size,
                ],
                max_mixed_bricks: self.effective_max_mixed,
                brick_pool_offset: bp_off,
                voxel_pool_offset: vp_off,
                edit_count: self.edits.len() as u32,
                _pad1: 0,
            };

            queue.write_buffer(&self.gen_uniform_buf, 0, bytemuck::bytes_of(&gen_uniforms));
            queue.write_buffer(&self.alloc_counter_buf, 0, &[0u8; 4]);

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("TerraForge Gen"),
            });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("TerraForge Gen Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.gen_pipeline);
                cpass.set_bind_group(0, &self.gen_bg, &[]);
                cpass.dispatch_workgroups(CHUNK_DIM_BRICKS, CHUNK_DIM_BRICKS, CHUNK_DIM_BRICKS);
            }
            queue.submit(std::iter::once(encoder.finish()));

            self.chunk_slots[slot_idx] = ChunkSlot {
                pos: chunk_pos,
                loaded: true,
                last_used_frame: frame,
            };
            self.chunk_table_cpu[slot_idx] = ChunkInfo {
                pos: chunk_pos,
                status: 1,
                brick_pool_offset: bp_off,
                voxel_pool_offset: vp_off,
                _pad: [0; 2],
            };
        }

        queue.write_buffer(
            &self.chunk_table_buf,
            0,
            bytemuck::cast_slice(&self.chunk_table_cpu),
        );
    }

    #[cfg(test)]
    pub(crate) fn find_planet_chunks(planet_radius: f32, chunk_world_size: f32) -> Vec<[i32; 3]> {
        Self::find_surface_chunks_near(
            [0.0, 0.0, planet_radius * 2.0],
            planet_radius,
            chunk_world_size,
        )
    }
}

// ── Utility ──────────────────────────────────────────────────────────────────

pub(crate) fn sphere_aabb_test(
    center: [f32; 3],
    radius: f32,
    bmin: [f32; 3],
    bmax: [f32; 3],
) -> bool {
    let nx = center[0].clamp(bmin[0], bmax[0]);
    let ny = center[1].clamp(bmin[1], bmax[1]);
    let nz = center[2].clamp(bmin[2], bmax[2]);
    let dx = nx - center[0];
    let dy = ny - center[1];
    let dz = nz - center[2];
    dx * dx + dy * dy + dz * dz <= radius * radius
}
