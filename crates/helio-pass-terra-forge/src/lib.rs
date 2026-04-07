//! Terra Forge — Chunked brick-map voxel pass with dynamic streaming and SDF editing.
//!
//! Architecture: Multiple chunks (each 32³ bricks × 8³ voxels = 256³ voxels)
//! dynamically streamed around the camera into a pre-allocated GPU buffer pool.
//! An indirection grid provides O(1) chunk lookup during ray marching.
//! Far-field: procedural SDF sphere tracing for unloaded terrain.
//! Edits: Smooth CSG operations (Quilez-style) applied during generation.
//!
//! Generation: GPU compute shader evaluates terrain SDF + edits per chunk.
//! Rendering:  Chunk DDA → Brick DDA → Voxel DDA (compute), then fullscreen shade.

pub mod gpu_types;

use bytemuck::Zeroable;
use gpu_types::{
    BrickMeta, ChunkInfo, EditOp, GenUniforms, GpuMaterial, GpuUniforms, BRICKS_PER_CHUNK,
    BRICK_DIM, BRICK_EMPTY, CHUNK_DIM_BRICKS, INDIR_EMPTY, INDIR_GRID_DIM, MAX_EDITS,
    MAX_LOADED_CHUNKS, MAX_MIXED_BRICKS_PER_CHUNK, WORDS_PER_BRICK,
};
use helio_v3::{traits::RenderPass, PassContext, PrepareContext, Result};

/// Default voxel size in world units (0.1 m = 10 cm).
pub const VOXEL_SIZE: f32 = 0.1;

/// Default planet radius in world units.
pub const DEFAULT_PLANET_RADIUS: f32 = 1000.0;

/// Max chunks to generate per frame (streaming budget).
const CHUNKS_PER_FRAME: usize = 8;

// ── Brick-map sphere generation (CPU, kept for unit tests) ───────────────────

/// Result of CPU brick-map generation (used only in tests).
pub struct BrickMapData {
    pub brick_grid: Vec<BrickMeta>,
    pub voxel_pool: Vec<u32>,
    pub allocated_bricks: u32,
}

/// Generate a brick-map containing a voxelized sphere (CPU only, for tests).
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

// ── Default material palette ─────────────────────────────────────────────────

fn default_palette() -> Vec<GpuMaterial> {
    let mut palette = vec![
        GpuMaterial {
            color: [0.0, 0.0, 0.0],
            roughness: 1.0,
        };
        256
    ];
    // 1 = rock (deep)
    palette[1] = GpuMaterial {
        color: [0.6, 0.55, 0.5],
        roughness: 0.9,
    };
    // 2 = grass (surface)
    palette[2] = GpuMaterial {
        color: [0.3, 0.6, 0.2],
        roughness: 0.85,
    };
    // 3 = dirt (shallow)
    palette[3] = GpuMaterial {
        color: [0.5, 0.35, 0.2],
        roughness: 0.95,
    };
    palette
}

// ── Chunk slot tracking ──────────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct ChunkSlot {
    pos: [i32; 3],
    loaded: bool,
    last_used_frame: u64,
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

// ── TerraForgePass ───────────────────────────────────────────────────────────

pub struct TerraForgePass {
    // Per-frame uniforms
    uniform_buf: wgpu::Buffer,

    // Own unjittered camera buffer (avoids renderer TAA jitter)
    camera_buf: wgpu::Buffer,

    // Chunk data buffers
    chunk_table_buf: wgpu::Buffer,
    indir_grid_buf: wgpu::Buffer,
    brick_pool_buf: wgpu::Buffer,
    voxel_pool_buf: wgpu::Buffer,
    palette_buf: wgpu::Buffer,

    // Edit buffer
    edit_buf: wgpu::Buffer,

    // Half-res render targets
    #[allow(dead_code)]
    mat_tex: wgpu::Texture,
    mat_view: wgpu::TextureView,
    #[allow(dead_code)]
    norm_tex: wgpu::Texture,
    norm_view: wgpu::TextureView,

    // Ray march pipeline
    ray_march_pipeline: wgpu::ComputePipeline,
    ray_march_bgl: wgpu::BindGroupLayout,
    ray_march_bind_group: wgpu::BindGroup,

    // Shade pipeline
    shade_pipeline: wgpu::RenderPipeline,
    shade_bgl: wgpu::BindGroupLayout,
    shade_bind_group: wgpu::BindGroup,

    // Gen pipeline
    gen_pipeline: wgpu::ComputePipeline,
    #[allow(dead_code)]
    gen_bgl: wgpu::BindGroupLayout,
    gen_bg: wgpu::BindGroup,
    gen_uniform_buf: wgpu::Buffer,
    alloc_counter_buf: wgpu::Buffer,

    // Streaming state (CPU mirrors)
    chunk_slots: Vec<ChunkSlot>,
    chunk_table_cpu: Vec<ChunkInfo>,
    indir_grid_cpu: Vec<u32>,
    initialized: bool,

    // Edit state
    edits: Vec<EditOp>,
    edits_dirty: bool,

    // Config
    voxel_size: f32,
    planet_radius: f32,
    effective_max_mixed: u32,
    chunk_world_size: f32,
    indir_origin: [i32; 3],
    ray_w: u32,
    ray_h: u32,
}

impl TerraForgePass {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        Self::with_radius(
            device,
            queue,
            width,
            height,
            surface_format,
            DEFAULT_PLANET_RADIUS,
        )
    }

    pub fn with_radius(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        surface_format: wgpu::TextureFormat,
        planet_radius: f32,
    ) -> Self {
        let voxel_size = VOXEL_SIZE;
        let chunk_world_size = CHUNK_DIM_BRICKS as f32 * BRICK_DIM as f32 * voxel_size;

        log::info!(
            "Terra Forge: radius={:.0}m, chunk={:.1}m, voxel={:.2}m",
            planet_radius,
            chunk_world_size,
            voxel_size,
        );

        // ── Buffer pool allocation ───────────────────────────────────────

        let max_buf = device.limits().max_storage_buffer_binding_size as u64;
        let desired_voxel_bytes = MAX_LOADED_CHUNKS as u64
            * MAX_MIXED_BRICKS_PER_CHUNK as u64
            * WORDS_PER_BRICK as u64
            * 4;
        let voxel_pool_bytes = desired_voxel_bytes.min(max_buf);
        let effective_max_mixed =
            (voxel_pool_bytes / (WORDS_PER_BRICK as u64 * 4 * MAX_LOADED_CHUNKS as u64)) as u32;

        log::info!(
            "  Voxel pool: {:.1} MB (max_mixed/chunk={}), Brick pool: {:.1} MB",
            voxel_pool_bytes as f64 / (1024.0 * 1024.0),
            effective_max_mixed,
            (MAX_LOADED_CHUNKS as u64 * BRICKS_PER_CHUNK as u64 * 8) as f64 / (1024.0 * 1024.0),
        );

        let brick_pool_bytes = MAX_LOADED_CHUNKS as u64
            * BRICKS_PER_CHUNK as u64
            * std::mem::size_of::<BrickMeta>() as u64;
        let brick_pool_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TerraForge BrickPool"),
            size: brick_pool_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let voxel_pool_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TerraForge VoxelPool"),
            size: voxel_pool_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let chunk_table_bytes = MAX_LOADED_CHUNKS as u64 * std::mem::size_of::<ChunkInfo>() as u64;
        let chunk_table_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TerraForge ChunkTable"),
            size: chunk_table_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let indir_grid_count = INDIR_GRID_DIM * INDIR_GRID_DIM * INDIR_GRID_DIM;
        let indir_grid_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TerraForge IndirGrid"),
            size: indir_grid_count as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Edit buffer.
        let edit_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TerraForge EditBuffer"),
            size: (MAX_EDITS as u64) * std::mem::size_of::<EditOp>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Initialize brick pool to BRICK_EMPTY.
        {
            let empty_brick = BrickMeta {
                data_offset: BRICK_EMPTY,
                occupancy: 0,
            };
            let total_bricks = (MAX_LOADED_CHUNKS * BRICKS_PER_CHUNK) as usize;
            let init_data: Vec<BrickMeta> = vec![empty_brick; total_bricks];
            queue.write_buffer(&brick_pool_buf, 0, bytemuck::cast_slice(&init_data));
        }

        // Initialize indirection grid to INDIR_EMPTY.
        let indir_grid_cpu = vec![INDIR_EMPTY; indir_grid_count as usize];
        queue.write_buffer(&indir_grid_buf, 0, bytemuck::cast_slice(&indir_grid_cpu));

        // ── Gen pipeline (5 bindings: uniforms + brick + voxel + alloc + edits) ──

        let gen_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TerraForge GenUniforms"),
            size: std::mem::size_of::<GenUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let alloc_counter_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TerraForge AllocCounter"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let gen_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("TerraForge Gen Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/terra_gen.wgsl").into()),
        });

        let gen_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("TerraForge Gen BGL"),
            entries: &[
                bgl_uniform(0),
                bgl_storage_rw(1),
                bgl_storage_rw(2),
                bgl_storage_rw(3),
                bgl_storage(4), // edit buffer (read-only)
            ],
        });

        let gen_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("TerraForge Gen"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("TerraForge Gen PL"),
                    bind_group_layouts: &[Some(&gen_bgl)],
                    immediate_size: 0,
                }),
            ),
            module: &gen_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let gen_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TerraForge Gen BG"),
            layout: &gen_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gen_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: brick_pool_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: voxel_pool_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: alloc_counter_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: edit_buf.as_entire_binding(),
                },
            ],
        });

        // ── Uniform buffer ───────────────────────────────────────────────

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TerraForge Uniforms"),
            size: std::mem::size_of::<GpuUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Palette.
        let palette = default_palette();
        let palette_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TerraForge Palette"),
            size: (palette.len() * std::mem::size_of::<GpuMaterial>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&palette_buf, 0, bytemuck::cast_slice(&palette));

        // Full-res textures (no half-res to avoid nearest-neighbor upsample flicker).
        let ray_w = width;
        let ray_h = height;
        let (mat_tex, mat_view) = Self::create_tex(
            device,
            ray_w,
            ray_h,
            wgpu::TextureFormat::R32Uint,
            "Material",
        );
        let (norm_tex, norm_view) = Self::create_tex(
            device,
            ray_w,
            ray_h,
            wgpu::TextureFormat::Rgba16Float,
            "Normal",
        );

        // ── Ray march compute pipeline (9 bindings: +edit_buf) ──────────

        let ray_march_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("TerraForge RayMarch Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/terra_ray_march.wgsl").into(),
            ),
        });

        let ray_march_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("TerraForge RayMarch BGL"),
            entries: &[
                bgl_uniform(0),                                       // uniforms
                bgl_uniform(1),                                       // camera
                bgl_storage(2),                                       // chunk_table
                bgl_storage(3),                                       // indir_grid
                bgl_storage(4),                                       // brick_pool
                bgl_storage(5),                                       // voxel_pool
                bgl_storage_tex(6, wgpu::TextureFormat::R32Uint),     // out_material
                bgl_storage_tex(7, wgpu::TextureFormat::Rgba16Float), // out_normal
                bgl_storage(8),                                       // edit_buf
            ],
        });

        let ray_march_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("TerraForge RayMarch"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("TerraForge RayMarch PL"),
                    bind_group_layouts: &[Some(&ray_march_bgl)],
                    immediate_size: 0,
                }),
            ),
            module: &ray_march_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ── Shade render pipeline ────────────────────────────────────────

        let shade_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("TerraForge Shade Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/terra_shade.wgsl").into()),
        });

        let shade_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("TerraForge Shade BGL"),
            entries: &[
                bgl_uniform_frag(0),
                bgl_tex_uint(1),
                bgl_tex_float(2),
                bgl_storage_frag(3),
            ],
        });

        let shade_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("TerraForge Shade"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("TerraForge Shade PL"),
                    bind_group_layouts: &[Some(&shade_bgl)],
                    immediate_size: 0,
                }),
            ),
            vertex: wgpu::VertexState {
                module: &shade_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shade_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        // Own unjittered camera buffer (avoids renderer TAA jitter for stable rays).
        let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TerraForge Camera"),
            size: std::mem::size_of::<helio_v3::GpuCameraUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let ray_march_bind_group = Self::mk_ray_march_bg(
            device,
            &ray_march_bgl,
            &uniform_buf,
            &camera_buf,
            &chunk_table_buf,
            &indir_grid_buf,
            &brick_pool_buf,
            &voxel_pool_buf,
            &mat_view,
            &norm_view,
            &edit_buf,
        );
        let shade_bind_group = Self::mk_shade_bg(
            device,
            &shade_bgl,
            &camera_buf,
            &mat_view,
            &norm_view,
            &palette_buf,
        );

        let chunk_slots = vec![ChunkSlot::default(); MAX_LOADED_CHUNKS as usize];
        let chunk_table_cpu = vec![ChunkInfo::zeroed(); MAX_LOADED_CHUNKS as usize];

        Self {
            uniform_buf,
            camera_buf,
            chunk_table_buf,
            indir_grid_buf,
            brick_pool_buf,
            voxel_pool_buf,
            palette_buf,
            edit_buf,
            mat_tex,
            mat_view,
            norm_tex,
            norm_view,
            ray_march_pipeline,
            ray_march_bgl,
            ray_march_bind_group,
            shade_pipeline,
            shade_bgl,
            shade_bind_group,
            gen_pipeline,
            gen_bgl,
            gen_bg,
            gen_uniform_buf,
            alloc_counter_buf,
            chunk_slots,
            chunk_table_cpu,
            indir_grid_cpu,
            initialized: false,
            edits: Vec::new(),
            edits_dirty: false,
            voxel_size,
            planet_radius,
            effective_max_mixed,
            chunk_world_size,
            indir_origin: [0; 3],
            ray_w,
            ray_h,
        }
    }

    // ── Public edit API ──────────────────────────────────────────────────

    /// Add an SDF edit operation. Affected chunks are regenerated next frame.
    pub fn add_edit(&mut self, edit: EditOp) {
        if self.edits.len() >= MAX_EDITS as usize {
            log::warn!("Terra Forge: edit buffer full ({} max)", MAX_EDITS);
            return;
        }
        // Mark affected chunks dirty (will be regenerated).
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
                // Force reload by marking unloaded — will be picked up by streaming.
                slot.loaded = false;
            }
        }
        self.edits.push(edit);
        self.edits_dirty = true;
        log::info!("Terra Forge: edit added ({} total)", self.edits.len());
    }

    /// Current planet radius.
    pub fn planet_radius(&self) -> f32 {
        self.planet_radius
    }

    // ── Chunk streaming ──────────────────────────────────────────────────

    /// Find surface chunks near camera that fit within the indirection grid.
    fn find_surface_chunks_near(
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

                    // Nearest point on AABB to origin (planet center).
                    let nx = 0.0f32.clamp(cmin[0], cmax[0]);
                    let ny = 0.0f32.clamp(cmin[1], cmax[1]);
                    let nz = 0.0f32.clamp(cmin[2], cmax[2]);
                    let near_dist2 = nx * nx + ny * ny + nz * nz;

                    // Farthest point on AABB from origin.
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

                    let inner_r = planet_radius * 0.90; // below surface - noise
                    let outer_r = planet_radius * 1.10; // above surface + noise

                    // Chunk intersects the surface shell if:
                    // farthest point >= inner_r  AND  nearest point <= outer_r
                    if far_dist2 >= inner_r * inner_r && near_dist2 <= outer_r * outer_r {
                        // Sort by distance from camera.
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

    /// Find a free chunk slot (not loaded).
    fn find_free_slot(&self) -> Option<usize> {
        self.chunk_slots.iter().position(|s| !s.loaded)
    }

    /// Evict the LRU loaded chunk (furthest from camera or least recently used).
    fn evict_lru_chunk(&mut self) -> Option<usize> {
        self.chunk_slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.loaded)
            .min_by_key(|(_, s)| s.last_used_frame)
            .map(|(i, _)| i)
    }

    /// Clear a chunk slot's GPU data (brick_pool + voxel_pool ranges).
    fn clear_chunk_gpu_data(&self, slot_idx: usize, queue: &wgpu::Queue) {
        let bp_off = slot_idx as u64 * BRICKS_PER_CHUNK as u64;
        let empty_bricks = vec![
            BrickMeta {
                data_offset: BRICK_EMPTY,
                occupancy: 0
            };
            BRICKS_PER_CHUNK as usize
        ];
        queue.write_buffer(
            &self.brick_pool_buf,
            bp_off * std::mem::size_of::<BrickMeta>() as u64,
            bytemuck::cast_slice(&empty_bricks),
        );

        // Clear voxel pool range to zero (critical: atomicOr leaves stale data).
        let vp_off = slot_idx as u64 * self.effective_max_mixed as u64;
        let vp_words = self.effective_max_mixed as u64 * WORDS_PER_BRICK as u64;
        let vp_bytes = vp_words * 4;
        // Write in chunks to avoid huge temp allocations.
        const CLEAR_CHUNK: u64 = 1024 * 1024; // 1MB at a time
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

    /// Rebuild the entire indirection grid from currently loaded chunks.
    fn rebuild_indir_grid(&mut self, queue: &wgpu::Queue) {
        let dim = INDIR_GRID_DIM as i32;
        self.indir_grid_cpu.fill(INDIR_EMPTY);

        for (slot_idx, slot) in self.chunk_slots.iter().enumerate() {
            if !slot.loaded {
                continue;
            }
            // Only include chunks within the current grid window.
            // Out-of-window chunks must be skipped — their wrap indices would
            // alias to wrong positions inside the window.
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

    /// Upload edit buffer to GPU.
    fn upload_edits(&mut self, queue: &wgpu::Queue) {
        if !self.edits.is_empty() {
            queue.write_buffer(&self.edit_buf, 0, bytemuck::cast_slice(&self.edits));
        }
        self.edits_dirty = false;
    }

    /// Generate a batch of chunks — one submit per chunk for correctness.
    /// (queue.write_buffer to the same offset overwrites, so each dispatch
    /// needs its own submit to see correct uniforms.)
    fn generate_chunks(
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

            // Clear stale data before reuse.
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

            // One encoder + submit per chunk so each dispatch sees correct uniforms.
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

            // Update CPU tracking.
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

        // Upload chunk table.
        queue.write_buffer(
            &self.chunk_table_buf,
            0,
            bytemuck::cast_slice(&self.chunk_table_cpu),
        );
    }

    // ── Helper methods ───────────────────────────────────────────────────

    fn create_tex(
        device: &wgpu::Device,
        w: u32,
        h: u32,
        format: wgpu::TextureFormat,
        label: &str,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&Default::default());
        (tex, view)
    }

    fn mk_ray_march_bg(
        device: &wgpu::Device,
        bgl: &wgpu::BindGroupLayout,
        uniform_buf: &wgpu::Buffer,
        camera_buf: &wgpu::Buffer,
        chunk_table_buf: &wgpu::Buffer,
        indir_grid_buf: &wgpu::Buffer,
        brick_pool_buf: &wgpu::Buffer,
        voxel_pool_buf: &wgpu::Buffer,
        mat_view: &wgpu::TextureView,
        norm_view: &wgpu::TextureView,
        edit_buf: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TerraForge RayMarch BG"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: chunk_table_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: indir_grid_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: brick_pool_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: voxel_pool_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(mat_view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(norm_view),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: edit_buf.as_entire_binding(),
                },
            ],
        })
    }

    fn mk_shade_bg(
        device: &wgpu::Device,
        bgl: &wgpu::BindGroupLayout,
        camera_buf: &wgpu::Buffer,
        mat_view: &wgpu::TextureView,
        norm_view: &wgpu::TextureView,
        palette_buf: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TerraForge Shade BG"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(mat_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(norm_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: palette_buf.as_entire_binding(),
                },
            ],
        })
    }

    /// Keep the old static method for unit tests.
    #[cfg(test)]
    fn find_planet_chunks(planet_radius: f32, chunk_world_size: f32) -> Vec<[i32; 3]> {
        Self::find_surface_chunks_near(
            [0.0, 0.0, planet_radius * 2.0],
            planet_radius,
            chunk_world_size,
        )
    }
}

// ── Utility ──────────────────────────────────────────────────────────────────

fn sphere_aabb_test(center: [f32; 3], radius: f32, bmin: [f32; 3], bmax: [f32; 3]) -> bool {
    let nx = center[0].clamp(bmin[0], bmax[0]);
    let ny = center[1].clamp(bmin[1], bmax[1]);
    let nz = center[2].clamp(bmin[2], bmax[2]);
    let dx = nx - center[0];
    let dy = ny - center[1];
    let dz = nz - center[2];
    dx * dx + dy * dy + dz * dz <= radius * radius
}

// ── BGL helpers ──────────────────────────────────────────────────────────────

fn bgl_uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_storage(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_storage_rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_storage_tex(binding: u32, format: wgpu::TextureFormat) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

fn bgl_uniform_frag(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_tex_uint(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Uint,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn bgl_tex_float(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn bgl_storage_frag(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

// ── RenderPass impl ──────────────────────────────────────────────────────────

impl RenderPass for TerraForgePass {
    fn name(&self) -> &'static str {
        "TerraForge"
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> Result<()> {
        let cam_pos = ctx.scene.camera.position();
        let frame = ctx.scene.frame_count;

        // ── Dynamic chunk streaming ──────────────────────────────────────
        // find_surface_chunks_near returns ALL surface chunks in the 16³ grid,
        // sorted by camera distance. We only want the closest MAX_LOADED_CHUNKS.
        let all_needed =
            Self::find_surface_chunks_near(cam_pos, self.planet_radius, self.chunk_world_size);
        let needed: Vec<[i32; 3]> = all_needed
            .into_iter()
            .take(MAX_LOADED_CHUNKS as usize)
            .collect();

        // Mark currently-needed loaded chunks as "used this frame".
        let needed_set: std::collections::HashSet<[i32; 3]> = needed.iter().copied().collect();
        for slot in &mut self.chunk_slots {
            if slot.loaded && needed_set.contains(&slot.pos) {
                slot.last_used_frame = frame;
            }
        }

        // Find which needed chunks are not yet loaded.
        let loaded_set: std::collections::HashSet<[i32; 3]> = self
            .chunk_slots
            .iter()
            .filter(|s| s.loaded)
            .map(|s| s.pos)
            .collect();

        // Count how many free or evictable (not-needed) slots we have.
        let free_slots = self.chunk_slots.iter().filter(|s| !s.loaded).count();
        let evictable = self
            .chunk_slots
            .iter()
            .filter(|s| s.loaded && !needed_set.contains(&s.pos))
            .count();
        let available = free_slots + evictable;

        let budget = if self.initialized {
            CHUNKS_PER_FRAME.min(available)
        } else {
            available
        };

        let to_load: Vec<[i32; 3]> = needed
            .iter()
            .filter(|p| !loaded_set.contains(*p))
            .copied()
            .take(budget)
            .collect();

        if !to_load.is_empty() {
            if !self.initialized {
                log::info!(
                    "Terra Forge: initial load {} chunks (needed={})",
                    to_load.len(),
                    needed.len(),
                );
            }
            // Upload edits before generation so gen shader can use them.
            if self.edits_dirty {
                self.upload_edits(ctx.queue);
            }
            self.generate_chunks(&to_load, ctx.queue, ctx.device, frame);
        }

        // Update indirection grid origin to track camera.
        let half_grid = (INDIR_GRID_DIM / 2) as i32;
        let cam_chunk = [
            (cam_pos[0] / self.chunk_world_size).floor() as i32,
            (cam_pos[1] / self.chunk_world_size).floor() as i32,
            (cam_pos[2] / self.chunk_world_size).floor() as i32,
        ];
        self.indir_origin = [
            cam_chunk[0] - half_grid,
            cam_chunk[1] - half_grid,
            cam_chunk[2] - half_grid,
        ];

        // Rebuild full indirection grid every frame (critic: prevents stale entries).
        self.rebuild_indir_grid(ctx.queue);

        // Always upload chunk table for GPU consistency (even if no chunks generated).
        ctx.queue.write_buffer(
            &self.chunk_table_buf,
            0,
            bytemuck::cast_slice(&self.chunk_table_cpu),
        );

        self.initialized = true;

        // ── Upload uniforms ──────────────────────────────────────────────
        // Compute far-field cell size once per frame based on camera-planet distance.
        // This ensures all far-field calls in the shader use the same cell grid,
        // eliminating seams between adjacent empty chunks.
        let cam_dist_to_planet =
            (cam_pos[0] * cam_pos[0] + cam_pos[1] * cam_pos[1] + cam_pos[2] * cam_pos[2]).sqrt();
        let surface_dist = (cam_dist_to_planet - self.planet_radius).abs().max(1.0);
        // Snap to discrete power-of-2 levels to prevent the far-field virtual grid
        // from continuously shifting as the camera moves.
        let raw_cell = (0.001 * surface_dist).max(0.8);
        let ff_cell_size = 2.0f32.powi(raw_cell.log2().ceil() as i32);

        let uniforms = GpuUniforms {
            width: self.ray_w,
            height: self.ray_h,
            brick_dim: BRICK_DIM,
            chunk_dim_bricks: CHUNK_DIM_BRICKS,
            voxel_size: self.voxel_size,
            planet_radius: self.planet_radius,
            indir_grid_dim: INDIR_GRID_DIM,
            edit_count: self.edits.len() as u32,
            indir_origin: self.indir_origin,
            ff_cell_size,
            camera_offset: cam_pos,
            _pad_cam: 0.0,
        };
        ctx.queue
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        // ── Upload unjittered camera ─────────────────────────────────────
        // The renderer always applies TAA Halton jitter to the projection
        // by modifying column 2 of the projection matrix (elements [8],[9]
        // in column-major layout). We zero those elements directly instead of
        // using an inverse matrix multiply — this is exact and avoids the
        // numerical amplification that the extreme near/far ratio would cause
        // through matrix inverse (condition number ~10^7 for near=0.01, far=100000).
        {
            let cam_data = ctx.scene.camera.data();
            let mut proj_cols = cam_data.proj;
            // col2 row0 and col2 row1 are the only elements affected by jitter.
            // For a standard symmetric perspective projection, these are 0.
            proj_cols[8] = 0.0;
            proj_cols[9] = 0.0;
            let unjittered_proj = glam::Mat4::from_cols_array(&proj_cols);
            let view = glam::Mat4::from_cols_array(&cam_data.view);
            let unjittered_vp = unjittered_proj * view;
            let unjittered_inv_vp = unjittered_vp.inverse();

            let mut clean = *cam_data;
            clean.proj = unjittered_proj.to_cols_array();
            clean.view_proj = unjittered_vp.to_cols_array();
            clean.inv_view_proj = unjittered_inv_vp.to_cols_array();
            clean.jitter_frame = [0.0, 0.0, cam_data.jitter_frame[2], 0.0];
            ctx.queue
                .write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&clean));
        }

        // ── Rebuild bind groups ──────────────────────────────────────────
        self.ray_march_bind_group = Self::mk_ray_march_bg(
            ctx.device,
            &self.ray_march_bgl,
            &self.uniform_buf,
            &self.camera_buf,
            &self.chunk_table_buf,
            &self.indir_grid_buf,
            &self.brick_pool_buf,
            &self.voxel_pool_buf,
            &self.mat_view,
            &self.norm_view,
            &self.edit_buf,
        );
        self.shade_bind_group = Self::mk_shade_bg(
            ctx.device,
            &self.shade_bgl,
            &self.camera_buf,
            &self.mat_view,
            &self.norm_view,
            &self.palette_buf,
        );
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        // Compute ray march.
        {
            let mut cpass = ctx
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("TerraForge RayMarch"),
                    timestamp_writes: None,
                });
            cpass.set_pipeline(&self.ray_march_pipeline);
            cpass.set_bind_group(0, &self.ray_march_bind_group, &[]);
            cpass.dispatch_workgroups((self.ray_w + 7) / 8, (self.ray_h + 7) / 8, 1);
        }

        // Fullscreen shade → swapchain target.
        {
            let mut rpass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("TerraForge Shade"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: ctx.target,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rpass.set_pipeline(&self.shade_pipeline);
            rpass.set_bind_group(0, &self.shade_bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }
        Ok(())
    }

    fn on_resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if width == self.ray_w && height == self.ray_h {
            return;
        }
        self.ray_w = width;
        self.ray_h = height;
        let (mt, mv) = Self::create_tex(
            device,
            width,
            height,
            wgpu::TextureFormat::R32Uint,
            "Material",
        );
        let (nt, nv) = Self::create_tex(
            device,
            width,
            height,
            wgpu::TextureFormat::Rgba16Float,
            "Normal",
        );
        self.mat_tex = mt;
        self.mat_view = mv;
        self.norm_tex = nt;
        self.norm_view = nv;
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

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
        assert_eq!(std::mem::size_of::<GpuUniforms>(), 64);
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
