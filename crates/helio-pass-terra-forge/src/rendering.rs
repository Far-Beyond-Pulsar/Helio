use crate::biome::default_palette;
use crate::gpu_types::*;
use crate::terrain::ChunkSlot;
use crate::{TerraForgePass, CHUNKS_PER_FRAME, DEFAULT_PLANET_RADIUS, VOXEL_SIZE};
use bytemuck::{bytes_of, cast_slice, Zeroable};
use helio_v3::graph::{ResourceBuilder, ResourceFormat, ResourceSize};
use helio_v3::{traits::RenderPass, PassContext, PrepareContext, Result};

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

// ── Constructors ─────────────────────────────────────────────────────────────

impl TerraForgePass {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        Self::with_radius(device, queue, width, height, surface_format, DEFAULT_PLANET_RADIUS)
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

        let edit_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TerraForge EditBuffer"),
            size: (MAX_EDITS as u64) * std::mem::size_of::<EditOp>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        {
            let empty_brick = BrickMeta {
                data_offset: BRICK_EMPTY,
                occupancy: 0,
            };
            let total_bricks = (MAX_LOADED_CHUNKS * BRICKS_PER_CHUNK) as usize;
            let init_data: Vec<BrickMeta> = vec![empty_brick; total_bricks];
            queue.write_buffer(&brick_pool_buf, 0, cast_slice(&init_data));
        }

        let indir_grid_cpu = vec![INDIR_EMPTY; indir_grid_count as usize];
        queue.write_buffer(&indir_grid_buf, 0, cast_slice(&indir_grid_cpu));

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
                bgl_storage(4),
            ],
        });

        let gen_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("TerraForge Gen"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("TerraForge Gen PL"),
                bind_group_layouts: &[Some(&gen_bgl)],
                immediate_size: 0,
            })),
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

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TerraForge Uniforms"),
            size: std::mem::size_of::<GpuUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let palette = default_palette();
        let palette_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TerraForge Palette"),
            size: (palette.len() * std::mem::size_of::<GpuMaterial>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&palette_buf, 0, cast_slice(&palette));

        let ray_w = width;
        let ray_h = height;
        let ray_w_half = (ray_w + 1) / 2;
        let ray_h_half = (ray_h + 1) / 2;

        let (mat_tex, mat_view) = Self::create_tex(device, ray_w, ray_h, wgpu::TextureFormat::R32Uint, "Material");
        let (norm_tex, norm_view) = Self::create_tex(device, ray_w, ray_h, wgpu::TextureFormat::Rgba16Float, "Normal");
        let (mat_tex_half, mat_view_half) = Self::create_tex(device, ray_w_half, ray_h_half, wgpu::TextureFormat::R32Uint, "Material Half");
        let (norm_tex_half, norm_view_half) = Self::create_tex(device, ray_w_half, ray_h_half, wgpu::TextureFormat::Rgba16Float, "Normal Half");

        let ray_march_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("TerraForge RayMarch Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/terra_ray_march.wgsl").into()),
        });

        let ray_march_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("TerraForge RayMarch BGL"),
            entries: &[
                bgl_uniform(0),
                bgl_uniform(1),
                bgl_storage(2),
                bgl_storage(3),
                bgl_storage(4),
                bgl_storage(5),
                bgl_storage_tex(6, wgpu::TextureFormat::R32Uint),
                bgl_storage_tex(7, wgpu::TextureFormat::Rgba16Float),
                bgl_storage(8),
            ],
        });

        let ray_march_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("TerraForge RayMarch"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("TerraForge RayMarch PL"),
                bind_group_layouts: &[Some(&ray_march_bgl)],
                immediate_size: 0,
            })),
            module: &ray_march_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

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
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("TerraForge Shade PL"),
                bind_group_layouts: &[Some(&shade_bgl)],
                immediate_size: 0,
            })),
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

        let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TerraForge Camera"),
            size: std::mem::size_of::<helio_v3::GpuCameraUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let ray_march_bind_group = Self::mk_ray_march_bg(
            device, &ray_march_bgl, &uniform_buf, &camera_buf,
            &chunk_table_buf, &indir_grid_buf, &brick_pool_buf, &voxel_pool_buf,
            &mat_view_half, &norm_view_half, &edit_buf,
        );
        let shade_bind_group = Self::mk_shade_bg(
            device, &shade_bgl, &camera_buf, &mat_view_half, &norm_view_half, &palette_buf,
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
            ray_w_half,
            ray_h_half,
            mat_tex_half,
            mat_view_half,
            norm_tex_half,
            norm_view_half,
            surface_format,
        }
    }

    fn create_tex(
        device: &wgpu::Device,
        w: u32,
        h: u32,
        format: wgpu::TextureFormat,
        label: &str,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
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
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: camera_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: chunk_table_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: indir_grid_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: brick_pool_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: voxel_pool_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(mat_view) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(norm_view) },
                wgpu::BindGroupEntry { binding: 8, resource: edit_buf.as_entire_binding() },
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
                wgpu::BindGroupEntry { binding: 0, resource: camera_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(mat_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(norm_view) },
                wgpu::BindGroupEntry { binding: 3, resource: palette_buf.as_entire_binding() },
            ],
        })
    }
}

// ── RenderPass impl ──────────────────────────────────────────────────────────

impl RenderPass for TerraForgePass {
    fn name(&self) -> &'static str {
        "TerraForge"
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.write_color(
            "pre_aa",
            ResourceFormat::from(self.surface_format),
            ResourceSize::MatchSurface,
        );
    }

    fn publish<'a>(&'a self, _frame: &mut libhelio::FrameResources<'a>) {}

    fn writes(&self) -> &'static [&'static str] {
        &["pre_aa"]
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> Result<()> {
        let cam_pos = ctx.scene.camera.position();
        let frame = ctx.scene.frame_count;

        let all_needed = TerraForgePass::find_surface_chunks_near(cam_pos, self.planet_radius, self.chunk_world_size);
        let needed: Vec<[i32; 3]> = all_needed
            .into_iter()
            .take(MAX_LOADED_CHUNKS as usize)
            .collect();

        let needed_set: std::collections::HashSet<[i32; 3]> = needed.iter().copied().collect();
        for slot in &mut self.chunk_slots {
            if slot.loaded && needed_set.contains(&slot.pos) {
                slot.last_used_frame = frame;
            }
        }

        let loaded_set: std::collections::HashSet<[i32; 3]> = self
            .chunk_slots
            .iter()
            .filter(|s| s.loaded)
            .map(|s| s.pos)
            .collect();

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
            if self.edits_dirty {
                self.upload_edits(ctx.queue);
            }
            self.generate_chunks(&to_load, ctx.queue, ctx.device, frame);
        }

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

        self.rebuild_indir_grid(ctx.queue);

        ctx.queue.write_buffer(
            &self.chunk_table_buf,
            0,
            cast_slice(&self.chunk_table_cpu),
        );

        self.initialized = true;

        let cam_dist_to_planet =
            (cam_pos[0] * cam_pos[0] + cam_pos[1] * cam_pos[1] + cam_pos[2] * cam_pos[2]).sqrt();
        let surface_dist = (cam_dist_to_planet - self.planet_radius).abs().max(1.0);
        let raw_cell = (0.0004 * surface_dist).max(0.8);
        let ff_cell_size = 2.0f32.powi(raw_cell.log2().ceil() as i32);

        let jitter = libhelio::temporal_jitter(ctx.frame_num);
        let uniforms = GpuUniforms {
            width: self.ray_w_half,
            height: self.ray_h_half,
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
            jitter,
            _jitter_pad: [0.0; 2],
        };
        ctx.queue.write_buffer(&self.uniform_buf, 0, bytes_of(&uniforms));

        {
            let cam_data = ctx.scene.camera.data();
            let mut proj_cols = cam_data.proj;
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
            ctx.queue.write_buffer(&self.camera_buf, 0, bytes_of(&clean));
        }

        self.ray_march_bind_group = Self::mk_ray_march_bg(
            ctx.device, &self.ray_march_bgl, &self.uniform_buf, &self.camera_buf,
            &self.chunk_table_buf, &self.indir_grid_buf, &self.brick_pool_buf, &self.voxel_pool_buf,
            &self.mat_view_half, &self.norm_view_half, &self.edit_buf,
        );
        self.shade_bind_group = Self::mk_shade_bg(
            ctx.device, &self.shade_bgl, &self.camera_buf,
            &self.mat_view_half, &self.norm_view_half, &self.palette_buf,
        );
        Ok(())
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        let pre_aa_view = resources.pre_aa.read("TerraForge")?;
        let color_attachments: &'a [Option<wgpu::RenderPassColorAttachment<'a>>] = Box::leak(Box::new([
            Some(wgpu::RenderPassColorAttachment {
                view: pre_aa_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            }),
        ]));
        Some(wgpu::RenderPassDescriptor {
            label: Some("TerraForge Shade"),
            color_attachments,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        {
            let mut cpass = unsafe { &mut *ctx.compute_encoder_ptr }
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("TerraForge RayMarch"),
                    timestamp_writes: None,
                });
            cpass.set_pipeline(&self.ray_march_pipeline);
            cpass.set_bind_group(0, &self.ray_march_bind_group, &[]);
            cpass.dispatch_workgroups((self.ray_w_half + 7) / 8, (self.ray_h_half + 7) / 8, 1);
        }

        {
            let rp = unsafe { &mut *ctx.active_render_pass_ptr().unwrap() };
            rp.set_pipeline(&self.shade_pipeline);
            rp.set_bind_group(0, &self.shade_bind_group, &[]);
            rp.draw(0..3, 0..1);
        }
        Ok(())
    }

    fn on_resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if width == self.ray_w && height == self.ray_h {
            return;
        }
        self.ray_w = width;
        self.ray_h = height;
        self.ray_w_half = (width + 1) / 2;
        self.ray_h_half = (height + 1) / 2;
        let (mt, mv) = Self::create_tex(device, width, height, wgpu::TextureFormat::R32Uint, "Material");
        let (nt, nv) = Self::create_tex(device, width, height, wgpu::TextureFormat::Rgba16Float, "Normal");
        let (mth, mvh) = Self::create_tex(device, self.ray_w_half, self.ray_h_half, wgpu::TextureFormat::R32Uint, "Material Half");
        let (nth, nvh) = Self::create_tex(device, self.ray_w_half, self.ray_h_half, wgpu::TextureFormat::Rgba16Float, "Normal Half");
        self.mat_tex = mt;
        self.mat_view = mv;
        self.norm_tex = nt;
        self.norm_view = nv;
        self.mat_tex_half = mth;
        self.mat_view_half = mvh;
        self.norm_tex_half = nth;
        self.norm_view_half = nvh;
    }
}
