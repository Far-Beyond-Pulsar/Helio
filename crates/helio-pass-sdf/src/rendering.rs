use crate::edit_list::{GpuSdfEdit, SdfEdit, SdfEditList};
use crate::gpu_bvh::{build_flat_bvh, GpuBvhNode};
use crate::primitives::SdfShapeType;
use crate::terrain::{GpuTerrainParams, TerrainConfig};
use crate::uniforms::SdfGridParams;
use crate::{SdfPass, INITIAL_BVH_CAPACITY, INITIAL_EDIT_CAPACITY, MAX_BRICKS_PER_LEVEL};
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

// ═══════════════════════════════════════════════════════════════════════════════
// GPU structs (WGSL mirrors)
// ═══════════════════════════════════════════════════════════════════════════════

/// Static clip configuration uploaded once (and whenever edits/terrain change).
/// Matches `ClipConfig` in `sdf_scroll.wgsl` and `sdf_classify.wgsl` (96 bytes).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GpuClipConfig {
    pub level_count: u32,
    pub grid_dim: u32,
    pub brick_size: u32,
    pub brick_grid_dim: u32,
    pub bricks_per_level: u32,
    pub atlas_bricks_per_axis: u32,
    pub base_voxel_size: f32,
    pub edit_count: u32,
    pub bvh_node_count: u32,
    pub terrain_enabled: u32,
    pub terrain_y_min: f32,
    pub terrain_y_max: f32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    pub _pad3: u32,
    pub voxel_sizes_lo: [f32; 4],
    pub voxel_sizes_hi: [f32; 4],
}

const _: () = assert!(
    std::mem::size_of::<GpuClipConfig>() == 96,
    "GpuClipConfig must be 96 bytes"
);

/// Persistent GPU scroll state (read-write by the scroll shader).
/// Matches `ScrollState` in both scroll and classify shaders (144 bytes).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GpuScrollState {
    pub snap_origins: [[i32; 4]; 8],
    pub edit_gen: u32,
    pub prev_edit_gen: u32,
    _pad0: u32,
    _pad1: u32,
}

const _: () = assert!(
    std::mem::size_of::<GpuScrollState>() == 144,
    "GpuScrollState must be 144 bytes"
);

// ═══════════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════════

const DEFAULT_GRID_DIM: u32 = 128;
const DEFAULT_CLIP_LEVELS: u32 = 8;
const DEFAULT_BRICK_SIZE: u32 = 8;
const EDIT_LIST_STRIDE: u32 = 65;

// ═══════════════════════════════════════════════════════════════════════════════
// Constructors
// ═══════════════════════════════════════════════════════════════════════════════

impl SdfPass {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        terrain: Option<TerrainConfig>,
    ) -> Self {
        Self::with_grid(
            device,
            surface_format,
            DEFAULT_GRID_DIM,
            [-50.0; 3],
            [50.0; 3],
            terrain,
        )
    }

    pub fn with_grid(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        grid_dim: u32,
        volume_min: [f32; 3],
        volume_max: [f32; 3],
        terrain: Option<TerrainConfig>,
    ) -> Self {
        let level_count = DEFAULT_CLIP_LEVELS;
        let brick_size = DEFAULT_BRICK_SIZE;
        let brick_grid_dim = grid_dim / brick_size;
        let bricks_per_level = brick_grid_dim * brick_grid_dim * brick_grid_dim;

        let range = volume_max[0] - volume_min[0];
        let base_voxel_size = range / grid_dim as f32;
        let padded_brick_voxels = (brick_size + 1) * (brick_size + 1) * (brick_size + 1);

        let atlas_word_count = (bricks_per_level * padded_brick_voxels + 3) / 4;
        let atlas_buffers: Vec<wgpu::Buffer> = (0..level_count as usize)
            .map(|i| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("SDF Atlas L{i}")),
                    size: (atlas_word_count * 4) as u64,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                })
            })
            .collect();

        let level_params_buffers: Vec<wgpu::Buffer> = (0..level_count as usize)
            .map(|i| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("SDF Level Params L{i}")),
                    size: std::mem::size_of::<SdfGridParams>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();

        let edit_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Edit Buffer"),
            size: (INITIAL_EDIT_CAPACITY * std::mem::size_of::<GpuSdfEdit>()).max(64) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let terrain_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Terrain Params"),
            size: std::mem::size_of::<GpuTerrainParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bvh_nodes_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF BVH Nodes"),
            size: (INITIAL_BVH_CAPACITY * std::mem::size_of::<GpuBvhNode>()).max(64) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let scroll_state_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Scroll State"),
            size: std::mem::size_of::<GpuScrollState>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let dirty_flags_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Dirty Flags"),
            size: (level_count * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let clip_config_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Clip Config"),
            size: std::mem::size_of::<GpuClipConfig>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let per_brick_hashes_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Per-Brick Hashes"),
            size: (level_count * bricks_per_level * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let per_brick_edit_lists_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Per-Brick Edit Lists"),
            size: (level_count * bricks_per_level * EDIT_LIST_STRIDE * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let all_brick_indices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF All Brick Indices"),
            size: (level_count * bricks_per_level * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let dirty_bricks_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Dirty Bricks"),
            size: (level_count * MAX_BRICKS_PER_LEVEL * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let indirect_byte_size = (level_count * 3 * 4) as u64;
        let eval_indirect_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Eval Indirect"),
            size: indirect_byte_size,
            usage: wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let eval_indirect_template_buffer = {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("SDF Eval Indirect Template"),
                size: indirect_byte_size,
                usage: wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: true,
            });
            let template_data: Vec<u32> = (0..level_count).flat_map(|_| [0u32, 1, 1]).collect();
            buf.slice(..)
                .get_mapped_range_mut()
                .expect("indirect template buffer should be mapped")
                .copy_from_slice(bytemuck::cast_slice(&template_data));
            buf.unmap();
            buf
        };

        let scroll_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SDF Scroll"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sdf_scroll.wgsl").into()),
        });
        let scroll_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SDF Scroll Pipeline"),
            layout: None,
            module: &scroll_shader,
            entry_point: Some("cs_scroll"),
            compilation_options: Default::default(),
            cache: None,
        });

        let classify_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SDF Classify"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sdf_classify.wgsl").into()),
        });
        let classify_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SDF Classify Pipeline"),
            layout: None,
            module: &classify_shader,
            entry_point: Some("cs_classify"),
            compilation_options: Default::default(),
            cache: None,
        });

        let eval_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SDF Evaluate"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sdf_evaluate.wgsl").into()),
        });
        let eval_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SDF Evaluate Pipeline"),
            layout: None,
            module: &eval_shader,
            entry_point: Some("cs_evaluate_sparse"),
            compilation_options: Default::default(),
            cache: None,
        });

        let classify_bgl = classify_pipeline.get_bind_group_layout(0);
        let classify_bg = Self::build_classify_bg_impl(
            device,
            &classify_bgl,
            &clip_config_buffer,
            &scroll_state_buffer,
            &dirty_flags_buffer,
            &bvh_nodes_buffer,
            &per_brick_hashes_buffer,
            &per_brick_edit_lists_buffer,
            &all_brick_indices_buffer,
            &dirty_bricks_buffer,
            &eval_indirect_buffer,
        );

        let eval_bgl = eval_pipeline.get_bind_group_layout(0);
        let eval_bgs = Self::build_eval_bgs_impl(
            device,
            &eval_bgl,
            &level_params_buffers,
            &edit_buffer,
            &atlas_buffers,
            &dirty_bricks_buffer,
            &per_brick_edit_lists_buffer,
            &terrain_params_buffer,
        );

        let march_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SDF Ray March"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sdf_ray_march.wgsl").into()),
        });

        let march_bgl = Self::build_march_bgl(device, level_count as usize);
        let march_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("SDF Ray March PL"),
                bind_group_layouts: &[Some(&march_bgl)],
                immediate_size: 0,
            });
        let march_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SDF Ray March Pipeline"),
            layout: Some(&march_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &march_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &march_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            scroll_pipeline,
            classify_pipeline,
            eval_pipeline,
            march_pipeline,
            march_bgl,
            edit_buffer,
            terrain_params_buffer,
            bvh_nodes_buffer,
            scroll_state_buffer,
            dirty_flags_buffer,
            clip_config_buffer,
            per_brick_hashes_buffer,
            per_brick_edit_lists_buffer,
            all_brick_indices_buffer,
            dirty_bricks_buffer,
            eval_indirect_buffer,
            eval_indirect_template_buffer,
            atlas_buffers,
            level_params_buffers,
            scroll_bg: None,
            scroll_bg_camera_key: 0,
            classify_bg,
            eval_bgs,
            march_bg: None,
            march_bg_camera_key: 0,
            edit_list: SdfEditList::new(),
            terrain_config: terrain,
            last_gen: u64::MAX,
            edit_generation: 0,
            bindings_dirty: false,
            debug_mode: false,
            enabled: true,
            preserve_framebuffer: false,
            level_count,
            bricks_per_level,
            brick_grid_dim,
            brick_size,
            grid_dim,
            base_voxel_size,
            padded_brick_voxels,
            volume_min,
            volume_max,
            surface_format,
            cached_snap_origins: [[i32::MIN; 3]; 8],
            gpu_passes_clean: false,
        }
    }

    // ── Bind group builders ───────────────────────────────────────────────

    fn build_scroll_bg(
        device: &wgpu::Device,
        scroll_bgl: &wgpu::BindGroupLayout,
        camera_buf: &wgpu::Buffer,
        clip_config_buf: &wgpu::Buffer,
        scroll_state_buf: &wgpu::Buffer,
        dirty_flags_buf: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SDF Scroll BG"),
            layout: scroll_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: clip_config_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: scroll_state_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dirty_flags_buf.as_entire_binding(),
                },
            ],
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn build_classify_bg_impl(
        device: &wgpu::Device,
        bgl: &wgpu::BindGroupLayout,
        clip_config_buf: &wgpu::Buffer,
        scroll_state_buf: &wgpu::Buffer,
        dirty_flags_buf: &wgpu::Buffer,
        bvh_nodes_buf: &wgpu::Buffer,
        per_brick_hashes_buf: &wgpu::Buffer,
        per_brick_edit_lists_buf: &wgpu::Buffer,
        all_brick_indices_buf: &wgpu::Buffer,
        dirty_bricks_buf: &wgpu::Buffer,
        eval_indirect_buf: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SDF Classify BG"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: clip_config_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: scroll_state_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dirty_flags_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: bvh_nodes_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: per_brick_hashes_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: per_brick_edit_lists_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: all_brick_indices_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: dirty_bricks_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: eval_indirect_buf.as_entire_binding(),
                },
            ],
        })
    }

    fn build_eval_bgs_impl(
        device: &wgpu::Device,
        bgl: &wgpu::BindGroupLayout,
        level_params_bufs: &[wgpu::Buffer],
        edit_buf: &wgpu::Buffer,
        atlas_bufs: &[wgpu::Buffer],
        dirty_bricks_buf: &wgpu::Buffer,
        per_brick_edit_lists_buf: &wgpu::Buffer,
        terrain_params_buf: &wgpu::Buffer,
    ) -> Vec<wgpu::BindGroup> {
        level_params_bufs
            .iter()
            .enumerate()
            .zip(atlas_bufs.iter())
            .map(|((i, params_buf), atlas_buf)| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("SDF Eval BG L{i}")),
                    layout: bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: params_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: edit_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: atlas_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: dirty_bricks_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: per_brick_edit_lists_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: terrain_params_buf.as_entire_binding(),
                        },
                    ],
                })
            })
            .collect()
    }

    fn build_march_bgl(device: &wgpu::Device, level_count: usize) -> wgpu::BindGroupLayout {
        let mut entries = vec![
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ];
        for i in 0..level_count {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: (3 + i) as u32,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: (3 + level_count) as u32,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SDF Ray March BGL"),
            entries: &entries,
        })
    }

    fn build_march_bg(
        device: &wgpu::Device,
        bgl: &wgpu::BindGroupLayout,
        camera_buf: &wgpu::Buffer,
        clip_config_buf: &wgpu::Buffer,
        scroll_state_buf: &wgpu::Buffer,
        atlas_bufs: &[wgpu::Buffer],
        all_brick_indices_buf: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let level_count = atlas_bufs.len();
        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: clip_config_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: scroll_state_buf.as_entire_binding(),
            },
        ];
        for (i, atlas) in atlas_bufs.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: (3 + i) as u32,
                resource: atlas.as_entire_binding(),
            });
        }
        entries.push(wgpu::BindGroupEntry {
            binding: (3 + level_count) as u32,
            resource: all_brick_indices_buf.as_entire_binding(),
        });
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SDF Ray March BG"),
            layout: bgl,
            entries: &entries,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// RenderPass impl
// ═══════════════════════════════════════════════════════════════════════════════

impl RenderPass for SdfPass {
    fn name(&self) -> &'static str {
        "SDF"
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        if !self.enabled {
            return Ok(());
        }

        let gen = self.edit_list.generation();
        let needs_upload = gen != self.last_gen;

        if needs_upload {
            let gpu_edits = self.edit_list.flush_gpu_data();
            let edit_count = gpu_edits.len() as u32;

            let required = (gpu_edits.len() * std::mem::size_of::<GpuSdfEdit>()).max(64) as u64;
            if required > self.edit_buffer.size() {
                let new_size = (required * 2).max(64);
                self.edit_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("SDF Edit Buffer"),
                    size: new_size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.bindings_dirty = true;
            }
            if !gpu_edits.is_empty() {
                ctx.queue
                    .write_buffer(&self.edit_buffer, 0, bytemuck::cast_slice(&gpu_edits));
            }

            let bounds: Vec<(glam::Vec3, f32)> = self
                .edit_list
                .edits()
                .iter()
                .map(SdfPass::sdf_edit_bounds)
                .collect();
            let bvh = build_flat_bvh(&bounds);
            let bvh_node_count = bvh.len() as u32;

            let bvh_bytes = bytemuck::cast_slice::<_, u8>(&bvh);
            let bvh_required = bvh_bytes.len().max(64) as u64;
            if bvh_required > self.bvh_nodes_buffer.size() {
                let new_size = (bvh_required * 2).max(64);
                self.bvh_nodes_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("SDF BVH Nodes"),
                    size: new_size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.bindings_dirty = true;
            }
            ctx.queue.write_buffer(&self.bvh_nodes_buffer, 0, bvh_bytes);

            let terrain_gpu = self
                .terrain_config
                .as_ref()
                .map(|c| c.build_gpu_params())
                .unwrap_or_else(GpuTerrainParams::disabled);
            ctx.queue.write_buffer(
                &self.terrain_params_buffer,
                0,
                bytemuck::bytes_of(&terrain_gpu),
            );

            let clip_cfg =
                self.build_clip_config(edit_count, bvh_node_count, self.terrain_config.as_ref());
            ctx.queue
                .write_buffer(&self.clip_config_buffer, 0, bytemuck::bytes_of(&clip_cfg));

            for level in 0..self.level_count {
                let params = self.build_level_params(level, edit_count);
                ctx.queue.write_buffer(
                    &self.level_params_buffers[level as usize],
                    0,
                    bytemuck::bytes_of(&params),
                );
            }

            self.edit_generation = self.edit_generation.wrapping_add(1);
            let edit_gen_offset: u64 = 128;
            ctx.queue.write_buffer(
                &self.scroll_state_buffer,
                edit_gen_offset,
                bytemuck::bytes_of(&self.edit_generation),
            );

            self.last_gen = gen;
        }

        if self.bindings_dirty {
            let classify_bgl = self.classify_pipeline.get_bind_group_layout(0);
            self.classify_bg = Self::build_classify_bg_impl(
                ctx.device,
                &classify_bgl,
                &self.clip_config_buffer,
                &self.scroll_state_buffer,
                &self.dirty_flags_buffer,
                &self.bvh_nodes_buffer,
                &self.per_brick_hashes_buffer,
                &self.per_brick_edit_lists_buffer,
                &self.all_brick_indices_buffer,
                &self.dirty_bricks_buffer,
                &self.eval_indirect_buffer,
            );
            let eval_bgl = self.eval_pipeline.get_bind_group_layout(0);
            self.eval_bgs = Self::build_eval_bgs_impl(
                ctx.device,
                &eval_bgl,
                &self.level_params_buffers,
                &self.edit_buffer,
                &self.atlas_buffers,
                &self.dirty_bricks_buffer,
                &self.per_brick_edit_lists_buffer,
                &self.terrain_params_buffer,
            );
            self.march_bg = None;
            self.march_bg_camera_key = 0;
            self.bindings_dirty = false;
        }

        let cam_pos = ctx.scene.camera.position();
        let mut any_level_dirty = false;
        for level in 0..self.level_count as usize {
            let vs = self.voxel_size_for_level(level as u32);
            let brick_step = vs * self.brick_size as f32;
            let new_snap = [
                (cam_pos[0] / brick_step).floor() as i32,
                (cam_pos[1] / brick_step).floor() as i32,
                (cam_pos[2] / brick_step).floor() as i32,
            ];
            if new_snap != self.cached_snap_origins[level] {
                any_level_dirty = true;
                self.cached_snap_origins[level] = new_snap;
            }
        }
        self.gpu_passes_clean = !needs_upload && !any_level_dirty;

        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        if !self.enabled {
            return Ok(());
        }

        let camera_ptr = ctx.scene.camera as *const _ as usize;
        if self.scroll_bg_camera_key != camera_ptr || self.scroll_bg.is_none() {
            let scroll_bgl = self.scroll_pipeline.get_bind_group_layout(0);
            self.scroll_bg = Some(Self::build_scroll_bg(
                ctx.device,
                &scroll_bgl,
                ctx.scene.camera,
                &self.clip_config_buffer,
                &self.scroll_state_buffer,
                &self.dirty_flags_buffer,
            ));
            self.scroll_bg_camera_key = camera_ptr;
        }
        if self.march_bg_camera_key != camera_ptr || self.march_bg.is_none() {
            self.march_bg = Some(Self::build_march_bg(
                ctx.device,
                &self.march_bgl,
                ctx.scene.camera,
                &self.clip_config_buffer,
                &self.scroll_state_buffer,
                &self.atlas_buffers,
                &self.all_brick_indices_buffer,
            ));
            self.march_bg_camera_key = camera_ptr;
        }

        if !self.gpu_passes_clean {
            unsafe { &mut *ctx.encoder_ptr }.copy_buffer_to_buffer(
                &self.eval_indirect_template_buffer,
                0,
                &self.eval_indirect_buffer,
                0,
                self.level_count as u64 * 3 * 4,
            );
            unsafe { &mut *ctx.encoder_ptr }.clear_buffer(&self.dirty_flags_buffer, 0, None);
            unsafe { &mut *ctx.encoder_ptr }.clear_buffer(&self.dirty_bricks_buffer, 0, None);

            {
                let mut cpass = ctx.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("SDF Scroll"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.scroll_pipeline);
                cpass.set_bind_group(0, self.scroll_bg.as_ref().unwrap(), &[]);
                cpass.dispatch_workgroups(1, 1, 1);
            }

            {
                let wgs_x = self.bricks_per_level / 64;
                let mut cpass = ctx.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("SDF Classify"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.classify_pipeline);
                cpass.set_bind_group(0, &self.classify_bg, &[]);
                cpass.dispatch_workgroups(wgs_x, self.level_count, 1);
            }

            {
                let mut cpass = ctx.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("SDF Evaluate"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.eval_pipeline);
                for (level, eval_bg) in self.eval_bgs.iter().enumerate() {
                    cpass.set_bind_group(0, eval_bg, &[]);
                    cpass.dispatch_workgroups_indirect(
                        &self.eval_indirect_buffer,
                        (level * 3 * 4) as u64,
                    );
                }
            }
        }

        {
            let depth_view = ctx.resources.full_res_depth.get().unwrap_or(ctx.depth);
            let color_load_op = if self.preserve_framebuffer {
                wgpu::LoadOp::Load
            } else {
                wgpu::LoadOp::Clear(wgpu::Color {
                    r: 0.53,
                    g: 0.72,
                    b: 0.90,
                    a: 1.0,
                })
            };
            let depth_load_op = if self.preserve_framebuffer {
                wgpu::LoadOp::Load
            } else {
                wgpu::LoadOp::Clear(1.0)
            };

            let desc = wgpu::RenderPassDescriptor {
                label: Some("SDF Ray March"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: ctx.target,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: color_load_op,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: depth_load_op,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            };

            let mut rpass = ctx.begin_render_pass(&desc);
            rpass.set_pipeline(&self.march_pipeline);
            rpass.set_bind_group(0, self.march_bg.as_ref().unwrap(), &[]);
            rpass.draw(0..3, 0..1);
        }

        Ok(())
    }
}
