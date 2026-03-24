//! SDF Clipmap pass — sparse brick atlas + toroidal clip-map + fullscreen ray march.
//!
//! Replaces the earlier JFA screen-space SDF with a true volumetric SDF system:
//!
//! - Up to 8 nested clip-map levels (level 0 = finest, level 7 = coarsest)
//! - Each level: 16^3 grid of 8^3-voxel bricks stored in a 8^3-brick atlas
//! - Per-frame CPU work: O(1) — only the newly-visible toroidal shell is reclassified
//! - GPU: compute pass bakes SDF into occupied bricks, fragment pass sphere-traces
//!
//! # Usage
//! ```no_run
//! let mut sdf = SdfClipmapPass::new(&device, camera_buf, surface_format);
//! sdf.set_terrain(TerrainConfig::rolling());
//! renderer.add_pass(Box::new(sdf));
//! ```

pub mod primitives;
pub mod edit_list;
pub mod terrain;
pub mod noise;
pub mod edit_bvh;
pub mod uniforms;
pub mod brick;
pub mod clip_map;

use bytemuck;
use glam::Vec3;
use helio_v3::{RenderPass, PassContext, PrepareContext, Result as HelioResult};

use edit_list::{SdfEdit, GpuSdfEdit, SdfEditList};
use edit_bvh::{Aabb, EditBvh};
use terrain::{TerrainConfig, GpuTerrainParams};
use clip_map::{SdfClipMap, LEVEL_COUNT};
use uniforms::SdfGridParams;

const MAX_EDITS: usize = 4096;

// -- AABB helper ---------------------------------------------------------------

fn edit_aabb(edit: &SdfEdit) -> Aabb {
    use primitives::SdfShapeType;
    let p = &edit.params;
    let local_max = match edit.shape {
        SdfShapeType::Sphere   => Vec3::splat(p.param0),
        SdfShapeType::Cube     => Vec3::new(p.param0, p.param1, p.param2),
        SdfShapeType::Capsule  => Vec3::new(p.param0, p.param0 + p.param1, p.param0),
        SdfShapeType::Torus    => Vec3::new(p.param0 + p.param1, p.param1, p.param0 + p.param1),
        SdfShapeType::Cylinder => Vec3::new(p.param0, p.param1, p.param0),
    };
    let corners = [
        Vec3::new(-local_max.x, -local_max.y, -local_max.z),
        Vec3::new( local_max.x, -local_max.y, -local_max.z),
        Vec3::new(-local_max.x,  local_max.y, -local_max.z),
        Vec3::new( local_max.x,  local_max.y, -local_max.z),
        Vec3::new(-local_max.x, -local_max.y,  local_max.z),
        Vec3::new( local_max.x, -local_max.y,  local_max.z),
        Vec3::new(-local_max.x,  local_max.y,  local_max.z),
        Vec3::new( local_max.x,  local_max.y,  local_max.z),
    ];
    let mut wmin = Vec3::splat(f32::INFINITY);
    let mut wmax = Vec3::splat(f32::NEG_INFINITY);
    for c in &corners {
        let wc = edit.transform.transform_point3(*c);
        wmin = wmin.min(wc);
        wmax = wmax.max(wc);
    }
    Aabb::new(wmin, wmax)
}

// -- BGL helpers ---------------------------------------------------------------

fn bgl_uniform(binding: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: vis,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_storage_r(binding: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: vis,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_storage_rw(binding: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: vis,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

// -- Pass struct ---------------------------------------------------------------

/// Volumetric SDF clipmap pass: sparse brick atlas + fullscreen ray march.
pub struct SdfClipmapPass {
    // -- CPU state -------------------------------------------------------------
    edit_list:   SdfEditList,
    edit_bvh:    EditBvh,
    terrain:     Option<TerrainConfig>,
    clip_map:    SdfClipMap,
    first_frame: bool,
    edits_dirty: bool,

    // -- Global GPU buffers ----------------------------------------------------
    edits_buf:   wgpu::Buffer,
    terrain_buf: wgpu::Buffer,

    // -- Compute pipeline ------------------------------------------------------
    compute_pipeline:  wgpu::ComputePipeline,
    level_compute_bgs: Vec<wgpu::BindGroup>,

    // -- Render pipeline -------------------------------------------------------
    render_pipeline: wgpu::RenderPipeline,
    group0_bg:       wgpu::BindGroup,
    group1_bg:       wgpu::BindGroup,
}

impl SdfClipmapPass {
    /// Create the pass.
    /// * `camera_buf`     - GpuCameraUniforms buffer (368 bytes)
    /// * `surface_format` - swapchain color format
    pub fn new(
        device: &wgpu::Device,
        camera_buf: &wgpu::Buffer,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let base_voxel_size = 0.5_f32;

        let mut clip_map = SdfClipMap::new(base_voxel_size, Vec3::ZERO);
        clip_map.create_gpu_resources(device);

        let edits_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Edits"),
            size: (MAX_EDITS * std::mem::size_of::<GpuSdfEdit>()).max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let terrain_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Terrain"),
            size: std::mem::size_of::<GpuTerrainParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Compute pipeline.
        let cs_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sdf_evaluate_sparse"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/sdf_evaluate_sparse.wgsl").into(),
            ),
        });
        let compute_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SdfCompute BGL"),
            entries: &[
                bgl_uniform(0, wgpu::ShaderStages::COMPUTE),
                bgl_storage_r(1, wgpu::ShaderStages::COMPUTE),
                bgl_storage_rw(2, wgpu::ShaderStages::COMPUTE),
                bgl_storage_r(3, wgpu::ShaderStages::COMPUTE),
                bgl_storage_r(4, wgpu::ShaderStages::COMPUTE),
                bgl_uniform(5, wgpu::ShaderStages::COMPUTE),
                bgl_storage_r(6, wgpu::ShaderStages::COMPUTE),
                bgl_storage_r(7, wgpu::ShaderStages::COMPUTE),
            ],
        });
        let compute_pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SdfCompute PLL"),
            bind_group_layouts: &[Some(&compute_bgl)],
            immediate_size: 0,
        });
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SdfCompute"),
            layout: Some(&compute_pll),
            module: &cs_mod,
            entry_point: Some("cs_evaluate_sparse"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let level_compute_bgs: Vec<wgpu::BindGroup> = clip_map.levels.iter().map(|level| {
            let bm = &level.brick_map;
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("SdfComputeBG"),
                layout: &compute_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: bm.params_buf.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: edits_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: bm.atlas_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: bm.active_brick_buf.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: bm.brick_index_buf.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: terrain_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: bm.edit_list_offsets_buf.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: bm.edit_list_data_buf.as_ref().unwrap().as_entire_binding() },
                ],
            })
        }).collect();

        // Render pipeline.
        let fs_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sdf_ray_march_clipmap"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/sdf_ray_march_clipmap.wgsl").into(),
            ),
        });
        let vs_fs = wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT;
        let group0_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SdfRender G0 BGL"),
            entries: &[bgl_uniform(0, vs_fs)],
        });
        let mut g1_entries = vec![bgl_uniform(0, wgpu::ShaderStages::FRAGMENT)];
        for i in 0..(LEVEL_COUNT as u32) {
            g1_entries.push(bgl_storage_r(1 + i, wgpu::ShaderStages::FRAGMENT));
        }
        g1_entries.push(bgl_storage_r(1 + LEVEL_COUNT as u32, wgpu::ShaderStages::FRAGMENT));
        let group1_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SdfRender G1 BGL"),
            entries: &g1_entries,
        });
        let render_pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SdfRender PLL"),
            bind_group_layouts: &[Some(&group0_bgl), Some(&group1_bgl)],
            immediate_size: 0,
        });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SdfRayMarch"),
            layout: Some(&render_pll),
            vertex: wgpu::VertexState {
                module: &fs_mod,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &fs_mod,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
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

        let group0_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SdfRender G0"),
            layout: &group0_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buf.as_entire_binding(),
            }],
        });

        let mut g1_bg_entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: clip_map.clip_params_buf.as_ref().unwrap().as_entire_binding(),
        }];
        for (i, level) in clip_map.levels.iter().enumerate() {
            g1_bg_entries.push(wgpu::BindGroupEntry {
                binding: 1 + i as u32,
                resource: level.brick_map.atlas_buffer.as_ref().unwrap().as_entire_binding(),
            });
        }
        g1_bg_entries.push(wgpu::BindGroupEntry {
            binding: 1 + LEVEL_COUNT as u32,
            resource: clip_map.all_brick_indices_buf.as_ref().unwrap().as_entire_binding(),
        });
        let group1_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SdfRender G1"),
            layout: &group1_bgl,
            entries: &g1_bg_entries,
        });

        Self {
            edit_list: SdfEditList::new(),
            edit_bvh: EditBvh::new(),
            terrain: None,
            clip_map,
            first_frame: true,
            edits_dirty: false,
            edits_buf,
            terrain_buf,
            compute_pipeline,
            level_compute_bgs,
            render_pipeline,
            group0_bg,
            group1_bg,
        }
    }

    /// Add an SDF edit. Returns its stable index for future reference.
    pub fn add_edit(&mut self, edit: SdfEdit) -> usize {
        let aabb = edit_aabb(&edit);
        let idx = self.edit_list.add(edit);
        self.edit_bvh.insert(idx, aabb);
        self.edits_dirty = true;
        idx
    }

    /// Remove an edit by index (rebuilds BVH, forces full reclassify).
    pub fn remove_edit(&mut self, idx: usize) {
        self.edit_list.remove(idx);
        self.edit_bvh = EditBvh::new();
        for (i, edit) in self.edit_list.edits().iter().enumerate() {
            self.edit_bvh.insert(i, edit_aabb(edit));
        }
        self.edits_dirty = true;
        self.first_frame = true;
    }

    /// Set terrain configuration (forces full reclassify on next frame).
    pub fn set_terrain(&mut self, config: TerrainConfig) {
        self.terrain = Some(config);
        self.first_frame = true;
    }
}

impl RenderPass for SdfClipmapPass {
    fn name(&self) -> &'static str { "SdfClipmapPass" }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let cam_pos = Vec3::from(ctx.scene.camera.position());

        // Snapshot previous origins before update_center moves them.
        let mut prev_origins = [[0i32; 3]; LEVEL_COUNT];
        for (i, level) in self.clip_map.levels.iter().enumerate() {
            prev_origins[i] = level.brick_map.toroidal_origin;
        }

        let dirty_mask: u8 = if self.first_frame {
            self.clip_map.update_center(cam_pos);
            0xFF
        } else {
            self.clip_map.update_center(cam_pos)
        };

        // Build GPU edit list (needed for brick-level edit lists).
        let edits_gpu: Vec<GpuSdfEdit> =
            self.edit_list.edits().iter().map(|e| e.to_gpu()).collect();
        let terrain = self.terrain.as_ref();

        // Classify bricks. Separate field borrows are disjoint — the borrow checker allows this.
        if self.first_frame || self.edits_dirty {
            self.clip_map.classify_all(&self.edit_bvh, &edits_gpu, terrain);
            self.first_frame = false;
        } else if dirty_mask != 0 {
            self.clip_map.classify_toroidal_levels(
                dirty_mask, &prev_origins, &self.edit_bvh, &edits_gpu, terrain,
            );
        }

        let effective_mask = if self.edits_dirty { 0xFF } else { dirty_mask };

        // Upload edits GPU buffer.
        if self.edits_dirty {
            let gpu_data = self.edit_list.flush_gpu_data();
            if !gpu_data.is_empty() {
                ctx.queue.write_buffer(&self.edits_buf, 0, bytemuck::cast_slice(&gpu_data));
            }
            self.edits_dirty = false;
        }

        // Upload terrain.
        if let Some(tc) = &self.terrain {
            ctx.queue.write_buffer(&self.terrain_buf, 0, bytemuck::bytes_of(&tc.build_gpu_params()));
        }

        // Upload per-level grid params.
        let edit_count = self.edit_list.len() as u32;
        let terrain_on = self.terrain.is_some();
        for level in &self.clip_map.levels {
            if let Some(buf) = &level.brick_map.params_buf {
                let p = level.brick_map.build_grid_params(edit_count, terrain_on);
                ctx.queue.write_buffer(buf, 0, bytemuck::bytes_of(&p));
            }
        }

        // Upload dirty brick buffers.
        self.clip_map.upload_dirty(effective_mask, ctx.queue);

        // Rebuild and upload all-brick-indices.
        self.clip_map.update_cached_indices(ctx.queue);

        // Upload clip-map params.
        self.clip_map.upload_clip_params(ctx.queue);

        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        // -- Compute: bake SDF into brick atlases --
        {
            let mut cpass = ctx.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SdfClipmap Evaluate"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.compute_pipeline);
            for (i, bg) in self.level_compute_bgs.iter().enumerate() {
                let active = self.clip_map.levels[i].brick_map.active_count();
                if active == 0 { continue; }
                cpass.set_bind_group(0, bg, &[]);
                cpass.dispatch_workgroups((active + 255) / 256, 1, 1);
            }
        }

        // -- Render: fullscreen SDF ray march --
        let color_attachments = [Some(wgpu::RenderPassColorAttachment {
            view: ctx.target,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.05, g: 0.05, b: 0.1, a: 1.0 }),
                store: wgpu::StoreOp::Store,
            },
        })];
        let depth_stencil = Some(wgpu::RenderPassDepthStencilAttachment {
            view: ctx.depth,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        });
        let rp_desc = wgpu::RenderPassDescriptor {
            label: Some("SdfClipmap RayMarch"),
            color_attachments: &color_attachments,
            depth_stencil_attachment: depth_stencil,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        };
        let mut pass = ctx.begin_render_pass(&rp_desc);
        pass.set_pipeline(&self.render_pipeline);
        pass.set_bind_group(0, &self.group0_bg, &[]);
        pass.set_bind_group(1, &self.group1_bg, &[]);
        pass.draw(0..3, 0..1);
        Ok(())
    }
}
