//! Shadow depth pass - renders depth into the shadow atlas

use crate::buffer_pool::SharedPoolBuffer;
use crate::gpu_scene::MaterialRange;
use crate::graph::{RenderPass, PassContext, PassResourceBuilder, ResourceHandle};
use crate::Result;
use std::sync::{Arc, Mutex, atomic::{AtomicU32, Ordering}};
use wgpu::util::DeviceExt;

/// Per-light data written by the `Renderer` each frame and read by `ShadowPass::execute`
/// to skip draw calls that cannot contribute to a given shadow face.
///
/// Culling strategy: **Aggressive shadow rendering**
/// 1. **Range cull** – extended to 5x for point/spot lights (instead of 1x) to fill
///    the atlas with quality shadows for visible objects.
/// 2. **Hemisphere cull** (point lights only) – skip the mesh if it lies entirely
///    in the hemisphere opposite the cube face being rendered.
#[derive(Clone, Copy, Default)]
pub struct ShadowCullLight {
    pub position:       [f32; 3],
    pub direction:      [f32; 3],
    pub range:          f32,
    pub is_directional: bool,
    pub is_point:       bool,
    /// FNV-1a hash of this light's computed shadow matrix/matrices.
    pub matrix_hash:    u64,
}

/// GPU-side light data for shadow culling (uploaded from ShadowCullLight each frame).
/// 32 bytes; layout matches the WGSL struct in shadow_cull.wgsl.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ShadowCullGpuLight {
    position:   [f32; 3],  // offset  0
    light_type: u32,       // offset 12  (0=point, 1=directional, 2=spot)
    direction:  [f32; 3],  // offset 16
    range:      f32,       // offset 28
    // total 32 bytes
}

/// Params uniform for shadow_cull.wgsl (16 bytes).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ShadowCullParams {
    draw_count:          u32,
    light_count:         u32,
    shadow_max_distance: f32,
    _pad:                u32,
}

/// Per-light shadow cache entry.
#[derive(Clone, Copy, Default)]
struct ShadowLightCache {
    light_hash: u64,
    geom_hash:  u64,
    valid:      bool,
}

pub struct ShadowPass {
    device: Arc<wgpu::Device>,
    queue:  Arc<wgpu::Queue>,
    layer_views: Vec<Arc<wgpu::TextureView>>,
    /// Kept alive so bind groups referencing it are never invalidated.
    _shadow_matrix_buffer: Arc<wgpu::Buffer>,
    light_count: Arc<AtomicU32>,
    light_face_counts: Arc<Mutex<Vec<u8>>>,
    cull_lights: Arc<Mutex<Vec<ShadowCullLight>>>,
    pipeline: Arc<wgpu::RenderPipeline>,
    /// Per-face (light_idx*6+face) bind group.
    slot_bind_groups: Vec<wgpu::BindGroup>,
    /// Kept alive so slot_bind_groups referencing them are never invalidated.
    _slot_idx_buffers: Vec<wgpu::Buffer>,
    /// Pool VB/IB for unified geometry.
    pool_vertex_buffer: SharedPoolBuffer,
    pool_index_buffer:  SharedPoolBuffer,

    // ── GPU indirect shadow cull ───────────────────────────────────────────
    /// Opaque draw-call buffer from GpuScene (Arc refreshed each frame via shared slot).
    shared_draw_call_buf: Arc<Mutex<Option<Arc<wgpu::Buffer>>>>,
    /// Material ranges (same as used by GBuffer pass).
    shared_material_ranges: Arc<Mutex<Vec<MaterialRange>>>,
    /// True when wgpu::Features::MULTI_DRAW_INDIRECT is available.
    has_multi_draw: bool,

    // GPU resources for shadow cull compute
    shadow_cull_pipeline:    Option<Arc<wgpu::ComputePipeline>>,
    shadow_cull_params_buf:  Option<wgpu::Buffer>,
    shadow_cull_bg:          Option<wgpu::BindGroup>,
    shadow_light_buf:        Option<Arc<wgpu::Buffer>>,
    shadow_light_buf_cap:    u32,
    shadow_indirect_buf:     Option<Arc<wgpu::Buffer>>,
    shadow_indirect_draw_cap: u32,  // indirect capacity in draw-call slots
    shadow_indirect_light_cap: u32, // indirect capacity in light slots

    // Bind-group staleness detection (pointer comparison)
    last_dc_buf_ptr:    usize,
    last_sl_buf_ptr:    usize,
    last_ind_buf_ptr:   usize,

    /// Per-light shadow cache (light_hash + geom_hash) to skip unchanged lights.
    shadow_cache: Vec<ShadowLightCache>,
}

impl ShadowPass {
    pub fn new(
        light_face_counts: Arc<Mutex<Vec<u8>>>,
        cull_lights: Arc<Mutex<Vec<ShadowCullLight>>>,
        layer_views: Vec<Arc<wgpu::TextureView>>,
        shadow_matrix_buffer: Arc<wgpu::Buffer>,
        light_count: Arc<AtomicU32>,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        material_layout: &wgpu::BindGroupLayout,
        instance_data_buffer: &wgpu::Buffer,
        pool_vertex_buffer: SharedPoolBuffer,
        pool_index_buffer: SharedPoolBuffer,
        shared_draw_call_buf: Arc<Mutex<Option<Arc<wgpu::Buffer>>>>,
        shared_material_ranges: Arc<Mutex<Vec<MaterialRange>>>,
        has_multi_draw: bool,
    ) -> Self {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shadow Matrix BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shadow Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/passes/shadow.wgsl").into(),
            ),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Shadow Pipeline Layout"),
            bind_group_layouts: &[Some(&bgl), Some(material_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Shadow Pipeline"),
            layout: Some(&pipeline_layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: 32,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 0,
                                shader_location: 0,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 16,
                                shader_location: 2,
                            },
                        ],
                    },
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 0,
                    slope_scale: 1.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
        });

        let max_lights = layer_views.len() / 6;
        let total_slots = max_lights * 6;

        let mut slot_idx_buffers = Vec::with_capacity(total_slots);
        let mut slot_bind_groups = Vec::with_capacity(total_slots);
        for slot in 0..total_slots {
            let idx_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("shadow_layer_idx"),
                contents: &(slot as u32).to_le_bytes(),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Shadow Slot BG"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: shadow_matrix_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: idx_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: instance_data_buffer.as_entire_binding(),
                    },
                ],
            });
            slot_idx_buffers.push(idx_buf);
            slot_bind_groups.push(bg);
        }

        Self {
            device,
            queue,
            layer_views,
            _shadow_matrix_buffer: shadow_matrix_buffer,
            light_count,
            light_face_counts,
            cull_lights,
            pipeline: Arc::new(pipeline),
            slot_bind_groups,
            _slot_idx_buffers: slot_idx_buffers,
            pool_vertex_buffer,
            pool_index_buffer,
            shared_draw_call_buf,
            shared_material_ranges,
            has_multi_draw,
            shadow_cull_pipeline:     None,
            shadow_cull_params_buf:   None,
            shadow_cull_bg:           None,
            shadow_light_buf:         None,
            shadow_light_buf_cap:     0,
            shadow_indirect_buf:      None,
            shadow_indirect_draw_cap:  0,
            shadow_indirect_light_cap: 0,
            last_dc_buf_ptr:  0,
            last_sl_buf_ptr:  0,
            last_ind_buf_ptr: 0,
            shadow_cache: vec![ShadowLightCache::default(); max_lights],
        }
    }

    /// Build the shadow-cull compute pipeline (once on first use).
    fn ensure_cull_pipeline(&mut self) {
        if self.shadow_cull_pipeline.is_some() { return; }

        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shadow Cull Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/passes/shadow_cull.wgsl").into(),
            ),
        });
        let bgl = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shadow Cull BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                // binding 4: shadow matrices — used by directional light cascade frustum cull
                wgpu::BindGroupLayoutEntry {
                    binding: 4, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
            ],
        });
        let pl = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Shadow Cull Pipeline Layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        self.shadow_cull_pipeline = Some(Arc::new(self.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("Shadow Cull Pipeline"),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some("shadow_cull"),
                compilation_options: Default::default(),
                cache: None,
            },
        )));
        self.shadow_cull_params_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shadow Cull Params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
    }
}

impl RenderPass for ShadowPass {
    fn name(&self) -> &str { "shadow" }

    fn declare_resources(&self, builder: &mut PassResourceBuilder) {
        builder.write(ResourceHandle::named("shadow_atlas"));
    }

    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        let light_count  = self.light_count.load(Ordering::Relaxed) as usize;
        let actual_count = light_count.min(self.layer_views.len() / 6);
        if actual_count == 0 { return Ok(()); }

        // Get the current opaque draw-call buffer; skip if not yet populated.
        let draw_call_buf = self.shared_draw_call_buf.lock().unwrap().clone();
        let Some(draw_call_buf) = draw_call_buf else { return Ok(()); };

        let material_ranges: Vec<MaterialRange> = self.shared_material_ranges.lock().unwrap().clone();
        let draw_count: u32 = material_ranges.iter().map(|r| r.count).sum();
        if draw_count == 0 { return Ok(()); }

        let face_counts: Vec<u8> = self.light_face_counts.lock().unwrap().clone();
        let cull_lights: Vec<ShadowCullLight> = self.cull_lights.lock().unwrap().clone();

        // ── Shadow cache check (O(N_lights) CPU work) ────────────────────────
        // geom_hash uses the draw_list_generation counter from PassContext —
        // incremented whenever objects are added or removed from the scene.
        let geom_hash = ctx.draw_list_generation;

        let mut dirty_lights: Vec<usize> = Vec::new();
        let mut light_data:   Vec<(u64, u32)> = Vec::with_capacity(actual_count); // (light_hash, max_faces)

        for i in 0..actual_count {
            let max_faces = face_counts.get(i).copied().unwrap_or(6) as u32;
            let light = cull_lights.get(i).copied().unwrap_or_default();

            // FNV-1a hash of the per-light properties that affect shadow output.
            let mut h: u64 = 0xcbf29ce484222325;
            for &v in &[
                light.matrix_hash,
                light.range.to_bits() as u64,
                light.is_directional as u64,
                light.is_point as u64,
                max_faces as u64,
            ] {
                h ^= v;
                h = h.wrapping_mul(0x100000001b3);
            }
            let light_hash = h;
            light_data.push((light_hash, max_faces));

            let cached = &self.shadow_cache[i];
            if !(cached.valid && cached.light_hash == light_hash && cached.geom_hash == geom_hash) {
                dirty_lights.push(i);
            }
        }

        if dirty_lights.is_empty() { return Ok(()); }

        // ── Ensure GPU cull pipeline ──────────────────────────────────────────
        self.ensure_cull_pipeline();

        // ── Ensure shadow light buffer ────────────────────────────────────────
        if self.shadow_light_buf.is_none() || self.shadow_light_buf_cap < actual_count as u32 {
            let new_cap = (actual_count as u32).next_power_of_two().max(8);
            self.shadow_light_buf = Some(Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Shadow Cull Light Buf"),
                size: new_cap as u64 * std::mem::size_of::<ShadowCullGpuLight>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })));
            self.shadow_light_buf_cap = new_cap;
            self.last_sl_buf_ptr = 0;
        }

        // Upload shadow light data each frame (only actual_count entries, < 2 KB).
        let gpu_lights: Vec<ShadowCullGpuLight> = cull_lights[..actual_count].iter().map(|l| {
            ShadowCullGpuLight {
                position:   l.position,
                light_type: if l.is_directional { 1 } else if l.is_point { 0 } else { 2 },
                direction:  l.direction,
                range:      l.range,
            }
        }).collect();
        self.queue.write_buffer(self.shadow_light_buf.as_ref().unwrap(), 0,
            bytemuck::cast_slice(&gpu_lights));

        // ── Ensure shadow indirect buffer ─────────────────────────────────────
        // Layout: [light_face * draw_count + draw_idx] (20 bytes each)
        let needs_resize = self.shadow_indirect_buf.is_none()
            || self.shadow_indirect_draw_cap  < draw_count
            || self.shadow_indirect_light_cap < actual_count as u32;
        if needs_resize {
            let dc_cap = draw_count.next_power_of_two().max(64);
            let li_cap = (actual_count as u32).next_power_of_two().max(8);
            self.shadow_indirect_buf = Some(Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Shadow Indirect Buf"),
                size: li_cap as u64 * 6 * dc_cap as u64 * 20,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })));
            self.shadow_indirect_draw_cap  = dc_cap;
            self.shadow_indirect_light_cap = li_cap;
            self.last_ind_buf_ptr = 0;
        }

        // ── Rebuild bind group when any buffer changes identity ───────────────
        let dc_ptr  = Arc::as_ptr(&draw_call_buf) as usize;
        let sl_ptr  = Arc::as_ptr(self.shadow_light_buf.as_ref().unwrap()) as usize;
        let ind_ptr = Arc::as_ptr(self.shadow_indirect_buf.as_ref().unwrap()) as usize;

        let bg_stale = self.shadow_cull_bg.is_none()
            || dc_ptr  != self.last_dc_buf_ptr
            || sl_ptr  != self.last_sl_buf_ptr
            || ind_ptr != self.last_ind_buf_ptr;

        if bg_stale {
            let pipeline = self.shadow_cull_pipeline.as_ref().unwrap();
            let params   = self.shadow_cull_params_buf.as_ref().unwrap();
            self.shadow_cull_bg = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Shadow Cull BG"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: draw_call_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: self.shadow_light_buf.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: self.shadow_indirect_buf.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: params.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: self._shadow_matrix_buffer.as_entire_binding() },
                ],
            }));
            self.last_dc_buf_ptr  = dc_ptr;
            self.last_sl_buf_ptr  = sl_ptr;
            self.last_ind_buf_ptr = ind_ptr;
        }

        // ── Upload cull params and dispatch compute cull ─────────────────────
        const SHADOW_MAX_DISTANCE: f32 = 300.0;
        self.queue.write_buffer(
            self.shadow_cull_params_buf.as_ref().unwrap(),
            0,
            bytemuck::bytes_of(&ShadowCullParams {
                draw_count,
                light_count:         actual_count as u32,
                shadow_max_distance: SHADOW_MAX_DISTANCE,
                _pad: 0,
            }),
        );

        // Clear the indirect buffer region used this frame so culled entries are zero.
        let indirect = self.shadow_indirect_buf.as_ref().unwrap().clone();
        ctx.encoder.clear_buffer(
            &indirect, 0,
            Some(actual_count as u64 * 6 * draw_count as u64 * 20),
        );

        {
            let pipeline = self.shadow_cull_pipeline.as_ref().unwrap();
            let bg       = self.shadow_cull_bg.as_ref().unwrap();
            let mut cpass = ctx.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Shadow Cull"), timestamp_writes: None,
            });
            cpass.set_pipeline(pipeline);
            cpass.set_bind_group(0, bg, &[]);
            let wx = (draw_count + 63) / 64;
            let wy = actual_count as u32 * 6;
            cpass.dispatch_workgroups(wx, wy, 1);
        }

        // ── Render dirty lights ───────────────────────────────────────────────
        let vb = self.pool_vertex_buffer.lock().unwrap().clone();
        let ib = self.pool_index_buffer.lock().unwrap().clone();

        for &i in &dirty_lights {
            let (light_hash, max_faces) = light_data[i];
            let t_light = ctx.scope_begin(&format!("shadow/light_{i}"));

            for face in 0u32..max_faces {
                let layer_idx  = i as u32 * 6 + face;
                let light_face = i as u32 * 6 + face;
                let t_face = ctx.scope_begin(&format!("shadow/light_{i}/face_{face}"));

                let mut pass = ctx.begin_render_pass(
                    &format!("Shadow Light {i} Face {face}"),
                    &[],
                    Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.layer_views[layer_idx as usize],
                        depth_ops: Some(wgpu::Operations {
                            load:  wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                );

                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.slot_bind_groups[layer_idx as usize], &[]);
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);

                // O(N_materials) — one multi-draw per material range per face.
                for range in material_ranges.iter() {
                    pass.set_bind_group(1, Some(range.bind_group.as_ref()), &[]);
                    let face_base = light_face as u64 * draw_count as u64 * 20;
                    let mat_off   = face_base + range.start as u64 * 20;
                    if self.has_multi_draw {
                        pass.multi_draw_indexed_indirect(&indirect, mat_off, range.count);
                    } else {
                        for j in 0..range.count {
                            pass.draw_indexed_indirect(&indirect, mat_off + j as u64 * 20);
                        }
                    }
                }

                drop(pass);
                ctx.scope_end(t_face);
            }

            ctx.scope_end(t_light);
            self.shadow_cache[i] = ShadowLightCache { light_hash, geom_hash, valid: true };
        }

        Ok(())
    }
}


