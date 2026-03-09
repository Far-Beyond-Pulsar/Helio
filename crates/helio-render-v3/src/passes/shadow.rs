/// Shadow pass — cubemap-atlas shadow rendering.
///
/// Each shadow-casting light (point = 6 faces, spot = 1, directional = N cascades)
/// gets a slice of the depth atlas array.
///
/// Bundle cache: one RenderBundle per atlas slot. Hash = FNV-1a over
/// `(hism_handle: u32, instance_count: u32)` — deterministic, NOT pointer addresses.
///
/// Pre-baked per-slot bind groups are created in new() for all max_lights×6 slots.
/// The u32 layer-index buffers that back them are kept alive in `slot_idx_buffers`
/// (they must outlive their bind groups).
use std::sync::Arc;
use glam::{Mat4, Vec3};
use crate::{
    graph::pass::{RenderPass, PassContext},
    mesh::{PackedVertex, DrawCall},
    hism::shadow_draw_list_hash,
};

const MAX_FACES_PER_LIGHT: u32 = 6;

#[derive(Clone, Debug)]
pub struct ShadowConfig {
    pub atlas_size:       u32,  // e.g. 2048
    pub max_shadow_lights: u32, // e.g. 8
}

pub struct ShadowPass {
    pipeline:          Arc<wgpu::RenderPipeline>,
    pub atlas:         Arc<wgpu::Texture>,
    pub atlas_view:    wgpu::TextureView,
    slot_bind_groups:  Vec<wgpu::BindGroup>,     // pre-baked: one per atlas layer
    slot_idx_buffers:  Vec<wgpu::Buffer>,         // MUST outlive slot_bind_groups
    shadow_matrix_buf: Arc<wgpu::Buffer>,         // STORAGE of GpuShadowMatrix
    slot_bundles:      Vec<Option<(wgpu::RenderBundle, u64)>>,  // (bundle, hash)
    config:            ShadowConfig,
}

/// One shadow projection matrix in the atlas — 96 bytes to match deferred_lighting.wgsl.
/// Layout must match WGSL `ShadowMatrix` struct exactly.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuShadowMatrix {
    pub view_proj:       [[f32; 4]; 4],  // offset  0, size 64
    pub light_dir:       [f32; 3],       // offset 64, size 12
    pub atlas_layer:     u32,            // offset 76, size  4
    pub atlas_uv_offset: [f32; 2],       // offset 80, size  8
    pub atlas_uv_scale:  f32,            // offset 88, size  4
    pub _pad:            f32,            // offset 92, size  4  → total 96
}

impl ShadowPass {
    pub fn new(
        device:          &wgpu::Device,
        config:          ShadowConfig,
        camera_bgl:      &wgpu::BindGroupLayout,
    ) -> Self {
        let total_slots = config.max_shadow_lights * MAX_FACES_PER_LIGHT;

        let atlas = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label:           Some("shadow_atlas"),
            size:            wgpu::Extent3d { width: config.atlas_size, height: config.atlas_size, depth_or_array_layers: total_slots },
            mip_level_count: 1,
            sample_count:    1,
            dimension:       wgpu::TextureDimension::D2,
            format:          wgpu::TextureFormat::Depth32Float,
            usage:           wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats:    &[],
        }));

        let atlas_view = atlas.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        // Shadow matrix storage buffer
        let mat_size = (std::mem::size_of::<GpuShadowMatrix>() * total_slots as usize).max(16) as u64;
        let shadow_matrix_buf = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("shadow_matrices"),
            size:               mat_size,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // Per-slot bind group layout (minimal — just shadow matrix storage + layer idx uniform)
        let slot_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow_slot_bgl"),
            entries: &[
                // 0: shadow matrix storage
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                // 1: layer index uniform (immutable per slot — baked at construction)
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
            ],
        });

        // Build one idx buffer + bind group per slot UP FRONT in new().
        // Doing this at render time would require stalling.
        use wgpu::util::DeviceExt;
        let mut slot_idx_buffers = Vec::with_capacity(total_slots as usize);
        let mut slot_bind_groups = Vec::with_capacity(total_slots as usize);

        for slot in 0..total_slots {
            let layer_idx: u32 = slot;
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label:    None,
                contents: bytemuck::bytes_of(&layer_idx),
                usage:    wgpu::BufferUsages::UNIFORM,
            });
            slot_idx_buffers.push(buf);
        }

        // Create bind groups AFTER all buffers exist (borrows into slot_idx_buffers are stable now).
        for slot in 0..total_slots as usize {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label:   None,
                layout:  &slot_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: shadow_matrix_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: slot_idx_buffers[slot].as_entire_binding() },
                ],
            });
            slot_bind_groups.push(bg);
        }

        let pipeline = Arc::new(build_pipeline(device, &slot_bgl));
        let slot_bundles = (0..total_slots).map(|_| None).collect();

        ShadowPass {
            pipeline, atlas, atlas_view,
            slot_bind_groups, slot_idx_buffers, shadow_matrix_buf,
            slot_bundles, config,
        }
    }

    pub fn shadow_matrix_buffer(&self) -> &wgpu::Buffer { &self.shadow_matrix_buf }
    pub fn shadow_matrix_buf_arc(&self) -> Arc<wgpu::Buffer> { self.shadow_matrix_buf.clone() }

    /// Upload shadow projection matrices for this frame (called from renderer's prep).
    pub fn upload_matrices(&self, queue: &wgpu::Queue, matrices: &[GpuShadowMatrix]) {
        if matrices.is_empty() { return; }
        queue.write_buffer(&self.shadow_matrix_buf, 0, bytemuck::cast_slice(matrices));
    }

    fn rebuild_slot_bundle(&mut self, device: &wgpu::Device, slot: usize, draws: &[DrawCall]) {
        let mut enc = device.create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
            label:            Some("shadow_bundle"),
            color_formats:    &[],
            depth_stencil:    Some(wgpu::RenderBundleDepthStencil {
                format:            wgpu::TextureFormat::Depth32Float,
                depth_read_only:   false,
                stencil_read_only: true,
            }),
            sample_count: 1, multiview: None,
        });

        enc.set_pipeline(&self.pipeline);
        enc.set_bind_group(0, &self.slot_bind_groups[slot], &[]);

        for draw in draws {
            enc.set_vertex_buffer(0, draw.vertex_buffer.slice(..));
            enc.set_vertex_buffer(1, draw.instance_buffer.slice(draw.instance_buffer_offset..));
            enc.set_index_buffer(draw.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            enc.draw_indexed(0..draw.index_count, 0, 0..draw.instance_count);
        }

        let bundle = enc.finish(&wgpu::RenderBundleDescriptor { label: Some("shadow_bundle") });
        let hash = shadow_draw_list_hash(draws);
        self.slot_bundles[slot] = Some((bundle, hash));
    }
}

impl RenderPass for ShadowPass {
    fn execute(&mut self, ctx: &mut PassContext) {
        let shadow_lights: Vec<usize> = ctx.lights.iter().enumerate()
            .filter(|(_, l)| l.cos_angles_shadow[2] > 0.5)
            .map(|(i, _)| i)
            .take(self.config.max_shadow_lights as usize)
            .collect();

        if shadow_lights.is_empty() { return; }

        // Build and upload shadow matrices for this frame.
        let total_slots = (self.config.max_shadow_lights * MAX_FACES_PER_LIGHT) as usize;
        let mut matrices = vec![GpuShadowMatrix { view_proj: [[0.0;4];4], light_dir: [0.0;3], atlas_layer: 0, atlas_uv_offset: [0.0;2], atlas_uv_scale: 0.0, _pad: 0.0 }; total_slots];
        for (slot, &light_idx) in shadow_lights.iter().enumerate() {
            if slot >= total_slots { break; }
            let light = &ctx.lights[light_idx];
            let pos = Vec3::from_slice(&light.position_type[0..3]);
            let dir = Vec3::from_slice(&light.direction_range[0..3]);
            let range = light.direction_range[3].max(1.0);
            let light_type = light.position_type[3]; // 0=dir, 1=point, 2=spot

            let up = if dir.normalize().abs().dot(Vec3::Y) < 0.99 { Vec3::Y } else { Vec3::X };
            let view_proj = if light_type == 0.0 {
                // Directional: ortho, positioned along reverse direction
                let eye = -dir.normalize() * 50.0;
                let v = Mat4::look_at_rh(eye, Vec3::ZERO, up);
                let p = Mat4::orthographic_rh(-30.0, 30.0, -30.0, 30.0, 0.1, 200.0);
                p * v
            } else {
                // Point / Spot: perspective from light position along direction
                let target = pos + dir.normalize();
                let v = Mat4::look_at_rh(pos, target, up);
                let p = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1, range);
                p * v
            };

            matrices[slot] = GpuShadowMatrix {
                view_proj:       view_proj.to_cols_array_2d(),
                light_dir:       dir.normalize().to_array(),
                atlas_layer:     slot as u32,
                atlas_uv_offset: [0.0, 0.0],
                atlas_uv_scale:  1.0,
                _pad:            0.0,
            };
        }
        self.upload_matrices(ctx.queue, &matrices);

        let current_hash = shadow_draw_list_hash(ctx.opaque_draws);

        for (slot_base, _light_idx) in shadow_lights.iter().enumerate() {
            // For now treat each shadow light as a single slot (extend to 6 faces for point lights).
            let slot = slot_base;
            if slot >= self.slot_bundles.len() { break; }

            let needs_rebuild = match &self.slot_bundles[slot] {
                None             => true,
                Some((_, h)) => *h != current_hash,
            };

            if needs_rebuild {
                self.rebuild_slot_bundle(ctx.device, slot, ctx.opaque_draws);
            }

            let bundle = match &self.slot_bundles[slot] { Some((b, _)) => b, None => continue };

            let layer_view = self.atlas.create_view(&wgpu::TextureViewDescriptor {
                dimension:          Some(wgpu::TextureViewDimension::D2),
                base_array_layer:   slot as u32,
                array_layer_count:  Some(1),
                ..Default::default()
            });

            let mut rpass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow_slot"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &layer_view,
                    depth_ops: Some(wgpu::Operations {
                        load:  wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            rpass.execute_bundles(std::iter::once(bundle));
        }
    }
}

fn build_pipeline(device: &wgpu::Device, slot_bgl: &wgpu::BindGroupLayout) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label:  Some("shadow_shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/shadow.wgsl").into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label:                Some("shadow_layout"),
        bind_group_layouts:   &[Some(slot_bgl)],
        immediate_size:       0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label:  Some("shadow"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader, entry_point: Some("vs_main"),
            buffers: &[PackedVertex::vertex_buffer_layout(), PackedVertex::instance_buffer_layout()],
            compilation_options: Default::default(),
        },
        primitive: wgpu::PrimitiveState {
            topology:   wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode:  Some(wgpu::Face::Front), // front-face culling reduces peter-panning
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format:              wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare:       Some(wgpu::CompareFunction::Less),
            stencil:             wgpu::StencilState::default(),
            bias:                wgpu::DepthBiasState {
                constant: 2,
                slope_scale: 2.0,
                clamp: 0.0,
            },
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment:    None,
        multiview_mask:   None, cache: None,
    })
}
