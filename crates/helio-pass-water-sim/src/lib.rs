//! Water heightfield simulation pass.
//!
//! Runs a shallow-water wave simulation on a 256x256 `Rgba16Float` ping-pong texture pair.
//! Each frame the pass executes in this order:
//!
//!   1. (optional) AABB hitbox displacement -- water rises/falls where rigid bodies move
//!   2. (optional) Cosine-falloff drop ripple -- queued via [`WaterSimPass::add_drop`]
//!   3. 2x wave-propagation update steps (shallow-water equation)
//!   4. Normal recomputation from height gradient
//!
//! Publishes `frame.water_sim_texture` and `frame.water_sim_sampler` for downstream
//! passes (caustics, surface, underwater).

use bytemuck::{Pod, Zeroable};
use helio_v3::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

const SIM_SIZE: u32 = 256;
const MAX_DROPS_BUFFERED: usize = 16;

// ---- GPU uniform structs --------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DropUniform {
    center: [f32; 2],
    radius: f32,
    strength: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DeltaUniform {
    delta: [f32; 2],
    _pad: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HitboxCountUniform {
    count: u32,
    _pad: [u32; 3],
}

// ---- Pass struct ----------------------------------------------------------------

pub struct WaterSimPass {
    sim_bgl: wgpu::BindGroupLayout,
    hitbox_bgl: wgpu::BindGroupLayout,

    drop_pipeline: wgpu::RenderPipeline,
    update_pipeline: wgpu::RenderPipeline,
    normal_pipeline: wgpu::RenderPipeline,
    hitbox_pipeline: wgpu::RenderPipeline,

    _tex_a: wgpu::Texture,
    _tex_b: wgpu::Texture,
    view_a: wgpu::TextureView,
    view_b: wgpu::TextureView,
    front: bool,

    sampler: wgpu::Sampler,
    output_sampler: wgpu::Sampler,

    drop_buf: wgpu::Buffer,
    update_buf: wgpu::Buffer,
    normal_buf: wgpu::Buffer,
    hitbox_count_buf: wgpu::Buffer,

    pending_drops: std::collections::VecDeque<DropUniform>,
    drop_staged: bool,
}

impl WaterSimPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let vert = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("WaterSim VS"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/fullscreen.vert.wgsl").into(),
            ),
        });
        let drop_frag = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("WaterSim Drop FS"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/drop.frag.wgsl").into()),
        });
        let update_frag = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("WaterSim Update FS"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/update.frag.wgsl").into()),
        });
        let normal_frag = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("WaterSim Normal FS"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/normal.frag.wgsl").into()),
        });
        let hitbox_frag = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("WaterSim Hitbox FS"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/hitbox.frag.wgsl").into()),
        });

        let sim_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("WaterSim BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let hitbox_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("WaterSim Hitbox BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let sim_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("WaterSim PL"),
            bind_group_layouts: &[Some(&sim_bgl)],
            immediate_size: 0,
        });
        let hitbox_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("WaterSim Hitbox PL"),
            bind_group_layouts: &[Some(&hitbox_bgl)],
            immediate_size: 0,
        });

        let make_pipeline = |label, layout: &wgpu::PipelineLayout, frag: &wgpu::ShaderModule| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: &vert,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: frag,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };

        let drop_pipeline   = make_pipeline("WaterSim Drop",   &sim_pl,    &drop_frag);
        let update_pipeline = make_pipeline("WaterSim Update", &sim_pl,    &update_frag);
        let normal_pipeline = make_pipeline("WaterSim Normal", &sim_pl,    &normal_frag);
        let hitbox_pipeline = make_pipeline("WaterSim Hitbox", &hitbox_pl, &hitbox_frag);

        let make_sim_tex = |label| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: SIM_SIZE,
                    height: SIM_SIZE,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };

        let tex_a = make_sim_tex("WaterSim Tex A");
        let tex_b = make_sim_tex("WaterSim Tex B");
        let view_a = tex_a.create_view(&wgpu::TextureViewDescriptor::default());
        let view_b = tex_b.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("WaterSim Internal Sampler"),
            min_filter: wgpu::FilterMode::Linear,
            mag_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let output_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("WaterSim Output Sampler"),
            min_filter: wgpu::FilterMode::Linear,
            mag_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            ..Default::default()
        });

        let make_ubuf = |label, size: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        let drop_buf         = make_ubuf("WaterSim Drop Uniform",  std::mem::size_of::<DropUniform>());
        let update_buf       = make_ubuf("WaterSim Update Uniform", std::mem::size_of::<DeltaUniform>());
        let normal_buf       = make_ubuf("WaterSim Normal Uniform", std::mem::size_of::<DeltaUniform>());
        let hitbox_count_buf = make_ubuf("WaterSim Hitbox Count",   std::mem::size_of::<HitboxCountUniform>());

        Self {
            sim_bgl,
            hitbox_bgl,
            drop_pipeline,
            update_pipeline,
            normal_pipeline,
            hitbox_pipeline,
            _tex_a: tex_a,
            _tex_b: tex_b,
            view_a,
            view_b,
            front: true,
            sampler,
            output_sampler,
            drop_buf,
            update_buf,
            normal_buf,
            hitbox_count_buf,
            pending_drops: std::collections::VecDeque::new(),
            drop_staged: false,
        }
    }

    /// Queue a water-drop ripple to be applied next frame.
    ///
    /// `center_x`, `center_z` are in [-1, 1] sim-texture space.
    /// `radius` is a fraction of texture space (e.g. 0.05).
    /// `strength` is the height increment at the drop centre.
    pub fn add_drop(&mut self, center_x: f32, center_z: f32, radius: f32, strength: f32) {
        if self.pending_drops.len() < MAX_DROPS_BUFFERED {
            self.pending_drops.push_back(DropUniform {
                center: [center_x, center_z],
                radius,
                strength,
            });
        }
    }
}

impl RenderPass for WaterSimPass {
    fn name(&self) -> &'static str {
        "WaterSim"
    }

    fn publish<'a>(&'a self, frame: &mut libhelio::FrameResources<'a>) {
        let view = if self.front { &self.view_a } else { &self.view_b };
        frame.water_sim_texture = Some(view);
        frame.water_sim_sampler = Some(&self.output_sampler);
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let delta = DeltaUniform {
            delta: [1.0 / SIM_SIZE as f32, 1.0 / SIM_SIZE as f32],
            _pad: [0.0; 2],
        };
        ctx.write_buffer(&self.update_buf, 0, bytemuck::bytes_of(&delta));
        ctx.write_buffer(&self.normal_buf, 0, bytemuck::bytes_of(&delta));

        let count = ctx.frame_resources.water_hitbox_count;
        ctx.write_buffer(
            &self.hitbox_count_buf,
            0,
            bytemuck::bytes_of(&HitboxCountUniform { count, _pad: [0; 3] }),
        );

        self.drop_staged = false;
        if let Some(drop) = self.pending_drops.pop_front() {
            ctx.write_buffer(&self.drop_buf, 0, bytemuck::bytes_of(&drop));
            self.drop_staged = true;
        }

        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        // ---- 1. Hitbox displacement ------------------------------------------
        if ctx.frame.water_hitbox_count > 0 {
            if let Some(hitboxes_buf) = ctx.frame.water_hitboxes {
                // SAFETY: view_a and view_b are separate, non-overlapping wgpu
                // TextureView allocations. We always render FROM src (bind group)
                // INTO dst (colour attachment) -- never the same texture for both.
                // The raw pointer is valid for the lifetime of self and is only
                // dereferenced while no safe borrow of the same field is live.
                let src: &wgpu::TextureView =
                    if self.front { &self.view_a } else { &self.view_b };
                let dst_ptr: *const wgpu::TextureView =
                    if self.front { &self.view_b } else { &self.view_a };

                let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("WaterSim Hitbox BG"),
                    layout: &self.hitbox_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(src),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.hitbox_count_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: hitboxes_buf.as_entire_binding(),
                        },
                    ],
                });

                let dst = unsafe { &*dst_ptr };
                let color_attachments = [Some(wgpu::RenderPassColorAttachment {
                    view: dst,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })];
                {
                    let desc = wgpu::RenderPassDescriptor {
                        label: Some("WaterSim Hitbox"),
                        color_attachments: &color_attachments,
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    };
                    let mut pass = ctx.begin_render_pass(&desc);
                    pass.set_pipeline(&self.hitbox_pipeline);
                    pass.set_bind_group(0, &bg, &[]);
                    pass.draw(0..6, 0..1);
                }
                self.front = !self.front;
            }
        }

        // ---- 2. Drop ripple --------------------------------------------------
        if self.drop_staged {
            // SAFETY: same as hitbox block above.
            let src: &wgpu::TextureView =
                if self.front { &self.view_a } else { &self.view_b };
            let dst_ptr: *const wgpu::TextureView =
                if self.front { &self.view_b } else { &self.view_a };

            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("WaterSim Drop BG"),
                layout: &self.sim_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(src),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.drop_buf.as_entire_binding(),
                    },
                ],
            });

            let dst = unsafe { &*dst_ptr };
            let color_attachments = [Some(wgpu::RenderPassColorAttachment {
                view: dst,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })];
            {
                let desc = wgpu::RenderPassDescriptor {
                    label: Some("WaterSim Drop"),
                    color_attachments: &color_attachments,
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                };
                let mut pass = ctx.begin_render_pass(&desc);
                pass.set_pipeline(&self.drop_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.draw(0..6, 0..1);
            }
            self.front = !self.front;
        }

        // ---- 3. Wave propagation (2 steps per frame for numerical stability) -
        for i in 0..2u32 {
            // SAFETY: same as hitbox block above.
            let src: &wgpu::TextureView =
                if self.front { &self.view_a } else { &self.view_b };
            let dst_ptr: *const wgpu::TextureView =
                if self.front { &self.view_b } else { &self.view_a };

            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("WaterSim Update BG"),
                layout: &self.sim_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(src),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.update_buf.as_entire_binding(),
                    },
                ],
            });

            let dst = unsafe { &*dst_ptr };
            let label = if i == 0 { "WaterSim Update 1" } else { "WaterSim Update 2" };
            let color_attachments = [Some(wgpu::RenderPassColorAttachment {
                view: dst,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })];
            {
                let desc = wgpu::RenderPassDescriptor {
                    label: Some(label),
                    color_attachments: &color_attachments,
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                };
                let mut pass = ctx.begin_render_pass(&desc);
                pass.set_pipeline(&self.update_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.draw(0..6, 0..1);
            }
            self.front = !self.front;
        }

        // ---- 4. Normal recomputation ----------------------------------------
        {
            // SAFETY: same as hitbox block above.
            let src: &wgpu::TextureView =
                if self.front { &self.view_a } else { &self.view_b };
            let dst_ptr: *const wgpu::TextureView =
                if self.front { &self.view_b } else { &self.view_a };

            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("WaterSim Normal BG"),
                layout: &self.sim_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(src),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.normal_buf.as_entire_binding(),
                    },
                ],
            });

            let dst = unsafe { &*dst_ptr };
            let color_attachments = [Some(wgpu::RenderPassColorAttachment {
                view: dst,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })];
            {
                let desc = wgpu::RenderPassDescriptor {
                    label: Some("WaterSim Normal"),
                    color_attachments: &color_attachments,
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                };
                let mut pass = ctx.begin_render_pass(&desc);
                pass.set_pipeline(&self.normal_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.draw(0..6, 0..1);
            }
            self.front = !self.front;
        }

        Ok(())
    }
}
