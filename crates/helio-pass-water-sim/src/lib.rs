pub mod pipeline;
pub mod simulation;

use helio_core::graph::{ResourceBuilder, ResourceFormat, ResourceSize};
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};
use wgpu::util::DeviceExt;

/// Simple fullscreen blit: copies a texture to the render target as-is.
const BLIT_WGSL: &str = "
@group(0) @binding(0) var blit_tex:  texture_2d<f32>;
@group(0) @binding(1) var blit_samp: sampler;
struct V { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }
@vertex fn vs(@builtin(vertex_index) vi: u32) -> V {
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    return V(vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0), vec2<f32>(x, y));
}
@fragment fn fs(in: V) -> @location(0) vec4<f32> {
    return textureSample(blit_tex, blit_samp, in.uv);
}
";

const SIM_SIZE: u32 = 256;
const CAUSTICS_SIZE: u32 = 256;
const MAX_DROPS_BUFFERED: usize = 16;

// ---- Mesh helpers ----------------------------------------------------------------

fn make_surface_mesh(device: &wgpu::Device) -> (wgpu::Buffer, wgpu::Buffer, u32) {
    const DETAIL: u32 = 128;
    let n = DETAIL + 1;
    let mut verts: Vec<[f32; 3]> = Vec::with_capacity((n * n) as usize);
    for j in 0..n {
        for i in 0..n {
            let x = i as f32 / DETAIL as f32 * 2.0 - 1.0;
            let y = j as f32 / DETAIL as f32 * 2.0 - 1.0;
            verts.push([x, y, 0.0]);
        }
    }
    let mut indices: Vec<u32> = Vec::with_capacity((DETAIL * DETAIL * 6) as usize);
    for j in 0..DETAIL {
        for i in 0..DETAIL {
            let tl = j * n + i;
            let tr = j * n + (i + 1);
            let bl = (j + 1) * n + i;
            let br = (j + 1) * n + (i + 1);
            indices.extend_from_slice(&[tl, bl, tr, tr, bl, br]);
        }
    }
    let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Water Surface VB"),
        contents: bytemuck::cast_slice(&verts),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Water Surface IB"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    (vbuf, ibuf, indices.len() as u32)
}

fn make_volume_box_mesh(device: &wgpu::Device) -> (wgpu::Buffer, wgpu::Buffer, u32) {
    let verts: Vec<[f32; 3]> = vec![
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, -1.0, 1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0],
        [1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0],
    ];
    let indices: Vec<u32> = vec![
        0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7, 8, 9, 10, 8, 10, 11, 12, 13, 14, 12, 14, 15, 16, 17,
        18, 16, 18, 19,
    ];
    let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Water Volume Box VB"),
        contents: bytemuck::cast_slice(&verts),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Water Volume Box IB"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    (vbuf, ibuf, indices.len() as u32)
}

fn vec3_vbl() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: 12,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x3,
            offset: 0,
            shader_location: 0,
        }],
    }
}

// ---- Pass struct ----------------------------------------------------------------

pub struct WaterSimPass {
    pub(crate) sim_bgl: wgpu::BindGroupLayout,
    pub(crate) hitbox_bgl: wgpu::BindGroupLayout,

    pub(crate) drop_pipeline: wgpu::RenderPipeline,
    pub(crate) update_pipeline: wgpu::RenderPipeline,
    pub(crate) normal_pipeline: wgpu::RenderPipeline,
    pub(crate) hitbox_pipeline: wgpu::RenderPipeline,

    pub(crate) _tex_a: wgpu::Texture,
    pub(crate) _tex_b: wgpu::Texture,
    pub(crate) view_a: wgpu::TextureView,
    pub(crate) view_b: wgpu::TextureView,
    pub(crate) front: bool,

    pub(crate) sampler: wgpu::Sampler,
    pub(crate) output_sampler: wgpu::Sampler,
    pub(crate) depth_sampler: wgpu::Sampler,

    pub(crate) drop_buf: wgpu::Buffer,
    pub(crate) update_buf: wgpu::Buffer,
    pub(crate) normal_buf: wgpu::Buffer,
    pub(crate) hitbox_count_buf: wgpu::Buffer,

    pub(crate) pending_drops: std::collections::VecDeque<simulation::DropUniform>,
    pub(crate) drop_staged: bool,

    pub(crate) surface_vbuf: wgpu::Buffer,
    pub(crate) surface_ibuf: wgpu::Buffer,
    pub(crate) surface_index_count: u32,

    pub(crate) volume_vbuf: wgpu::Buffer,
    pub(crate) volume_ibuf: wgpu::Buffer,
    pub(crate) volume_index_count: u32,

    pub(crate) caustics_sampler: wgpu::Sampler,

    pub(crate) caustics_render_bgl: wgpu::BindGroupLayout,
    pub(crate) render_bgl: wgpu::BindGroupLayout,
    pub(crate) render_bg: Option<wgpu::BindGroup>,
    pub(crate) render_bg_key: Option<(usize, usize, usize, usize)>,
    pub(crate) normal_bg: Option<wgpu::BindGroup>,
    pub(crate) normal_bg_key: Option<usize>,

    pub(crate) hitbox_bg: Option<wgpu::BindGroup>,
    pub(crate) hitbox_bg_key: Option<(usize, usize)>,
    pub(crate) drop_bg: Option<wgpu::BindGroup>,
    pub(crate) drop_bg_key: Option<usize>,
    pub(crate) update_bg: Option<wgpu::BindGroup>,
    pub(crate) update_bg_key: Option<usize>,
    pub(crate) underwater_tint_bg: Option<wgpu::BindGroup>,
    pub(crate) underwater_tint_bg_key: Option<(usize, usize)>,

    pub(crate) caustics_pipeline: wgpu::RenderPipeline,
    pub(crate) surface_above_pipeline: wgpu::RenderPipeline,
    pub(crate) surface_under_pipeline: wgpu::RenderPipeline,
    pub(crate) volume_walls_pipeline: wgpu::RenderPipeline,

    pub(crate) _pre_aa_fallback_tex: wgpu::Texture,
    pub(crate) pre_aa_fallback_view: wgpu::TextureView,

    pub(crate) _gbuffer_fallback_tex: wgpu::Texture,
    pub(crate) gbuffer_fallback_view: wgpu::TextureView,

    pub(crate) _depth_copy_tex: wgpu::Texture,
    pub(crate) depth_copy_view: wgpu::TextureView,

    pub(crate) internal_width: u32,
    pub(crate) internal_height: u32,
    pub(crate) surface_format: wgpu::TextureFormat,
    pub(crate) viewport_buf: wgpu::Buffer,

    pub(crate) blit_bgl: wgpu::BindGroupLayout,
    pub(crate) blit_pipeline: wgpu::RenderPipeline,
    pub(crate) blit_bg: Option<wgpu::BindGroup>,
    pub(crate) blit_bg_key: Option<usize>,

    pub(crate) water_output_view: Option<wgpu::TextureView>,

    pub(crate) caustics_bg_key: Option<(usize, usize)>,
    pub(crate) caustics_bg: Option<wgpu::BindGroup>,

    pub(crate) _tint_scratch_tex: wgpu::Texture,
    pub(crate) tint_scratch_view: wgpu::TextureView,
    pub(crate) underwater_tint_bgl: wgpu::BindGroupLayout,
    pub(crate) underwater_tint_pipeline: wgpu::RenderPipeline,

    pub(crate) wave_spring: f32,
    pub(crate) wave_damping: f32,

    pub(crate) wind_direction: [f32; 2],
    pub(crate) wind_strength: f32,
    pub(crate) wave_scale: f32,
    pub(crate) wave_speed: f32,
    pub(crate) sim_time: f32,
}

// ---- Public API ----------------------------------------------------------------

impl WaterSimPass {
    pub fn set_sim_dynamics(&mut self, spring: f32, damping: f32) {
        self.wave_spring = spring.clamp(0.1, 2.0);
        self.wave_damping = damping.clamp(0.0, 1.0);
    }

    pub fn set_wind(&mut self, direction: [f32; 2], strength: f32) {
        let len = (direction[0] * direction[0] + direction[1] * direction[1]).sqrt();
        self.wind_direction = if len > 1e-6 {
            [direction[0] / len, direction[1] / len]
        } else {
            [0.0, 0.0]
        };
        self.wind_strength = strength.max(0.0);
    }

    pub fn set_wave_scale(&mut self, scale: f32) {
        self.wave_scale = scale.max(0.01);
    }

    pub fn set_wave_speed(&mut self, speed: f32) {
        self.wave_speed = speed.max(0.0);
    }

    pub fn add_drop(&mut self, center_x: f32, center_z: f32, radius: f32, strength: f32) {
        if self.pending_drops.len() < MAX_DROPS_BUFFERED {
            self.pending_drops.push_back(simulation::DropUniform {
                center: [center_x, center_z],
                radius,
                strength,
            });
        }
    }

    pub fn resize_internal(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let depth_copy_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Water Depth Copy"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.depth_copy_view = depth_copy_tex.create_view(&wgpu::TextureViewDescriptor::default());
        self._depth_copy_tex = depth_copy_tex;

        let tint_scratch_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Water Tint Scratch"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.surface_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.tint_scratch_view =
            tint_scratch_tex.create_view(&wgpu::TextureViewDescriptor::default());
        self._tint_scratch_tex = tint_scratch_tex;

        self.viewport_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Water Viewport"),
            contents: bytemuck::cast_slice(&[
                width as f32,
                height as f32,
                1.0 / width as f32,
                1.0 / height as f32,
            ]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        self.internal_width = width;
        self.internal_height = height;

        self.water_output_view = None;
        self.render_bg = None;
        self.render_bg_key = None;
        self.blit_bg = None;
        self.blit_bg_key = None;
    }
}

// ---- RenderPass impl ----------------------------------------------------------------

impl RenderPass for WaterSimPass {
    fn name(&self) -> &'static str {
        "WaterSim"
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn on_resize(&mut self, _device: &wgpu::Device, _width: u32, _height: u32) {}

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.read("pre_aa");
        builder.write_color(
            "water_output",
            ResourceFormat::from(self.surface_format),
            ResourceSize::MatchSurface,
        );
        builder.write_color_raw(
            "water_caustics",
            wgpu::TextureFormat::Rgba16Float,
            ResourceSize::Absolute {
                width: CAUSTICS_SIZE,
                height: CAUSTICS_SIZE,
            },
        );
    }

    fn reads(&self) -> &'static [&'static str] {
        &[
            "gbuffer",
            "depth",
            "pre_aa",
            "depth_texture",
            "water_hitbox_count",
            "water_hitboxes",
            "water_volume_count",
            "water_volumes",
            "water_caustics",
        ]
    }
    fn writes(&self) -> &'static [&'static str] {
        &[
            "water_sim_texture",
            "water_sim_sampler",
            "water_caustics",
            "pre_aa",
        ]
    }

    fn publish<'a>(&'a self, frame: &mut libhelio::FrameResources<'a>) {
        let view = if self.front {
            &self.view_a
        } else {
            &self.view_b
        };
        frame.water_sim_texture.write(view, "WaterSim");
        frame
            .water_sim_sampler
            .write(&self.output_sampler, "WaterSim");
        if let Some(view) = &self.water_output_view {
            frame.pre_aa.write(view, "WaterSim");
        }
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        self.sim_time += self.wave_speed / 60.0;
        let step_dt = 1.0 / 120.0;
        let delta = simulation::DeltaUniform {
            delta: [1.0 / SIM_SIZE as f32, 1.0 / SIM_SIZE as f32],
            spring: self.wave_spring,
            damping: self.wave_damping,
            wind_dir: self.wind_direction,
            wind_strength: self.wind_strength,
            time: self.sim_time,
            wave_scale: self.wave_scale,
            time_step: step_dt,
        };
        ctx.write_buffer(&self.update_buf, 0, bytemuck::bytes_of(&delta));
        ctx.write_buffer(&self.normal_buf, 0, bytemuck::bytes_of(&delta));

        let count = ctx.frame_resources.water_hitbox_count;
        ctx.write_buffer(
            &self.hitbox_count_buf,
            0,
            bytemuck::bytes_of(&simulation::HitboxCountUniform {
                count,
                _pad: [0; 3],
            }),
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
        if ctx.resources.water_hitbox_count > 0 {
            if let Some(hitboxes_buf) = ctx.resources.water_hitboxes.get() {
                let src: &wgpu::TextureView = if self.front {
                    &self.view_a
                } else {
                    &self.view_b
                };
                let dst_ptr: *const wgpu::TextureView = if self.front {
                    &self.view_b
                } else {
                    &self.view_a
                };

                let src_key = src as *const wgpu::TextureView as usize;
                let hitboxes_key = hitboxes_buf as *const wgpu::Buffer as usize;
                let new_key = (src_key, hitboxes_key);
                if self.hitbox_bg_key != Some(new_key) {
                    self.hitbox_bg =
                        Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                        }));
                    self.hitbox_bg_key = Some(new_key);
                }
                let bg = self.hitbox_bg.as_ref().unwrap();

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
                pass.set_bind_group(0, bg, &[]);
                pass.draw(0..6, 0..1);
                drop(pass);
                self.front = !self.front;
            }
        }

        // ---- 2. Drop ripple --------------------------------------------------
        if self.drop_staged {
            let src: &wgpu::TextureView = if self.front {
                &self.view_a
            } else {
                &self.view_b
            };
            let dst_ptr: *const wgpu::TextureView = if self.front {
                &self.view_b
            } else {
                &self.view_a
            };

            let src_key = src as *const wgpu::TextureView as usize;
            if self.drop_bg_key != Some(src_key) {
                self.drop_bg = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                }));
                self.drop_bg_key = Some(src_key);
            }
            let bg = self.drop_bg.as_ref().unwrap();

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
            pass.set_bind_group(0, bg, &[]);
            pass.draw(0..6, 0..1);
            drop(pass);
            self.front = !self.front;
        }

        // ---- 3. Wave propagation (2 steps per frame) ------------------------
        for i in 0..2u32 {
            let src: &wgpu::TextureView = if self.front {
                &self.view_a
            } else {
                &self.view_b
            };
            let dst_ptr: *const wgpu::TextureView = if self.front {
                &self.view_b
            } else {
                &self.view_a
            };

            let src_key = src as *const wgpu::TextureView as usize;
            if self.update_bg_key != Some(src_key) {
                self.update_bg = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                }));
                self.update_bg_key = Some(src_key);
            }
            let bg = self.update_bg.as_ref().unwrap();

            let dst = unsafe { &*dst_ptr };
            let label = if i == 0 {
                "WaterSim Update 1"
            } else {
                "WaterSim Update 2"
            };
            let color_attachments = [Some(wgpu::RenderPassColorAttachment {
                view: dst,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })];
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
            pass.set_bind_group(0, bg, &[]);
            pass.draw(0..6, 0..1);
            drop(pass);
            self.front = !self.front;
        }

        // ---- 4. Normal recomputation ----------------------------------------
        {
            let src: &wgpu::TextureView = if self.front {
                &self.view_a
            } else {
                &self.view_b
            };
            let dst_ptr: *const wgpu::TextureView = if self.front {
                &self.view_b
            } else {
                &self.view_a
            };

            let src_key = src as *const wgpu::TextureView as usize;
            if self.normal_bg_key != Some(src_key) {
                self.normal_bg = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                }));
                self.normal_bg_key = Some(src_key);
            }
            let bg = self.normal_bg.as_ref().unwrap();

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
            pass.set_bind_group(0, bg, &[]);
            pass.draw(0..6, 0..1);
            drop(pass);
            self.front = !self.front;
        }

        // ---- 5. Caustics projection ------------------------------------------
        if ctx.resources.water_volume_count > 0 {
            if let Some(vols_buf) = ctx.resources.water_volumes.get() {
                let sim_view = if self.front {
                    &self.view_a
                } else {
                    &self.view_b
                };

                let vols_key = vols_buf as *const wgpu::Buffer as usize;
                let sim_key = sim_view as *const wgpu::TextureView as usize;
                let new_key = (vols_key, sim_key);

                if self.caustics_bg_key != Some(new_key) {
                    self.caustics_bg =
                        Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("Water Caustics BG"),
                            layout: &self.caustics_render_bgl,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: vols_buf.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::TextureView(sim_view),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: wgpu::BindingResource::Sampler(&self.output_sampler),
                                },
                            ],
                        }));
                    self.caustics_bg_key = Some(new_key);
                }

                let caustics_view = ctx.resources.water_caustics.read("WaterSim").unwrap();
                let cau_attachments = [Some(wgpu::RenderPassColorAttachment {
                    view: caustics_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })];
                let desc = wgpu::RenderPassDescriptor {
                    label: Some("Water Caustics"),
                    color_attachments: &cau_attachments,
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                };
                let mut pass = ctx.begin_render_pass(&desc);
                pass.set_pipeline(&self.caustics_pipeline);
                pass.set_bind_group(0, self.caustics_bg.as_ref().unwrap(), &[]);
                pass.set_vertex_buffer(0, self.surface_vbuf.slice(..));
                pass.set_index_buffer(self.surface_ibuf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..self.surface_index_count, 0, 0..1);
                drop(pass);
            }
        }

        // ---- 6. Blit pre_aa -> water_output (scene baseline) -----------------
        if self.water_output_view.is_none() {
            self.water_output_view = ctx.resource_pool.get_view("water_output").cloned();
        }
        let water_output_view = self
            .water_output_view
            .as_ref()
            .expect("water_output view from graph");
        let scene_view: &wgpu::TextureView = ctx
            .resources
            .pre_aa
            .get()
            .unwrap_or(&self.pre_aa_fallback_view);
        let blit_key = scene_view as *const _ as usize;
        if self.blit_bg_key != Some(blit_key) {
            self.blit_bg = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Water Blit BG"),
                layout: &self.blit_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(scene_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.output_sampler),
                    },
                ],
            }));
            self.blit_bg_key = Some(blit_key);
        }
        {
            let attachments = [Some(wgpu::RenderPassColorAttachment {
                view: water_output_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })];
            let desc = wgpu::RenderPassDescriptor {
                label: Some("Water Blit"),
                color_attachments: &attachments,
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            };
            let mut pass = ctx.begin_render_pass(&desc);
            pass.set_pipeline(&self.blit_pipeline);
            pass.set_bind_group(0, self.blit_bg.as_ref().unwrap(), &[]);
            pass.draw(0..3, 0..1);
        }

        // ---- 7. Water surface render -> water_output --------------------------
        if ctx.resources.water_volume_count > 0 {
            if let Some(vols_buf) = ctx.resources.water_volumes.get() {
                let sim_view = if self.front {
                    &self.view_a
                } else {
                    &self.view_b
                };

                let src_depth_tex = ctx.resources.depth_texture.get().ok_or_else(|| {
                    helio_core::Error::InvalidPassConfig(
                        "Water SSR requires depth_texture in FrameResources".to_string(),
                    )
                })?;
                unsafe { &mut *ctx.encoder_ptr }.copy_texture_to_texture(
                    src_depth_tex.as_image_copy(),
                    self._depth_copy_tex.as_image_copy(),
                    wgpu::Extent3d {
                        width: self.internal_width,
                        height: self.internal_height,
                        depth_or_array_layers: 1,
                    },
                );

                let gbuffer_normal_view = ctx
                    .resources
                    .gbuffer
                    .get()
                    .map(|gb| gb.normal)
                    .unwrap_or(&self.gbuffer_fallback_view);

                let scene_key = scene_view as *const wgpu::TextureView as usize;
                let gbuffer_key = gbuffer_normal_view as *const wgpu::TextureView as usize;
                let new_key = (
                    vols_buf as *const wgpu::Buffer as usize,
                    sim_view as *const wgpu::TextureView as usize,
                    scene_key,
                    gbuffer_key,
                );
                if self.render_bg_key != Some(new_key) {
                    self.render_bg =
                        Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("Water Render BG"),
                            layout: &self.render_bgl,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: ctx.scene.camera.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: vols_buf.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: wgpu::BindingResource::TextureView(sim_view),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 3,
                                    resource: wgpu::BindingResource::Sampler(&self.output_sampler),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 4,
                                    resource: wgpu::BindingResource::TextureView(
                                        ctx.resources.water_caustics.read("WaterSim").unwrap(),
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 5,
                                    resource: wgpu::BindingResource::Sampler(
                                        &self.caustics_sampler,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 6,
                                    resource: wgpu::BindingResource::TextureView(scene_view),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 7,
                                    resource: self.viewport_buf.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 8,
                                    resource: wgpu::BindingResource::TextureView(
                                        &self.depth_copy_view,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 9,
                                    resource: wgpu::BindingResource::Sampler(&self.depth_sampler),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 10,
                                    resource: wgpu::BindingResource::TextureView(
                                        gbuffer_normal_view,
                                    ),
                                },
                            ],
                        }));
                    self.render_bg_key = Some(new_key);
                }
                let render_bg = self.render_bg.as_ref().unwrap();

                let depth_view = ctx.depth;
                let color_attachments = [Some(wgpu::RenderPassColorAttachment {
                    view: water_output_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })];

                // 1. Water volume walls
                {
                    let mut pass = unsafe { &mut *ctx.encoder_ptr }.begin_render_pass(
                        &wgpu::RenderPassDescriptor {
                            label: Some("Water Volume Walls"),
                            color_attachments: &color_attachments,
                            depth_stencil_attachment: Some(
                                wgpu::RenderPassDepthStencilAttachment {
                                    view: depth_view,
                                    depth_ops: Some(wgpu::Operations {
                                        load: wgpu::LoadOp::Load,
                                        store: wgpu::StoreOp::Store,
                                    }),
                                    stencil_ops: None,
                                },
                            ),
                            timestamp_writes: None,
                            occlusion_query_set: None,
                            multiview_mask: None,
                        },
                    );
                    pass.set_pipeline(&self.volume_walls_pipeline);
                    pass.set_bind_group(0, render_bg, &[]);
                    pass.set_vertex_buffer(0, self.volume_vbuf.slice(..));
                    pass.set_index_buffer(self.volume_ibuf.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..self.volume_index_count, 0, 0..1);
                }

                // 2. Water surface above
                {
                    let mut pass = unsafe { &mut *ctx.encoder_ptr }.begin_render_pass(
                        &wgpu::RenderPassDescriptor {
                            label: Some("Water Surface Above"),
                            color_attachments: &color_attachments,
                            depth_stencil_attachment: Some(
                                wgpu::RenderPassDepthStencilAttachment {
                                    view: depth_view,
                                    depth_ops: Some(wgpu::Operations {
                                        load: wgpu::LoadOp::Load,
                                        store: wgpu::StoreOp::Store,
                                    }),
                                    stencil_ops: None,
                                },
                            ),
                            timestamp_writes: None,
                            occlusion_query_set: None,
                            multiview_mask: None,
                        },
                    );
                    pass.set_pipeline(&self.surface_above_pipeline);
                    pass.set_bind_group(0, render_bg, &[]);
                    pass.set_vertex_buffer(0, self.surface_vbuf.slice(..));
                    pass.set_index_buffer(self.surface_ibuf.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..self.surface_index_count, 0, 0..1);
                }

                // 3. Water surface under
                {
                    let mut pass = unsafe { &mut *ctx.encoder_ptr }.begin_render_pass(
                        &wgpu::RenderPassDescriptor {
                            label: Some("Water Surface Under"),
                            color_attachments: &color_attachments,
                            depth_stencil_attachment: Some(
                                wgpu::RenderPassDepthStencilAttachment {
                                    view: depth_view,
                                    depth_ops: Some(wgpu::Operations {
                                        load: wgpu::LoadOp::Load,
                                        store: wgpu::StoreOp::Store,
                                    }),
                                    stencil_ops: None,
                                },
                            ),
                            timestamp_writes: None,
                            occlusion_query_set: None,
                            multiview_mask: None,
                        },
                    );
                    pass.set_pipeline(&self.surface_under_pipeline);
                    pass.set_bind_group(0, render_bg, &[]);
                    pass.set_vertex_buffer(0, self.surface_vbuf.slice(..));
                    pass.set_index_buffer(self.surface_ibuf.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..self.surface_index_count, 0, 0..1);
                }

                // 4. Underwater effect
                {
                    let vols_key = vols_buf as *const wgpu::Buffer as usize;
                    let water_output_key = water_output_view as *const wgpu::TextureView as usize;
                    let new_tint_key = (vols_key, water_output_key);
                    if self.underwater_tint_bg_key != Some(new_tint_key) {
                        self.underwater_tint_bg =
                            Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("Water Underwater Tint BG"),
                                layout: &self.underwater_tint_bgl,
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: ctx.scene.camera.as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource: vols_buf.as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 2,
                                        resource: wgpu::BindingResource::TextureView(
                                            water_output_view,
                                        ),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 3,
                                        resource: wgpu::BindingResource::Sampler(
                                            &self.depth_sampler,
                                        ),
                                    },
                                ],
                            }));
                        self.underwater_tint_bg_key = Some(new_tint_key);
                    }
                    let tint_bg = self.underwater_tint_bg.as_ref().unwrap();
                    let tint_attachments = [Some(wgpu::RenderPassColorAttachment {
                        view: &self.tint_scratch_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })];
                    let mut tint_pass = unsafe { &mut *ctx.encoder_ptr }.begin_render_pass(
                        &wgpu::RenderPassDescriptor {
                            label: Some("Water Underwater Tint"),
                            color_attachments: &tint_attachments,
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                            multiview_mask: None,
                        },
                    );
                    tint_pass.set_pipeline(&self.underwater_tint_pipeline);
                    tint_pass.set_bind_group(0, tint_bg, &[]);
                    tint_pass.draw(0..3, 0..1);
                    drop(tint_pass);

                    let scratch_blit_bg =
                        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("Water Tint Blit BG"),
                            layout: &self.blit_bgl,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::TextureView(
                                        &self.tint_scratch_view,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::Sampler(&self.output_sampler),
                                },
                            ],
                        });
                    let blit_attachments = [Some(wgpu::RenderPassColorAttachment {
                        view: water_output_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })];
                    let mut blit_pass = unsafe { &mut *ctx.encoder_ptr }.begin_render_pass(
                        &wgpu::RenderPassDescriptor {
                            label: Some("Water Tint Blit Back"),
                            color_attachments: &blit_attachments,
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                            multiview_mask: None,
                        },
                    );
                    blit_pass.set_pipeline(&self.blit_pipeline);
                    blit_pass.set_bind_group(0, &scratch_blit_bg, &[]);
                    blit_pass.draw(0..3, 0..1);
                }
            }
        }

        Ok(())
    }
}
