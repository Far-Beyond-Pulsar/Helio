use std::sync::{Arc, Mutex};

use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

use super::renderer_impl::{DebugBatch, DebugVertex, Renderer};

pub struct DebugDrawState {
    pub editor_enabled: bool,
    pub camera_position: glam::Vec3,
    pub user_lines: Vec<DebugVertex>,
    pub user_lines_generation: u64,
    pub user_tris: Vec<DebugVertex>,
    pub user_tris_generation: u64,
}

impl Default for DebugDrawState {
    fn default() -> Self {
        Self {
            editor_enabled: false,
            camera_position: glam::Vec3::ZERO,
            user_lines: Vec::new(),
            user_lines_generation: 0,
            user_tris: Vec::new(),
            user_tris_generation: 0,
        }
    }
}

const MAX_DEBUG_VERTS: u32 = 65536;
const MAX_DEBUG_TRIS: u32 = 65536;

pub struct DebugPass {
    pipeline_depth: wgpu::RenderPipeline,
    pipeline_no_depth: wgpu::RenderPipeline,
    pipeline_tri_depth: wgpu::RenderPipeline,
    pipeline_tri_no_depth: wgpu::RenderPipeline,
    #[allow(dead_code)]
    bgl: wgpu::BindGroupLayout,
    camera_buf: wgpu::Buffer,
    bind_group: Option<wgpu::BindGroup>,
    bind_group_key: Option<usize>,
    vertex_buf: wgpu::Buffer,
    pub vertex_count: u32,
    tri_buf: wgpu::Buffer,
    pub tri_count: u32,
    depth_test_enabled: bool,
}

impl DebugPass {
    pub fn new(
        device: &wgpu::Device,
        camera_buf: &wgpu::Buffer,
        target_format: wgpu::TextureFormat,
        depth_test: bool,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Debug Draw Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/debug_draw.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Debug Draw BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Debug Draw PL"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Debug Vertex Buffer"),
            size: (MAX_DEBUG_VERTS as usize * std::mem::size_of::<DebugVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline_depth = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Debug Draw Pipeline Depth"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<DebugVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 16,
                            shader_location: 1,
                        },
                    ],
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let pipeline_no_depth = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Debug Draw Pipeline NoDepth"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<DebugVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 16,
                            shader_location: 1,
                        },
                    ],
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let tri_attribs = [
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 16,
                shader_location: 1,
            },
        ];
        let tri_vbl = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<DebugVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &tri_attribs,
        };
        let tri_target = wgpu::ColorTargetState {
            format: target_format,
            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
            write_mask: wgpu::ColorWrites::ALL,
        };
        let tri_prim = wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..Default::default()
        };

        let pipeline_tri_depth = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Debug Tri Pipeline Depth"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(tri_vbl.clone())],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(tri_target.clone())],
            }),
            primitive: tri_prim,
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let pipeline_tri_no_depth =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Debug Tri Pipeline NoDepth"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[Some(tri_vbl)],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(tri_target)],
                }),
                primitive: tri_prim,
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

        let tri_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Debug Tri Buffer"),
            size: (MAX_DEBUG_TRIS as usize * std::mem::size_of::<DebugVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline_depth,
            pipeline_no_depth,
            pipeline_tri_depth,
            pipeline_tri_no_depth,
            bgl,
            camera_buf: camera_buf.clone(),
            bind_group: None,
            bind_group_key: None,
            vertex_buf,
            vertex_count: 0,
            tri_buf,
            tri_count: 0,
            depth_test_enabled: depth_test,
        }
    }

    pub fn update_lines(&mut self, queue: &wgpu::Queue, verts: &[DebugVertex]) {
        let count = verts.len().min(MAX_DEBUG_VERTS as usize);
        if count > 0 {
            helio_core::upload::write_buffer(
                queue,
                &self.vertex_buf,
                0,
                bytemuck::cast_slice(&verts[..count]),
            );
        }
        self.vertex_count = count as u32;
    }

    pub fn update_lines_at(
        &mut self,
        queue: &wgpu::Queue,
        first_vertex: usize,
        verts: &[DebugVertex],
    ) {
        if verts.is_empty() || first_vertex >= MAX_DEBUG_VERTS as usize {
            return;
        }
        let max_writable = (MAX_DEBUG_VERTS as usize).saturating_sub(first_vertex);
        let count = verts.len().min(max_writable);
        if count == 0 {
            return;
        }
        let offset_bytes = (first_vertex * std::mem::size_of::<DebugVertex>()) as u64;
        helio_core::upload::write_buffer(
            queue,
            &self.vertex_buf,
            offset_bytes,
            bytemuck::cast_slice(&verts[..count]),
        );
    }

    pub fn set_line_vertex_count(&mut self, count: usize) {
        self.vertex_count = count.min(MAX_DEBUG_VERTS as usize) as u32;
    }

    pub fn update_tris(&mut self, queue: &wgpu::Queue, verts: &[DebugVertex]) {
        let count = verts.len().min(MAX_DEBUG_TRIS as usize);
        if count > 0 {
            helio_core::upload::write_buffer(
                queue,
                &self.tri_buf,
                0,
                bytemuck::cast_slice(&verts[..count]),
            );
        }
        self.tri_count = count as u32;
    }

    pub fn clear(&mut self) {
        self.vertex_count = 0;
        self.tri_count = 0;
    }

    pub fn set_depth_test(&mut self, enabled: bool) {
        self.depth_test_enabled = enabled;
    }
}

impl DebugPass {
    fn draw_commands(&self, rp: &mut wgpu::RenderPass) {
        rp.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);

        if self.vertex_count > 0 {
            if self.depth_test_enabled {
                rp.set_pipeline(&self.pipeline_depth);
            } else {
                rp.set_pipeline(&self.pipeline_no_depth);
            }
            rp.set_vertex_buffer(0, self.vertex_buf.slice(..));
            rp.draw(0..self.vertex_count, 0..1);
        }

        if self.tri_count > 0 {
            if self.depth_test_enabled {
                rp.set_pipeline(&self.pipeline_tri_depth);
            } else {
                rp.set_pipeline(&self.pipeline_tri_no_depth);
            }
            rp.set_vertex_buffer(0, self.tri_buf.slice(..));
            rp.draw(0..self.tri_count, 0..1);
        }
    }

    fn ensure_bind_group(&mut self, device: &wgpu::Device) {
        let camera_key = &self.camera_buf as *const _ as usize;
        if self.bind_group_key != Some(camera_key) {
            self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Debug Draw BG"),
                layout: &self.bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.camera_buf.as_entire_binding(),
                }],
            }));
            self.bind_group_key = Some(camera_key);
        }
    }

    pub fn execute_on_target(
        &mut self,
        ctx: &mut PassContext,
        target: &wgpu::TextureView,
    ) -> HelioResult<()> {
        if self.vertex_count == 0 && self.tri_count == 0 {
            return Ok(());
        }

        self.ensure_bind_group(ctx.device);

        let depth_attachment = if self.depth_test_enabled {
            let depth_view = if let Some(frd) = ctx.resources.full_res_depth.get() {
                frd
            } else {
                ctx.depth
            };
            Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            })
        } else {
            None
        };

        let color_attachment = wgpu::RenderPassColorAttachment {
            view: target,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        };
        let color_attachments = [Some(color_attachment)];
        let desc = wgpu::RenderPassDescriptor {
            label: Some("DebugDraw"),
            color_attachments: &color_attachments,
            depth_stencil_attachment: depth_attachment,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        };

        let mut pass = unsafe { &mut *ctx.encoder_ptr }.begin_render_pass(&desc);
        self.draw_commands(&mut pass);
        Ok(())
    }
}

impl RenderPass for DebugPass {
    fn name(&self) -> &'static str {
        "DebugDraw"
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        target: &'a wgpu::TextureView,
        depth: &'a wgpu::TextureView,
        resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        let depth_attachment = if self.depth_test_enabled {
            let depth_view = if let Some(frd) = resources.full_res_depth.get() {
                frd
            } else {
                depth
            };
            Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            })
        } else {
            None
        };
        let color_attachments: &'a [Option<wgpu::RenderPassColorAttachment<'a>>] =
            Box::leak(Box::new([Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })]));
        Some(wgpu::RenderPassDescriptor {
            label: Some("DebugDraw"),
            color_attachments,
            depth_stencil_attachment: depth_attachment,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    fn prepare(&mut self, _ctx: &PrepareContext) -> HelioResult<()> {
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        if self.vertex_count == 0 && self.tri_count == 0 {
            return Ok(());
        }
        self.ensure_bind_group(ctx.device);
        let rp = unsafe { &mut *ctx.active_render_pass_ptr().unwrap() };
        self.draw_commands(rp);
        Ok(())
    }
}

pub struct DebugDrawPass {
    pass: DebugPass,
    state: Arc<Mutex<DebugDrawState>>,
    editor_mode: bool,
    cached_line_gen: u64,
    cached_tri_gen: u64,
    editor_grid_cache: Vec<DebugVertex>,
    editor_marker_lines: [DebugVertex; 6],
    editor_last_key: Option<(bool, i32, i32, i32)>,
    editor_last_cam: Option<[f32; 3]>,
}

impl DebugDrawPass {
    pub fn new(
        device: &wgpu::Device,
        camera_buf: &wgpu::Buffer,
        surface_format: wgpu::TextureFormat,
        state: Arc<Mutex<DebugDrawState>>,
        depth_test: bool,
        editor_mode: bool,
    ) -> Self {
        Self {
            pass: DebugPass::new(device, camera_buf, surface_format, depth_test),
            state,
            editor_mode,
            cached_line_gen: u64::MAX,
            cached_tri_gen: u64::MAX,
            editor_grid_cache: Vec::new(),
            editor_marker_lines: [DebugVertex {
                position: [0.0, 0.0, 0.0],
                _pad: 0.0,
                color: [0.0, 1.0, 1.0, 1.0],
            }; 6],
            editor_last_key: None,
            editor_last_cam: None,
        }
    }

    pub fn set_depth_test(&mut self, enabled: bool) {
        self.pass.set_depth_test(enabled);
    }

    fn rebuild_editor_grid_cache(&mut self, center_x: f32, center_z: f32, grid_step: f32) {
        self.editor_grid_cache.clear();

        let minor_color: [f32; 4] = [0.25, 0.25, 0.25, 1.0];
        let major_color: [f32; 4] = [0.5, 0.5, 0.5, 1.0];
        let axis_color_x: [f32; 4] = [1.0, 0.2, 0.2, 1.0];
        let axis_color_z: [f32; 4] = [0.2, 1.0, 0.2, 1.0];

        let range: f32 = 40.0;
        let count = (range / grid_step).ceil() as i32;

        for i in -count..=count {
            let x = center_x + i as f32 * grid_step;
            let z = center_z + i as f32 * grid_step;
            let x_color = if i.rem_euclid(5) == 0 {
                major_color
            } else {
                minor_color
            };
            let z_color = if i.rem_euclid(5) == 0 {
                major_color
            } else {
                minor_color
            };

            self.editor_grid_cache.push(DebugVertex {
                position: [x, 0.0, center_z - range],
                _pad: 0.0,
                color: if x.abs() < 0.01 {
                    axis_color_x
                } else {
                    x_color
                },
            });
            self.editor_grid_cache.push(DebugVertex {
                position: [x, 0.0, center_z + range],
                _pad: 0.0,
                color: if x.abs() < 0.01 {
                    axis_color_x
                } else {
                    x_color
                },
            });
            self.editor_grid_cache.push(DebugVertex {
                position: [center_x - range, 0.0, z],
                _pad: 0.0,
                color: if z.abs() < 0.01 {
                    axis_color_z
                } else {
                    z_color
                },
            });
            self.editor_grid_cache.push(DebugVertex {
                position: [center_x + range, 0.0, z],
                _pad: 0.0,
                color: if z.abs() < 0.01 {
                    axis_color_z
                } else {
                    z_color
                },
            });
        }

        let origin_color = [1.0, 1.0, 0.0, 1.0];
        self.editor_grid_cache.push(DebugVertex {
            position: [-3.0, 0.0, 0.0],
            _pad: 0.0,
            color: origin_color,
        });
        self.editor_grid_cache.push(DebugVertex {
            position: [3.0, 0.0, 0.0],
            _pad: 0.0,
            color: origin_color,
        });
        self.editor_grid_cache.push(DebugVertex {
            position: [0.0, 0.0, -3.0],
            _pad: 0.0,
            color: origin_color,
        });
        self.editor_grid_cache.push(DebugVertex {
            position: [0.0, 0.0, 3.0],
            _pad: 0.0,
            color: origin_color,
        });
    }

    fn update_editor_marker(&mut self, cam: glam::Vec3) {
        let camera_marker_color = [0.0, 1.0, 1.0, 1.0];
        self.editor_marker_lines = [
            DebugVertex {
                position: [cam.x - 0.3, cam.y, cam.z],
                _pad: 0.0,
                color: camera_marker_color,
            },
            DebugVertex {
                position: [cam.x + 0.3, cam.y, cam.z],
                _pad: 0.0,
                color: camera_marker_color,
            },
            DebugVertex {
                position: [cam.x, cam.y - 0.3, cam.z],
                _pad: 0.0,
                color: camera_marker_color,
            },
            DebugVertex {
                position: [cam.x, cam.y + 0.3, cam.z],
                _pad: 0.0,
                color: camera_marker_color,
            },
            DebugVertex {
                position: [cam.x, cam.y, cam.z - 0.3],
                _pad: 0.0,
                color: camera_marker_color,
            },
            DebugVertex {
                position: [cam.x, cam.y, cam.z + 0.3],
                _pad: 0.0,
                color: camera_marker_color,
            },
        ];
    }
}

impl RenderPass for DebugDrawPass {
    fn name(&self) -> &'static str {
        "DebugDraw"
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        target: &'a wgpu::TextureView,
        depth: &'a wgpu::TextureView,
        resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        self.pass.render_pass_descriptor(target, depth, resources)
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let state_arc = Arc::clone(&self.state);
        let state = state_arc.lock().unwrap();

        if self.editor_mode {
            let editor_enabled = state.editor_enabled;
            let cam = state.camera_position;
            drop(state);

            if !editor_enabled {
                if self.cached_line_gen != 0 {
                    self.pass.update_lines(ctx.queue, &[]);
                    self.cached_line_gen = 0;
                }
                self.editor_last_key = None;
                self.editor_last_cam = None;
                self.pass.update_tris(ctx.queue, &[]);
                self.cached_tri_gen = 0;
                return Ok(());
            }

            let cam_dist = cam.length();
            let (grid_step, step_index) = if cam_dist < 20.0_f32 {
                (1.0_f32, 0)
            } else if cam_dist < 60.0_f32 {
                (2.0_f32, 1)
            } else if cam_dist < 150.0_f32 {
                (5.0_f32, 2)
            } else if cam_dist < 300.0_f32 {
                (10.0_f32, 3)
            } else {
                (20.0_f32, 4)
            };

            let center_x = (cam.x / grid_step).round() * grid_step;
            let center_z = (cam.z / grid_step).round() * grid_step;
            let key = (
                true,
                step_index,
                (center_x * 1000.0) as i32,
                (center_z * 1000.0) as i32,
            );
            let mut grid_rebuilt = false;
            if self.editor_last_key != Some(key) {
                self.rebuild_editor_grid_cache(center_x, center_z, grid_step);
                self.editor_last_key = Some(key);
                grid_rebuilt = true;
            }

            let cam_arr = [cam.x, cam.y, cam.z];
            if self.editor_last_cam != Some(cam_arr)
                || self.cached_line_gen == u64::MAX
                || grid_rebuilt
            {
                self.update_editor_marker(cam);

                if grid_rebuilt || self.cached_line_gen == u64::MAX {
                    let mut lines = Vec::with_capacity(
                        self.editor_grid_cache.len() + self.editor_marker_lines.len(),
                    );
                    lines.extend_from_slice(&self.editor_grid_cache);
                    lines.extend_from_slice(&self.editor_marker_lines);
                    self.pass.update_lines(ctx.queue, &lines);
                } else {
                    self.pass.update_lines_at(
                        ctx.queue,
                        self.editor_grid_cache.len(),
                        &self.editor_marker_lines,
                    );
                    self.pass.set_line_vertex_count(
                        self.editor_grid_cache.len() + self.editor_marker_lines.len(),
                    );
                }

                self.editor_last_cam = Some(cam_arr);
                self.cached_line_gen = self.cached_line_gen.wrapping_add(1);
            }

            if self.cached_tri_gen != 0 {
                self.pass.update_tris(ctx.queue, &[]);
                self.cached_tri_gen = 0;
            }
            return Ok(());
        }

        let user_lines_generation = state.user_lines_generation;
        let user_tris_generation = state.user_tris_generation;

        if user_lines_generation != self.cached_line_gen {
            self.pass.update_lines(ctx.queue, &state.user_lines);
            self.cached_line_gen = user_lines_generation;
        }
        if user_tris_generation != self.cached_tri_gen {
            self.pass.update_tris(ctx.queue, &state.user_tris);
            self.cached_tri_gen = user_tris_generation;
        }
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let target_view = if self.editor_mode {
            ctx.resources.pre_aa.get().unwrap_or(ctx.target)
        } else {
            ctx.target
        };

        let previous_target = ctx.target;
        ctx.target = target_view;

        let res = self.pass.execute(ctx);

        ctx.target = previous_target;

        res
    }
}

impl Renderer {
    pub fn debug_clear(&mut self) {
        if let Ok(mut s) = self.debug_state.lock() {
            if !s.user_lines.is_empty() {
                s.user_lines_generation = s.user_lines_generation.wrapping_add(1);
            }
            s.user_lines.clear();
            if !s.user_tris.is_empty() {
                s.user_tris_generation = s.user_tris_generation.wrapping_add(1);
            }
            s.user_tris.clear();
        }
    }

    pub fn debug_batch<F>(&mut self, f: F)
    where
        F: FnOnce(&mut DebugBatch<'_>),
    {
        if let Ok(mut s) = self.debug_state.lock() {
            let mut batch = DebugBatch {
                state: &mut s,
                lines_changed: false,
                tris_changed: false,
            };
            f(&mut batch);
            batch.finish();
        }
    }

    pub fn debug_line(&mut self, from: [f32; 3], to: [f32; 3], color: [f32; 4]) {
        if let Ok(mut s) = self.debug_state.lock() {
            s.user_lines.push(DebugVertex {
                position: from,
                _pad: 0.0,
                color,
            });
            s.user_lines.push(DebugVertex {
                position: to,
                _pad: 0.0,
                color,
            });
            s.user_lines_generation = s.user_lines_generation.wrapping_add(1);
        }
    }

    pub fn debug_tri(&mut self, v0: [f32; 3], v1: [f32; 3], v2: [f32; 3], color: [f32; 4]) {
        if let Ok(mut s) = self.debug_state.lock() {
            s.user_tris.push(DebugVertex {
                position: v0,
                _pad: 0.0,
                color,
            });
            s.user_tris.push(DebugVertex {
                position: v1,
                _pad: 0.0,
                color,
            });
            s.user_tris.push(DebugVertex {
                position: v2,
                _pad: 0.0,
                color,
            });
            s.user_tris_generation = s.user_tris_generation.wrapping_add(1);
        }
    }

    pub fn debug_filled_disk(
        &mut self,
        center: [f32; 3],
        normal: [f32; 3],
        radius: f32,
        color: [f32; 4],
        segments: u32,
    ) {
        if segments < 3 {
            return;
        }
        let c = glam::Vec3::from(center);
        let n = glam::Vec3::from(normal).normalize_or_zero();
        let up = if n.abs_diff_eq(glam::Vec3::Y, 1e-5) {
            glam::Vec3::X
        } else {
            glam::Vec3::Y
        };
        let tangent = n.cross(up).normalize_or_zero();
        let bitangent = n.cross(tangent).normalize_or_zero();
        let mut prev = c + tangent * radius;
        for i in 1..=segments {
            let theta = i as f32 / segments as f32 * std::f32::consts::TAU;
            let cur = c + (tangent * theta.cos() + bitangent * theta.sin()) * radius;
            self.debug_tri(c.to_array(), prev.to_array(), cur.to_array(), color);
            prev = cur;
        }
    }

    pub fn debug_filled_cone(
        &mut self,
        apex: [f32; 3],
        axis: [f32; 3],
        height: f32,
        base_radius: f32,
        color: [f32; 4],
        segments: u32,
    ) {
        if segments < 3 {
            return;
        }
        let apex_v = glam::Vec3::from(apex);
        let dir = glam::Vec3::from(axis).normalize_or_zero();
        let base = apex_v + dir * height;
        let up = if dir.cross(glam::Vec3::Y).length_squared() < 1e-8 {
            glam::Vec3::X
        } else {
            glam::Vec3::Y
        };
        let tangent = dir.cross(up).normalize_or_zero();
        let bitangent = dir.cross(tangent).normalize_or_zero();
        let mut prev = base + tangent * base_radius;
        for i in 1..=segments {
            let theta = i as f32 / segments as f32 * std::f32::consts::TAU;
            let cur = base + (tangent * theta.cos() + bitangent * theta.sin()) * base_radius;
            self.debug_tri(apex_v.to_array(), prev.to_array(), cur.to_array(), color);
            self.debug_tri(base.to_array(), cur.to_array(), prev.to_array(), color);
            prev = cur;
        }
    }

    pub fn debug_filled_box(&mut self, center: [f32; 3], half: f32, color: [f32; 4]) {
        let c = glam::Vec3::from(center);
        let h = half;
        let corners = [
            c + glam::Vec3::new(-h, -h, -h),
            c + glam::Vec3::new(h, -h, -h),
            c + glam::Vec3::new(h, h, -h),
            c + glam::Vec3::new(-h, h, -h),
            c + glam::Vec3::new(-h, -h, h),
            c + glam::Vec3::new(h, -h, h),
            c + glam::Vec3::new(h, h, h),
            c + glam::Vec3::new(-h, h, h),
        ];
        let quads: [[usize; 4]; 6] = [
            [0, 3, 2, 1],
            [4, 5, 6, 7],
            [0, 4, 7, 3],
            [1, 2, 6, 5],
            [0, 1, 5, 4],
            [3, 7, 6, 2],
        ];
        for [a, b, cc, d] in quads {
            self.debug_tri(
                corners[a].to_array(),
                corners[b].to_array(),
                corners[cc].to_array(),
                color,
            );
            self.debug_tri(
                corners[a].to_array(),
                corners[cc].to_array(),
                corners[d].to_array(),
                color,
            );
        }
    }

    pub fn debug_circle(&mut self, center: [f32; 3], radius: f32, color: [f32; 4], segments: u32) {
        if segments < 3 {
            return;
        }
        let (cx, cy, cz) = (center[0], center[1], center[2]);
        let step = std::f32::consts::TAU / segments as f32;
        let mut last = (cx + radius, cy, cz);
        for i in 1..=segments {
            let theta = i as f32 * step;
            let next = (cx + radius * theta.cos(), cy, cz + radius * theta.sin());
            self.debug_line([last.0, last.1, last.2], [next.0, next.1, next.2], color);
            last = next;
        }
    }

    pub fn debug_sphere(&mut self, center: [f32; 3], radius: f32, color: [f32; 4], segments: u32) {
        if segments < 4 {
            return;
        }
        for plane in 0..3 {
            let mut prev = glam::Vec3::ZERO;
            for i in 0..=segments {
                let theta = i as f32 / segments as f32 * std::f32::consts::TAU;
                let pos = match plane {
                    0 => glam::Vec3::new(radius * theta.cos(), radius * theta.sin(), 0.0),
                    1 => glam::Vec3::new(radius * theta.cos(), 0.0, radius * theta.sin()),
                    _ => glam::Vec3::new(0.0, radius * theta.cos(), radius * theta.sin()),
                } + glam::Vec3::from(center);
                if i > 0 {
                    self.debug_line(prev.to_array(), pos.to_array(), color);
                }
                prev = pos;
            }
        }
    }

    pub fn debug_torus(
        &mut self,
        center: [f32; 3],
        normal: [f32; 3],
        major_radius: f32,
        minor_radius: f32,
        color: [f32; 4],
        major_segments: u32,
        minor_segments: u32,
    ) {
        if major_segments < 3 || minor_segments < 3 {
            return;
        }
        let c = glam::Vec3::from(center);
        let n = glam::Vec3::from(normal).normalize_or_zero();
        let up = if n.abs_diff_eq(glam::Vec3::Y, 1e-6) {
            glam::Vec3::X
        } else {
            glam::Vec3::Y
        };
        let tangent = n.cross(up).normalize_or_zero();
        let bitangent = n.cross(tangent).normalize_or_zero();

        for j in 0..major_segments {
            let theta0 = 2.0 * std::f32::consts::TAU * (j as f32) / (major_segments as f32);
            let theta1 = 2.0 * std::f32::consts::TAU * ((j + 1) as f32) / (major_segments as f32);
            let center0 = c + (tangent * theta0.cos() + bitangent * theta0.sin()) * major_radius;
            let center1 = c + (tangent * theta1.cos() + bitangent * theta1.sin()) * major_radius;

            let mut pprev0 = center0 + (n * minor_radius);
            let mut pprev1 = center1 + (n * minor_radius);
            for i in 1..=minor_segments {
                let phi = 2.0 * std::f32::consts::TAU * (i as f32) / (minor_segments as f32);
                let offset = (n * phi.cos()
                    + (tangent * theta0.cos() + bitangent * theta0.sin()) * phi.sin())
                .normalize_or_zero()
                    * minor_radius;
                let cur0 = center0 + offset;
                let offset1 = (n * phi.cos()
                    + (tangent * theta1.cos() + bitangent * theta1.sin()) * phi.sin())
                .normalize_or_zero()
                    * minor_radius;
                let cur1 = center1 + offset1;

                self.debug_line(pprev0.to_array(), cur0.to_array(), color);
                self.debug_line(pprev1.to_array(), cur1.to_array(), color);
                self.debug_line(pprev0.to_array(), pprev1.to_array(), color);

                pprev0 = cur0;
                pprev1 = cur1;
            }
        }
    }

    pub fn debug_cylinder(
        &mut self,
        base_center: [f32; 3],
        axis: [f32; 3],
        height: f32,
        radius: f32,
        color: [f32; 4],
        segments: u32,
    ) {
        if segments < 3 {
            return;
        }
        let base = glam::Vec3::from(base_center);
        let dir = glam::Vec3::from(axis).normalize_or_zero();
        let top = base + dir * height;
        let up = if dir.abs_diff_eq(glam::Vec3::Y, 1e-5) {
            glam::Vec3::X
        } else {
            glam::Vec3::Y
        };
        let tangent = dir.cross(up).normalize_or_zero();
        let bitangent = dir.cross(tangent).normalize_or_zero();
        let mut prev_base = base + tangent * radius;
        let mut prev_top = top + tangent * radius;
        for i in 1..=segments {
            let theta = i as f32 / segments as f32 * std::f32::consts::TAU;
            let dir_circle = tangent * theta.cos() + bitangent * theta.sin();
            let cur_base = base + dir_circle * radius;
            let cur_top = top + dir_circle * radius;
            self.debug_line(prev_base.to_array(), cur_base.to_array(), color);
            self.debug_line(prev_top.to_array(), cur_top.to_array(), color);
            self.debug_line(prev_base.to_array(), prev_top.to_array(), color);
            prev_base = cur_base;
            prev_top = cur_top;
        }
    }

    pub fn debug_cone(
        &mut self,
        apex: [f32; 3],
        axis: [f32; 3],
        height: f32,
        base_radius: f32,
        color: [f32; 4],
        segments: u32,
    ) {
        if segments < 3 {
            return;
        }
        let apex_v = glam::Vec3::from(apex);
        let dir = glam::Vec3::from(axis).normalize_or_zero();
        let base = apex_v + dir * height;
        let up = if dir.cross(glam::Vec3::Y).length_squared() < 1e-8 {
            glam::Vec3::X
        } else {
            glam::Vec3::Y
        };
        let tangent = dir.cross(up).normalize_or_zero();
        let bitangent = dir.cross(tangent).normalize_or_zero();
        let mut prev = base + tangent * base_radius;
        for i in 1..=segments {
            let theta = i as f32 / segments as f32 * std::f32::consts::TAU;
            let cur = base + (tangent * theta.cos() + bitangent * theta.sin()) * base_radius;
            self.debug_line(prev.to_array(), cur.to_array(), color);
            self.debug_line(cur.to_array(), apex_v.to_array(), color);
            prev = cur;
        }
    }

    pub fn debug_frustum(
        &mut self,
        origin: [f32; 3],
        forward: [f32; 3],
        up: [f32; 3],
        fov_y: f32,
        aspect: f32,
        near: f32,
        far: f32,
        color: [f32; 4],
    ) {
        let o = glam::Vec3::from(origin);
        let fwd = glam::Vec3::from(forward).normalize_or_zero();
        let upv = glam::Vec3::from(up).normalize_or_zero();
        let rightv = fwd.cross(upv).normalize_or_zero();
        let n_center = o + fwd * near;
        let f_center = o + fwd * far;
        let nh = (fov_y * 0.5).tan() * near;
        let nw = nh * aspect;
        let fh = (fov_y * 0.5).tan() * far;
        let fw = fh * aspect;

        let n = [
            n_center + upv * nh - rightv * nw,
            n_center + upv * nh + rightv * nw,
            n_center - upv * nh + rightv * nw,
            n_center - upv * nh - rightv * nw,
        ];
        let f = [
            f_center + upv * fh - rightv * fw,
            f_center + upv * fh + rightv * fw,
            f_center - upv * fh + rightv * fw,
            f_center - upv * fh - rightv * fw,
        ];

        for i in 0..4 {
            self.debug_line(n[i].to_array(), n[(i + 1) % 4].to_array(), color);
            self.debug_line(f[i].to_array(), f[(i + 1) % 4].to_array(), color);
            self.debug_line(n[i].to_array(), f[i].to_array(), color);
        }
    }
}
