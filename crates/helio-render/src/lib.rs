use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use helio_core::{MeshBuffer, PackedVertex};
use helio_features::{FeatureContext, FeatureRegistry, MeshData};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Camera uniforms (group 0, binding 0)
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CameraUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub position: [f32; 3],
    pub time: f32,
}

impl CameraUniforms {
    pub fn new(view_proj: Mat4, position: Vec3, time: f32) -> Self {
        Self { view_proj: view_proj.to_cols_array_2d(), position: position.to_array(), time }
    }
}

/// Per-object transform uniforms (group 1, binding 0 with dynamic offset)
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct TransformUniforms {
    pub model: [[f32; 4]; 4],
}

impl TransformUniforms {
    pub fn from_matrix(mat: Mat4) -> Self { Self { model: mat.to_cols_array_2d() } }
    pub fn identity() -> Self { Self { model: Mat4::IDENTITY.to_cols_array_2d() } }
}

/// Simple FPS-style camera.
pub struct FpsCamera {
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub move_speed: f32,
    pub look_speed: f32,
}

impl FpsCamera {
    pub fn new(position: Vec3) -> Self {
        Self { position, yaw: -90.0_f32.to_radians(), pitch: 0.0, move_speed: 5.0, look_speed: 0.1 }
    }
    pub fn forward(&self) -> Vec3 {
        Vec3::new(self.yaw.cos() * self.pitch.cos(), self.pitch.sin(), self.yaw.sin() * self.pitch.cos()).normalize()
    }
    pub fn right(&self) -> Vec3 { self.forward().cross(Vec3::Y).normalize() }
    pub fn update_movement(&mut self, forward: f32, right: f32, up: f32, dt: f32) {
        let fwd = self.forward(); let rgt = self.right(); let speed = self.move_speed * dt;
        self.position += fwd * forward * speed + rgt * right * speed;
        self.position.y += up * speed;
    }
    pub fn handle_mouse_delta(&mut self, dx: f32, dy: f32) {
        self.yaw += dx * self.look_speed * 0.01;
        self.pitch = (self.pitch - dy * self.look_speed * 0.01).clamp(-89.0_f32.to_radians(), 89.0_f32.to_radians());
    }
    pub fn view_matrix(&self) -> Mat4 { Mat4::look_at_rh(self.position, self.position + self.forward(), Vec3::Y) }
    pub fn build_camera_uniforms(&self, fov_deg: f32, aspect: f32, time: f32) -> CameraUniforms {
        let proj = Mat4::perspective_rh(fov_deg.to_radians(), aspect, 0.1, 10000.0);
        CameraUniforms::new(proj * self.view_matrix(), self.position, time)
    }
}

const MAX_OBJECTS: u64 = 1024;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

pub struct FeatureRenderer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::RenderPipeline,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    transform_buffer: wgpu::Buffer,
    transform_bind_group_layout: wgpu::BindGroupLayout,
    transform_bind_group: wgpu::BindGroup,
    transform_stride: u32,
    registry: FeatureRegistry,
    frame_index: u64,
    base_shader: String,
    surface_format: wgpu::TextureFormat,
}

impl FeatureRenderer {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        surface_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        mut registry: FeatureRegistry,
        base_shader: &str,
    ) -> Result<Self, String> {
        let feature_ctx = FeatureContext::new(device.clone(), queue.clone(), (width, height), DEPTH_FORMAT, surface_format);
        registry.init_all(&feature_ctx);

        let composed = registry.compose_shader(base_shader);
        log::debug!("Composed shader ({} features)", registry.enabled_count());

        // Camera BGL: group 0
        let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }],
        });

        // Transform BGL: group 1, dynamic offset
        let transform_align = device.limits().min_uniform_buffer_offset_alignment;
        let transform_stride = helio_core::align_to(std::mem::size_of::<TransformUniforms>() as u32, transform_align);
        let transform_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("transform_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: true, min_binding_size: None },
                count: None,
            }],
        });

        // Gather extra BGLs from features (group 2+)
        let mut extra_bgls: Vec<(u32, wgpu::BindGroupLayout)> = Vec::new();
        for feature in registry.features() {
            if feature.is_enabled() {
                if let Some(bgl_entry) = feature.main_pass_bind_group_layout(&device) {
                    extra_bgls.push(bgl_entry);
                }
            }
        }
        extra_bgls.sort_by_key(|(g, _)| *g);

        let mut bgl_refs: Vec<&wgpu::BindGroupLayout> = vec![&camera_bgl, &transform_bgl];
        for (_, bgl) in &extra_bgls { bgl_refs.push(bgl); }

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("main_pipeline_layout"),
            bind_group_layouts: &bgl_refs,
            push_constant_ranges: &[],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("main_shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(composed.clone())),
        });

        let (depth_texture, depth_view) = create_depth(device.as_ref(), width, height);

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("main_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[PackedVertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState { format: surface_format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
        });

        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera_buffer"),
            size: std::mem::size_of::<CameraUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera_bg"),
            layout: &camera_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: camera_buffer.as_entire_binding() }],
        });

        let transform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("transform_buffer"),
            size: MAX_OBJECTS * transform_stride as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let transform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("transform_bg"),
            layout: &transform_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &transform_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(std::mem::size_of::<TransformUniforms>() as u64),
                }),
            }],
        });

        Ok(Self {
            device, queue, pipeline,
            depth_texture, depth_view,
            camera_buffer, camera_bind_group,
            transform_buffer, transform_bind_group_layout: transform_bgl, transform_bind_group,
            transform_stride,
            registry,
            frame_index: 0,
            base_shader: base_shader.to_string(),
            surface_format,
        })
    }

    pub fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        camera: CameraUniforms,
        meshes: &[(TransformUniforms, &MeshBuffer)],
        delta_time: f32,
    ) {
        let mut ctx = FeatureContext::new(self.device.clone(), self.queue.clone(), (0, 0), DEPTH_FORMAT, self.surface_format);
        ctx.update_frame(self.frame_index, delta_time);
        ctx.camera_position = camera.position;

        self.registry.prepare_frame(&ctx);

        // Build MeshData for shadow pass
        let mesh_data: Vec<MeshData> = meshes.iter().map(|(t, m)| MeshData {
            transform: t.model,
            vertex_buffer: Arc::clone(&m.vertex_buffer),
            index_buffer: Arc::clone(&m.index_buffer),
            index_count: m.index_count,
        }).collect();

        self.registry.execute_shadow_passes(encoder, &ctx, &mesh_data);
        self.registry.execute_pre_passes(encoder, &ctx);

        // Upload camera + transforms
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&camera));
        for (i, (transform, _)) in meshes.iter().enumerate() {
            let offset = i as u64 * self.transform_stride as u64;
            self.queue.write_buffer(&self.transform_buffer, offset, bytemuck::bytes_of(transform));
        }

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.camera_bind_group, &[]);

            // Bind feature bind groups (group 2+)
            for feature in self.registry.features() {
                if feature.is_enabled() {
                    if let Some((group, bg)) = feature.main_pass_bind_group() {
                        pass.set_bind_group(group, bg, &[]);
                    }
                }
            }

            for (i, (_, mesh)) in meshes.iter().enumerate() {
                let offset = i as u32 * self.transform_stride;
                pass.set_bind_group(1, &self.transform_bind_group, &[offset]);
                pass.set_vertex_buffer(0, mesh.vertex_buffer().slice(..));
                pass.set_index_buffer(mesh.index_buffer().slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..mesh.index_count, 0, 0..1);
            }
        }

        self.registry.execute_post_passes(encoder, &ctx);
        self.frame_index += 1;
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        let (dt, dv) = create_depth(self.device.as_ref(), width, height);
        self.depth_texture = dt;
        self.depth_view = dv;
    }

    pub fn rebuild_pipeline(&mut self) {
        log::info!("Rebuilding pipeline");
        let composed = self.registry.compose_shader(&self.base_shader);
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("main_shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(composed)),
        });

        let camera_bgl = self.camera_bind_group_layout();
        let extra_bgls = self.extra_bgls();
        let mut bgl_refs: Vec<&wgpu::BindGroupLayout> = vec![&camera_bgl, &self.transform_bind_group_layout];
        for (_, bgl) in &extra_bgls { bgl_refs.push(bgl); }
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &bgl_refs, push_constant_ranges: &[],
        });

        self.pipeline = self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("main_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: "vs_main", buffers: &[PackedVertex::desc()], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: "fs_main", targets: &[Some(wgpu::ColorTargetState { format: self.surface_format, blend: None, write_mask: wgpu::ColorWrites::ALL })], compilation_options: Default::default() }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: Some(wgpu::Face::Back), ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState { format: DEPTH_FORMAT, depth_write_enabled: true, depth_compare: wgpu::CompareFunction::Less, stencil: Default::default(), bias: Default::default() }),
            multisample: Default::default(),
            multiview: None,
        });
    }

    fn camera_bind_group_layout(&self) -> wgpu::BindGroupLayout {
        self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None }],
        })
    }

    fn extra_bgls(&self) -> Vec<(u32, wgpu::BindGroupLayout)> {
        let mut out = Vec::new();
        for feature in self.registry.features() {
            if feature.is_enabled() {
                if let Some(e) = feature.main_pass_bind_group_layout(&self.device) { out.push(e); }
            }
        }
        out.sort_by_key(|(g, _)| *g);
        out
    }

    pub fn registry(&self) -> &FeatureRegistry { &self.registry }
    pub fn registry_mut(&mut self) -> &mut FeatureRegistry { &mut self.registry }
    pub fn frame_index(&self) -> u64 { self.frame_index }
    pub fn device(&self) -> &Arc<wgpu::Device> { &self.device }
    pub fn queue(&self) -> &Arc<wgpu::Queue> { &self.queue }

    pub fn toggle_and_rebuild(&mut self, name: &str) -> Result<bool, helio_features::FeatureError> {
        let state = self.registry.toggle_feature(name)?;
        self.rebuild_pipeline();
        Ok(state)
    }
}

fn create_depth(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}
