use bytemuck::{Pod, Zeroable};
use glam;
use helio_core::PackedVertex;
use helio_features::{Feature, FeatureContext, MeshData, ShaderInjection, ShaderInjectionPoint};
use std::sync::Arc;
use wgpu::util::DeviceExt;

pub const MAX_SHADOW_LIGHTS: usize = 8;
const SHADOW_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LightType {
    Directional,
    Point,
    Spot { inner_angle: f32, outer_angle: f32 },
    Rect { width: f32, height: f32 },
}

#[derive(Debug, Clone, Copy)]
pub struct LightConfig {
    pub light_type: LightType,
    pub position: glam::Vec3,
    pub direction: glam::Vec3,
    pub intensity: f32,
    pub color: glam::Vec3,
    pub attenuation_radius: f32,
    pub attenuation_falloff: f32,
}

impl Default for LightConfig {
    fn default() -> Self {
        Self {
            light_type: LightType::Directional,
            position: glam::Vec3::new(10.0, 15.0, 10.0),
            direction: glam::Vec3::new(0.5, -1.0, 0.3).normalize(),
            intensity: 1.0, color: glam::Vec3::ONE,
            attenuation_radius: 10.0, attenuation_falloff: 2.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuLight {
    pub view_proj: [[f32; 4]; 4],
    pub position_and_type: [f32; 4],
    pub direction_and_radius: [f32; 4],
    pub color_and_intensity: [f32; 4],
    pub params: [f32; 4],
}

impl GpuLight {
    fn from_config(config: &LightConfig, shadow_layer: u32) -> Self {
        let light_type = match config.light_type { LightType::Directional => 0.0, LightType::Point => 1.0, LightType::Spot { .. } => 2.0, LightType::Rect { .. } => 3.0 };
        let (ia, oa) = match config.light_type { LightType::Spot { inner_angle, outer_angle } => (inner_angle, outer_angle), _ => (0.0, 0.0) };
        Self {
            view_proj: [[0.0;4];4],
            position_and_type: [config.position.x, config.position.y, config.position.z, light_type],
            direction_and_radius: [config.direction.x, config.direction.y, config.direction.z, config.attenuation_radius],
            color_and_intensity: [config.color.x, config.color.y, config.color.z, config.intensity],
            params: [ia, oa, config.attenuation_falloff, shadow_layer as f32],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ShadowUniforms {
    pub light_view_proj: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct LightingUniforms {
    pub light_count: [f32; 4],
    pub lights: [GpuLight; MAX_SHADOW_LIGHTS],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ObjectUniforms {
    pub model: [[f32; 4]; 4],
}

pub struct ProceduralShadows {
    enabled: bool,
    shadow_map_size: u32,
    lights: Vec<LightConfig>,
    ambient: f32,
    // GPU resources (created in init)
    shadow_map: Option<wgpu::Texture>,
    shadow_map_array_view: Option<wgpu::TextureView>,
    shadow_sampler: Option<wgpu::Sampler>,
    shadow_pipeline: Option<wgpu::RenderPipeline>,
    shadow_bgl: Option<wgpu::BindGroupLayout>,
    shadow_bind_group: Option<wgpu::BindGroup>,
    lighting_buffer: Option<wgpu::Buffer>,
    shadow_uniforms_buffer: Option<wgpu::Buffer>,
    object_uniforms_buffer: Option<wgpu::Buffer>,
    shadow_ubo_stride: u32,
    object_ubo_stride: u32,
    device: Option<Arc<wgpu::Device>>,
    queue: Option<Arc<wgpu::Queue>>,
    // Spotlight billboard texture (visual only)
    spotlight_icon_texture: Option<helio_core::TextureId>,
}

impl ProceduralShadows {
    pub fn new() -> Self {
        Self {
            enabled: true, shadow_map_size: 1024, lights: Vec::new(), ambient: 0.1,
            shadow_map: None, shadow_map_array_view: None, shadow_sampler: None,
            shadow_pipeline: None, shadow_bgl: None, shadow_bind_group: None,
            lighting_buffer: None, shadow_uniforms_buffer: None, object_uniforms_buffer: None,
            shadow_ubo_stride: 0, object_ubo_stride: 0,
            device: None, queue: None,
            spotlight_icon_texture: None,
        }
    }
    pub fn with_size(mut self, size: u32) -> Self { self.shadow_map_size = size; self }
    pub fn with_ambient(mut self, ambient: f32) -> Self { self.ambient = ambient; self }
    pub fn set_ambient(&mut self, ambient: f32) { self.ambient = ambient; }
    pub fn ambient(&self) -> f32 { self.ambient }
    pub fn add_light(&mut self, config: LightConfig) -> Result<(), String> {
        if self.lights.len() >= MAX_SHADOW_LIGHTS { return Err(format!("Max {} lights", MAX_SHADOW_LIGHTS)); }
        self.lights.push(config); Ok(())
    }
    pub fn clear_lights(&mut self) { self.lights.clear(); }
    pub fn lights(&self) -> &[LightConfig] { &self.lights }
    pub fn lights_mut(&mut self) -> &mut Vec<LightConfig> { &mut self.lights }
    pub fn with_light(mut self, config: LightConfig) -> Self { self.lights = vec![config]; self }
    pub fn with_directional_light(mut self, direction: glam::Vec3) -> Self {
        let mut c = LightConfig::default(); c.light_type = LightType::Directional; c.direction = direction.normalize();
        self.lights = vec![c]; self
    }
    pub fn with_point_light(mut self, position: glam::Vec3, radius: f32) -> Self {
        let mut c = LightConfig::default(); c.light_type = LightType::Point; c.position = position; c.attenuation_radius = radius;
        self.lights = vec![c]; self
    }
    pub fn with_spot_light(mut self, position: glam::Vec3, direction: glam::Vec3, inner_angle: f32, outer_angle: f32, radius: f32) -> Self {
        let mut c = LightConfig::default(); c.light_type = LightType::Spot { inner_angle, outer_angle }; c.position = position; c.direction = direction.normalize(); c.attenuation_radius = radius;
        self.lights = vec![c]; self
    }
    pub fn set_spotlight_icon(&mut self, id: helio_core::TextureId) { self.spotlight_icon_texture = Some(id); }
    pub fn spotlight_icon_texture(&self) -> Option<helio_core::TextureId> { self.spotlight_icon_texture }
    pub fn generate_light_billboards(&self) -> Vec<(glam::Vec3, f32)> {
        self.lights.iter().filter_map(|l| {
            let size = match l.light_type { LightType::Directional => return None, LightType::Point => 0.5, LightType::Spot { .. } => 0.6, LightType::Rect { width, height } => width.max(height) * 0.5 };
            Some((l.position, size))
        }).collect()
    }
    pub fn light_config(&self) -> Option<&LightConfig> { self.lights.first() }
    pub fn set_light_config(&mut self, config: LightConfig) { self.lights = vec![config]; }
    pub fn with_rect_light(mut self, position: glam::Vec3, direction: glam::Vec3, width: f32, height: f32, radius: f32) -> Self {
        let mut c = LightConfig::default(); c.light_type = LightType::Rect { width, height }; c.position = position; c.direction = direction.normalize(); c.attenuation_radius = radius;
        self.lights = vec![c]; self
    }
    pub fn set_texture_manager(&mut self, _: Arc<helio_core::TextureManager>) {}

    fn build_lighting_uniforms(&self) -> LightingUniforms {
        let mut lights = [GpuLight { view_proj: [[0.0;4];4], position_and_type: [0.0;4], direction_and_radius: [0.0;4], color_and_intensity: [0.0;4], params: [0.0;4] }; MAX_SHADOW_LIGHTS];
        for (i, config) in self.lights.iter().take(MAX_SHADOW_LIGHTS).enumerate() {
            let base_layer = (i * 6) as u32;
            let vp = Self::get_light_view_proj(config);
            let mut gl = GpuLight::from_config(config, base_layer);
            gl.view_proj = vp.to_cols_array_2d();
            lights[i] = gl;
        }
        LightingUniforms { light_count: [self.lights.len().min(MAX_SHADOW_LIGHTS) as f32, self.ambient, 0.0, 0.0], lights }
    }

    const CUBE_FACE_DIRS: [(glam::Vec3, glam::Vec3); 6] = [
        (glam::Vec3::X, glam::Vec3::NEG_Y),
        (glam::Vec3::NEG_X, glam::Vec3::NEG_Y),
        (glam::Vec3::Y, glam::Vec3::Z),
        (glam::Vec3::NEG_Y, glam::Vec3::NEG_Z),
        (glam::Vec3::Z, glam::Vec3::NEG_Y),
        (glam::Vec3::NEG_Z, glam::Vec3::NEG_Y),
    ];

    fn get_shadow_render_matrices(config: &LightConfig) -> Vec<glam::Mat4> {
        match config.light_type {
            LightType::Directional => {
                let lp = -config.direction * 20.0;
                let v = glam::Mat4::look_at_rh(lp, glam::Vec3::ZERO, glam::Vec3::Y);
                let p = glam::Mat4::orthographic_rh(-8.0, 8.0, -8.0, 8.0, 0.1, 40.0);
                vec![p * v]
            }
            LightType::Point => {
                if config.attenuation_radius <= 0.1 { return Vec::new(); }
                let p = glam::Mat4::perspective_rh(90.0_f32.to_radians(), 1.0, 0.1, config.attenuation_radius);
                Self::CUBE_FACE_DIRS.iter().map(|(fwd, up)| {
                    let v = glam::Mat4::look_at_rh(config.position, config.position + *fwd, *up);
                    p * v
                }).collect()
            }
            LightType::Spot { outer_angle, .. } => {
                if outer_angle <= 0.0 || config.attenuation_radius <= 0.1 { return Vec::new(); }
                let v = glam::Mat4::look_at_rh(config.position, config.position + config.direction, glam::Vec3::Y);
                let p = glam::Mat4::perspective_rh(outer_angle * 2.0, 1.0, 0.1, config.attenuation_radius);
                vec![p * v]
            }
            LightType::Rect { width, height } => {
                if width <= 0.0 || height <= 0.0 || config.attenuation_radius <= 0.1 { return Vec::new(); }
                let v = glam::Mat4::look_at_rh(config.position, config.position + config.direction, glam::Vec3::Y);
                let (hw, hh) = (width / 2.0, height / 2.0);
                let p = glam::Mat4::orthographic_rh(-hw, hw, -hh, hh, 0.1, config.attenuation_radius);
                vec![p * v]
            }
        }
    }

    fn get_light_view_proj(config: &LightConfig) -> glam::Mat4 {
        match config.light_type {
            LightType::Directional => {
                let lp = -config.direction * 20.0;
                let v = glam::Mat4::look_at_rh(lp, glam::Vec3::ZERO, glam::Vec3::Y);
                glam::Mat4::orthographic_rh(-8.0, 8.0, -8.0, 8.0, 0.1, 40.0) * v
            }
            LightType::Point => {
                let (fwd, up) = Self::CUBE_FACE_DIRS[0];
                let v = glam::Mat4::look_at_rh(config.position, config.position + fwd, up);
                glam::Mat4::perspective_rh(90.0_f32.to_radians(), 1.0, 0.1, config.attenuation_radius) * v
            }
            LightType::Spot { outer_angle, .. } => {
                let v = glam::Mat4::look_at_rh(config.position, config.position + config.direction, glam::Vec3::Y);
                glam::Mat4::perspective_rh(outer_angle * 2.0, 1.0, 0.1, config.attenuation_radius) * v
            }
            LightType::Rect { width, height } => {
                let v = glam::Mat4::look_at_rh(config.position, config.position + config.direction, glam::Vec3::Y);
                let (hw, hh) = (width / 2.0, height / 2.0);
                glam::Mat4::orthographic_rh(-hw, hw, -hh, hh, 0.1, config.attenuation_radius) * v
            }
        }
    }
}

impl Default for ProceduralShadows {
    fn default() -> Self { Self::new() }
}

impl Feature for ProceduralShadows {
    fn name(&self) -> &str { "procedural_shadows" }
    fn is_enabled(&self) -> bool { self.enabled }
    fn set_enabled(&mut self, e: bool) { self.enabled = e; }

    fn init(&mut self, ctx: &FeatureContext) {
        log::info!("Initializing procedural shadows ({}x{}, {} layers)", self.shadow_map_size, self.shadow_map_size, MAX_SHADOW_LIGHTS * 6);
        self.device = Some(ctx.device.clone());
        self.queue = Some(ctx.queue.clone());
        let device = &ctx.device;

        let align = device.limits().min_uniform_buffer_offset_alignment;
        self.shadow_ubo_stride = helio_core::align_to(std::mem::size_of::<ShadowUniforms>() as u32, align);
        self.object_ubo_stride = helio_core::align_to(std::mem::size_of::<ObjectUniforms>() as u32, align);

        // Create shadow map array texture
        let shadow_map = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow_map_array"),
            size: wgpu::Extent3d { width: self.shadow_map_size, height: self.shadow_map_size, depth_or_array_layers: (MAX_SHADOW_LIGHTS * 6) as u32 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SHADOW_DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let array_view = shadow_map.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            aspect: wgpu::TextureAspect::DepthOnly,
            array_layer_count: Some((MAX_SHADOW_LIGHTS * 6) as u32),
            ..Default::default()
        });
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        // Shadow pass pipeline
        let shadow_vert_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow_vert_bgl"),
            entries: &[
                bgl_entry(0, wgpu::ShaderStages::VERTEX, wgpu::BufferBindingType::Uniform, true),
                bgl_entry(1, wgpu::ShaderStages::VERTEX, wgpu::BufferBindingType::Uniform, true),
            ],
        });
        let shadow_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&shadow_vert_bgl], push_constant_ranges: &[],
        });
        let shadow_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shadow_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/shadow_pass.wgsl").into()),
        });
        let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow_pipeline"),
            layout: Some(&shadow_pipeline_layout),
            vertex: wgpu::VertexState { module: &shadow_shader, entry_point: "vs_main", buffers: &[PackedVertex::desc()], compilation_options: Default::default() },
            fragment: None,
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState { format: SHADOW_DEPTH_FORMAT, depth_write_enabled: true, depth_compare: wgpu::CompareFunction::Less, stencil: Default::default(), bias: wgpu::DepthBiasState { constant: 2, slope_scale: 2.0, clamp: 0.0 } }),
            multisample: Default::default(),
            multiview: None,
        });

        // Uniform buffers for shadow pass
        let max_faces = MAX_SHADOW_LIGHTS * 6;
        let shadow_uniforms_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow_uniforms"), size: max_faces as u64 * self.shadow_ubo_stride as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        let object_uniforms_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow_object_uniforms"), size: 1024 * self.object_ubo_stride as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });

        // Main pass shadow BGL: group 2
        let shadow_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow_bgl"),
            entries: &[
                bgl_entry_uniform(0, wgpu::ShaderStages::FRAGMENT, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: wgpu::TextureViewDimension::D2Array, multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });

        let lighting_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lighting_uniforms"), size: std::mem::size_of::<LightingUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });

        let shadow_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow_bg"),
            layout: &shadow_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: lighting_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&array_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&shadow_sampler) },
            ],
        });

        self.shadow_map = Some(shadow_map);
        self.shadow_map_array_view = Some(array_view);
        self.shadow_sampler = Some(shadow_sampler);
        self.shadow_pipeline = Some(shadow_pipeline);
        self.shadow_bgl = Some(shadow_bgl);
        self.shadow_bind_group = Some(shadow_bind_group);
        self.lighting_buffer = Some(lighting_buffer);
        self.shadow_uniforms_buffer = Some(shadow_uniforms_buffer);
        self.object_uniforms_buffer = Some(object_uniforms_buffer);
    }

    fn prepare_frame(&mut self, _ctx: &FeatureContext) {
        if let (Some(queue), Some(buf)) = (&self.queue, &self.lighting_buffer) {
            let uniforms = self.build_lighting_uniforms();
            queue.write_buffer(buf, 0, bytemuck::bytes_of(&uniforms));
        }
    }

    fn shader_injections(&self) -> Vec<ShaderInjection> {
        vec![ShaderInjection::new(ShaderInjectionPoint::FragmentPreamble, include_str!("../shaders/shadow_functions.wgsl"))]
    }

    fn main_pass_bind_group_layout(&self, device: &wgpu::Device) -> Option<(u32, wgpu::BindGroupLayout)> {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow_bgl"),
            entries: &[
                bgl_entry_uniform(0, wgpu::ShaderStages::FRAGMENT, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: wgpu::TextureViewDimension::D2Array, multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });
        Some((2, bgl))
    }

    fn main_pass_bind_group(&self) -> Option<(u32, &wgpu::BindGroup)> {
        self.shadow_bind_group.as_ref().map(|bg| (2u32, bg))
    }

    fn render_shadow_pass(&mut self, encoder: &mut wgpu::CommandEncoder, _ctx: &FeatureContext, meshes: &[MeshData]) {
        let shadow_map = match &self.shadow_map { Some(s) => s, None => return };
        let pipeline = match &self.shadow_pipeline { Some(p) => p, None => return };
        let sub = match &self.shadow_uniforms_buffer { Some(b) => b, None => return };
        let oub = match &self.object_uniforms_buffer { Some(b) => b, None => return };
        let queue = match &self.queue { Some(q) => q, None => return };
        let device = match &self.device { Some(d) => d, None => return };

        // Upload object transforms
        for (i, mesh) in meshes.iter().enumerate().take(1024) {
            let off = i as u64 * self.object_ubo_stride as u64;
            queue.write_buffer(oub, off, bytemuck::cast_slice(&mesh.transform));
        }

        let shadow_vert_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                bgl_entry(0, wgpu::ShaderStages::VERTEX, wgpu::BufferBindingType::Uniform, true),
                bgl_entry(1, wgpu::ShaderStages::VERTEX, wgpu::BufferBindingType::Uniform, true),
            ],
        });
        let shadow_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &shadow_vert_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: sub, offset: 0, size: wgpu::BufferSize::new(std::mem::size_of::<ShadowUniforms>() as u64) }) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: oub, offset: 0, size: wgpu::BufferSize::new(std::mem::size_of::<ObjectUniforms>() as u64) }) },
            ],
        });

        let mut face_index = 0u32;
        for (light_idx, config) in self.lights.iter().enumerate() {
            let matrices = Self::get_shadow_render_matrices(config);
            for mat in &matrices {
                let su = ShadowUniforms { light_view_proj: mat.to_cols_array_2d() };
                let off = face_index as u64 * self.shadow_ubo_stride as u64;
                queue.write_buffer(sub, off, bytemuck::bytes_of(&su));

                let layer = light_idx * 6 + (face_index as usize % 6);
                let face_view = shadow_map.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    aspect: wgpu::TextureAspect::DepthOnly,
                    base_array_layer: layer as u32, array_layer_count: Some(1),
                    ..Default::default()
                });

                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("shadow_pass"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &face_view,
                        depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None, occlusion_query_set: None,
                });

                pass.set_pipeline(pipeline);
                let su_off = face_index * self.shadow_ubo_stride;
                for (mesh_idx, mesh) in meshes.iter().enumerate().take(1024) {
                    let obj_off = mesh_idx as u32 * self.object_ubo_stride;
                    pass.set_bind_group(0, &shadow_bg, &[su_off, obj_off]);
                    pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                }

                face_index += 1;
            }
        }
    }
}

fn bgl_entry(binding: u32, vis: wgpu::ShaderStages, ty: wgpu::BufferBindingType, dynamic_offset: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: vis, ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: dynamic_offset, min_binding_size: None }, count: None }
}
fn bgl_entry_uniform(binding: u32, vis: wgpu::ShaderStages, dynamic_offset: bool) -> wgpu::BindGroupLayoutEntry {
    bgl_entry(binding, vis, wgpu::BufferBindingType::Uniform, dynamic_offset)
}
