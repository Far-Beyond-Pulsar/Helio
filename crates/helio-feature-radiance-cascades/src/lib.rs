use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use helio_core::PackedVertex;
use helio_features::{Feature, FeatureContext, MeshData, ShaderInjection, ShaderInjectionPoint};
use std::sync::Arc;
use wgpu::util::DeviceExt;

pub use config::{GIQuality, IntegrationMode, RadianceCascadesConfig};
pub use cascade::{CascadeData, CascadeType};
pub use uniforms::{GpuCascade, RadianceCascadesUniforms};

mod config;
mod cascade;
mod uniforms;

// ======================== Light API ========================

pub const MAX_LIGHTS: usize = 8;
const SHADOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const SHADOW_MAP_SIZE: u32 = 1024;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LightType {
    Directional,
    Point,
    Spot { inner_angle: f32, outer_angle: f32 },
}

#[derive(Debug, Clone, Copy)]
pub struct LightConfig {
    pub light_type: LightType,
    pub position: Vec3,
    pub direction: Vec3,
    pub intensity: f32,
    pub color: Vec3,
    pub attenuation_radius: f32,
    pub attenuation_falloff: f32,
}

impl Default for LightConfig {
    fn default() -> Self {
        Self {
            light_type: LightType::Directional,
            position: Vec3::new(10.0, 15.0, 10.0),
            direction: Vec3::new(0.5, -1.0, 0.3).normalize(),
            intensity: 1.0,
            color: Vec3::ONE,
            attenuation_radius: 20.0,
            attenuation_falloff: 2.0,
        }
    }
}

// ======================== GPU Layout ========================

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuLight {
    view_proj: [[f32; 4]; 4],
    position_and_type: [f32; 4],    // xyz=pos, w=type (0=dir,1=point,2=spot)
    direction_and_radius: [f32; 4], // xyz=dir, w=attenuation_radius
    color_and_intensity: [f32; 4],  // xyz=color, w=intensity
    params: [f32; 4],               // x=inner_angle, y=outer_angle, z=falloff, w=base_shadow_layer
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LightingUniforms {
    light_count: [f32; 4], // x=count, y=ambient, zw=unused
    lights: [GpuLight; MAX_LIGHTS],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ShadowVpUniforms {
    view_proj: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ShadowObjUniforms {
    model: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CascadeIndexUniform {
    index: u32,
    _pad: [u32; 3],
}

// ======================== RadianceCascades ========================

pub struct RadianceCascades {
    enabled: bool,
    pub config: RadianceCascadesConfig,
    lights: Vec<LightConfig>,
    cascades: Vec<CascadeData>,
    device: Option<Arc<wgpu::Device>>,
    queue: Option<Arc<wgpu::Queue>>,
    uniforms_buffer: Option<wgpu::Buffer>,
    lighting_buffer: Option<wgpu::Buffer>,
    cascade_index_bufs: Vec<wgpu::Buffer>,
    // Shadow rendering
    shadow_map: Option<wgpu::Texture>,
    shadow_map_array_view: Option<wgpu::TextureView>,
    shadow_sampler: Option<wgpu::Sampler>,
    shadow_pipeline: Option<wgpu::RenderPipeline>,
    shadow_vp_buf: Option<wgpu::Buffer>,
    shadow_obj_buf: Option<wgpu::Buffer>,
    shadow_vp_stride: u32,
    shadow_obj_stride: u32,
    // Injection compute
    radiance_injection_pipeline: Option<wgpu::ComputePipeline>,
    injection_bgl: Option<wgpu::BindGroupLayout>,
    // Fragment sampling (group 2)
    gi_sampler: Option<wgpu::Sampler>,
    gi_bind_group: Option<wgpu::BindGroup>,
}

impl RadianceCascades {
    pub fn new() -> Self { Self::with_config(RadianceCascadesConfig::default()) }

    pub fn with_config(config: RadianceCascadesConfig) -> Self {
        Self {
            enabled: true, config, lights: Vec::new(), cascades: Vec::new(),
            device: None, queue: None,
            uniforms_buffer: None, lighting_buffer: None, cascade_index_bufs: Vec::new(),
            shadow_map: None, shadow_map_array_view: None, shadow_sampler: None,
            shadow_pipeline: None, shadow_vp_buf: None, shadow_obj_buf: None,
            shadow_vp_stride: 0, shadow_obj_stride: 0,
            radiance_injection_pipeline: None, injection_bgl: None,
            gi_sampler: None, gi_bind_group: None,
        }
    }

    pub fn add_light(&mut self, config: LightConfig) -> Result<(), String> {
        if self.lights.len() >= MAX_LIGHTS {
            return Err(format!("Max {} lights reached", MAX_LIGHTS));
        }
        self.lights.push(config);
        Ok(())
    }
    pub fn set_lights(&mut self, lights: Vec<LightConfig>) { self.lights = lights; }
    pub fn clear_lights(&mut self)                         { self.lights.clear(); }
    pub fn lights(&self)     -> &[LightConfig]             { &self.lights }
    pub fn lights_mut(&mut self) -> &mut Vec<LightConfig>  { &mut self.lights }

    const CUBE_FACE_DIRS: [(Vec3, Vec3); 6] = [
        (Vec3::X,     Vec3::NEG_Y),
        (Vec3::NEG_X, Vec3::NEG_Y),
        (Vec3::Y,     Vec3::Z),
        (Vec3::NEG_Y, Vec3::NEG_Z),
        (Vec3::Z,     Vec3::NEG_Y),
        (Vec3::NEG_Z, Vec3::NEG_Y),
    ];

    fn shadow_layers(lt: &LightType) -> u32 {
        match lt { LightType::Point => 6, _ => 1 }
    }

    fn get_shadow_render_matrices(config: &LightConfig) -> Vec<Mat4> {
        match config.light_type {
            LightType::Directional => {
                let lp = -config.direction * 20.0;
                let v = Mat4::look_at_rh(lp, Vec3::ZERO, Vec3::Y);
                vec![Mat4::orthographic_rh(-15.0, 15.0, -15.0, 15.0, 0.1, 50.0) * v]
            }
            LightType::Point => {
                if config.attenuation_radius <= 0.1 { return Vec::new(); }
                let p = Mat4::perspective_rh(90.0_f32.to_radians(), 1.0, 0.1, config.attenuation_radius);
                Self::CUBE_FACE_DIRS.iter().map(|(fwd, up)| {
                    p * Mat4::look_at_rh(config.position, config.position + *fwd, *up)
                }).collect()
            }
            LightType::Spot { outer_angle, .. } => {
                if config.attenuation_radius <= 0.1 { return Vec::new(); }
                let v = Mat4::look_at_rh(config.position, config.position + config.direction, Vec3::Y);
                let p = Mat4::perspective_rh(outer_angle * 2.0, 1.0, 0.1, config.attenuation_radius);
                vec![p * v]
            }
        }
    }

    fn get_light_view_proj(config: &LightConfig) -> Mat4 {
        match config.light_type {
            LightType::Directional => {
                let lp = -config.direction * 20.0;
                let v = Mat4::look_at_rh(lp, Vec3::ZERO, Vec3::Y);
                Mat4::orthographic_rh(-15.0, 15.0, -15.0, 15.0, 0.1, 50.0) * v
            }
            LightType::Point => {
                let (fwd, up) = Self::CUBE_FACE_DIRS[0];
                let v = Mat4::look_at_rh(config.position, config.position + fwd, up);
                Mat4::perspective_rh(90.0_f32.to_radians(), 1.0, 0.1, config.attenuation_radius) * v
            }
            LightType::Spot { outer_angle, .. } => {
                let v = Mat4::look_at_rh(config.position, config.position + config.direction, Vec3::Y);
                Mat4::perspective_rh(outer_angle * 2.0, 1.0, 0.1, config.attenuation_radius) * v
            }
        }
    }

    fn build_lighting_uniforms(&self) -> LightingUniforms {
        let zero = GpuLight {
            view_proj: [[0.0; 4]; 4], position_and_type: [0.0; 4],
            direction_and_radius: [0.0; 4], color_and_intensity: [0.0; 4], params: [0.0; 4],
        };
        let mut lights = [zero; MAX_LIGHTS];
        let mut base_layer = 0u32;
        for (i, cfg) in self.lights.iter().take(MAX_LIGHTS).enumerate() {
            let lt = match cfg.light_type { LightType::Directional => 0.0, LightType::Point => 1.0, LightType::Spot { .. } => 2.0 };
            let (ia, oa) = match cfg.light_type { LightType::Spot { inner_angle, outer_angle } => (inner_angle, outer_angle), _ => (0.0, 0.0) };
            lights[i] = GpuLight {
                view_proj: Self::get_light_view_proj(cfg).to_cols_array_2d(),
                position_and_type: [cfg.position.x, cfg.position.y, cfg.position.z, lt],
                direction_and_radius: [cfg.direction.x, cfg.direction.y, cfg.direction.z, cfg.attenuation_radius],
                color_and_intensity: [cfg.color.x, cfg.color.y, cfg.color.z, cfg.intensity],
                params: [ia, oa, cfg.attenuation_falloff, base_layer as f32],
            };
            base_layer += Self::shadow_layers(&cfg.light_type);
        }
        LightingUniforms {
            light_count: [self.lights.len().min(MAX_LIGHTS) as f32, self.config.ambient, 0.0, 0.0],
            lights,
        }
    }

    fn create_cascade_texture(device: &wgpu::Device, name: &str, res: u32) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(name),
            size: wgpu::Extent3d { width: res, height: res, depth_or_array_layers: res },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            array_layer_count: Some(res),
            ..Default::default()
        });
        (tex, view)
    }

    fn rebuild_gi_bind_group(&mut self, device: &wgpu::Device) {
        let (Some(cv), Some(ub), Some(gs)) = (
            self.cascades.first().and_then(|c| c.radiance_view.as_ref()),
            self.uniforms_buffer.as_ref(),
            self.gi_sampler.as_ref(),
        ) else { return };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gi_frag_bgl"),
            entries: &[
                bgl_tex2darray(0, wgpu::ShaderStages::FRAGMENT),
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None,
                },
                bgl_uniform(2, wgpu::ShaderStages::FRAGMENT),
            ],
        });
        self.gi_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gi_frag_bg"), layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(cv) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(gs) },
                wgpu::BindGroupEntry { binding: 2, resource: ub.as_entire_binding() },
            ],
        }));
    }
}

impl Default for RadianceCascades { fn default() -> Self { Self::new() } }

impl Feature for RadianceCascades {
    fn name(&self) -> &str { "radiance_cascades" }
    fn is_enabled(&self) -> bool { self.enabled }
    fn set_enabled(&mut self, e: bool) { self.enabled = e; }

    fn init(&mut self, ctx: &FeatureContext) {
        log::info!("Initializing RadianceCascades GI (self-contained lighting + shadows)");
        let device = &ctx.device;
        self.device = Some(ctx.device.clone());
        self.queue  = Some(ctx.queue.clone());

        let align = device.limits().min_uniform_buffer_offset_alignment;
        self.shadow_vp_stride  = align_to(std::mem::size_of::<ShadowVpUniforms>()  as u32, align);
        self.shadow_obj_stride = align_to(std::mem::size_of::<ShadowObjUniforms>() as u32, align);

        // Cascade textures
        let cascade_count = self.config.quality.cascade_count() as usize;
        let probe_res     = self.config.quality.probe_resolution();
        self.cascades.clear();
        self.cascade_index_bufs.clear();
        for i in 0..cascade_count {
            let ct = if self.config.enable_2d_distant_cascades && i == cascade_count - 1 {
                CascadeType::HeightField2D
            } else { CascadeType::Volumetric3D };
            let extent = 10.0 * self.config.cascade_spacing_factor.powi(i as i32);
            let mut cascade = CascadeData::new(i as u32, ct, Vec3::ZERO, extent, probe_res);
            let (rt, rv) = Self::create_cascade_texture(device, &format!("cascade_{}_radiance", i), probe_res);
            let (ht, hv) = Self::create_cascade_texture(device, &format!("cascade_{}_history", i), probe_res);
            cascade.radiance_texture = Some(rt); cascade.radiance_view   = Some(rv);
            cascade.radiance_history = Some(ht); cascade.history_view    = Some(hv);
            self.cascades.push(cascade);
            self.cascade_index_bufs.push(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("cascade_{}_idx", i)),
                contents: bytemuck::bytes_of(&CascadeIndexUniform { index: i as u32, _pad: [0; 3] }),
                usage: wgpu::BufferUsages::UNIFORM,
            }));
        }

        self.uniforms_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rc_uniforms"),
            size: std::mem::size_of::<RadianceCascadesUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.lighting_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rc_lighting"),
            size: std::mem::size_of::<LightingUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // Shadow map array: MAX_LIGHTS * 6 layers (point lights use 6 faces each)
        let total_layers = (MAX_LIGHTS * 6) as u32;
        let shadow_map = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("rc_shadow_maps"),
            size: wgpu::Extent3d { width: SHADOW_MAP_SIZE, height: SHADOW_MAP_SIZE, depth_or_array_layers: total_layers },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SHADOW_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_array_view = shadow_map.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            aspect: wgpu::TextureAspect::DepthOnly,
            array_layer_count: Some(total_layers),
            ..Default::default()
        });
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("rc_shadow_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        // Shadow depth-only pipeline
        let shad_vert_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rc_shadow_vert_bgl"),
            entries: &[bgl_uniform_dynamic(0, wgpu::ShaderStages::VERTEX), bgl_uniform_dynamic(1, wgpu::ShaderStages::VERTEX)],
        });
        let shad_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&shad_vert_bgl], push_constant_ranges: &[],
        });
        let shad_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rc_shadow_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/shadow_pass.wgsl").into()),
        });
        let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("rc_shadow_pipeline"), layout: Some(&shad_layout),
            vertex: wgpu::VertexState { module: &shad_shader, entry_point: "vs_main", buffers: &[PackedVertex::desc()], compilation_options: Default::default() },
            fragment: None,
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: SHADOW_FORMAT, depth_write_enabled: true, depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: wgpu::DepthBiasState { constant: 2, slope_scale: 2.0, clamp: 0.0 },
            }),
            multisample: Default::default(), multiview: None,
        });

        let max_faces = (MAX_LIGHTS * 6) as u64;
        let shadow_vp_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rc_shadow_vp"),
            size: max_faces * self.shadow_vp_stride as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let shadow_obj_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rc_shadow_obj"),
            size: 1024 * self.shadow_obj_stride as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Injection compute pipeline
        let injection_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rc_injection_bgl"),
            entries: &[
                bgl_uniform(0, wgpu::ShaderStages::COMPUTE),
                bgl_storage_tex_write(1, wgpu::ShaderStages::COMPUTE),
                bgl_tex2darray(2, wgpu::ShaderStages::COMPUTE),
                bgl_uniform(3, wgpu::ShaderStages::COMPUTE),
                wgpu::BindGroupLayoutEntry {
                    binding: 4, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: wgpu::TextureViewDimension::D2Array, multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison), count: None,
                },
                bgl_uniform(6, wgpu::ShaderStages::COMPUTE),
            ],
        });
        let injection_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&injection_bgl], push_constant_ranges: &[],
        });
        let injection_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rc_injection_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/radiance_injection.wgsl").into()),
        });
        self.radiance_injection_pipeline = Some(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rc_injection"), layout: Some(&injection_layout),
            module: &injection_shader, entry_point: "cs_main",
            compilation_options: Default::default(),
        }));
        self.injection_bgl = Some(injection_bgl);

        self.shadow_map            = Some(shadow_map);
        self.shadow_map_array_view = Some(shadow_array_view);
        self.shadow_sampler        = Some(shadow_sampler);
        self.shadow_pipeline       = Some(shadow_pipeline);
        self.shadow_vp_buf         = Some(shadow_vp_buf);
        self.shadow_obj_buf        = Some(shadow_obj_buf);

        self.gi_sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("gi_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));
        self.rebuild_gi_bind_group(device);
    }

    fn prepare_frame(&mut self, _ctx: &FeatureContext) {
        let (Some(queue), Some(ubuf), Some(lbuf)) = (
            self.queue.as_ref(), self.uniforms_buffer.as_ref(), self.lighting_buffer.as_ref()
        ) else { return };

        let lighting = self.build_lighting_uniforms();
        log::debug!("RC prepare_frame: {} lights, first light type={}", self.lights.len(),
            if self.lights.is_empty() { -1.0 } else { self.lights[0].intensity });

        let mut uniforms = RadianceCascadesUniforms::new(
            self.config.quality.cascade_count(), self.config.gi_intensity,
            self.config.integration_mode as u32, self.config.temporal_blend_factor,
        );
        for (i, c) in self.cascades.iter().enumerate().take(4) {
            uniforms.cascades[i] = GpuCascade {
                center_and_extent: [c.world_center.x, c.world_center.y, c.world_center.z, c.world_extent],
                resolution_and_type: [c.probe_resolution as f32, c.cascade_type as u32 as f32, 0.0, 0.0],
                texture_layer: i as u32, _pad0: 0, _pad1: 0, _pad2: 0,
            };
        }
        queue.write_buffer(ubuf, 0, bytemuck::bytes_of(&uniforms));
        queue.write_buffer(lbuf, 0, bytemuck::bytes_of(&lighting));
    }

    fn render_shadow_pass(&mut self, encoder: &mut wgpu::CommandEncoder, _ctx: &FeatureContext, meshes: &[MeshData]) {
        if self.lights.is_empty() { return; }
        let (Some(shadow_map), Some(pipeline), Some(vp_buf), Some(obj_buf), Some(queue), Some(device)) = (
            self.shadow_map.as_ref(), self.shadow_pipeline.as_ref(),
            self.shadow_vp_buf.as_ref(), self.shadow_obj_buf.as_ref(),
            self.queue.as_ref(), self.device.as_ref(),
        ) else { return };

        // Upload object transforms
        for (i, mesh) in meshes.iter().enumerate().take(1024) {
            queue.write_buffer(obj_buf, i as u64 * self.shadow_obj_stride as u64,
                bytemuck::bytes_of(&ShadowObjUniforms { model: mesh.transform }));
        }

        // Compute per-light base layers + pre-upload all VP matrices
        let mut layer_bases: Vec<u32> = Vec::new();
        let mut face_matrices: Vec<Vec<Mat4>> = Vec::new();
        let mut cur_layer = 0u32;
        let mut global_face = 0u32;
        for cfg in &self.lights {
            layer_bases.push(cur_layer);
            let mats = Self::get_shadow_render_matrices(cfg);
            for mat in &mats {
                queue.write_buffer(vp_buf, global_face as u64 * self.shadow_vp_stride as u64,
                    bytemuck::bytes_of(&ShadowVpUniforms { view_proj: mat.to_cols_array_2d() }));
                global_face += 1;
            }
            cur_layer += mats.len() as u32;
            face_matrices.push(mats);
        }

        // Shadow bind group (dynamic offsets: vp at binding 0, obj at binding 1)
        let shad_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[bgl_uniform_dynamic(0, wgpu::ShaderStages::VERTEX), bgl_uniform_dynamic(1, wgpu::ShaderStages::VERTEX)],
        });
        let shad_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &shad_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: vp_buf, offset: 0, size: wgpu::BufferSize::new(std::mem::size_of::<ShadowVpUniforms>() as u64),
                })},
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: obj_buf, offset: 0, size: wgpu::BufferSize::new(std::mem::size_of::<ShadowObjUniforms>() as u64),
                })},
            ],
        });

        // One depth pass per face
        global_face = 0;
        for (light_i, mats) in face_matrices.iter().enumerate() {
            for (face_in_light, _) in mats.iter().enumerate() {
                let layer      = layer_bases[light_i] + face_in_light as u32;
                let vp_offset  = global_face * self.shadow_vp_stride;
                let face_view  = shadow_map.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    aspect: wgpu::TextureAspect::DepthOnly,
                    base_array_layer: layer, array_layer_count: Some(1),
                    ..Default::default()
                });
                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("rc_shadow"),
                        color_attachments: &[],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &face_view,
                            depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None, occlusion_query_set: None,
                    });
                    pass.set_pipeline(pipeline);
                    for (mesh_i, mesh) in meshes.iter().enumerate().take(1024) {
                        let obj_off = mesh_i as u32 * self.shadow_obj_stride;
                        pass.set_bind_group(0, &shad_bg, &[vp_offset, obj_off]);
                        pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                        pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                    }
                }
                global_face += 1;
            }
        }
    }

    fn pre_render_pass(&mut self, encoder: &mut wgpu::CommandEncoder, _ctx: &FeatureContext) {
        let (Some(pipeline), Some(bgl), Some(ubuf), Some(lbuf), Some(shad_view), Some(shad_samp), Some(device)) = (
            self.radiance_injection_pipeline.as_ref(), self.injection_bgl.as_ref(),
            self.uniforms_buffer.as_ref(), self.lighting_buffer.as_ref(),
            self.shadow_map_array_view.as_ref(), self.shadow_sampler.as_ref(),
            self.device.as_ref(),
        ) else { return };

        for i in 0..self.cascades.len() {
            let (Some(rv), Some(hv)) = (self.cascades[i].radiance_view.as_ref(), self.cascades[i].history_view.as_ref())
                else { continue };
            let Some(idx_buf) = self.cascade_index_bufs.get(i) else { continue };

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None, layout: bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(rv) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(hv) },
                    wgpu::BindGroupEntry { binding: 3, resource: lbuf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(shad_view) },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(shad_samp) },
                    wgpu::BindGroupEntry { binding: 6, resource: idx_buf.as_entire_binding() },
                ],
            });

            let res = self.cascades[i].probe_resolution;
            let xy  = (res + 7) / 8;
            let z   = (res + 3) / 4;
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("rc_injection"), timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(xy, xy, z);
            }
        }
    }

    fn shader_injections(&self) -> Vec<ShaderInjection> {
        vec![
            ShaderInjection::with_priority(ShaderInjectionPoint::FragmentPreamble,    include_str!("../shaders/gi_functions.wgsl"), 10),
            ShaderInjection::with_priority(ShaderInjectionPoint::FragmentColorCalculation, include_str!("../shaders/gi_sampling.wgsl"),  15),
        ]
    }

    fn main_pass_bind_group_layout(&self, device: &wgpu::Device) -> Option<(u32, wgpu::BindGroupLayout)> {
        Some((2, device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gi_frag_bgl"),
            entries: &[
                bgl_tex2darray(0, wgpu::ShaderStages::FRAGMENT),
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None,
                },
                bgl_uniform(2, wgpu::ShaderStages::FRAGMENT),
            ],
        })))
    }

    fn main_pass_bind_group(&self) -> Option<(u32, &wgpu::BindGroup)> {
        self.gi_bind_group.as_ref().map(|bg| (2u32, bg))
    }
}

// ======================== Helpers ========================

fn align_to(val: u32, alignment: u32) -> u32 { (val + alignment - 1) & !(alignment - 1) }

fn bgl_uniform(binding: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: vis,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
        count: None }
}
fn bgl_uniform_dynamic(binding: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: vis,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: true, min_binding_size: None },
        count: None }
}
fn bgl_storage_tex_write(binding: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: vis,
        ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba16Float, view_dimension: wgpu::TextureViewDimension::D2Array },
        count: None }
}
fn bgl_tex2darray(binding: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: vis,
        ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2Array, multisampled: false },
        count: None }
}

