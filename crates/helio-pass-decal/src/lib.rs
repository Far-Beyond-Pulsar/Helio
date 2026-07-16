use helio_core::graph::ResourceBuilder;
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};
use std::sync::Arc;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DecalGlobals { decal_count: u32, _pad0: u32, _pad1: u32, _pad2: u32 }

pub struct DecalPass {
    collect_pipeline: wgpu::ComputePipeline,
    apply_pipeline: wgpu::ComputePipeline,
    bgl_collect: wgpu::BindGroupLayout,
    bgl_apply: wgpu::BindGroupLayout,
    bg_collect: Option<wgpu::BindGroup>,
    bg_apply: Option<wgpu::BindGroup>,
    bg_collect_key: Option<(usize, usize, usize, usize, usize, usize, u64)>,
    bg_apply_key: Option<(usize, usize, usize, usize, usize, usize, u64)>,
    globals_buf: wgpu::Buffer,
    temp_albedo: Option<(wgpu::Texture, wgpu::TextureView)>,
    temp_normal: Option<(wgpu::Texture, wgpu::TextureView)>,
    temp_orm: Option<(wgpu::Texture, wgpu::TextureView)>,
    temp_emissive: Option<(wgpu::Texture, wgpu::TextureView)>,
    device: Arc<wgpu::Device>,
    fallback_samp: wgpu::Sampler,
    last_w: u32, last_h: u32,
}

impl DecalPass {
    pub fn new(device: &wgpu::Device, _queue: &wgpu::Queue, decal_buf: &wgpu::Buffer,
               camera_buf: &wgpu::Buffer, _w: u32, _h: u32) -> Self {
        let collect_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Decal Collect"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/decal_collect.wgsl").into()),
        });
        let apply_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Decal Apply"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/decal_apply.wgsl").into()),
        });
        let globals_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DecalGlobals"), size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // BGL collect: camera(0) + globals(1) + decals(2) + depth(3) + gbuf(4-7) + temp_out(8-11)
        let bgl_collect = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Decal Collect BGL"),
            entries: &[
                bgl_entry_buf(0, wgpu::BufferBindingType::Uniform),
                bgl_entry_buf(1, wgpu::BufferBindingType::Uniform),
                bgl_entry_buf(2, wgpu::BufferBindingType::Storage { read_only: true }),
                bgl_entry_tex(3, wgpu::TextureSampleType::Depth),
                bgl_entry_tex(4, wgpu::TextureSampleType::Float { filterable: false }),
                bgl_entry_tex(5, wgpu::TextureSampleType::Float { filterable: false }),
                bgl_entry_tex(6, wgpu::TextureSampleType::Float { filterable: false }),
                bgl_entry_tex(7, wgpu::TextureSampleType::Float { filterable: false }),
                bgl_entry_tex_storage(8, wgpu::StorageTextureAccess::WriteOnly, wgpu::TextureFormat::Rgba8Unorm),
                bgl_entry_tex_storage(9, wgpu::StorageTextureAccess::WriteOnly, wgpu::TextureFormat::Rgba16Float),
                bgl_entry_tex_storage(10, wgpu::StorageTextureAccess::WriteOnly, wgpu::TextureFormat::Rgba8Unorm),
                bgl_entry_tex_storage(11, wgpu::StorageTextureAccess::WriteOnly, wgpu::TextureFormat::Rgba16Float),
                bgl_entry_tex(12, wgpu::TextureSampleType::Float { filterable: true }),
                wgpu::BindGroupLayoutEntry {
                    binding: 13, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let collect_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Decal Collect PL"), bind_group_layouts: &[Some(&bgl_collect)], immediate_size: 0,
        });
        let collect_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Decal Collect"), layout: Some(&collect_pl), module: &collect_mod,
            entry_point: Some("cs_main"), compilation_options: Default::default(), cache: None,
        });

        // BGL apply: camera(0) + globals(1) + decals(2) + temp_in(3-6) + gbuf_out(7-10 storage)
        let bgl_apply = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Decal Apply BGL"),
            entries: &[
                bgl_entry_buf(0, wgpu::BufferBindingType::Uniform),
                bgl_entry_buf(1, wgpu::BufferBindingType::Uniform),
                bgl_entry_buf(2, wgpu::BufferBindingType::Storage { read_only: true }),
                bgl_entry_tex(3, wgpu::TextureSampleType::Float { filterable: false }),
                bgl_entry_tex(4, wgpu::TextureSampleType::Float { filterable: false }),
                bgl_entry_tex(5, wgpu::TextureSampleType::Float { filterable: false }),
                bgl_entry_tex(6, wgpu::TextureSampleType::Float { filterable: false }),
                bgl_entry_tex_storage(7, wgpu::StorageTextureAccess::WriteOnly, wgpu::TextureFormat::Rgba8Unorm),
                bgl_entry_tex_storage(8, wgpu::StorageTextureAccess::WriteOnly, wgpu::TextureFormat::Rgba16Float),
                bgl_entry_tex_storage(9, wgpu::StorageTextureAccess::WriteOnly, wgpu::TextureFormat::Rgba8Unorm),
                bgl_entry_tex_storage(10, wgpu::StorageTextureAccess::WriteOnly, wgpu::TextureFormat::Rgba16Float),
            ],
        });
        let apply_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Decal Apply PL"), bind_group_layouts: &[Some(&bgl_apply)], immediate_size: 0,
        });
        let apply_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Decal Apply"), layout: Some(&apply_pl), module: &apply_mod,
            entry_point: Some("cs_main"), compilation_options: Default::default(), cache: None,
        });

        Self {
            collect_pipeline, apply_pipeline, bgl_collect, bgl_apply,
            bg_collect: None, bg_apply: None, bg_collect_key: None, bg_apply_key: None,
            globals_buf, temp_albedo: None, temp_normal: None, temp_orm: None, temp_emissive: None,
            device: Arc::new(device.clone()),
            fallback_samp: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("Decal Fallback Sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                ..Default::default()
            }),
            last_w: 0, last_h: 0,
        }
    }

    fn ensure_temp(&mut self, w: u32, h: u32, device: &wgpu::Device) {
        if w == self.last_w && h == self.last_h && self.temp_albedo.is_some() { return; }
        self.last_w = w; self.last_h = h;
        let size = wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 };
        self.temp_albedo = Some(make_temp(device, wgpu::TextureFormat::Rgba8Unorm, "DecalTemp_Albedo", size));
        self.temp_normal = Some(make_temp(device, wgpu::TextureFormat::Rgba16Float, "DecalTemp_Normal", size));
        self.temp_orm = Some(make_temp(device, wgpu::TextureFormat::Rgba8Unorm, "DecalTemp_ORM", size));
        self.temp_emissive = Some(make_temp(device, wgpu::TextureFormat::Rgba16Float, "DecalTemp_Emissive", size));
        self.bg_collect = None; self.bg_apply = None;
    }
}

fn make_temp(device: &wgpu::Device, format: wgpu::TextureFormat, label: &str, size: wgpu::Extent3d) -> (wgpu::Texture, wgpu::TextureView) {
    let t = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label), size, mip_level_count: 1, sample_count: 1,
        dimension: wgpu::TextureDimension::D2, format,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let v = t.create_view(&Default::default());
    (t, v)
}

fn bgl_entry_buf(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
        count: None }
}
fn bgl_entry_tex(binding: u32, sample_type: wgpu::TextureSampleType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture { sample_type, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
        count: None }
}
fn bgl_entry_tex_storage(binding: u32, access: wgpu::StorageTextureAccess, format: wgpu::TextureFormat) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture { access, format, view_dimension: wgpu::TextureViewDimension::D2 },
        count: None }
}

impl RenderPass for DecalPass {
    fn name(&self) -> &'static str { "DecalApply" }
    fn declare_resources(&self, builder: &mut ResourceBuilder) { builder.read("gbuffer"); builder.read("depth"); }
    fn publish<'a>(&'a self, _: &mut libhelio::FrameResources<'a>) {}
    fn render_pass_descriptor<'a>(&'a self, _: &'a wgpu::TextureView, _: &'a wgpu::TextureView, _: &'a libhelio::FrameResources<'a>) -> Option<wgpu::RenderPassDescriptor<'a>> { None }
    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        ctx.write_buffer(&self.globals_buf, 0, bytemuck::bytes_of(&DecalGlobals { decal_count: ctx.scene.decals.len() as u32, _pad0: 0, _pad1: 0, _pad2: 0 }));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        if ctx.scene.decal_count == 0 { return Ok(()); }
        let gb = match ctx.resources.gbuffer.read(self.name()) { Some(g) => g, None => return Ok(()) };
        let dv = ctx.depth;
        let camera_ptr = ctx.scene.camera as *const _ as usize;
        let decal_ptr = ctx.scene.decals as *const _ as usize;
        self.ensure_temp(ctx.width, ctx.height, ctx.device);
        let (_, ta) = self.temp_albedo.as_ref().unwrap();
        let (_, tn) = self.temp_normal.as_ref().unwrap();
        let (_, to) = self.temp_orm.as_ref().unwrap();
        let (_, te) = self.temp_emissive.as_ref().unwrap();

        let ck = (camera_ptr, decal_ptr, dv as *const _ as usize, gb.albedo as *const _ as usize,
                   gb.normal as *const _ as usize, gb.orm as *const _ as usize,
                   u64::from(self.last_w) | (u64::from(self.last_h) << 32));
        if self.bg_collect_key != Some(ck) {
            // Find a decal texture view + sampler (use first scene texture or a fallback)
            let (dtv, dts) = ctx.resources.main_scene.read(self.name())
                .and_then(|ms| {
                    let tex = ms.material_textures.texture_views.first().copied();
                    let samp = ms.material_textures.samplers.first().copied();
                    tex.zip(samp)
                })
                .unwrap_or((gb.albedo, &self.fallback_samp));
            self.bg_collect = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Decal Collect BG"), layout: &self.bgl_collect,
                entries: &[
                    bind_buf(0, ctx.scene.camera), bind_buf(1, &self.globals_buf),
                    bind_buf(2, ctx.scene.decals), bind_tex(3, dv),
                    bind_tex(4, gb.albedo), bind_tex(5, gb.normal), bind_tex(6, gb.orm), bind_tex(7, gb.emissive),
                    bind_tex(8, ta), bind_tex(9, tn), bind_tex(10, to), bind_tex(11, te),
                    bind_tex(12, dtv), wgpu::BindGroupEntry { binding: 13, resource: wgpu::BindingResource::Sampler(dts) },
                ],
            }));
            self.bg_collect_key = Some(ck);
        }

        {
            let mut cp = unsafe { &mut *ctx.encoder_ptr }
                .begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("DecalCollect"), timestamp_writes: None });
            cp.set_pipeline(&self.collect_pipeline);
            cp.set_bind_group(0, self.bg_collect.as_ref().unwrap(), &[]);
            cp.dispatch_workgroups((ctx.width + 15) / 16, (ctx.height + 15) / 16, 1);
        }

        let ak = (camera_ptr, decal_ptr, ta as *const _ as usize, tn as *const _ as usize,
                   to as *const _ as usize, te as *const _ as usize,
                   u64::from(self.last_w) | (u64::from(self.last_h) << 32));
        if self.bg_apply_key != Some(ak) {
            self.bg_apply = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Decal Apply BG"), layout: &self.bgl_apply,
                entries: &[
                    bind_buf(0, ctx.scene.camera), bind_buf(1, &self.globals_buf),
                    bind_buf(2, ctx.scene.decals),
                    bind_tex(3, ta), bind_tex(4, tn), bind_tex(5, to), bind_tex(6, te),
                    bind_tex(7, gb.albedo), bind_tex(8, gb.normal), bind_tex(9, gb.orm), bind_tex(10, gb.emissive),
                ],
            }));
            self.bg_apply_key = Some(ak);
        }

        {
            let mut cp = unsafe { &mut *ctx.encoder_ptr }
                .begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("DecalApply"), timestamp_writes: None });
            cp.set_pipeline(&self.apply_pipeline);
            cp.set_bind_group(0, self.bg_apply.as_ref().unwrap(), &[]);
            cp.dispatch_workgroups((ctx.width + 15) / 16, (ctx.height + 15) / 16, 1);
        }

        Ok(())
    }

    fn reads(&self) -> &'static [&'static str] { &["gbuffer", "depth"] }
    fn writes(&self) -> &'static [&'static str] { &["gbuffer"] }
}

fn bind_buf<'a>(binding: u32, buf: &'a wgpu::Buffer) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry { binding, resource: buf.as_entire_binding() }
}
fn bind_tex<'a>(binding: u32, view: &'a wgpu::TextureView) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry { binding, resource: wgpu::BindingResource::TextureView(view) }
}
