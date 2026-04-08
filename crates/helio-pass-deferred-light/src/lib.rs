use bytemuck::{Pod, Zeroable};
use helio_v3::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DeferredGlobals {
    frame: u32,
    delta_time: f32,
    light_count: u32,
    ambient_intensity: f32,
    ambient_color: [f32; 4],
    rc_world_min: [f32; 4],
    rc_world_max: [f32; 4],
    csm_splits: [f32; 4],
    debug_mode: u32,
    _pad0: u32,
    /// Number of tiles in the X dimension for tiled light culling.
    num_tiles_x: u32,
    _pad2: u32,
}

pub struct DeferredLightPass {
    pipeline: wgpu::RenderPipeline,
    globals_buf: wgpu::Buffer,
    shadow_config_buf: wgpu::Buffer,
    bgl_0: wgpu::BindGroupLayout,
    bgl_1: wgpu::BindGroupLayout,
    bgl_2: wgpu::BindGroupLayout,
    bgl_3: wgpu::BindGroupLayout,
    bind_group_0: wgpu::BindGroup,
    bind_group_1: Option<wgpu::BindGroup>,
    bind_group_2: Option<wgpu::BindGroup>,
    bind_group_3: Option<wgpu::BindGroup>,
    bind_group_1_key: Option<(usize, usize, usize, usize, usize, usize)>,
    bind_group_2_key: Option<(usize, usize, usize, usize, usize, usize, usize)>,
    bind_group_3_key: Option<(usize, usize)>,
    width: u32,
    height: u32,
    fallback_tile_lists: wgpu::Buffer,
    fallback_tile_counts: wgpu::Buffer,
    pre_aa_texture: wgpu::Texture,
    pre_aa_view: wgpu::TextureView,
    pre_aa_format: wgpu::TextureFormat,
    fallback_shadow_view: wgpu::TextureView,
    fallback_static_shadow_view: wgpu::TextureView,
    fallback_shadow_sampler: wgpu::Sampler,
    shadow_depth_sampler: wgpu::Sampler,
    fallback_env_view: wgpu::TextureView,
    fallback_env_sampler: wgpu::Sampler,
    fallback_rc_view: wgpu::TextureView,
    fallback_caustics_view: wgpu::TextureView,
    caustics_sampler: wgpu::Sampler,
    fallback_water_volumes: wgpu::Buffer,
    /// 1×1 white R8Unorm fallback used when neither SSAO nor baked AO is available.
    fallback_ao_view: wgpu::TextureView,
    fallback_ao_sampler: wgpu::Sampler,
    /// 1×1 black Rgba16Float fallback used when baked lightmap is not available.
    fallback_lightmap_view: wgpu::TextureView,
    fallback_lightmap_sampler: wgpu::Sampler,
    /// 1×1 black Rg16Float fallback for lightmap UVs when not available.
    fallback_lightmap_uv_view: wgpu::TextureView,
    pub debug_mode: u32,
}

impl DeferredLightPass {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        camera_buf: &wgpu::Buffer,
        width: u32,
        height: u32,
        pre_aa_format: wgpu::TextureFormat,
    ) -> Self {
        let _height = height; // kept for future use (resize)

        // Fallback 1-entry storage buffers used when LightCullPass is absent.
        let fallback_tile_lists = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Deferred Fallback TileLists"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let fallback_tile_counts = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Deferred Fallback TileCounts"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Deferred Lighting Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/deferred_lighting.wgsl").into(),
            ),
        });

        let globals_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Deferred Globals"),
            size: std::mem::size_of::<DeferredGlobals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shadow_config_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shadow Config"),
            size: std::mem::size_of::<libhelio::ShadowConfig>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &shadow_config_buf,
            0,
            bytemuck::bytes_of(&libhelio::ShadowConfig::from_quality(
                libhelio::ShadowQuality::Medium,
            )),
        );

        let bgl_0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DeferredLight BGL0"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
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
        let bgl_1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DeferredLight BGL1"),
            entries: &[
                texture_entry(0, wgpu::TextureSampleType::Float { filterable: false }),
                texture_entry(1, wgpu::TextureSampleType::Float { filterable: false }),
                texture_entry(2, wgpu::TextureSampleType::Float { filterable: false }),
                texture_entry(3, wgpu::TextureSampleType::Float { filterable: false }),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Screen-space AO (SSAO result or pre-baked AO). Filterable so the
                // bilinear sampler can soften the AO at the edges of the screen.
                texture_entry(5, wgpu::TextureSampleType::Float { filterable: true }),
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Lightmap UVs from GBuffer (binding 7, Rg16Float)
                texture_entry(7, wgpu::TextureSampleType::Float { filterable: false }),
            ],
        });
        let bgl_2 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DeferredLight BGL2"),
            entries: &[
                storage_entry(0),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                storage_entry(4),
                texture_entry(5, wgpu::TextureSampleType::Float { filterable: false }),
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Water caustics texture
                texture_entry(8, wgpu::TextureSampleType::Float { filterable: true }),
                // Caustics sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Water volumes buffer
                storage_entry(10),
                // Static shadow atlas (cached, rendered only when Static/Stationary topology changes)
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                // Baked lightmap atlas texture
                texture_entry(12, wgpu::TextureSampleType::Float { filterable: true }),
                // Baked lightmap sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 13,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Group 3: tiled light culling results (tile_light_lists, tile_light_counts).
        // These are storage buffers written by LightCullPass and consumed here.
        let bgl_3 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DeferredLight BGL3"),
            entries: &[
                storage_entry(0), // tile_light_lists
                storage_entry(1), // tile_light_counts
            ],
        });

        let bind_group_0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DeferredLight BG0"),
            layout: &bgl_0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: globals_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: shadow_config_buf.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DeferredLight PL"),
            bind_group_layouts: &[Some(&bgl_0), Some(&bgl_1), Some(&bgl_2), Some(&bgl_3)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("DeferredLight Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: pre_aa_format,
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
        });

        let (pre_aa_texture, pre_aa_view) =
            color_texture(device, width, height, pre_aa_format, "Deferred PreAA");
        let (_fallback_shadow_tex, fallback_shadow_view) = fallback_shadow_texture(device);
        let (_fallback_static_shadow_tex, fallback_static_shadow_view) = fallback_shadow_texture(device);
        let fallback_shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Deferred Fallback Shadow Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        let shadow_depth_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Shadow Depth Sampler (PCSS)"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: None, // No comparison - returns actual depth values for PCSS blocker search
            ..Default::default()
        });
        let (fallback_env_texture, fallback_env_view) = black_cube_texture(device, queue);
        let fallback_env_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Deferred Env Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let (fallback_rc_texture, fallback_rc_view) =
            black_2d_texture(device, queue, "Deferred Fallback RC");

        // Fallback caustics texture (black 1x1 RGBA16Float)
        let (fallback_caustics_texture, fallback_caustics_view) =
            black_2d_texture(device, queue, "Deferred Fallback Caustics");

        // Caustics sampler
        let caustics_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Caustics Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        // Fallback water volumes buffer (empty)
        let fallback_water_volumes = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fallback Water Volumes"),
            size: 256, // Minimum size for empty buffer
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Fallback 1×1 white R8Unorm AO texture.
        // Used when neither SSAO nor pre-baked AO is available so the shader sees
        // AO = 1.0 (fully unoccluded) rather than undefined data.
        let fallback_ao_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Deferred Fallback AO"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &fallback_ao_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[255u8], // white = AO 1.0
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(1), rows_per_image: Some(1) },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        let fallback_ao_view = fallback_ao_tex.create_view(&Default::default());
        let fallback_ao_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Deferred Fallback AO Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Fallback 1×1 black Rgba16Float lightmap texture.
        // Used when baked lightmap is not available (no indirect lighting).
        let (fallback_lightmap_tex, fallback_lightmap_view) =
            black_2d_texture(device, queue, "Deferred Fallback Lightmap");
        let fallback_lightmap_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Deferred Fallback Lightmap Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Fallback 1×1 zero Rg16Float lightmap UV texture.
        // Used when lightmap UVs are not available from GBuffer.
        let fallback_lightmap_uv_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Deferred Fallback Lightmap UV"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &fallback_lightmap_uv_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[0u8; 4], // (0.0, 0.0) UV coords
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        let fallback_lightmap_uv_view = fallback_lightmap_uv_tex.create_view(&Default::default());

        Self {
            pipeline,
            globals_buf,
            shadow_config_buf,
            bgl_0,
            bgl_1,
            bgl_2,
            bgl_3,
            bind_group_0,
            bind_group_1: None,
            bind_group_2: None,
            bind_group_3: None,
            bind_group_1_key: None,
            bind_group_2_key: None,
            bind_group_3_key: None,
            width,
            height,
            fallback_tile_lists,
            fallback_tile_counts,
            pre_aa_texture,
            pre_aa_view,
            pre_aa_format,
            fallback_shadow_view,
            fallback_static_shadow_view,
            fallback_shadow_sampler,
            shadow_depth_sampler,
            fallback_env_view,
            fallback_env_sampler,
            fallback_rc_view,
            fallback_caustics_view,
            caustics_sampler,
            fallback_water_volumes,
            fallback_ao_view,
            fallback_ao_sampler,
            fallback_lightmap_view,
            fallback_lightmap_sampler,
            fallback_lightmap_uv_view,
            debug_mode: 0,
        }
    }

    /// Set the debug visualisation mode:
    /// - 0  = normal PBR lighting
    /// - 10 = shadow factor greyscale (white=lit, black=shadowed)
    /// - 11 = raw shadow atlas depth slice 0 (unmipped, linear)
    pub fn set_debug_mode(&mut self, mode: u32) {
        self.debug_mode = mode;
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let (texture, view) =
            color_texture(device, width, height, self.pre_aa_format, "Deferred PreAA");
        self.pre_aa_texture = texture;
        self.pre_aa_view = view;
        self.width = width;
        self.height = height;
    }

    /// Set shadow quality at runtime (zero CPU cost per frame, one-time buffer write).
    pub fn set_shadow_quality(&mut self, quality: libhelio::ShadowQuality, queue: &wgpu::Queue) {
        let config = libhelio::ShadowConfig::from_quality(quality);
        queue.write_buffer(&self.shadow_config_buf, 0, bytemuck::bytes_of(&config));
    }
}

impl RenderPass for DeferredLightPass {
    fn name(&self) -> &'static str {
        "DeferredLight"
    }

    fn publish<'a>(&'a self, frame: &mut libhelio::FrameResources<'a>) {
        if frame.pre_aa.is_none() {
            frame.pre_aa = Some(&self.pre_aa_view);
        }
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let main_scene = ctx.frame_resources.main_scene.as_ref();
        let (ambient_color, ambient_intensity) = if let Some(main_scene) = main_scene {
            (main_scene.ambient_color, main_scene.ambient_intensity)
        } else {
            ([0.5, 0.5, 0.6], 1.0) // Brighter fallback ambient: sky-blue tint
        };
        // Get RC bounds from frame resources (dual-tier GI: RC near, ambient far)
        let (rc_min, rc_max) = if let Some(main) = main_scene {
            (main.rc_world_min, main.rc_world_max)
        } else {
            ([0.0; 3], [0.0; 3]) // Fallback: RC disabled
        };

        let globals = DeferredGlobals {
            frame: ctx.frame_num as u32,
            delta_time: ctx.delta_time,
            light_count: ctx.scene.movable_light_count, // Only movable lights (static/stationary are baked)
            ambient_intensity,
            ambient_color: [ambient_color[0], ambient_color[1], ambient_color[2], 1.0],
            rc_world_min: [rc_min[0], rc_min[1], rc_min[2], 0.0],
            rc_world_max: [rc_max[0], rc_max[1], rc_max[2], 0.0],
            // Must match CSM_SPLITS constant in shadow_matrices.wgsl ([16,80,300,1400]).
            // The shadow matrices are computed for these distances, so cascade selection
            // must use the same values or shadow maps will be sampled outside their valid range.
            csm_splits: libhelio::CSM_SPLITS,
            debug_mode: self.debug_mode,
            _pad0: 0,
            num_tiles_x: self.width.div_ceil(16),
            _pad2: 0,
        };
        ctx.write_buffer(&self.globals_buf, 0, bytemuck::bytes_of(&globals));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let gbuffer = ctx.resources.gbuffer.as_ref().ok_or_else(|| {
            helio_v3::Error::InvalidPassConfig(
                "DeferredLight requires published gbuffer resources".to_string(),
            )
        })?;

        // Screen-space AO: use baked AO (via frame.ssao, which SsaoPass publishes as override
        // when a baked AO texture is present) or fall back to the 1×1 white texture.
        let ao_view = ctx.resources.ssao.unwrap_or(&self.fallback_ao_view);
        
        // Lightmap UVs from GBuffer
        let lightmap_uv_view = ctx.resources.gbuffer_lightmap_uv.unwrap_or(&self.fallback_lightmap_uv_view);

        let gbuffer_key = (
            gbuffer.albedo as *const _ as usize,
            gbuffer.normal as *const _ as usize,
            gbuffer.orm as *const _ as usize,
            gbuffer.emissive as *const _ as usize,
            ctx.depth as *const _ as usize,
            ao_view as *const _ as usize,
        );
        if self.bind_group_1_key != Some(gbuffer_key) {
            self.bind_group_1 = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("DeferredLight BG1"),
                layout: &self.bgl_1,
                entries: &[
                    texture_view_entry(0, gbuffer.albedo),
                    texture_view_entry(1, gbuffer.normal),
                    texture_view_entry(2, gbuffer.orm),
                    texture_view_entry(3, gbuffer.emissive),
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(ctx.depth),
                    },
                    // Screen-space AO (binding 5): SSAO or pre-baked AO from SsaoPass.publish()
                    texture_view_entry(5, ao_view),
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::Sampler(&self.fallback_ao_sampler),
                    },
                    // Lightmap UVs from GBuffer (binding 7)
                    texture_view_entry(7, lightmap_uv_view),
                ],
            }));
            self.bind_group_1_key = Some(gbuffer_key);
        }

        let shadow_view = ctx.resources.shadow_atlas.unwrap_or(&self.fallback_shadow_view);
        let static_shadow_view = ctx.resources.static_shadow_atlas.unwrap_or(&self.fallback_static_shadow_view);
        let shadow_sampler = ctx
            .resources
            .shadow_sampler
            .unwrap_or(&self.fallback_shadow_sampler);
        let rc_view = &self.fallback_rc_view;
        let env_view = &self.fallback_env_view;
        
        // Baked lightmap atlas from bake inject pass
        let lightmap_view = ctx.resources.baked_lightmap.unwrap_or(&self.fallback_lightmap_view);
        let lightmap_sampler = ctx.resources.baked_lightmap_sampler.unwrap_or(&self.fallback_lightmap_sampler);
        
        let scene_key = (
            ctx.scene.lights as *const _ as usize,
            shadow_view as *const _ as usize,
            static_shadow_view as *const _ as usize,
            shadow_sampler as *const _ as usize,
            env_view as *const _ as usize,
            ctx.scene.shadow_matrices as *const _ as usize,
            rc_view as *const _ as usize,
        );
        if self.bind_group_2_key != Some(scene_key) {
            self.bind_group_2 = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("DeferredLight BG2"),
                layout: &self.bgl_2,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ctx.scene.lights.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(shadow_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(shadow_sampler),
                    },
                    texture_view_entry(3, env_view),
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: ctx.scene.shadow_matrices.as_entire_binding(),
                    },
                    texture_view_entry(5, rc_view),
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::Sampler(&self.fallback_env_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::Sampler(&self.shadow_depth_sampler),
                    },
                    // Water caustics texture (binding 8)
                    texture_view_entry(8, ctx.resources.water_caustics.unwrap_or(&self.fallback_caustics_view)),
                    // Caustics sampler (binding 9)
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: wgpu::BindingResource::Sampler(&self.caustics_sampler),
                    },
                    // Water volumes buffer (binding 10)
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: ctx.resources.water_volumes.unwrap_or(&self.fallback_water_volumes).as_entire_binding(),
                    },
                    // Static shadow atlas (binding 11) — cached, only changes with Static topology
                    texture_view_entry(11, static_shadow_view),
                    // Baked lightmap atlas (binding 12)
                    texture_view_entry(12, lightmap_view),
                    // Baked lightmap sampler (binding 13)
                    wgpu::BindGroupEntry {
                        binding: 13,
                        resource: wgpu::BindingResource::Sampler(lightmap_sampler),
                    },
                ],
            }));
            self.bind_group_2_key = Some(scene_key);
        }

        // ── Bind group 3: tile light culling results ──────────────────────────
        let tile_lists   = ctx.resources.tile_light_lists.unwrap_or(&self.fallback_tile_lists);
        let tile_counts  = ctx.resources.tile_light_counts.unwrap_or(&self.fallback_tile_counts);
        let tile_key = (tile_lists as *const _ as usize, tile_counts as *const _ as usize);
        if self.bind_group_3_key != Some(tile_key) {
            self.bind_group_3 = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("DeferredLight BG3"),
                layout: &self.bgl_3,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: tile_lists.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: tile_counts.as_entire_binding() },
                ],
            }));
            self.bind_group_3_key = Some(tile_key);
        }

        let load_op = if ctx.resources.sky_lut.is_some() {
            wgpu::LoadOp::Load
        } else {
            wgpu::LoadOp::Clear(wgpu::Color::BLACK)
        };

        let pre_aa_view = ctx.resources.pre_aa.unwrap_or(&self.pre_aa_view);
        let color_attachments = [Some(wgpu::RenderPassColorAttachment {
            view: pre_aa_view,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: load_op,
                store: wgpu::StoreOp::Store,
            },
        })];
        let desc = wgpu::RenderPassDescriptor {
            label: Some("DeferredLight"),
            color_attachments: &color_attachments,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        };
        let mut pass = ctx.begin_render_pass(&desc);
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group_0, &[]);
        pass.set_bind_group(1, self.bind_group_1.as_ref().unwrap(), &[]);
        pass.set_bind_group(2, self.bind_group_2.as_ref().unwrap(), &[]);
        pass.set_bind_group(3, self.bind_group_3.as_ref().unwrap(), &[]);
        pass.draw(0..3, 0..1);
        Ok(())
    }
}

fn storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn texture_entry(binding: u32, sample_type: wgpu::TextureSampleType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn texture_view_entry<'a>(binding: u32, view: &'a wgpu::TextureView) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(view),
    }
}

fn color_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn fallback_shadow_texture(device: &wgpu::Device) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Deferred Fallback Shadow"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    (texture, view)
}

fn black_2d_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let zero = [0u8; 8];
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &zero,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(8),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn black_cube_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Deferred Fallback Env Cube"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 6,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let zero = [0u8; 8 * 6];
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &zero,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(8),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 6,
        },
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    });
    (texture, view)
}

