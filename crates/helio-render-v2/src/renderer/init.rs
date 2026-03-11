//! Renderer construction, material creation, and lighting bind group helpers.

use super::*;
use super::uniforms::{GlobalsUniform, SkyUniform};
use super::helpers::{create_depth_texture, create_gbuffer_textures, create_gbuffer_bind_group};
use super::shadow_math::CSM_SPLITS;
use crate::features::lighting::MAX_LIGHTS;
use crate::buffer_pool::GpuBufferPool;
use crate::gpu_scene::MaterialRange;
use crate::passes::ShadowMatrixPass;

impl Renderer {
    /// Create a new renderer
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        config: RendererConfig,
    ) -> Result<Self> {
        log::trace!("Creating Helio Render V2");
        log::trace!("  Surface format: {:?}", config.surface_format);
        log::trace!("  Resolution: {}x{}", config.width, config.height);

        let mut resources = ResourceManager::new(device.clone());
        let lighting_layout = resources.bind_group_layouts.lighting.clone();
        let bind_group_layouts = Arc::new(resources.bind_group_layouts.clone());
        let mut pipelines = PipelineCache::new(device.clone(), bind_group_layouts.clone(), config.surface_format);
        let mut graph = RenderGraph::new();
        let mut features = config.features;

        // ── Uniform buffers ──────────────────────────────────────────────────
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera Uniform Buffer"),
            size: std::mem::size_of::<Camera>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let globals_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Globals Uniform Buffer"),
            size: std::mem::size_of::<GlobalsUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // GpuScene must be created BEFORE global_bind_group because binding 2
        // needs the instance buffer.
        let gpu_scene = GpuScene::new(&device);

        let global_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Global Bind Group"),
            layout: &resources.bind_group_layouts.global,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: camera_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: globals_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: gpu_scene.instance_buffer().as_entire_binding() },
            ],
        });

        // ── GPU-driven infrastructure ─────────────────────────────────────────
        // Unified geometry pool — all pool-allocated meshes share these buffers.
        let buffer_pool = GpuBufferPool::new(device.clone());
        let pool_vb = buffer_pool.shared_vertex_buffer.clone();
        let pool_ib = buffer_pool.shared_index_buffer.clone();

        // IndirectDispatchPass is driven from renderer.render() — NOT added to graph.
        let indirect_dispatch = IndirectDispatchPass::new();

        // ShadowMatrixPass: GPU-driven shadow matrix computation (runs before shadow pass).
        let shadow_matrix_pass = ShadowMatrixPass::new(&device)?;

        // Shared Arcs: written from render() after dispatch, read by geometry passes.
        let shared_indirect_buf: Arc<Mutex<Option<Arc<wgpu::Buffer>>>> =
            Arc::new(Mutex::new(None));
        let shared_material_ranges: Arc<Mutex<Vec<MaterialRange>>> =
            Arc::new(Mutex::new(Vec::new()));

        // ── Geometry pass — must be added to graph BEFORE features so it ────────
        // ── executes first (billboard/post passes render on top).           ────────
        let draw_list: Arc<Mutex<Vec<DrawCall>>> = Arc::new(Mutex::new(Vec::new()));
        let debug_shapes: Arc<Mutex<Vec<DebugShape>>> = Arc::new(Mutex::new(Vec::new()));
        let debug_batch: Arc<Mutex<Option<DebugDrawBatch>>> = Arc::new(Mutex::new(None));

        // Shadow matrix buffer: MAX_LIGHTS × 6 faces × mat4x4<f32>
        let shadow_matrix_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shadow Matrix Buffer"),
            size: (MAX_LIGHTS as u64) * 6 * 64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // GPU-driven shadow matrix computation: per-light dirty flags (u32)
        let shadow_dirty_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shadow Dirty Flags Buffer"),
            size: (MAX_LIGHTS as u64) * 4, // u32 per light
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // GPU-driven shadow matrix computation: per-light FNV hashes (u32)
        let shadow_hash_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shadow Hash Buffer"),
            size: (MAX_LIGHTS as u64) * 4, // u32 per light
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // Shared light count — updated each frame; read by ShadowPass
        let light_count_arc = Arc::new(AtomicU32::new(0));
        // Per-light face counts — updated each frame; read by ShadowPass
        let light_face_counts: Arc<std::sync::Mutex<Vec<u8>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        // Per-light cull data (position, range, type) — updated each frame; read by ShadowPass
        let shadow_cull_lights: Arc<std::sync::Mutex<Vec<ShadowCullLight>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));

        // ── Sky pass (added FIRST so it runs before geometry) ─────────────────
        let sky_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sky Uniform Buffer"),
            size: std::mem::size_of::<SkyUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Sky-View LUT: small panoramic Rgba16Float texture the SkyLutPass writes
        // into every frame. The main SkyPass samples it instead of running the
        // full atmosphere ray-march per pixel (~46× cheaper at 1280×720).
        let sky_lut_tex = device.create_texture(&wgpu::TextureDescriptor {
            label:           Some("Sky View LUT"),
            size:            wgpu::Extent3d { width: SKY_LUT_W, height: SKY_LUT_H, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count:    1,
            dimension:       wgpu::TextureDimension::D2,
            format:          SKY_LUT_FORMAT,
            usage:           wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats:    &[],
        });
        let sky_lut_view = Arc::new(sky_lut_tex.create_view(&wgpu::TextureViewDescriptor::default()));
        let sky_lut_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label:     Some("Sky LUT Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::Repeat, // wrap azimuth
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        // Uniform-only bind group for the LUT pass (no LUT texture – it produces it)
        let sky_uniform_bg = Arc::new(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("Sky Uniform Bind Group"),
            layout:  &resources.bind_group_layouts.sky_uniform,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: sky_uniform_buffer.as_entire_binding() },
            ],
        }));

        // Full sky bind group (uniform + LUT texture + sampler) for the SkyPass
        let sky_bind_group = Arc::new(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("Sky Bind Group"),
            layout:  &resources.bind_group_layouts.sky,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: sky_uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&sky_lut_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sky_lut_sampler) },
            ],
        }));

        // SkyLutPass: renders atmosphere into the small LUT each frame
        let sky_lut_pass = SkyLutPass::new(
            &device,
            &resources.bind_group_layouts.sky_uniform,
            &resources.bind_group_layouts.global,
            sky_uniform_bg,
            sky_lut_view,
        );
        graph.add_pass(sky_lut_pass);

        let sky_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sky Pipeline Layout"),
            bind_group_layouts: &[
                Some(resources.bind_group_layouts.global.as_ref()),
                Some(resources.bind_group_layouts.sky.as_ref()),
            ],
            immediate_size: 0,
        });
        let sky_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sky Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/passes/sky.wgsl").into()),
        });
        let sky_pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Sky Pipeline"),
            layout: Some(&sky_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &sky_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &sky_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        }));

        let sky_pass = SkyPass::new(sky_pipeline, sky_bind_group.clone());
        graph.add_pass(sky_pass);

        // ── Deferred: depth + G-buffer textures (created before graph build) ────
        let defines = features.collect_shader_defines();

        // Depth texture with TEXTURE_BINDING so the deferred lighting pass can sample it.
        let (depth_texture, depth_view, depth_sample_view) =
            create_depth_texture(&device, config.width, config.height);

        // Four G-buffer render targets.
        let (gbuf_albedo_texture, gbuf_normal_texture, gbuf_orm_texture, gbuf_emissive_texture, gbuf_targets) =
            create_gbuffer_textures(&device, config.width, config.height);
        let gbuffer_targets = Arc::new(Mutex::new(gbuf_targets));

        // ── GPU-driven indirect dispatch pass ──────────────────────────────────
        // NOTE: IndirectDispatchPass is NOT added to the graph — it is driven
        // manually from render() before graph.execute() so the indirect buffer
        // is ready when geometry passes run.

        // ── Default material views + bind group (needed before pass construction) ──
        let default_material_views = DefaultMaterialViews::new(&device, &queue);

        let mat_uniform = MaterialUniform {
            base_color: [1.0, 1.0, 1.0, 1.0],
            metallic: 0.0, roughness: 0.5, emissive_factor: 0.0, ao: 1.0,
            emissive_color: [0.0; 3], alpha_cutoff: 0.0,
        };
        let material_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Default Material Uniform"),
            contents: bytemuck::bytes_of(&mat_uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let default_material_bind_group = Arc::new(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Default Material Bind Group"),
            layout: &resources.bind_group_layouts.material,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: material_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&default_material_views.white_srgb) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&default_material_views.flat_normal) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&default_material_views.sampler) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&default_material_views.white_orm) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&default_material_views.black_emissive) },
            ],
        }));

        // Depth prepass (early-Z rejection before GBuffer)
        let depth_pipeline = pipelines.get_or_create(
            include_str!("../../shaders/passes/gbuffer.wgsl"),
            "depth_prepass".to_string(),
            &defines,
            PipelineVariant::DepthOnly,
        )?;
        let has_multi_draw = device.features().contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT);

        let depth_prepass = DepthPrepassPass::new(
            depth_pipeline.clone(),
            pool_vb.clone(),
            pool_ib.clone(),
            shared_indirect_buf.clone(),
            shared_material_ranges.clone(),
            default_material_bind_group.clone(),
            has_multi_draw,
        );
        graph.add_pass(depth_prepass);

        // G-buffer write pass (replaces forward GeometryPass)
        let gbuf_pipeline = pipelines.get_or_create(
            include_str!("../../shaders/passes/gbuffer.wgsl"),
            "gbuffer".to_string(),
            &defines,
            PipelineVariant::GBufferWrite,
        )?;
        let gbuffer_pass = GBufferPass::new(
            gbuffer_targets.clone(),
            gbuf_pipeline.clone(),
            pool_vb.clone(),
            pool_ib.clone(),
            shared_indirect_buf.clone(),
            shared_material_ranges.clone(),
            has_multi_draw,
        );
        graph.add_pass(gbuffer_pass);

        // ── Register features (adds BillboardPass etc. after GeometryPass) ───────
        let shared_shadow_draw_call_buf: Arc<Mutex<Option<Arc<wgpu::Buffer>>>> =
            Arc::new(Mutex::new(None));
        let (feat_light_buf, feat_shadow_view, feat_shadow_sampler, feat_rc_view, feat_rc_bounds) = {
            let mut ctx = FeatureContext::new(
                &device, &queue, &mut graph, &mut resources, config.surface_format,
                device.clone(),
                queue.clone(),
                draw_list.clone(),
                shadow_matrix_buffer.clone(),
                light_count_arc.clone(),
                light_face_counts.clone(),
                shadow_cull_lights.clone(),
                gpu_scene.instance_buffer(),
                pool_vb.clone(),
                pool_ib.clone(),
                shared_shadow_draw_call_buf.clone(),
                shared_material_ranges.clone(),
                has_multi_draw,
            );
            features.register_all(&mut ctx)?;
            (ctx.light_buffer, ctx.shadow_atlas_view, ctx.shadow_sampler,
             ctx.rc_cascade0_view, ctx.rc_world_bounds)
        };

        // ── Lighting bind group (group 2) ────────────────────────────────────
        // Fallback null light buffer when LightingFeature is not registered
        let null_light_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Null Light Buffer"),
            size: 64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Fallback 1-layer shadow atlas
        let default_shadow_atlas = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Default Shadow Atlas"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let default_shadow_atlas_view = default_shadow_atlas.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        let default_comparison_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Default Comparison Sampler"),
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        // Fallback black cube (env map)
        let cube_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Default Env Cube"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 6 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        for face in 0..6u32 {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &cube_tex, mip_level: 0,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: face },
                    aspect: wgpu::TextureAspect::All,
                },
                &[0u8, 0, 0, 255],
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) },
                wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            );
        }
        let cube_view = cube_tex.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
            array_layer_count: Some(6),
            ..Default::default()
        });

        // Pick the real resources if features provided them, otherwise use defaults
        let light_buf     = feat_light_buf.unwrap_or_else(|| Arc::new(null_light_buf));
        let shadow_view   = feat_shadow_view.unwrap_or_else(|| Arc::new(default_shadow_atlas_view));
        let shadow_samp   = feat_shadow_sampler.unwrap_or_else(|| Arc::new(default_comparison_sampler));

        // Fallback 1×1 black texture for RC when feature not registered.  Use
        // 8‑bit on wasm where 16‑bit float support can be flaky.
        let rc_format = if cfg!(target_arch = "wasm32") {
            wgpu::TextureFormat::Rgba8Unorm
        } else {
            wgpu::TextureFormat::Rgba16Float
        };
        let default_rc_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Default RC Cascade 0"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: rc_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let default_rc_view = default_rc_tex.create_view(&Default::default());
        if feat_rc_view.is_some() {
            log::trace!("RC feature provided valid output texture view");
        } else {
            log::warn!("RC feature output view not set - using dummy black 1x1 texture!");
        }
        let rc_view = feat_rc_view.unwrap_or_else(|| Arc::new(default_rc_view));
        let rc_world_min = feat_rc_bounds.map(|(mn, _)| mn).unwrap_or([0.0; 3]);
        let rc_world_max = feat_rc_bounds.map(|(_, mx)| mx).unwrap_or([0.0; 3]);

        // Linear sampler for env-IBL reads in the deferred lighting shader (group 2 binding 6).
        let env_linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Env Linear Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        let lighting_bind_group = Arc::new(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Lighting Bind Group"),
            layout: &resources.bind_group_layouts.lighting,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: light_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&shadow_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&shadow_samp) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&cube_view) },
                wgpu::BindGroupEntry { binding: 4, resource: shadow_matrix_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&rc_view) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&env_linear_sampler) },
            ],
        }));

        // ── Deferred lighting pass (fullscreen PBR over the G-buffer) ─────────
        let gbuf_bg_inner = create_gbuffer_bind_group(
            &device,
            &resources.bind_group_layouts.gbuffer_read,
            &*gbuffer_targets.lock().unwrap(),
            &depth_sample_view,
        );
        let deferred_bg = Arc::new(Mutex::new(Arc::new(gbuf_bg_inner)));

        let deferred_pipeline = pipelines.get_or_create(
            include_str!("../../shaders/passes/deferred_lighting.wgsl"),
            "deferred_lighting".to_string(),
            &defines,
            PipelineVariant::DeferredLighting,
        )?;
        let deferred_pass = DeferredLightingPass::new(deferred_bg.clone(), deferred_pipeline);
        graph.add_pass(deferred_pass);

        // ── Transparent pass (forward blended, depth read-only) ───────────────
        let transparent_pipeline = pipelines.get_or_create(
            include_str!("../../shaders/passes/geometry.wgsl"),
            "transparent_forward".to_string(),
            &defines,
            PipelineVariant::TransparentForward,
        )?;
        let transparent_pass = TransparentPass::new(transparent_pipeline, draw_list.clone(), pool_vb.clone(), pool_ib.clone());
        graph.add_pass(transparent_pass);

        // ── Debug draw pass (overlay after deferred + feature passes) ──────────
        let debug_draw_pass = DebugDrawPass::new(
            &device,
            &resources.bind_group_layouts.global,
            config.surface_format,
            debug_batch.clone(),
        );
        graph.add_pass(debug_draw_pass);

        // Build the render graph
        graph.build()?;

        // Cache pass names once — avoids a per-frame Vec<String> allocation when the
        // portal is open.  Pass order never changes after build().
        let cached_pass_names = graph.execution_pass_names();

        // Create profiler if TIMESTAMP_QUERY was requested on this device
        let profiler = GpuProfiler::new(&device, &queue);

        // Create pre-AA texture for post-processing
        let pre_aa_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Pre-AA Texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: config.surface_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let pre_aa_view = pre_aa_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create AA passes based on config
        let (fxaa_pass, smaa_pass, taa_pass) = match config.aa_mode {
            AntiAliasingMode::Fxaa => {
                log::trace!("AA: FXAA enabled");
                let pass = FxaaPass::new(&device, config.surface_format);
                (Some(pass), None, None)
            }
            AntiAliasingMode::Smaa => {
                log::trace!("AA: SMAA enabled");
                let pass = SmaaPass::new(&device, &queue, config.surface_format);
                (None, Some(pass), None)
            }
            AntiAliasingMode::Taa => {
                log::trace!("AA: TAA enabled");
                let pass = TaaPass::new(&device, config.surface_format, config.taa_config);
                (None, None, Some(pass))
            }
            AntiAliasingMode::Msaa(samples) => {
                log::trace!("AA: MSAA {:?} enabled", samples);
                (None, None, None)
            }
            AntiAliasingMode::None => {
                (None, None, None)
            }
        };

        // Cache AA bind groups; update only when source views change (e.g. resize).
        let fxaa_bind_group = fxaa_pass
            .as_ref()
            .map(|p| p.create_bind_group(&device, &pre_aa_view));
        let smaa_bind_group = smaa_pass
            .as_ref()
            .map(|p| p.create_bind_group(&device, &pre_aa_view));
        let taa_bind_group = taa_pass
            .as_ref()
            .map(|p| p.create_bind_group(&device, &pre_aa_view, &depth_view));

        log::trace!("Helio Render V2 initialized successfully");
        log::info!("GPU-driven rendering active: frustum culling via compute, indirect draws, storage buffer transforms");

        let gpu_light_scene = gpu_light_scene::GpuLightScene::new(
            light_buf.clone(),
            shadow_matrix_buffer.clone(),
            shadow_dirty_buffer.clone(),
            shadow_hash_buffer.clone(),
        );

        Ok(Self {
            device,
            queue,
            resources,
            graph,
            pipelines,
            features,
            camera_buffer,
            globals_buffer,
            global_bind_group,
            lighting_bind_group,
            lighting_layout,
            default_material_bind_group,
            draw_list,
            debug_shapes,
            debug_batch,
            lighting_shadow_view: shadow_view,
            lighting_shadow_sampler: shadow_samp,
            lighting_env_cube_view: Arc::new(cube_view),
            lighting_rc_view: rc_view,
            lighting_env_sampler: Arc::new(env_linear_sampler),
            light_count_arc,
            light_face_counts,
            shadow_cull_lights,
            scene_ambient_color: [0.0, 0.0, 0.0],
            scene_ambient_intensity: 0.0,
            scene_light_count: 0,
            scene_sky_color: [0.0, 0.0, 0.0],
            scene_has_sky: false,
            scene_csm_splits: CSM_SPLITS,
            rc_world_min,
            rc_world_max,
            sky_uniform_buffer,
            sky_bind_group,
            depth_texture,
            depth_view,
            depth_sample_view,
            gbuf_albedo_texture,
            gbuf_normal_texture,
            gbuf_orm_texture,
            gbuf_emissive_texture,
            gbuffer_targets,
            deferred_bg,
            default_material_views,
            enable_ssao: config.enable_ssao,
            ssao_texture: None,
            ssao_view: None,
            aa_mode: config.aa_mode,
            pre_aa_texture,
            pre_aa_view,
            fxaa_pass,
            smaa_pass,
            taa_pass,
            fxaa_bind_group,
            smaa_bind_group,
            taa_bind_group,
            // ── GPU-resident scene + lights ───────────────────────────────────────
            gpu_scene,
            gpu_light_scene,
            // ── GPU-driven indirect rendering ────────────────────────────────────
            buffer_pool,
            indirect_dispatch,
            shadow_matrix_pass,
            shared_indirect_buf,
            shared_material_ranges,
            shared_shadow_draw_call_buf,
            // ── Frame state ────────────────────────────────────────────────────
            frame_count: 0,
            width: config.width,
            height: config.height,
            last_frame_start: None,
            last_frame_end: None,
            profiler,
            #[cfg(feature = "live-portal")]
            live_portal: None,
            #[cfg(feature = "live-portal")]
            latest_scene_layout: None,
            #[cfg(feature = "live-portal")]
            previous_scene_layout: None,
            #[cfg(feature = "live-portal")]
            pending_layout_changed: false,
            #[cfg(feature = "live-portal")]
            portal_scene_key: (0, 0, 0),
            cached_pass_names,
            // ── Persistent scene environment state ────────────────────────────
            scene_state: SceneState::default(),
            sky_state_changed: true, // first frame always renders the sky LUT
            draw_list_generation:     0,
            persistent_draw_count:   0,
            cached_draw_list_gen:    u64::MAX, // force rebuild on first frame
        })
    }

    // ── Material creation ─────────────────────────────────────────────────────

    /// Upload a [`Material`] to the GPU and return a [`GpuMaterial`] ready for use in
    /// [`Scene::add_object_with_material`] or [`Renderer::draw_mesh_with_material`].
    pub fn create_material(&self, mat: &Material) -> GpuMaterial {
        build_gpu_material(
            &self.device,
            &self.queue,
            &self.resources.bind_group_layouts.material,
            mat,
            &self.default_material_views,
        )
    }

    pub(super) fn rebuild_lighting_bind_group(&mut self) {
        self.lighting_bind_group = Arc::new(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Lighting Bind Group"),
            layout: &self.lighting_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.gpu_light_scene.light_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.lighting_shadow_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.lighting_shadow_sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.lighting_env_cube_view) },
                wgpu::BindGroupEntry { binding: 4, resource: self.gpu_light_scene.shadow_matrix_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.lighting_rc_view) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&self.lighting_env_sampler) },
            ],
        }));
    }
}
