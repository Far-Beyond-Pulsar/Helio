//! Volumetric fog — froxel grid.
//!
//! A view-space 3D grid (Hillaire, "Physically Based and Unified Volumetric
//! Rendering in Frostbite", SIGGRAPH 2015), rather than a raymarch per pixel:
//!
//! 1. **Classify** — compact active SceneDB volume/light rows once per frame.
//! 2. **Inject** — one thread per froxel: density, one shadow tap per opted-in
//!    light, blended against the reprojected previous frame.
//! 3. **Integrate** — one thread per (x,y) column: marches z once, producing
//!    accumulated in-scattering + transmittance.
//! 4. **Composite** (in `postprocess.wgsl`) — one trilinear 3D fetch at the
//!    pixel's depth.
//!
//! # Why a grid
//!
//! Cost is decoupled from screen resolution: ~2.65M froxels lit once each, against
//! ~59M samples for a 1280x720 per-pixel march at 64 steps. The trilinear fetch
//! filters in depth as well as x/y, so there is no reduced-resolution upsample to
//! hide — which is what made the earlier per-pixel version pixelate the geometry
//! seen through it.
//!
//! Temporal reprojection is what makes one shadow tap per froxel sufficient;
//! without it the grid is far too noisy to use.
//!
//! # Placement
//!
//! Builds the grid before AA; post-processing composites it afterward. Fog has
//! its own temporal history. Optional scene lights/shadow atlas illuminate the
//! medium; camera defaults and SceneDB PP volume rows define its density.
//!
//! # Owned resources
//!
//! The graph's texture pool is 2D-only, so the three 3D textures are owned here
//! and handed to later passes via [`RenderPass::publish`].

use bytemuck::{Pod, Zeroable};
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

pub mod components;
pub use components::{FogComponent, FogSceneBinding};

/// Froxel grid dimensions.
///
/// Fixed screen footprint and 128 logarithmic depth slices. Empty froxels
/// skip lighting, and active scene rows are compacted once per frame. The
/// three RGBA16F grids allocate ~60.75 MiB, independent of output resolution.
const FROXEL_W: u32 = 192;
const FROXEL_H: u32 = 108;
const FROXEL_D: u32 = 128;

const WG_X: u32 = 8;
const WG_Y: u32 = 8;

/// Weight of the current frame in the temporal blend.
///
/// 0.05 leans hard on history, which is what buys a clean image from one sample
/// per froxel. The cost is latency: a light that snaps on takes ~20 frames to
/// fully appear. Raise it if that lag matters more than the noise.
const TEMPORAL_BLEND: f32 = 0.05;

const FMT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
// FogVolume's WGSL opaque prefix/suffix must agree with the PP row ABI.
const _: () = assert!(helio_pass_postprocess::GpuPostProcessUniforms::FOG_BLOCK_OFFSET == 304);
const _: () = assert!(std::mem::size_of::<helio_pass_postprocess::GpuPostProcessVolume>() == 528);

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct FogGlobals {
    csm_splits: [f32; 4],
    _reserved: u32,
    frame: u32,
    history_valid: u32,
    temporal_blend: f32,
    time: f32,
    _pad: [f32; 3],
}

pub struct VolumetricFogPass {
    timing_query: Option<wgpu::QuerySet>,
    classify_pipeline: wgpu::ComputePipeline,
    inject_pipeline: wgpu::ComputePipeline,
    integrate_pipeline: wgpu::ComputePipeline,
    inject_bgl: wgpu::BindGroupLayout,
    /// Group 0 for cs_integrate: camera + fog only.
    ///
    /// Deliberately *not* inject_bgl. That one binds the scattering grid as a
    /// write-only storage texture, and cs_integrate samples the same grid from
    /// group 1 — binding both in one dispatch is a usage conflict wgpu rejects
    /// outright (STORAGE_WRITE_ONLY is exclusive).
    integrate_g0_bgl: wgpu::BindGroupLayout,
    integrate_bgl: wgpu::BindGroupLayout,

    /// The fog block, copied out of the post-process uniform buffer each frame.
    fog_uniform_buf: wgpu::Buffer,
    globals_buf: wgpu::Buffer,
    shadow_sampler: wgpu::Sampler,
    linear_sampler: wgpu::Sampler,
    active_media_buf: wgpu::Buffer,
    fallback_volumes: wgpu::Buffer,
    fallback_lights: wgpu::Buffer,
    fallback_shadow: wgpu::TextureView,

    /// Ping-ponged scattering grids: one is read as history while the other is
    /// written. Sampling and storing to one texture in a single dispatch is a
    /// data race, hence two.
    scatter_view: [wgpu::TextureView; 2],
    _scatter: [wgpu::Texture; 2],
    integrated_view: wgpu::TextureView,
    _integrated: wgpu::Texture,

    /// Index of the scatter grid written this frame; `1 - write_idx` is history.
    write_idx: usize,

    inject_bg: [Option<wgpu::BindGroup>; 2],
    inject_bg_key: Option<[wgpu::Buffer; 4]>,
    inject_shadow: Option<wgpu::TextureView>,
    integrate_g0_bg: Option<wgpu::BindGroup>,
    integrate_bg: [Option<wgpu::BindGroup>; 2],

    frame: u32,
    history_valid: bool,
    temporal_blend: f32,
    time: f32,
}

fn make_grid(device: &wgpu::Device, label: &str) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: FROXEL_W,
            height: FROXEL_H,
            depth_or_array_layers: FROXEL_D,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: FMT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D3),
        ..Default::default()
    });
    (tex, view)
}

impl VolumetricFogPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = helio_core::shader::module(
            device,
            "Volumetric Fog Shader",
            include_str!("../shaders/volumetric_fog.wgsl"),
        );

        let cv = wgpu::ShaderStages::COMPUTE;
        let uniform = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: cv,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage_ro = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: cv,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let tex3d = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: cv,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D3,
                multisampled: false,
            },
            count: None,
        };
        let storage3d = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: cv,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: FMT,
                view_dimension: wgpu::TextureViewDimension::D3,
            },
            count: None,
        };

        let inject_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Volumetric Fog Inject BGL"),
            entries: &[
                // camera
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: cv,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                uniform(1),    // fog
                uniform(2),    // globals
                storage_ro(3), // lights
                storage_ro(4), // shadow matrices
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: cv,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: cv,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
                tex3d(7), // scatter history
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: cv,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                storage3d(9), // scatter out
                storage_ro(11), // SceneDB post-process volumes
                wgpu::BindGroupLayoutEntry {
                    binding: 12,
                    visibility: cv,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Only what cs_integrate actually reads — see the field's doc comment.
        let integrate_g0_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Volumetric Fog Integrate Group0 BGL"),
            entries: &[storage_ro(0), uniform(1)],
        });

        let integrate_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Volumetric Fog Integrate BGL"),
            entries: &[tex3d(0), storage3d(1)],
        });

        let inject_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Volumetric Fog Inject PL"),
            bind_group_layouts: &[Some(&inject_bgl)],
            immediate_size: 0,
        });
        let integrate_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Volumetric Fog Integrate PL"),
            bind_group_layouts: &[Some(&integrate_g0_bgl), Some(&integrate_bgl)],
            immediate_size: 0,
        });

        let classify_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Volumetric Fog Classify"),
            layout: Some(&inject_pl),
            module: &shader,
            entry_point: Some("cs_classify"),
            compilation_options: Default::default(),
            cache: None,
        });
        let inject_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Volumetric Fog Inject"),
            layout: Some(&inject_pl),
            module: &shader,
            entry_point: Some("cs_inject"),
            compilation_options: Default::default(),
            cache: None,
        });
        let integrate_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Volumetric Fog Integrate"),
            layout: Some(&integrate_pl),
            module: &shader,
            entry_point: Some("cs_integrate"),
            compilation_options: Default::default(),
            cache: None,
        });

        let fog_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Volumetric Fog Uniforms"),
            size: helio_pass_postprocess::GpuPostProcessUniforms::FOG_BLOCK_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let globals_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Volumetric Fog Globals"),
            size: std::mem::size_of::<FogGlobals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Volumetric Fog Shadow Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        // Clamped + trilinear: history reprojection and the composite both rely on
        // filtering across all three axes.
        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Volumetric Fog Linear Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let (s0, v0) = make_grid(device, "Fog Scatter 0");
        let (s1, v1) = make_grid(device, "Fog Scatter 1");
        let (integrated, integrated_view) = make_grid(device, "Fog Integrated");

        let active_media_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fog Active Volume and Light Indices"),
            size: (4 + 64 + 256) * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let fallback_volumes = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fog Empty SceneDB Volumes"),
            size: std::mem::size_of::<helio_pass_postprocess::GpuPostProcessVolume>() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let fallback_lights = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fog Empty Lights"),
            size: 128,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let fallback_shadow = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Fog Empty Shadow Atlas"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }).create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        Self {
            timing_query: None,
            classify_pipeline,
            active_media_buf,
            fallback_volumes,
            fallback_lights,
            fallback_shadow,
            inject_pipeline,
            integrate_pipeline,
            inject_bgl,
            integrate_g0_bgl,
            integrate_bgl,
            fog_uniform_buf,
            globals_buf,
            shadow_sampler,
            linear_sampler,
            scatter_view: [v0, v1],
            _scatter: [s0, s1],
            integrated_view,
            _integrated: integrated,
            write_idx: 0,
            inject_bg: [None, None],
            inject_bg_key: None,
            inject_shadow: None,
            integrate_g0_bg: None,
            integrate_bg: [None, None],
            frame: 0,
            history_valid: false,
            temporal_blend: TEMPORAL_BLEND,
            time: 0.0,
        }
    }

    /// Weight of the current frame in the temporal blend, 0..1.
    ///
    /// Lower is steadier but slower to react to lighting changes; higher reacts
    /// faster but lets the single-sample noise through.
    pub fn set_temporal_blend(&mut self, blend: f32) {
        self.temporal_blend = blend.clamp(0.01, 1.0);
    }

    /// Drop the temporal history — call after a camera cut, or reprojected fog
    /// from the previous shot smears across the first frames of the new one.
    pub fn reset_history(&mut self) {
        self.history_valid = false;
    }

    /// Optional GPU timestamps around classification, injection and integration.
    pub fn enable_timing(&mut self, device: &wgpu::Device) -> bool {
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS) {
            return false;
        }
        self.timing_query = Some(device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("Volumetric fog timings"),
            ty: wgpu::QueryType::Timestamp,
            count: 4,
        }));
        true
    }

    pub fn timing_query(&self) -> Option<&wgpu::QuerySet> {
        self.timing_query.as_ref()
    }
}

impl RenderPass for VolumetricFogPass {
    fn name(&self) -> &'static str {
        "VolumetricFogPass"
    }

    fn writes(&self) -> &'static [&'static str] {
        &["fog_accum"]
    }

    fn publish<'a>(&self, frame: &mut helio_core::ResourceRegistry<'a>) {
        // The graph's pool is 2D-only, so this texture is pass-owned and handed
        // over here rather than routed by name.
        let view: &'a wgpu::TextureView = unsafe { std::mem::transmute(&self.integrated_view) };
        frame.write_texture_view(helio_core::ResourceKey::new("fog_accum"), view, "VolumetricFogPass");
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a helio_core::ResourceRegistry<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn chain_transparent(&self) -> bool {
        // execute() only touches ctx.compute_encoder_ptr.
        true
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        self.frame = ctx.frame_num as u32;
        self.time += ctx.delta_time.max(0.0);

        let globals = FogGlobals {
            csm_splits: helio_pass_shadow_matrix::CSM_SPLITS,
            _reserved: 0,
            frame: self.frame,
            history_valid: self.history_valid as u32,
            temporal_blend: self.temporal_blend,
            time: self.time,
            _pad: [0.0; 3],
        };
        ctx.queue
            .write_buffer(&self.globals_buf, 0, bytemuck::bytes_of(&globals));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let Some(postprocess_buf) = ctx.registry.get(helio_core::ResourceKey::new("postprocess_uniforms")) else {
            return Ok(());
        };
        let shadow_atlas = ctx.registry.get(helio_core::ResourceKey::new("shadow_atlas"))
            .unwrap_or(&self.fallback_shadow);

        let camera_buf = ctx.camera;
        let lights_buf = ctx
            .scene_buffers
            .get(helio_core::BufferKey::of("scene_lights"))
            .map(|handle| &handle.buffer)
            .unwrap_or(&self.fallback_lights);
        let volumes_buf = ctx.scene_buffers
            .get(helio_core::BufferKey::of("post_process_volumes"))
            .map(|handle| &handle.buffer)
            .unwrap_or(&self.fallback_volumes);
        let shadow_matrices = ctx
            .registry
            .get::<helio_pass_shadow_matrix::ShadowMatricesFrameData<'_>>(
                helio_core::resource_keys::shadow_matrices(),
            )
            .map(|s| s.shadow_matrices)
            .unwrap_or(ctx.camera);

        // Swap the ping-pong: last frame's write target is this frame's history.
        self.write_idx ^= 1;
        let write_idx = self.write_idx;
        let history_idx = 1 - write_idx;

        let key = [camera_buf.clone(), lights_buf.clone(), shadow_matrices.clone(), volumes_buf.clone()];
        if self.inject_bg_key.as_ref() != Some(&key)
            || self.inject_shadow.as_ref() != Some(shadow_atlas)
        {
            // Both sides are rebuilt together: each pins a fixed history/write
            // pair, so a stale one would read the grid it is also writing.
            self.inject_bg = [None, None];
            self.inject_bg_key = Some(key);
            self.inject_shadow = Some(shadow_atlas.clone());
            self.integrate_g0_bg = None;
        }

        if self.inject_bg[write_idx].is_none() {
            self.inject_bg[write_idx] =
                Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Volumetric Fog Inject BG"),
                    layout: &self.inject_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 11,
                            resource: volumes_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 12,
                            resource: self.active_media_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: camera_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.fog_uniform_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.globals_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: lights_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: shadow_matrices.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::TextureView(shadow_atlas),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: wgpu::BindingResource::Sampler(&self.shadow_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: wgpu::BindingResource::TextureView(
                                &self.scatter_view[history_idx],
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 8,
                            resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 9,
                            resource: wgpu::BindingResource::TextureView(
                                &self.scatter_view[write_idx],
                            ),
                        },
                    ],
                }));
        }

        if self.integrate_g0_bg.is_none() {
            self.integrate_g0_bg = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Volumetric Fog Integrate Group0 BG"),
                layout: &self.integrate_g0_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: camera_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.fog_uniform_buf.as_entire_binding(),
                    },
                ],
            }));
        }

        if self.integrate_bg[write_idx].is_none() {
            self.integrate_bg[write_idx] =
                Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Volumetric Fog Integrate BG"),
                    layout: &self.integrate_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &self.scatter_view[write_idx],
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&self.integrated_view),
                        },
                    ],
                }));
        }

        let (Some(inject_bg), Some(integrate_g0_bg), Some(integrate_bg)) = (
            self.inject_bg[write_idx].as_ref(),
            self.integrate_g0_bg.as_ref(),
            self.integrate_bg[write_idx].as_ref(),
        ) else {
            return Ok(());
        };

        let ce = ctx.compute_encoder_ptr;

        // Pull this frame's fog config out of the post-process uniform buffer,
        // keeping PostProcessSettings the single source of truth.
        unsafe { &mut *ce }.copy_buffer_to_buffer(
            postprocess_buf,
            helio_pass_postprocess::GpuPostProcessUniforms::FOG_BLOCK_OFFSET,
            &self.fog_uniform_buf,
            0,
            helio_pass_postprocess::GpuPostProcessUniforms::FOG_BLOCK_SIZE,
        );

        if let Some(query) = &self.timing_query {
            unsafe { &mut *ce }.write_timestamp(query, 0);
        }
        {
            let mut cpass = unsafe { &mut *ce }.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Volumetric Fog Classify"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.classify_pipeline);
            cpass.set_bind_group(0, inject_bg, &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }
        if let Some(query) = &self.timing_query {
            unsafe { &mut *ce }.write_timestamp(query, 1);
        }
        {
            let mut cpass = unsafe { &mut *ce }.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Volumetric Fog Inject"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.inject_pipeline);
            cpass.set_bind_group(0, inject_bg, &[]);
            cpass.dispatch_workgroups(FROXEL_W.div_ceil(WG_X), FROXEL_H.div_ceil(WG_Y), FROXEL_D);
        }

        if let Some(query) = &self.timing_query {
            unsafe { &mut *ce }.write_timestamp(query, 2);
        }
        {
            // One thread per (x,y) column — each marches all FROXEL_D slices, so
            // z is 1 here, not FROXEL_D.
            let mut cpass = unsafe { &mut *ce }.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Volumetric Fog Integrate"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.integrate_pipeline);
            cpass.set_bind_group(0, integrate_g0_bg, &[]);
            cpass.set_bind_group(1, integrate_bg, &[]);
            cpass.dispatch_workgroups(FROXEL_W.div_ceil(WG_X), FROXEL_H.div_ceil(WG_Y), 1);
        }

        if let Some(query) = &self.timing_query {
            unsafe { &mut *ce }.write_timestamp(query, 3);
        }
        // History is only meaningful once a grid has actually been written.
        self.history_valid = true;

        Ok(())
    }
}
