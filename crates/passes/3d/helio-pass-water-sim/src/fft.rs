//! Tessendorf FFT ocean (a port of Ultra-Water's `OceanSim.js`).
//!
//! Four cascades of 256x256 are stacked in one 256x1024 target so every
//! inverse-FFT stage is a single draw. Each frame: initial spectrum from volume
//! row 0 -> evolve to time t -> 16 FFT stages -> assemble into the displacement
//! and slope arrays -> mip chains. The surface shader reads those arrays; the
//! height-field simulation (`update.frag.wgsl`) now only carries object
//! interaction on top.
//!
//! One sea is shared by every volume: the cascades tile in world space (the
//! lengths divide [`crate::WATER_WORLD_PERIOD`]), so each volume simply shows
//! its patch of the same ocean.

use helio_core::PassContext;

pub(crate) const FFT_N: u32 = 256;
pub(crate) const FFT_CASCADES: u32 = 4;
/// Tile length of each cascade in metres; each divides `WATER_WORLD_PERIOD`
/// and `surface.wgsl`'s `FFT_LENGTHS` must match. The last is the
/// gravity-capillary layer (slopes only).
pub(crate) const FFT_LENGTHS: [f32; 4] = [270.0, 45.0, 9.0, 2.0];
pub(crate) const FFT_MIPS: u32 = 9;
/// The sea loops seamlessly after this many seconds (`fft_evolve.wgsl`).
const FFT_LOOP: f32 = 300.0;
const CAPILLARY_FADE: f32 = 0.08;
const CHOPPINESS: f32 = 0.85;
const FOAM_RECOVERY: f32 = 0.7;

const FFT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;
const OUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

const STAGES: u32 = 16;
const SLOT_INIT: u32 = 0;
const SLOT_EVOLVE: u32 = 1;
const SLOT_STAGE: u32 = 2;
const SLOT_ASSEMBLE: u32 = SLOT_STAGE + STAGES;
const SLOT_MIP: u32 = SLOT_ASSEMBLE + FFT_CASCADES;
const SLOT_COUNT: u32 = SLOT_MIP + FFT_CASCADES * (FFT_MIPS - 1);
/// `min_uniform_buffer_offset_alignment` is at most 256 on every backend.
const SLOT_STRIDE: u64 = 256;

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct FftUniform {
    lengths: [f32; 4],
    cut: [f32; 4],
    time: f32,
    cap_fade: f32,
    lambda: f32,
    dt: f32,
    foam_recovery: f32,
    seed: u32,
    sub: i32,
    vertical: i32,
    cascade: i32,
    level: i32,
    _pad: [i32; 2],
}

/// Everything that needs a full-size allocation; freed when the water leaves.
struct FftTargets {
    _dummy: wgpu::Texture,
    uniform_buf: wgpu::Buffer,

    h0: wgpu::Texture,
    ping: [wgpu::Texture; 2],
    pong: [wgpu::Texture; 2],

    disp: [wgpu::Texture; 2],
    slope: [wgpu::Texture; 2],
    /// All layers and mips, for the surface to sample.
    disp_array: [wgpu::TextureView; 2],
    slope_array: [wgpu::TextureView; 2],
    /// One view per (parity, cascade, mip), `(p * 4 + c) * FFT_MIPS + m`.
    disp_mips: Vec<wgpu::TextureView>,
    slope_mips: Vec<wgpu::TextureView>,

    h0_view: wgpu::TextureView,
    ping_views: [wgpu::TextureView; 2],
    pong_views: [wgpu::TextureView; 2],

    bg_init: wgpu::BindGroup,
    bg_evolve: wgpu::BindGroup,
    bg_ping: wgpu::BindGroup,
    bg_pong: wgpu::BindGroup,
    /// Indexed `parity * 4 + cascade`: reads `ping` and that parity's turbulence.
    bg_assemble: Vec<wgpu::BindGroup>,
    /// Indexed `((parity * 4 + cascade) * (FFT_MIPS - 1)) + (mip - 1)`.
    bg_mip: Vec<wgpu::BindGroup>,
}

pub(crate) struct FftOcean {
    group0: wgpu::BindGroupLayout,
    group1: wgpu::BindGroupLayout,
    init: wgpu::RenderPipeline,
    evolve: wgpu::RenderPipeline,
    stage: wgpu::RenderPipeline,
    assemble: wgpu::RenderPipeline,
    mip: wgpu::RenderPipeline,
    pub(crate) sampler: wgpu::Sampler,

    targets: Option<FftTargets>,
    /// Which output set the last assemble wrote.
    parity: usize,
    needs_clear: bool,
    rows_bg: Option<(wgpu::Buffer, wgpu::BindGroup)>,
}

fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

impl FftOcean {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let vert = helio_core::shader::module(
            device,
            "WaterFft VS",
            helio_core::include_wgsl!("../shaders/fullscreen.vert.wgsl"),
        );
        let module = |label, src| helio_core::shader::module(device, label, src);
        let init_fs = module(
            "WaterFft Init FS",
            helio_core::include_wgsl!("../shaders/fft_init.wgsl"),
        );
        let evolve_fs = module(
            "WaterFft Evolve FS",
            helio_core::include_wgsl!("../shaders/fft_evolve.wgsl"),
        );
        let stage_fs = module(
            "WaterFft Stage FS",
            helio_core::include_wgsl!("../shaders/fft_stage.wgsl"),
        );
        let assemble_fs = module(
            "WaterFft Assemble FS",
            helio_core::include_wgsl!("../shaders/fft_assemble.wgsl"),
        );
        let mip_fs = module(
            "WaterFft Mip FS",
            helio_core::include_wgsl!("../shaders/fft_mip.wgsl"),
        );

        let group0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("WaterFft BGL"),
            entries: &[
                texture_entry(0),
                texture_entry(1),
                texture_entry(2),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        // The initial spectrum also reads the water volume rows.
        let group1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("WaterFft Rows BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let layout0 = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("WaterFft PL"),
            bind_group_layouts: &[Some(&group0)],
            immediate_size: 0,
        });
        let layout_init = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("WaterFft Init PL"),
            bind_group_layouts: &[Some(&group0), Some(&group1)],
            immediate_size: 0,
        });

        let make = |label: &str,
                    layout: &wgpu::PipelineLayout,
                    frag: &wgpu::ShaderModule,
                    formats: &[wgpu::TextureFormat]| {
            let targets: Vec<Option<wgpu::ColorTargetState>> = formats
                .iter()
                .map(|&format| {
                    Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })
                })
                .collect();
            helio_core::shader::create_render_pipeline(
                device,
                wgpu::RenderPipelineDescriptor {
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
                        targets: &targets,
                    }),
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                },
            )
        };
        let init = make("WaterFft Init", &layout_init, &init_fs, &[FFT_FORMAT]);
        let evolve = make(
            "WaterFft Evolve",
            &layout0,
            &evolve_fs,
            &[FFT_FORMAT, FFT_FORMAT],
        );
        let stage = make(
            "WaterFft Stage",
            &layout0,
            &stage_fs,
            &[FFT_FORMAT, FFT_FORMAT],
        );
        let assemble = make(
            "WaterFft Assemble",
            &layout0,
            &assemble_fs,
            &[OUT_FORMAT, OUT_FORMAT],
        );
        let mip = make("WaterFft Mip", &layout0, &mip_fs, &[OUT_FORMAT, OUT_FORMAT]);

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("WaterFft Sampler"),
            min_filter: wgpu::FilterMode::Linear,
            mag_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            ..Default::default()
        });

        Self {
            group0,
            group1,
            init,
            evolve,
            stage,
            assemble,
            mip,
            sampler,
            targets: None,
            parity: 0,
            needs_clear: true,
            rows_bg: None,
        }
    }

    pub(crate) fn is_allocated(&self) -> bool {
        self.targets.is_some()
    }

    /// Allocates (`true`) or frees (`false`) the full-size targets.
    pub(crate) fn set_allocated(&mut self, device: &wgpu::Device, allocated: bool) {
        self.parity = 0;
        self.needs_clear = true;
        self.rows_bg = None;
        self.targets = allocated.then(|| self.allocate(device));
    }

    /// The displacement and slope arrays the surface samples this frame.
    pub(crate) fn output_views(&self) -> Option<(&wgpu::TextureView, &wgpu::TextureView)> {
        self.targets
            .as_ref()
            .map(|t| (&t.disp_array[self.parity], &t.slope_array[self.parity]))
    }

    fn allocate(&self, device: &wgpu::Device) -> FftTargets {
        let stack = |label: &str| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: FFT_N,
                    height: FFT_N * FFT_CASCADES,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: FFT_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };
        let dummy = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("WaterFft Dummy"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FFT_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let dummy_view = dummy.create_view(&Default::default());

        let h0 = stack("WaterFft H0");
        let ping = [stack("WaterFft Ping 0"), stack("WaterFft Ping 1")];
        let pong = [stack("WaterFft Pong 0"), stack("WaterFft Pong 1")];
        let h0_view = h0.create_view(&Default::default());
        let ping_views = [
            ping[0].create_view(&Default::default()),
            ping[1].create_view(&Default::default()),
        ];
        let pong_views = [
            pong[0].create_view(&Default::default()),
            pong[1].create_view(&Default::default()),
        ];

        let out = |label: &str| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: FFT_N,
                    height: FFT_N,
                    depth_or_array_layers: FFT_CASCADES,
                },
                mip_level_count: FFT_MIPS,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: OUT_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };
        let disp = [out("WaterFft Disp 0"), out("WaterFft Disp 1")];
        let slope = [out("WaterFft Slope 0"), out("WaterFft Slope 1")];
        let array = |tex: &wgpu::Texture| {
            tex.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            })
        };
        let disp_array = [array(&disp[0]), array(&disp[1])];
        let slope_array = [array(&slope[0]), array(&slope[1])];
        let mips = |tex: [&wgpu::Texture; 2]| -> Vec<wgpu::TextureView> {
            let mut views = Vec::new();
            for tex in tex {
                for c in 0..FFT_CASCADES {
                    for m in 0..FFT_MIPS {
                        views.push(tex.create_view(&wgpu::TextureViewDescriptor {
                            dimension: Some(wgpu::TextureViewDimension::D2),
                            base_mip_level: m,
                            mip_level_count: Some(1),
                            base_array_layer: c,
                            array_layer_count: Some(1),
                            ..Default::default()
                        }));
                    }
                }
            }
            views
        };
        let disp_mips = mips([&disp[0], &disp[1]]);
        let slope_mips = mips([&slope[0], &slope[1]]);

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("WaterFft Uniforms"),
            size: SLOT_STRIDE * u64::from(SLOT_COUNT),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_binding = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &uniform_buf,
            offset: 0,
            size: wgpu::BufferSize::new(std::mem::size_of::<FftUniform>() as u64),
        });
        let make_bg = |label: &str,
                       t0: &wgpu::TextureView,
                       t1: &wgpu::TextureView,
                       t2: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &self.group0,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(t0),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(t1),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(t2),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: uniform_binding.clone(),
                    },
                ],
            })
        };

        let bg_init = make_bg("WaterFft Init BG", &dummy_view, &dummy_view, &dummy_view);
        let bg_evolve = make_bg("WaterFft Evolve BG", &h0_view, &dummy_view, &dummy_view);
        let bg_ping = make_bg(
            "WaterFft Ping BG",
            &ping_views[0],
            &ping_views[1],
            &dummy_view,
        );
        let bg_pong = make_bg(
            "WaterFft Pong BG",
            &pong_views[0],
            &pong_views[1],
            &dummy_view,
        );
        let mip_index = |p: u32, c: u32, m: u32| ((p * FFT_CASCADES + c) * FFT_MIPS + m) as usize;
        let mut bg_assemble = Vec::new();
        let mut bg_mip = Vec::new();
        for p in 0..2 {
            for c in 0..FFT_CASCADES {
                // the turbulence the new frame continues from
                bg_assemble.push(make_bg(
                    "WaterFft Assemble BG",
                    &ping_views[0],
                    &ping_views[1],
                    &disp_mips[mip_index(p, c, 0)],
                ));
                for m in 1..FFT_MIPS {
                    bg_mip.push(make_bg(
                        "WaterFft Mip BG",
                        &disp_mips[mip_index(p, c, m - 1)],
                        &slope_mips[mip_index(p, c, m - 1)],
                        &dummy_view,
                    ));
                }
            }
        }

        FftTargets {
            _dummy: dummy,
            uniform_buf,
            h0,
            ping,
            pong,
            disp,
            slope,
            disp_array,
            slope_array,
            disp_mips,
            slope_mips,
            h0_view,
            ping_views,
            pong_views,
            bg_init,
            bg_evolve,
            bg_ping,
            bg_pong,
            bg_assemble,
            bg_mip,
        }
    }

    /// Stages every pass's uniforms for this frame. `time` is the sea's clock,
    /// `dt` the simulated seconds since the last update.
    pub(crate) fn write_uniforms(&self, ctx: &helio_core::PrepareContext, time: f32, dt: f32) {
        let Some(t) = &self.targets else { return };
        let mut cut = [0.0f32; 4];
        for c in 0..3 {
            cut[c] = std::f32::consts::TAU / FFT_LENGTHS[c + 1] * 6.0;
        }
        let base = FftUniform {
            lengths: FFT_LENGTHS,
            cut,
            time: time.rem_euclid(FFT_LOOP),
            cap_fade: CAPILLARY_FADE,
            lambda: CHOPPINESS,
            dt,
            foam_recovery: FOAM_RECOVERY,
            seed: 1337,
            ..Default::default()
        };
        let mut bytes = vec![0u8; (SLOT_STRIDE * u64::from(SLOT_COUNT)) as usize];
        let mut put = |slot: u32, u: FftUniform| {
            let at = slot as usize * SLOT_STRIDE as usize;
            let src = bytemuck::bytes_of(&u);
            bytes[at..at + src.len()].copy_from_slice(src);
        };
        put(SLOT_INIT, base);
        put(SLOT_EVOLVE, base);
        for k in 0..STAGES {
            let (dir, s) = (k / 8, k % 8 + 1);
            put(
                SLOT_STAGE + k,
                FftUniform {
                    sub: 1 << s,
                    vertical: dir as i32,
                    ..base
                },
            );
        }
        for c in 0..FFT_CASCADES {
            put(
                SLOT_ASSEMBLE + c,
                FftUniform {
                    cascade: c as i32,
                    ..base
                },
            );
        }
        ctx.write_buffer(&t.uniform_buf, 0, &bytes);
    }

    /// Runs the whole FFT chain once. `rows` is the `"water_volumes"` buffer.
    pub(crate) fn execute(&mut self, ctx: &mut PassContext, rows: &wgpu::Buffer) {
        let Some(t) = &self.targets else { return };

        if self.rows_bg.as_ref().map(|(b, _)| b) != Some(rows) {
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("WaterFft Rows BG"),
                layout: &self.group1,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: rows.as_entire_binding(),
                }],
            });
            self.rows_bg = Some((rows.clone(), bg));
        }
        let rows_bg = &self.rows_bg.as_ref().unwrap().1;

        let mip_index = |p: usize, c: u32, m: u32| (p as u32 * FFT_CASCADES + c) * FFT_MIPS + m;
        let offset = |slot: u32| (u64::from(slot) * SLOT_STRIDE) as u32;

        if self.needs_clear {
            // turbulence starts at 1 (no foam), slopes at 0
            for p in 0..2usize {
                for c in 0..FFT_CASCADES {
                    let i = mip_index(p, c, 0) as usize;
                    let attachment = |view, color| {
                        Some(wgpu::RenderPassColorAttachment {
                            view,
                            resolve_target: None,
                            depth_slice: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(color),
                                store: wgpu::StoreOp::Store,
                            },
                        })
                    };
                    let attachments = [
                        attachment(
                            &t.disp_mips[i],
                            wgpu::Color {
                                r: 0.0,
                                g: 0.0,
                                b: 0.0,
                                a: 1.0,
                            },
                        ),
                        attachment(&t.slope_mips[i], wgpu::Color::TRANSPARENT),
                    ];
                    let desc = wgpu::RenderPassDescriptor {
                        label: Some("WaterFft Clear"),
                        color_attachments: &attachments,
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    };
                    drop(ctx.begin_render_pass(&desc));
                }
            }
            self.needs_clear = false;
        }

        run(
            ctx,
            "WaterFft Init",
            &self.init,
            &t.bg_init,
            offset(SLOT_INIT),
            Some(rows_bg),
            &[&t.h0_view],
        );
        run(
            ctx,
            "WaterFft Evolve",
            &self.evolve,
            &t.bg_evolve,
            offset(SLOT_EVOLVE),
            None,
            &[&t.ping_views[0], &t.ping_views[1]],
        );
        for k in 0..STAGES {
            let (bg, dst) = if k % 2 == 0 {
                (&t.bg_ping, &t.pong_views)
            } else {
                (&t.bg_pong, &t.ping_views)
            };
            run(
                ctx,
                "WaterFft Stage",
                &self.stage,
                bg,
                offset(SLOT_STAGE + k),
                None,
                &[&dst[0], &dst[1]],
            );
        }

        let next = self.parity ^ 1;
        for c in 0..FFT_CASCADES {
            let i = mip_index(next, c, 0) as usize;
            run(
                ctx,
                "WaterFft Assemble",
                &self.assemble,
                &t.bg_assemble[self.parity * FFT_CASCADES as usize + c as usize],
                offset(SLOT_ASSEMBLE + c),
                None,
                &[&t.disp_mips[i], &t.slope_mips[i]],
            );
            for m in 1..FFT_MIPS {
                let i = mip_index(next, c, m) as usize;
                let bg = (next as u32 * FFT_CASCADES + c) * (FFT_MIPS - 1) + (m - 1);
                run(
                    ctx,
                    "WaterFft Mip",
                    &self.mip,
                    &t.bg_mip[bg as usize],
                    offset(SLOT_MIP),
                    None,
                    &[&t.disp_mips[i], &t.slope_mips[i]],
                );
            }
        }
        self.parity = next;
    }
}

/// One fullscreen draw that overwrites `views`.
fn run(
    ctx: &mut PassContext,
    label: &str,
    pipeline: &wgpu::RenderPipeline,
    bg0: &wgpu::BindGroup,
    offset: u32,
    bg1: Option<&wgpu::BindGroup>,
    views: &[&wgpu::TextureView],
) {
    let attachments: Vec<Option<wgpu::RenderPassColorAttachment>> = views
        .iter()
        .map(|view| {
            Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })
        })
        .collect();
    let desc = wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &attachments,
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    };
    let mut pass = ctx.begin_render_pass(&desc);
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bg0, &[offset]);
    if let Some(bg1) = bg1 {
        pass.set_bind_group(1, bg1, &[]);
    }
    pass.draw(0..6, 0..1);
}
