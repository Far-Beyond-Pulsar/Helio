//! Helio's denoiser: the frame's `helio_frame_api::denoise` contract,
//! implemented over frame contracts alone (the signal's inputs, the frame's
//! history, the G-buffer's depth and normals), so it serves any producer of
//! any kind and lives below every plugin.
//!
//! **Stages** (`shaders/irradiance.wgsl`), for a signal traced at one pixel
//! of each block of `side x side` internal pixels:
//!
//! 1. **Temporal accumulation through the frame's history.** A pixel's new
//!    sample is the traced pixels of its four nearest blocks, each weighted
//!    by a tent of one block over its distance (a partition of unity in
//!    expectation: one sample a block's area) times the geometry weight
//!    below; their sum is how much this frame says about the pixel. Its
//!    history is read where the frame's history says it was, from the
//!    bilinear taps that held the same surface, renormalised. The history's
//!    length is multiplied by the producer's confidence and by the share of
//!    the footprint that held the surface, then grows by this frame's
//!    weight up to [`DenoiseSettings::max_history`]; the running mean
//!    blends by weight over length.
//! 2. **A wide reconstruction where history is short** (below
//!    [`FIX_HISTORY`] frames): the neighbours' accumulated values on a 5x5
//!    of taps two blocks each way (one once half the length is there),
//!    weighted by their own history length (inverse variance) and the
//!    geometry. Its result is the history the next frame reads, with the
//!    pixel's own length, so new samples replace it as fast as before.
//! 3. **A variance-guided spatial stage**: an a-trous filter of the linear
//!    B-spline's 3x3, four iterations at steps of one, two, four and eight
//!    blocks (the reach of three of the cubic B-spline's 5x5 at half the
//!    taps; a step of one block makes every tap an independent sample), a
//!    pixel already within the half floats' rounding left as it is, each tap
//!    weighted by the
//!    luminance stop (the difference of two estimates against the standard
//!    error of their difference, from the history's two moments over its
//!    length; variance propagated through the iterations) and the geometry
//!    weight.
//!
//! **Geometry weight** (one function in all three): irradiance caching's
//! error bound (Ward, Rubinstein and Clear 1988; Tabellion and Lamorlette
//! 2004): a record holds over its harmonic mean hit distance (the producer's
//! hit distance, accumulated as its inverse), so near occluders keep contact
//! detail and open ground filters wide; never narrower than its pixel's
//! diagonal. Times a normal stop (the share of the hemisphere a tilted
//! normal sees that the sample never did) and a plane stop (a step leaves
//! the plane by its whole separation). The constants' derivations are in the
//! shader.
//!
//! **Cost** (internal pixels `P`, a view): temporal 4 sample and 4 history
//! taps, fix 25 taps on the pixels below [`FIX_HISTORY`] only (the rest copy),
//! spatial 4 x 9 taps; every tap's geometry one load of the guides (raw depth
//! and octahedral normal, written once). Memory: history 24 bytes a pixel
//! (the luminance's two moments in f32: their difference is the variance,
//! which half floats cannot resolve), the temporal result (also the spatial
//! stage's ping-pong) 24, the output 16, the guides 16: 80 bytes a pixel a
//! signal and view (166 MB at 1920x1080). Nothing scales with the scene.

use std::sync::{Arc, Mutex};

use bytemuck::{Pod, Zeroable};
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};
use helio_feature_api::{BuildEnv, Feature, FeatureRegistrar, NodeBuild};
use helio_frame_api::denoise::{self, Denoiser, RegisterDenoiser, Signal, SignalKind};
use helio_frame_api::history::{self, Need};
use helio_settings::{number, Group, Resolved};

/// Where history is shorter than this many frames the wide reconstruction
/// runs. The spatial stage's first iteration gathers the 25 taps of one
/// block step; once a pixel's own history holds as many samples as the
/// fix's 5x5 would add over its usable share (a quarter, after the geometry
/// rejects what is not its surface: about 6), the fix adds nothing the
/// spatial stage does not. Four keeps it below that.
pub const FIX_HISTORY: f32 = 4.0;

/// The denoiser's settings (a renderer config's `features`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DenoiseSettings {
    /// The most frames of history a pixel keeps. The running mean's noise
    /// falls as the square root of it (32: a factor of 5.7, NRD's 30 alike);
    /// light that changes without the producer's confidence saying so lags
    /// by it (32 frames: half a second at 60 Hz). A producer with a
    /// confidence may take more.
    pub max_history: f32,
}

impl Default for DenoiseSettings {
    fn default() -> Self {
        Self { max_history: 32.0 }
    }
}

/// The denoiser's host settings.
pub static DENOISE: Group = Group {
    key: "denoise",
    title: "Denoiser",
    description: "The frame's denoiser: temporal accumulation through the frame's history, a wide \
                  reconstruction where history is short, a variance-guided spatial stage",
    settings: &[number(
        "max_history",
        "Denoiser History",
        "Denoiser",
        (4.0, 128.0, 1.0),
        32.0,
        "The most frames of history a pixel keeps: more is less noise, and more lag where the \
         light changes without its producer saying so",
    )],
};

impl DenoiseSettings {
    /// The settings `settings` (of [`DENOISE`]) hold.
    pub fn from_settings(settings: &Resolved) -> Self {
        let default = Self::default();
        Self {
            max_history: settings
                .optional("max_history")
                .map_or(default.max_history, |v| v as f32),
        }
    }

    fn clamped(self) -> Self {
        Self {
            max_history: self.max_history.clamp(FIX_HISTORY, 1024.0),
        }
    }
}

/// The denoiser as a feature: it registers itself with the frame's
/// denoiser contract.
pub struct DenoiseFeature;

impl Feature for DenoiseFeature {
    fn id(&self) -> &'static str {
        "helio.denoise"
    }

    fn register(&self, registrar: &mut FeatureRegistrar<'_>) {
        registrar.denoiser(denoiser());
    }
}

/// The denoiser this crate registers (a test or a host without features
/// offers it with `denoise::offer`).
pub fn denoiser() -> Denoiser {
    Denoiser {
        name: "helio",
        supports: Arc::new(|_, signal| signal.kind == SignalKind::Irradiance),
        filter: Arc::new(filter),
    }
}

fn settings(env: &BuildEnv<'_>) -> DenoiseSettings {
    env.setting::<DenoiseSettings>()
        .copied()
        .unwrap_or_default()
        .clamped()
}

fn filter(build: &mut NodeBuild<'_, '_>, signal: &Signal) -> Result<(), String> {
    // Absent a publisher, every frame starts afresh (no history).
    let history = history::request(build, Need::Pixels)?;
    let pass = DenoisePass::new(build.env.device, *signal, settings(build.env), history);
    build.graph.add_pass(Box::new(pass));
    Ok(())
}

/// The snippets `irradiance.wgsl` is built with.
pub const SNIPPETS: [helio_core::shader::ShaderSnippet; 2] =
    [history::SNIPPET, denoise::IRRADIANCE_SNIPPET];

/// `DenoiseParams` (irradiance.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    inv_proj: [f32; 16],
    size: [u32; 2],
    traced: [u32; 2],
    side: u32,
    frame: u32,
    flags: u32,
    max_history: f32,
    fix_history: f32,
    pixel_scale: f32,
    _pad: [u32; 2],
}

const PARAMS_BYTES: u64 = std::mem::size_of::<Params>() as u64;

/// Steps of the spatial stage, in blocks.
const SPATIAL_STEPS: [u32; 4] = [1, 2, 4, 8];

/// One view's textures (all at the internal size).
struct Targets {
    size: [u32; 2],
    /// History: (a, Co, Cg, 1 / hit), (b, length), the luminance's moments.
    history: [wgpu::TextureView; 3],
    /// The temporal result, read by the fix; the spatial stage's ping-pong.
    temp: [wgpu::TextureView; 3],
    /// The signal's output (`Signal::outputs`).
    out: [wgpu::TextureView; 2],
    /// Each pixel's raw depth and octahedral normal, for the stages' taps.
    guides: wgpu::TextureView,
    /// History was written for this view's previous frame.
    primed: bool,
    /// The textures behind the views, in their order (tests read them back).
    textures: Vec<wgpu::Texture>,
}

impl Targets {
    fn new(device: &wgpu::Device, size: [u32; 2]) -> Self {
        let made = std::cell::RefCell::new(Vec::new());
        let texture = |label: &str, format: wgpu::TextureFormat| {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: size[0].max(1),
                    height: size[1].max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                // Copies: tests read the output and history back.
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = texture.create_view(&Default::default());
            made.borrow_mut().push(texture);
            view
        };
        let half = wgpu::TextureFormat::Rgba16Float;
        let moment = wgpu::TextureFormat::Rg32Float;
        let mut targets = Self {
            size,
            history: [
                texture("Denoise history A", half),
                texture("Denoise history B", half),
                texture("Denoise history moments", moment),
            ],
            temp: [
                texture("Denoise temporal A", half),
                texture("Denoise temporal B", half),
                texture("Denoise temporal moments", moment),
            ],
            out: [
                texture("Denoise output A", half),
                texture("Denoise output B", half),
            ],
            guides: texture("Denoise guides", wgpu::TextureFormat::Rgba32Float),
            primed: false,
            textures: Vec::new(),
        };
        targets.textures = made.into_inner();
        targets
    }
}

struct View {
    id: u32,
    used: u64,
    targets: Option<Targets>,
}

/// Views a denoiser keeps history for at once.
const MAX_VIEWS: usize = 8;

struct Kernels {
    shared_layout: wgpu::BindGroupLayout,
    guides_layout: wgpu::BindGroupLayout,
    guides: wgpu::ComputePipeline,
    temporal_layout: wgpu::BindGroupLayout,
    fix_layout: wgpu::BindGroupLayout,
    spatial_layout: wgpu::BindGroupLayout,
    temporal: wgpu::ComputePipeline,
    fix: wgpu::ComputePipeline,
    spatial: wgpu::ComputePipeline,
    params: wgpu::Buffer,
    /// A uniform each spatial iteration (step, iteration).
    iterations: Vec<wgpu::Buffer>,
    /// Bound where the frame publishes no history.
    no_history: wgpu::TextureView,
}

impl Kernels {
    fn new(device: &wgpu::Device) -> Self {
        let float = wgpu::TextureSampleType::Float { filterable: false };
        let texture = |binding, sample_type| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let storage = |binding, format| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        };
        let uniform = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let half = wgpu::TextureFormat::Rgba16Float;
        let moment = wgpu::TextureFormat::Rg32Float;
        let layout = |label, entries: &[wgpu::BindGroupLayoutEntry]| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries,
            })
        };
        let shared_layout = layout(
            "Denoise shared",
            &[
                texture(0, wgpu::TextureSampleType::Depth),
                texture(1, float),
                texture(2, float),
                uniform(3),
            ],
        );
        let guides_layout = layout(
            "Denoise guides",
            &[storage(20, wgpu::TextureFormat::Rgba32Float)],
        );
        let temporal_layout = layout(
            "Denoise temporal",
            &[
                texture(21, float),
                texture(0, float),
                texture(1, float),
                texture(2, float),
                texture(3, float),
                texture(4, float),
                storage(5, half),
                storage(6, half),
                storage(7, moment),
            ],
        );
        let fix_layout = layout(
            "Denoise fix",
            &[
                texture(21, float),
                texture(8, float),
                texture(9, float),
                texture(10, float),
                storage(11, half),
                storage(12, half),
                storage(13, moment),
            ],
        );
        let spatial_layout = layout(
            "Denoise spatial",
            &[
                texture(21, float),
                texture(14, float),
                texture(15, float),
                texture(16, float),
                storage(17, half),
                storage(18, half),
                uniform(19),
            ],
        );
        let module = helio_core::shader::module_with(
            device,
            "Denoise irradiance",
            helio_core::include_wgsl!("../shaders/irradiance.wgsl"),
            &SNIPPETS,
        );
        let pipeline = |entry: &str, stage: &wgpu::BindGroupLayout| {
            helio_core::shader::create_compute_pipeline(
                device,
                wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(
                        &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            label: Some(entry),
                            bind_group_layouts: &[Some(&shared_layout), Some(stage)],
                            immediate_size: 0,
                        }),
                    ),
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                },
            )
        };
        let buffer = |label: &str, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let iterations: Vec<_> = SPATIAL_STEPS
            .iter()
            .map(|_| buffer("Denoise iteration", 16))
            .collect();
        let no_history = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("Denoise no history"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: history::PIXELS_FORMAT.to_wgpu(),
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default());
        Self {
            guides: pipeline("guides", &guides_layout),
            guides_layout,
            temporal: pipeline("temporal", &temporal_layout),
            fix: pipeline("fix", &fix_layout),
            spatial: pipeline("spatial", &spatial_layout),
            shared_layout,
            temporal_layout,
            fix_layout,
            spatial_layout,
            params: buffer("Denoise params", PARAMS_BYTES),
            iterations,
            no_history,
        }
    }
}

/// The denoiser's passes for one signal (module docs): one pass, its stages
/// dispatches of their own, each under a GPU scope (`Denoise: temporal`,
/// `Denoise: fix`, `Denoise: spatial`).
pub struct DenoisePass {
    signal: Signal,
    settings: DenoiseSettings,
    /// The frame publishes a history.
    history: bool,
    kernels: Kernels,
    views: Mutex<Vec<View>>,
    active: Option<usize>,
    /// The step of each spatial iteration was uploaded for this block size.
    uploaded_steps: bool,
    cache: helio_core::gpu_cache::GpuCache,
    /// This frame filtered the signal (its inputs were there).
    ran: bool,
    reads: &'static [&'static str],
    writes: &'static [&'static str],
}

impl DenoisePass {
    fn new(
        device: &wgpu::Device,
        signal: Signal,
        settings: DenoiseSettings,
        history: bool,
    ) -> Self {
        let mut reads = signal.inputs.to_vec();
        if history {
            reads.push(history::PIXELS);
        }
        Self {
            signal,
            settings,
            history,
            kernels: Kernels::new(device),
            views: Mutex::new(Vec::new()),
            active: None,
            uploaded_steps: false,
            cache: helio_core::gpu_cache::GpuCache::new(),
            ran: false,
            reads: helio_frame_api::static_names(&reads),
            writes: helio_frame_api::static_names(&signal.outputs),
        }
    }

    /// The signal this pass filters.
    pub fn signal(&self) -> &Signal {
        &self.signal
    }

    /// The active view's history `[(a, Co, Cg, 1 / hit), (b, length),
    /// moment]` and output `[out0, out1]` textures (for tests and tools that
    /// read them back).
    pub fn textures(&self) -> Option<([wgpu::Texture; 3], [wgpu::Texture; 2])> {
        let views = self.views.lock().unwrap();
        let targets = views.get(self.active?)?.targets.as_ref()?;
        let t = &targets.textures;
        Some((
            [t[0].clone(), t[1].clone(), t[2].clone()],
            [t[6].clone(), t[7].clone()],
        ))
    }
}

fn texture_entry(binding: u32, view: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(view),
    }
}

fn buffer_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

impl RenderPass for DenoisePass {
    fn name(&self) -> &'static str {
        "Denoise"
    }

    fn reads(&self) -> &'static [&'static str] {
        self.reads
    }

    fn writes(&self) -> &'static [&'static str] {
        self.writes
    }

    fn declare_resources(&self, builder: &mut helio_core::graph::ResourceBuilder) {
        builder.read(helio_frame_api::gbuffer::GROUP);
        builder.read(helio_frame_api::gbuffer::DEPTH);
    }

    fn on_resize(&mut self, _device: &wgpu::Device, _size: helio_core::graph::RenderSize) {
        for view in self.views.get_mut().unwrap() {
            view.targets = None;
        }
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let camera = ctx.camera_data;
        let id = camera.jitter_frame[3].to_bits();
        let size = [ctx.width.max(1), ctx.height.max(1)];
        let views = self.views.get_mut().unwrap();
        let index = match views.iter().position(|v| v.id == id) {
            Some(i) => i,
            None => {
                if views.len() >= MAX_VIEWS {
                    let oldest = (0..views.len()).min_by_key(|&i| views[i].used).unwrap();
                    views.swap_remove(oldest);
                }
                views.push(View {
                    id,
                    used: 0,
                    targets: None,
                });
                views.len() - 1
            }
        };
        views[index].used = ctx.frame_num;
        if views[index].targets.as_ref().is_none_or(|t| t.size != size) {
            views[index].targets = Some(Targets::new(ctx.device, size));
        }
        self.active = Some(index);
        let proj = glam::DMat4::from_cols_array(&camera.proj.map(f64::from));
        let inv_proj = proj.inverse().to_cols_array().map(|v| v as f32);
        let side = self.signal.block.side();
        let traced = self.signal.traced_size(size);
        // A pixel's width a metre of view depth (the projection's vertical
        // scale: 2 / (P[1][1] h)).
        let pixel_scale = 2.0 / (camera.proj[5].abs().max(1e-12) * size[1] as f32);
        let primed = views[index].targets.as_ref().unwrap().primed;
        let params = Params {
            inv_proj,
            size,
            traced,
            side,
            frame: ctx.frame_num as u32,
            flags: u32::from(self.signal.confidence) | (u32::from(self.history && primed) << 1),
            max_history: self.settings.max_history,
            fix_history: FIX_HISTORY,
            pixel_scale,
            _pad: [0; 2],
        };
        ctx.write_buffer(&self.kernels.params, 0, bytemuck::bytes_of(&params));
        if !self.uploaded_steps {
            for (i, (buffer, step)) in self
                .kernels
                .iterations
                .iter()
                .zip(SPATIAL_STEPS)
                .enumerate()
            {
                ctx.write_buffer(
                    buffer,
                    0,
                    bytemuck::cast_slice(&[step * side, i as u32, 0, 0]),
                );
            }
            self.uploaded_steps = true;
        }
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        self.ran = false;
        let key = |name| helio_core::ResourceKey::new(name);
        let (Some(signal0), Some(signal1)) = (
            ctx.registry.texture_view(key(self.signal.inputs[0])),
            ctx.registry.texture_view(key(self.signal.inputs[1])),
        ) else {
            // The producer wrote nothing this frame: history is kept.
            return Ok(());
        };
        let Some(gbuffer) = ctx.registry.view_group::<4>(helio_core::ResourceKey::new(
            helio_frame_api::gbuffer::GROUP,
        )) else {
            return Ok(());
        };
        let pixels = if self.history {
            ctx.registry.texture_view(key(history::PIXELS))
        } else {
            None
        };
        let k = &self.kernels;
        let pixels = pixels.unwrap_or(&k.no_history);
        let mut views = self.views.lock().unwrap();
        let Some(targets) = self
            .active
            .and_then(|i| views.get_mut(i))
            .and_then(|v| v.targets.as_mut())
        else {
            return Ok(());
        };
        self.cache.end_frame();
        let device = ctx.device;
        let group = |layout: &wgpu::BindGroupLayout, entries: &[wgpu::BindGroupEntry<'_>]| {
            self.cache.bind_group(
                device,
                &wgpu::BindGroupDescriptor {
                    label: Some("Denoise"),
                    layout,
                    entries,
                },
            )
        };
        let shared = group(
            &k.shared_layout,
            &[
                texture_entry(0, ctx.depth),
                texture_entry(1, gbuffer.views[1]),
                texture_entry(2, pixels),
                buffer_entry(3, &k.params),
            ],
        );
        let [ha, hb, hm] = &targets.history;
        let [ta, tb, tm] = &targets.temp;
        let [oa, ob] = &targets.out;
        let g = &targets.guides;
        let guides = group(&k.guides_layout, &[texture_entry(20, g)]);
        let temporal = group(
            &k.temporal_layout,
            &[
                texture_entry(21, g),
                texture_entry(0, signal0),
                texture_entry(1, signal1),
                texture_entry(2, ha),
                texture_entry(3, hb),
                texture_entry(4, hm),
                texture_entry(5, ta),
                texture_entry(6, tb),
                texture_entry(7, tm),
            ],
        );
        let fix = group(
            &k.fix_layout,
            &[
                texture_entry(21, g),
                texture_entry(8, ta),
                texture_entry(9, tb),
                texture_entry(10, tm),
                texture_entry(11, ha),
                texture_entry(12, hb),
                texture_entry(13, hm),
            ],
        );
        // history -> temp -> out -> temp -> out.
        let iteration =
            |i: usize, input: [&wgpu::TextureView; 2], output: [&wgpu::TextureView; 2]| {
                group(
                    &k.spatial_layout,
                    &[
                        texture_entry(21, g),
                        texture_entry(14, input[0]),
                        texture_entry(15, input[1]),
                        texture_entry(16, hm),
                        texture_entry(17, output[0]),
                        texture_entry(18, output[1]),
                        buffer_entry(19, &k.iterations[i]),
                    ],
                )
            };
        let spatial = [
            iteration(0, [ha, hb], [ta, tb]),
            iteration(1, [ta, tb], [oa, ob]),
            iteration(2, [oa, ob], [ta, tb]),
            iteration(3, [ta, tb], [oa, ob]),
        ];
        let groups = [targets.size[0].div_ceil(8), targets.size[1].div_ceil(8)];
        let mut cmds = ctx.graphics_cmds();
        let stage = |cmds: &mut helio_core::CommandRecorder<'_>,
                     ctx: &mut PassContext,
                     scope: &'static str,
                     pipeline: &wgpu::ComputePipeline,
                     stage: &[&wgpu::BindGroup]| {
            ctx.begin_gpu_scope(cmds, scope);
            {
                let mut pass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(scope),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, Some(&shared), &[]);
                for group in stage {
                    pass.set_bind_group(1, Some(*group), &[]);
                    pass.dispatch_workgroups(groups[0], groups[1], 1);
                }
            }
            ctx.end_gpu_scope(cmds, scope);
        };
        stage(&mut cmds, ctx, "Denoise: guides", &k.guides, &[&guides]);
        stage(
            &mut cmds,
            ctx,
            "Denoise: temporal",
            &k.temporal,
            &[&temporal],
        );
        stage(&mut cmds, ctx, "Denoise: fix", &k.fix, &[&fix]);
        stage(
            &mut cmds,
            ctx,
            "Denoise: spatial",
            &k.spatial,
            &[&spatial[0], &spatial[1], &spatial[2], &spatial[3]],
        );
        targets.primed = true;
        drop(views);
        self.ran = true;
        Ok(())
    }

    fn publish<'a>(&self, frame: &mut helio_core::ResourceRegistry<'a>) {
        if !self.ran {
            return;
        }
        let views = self.views.lock().unwrap();
        let Some(targets) = self
            .active
            .and_then(|i| views.get(i))
            .and_then(|v| v.targets.as_ref())
        else {
            return;
        };
        for (name, view) in self.signal.outputs.iter().zip(&targets.out) {
            frame.write_texture_binding(*name, view, "Denoise");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate(source: &str) -> wgpu::naga::Module {
        let text = helio_core::shader::resolve_with(source, &SNIPPETS);
        let module = wgpu::naga::front::wgsl::parse_str(&text)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&text)));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&text)));
        module
    }

    #[test]
    fn the_shader_validates_and_its_params_have_their_size() {
        let module = validate(include_str!("../shaders/irradiance.wgsl"));
        let mut layouter = wgpu::naga::proc::Layouter::default();
        layouter.update(module.to_ctx()).unwrap();
        let (handle, _) = module
            .types
            .iter()
            .find(|(_, ty)| ty.name.as_deref() == Some("DenoiseParams"))
            .unwrap();
        assert_eq!(u64::from(layouter[handle].size), PARAMS_BYTES);
    }

    #[test]
    fn host_defaults_are_the_default_settings() {
        let host = DenoiseSettings::from_settings(&DENOISE.defaults());
        assert_eq!(host, DenoiseSettings::default());
    }
}
