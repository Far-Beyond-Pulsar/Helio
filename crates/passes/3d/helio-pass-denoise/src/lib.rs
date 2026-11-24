//! Helio's denoiser: the frame's `helio_frame_api::denoise` contract,
//! implemented over frame contracts alone (the signal's inputs, the frame's
//! history, the G-buffer's depth and normals), so it serves any producer of
//! any kind and lives below every plugin.
//!
//! **Stages** (`shaders/irradiance.wgsl`), for a signal traced at one pixel
//! of each block of `side x side` internal pixels. Light is spread where the
//! information is, once a traced sample; only what needs a pixel's own
//! geometry runs once a pixel:
//!
//! 1. **Temporal accumulation through the frame's history** (a pixel). A
//!    pixel's new sample is the traced pixels of its four nearest blocks,
//!    each weighted by a tent of one block over its distance (a partition of
//!    unity in expectation: one sample a block's area) times the geometry
//!    weight's surface and occluder stops (below; no hit-distance radius:
//!    a sample's own ray length goes with the light it found, and contact
//!    detail finer than a block is the producer's, a gather's near field);
//!    their sum is how much this frame says about the pixel.
//!    Its history is read where the frame's history says it was, from the
//!    bilinear taps that held the same surface, renormalised. The history's
//!    length is multiplied by the producer's confidence and by the share of
//!    the footprint that held the surface, then grows by this frame's weight
//!    up to [`DenoiseSettings::max_history`]; the running mean blends by
//!    weight over length.
//! 2. **Records** (a block; blocks of 4 and 16): the temporal result at the
//!    block's representative pixel (its centre). Traced at every pixel, the
//!    pixels are the records.
//! 3. **Spatial** (a block), twice: an a-trous filter of the linear
//!    B-spline's 3x3 over the records, at steps of s and 2s blocks, `s =
//!    sqrt(16 / length)` within 1 to 4 (the footprint adds about s^2
//!    records, so every estimate rests on about 16 frames' worth: wide where
//!    history is short, which is the history fix, one block from 16 frames
//!    on), each tap weighted by geometry alone (below). Each record also
//!    carries how many samples' worth it holds: a weighted mean of
//!    estimates of `n_q` samples holds `(sum w)^2 / sum (w^2 / n_q)`.
//! 4. **Reconstruction** (a pixel): the filtered records of the pixel's
//!    nearest 2x2 blocks (a tent of the block times the geometry weight; a
//!    4x4 where none holds the pixel's surface), at the pixel's own normal,
//!    combined with the pixel's own temporal result by the samples each
//!    rests on (the pixel's own its length; the records' what the spatial
//!    stage left them, times the share of the pixel's tent their geometry
//!    lets them serve, so at a contact the pixel keeps its own). The result
//!    is the output and the next frame's history: the spatial stage is
//!    recurrent, as ReBLUR's post-blur history is, while the moments stay
//!    the raw samples' (the variance never sees the filter).
//!
//! **The mean is kept.** No weight that decides how much a sample counts
//! depends on the light it carries: tents, geometry, the hit distance's
//! radius, history lengths and sample counts. Every stage is then a
//! normalised linear filter with weights independent of the values, whose
//! expectation for a stationary signal is the signal's mean, and chroma is
//! averaged weighted by the luminance it scales (a ratio's mean so weighted
//! is the ratio of the means). A luminance stop or an inverse-variance
//! combination with the variance estimated from the samples themselves (the
//! first version's) weighs a positive heavy-tailed signal's bright samples
//! down, since those are the ones whose estimated variance is high.
//!
//! **Geometry weight** (one function in every stage): irradiance caching's
//! error bound (Ward, Rubinstein and Clear 1988; Tabellion and Lamorlette
//! 2004): a record holds over its harmonic mean hit distance (the producer's
//! hit distance, accumulated as its inverse), so near occluders keep contact
//! detail and open ground filters wide; never narrower than its pixel's
//! diagonal. The spatial stage takes one radius for all its taps, the
//! harmonic mean over the footprint's records counted alike: a record's own
//! ray lengths go with its light (a near source is short), so a radius of
//! its own would weigh it by its light. Times a normal stop (the share of the hemisphere a tilted
//! normal sees that the sample never did) and a plane stop (a step leaves
//! the plane by its whole separation), in view space. And an occluder
//! stop: nothing the frame's depth shows between the two pixels may stand
//! out of both receivers' tangent planes by more than the plane stop
//! accepts, which tells the floor either side of a wall's foot apart (same
//! plane, same normal, close; a hit distance cannot, the two sides' rays
//! meeting the wall alike). Positions and normals are the frame's guides
//! (`helio_frame_api::history::DEPTH`, `NORMAL`), one copy a view.
//!
//! **Cost** (internal pixels `P`, blocks `P / side^2`): temporal 4 sample and
//! 4 history taps a pixel, reconstruction 4 record taps a pixel, records and
//! 2 x 9 spatial taps a block, each tap that passes the geometry weight also
//! reading the depth of the pixels between (the occluder stop: under a
//! block's side a tap in the temporal stage and the reconstruction, the
//! step in the spatial stage). Memory: a pixel's history 16 bytes (its A
//! and B, the outputs), its moments 2 x 16 (length and two f32 moments,
//! swapped with the temporal result's, never copied), the temporal result
//! 16: 64 bytes a pixel; records 24 bytes a block in two or three sets.
//! Nothing scales with the scene.

use std::sync::{Arc, Mutex};

use bytemuck::{Pod, Zeroable};
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};
use helio_feature_api::{BuildEnv, Feature, FeatureRegistrar, NodeBuild};
use helio_frame_api::denoise::{self, Denoiser, RegisterDenoiser, Signal, SignalKind};
use helio_frame_api::history::{self, Need};
use helio_settings::{number, Group, Resolved};

/// The least history a setting may ask: the standard error the output
/// reports comes from a pixel's moments, which over fewer than four samples
/// are dominated by their own noise (SVGF's threshold for trusting them).
pub const MIN_HISTORY: f32 = 4.0;

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
                  reconstruction where history is short, a spatial stage stopped by geometry",
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
            max_history: self.max_history.clamp(MIN_HISTORY, 1024.0),
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
    // The guides, history and where each pixel was are the frame's.
    if !history::request(build, Need::Pixels)? {
        return Err("the denoiser reads the frame's history, which no publisher offers".into());
    }
    let pass = DenoisePass::new(
        build.env.device,
        build.env.queue,
        *signal,
        settings(build.env),
    );
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
    view_rotation: [f32; 16],
    proj_zw: [f32; 4],
    size: [u32; 2],
    traced: [u32; 2],
    side: u32,
    frame: u32,
    flags: u32,
    max_history: f32,
    pixel_scale: f32,
    _pad: [u32; 7],
}

const PARAMS_BYTES: u64 = std::mem::size_of::<Params>() as u64;

/// Iterations of the spatial stage (steps of s and 2s blocks).
const SPATIAL_ITERATIONS: usize = 2;

const HALF: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const MOMENTS: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;
const LENGTH: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Float;

/// One view's textures.
struct Targets {
    size: [u32; 2],
    /// A pixel's state (internal size): A (a, Co / Y, Cg / Y, standard
    /// error), B (b, 1 / hit); the history's are the signal's outputs.
    history: [wgpu::TextureView; 2],
    /// The temporal result, the same layout.
    temp: [wgpu::TextureView; 2],
    /// M (length, mean, mean square), two: the temporal stage reads the
    /// history's and writes the other, which is the next frame's history's.
    moments: [wgpu::TextureView; 2],
    /// The moments the last frame wrote.
    moments_latest: usize,
    /// Records (traced size): A, B and N (length, samples). The blocks' records first
    /// (none for a signal traced at every pixel: its pixels are its
    /// records), then the spatial stage's ping-pong.
    records: Vec<[wgpu::TextureView; 3]>,
    /// History was written for this view's previous frame.
    primed: bool,
    /// The textures behind the history's views (tests read them back).
    textures: Vec<wgpu::Texture>,
}

impl Targets {
    fn new(device: &wgpu::Device, size: [u32; 2], traced: [u32; 2], side: u32) -> Self {
        let made = std::cell::RefCell::new(Vec::new());
        let texture = |label: &str, size: [u32; 2], format: wgpu::TextureFormat| {
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
        let history = [
            texture("Denoise history A", size, HALF),
            texture("Denoise history B", size, HALF),
        ];
        let moments = [
            texture("Denoise moments A", size, MOMENTS),
            texture("Denoise moments B", size, MOMENTS),
        ];
        let temp = [
            texture("Denoise temporal A", size, HALF),
            texture("Denoise temporal B", size, HALF),
        ];
        let sets = if side == 1 { 2 } else { 3 };
        let records = (0..sets)
            .map(|_| {
                [
                    texture("Denoise records A", traced, HALF),
                    texture("Denoise records B", traced, HALF),
                    texture("Denoise records length", traced, LENGTH),
                ]
            })
            .collect();
        let textures = made.into_inner();
        Self {
            size,
            history,
            temp,
            moments,
            moments_latest: 0,
            records,
            primed: false,
            textures,
        }
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
    temporal_layout: wgpu::BindGroupLayout,
    records_layout: wgpu::BindGroupLayout,
    spatial_layout: wgpu::BindGroupLayout,
    reconstruct_layout: wgpu::BindGroupLayout,
    temporal: wgpu::ComputePipeline,
    records: wgpu::ComputePipeline,
    spatial: wgpu::ComputePipeline,
    reconstruct: wgpu::ComputePipeline,
    params: wgpu::Buffer,
    /// A uniform each spatial iteration (its index), written once.
    iterations: Vec<wgpu::Buffer>,
}

impl Kernels {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let texture = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
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
        let layout = |label, entries: &[wgpu::BindGroupLayoutEntry]| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries,
            })
        };
        let shared_layout = layout(
            "Denoise shared",
            &[texture(0), texture(1), texture(2), uniform(3)],
        );
        let temporal_layout = layout(
            "Denoise temporal",
            &[
                texture(0),
                texture(1),
                texture(2),
                texture(3),
                texture(4),
                storage(5, HALF),
                storage(6, HALF),
                storage(7, MOMENTS),
            ],
        );
        let records_layout = layout(
            "Denoise records",
            &[
                texture(8),
                texture(9),
                texture(10),
                storage(11, HALF),
                storage(12, HALF),
                storage(13, LENGTH),
            ],
        );
        let spatial_layout = layout(
            "Denoise spatial",
            &[
                texture(14),
                texture(15),
                texture(16),
                storage(17, HALF),
                storage(18, HALF),
                storage(19, LENGTH),
                uniform(20),
            ],
        );
        let reconstruct_layout = layout(
            "Denoise reconstruct",
            &[
                texture(21),
                texture(22),
                texture(23),
                texture(24),
                texture(25),
                storage(26, HALF),
                storage(27, HALF),
                texture(28),
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
        let iterations: Vec<_> = (0..SPATIAL_ITERATIONS)
            .map(|i| {
                let b = buffer("Denoise iteration", 16);
                helio_core::upload::write_buffer(
                    queue,
                    &b,
                    0,
                    bytemuck::cast_slice(&[i as u32, 0, 0, 0]),
                );
                b
            })
            .collect();
        Self {
            temporal: pipeline("temporal", &temporal_layout),
            records: pipeline("records", &records_layout),
            spatial: pipeline("spatial", &spatial_layout),
            reconstruct: pipeline("reconstruct", &reconstruct_layout),
            shared_layout,
            temporal_layout,
            records_layout,
            spatial_layout,
            reconstruct_layout,
            params: buffer("Denoise params", PARAMS_BYTES),
            iterations,
        }
    }
}

/// The denoiser's passes for one signal (module docs): one pass, its stages
/// dispatches of their own, each under a GPU scope (`Denoise: temporal`,
/// `Denoise: records`, `Denoise: spatial`, `Denoise: reconstruct`).
pub struct DenoisePass {
    signal: Signal,
    settings: DenoiseSettings,
    kernels: Kernels,
    views: Mutex<Vec<View>>,
    active: Option<usize>,
    cache: helio_core::gpu_cache::GpuCache,
    /// This frame filtered the signal (its inputs were there).
    ran: bool,
    reads: &'static [&'static str],
    writes: &'static [&'static str],
}

impl DenoisePass {
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        signal: Signal,
        settings: DenoiseSettings,
    ) -> Self {
        let mut reads = signal.inputs.to_vec();
        reads.extend([history::PIXELS, history::DEPTH, history::NORMAL]);
        Self {
            signal,
            settings,
            kernels: Kernels::new(device, queue),
            views: Mutex::new(Vec::new()),
            active: None,
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

    /// The active view's history `[A, B, M]` (A and B are the outputs; M the
    /// length and the luminance's moments) and output `[out0, out1]`
    /// textures (for tests and tools that read them back).
    pub fn textures(&self) -> Option<([wgpu::Texture; 3], [wgpu::Texture; 2])> {
        let views = self.views.lock().unwrap();
        let targets = views.get(self.active?)?.targets.as_ref()?;
        // Made in order: history A, B, moments A, B, temporal A, B, ...
        let t = &targets.textures;
        Some((
            [
                t[0].clone(),
                t[1].clone(),
                t[2 + targets.moments_latest].clone(),
            ],
            [t[0].clone(), t[1].clone()],
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

    fn on_resize(&mut self, _device: &wgpu::Device, _size: helio_core::graph::RenderSize) {
        for view in self.views.get_mut().unwrap() {
            view.targets = None;
        }
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let camera = ctx.camera_data;
        let id = camera.jitter_frame[3].to_bits();
        let size = [ctx.width.max(1), ctx.height.max(1)];
        let side = self.signal.block.side();
        let traced = self.signal.traced_size(size);
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
            views[index].targets = Some(Targets::new(ctx.device, size, traced, side));
        }
        self.active = Some(index);
        let proj = glam::DMat4::from_cols_array(&camera.proj.map(f64::from));
        let inv_proj = proj.inverse().to_cols_array().map(|v| v as f32);
        let p = camera.proj;
        // A pixel's width a metre of view depth (the projection's vertical
        // scale: 2 / (P[1][1] h)).
        let pixel_scale = 2.0 / (p[5].abs().max(1e-12) * size[1] as f32);
        let primed = views[index].targets.as_ref().unwrap().primed;
        // The view matrix's rotation alone (frame space to view space).
        let v = camera.view;
        let view_rotation = [
            v[0], v[1], v[2], 0.0, v[4], v[5], v[6], 0.0, v[8], v[9], v[10], 0.0, 0.0, 0.0, 0.0,
            1.0,
        ];
        let params = Params {
            inv_proj,
            view_rotation,
            // P[2][2], P[2][3], P[3][2], P[3][3] (column-major).
            proj_zw: [p[10], p[11], p[14], p[15]],
            size,
            traced,
            side,
            frame: ctx.frame_num as u32,
            flags: u32::from(self.signal.confidence) | (u32::from(primed) << 1),
            max_history: self.settings.max_history,
            pixel_scale,
            _pad: [0; 7],
        };
        ctx.write_buffer(&self.kernels.params, 0, bytemuck::bytes_of(&params));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        self.ran = false;
        let key = |name| helio_core::ResourceKey::new(name);
        let view = |name| ctx.registry.texture_view(key(name));
        let (Some(signal0), Some(signal1), Some(pixels), Some(depth), Some(normal)) = (
            view(self.signal.inputs[0]),
            view(self.signal.inputs[1]),
            view(history::PIXELS),
            view(history::DEPTH),
            view(history::NORMAL),
        ) else {
            // The producer or the history wrote nothing this frame: history
            // is kept.
            return Ok(());
        };
        let k = &self.kernels;
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
                texture_entry(0, depth),
                texture_entry(1, normal),
                texture_entry(2, pixels),
                buffer_entry(3, &k.params),
            ],
        );
        let [ha, hb] = &targets.history;
        let [ta, tb] = &targets.temp;
        let latest = targets.moments_latest;
        let (hm, tm) = (&targets.moments[latest], &targets.moments[latest ^ 1]);
        let temporal = group(
            &k.temporal_layout,
            &[
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
        // The records: the blocks' (a pass of their own), or the pixels'
        // temporal result itself.
        let (records, ping, pong) = if targets.records.len() == 3 {
            let [ra, rb, rn] = &targets.records[0];
            let g = group(
                &k.records_layout,
                &[
                    texture_entry(8, ta),
                    texture_entry(9, tb),
                    texture_entry(10, tm),
                    texture_entry(11, ra),
                    texture_entry(12, rb),
                    texture_entry(13, rn),
                ],
            );
            (Some(g), 1, 2)
        } else {
            (None, 0, 1)
        };
        let first: [&wgpu::TextureView; 3] = if records.is_some() {
            let [a, b, n] = &targets.records[0];
            [a, b, n]
        } else {
            [ta, tb, tm]
        };
        let iteration =
            |i: usize, input: [&wgpu::TextureView; 3], output: &[wgpu::TextureView; 3]| {
                group(
                    &k.spatial_layout,
                    &[
                        texture_entry(14, input[0]),
                        texture_entry(15, input[1]),
                        texture_entry(16, input[2]),
                        texture_entry(17, &output[0]),
                        texture_entry(18, &output[1]),
                        texture_entry(19, &output[2]),
                        buffer_entry(20, &k.iterations[i]),
                    ],
                )
            };
        let [pa, pb, pn] = &targets.records[ping];
        let spatial = [
            iteration(0, first, &targets.records[ping]),
            iteration(1, [pa, pb, pn], &targets.records[pong]),
        ];
        let [fa, fb, fn_] = &targets.records[pong];
        let reconstruct = group(
            &k.reconstruct_layout,
            &[
                texture_entry(21, ta),
                texture_entry(22, tb),
                texture_entry(23, tm),
                texture_entry(24, fa),
                texture_entry(25, fb),
                texture_entry(26, ha),
                texture_entry(27, hb),
                texture_entry(28, fn_),
            ],
        );
        let pixel_groups = [targets.size[0].div_ceil(8), targets.size[1].div_ceil(8)];
        let traced = self.signal.traced_size(targets.size);
        let traced_groups = [traced[0].div_ceil(8), traced[1].div_ceil(8)];
        let mut cmds = ctx.graphics_cmds();
        let stage = |cmds: &mut helio_core::CommandRecorder<'_>,
                     ctx: &mut PassContext,
                     scope: &'static str,
                     pipeline: &wgpu::ComputePipeline,
                     stage: &[&wgpu::BindGroup],
                     groups: [u32; 2]| {
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
        stage(
            &mut cmds,
            ctx,
            "Denoise: temporal",
            &k.temporal,
            &[&temporal],
            pixel_groups,
        );
        if let Some(records) = &records {
            stage(
                &mut cmds,
                ctx,
                "Denoise: records",
                &k.records,
                &[records],
                traced_groups,
            );
        }
        stage(
            &mut cmds,
            ctx,
            "Denoise: spatial",
            &k.spatial,
            &[&spatial[0], &spatial[1]],
            traced_groups,
        );
        stage(
            &mut cmds,
            ctx,
            "Denoise: reconstruct",
            &k.reconstruct,
            &[&reconstruct],
            pixel_groups,
        );
        targets.primed = true;
        targets.moments_latest = latest ^ 1;
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
        // The history's A and B are the signal's outputs.
        for (name, view) in self.signal.outputs.iter().zip(&targets.history) {
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
