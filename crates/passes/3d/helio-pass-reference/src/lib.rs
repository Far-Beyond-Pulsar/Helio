//! The reference path tracer: the image every lighting technique is judged
//! against (`docs/architecture/rendering.md`, section 4, "Direct light
//! samples and the reference").
//!
//! A consumer of the frame's ray streams (`helio_frame_api::rays::stream`)
//! and light samples (`rays::light_samples`) that knows no kind: whatever
//! kinds registered a tracer are its geometry, whatever kinds registered a
//! light sampler are its lights, and every surface is lit through the
//! frame's BSDF (`gbuffer::BSDF_SNIPPET`), the one the lighting models use.
//!
//! **A frame** (`reference.wgsl`) traces one path for each pixel of a window
//! of the image ([`ReferenceSettings::paths_per_frame`]; the window moves
//! on each frame, so a pixel gets a sample every `pixels / paths` frames),
//! as unrolled stages of the graph ([`ReferenceSettings::max_depth`]):
//!
//! 1. [`ReferenceBegin`]: camera rays from the eye, jittered within their
//!    pixel, as one run of the depth-0 stream (`Stream::with_run`: every ray
//!    shares the eye's origin, so a tracer that places an origin in its own
//!    structure does it once).
//! 2. At each depth: the kinds' tracers (`stream::trace`: nearest hits and
//!    their `Full` surfaces); [`ReferenceStage`] `Shade` (emission, or the
//!    sky on a miss; a light query a hit; the BSDF's sample for the next ray,
//!    Russian roulette from [`ReferenceSettings::roulette_depth`], live
//!    paths compacted into the next depth's stream); the light samplers
//!    (`light_samples::sample`); `Shadow` (the sample weighted by the BSDF and
//!    by multiple importance sampling against the BSDF's density, a shadow
//!    ray toward it); the tracers again (occlusion); `Resolve` (what got
//!    through).
//! 3. [`ReferenceAccumulate`]: each path's radiance into its pixel's 32-bit
//!    float sum and count. No cache, no denoiser, no clamp.
//! 4. [`ReferenceDisplay`]: the mean over the scene colour, where the host
//!    chose the reference's debug view ([`REFERENCE_VIEW`],
//!    [`REFERENCE_DIFFUSE_VIEW`]).
//!
//! The sum restarts when the camera (`camera_generation`), the frame's
//! origin, the image or what
//! the scene's buffers hold changes (`SceneBufferProjection::content_signature`).
//! Outside its views every pass records nothing and holds no buffer.
//!
//! **Why here.** A Helio pass crate (the feature layer): it needs only frame
//! contracts (streams, light samples, the surface and its BSDF, the sky, the
//! scene colour), so it can live below every plugin, and every host that
//! builds Helio's presets has it.

use std::sync::{Arc, Mutex};

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};
use helio_feature_api::{
    BuildEnv, Feature, FeatureRegistrar, NodeBuild, NodeDescriptor, ParamDesc,
};
use helio_frame_api::lighting::sky::{
    SKY_FRAME, SKY_FRAME_BYTES, SKY_SNIPPET, SKY_TABLE, SKY_TABLE_BYTES, SKY_TABLE_SNIPPET,
};
use helio_frame_api::rays::light_samples::{self, LightBatch, LightBatchBuffers};
use helio_frame_api::rays::stream::{self, RayClass, Stream, StreamBuffers, SurfaceDetail};
use helio_frame_api::{FrameNode, Slot};
use helio_settings::{number, Group, Resolved};

mod stages;
pub use stages::{ReferenceAccumulate, ReferenceDisplay, ReferenceImage, ReferenceStage, Step};

/// The debug view of the reference's image.
pub const REFERENCE_VIEW: u32 = 62;
/// The debug view of its indirect diffuse: the light that reached the
/// camera's hit by a first bounce off a diffuse lobe (what a real-time
/// indirect diffuse, the deferred lighting's own debug view 23, holds).
pub const REFERENCE_DIFFUSE_VIEW: u32 = 63;

/// The most stages a graph unrolls.
pub const MAX_DEPTH: u32 = 12;

/// The reference's settings (a renderer config's `features`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReferenceSettings {
    /// The image's size against the frame's internal resolution (0.1..1).
    pub resolution_scale: f32,
    /// Paths a frame (its rays budget: about three rays a path a depth).
    pub paths_per_frame: u32,
    /// Surfaces a path meets at most (1: direct light only).
    pub max_depth: u32,
    /// The depth from which Russian roulette ends paths by their throughput.
    pub roulette_depth: u32,
}

impl Default for ReferenceSettings {
    fn default() -> Self {
        Self {
            resolution_scale: 1.0,
            paths_per_frame: 1 << 19,
            max_depth: 6,
            roulette_depth: 3,
        }
    }
}

/// The reference's host settings.
pub static REFERENCE: Group = Group {
    key: "reference",
    title: "Reference Path Tracer",
    description: "The brute-force path tracer shown by the reference debug views: no cache, no \
                  denoiser, accumulated while the camera and the scene hold still",
    settings: &[
        number(
            "resolution_scale",
            "Reference Resolution Scale",
            "Reference Path Tracer",
            (0.1, 1.0, 0.05),
            1.0,
            "The reference image's size against the frame's internal resolution",
        ),
        number(
            "paths_per_frame",
            "Reference Paths a Frame",
            "Reference Path Tracer",
            (1024.0, 4194304.0, 1024.0),
            524288.0,
            "Paths traced a frame (a pixel each): fewer keep the editor responsive, more converge faster",
        ),
        number(
            "max_depth",
            "Reference Bounces",
            "Reference Path Tracer",
            (1.0, 12.0, 1.0),
            6.0,
            "Surfaces a path meets at most (1: direct light only)",
        ),
    ],
};

impl ReferenceSettings {
    /// The settings `settings` (of [`REFERENCE`]) hold.
    pub fn from_settings(settings: &Resolved) -> Self {
        let default = Self::default();
        Self {
            resolution_scale: settings
                .optional("resolution_scale")
                .map_or(default.resolution_scale, |v| v as f32),
            paths_per_frame: settings
                .optional("paths_per_frame")
                .map_or(default.paths_per_frame, |v| v as u32),
            max_depth: settings
                .optional("max_depth")
                .map_or(default.max_depth, |v| v as u32),
            roulette_depth: default.roulette_depth,
        }
    }

    fn clamped(self) -> Self {
        Self {
            resolution_scale: self.resolution_scale.clamp(0.1, 1.0),
            paths_per_frame: self.paths_per_frame.clamp(64, 1 << 22),
            max_depth: self.max_depth.clamp(1, MAX_DEPTH),
            roulette_depth: self.roulette_depth.max(1),
        }
    }
}

/// The snippets `reference.wgsl` is built with.
pub const SNIPPETS: [helio_core::shader::ShaderSnippet; 7] = [
    stream::SNIPPET,
    stream::SURFACE_SNIPPET,
    helio_frame_api::gbuffer::SURFACE_SNIPPET,
    helio_frame_api::gbuffer::BSDF_SNIPPET,
    light_samples::SNIPPET,
    SKY_SNIPPET,
    SKY_TABLE_SNIPPET,
];

/// `ReferenceFrame` (`reference.wgsl`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuFrame {
    clip_to_frame: [f32; 16],
    eye: [f32; 4],
    size: [u32; 4],
    info: [u32; 4],
    sky: [f32; 4],
    ground: [f32; 4],
}

const PATH_BYTES: u64 = 48;
const OWNER_BYTES: u64 = 32;

/// The streams and batches of a graph's stages (names `reference_d<d>`,
/// `reference_shadow_d<d>`).
#[derive(Clone)]
struct Streams {
    nearest: Vec<Stream>,
    shadows: Vec<Stream>,
    batches: Vec<LightBatch>,
}

impl Streams {
    fn new(settings: &ReferenceSettings) -> Self {
        let capacity = settings.paths_per_frame;
        let name = |what: String| helio_frame_api::static_name(&what);
        let mut nearest = Vec::new();
        let mut shadows = Vec::new();
        let mut batches = Vec::new();
        for d in 0..settings.max_depth {
            let mut s = Stream::new(name(format!("reference_d{d}")), RayClass::Nearest, capacity)
                .with_detail(SurfaceDetail::Full);
            if d == 0 {
                // Every camera ray leaves the eye: one run.
                s = s.with_run(capacity);
            }
            nearest.push(s);
            shadows.push(Stream::new(
                name(format!("reference_shadow_d{d}")),
                RayClass::Occlusion,
                capacity,
            ));
            batches.push(LightBatch::new(name(format!("reference_d{d}")), capacity));
        }
        Self {
            nearest,
            shadows,
            batches,
        }
    }
}

/// The reference's buffers while one of its views is shown.
struct Buffers {
    capacity: u32,
    size: [u32; 2],
    paths: wgpu::Buffer,
    /// The two nearest streams the stages alternate, their rows' paths.
    rays: [StreamBuffers; 2],
    rows: [wgpu::Buffer; 2],
    lights: LightBatchBuffers,
    owners: wgpu::Buffer,
    shadows: StreamBuffers,
    pending: wgpu::Buffer,
    image: wgpu::Buffer,
    diffuse: wgpu::Buffer,
    snapshot: wgpu::Buffer,
}

impl Buffers {
    fn new(device: &wgpu::Device, streams: &Streams, capacity: u32, size: [u32; 2]) -> Self {
        let storage = |label: &str, bytes: u64, extra: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(16),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | extra,
                mapped_at_creation: false,
            })
        };
        let rows = u64::from(capacity);
        let pixels = u64::from(size[0]) * u64::from(size[1]);
        let copy = wgpu::BufferUsages::COPY_SRC;
        Self {
            capacity,
            size,
            paths: storage(
                "Reference paths",
                rows * PATH_BYTES,
                wgpu::BufferUsages::empty(),
            ),
            rays: [
                StreamBuffers::new(device, &streams.nearest[0]),
                StreamBuffers::new(device, &streams.nearest[0]),
            ],
            rows: [
                storage("Reference rows A", rows * 4, wgpu::BufferUsages::empty()),
                storage("Reference rows B", rows * 4, wgpu::BufferUsages::empty()),
            ],
            lights: LightBatchBuffers::new(device, &streams.batches[0]),
            owners: storage(
                "Reference owners",
                rows * OWNER_BYTES,
                wgpu::BufferUsages::empty(),
            ),
            shadows: StreamBuffers::new(device, &streams.shadows[0]),
            pending: storage("Reference pending", rows * 16, wgpu::BufferUsages::empty()),
            image: storage("Reference image", pixels * 16, copy),
            diffuse: storage("Reference indirect diffuse", pixels * 16, copy),
            snapshot: storage("Reference snapshot", pixels * 16, copy),
        }
    }
}

/// What the reference's passes share, kept over graph rebuilds.
struct State {
    buffers: Option<Arc<Buffers>>,
    /// What the sum was taken under (camera, origin, image, scene); a
    /// change restarts it.
    key: Option<Vec<u64>>,
    /// Frames since the sum restarted, and the window's first pixel.
    frames: u32,
    cursor: u32,
    /// Paths this frame (0: nothing to do).
    paths: u32,
    clear: bool,
    /// The debug view shown.
    mode: u32,
    /// A host's request for a snapshot of the real-time scene colour.
    snapshot_requested: bool,
    snapshot_taken: bool,
}

impl State {
    fn active(&self) -> bool {
        self.mode == REFERENCE_VIEW || self.mode == REFERENCE_DIFFUSE_VIEW
    }
}

/// The pipelines every pass of a graph's reference shares.
struct Kernels {
    layout: wgpu::BindGroupLayout,
    camera: wgpu::ComputePipeline,
    shade: wgpu::ComputePipeline,
    shade_headers: wgpu::ComputePipeline,
    shadow: wgpu::ComputePipeline,
    shadow_header: wgpu::ComputePipeline,
    resolve: wgpu::ComputePipeline,
    accumulate: wgpu::ComputePipeline,
    frame: wgpu::Buffer,
    /// A uniform a stage holding its depth.
    stages: Vec<wgpu::Buffer>,
    no_sky: wgpu::Buffer,
    no_sky_table: wgpu::Buffer,
    stats: wgpu::Buffer,
    stat: helio_core::StatSlot,
    stats_handle: helio_core::FrameStats,
}

struct Shared {
    settings: ReferenceSettings,
    streams: Streams,
    kernels: Kernels,
    state: Mutex<State>,
    cache: helio_core::gpu_cache::GpuCache,
}

fn layout_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

impl Kernels {
    fn new(
        device: &wgpu::Device,
        settings: &ReferenceSettings,
        stats: &helio_core::FrameStats,
    ) -> Self {
        let storage =
            |b, read_only| layout_entry(b, wgpu::BufferBindingType::Storage { read_only });
        let uniform = |b| layout_entry(b, wgpu::BufferBindingType::Uniform);
        let mut entries = vec![uniform(0)];
        for b in 1..=20 {
            entries.push(storage(b, b == 5 || b == 13));
        }
        entries.extend([uniform(21), uniform(22), storage(23, false), uniform(24)]);
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Reference"),
            entries: &entries,
        });
        let module = helio_core::shader::module_with(
            device,
            "Reference",
            helio_core::include_wgsl!("../shaders/reference.wgsl"),
            &SNIPPETS,
        );
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Reference"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let constants = stats.slot_constants(&[("REFERENCE_STAT", "rays.reference")]);
        let pipeline = |entry: &str| {
            helio_core::shader::create_compute_pipeline(
                device,
                wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &constants,
                        ..Default::default()
                    },
                    cache: None,
                },
            )
        };
        let uniform_buffer = |label: &str, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        Self {
            camera: pipeline("reference_camera"),
            shade: pipeline("reference_shade"),
            shade_headers: pipeline("reference_shade_headers"),
            shadow: pipeline("reference_shadow"),
            shadow_header: pipeline("reference_shadow_header"),
            resolve: pipeline("reference_resolve"),
            accumulate: pipeline("reference_accumulate"),
            layout,
            frame: uniform_buffer("Reference frame", std::mem::size_of::<GpuFrame>() as u64),
            stages: (0..settings.max_depth)
                .map(|_| uniform_buffer("Reference stage", 16))
                .collect(),
            no_sky: uniform_buffer("Reference no sky", SKY_FRAME_BYTES),
            no_sky_table: uniform_buffer("Reference no sky table", SKY_TABLE_BYTES),
            stats: stats.buffer().clone(),
            stat: stats.slot("rays.reference"),
            stats_handle: stats.clone(),
        }
    }
}

impl Shared {
    /// Depth `d`'s bind group: its stream (`d % 2`) current, the other next.
    fn group(
        &self,
        device: &wgpu::Device,
        b: &Buffers,
        d: u32,
        sky: [&wgpu::Buffer; 2],
    ) -> wgpu::BindGroup {
        let k = &self.kernels;
        let (cur, next) = ((d % 2) as usize, ((d + 1) % 2) as usize);
        let surfaces = b.rays[cur]
            .surfaces
            .as_ref()
            .expect("nearest streams have surfaces");
        self.cache.bind_group(
            device,
            &wgpu::BindGroupDescriptor {
                label: Some("Reference"),
                layout: &k.layout,
                entries: &[
                    entry(0, &k.frame),
                    entry(1, &b.paths),
                    entry(2, &b.rays[cur].header),
                    entry(3, &b.rays[cur].rays),
                    entry(4, &b.rays[cur].hits),
                    entry(5, surfaces),
                    entry(6, &b.rows[cur]),
                    entry(7, &b.rays[next].header),
                    entry(8, &b.rays[next].rays),
                    entry(9, &b.rays[next].hits),
                    entry(10, &b.rows[next]),
                    entry(11, &b.lights.header),
                    entry(12, &b.lights.queries),
                    entry(13, &b.lights.samples),
                    entry(14, &b.owners),
                    entry(15, &b.shadows.header),
                    entry(16, &b.shadows.rays),
                    entry(17, &b.shadows.hits),
                    entry(18, &b.pending),
                    entry(19, &b.image),
                    entry(20, &b.diffuse),
                    entry(21, sky[0]),
                    entry(22, sky[1]),
                    entry(23, &k.stats),
                    entry(24, &k.stages[d as usize]),
                ],
            },
        )
    }

    fn sky<'a>(&'a self, registry: &helio_core::ResourceRegistry<'a>) -> [&'a wgpu::Buffer; 2] {
        let get = |name| registry.get::<&'a wgpu::Buffer>(helio_core::ResourceKey::new(name));
        [
            get(SKY_FRAME).unwrap_or(&self.kernels.no_sky),
            get(SKY_TABLE).unwrap_or(&self.kernels.no_sky_table),
        ]
    }
}

/// The reference as a feature: its node, its views, its settings.
pub struct ReferenceFeature;

impl Feature for ReferenceFeature {
    fn id(&self) -> &'static str {
        "helio.reference"
    }

    fn register(&self, registrar: &mut FeatureRegistrar<'_>) {
        registrar.node(
            NodeDescriptor::new(
                "helio.pass.reference",
                "Debug: Reference Path Tracer",
                "Passes: Debug",
                "A brute-force path tracer over every kind's ray tracer and light sampler, shown \
                 by its debug views (62: the image, 63: its indirect diffuse). Records nothing \
                 while neither is shown; builds nothing where nothing traces rays.",
                build,
            )
            .with_param(ParamDesc::boolean("enabled", "Enabled", true))
            .with_generated_pins()
            .once_per_frame()
            .in_slot(Slot::Lighting, 900),
        );
    }
}

fn settings(env: &BuildEnv<'_>) -> ReferenceSettings {
    env.setting::<ReferenceSettings>()
        .copied()
        .unwrap_or_default()
        .clamped()
}

/// The reference's passes (module docs).
fn build(build: &mut NodeBuild<'_, '_>) -> Result<(), String> {
    if stream::traced(build.env) == 0 {
        return Ok(());
    }
    let settings = settings(build.env);
    let streams = Streams::new(&settings);
    let device = build.env.device.clone();
    let stats = build.env.stats.clone();
    let shared = Arc::new(Shared {
        kernels: Kernels::new(&device, &settings, &stats),
        streams: streams.clone(),
        settings,
        state: Mutex::new(State {
            buffers: None,
            key: None,
            frames: 0,
            cursor: 0,
            paths: 0,
            clear: true,
            mode: helio_frame_api::lighting::lighting_settings(build.env).debug_mode,
            snapshot_requested: false,
            snapshot_taken: false,
        }),
        cache: helio_core::gpu_cache::GpuCache::new(),
    });
    for k in shared.kernels.stages.iter().enumerate() {
        helio_core::upload::write_buffer(
            build.env.queue,
            k.1,
            0,
            bytemuck::cast_slice(&[k.0 as u32, 0, 0, 0]),
        );
    }
    build.graph.add_pass(Box::new(ReferenceBegin {
        shared: shared.clone(),
    }));
    for d in 0..settings.max_depth {
        let d_usize = d as usize;
        stream::trace(build, &streams.nearest[d_usize])?;
        build.graph.add_pass(Box::new(ReferenceStage::new(
            shared.clone(),
            d,
            Step::Shade,
        )));
        light_samples::sample(build, &streams.batches[d_usize])?;
        build.graph.add_pass(Box::new(ReferenceStage::new(
            shared.clone(),
            d,
            Step::Shadow,
        )));
        stream::trace(build, &streams.shadows[d_usize])?;
        build.graph.add_pass(Box::new(ReferenceStage::new(
            shared.clone(),
            d,
            Step::Resolve,
        )));
    }
    build.graph.add_pass(Box::new(ReferenceAccumulate {
        shared: shared.clone(),
    }));
    let color = helio_frame_api::color::current(build.env);
    build.graph.add_pass(Box::new(ReferenceDisplay::new(
        &device,
        shared,
        color,
        build.env.scene_format(),
    )));
    Ok(())
}

/// Internal resource chaining the reference's passes in order.
const PATHS: &str = "reference_paths";
const IMAGE: &str = "reference_image";

/// The frame's start: buffers while a view is shown, the sum restarted on
/// change, the camera rays.
pub struct ReferenceBegin {
    shared: Arc<Shared>,
}

/// The camera's clip-to-frame transform without its jitter.
fn clip_to_frame(camera: &helio_core::GpuCameraUniforms) -> Mat4 {
    let view_proj = Mat4::from_cols_array(&camera.view_proj);
    let jitter = Mat4::from_translation(Vec3::new(
        camera.jitter_frame[0],
        camera.jitter_frame[1],
        0.0,
    ));
    (jitter.inverse() * view_proj).inverse()
}

impl RenderPass for ReferenceBegin {
    fn name(&self) -> &'static str {
        "ReferenceBegin"
    }

    fn writes(&self) -> &'static [&'static str] {
        let s = &self.shared.streams;
        let mut names = vec![PATHS];
        for d in 0..s.nearest.len() {
            names.extend([s.nearest[d].header, s.nearest[d].rays, s.nearest[d].hits]);
            names.extend(s.nearest[d].surfaces);
            names.extend([s.shadows[d].header, s.shadows[d].rays, s.shadows[d].hits]);
            names.extend([
                s.batches[d].header,
                s.batches[d].queries,
                s.batches[d].samples,
            ]);
        }
        helio_frame_api::static_names(&names)
    }

    fn reads(&self) -> &'static [&'static str] {
        &[
            SKY_FRAME,
            SKY_TABLE,
            helio_core::resource_keys::RENDER_ENVIRONMENT,
        ]
    }

    fn inherit_persistent_state(&mut self, previous: &mut dyn RenderPass) -> bool {
        let Some(previous) = previous.as_any_mut().downcast_mut::<Self>() else {
            return false;
        };
        if previous.shared.settings != self.shared.settings {
            return false;
        }
        let mut old = previous.shared.state.lock().unwrap();
        let mut new = self.shared.state.lock().unwrap();
        std::mem::swap(&mut *old, &mut *new);
        true
    }

    fn set_debug_mode(&mut self, mode: u32) {
        self.shared.state.lock().unwrap().mode = mode;
    }

    fn debug_views(&self) -> &'static [helio_core::DebugViewDescriptor] {
        static VIEWS: &[helio_core::DebugViewDescriptor] = &[
            helio_core::DebugViewDescriptor {
                name: "Reference Path Tracer",
                debug_mode: REFERENCE_VIEW,
                description: "The brute-force path tracer's image over every kind's tracer and light sampler, accumulated while the camera and scene hold still",
            },
            helio_core::DebugViewDescriptor {
                name: "Reference Indirect Diffuse",
                debug_mode: REFERENCE_DIFFUSE_VIEW,
                description: "The reference's light reaching the camera's hit by a first bounce off a diffuse lobe",
            },
        ];
        VIEWS
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let shared = &self.shared;
        let mut state = shared.state.lock().unwrap();
        state.paths = 0;
        if !state.active() {
            state.buffers = None;
            state.key = None;
            return Ok(());
        }
        let scale = shared.settings.resolution_scale;
        let size = [
            ((ctx.width as f32 * scale).round() as u32).max(1),
            ((ctx.height as f32 * scale).round() as u32).max(1),
        ];
        let capacity = shared.settings.paths_per_frame;
        if state
            .buffers
            .as_ref()
            .is_none_or(|b| b.size != size || b.capacity != capacity)
        {
            state.buffers = Some(Arc::new(Buffers::new(
                ctx.device,
                &shared.streams,
                capacity,
                size,
            )));
            state.key = None;
        }
        let camera = ctx.camera_data;
        let clip = clip_to_frame(camera);
        let origin = ctx.world_origin;
        // The camera as its host supplied it (`camera_generation`: the
        // renderer's jitter moves the uploaded matrices every frame and is
        // not a camera change; unjittering them on the GPU's side rounds
        // differently each frame).
        let mut key: Vec<u64> = vec![ctx.camera_generation];
        key.extend([origin.x, origin.y, origin.z].map(f64::to_bits));
        key.extend([
            u64::from(size[0]),
            u64::from(size[1]),
            u64::from(camera.view_info[3]),
        ]);
        key.push(ctx.scene_buffers.content_signature());
        if state.key.as_ref() != Some(&key) {
            if let Some(old) = &state.key {
                let scene = old.last() != key.last();
                log::debug!(
                    "reference: the sum restarts after {} frames ({})",
                    state.frames,
                    if scene {
                        "the scene's buffers changed"
                    } else {
                        "the camera or image changed"
                    }
                );
            }
            state.key = Some(key);
            state.frames = 0;
            state.cursor = 0;
            state.clear = true;
        }
        let pixels = size[0] * size[1];
        let paths = capacity.min(pixels);
        // A pixel's spread: the angle between neighbouring pixels' rays at
        // the image's centre.
        let ray = |ndc: Vec4| {
            let near = clip * ndc.with_z(1.0);
            let mid = clip * ndc.with_z(0.5);
            (mid.truncate() / mid.w - near.truncate() / near.w).normalize()
        };
        let spread = ray(Vec4::new(0.0, 0.0, 1.0, 1.0)).angle_between(ray(Vec4::new(
            0.0,
            2.0 / size[1] as f32,
            1.0,
            1.0,
        )));
        let environment = ctx
            .registry
            .get::<helio_core::RenderEnvironment>(helio_core::resource_keys::render_environment());
        let sky = environment.map_or(Vec3::ZERO, |e| {
            Vec3::from(e.ambient_color) * e.ambient_intensity
        });
        let gpu = GpuFrame {
            clip_to_frame: clip.to_cols_array(),
            eye: [
                camera.position_near[0],
                camera.position_near[1],
                camera.position_near[2],
                spread,
            ],
            size: [size[0], size[1], pixels, state.cursor],
            info: [
                paths,
                state.frames.wrapping_mul(0x9e37_79b9) ^ 0x6a09_e667,
                camera.view_info[3],
                shared.settings.max_depth - 1,
            ],
            sky: sky.extend(shared.settings.roulette_depth as f32).to_array(),
            ground: (sky * 0.15).extend(0.0).to_array(),
        };
        helio_core::upload::write_buffer(
            ctx.queue,
            &shared.kernels.frame,
            0,
            bytemuck::bytes_of(&gpu),
        );
        let buffers = state.buffers.as_ref().unwrap();
        buffers.rays[0].set_count(ctx.queue, paths);
        shared
            .kernels
            .stats_handle
            .add(shared.kernels.stat, u64::from(paths));
        state.paths = paths;
        state.cursor = (state.cursor + paths) % pixels;
        state.frames += 1;
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let shared = &self.shared;
        let mut state = shared.state.lock().unwrap();
        if state.paths == 0 {
            return Ok(());
        }
        let sky = shared.sky(ctx.registry);
        let buffers = state.buffers.as_ref().unwrap();
        shared.cache.end_frame();
        let group = shared.group(ctx.device, buffers, 0, sky);
        let mut cmds = ctx.graphics_cmds();
        if state.clear {
            cmds.clear_buffer(&buffers.image, 0, None);
            cmds.clear_buffer(&buffers.diffuse, 0, None);
        }
        let paths = state.paths;
        {
            let mut pass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Reference camera"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, Some(&group), &[]);
            pass.set_pipeline(&shared.kernels.camera);
            pass.dispatch_workgroups(paths.div_ceil(64), 1, 1);
        }
        drop(cmds);
        state.clear = false;
        Ok(())
    }

    fn publish<'a>(&self, frame: &mut helio_core::ResourceRegistry<'a>) {
        let state = self.shared.state.lock().unwrap();
        let (Some(b), true) = (state.buffers.as_ref(), state.paths > 0) else {
            return;
        };
        let s = &self.shared.streams;
        for d in 0..s.nearest.len() {
            // The stages run in order, each reading the stream its previous
            // stage wrote: two nearest streams, one batch and one shadow
            // stream serve every depth.
            b.rays[d % 2].publish(&s.nearest[d], frame, "ReferenceBegin");
            b.lights.publish(&s.batches[d], frame, "ReferenceBegin");
            b.shadows.publish(&s.shadows[d], frame, "ReferenceBegin");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate(source: &str, snippets: &[helio_core::shader::ShaderSnippet]) {
        let text = helio_core::shader::resolve_with(source, snippets);
        let module = wgpu::naga::front::wgsl::parse_str(&text)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&text)));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&text)));
    }

    #[test]
    fn the_shaders_validate() {
        validate(include_str!("../shaders/reference.wgsl"), &SNIPPETS);
        validate(include_str!("../shaders/reference_display.wgsl"), &[]);
    }

    /// The shader's rows are the sizes the buffers are made for.
    #[test]
    fn rows_have_their_sizes() {
        assert_eq!(std::mem::size_of::<GpuFrame>(), 144);
        let module = wgpu::naga::front::wgsl::parse_str(&helio_core::shader::resolve_with(
            include_str!("../shaders/reference.wgsl"),
            &SNIPPETS,
        ))
        .unwrap();
        let mut layouter = wgpu::naga::proc::Layouter::default();
        layouter.update(module.to_ctx()).unwrap();
        for (name, bytes) in [
            ("Path", PATH_BYTES),
            ("Owner", OWNER_BYTES),
            ("ReferenceFrame", 144),
        ] {
            let (handle, _) = module
                .types
                .iter()
                .find(|(_, ty)| ty.name.as_deref() == Some(name))
                .unwrap();
            assert_eq!(u64::from(layouter[handle].size), bytes, "{name}");
        }
    }

    #[test]
    fn host_defaults_are_the_default_settings() {
        let host = ReferenceSettings::from_settings(&REFERENCE.defaults());
        let default = ReferenceSettings::default();
        assert_eq!(host.max_depth, default.max_depth);
        assert_eq!(host.paths_per_frame, default.paths_per_frame);
    }

    /// Every depth's streams are their own resources; the camera's is one
    /// run.
    #[test]
    fn the_stages_name_their_streams() {
        let s = Streams::new(&ReferenceSettings::default());
        assert_eq!(s.nearest.len(), 6);
        assert_eq!(
            s.nearest[0].run,
            ReferenceSettings::default().paths_per_frame
        );
        assert_eq!(s.nearest[1].run, 1);
        assert_ne!(s.nearest[0].rays, s.nearest[1].rays);
        assert_eq!(s.shadows[2].class, RayClass::Occlusion);
    }
}
