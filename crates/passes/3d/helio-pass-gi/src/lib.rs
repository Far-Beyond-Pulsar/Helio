//! World-space diffuse global illumination (README.md): a camera-centred
//! clipmap of irradiance probes, kept up to date by rays the scene's tracers
//! trace, read by deferred lighting at every pixel.
//!
//! [`GiPass`] runs before the tracers: it integrates last frame's hits into
//! their probes, resets the probes that came into the clipmap and writes
//! this frame's rays (`gi_rays`). Tracers (the voxel terrain's today) write
//! a hit a ray (`gi_hits`). [`GiResolvePass`] runs after them, before
//! deferred lighting: the probes' irradiance at every pixel
//! (`gi_irradiance`).
//!
//! The clipmap is a function of the eye's world position alone
//! ([`Clipmap::around`], [`eye`]): a tracer that keeps state per probe (the
//! voxel terrain locates each probe's cell once) follows it by computing it
//! too.

use bytemuck::{Pod, Zeroable};
use glam::{DVec3, I64Vec3, Vec3};
use helio_core::graph::ResourceBuilder;
use helio_core::{PassContext, PrepareContext, RenderPass, ResourceKey, Result as HelioResult};
use helio_pass_sky::atmosphere::{
    ATMOSPHERE_FRAME, ATMOSPHERE_FRAME_BYTES, ATMOSPHERE_SKY_TABLE, ATMOSPHERE_SKY_TABLE_BYTES,
    ATMOSPHERE_SNIPPET,
};
use std::collections::VecDeque;

/// Levels of the clipmap, each twice as far apart as the one inside it.
pub const LEVELS: usize = 6;
/// Probes along a level's side.
pub const SIDE: i64 = 24;
pub const LEVEL_PROBES: u32 = (SIDE * SIDE * SIDE) as u32;
pub const PROBES: u32 = LEVEL_PROBES * LEVELS as u32;
/// The finest level's probe spacing (metres).
pub const FINEST_SPACING: f64 = 1.0;
/// Rays a probe update traces.
pub const RAYS: u32 = 64;
/// Probes updated a frame, a level.
pub const LEVEL_UPDATES: usize = 680;
pub const MAX_UPDATES: u32 = (LEVEL_UPDATES * LEVELS) as u32;
pub const MAX_RAYS: u32 = MAX_UPDATES * RAYS;
/// Words of a probe's record (`gi_common.wgsl`).
pub const PROBE_WORDS: u32 = 72;

/// Graph keys of what the passes publish.
pub const GI_FRAME: &str = "gi_frame";
pub const GI_PROBES: &str = "gi_probes";
pub const GI_RAYS: &str = "gi_rays";
pub const GI_HITS: &str = "gi_hits";
pub const GI_SUN: &str = "gi_sun";
pub const GI_IRRADIANCE: &str = "gi_irradiance";
/// Bytes of a `GiRay` and a `GiHit`.
pub const RAY_BYTES: u64 = 32;
pub const HIT_BYTES: u64 = 16;

/// The probes' shared WGSL (`gi_common.wgsl`: the clipmap, a probe's record,
/// rays and hits): a tracer opts in with `//!use gi_common`.
pub const GI_COMMON_SNIPPET: helio_core::shader::ShaderSnippet =
    helio_core::wgsl_snippet!("//!use gi_common", "../shaders/gi_common.wgsl");
/// Irradiance from the probes at a point (`gi_sample.wgsl`, with
/// `gi_common`): `//!use gi_sample`.
pub const GI_SAMPLE_SNIPPET: helio_core::shader::ShaderSnippet =
    helio_core::wgsl_snippet!("//!use gi_sample", "../shaders/gi_sample.wgsl");

/// The probe spacing of `level`.
pub fn spacing(level: usize) -> f64 {
    FINEST_SPACING * f64::from(1u32 << level)
}

/// The eye's world position this frame: the frame's origin and the
/// camera's position in it.
pub fn eye(ctx: &PrepareContext) -> DVec3 {
    ctx.world_origin.unwrap_or_default() + Vec3::from_slice(&ctx.camera_data.position_near[..3]).as_dvec3()
}

/// Where each level's probes are: the world coordinates (in its spacings)
/// of its logical probe 0. Probes sit at whole multiples of their spacing
/// and are stored toroidally (storage coordinates are world coordinates
/// modulo [`SIDE`]), so a probe keeps its storage while it stays in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Clipmap {
    pub first: [I64Vec3; LEVELS],
}

impl Clipmap {
    /// The clipmap around world point `eye`: each level's grid with the eye
    /// between its middle probes.
    pub fn around(eye: DVec3) -> Self {
        let mut first = [I64Vec3::ZERO; LEVELS];
        for (level, first) in first.iter_mut().enumerate() {
            let cell = (eye / spacing(level)).floor();
            *first = I64Vec3::new(cell.x as i64, cell.y as i64, cell.z as i64) - (SIDE / 2 - 1);
        }
        Self { first }
    }

    /// The storage index of world probe `world` (in spacings) of `level`.
    pub fn index(level: usize, world: I64Vec3) -> u32 {
        let s = world.rem_euclid(I64Vec3::splat(SIDE));
        level as u32 * LEVEL_PROBES + (s.x + SIDE * (s.y + SIDE * s.z)) as u32
    }

    /// Probe `index`'s level and world coordinates (in its spacings).
    pub fn probe(&self, index: u32) -> (usize, I64Vec3) {
        let level = (index / LEVEL_PROBES) as usize;
        let at = i64::from(index % LEVEL_PROBES);
        let s = I64Vec3::new(at % SIDE, (at / SIDE) % SIDE, at / (SIDE * SIDE));
        let logical = (s - self.first[level]).rem_euclid(I64Vec3::splat(SIDE));
        (level, self.first[level] + logical)
    }

    /// Probe `index`'s world position.
    pub fn position(&self, index: u32) -> DVec3 {
        let (level, world) = self.probe(index);
        world.as_dvec3() * spacing(level)
    }

    /// Whether world probe `world` of `level` is in the clipmap.
    pub fn holds(&self, level: usize, world: I64Vec3) -> bool {
        let d = world - self.first[level];
        d.cmpge(I64Vec3::ZERO).all() && d.cmplt(I64Vec3::splat(SIDE)).all()
    }

    /// The storage indices of the probes in this clipmap that were not in
    /// `before` (all of them without one).
    pub fn entered(&self, before: Option<&Clipmap>) -> Vec<u32> {
        let mut out = Vec::new();
        for level in 0..LEVELS {
            if before.is_some_and(|b| b.first[level] == self.first[level]) {
                continue;
            }
            for z in 0..SIDE {
                for y in 0..SIDE {
                    for x in 0..SIDE {
                        let world = self.first[level] + I64Vec3::new(x, y, z);
                        if !before.is_some_and(|b| b.holds(level, world)) {
                            out.push(Self::index(level, world));
                        }
                    }
                }
            }
        }
        out
    }

    /// The frame's levels for the GPU, relative to world point `eye`.
    pub fn levels(&self, eye: DVec3) -> [GpuLevel; LEVELS] {
        std::array::from_fn(|level| {
            let s = spacing(level);
            let origin = (self.first[level].as_dvec3() * s - eye).as_vec3();
            let base = self.first[level].rem_euclid(I64Vec3::splat(SIDE));
            GpuLevel {
                origin: [origin.x, origin.y, origin.z, s as f32],
                base: [base.x as u32, base.y as u32, base.z as u32, 0],
            }
        })
    }
}

/// `GiLevel` (`gi_common.wgsl`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuLevel {
    pub origin: [f32; 4],
    pub base: [u32; 4],
}

/// `GiFrame` (`gi_common.wgsl`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuFrame {
    pub levels: [GpuLevel; LEVELS],
    pub sky: [f32; 4],
    pub ground: [f32; 4],
    pub counts: [u32; 4],
    pub rotation: [f32; 4],
    pub prev_rotation: [f32; 4],
}

pub const FRAME_BYTES: u64 = std::mem::size_of::<GpuFrame>() as u64;
const _: () = assert!(FRAME_BYTES == 272);

/// Which probes update each frame: per level, those that came in (oldest
/// first), then the others in turn.
struct Schedule {
    entered: [VecDeque<u32>; LEVELS],
    queued: Vec<bool>,
    cursor: [u32; LEVELS],
}

impl Schedule {
    fn new() -> Self {
        Self { entered: Default::default(), queued: vec![false; PROBES as usize], cursor: [0; LEVELS] }
    }

    fn enter(&mut self, index: u32) {
        if !std::mem::replace(&mut self.queued[index as usize], true) {
            self.entered[(index / LEVEL_PROBES) as usize].push_back(index);
        }
    }

    fn next(&mut self, out: &mut Vec<u32>) {
        out.clear();
        for level in 0..LEVELS {
            let start = out.len();
            while out.len() - start < LEVEL_UPDATES {
                let Some(index) = self.entered[level].pop_front() else { break };
                self.queued[index as usize] = false;
                out.push(index);
            }
            while out.len() - start < LEVEL_UPDATES {
                out.push(level as u32 * LEVEL_PROBES + self.cursor[level]);
                self.cursor[level] = (self.cursor[level] + 1) % LEVEL_PROBES;
            }
        }
    }
}

/// A unit quaternion, uniformly random from `seed` (Shoemake).
fn rotation(seed: u64) -> [f32; 4] {
    let mut h = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0xd1b5_4a32_d192_ed03;
    let mut next = || {
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 33;
        (h >> 11) as f64 / (1u64 << 53) as f64
    };
    let (u1, u2, u3) = (next(), next(), next());
    let tau = std::f64::consts::TAU;
    let (a, b) = ((1.0 - u1).sqrt(), u1.sqrt());
    [
        (a * (tau * u2).sin()) as f32,
        (a * (tau * u2).cos()) as f32,
        (b * (tau * u3).sin()) as f32,
        (b * (tau * u3).cos()) as f32,
    ]
}

fn storage(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform(binding: u32, size: u64) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(size),
        },
        count: None,
    }
}

fn buffer(device: &wgpu::Device, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false })
}

/// The probes' state across frames (kept over graph rebuilds).
struct Probes {
    frame: wgpu::Buffer,
    probes: wgpu::Buffer,
    rays: wgpu::Buffer,
    hits: wgpu::Buffer,
    lists: wgpu::Buffer,
    sun: wgpu::Buffer,
    clipmap: Option<Clipmap>,
    schedule: Schedule,
    /// Probes updated last frame (their rays' hits are in `hits`).
    updated: u32,
    parity: u32,
    prev_rotation: [f32; 4],
}

impl Probes {
    fn new(device: &wgpu::Device) -> Self {
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        Self {
            frame: buffer(device, "GI frame", FRAME_BYTES, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST),
            probes: buffer(device, "GI probes", u64::from(PROBES * PROBE_WORDS) * 4, storage),
            rays: buffer(device, "GI rays", u64::from(MAX_RAYS) * RAY_BYTES, storage),
            hits: buffer(device, "GI hits", u64::from(MAX_RAYS) * HIT_BYTES, storage),
            lists: buffer(device, "GI lists", u64::from(2 * MAX_UPDATES + PROBES) * 4, storage),
            sun: buffer(device, "GI sun", 32, storage),
            clipmap: None,
            schedule: Schedule::new(),
            updated: 0,
            parity: 0,
            prev_rotation: [0.0, 0.0, 0.0, 1.0],
        }
    }
}

/// The probes' update (module docs): before the tracers.
pub struct GiPass {
    state: Probes,
    layout: wgpu::BindGroupLayout,
    sun: wgpu::ComputePipeline,
    integrate: wgpu::ComputePipeline,
    reset: wgpu::ComputePipeline,
    rays: wgpu::ComputePipeline,
    fallback_atmosphere: wgpu::Buffer,
    fallback_sky_table: wgpu::Buffer,
    fallback_lights: wgpu::Buffer,
    group: Option<(wgpu::BindGroup, [wgpu::Buffer; 3])>,
    /// This frame's probe counts: updated, reset.
    counts: [u32; 2],
    list: Vec<u32>,
}

impl GiPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GI update"),
            entries: &[
                uniform(0, FRAME_BYTES),
                storage(1, false),
                storage(2, false),
                storage(3, true),
                storage(4, true),
                storage(5, false),
                storage(6, true),
                uniform(7, ATMOSPHERE_FRAME_BYTES),
                uniform(8, ATMOSPHERE_SKY_TABLE_BYTES),
            ],
        });
        let module = helio_core::shader::module_with(
            device,
            "GI update",
            helio_core::include_wgsl!("../shaders/gi_update.wgsl"),
            &[ATMOSPHERE_SNIPPET, GI_COMMON_SNIPPET, GI_SAMPLE_SNIPPET],
        );
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("GI update"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let uniform = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
        Self {
            state: Probes::new(device),
            sun: pipeline("gi_sun_main"),
            integrate: pipeline("gi_integrate"),
            reset: pipeline("gi_reset"),
            rays: pipeline("gi_rays_main"),
            layout,
            fallback_atmosphere: buffer(device, "GI no atmosphere", ATMOSPHERE_FRAME_BYTES, uniform),
            fallback_sky_table: buffer(device, "GI no sky table", ATMOSPHERE_SKY_TABLE_BYTES, uniform),
            fallback_lights: buffer(device, "GI no lights", 128, wgpu::BufferUsages::STORAGE),
            group: None,
            counts: [0; 2],
            list: Vec::with_capacity(MAX_UPDATES as usize),
        }
    }
}

impl RenderPass for GiPass {
    fn name(&self) -> &'static str {
        "GI"
    }

    fn reads(&self) -> &'static [&'static str] {
        &[ATMOSPHERE_FRAME, ATMOSPHERE_SKY_TABLE, "render_environment"]
    }

    fn writes(&self) -> &'static [&'static str] {
        &[GI_FRAME, GI_PROBES, GI_RAYS, GI_HITS, GI_SUN]
    }

    fn declare_resources(&self, _builder: &mut ResourceBuilder) {}

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a helio_core::ResourceRegistry<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn inherit_persistent_state(&mut self, previous: &mut dyn RenderPass) -> bool {
        let Some(previous) = previous.as_any_mut().downcast_mut::<Self>() else { return false };
        std::mem::swap(&mut self.state, &mut previous.state);
        true
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let state = &mut self.state;
        let eye = eye(ctx);
        let clipmap = Clipmap::around(eye);
        let entered = clipmap.entered(state.clipmap.as_ref());
        for &index in &entered {
            state.schedule.enter(index);
        }
        state.clipmap = Some(clipmap);
        state.schedule.next(&mut self.list);
        let parity = 1 - state.parity;
        let rotation = rotation(ctx.frame_num);
        let environment = ctx.registry.get::<helio_core::RenderEnvironment>(helio_core::resource_keys::render_environment());
        let (ambient, intensity) = environment.map_or(([0.5, 0.5, 0.6], 1.0), |e| (e.ambient_color, e.ambient_intensity));
        let sky = Vec3::from(ambient) * intensity;
        let frame = GpuFrame {
            levels: clipmap.levels(eye),
            sky: sky.extend(0.0).to_array(),
            ground: (sky * 0.15).extend(0.0).to_array(),
            counts: [self.list.len() as u32, state.updated, entered.len() as u32, parity],
            rotation,
            prev_rotation: state.prev_rotation,
        };
        ctx.write_buffer(&state.frame, 0, bytemuck::bytes_of(&frame));
        ctx.write_buffer(&state.lists, u64::from(parity * MAX_UPDATES) * 4, bytemuck::cast_slice(&self.list));
        if !entered.is_empty() {
            ctx.write_buffer(&state.lists, u64::from(2 * MAX_UPDATES) * 4, bytemuck::cast_slice(&entered));
        }
        self.counts = [self.list.len() as u32, entered.len() as u32];
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let atmosphere = ctx.registry.get::<&wgpu::Buffer>(ResourceKey::new(ATMOSPHERE_FRAME)).unwrap_or(&self.fallback_atmosphere);
        let sky_table = ctx.registry.get::<&wgpu::Buffer>(ResourceKey::new(ATMOSPHERE_SKY_TABLE)).unwrap_or(&self.fallback_sky_table);
        let lights = ctx.scene_buffers.get(helio_core::BufferKey::of("scene_lights")).map_or(&self.fallback_lights, |h| &h.buffer);
        let key = [atmosphere.clone(), sky_table.clone(), lights.clone()];
        if self.group.as_ref().is_none_or(|(_, k)| *k != key) {
            let s = &self.state;
            let group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("GI update"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: s.frame.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: s.probes.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: s.rays.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: s.hits.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: s.lists.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: s.sun.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: lights.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: atmosphere.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: sky_table.as_entire_binding() },
                ],
            });
            self.group = Some((group, key));
        }
        let Some((group, _)) = &self.group else { return Ok(()) };
        let state = &mut self.state;
        let mut cmds = ctx.graphics_cmds();
        {
            let mut pass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("GI update"), timestamp_writes: None });
            pass.set_bind_group(0, Some(group), &[]);
            pass.set_pipeline(&self.sun);
            pass.dispatch_workgroups(1, 1, 1);
            if state.updated > 0 {
                pass.set_pipeline(&self.integrate);
                pass.dispatch_workgroups(state.updated, 1, 1);
            }
            if self.counts[1] > 0 {
                pass.set_pipeline(&self.reset);
                pass.dispatch_workgroups(self.counts[1].div_ceil(64), 1, 1);
            }
            pass.set_pipeline(&self.rays);
            pass.dispatch_workgroups((self.counts[0] * RAYS).div_ceil(64), 1, 1);
        }
        // The tracers write each ray's hit, or leave it none.
        cmds.clear_buffer(&state.hits, 0, None);
        state.updated = self.counts[0];
        state.parity = 1 - state.parity;
        state.prev_rotation = rotation(ctx.frame_num);
        Ok(())
    }

    fn publish<'a>(&self, frame: &mut helio_core::ResourceRegistry<'a>) {
        let s = &self.state;
        frame.write_buffer_binding(GI_FRAME, &s.frame, "GI");
        frame.write_buffer_binding(GI_PROBES, &s.probes, "GI");
        frame.write_buffer_binding(GI_RAYS, &s.rays, "GI");
        frame.write_buffer_binding(GI_HITS, &s.hits, "GI");
        frame.write_buffer_binding(GI_SUN, &s.sun, "GI");
    }
}

/// The probes' irradiance at every pixel (module docs): after the tracers,
/// before deferred lighting.
pub struct GiResolvePass {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    out: Option<(wgpu::Buffer, [u32; 2])>,
    group: Option<(wgpu::BindGroup, (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, usize, usize, wgpu::Buffer))>,
}

impl GiResolvePass {
    pub fn new(device: &wgpu::Device) -> Self {
        let texture = |binding: u32, sample_type: wgpu::TextureSampleType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GI resolve"),
            entries: &[
                uniform(0, FRAME_BYTES),
                storage(1, true),
                storage(2, true),
                texture(3, wgpu::TextureSampleType::Float { filterable: false }),
                texture(4, wgpu::TextureSampleType::Depth),
                storage(5, false),
            ],
        });
        let module = helio_core::shader::module_with(
            device,
            "GI resolve",
            helio_core::include_wgsl!("../shaders/gi_resolve.wgsl"),
            &[GI_COMMON_SNIPPET, GI_SAMPLE_SNIPPET],
        );
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("GI resolve"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gi_resolve"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("gi_resolve"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { layout, pipeline, out: None, group: None }
    }
}

impl RenderPass for GiResolvePass {
    fn name(&self) -> &'static str {
        "GIResolve"
    }

    fn reads(&self) -> &'static [&'static str] {
        &["gbuffer", "depth", GI_FRAME, GI_PROBES]
    }

    fn writes(&self) -> &'static [&'static str] {
        &[GI_IRRADIANCE]
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.read("gbuffer");
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a helio_core::ResourceRegistry<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let (Some(frame), Some(probes)) = (ctx.registry.buffer_binding(GI_FRAME), ctx.registry.buffer_binding(GI_PROBES)) else {
            self.out = None;
            return Ok(());
        };
        let Some(gbuffer) = ctx.registry.get::<helio_core::ViewGroup<'_, 4>>(ResourceKey::new("gbuffer")) else {
            return Err(helio_core::Error::ResourceNotFound("gbuffer".into()));
        };
        let size = [ctx.width.max(1), ctx.height.max(1)];
        if self.out.as_ref().is_none_or(|(_, s)| *s != size) {
            let bytes = u64::from(size[0]) * u64::from(size[1]) * 8;
            self.out = Some((buffer(ctx.device, "GI irradiance", bytes, wgpu::BufferUsages::STORAGE), size));
        }
        let Some((out, _)) = &self.out else { return Ok(()) };
        // Views have no identity to compare: their addresses stand in (the
        // graph keeps its views for the frame's lifetime and rebuilds them
        // on resize, with a new `out`).
        let key = (
            frame.clone(),
            probes.clone(),
            ctx.camera.clone(),
            gbuffer.views[1] as *const wgpu::TextureView as usize,
            ctx.depth as *const wgpu::TextureView as usize,
            out.clone(),
        );
        if self.group.as_ref().is_none_or(|(_, k)| *k != key) {
            let group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("GI resolve"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: frame.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: probes.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: ctx.camera.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(gbuffer.views[1]) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(ctx.depth) },
                    wgpu::BindGroupEntry { binding: 5, resource: out.as_entire_binding() },
                ],
            });
            self.group = Some((group, key));
        }
        let Some((group, _)) = &self.group else { return Ok(()) };
        let mut cmds = ctx.graphics_cmds();
        let mut pass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("GI resolve"), timestamp_writes: None });
        pass.set_bind_group(0, Some(group), &[]);
        pass.set_pipeline(&self.pipeline);
        pass.dispatch_workgroups(size[0].div_ceil(8), size[1].div_ceil(8), 1);
        Ok(())
    }

    fn publish<'a>(&self, frame: &mut helio_core::ResourceRegistry<'a>) {
        if let Some((out, _)) = &self.out {
            frame.write_buffer_binding(GI_IRRADIANCE, out, "GIResolve");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probes_keep_their_storage_while_they_stay_in() {
        let a = Clipmap::around(DVec3::new(10.3, -4.0, 1e6 + 0.5));
        let b = Clipmap::around(DVec3::new(13.9, -4.0, 1e6 + 0.5));
        for index in 0..PROBES {
            let (level, world) = b.probe(index);
            assert_eq!(Clipmap::index(level, world), index);
            assert!(b.holds(level, world));
        }
        let entered = b.entered(Some(&a));
        // Level 0 moved by 3 probes in x, level 1 by 1 or 2, coarser by at
        // most 1: whole slabs came in.
        for &index in &entered {
            let (level, world) = b.probe(index);
            assert!(!a.holds(level, world));
        }
        let count = |level: usize| entered.iter().filter(|&&i| (i / LEVEL_PROBES) as usize == level).count() as i64;
        let moved = |level: usize| (b.first[level].x - a.first[level].x).abs();
        for level in 0..LEVELS {
            assert_eq!(count(level), moved(level) * SIDE * SIDE);
        }
        assert_eq!(a.entered(None).len(), PROBES as usize);
    }

    #[test]
    fn the_eye_is_between_each_level_middle_probes() {
        let eye = DVec3::new(-123.45, 6.0, 789.0);
        let c = Clipmap::around(eye);
        for level in 0..LEVELS {
            let local = eye / spacing(level) - c.first[level].as_dvec3();
            assert!(local.cmpge(DVec3::splat(11.0)).all() && local.cmplt(DVec3::splat(12.0)).all(), "{level} {local}");
        }
    }

    #[test]
    fn updates_take_entered_probes_first_then_all_in_turn() {
        let mut s = Schedule::new();
        s.enter(5);
        s.enter(5);
        s.enter(LEVEL_PROBES + 7);
        let mut out = Vec::new();
        s.next(&mut out);
        assert_eq!(out.len(), MAX_UPDATES as usize);
        assert_eq!(out[0], 5);
        assert_eq!(out[1], 0);
        assert_eq!(out[LEVEL_UPDATES], LEVEL_PROBES + 7);
        // Every probe of a level comes round.
        let mut seen = vec![false; LEVEL_PROBES as usize];
        for _ in 0..(LEVEL_PROBES as usize).div_ceil(LEVEL_UPDATES) + 1 {
            s.next(&mut out);
            for &i in &out[..LEVEL_UPDATES] {
                seen[i as usize] = true;
            }
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn rotations_are_unit() {
        for seed in 0..64 {
            let q = rotation(seed);
            let n: f32 = q.iter().map(|c| c * c).sum();
            assert!((n - 1.0).abs() < 1e-5);
        }
    }
}

#[cfg(test)]
mod shader_tests {
    use super::*;

    fn validate(label: &str, source: &str, snippets: &[helio_core::shader::ShaderSnippet]) {
        let text = helio_core::shader::resolve_with(source, snippets);
        let module = wgpu::naga::front::wgsl::parse_str(&text).unwrap_or_else(|e| panic!("{label}: {}", e.emit_to_string(&text)));
        wgpu::naga::valid::Validator::new(wgpu::naga::valid::ValidationFlags::all(), wgpu::naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{label}: {}", e.emit_to_string(&text)));
    }

    #[test]
    fn shaders_validate() {
        validate("gi_update", include_str!("../shaders/gi_update.wgsl"), &[ATMOSPHERE_SNIPPET, GI_COMMON_SNIPPET, GI_SAMPLE_SNIPPET]);
        validate("gi_resolve", include_str!("../shaders/gi_resolve.wgsl"), &[GI_COMMON_SNIPPET, GI_SAMPLE_SNIPPET]);
    }
}
