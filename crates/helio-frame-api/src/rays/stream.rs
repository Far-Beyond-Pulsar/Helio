//! Ray streams: how anything in the frame traces rays against every kind of
//! geometry, without knowing a kind.
//!
//! A **consumer** (GI's probes, reflections, a radiance cache's update, a
//! reference path tracer, a lighting model's shadow rays) declares a
//! [`Stream`]: a batch of rays it writes each frame ([`SceneRay`] rows and a
//! [`StreamHeader`]) and what it wants back ([`RayClass`]). A **tracer** is a
//! kind's answer: the meshes trace the frame's TLAS, a voxel terrain walks
//! its chunks, and each writes the hits of its own geometry into the
//! stream's [`SceneHit`] rows, keeping the nearer. For a nearest-hit stream
//! the tracer that owns a hit then writes its [`SceneSurface`]: what the
//! consumer lights.
//!
//! A kind registers its tracer once ([`RegisterRayTracer::ray_tracer`]). A
//! consumer's node builds its generator pass, then calls [`trace`], which
//! adds every tracer's passes for that stream right there in the graph: the
//! consumer names no kind, a kind names no consumer, and an absent kind is a
//! tracer that is not there. [`traced`] says whether anything would answer,
//! for a consumer that builds nothing otherwise, and a stream's [`Answers`]
//! whether every tracer answered in full this frame (none was still
//! compiling or loading what it traces).
//!
//! Why batches instead of one `trace()` function spliced into every
//! consumer's shader:
//!
//! - **Kinds are plugins and not all triangles.** An inline megakernel needs
//!   every kind's intersection code, material code and bindings in every
//!   consumer's shader (a voxel walk binds its own chunk trees and pages).
//!   DXR and Vulkan solve that with ray-tracing pipelines and shader tables;
//!   wgpu has ray queries only. A stream is the shader table as data: the
//!   tracer id in a hit is the hit group.
//! - **Coherence.** Path tracers on ray-tracing pipelines need shader
//!   execution reordering to get material coherence back (NVIDIA's
//!   path-tracing guidance, RTXPT). A stream's hits are already rows a pass
//!   can bin by tracer and material, as the wavefront path tracers do (Laine
//!   et al., "Megakernels Considered Harmful", HPG 2013; Cycles and the
//!   production GPU renderers since).
//! - **Budgets.** Rays per signal are explicit rows with a count, so a
//!   consumer can hold a ray budget per frame (Call of Duty's variable-rate
//!   ray tracing) and the frame can report rays per signal.
//!
//! What it costs: a ray is 32 bytes, a hit 16, a surface 32, written and
//! read once each, and a stream's hit arrives in the same frame only after
//! the tracers' passes ran. A consumer that needs one visibility test per
//! pixel against triangles alone can still query the TLAS inline
//! ([`super::SCENE_RAYS`]); it then sees no other kind.
//!
//! Cost scales with the rays a consumer asks for, never with the scene's
//! size: a tracer's pass is one indirect dispatch over the stream's count.
//! A consumer whose rays leave one origin in runs says so
//! ([`Stream::with_run`]), and a tracer pays its per-origin work once a run.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use helio_core::shader::ShaderSnippet;
use helio_feature_api::{BuildEnv, FeatureRegistrar, NodeBuild};

/// Marker opting a shader into [`SNIPPET`].
pub const MARKER: &str = "//!use frame_ray_stream";
/// The rows' WGSL (`shaders/ray_stream.wgsl`): `RayStreamHeader`,
/// `SceneRay` (and its coordinate space), `SceneHit`, and the functions that
/// pack and read them (unit vectors, radiance).
pub const SNIPPET: ShaderSnippet = helio_core::wgsl_snippet!(MARKER, "../shaders/ray_stream.wgsl");

/// Marker opting a shader into [`SURFACE_SNIPPET`].
pub const SURFACE_MARKER: &str = "//!use frame_ray_surface";
/// A nearest-hit stream's surface rows (`shaders/ray_surface.wgsl`, with
/// [`SNIPPET`] and the frame's `frame_surface`,
/// [`crate::gbuffer::SURFACE_SNIPPET`]): `SceneSurface`,
/// `scene_surface_pack(surface, face)`, `scene_surface_unpack(row)` and
/// `scene_surface_face(row)`, the ray side's encode and decode of the
/// frame's `SurfaceData`. Apart from [`SNIPPET`] so a shader reading rays or
/// the packings alone needs no `SurfaceData`.
pub const SURFACE_SNIPPET: ShaderSnippet =
    helio_core::wgsl_snippet!(SURFACE_MARKER, "../shaders/ray_surface.wgsl");

/// Invocations of a tracer's workgroup: [`StreamHeader::groups`] is the
/// dispatch for workgroups this size (`RAY_STREAM_GROUP` in WGSL).
pub const GROUP: u32 = 64;

/// TLAS instance and ray mask bit: casters that never move.
pub const MASK_STATIC: u8 = 1;
/// TLAS instance and ray mask bit: casters that may move.
pub const MASK_MOVABLE: u8 = 2;
/// A ray that sees every caster (a ray's mask holds four bits).
pub const MASK_ALL: u8 = 0x0f;

/// Tracers a frame can hold: a hit names its tracer in four bits, 0 being
/// none.
pub const MAX_TRACERS: usize = 15;

/// `RayStreamHeader`: the indirect dispatch over a stream's rays, then
/// their count.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct StreamHeader {
    /// `[ceil(count / GROUP), 1, 1]`.
    pub groups: [u32; 3],
    pub count: u32,
}

impl StreamHeader {
    pub const BYTES: u64 = std::mem::size_of::<Self>() as u64;

    pub fn of(count: u32) -> Self {
        Self {
            groups: [count.div_ceil(GROUP), 1, 1],
            count,
        }
    }
}

/// `SceneRay` (frame space; see the WGSL), 32 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct SceneRay {
    pub origin: [f32; 3],
    /// The most distance the ray travels; 0 or less: no ray in this row.
    pub t_max: f32,
    pub dir: [f32; 3],
    /// The ray's cone ([`Self::cone`]), caster mask and coordinate space
    /// ([`Self::in_space`]).
    pub cone: u32,
}

/// A ray's coordinate space ([`SceneRay::space`]) when it sees every space
/// (as an unrestricted camera, `view_info.w` 0, does).
pub const SPACE_ANY: u32 = 0;

/// A ray's coordinate space ([`SceneRay::space`]) in authored space slot
/// `slot` (0: the world): `slot + 1`, a camera's `view_info.w`
/// (`helio_core::GpuCameraUniforms::with_coordinate_space`).
pub const fn space_of_slot(slot: u32) -> u32 {
    slot + 1
}

/// Where a ray's cone holds its space (9 bits: up to slot 255 plus one).
const SPACE_SHIFT: u32 = 15;
const SPACE_BITS: u32 = 0x1ff;

impl SceneRay {
    pub const BYTES: u64 = std::mem::size_of::<Self>() as u64;

    /// A ray from `origin` along unit `dir`, at most `t_max` long, with
    /// cone `cone` ([`Self::cone`]).
    pub fn new(origin: [f32; 3], t_max: f32, dir: [f32; 3], cone: u32) -> Self {
        Self {
            origin,
            t_max,
            dir,
            cone,
        }
    }

    /// `scene_cone_in_space`: the ray in coordinate space `space` (its
    /// encoding, [`SPACE_ANY`] or [`space_of_slot`]).
    pub fn in_space(mut self, space: u32) -> Self {
        self.cone =
            (self.cone & !(SPACE_BITS << SPACE_SHIFT)) | ((space & SPACE_BITS) << SPACE_SHIFT);
        self
    }

    /// `scene_ray_space`: the coordinate space it travels in.
    pub fn space(&self) -> u32 {
        (self.cone >> SPACE_SHIFT) & SPACE_BITS
    }

    /// `scene_ray_sees_space`: whether the ray sees geometry authored in
    /// space slot `slot`.
    pub fn sees_space(&self, slot: u32) -> bool {
        let space = self.space();
        space == SPACE_ANY || space == slot + 1
    }

    /// `scene_ray_cone`: a footprint `width` across at the origin that grows
    /// by `spread` a metre, seeing the casters of `mask` (four bits), in
    /// every coordinate space.
    pub fn cone(width: f32, spread: f32, mask: u8) -> u32 {
        let w = if width > 0.0 {
            (width.log2() * 8.0 + 160.0).round().clamp(1.0, 255.0) as u32
        } else {
            0
        };
        let s = ((u32::from(half_bits(spread)) + 8) >> 4).min(0x7bf);
        s | ((u32::from(mask) & 0xf) << 11) | (w << 24)
    }

    /// `scene_ray_spread`.
    pub fn spread(&self) -> f32 {
        half_value(((self.cone & 0x7ff) << 4) as u16)
    }

    /// `scene_ray_width`.
    pub fn width(&self) -> f32 {
        match self.cone >> 24 {
            0 => 0.0,
            w => ((w as f32 - 160.0) / 8.0).exp2(),
        }
    }

    /// `scene_ray_mask`.
    pub fn mask(&self) -> u8 {
        ((self.cone >> 11) & 0xf) as u8
    }
}

/// The half float nearest `value` (finite, zero or more; past the largest
/// half: the largest).
fn half_bits(value: f32) -> u16 {
    if value.is_nan() || value <= 0.0 {
        return 0;
    }
    let bits = value.to_bits();
    let exponent = (bits >> 23) as i32 - 127;
    let mantissa = bits & 0x007f_ffff;
    if exponent > 15 {
        return 0x7bff;
    }
    if exponent >= -14 {
        // Normal: 10 bits of mantissa, rounded to nearest (a carry runs
        // into the exponent, which is the next half).
        let half = (((exponent + 15) as u32) << 10) | (mantissa >> 13);
        let round = (mantissa >> 12) & 1;
        return (half + round).min(0x7bff) as u16;
    }
    if exponent >= -25 {
        // Subnormal: the implicit one shifted down.
        let shift = (-14 - exponent) as u32;
        let full = (mantissa | 0x0080_0000) >> (13 + shift);
        let round = ((mantissa | 0x0080_0000) >> (12 + shift)) & 1;
        return (full + round) as u16;
    }
    0
}

/// The value of half float `bits` (finite ones).
fn half_value(bits: u16) -> f32 {
    let exponent = i32::from((bits >> 10) & 0x1f);
    let mantissa = f32::from(bits & 0x3ff);
    let value = if exponent == 0 {
        mantissa * (-24.0f32).exp2()
    } else {
        (1.0 + mantissa / 1024.0) * ((exponent - 15) as f32).exp2()
    };
    if bits & 0x8000 != 0 {
        -value
    } else {
        value
    }
}

/// `SceneHit` (see the WGSL).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct SceneHit {
    pub t: f32,
    pub a: u32,
    pub b: u32,
    pub c: u32,
}

impl SceneHit {
    pub const BYTES: u64 = std::mem::size_of::<Self>() as u64;
    /// [`Self::a`]: the ray met the surface from its inside.
    pub const INSIDE: u32 = 0x0800_0000;
    /// [`Self::a`]: the tracer's own bits.
    pub const PAYLOAD: u32 = 0x07ff_ffff;

    /// The tracer that found the hit (0: no hit).
    pub fn tracer(&self) -> TracerId {
        TracerId(self.a >> 28)
    }

    pub fn inside(&self) -> bool {
        self.a & Self::INSIDE != 0
    }

    pub fn payload(&self) -> u32 {
        self.a & Self::PAYLOAD
    }

    /// What an occlusion stream's row lets through, per channel.
    pub fn through(&self) -> [f32; 3] {
        [0, 10, 20].map(|shift| 1.0 - ((self.b >> shift) & 1023) as f32 / 1023.0)
    }
}

/// `SceneSurface` (see the WGSL): a `SurfaceData` (the frame's surface at a
/// lit point, `crate::gbuffer::SURFACE_SNIPPET`) and the face's normal, in
/// 32 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct SceneSurface {
    pub normal: u32,
    pub face: u32,
    pub base_color: u32,
    pub params: u32,
    pub emissive: u32,
    pub specular: u32,
    pub subsurface: u32,
    pub anisotropy: u32,
}

/// `SurfaceData::flags` (`SURFACE_FLAG_*` in WGSL).
pub mod surface_flags {
    pub const SUBSURFACE: u32 = 1;
    pub const ANISOTROPIC: u32 = 2;
    pub const LOW_SPECULAR: u32 = 4;
    /// A thin two-sided surface: `subsurface` is the share of the light
    /// behind it that passes through, by colour.
    pub const TWO_SIDED: u32 = 8;
}

impl SceneSurface {
    pub const BYTES: u64 = std::mem::size_of::<Self>() as u64;

    /// Base colour (linear RGB) and metalness.
    pub fn base_color(&self) -> [f32; 4] {
        self.base_color
            .to_le_bytes()
            .map(|byte| f32::from(byte) / 255.0)
    }

    pub fn roughness(&self) -> f32 {
        (self.params & 0xff) as f32 / 255.0
    }

    /// The material's own occlusion.
    pub fn ao(&self) -> f32 {
        ((self.params >> 8) & 0xff) as f32 / 255.0
    }

    /// `SurfaceData::flags` ([`surface_flags`]).
    pub fn flags(&self) -> u32 {
        self.params >> 24
    }

    /// Reflectance at normal incidence.
    pub fn specular_f0(&self) -> [f32; 3] {
        [0, 10, 20].map(|shift| ((self.specular >> shift) & 1023) as f32 / 1023.0)
    }

    /// The share of the light behind the surface that passes through, by
    /// colour ([`surface_flags::TWO_SIDED`]; none: zero).
    pub fn transmission(&self) -> [f32; 3] {
        if self.flags() & surface_flags::TWO_SIDED != 0 {
            unpack_radiance(self.subsurface)
        } else {
            [0.0; 3]
        }
    }
}

/// `scene_pack_unit`: a unit vector in 32 bits.
pub fn pack_unit(n: [f32; 3]) -> u32 {
    let sum = (n[0].abs() + n[1].abs() + n[2].abs()).max(1e-20);
    let mut p = [n[0] / sum, n[1] / sum];
    if n[2] < 0.0 {
        let sign = |v: f32| if v >= 0.0 { 1.0 } else { -1.0 };
        p = [
            (1.0 - p[1].abs()) * sign(p[0]),
            (1.0 - p[0].abs()) * sign(p[1]),
        ];
    }
    let q = p.map(|v| ((v * 0.5 + 0.5).clamp(0.0, 1.0) * 65535.0 + 0.5) as u32);
    q[0] | (q[1] << 16)
}

/// `scene_unpack_unit`.
pub fn unpack_unit(p: u32) -> [f32; 3] {
    let e = [p & 0xffff, p >> 16].map(|v| v as f32 / 65535.0 * 2.0 - 1.0);
    let mut n = [e[0], e[1], 1.0 - e[0].abs() - e[1].abs()];
    let fold = (-n[2]).max(0.0);
    n[0] += if n[0] >= 0.0 { -fold } else { fold };
    n[1] += if n[1] >= 0.0 { -fold } else { fold };
    let length = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    n.map(|v| v / length)
}

/// `scene_pack_radiance`: linear radiance in 32 bits (RGB9E5).
pub fn pack_radiance(c: [f32; 3]) -> u32 {
    let v = c.map(|v| v.clamp(0.0, 65408.0));
    let most = v[0].max(v[1]).max(v[2]);
    let mut e = (most.max(1e-10).log2().floor() + 1.0).clamp(-15.0, 16.0);
    if (most * (9.0 - e).exp2() + 0.5).floor() >= 512.0 {
        e += 1.0;
    }
    let q = v.map(|v| (v * (9.0 - e).exp2() + 0.5).floor() as u32);
    q[0] | (q[1] << 9) | (q[2] << 18) | (((e + 15.0) as u32) << 27)
}

/// `scene_unpack_radiance`.
pub fn unpack_radiance(p: u32) -> [f32; 3] {
    let scale = ((p >> 27) as f32 - 24.0).exp2();
    [p & 511, (p >> 9) & 511, (p >> 18) & 511].map(|v| v as f32 * scale)
}

/// What a consumer wants of its rays.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RayClass {
    /// The nearest surface along each ray: a [`SceneHit`] and its
    /// [`SceneSurface`].
    Nearest,
    /// How much light gets from each ray's origin to its end
    /// ([`SceneHit::through`]): no surface is written, and a tracer may stop
    /// at the first thing that blocks it.
    Occlusion,
}

/// How much of the surface at a hit a nearest-hit stream needs
/// ([`Stream::detail`]). A tracer answers at least what a stream asks, and
/// may use a cheaper kernel for a lower level; every level writes a whole
/// [`SceneSurface`], a lower one leaving what it does not evaluate at
/// `surface_data`'s neutral values.
///
/// A property of the stream, not of each ray: a tracer picks its kernel per
/// stream when the graph is built (no per-ray branch over material code, no
/// bit in a full ray row), and a consumer with rays of several classes (a
/// path tracer's camera rays and its diffuse bounces) writes them into a
/// stream per class, which is how ray classes are budgeted anyway.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SurfaceDetail {
    /// What a diffuse consumer lights (probes, a radiance cache, diffuse
    /// bounces): base colour, emission, the shading normal and the share let
    /// through from behind, at the ray's footprint (`scene_ray_footprint`:
    /// textures at the mip the cone covers); roughness and metalness as the
    /// material's constants.
    Diffuse,
    /// The material in full, as the G-buffer's drawers evaluate it (its
    /// template or graph, normal maps, every `SurfaceData` field), at the
    /// ray's footprint: reflections, a specular consumer, the reference.
    #[default]
    Full,
}

/// A batch of rays a consumer writes each frame and the kinds' tracers
/// answer (module docs). Its buffers are graph resources under these names;
/// the consumer owns them ([`StreamBuffers`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Stream {
    pub name: &'static str,
    pub class: RayClass,
    /// The most rays it holds.
    pub capacity: u32,
    /// Rays come in runs of this many rows from one origin ([`Self::run`];
    /// 1: no runs).
    pub run: u32,
    /// What a nearest-hit stream needs of each hit's surface
    /// ([`Self::with_detail`]; [`SurfaceDetail::Full`] unless the
    /// consumer asks for less; occlusion streams write none).
    pub detail: SurfaceDetail,
    /// [`StreamHeader`], one.
    pub header: &'static str,
    /// [`SceneRay`] rows.
    pub rays: &'static str,
    /// [`SceneHit`] rows, one a ray.
    pub hits: &'static str,
    /// [`SceneSurface`] rows, one a ray ([`RayClass::Nearest`] only).
    pub surfaces: Option<&'static str>,
}

impl Stream {
    /// The stream `name` (unique in a graph: its resources are
    /// `<name>_ray_header`, `<name>_rays`, `<name>_hits`, `<name>_surfaces`).
    pub fn new(name: &'static str, class: RayClass, capacity: u32) -> Self {
        let resource = |suffix: &str| crate::static_name(&format!("{name}_{suffix}"));
        Self {
            name,
            class,
            capacity,
            run: 1,
            detail: SurfaceDetail::Full,
            header: resource("ray_header"),
            rays: resource("rays"),
            hits: resource("hits"),
            surfaces: (class == RayClass::Nearest).then(|| resource("surfaces")),
        }
    }

    /// A stream whose rays come in runs of `run` consecutive rows, from row
    /// 0 (the last run may be short), the rays of a run sharing their origin,
    /// caster mask and coordinate space: a probe's rays, a hit's shadow rays toward several
    /// lights, a pixel's primary and its samples. A tracer whose cost has a
    /// part that depends on the origin alone (placing it in the kind's own
    /// structure: a voxel terrain's chunk trees) may pay it once a run; one
    /// with none (a TLAS query) ignores it. This is the origin coherence a
    /// wavefront tracer recovers by sorting rays, stated by the consumer
    /// that made the rays that way.
    pub fn with_run(mut self, run: u32) -> Self {
        self.run = run.max(1);
        self
    }

    /// A stream whose hits need only `detail` of their surface
    /// ([`SurfaceDetail`]).
    pub fn with_detail(mut self, detail: SurfaceDetail) -> Self {
        self.detail = detail;
        self
    }

    /// The frame statistic counting this stream's rays (`rays.<name>`): the
    /// consumer adds its count each frame, so a host reads rays per signal.
    pub fn stat(&self, stats: &helio_core::FrameStats) -> helio_core::StatSlot {
        stats.slot(&format!("rays.{}", self.name))
    }
}

/// A stream's buffers, made and held by its consumer's generator pass, which
/// writes the header and the rays, clears the hits each frame
/// ([`Self::clear`]) and publishes them ([`Self::publish`]).
pub struct StreamBuffers {
    pub header: wgpu::Buffer,
    pub rays: wgpu::Buffer,
    pub hits: wgpu::Buffer,
    pub surfaces: Option<wgpu::Buffer>,
}

impl StreamBuffers {
    pub fn new(device: &wgpu::Device, stream: &Stream) -> Self {
        let rows = u64::from(stream.capacity.max(1));
        let buffer = |what: &str, size: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{} {what}", stream.name)),
                size,
                usage: usage | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        Self {
            header: buffer(
                "ray header",
                StreamHeader::BYTES,
                wgpu::BufferUsages::INDIRECT,
            ),
            rays: buffer("rays", rows * SceneRay::BYTES, wgpu::BufferUsages::empty()),
            hits: buffer("hits", rows * SceneHit::BYTES, wgpu::BufferUsages::COPY_SRC),
            surfaces: stream.surfaces.map(|_| {
                buffer(
                    "surfaces",
                    rows * SceneSurface::BYTES,
                    wgpu::BufferUsages::COPY_SRC,
                )
            }),
        }
    }

    /// Writes the header of a frame holding `count` rays, for a consumer
    /// that knows its count on the CPU (one that compacts on the GPU writes
    /// the header there).
    pub fn set_count(&self, queue: &wgpu::Queue, count: u32) {
        helio_core::upload::write_buffer(
            queue,
            &self.header,
            0,
            bytemuck::bytes_of(&StreamHeader::of(count)),
        );
    }

    /// Clears the hits: no ray has met anything, every occlusion ray gets
    /// through. The generator records it before the tracers run.
    pub fn clear(&self, cmds: &mut helio_core::CommandRecorder<'_>) {
        cmds.clear_buffer(&self.hits, 0, None);
    }

    /// Publishes the buffers under `stream`'s resource names.
    pub fn publish(
        &self,
        stream: &Stream,
        frame: &mut helio_core::ResourceRegistry<'_>,
        writer: &'static str,
    ) {
        frame.write_buffer_binding(stream.header, &self.header, writer);
        frame.write_buffer_binding(stream.rays, &self.rays, writer);
        frame.write_buffer_binding(stream.hits, &self.hits, writer);
        if let (Some(name), Some(surfaces)) = (stream.surfaces, &self.surfaces) {
            frame.write_buffer_binding(name, surfaces, writer);
        }
    }
}

/// Whether a stream's rays were answered in full this frame: by every tracer,
/// for all the geometry it knows of. A tracer that knows geometry it could
/// not trace this frame (its kernels still compiling, what it walks not yet
/// drawn) says so ([`Self::short`]), and its rows then miss that geometry:
/// a consumer that keeps what its rays find over frames (a radiance cache, a
/// probe's history) holds instead of taking them ([`Self::complete`]), so a
/// tracer that is late is not taken for geometry that is not there. Geometry
/// that is not there (nothing of the kind in the scene, chunks or meshes
/// still streaming in, whose stand-ins a tracer does trace) is no shortfall.
///
/// One a stream a graph ([`answers`]), set on the CPU while the frame
/// records, in graph order: the consumer's generator pass starts the frame
/// ([`Self::begin`]) in its `execute`, before the tracers' passes; a
/// tracer's pass marks it in its `execute`; the consumer's pass that reads
/// the hits asks in its `execute`, after them.
#[derive(Debug, Default)]
pub struct Answers {
    short: AtomicBool,
}

impl Answers {
    /// A new frame's rays: none short yet (the consumer's generator).
    pub fn begin(&self) {
        self.short.store(false, Ordering::Relaxed);
    }

    /// A tracer could not trace geometry it knows of this frame.
    pub fn short(&self) {
        self.short.store(true, Ordering::Relaxed);
    }

    /// Whether every tracer answered in full since [`Self::begin`].
    pub fn complete(&self) -> bool {
        !self.short.load(Ordering::Relaxed)
    }
}

#[derive(Default)]
struct StreamAnswers(Vec<(&'static str, Arc<Answers>)>);

/// The [`Answers`] of `stream` in the graph being built: the same for its
/// consumer's passes and every tracer's.
pub fn answers(env: &mut BuildEnv<'_>, stream: &Stream) -> Arc<Answers> {
    let all = &mut env.shared(StreamAnswers::default).0;
    if let Some((_, answers)) = all.iter().find(|(name, _)| *name == stream.name) {
        return answers.clone();
    }
    let answers = Arc::new(Answers::default());
    all.push((stream.name, answers.clone()));
    answers
}

/// A tracer's number in the hits it writes (`scene_nearest`'s `tracer`): 1 and
/// up, in registration order; 0 is no hit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TracerId(pub u32);

/// Adds a tracer's passes for one stream to the graph being built.
pub type TracerBuild =
    Arc<dyn Fn(&mut NodeBuild<'_, '_>, &Stream, TracerId) -> Result<(), String> + Send + Sync>;

/// A kind's answer to every ray stream (module docs).
#[derive(Clone)]
pub struct Tracer {
    /// The kind, for diagnostics (`meshes`, `voxel`).
    pub name: &'static str,
    /// Whether the tracer can run in this graph (hardware ray queries on the
    /// device, say). `false`: it adds no pass, and its geometry is not seen.
    pub supported: Arc<dyn Fn(&BuildEnv<'_>) -> bool + Send + Sync>,
    /// Adds the passes that trace `stream`: they read its header and rays
    /// and write, into each row of its hits, the nearer of what the row
    /// holds and this kind's hit (`scene_hit_reach`, `scene_nearest`), from
    /// the ray's origin itself (a consumer whose rays leave a surface starts
    /// them off it: a tracer adds no offset of its own); for
    /// [`RayClass::Occlusion`], what the row lets through times what this
    /// kind's geometry does (`scene_occlusion`). One indirect dispatch over
    /// the header's groups ([`GROUP`] invocations each). On a frame where
    /// the kind has nothing to trace the pass does nothing; on one where it
    /// knows geometry it cannot trace yet, it marks the stream's
    /// [`Answers::short`] ([`answers`]).
    pub trace: TracerBuild,
    /// Adds the passes that write the [`SceneSurface`] of every hit whose
    /// tracer is this one, once every tracer has traced
    /// ([`RayClass::Nearest`] streams only).
    pub surface: TracerBuild,
}

#[derive(Default)]
struct Tracers(Vec<Tracer>);

/// What a feature registers its kind's tracer through.
pub trait RegisterRayTracer {
    /// Registers `tracer` for every graph built with this feature.
    fn ray_tracer(&mut self, tracer: Tracer);
}

impl RegisterRayTracer for FeatureRegistrar<'_> {
    fn ray_tracer(&mut self, tracer: Tracer) {
        self.prepare(Arc::new(move |env| offer(env, tracer.clone())));
    }
}

/// Offers `tracer` to the graph being built ([`RegisterRayTracer`] does it
/// for a feature; a test or a host without features calls it directly).
///
/// # Panics
///
/// Past [`MAX_TRACERS`] tracers.
pub fn offer(env: &mut BuildEnv<'_>, tracer: Tracer) {
    let tracers = &mut env.shared(Tracers::default).0;
    assert!(
        tracers.len() < MAX_TRACERS,
        "a frame holds at most {MAX_TRACERS} ray tracers"
    );
    tracers.push(tracer);
}

/// The tracers that can run in this graph, each with its id.
fn supported(env: &mut BuildEnv<'_>) -> Vec<(TracerId, Tracer)> {
    let tracers = env.shared(Tracers::default).0.clone();
    tracers
        .into_iter()
        .enumerate()
        .filter(|(_, tracer)| (tracer.supported)(env))
        .map(|(index, tracer)| (TracerId(index as u32 + 1), tracer))
        .collect()
}

/// How many tracers would answer a stream in this graph: a consumer whose
/// rays nothing can trace builds nothing.
pub fn traced(env: &mut BuildEnv<'_>) -> usize {
    supported(env).len()
}

/// The names of the tracers that would answer a stream in this graph.
pub fn tracer_names(env: &mut BuildEnv<'_>) -> Vec<&'static str> {
    supported(env)
        .into_iter()
        .map(|(_, tracer)| tracer.name)
        .collect()
}

/// Adds every tracer's passes for `stream`, after the consumer's generator
/// pass and before whatever reads the hits: each tracer's trace passes in
/// registration order, then (for [`RayClass::Nearest`]) each one's surface
/// passes. Returns how many tracers answer.
pub fn trace(build: &mut NodeBuild<'_, '_>, stream: &Stream) -> Result<usize, String> {
    let tracers = supported(build.env);
    for (id, tracer) in &tracers {
        (tracer.trace)(build, stream, *id)
            .map_err(|error| format!("{}'s tracer of `{}`: {error}", tracer.name, stream.name))?;
    }
    if stream.class == RayClass::Nearest {
        for (id, tracer) in &tracers {
            (tracer.surface)(build, stream, *id).map_err(|error| {
                format!("{}'s surfaces of `{}`: {error}", tracer.name, stream.name)
            })?;
        }
    }
    Ok(tracers.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_have_their_contract_sizes() {
        assert_eq!(StreamHeader::BYTES, 16);
        assert_eq!(SceneRay::BYTES, 32);
        assert_eq!(SceneHit::BYTES, 16);
        assert_eq!(SceneSurface::BYTES, 32);
    }

    #[test]
    fn a_cone_keeps_its_spread_mask_and_width() {
        let ray = SceneRay::new(
            [0.0; 3],
            1.0,
            [0.0, 1.0, 0.0],
            SceneRay::cone(0.0, 0.443, MASK_ALL),
        );
        // The spread within its 6 mantissa bits (0.8%).
        assert!(
            (ray.spread() / 0.443 - 1.0).abs() < 0.008,
            "{}",
            ray.spread()
        );
        assert_eq!(ray.mask(), MASK_ALL);
        assert_eq!(ray.width(), 0.0);
        // Widths keep within their step (2^(1/16) either way), from a
        // micrometre to kilometres; the other fields are untouched.
        for width in [1e-6f32, 0.002, 0.1, 1.0, 37.5, 3000.0] {
            let ray = SceneRay::new(
                [0.0; 3],
                1.0,
                [0.0, 1.0, 0.0],
                SceneRay::cone(width, 0.01, MASK_STATIC),
            );
            let ratio = ray.width() / width;
            assert!(
                (0.957..1.045).contains(&ratio),
                "{width} -> {}",
                ray.width()
            );
            assert!((ray.spread() / 0.01 - 1.0).abs() < 0.008);
            assert_eq!(ray.mask(), MASK_STATIC);
            assert_eq!(ray.space(), SPACE_ANY);
        }
        // Halves round to nearest and survive their range's ends.
        assert_eq!(half_value(half_bits(1.0)), 1.0);
        assert_eq!(half_value(half_bits(0.5)), 0.5);
        assert_eq!(half_value(half_bits(65504.0)), 65504.0);
        assert_eq!(half_value(half_bits(1e9)), 65504.0);
        assert_eq!(half_bits(0.0), 0);
        let tiny = half_value(half_bits(1e-6));
        assert!((tiny - 1e-6).abs() < 6e-8, "{tiny}");
        // A spread past the largest the cone holds: the largest.
        let wide = SceneRay::new([0.0; 3], 1.0, [0.0, 1.0, 0.0], SceneRay::cone(0.0, 1e9, 0));
        assert_eq!(wide.spread(), 65024.0);
        // A space takes its bits and leaves the cone's.
        let placed = wide.in_space(space_of_slot(255));
        assert_eq!(placed.space(), 256);
        assert_eq!(placed.spread(), wide.spread());
        assert_eq!(placed.width(), wide.width());
        assert_eq!(placed.mask(), 0);
        assert_eq!(placed.in_space(SPACE_ANY), wide);
    }

    #[test]
    fn a_header_covers_its_rays() {
        assert_eq!(StreamHeader::of(0).groups, [0, 1, 1]);
        assert_eq!(StreamHeader::of(1).groups, [1, 1, 1]);
        assert_eq!(StreamHeader::of(64).groups, [1, 1, 1]);
        assert_eq!(StreamHeader::of(65).groups, [2, 1, 1]);
    }

    #[test]
    fn unit_vectors_survive_packing() {
        // The angle between the two, by their cross product in f64 (an f32
        // dot product cannot tell angles this small from none).
        let mut worst = 0.0f64;
        for i in 0..2000 {
            // A spherical Fibonacci set: every octant, both hemispheres.
            let z = 1.0 - 2.0 * (i as f32 + 0.5) / 2000.0;
            let r = (1.0 - z * z).sqrt();
            let phi = i as f32 * 2.399_963_1;
            let n = [r * phi.cos(), r * phi.sin(), z];
            let [a, b, c] = n.map(f64::from);
            let [x, y, z] = unpack_unit(pack_unit(n)).map(f64::from);
            let cross = [b * z - c * y, c * x - a * z, a * y - b * x];
            let sine = cross.iter().map(|v| v * v).sum::<f64>().sqrt();
            worst = worst.max(sine.asin());
        }
        // Under a hundredth of a degree (16 bits an axis).
        assert!(worst < 1e-4, "{worst} rad");
        for axis in [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]] {
            let back = unpack_unit(pack_unit(axis));
            for (a, b) in axis.iter().zip(back) {
                assert!((a - b).abs() < 1e-4, "{axis:?} -> {back:?}");
            }
        }
    }

    #[test]
    fn radiance_survives_packing_within_its_precision() {
        for c in [
            [0.0, 0.0, 0.0],
            [1.0, 0.5, 0.25],
            [0.001, 0.002, 0.003],
            [1000.0, 10.0, 0.0],
            [60000.0, 60000.0, 60000.0],
            // The largest channel rounds up to the next exponent.
            [0.999_9, 0.5, 0.1],
        ] {
            let back = unpack_radiance(pack_radiance(c));
            let most = c[0].max(c[1]).max(c[2]);
            for (a, b) in c.iter().zip(back) {
                // Half a step of the shared exponent's 9 bits.
                assert!((a - b).abs() <= most / 256.0 + 1e-6, "{c:?} -> {back:?}");
            }
        }
        // Past the range: clamped, not wrapped.
        let back = unpack_radiance(pack_radiance([1e9, 0.0, 0.0]));
        assert_eq!(back, [65408.0, 0.0, 0.0]);
    }

    #[test]
    fn a_cleared_occlusion_row_lets_everything_through() {
        assert_eq!(SceneHit::default().through(), [1.0; 3]);
        let blocked = SceneHit {
            t: 1.0,
            a: 0,
            b: 1023 | (1023 << 10) | (1023 << 20),
            c: 0,
        };
        assert_eq!(blocked.through(), [0.0; 3]);
    }

    #[test]
    fn a_stream_has_one_answers_a_graph_and_a_short_tracer_marks_it() {
        crate::test_env::with_env(|env| {
            let probes = Stream::new("test_probe", RayClass::Nearest, 128);
            let shadows = Stream::new("test_shadow", RayClass::Occlusion, 128);
            let consumer = answers(env, &probes);
            let tracer = answers(env, &probes);
            assert!(Arc::ptr_eq(&consumer, &tracer));
            assert!(!Arc::ptr_eq(&consumer, &answers(env, &shadows)));
            consumer.begin();
            assert!(consumer.complete());
            tracer.short();
            assert!(!consumer.complete());
            assert!(answers(env, &shadows).complete());
            // The next frame starts complete.
            consumer.begin();
            assert!(consumer.complete());
        });
    }

    #[test]
    fn a_stream_names_its_resources() {
        let nearest = Stream::new("test_probe", RayClass::Nearest, 128);
        assert_eq!(nearest.rays, "test_probe_rays");
        assert_eq!(nearest.hits, "test_probe_hits");
        assert_eq!(nearest.header, "test_probe_ray_header");
        assert_eq!(nearest.surfaces, Some("test_probe_surfaces"));
        let occlusion = Stream::new("test_shadow", RayClass::Occlusion, 128);
        assert_eq!(occlusion.surfaces, None);
        // The same stream declared twice names the same resources.
        assert_eq!(Stream::new("test_probe", RayClass::Nearest, 128), nearest);
        // No runs unless the consumer says so; a run is at least one ray.
        assert_eq!(nearest.run, 1);
        assert_eq!(nearest.with_run(64).run, 64);
        assert_eq!(nearest.with_run(0).run, 1);
    }

    /// A stream asks for the material in full unless its consumer asks for
    /// less.
    #[test]
    fn a_stream_asks_for_its_surface_detail() {
        let nearest = Stream::new("test_probe", RayClass::Nearest, 128);
        assert_eq!(nearest.detail, SurfaceDetail::Full);
        let diffuse = nearest.with_detail(SurfaceDetail::Diffuse);
        assert_eq!(diffuse.detail, SurfaceDetail::Diffuse);
        assert!(SurfaceDetail::Diffuse < SurfaceDetail::Full);
        // Another detail is another stream's permutation, same resources.
        assert_ne!(diffuse, nearest);
        assert_eq!(diffuse.rays, nearest.rays);
    }

    /// A ray's space is a camera's encoding: any space, or a slot plus one;
    /// overlapping places in two spaces are told apart by it alone.
    #[test]
    fn a_ray_sees_the_spaces_a_camera_would() {
        let ray = SceneRay::new(
            [1.0, 2.0, 3.0],
            10.0,
            [0.0, 1.0, 0.0],
            SceneRay::cone(0.5, 0.25, MASK_ALL),
        );
        assert_eq!(ray.space(), SPACE_ANY);
        assert!(ray.sees_space(0) && ray.sees_space(7));
        let world = ray.in_space(space_of_slot(0));
        assert!(world.sees_space(0) && !world.sees_space(7));
        let pocket = ray.in_space(space_of_slot(7));
        assert!(pocket.sees_space(7) && !pocket.sees_space(0));
        assert_eq!(pocket.origin, world.origin);
        // The highest slot (255) still encodes.
        assert!(ray.in_space(space_of_slot(255)).sees_space(255));
    }
}
