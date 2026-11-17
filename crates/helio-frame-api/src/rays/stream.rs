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
//! for a consumer that builds nothing otherwise.
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
//! What it costs: a ray is 32 bytes, a hit 16, a surface 20, written and
//! read once each, and a stream's hit arrives in the same frame only after
//! the tracers' passes ran. A consumer that needs one visibility test per
//! pixel against triangles alone can still query the TLAS inline
//! ([`super::SCENE_RAYS`]); it then sees no other kind.
//!
//! Cost scales with the rays a consumer asks for, never with the scene's
//! size: a tracer's pass is one indirect dispatch over the stream's count.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use helio_core::shader::ShaderSnippet;
use helio_feature_api::{BuildEnv, FeatureRegistrar, NodeBuild};

/// Marker opting a shader into [`SNIPPET`].
pub const MARKER: &str = "//!use frame_ray_stream";
/// The rows' WGSL (`shaders/ray_stream.wgsl`): `RayStreamHeader`,
/// `SceneRay`, `SceneHit`, `SceneSurface`, and the functions that pack and
/// read them.
pub const SNIPPET: ShaderSnippet = helio_core::wgsl_snippet!(MARKER, "../shaders/ray_stream.wgsl");

/// Invocations of a tracer's workgroup: [`StreamHeader::groups`] is the
/// dispatch for workgroups this size (`RAY_STREAM_GROUP` in WGSL).
pub const GROUP: u32 = 64;

/// TLAS instance and ray mask bit: casters that never move.
pub const MASK_STATIC: u8 = 1;
/// TLAS instance and ray mask bit: casters that may move.
pub const MASK_MOVABLE: u8 = 2;
/// A ray that sees every caster.
pub const MASK_ALL: u8 = 0xff;

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

/// `SceneRay` (frame space; see the WGSL).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct SceneRay {
    pub origin: [f32; 3],
    /// The most distance the ray travels; 0 or less: no ray in this row.
    pub t_max: f32,
    pub dir: [f32; 3],
    /// Spread (half float, bits 0..16) and caster mask (bits 16..24).
    pub cone: u32,
}

impl SceneRay {
    pub const BYTES: u64 = std::mem::size_of::<Self>() as u64;
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

/// `SceneSurface` (see the WGSL).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct SceneSurface {
    pub normal: u32,
    pub face: u32,
    pub base_color: u32,
    pub params: u32,
    pub emissive: u32,
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

    /// The share of light let through from the far side.
    pub fn transmission(&self) -> f32 {
        ((self.params >> 8) & 0xff) as f32 / 255.0
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

/// A batch of rays a consumer writes each frame and the kinds' tracers
/// answer (module docs). Its buffers are graph resources under these names;
/// the consumer owns them ([`StreamBuffers`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Stream {
    pub name: &'static str,
    pub class: RayClass,
    /// The most rays it holds.
    pub capacity: u32,
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
            header: resource("ray_header"),
            rays: resource("rays"),
            hits: resource("hits"),
            surfaces: (class == RayClass::Nearest).then(|| resource("surfaces")),
        }
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

/// A tracer's number in the hits it writes (`scene_hit`'s `tracer`): 1 and
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
    /// holds and this kind's hit (`scene_hit_reach`); for
    /// [`RayClass::Occlusion`], what the row lets through times what this
    /// kind's geometry does (`scene_occlusion`). One indirect dispatch over
    /// the header's groups ([`GROUP`] invocations each). On a frame where
    /// the kind has nothing to trace the pass does nothing.
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
        assert_eq!(SceneSurface::BYTES, 20);
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
    }
}
