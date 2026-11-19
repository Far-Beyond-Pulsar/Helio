//! Direct light at a point: how anything in the frame asks every kind of
//! light for a sample, without knowing a kind (the counterpart of
//! [`super::stream`] for lights).
//!
//! A **consumer** (GI's probe hits, a radiance cache's update, the
//! reference path tracer's next-event estimation, a reflection's hit)
//! declares a [`LightBatch`]: a batch of points ([`LightQuery`] rows, 32
//! bytes: a frame-space position, the footprint width it stands for, its
//! normal, two uniform numbers for where on a light and one for the choices,
//! its coordinate space) and a [`StreamHeader`](super::stream::StreamHeader)
//! for their count. A **sampler** is a kind of light's answer: the lights
//! plugin samples its point, spot and directional lights; an emissive
//! surface, the sky or an area light kind would register its own. Each
//! sampler's pass offers one sample of its own lights at each point, and the
//! row keeps one ([`LightSample`], 32 bytes): the samplers' offers merged by
//! weighted reservoir sampling (`light_sample_offer`: weight the offer's
//! luminance), so the row is one unbiased sample of every registered kind's
//! light, whichever registered.
//!
//! The consumer then turns each sample into an occlusion-stream ray toward
//! the light (`light_sample_dir`, `light_sample_distance`, from the point in
//! its space) and weights what gets through by its own BSDF:
//! `bsdf * cos * light_sample_radiance(s) * visibility`. For multiple
//! importance sampling against its BSDF sample, `light_sample_pdf(s)` is the
//! direction's solid-angle density, the choices included; 0 marks a delta
//! light (point, spot, directional), which a BSDF sample cannot hit (its
//! light sample's weight is 1).
//!
//! Why a contract and not the lights' rows read by every consumer: GI read
//! the light list itself (`local_lights`, a stand-in), and a reflection, a
//! cache and the reference would each have grown a copy; how a light is
//! chosen (uniform or by power today, ReSTIR or a light tree in r5) is the
//! lights' business, behind this one entry point. The batch is the
//! wavefront form of next-event estimation (rows a pass reads, as the ray
//! streams are): no light code in any consumer's shader.
//!
//! Cost scales with the points asked for, times the samplers that register
//! (one in practice), never with the lights' count beyond what a sampler's
//! own choice costs.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use helio_core::shader::ShaderSnippet;
use helio_feature_api::{BuildEnv, FeatureRegistrar, NodeBuild};

use super::stream::StreamHeader;

/// Marker opting a shader into [`SNIPPET`] (with `frame_ray_stream`, whose
/// packings it uses).
pub const MARKER: &str = "//!use frame_light_samples";
/// The rows' WGSL (`shaders/light_samples.wgsl`): `LightQuery`,
/// `LightSample`, the queries' helpers (`light_query_point`,
/// `light_query_uniform`, `light_query_position_uniforms`,
/// `light_query_sees_space`, `light_query_faces`), a sampler's
/// (`light_sample_of`, `light_sample_offer`) and a consumer's
/// (`light_sample_found`, `light_sample_dir`, `light_sample_distance`,
/// `light_sample_radiance`, `light_sample_pdf`).
pub const SNIPPET: ShaderSnippet =
    helio_core::wgsl_snippet!(MARKER, "../shaders/light_samples.wgsl");

/// Samplers a frame can hold: a sample names its sampler in four bits.
pub const MAX_SAMPLERS: usize = 15;

/// `LightQuery::space` bit: the point is lit from every side.
pub const QUERY_ANY_SIDE: u32 = 0x8000_0000;

/// `LightSample::light` of a row without a sample.
pub const SAMPLE_NONE: u32 = u32::MAX;

/// How far a directional light is (`LightSample::distance`).
pub const SAMPLE_FAR: f32 = 1e30;

/// `LightQuery` (see the WGSL), 32 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct LightQuery {
    pub position: [f32; 3],
    /// The footprint the point stands for (metres across).
    pub width: f32,
    /// The side it is lit from (`pack_unit`).
    pub normal: u32,
    /// Two uniform numbers for where on a light, 16 bits each (low first).
    pub random: u32,
    /// A uniform number for the choices among lights and samplers.
    pub pick: u32,
    /// Bits 0..16 its coordinate space (`SceneRay::space`'s encoding), bit
    /// 31 [`QUERY_ANY_SIDE`].
    pub space: u32,
}

impl LightQuery {
    pub const BYTES: u64 = std::mem::size_of::<Self>() as u64;
}

/// `LightSample` (see the WGSL), 32 bytes. A cleared row (zeros) holds no
/// sample.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct LightSample {
    /// Incident radiance over its probability within its sampler.
    pub radiance: [f32; 3],
    pub distance: f32,
    /// Toward the light (`pack_unit`).
    pub dir: u32,
    /// Solid-angle density within its sampler (0: a delta light).
    pub pdf: f32,
    /// The sum of the samplers' offers' weights (0: no sample).
    pub weight: f32,
    /// The sampler's number in bits 28..32, its own id of the light below.
    pub light: u32,
}

/// `light_luminance`.
fn luminance(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

impl LightSample {
    pub const BYTES: u64 = std::mem::size_of::<Self>() as u64;

    /// `light_sample_offer`, with the uniform number `u` the WGSL draws
    /// from the query for this sampler.
    pub fn offer(self, offer: LightSample, u: f32) -> LightSample {
        let w = luminance(offer.radiance);
        if w.is_nan() || w <= 0.0 {
            return self;
        }
        let sum = self.weight + w;
        let mut out = if u * sum < w { offer } else { self };
        out.weight = sum;
        out
    }

    /// `light_sample_found`.
    pub fn found(&self) -> bool {
        self.weight > 0.0
    }

    /// `light_sample_radiance`: over the whole choice's probability.
    pub fn contribution(&self) -> [f32; 3] {
        let w = luminance(self.radiance);
        if self.weight > 0.0 && w > 0.0 {
            self.radiance.map(|c| c * self.weight / w)
        } else {
            [0.0; 3]
        }
    }

    /// `light_sample_pdf`.
    pub fn pdf(&self) -> f32 {
        let w = luminance(self.radiance);
        if self.weight > 0.0 && w > 0.0 {
            self.pdf * w / self.weight
        } else {
            0.0
        }
    }

    /// The sampler that gave the sample.
    pub fn sampler(&self) -> SamplerId {
        SamplerId(self.light >> 28)
    }
}

/// A batch of points a consumer asks for direct light at each frame (module
/// docs). Its buffers are graph resources under these names; the consumer
/// owns them ([`LightBatchBuffers`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LightBatch {
    pub name: &'static str,
    /// The most points it holds.
    pub capacity: u32,
    /// `StreamHeader`, one: the dispatch over the points (groups of
    /// [`super::stream::GROUP`]) and their count.
    pub header: &'static str,
    /// [`LightQuery`] rows.
    pub queries: &'static str,
    /// [`LightSample`] rows, one a query.
    pub samples: &'static str,
}

impl LightBatch {
    /// The batch `name` (unique in a graph: its resources are
    /// `<name>_light_header`, `<name>_light_queries`,
    /// `<name>_light_samples`).
    pub fn new(name: &'static str, capacity: u32) -> Self {
        let resource = |suffix: &str| crate::static_name(&format!("{name}_{suffix}"));
        Self {
            name,
            capacity,
            header: resource("light_header"),
            queries: resource("light_queries"),
            samples: resource("light_samples"),
        }
    }
}

/// A batch's buffers, made and held by its consumer, which writes the header
/// and the queries, clears the samples each frame ([`Self::clear`]) and
/// publishes them ([`Self::publish`]).
pub struct LightBatchBuffers {
    pub header: wgpu::Buffer,
    pub queries: wgpu::Buffer,
    pub samples: wgpu::Buffer,
}

impl LightBatchBuffers {
    pub fn new(device: &wgpu::Device, batch: &LightBatch) -> Self {
        let rows = u64::from(batch.capacity.max(1));
        let buffer = |what: &str, size: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{} {what}", batch.name)),
                size,
                usage: usage | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        Self {
            header: buffer(
                "light header",
                StreamHeader::BYTES,
                wgpu::BufferUsages::INDIRECT,
            ),
            queries: buffer(
                "light queries",
                rows * LightQuery::BYTES,
                wgpu::BufferUsages::empty(),
            ),
            samples: buffer(
                "light samples",
                rows * LightSample::BYTES,
                wgpu::BufferUsages::COPY_SRC,
            ),
        }
    }

    /// Clears the samples: no point has one. Recorded before the samplers
    /// run.
    pub fn clear(&self, cmds: &mut helio_core::CommandRecorder<'_>) {
        cmds.clear_buffer(&self.samples, 0, None);
    }

    /// Publishes the buffers under `batch`'s resource names.
    pub fn publish(
        &self,
        batch: &LightBatch,
        frame: &mut helio_core::ResourceRegistry<'_>,
        writer: &'static str,
    ) {
        frame.write_buffer_binding(batch.header, &self.header, writer);
        frame.write_buffer_binding(batch.queries, &self.queries, writer);
        frame.write_buffer_binding(batch.samples, &self.samples, writer);
    }
}

/// A sampler's number in the samples it writes (bits 28..32 of
/// `LightSample::light`): 1 and up, in registration order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SamplerId(pub u32);

/// Adds a sampler's passes for one batch to the graph being built.
pub type SamplerBuild =
    Arc<dyn Fn(&mut NodeBuild<'_, '_>, &LightBatch, SamplerId) -> Result<(), String> + Send + Sync>;

/// A kind of light's answer to every light batch (module docs).
#[derive(Clone)]
pub struct LightSampler {
    /// The kind, for diagnostics (`lights`).
    pub name: &'static str,
    /// Whether it samples in this graph. `false`: it adds no pass, and its
    /// lights are not sampled.
    pub supported: Arc<dyn Fn(&BuildEnv<'_>) -> bool + Send + Sync>,
    /// Adds the passes that sample `batch`: one indirect dispatch over its
    /// header's groups, which for each query chooses one of this kind's
    /// lights that the query's space sees and its side faces (uniformly or
    /// by power; r5 replaces the choice), draws a point on it with the
    /// query's position numbers, and merges the sample into the row with
    /// `light_sample_offer(row, light_sample_of(..), query, id)`. A point
    /// none of its lights reach gets no offer. On a frame with no lights the
    /// pass records nothing.
    pub sample: SamplerBuild,
}

#[derive(Default)]
struct Samplers(Vec<LightSampler>);

/// What a feature registers its kind's light sampler through.
pub trait RegisterLightSampler {
    /// Registers `sampler` for every graph built with this feature.
    fn light_sampler(&mut self, sampler: LightSampler);
}

impl RegisterLightSampler for FeatureRegistrar<'_> {
    fn light_sampler(&mut self, sampler: LightSampler) {
        self.prepare(Arc::new(move |env| offer(env, sampler.clone())));
    }
}

/// Offers `sampler` to the graph being built ([`RegisterLightSampler`] does
/// it for a feature; a test or a host without features calls it directly).
///
/// # Panics
///
/// Past [`MAX_SAMPLERS`] samplers.
pub fn offer(env: &mut BuildEnv<'_>, sampler: LightSampler) {
    let samplers = &mut env.shared(Samplers::default).0;
    assert!(
        samplers.len() < MAX_SAMPLERS,
        "a frame holds at most {MAX_SAMPLERS} light samplers"
    );
    samplers.push(sampler);
}

/// The samplers that can run in this graph, each with its id.
fn supported(env: &mut BuildEnv<'_>) -> Vec<(SamplerId, LightSampler)> {
    let samplers = env.shared(Samplers::default).0.clone();
    samplers
        .into_iter()
        .enumerate()
        .filter(|(_, sampler)| (sampler.supported)(env))
        .map(|(index, sampler)| (SamplerId(index as u32 + 1), sampler))
        .collect()
}

/// How many samplers would answer a batch in this graph: a consumer whose
/// points no light kind samples builds nothing for its direct light.
pub fn sampled(env: &mut BuildEnv<'_>) -> usize {
    supported(env).len()
}

/// The names of the samplers that would answer a batch in this graph.
pub fn sampler_names(env: &mut BuildEnv<'_>) -> Vec<&'static str> {
    supported(env)
        .into_iter()
        .map(|(_, sampler)| sampler.name)
        .collect()
}

/// Adds every sampler's passes for `batch`, after the consumer's pass that
/// wrote its queries (and cleared its samples) and before whatever reads the
/// samples, in registration order. Returns how many samplers answer.
pub fn sample(build: &mut NodeBuild<'_, '_>, batch: &LightBatch) -> Result<usize, String> {
    let samplers = supported(build.env);
    for (id, sampler) in &samplers {
        (sampler.sample)(build, batch, *id)
            .map_err(|error| format!("{}'s samples of `{}`: {error}", sampler.name, batch.name))?;
    }
    Ok(samplers.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_have_their_contract_sizes() {
        assert_eq!(LightQuery::BYTES, 32);
        assert_eq!(LightSample::BYTES, 32);
    }

    #[test]
    fn a_batch_names_its_resources() {
        let batch = LightBatch::new("test_points", 64);
        assert_eq!(batch.header, "test_points_light_header");
        assert_eq!(batch.queries, "test_points_light_queries");
        assert_eq!(batch.samples, "test_points_light_samples");
        assert_eq!(LightBatch::new("test_points", 64), batch);
    }

    fn sample(radiance: [f32; 3], light: u32) -> LightSample {
        LightSample {
            radiance,
            distance: 1.0,
            dir: 0,
            pdf: 0.0,
            weight: 0.0,
            light,
        }
    }

    /// One sampler: the row is its offer, at its own value.
    #[test]
    fn one_sampler_keeps_its_offer() {
        let offer = sample([2.0, 1.0, 0.5], (1 << 28) | 7);
        let row = LightSample::default().offer(offer, 0.99);
        assert!(row.found());
        assert_eq!(row.sampler(), SamplerId(1));
        let c = row.contribution();
        for (a, b) in c.iter().zip(offer.radiance) {
            assert!((a - b).abs() < 1e-5, "{c:?}");
        }
        // No offer (black): no sample.
        let none = LightSample::default().offer(sample([0.0; 3], 1 << 28), 0.0);
        assert!(!none.found());
        assert_eq!(none.contribution(), [0.0; 3]);
    }

    /// Several samplers: the expected contribution is the sum of their
    /// offers (an unbiased one-sample choice among them), over the uniform
    /// numbers the merge draws.
    #[test]
    fn the_merge_of_samplers_is_unbiased() {
        let offers = [
            sample([4.0, 3.0, 1.0], 1 << 28),
            sample([0.1, 0.2, 0.9], 2 << 28),
            sample([0.0, 0.0, 0.0], 3 << 28),
            sample([1.0, 1.0, 1.0], 4 << 28),
        ];
        let n = 200_000;
        let mut mean = [0.0f64; 3];
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut uniform = || {
            // xorshift64*: independent uniforms for each merge.
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            (state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40) as f32 / (1u64 << 24) as f32
        };
        for _ in 0..n {
            let mut row = LightSample::default();
            for offer in &offers {
                row = row.offer(*offer, uniform());
            }
            for (m, c) in mean.iter_mut().zip(row.contribution()) {
                *m += f64::from(c) / n as f64;
            }
        }
        let expected = [5.1, 4.2, 2.9];
        for (m, e) in mean.iter().zip(expected) {
            assert!((m - e).abs() < 0.02 * e, "{mean:?} against {expected:?}");
        }
    }

    /// The pdf a consumer weighs its BSDF sample against includes the
    /// choice among samplers.
    #[test]
    fn the_pdf_includes_the_choice() {
        let mut a = sample([1.0, 1.0, 1.0], 1 << 28);
        a.pdf = 2.0;
        let b = sample([3.0, 3.0, 3.0], 2 << 28);
        // `a` chosen (u low), against a total weight of 4: chance 1/4.
        let row = LightSample::default().offer(a, 0.0).offer(b, 0.9);
        assert_eq!(row.sampler(), SamplerId(1));
        assert!((row.pdf() - 0.5).abs() < 1e-6);
        let c = row.contribution();
        assert!((c[0] - 4.0).abs() < 1e-5, "{c:?}");
    }

    #[test]
    fn samplers_register_in_order_and_unsupported_ones_drop_out() {
        crate::test_env::with_env(|env| {
            assert_eq!(sampled(env), 0);
            let build: SamplerBuild = Arc::new(|_, _, _| Ok(()));
            offer(
                env,
                LightSampler {
                    name: "first",
                    supported: Arc::new(|_| true),
                    sample: build.clone(),
                },
            );
            offer(
                env,
                LightSampler {
                    name: "absent",
                    supported: Arc::new(|_| false),
                    sample: build.clone(),
                },
            );
            offer(
                env,
                LightSampler {
                    name: "third",
                    supported: Arc::new(|_| true),
                    sample: build,
                },
            );
            assert_eq!(sampled(env), 2);
            assert_eq!(sampler_names(env), vec!["first", "third"]);
            // Ids keep registration order (the unsupported one's is unused).
            let ids: Vec<_> = supported(env).into_iter().map(|(id, _)| id).collect();
            assert_eq!(ids, vec![SamplerId(1), SamplerId(3)]);
        });
    }
}
