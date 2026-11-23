//! The frame's denoiser contract: one way for every noisy signal (the
//! pixels' indirect diffuse now; reflections in r4, visibility in r5) to be
//! filtered, by whichever denoiser the frame has (`docs/architecture/
//! rendering.md`, section 5).
//!
//! A **producer** describes its [`Signal`] (its kind, the block of pixels
//! one traced sample stands for, whether it writes a confidence), writes its
//! inputs each frame under the signal's resource names, and calls
//! [`filter`] right after the passes that write them. A **denoiser** (a
//! feature: `helio-pass-denoise`, a vendor's reconstruction later) registers
//! once ([`RegisterDenoiser::denoiser`]); [`filter`] adds its passes there,
//! as `rays::stream::trace` adds the tracers'. With none registered the
//! signal passes through: [`Filtered`] names the producer's own inputs, at
//! their traced resolution, so a producer needs no denoiser to build.
//!
//! **Irradiance** ([`SignalKind::Irradiance`], the only kind built): the
//! light arriving at a surface from its hemisphere, as first-order harmonics
//! of its luminance and two chroma terms. A sample of incident light gives
//! `c = L cos / pdf` (an unbiased irradiance estimate at the traced normal
//! `n0`, as RGB) from direction `w`; the producer writes, summed over its
//! samples of a traced pixel ([`IRRADIANCE_SNIPPET`]'s
//! `denoise_irradiance_add`):
//!
//! | input | format | x | y | z | w |
//! |---|---|---|---|---|---|
//! | 0 | `Rgba16Float` | luminance `Y(c)` | chroma `Co(c)` | chroma `Cg(c)` | hit distance (m; past `f16::MAX` or infinite: none) |
//! | 1 | `Rgba16Float` | `Y(c) w.x` | `Y(c) w.y` | `Y(c) w.z` | confidence in [0, 1] ([`Signal::confidence`]) |
//!
//! (YCoCg: `Y = (r + 2g + b) / 4`, `Co = (r - b) / 2`, `Cg = (-r + 2g - b)
//! / 4`.) The denoiser turns a sample into the linear irradiance
//! `E(n) = a + b . n` exact at `n0` (`b` = 3/4 of the luminance-weighted direction:
//! the first-order harmonics' cosine convolution over a uniformly lit
//! hemisphere; `a = Y - b . n0`), so samples traced at other normals
//! average and a pixel takes its light by its own normal. Its output, at
//! the internal resolution:
//!
//! | output | format | x | y | z | w |
//! |---|---|---|---|---|---|
//! | 0 | `Rgba16Float` | `a` | `Co / Y` | `Cg / Y` | the luminance's standard error |
//! | 1 | `Rgba16Float` | `b.x` | `b.y` | `b.z` | 1 / hit distance (harmonic mean) |
//!
//! read with `denoise_irradiance(out0, out1, normal)` (RGB irradiance). A
//! pass-through signal is read with `denoise_irradiance_input(in0, in1,
//! normal0, normal)`.
//!
//! **Which pixel is traced.** One in each block of `side x side` internal
//! pixels ([`TraceBlock`]), rotating over the block's pixels so every pixel is
//! traced once in `side^2` frames, neighbouring blocks out of phase (the
//! Bayer order of the block's place in its own `side x side` group):
//! [`traced_offset`] / `denoise_traced_offset(block, frame, side)`, frame
//! being the frame number (`PassContext::frame_num`, truncated).
//!
//! What a later kind adds: a [`SignalKind`] variant with its two tables and
//! its snippet; [`Signal`], [`filter`] and the registration stay.

use std::sync::Arc;

use helio_core::graph::ResourceFormat;
use helio_core::shader::ShaderSnippet;
use helio_feature_api::{BuildEnv, FeatureRegistrar, NodeBuild};

/// Marker opting a shader into [`IRRADIANCE_SNIPPET`].
pub const IRRADIANCE_MARKER: &str = "//!use frame_denoise_irradiance";
/// The irradiance signal's encodings (`shaders/denoise_irradiance.wgsl`):
/// `denoise_traced_offset`, `denoise_irradiance_add` (a producer's sample),
/// `denoise_irradiance` (a reader's output decode),
/// `denoise_irradiance_input` (the pass-through decode), the YCoCg pair.
pub const IRRADIANCE_SNIPPET: ShaderSnippet =
    helio_core::wgsl_snippet!(IRRADIANCE_MARKER, "shaders/denoise_irradiance.wgsl");

/// What a signal holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SignalKind {
    /// Irradiance as first-order harmonics of luminance and two chroma terms
    /// (module docs).
    Irradiance,
}

/// How many internal pixels one traced sample stands for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TraceBlock {
    /// Every pixel traced.
    #[default]
    One,
    /// One pixel of each 2x2.
    Four,
    /// One pixel of each 4x4.
    Sixteen,
}

impl TraceBlock {
    /// The block's side in pixels.
    pub const fn side(self) -> u32 {
        match self {
            Self::One => 1,
            Self::Four => 2,
            Self::Sixteen => 4,
        }
    }

    /// The block of `pixels` per traced sample: 1, 4 or 16.
    pub fn of_pixels(pixels: u32) -> Option<Self> {
        match pixels {
            1 => Some(Self::One),
            4 => Some(Self::Four),
            16 => Some(Self::Sixteen),
            _ => None,
        }
    }
}

/// The Bayer rank of `(x, y)` within a `side x side` group (side 1, 2 or 4).
fn bayer_rank(x: u32, y: u32, side: u32) -> u32 {
    let a = x ^ y;
    match side {
        2 => ((a & 1) << 1) | (y & 1),
        4 => ((a & 1) << 3) | ((y & 1) << 2) | (((a >> 1) & 1) << 1) | ((y >> 1) & 1),
        _ => 0,
    }
}

/// The place of Bayer rank `r` within a `side x side` group.
fn bayer_place(r: u32, side: u32) -> [u32; 2] {
    let (a, y) = match side {
        2 => ((r >> 1) & 1, r & 1),
        4 => (
            ((r >> 3) & 1) | (((r >> 1) & 1) << 1),
            ((r >> 2) & 1) | ((r & 1) << 1),
        ),
        _ => (0, 0),
    };
    [a ^ y, y]
}

/// The pixel of block `block` traced on frame `frame`, within the block
/// (`denoise_traced_offset`).
pub fn traced_offset(block: [u32; 2], frame: u32, side: u32) -> [u32; 2] {
    let n = side * side;
    let phase = bayer_rank(block[0] % side, block[1] % side, side);
    bayer_place(frame.wrapping_add(phase) % n, side)
}

/// The internal pixel block `block` traces on frame `frame`
/// (`denoise_traced_pixel`): the last row and column of blocks of an
/// internal image of `size` clamped into it.
pub fn traced_pixel(block: [u32; 2], frame: u32, side: u32, size: [u32; 2]) -> [u32; 2] {
    let o = traced_offset(block, frame, side);
    [
        (block[0] * side + o[0]).min(size[0] - 1),
        (block[1] * side + o[1]).min(size[1] - 1),
    ]
}

/// A noisy signal a producer writes every frame (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Signal {
    /// Unique in a graph: its resources are `<name>_signal0`, `_signal1`
    /// (the producer's) and `<name>_denoised0`, `_denoised1` (the
    /// denoiser's).
    pub name: &'static str,
    pub kind: SignalKind,
    pub block: TraceBlock,
    /// The producer writes a confidence (input 1's w) that the signal's
    /// history still describes the light: 1 where nothing it knows of
    /// changed, toward 0 where a revalidated sample or a cache entry it
    /// read changed. Without one the denoiser keeps history by geometry
    /// alone.
    pub confidence: bool,
    /// The producer's textures at the traced resolution ([`Self::traced_size`]).
    pub inputs: [&'static str; 2],
    /// The denoiser's textures at the internal resolution.
    pub outputs: [&'static str; 2],
}

impl Signal {
    /// An irradiance signal `name` traced at one pixel of each `block`.
    pub fn irradiance(name: &'static str, block: TraceBlock) -> Self {
        let resource = |suffix: &str| crate::static_name(&format!("{name}_{suffix}"));
        Self {
            name,
            kind: SignalKind::Irradiance,
            block,
            confidence: false,
            inputs: [resource("signal0"), resource("signal1")],
            outputs: [resource("denoised0"), resource("denoised1")],
        }
    }

    /// The signal with a producer's confidence ([`Self::confidence`]).
    pub fn with_confidence(mut self) -> Self {
        self.confidence = true;
        self
    }

    /// The traced resolution of an internal `size`: a texel a block, the
    /// last row and column of blocks possibly partial.
    pub fn traced_size(&self, size: [u32; 2]) -> [u32; 2] {
        let side = self.block.side();
        [size[0].div_ceil(side), size[1].div_ceil(side)]
    }

    /// The format of input `i` and output `i`.
    pub fn format(&self, _i: usize) -> ResourceFormat {
        match self.kind {
            SignalKind::Irradiance => ResourceFormat::Rgba16Float,
        }
    }
}

/// What a producer reads after [`filter`]: the textures and the block they
/// are at (1 when denoised, the signal's own when passed through).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Filtered {
    pub textures: [&'static str; 2],
    pub block: TraceBlock,
    /// The denoiser that filtered it, if any.
    pub denoiser: Option<&'static str>,
}

/// Adds a denoiser's passes for one signal to the graph being built.
pub type DenoiseBuild =
    Arc<dyn Fn(&mut NodeBuild<'_, '_>, &Signal) -> Result<(), String> + Send + Sync>;

/// Whether a denoiser filters a signal in a graph.
pub type SupportsFn = Arc<dyn Fn(&BuildEnv<'_>, &Signal) -> bool + Send + Sync>;

/// A denoiser (module docs).
#[derive(Clone)]
pub struct Denoiser {
    /// For diagnostics (`helio`).
    pub name: &'static str,
    /// Whether it filters `signal` in this graph (its kind, the device).
    pub supports: SupportsFn,
    /// Adds the passes that read the signal's inputs (and what the frame
    /// offers: its history, the G-buffer) and write its outputs, every
    /// frame the producer's passes ran.
    pub filter: DenoiseBuild,
}

#[derive(Default)]
struct Denoisers(Vec<Denoiser>);

/// What a feature registers its denoiser through.
pub trait RegisterDenoiser {
    /// Registers `denoiser` for every graph built with this feature.
    fn denoiser(&mut self, denoiser: Denoiser);
}

impl RegisterDenoiser for FeatureRegistrar<'_> {
    fn denoiser(&mut self, denoiser: Denoiser) {
        self.prepare(Arc::new(move |env| offer(env, denoiser.clone())));
    }
}

/// Offers `denoiser` to the graph being built ([`RegisterDenoiser`] does it
/// for a feature; a test or a host without features calls it directly).
pub fn offer(env: &mut BuildEnv<'_>, denoiser: Denoiser) {
    env.shared(Denoisers::default).0.push(denoiser);
}

/// The denoiser that would filter `signal` in this graph: the first
/// registered that supports it.
pub fn denoiser_for(env: &mut BuildEnv<'_>, signal: &Signal) -> Option<&'static str> {
    chosen(env, signal).map(|d| d.name)
}

fn chosen(env: &mut BuildEnv<'_>, signal: &Signal) -> Option<Denoiser> {
    let denoisers = env.shared(Denoisers::default).0.clone();
    denoisers.into_iter().find(|d| (d.supports)(env, signal))
}

/// Adds the frame's denoiser's passes for `signal` here, after the
/// producer's passes that write its inputs; returns what to read. With no
/// denoiser for it, adds nothing: the inputs pass through.
pub fn filter(build: &mut NodeBuild<'_, '_>, signal: &Signal) -> Result<Filtered, String> {
    match chosen(build.env, signal) {
        Some(denoiser) => {
            (denoiser.filter)(build, signal)
                .map_err(|e| format!("{} denoising `{}`: {e}", denoiser.name, signal.name))?;
            Ok(Filtered {
                textures: signal.outputs,
                block: TraceBlock::One,
                denoiser: Some(denoiser.name),
            })
        }
        None => Ok(Filtered {
            textures: signal.inputs,
            block: signal.block,
            denoiser: None,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every pixel of a block is traced once in `side^2` frames, and the
    /// blocks of a group start at different pixels.
    #[test]
    fn the_traced_pixel_visits_its_whole_block() {
        for side in [1u32, 2, 4] {
            for by in 0..side {
                for bx in 0..side {
                    let mut seen = vec![false; (side * side) as usize];
                    for frame in 0..side * side {
                        let [x, y] = traced_offset([bx, by], frame + 1000, side);
                        assert!(x < side && y < side);
                        seen[(y * side + x) as usize] = true;
                    }
                    assert!(seen.iter().all(|s| *s), "side {side} block {bx},{by}");
                }
            }
            let mut first: Vec<_> = (0..side * side)
                .map(|i| traced_offset([i % side, i / side], 0, side))
                .collect();
            first.sort();
            first.dedup();
            assert_eq!(first.len() as u32, side * side, "side {side}");
        }
        // The Bayer order: the rank and the place are inverse.
        for side in [2u32, 4] {
            for r in 0..side * side {
                let [x, y] = bayer_place(r, side);
                assert_eq!(bayer_rank(x, y, side), r);
            }
        }
        assert_eq!(bayer_rank(1, 0, 4), 8);
        assert_eq!(bayer_rank(0, 1, 4), 12);
    }

    #[test]
    fn a_signal_names_its_resources_and_sizes() {
        let s = Signal::irradiance("test_diffuse", TraceBlock::Sixteen);
        assert_eq!(s.inputs, ["test_diffuse_signal0", "test_diffuse_signal1"]);
        assert_eq!(
            s.outputs,
            ["test_diffuse_denoised0", "test_diffuse_denoised1"]
        );
        assert_eq!(s.traced_size([1920, 1081]), [480, 271]);
        assert!(!s.confidence && s.with_confidence().confidence);
        assert_eq!(TraceBlock::of_pixels(4), Some(TraceBlock::Four));
        assert_eq!(TraceBlock::of_pixels(3), None);
    }

    #[test]
    fn without_a_denoiser_the_signal_passes_through() {
        crate::test_env::with_env(|env| {
            let s = Signal::irradiance("test_pass", TraceBlock::Four);
            assert_eq!(denoiser_for(env, &s), None);
            offer(
                env,
                Denoiser {
                    name: "test",
                    supports: Arc::new(|_, s| s.kind == SignalKind::Irradiance),
                    filter: Arc::new(|_, _| Ok(())),
                },
            );
            assert_eq!(denoiser_for(env, &s), Some("test"));
        });
    }
}
