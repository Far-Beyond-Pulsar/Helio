//! A pixel's history: what the previous frame of a view held at a pixel, and
//! where this frame's pixel was then. One owner for every temporal pass (TSR,
//! the denoiser, a lighting model's own filters), instead of each keeping a
//! previous depth and normal, reprojecting and deciding disocclusion with its
//! own test (`docs/architecture/rendering.md`, section 5).
//!
//! The spine publishes it for each view, at the internal resolution, when a
//! reader asks while its graph builds:
//!
//! - [`request`]`(build, Need::Reprojection)`: [`REPROJECTION`], a uniform
//!   ([`HistoryReprojection`]) holding the f64 composition from this frame's
//!   clip space to the previous frame's, rebased across a frame-origin
//!   change. CPU work only.
//! - [`request`]`(build, Need::Pixels)`: that, and one compute pass a frame
//!   publishing [`PIXELS`] ("where a pixel was": its previous position and
//!   which of the bilinear taps there held the same surface), this frame's
//!   guides, [`DEPTH`] and [`NORMAL`] (view depth, normal and roughness:
//!   one copy a view, which every temporal pass and denoised signal reads
//!   instead of keeping its own), and the previous guides,
//!   [`PREVIOUS_DEPTH`] and [`PREVIOUS_NORMAL`] (last frame's; the two
//!   swap, nothing is copied).
//!
//! The first reader of a graph adds the publisher's passes where it asks
//! (after the G-buffer, before itself); later readers find them there. A
//! graph none of whose nodes asks schedules neither. Each graph keeps the
//! history of the views it renders (by view id: mono, each XR eye; a capture
//! is a graph of its own), so captures and stereo eyes never share one.
//!
//! **Conventions.** Positions are on the rasters as drawn: this frame's and
//! the previous frame's, each with its own camera jitter (the G-buffer's
//! texel `p` holds the surface at `p + 0.5` of this frame's jittered raster;
//! the previous guides hold last frame's at its jittered raster). A reader
//! whose own history is on another grid (TSR's unjittered output) removes
//! the jitter itself. View depth is positive along the view axis, 0 for
//! sky. Normals are frame space (the frame's axes are the world's: an origin
//! change moves no normal).
//!
//! Absent publisher (a host that builds no spine), [`request`] answers
//! `false` and the reader runs without history.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use helio_core::graph::ResourceFormat;
use helio_core::shader::ShaderSnippet;
use helio_feature_api::{BuildEnv, FeatureRegistrar, NodeBuild};

/// Marker opting a shader into [`SNIPPET`].
pub const MARKER: &str = "//!use frame_history";
/// `HistoryReprojection`, `history_reproject`, the [`PIXELS`] texel's decode
/// (`HistoryPixel`, `history_pixel`) and the bilinear taps it validates
/// (`history_taps`) (`shaders/history.wgsl`). Needs the prelude.
pub const SNIPPET: ShaderSnippet = helio_core::wgsl_snippet!(MARKER, "shaders/history.wgsl");

/// Marker opting a shader into [`SURFACE_STOPS_SNIPPET`].
pub const SURFACE_STOPS_MARKER: &str = "//!use frame_surface_stops";
/// The surface stops (`shaders/surface_stops.wgsl`): whether two points lie
/// on one surface (`surface_stops_weight`: the normal and plane stops), the
/// one test the denoiser's weights, a resampler's reuse across pixels and
/// the radiance cache's reads take.
pub const SURFACE_STOPS_SNIPPET: ShaderSnippet =
    helio_core::wgsl_snippet!(SURFACE_STOPS_MARKER, "shaders/surface_stops.wgsl");

/// Marker opting a shader into [`SURFACE_SEEN_SNIPPET`].
pub const SURFACE_SEEN_MARKER: &str = "//!use frame_surface_seen";
/// The surface stops' depth walk (`shaders/surface_stops_seen.wgsl`, with
/// [`SURFACE_STOPS_SNIPPET`]): whether the frame's depth shows an occluder
/// standing between two pixels (`surface_stops_seen`, over the shader's own
/// `surface_stops_point`).
pub const SURFACE_SEEN_SNIPPET: ShaderSnippet =
    helio_core::wgsl_snippet!(SURFACE_SEEN_MARKER, "shaders/surface_stops_seen.wgsl");

/// The view's reprojection: a uniform buffer holding one [`HistoryReprojection`].
pub const REPROJECTION: &str = "history_reprojection";
/// This frame's view depth (`R32Float`, positive, 0 for sky or nothing):
/// the guides every temporal pass and the denoiser read at a pixel, written
/// once a frame by the publisher. Next frame they are [`PREVIOUS_DEPTH`]
/// (the textures swap; nothing is copied).
pub const DEPTH: &str = "history_depth";
/// This frame's shading normal (xyz, frame space) and roughness (w)
/// (`Rgba16Float`); next frame's [`PREVIOUS_NORMAL`].
pub const NORMAL: &str = "history_normal";
/// Last frame's view depth (`R32Float`, positive, 0 for sky or nothing).
pub const PREVIOUS_DEPTH: &str = "history_previous_depth";
/// Last frame's shading normal (xyz, frame space) and roughness (w)
/// (`Rgba16Float`).
pub const PREVIOUS_NORMAL: &str = "history_previous_normal";
/// Where each pixel was (`Rgba16Float`): xy its surface's position in the
/// previous frame minus its own centre, in pixels (the previous position is
/// `p + 0.5 + xy`); z the [`TAP_BITS`] of the 2x2 texels around that
/// position that held the same surface; w the [`pixel_flags`]. Integers are
/// exact in a half float.
pub const PIXELS: &str = "history_pixels";

pub const PREVIOUS_DEPTH_FORMAT: ResourceFormat = ResourceFormat::R32Float;
pub const PREVIOUS_NORMAL_FORMAT: ResourceFormat = ResourceFormat::Rgba16Float;
pub const PIXELS_FORMAT: ResourceFormat = ResourceFormat::Rgba16Float;

/// [`PIXELS`]' z: tap `i` of the 2x2 at `floor(previous - 0.5) + (i & 1, i
/// >> 1)` held the same surface (bit `i`).
pub const TAP_BITS: u32 = 0xf;

/// [`PIXELS`]' w (`HISTORY_PIXEL_*` in WGSL).
pub mod pixel_flags {
    /// The surface moves by itself (the G-buffer's velocity, bit 0 of its
    /// w): its previous position is the velocity's, not the camera's.
    pub const MOVES: u32 = 1;
    /// The view has a previous frame whose guides this frame compared
    /// against (none on a view's first frame, after a resize or a gap).
    pub const HAS_PREVIOUS: u32 = 2;
}

/// `HistoryReprojection` (`shaders/history.wgsl`), 128 bytes: how a point of
/// this frame's clip space (`c = (ndc, raw depth, 1)`, jittered) maps to the
/// previous frame's, composed on the CPU in f64 from this frame's
/// view-projection, the previous frame's (both jittered, the camera's
/// `view_proj` and `prev_view_proj`) and the previous view (UE's
/// ClipToPrevClip). Through world space in f32 instead, a near point moves by
/// the rounding of the camera's position (hundreds of metres off the frame
/// origin) frame to frame, and history accumulated over dozens of frames
/// smears by it.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct HistoryReprojection {
    /// x: 1 when the view has a previous frame (0: the previous matrices are
    /// this frame's), y: frames the view has rendered (saturating), z and w 0.
    pub info: [u32; 4],
    /// The previous frame's homogeneous clip position of the point is this
    /// times `c`.
    pub clip_to_previous_clip: [f32; 16],
    /// The point's homogeneous w is `dot(this, c)`.
    pub clip_to_world_w: [f32; 4],
    /// The previous frame's clip position of this frame's eye (sky pixels
    /// reproject as directions: the point minus w times the eye).
    pub eye_to_previous_clip: [f32; 4],
    /// The previous view's z times w is `dot(this, c)`.
    pub clip_to_previous_view_z: [f32; 4],
}

impl HistoryReprojection {
    pub const BYTES: u64 = std::mem::size_of::<Self>() as u64;

    /// From this frame's view-projection (`view_proj`, jittered) and eye,
    /// the previous frame's view-projection and view, both already in this
    /// frame's coordinates (rebased: [`rebase_previous_view`] and
    /// `helio_core::temporal::rebase_previous_projection`); all
    /// column-major.
    pub fn new(
        view_proj: [f32; 16],
        eye: [f32; 3],
        previous_view_proj: [f32; 16],
        previous_view: [f32; 16],
    ) -> Self {
        let matrix = |m: [f32; 16]| glam::DMat4::from_cols_array(&m.map(f64::from));
        let clip_to_world = matrix(view_proj).inverse();
        let previous_view_proj = matrix(previous_view_proj);
        let clip_to_previous_clip = previous_view_proj * clip_to_world;
        let clip_to_previous_view = matrix(previous_view) * clip_to_world;
        let eye = glam::DVec4::new(f64::from(eye[0]), f64::from(eye[1]), f64::from(eye[2]), 1.0);
        let f32s = |v: glam::DVec4| v.to_array().map(|x| x as f32);
        Self {
            clip_to_previous_clip: clip_to_previous_clip.to_cols_array().map(|x| x as f32),
            clip_to_world_w: f32s(clip_to_world.row(3)),
            eye_to_previous_clip: f32s(previous_view_proj * eye),
            clip_to_previous_view_z: f32s(clip_to_previous_view.row(2)),
            info: [0; 4],
        }
    }

    /// This reprojection, marked as having a previous frame (`frames` the
    /// view has rendered).
    pub fn with_previous(mut self, has_previous: bool, frames: u32) -> Self {
        self.info = [u32::from(has_previous), frames, 0, 0];
        self
    }

    /// `history_reproject` on the CPU, in f64 (for tests): the previous
    /// raster uv (jittered), whether the point was in front of the previous
    /// camera, and its previous view depth (0 for sky).
    pub fn reproject(&self, raster_uv: [f64; 2], depth: f64) -> ([f64; 2], bool, f64) {
        let c = glam::DVec4::new(
            raster_uv[0] * 2.0 - 1.0,
            1.0 - raster_uv[1] * 2.0,
            depth,
            1.0,
        );
        let v = |a: [f32; 4]| glam::DVec4::from_array(a.map(f64::from));
        let w = v(self.clip_to_world_w).dot(c);
        let sky = depth <= 0.0;
        let m = glam::DMat4::from_cols_array(&self.clip_to_previous_clip.map(f64::from));
        let previous = m * c - if sky { w } else { 0.0 } * v(self.eye_to_previous_clip);
        let uv = [
            previous.x / previous.w * 0.5 + 0.5,
            0.5 - previous.y / previous.w * 0.5,
        ];
        let front = if sky { previous.w } else { previous.w * w };
        let depth = if sky {
            0.0
        } else {
            -v(self.clip_to_previous_view_z).dot(c) / w
        };
        (uv, front > 0.0, depth)
    }
}

/// The previous view matrix (`previous`, column-major) in a frame whose
/// origin moved by `origin_shift` (new origin minus old): this frame's
/// coordinates are the previous frame's minus the shift, so the view takes
/// `V * T(shift)`; only its translation column changes. Composed in f64, so
/// a large shift keeps the camera's small residual.
pub fn rebase_previous_view(previous: [f32; 16], origin_shift: [f64; 3]) -> [f32; 16] {
    assert!(
        origin_shift.iter().all(|v| v.is_finite()),
        "camera origin shift must be finite"
    );
    let mut rebased = previous;
    for row in 0..4 {
        rebased[12 + row] = (f64::from(previous[row]) * origin_shift[0]
            + f64::from(previous[4 + row]) * origin_shift[1]
            + f64::from(previous[8 + row]) * origin_shift[2]
            + f64::from(previous[12 + row])) as f32;
    }
    rebased
}

/// What a reader needs of the history.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Need {
    /// [`REPROJECTION`] alone (a reader with its own history at another
    /// resolution, as TSR's at the output's).
    Reprojection,
    /// [`REPROJECTION`], [`PIXELS`], this frame's guides ([`DEPTH`],
    /// [`NORMAL`]) and last frame's ([`PREVIOUS_DEPTH`],
    /// [`PREVIOUS_NORMAL`]).
    Pixels,
}

/// Adds the publisher's passes for one [`Need`] to the graph being built.
pub type PublishBuild = Arc<dyn Fn(&mut NodeBuild<'_, '_>) -> Result<(), String> + Send + Sync>;

/// Who publishes the history (the spine): a pass for [`REPROJECTION`] and
/// one for [`PIXELS`] and the guides, which reads the first.
#[derive(Clone)]
pub struct HistoryPublisher {
    pub reprojection: PublishBuild,
    pub pixels: PublishBuild,
}

#[derive(Default)]
struct Publisher {
    publisher: Option<HistoryPublisher>,
    reprojection: bool,
    pixels: bool,
}

/// What the spine registers its publisher through.
pub trait RegisterHistory {
    /// Registers `publisher` for every graph built with this feature.
    fn history(&mut self, publisher: HistoryPublisher);
}

impl RegisterHistory for FeatureRegistrar<'_> {
    fn history(&mut self, publisher: HistoryPublisher) {
        self.prepare(Arc::new(move |env| offer(env, publisher.clone())));
    }
}

/// Offers `publisher` to the graph being built ([`RegisterHistory`] does it
/// for a feature; a test or a host without features calls it directly).
///
/// # Panics
///
/// When a graph is offered two publishers: the history has one owner.
pub fn offer(env: &mut BuildEnv<'_>, publisher: HistoryPublisher) {
    let state = env.shared(Publisher::default);
    assert!(
        state.publisher.is_none(),
        "a frame's history has one publisher"
    );
    state.publisher = Some(publisher);
}

/// Asks for the history `need` at this point of the graph being built: the
/// first reader that needs a part adds its publisher's passes here (after
/// whatever was built before it, the G-buffer included), so call it before
/// adding the passes that read it. Returns whether a publisher answers; a
/// reader without one runs without history.
pub fn request(build: &mut NodeBuild<'_, '_>, need: Need) -> Result<bool, String> {
    let state = build.env.shared(Publisher::default);
    let Some(publisher) = state.publisher.clone() else {
        return Ok(false);
    };
    let add_reprojection = !std::mem::replace(&mut state.reprojection, true);
    let add_pixels = need == Need::Pixels && !std::mem::replace(&mut state.pixels, true);
    if add_reprojection {
        (publisher.reprojection)(build).map_err(|e| format!("the history's reprojection: {e}"))?;
    }
    if add_pixels {
        (publisher.pixels)(build).map_err(|e| format!("the history's pixels: {e}"))?;
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_reprojection_has_its_contract_size() {
        assert_eq!(HistoryReprojection::BYTES, 128);
    }

    #[test]
    fn a_previous_view_rebase_keeps_physical_coordinates_and_f64_cancellation() {
        // A 90-degree view rotation maps (x, y, z) to (-z, y, x), plus
        // translation.
        let view = [
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, -3.0, 1.0,
        ];
        let rebased = rebase_previous_view(view, [2.5, -1.25, 7.0]);
        assert_eq!(&rebased[..12], &view[..12]);
        assert_eq!(&rebased[12..], &[-6.0, 0.75, -0.5, 1.0]);
        let mut translated = [0.0; 16];
        for i in [0, 5, 10, 15] {
            translated[i] = 1.0;
        }
        translated[14] = -16_777_216.0;
        assert_eq!(
            rebase_previous_view(translated, [0.0, 0.0, 16_777_216.25])[14],
            0.25,
            "casting the origin shift to f32 before composition loses the physical residual"
        );
    }

    /// A still camera reprojects every point onto itself; a camera that
    /// moved sideways moves it by the parallax of its depth.
    #[test]
    fn the_cpu_reprojection_follows_the_camera() {
        let proj = helio_core::depth::perspective(1.0, 1.0, 0.1);
        let view = |x: f32| glam::Mat4::from_translation(glam::Vec3::new(-x, 0.0, 0.0));
        let (now, before) = (proj * view(0.5), proj * view(0.0));
        let still = HistoryReprojection::new(
            now.to_cols_array(),
            [0.5, 0.0, 0.0],
            now.to_cols_array(),
            view(0.5).to_cols_array(),
        );
        let (uv, front, depth) = still.reproject([0.3, 0.7], 0.05);
        assert!(front && (uv[0] - 0.3).abs() < 1e-6 && (uv[1] - 0.7).abs() < 1e-6);
        assert!((depth - 0.1 / 0.05).abs() < 1e-4, "{depth}");
        let moved = HistoryReprojection::new(
            now.to_cols_array(),
            [0.5, 0.0, 0.0],
            before.to_cols_array(),
            view(0.0).to_cols_array(),
        );
        let (uv, _, _) = moved.reproject([0.5, 0.5], 0.05);
        // The camera moved 0.5 m right; the point 2 m ahead sat 0.5 m to the
        // right of the previous view's axis: x_ndc = 0.5 / 2 / tan(0.5).
        let expected = 0.5 + 0.5 * (0.5 / 2.0) / (0.5f64).tan();
        assert!((uv[0] - expected).abs() < 1e-5, "{} {expected}", uv[0]);
    }
}
