//! The G-buffer layout: what every drawer of [`crate::Phase::GBuffer`] writes
//! and the lighting model reads.
//!
//! The first four targets are one group (`gbuffer`, read together as a
//! `ViewGroup<4>`); the rest are read one by one. A drawer that has nothing
//! for a target still binds it with the same format, so drawers fuse.

use helio_core::graph::ResourceFormat;
use helio_feature_api::BuildEnv;

/// The bundled albedo/normal/ORM/emissive group, as readers name it.
pub const GROUP: &str = "gbuffer";
pub const ALBEDO: &str = "gbuffer_albedo";
pub const NORMAL: &str = "gbuffer_normal";
pub const ORM: &str = "gbuffer_orm";
pub const EMISSIVE: &str = "gbuffer_emissive";
/// The surface channel (`Rg16Float`): `x < 0` holds the frame's surface tags
/// (the prelude's `GBUFFER_TAG_*`: a plain surface, one with its own
/// occlusion, a traced one, a view remap's chain, see
/// [`crate::lighting::VIEW_REMAPS`]); `x >= 0` holds the payload one feature
/// requested for the graph ([`request_payload`]), which the drawers that carry
/// it write. Phase openers clear it to the plain tag ([`SURFACE_CLEAR`]).
///
/// The G-buffer is at WebGPU's limit of eight colour targets, so a feature's
/// per-pixel extension shares this channel under a declared owner rather
/// than adding a target (Unreal packs its shading models' data into shared
/// custom-data channels the same way).
pub const SURFACE: &str = "gbuffer_surface";
pub const SSS: &str = "gbuffer_sss";
pub const EXTRA: &str = "gbuffer_extra";
/// Per-texel motion vectors, written by every G-buffer drawer.
///
/// `xy` is the texel's motion since last frame in pixels: its raster position
/// this frame minus the same surface point's last frame, both with that
/// frame's camera jitter (fragment position minus the point projected by the
/// previous, jittered view-projection and its previous transform or
/// animation). Bit 0 of `w` (as an integer) is set for geometry that moves
/// by itself: a moving or skinned instance, animated foliage. Elsewhere
/// (static geometry, drawers that write none, the cleared target) a reader
/// (TSR) derives the camera's motion from depth, which is exact where a
/// writer's camera-relative matrices round (UE's velocity buffer holds only
/// dynamic geometry the same way). `z` is the FP16 view-depth residual and
/// `w` = 2 marks a valid correction for precise ray-query receiver origins.
pub const VELOCITY: &str = "gbuffer_velocity";
/// The scene depth (see [`crate::depth`] for the rest of the frame's depth
/// resources).
pub const DEPTH: &str = helio_core::resource_keys::DEPTH;

pub const ALBEDO_FORMAT: ResourceFormat = ResourceFormat::Rgba8Unorm;
pub const NORMAL_FORMAT: ResourceFormat = ResourceFormat::Rgba16Float;
pub const ORM_FORMAT: ResourceFormat = ResourceFormat::Rgba8Unorm;
pub const EMISSIVE_FORMAT: ResourceFormat = ResourceFormat::Rgba16Float;
pub const SURFACE_FORMAT: ResourceFormat = ResourceFormat::Rg16Float;
pub const SSS_FORMAT: ResourceFormat = ResourceFormat::Rgba16Float;
pub const EXTRA_FORMAT: ResourceFormat = ResourceFormat::Rgba16Float;
pub const VELOCITY_FORMAT: ResourceFormat = ResourceFormat::Rgba16Float;

/// Every target a drawer of [`crate::Phase::GBuffer`] binds, in fragment
/// location order (`@location(0)` to `@location(7)`), with its format. A
/// drawer builds its pipeline's colour targets from this (or
/// [`TARGET_FORMATS`]), never from its own list.
pub const TARGETS: [(&str, ResourceFormat); 8] = [
    (ALBEDO, ALBEDO_FORMAT),
    (NORMAL, NORMAL_FORMAT),
    (ORM, ORM_FORMAT),
    (EMISSIVE, EMISSIVE_FORMAT),
    (SURFACE, SURFACE_FORMAT),
    (SSS, SSS_FORMAT),
    (EXTRA, EXTRA_FORMAT),
    (VELOCITY, VELOCITY_FORMAT),
];

/// Marker opting a shader into [`OUTPUT_SNIPPET`].
pub const OUTPUT_MARKER: &str = "//!use gbuffer_output";
/// `GBufferOutput` (the fragment outputs for [`TARGETS`], in location order)
/// and `GBufferDepthOutput` (the same with `frag_depth`): every G-buffer
/// drawer's fragment returns one, so the layout has one owner.
pub const OUTPUT_SNIPPET: helio_core::shader::ShaderSnippet =
    helio_core::wgsl_snippet!(OUTPUT_MARKER, "shaders/gbuffer_output.wgsl");

/// [`TARGETS`]' formats, in location order.
pub const TARGET_FORMATS: [wgpu::TextureFormat; 8] = [
    ALBEDO_FORMAT.to_wgpu(),
    NORMAL_FORMAT.to_wgpu(),
    ORM_FORMAT.to_wgpu(),
    EMISSIVE_FORMAT.to_wgpu(),
    SURFACE_FORMAT.to_wgpu(),
    SSS_FORMAT.to_wgpu(),
    EXTRA_FORMAT.to_wgpu(),
    VELOCITY_FORMAT.to_wgpu(),
];

/// A drawer's pipeline colour targets for [`TARGETS`]: every target, no
/// blending, all channels written. (A drawer that leaves targets unwritten
/// masks them from this list.)
pub fn color_targets() -> [Option<wgpu::ColorTargetState>; 8] {
    TARGET_FORMATS.map(|format| {
        Some(wgpu::ColorTargetState {
            format,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })
    })
}

/// The group's members in binding order, with their formats.
pub const GROUP_MEMBERS: [(&str, ResourceFormat); 4] = [
    (ALBEDO, ALBEDO_FORMAT),
    (NORMAL, NORMAL_FORMAT),
    (ORM, ORM_FORMAT),
    (EMISSIVE, EMISSIVE_FORMAT),
];

/// The value a G-buffer phase opener clears [`SURFACE`] to: the plain tag
/// (`GBUFFER_TAG_PLAIN` in the prelude), so a pixel nothing drew carries no
/// payload. Every other target clears to zero.
pub const SURFACE_CLEAR: wgpu::Color = wgpu::Color {
    r: -1.0,
    g: -1.0,
    b: 0.0,
    a: 0.0,
};

/// The value a G-buffer phase opener clears target `name` to.
pub fn clear_value(name: &str) -> wgpu::Color {
    if name == SURFACE {
        SURFACE_CLEAR
    } else {
        wgpu::Color::TRANSPARENT
    }
}

/// What a graph's drawers write into [`SURFACE`]'s payload (`x >= 0`): an id
/// the requesting feature's api crate declares (baked lighting's lightmap
/// UV). One per graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SurfacePayload(pub &'static str);

/// Requests [`SURFACE`]'s payload for the graph being built: the drawers that
/// carry `payload` write it (they ask [`payload`] when they build), every
/// other surface keeps its tag. Call from a feature's prepare hook, before
/// the drawers build. Two features asking for different payloads is a build
/// error: the channel has one owner.
///
/// # Panics
///
/// When another payload was requested for this graph.
pub fn request_payload(env: &mut BuildEnv<'_>, payload: SurfacePayload) {
    match env.setting::<SurfacePayload>() {
        Some(existing) if *existing != payload => panic!(
            "the G-buffer surface payload is `{}`; `{}` cannot share it",
            existing.0, payload.0
        ),
        Some(_) => {}
        None => env.insert_setting(payload),
    }
}

/// The surface payload requested for the graph being built, if any.
pub fn payload(env: &BuildEnv<'_>) -> Option<SurfacePayload> {
    env.setting::<SurfacePayload>().copied()
}
