//! The G-buffer layout: what every drawer of [`crate::Phase::GBuffer`] writes
//! and the lighting model reads.
//!
//! The first four targets are one group (`gbuffer`, read together as a
//! `ViewGroup<4>`); the rest are read one by one. A drawer that has nothing
//! for a target still binds it with the same format, so drawers fuse.

use helio_core::graph::ResourceFormat;

/// The bundled albedo/normal/ORM/emissive group, as readers name it.
pub const GROUP: &str = "gbuffer";
pub const ALBEDO: &str = "gbuffer_albedo";
pub const NORMAL: &str = "gbuffer_normal";
pub const ORM: &str = "gbuffer_orm";
pub const EMISSIVE: &str = "gbuffer_emissive";
/// Lightmap UVs; a view remap's chain is tagged here too
/// ([`crate::lighting::VIEW_REMAPS`]).
pub const LIGHTMAP_UV: &str = "gbuffer_lightmap_uv";
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
pub const LIGHTMAP_UV_FORMAT: ResourceFormat = ResourceFormat::Rg16Float;
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
    (LIGHTMAP_UV, LIGHTMAP_UV_FORMAT),
    (SSS, SSS_FORMAT),
    (EXTRA, EXTRA_FORMAT),
    (VELOCITY, VELOCITY_FORMAT),
];

/// [`TARGETS`]' formats, in location order.
pub const TARGET_FORMATS: [wgpu::TextureFormat; 8] = [
    ALBEDO_FORMAT.to_wgpu(),
    NORMAL_FORMAT.to_wgpu(),
    ORM_FORMAT.to_wgpu(),
    EMISSIVE_FORMAT.to_wgpu(),
    LIGHTMAP_UV_FORMAT.to_wgpu(),
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

