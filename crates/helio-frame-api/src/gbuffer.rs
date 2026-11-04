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
/// XY: pixel motion; Z: the FP16 view-depth residual; W = 2 marks a valid
/// correction for precise ray-query receiver origins.
pub const VELOCITY: &str = "gbuffer_velocity";
/// The scene depth.
pub const DEPTH: &str = "depth";

pub const ALBEDO_FORMAT: ResourceFormat = ResourceFormat::Rgba8Unorm;
pub const NORMAL_FORMAT: ResourceFormat = ResourceFormat::Rgba16Float;
pub const ORM_FORMAT: ResourceFormat = ResourceFormat::Rgba8Unorm;
pub const EMISSIVE_FORMAT: ResourceFormat = ResourceFormat::Rgba16Float;
pub const LIGHTMAP_UV_FORMAT: ResourceFormat = ResourceFormat::Rg16Float;
pub const SSS_FORMAT: ResourceFormat = ResourceFormat::Rgba16Float;
pub const EXTRA_FORMAT: ResourceFormat = ResourceFormat::Rgba16Float;
pub const VELOCITY_FORMAT: ResourceFormat = ResourceFormat::Rgba16Float;

/// The group's members in binding order, with their formats.
pub const GROUP_MEMBERS: [(&str, ResourceFormat); 4] = [
    (ALBEDO, ALBEDO_FORMAT),
    (NORMAL, NORMAL_FORMAT),
    (ORM, ORM_FORMAT),
    (EMISSIVE, EMISSIVE_FORMAT),
];
