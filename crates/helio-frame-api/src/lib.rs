//! The frame: where every kind of object meets, without the frame knowing
//! any kind (docs/architecture/plugin-migration.md §2).
//!
//! - [`Slot`]s and draw [`Phase`]s are the frame's extension points. A
//!   feature places its nodes at a slot, or registers **drawers** into a phase
//!   (and compute that must run before it); a preset lists slots and phases,
//!   never a feature's nodes.
//! - [`gbuffer`]: the G-buffer layout every GBuffer drawer writes, and its
//!   surface tags.
//! - [`depth`]: the frame's depth resources (full-resolution depth, Hi-Z).
//! - [`color`]: the scene colour chain post features read and advance.
//! - [`media`]: the participating medium integrated along view rays, for
//!   what composites at its own depth.
//! - [`lighting`]: the inputs a lighting model reads, as generic contracts
//!   with stand-ins: indirect diffuse and specular, the sun's visibility and
//!   transmittance, transmission, the sky, the light list, view remaps.
//! - [`rays`]: the frame's TLAS and the buffers that follow it, published by
//!   whatever builds it.
//! - [`static_names`]: the read lists of a pass whose reads are what a
//!   publisher named (media, the sun's transmittance).
//! - [`host_input`]: resources a host supplies every frame, declared by the
//!   feature that reads them.
//!
//! Depends on `helio-core` and `helio-feature-api` only.

pub mod color;
pub mod depth;
pub mod diagnostics;
pub mod gbuffer;
mod host;
pub mod lighting;
pub mod media;
mod phase;
pub mod rays;

pub use host::{host_input, host_kind, HOST_INPUTS};

/// The scene buffers the frame defines, by key: any feature may read them,
/// whoever produces them (the renderer's entity generations and camera, a
/// lighting input's probe rows, a portal's view remaps). A host's catalog
/// links a feature's read of one of these to the frame rather than to a
/// kind's contract.
pub const FRAME_BUFFERS: &[&str] = &[
    helio_core::ENTITY_GENERATIONS_KEY.as_str(),
    helio_core::CAMERA_BUFFER_KEY.as_str(),
    lighting::ENVIRONMENT_PROBES,
    lighting::VIEW_REMAPS,
];
pub use phase::{
    phase_settings, Attachment, AttachmentFormat, FrameNode, Phase, PhaseList, PhaseViews, Slot,
};

/// `names` as a `'static` slice, one per distinct list: a graph's passes
/// declare `'static` read and write lists, and a pass that reads what a
/// publisher named ([`media`], [`lighting::sun_transmittance`]) knows its
/// list only when it is built.
pub fn static_names(names: &[&'static str]) -> &'static [&'static str] {
    static LISTS: std::sync::OnceLock<std::sync::Mutex<Vec<&'static [&'static str]>>> =
        std::sync::OnceLock::new();
    let mut lists = LISTS.get_or_init(Default::default).lock().unwrap();
    if let Some(list) = lists.iter().find(|list| **list == names) {
        return list;
    }
    let list: &'static [&'static str] = Box::leak(names.to_vec().into_boxed_slice());
    lists.push(list);
    list
}
