//! The frame: where every kind of object meets, without the frame knowing
//! any kind (docs/architecture/plugin-migration.md §2).
//!
//! - [`Slot`]s and draw [`Phase`]s are the frame's extension points. A
//!   feature places its nodes at a slot, or registers **drawers** into a phase
//!   (and compute that must run before it); a preset lists slots and phases,
//!   never a feature's nodes.
//! - [`gbuffer`]: the G-buffer layout every GBuffer drawer writes.
//! - [`color`]: the scene colour chain post features read and advance.
//! - [`media`]: the participating medium integrated along view rays, for
//!   what composites at its own depth.
//! - [`lighting`]: the inputs a lighting model reads, as generic contracts
//!   with stand-ins: indirect diffuse and specular, the sun's visibility,
//!   transmission, the sky, the light list, view remaps.
//! - [`host_input`]: resources a host supplies every frame, declared by the
//!   feature that reads them.
//!
//! Depends on `helio-core` and `helio-feature-api` only.

pub mod color;
pub mod gbuffer;
mod host;
pub mod lighting;
pub mod media;
mod phase;

pub use host::{host_input, host_kind, HOST_INPUTS};
pub use phase::{
    phase_settings, Attachment, AttachmentFormat, FrameNode, Phase, PhaseList, PhaseViews, Slot,
};
