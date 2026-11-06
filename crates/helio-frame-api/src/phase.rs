//! The frame's extension points: slots, and the draw phases kinds register
//! drawers into.
//!
//! How others do it, and why this shape: Bevy's `core_pipeline` owns the
//! phases (opaque, alpha mask, transparent, shadow) and their views; a plugin
//! queues phase items per view, and one phase node draws them all in one
//! render pass. Unity SRP injects renderer features at named events. Here a
//! **drawer** is a node whose passes draw into a phase's attachments; a phase
//! is a place in the preset graph with an attachment contract and three
//! ordered stages:
//!
//! 1. **before**: compute a drawer needs done before any of the phase's
//!    render passes (foliage placement, GI's ray generation). Kept out of the
//!    drawers, because a compute pass between two render passes that share
//!    attachments breaks the graph's subpass fusion.
//! 2. **open**: the drawer that begins the phase by clearing its
//!    attachments (the mesh G-buffer pass). A phase with drawers but no
//!    opener is an error, never a frame of last frame's pixels.
//! 3. **draw**: every other drawer, loading the attachments. Adjacent
//!    drawers whose attachments match exactly fuse into one render pass
//!    (`RenderGraph`'s subpass chains), which is Bevy's one-pass-per-phase
//!    without a central draw loop: each kind keeps its own pass, culling and
//!    pipelines.
//!
//! The contract is checked when a drawer builds: every attachment of its
//! phase a drawer's pass declares must have the phase's format, so a drawer
//! cannot silently break fusion or write a G-buffer layout of its own.

use std::sync::Arc;

use helio_core::graph::{ResourceAccess, ResourceFormat};
use helio_core::RenderGraph;
use helio_feature_api::{BuildEnv, ExtensionPoint, NodeDescriptor};

use crate::gbuffer;

/// Where in the frame a feature's node runs, outside the draw phases. A
/// preset places the nodes registered at each slot there (by order, then
/// kind), so a feature joins the pipeline without the preset naming it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Slot {
    /// After the scene's batches are gathered, before the shadow views:
    /// per-frame tables everything after reads (the atmosphere's).
    Prepare,
    /// After the G-buffer phase, before decals: what reads the complete
    /// G-buffer (GI's tracers).
    Geometry,
    /// After decals, before the lighting model: what lighting reads that is
    /// not the G-buffer (indirect diffuse and specular, the sun's
    /// visibility, transmission).
    LightingInputs,
    /// After the scene is lit, before the overlays, media and transparency:
    /// resolves that write the lit colour and keep the pixels they do not
    /// touch (the sky's composite).
    Lighting,
    /// After the scene colour is lit and overlaid, before transparency:
    /// participating media composited over it (fog), and the post settings
    /// media and the tone map read.
    Media,
    /// After anti-aliasing, before the tone map: effects on the resolved
    /// scene-linear colour (the camera's lens).
    Resolved,
    /// After the tone map, over the final image: debug views, 2D overlays.
    Display,
    /// After everything drawn over the final image: consumers of the
    /// finished frame (captures, readbacks).
    Output,
}

impl Slot {
    pub const ALL: [Slot; 8] = [
        Slot::Prepare,
        Slot::Geometry,
        Slot::LightingInputs,
        Slot::Lighting,
        Slot::Media,
        Slot::Resolved,
        Slot::Display,
        Slot::Output,
    ];

    pub fn point(self) -> ExtensionPoint {
        ExtensionPoint(match self {
            Slot::Prepare => "slot.prepare",
            Slot::Geometry => "slot.geometry",
            Slot::LightingInputs => "slot.lighting_inputs",
            Slot::Lighting => "slot.lighting",
            Slot::Media => "slot.media",
            Slot::Resolved => "slot.resolved",
            Slot::Display => "slot.display",
            Slot::Output => "slot.output",
        })
    }
}

/// A draw phase: a place in the frame where every kind that has something to
/// draw there draws, into attachments the frame owns.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Phase {
    /// Depth into the shadow views (the shadow atlas faces).
    ShadowCasters,
    /// Depth only, before the G-buffer (no preset runs it yet).
    DepthPrepass,
    /// Opaque surfaces into the G-buffer and the scene depth.
    GBuffer,
    /// Lit opaque surfaces straight into the scene colour (pipelines without
    /// a G-buffer).
    Forward,
    /// Alpha-blended surfaces over the scene colour, depth-tested.
    Transparent,
    /// Over the lit scene colour, depth-tested, before media: light
    /// sprites, glows, editor gizmos.
    Overlay,
}

/// Which views a phase is drawn for. A graph is built for one camera (stereo
/// eyes are layers of it), so a phase's drawer list is per view: the camera's
/// phases draw for it, the shadow casters for every shadow face the lights'
/// shadow views pick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhaseViews {
    Camera,
    ShadowFaces,
}

/// The format an attachment of a phase has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttachmentFormat {
    Fixed(ResourceFormat),
    /// The scene-linear colour's format (`BuildEnv::scene_format`).
    SceneColor,
}

/// One attachment of a phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Attachment {
    pub name: &'static str,
    pub format: AttachmentFormat,
}

const fn fixed(name: &'static str, format: ResourceFormat) -> Attachment {
    Attachment {
        name,
        format: AttachmentFormat::Fixed(format),
    }
}

const fn scene(name: &'static str) -> Attachment {
    Attachment {
        name,
        format: AttachmentFormat::SceneColor,
    }
}

const GBUFFER_ATTACHMENTS: [Attachment; 9] = [
    fixed(gbuffer::ALBEDO, gbuffer::ALBEDO_FORMAT),
    fixed(gbuffer::NORMAL, gbuffer::NORMAL_FORMAT),
    fixed(gbuffer::ORM, gbuffer::ORM_FORMAT),
    fixed(gbuffer::EMISSIVE, gbuffer::EMISSIVE_FORMAT),
    fixed(gbuffer::LIGHTMAP_UV, gbuffer::LIGHTMAP_UV_FORMAT),
    fixed(gbuffer::SSS, gbuffer::SSS_FORMAT),
    fixed(gbuffer::EXTRA, gbuffer::EXTRA_FORMAT),
    fixed(gbuffer::VELOCITY, gbuffer::VELOCITY_FORMAT),
    fixed(gbuffer::DEPTH, ResourceFormat::Depth32Float),
];

impl Phase {
    pub const ALL: [Phase; 6] = [
        Phase::ShadowCasters,
        Phase::DepthPrepass,
        Phase::GBuffer,
        Phase::Forward,
        Phase::Transparent,
        Phase::Overlay,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Phase::ShadowCasters => "shadow_casters",
            Phase::DepthPrepass => "depth_prepass",
            Phase::GBuffer => "gbuffer",
            Phase::Forward => "forward",
            Phase::Transparent => "transparent",
            Phase::Overlay => "overlay",
        }
    }

    pub fn views(self) -> PhaseViews {
        match self {
            Phase::ShadowCasters => PhaseViews::ShadowFaces,
            _ => PhaseViews::Camera,
        }
    }

    /// The attachments the phase's drawers write, with the formats every
    /// drawer must declare them in. The colour phases draw into the scene
    /// colour current where they run ([`crate::color`]); its format is
    /// checked, its name is the chain's.
    pub fn attachments(self) -> &'static [Attachment] {
        const SHADOW: [Attachment; 1] = [fixed("shadow_atlas", ResourceFormat::Depth32Float)];
        const DEPTH: [Attachment; 1] = [fixed(gbuffer::DEPTH, ResourceFormat::Depth32Float)];
        const COLOUR: [Attachment; 1] = [scene(crate::color::SCENE_COLOR)];
        match self {
            Phase::ShadowCasters => &SHADOW,
            Phase::DepthPrepass => &DEPTH,
            Phase::GBuffer => &GBUFFER_ATTACHMENTS,
            Phase::Forward | Phase::Overlay => &COLOUR,
            Phase::Transparent => &[],
        }
    }

    /// Compute run before the phase's render passes.
    pub fn before(self) -> ExtensionPoint {
        ExtensionPoint(match self {
            Phase::ShadowCasters => "phase.shadow_casters.before",
            Phase::DepthPrepass => "phase.depth_prepass.before",
            Phase::GBuffer => "phase.gbuffer.before",
            Phase::Forward => "phase.forward.before",
            Phase::Transparent => "phase.transparent.before",
            Phase::Overlay => "phase.overlay.before",
        })
    }

    /// The drawer that clears the attachments.
    pub fn open(self) -> ExtensionPoint {
        ExtensionPoint(match self {
            Phase::ShadowCasters => "phase.shadow_casters.open",
            Phase::DepthPrepass => "phase.depth_prepass.open",
            Phase::GBuffer => "phase.gbuffer.open",
            Phase::Forward => "phase.forward.open",
            Phase::Transparent => "phase.transparent.open",
            Phase::Overlay => "phase.overlay.open",
        })
    }

    /// Drawers that load the attachments.
    pub fn draw(self) -> ExtensionPoint {
        ExtensionPoint(match self {
            Phase::ShadowCasters => "phase.shadow_casters.draw",
            Phase::DepthPrepass => "phase.depth_prepass.draw",
            Phase::GBuffer => "phase.gbuffer.draw",
            Phase::Forward => "phase.forward.draw",
            Phase::Transparent => "phase.transparent.draw",
            Phase::Overlay => "phase.overlay.draw",
        })
    }

    /// Whether the phase's attachments start cleared by someone else (the
    /// colour phases draw over the lit or background colour), so it needs no
    /// opener.
    pub fn needs_opener(self) -> bool {
        matches!(
            self,
            Phase::ShadowCasters | Phase::DepthPrepass | Phase::GBuffer
        )
    }
}

/// The phases a graph runs: a setting a preset supplies, so a drawer can tell
/// what else the frame has (the forward drawer draws every opaque surface
/// when there is no G-buffer phase).
#[derive(Clone, Debug, Default)]
pub struct PhaseList(pub Vec<Phase>);

impl PhaseList {
    pub fn runs(&self, phase: Phase) -> bool {
        self.0.contains(&phase)
    }
}

/// The phases the graph being built runs (none when its builder says nothing).
pub fn phase_settings(env: &BuildEnv<'_>) -> PhaseList {
    env.setting::<PhaseList>().cloned().unwrap_or_default()
}

/// Placing a node in the frame.
pub trait FrameNode: Sized {
    /// Joins preset graphs at `slot`, `order` among the slot's nodes.
    fn in_slot(self, slot: Slot, order: i32) -> Self;
    /// Draws into `phase`, loading its attachments; checked against the
    /// phase's attachment contract when it builds.
    fn drawer(self, phase: Phase, order: i32) -> Self;
    /// Opens `phase`: clears its attachments, then draws.
    fn opens(self, phase: Phase, order: i32) -> Self;
    /// Compute run before `phase`'s render passes.
    fn before_phase(self, phase: Phase, order: i32) -> Self;
}

impl FrameNode for NodeDescriptor {
    fn in_slot(self, slot: Slot, order: i32) -> Self {
        self.at(slot.point(), order)
    }

    fn drawer(self, phase: Phase, order: i32) -> Self {
        self.at(phase.draw(), order).with_check(contract_check(phase))
    }

    fn opens(self, phase: Phase, order: i32) -> Self {
        self.at(phase.open(), order).with_check(contract_check(phase))
    }

    fn before_phase(self, phase: Phase, order: i32) -> Self {
        self.at(phase.before(), order)
    }
}

/// Checks that the passes a drawer of `phase` added declare the phase's
/// attachments in its formats.
fn contract_check(phase: Phase) -> helio_feature_api::CheckFn {
    Arc::new(move |env, graph, passes| check_attachments(phase, env, graph, passes))
}

fn check_attachments(
    phase: Phase,
    env: &BuildEnv<'_>,
    graph: &RenderGraph,
    passes: std::ops::Range<usize>,
) -> Result<(), String> {
    let scene = ResourceFormat::from(env.scene_format());
    for index in passes {
        let Some(pass) = graph.pass_declaration(index) else {
            continue;
        };
        for decl in graph.pass_resources(index) {
            if decl.access != ResourceAccess::Write {
                continue;
            }
            let Some(format) = decl.format else { continue };
            let Some(attachment) = phase.attachments().iter().find(|a| a.name == decl.name) else {
                continue;
            };
            let expected = match attachment.format {
                AttachmentFormat::Fixed(format) => format,
                AttachmentFormat::SceneColor => scene,
            };
            if format != expected {
                return Err(format!(
                    "drawer pass `{}` of phase `{}` writes `{}` as {format:?}; the phase's \
                     contract is {expected:?}",
                    pass.pass,
                    phase.name(),
                    decl.name
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_point_is_distinct() {
        let mut points: Vec<&str> = Phase::ALL
            .iter()
            .flat_map(|p| [p.before().0, p.open().0, p.draw().0])
            .chain(
                Slot::ALL.map(|s| s.point().0),
            )
            .collect();
        let count = points.len();
        points.sort_unstable();
        points.dedup();
        assert_eq!(points.len(), count);
    }

    #[test]
    fn the_gbuffer_contract_lists_the_layout() {
        let names: Vec<_> = Phase::GBuffer.attachments().iter().map(|a| a.name).collect();
        assert!(names.contains(&gbuffer::ALBEDO) && names.contains(&gbuffer::DEPTH));
        assert_eq!(names.len(), 9);
    }
}
