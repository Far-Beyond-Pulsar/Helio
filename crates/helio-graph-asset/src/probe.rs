//! Pins generated from what passes declare.
//!
//! A node that adds passes has no pin list of its own to keep in step: build it
//! once on a device, read what its passes declare (`reads()`, `writes()`,
//! `declare_resources`, [`RenderPass::scene_buffers`](helio_core::RenderPass::scene_buffers)),
//! and derive the pins from that. Nothing here names a pass or a resource.
//!
//! For each node, over the passes it added, in graph order:
//!
//! * **outputs**: every resource a pass writes;
//! * **inputs**: every resource a pass reads that an earlier node wrote, or
//!   that the host supplies (a Scene pin). A resource only a *later* pass
//!   writes is last frame's value (feedback) and is not an input, and one
//!   nothing writes is an optional input with a fallback;
//! * **host inputs offered**: a node that declares a resource as supplied by the host
//!   (`declare_external_input`) gets a Scene output pin for it;
//! * **SceneDB inputs**: a Buffer pin `scenedb.<key>` for each SceneDB buffer a
//!   pass consumes.
//!
//! The kind of a pin is the kind its producer declares: a buffer for
//! `write_buffer`, a depth or colour texture for the others.

use std::collections::{BTreeSet, HashMap, HashSet};

use helio_core::{DeclaredKind, PassDeclaration};

use crate::asset::GraphAsset;
use crate::compile::build_graph;
use crate::pins::{PinDesc, PinType};
use crate::registry::{BuildEnv, NodePins, NodeRegistry};
use crate::validate::Diagnostic;

/// Prefix of the resource name of a SceneDB buffer pin.
pub const SCENEDB_PREFIX: &str = "scenedb.";

/// What probing found.
#[derive(Clone, Debug, Default)]
pub struct Probe {
    /// Pins by node kind (the first node of a kind decides).
    pub nodes: HashMap<String, NodePins>,
    /// Every SceneDB GPU buffer key: those the mirror registered and those a
    /// pass declares it consumes.
    pub scene_buffers: Vec<String>,
}

fn pin_type(kind: DeclaredKind) -> Option<PinType> {
    match kind {
        DeclaredKind::Texture => Some(PinType::TEXTURE),
        DeclaredKind::Depth => Some(PinType::DEPTH),
        DeclaredKind::Buffer => Some(PinType::BUFFER),
        DeclaredKind::Unknown => None,
    }
}

/// The pin an output named `resource` gets. Its id differs from an input of
/// the same resource, because a node may read and write one (`pre_aa`).
pub fn output_pin(ty: PinType, resource: &str) -> PinDesc {
    PinDesc { name: format!("out_{resource}"), ty, resource: resource.to_string() }
}

/// Builds `asset` on `env`'s device without locking it, and derives the pins of
/// every node kind in it from what its passes declare.
pub fn probe(
    asset: &GraphAsset,
    registry: &NodeRegistry,
    env: &mut BuildEnv<'_>,
) -> Result<Probe, Vec<Diagnostic>> {
    let mut observed: Vec<(String, std::ops::Range<usize>, HashSet<&'static str>)> = Vec::new();
    let graph = build_graph(asset, registry, env, false, &mut |node, graph, range| {
        let externals = graph.external_input_names().into_iter().collect();
        observed.push((node.kind.clone(), range, externals));
    })?;

    let declarations: Vec<PassDeclaration> =
        (0..graph.pass_count()).filter_map(|index| graph.pass_declaration(index)).collect();
    let external: HashSet<&'static str> = graph.external_input_names().into_iter().collect();

    // The kind a resource has is the one its first precise producer declares.
    let mut kinds: HashMap<&'static str, PinType> = HashMap::new();
    for declaration in &declarations {
        for write in &declaration.writes {
            if let Some(ty) = pin_type(write.kind) {
                kinds.entry(write.name).or_insert(ty);
            }
        }
    }
    let kind_of = |name: &str| kinds.get(name).copied().unwrap_or(PinType::TEXTURE);

    let mut probe = Probe::default();
    let mut scene_buffers: BTreeSet<String> = env
        .scene_db
        .store()
        .buffer_registry()
        .keys()
        .into_iter()
        .map(|key| key.as_str().to_string())
        .collect();

    let mut declared_externals: HashSet<&'static str> = HashSet::new();
    for (kind, range, externals) in observed {
        let written_before: HashSet<&'static str> = declarations[..range.start]
            .iter()
            .flat_map(|declaration| declaration.writes.iter().map(|write| write.name))
            .collect();

        let mut inputs = vec![PinDesc::flow("in")];
        let mut outputs = vec![PinDesc::flow("out")];
        // What the node declared as host-supplied is what it offers on its outputs.
        let mut offered: Vec<&'static str> =
            externals.difference(&declared_externals).copied().collect();
        offered.sort_unstable();
        for name in offered {
            outputs.push(PinDesc::resource(PinType::SCENE, name));
        }
        declared_externals.extend(externals);
        let mut seen_inputs: HashSet<&'static str> = HashSet::new();
        let mut seen_outputs: HashSet<&'static str> = HashSet::new();
        let mut written_inside: HashSet<&'static str> = HashSet::new();
        let mut consumed: BTreeSet<&'static str> = BTreeSet::new();

        for declaration in &declarations[range.clone()] {
            for read in &declaration.reads {
                if written_inside.contains(read.name) || !seen_inputs.insert(read.name) {
                    continue;
                }
                if written_before.contains(read.name) {
                    inputs.push(PinDesc::resource(kind_of(read.name), read.name));
                } else if external.contains(read.name) {
                    inputs.push(PinDesc::resource(PinType::SCENE, read.name));
                } else {
                    seen_inputs.remove(read.name);
                }
            }
            consumed.extend(declaration.scene_buffers.iter().copied());
            for write in &declaration.writes {
                written_inside.insert(write.name);
                if seen_outputs.insert(write.name) {
                    outputs.push(output_pin(kind_of(write.name), write.name));
                }
            }
        }
        for key in consumed {
            scene_buffers.insert(key.to_string());
            inputs.push(PinDesc::resource(PinType::BUFFER, format!("{SCENEDB_PREFIX}{key}")));
        }
        probe.nodes.entry(kind).or_insert(NodePins { inputs, outputs });
    }

    probe.scene_buffers = scene_buffers.into_iter().collect();
    Ok(probe)
}
