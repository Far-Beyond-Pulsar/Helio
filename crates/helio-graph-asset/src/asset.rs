//! The saved form of a render graph: nodes with parameters, and wires.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// File extension of a render-graph asset.
pub const EXTENSION: &str = "rendergraph";
/// Current `format_version`.
pub const FORMAT_VERSION: u32 = 1;

/// Node parameters by key.
pub type Params = BTreeMap<String, Value>;

/// One end of a wire.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PinRef {
    pub node: String,
    pub pin: String,
}

impl PinRef {
    pub fn new(node: impl Into<String>, pin: impl Into<String>) -> Self {
        Self { node: node.into(), pin: pin.into() }
    }
}

/// A wire from an output pin to an input pin.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Connection {
    pub from: PinRef,
    pub to: PinRef,
}

/// A node placed in the graph.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NodeInstance {
    /// Unique within the asset.
    pub id: String,
    /// The [`NodeDescriptor::kind`](crate::NodeDescriptor::kind) it instantiates.
    pub kind: String,
    #[serde(default)]
    pub params: Params,
    /// Canvas position, editor-only.
    #[serde(default)]
    pub position: [f32; 2],
}

/// A labelled box drawn around a group of nodes, editor-only.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphComment {
    pub id: String,
    pub text: String,
    pub position: [f32; 2],
    pub size: [f32; 2],
    /// Normalised RGBA.
    pub color: [f32; 4],
    /// Nodes the box surrounds.
    #[serde(default)]
    pub nodes: Vec<String>,
}

/// A pin a section shows on the outside: where a wire crosses its boundary.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SectionPort {
    /// Id of the pin on the section's node, and of the matching pin on its entry
    /// or exit inside.
    pub name: String,
    pub ty: crate::pins::PinType,
    /// The resource the pin carries, empty for flow.
    #[serde(default)]
    pub resource: String,
}

impl SectionPort {
    pub fn to_pin(&self) -> crate::pins::PinDesc {
        crate::pins::PinDesc {
            name: self.name.clone(),
            ty: self.ty,
            resource: self.resource.clone(),
            history: false,
            optional: false,
        }
    }
}

/// A named group of nodes shown as one node (a collapsed graph in the editor).
/// A section is organisation only: it is expanded in place before the graph is
/// checked or built, so the pipeline is the one its nodes would make unboxed.
///
/// Inside, [`SECTION_ENTRY`] stands for the section's input pins (as the source
/// of wires) and [`SECTION_EXIT`] for its output pins (as their target).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphSection {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub inputs: Vec<SectionPort>,
    #[serde(default)]
    pub outputs: Vec<SectionPort>,
    #[serde(default)]
    pub nodes: Vec<NodeInstance>,
    #[serde(default)]
    pub connections: Vec<Connection>,
}

/// Prefix of the kind of a node standing for a section: `section:<id>`.
pub const SECTION_PREFIX: &str = "section:";
/// The pseudo node inside a section whose output pins are the section's inputs.
pub const SECTION_ENTRY: &str = "$entry";
/// The pseudo node inside a section whose input pins are the section's outputs.
pub const SECTION_EXIT: &str = "$exit";

/// A render graph asset (`.rendergraph`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GraphAsset {
    pub format_version: u32,
    pub name: String,
    pub description: String,
    pub nodes: Vec<NodeInstance>,
    pub connections: Vec<Connection>,
    #[serde(default)]
    pub comments: Vec<GraphComment>,
    /// Groups of nodes shown as single nodes; each is placed by a `section:<id>` node.
    #[serde(default)]
    pub sections: Vec<GraphSection>,
}

impl Default for GraphAsset {
    fn default() -> Self {
        Self {
            format_version: FORMAT_VERSION,
            name: String::new(),
            description: String::new(),
            nodes: Vec::new(),
            connections: Vec::new(),
            comments: Vec::new(),
            sections: Vec::new(),
        }
    }
}

impl GraphAsset {
    pub fn node(&self, id: &str) -> Option<&NodeInstance> {
        self.nodes.iter().find(|node| node.id == id)
    }

    /// The section a `section:<id>` node stands for.
    pub fn section_of(&self, node: &NodeInstance) -> Option<&GraphSection> {
        let id = node.kind.strip_prefix(SECTION_PREFIX)?;
        self.sections.iter().find(|section| section.id == id)
    }

    /// An id not used by any node, `prefix` followed by a number.
    pub fn fresh_id(&self, prefix: &str) -> String {
        (1..)
            .map(|n| format!("{prefix}{n}"))
            .find(|id| self.node(id).is_none())
            .expect("an unbounded range yields an unused id")
    }

    pub fn from_json(text: &str) -> Result<Self, String> {
        let asset: Self =
            serde_json::from_str(text).map_err(|e| format!("invalid render graph: {e}"))?;
        if asset.format_version > FORMAT_VERSION {
            return Err(format!(
                "render graph format {} is newer than this build supports ({FORMAT_VERSION})",
                asset.format_version
            ));
        }
        Ok(asset)
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("a graph asset is always serializable")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_and_rejects_newer_formats() {
        let mut asset = GraphAsset { name: "x".into(), ..Default::default() };
        asset.nodes.push(NodeInstance {
            id: "n1".into(),
            kind: "core.output".into(),
            params: Params::new(),
            position: [1.0, 2.0],
        });
        assert_eq!(GraphAsset::from_json(&asset.to_json()).unwrap(), asset);
        assert!(GraphAsset::from_json(r#"{"format_version": 99}"#).is_err());
        assert_eq!(asset.fresh_id("n"), "n2");
    }
}
