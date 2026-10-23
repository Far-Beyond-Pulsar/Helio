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

/// A render graph asset (`.rendergraph`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GraphAsset {
    pub format_version: u32,
    pub name: String,
    pub description: String,
    pub nodes: Vec<NodeInstance>,
    pub connections: Vec<Connection>,
}

impl Default for GraphAsset {
    fn default() -> Self {
        Self {
            format_version: FORMAT_VERSION,
            name: String::new(),
            description: String::new(),
            nodes: Vec::new(),
            connections: Vec::new(),
        }
    }
}

impl GraphAsset {
    pub fn node(&self, id: &str) -> Option<&NodeInstance> {
        self.nodes.iter().find(|node| node.id == id)
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
