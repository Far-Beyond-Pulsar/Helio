//! What flows between render-graph nodes, and how each kind is drawn.
//!
//! A pin's [`PinType`] says what kind of GPU object it carries and whether it
//! survives from one frame to the next. Both decide its shape, so a graph reads
//! at a glance: a ringed pin is state that outlives the frame.

use serde::{Deserialize, Serialize};

/// The kind of object a pin carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PinKind {
    /// Pass order only: no data, "run after".
    Flow,
    /// A colour texture.
    Texture,
    /// A depth texture.
    Depth,
    /// A GPU buffer.
    Buffer,
    /// Frame inputs the host supplies (camera, SceneDB columns, environment).
    Scene,
}

/// Outline drawn for a pin.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PinShape {
    /// Right-pointing arrow, like an execution pin.
    Arrow,
    Circle,
    Diamond,
    Square,
    Hexagon,
}

/// A pin's type: its kind and whether the data persists across frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PinType {
    pub kind: PinKind,
    /// Kept between frames and across graph rebuilds (history textures,
    /// accumulation buffers) instead of recycled each frame.
    #[serde(default)]
    pub persistent: bool,
}

impl PinType {
    pub const FLOW: Self = Self::new(PinKind::Flow, false);
    pub const TEXTURE: Self = Self::new(PinKind::Texture, false);
    pub const DEPTH: Self = Self::new(PinKind::Depth, false);
    pub const BUFFER: Self = Self::new(PinKind::Buffer, false);
    pub const SCENE: Self = Self::new(PinKind::Scene, false);
    pub const PERSISTENT_TEXTURE: Self = Self::new(PinKind::Texture, true);
    pub const PERSISTENT_BUFFER: Self = Self::new(PinKind::Buffer, true);

    pub const fn new(kind: PinKind, persistent: bool) -> Self {
        Self { kind, persistent }
    }

    /// The short tag the type is written with in a buffer list (`tex`, `ptex`,
    /// `depth`, `buf`, `pbuf`, `scene`); `None` for flow, which is no buffer.
    pub fn prefix(self) -> Option<&'static str> {
        Some(match (self.kind, self.persistent) {
            (PinKind::Flow, _) => return None,
            (PinKind::Texture, false) => "tex",
            (PinKind::Texture, true) => "ptex",
            (PinKind::Depth, _) => "depth",
            (PinKind::Buffer, false) => "buf",
            (PinKind::Buffer, true) => "pbuf",
            (PinKind::Scene, _) => "scene",
        })
    }

    /// The inverse of [`prefix`](Self::prefix).
    pub fn from_prefix(prefix: &str) -> Option<Self> {
        Some(match prefix {
            "tex" => Self::TEXTURE,
            "ptex" => Self::PERSISTENT_TEXTURE,
            "depth" => Self::DEPTH,
            "buf" => Self::BUFFER,
            "pbuf" => Self::PERSISTENT_BUFFER,
            "scene" => Self::SCENE,
            _ => return None,
        })
    }

    pub fn shape(self) -> PinShape {
        match self.kind {
            PinKind::Flow => PinShape::Arrow,
            PinKind::Texture => PinShape::Circle,
            PinKind::Depth => PinShape::Diamond,
            PinKind::Buffer => PinShape::Square,
            PinKind::Scene => PinShape::Hexagon,
        }
    }

    /// Drawn with a second outline: the data outlives the frame.
    pub fn ringed(self) -> bool {
        self.persistent && self.kind != PinKind::Flow
    }

    /// RGB used for the pin and its wire.
    pub fn color(self) -> [u8; 3] {
        match self.kind {
            PinKind::Flow => [0xE8, 0xE8, 0xE8],
            PinKind::Texture => [0x4F, 0xC3, 0xF7],
            PinKind::Depth => [0xB3, 0x9D, 0xDB],
            PinKind::Buffer => [0xFF, 0xB7, 0x4D],
            PinKind::Scene => [0x81, 0xC7, 0x84],
        }
    }
}

/// A pin on a node type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PinDesc {
    /// Id within the node; unique per direction.
    pub name: String,
    pub ty: PinType,
    /// The Helio resource name this pin binds, empty for [`PinKind::Flow`].
    /// Helio resolves resources by name, so two pins connect only when they
    /// carry the same name.
    pub resource: String,
    /// An input that reads the value from the previous frame (temporal feedback:
    /// Hi-Z reading last frame's depth). It connects to the resource's producer
    /// like any input, but does not order the graph, so it cannot form a cycle.
    pub history: bool,
    /// An input nothing in the graph has to supply: the pass binds a fallback
    /// when it is absent. Unconnected, it is not a problem.
    pub optional: bool,
}

impl PinDesc {
    pub fn flow(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ty: PinType::FLOW,
            resource: String::new(),
            history: false,
            optional: false,
        }
    }

    /// Marks this input as reading the previous frame's value.
    pub fn history(mut self) -> Self {
        self.history = true;
        self
    }

    /// Marks this input as optional.
    pub fn optional(mut self) -> Self {
        self.optional = true;
        self
    }

    /// A resource pin named after the resource it binds.
    pub fn resource(ty: PinType, resource: impl Into<String>) -> Self {
        let resource = resource.into();
        Self {
            name: resource.clone(),
            ty,
            resource,
            history: false,
            optional: false,
        }
    }

    /// Whether a wire from `self` (an output) to `to` (an input) is legal.
    pub fn connects_to(&self, to: &PinDesc) -> bool {
        self.ty == to.ty && self.resource == to.resource
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes_follow_kind_and_rings_follow_persistence() {
        assert_eq!(PinType::FLOW.shape(), PinShape::Arrow);
        assert_eq!(PinType::TEXTURE.shape(), PinShape::Circle);
        assert_eq!(PinType::DEPTH.shape(), PinShape::Diamond);
        assert_eq!(PinType::BUFFER.shape(), PinShape::Square);
        assert_eq!(PinType::SCENE.shape(), PinShape::Hexagon);
        assert!(PinType::PERSISTENT_TEXTURE.ringed());
        assert!(PinType::PERSISTENT_BUFFER.ringed());
        assert!(!PinType::TEXTURE.ringed());
        assert!(!PinType::new(PinKind::Flow, true).ringed());
    }

    #[test]
    fn pins_connect_on_type_and_resource_name() {
        let a = PinDesc::resource(PinType::TEXTURE, "pre_aa");
        assert!(a.connects_to(&PinDesc::resource(PinType::TEXTURE, "pre_aa")));
        assert!(!a.connects_to(&PinDesc::resource(PinType::TEXTURE, "ssao")));
        assert!(!a.connects_to(&PinDesc::resource(PinType::PERSISTENT_TEXTURE, "pre_aa")));
        assert!(PinDesc::flow("out").connects_to(&PinDesc::flow("in")));
    }
}
