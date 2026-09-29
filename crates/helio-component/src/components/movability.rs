//! The authored mobility of a mesh or light (Pulsar-Native#837).
//!
//! Editor-facing twin of [`helio::Movability`]: a reflected enum the
//! properties panel shows as a dropdown, with each variant's doc comment as
//! its tooltip. The renderer never reads this field directly; the scene
//! bridge projects it into the SceneDB [`helio::Movability`] component and
//! the object row's movable flag, which is what passes key their caches on.

use pulsar_reflection::Reflectable;
use serde::{Deserialize, Deserializer, Serialize};

#[derive(Clone, Copy, Debug, Default, Serialize, PartialEq, Eq, Hash, Reflectable)]
pub enum ObjectMovability {
    /// Never moves or changes. Cheapest: drawn into the cached static shadow
    /// atlas and its ray-tracing geometry is never re-read. Moving it
    /// anyway leaves its old shadow behind until the static cache rebuilds.
    #[default]
    Static,
    /// Never moves, but a light's colour or intensity may change. Cached
    /// like Static, and still receives shadows from movable objects.
    Stationary,
    /// May move every frame. Its shadow is redrawn into the dynamic shadow
    /// atlas each frame, which costs more than Static.
    Movable,
    /// May move and deform every frame. Most expensive: everything derived
    /// from its geometry (ray-tracing BLAS) is re-checked on every change.
    Dynamic,
}

impl ObjectMovability {
    pub const ALL: [Self; 4] = [Self::Static, Self::Stationary, Self::Movable, Self::Dynamic];

    pub fn name(self) -> &'static str {
        match self {
            Self::Static => "Static",
            Self::Stationary => "Stationary",
            Self::Movable => "Movable",
            Self::Dynamic => "Dynamic",
        }
    }

    pub fn from_index(index: u64) -> Option<Self> {
        Self::ALL.get(index as usize).copied()
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|m| m.name().eq_ignore_ascii_case(name))
    }

    /// Whether its transform is promised never to change.
    pub fn is_fixed(self) -> bool {
        !helio::Movability::from(self).can_move()
    }
}

impl From<ObjectMovability> for helio::Movability {
    fn from(value: ObjectMovability) -> Self {
        match value {
            ObjectMovability::Static => Self::Static,
            ObjectMovability::Stationary => Self::Stationary,
            ObjectMovability::Movable => Self::Movable,
            ObjectMovability::Dynamic => Self::Dynamic,
        }
    }
}

/// Accepts the variant name (serde's form) and the variant index (the
/// reflection JSON form); levels contain both, as they do for `LightType`.
impl<'de> Deserialize<'de> for ObjectMovability {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Repr {
            Name(String),
            Index(u64),
        }
        let parsed = match Repr::deserialize(deserializer)? {
            Repr::Name(name) => Self::from_name(&name),
            Repr::Index(index) => Self::from_index(index),
        };
        parsed.ok_or_else(|| serde::de::Error::custom("unknown movability"))
    }
}

#[cfg(test)]
mod tests {
    use super::ObjectMovability;

    #[test]
    fn reads_names_and_indices() {
        let by_name: ObjectMovability = serde_json::from_value(serde_json::json!("Movable")).unwrap();
        let by_index: ObjectMovability = serde_json::from_value(serde_json::json!(3)).unwrap();
        assert_eq!(by_name, ObjectMovability::Movable);
        assert_eq!(by_index, ObjectMovability::Dynamic);
        assert_eq!(serde_json::to_value(ObjectMovability::Static).unwrap(), "Static");
    }

    #[test]
    fn only_movable_modes_may_move() {
        assert!(ObjectMovability::Static.is_fixed());
        assert!(ObjectMovability::Stationary.is_fixed());
        assert!(!ObjectMovability::Movable.is_fixed());
        assert!(!ObjectMovability::Dynamic.is_fixed());
    }
}
