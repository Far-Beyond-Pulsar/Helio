/// Mobility of a scene object, following Unreal Engine's mobility model
/// (Static / Stationary / Movable) plus `Dynamic` for deforming geometry.
///
/// Mobility is a promise the author makes about what will change at runtime.
/// Passes use it to decide what they may cache; each pass defines what the
/// promise buys it. The enum names no specific pass or scene-object type.
///
/// | Variant      | Transform changes | Vertices change |
/// |--------------|-------------------|-----------------|
/// | `Static`     | no                | no              |
/// | `Stationary` | no                | no              |
/// | `Movable`    | yes               | no              |
/// | `Dynamic`    | yes               | yes             |
///
/// `Stationary` differs from `Static` only in non-geometric properties: a
/// stationary light keeps its position but may change colour or intensity and
/// still needs dynamic shadows from movable casters.
///
/// # SceneDB usage
///
/// `Movability` is itself a SceneDB component. Insert it on a mesh entity to
/// declare whether that mesh's vertices may change, and on an object entity
/// to declare whether its transform may change. A pass that reads it must
/// treat an absent component as the least restrictive behaviour it already
/// supported, so opting in never changes results, only cost.
///
/// Breaking a promise is not detected: a pass may keep using its cached copy.
/// To change a `Static` mesh's vertices, spawn a new mesh entity or mark the
/// mesh `Dynamic` first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Movability {
    /// Never moves and never deforms. Everything derived from it can be cached.
    Static = 0,
    /// Never moves and never deforms, but other properties (a light's colour
    /// or intensity) may change and it still receives dynamic shadows.
    Stationary = 1,
    /// Transform may change every frame; geometry never deforms.
    Movable = 2,
    /// Transform and geometry may both change every frame.
    Dynamic = 3,
}

impl Default for Movability {
    /// Default to Static for maximum performance.
    /// Users must explicitly opt-in to Movable for dynamic objects.
    fn default() -> Self {
        Movability::Static
    }
}

impl Movability {
    /// Returns true if this object can have its transform updated.
    pub fn can_move(self) -> bool {
        matches!(self, Movability::Movable | Movability::Dynamic)
    }

    /// Returns true if this object's geometry (vertices or indices) may change
    /// after it is first created.
    pub fn can_deform(self) -> bool {
        matches!(self, Movability::Dynamic)
    }

    /// Returns true if this object is fully static (no movement, no dynamic shadows).
    pub fn is_fully_static(self) -> bool {
        matches!(self, Movability::Static)
    }

    /// Returns true if this mobility allows dynamic shadow updates.
    pub fn allows_dynamic_shadows(self) -> bool {
        !self.is_fully_static()
    }
}

#[cfg(test)]
mod tests {
    use super::Movability;

    #[test]
    fn capabilities_follow_the_mobility_table() {
        let table = [
            (Movability::Static, false, false),
            (Movability::Stationary, false, false),
            (Movability::Movable, true, false),
            (Movability::Dynamic, true, true),
        ];
        for (mobility, moves, deforms) in table {
            assert_eq!(mobility.can_move(), moves, "{mobility:?}");
            assert_eq!(mobility.can_deform(), deforms, "{mobility:?}");
        }
    }
}
