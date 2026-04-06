/// Actor mobility mode, similar to Unreal Engine's mobility system.
///
/// Determines whether an actor can move at runtime and affects caching optimizations.
///
/// # Caching Behavior
///
/// - **Static**: Objects never move. Shadows, occlusion, and lighting are heavily cached.
///   Attempting to update transform will log a warning and no-op.
/// - **Stationary**: Objects don't move but can cast dynamic shadows (lights only).
///   Used for lights that remain fixed but need to respond to dynamic objects.
/// - **Movable**: Objects can move freely. Minimal caching, full dynamic updates.
///
/// # Performance Impact
///
/// - Static objects provide maximum caching (shadows skip rendering when scene static)
/// - Movable objects force cache invalidation each frame
/// - Stationary is a middle ground (for lights: static light pos, dynamic shadow casters)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Movability {
    /// Object cannot move at runtime. Maximum caching, transform updates will warn and no-op.
    Static = 0,
    /// Object position is static but responds to dynamic objects (lights only).
    Stationary = 1,
    /// Object can move freely. Minimal caching, full dynamic behavior.
    Movable = 2,
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
        matches!(self, Movability::Movable)
    }

    /// Returns true if this object is fully static (no movement, no dynamic shadows).
    pub fn is_fully_static(self) -> bool {
        matches!(self, Movability::Static)
    }

    /// Returns true if this mobility allows dynamic shadow updates.
    pub fn allows_dynamic_shadows(self) -> bool {
        matches!(self, Movability::Stationary | Movability::Movable)
    }
}
