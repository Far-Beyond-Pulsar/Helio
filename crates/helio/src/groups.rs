//! Object group membership system.
//!
//! Any object can belong to any number of groups simultaneously.  Groups are
//! stored as a compact `u64` bitmask (`GroupMask`), supporting up to 64
//! distinct groups with zero per-object overhead beyond the 8 bytes of the
//! mask itself.
//!
//! # Workflow
//!
//! ```rust,ignore
//! // Tag all editor billboards with the EDITOR group.
//! scene.set_object_groups(billboard_id, GroupMask::from(GroupId::EDITOR));
//!
//! // Hide every editor-only object in one call.
//! scene.hide_group(GroupId::EDITOR);
//!
//! // Move every physics prop 5 units upward.
//! scene.translate_group(GroupId::PHYSICS, Vec3::new(0.0, 5.0, 0.0));
//! ```
//!
//! # Visibility semantics
//!
//! An object is **hidden** if **any** of its groups are currently hidden.
//! An object with `GroupMask::NONE` (ungrouped) is always visible regardless of
//! which groups are hidden.

/// Index of a single group — the bit position within a `GroupMask`.
///
/// Valid range: `0..=63`.  Values outside this range saturate to 63.
///
/// # Pre-defined group names
///
/// Well-known groups are exposed as associated constants so code can use
/// `GroupId::EDITOR` instead of a bare magic number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GroupId(u8);

impl GroupId {
    // ── Well-known built-in groups ────────────────────────────────────────────

    /// In-editor helpers (billboards, gizmos, grid overlays, etc.).
    /// These are hidden at runtime and should never appear in shipped builds.
    pub const EDITOR: GroupId = GroupId(0);

    /// Default group for user-created scene objects.
    pub const DEFAULT: GroupId = GroupId(1);

    /// Static world geometry (floors, walls, terrain, props that never move).
    pub const STATIC: GroupId = GroupId(2);

    /// Dynamically simulated or animated objects.
    pub const DYNAMIC: GroupId = GroupId(3);

    /// UI elements rendered in world-space (health bars, nameplates, etc.).
    pub const WORLD_UI: GroupId = GroupId(4);

    /// VFX / particle system visual objects.
    pub const VFX: GroupId = GroupId(5);

    /// Shadow-caster hint group (can be used to mass-disable shadows for props).
    pub const SHADOW_CASTERS: GroupId = GroupId(6);

    /// Logical debug visualisers (AABBs, nav-mesh overlays, etc.).
    pub const DEBUG: GroupId = GroupId(7);

    // ─────────────────────────────────────────────────────────────────────────

    /// Create a group from a raw bit index in `0..=63`.
    ///
    /// Values above 63 are silently clamped to 63.
    #[inline(always)]
    pub const fn new(index: u8) -> Self {
        if index > 63 {
            GroupId(63)
        } else {
            GroupId(index)
        }
    }

    /// The bit-index (0–63) for this group.
    #[inline(always)]
    pub const fn index(self) -> u8 {
        self.0
    }

    /// Return a `GroupMask` with only this group's bit set.
    #[inline(always)]
    pub const fn mask(self) -> GroupMask {
        GroupMask(1u64 << self.0)
    }
}

// ─── GroupMask ───────────────────────────────────────────────────────────────

/// A bitmask representing membership across up to 64 groups.
///
/// Bit *N* is set when the object belongs to `GroupId(N)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct GroupMask(pub u64);

impl GroupMask {
    /// The empty mask — not a member of any group.
    pub const NONE: GroupMask = GroupMask(0);

    /// Mask with every group set.
    pub const ALL: GroupMask = GroupMask(u64::MAX);

    /// Return the raw bit pattern.
    #[inline(always)]
    pub const fn bits(self) -> u64 {
        self.0
    }

    /// True if no groups are set.
    #[inline(always)]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Return a copy of this mask with `group`'s bit added.
    #[inline(always)]
    pub const fn with(self, group: GroupId) -> Self {
        GroupMask(self.0 | group.mask().0)
    }

    /// Return a copy of this mask with `group`'s bit removed.
    #[inline(always)]
    pub const fn without(self, group: GroupId) -> Self {
        GroupMask(self.0 & !group.mask().0)
    }

    /// True if this mask contains `group`.
    #[inline(always)]
    pub const fn contains(self, group: GroupId) -> bool {
        self.0 & group.mask().0 != 0
    }

    /// True if this mask and `other` share at least one common group.
    #[inline(always)]
    pub const fn intersects(self, other: GroupMask) -> bool {
        self.0 & other.0 != 0
    }

    /// Union of two masks.
    #[inline(always)]
    pub const fn union(self, other: GroupMask) -> Self {
        GroupMask(self.0 | other.0)
    }

    /// Intersection of two masks.
    #[inline(always)]
    pub const fn intersection(self, other: GroupMask) -> Self {
        GroupMask(self.0 & other.0)
    }
}

impl From<GroupId> for GroupMask {
    #[inline(always)]
    fn from(g: GroupId) -> Self {
        g.mask()
    }
}

impl std::ops::BitOr for GroupMask {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        GroupMask(self.0 | rhs.0)
    }
}

impl std::ops::BitAnd for GroupMask {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self {
        GroupMask(self.0 & rhs.0)
    }
}

impl std::ops::BitOrAssign for GroupMask {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl std::ops::BitAndAssign for GroupMask {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl std::ops::Not for GroupMask {
    type Output = Self;
    fn not(self) -> Self {
        GroupMask(!self.0)
    }
}

