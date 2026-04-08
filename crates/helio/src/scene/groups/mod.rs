//! Group management for batch visibility control and transforms.
//!
//! Groups allow you to control visibility and apply transforms to multiple objects
//! at once. Objects can belong to multiple groups simultaneously via a bitmask.
//!
//! # Group Semantics
//!
//! - Objects can be members of 0 to 64 groups (via [`GroupMask`](crate::groups::GroupMask))
//! - An object is **hidden** if **any** of its groups are currently hidden
//! - Ungrouped objects ([`GroupMask::NONE`](crate::groups::GroupMask::NONE)) are **always visible**
//!
//! # Module Organization
//!
//! - [`membership`]: Group membership management (add/remove objects to/from groups)
//! - [`visibility`]: Group visibility control (hide/show groups)
//! - [`transforms`]: Group transform operations (move/translate entire groups)
//!
//! # Example
//!
//! ```ignore
//! use helio::groups::{GroupId, GroupMask};
//!
//! // Create objects in different groups
//! let group_enemies = GroupId(0);
//! let group_ui = GroupId(1);
//!
//! let enemy_id = scene.insert_object(ObjectDescriptor {
//!     groups: GroupMask::from_id(group_enemies),
//!     ..desc
//! })?;
//!
//! // Hide all enemies
//! scene.hide_group(group_enemies);
//!
//! // Move all UI elements
//! scene.translate_group(group_ui, Vec3::new(10.0, 0.0, 0.0));
//! ```

mod membership;
mod transforms;
mod visibility;

