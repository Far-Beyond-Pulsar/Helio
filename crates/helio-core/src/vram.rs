//! Who owns Helio's video memory.
//!
//! Every GPU allocation is named by its owner in the device's allocator
//! report through wgpu's allocation owners (`wgpu::wgt::allocation_owner`, a
//! Helio patch of the engine's wgpu): the code that runs for an owner opens a
//! scope, and whatever it allocates on that thread is the owner's. Helio opens
//! them where its work starts, so no pass names an owner itself:
//!
//! - building a graph node: `Helio/<node kind>` (the node's passes are built
//!   here and remember it, see [`crate::RenderGraph::add_pass`]);
//! - a pass's `prepare` and `execute`: the owner its pass was built under;
//! - a scene derivation: `Helio/<derivation>`;
//! - the graph's targets, which it allocates and aliases for all passes:
//!   [`GRAPH_TARGETS`];
//! - everything else the renderer does (its own targets, uploads, the
//!   profiler, the texture table): [`HELIO`].
//!
//! Owner names live as long as the process ([`intern`]): opening a scope on a
//! frame allocates nothing.

use std::collections::HashSet;
use std::sync::Mutex;

pub use wgpu::wgt::allocation_owner::{current, root_scope, scope, OwnerScope};

/// The owner of what the renderer allocates outside any pass or derivation.
pub const HELIO: &str = "Helio";

/// The owner of the render graph's targets (the G-buffer, history, the
/// resources passes declare): the graph allocates and aliases them for all
/// passes, so no one pass owns them.
pub const GRAPH_TARGETS: &str = "Helio/graph targets";

/// `name` as a string that lives as long as the process; equal names share
/// one. Owner names are few (node kinds, derivations), so the set stays small.
pub fn intern(name: &str) -> &'static str {
    static NAMES: Mutex<Option<HashSet<&'static str>>> = Mutex::new(None);
    let mut names = NAMES.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let names = names.get_or_insert_with(HashSet::new);
    if let Some(known) = names.get(name) {
        return known;
    }
    let leaked: &'static str = Box::leak(name.to_owned().into_boxed_str());
    names.insert(leaked);
    leaked
}

/// The scope for work done for the renderer itself.
pub fn helio() -> OwnerScope {
    root_scope(HELIO)
}

/// The scope a graph node's passes are built in: `Helio/<kind>`.
pub fn node(kind: &str) -> OwnerScope {
    root_scope(intern(&format!("{HELIO}/{kind}")))
}

/// The scope a scene derivation runs in: `Helio/<name>`, no allocation.
pub fn derivation(name: &'static str) -> [OwnerScope; 2] {
    [root_scope(HELIO), scope(name)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_shared_and_scopes_name_helio() {
        assert!(std::ptr::eq(intern("a/b"), intern("a/b")));
        {
            let _node = node("shadow");
            assert_eq!(current().as_deref(), Some("Helio/shadow"));
        }
        let _derivation = derivation("SceneJoin");
        assert_eq!(current().as_deref(), Some("Helio/SceneJoin"));
    }
}
