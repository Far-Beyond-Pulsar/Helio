//! Portals: a pair of poses whose `pair_map_inverse` is one more coordinate
//! space, used to draw a clipped duplicate of nearby geometry.
//!
//! Unlike a sublevel, a portal tags no member objects. What's visible through
//! a portal is decided fresh every frame by `helio-pass-portal-cull`: it
//! frustum-tests every draw-call group's bounds *mapped through the portal's
//! `pair_map_inverse`* against the main camera, exactly like the ordinary
//! frustum cull already does for world space — content that isn't actually
//! near the portal's other side maps nowhere near the camera and is rejected
//! by that same test. Survivors are drawn a second time by
//! `helio-pass-portal-instances`, through the *same* main camera (no separate
//! "eye" — see `docs/` for why that's mathematically equivalent to one), and
//! clipped in the fragment shader to the portal's opening.
//!
//! The CPU-side pose algebra (`pair_map`, crossing detection, teleport) is
//! `helio_portal_core`, re-exported here — it is unaffected by any of the
//! above and unchanged from the design's first version.

use glam::Vec2;
use helio_portal_core::{PortalPair, PortalPose};

use crate::handles::PortalId;
use crate::scene::errors::{invalid, Result, SceneError};
use crate::scene::Scene;

/// Configuration for [`Scene::add_portal`].
#[derive(Debug, Clone, Copy)]
pub struct PortalDescriptor {
    /// The "near" surface — the one the viewer looks through.
    pub a: PortalPose,
    /// The "far" surface — what's actually near `b` is what gets duplicated
    /// and drawn as if seen through `a`.
    pub b: PortalPose,
    /// Half-extent of the portal opening in `a`'s local X/Y — the fragment
    /// clip test's bounds. Content mapped outside this (or behind the
    /// surface) is discarded rather than drawn.
    pub half_extent: Vec2,
}

/// Internal record for a portal.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PortalRecord {
    pub pair: PortalPair,
    pub half_extent: Vec2,
    /// GPU coordinate-space slot holding `pair.pair_map_inverse()`. Stable
    /// for the portal's lifetime.
    pub coordinate_space: u32,
}

impl Scene {
    /// Create a portal: allocate a coordinate-space slot and write
    /// `pair_map_inverse()` (B's frame → A's frame) into it.
    ///
    /// # Errors
    /// [`SceneError::CoordinateSpaceCapacityExceeded`] if all coordinate-space
    /// slots are already claimed by other sublevels/portals.
    pub fn add_portal(&mut self, desc: PortalDescriptor) -> Result<PortalId> {
        let slot = self
            .alloc_coordinate_space()
            .ok_or(SceneError::CoordinateSpaceCapacityExceeded)?;
        let pair = PortalPair { a: desc.a, b: desc.b };
        self.gpu_scene
            .coordinate_spaces
            .update_slot(slot, pair.pair_map_inverse().to_cols_array());

        let record = PortalRecord {
            pair,
            half_extent: desc.half_extent,
            coordinate_space: slot,
        };
        let (id, _slot_index, _fresh) = self.portals.insert(record);
        self.republish_portal_views();
        // Adding a portal changes which chains exist (a new leaf appears at
        // every depth) — pose updates don't need this, only set changes.
        self.republish_portal_chains();
        Ok(id)
    }

    /// Update a portal's pose pair. **O(1)** — one coordinate-space matrix
    /// write, no scene object is touched.
    pub fn update_portal_pose(&mut self, portal: PortalId, a: PortalPose, b: PortalPose) -> Result<()> {
        let (_slot_index, record) = self
            .portals
            .get_mut_with_slot(portal)
            .ok_or_else(|| invalid("portal"))?;
        record.pair = PortalPair { a, b };
        let (space, inverse) = (record.coordinate_space, record.pair.pair_map_inverse());
        self.gpu_scene
            .coordinate_spaces
            .update_slot(space, inverse.to_cols_array());
        self.republish_portal_views();
        Ok(())
    }

    /// Update a portal's clip half-extent (the opening's size in `a`'s local X/Y).
    pub fn update_portal_half_extent(&mut self, portal: PortalId, half_extent: Vec2) -> Result<()> {
        let (_slot_index, record) = self
            .portals
            .get_mut_with_slot(portal)
            .ok_or_else(|| invalid("portal"))?;
        record.half_extent = half_extent;
        self.republish_portal_views();
        Ok(())
    }

    /// Current pose pair for a portal — e.g. to run
    /// [`helio_portal_core::crossing_detected`] against either surface, or
    /// `pair.teleport_ray(..)` to carry a camera/player through.
    pub fn portal_pair(&self, portal: PortalId) -> Option<PortalPair> {
        self.portals.get(portal).map(|r| r.pair)
    }

    /// Current clip half-extent for a portal.
    pub fn portal_half_extent(&self, portal: PortalId) -> Option<Vec2> {
        self.portals.get(portal).map(|r| r.half_extent)
    }

    /// Remove a portal and free its GPU coordinate-space slot for reuse.
    pub fn remove_portal(&mut self, portal: PortalId) -> Result<()> {
        let (_slot_index, record) = self
            .portals
            .remove(portal)
            .ok_or_else(|| invalid("portal"))?;
        self.free_coordinate_space(record.coordinate_space);
        self.republish_portal_views();
        self.republish_portal_chains();
        Ok(())
    }

    /// Rewrites the GPU-facing portal view list from scratch.
    ///
    /// Portal counts are always small (a handful at most), so republishing
    /// the whole list on every add/remove/pose-update is simpler than
    /// dirty-tracking it and costs nothing measurable — this is not a
    /// per-frame cost, only a per-edit one.
    fn republish_portal_views(&mut self) {
        let views: Vec<libhelio::GpuPortalView> = (0..self.portals.slot_len())
            .filter_map(|slot| self.portals.get_by_slot(slot))
            .map(|record| libhelio::GpuPortalView {
                transform: record.pair.a.transform.to_cols_array(),
                inverse_transform: record.pair.a.transform.inverse().to_cols_array(),
                half_extent: record.half_extent.to_array(),
                coordinate_space: record.coordinate_space,
                _pad: 0,
            })
            .collect();
        self.gpu_scene.portal_views.set_data(views);
    }

    /// Rebuilds every valid portal chain from scratch — this is what makes
    /// portals reflect each other recursively with zero manual authoring.
    /// See `libhelio::GpuPortalChain`'s docs for the shape and
    /// `helio-pass-portal-cull`/`helio-pass-portal-instances` for how the
    /// list gets consumed. Only called on add/remove (the set of valid
    /// index sequences depends only on *how many* portals exist, not where
    /// they are — pose updates don't touch this).
    ///
    /// Generates every sequence of active-portal indices, length
    /// `1..=MAX_CHAIN_DEPTH`, **allowing repeats** — `[P, P, P]` (the same
    /// portal three times over) is exactly what makes a single mirror-style
    /// portal, or a room whose portal faces its own reflection, read as
    /// infinite. A plain depth-first walk of the "append any portal" tree,
    /// stopping once `MAX_PORTAL_CHAINS` is hit (see that constant's docs —
    /// scenes are expected to stay well under it).
    fn republish_portal_chains(&mut self) {
        let records: Vec<PortalRecord> = (0..self.portals.slot_len())
            .filter_map(|slot| self.portals.get_by_slot(slot))
            .copied()
            .collect();

        let mut chains = Vec::new();
        if !records.is_empty() {
            let mut prefix = Vec::with_capacity(libhelio::MAX_CHAIN_DEPTH);
            generate_chains(&records, &mut prefix, &mut chains);
        }
        self.gpu_scene.portal_chains.set_data(chains);
    }
}

/// Depth-first: append `chains` with every non-empty prefix reachable by
/// picking `0..records.len()` at each of up to `MAX_CHAIN_DEPTH` steps,
/// pruned to steps where the next portal is actually reachable through the
/// current innermost one — see `portal_reachable_through`.
fn generate_chains(records: &[PortalRecord], prefix: &mut Vec<u32>, chains: &mut Vec<libhelio::GpuPortalChain>) {
    if chains.len() >= libhelio::MAX_PORTAL_CHAINS {
        return;
    }
    if !prefix.is_empty() {
        let mut portals = [0u32; libhelio::MAX_CHAIN_DEPTH];
        portals[..prefix.len()].copy_from_slice(prefix);
        chains.push(libhelio::GpuPortalChain { portals, depth: prefix.len() as u32 });
    }
    if prefix.len() >= libhelio::MAX_CHAIN_DEPTH {
        return;
    }
    for p in 0..records.len() as u32 {
        if chains.len() >= libhelio::MAX_PORTAL_CHAINS {
            return;
        }
        // Only extend the chain with `p` as the new innermost portal when
        // its own opening is plausibly reachable through whichever portal
        // is currently innermost (`prefix`'s last entry) — i.e. `p.a` sits
        // inside that portal's `b` window. Without this, chain generation
        // combinatorially pairs up completely unrelated portals (any two
        // portals leading to two unrelated, non-recursive rooms, say), and
        // the per-instance cull/clip tests can't fully catch the result:
        // the outermost stage is deliberately loose on X/Y (see
        // `gbuffer_portal.wgsl`'s module doc — wide content behind a
        // narrow opening needs to still show), so a nonsense multi-hop
        // transform can still slip through the outer portal's mask as
        // faint, wrongly-positioned "ghost" content. A scene where a
        // portal genuinely IS reachable through the previous one —
        // `portal_cube`'s opposite-wall doors, or a portal paired with
        // itself — passes this check fine and keeps recursing exactly as
        // before; a scene of unrelated single-hop portals (`portal_rooms`)
        // now simply never generates the bogus deeper chains at all.
        if let Some(&prev_idx) = prefix.last() {
            if !portal_reachable_through(&records[prev_idx as usize], &records[p as usize]) {
                continue;
            }
        }
        prefix.push(p);
        generate_chains(records, prefix, chains);
        prefix.pop();
    }
}

/// True when `next`'s real opening (`next.a`) plausibly sits within `prev`'s
/// far surface (`prev.b`)'s window — i.e. composing `prev` after `next` in a
/// chain corresponds to an actual physical adjacency, not an arbitrary
/// pairing of unrelated portals. Position-only (orientation isn't checked):
/// two surfaces occupying the same physical opening but facing opposite
/// ways — exactly `prev.b` and the opposite-facing portal whose `a` sits at
/// that same spot, as in `portal_cube` — is precisely the legitimate case
/// this needs to keep allowing.
fn portal_reachable_through(prev: &PortalRecord, next: &PortalRecord) -> bool {
    let local = prev.pair.b.transform.inverse().transform_point3(next.pair.a.position());
    let tolerance = prev.half_extent.x.max(prev.half_extent.y);
    local.x.abs() <= prev.half_extent.x && local.y.abs() <= prev.half_extent.y && local.z.abs() <= tolerance
}

/// Placement helper: build a rigid [`PortalPose`] at `position` looking toward
/// `forward` (its outward normal), with `up` as the surface's up vector.
///
/// Thin convenience wrapper — [`PortalPose::from_look_at`] takes a look-at
/// *target*, not a direction; this is the more natural call shape for placing
/// a portal in a level (`portal_pose_facing(pos, Vec3::Z, Vec3::Y)` rather
/// than having to compute `pos + forward` at every call site).
pub fn portal_pose_facing(position: glam::Vec3, forward: glam::Vec3, up: glam::Vec3) -> PortalPose {
    PortalPose::from_look_at(position, position + forward, up)
}
