//! Non-Euclidean portals — seamless, real-time-rendered openings between two
//! regions of the world.
//!
//! See `docs/portals_and_sublevels.md` for the full design and
//! [`super::secondary_views`] for the per-frame eye construction and
//! composite publish. This module only owns the CPU-side portal-pair
//! registry: creation, pose updates, and removal.
//!
//! Portals take the *whole* visible scene as their draw set (no membership
//! mask, unlike sublevels) — see the design doc §5 — so, unlike
//! [`super::sublevels`], there is no group/flag bookkeeping here.

use glam::{Vec2, Vec3};
use helio_portal_core::PortalPose;

use crate::handles::PortalId;

use super::core::Scene;
use super::errors::{invalid, Result};

/// Describes a new portal pair.
#[derive(Debug, Clone, Copy)]
pub struct PortalDescriptor {
    /// The surface the player looks through.
    pub a: PortalPose,
    /// The surface the player sees (and arrives at, on crossing `a`).
    pub b: PortalPose,
    /// Half-extent (local X, Y) of the portal's rectangular opening, used for
    /// crossing detection (`helio_portal_core::crossing_detected`) and the
    /// on-screen quad's world-space corners.
    pub half_extent: Vec2,
    /// Application-defined tag, mirroring [`super::types::ObjectDescriptor::user_tag`].
    pub user_tag: u64,
}

pub(in crate::scene) struct PortalRecord {
    pub a: PortalPose,
    pub b: PortalPose,
    pub half_extent: Vec2,
    #[allow(dead_code)]
    pub user_tag: u64,
}

impl Scene {
    /// Register a portal pair.
    ///
    /// Slot allocation happens per frame in
    /// [`Scene::update_secondary_views`](super::secondary_views) (portal
    /// visibility, unlike sublevel activity, can change every frame as the
    /// camera moves), not here — `add_portal` only registers the pair.
    pub fn add_portal(&mut self, desc: PortalDescriptor) -> PortalId {
        let (id, _) = self.portals.insert(PortalRecord {
            a: desc.a,
            b: desc.b,
            half_extent: desc.half_extent,
            user_tag: desc.user_tag,
        });
        self.secondary_dirty = true;
        id
    }

    /// Update a portal pair's poses (e.g. a moving portal surface).
    pub fn update_portal_pose(&mut self, id: PortalId, a: PortalPose, b: PortalPose) -> Result<()> {
        let record = self.portals.get_mut(id).ok_or_else(|| invalid("portal"))?;
        record.a = a;
        record.b = b;
        self.secondary_dirty = true;
        Ok(())
    }

    /// Current poses of a portal pair.
    pub fn portal_poses(&self, id: PortalId) -> Result<(PortalPose, PortalPose)> {
        self.portals
            .get(id)
            .map(|r| (r.a, r.b))
            .ok_or_else(|| invalid("portal"))
    }

    /// Remove a portal pair.
    pub fn remove_portal(&mut self, id: PortalId) -> Result<()> {
        self.portals.remove(id).ok_or_else(|| invalid("portal"))?;
        self.secondary_dirty = true;
        Ok(())
    }

    /// Every registered portal's `a` surface the ray (`origin`, unit `dir`)
    /// crosses at a positive distance, as `(pair, t)` — `t` is the distance
    /// from `origin` along `dir` to the crossing point.
    ///
    /// Used by `ScenePicker::cast_ray_through_portals` to implement
    /// pick-through-portal (design doc §9): a ray that would cross a portal
    /// surface before hitting anything solid continues on the other side,
    /// mapped by `pair.map_point`/`map_direction` — "the same chain the
    /// camera uses to teleport" (§7.4), depth-capped at 1 to match this
    /// pass's portal recursion scope cut.
    pub fn portal_ray_crossings(&self, origin: Vec3, dir: Vec3) -> Vec<(helio_portal_core::PortalPair, f32)> {
        self.portals
            .iter()
            .filter_map(|(_, record)| {
                let n = record.a.forward();
                let denom = dir.dot(n);
                if denom.abs() < 1e-6 {
                    return None; // parallel to the portal plane
                }
                let t = (record.a.position() - origin).dot(n) / denom;
                if t <= 1e-4 {
                    return None; // behind the ray, or touching the origin
                }
                let hit = origin + dir * t;
                let local = record.a.transform.inverse().transform_point3(hit);
                if local.x.abs() > record.half_extent.x || local.y.abs() > record.half_extent.y {
                    return None; // outside the portal's rectangular opening
                }
                Some((helio_portal_core::PortalPair { a: record.a, b: record.b }, t))
            })
            .collect()
    }
}
