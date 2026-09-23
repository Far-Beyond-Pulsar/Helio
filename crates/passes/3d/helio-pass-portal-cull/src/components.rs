//! SceneDB-owned portal GPU projections.

use pulsar_scenedb_derive::SceneStore;

/// SceneDB-authored peer reference for a portal.
///
/// `Entity` is a packed generational handle, but it is not itself a GPU/POD
/// type. Keeping its raw bits in the mirrored row preserves the authoritative
/// peer relationship and lets the future resolver reconstruct the validated
/// SceneDB handle before generating portal-view and chain projections.
pub const NO_PORTAL_PEER: u64 = u64::MAX;

/// Authored portal relationship and local aperture data.
///
/// This is intentionally distinct from [`PortalViewComponent`] and
/// [`PortalChainComponent`]. Those rows are derived GPU projections consumed by
/// the current render passes; this row is the persistent SceneDB-facing source
/// of truth. A portal can point to a peer in the same sublevel, another
/// sublevel, or a nested actor context without storing a sublevel index of its
/// own.
#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "portals")]
pub struct PortalComponent {
    #[gpu]
    pub peer: u64,
    #[gpu]
    pub flags: u32,
    #[gpu]
    pub _pad: u32,
    #[gpu]
    pub transform: [[f32; 4]; 4],
    #[gpu]
    pub half_extent: [f32; 2],
}

impl PortalComponent {
    /// Bit in [`Self::flags`] that enables this authored portal.
    pub const FLAG_ENABLED: u32 = 1 << 0;

    pub fn new(
        peer: Option<pulsar_scenedb::Entity>,
        transform: glam::Mat4,
        half_extent: [f32; 2],
    ) -> Self {
        Self {
            peer: peer.map_or(NO_PORTAL_PEER, pulsar_scenedb::Entity::bits),
            flags: Self::FLAG_ENABLED,
            _pad: 0,
            transform: transform.to_cols_array_2d(),
            half_extent,
        }
    }

    pub fn peer_entity(&self) -> Option<pulsar_scenedb::Entity> {
        (self.peer != NO_PORTAL_PEER).then(|| pulsar_scenedb::Entity::from_bits(self.peer))
    }

    pub fn transform(&self) -> glam::Mat4 {
        glam::Mat4::from_cols_array_2d(&self.transform)
    }

    pub fn is_enabled(&self) -> bool {
        self.flags & Self::FLAG_ENABLED != 0
    }
}

#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "portal_views")]
pub struct PortalViewComponent {
    #[gpu]
    pub transform: [f32; 16],
    #[gpu]
    pub inverse_transform: [f32; 16],
    #[gpu]
    pub half_extent: [f32; 2],
    #[gpu]
    pub coordinate_space: u32,
    /// Reserved compatibility word. Zero means the row is a physical portal
    /// for legacy manually-authored rows. The projection bridge sets
    /// [`Self::FLAG_MASK_HIDDEN`] on virtual nested rows so they can still be
    /// referenced by a chain without being stamped as another opening.
    #[gpu]
    pub _pad: u32,
}

impl PortalViewComponent {
    /// The row is a nested projection-only portal surface, not a physical
    /// opening that should contribute to the screen-space portal mask.
    pub const FLAG_MASK_HIDDEN: u32 = 1 << 0;
}

/// One derived portal chain row.
///
/// The chain is deliberately a variable-length SceneDB field. SceneDB mirrors
/// `portals` through one growable `u32` payload pool and one growable
/// `VarLenHandle` table; the renderer binds both buffers and uses the handle's
/// `count` as the runtime recursion depth. This is the single representation
/// for every depth, rather than a family of fixed `[u32; N]` component types.
#[derive(SceneStore, Clone, Debug, PartialEq)]
pub struct PortalChainComponent {
    #[gpu(buffer = "PortalChainComponent::portals")]
    pub portals: Vec<u32>,
}

/// Derived frame counts for the portal projection tables.
///
/// The authored [`PortalComponent`] remains the source of truth. This row is
/// published by the projection bridge alongside the dense derived view and
/// chain rows so a renderer integration can distinguish active rows from a
/// growable SceneDB buffer's reserved capacity without changing the authored
/// data model.
#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "portal_projection_counts")]
pub struct PortalProjectionCountsComponent {
    #[gpu]
    pub portal_view_count: u32,
    #[gpu]
    pub portal_chain_count: u32,
    #[gpu]
    pub coordinate_space_count: u32,
    #[gpu]
    pub _pad: u32,
}

impl From<crate::GpuPortalView> for PortalViewComponent {
    fn from(v: crate::GpuPortalView) -> Self {
        bytemuck::cast(v)
    }
}
impl From<PortalViewComponent> for crate::GpuPortalView {
    fn from(v: PortalViewComponent) -> Self {
        bytemuck::cast(v)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::Zeroable;
    #[test]
    fn scene_records_preserve_gpu_abi() {
        assert_eq!(std::mem::size_of::<PortalComponent>(), 88);
        assert_eq!(std::mem::size_of::<PortalViewComponent>(), 144);
        assert_eq!(std::mem::size_of::<PortalProjectionCountsComponent>(), 16);
    }

    #[test]
    fn authored_portal_round_trips_peer_handle() {
        let mut world = pulsar_scenedb::World::new();
        let portal = world.spawn();
        let peer = world.spawn();
        let authored = PortalComponent::new(Some(peer), glam::Mat4::IDENTITY, [1.0, 2.0]);
        world.insert(portal, authored);

        let stored = world.get::<PortalComponent>(portal).unwrap();
        assert_eq!(stored.peer_entity(), Some(peer));
        assert_eq!(stored.half_extent, [1.0, 2.0]);
    }

    #[test]
    fn scene_rows_support_insert_mutate_remove() {
        let mut world = pulsar_scenedb::World::new();
        let entity = world.spawn();
        let view = PortalViewComponent::zeroed();
        let chain = PortalChainComponent {
            portals: vec![entity.index()],
        };
        world.insert(entity, view);
        world.insert(entity, chain.clone());

        world
            .get_mut::<PortalViewComponent>(entity)
            .unwrap()
            .half_extent = [2.0, 3.0];
        assert_eq!(
            world
                .get::<PortalViewComponent>(entity)
                .unwrap()
                .half_extent,
            [2.0, 3.0]
        );
        assert_eq!(
            world.remove::<PortalViewComponent>(entity),
            Some(PortalViewComponent {
                half_extent: [2.0, 3.0],
                ..view
            })
        );
        assert_eq!(world.remove::<PortalChainComponent>(entity), Some(chain));
    }
}
