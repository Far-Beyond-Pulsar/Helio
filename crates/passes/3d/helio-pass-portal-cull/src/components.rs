//! SceneDB-owned portal GPU projections.

use pulsar_scenedb_derive::SceneStore;

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
    #[gpu]
    pub _pad: u32,
}

#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "portal_chains")]
pub struct PortalChainComponent {
    #[gpu]
    pub portals: [u32; 3],
    #[gpu]
    pub depth: u32,
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
impl From<crate::GpuPortalChain> for PortalChainComponent {
    fn from(v: crate::GpuPortalChain) -> Self {
        bytemuck::cast(v)
    }
}
impl From<PortalChainComponent> for crate::GpuPortalChain {
    fn from(v: PortalChainComponent) -> Self {
        bytemuck::cast(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::Zeroable;
    #[test]
    fn scene_records_preserve_gpu_abi() {
        assert_eq!(std::mem::size_of::<PortalViewComponent>(), 144);
        assert_eq!(std::mem::size_of::<PortalChainComponent>(), 16);
    }

    #[test]
    fn scene_rows_support_insert_mutate_remove() {
        let mut world = pulsar_scenedb::World::new();
        let entity = world.spawn();
        let view = PortalViewComponent::zeroed();
        let chain = PortalChainComponent {
            portals: [entity.index(), 0, 0],
            depth: 1,
        };
        world.insert(entity, view);
        world.insert(entity, chain);

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
