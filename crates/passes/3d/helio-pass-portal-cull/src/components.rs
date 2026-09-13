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

impl From<libhelio::GpuPortalView> for PortalViewComponent {
    fn from(v: libhelio::GpuPortalView) -> Self {
        bytemuck::cast(v)
    }
}
impl From<PortalViewComponent> for libhelio::GpuPortalView {
    fn from(v: PortalViewComponent) -> Self {
        bytemuck::cast(v)
    }
}
impl From<libhelio::GpuPortalChain> for PortalChainComponent {
    fn from(v: libhelio::GpuPortalChain) -> Self {
        bytemuck::cast(v)
    }
}
impl From<PortalChainComponent> for libhelio::GpuPortalChain {
    fn from(v: PortalChainComponent) -> Self {
        bytemuck::cast(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scene_records_preserve_gpu_abi() {
        assert_eq!(std::mem::size_of::<PortalViewComponent>(), 144);
        assert_eq!(std::mem::size_of::<PortalChainComponent>(), 16);
    }
}
