//! SceneDB-owned foliage authoring records.

use helio_foliage_core::{GpuFoliageLayer, GpuFoliageType};
use pulsar_scenedb_derive::SceneStore;

#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "foliage_types")]
pub struct FoliageTypeComponent {
    #[gpu]
    pub density: f32,
    #[gpu]
    pub height_range: [f32; 2],
    #[gpu]
    pub width_range: [f32; 2],
    #[gpu]
    pub slope_range: [f32; 2],
    #[gpu]
    pub altitude_range: [f32; 2],
    #[gpu]
    pub lod_distances: [f32; 4],
    #[gpu]
    pub wind_response: [f32; 3],
    #[gpu]
    pub interaction_stiffness: f32,
    #[gpu]
    pub material_id: u32,
    #[gpu]
    pub density_layer: u32,
    #[gpu]
    pub kind_and_flags: u32,
    #[gpu]
    pub mesh_or_impostor_id: u32,
    #[gpu]
    pub _pad: [u32; 3],
}

#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "foliage_layers")]
pub struct FoliageLayerComponent {
    #[gpu]
    pub bounds_min: [f32; 4],
    #[gpu]
    pub bounds_max: [f32; 4],
}

#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "foliage_interactors")]
pub struct FoliageInteractorComponent {
    #[gpu]
    pub position_radius: [f32; 4],
    #[gpu]
    pub velocity: [f32; 4],
}

#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "foliage_wind")]
pub struct FoliageWindComponent {
    #[gpu]
    pub direction_speed: [f32; 4],
    #[gpu]
    pub gust: [f32; 4],
    #[gpu]
    pub time_prev_time: [f32; 2],
    #[gpu]
    pub _pad: [f32; 2],
}

impl From<GpuFoliageType> for FoliageTypeComponent {
    fn from(v: GpuFoliageType) -> Self {
        bytemuck::cast(v)
    }
}
impl From<FoliageTypeComponent> for GpuFoliageType {
    fn from(v: FoliageTypeComponent) -> Self {
        bytemuck::cast(v)
    }
}
impl From<GpuFoliageLayer> for FoliageLayerComponent {
    fn from(v: GpuFoliageLayer) -> Self {
        bytemuck::cast(v)
    }
}
impl From<FoliageLayerComponent> for GpuFoliageLayer {
    fn from(v: FoliageLayerComponent) -> Self {
        bytemuck::cast(v)
    }
}
impl From<libhelio::GpuWind> for FoliageWindComponent {
    fn from(v: libhelio::GpuWind) -> Self {
        bytemuck::cast(v)
    }
}
impl From<FoliageWindComponent> for libhelio::GpuWind {
    fn from(v: FoliageWindComponent) -> Self {
        bytemuck::cast(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scene_records_preserve_gpu_abi() {
        assert_eq!(std::mem::size_of::<FoliageTypeComponent>(), 96);
        assert_eq!(std::mem::size_of::<FoliageLayerComponent>(), 32);
        assert_eq!(std::mem::size_of::<FoliageInteractorComponent>(), 32);
        assert_eq!(std::mem::size_of::<FoliageWindComponent>(), 48);
    }
}
