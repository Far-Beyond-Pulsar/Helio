//! SceneDB-owned volumetric-fog parameters.
//!
//! Fog used to be available only as part of Helio's post-process uniform
//! resource.  That made it impossible for the scene database to own the
//! authored environment state.  This record is the persistent, packed row;
//! the froxel grids and per-frame light integration remain pass-owned
//! transient resources.

use std::marker::PhantomData;

use pulsar_scenedb::gpu::{BufferHandle, BufferKey, GpuMirrorHandle};
use pulsar_scenedb_derive::SceneStore;

/// Persistent fog settings.  The layout intentionally matches
/// `libhelio::postprocess::GpuFogUniforms` exactly.
#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "fog_components")]
pub struct FogComponent {
    #[gpu]
    pub fog_enabled: u32,
    #[gpu]
    pub fog_mode: u32,
    #[gpu]
    pub fog_density: f32,
    #[gpu]
    pub fog_height_falloff: f32,
    #[gpu]
    pub fog_start_distance: f32,
    #[gpu]
    pub fog_max_distance: f32,
    #[gpu]
    pub fog_height: f32,
    #[gpu]
    pub fog_scattering_anisotropy: f32,
    #[gpu]
    pub fog_color: [f32; 3],
    #[gpu]
    pub pad_fog_color: f32,
    #[gpu]
    pub fog_emissive: [f32; 3],
    #[gpu]
    pub pad_fog_emissive: f32,
}

impl Default for FogComponent {
    fn default() -> Self {
        Self {
            fog_enabled: 0,
            fog_mode: 0,
            fog_density: 0.02,
            fog_height_falloff: 0.0,
            fog_start_distance: 0.0,
            fog_max_distance: 10000.0,
            fog_height: 0.0,
            fog_scattering_anisotropy: 0.0,
            fog_color: [0.7, 0.8, 1.0],
            pad_fog_color: 0.0,
            fog_emissive: [0.0; 3],
            pad_fog_emissive: 0.0,
        }
    }
}

impl From<libhelio::postprocess::GpuFogUniforms> for FogComponent {
    fn from(value: libhelio::postprocess::GpuFogUniforms) -> Self {
        bytemuck::cast(value)
    }
}

impl From<FogComponent> for libhelio::postprocess::GpuFogUniforms {
    fn from(value: FogComponent) -> Self {
        bytemuck::cast(value)
    }
}

/// Read-only frame binding resolved from the SceneDB GPU mirror.
#[derive(Clone)]
pub struct FogSceneBinding {
    handle: BufferHandle,
    _record: PhantomData<FogComponent>,
}

impl FogSceneBinding {
    pub fn resolve(mirror: &GpuMirrorHandle) -> Option<Self> {
        mirror
            .store()
            .resolve_buffer_handle(BufferKey::of("fog_components"))
            .map(|handle| Self {
                handle,
                _record: PhantomData,
            })
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.handle.buffer
    }
    pub fn epoch(&self) -> u64 {
        self.handle.epoch
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fog_component_matches_the_helio_uniform_abi() {
        assert_eq!(std::mem::size_of::<FogComponent>(), 64);
        assert_eq!(
            std::mem::size_of::<FogComponent>(),
            std::mem::size_of::<libhelio::postprocess::GpuFogUniforms>()
        );
        assert_eq!(FogComponent::default().fog_max_distance, 10000.0);
    }

    #[test]
    fn fog_component_round_trips_without_a_scene_lock() {
        let source = libhelio::postprocess::GpuFogUniforms {
            fog_enabled: 1,
            fog_mode: 1,
            fog_density: 0.4,
            ..Default::default()
        };
        let round_trip: libhelio::postprocess::GpuFogUniforms = FogComponent::from(source).into();
        assert_eq!(round_trip.fog_enabled, 1);
        assert_eq!(round_trip.fog_density, 0.4);
    }
}
