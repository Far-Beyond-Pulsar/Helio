//! SceneDB-owned records consumed by the sky pass.
//!
//! Sky is a singleton from the renderer's point of view, but it is still an
//! ordinary SceneDB component.  Frontends attach it to their environment
//! entity (the first row is used by the sky pass) and SceneDB owns both the
//! CPU column and its packed GPU projection.  The pass never reads the CPU
//! world and never takes a renderer-side lock.

use std::marker::PhantomData;

use pulsar_scenedb::gpu::{BufferHandle, BufferKey, GpuMirrorHandle};
use pulsar_scenedb_derive::SceneStore;

/// Persistent atmospheric and volumetric-cloud parameters.
///
/// The field grouping intentionally matches the WGSL `SkyComponent` record.
/// All authored fields are `#[gpu]`; changing a value through `World::get_mut`
/// therefore queues a row update which is coalesced by SceneDB's normal flush.
#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "sky_components")]
pub struct SkyComponent {
    #[gpu]
    pub sun_direction: [f32; 3],
    #[gpu]
    pub sun_intensity: f32,
    #[gpu]
    pub rayleigh_scatter: [f32; 3],
    #[gpu]
    pub rayleigh_h_scale: f32,
    #[gpu]
    pub mie_scatter: f32,
    #[gpu]
    pub mie_h_scale: f32,
    #[gpu]
    pub mie_g: f32,
    #[gpu]
    pub sun_disk_cos: f32,
    #[gpu]
    pub earth_radius: f32,
    #[gpu]
    pub atm_radius: f32,
    #[gpu]
    pub exposure: f32,
    #[gpu]
    pub clouds_enabled: u32,
    #[gpu]
    pub cloud_coverage: f32,
    #[gpu]
    pub cloud_density: f32,
    #[gpu]
    pub cloud_base: f32,
    #[gpu]
    pub cloud_top: f32,
    #[gpu]
    pub cloud_wind_x: f32,
    #[gpu]
    pub cloud_wind_z: f32,
    #[gpu]
    pub cloud_speed: f32,
    #[gpu]
    pub time_sky: f32,
    #[gpu]
    pub skylight_intensity: f32,
    #[gpu]
    pub cloud_mode: u32,
    #[gpu]
    pub cloud_quality: u32,
    #[gpu]
    pub cloud_resolution: u32,
}

/// Atmosphere and cloud settings are intentionally one SceneDB row.  Helio's
/// former `SkyActor` treated these as a singleton resource, while the scene
/// contract requires every authored value to have an entity owner.  These
/// aliases keep the domain vocabulary explicit without introducing parallel
/// storage or duplicate GPU columns.
pub type AtmosphereComponent = SkyComponent;
pub type CloudscapeComponent = SkyComponent;

impl Default for SkyComponent {
    fn default() -> Self {
        Self {
            sun_direction: [0.0, 0.9, 0.4],
            sun_intensity: 22.0,
            rayleigh_scatter: [5.8e-3, 1.35e-2, 3.31e-2],
            rayleigh_h_scale: 0.1,
            mie_scatter: 2.1e-3,
            mie_h_scale: 0.075,
            mie_g: 0.76,
            sun_disk_cos: 0.9998,
            earth_radius: 6360.0,
            atm_radius: 6420.0,
            exposure: 0.1,
            clouds_enabled: 0,
            cloud_coverage: 0.0,
            cloud_density: 0.0,
            cloud_base: 0.0,
            cloud_top: 0.0,
            cloud_wind_x: 0.0,
            cloud_wind_z: 0.0,
            cloud_speed: 0.0,
            time_sky: 0.0,
            skylight_intensity: 0.0,
            cloud_mode: 1,
            cloud_quality: 2,
            cloud_resolution: 4,
        }
    }
}

/// Re-resolvable borrowed GPU input for the sky pass.
#[derive(Clone)]
pub struct SkySceneBinding {
    handle: BufferHandle,
    _record: PhantomData<SkyComponent>,
}

impl SkySceneBinding {
    pub fn resolve(mirror: &GpuMirrorHandle) -> Option<Self> {
        mirror
            .store()
            .resolve_buffer_handle(BufferKey::of("sky_components"))
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
