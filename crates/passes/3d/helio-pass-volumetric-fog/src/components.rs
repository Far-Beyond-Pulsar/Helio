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

/// World-space participating medium. One world unit is one metre. Overlaps add.
/// `extinction` is sigma_a + sigma_s in m^-1, `albedo` is sigma_s / sigma_t,
/// and `emission` is scene-linear radiance emitted per metre (independent of density).
/// Mode: 0 uniform, 1 exponential height, 2 animated smoke with a height envelope.
/// Zeroed/deleted rows are inert. No camera-distance gates apply to these media.
#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "global_fog_media")]
pub struct GlobalFogComponent {
    #[gpu]
    pub enabled: u32,
    #[gpu]
    pub mode: u32,
    #[gpu]
    pub extinction: f32,
    #[gpu]
    pub height_falloff: f32,
    #[gpu]
    pub height: f32,
    #[gpu]
    pub anisotropy: f32,
    #[gpu]
    pub _pad: [f32; 2],
    #[gpu]
    pub albedo: [f32; 3],
    #[gpu]
    pub _pad_albedo: f32,
    #[gpu]
    pub emission: [f32; 3],
    #[gpu]
    pub _pad_emission: f32,
}

impl Default for GlobalFogComponent {
    fn default() -> Self {
        Self {
            enabled: 1,
            mode: 0,
            extinction: 0.02,
            height_falloff: 0.0,
            height: 0.0,
            anisotropy: 0.0,
            _pad: [0.0; 2],
            albedo: [1.0; 3],
            _pad_albedo: 0.0,
            emission: [0.0; 3],
            _pad_emission: 0.0,
        }
    }
}

/// Axis-aligned world volume. `medium` stores raw bytes in a reflection-supported
/// float array (not numeric float conversions), with the GlobalFogComponent layout;
/// use `new` / `set_medium` instead of packing words manually. `edge_fade` is an
/// inward fade distance in metres; zero gives a hard boundary. No PP priority.
#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "local_fog_media")]
pub struct LocalFogVolumeComponent {
    #[gpu]
    pub bounds_min: [f32; 4],
    #[gpu]
    pub bounds_max: [f32; 4],
    #[gpu]
    pub medium: [f32; 16],
    #[gpu]
    pub edge_fade: f32,
    #[gpu]
    pub _pad: [f32; 3],
}

impl LocalFogVolumeComponent {
    pub fn new(bounds_min: [f32; 3], bounds_max: [f32; 3], medium: GlobalFogComponent) -> Self {
        Self {
            bounds_min: [bounds_min[0], bounds_min[1], bounds_min[2], 0.0],
            bounds_max: [bounds_max[0], bounds_max[1], bounds_max[2], 0.0],
            medium: bytemuck::cast(medium),
            edge_fade: 0.0,
            _pad: [0.0; 3],
        }
    }
    pub fn set_medium(&mut self, medium: GlobalFogComponent) {
        self.medium = bytemuck::cast(medium);
    }
    pub fn medium(&self) -> GlobalFogComponent {
        bytemuck::cast(self.medium)
    }
}

/// GPU-resolved per-view scalability. Match `view_id` to the u32 bits in
/// Camera.jitter_frame.w; u32::MAX is a fallback for every view. Exact matches
/// beat fallbacks, and the lowest entity row wins ties. `active=0` is a tombstone;
/// `enabled=0` on an active row explicitly disables rendering for that view.
/// Quality 0: 96x54x64; quality 1: 192x108x128 (aspect adjusted within budget).
/// Increase `history_epoch` for a camera cut or explicit history invalidation.
#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "volumetric_fog_settings")]
pub struct VolumetricFogSettingsComponent {
    #[gpu]
    pub active: u32,
    #[gpu]
    pub view_id: u32,
    #[gpu]
    pub quality: u32,
    #[gpu]
    pub enabled: u32,
    #[gpu]
    pub max_distance: f32,
    #[gpu]
    pub light_max_distance: f32,
    #[gpu]
    pub temporal_blend: f32,
    #[gpu]
    /// Relative density change at which history starts to be discarded (a
    /// medium appearing or vanishing). Lighting changes never reject: samples
    /// are jittered, so they differ by design and the blend integrates them.
    pub history_rejection: f32,
    #[gpu]
    pub light_samples: u32,
    #[gpu]
    pub history_epoch: u32,
    #[gpu]
    pub _pad: [f32; 2],
}

impl Default for VolumetricFogSettingsComponent {
    fn default() -> Self {
        Self {
            active: 1,
            view_id: u32::MAX,
            quality: 0,
            enabled: 1,
            max_distance: 1000.0,
            light_max_distance: 1000.0,
            temporal_blend: 0.05,
            history_rejection: 0.8,
            light_samples: 0,
            history_epoch: 0,
            _pad: [0.0; 2],
        }
    }
}

// SceneDB's packed rows need explicit removal callbacks in the pinned revision.
macro_rules! tombstone {
    ($ty:ty, $packed:ty, $clear:ident) => {
        pulsar_reflection::inventory::submit! {
            pulsar_scenedb::gpu::world_mirror::VarLenReleaseRegistration {
                component_id: pulsar_scenedb::component::component_id::<$ty>, release: $clear,
            }
        }
        fn $clear(mirror: &GpuMirrorHandle, row: u32) {
            let zero: $ty = bytemuck::Zeroable::zeroed();
            mirror.store().mark_gpu_row_dirty(
                pulsar_scenedb::component::component_id::<$packed>(),
                row,
                bytemuck::bytes_of(&zero),
            );
        }
    };
}
tombstone!(
    GlobalFogComponent,
    __ScenedbGpuPacked_GlobalFogComponent,
    clear_global_fog
);
tombstone!(
    LocalFogVolumeComponent,
    __ScenedbGpuPacked_LocalFogVolumeComponent,
    clear_local_fog
);
tombstone!(
    VolumetricFogSettingsComponent,
    __ScenedbGpuPacked_VolumetricFogSettingsComponent,
    clear_fog_settings
);
tombstone!(
    FogComponent,
    __ScenedbGpuPacked_FogComponent,
    clear_legacy_fog
);

/// Persistent fog settings.  The layout intentionally matches
/// `helio_pass_postprocess::GpuFogUniforms` exactly.
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

impl From<helio_pass_postprocess::GpuFogUniforms> for FogComponent {
    fn from(value: helio_pass_postprocess::GpuFogUniforms) -> Self {
        bytemuck::cast(value)
    }
}

impl From<FogComponent> for helio_pass_postprocess::GpuFogUniforms {
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
    fn world_component_layouts_and_local_medium_round_trip() {
        assert_eq!(std::mem::size_of::<GlobalFogComponent>(), 64);
        assert_eq!(std::mem::size_of::<LocalFogVolumeComponent>(), 112);
        assert_eq!(std::mem::offset_of!(LocalFogVolumeComponent, medium), 32);
        assert_eq!(std::mem::offset_of!(GlobalFogComponent, albedo), 32);
        assert_eq!(std::mem::offset_of!(GlobalFogComponent, emission), 48);
        assert_eq!(std::mem::size_of::<VolumetricFogSettingsComponent>(), 48);
        let mut medium = GlobalFogComponent {
            mode: 2,
            anisotropy: -0.8,
            emission: [1.0, 2.0, 3.0],
            ..Default::default()
        };
        let mut local = LocalFogVolumeComponent::new([-1.0; 3], [1.0; 3], medium);
        assert_eq!(local.medium(), medium);
        medium.extinction = 4.0;
        local.set_medium(medium);
        assert_eq!(local.medium(), medium);
    }

    #[test]
    fn tier_dimensions_preserve_aspect_and_bound_allocations() {
        assert_eq!(crate::froxel_dimensions(1920, 1080, 0), [96, 54, 64]);
        assert_eq!(crate::froxel_dimensions(1920, 1080, 1), [192, 108, 128]);
        for (w, h) in [
            (0, 0),
            (1080, 1920),
            (1920, 800),
            (2560, 1600),
            (1, u32::MAX),
            (u32::MAX, 1),
        ] {
            let high = crate::froxel_dimensions(w, h, 1);
            let low = crate::froxel_dimensions(w, h, 0);
            assert!(high[0] <= 192 && high[1] <= 108);
            assert!(low.iter().all(|v| *v > 0));
            assert_eq!(low.map(|v| v * 2), high);
        }
    }

    #[test]
    fn fog_component_matches_the_helio_uniform_abi() {
        assert_eq!(std::mem::size_of::<FogComponent>(), 64);
        assert_eq!(
            std::mem::size_of::<FogComponent>(),
            std::mem::size_of::<helio_pass_postprocess::GpuFogUniforms>()
        );
        assert_eq!(FogComponent::default().fog_max_distance, 10000.0);
    }

    #[test]
    fn fog_component_round_trips_without_a_scene_lock() {
        let source = helio_pass_postprocess::GpuFogUniforms {
            fog_enabled: 1,
            fog_mode: 1,
            fog_density: 0.4,
            ..Default::default()
        };
        let round_trip: helio_pass_postprocess::GpuFogUniforms = FogComponent::from(source).into();
        assert_eq!(round_trip.fog_enabled, 1);
        assert_eq!(round_trip.fog_density, 0.4);
    }
}
