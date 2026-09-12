//! SceneDB-owned water inputs. Simulation textures and interaction results are
//! transient and remain owned by the water simulation pass.
use pulsar_scenedb::gpu::{BufferHandle, BufferKey, GpuMirrorHandle};
use pulsar_scenedb_derive::SceneStore;
use std::marker::PhantomData;

#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "water_volumes")]
pub struct WaterVolumeComponent {
    #[gpu]
    pub bounds_min: [f32; 4],
    #[gpu]
    pub bounds_max: [f32; 4],
    #[gpu]
    pub wave_params: [f32; 4],
    #[gpu]
    pub wave_direction: [f32; 4],
    #[gpu]
    pub water_color: [f32; 4],
    #[gpu]
    pub extinction: [f32; 4],
    #[gpu]
    pub reflection_refraction: [f32; 4],
    #[gpu]
    pub caustics_params: [f32; 4],
    #[gpu]
    pub fog_params: [f32; 4],
    #[gpu]
    pub sim_params: [f32; 4],
    #[gpu]
    pub shadow_params: [f32; 4],
    #[gpu]
    pub sun_direction: [f32; 4],
    #[gpu]
    pub ssr_params: [f32; 4],
    #[gpu]
    pub sim_dynamics: [f32; 4],
    #[gpu]
    pub wind_params: [f32; 4],
    #[gpu]
    pub _pad6: [f32; 4],
}
#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "water_hitboxes")]
pub struct WaterHitboxComponent {
    #[gpu]
    pub old_min: [f32; 4],
    #[gpu]
    pub old_max: [f32; 4],
    #[gpu]
    pub new_min: [f32; 4],
    #[gpu]
    pub new_max: [f32; 4],
    #[gpu]
    pub params: [f32; 4],
}

macro_rules! binding {
    ($name:ident, $ty:ty, $key:literal) => {
        #[derive(Clone)]
        pub struct $name {
            handle: BufferHandle,
            _marker: PhantomData<$ty>,
        }
        impl $name {
            pub fn resolve(m: &GpuMirrorHandle) -> Option<Self> {
                m.store()
                    .resolve_buffer_handle(BufferKey::of($key))
                    .map(|handle| Self {
                        handle,
                        _marker: PhantomData,
                    })
            }
            pub fn buffer(&self) -> &wgpu::Buffer {
                &self.handle.buffer
            }
            pub fn epoch(&self) -> u64 {
                self.handle.epoch
            }
        }
    };
}
binding!(
    WaterVolumeSceneBinding,
    WaterVolumeComponent,
    "water_volumes"
);
binding!(
    WaterHitboxSceneBinding,
    WaterHitboxComponent,
    "water_hitboxes"
);
impl From<libhelio::GpuWaterVolume> for WaterVolumeComponent {
    fn from(v: libhelio::GpuWaterVolume) -> Self {
        bytemuck::cast(v)
    }
}
impl From<WaterVolumeComponent> for libhelio::GpuWaterVolume {
    fn from(v: WaterVolumeComponent) -> Self {
        bytemuck::cast(v)
    }
}
impl From<libhelio::GpuWaterHitbox> for WaterHitboxComponent {
    fn from(v: libhelio::GpuWaterHitbox) -> Self {
        bytemuck::cast(v)
    }
}
impl From<WaterHitboxComponent> for libhelio::GpuWaterHitbox {
    fn from(v: WaterHitboxComponent) -> Self {
        bytemuck::cast(v)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn layout_matches_gpu_abi() {
        assert_eq!(std::mem::size_of::<WaterVolumeComponent>(), 256);
        assert_eq!(std::mem::size_of::<WaterHitboxComponent>(), 80);
    }
}
#[cfg(test)]
mod lifecycle_tests {
    use super::{WaterHitboxComponent, WaterVolumeComponent};
    #[test]
    fn water_rows_are_independently_added_and_removed() {
        let mut world = pulsar_scenedb::World::new();
        let volume = world.spawn();
        let hitbox = world.spawn();
        let v = WaterVolumeComponent {
            bounds_min: [0.0; 4],
            bounds_max: [1.0; 4],
            wave_params: [0.0; 4],
            wave_direction: [0.0; 4],
            water_color: [0.0; 4],
            extinction: [0.0; 4],
            reflection_refraction: [0.0; 4],
            caustics_params: [0.0; 4],
            fog_params: [0.0; 4],
            sim_params: [0.0; 4],
            shadow_params: [0.0; 4],
            sun_direction: [0.0; 4],
            ssr_params: [0.0; 4],
            sim_dynamics: [0.0; 4],
            wind_params: [0.0; 4],
            _pad6: [0.0; 4],
        };
        let h = WaterHitboxComponent {
            old_min: [0.0; 4],
            old_max: [1.0; 4],
            new_min: [0.0; 4],
            new_max: [1.0; 4],
            params: [1.0, 1.0, 0.0, 0.0],
        };
        world.insert(volume, v);
        world.insert(hitbox, h);
        assert_eq!(world.remove::<WaterVolumeComponent>(volume), Some(v));
        assert!(world.get::<WaterHitboxComponent>(hitbox).is_some());
        assert_eq!(world.remove::<WaterHitboxComponent>(hitbox), Some(h));
    }
}
