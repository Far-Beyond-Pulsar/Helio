//! SceneDB-owned decal records. Visible lists and screen-space work are
//! transient state owned by this pass.
use pulsar_scenedb::gpu::{BufferHandle, BufferKey, GpuMirrorHandle};
use pulsar_scenedb_derive::SceneStore;
use std::marker::PhantomData;

#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "decals")]
pub struct DecalComponent {
    #[gpu]
    pub transform: [f32; 16],
    #[gpu]
    pub color: [f32; 4],
    #[gpu]
    pub albedo_texture_index: u32,
    #[gpu]
    pub normal_texture_index: u32,
    #[gpu]
    pub roughness_texture_index: u32,
    #[gpu]
    pub metalness_texture_index: u32,
    #[gpu]
    pub blend_mode: u32,
    #[gpu]
    pub decal_type: u32,
    #[gpu]
    pub fade_time: f32,
    #[gpu]
    pub fade_start_delay: f32,
    #[gpu]
    pub age: f32,
    #[gpu]
    pub normal_adapt: u32,
    #[gpu]
    pub _pad0: f32,
    #[gpu]
    pub _pad1: f32,
}

impl From<libhelio::GpuDecal> for DecalComponent {
    fn from(v: libhelio::GpuDecal) -> Self {
        bytemuck::cast(v)
    }
}
impl From<DecalComponent> for libhelio::GpuDecal {
    fn from(v: DecalComponent) -> Self {
        bytemuck::cast(v)
    }
}

#[derive(Clone)]
pub struct DecalSceneBinding {
    handle: BufferHandle,
    _marker: PhantomData<DecalComponent>,
}
impl DecalSceneBinding {
    pub fn resolve(mirror: &GpuMirrorHandle) -> Option<Self> {
        mirror
            .store()
            .resolve_buffer_handle(BufferKey::of("decals"))
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

#[cfg(test)]
mod tests {
    use super::DecalComponent;
    #[test]
    fn layout_matches_gpu_abi() {
        assert_eq!(
            std::mem::size_of::<DecalComponent>(),
            std::mem::size_of::<libhelio::GpuDecal>()
        );
    }
}
#[cfg(test)]
mod lifecycle_tests {
    use super::DecalComponent;
    #[test]
    fn scene_row_lifecycle_is_entity_owned() {
        let mut world = pulsar_scenedb::World::new();
        let entity = world.spawn();
        let value = DecalComponent {
            transform: [0.0; 16],
            color: [1.0; 4],
            albedo_texture_index: u32::MAX,
            normal_texture_index: u32::MAX,
            roughness_texture_index: u32::MAX,
            metalness_texture_index: u32::MAX,
            blend_mode: 0,
            decal_type: 3,
            fade_time: 0.0,
            fade_start_delay: 0.0,
            age: 0.0,
            normal_adapt: 1,
            _pad0: 0.0,
            _pad1: 0.0,
        };
        world.insert(entity, value);
        assert_eq!(world.get::<DecalComponent>(entity), Some(&value));
        assert_eq!(world.remove::<DecalComponent>(entity), Some(value));
        assert!(world.get::<DecalComponent>(entity).is_none());
    }
}
