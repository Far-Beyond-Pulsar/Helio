use pulsar_scenedb_derive::SceneStore;

/// The authored Corona emitter record. The pass owns the schema; SceneDB owns
/// its packed CPU column and GPU projection.
#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "corona_emitters")]
pub struct CoronaEmitterComponent {
    #[gpu]
    pub transform: [[f32; 4]; 4],
    #[gpu]
    pub emit_params: [f32; 4],
    #[gpu]
    pub size_params: [f32; 4],
    #[gpu]
    pub start_color: [f32; 4],
    #[gpu]
    pub end_color: [f32; 4],
    #[gpu]
    pub velocity: [f32; 4],
    #[gpu]
    pub velocity_variation: [f32; 4],
    #[gpu]
    pub extras: [f32; 4],
    #[gpu]
    pub texture_index: i32,
    #[gpu]
    pub particle_offset: u32,
    #[gpu]
    pub particle_count: u32,
    #[gpu]
    pub spawn_cursor: u32,
    #[gpu]
    pub _pad: [f32; 12],
}

impl From<libhelio::GpuCoronaEmitter> for CoronaEmitterComponent {
    fn from(value: libhelio::GpuCoronaEmitter) -> Self {
        bytemuck::cast(value)
    }
}

impl From<CoronaEmitterComponent> for libhelio::GpuCoronaEmitter {
    fn from(value: CoronaEmitterComponent) -> Self {
        bytemuck::cast(value)
    }
}

#[cfg(test)]
mod tests {
    use super::CoronaEmitterComponent;

    #[test]
    fn scene_record_matches_corona_gpu_abi() {
        assert_eq!(
            std::mem::size_of::<CoronaEmitterComponent>(),
            std::mem::size_of::<libhelio::GpuCoronaEmitter>()
        );
        assert_eq!(std::mem::align_of::<CoronaEmitterComponent>(), 4);
    }
}
