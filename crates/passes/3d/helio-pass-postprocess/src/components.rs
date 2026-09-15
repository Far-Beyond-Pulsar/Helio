//! SceneDB-owned post-process volume records.
//!
//! One coarse editor-facing `PostProcessVolumeComponent` (bounds, priority,
//! blend weight, and every exposure/tonemap/fog/color-grading knob) authors
//! exactly one packed row here -- there is no separate "settings" component,
//! the whole editor property set is this one GPU-facing struct, field for
//! field identical to `libhelio::GpuPostProcessVolume`'s ABI.
use pulsar_scenedb_derive::SceneStore;

#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "post_process_volumes")]
pub struct PostProcessVolumeComponent {
    #[gpu]
    pub bounds_min: [f32; 4],
    #[gpu]
    pub bounds_max: [f32; 4],
    #[gpu]
    pub priority: f32,
    #[gpu]
    pub blend_radius: f32,
    /// Zero means "inactive" -- `postprocess.wgsl`'s `cs_volume_blend`
    /// skips a row with `blend_weight <= 0.0` before ever evaluating its
    /// (otherwise degenerate, all-zero) bounds, which is what makes an
    /// unused row safely inert. See `MAX_PP_VOLUMES`'s doc (helio-pass-
    /// postprocess) for the fixed-capacity iteration this enables.
    #[gpu]
    pub blend_weight: f32,
    #[gpu]
    pub unbound: u32,
    #[gpu]
    pub _pad: [f32; 4],
    /// `libhelio::GpuPostProcessUniforms`'s raw bytes, as `u32`s rather than
    /// the struct itself: SceneDB's packed-layout derive requires every
    /// `#[gpu]` field to implement its own `pulsar_scenedb::Pod` (a
    /// different trait than `bytemuck::Pod`, which is all
    /// `GpuPostProcessUniforms` implements), but the array blanket impl
    /// covers this. `bytemuck::cast` between the two OUTER structs below
    /// doesn't care about this field's internal shape, only that the two
    /// structs are the same total size -- see the `layout_matches_gpu_abi`
    /// test.
    #[gpu]
    pub settings: [u32; 116],
}
const _: () = assert!(
    std::mem::size_of::<[u32; 116]>() == std::mem::size_of::<libhelio::GpuPostProcessUniforms>()
);

impl From<libhelio::GpuPostProcessVolume> for PostProcessVolumeComponent {
    fn from(v: libhelio::GpuPostProcessVolume) -> Self {
        bytemuck::cast(v)
    }
}
impl From<PostProcessVolumeComponent> for libhelio::GpuPostProcessVolume {
    fn from(v: PostProcessVolumeComponent) -> Self {
        bytemuck::cast(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn layout_matches_gpu_abi() {
        // `bytemuck::cast` is a by-value reinterpret (like `mem::transmute`):
        // it requires equal size, not equal alignment, so only size is
        // checked here -- see `settings`'s field doc for why the two
        // structs' internal shapes differ (nested struct vs. flat `u32`s).
        assert_eq!(
            std::mem::size_of::<PostProcessVolumeComponent>(),
            std::mem::size_of::<libhelio::GpuPostProcessVolume>()
        );
    }
}

#[cfg(test)]
mod lifecycle_tests {
    use super::PostProcessVolumeComponent;
    #[test]
    fn scene_row_lifecycle_is_entity_owned() {
        let mut world = pulsar_scenedb::World::new();
        let entity = world.spawn();
        let value: PostProcessVolumeComponent = bytemuck::Zeroable::zeroed();
        world.insert(entity, value);
        assert_eq!(
            world.get::<PostProcessVolumeComponent>(entity),
            Some(&value)
        );
        assert_eq!(
            world.remove::<PostProcessVolumeComponent>(entity),
            Some(value)
        );
        assert!(world.get::<PostProcessVolumeComponent>(entity).is_none());
    }
}
