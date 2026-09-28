//! SceneDB-owned post-process volume records.
//!
//! One coarse editor-facing `PostProcessVolumeComponent` (bounds, priority,
//! blend weight, and every exposure/tonemap/fog/color-grading knob) authors
//! exactly one packed row here -- there is no separate "settings" component,
//! the whole editor property set is this one GPU-facing struct, field for
//! field identical to `crate::GpuPostProcessVolume`'s ABI.
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
    /// `PostProcessProperty` override bits, words 0..3 (`vec4<u32>` in WGSL).
    /// Scalars rather than `[u32; 4]`: reflection registers only specific
    /// array shapes, and this row derives reflection through `SceneStore`.
    #[gpu]
    pub override_mask_0: u32,
    #[gpu]
    pub override_mask_1: u32,
    #[gpu]
    pub override_mask_2: u32,
    #[gpu]
    pub override_mask_3: u32,
    /// `crate::GpuPostProcessUniforms`'s raw bytes, split at the lens block:
    /// reflection registers `[u32; 116]` and `[f32; 16]` but no 132-word
    /// array. `settings` is bytes 0..464 (everything before the lens block);
    /// `lens` is the 64-byte lens tail as raw bits, not numeric floats. The
    /// packed derive needs `pulsar_scenedb::Pod` per field, which
    /// `GpuPostProcessUniforms` lacks; `bytemuck::cast` between the outer
    /// structs only needs equal size -- see `layout_matches_gpu_abi`.
    #[gpu]
    pub settings: [u32; 116],
    #[gpu]
    pub lens: [f32; 16],
    /// Lens extension block (bytes 528..592), raw bits like `lens`.
    #[gpu]
    pub lens_ext: [f32; 16],
}
const _: () = assert!(
    std::mem::size_of::<[u32; 116]>() as u64 == crate::GpuPostProcessUniforms::LENS_BLOCK_OFFSET
);
const _: () = assert!(
    std::mem::size_of::<[u32; 116]>() + 2 * std::mem::size_of::<[f32; 16]>()
        == std::mem::size_of::<crate::GpuPostProcessUniforms>()
);

/// Split `GpuPostProcessUniforms` into the reflection-supported row parts:
/// bytes 0..464, the lens block 464..528 and the lens extension 528..592.
fn split_settings(settings: &crate::GpuPostProcessUniforms) -> ([u32; 116], [f32; 16], [f32; 16]) {
    let bytes = bytemuck::bytes_of(settings);
    let split = crate::GpuPostProcessUniforms::LENS_BLOCK_OFFSET as usize;
    let mut head = [0u32; 116];
    let mut lens = [0f32; 16];
    let mut lens_ext = [0f32; 16];
    bytemuck::bytes_of_mut(&mut head).copy_from_slice(&bytes[..split]);
    bytemuck::bytes_of_mut(&mut lens).copy_from_slice(&bytes[split..split + 64]);
    bytemuck::bytes_of_mut(&mut lens_ext).copy_from_slice(&bytes[split + 64..]);
    (head, lens, lens_ext)
}

fn join_settings(head: &[u32; 116], lens: &[f32; 16], lens_ext: &[f32; 16]) -> crate::GpuPostProcessUniforms {
    let mut settings: crate::GpuPostProcessUniforms = bytemuck::Zeroable::zeroed();
    let bytes = bytemuck::bytes_of_mut(&mut settings);
    let split = crate::GpuPostProcessUniforms::LENS_BLOCK_OFFSET as usize;
    bytes[..split].copy_from_slice(bytemuck::bytes_of(head));
    bytes[split..split + 64].copy_from_slice(bytemuck::bytes_of(lens));
    bytes[split + 64..].copy_from_slice(bytemuck::bytes_of(lens_ext));
    settings
}

// The pinned SceneDB version dispatches removals/despawns through this release
// registry, but only generates registrations for variable-length components.
// Packed PP rows need an explicit tombstone too: the shaders scan capacity,
// and stale blend_weight would otherwise keep a deleted volume alive forever.
pulsar_reflection::inventory::submit! {
    pulsar_scenedb::gpu::world_mirror::VarLenReleaseRegistration {
        component_id: pulsar_scenedb::component::component_id::<PostProcessVolumeComponent>,
        release: clear_volume_row,
    }
}

fn clear_volume_row(mirror: &pulsar_scenedb::gpu::GpuMirrorHandle, row: u32) {
    let zero: PostProcessVolumeComponent = bytemuck::Zeroable::zeroed();
    mirror.store().mark_gpu_row_dirty(
        pulsar_scenedb::component::component_id::<__ScenedbGpuPacked_PostProcessVolumeComponent>(),
        row,
        bytemuck::bytes_of(&zero),
    );
}

impl From<crate::GpuPostProcessVolume> for PostProcessVolumeComponent {
    fn from(v: crate::GpuPostProcessVolume) -> Self {
        bytemuck::cast(v)
    }
}
impl From<PostProcessVolumeComponent> for crate::GpuPostProcessVolume {
    fn from(v: PostProcessVolumeComponent) -> Self {
        bytemuck::cast(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn settings_split_round_trips_every_byte() {
        let mut settings = crate::PostProcessSettings::default();
        settings.lens_flare.enabled = true;
        settings.lens_flare.ghost_count = 7;
        settings.lens_flare.vignette = 0.25;
        settings.fog_density = 0.125;
        let gpu = settings.to_gpu();
        let camera = CameraPostProcessComponent::new(3, &settings);
        assert_eq!(bytemuck::bytes_of(&camera.settings()), bytemuck::bytes_of(&gpu));
        assert_eq!(camera.view_id, 3);

        let mut descriptor = crate::PostProcessVolumeDescriptor::default();
        descriptor.settings = settings;
        descriptor.override_mask = [1, 2, 3, 4];
        let row = PostProcessVolumeComponent::from(descriptor.to_gpu());
        assert_eq!([row.override_mask_0, row.override_mask_1, row.override_mask_2, row.override_mask_3], [1, 2, 3, 4]);
        let back = crate::GpuPostProcessVolume::from(row);
        assert_eq!(bytemuck::bytes_of(&back.settings), bytemuck::bytes_of(&gpu));
    }

    #[test]
    fn layout_matches_gpu_abi() {
        // `bytemuck::cast` is a by-value reinterpret (like `mem::transmute`):
        // it requires equal size, not equal alignment, so only size is
        // checked here -- see `settings`'s field doc for why the two
        // structs' internal shapes differ (nested struct vs. flat `u32`s).
        assert_eq!(
            std::mem::size_of::<PostProcessVolumeComponent>(),
            std::mem::size_of::<crate::GpuPostProcessVolume>()
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

/// Optional per-view baseline: the camera's own post-process settings,
/// including lens flare. The first enabled row matching the camera's view id
/// (u32 bits of `jitter_frame.w`) wins; with none, the resolver's project
/// defaults apply. PP volumes then override on top. Removing the component
/// clears its packed row. `settings`/`lens` split the uniforms exactly as
/// [`PostProcessVolumeComponent`] does.
#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "camera_postprocess")]
pub struct CameraPostProcessComponent {
    #[gpu]
    pub view_id: u32,
    #[gpu]
    pub enabled: u32,
    #[gpu]
    pub _pad0: u32,
    #[gpu]
    pub _pad1: u32,
    #[gpu]
    pub settings: [u32; 116],
    #[gpu]
    pub lens: [f32; 16],
    #[gpu]
    pub lens_ext: [f32; 16],
}
impl CameraPostProcessComponent {
    pub fn new(view_id: u32, settings: &crate::PostProcessSettings) -> Self {
        Self::from_gpu(view_id, &settings.to_gpu())
    }
    pub fn from_gpu(view_id: u32, settings: &crate::GpuPostProcessUniforms) -> Self {
        let (settings, lens, lens_ext) = split_settings(settings);
        Self { view_id, enabled: 1, _pad0: 0, _pad1: 0, settings, lens, lens_ext }
    }
    pub fn settings(&self) -> crate::GpuPostProcessUniforms {
        join_settings(&self.settings, &self.lens, &self.lens_ext)
    }
}
const _: () = assert!(std::mem::size_of::<CameraPostProcessComponent>() == 608);
const _: () = assert!(std::mem::offset_of!(CameraPostProcessComponent, settings) == 16);

pulsar_reflection::inventory::submit! {
    pulsar_scenedb::gpu::world_mirror::VarLenReleaseRegistration {
        component_id: pulsar_scenedb::component::component_id::<CameraPostProcessComponent>,
        release: clear_camera_row,
    }
}
fn clear_camera_row(mirror: &pulsar_scenedb::gpu::GpuMirrorHandle, row: u32) {
    let zero: CameraPostProcessComponent = bytemuck::Zeroable::zeroed();
    mirror.store().mark_gpu_row_dirty(
        pulsar_scenedb::component::component_id::<__ScenedbGpuPacked_CameraPostProcessComponent>(),
        row, bytemuck::bytes_of(&zero),
    );
}
