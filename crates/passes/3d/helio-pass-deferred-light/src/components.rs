//! SceneDB-owned reflection capture records.
//!
//! `ReflectionCaptureComponent` is field-for-field identical to
//! `libhelio::GpuReflectionCapture`'s ABI: one placed capture volume (sphere
//! or oriented box), its parallax transform, and which cube-array layer its
//! baked (or not-yet-baked) cubemap lives in.
//!
//! # Blend-order caveat
//!
//! `deferred_lighting.wgsl`'s `sample_reflection_environment` accumulates
//! captures front-to-back and saturates, so overlapping captures are meant
//! to be fed smallest-influence-first (a small capture gets first claim on
//! a pixel, matching how a level designer expects a small "override" probe
//! nested inside a large one to win). Iterating this SceneDB buffer in raw
//! entity-index order does not guarantee that ordering -- unlike `lights`/
//! `decals`/etc, whose zero-row default is inert regardless of iteration
//! order, `ReflectionCaptureComponent` rows are semantically order-sensitive
//! when captures overlap. A single capture, or non-overlapping captures,
//! blend identically either way; only deliberately-nested capture volumes
//! may pick a different (still plausible, not incorrect-looking) winner
//! than before. Sorting this buffer by influence volume is the same class
//! of cross-entity GPU-sort work already specified (not yet implemented)
//! for batched object instances -- tracked, not silently dropped.
use pulsar_scenedb_derive::SceneStore;

#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "reflection_captures")]
pub struct ReflectionCaptureComponent {
    #[gpu]
    pub position_radius: [f32; 4],
    #[gpu]
    pub extents_transition: [f32; 4],
    #[gpu]
    pub world_to_local: [[f32; 4]; 4],
    #[gpu]
    pub cubemap_index: i32,
    #[gpu]
    pub shape: u32,
    #[gpu]
    pub mobility: u32,
    #[gpu]
    pub brightness: f32,
}

impl From<libhelio::GpuReflectionCapture> for ReflectionCaptureComponent {
    fn from(v: libhelio::GpuReflectionCapture) -> Self {
        bytemuck::cast(v)
    }
}
impl From<ReflectionCaptureComponent> for libhelio::GpuReflectionCapture {
    fn from(v: ReflectionCaptureComponent) -> Self {
        bytemuck::cast(v)
    }
}

/// Fixed capacity for the `"reflection_captures"` SceneDB buffer --
/// `deferred_lighting.wgsl` already clamped its old CPU-driven count to this
/// exact number (`min(globals.reflection_capture_count, 64u)`), so keeping
/// it at 64 changes nothing about the shader's own worst-case cost, only
/// where the row data and the iteration bound come from.
pub const MAX_REFLECTION_CAPTURES: u32 = 64;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn layout_matches_gpu_abi() {
        assert_eq!(
            std::mem::size_of::<ReflectionCaptureComponent>(),
            std::mem::size_of::<libhelio::GpuReflectionCapture>()
        );
    }
}

#[cfg(test)]
mod lifecycle_tests {
    use super::ReflectionCaptureComponent;
    #[test]
    fn scene_row_lifecycle_is_entity_owned() {
        let mut world = pulsar_scenedb::World::new();
        let entity = world.spawn();
        let value: ReflectionCaptureComponent = bytemuck::Zeroable::zeroed();
        world.insert(entity, value);
        assert_eq!(
            world.get::<ReflectionCaptureComponent>(entity),
            Some(&value)
        );
        assert_eq!(
            world.remove::<ReflectionCaptureComponent>(entity),
            Some(value)
        );
        assert!(world.get::<ReflectionCaptureComponent>(entity).is_none());
    }
}
