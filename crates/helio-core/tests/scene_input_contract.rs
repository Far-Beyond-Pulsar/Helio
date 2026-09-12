use helio_core::{SceneBufferProjection, SceneInput};
use std::sync::{Arc, OnceLock};

fn accepts_generic_scene_input(_input: &dyn SceneInput) {}

struct FrontendProjection;

impl SceneInput for FrontendProjection {
    fn device(&self) -> &Arc<wgpu::Device> {
        todo!()
    }
    fn queue(&self) -> &Arc<wgpu::Queue> {
        todo!()
    }
    fn frame_count(&self) -> u64 {
        0
    }
    fn resources(&self) -> helio_core::SceneResources<'_> {
        todo!()
    }
    fn scene_buffers(&self) -> &SceneBufferProjection {
        static EMPTY: OnceLock<SceneBufferProjection> = OnceLock::new();
        EMPTY.get_or_init(SceneBufferProjection::empty)
    }
}

#[test]
fn generic_scene_input_is_the_only_core_contract() {
    fn assert_impl<T: SceneInput>() {}
    assert_impl::<FrontendProjection>();
    // Keep the object-safe boundary compile-checked as well.
    let _ = accepts_generic_scene_input;
}
