//! The specialized pass consumes opaque SceneDB buffers through Helio's
//! generic pass API. This compile/runtime contract intentionally does not add
//! any voxel types or branches to helio-core.

use helio_core::{BufferKey, PassContext, RenderPass, Result as HelioResult};

struct OpaqueBufferConsumer {
    key: BufferKey,
}

impl RenderPass for OpaqueBufferConsumer {
    fn name(&self) -> &'static str {
        "OpaqueBufferConsumer"
    }

    fn writes(&self) -> &'static [&'static str] {
        &["opaque_consumer_output"]
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        // A specialized pass declares/resolves a generic SceneDB key and
        // interprets its buffer locally. The central renderer sees only the
        // opaque key/handle and does not need the layout or semantic type.
        let _current_frame_buffer = ctx.scene_buffers.get(self.key);
        Ok(())
    }
}

#[test]
fn specialized_pass_uses_only_the_generic_scene_buffer_and_pass_contract() {
    let pass = OpaqueBufferConsumer {
        key: BufferKey::of("example_component_data"),
    };
    assert_eq!(pass.name(), "OpaqueBufferConsumer");
    assert_eq!(pass.writes(), &["opaque_consumer_output"]);
}
