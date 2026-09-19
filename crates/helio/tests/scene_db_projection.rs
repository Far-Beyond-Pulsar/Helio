//! Compile-time contract for the renderer/SceneDB boundary.
//!
//! The renderer receives only SceneDB's cloneable GPU projection. Keeping this
//! test intentionally device-free makes it useful on every CI target.

use helio::SceneDbHandle;

fn assert_read_side_projection<T: Clone + Send + Sync + 'static>() {}

#[test]
fn scene_db_handle_is_a_lockless_gpu_projection() {
    assert_read_side_projection::<SceneDbHandle>();
}
