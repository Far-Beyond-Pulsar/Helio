//! The uber shader against the Rust uniform ABI, on a GPU (an
//! adapter is required; the tests fail rather than skip).

use super::*;

fn gpu() -> (wgpu::Device, wgpu::Queue) {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(helio_core::wgpu_instance_descriptor().with_env());
        let adapter = instance
            .request_adapter(&Default::default())
            .await
            .expect("GPU adapter required for post-process ABI regressions");
        adapter.request_device(&Default::default()).await.unwrap()
    })
}

/// The uber shader and the other pipelines build against the same layout.
#[test]
fn the_uber_pipeline_builds_against_the_layout() {
    let (device, queue) = gpu();
    let _pass = PostProcessPass::new_with_user_effects(
        &device,
        &queue,
        64,
        64,
        wgpu::TextureFormat::Rgba8Unorm,
        None,
    );
}
