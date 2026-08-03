use std::sync::Arc;

use helio_pass_voxel_mesh::VoxelMeshPass;

async fn create_pass() -> Option<VoxelMeshPass> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let mut adapter = None;
    for force_fallback_adapter in [false, true] {
        if let Ok(candidate) = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter,
                apply_limit_buckets: false,
            })
            .await
        {
            adapter = Some(candidate);
            break;
        }
    }
    let adapter = adapter?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("Voxel Mesh Portability Test Device"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            ..Default::default()
        })
        .await
        .expect("adapter must create a device");
    device.on_uncaptured_error(Arc::new(|error| {
        panic!("voxel-mesh GPU validation error: {error:?}");
    }));

    Some(VoxelMeshPass::new_composited(
        &device,
        &queue,
        wgpu::TextureFormat::Rgba8Unorm,
    ))
}

#[test]
fn pipelines_compile_on_the_available_backend() {
    let Some(_pass) = pollster::block_on(create_pass()) else {
        eprintln!("skipping voxel-mesh portability test: no GPU adapter available");
        return;
    };
}
