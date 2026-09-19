//! Contract test for Tier 1 texture-object aliasing.

use helio_core::graph::{GraphTexturePool, TextureDescriptor};

async fn request_test_adapter(instance: &wgpu::Instance) -> Option<wgpu::Adapter> {
    instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: true,
            ..Default::default()
        })
        .await
        .ok()
}

#[test]
fn released_compatible_aliases_reuse_one_physical_texture() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let Some(adapter) = request_test_adapter(&instance).await else {
            eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: texture aliasing");
            return;
        };
        let (device, _queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Texture Aliasing Test Device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .expect("available adapter must create a device");

        let descriptor = |name: &str| TextureDescriptor {
            name: name.to_owned(),
            format: wgpu::TextureFormat::Rgba8Unorm,
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
            mip_level_count: 1,
            sample_count: 1,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            alias_group: Some("non_overlapping".to_owned()),
        };
        let mut pool = GraphTexturePool::new();
        pool.allocate(&device, descriptor("first"));
        let first_id = pool
            .allocation_id("first")
            .expect("first allocation exists");
        assert_eq!(pool.physical_allocation_count(), 1);

        pool.release("first");
        pool.allocate(&device, descriptor("second"));

        assert_eq!(pool.resource_count(), 2);
        assert_eq!(pool.physical_allocation_count(), 1);
        assert_eq!(pool.allocation_id("second"), Some(first_id));
    });
}
