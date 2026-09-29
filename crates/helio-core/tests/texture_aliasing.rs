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
fn smaller_alias_has_requested_attachment_extent() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let Ok(adapter) = instance.request_adapter(&Default::default()).await else {
            eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: texture alias extents");
            return;
        };
        eprintln!("GPU_TEXTURE_ALIAS_ADAPTER {:?}", adapter.get_info());
        let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();
        let mut pool = GraphTexturePool::new();
        let desc = |name: &str, width, height| TextureDescriptor {
            name: name.into(),
            format: wgpu::TextureFormat::Rgba16Float,
            width,
            height,
            depth_or_array_layers: 1,
            mip_level_count: 1,
            sample_count: 1,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            alias_group: Some("released_hdr".into()),
        };
        pool.allocate(&device, desc("large", 192, 108));
        pool.release("large");
        pool.allocate(&device, desc("small", 96, 54));
        assert_eq!(pool.get_texture("small").unwrap().width(), 96);
        assert_eq!(pool.get_texture("small").unwrap().height(), 54);
        assert_ne!(pool.allocation_id("large"), pool.allocation_id("small"));
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 96,
                height: 54,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let depth_view = depth.create_view(&Default::default());
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("small HDR and matching depth"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: pool.get_view("small").unwrap(),
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
        }
        queue.submit([encoder.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        assert!(validation.pop().await.is_none());
        pool.release("small");
        pool.allocate(&device, desc("small_again", 96, 54));
        assert_eq!(
            pool.allocation_id("small"),
            pool.allocation_id("small_again")
        );
        assert_eq!(pool.physical_allocation_count(), 2);
    });
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
