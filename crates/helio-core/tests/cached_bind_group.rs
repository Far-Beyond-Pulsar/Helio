//! `CachedBindGroup` reuses a group while its layout and bound resources are
//! the same handles, and rebuilds when either changes.

use helio_core::CachedBindGroup;

fn device() -> Option<wgpu::Device> {
    pollster::block_on(async {
        let instance = wgpu::Instance::default();
        let adapter = instance.request_adapter(&Default::default()).await.ok()?;
        let (device, _queue) = adapter.request_device(&Default::default()).await.ok()?;
        Some(device)
    })
}

fn layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

fn buffer(device: &wgpu::Device) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: false,
    })
}

fn group(
    cache: &mut CachedBindGroup,
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    cache
        .get_or_create(
            device,
            &wgpu::BindGroupDescriptor {
                label: None,
                layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            },
        )
        .clone()
}

#[test]
fn reused_until_a_bound_resource_or_the_layout_changes() {
    let Some(device) = device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let layout_a = layout(&device);
    let buffer_a = buffer(&device);
    let mut cache = CachedBindGroup::new();
    assert!(cache.get().is_none());

    let first = group(&mut cache, &device, &layout_a, &buffer_a);
    let again = group(&mut cache, &device, &layout_a, &buffer_a);
    assert!(first == again, "same layout and buffer: reused");

    let buffer_b = buffer(&device);
    let rebuilt = group(&mut cache, &device, &layout_a, &buffer_b);
    assert!(rebuilt != again, "new buffer: rebuilt");
    assert!(group(&mut cache, &device, &layout_a, &buffer_b) == rebuilt);

    let layout_b = layout(&device);
    let relaid = group(&mut cache, &device, &layout_b, &buffer_b);
    assert!(relaid != rebuilt, "new layout: rebuilt");

    cache.clear();
    assert!(cache.get().is_none());
    assert!(group(&mut cache, &device, &layout_b, &buffer_b) != relaid, "cleared: rebuilt");
}
