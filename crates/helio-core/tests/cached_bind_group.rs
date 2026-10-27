//! `CachedBindGroup` reuses a group while its layout and bound resources are
//! the same handles, and rebuilds when either changes.

use helio_core::CachedBindGroup;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct CountedAllocator;
thread_local! {
    static ALLOCATIONS: Cell<Option<usize>> = const { Cell::new(None) };
}

fn count_allocation() {
    let _ = ALLOCATIONS.try_with(|count| {
        if let Some(value) = count.get() {
            count.set(Some(value + 1));
        }
    });
}

unsafe impl GlobalAlloc for CountedAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count_allocation();
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count_allocation();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        count_allocation();
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountedAllocator = CountedAllocator;

fn device() -> Option<wgpu::Device> {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(helio_core::wgpu_instance_descriptor());
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

    // Count only this thread, after device creation and the first cache miss.
    // A steady hit must not allocate an owned resource key or a new group.
    ALLOCATIONS.with(|count| count.set(Some(0)));
    for _ in 0..1_000 {
        let _ = group(&mut cache, &device, &layout_a, &buffer_a);
    }
    let allocations = ALLOCATIONS.with(|count| count.replace(None).unwrap());
    assert_eq!(
        allocations, 0,
        "warmed bind-group cache hits allocate nothing"
    );

    let buffer_b = buffer(&device);
    let rebuilt = group(&mut cache, &device, &layout_a, &buffer_b);
    assert!(rebuilt != again, "new buffer: rebuilt");
    assert!(group(&mut cache, &device, &layout_a, &buffer_b) == rebuilt);

    let layout_b = layout(&device);
    let relaid = group(&mut cache, &device, &layout_b, &buffer_b);
    assert!(relaid != rebuilt, "new layout: rebuilt");

    cache.clear();
    assert!(cache.get().is_none());
    assert!(
        group(&mut cache, &device, &layout_b, &buffer_b) != relaid,
        "cleared: rebuilt"
    );
}
