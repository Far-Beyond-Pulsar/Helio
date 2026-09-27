//! `SceneBufferLiveness` treats unknown contents as live, reports all-zero
//! rows as dead once the readback lands, and goes live again as soon as
//! SceneDB reports new contents.

use helio_core::{BufferHandle, SceneBufferLiveness};

fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
    pollster::block_on(adapter.request_device(&Default::default())).ok()
}

fn handle(buffer: &wgpu::Buffer, content_generation: u64) -> BufferHandle {
    BufferHandle { buffer: buffer.clone(), epoch: 0, row_bytes: 16, content_generation }
}

/// Runs frames (update + device poll) until the answer for `handle` settles.
fn settle(liveness: &mut SceneBufferLiveness, device: &wgpu::Device, queue: &wgpu::Queue, handle: &BufferHandle) {
    for _ in 0..4 {
        liveness.update(device, queue, Some(handle));
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    }
    liveness.update(device, queue, Some(handle));
}

#[test]
fn zero_rows_are_dead_and_new_contents_are_live_until_read() {
    let Some((device, queue)) = device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let rows = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut liveness = SceneBufferLiveness::default();

    let empty = handle(&rows, 1);
    assert!(liveness.maybe_live(&empty), "unknown contents must count as live");
    settle(&mut liveness, &device, &queue, &empty);
    assert!(!liveness.maybe_live(&empty), "all-zero rows are dead");

    queue.write_buffer(&rows, 32, &[1, 0, 0, 0]);
    let written = handle(&rows, 2);
    assert!(liveness.maybe_live(&written), "new contents are live until read back");
    settle(&mut liveness, &device, &queue, &written);
    assert!(liveness.maybe_live(&written), "a non-zero row is live");
}
