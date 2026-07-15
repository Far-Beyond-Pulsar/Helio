use bytemuck::Pod;

/// Reads a GPU storage buffer back to the CPU synchronously.
///
/// This blocks the calling thread until the GPU has finished all previously
/// submitted work.  For offline baking this is fine; for in-editor use the
/// caller should submit work before calling this.
pub struct GpuReadback;

impl GpuReadback {
    pub fn read_buffer<T: Pod>(
        device: &wgpu::Device,
        queue:  &wgpu::Queue,
        src:    &wgpu::Buffer,
        len:    usize,
    ) -> Vec<T> {
        let byte_size = (std::mem::size_of::<T>() * len) as u64;

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("nebula_readback_staging"),
            size:               byte_size,
            usage:              wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("nebula_readback_enc"),
        });
        enc.copy_buffer_to_buffer(src, 0, &staging, 0, byte_size);
        queue.submit(std::iter::once(enc.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        rx.recv().expect("wgpu readback channel closed unexpectedly").expect("map_async failed");

        let mapped = slice
            .get_mapped_range()
            .expect("Nebula readback buffer should be mapped");
        let data: Vec<T> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        data
    }
}
