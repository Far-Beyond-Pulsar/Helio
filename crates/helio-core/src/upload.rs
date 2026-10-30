#[cfg(debug_assertions)]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(debug_assertions)]
use std::sync::{mpsc, OnceLock};

#[cfg(debug_assertions)]
static FRAME_UPLOAD_BYTES: AtomicU64 = AtomicU64::new(0);
#[cfg(debug_assertions)]
static REPORTER: OnceLock<mpsc::Sender<u64>> = OnceLock::new();

#[cfg(debug_assertions)]
fn reporter() -> &'static mpsc::Sender<u64> {
    REPORTER.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<u64>();
        std::thread::Builder::new()
            .name("helio-upload-reporter".to_string())
            .spawn(move || {
                let mut frames_since_print = 0u64;
                let mut bytes_since_print = 0u64;
                while let Ok(frame_bytes) = rx.recv() {
                    frames_since_print += 1;
                    bytes_since_print += frame_bytes;
                    if frames_since_print >= 100 {
                        let mib = bytes_since_print as f64 / (1024.0 * 1024.0);
                        let kib_per_frame = bytes_since_print as f64 / frames_since_print as f64 / 1024.0;
                        eprintln!(
                            "[helio][debug] GPU uploads last {} frames: {:.2} MiB total ({:.1} KiB/frame avg)",
                            frames_since_print,
                            mib,
                            kib_per_frame,
                        );
                        frames_since_print = 0;
                        bytes_since_print = 0;
                    }
                }
            })
            .expect("failed to spawn helio upload reporter thread");
        tx
    })
}

pub fn record_upload_bytes(bytes: u64) {
    #[cfg(debug_assertions)]
    FRAME_UPLOAD_BYTES.fetch_add(bytes, Ordering::Relaxed);

    #[cfg(not(debug_assertions))]
    let _ = bytes;
}

pub fn write_buffer(queue: &wgpu::Queue, buffer: &wgpu::Buffer, offset: u64, data: &[u8]) {
    record_upload_bytes(data.len() as u64);
    queue.write_buffer(buffer, offset, data);
}

pub fn write_texture(
    queue: &wgpu::Queue,
    texture: wgpu::TexelCopyTextureInfo<'_>,
    data: &[u8],
    data_layout: wgpu::TexelCopyBufferLayout,
    size: wgpu::Extent3d,
) {
    record_upload_bytes(data.len() as u64);
    queue.write_texture(texture, data, data_layout, size);
}

pub fn finish_frame() {
    #[cfg(debug_assertions)]
    {
        let frame_bytes = FRAME_UPLOAD_BYTES.swap(0, Ordering::Relaxed);
        let _ = reporter().send(frame_bytes);
    }
}

/// A frame's small buffer writes as one upload: each write's bytes go into
/// one CPU arena and a copy from the upload buffer into its target is
/// recorded where the write is made (so it lands in command order, before
/// whatever is recorded after it); [`Self::flush`] sends the arena up in one
/// `Queue::write_buffer`. A `Queue::write_buffer` a write makes a wgpu
/// staging buffer for each (created, mapped and destroyed for a few bytes,
/// and each freed again in the device's maintenance); here a frame makes
/// one a flush.
///
/// The upload buffer is a ring of one frame: writes go from its start again
/// once the frame is submitted ([`Self::end_frame`]). The queue runs a later
/// submission's writes after the earlier submission's copies, so a frame's
/// bytes never overwrite those of a frame still copying; within a frame
/// writes never overlap (a frame that outgrows the buffer flushes what it
/// has and goes on in one twice the size; the old one lives while its copies
/// are in flight).
///
/// The owner flushes before the encoders it wrote into are submitted (at the
/// end of each call that records writes) and ends the frame after the
/// submit. Writes can also be deferred ([`Self::defer`]): their copies are
/// recorded together into an encoder submitted before every other in the
/// submission ([`Self::record_deferred`]), so they mean what
/// `Queue::write_buffer` means (the render graph's uploads, through
/// `PrepareContext::write_buffer` and `PassContext::write_buffer`). Like [`crate::gpu_cache::GpuCache`], a value its owner holds,
/// usable through `&self`.
pub struct FrameUploads {
    device: wgpu::Device,
    queue: wgpu::Queue,
    label: &'static str,
    ring: std::sync::Mutex<Ring>,
}

/// The upload buffer; where the frame's next write goes and where the
/// bytes not flushed yet start in it; those bytes; the writes and flushes
/// since the frame began (diagnostics).
struct Ring {
    buffer: wgpu::Buffer,
    cursor: u64,
    flushed: u64,
    bytes: Vec<u8>,
    writes: u32,
    flushes: u32,
    /// Deferred copies: their upload buffer, offset there, target, offset
    /// there, bytes.
    deferred: Vec<(wgpu::Buffer, u64, wgpu::Buffer, u64, u64)>,
}

impl Ring {
    /// Room for `size` bytes at the cursor: what the buffer holds flushed,
    /// then a buffer twice the size.
    fn reserve(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, label: &'static str, size: u64) {
        if self.cursor + size > self.buffer.size() {
            self.flush(queue);
            let grown = (self.buffer.size() * 2).max(size.next_power_of_two());
            self.buffer = FrameUploads::buffer(device, label, grown);
            self.cursor = 0;
            self.flushed = 0;
        }
    }

    /// `data` at the cursor: where it went.
    fn push(&mut self, data: &[u8]) -> u64 {
        let at = self.cursor;
        self.bytes.extend_from_slice(data);
        self.cursor += data.len() as u64;
        self.writes += 1;
        record_upload_bytes(data.len() as u64);
        at
    }

    fn flush(&mut self, queue: &wgpu::Queue) {
        if self.bytes.is_empty() {
            return;
        }
        queue.write_buffer(&self.buffer, self.flushed, &self.bytes);
        self.flushed = self.cursor;
        self.bytes.clear();
        self.flushes += 1;
    }
}

impl FrameUploads {
    /// The first upload buffer's size (grows to a frame's need).
    const FIRST_BYTES: u64 = 64 << 10;

    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, label: &'static str) -> Self {
        Self {
            ring: std::sync::Mutex::new(Ring {
                buffer: Self::buffer(device, label, Self::FIRST_BYTES),
                cursor: 0,
                flushed: 0,
                bytes: Vec::new(),
                writes: 0,
                flushes: 0,
                deferred: Vec::new(),
            }),
            device: device.clone(),
            queue: queue.clone(),
            label,
        }
    }

    fn buffer(device: &wgpu::Device, label: &'static str, size: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn ring(&self) -> std::sync::MutexGuard<'_, Ring> {
        self.ring.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// `data` written into `target` at `offset` at this point of `encoder`'s
    /// commands (as `Queue::write_buffer`: `data`'s length a multiple of 4).
    pub fn write(
        &self,
        encoder: &mut crate::CommandRecorder<'_>,
        target: &wgpu::Buffer,
        offset: u64,
        data: &[u8],
    ) {
        if data.is_empty() {
            return;
        }
        let size = data.len() as u64;
        debug_assert!(
            size.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT),
            "{} writes {size} bytes",
            self.label
        );
        let mut ring = self.ring();
        ring.reserve(&self.device, &self.queue, self.label, size);
        encoder.copy_buffer_to_buffer(&ring.buffer, ring.cursor, target, offset, size);
        ring.push(data);
    }

    /// `data` written into `target` at `offset` before the submission's
    /// other work ([`Self::record_deferred`]), as `Queue::write_buffer`.
    pub fn defer(&self, target: &wgpu::Buffer, offset: u64, data: &[u8]) {
        if data.is_empty() {
            return;
        }
        let size = data.len() as u64;
        debug_assert!(
            size.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT),
            "{} writes {size} bytes",
            self.label
        );
        let mut ring = self.ring();
        ring.reserve(&self.device, &self.queue, self.label, size);
        let source = ring.buffer.clone();
        let at = ring.push(data);
        ring.deferred.push((source, at, target.clone(), offset, size));
    }

    /// The deferred writes' copies, in one encoder to submit before the
    /// submission's others (none: no encoder), their bytes flushed.
    pub fn record_deferred(&self, device: &wgpu::Device) -> Option<wgpu::CommandBuffer> {
        let mut ring = self.ring();
        if ring.deferred.is_empty() {
            return None;
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(self.label),
        });
        for (source, at, target, offset, size) in ring.deferred.drain(..) {
            encoder.copy_buffer_to_buffer(&source, at, &target, offset, size);
        }
        ring.flush(&self.queue);
        Some(encoder.finish())
    }

    /// The bytes written since the last flush, up in one queue write (before
    /// the encoders they were written into are submitted).
    pub fn flush(&self) {
        self.ring().flush(&self.queue);
    }

    /// The frame's encoders were submitted: the next frame writes from the
    /// buffer's start. Its writes and flushes, counted.
    pub fn end_frame(&self) -> (u32, u32) {
        let mut ring = self.ring();
        debug_assert!(
            ring.bytes.is_empty() && ring.deferred.is_empty(),
            "{}: writes not flushed before the submit",
            self.label
        );
        ring.flush(&self.queue);
        ring.cursor = 0;
        ring.flushed = 0;
        (std::mem::take(&mut ring.writes), std::mem::take(&mut ring.flushes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(crate::wgpu_instance_descriptor());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
        pollster::block_on(adapter.request_device(&Default::default())).ok()
    }

    fn read(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) -> Vec<u32> {
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
        queue.submit([encoder.finish()]);
        staging.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let words = bytemuck::cast_slice(&staging.slice(..).get_mapped_range().unwrap()).to_vec();
        words
    }

    /// Writes land in command order (a later write to the same words wins,
    /// a copy recorded between them sees the first), frames reuse the
    /// buffer, and a frame larger than it grows it without losing a write.
    #[test]
    fn writes_land_in_command_order_across_frames_and_growth() {
        let Some((device, queue)) = gpu() else { return };
        let buffer = |size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let (target, between) = (buffer(1 << 20), buffer(16));
        let uploads = FrameUploads::new(&device, &queue, "test uploads");
        for frame in 0..3u32 {
            let mut encoder = device.create_command_encoder(&Default::default());
            let mut recorder = crate::CommandRecorder::from_encoder(&mut encoder);
            uploads.write(&mut recorder, &target, 0, bytemuck::cast_slice(&[frame, 1, 2, 3]));
            recorder.copy_buffer_to_buffer(&target, 0, &between, 0, 16);
            uploads.write(&mut recorder, &target, 0, bytemuck::cast_slice(&[frame + 10]));
            // 256 KiB: more than the first buffer holds.
            let big: Vec<u32> = (0..(1u32 << 16)).map(|i| i ^ frame).collect();
            uploads.write(&mut recorder, &target, 16, bytemuck::cast_slice(&big));
            uploads.write(&mut recorder, &target, 16 + (1 << 18), bytemuck::cast_slice(&[7u32; 4]));
            uploads.flush();
            drop(recorder);
            queue.submit([encoder.finish()]);
            // The first frame grew the buffer twice, flushing before each.
            assert_eq!(uploads.end_frame(), (4, if frame == 0 { 3 } else { 1 }));
            let words = read(&device, &queue, &target);
            assert_eq!(&words[..4], &[frame + 10, 1, 2, 3]);
            assert!(words[4..4 + (1 << 16)]
                .iter()
                .enumerate()
                .all(|(i, &w)| w == i as u32 ^ frame));
            assert_eq!(&words[4 + (1 << 16)..8 + (1 << 16)], &[7; 4]);
            assert_eq!(read(&device, &queue, &between), vec![frame, 1, 2, 3]);
        }
    }

    /// Deferred writes land before the submission's other work, in the
    /// order written, as `Queue::write_buffer` (a copy recorded before the
    /// write still sees it), across frames and growth.
    #[test]
    fn deferred_writes_land_before_the_submission() {
        let Some((device, queue)) = gpu() else { return };
        let buffer = |size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let (target, seen) = (buffer(1 << 20), buffer(16));
        let uploads = FrameUploads::new(&device, &queue, "test deferred uploads");
        for frame in 0..3u32 {
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(&target, 0, &seen, 0, 16);
            uploads.defer(&target, 0, bytemuck::cast_slice(&[frame, 1, 2, 3]));
            uploads.defer(&target, 4, bytemuck::cast_slice(&[frame + 10]));
            let big: Vec<u32> = (0..(1u32 << 16)).map(|i| i ^ frame).collect();
            uploads.defer(&target, 16, bytemuck::cast_slice(&big));
            let first = uploads.record_deferred(&device).unwrap();
            assert!(uploads.record_deferred(&device).is_none());
            queue.submit([first, encoder.finish()]);
            uploads.end_frame();
            assert_eq!(read(&device, &queue, &seen), vec![frame, frame + 10, 2, 3]);
            let words = read(&device, &queue, &target);
            assert!(words[4..4 + (1 << 16)]
                .iter()
                .enumerate()
                .all(|(i, &w)| w == i as u32 ^ frame));
        }
    }
}
