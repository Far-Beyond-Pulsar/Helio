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

