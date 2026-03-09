//! Interval-based GPU transfer logging.
//!
//! Call [`track_upload`] / [`track_alloc`] / [`track_dealloc`] at each
//! `write_buffer` / `create_buffer` / buffer-drop site.  At the end of
//! every frame call [`end_frame`] — it accumulates totals and logs a
//! human-readable summary every `LOG_INTERVAL_FRAMES` frames.

use std::sync::atomic::{AtomicU64, Ordering};

static UPLOADED_BYTES: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);
static DEALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);
static FRAME_COUNTER: AtomicU64 = AtomicU64::new(0);

const LOG_INTERVAL_FRAMES: u64 = 60;

/// Record bytes uploaded to the GPU via `queue.write_buffer`.
#[inline]
pub fn track_upload(bytes: u64) {
    UPLOADED_BYTES.fetch_add(bytes, Ordering::Relaxed);
}

/// Record bytes allocated on the GPU via `device.create_buffer`.
#[inline]
pub fn track_alloc(bytes: u64) {
    ALLOCATED_BYTES.fetch_add(bytes, Ordering::Relaxed);
}

/// Record bytes deallocated from GPU (buffer replaced or dropped).
#[inline]
pub fn track_dealloc(bytes: u64) {
    DEALLOCATED_BYTES.fetch_add(bytes, Ordering::Relaxed);
}

/// Call once per frame at the end of `render()`.
/// Logs accumulated GPU transfer stats every `LOG_INTERVAL_FRAMES` frames.
pub fn end_frame() {
    let frame = FRAME_COUNTER.fetch_add(1, Ordering::Relaxed) + 1;
    if frame % LOG_INTERVAL_FRAMES != 0 {
        return;
    }

    let uploaded = UPLOADED_BYTES.swap(0, Ordering::Relaxed);
    let allocated = ALLOCATED_BYTES.swap(0, Ordering::Relaxed);
    let deallocated = DEALLOCATED_BYTES.swap(0, Ordering::Relaxed);

    if uploaded > 0 {
        eprintln!(
            "[helio] Uploaded {} additional data to GPU",
            format_bytes(uploaded),
        );
    }
    if allocated > 0 {
        eprintln!(
            "[helio] Allocated {} on GPU",
            format_bytes(allocated),
        );
    }
    if deallocated > 0 {
        eprintln!(
            "[helio] Requested dealloc of {} from GPU",
            format_bytes(deallocated),
        );
    }
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    } else {
        format!("{}B", bytes)
    }
}
