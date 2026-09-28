//! Whether a SceneDB row buffer holds any live row, without a per-frame CPU
//! scan.
//!
//! Hosts register component columns up front (Pulsar-Native's editor
//! registers water volumes, decals and billboards before any row exists), so
//! a buffer's presence says nothing about its contents, and passes that size
//! work to the buffer's capacity pay for it in scenes with no such content.
//! Dead rows are all-zero (`Zeroable`; SceneDB clears rows on removal), so a
//! pass reads the rows back whenever SceneDB reports new contents
//! (`content_generation`) or a reallocation (`epoch`) and records whether any
//! byte is non-zero.
//!
//! The readback is asynchronous and happens only when the rows change. Until
//! the answer for the current contents has arrived, the buffer is treated as
//! live, which is exactly a pass's behaviour without this check, so newly
//! added content is never skipped.

use std::sync::{Arc, Mutex};

use crate::BufferHandle;

type MapSlot = Arc<Mutex<Option<Result<(), wgpu::BufferAsyncError>>>>;

/// Buffers larger than this are not read back and are treated as live. The
/// copy only happens when SceneDB reports new contents, so a few MiB (e.g. a
/// 4096-row camera settings column at ~2.4 MiB) is a rare, bounded cost.
const MAX_READBACK_BYTES: u64 = 16 << 20;

/// `(epoch, content_generation)` of the buffer contents a result describes.
type ContentKey = (u64, u64);

pub struct SceneBufferLiveness {
    staging: Option<wgpu::Buffer>,
    slot: MapSlot,
    /// Contents copied into `staging` and waiting on `map_async`, with the
    /// row stride they were copied with.
    in_flight: Option<(ContentKey, u64)>,
    /// Latest harvested answer: whether those contents hold a live row.
    known: Option<(ContentKey, bool)>,
    /// Whether one row counts as live.
    row_is_live: fn(&[u8]) -> bool,
}

impl Default for SceneBufferLiveness {
    /// A row is live when any of its bytes is non-zero.
    fn default() -> Self {
        Self::with_row_predicate(|row| row.iter().any(|&byte| byte != 0))
    }
}

impl SceneBufferLiveness {
    /// Uses `row_is_live` instead of "any non-zero byte" to decide whether a
    /// row counts, so a pass can ask a narrower question of its rows (for
    /// example, whether any row enables a particular effect). Rows are
    /// `BufferHandle::row_bytes` long; a buffer without a row stride is
    /// passed as one slice.
    pub fn with_row_predicate(row_is_live: fn(&[u8]) -> bool) -> Self {
        Self { staging: None, slot: MapSlot::default(), in_flight: None, known: None, row_is_live }
    }

    /// Harvests a finished readback and starts one for new contents. Never
    /// waits on the GPU; the host's regular device polling completes maps.
    /// Call once per frame (from `prepare`) with the buffer's current handle.
    pub fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        handle: Option<&BufferHandle>,
    ) {
        if let Some((copied, row_bytes)) = self.in_flight {
            let result = self.slot.lock().expect("scene liveness slot poisoned").take();
            if let Some(result) = result {
                self.in_flight = None;
                if result.is_ok() {
                    let staging = self.staging.as_ref().expect("in-flight readback has staging");
                    let live = {
                        let bytes = staging.slice(..).get_mapped_range().expect("mapped staging");
                        if row_bytes == 0 {
                            (self.row_is_live)(&bytes)
                        } else {
                            bytes.chunks(row_bytes as usize).any(self.row_is_live)
                        }
                    };
                    staging.unmap();
                    self.known = Some((copied, live));
                }
            }
        }

        let Some(handle) = handle else {
            return;
        };
        let key = (handle.epoch, handle.content_generation);
        let size = handle.buffer.size();
        if self.in_flight.is_some()
            || self.known.is_some_and(|(known, _)| known == key)
            || size > MAX_READBACK_BYTES
            || size == 0
        {
            return;
        }

        if self.staging.as_ref().is_none_or(|staging| staging.size() != size) {
            self.staging = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("SceneDB Liveness Readback"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }));
        }
        let staging = self.staging.as_ref().expect("created above");
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("SceneDB Liveness Readback"),
        });
        encoder.copy_buffer_to_buffer(&handle.buffer, 0, staging, 0, size);
        queue.submit([encoder.finish()]);
        let slot = Arc::clone(&self.slot);
        staging.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            *slot.lock().expect("scene liveness slot poisoned") = Some(result);
        });
        self.in_flight = Some((key, handle.row_bytes));
    }

    /// False only when the current contents are known to hold no live row
    /// (per the row predicate).
    pub fn maybe_live(&self, handle: &BufferHandle) -> bool {
        let key = (handle.epoch, handle.content_generation);
        match self.known {
            Some((known, live)) if known == key => live,
            _ => true,
        }
    }
}
