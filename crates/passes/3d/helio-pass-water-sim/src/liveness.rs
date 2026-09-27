//! Whether any water volume row is live, without a per-frame CPU scan.
//!
//! Hosts register the `"water_volumes"` column up front (Pulsar-Native's
//! editor does), so the buffer exists in scenes with no water at all and its
//! presence says nothing. Dead rows are all-zero (`Zeroable`), so the pass
//! reads the rows back whenever SceneDB reports new contents
//! (`content_generation`) or a reallocation (`epoch`) and records whether
//! any byte is non-zero.
//!
//! The readback is asynchronous and happens only when the rows change. Until
//! the answer for the current contents has arrived, the volumes are treated
//! as live, which is exactly the pass's behaviour without this check, so a
//! newly added volume is never skipped.

use std::sync::{Arc, Mutex};

use helio_core::BufferHandle;

type MapSlot = Arc<Mutex<Option<Result<(), wgpu::BufferAsyncError>>>>;

/// Rows beyond this are not read back; a larger buffer is treated as live.
const MAX_READBACK_BYTES: u64 = 1 << 20;

/// `(epoch, content_generation)` of the buffer contents a result describes.
type ContentKey = (u64, u64);

#[derive(Default)]
pub(crate) struct VolumeLiveness {
    staging: Option<wgpu::Buffer>,
    slot: MapSlot,
    /// Contents copied into `staging` and waiting on `map_async`.
    in_flight: Option<ContentKey>,
    /// Latest harvested answer: any live row in those contents.
    known: Option<(ContentKey, bool)>,
}

impl VolumeLiveness {
    /// Harvests a finished readback and starts one for new contents. Never
    /// waits on the GPU; the host's regular device polling completes maps.
    pub(crate) fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        handle: Option<&BufferHandle>,
    ) {
        if let Some(copied) = self.in_flight {
            let result = self.slot.lock().expect("water liveness slot poisoned").take();
            if let Some(result) = result {
                self.in_flight = None;
                if result.is_ok() {
                    let staging = self.staging.as_ref().expect("in-flight readback has staging");
                    let live = {
                        let bytes = staging.slice(..).get_mapped_range().expect("mapped staging");
                        bytes.iter().any(|&byte| byte != 0)
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

        if self.staging.as_ref().map_or(true, |staging| staging.size() != size) {
            self.staging = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("WaterSim Volume Liveness Readback"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }));
        }
        let staging = self.staging.as_ref().expect("created above");
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("WaterSim Volume Liveness Readback"),
        });
        encoder.copy_buffer_to_buffer(&handle.buffer, 0, staging, 0, size);
        queue.submit([encoder.finish()]);
        let slot = Arc::clone(&self.slot);
        staging.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            *slot.lock().expect("water liveness slot poisoned") = Some(result);
        });
        self.in_flight = Some(key);
    }

    /// False only when the current contents are known to hold no live row.
    pub(crate) fn maybe_live(&self, handle: &BufferHandle) -> bool {
        let key = (handle.epoch, handle.content_generation);
        match self.known {
            Some((known, live)) if known == key => live,
            _ => true,
        }
    }
}
