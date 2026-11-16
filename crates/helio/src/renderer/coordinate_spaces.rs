//! The frame's coordinate-space table on the GPU (`helio_core::frame_inputs`):
//! slot 0 the frame's own space (identity), then the host's spaces (rigid
//! frames, placed sublevels), current and previous frame. The renderer owns
//! it because it is the frame's: every kind's culling and drawers place
//! their content by an instance's space through it, whichever plugins are
//! present.

use wgpu::util::DeviceExt;

/// Slots the table holds (an instance's space is 8 bits).
pub const COORDINATE_SPACE_SLOTS: usize = 256;

pub(crate) struct CoordinateSpaceTable {
    current: wgpu::Buffer,
    previous: wgpu::Buffer,
    /// What `current` holds, which becomes `previous` when the table next
    /// changes.
    last: Vec<f32>,
    /// `previous` differs from `current`: brought level next frame (a space
    /// that stops moving has no motion).
    previous_stale: bool,
}

/// The table as GPU words: slot 0 identity, then `spaces` from slot 1.
fn words(spaces: &[glam::Mat4]) -> Vec<f32> {
    let mut words = vec![0.0f32; COORDINATE_SPACE_SLOTS * 16];
    words[..16].copy_from_slice(&glam::Mat4::IDENTITY.to_cols_array());
    for (slot, space) in spaces.iter().take(COORDINATE_SPACE_SLOTS - 1).enumerate() {
        words[(slot + 1) * 16..(slot + 2) * 16].copy_from_slice(&space.to_cols_array());
    }
    words
}

impl CoordinateSpaceTable {
    pub fn new(device: &wgpu::Device) -> Self {
        let last = words(&[]);
        let buffer = |label: &str| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&last),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        };
        Self {
            current: buffer("Coordinate Spaces"),
            previous: buffer("Coordinate Spaces (previous frame)"),
            last,
            previous_stale: false,
        }
    }

    /// Uploads `spaces` (the host's, slot 1 on) for this frame.
    pub fn upload(&mut self, queue: &wgpu::Queue, spaces: &[glam::Mat4]) {
        let next = words(spaces);
        if next != self.last {
            queue.write_buffer(&self.current, 0, bytemuck::cast_slice(&next));
            queue.write_buffer(&self.previous, 0, bytemuck::cast_slice(&self.last));
            self.last = next;
            self.previous_stale = true;
        } else if self.previous_stale {
            queue.write_buffer(&self.previous, 0, bytemuck::cast_slice(&self.last));
            self.previous_stale = false;
        }
    }

    pub fn frame_data(&self) -> helio_core::CoordinateSpacesFrameData<'_> {
        helio_core::CoordinateSpacesFrameData {
            coordinate_spaces: &self.current,
            coordinate_spaces_prev: &self.previous,
        }
    }
}
