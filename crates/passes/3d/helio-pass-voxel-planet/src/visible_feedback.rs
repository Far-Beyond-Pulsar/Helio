use glam::DVec3;
use std::sync::{
    atomic::{AtomicU8, Ordering},
    Arc,
};

pub(super) const CAPACITY: usize = 256;
const COPY_BYTES: u64 = 16 + CAPACITY as u64 * 8;
const STORAGE_BYTES: u64 = COPY_BYTES + 4096 * 4;

#[derive(Clone, Copy)]
pub(super) struct View {
    pub eye: DVec3,
    pub forward: DVec3,
    pub size: [u32; 2],
    pub id: u32,
    pub frame: u32,
    pub max_eye_delta: f64,
}

impl View {
    fn accepts(self, old: Self) -> bool {
        self.size == old.size
            && self.id == old.id
            && self.frame.wrapping_sub(old.frame) <= 8
            && self.eye.distance(old.eye) <= self.max_eye_delta
            && self.forward.dot(old.forward) >= 0.75
    }
}

struct Readback {
    buffer: wgpu::Buffer,
    // 0 pending, 1 mapped, 2 mapping failed.
    state: Arc<AtomicU8>,
    stage: u8,
    view: Option<View>,
}

pub(super) struct Feedback {
    pub buffer: wgpu::Buffer,
    readbacks: Vec<Readback>,
    pub counts: [u32; 3],
    last_view: Option<View>,
}

impl Feedback {
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("planet visible block requests"),
            size: STORAGE_BYTES,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readbacks = (0..3)
            .map(|_| Readback {
                buffer: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("planet visible request readback"),
                    size: COPY_BYTES,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
                state: Arc::new(AtomicU8::new(0)),
                stage: 0,
                view: None,
            })
            .collect();
        Self {
            buffer,
            readbacks,
            counts: [0; 3],
            last_view: None,
        }
    }

    pub fn bytes(&self) -> u64 {
        STORAGE_BYTES + COPY_BYTES * self.readbacks.len() as u64
    }

    pub fn pending(&self) -> bool {
        self.readbacks.iter().any(|r| r.stage != 0)
    }

    pub fn view_changed(&self, current: View) -> bool {
        self.last_view.is_none_or(|old| {
            old.size != current.size
                || old.id != current.id
                || old.eye.distance(current.eye) > 0.01
                || old.forward.dot(current.forward) < 0.99999
        })
    }

    // The caller already polls the device without waiting. Mapping begins
    // only on the next encode, after the copy has been submitted.
    pub fn poll(&mut self, current: View) -> Vec<(u32, u32)> {
        let mut blocks = Vec::new();
        let mut newest_age = u32::MAX;
        for r in &mut self.readbacks {
            let state = r.state.load(Ordering::Acquire);
            if r.stage == 2 && state != 0 {
                if state == 1 {
                    if r.view.is_some_and(|old| {
                        current.accepts(old) && current.frame.wrapping_sub(old.frame) < newest_age
                    }) {
                        newest_age = current.frame.wrapping_sub(r.view.unwrap().frame);
                        let data = r.buffer.slice(..).get_mapped_range().unwrap();
                        for (n, count) in self.counts.iter_mut().enumerate() {
                            *count = u32::from_le_bytes(data[n * 4..n * 4 + 4].try_into().unwrap());
                        }
                        blocks = decode(&data);
                    }
                    r.buffer.unmap();
                }
                r.stage = 0;
                r.view = None;
                r.state.store(0, Ordering::Release);
            }
        }
        for r in &mut self.readbacks {
            if r.stage == 1 {
                let state = r.state.clone();
                r.buffer
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |result| {
                        state.store(if result.is_ok() { 1 } else { 2 }, Ordering::Release);
                    });
                r.stage = 2;
            }
        }
        // Prefer the newest completed frame, rather than replaying old views.
        blocks.sort_unstable();
        blocks.dedup();
        blocks
    }

    pub fn begin(&mut self, encoder: &mut wgpu::CommandEncoder, view: View) -> Option<usize> {
        let at = self.readbacks.iter().position(|r| r.stage == 0)?;
        encoder.clear_buffer(&self.buffer, 0, None);
        let r = &mut self.readbacks[at];
        r.view = Some(view);
        r.stage = 1;
        self.last_view = Some(view);
        Some(at)
    }

    pub fn copy(&self, encoder: &mut wgpu::CommandEncoder, at: usize) {
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &self.readbacks[at].buffer, 0, COPY_BYTES);
    }
}

fn decode(data: &[u8]) -> Vec<(u32, u32)> {
    if data.len() < COPY_BYTES as usize {
        return Vec::new();
    }
    let word = |at: usize| u32::from_le_bytes(data[at..at + 4].try_into().unwrap());
    let count = (word(0) as usize).min(CAPACITY);
    (0..count)
        .map(|i| (word(16 + i * 8), word(20 + i * 8)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feedback_decodes_bounded_full_keys_and_signed_indices() {
        let mut data = vec![0u8; COPY_BYTES as usize];
        data[..4].copy_from_slice(&u32::MAX.to_le_bytes());
        let key = 12u32 | (2 << 24) | (7 << 27);
        data[16..20].copy_from_slice(&key.to_le_bytes());
        data[20..24].copy_from_slice(&(-8i32).to_le_bytes());
        let keys = decode(&data);
        assert_eq!(keys.len(), CAPACITY);
        assert_eq!(keys[0], (key, (-8i32) as u32));
        assert!(decode(&data[..16]).is_empty());
    }

    #[test]
    fn feedback_rejects_old_camera_and_resize_without_double_frame_wrap() {
        let old = View {
            eye: DVec3::ZERO,
            forward: DVec3::Z,
            size: [1280, 720],
            id: 1,
            frame: u32::MAX - 1,
            max_eye_delta: 50.0,
        };
        let now = View {
            frame: 1,
            eye: DVec3::X * 2.0,
            ..old
        };
        assert!(now.accepts(old));
        assert!(!View {
            size: [720, 1280],
            ..now
        }
        .accepts(old));
        assert!(!View { id: 2, ..now }.accepts(old));
        assert!(!View {
            eye: DVec3::X * 100.0,
            ..now
        }
        .accepts(old));
        assert!(!View {
            forward: -DVec3::Z,
            ..now
        }
        .accepts(old));
        assert!(!View { frame: 10, ..now }.accepts(old));
    }
}
