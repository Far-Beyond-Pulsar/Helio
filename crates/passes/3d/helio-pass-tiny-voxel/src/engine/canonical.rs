//! Immutable published-source inputs for the canonical far-tracing experiment.
//! Generation may already target another revision; never expose its edit buffer
//! to rays traversing the old published cut.
use crate::{GpuEdit, World};
use std::sync::Arc;
use wgpu::util::DeviceExt;

pub(super) struct Source {
    pub edits: wgpu::Buffer,
    pub settings: wgpu::Buffer,
    world: Option<Arc<World>>,
}

#[cfg(test)]
#[path = "canonical_tests.rs"]
mod tests;

impl Source {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            edits: super::terrain::buffer(
                device,
                "canonical published edits",
                crate::world::MAX_EDITS as u64 * 32,
            ),
            settings: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("canonical clearance and published revision"),
                contents: bytemuck::cast_slice(&Self::settings(0)),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }),
            world: None,
        }
    }

    fn settings(edits: u32) -> [u32; 8] {
        let field = crate::landforms::default_clearance();
        [
            (field.lipschitz as f32).to_bits(),
            (field.quantization_guard as f32).to_bits(),
            (field.maximum_radius as f32).to_bits(),
            (crate::world::procedural_outer_radius() as f32).to_bits(),
            edits,
            0,
            0,
            0,
        ]
    }

    pub fn publish(&mut self, queue: &wgpu::Queue, world: &Arc<World>) {
        if self
            .world
            .as_ref()
            .is_some_and(|old| Arc::ptr_eq(old, world))
        {
            return;
        }
        let edits: Vec<_> = world
            .edits
            .iter()
            .map(|e| GpuEdit {
                cell: e.cell,
                material: e.material,
                radius: e.radius,
                radius_units: e.radius_units(),
                pad: [0.0; 2],
            })
            .collect();
        if !edits.is_empty() {
            queue.write_buffer(&self.edits, 0, bytemuck::cast_slice(&edits));
        }
        queue.write_buffer(
            &self.settings,
            0,
            bytemuck::cast_slice(&Self::settings(edits.len() as u32)),
        );
        self.world = Some(world.clone());
    }
}
