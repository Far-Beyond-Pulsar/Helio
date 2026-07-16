use crate::{
    transvoxel::generated, EXTRACTION_SAMPLE_COUNT, REGULAR_CASE_COUNT, TRANSVOXEL_CLASSIFY_WGSL,
};
use bytemuck::{Pod, Zeroable};
use helio_planet_voxel_core::{CellWord, PAGE_CELL_COUNT};
use wgpu::util::DeviceExt;

pub const TRANSVOXEL_CLASSIFY_WORKGROUP_SIZE: u32 = 64;
pub const TRANSVOXEL_CLASSIFY_WORKGROUPS: u32 =
    (PAGE_CELL_COUNT as u32).div_ceil(TRANSVOXEL_CLASSIFY_WORKGROUP_SIZE);

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuTransvoxelDispatch {
    pub dirty_microbricks_low: u32,
    pub dirty_microbricks_high: u32,
    pub generation_low: u32,
    pub generation_high: u32,
    pub cell_count: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

impl GpuTransvoxelDispatch {
    pub const fn new(generation: u64, dirty_microbricks: u64) -> Self {
        Self {
            dirty_microbricks_low: dirty_microbricks as u32,
            dirty_microbricks_high: (dirty_microbricks >> 32) as u32,
            generation_low: generation as u32,
            generation_high: (generation >> 32) as u32,
            cell_count: PAGE_CELL_COUNT as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }
    }

    pub const fn dirty_microbricks(self) -> u64 {
        self.dirty_microbricks_low as u64 | ((self.dirty_microbricks_high as u64) << 32)
    }

    pub const fn generation(self) -> u64 {
        self.generation_low as u64 | ((self.generation_high as u64) << 32)
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuTransvoxelCell {
    pub packed_case_class_counts: u32,
    pub generation_low: u32,
    pub generation_high: u32,
    pub _pad: u32,
}

impl GpuTransvoxelCell {
    const VALID_BIT: u32 = 1 << 31;

    pub const fn new(
        case_index: u8,
        class_index: u8,
        vertex_count: u8,
        triangle_count: u8,
        generation: u64,
    ) -> Self {
        assert!(triangle_count <= 0x0f);
        Self {
            packed_case_class_counts: case_index as u32
                | ((class_index as u32) << 8)
                | ((vertex_count as u32) << 16)
                | ((triangle_count as u32) << 24)
                | Self::VALID_BIT,
            generation_low: generation as u32,
            generation_high: (generation >> 32) as u32,
            _pad: 0,
        }
    }

    pub const fn case_index(self) -> u8 {
        self.packed_case_class_counts as u8
    }

    pub const fn class_index(self) -> u8 {
        (self.packed_case_class_counts >> 8) as u8
    }

    pub const fn vertex_count(self) -> u8 {
        (self.packed_case_class_counts >> 16) as u8
    }

    pub const fn triangle_count(self) -> u8 {
        ((self.packed_case_class_counts >> 24) & 0x0f) as u8
    }

    pub const fn generation(self) -> u64 {
        self.generation_low as u64 | ((self.generation_high as u64) << 32)
    }

    pub const fn is_valid_for(self, generation: u64) -> bool {
        self.packed_case_class_counts & Self::VALID_BIT != 0 && self.generation() == generation
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuTransvoxelClassifyCounters {
    pub visited_cells: u32,
    pub active_cells: u32,
    pub vertices: u32,
    pub triangles: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransvoxelClassifierResourceStats {
    pub buffers: u32,
    pub allocated_bytes: u64,
}

/// A bounded regular-cell Transvoxel classification stage. It owns no
/// size-dependent resources and writes one fixed record per page cell. The
/// output stays on the GPU for the later prefix-sum and emission stages;
/// readback is used only by validation and benchmark tooling.
pub struct TransvoxelGpuClassifier {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    dispatch_buffer: wgpu::Buffer,
    sample_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    counters_buffer: wgpu::Buffer,
    resource_stats: TransvoxelClassifierResourceStats,
}

impl TransvoxelGpuClassifier {
    pub fn new(device: &wgpu::Device) -> Result<Self, TransvoxelGpuError> {
        validate_device_limits(&device.limits())?;

        let regular_cell_class = generated::REGULAR_CELL_CLASS.map(u32::from);
        let regular_geometry_counts = generated::REGULAR_CELL_GEOMETRY_COUNTS.map(u32::from);
        let dispatch_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Planetary Transvoxel Dispatch"),
            contents: bytemuck::bytes_of(&GpuTransvoxelDispatch::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let sample_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Planetary Transvoxel Halo Samples"),
            size: sample_buffer_bytes(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let class_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Planetary Transvoxel Regular Classes"),
            contents: bytemuck::cast_slice(&regular_cell_class),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let geometry_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Planetary Transvoxel Regular Geometry Counts"),
            contents: bytemuck::cast_slice(&regular_geometry_counts),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Planetary Transvoxel Classified Cells"),
            size: output_buffer_bytes(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let counters_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Planetary Transvoxel Classify Counters"),
            size: counters_buffer_bytes(),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Planetary Transvoxel Classify Shader"),
            source: wgpu::ShaderSource::Wgsl(TRANSVOXEL_CLASSIFY_WGSL.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Planetary Transvoxel Classify Pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("classify_regular_cells"),
            compilation_options: Default::default(),
            cache: None,
        });
        let layout = pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Planetary Transvoxel Classify Bind Group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dispatch_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: sample_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: class_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: geometry_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: counters_buffer.as_entire_binding(),
                },
            ],
        });
        let resource_stats = TransvoxelClassifierResourceStats {
            buffers: 6,
            allocated_bytes: dispatch_buffer_bytes()
                + sample_buffer_bytes()
                + class_buffer_bytes()
                + geometry_buffer_bytes()
                + output_buffer_bytes()
                + counters_buffer_bytes(),
        };
        Ok(Self {
            pipeline,
            bind_group,
            dispatch_buffer,
            sample_buffer,
            output_buffer,
            counters_buffer,
            resource_stats,
        })
    }

    pub fn dispatch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        samples: &[CellWord],
        generation: u64,
        dirty_microbricks: u64,
    ) -> Result<wgpu::SubmissionIndex, TransvoxelGpuError> {
        if samples.len() != EXTRACTION_SAMPLE_COUNT {
            return Err(TransvoxelGpuError::SampleCount {
                actual: samples.len(),
                expected: EXTRACTION_SAMPLE_COUNT,
            });
        }
        queue.write_buffer(&self.sample_buffer, 0, bytemuck::cast_slice(samples));
        queue.write_buffer(
            &self.dispatch_buffer,
            0,
            bytemuck::bytes_of(&GpuTransvoxelDispatch::new(generation, dirty_microbricks)),
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Planetary Transvoxel Classify Encoder"),
        });
        encoder.clear_buffer(&self.counters_buffer, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Planetary Transvoxel Classify Dispatch"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(TRANSVOXEL_CLASSIFY_WORKGROUPS, 1, 1);
        }
        Ok(queue.submit([encoder.finish()]))
    }

    pub fn output_buffer(&self) -> &wgpu::Buffer {
        &self.output_buffer
    }

    pub fn counters_buffer(&self) -> &wgpu::Buffer {
        &self.counters_buffer
    }

    pub const fn resource_stats(&self) -> TransvoxelClassifierResourceStats {
        self.resource_stats
    }

    /// Classification owns no surface-size-dependent resources.
    pub fn resize(&mut self, _width: u32, _height: u32) {}
}

fn validate_device_limits(limits: &wgpu::Limits) -> Result<(), TransvoxelGpuError> {
    if limits.max_storage_buffers_per_shader_stage < 5 {
        return Err(TransvoxelGpuError::DeviceLimit {
            name: "storage buffers per shader stage",
            required: 5,
            available: u64::from(limits.max_storage_buffers_per_shader_stage),
        });
    }
    if limits.max_uniform_buffers_per_shader_stage < 1 {
        return Err(TransvoxelGpuError::DeviceLimit {
            name: "uniform buffers per shader stage",
            required: 1,
            available: u64::from(limits.max_uniform_buffers_per_shader_stage),
        });
    }
    if limits.max_compute_invocations_per_workgroup < TRANSVOXEL_CLASSIFY_WORKGROUP_SIZE
        || limits.max_compute_workgroup_size_x < TRANSVOXEL_CLASSIFY_WORKGROUP_SIZE
    {
        return Err(TransvoxelGpuError::DeviceLimit {
            name: "compute workgroup width",
            required: u64::from(TRANSVOXEL_CLASSIFY_WORKGROUP_SIZE),
            available: u64::from(
                limits
                    .max_compute_invocations_per_workgroup
                    .min(limits.max_compute_workgroup_size_x),
            ),
        });
    }
    if limits.max_compute_workgroups_per_dimension < TRANSVOXEL_CLASSIFY_WORKGROUPS {
        return Err(TransvoxelGpuError::DeviceLimit {
            name: "compute workgroups per dimension",
            required: u64::from(TRANSVOXEL_CLASSIFY_WORKGROUPS),
            available: u64::from(limits.max_compute_workgroups_per_dimension),
        });
    }
    for (name, bytes) in [
        ("halo samples", sample_buffer_bytes()),
        ("regular classes", class_buffer_bytes()),
        ("regular geometry counts", geometry_buffer_bytes()),
        ("classified cells", output_buffer_bytes()),
        ("classify counters", counters_buffer_bytes()),
    ] {
        let available = limits
            .max_buffer_size
            .min(limits.max_storage_buffer_binding_size);
        if bytes > available {
            return Err(TransvoxelGpuError::DeviceLimit {
                name,
                required: bytes,
                available,
            });
        }
    }
    if dispatch_buffer_bytes() > limits.max_uniform_buffer_binding_size {
        return Err(TransvoxelGpuError::DeviceLimit {
            name: "dispatch uniform",
            required: dispatch_buffer_bytes(),
            available: limits.max_uniform_buffer_binding_size,
        });
    }
    Ok(())
}

const fn dispatch_buffer_bytes() -> u64 {
    core::mem::size_of::<GpuTransvoxelDispatch>() as u64
}

const fn sample_buffer_bytes() -> u64 {
    (EXTRACTION_SAMPLE_COUNT * core::mem::size_of::<CellWord>()) as u64
}

const fn class_buffer_bytes() -> u64 {
    (REGULAR_CASE_COUNT * core::mem::size_of::<u32>()) as u64
}

const fn geometry_buffer_bytes() -> u64 {
    (generated::REGULAR_CELL_GEOMETRY_COUNTS.len() * core::mem::size_of::<u32>()) as u64
}

const fn output_buffer_bytes() -> u64 {
    (PAGE_CELL_COUNT * core::mem::size_of::<GpuTransvoxelCell>()) as u64
}

const fn counters_buffer_bytes() -> u64 {
    core::mem::size_of::<GpuTransvoxelClassifyCounters>() as u64
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum TransvoxelGpuError {
    #[error("Transvoxel classification received {actual} samples; expected {expected}")]
    SampleCount { actual: usize, expected: usize },
    #[error(
        "Transvoxel classification requires {required} {name}, but the device exposes {available}"
    )]
    DeviceLimit {
        name: &'static str,
        required: u64,
        available: u64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocation_is_fixed_and_exact() {
        assert_eq!(dispatch_buffer_bytes(), 32);
        assert_eq!(sample_buffer_bytes(), 157_216);
        assert_eq!(class_buffer_bytes(), 1_024);
        assert_eq!(geometry_buffer_bytes(), 64);
        assert_eq!(output_buffer_bytes(), 524_288);
        assert_eq!(counters_buffer_bytes(), 16);
        assert_eq!(TRANSVOXEL_CLASSIFY_WORKGROUPS, 512);
    }

    #[test]
    fn insufficient_device_limits_fail_before_allocation() {
        let limits = wgpu::Limits {
            max_storage_buffers_per_shader_stage: 4,
            ..wgpu::Limits::default()
        };
        assert!(matches!(
            validate_device_limits(&limits),
            Err(TransvoxelGpuError::DeviceLimit {
                name: "storage buffers per shader stage",
                required: 5,
                available: 4,
            })
        ));
    }

    #[test]
    fn packed_cell_preserves_full_generation_and_counts() {
        let cell = GpuTransvoxelCell::new(0xfe, 55, 12, 12, u64::MAX - 7);
        assert_eq!(cell.case_index(), 0xfe);
        assert_eq!(cell.class_index(), 55);
        assert_eq!(cell.vertex_count(), 12);
        assert_eq!(cell.triangle_count(), 12);
        assert_eq!(cell.generation(), u64::MAX - 7);
        assert!(cell.is_valid_for(u64::MAX - 7));
        assert!(!cell.is_valid_for(u64::MAX - 8));
        assert!(!GpuTransvoxelCell::default().is_valid_for(0));
    }
}
