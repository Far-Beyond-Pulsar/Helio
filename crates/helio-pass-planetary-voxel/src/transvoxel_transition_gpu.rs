use crate::{
    transvoxel::generated, GpuTerrainVertex, GpuTransvoxelCellOffset, GpuTransvoxelScanBlock,
    TRANSITION_ALL_FACE_SLAB_SAMPLE_COUNT, TRANSVOXEL_TRANSITION_GPU_WGSL,
};
use bytemuck::{Pod, Zeroable};
use helio_planet_voxel_core::{CellWord, TRANSITION_FACE_MASK};
use wgpu::util::DeviceExt;

pub const TRANSVOXEL_TRANSITION_FACE_COUNT: u32 = 6;
pub const TRANSVOXEL_TRANSITION_CELLS_PER_FACE: u32 = 32 * 32;
pub const TRANSVOXEL_TRANSITION_CELL_COUNT: u32 =
    TRANSVOXEL_TRANSITION_FACE_COUNT * TRANSVOXEL_TRANSITION_CELLS_PER_FACE;
pub const TRANSVOXEL_TRANSITION_CLASSIFY_WORKGROUP_SIZE: u32 = 64;
pub const TRANSVOXEL_TRANSITION_CLASSIFY_WORKGROUPS: u32 =
    TRANSVOXEL_TRANSITION_CELL_COUNT / TRANSVOXEL_TRANSITION_CLASSIFY_WORKGROUP_SIZE;
pub const TRANSVOXEL_TRANSITION_SCAN_WORKGROUP_SIZE: u32 = 256;
pub const TRANSVOXEL_TRANSITION_SCAN_BLOCKS: u32 =
    TRANSVOXEL_TRANSITION_CELL_COUNT / TRANSVOXEL_TRANSITION_SCAN_WORKGROUP_SIZE;
pub const TRANSVOXEL_TRANSITION_MAX_VERTICES: u32 = TRANSVOXEL_TRANSITION_CELL_COUNT * 12;
pub const TRANSVOXEL_TRANSITION_MAX_INDICES: u32 = TRANSVOXEL_TRANSITION_CELL_COUNT * 12 * 3;

const TRANSITION_CLASS_TABLE_VALUES: usize = 512;
const TRANSITION_GEOMETRY_TABLE_VALUES: usize = 56;
const TRANSITION_VERTEX_TABLE_VALUES: usize = 512 * 12;
const TRANSITION_TOPOLOGY_TABLE_VALUES: usize = 56 * 36;
const TRANSITION_TABLE_VALUES: usize = TRANSITION_CLASS_TABLE_VALUES
    + TRANSITION_GEOMETRY_TABLE_VALUES
    + TRANSITION_VERTEX_TABLE_VALUES
    + TRANSITION_TOPOLOGY_TABLE_VALUES;

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuTransvoxelTransitionDispatch {
    pub transition_mask: u32,
    pub generation_low: u32,
    pub generation_high: u32,
    pub cell_count: u32,
    pub max_vertices: u32,
    pub max_indices: u32,
    pub scan_block_count: u32,
    pub _pad: u32,
}

impl GpuTransvoxelTransitionDispatch {
    pub const fn new(
        transition_mask: u8,
        generation: u64,
        config: TransvoxelGpuTransitionExtractorConfig,
    ) -> Self {
        Self {
            transition_mask: transition_mask as u32,
            generation_low: generation as u32,
            generation_high: (generation >> 32) as u32,
            cell_count: TRANSVOXEL_TRANSITION_CELL_COUNT,
            max_vertices: config.max_vertices,
            max_indices: config.max_indices,
            scan_block_count: TRANSVOXEL_TRANSITION_SCAN_BLOCKS,
            _pad: 0,
        }
    }

    pub const fn generation(self) -> u64 {
        self.generation_low as u64 | ((self.generation_high as u64) << 32)
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuTransvoxelTransitionCell {
    pub packed_case_class_counts: u32,
    pub generation_low: u32,
    pub generation_high: u32,
    pub _pad: u32,
}

impl GpuTransvoxelTransitionCell {
    const VALID: u32 = 1 << 31;

    pub const fn new(
        case_index: u16,
        class_code: u8,
        vertex_count: u8,
        triangle_count: u8,
        generation: u64,
    ) -> Self {
        Self {
            packed_case_class_counts: case_index as u32
                | ((class_code as u32) << 9)
                | ((vertex_count as u32) << 17)
                | ((triangle_count as u32) << 21)
                | Self::VALID,
            generation_low: generation as u32,
            generation_high: (generation >> 32) as u32,
            _pad: 0,
        }
    }

    pub const fn case_index(self) -> u16 {
        (self.packed_case_class_counts & 0x1ff) as u16
    }

    pub const fn class_code(self) -> u8 {
        ((self.packed_case_class_counts >> 9) & 0xff) as u8
    }

    pub const fn class_index(self) -> u8 {
        self.class_code() & 0x7f
    }

    pub const fn reverse_winding(self) -> bool {
        self.class_code() & 0x80 != 0
    }

    pub const fn vertex_count(self) -> u8 {
        ((self.packed_case_class_counts >> 17) & 0x0f) as u8
    }

    pub const fn triangle_count(self) -> u8 {
        ((self.packed_case_class_counts >> 21) & 0x0f) as u8
    }

    pub const fn generation(self) -> u64 {
        self.generation_low as u64 | ((self.generation_high as u64) << 32)
    }

    pub const fn is_valid_for(self, generation: u64) -> bool {
        self.packed_case_class_counts & Self::VALID != 0 && self.generation() == generation
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuTransvoxelTransitionCounters {
    pub active_cells: u32,
    pub active_faces: u32,
    pub required_vertices: u32,
    pub required_indices: u32,
    pub emitted_vertices: u32,
    pub emitted_indices: u32,
    pub vertex_overflow: u32,
    pub index_overflow: u32,
    pub completed: u32,
    pub _pad: [u32; 3],
}

impl GpuTransvoxelTransitionCounters {
    pub const fn overflowed(self) -> bool {
        self.vertex_overflow != 0 || self.index_overflow != 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransvoxelGpuTransitionExtractorConfig {
    pub max_vertices: u32,
    pub max_indices: u32,
}

impl TransvoxelGpuTransitionExtractorConfig {
    pub fn new(max_vertices: u32, max_indices: u32) -> Result<Self, TransvoxelTransitionGpuError> {
        if max_vertices == 0 || max_indices == 0 {
            return Err(TransvoxelTransitionGpuError::InvalidExtractionCapacity {
                max_vertices,
                max_indices,
            });
        }
        Ok(Self {
            max_vertices,
            max_indices,
        })
    }
}

impl Default for TransvoxelGpuTransitionExtractorConfig {
    fn default() -> Self {
        Self {
            max_vertices: TRANSVOXEL_TRANSITION_MAX_VERTICES,
            max_indices: TRANSVOXEL_TRANSITION_MAX_INDICES,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransvoxelTransitionExtractorResourceStats {
    pub buffers: u32,
    pub allocated_bytes: u64,
}

/// Fixed-capacity GPU transition classifier, prefix allocator, and emitter.
/// Every dispatch processes at most six 32x32 coarse-page faces. Capacity
/// overflow suppresses the whole result instead of publishing a partial seam.
pub struct TransvoxelGpuTransitionExtractor {
    classify_pipeline: wgpu::ComputePipeline,
    scan_cells_pipeline: wgpu::ComputePipeline,
    scan_blocks_pipeline: wgpu::ComputePipeline,
    emit_pipeline: wgpu::ComputePipeline,
    classify_bind_group: wgpu::BindGroup,
    scan_cells_bind_group: wgpu::BindGroup,
    scan_blocks_bind_group: wgpu::BindGroup,
    emit_bind_group: wgpu::BindGroup,
    dispatch_buffer: wgpu::Buffer,
    sample_buffer: wgpu::Buffer,
    cells_buffer: wgpu::Buffer,
    offsets_buffer: wgpu::Buffer,
    blocks_buffer: wgpu::Buffer,
    vertices_buffer: wgpu::Buffer,
    indices_buffer: wgpu::Buffer,
    counters_buffer: wgpu::Buffer,
    config: TransvoxelGpuTransitionExtractorConfig,
    resource_stats: TransvoxelTransitionExtractorResourceStats,
}

impl TransvoxelGpuTransitionExtractor {
    pub fn new(
        device: &wgpu::Device,
        config: TransvoxelGpuTransitionExtractorConfig,
    ) -> Result<Self, TransvoxelTransitionGpuError> {
        validate_limits(&device.limits(), config)?;
        let dispatch_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Planetary Transvoxel Transition Dispatch"),
            size: dispatch_buffer_bytes(),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sample_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Planetary Transvoxel Transition Scalar Slabs"),
            size: sample_buffer_bytes(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let tables = packed_transition_tables();
        let tables_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Planetary Transvoxel Transition Tables"),
            contents: bytemuck::cast_slice(&tables),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let cells_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Planetary Transvoxel Transition Cells"),
            size: cells_buffer_bytes(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let offsets_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Planetary Transvoxel Transition Offsets"),
            size: offsets_buffer_bytes(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let blocks_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Planetary Transvoxel Transition Scan Blocks"),
            size: blocks_buffer_bytes(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let vertices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Planetary Transvoxel Transition Vertices"),
            size: vertices_buffer_bytes(config),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let indices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Planetary Transvoxel Transition Indices"),
            size: indices_buffer_bytes(config),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDEX
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let counters_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Planetary Transvoxel Transition Counters"),
            size: counters_buffer_bytes(),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Planetary Transvoxel Transition GPU Shader"),
            source: wgpu::ShaderSource::Wgsl(TRANSVOXEL_TRANSITION_GPU_WGSL.into()),
        });
        let classify_pipeline = compute_pipeline(device, &shader, "classify_transition_cells");
        let scan_cells_pipeline = compute_pipeline(device, &shader, "scan_transition_cells");
        let scan_blocks_pipeline = compute_pipeline(device, &shader, "scan_transition_blocks");
        let emit_pipeline = compute_pipeline(device, &shader, "emit_transition_cells");
        let classify_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Planetary Transvoxel Transition Classify Bind Group"),
            layout: &classify_pipeline.get_bind_group_layout(0),
            entries: &[
                entry(0, &dispatch_buffer),
                entry(1, &sample_buffer),
                entry(2, &tables_buffer),
                entry(3, &cells_buffer),
                entry(8, &counters_buffer),
            ],
        });
        let scan_cells_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Planetary Transvoxel Transition Cell Scan Bind Group"),
            layout: &scan_cells_pipeline.get_bind_group_layout(0),
            entries: &[
                entry(0, &dispatch_buffer),
                entry(3, &cells_buffer),
                entry(4, &offsets_buffer),
                entry(5, &blocks_buffer),
            ],
        });
        let scan_blocks_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Planetary Transvoxel Transition Block Scan Bind Group"),
            layout: &scan_blocks_pipeline.get_bind_group_layout(0),
            entries: &[
                entry(0, &dispatch_buffer),
                entry(5, &blocks_buffer),
                entry(8, &counters_buffer),
            ],
        });
        let emit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Planetary Transvoxel Transition Emit Bind Group"),
            layout: &emit_pipeline.get_bind_group_layout(0),
            entries: &[
                entry(0, &dispatch_buffer),
                entry(1, &sample_buffer),
                entry(2, &tables_buffer),
                entry(3, &cells_buffer),
                entry(4, &offsets_buffer),
                entry(5, &blocks_buffer),
                entry(6, &vertices_buffer),
                entry(7, &indices_buffer),
                entry(8, &counters_buffer),
            ],
        });
        let resource_stats = TransvoxelTransitionExtractorResourceStats {
            buffers: 9,
            allocated_bytes: dispatch_buffer_bytes()
                + sample_buffer_bytes()
                + tables_buffer_bytes()
                + cells_buffer_bytes()
                + offsets_buffer_bytes()
                + blocks_buffer_bytes()
                + vertices_buffer_bytes(config)
                + indices_buffer_bytes(config)
                + counters_buffer_bytes(),
        };
        Ok(Self {
            classify_pipeline,
            scan_cells_pipeline,
            scan_blocks_pipeline,
            emit_pipeline,
            classify_bind_group,
            scan_cells_bind_group,
            scan_blocks_bind_group,
            emit_bind_group,
            dispatch_buffer,
            sample_buffer,
            cells_buffer,
            offsets_buffer,
            blocks_buffer,
            vertices_buffer,
            indices_buffer,
            counters_buffer,
            config,
            resource_stats,
        })
    }

    pub fn dispatch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        face_slabs: &[CellWord],
        transition_mask: u8,
        generation: u64,
    ) -> Result<wgpu::SubmissionIndex, TransvoxelTransitionGpuError> {
        if face_slabs.len() != TRANSITION_ALL_FACE_SLAB_SAMPLE_COUNT {
            return Err(TransvoxelTransitionGpuError::SampleCount {
                actual: face_slabs.len(),
                expected: TRANSITION_ALL_FACE_SLAB_SAMPLE_COUNT,
            });
        }
        if transition_mask & !TRANSITION_FACE_MASK != 0 {
            return Err(TransvoxelTransitionGpuError::TransitionMask(
                transition_mask,
            ));
        }
        let dispatch =
            GpuTransvoxelTransitionDispatch::new(transition_mask, generation, self.config);
        queue.write_buffer(&self.sample_buffer, 0, bytemuck::cast_slice(face_slabs));
        queue.write_buffer(&self.dispatch_buffer, 0, bytemuck::bytes_of(&dispatch));
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Planetary Transvoxel Transition Extraction Encoder"),
        });
        encoder.clear_buffer(&self.counters_buffer, 0, None);
        encode_dispatch(
            &mut encoder,
            &self.classify_pipeline,
            &self.classify_bind_group,
            TRANSVOXEL_TRANSITION_CLASSIFY_WORKGROUPS,
            "Planetary Transvoxel Transition Classification",
        );
        encode_dispatch(
            &mut encoder,
            &self.scan_cells_pipeline,
            &self.scan_cells_bind_group,
            TRANSVOXEL_TRANSITION_SCAN_BLOCKS,
            "Planetary Transvoxel Transition Cell Scan",
        );
        encode_dispatch(
            &mut encoder,
            &self.scan_blocks_pipeline,
            &self.scan_blocks_bind_group,
            1,
            "Planetary Transvoxel Transition Block Scan",
        );
        encode_dispatch(
            &mut encoder,
            &self.emit_pipeline,
            &self.emit_bind_group,
            TRANSVOXEL_TRANSITION_CLASSIFY_WORKGROUPS,
            "Planetary Transvoxel Transition Emission",
        );
        Ok(queue.submit([encoder.finish()]))
    }

    pub fn cells_buffer(&self) -> &wgpu::Buffer {
        &self.cells_buffer
    }

    pub fn offsets_buffer(&self) -> &wgpu::Buffer {
        &self.offsets_buffer
    }

    pub fn blocks_buffer(&self) -> &wgpu::Buffer {
        &self.blocks_buffer
    }

    pub fn vertices_buffer(&self) -> &wgpu::Buffer {
        &self.vertices_buffer
    }

    pub fn indices_buffer(&self) -> &wgpu::Buffer {
        &self.indices_buffer
    }

    pub fn counters_buffer(&self) -> &wgpu::Buffer {
        &self.counters_buffer
    }

    pub const fn config(&self) -> TransvoxelGpuTransitionExtractorConfig {
        self.config
    }

    pub const fn resource_stats(&self) -> TransvoxelTransitionExtractorResourceStats {
        self.resource_stats
    }

    pub fn resize(&mut self, _width: u32, _height: u32) {}
}

fn compute_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    entry_point: &str,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry_point),
        layout: None,
        module: shader,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn encode_dispatch(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    workgroups: u32,
    label: &'static str,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(workgroups, 1, 1);
}

fn packed_transition_tables() -> Vec<u32> {
    let mut tables = Vec::with_capacity(TRANSITION_TABLE_VALUES);
    tables.extend(generated::TRANSITION_CELL_CLASS.map(u32::from));
    tables.extend(generated::TRANSITION_CELL_GEOMETRY_COUNTS.map(u32::from));
    for row in generated::TRANSITION_VERTEX_DATA {
        tables.extend(row.map(u32::from));
    }
    for row in generated::TRANSITION_CELL_VERTEX_INDEX {
        tables.extend(row.map(u32::from));
    }
    tables
}

fn validate_limits(
    limits: &wgpu::Limits,
    config: TransvoxelGpuTransitionExtractorConfig,
) -> Result<(), TransvoxelTransitionGpuError> {
    TransvoxelGpuTransitionExtractorConfig::new(config.max_vertices, config.max_indices)?;
    if limits.max_storage_buffers_per_shader_stage < 8 {
        return Err(TransvoxelTransitionGpuError::DeviceLimit {
            name: "storage buffers per shader stage",
            required: 8,
            available: u64::from(limits.max_storage_buffers_per_shader_stage),
        });
    }
    if limits.max_uniform_buffers_per_shader_stage < 1 {
        return Err(TransvoxelTransitionGpuError::DeviceLimit {
            name: "uniform buffers per shader stage",
            required: 1,
            available: u64::from(limits.max_uniform_buffers_per_shader_stage),
        });
    }
    let compute_width = limits
        .max_compute_invocations_per_workgroup
        .min(limits.max_compute_workgroup_size_x);
    if compute_width < TRANSVOXEL_TRANSITION_SCAN_WORKGROUP_SIZE {
        return Err(TransvoxelTransitionGpuError::DeviceLimit {
            name: "compute workgroup width",
            required: u64::from(TRANSVOXEL_TRANSITION_SCAN_WORKGROUP_SIZE),
            available: u64::from(compute_width),
        });
    }
    if limits.max_compute_workgroups_per_dimension < TRANSVOXEL_TRANSITION_CLASSIFY_WORKGROUPS {
        return Err(TransvoxelTransitionGpuError::DeviceLimit {
            name: "compute workgroups per dimension",
            required: u64::from(TRANSVOXEL_TRANSITION_CLASSIFY_WORKGROUPS),
            available: u64::from(limits.max_compute_workgroups_per_dimension),
        });
    }
    let available = limits
        .max_buffer_size
        .min(limits.max_storage_buffer_binding_size);
    for (name, required) in [
        ("transition scalar slabs", sample_buffer_bytes()),
        ("transition tables", tables_buffer_bytes()),
        ("transition cells", cells_buffer_bytes()),
        ("transition offsets", offsets_buffer_bytes()),
        ("transition scan blocks", blocks_buffer_bytes()),
        ("transition vertices", vertices_buffer_bytes(config)),
        ("transition indices", indices_buffer_bytes(config)),
        ("transition counters", counters_buffer_bytes()),
    ] {
        if required > available {
            return Err(TransvoxelTransitionGpuError::DeviceLimit {
                name,
                required,
                available,
            });
        }
    }
    if dispatch_buffer_bytes() > limits.max_uniform_buffer_binding_size {
        return Err(TransvoxelTransitionGpuError::DeviceLimit {
            name: "transition dispatch uniform",
            required: dispatch_buffer_bytes(),
            available: limits.max_uniform_buffer_binding_size,
        });
    }
    Ok(())
}

const fn dispatch_buffer_bytes() -> u64 {
    core::mem::size_of::<GpuTransvoxelTransitionDispatch>() as u64
}

const fn sample_buffer_bytes() -> u64 {
    (TRANSITION_ALL_FACE_SLAB_SAMPLE_COUNT * core::mem::size_of::<CellWord>()) as u64
}

const fn tables_buffer_bytes() -> u64 {
    (TRANSITION_TABLE_VALUES * core::mem::size_of::<u32>()) as u64
}

const fn cells_buffer_bytes() -> u64 {
    TRANSVOXEL_TRANSITION_CELL_COUNT as u64
        * core::mem::size_of::<GpuTransvoxelTransitionCell>() as u64
}

const fn offsets_buffer_bytes() -> u64 {
    TRANSVOXEL_TRANSITION_CELL_COUNT as u64 * core::mem::size_of::<GpuTransvoxelCellOffset>() as u64
}

const fn blocks_buffer_bytes() -> u64 {
    TRANSVOXEL_TRANSITION_SCAN_BLOCKS as u64 * core::mem::size_of::<GpuTransvoxelScanBlock>() as u64
}

const fn vertices_buffer_bytes(config: TransvoxelGpuTransitionExtractorConfig) -> u64 {
    config.max_vertices as u64 * core::mem::size_of::<GpuTerrainVertex>() as u64
}

const fn indices_buffer_bytes(config: TransvoxelGpuTransitionExtractorConfig) -> u64 {
    config.max_indices as u64 * core::mem::size_of::<u32>() as u64
}

const fn counters_buffer_bytes() -> u64 {
    core::mem::size_of::<GpuTransvoxelTransitionCounters>() as u64
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum TransvoxelTransitionGpuError {
    #[error(
        "Transvoxel transition extraction received {actual} scalar samples; expected {expected}"
    )]
    SampleCount { actual: usize, expected: usize },
    #[error("transition mask {0:#010b} uses bits outside the six page faces")]
    TransitionMask(u8),
    #[error("Transvoxel transition capacities must be nonzero (vertices={max_vertices}, indices={max_indices})")]
    InvalidExtractionCapacity { max_vertices: u32, max_indices: u32 },
    #[error("Transvoxel transition extraction requires {required} {name}, but the device exposes {available}")]
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
    fn worst_case_transition_budget_is_fixed_and_exact() {
        let config = TransvoxelGpuTransitionExtractorConfig::default();
        assert_eq!(TRANSVOXEL_TRANSITION_CELL_COUNT, 6_144);
        assert_eq!(TRANSVOXEL_TRANSITION_SCAN_BLOCKS, 24);
        assert_eq!(TRANSVOXEL_TRANSITION_MAX_VERTICES, 73_728);
        assert_eq!(TRANSVOXEL_TRANSITION_MAX_INDICES, 221_184);
        assert_eq!(dispatch_buffer_bytes(), 32);
        assert_eq!(sample_buffer_bytes(), 323_208);
        assert_eq!(tables_buffer_bytes(), 34_912);
        assert_eq!(cells_buffer_bytes(), 98_304);
        assert_eq!(offsets_buffer_bytes(), 98_304);
        assert_eq!(blocks_buffer_bytes(), 384);
        assert_eq!(vertices_buffer_bytes(config), 2_359_296);
        assert_eq!(indices_buffer_bytes(config), 884_736);
        assert_eq!(counters_buffer_bytes(), 48);
    }

    #[test]
    fn packed_transition_cell_preserves_case_class_counts_and_generation() {
        let cell = GpuTransvoxelTransitionCell::new(0x1fe, 0xb7, 12, 12, u64::MAX - 5);
        assert_eq!(cell.case_index(), 0x1fe);
        assert_eq!(cell.class_code(), 0xb7);
        assert_eq!(cell.class_index(), 0x37);
        assert!(cell.reverse_winding());
        assert_eq!(cell.vertex_count(), 12);
        assert_eq!(cell.triangle_count(), 12);
        assert_eq!(cell.generation(), u64::MAX - 5);
        assert!(cell.is_valid_for(u64::MAX - 5));
        assert!(!cell.is_valid_for(u64::MAX - 4));
        assert!(!GpuTransvoxelTransitionCell::default().is_valid_for(0));
    }

    #[test]
    fn zero_capacity_is_rejected() {
        assert!(matches!(
            TransvoxelGpuTransitionExtractorConfig::new(0, 1),
            Err(TransvoxelTransitionGpuError::InvalidExtractionCapacity { .. })
        ));
        assert!(matches!(
            TransvoxelGpuTransitionExtractorConfig::new(1, 0),
            Err(TransvoxelTransitionGpuError::InvalidExtractionCapacity { .. })
        ));
    }
}
