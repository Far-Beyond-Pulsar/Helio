use crate::{
    transvoxel::generated, GpuTerrainVertex, GpuTransvoxelDispatch, TransvoxelGpuClassifier,
    TransvoxelGpuError, TRANSVOXEL_CLASSIFY_WORKGROUPS, TRANSVOXEL_EMIT_WGSL,
    TRANSVOXEL_SCAN_BLOCKS, TRANSVOXEL_SCAN_WORKGROUP_SIZE,
};
use bytemuck::{Pod, Zeroable};
use helio_planet_voxel_core::{CellWord, PAGE_CELL_COUNT};
use wgpu::util::DeviceExt;

const REGULAR_VERTEX_TABLE_VALUES: usize = 256 * 12;
const REGULAR_TOPOLOGY_TABLE_VALUES: usize = 16 * 15;
const REGULAR_TABLE_VALUES: usize = REGULAR_VERTEX_TABLE_VALUES + REGULAR_TOPOLOGY_TABLE_VALUES;

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuTransvoxelCellOffset {
    pub first_vertex: u32,
    pub first_index: u32,
    pub generation_low: u32,
    pub generation_high: u32,
}

impl GpuTransvoxelCellOffset {
    pub const fn generation(self) -> u64 {
        self.generation_low as u64 | ((self.generation_high as u64) << 32)
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuTransvoxelScanBlock {
    pub vertex_count: u32,
    pub index_count: u32,
    pub first_vertex: u32,
    pub first_index: u32,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuTransvoxelEmissionCounters {
    pub required_vertices: u32,
    pub required_indices: u32,
    pub emitted_vertices: u32,
    pub emitted_indices: u32,
    pub vertex_overflow: u32,
    pub index_overflow: u32,
    pub completed: u32,
    pub _pad: u32,
}

impl GpuTransvoxelEmissionCounters {
    pub const fn overflowed(self) -> bool {
        self.vertex_overflow != 0 || self.index_overflow != 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransvoxelGpuExtractorConfig {
    pub max_vertices: u32,
    pub max_indices: u32,
}

impl TransvoxelGpuExtractorConfig {
    pub fn new(max_vertices: u32, max_indices: u32) -> Result<Self, TransvoxelGpuError> {
        if max_vertices == 0 || max_indices == 0 {
            return Err(TransvoxelGpuError::InvalidExtractionCapacity {
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

impl Default for TransvoxelGpuExtractorConfig {
    fn default() -> Self {
        Self {
            max_vertices: 393_216,
            max_indices: 491_520,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransvoxelExtractorResourceStats {
    pub buffers: u32,
    pub allocated_bytes: u64,
}

/// Deterministic regular-cell allocation and emission built on the classifier.
/// Prefix sums give every cell a stable range; capacity overflow suppresses the
/// entire emission rather than publishing a partial mesh.
pub struct TransvoxelGpuExtractor {
    classifier: TransvoxelGpuClassifier,
    scan_cells_pipeline: wgpu::ComputePipeline,
    scan_blocks_pipeline: wgpu::ComputePipeline,
    emit_pipeline: wgpu::ComputePipeline,
    scan_cells_bind_group: wgpu::BindGroup,
    scan_blocks_bind_group: wgpu::BindGroup,
    emit_bind_group: wgpu::BindGroup,
    offsets_buffer: wgpu::Buffer,
    blocks_buffer: wgpu::Buffer,
    vertices_buffer: wgpu::Buffer,
    indices_buffer: wgpu::Buffer,
    counters_buffer: wgpu::Buffer,
    config: TransvoxelGpuExtractorConfig,
    resource_stats: TransvoxelExtractorResourceStats,
}

impl TransvoxelGpuExtractor {
    pub fn new(
        device: &wgpu::Device,
        config: TransvoxelGpuExtractorConfig,
    ) -> Result<Self, TransvoxelGpuError> {
        validate_extractor_limits(&device.limits(), config)?;
        let classifier = TransvoxelGpuClassifier::new(device)?;
        let offsets_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Planetary Transvoxel Cell Offsets"),
            size: offsets_buffer_bytes(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let blocks_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Planetary Transvoxel Scan Blocks"),
            size: blocks_buffer_bytes(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let tables = packed_regular_tables();
        let tables_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Planetary Transvoxel Emission Tables"),
            contents: bytemuck::cast_slice(&tables),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let vertices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Planetary Transvoxel Vertices"),
            size: vertices_buffer_bytes(config),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let indices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Planetary Transvoxel Indices"),
            size: indices_buffer_bytes(config),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDEX
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let counters_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Planetary Transvoxel Emission Counters"),
            size: counters_buffer_bytes(),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Planetary Transvoxel Emission Shader"),
            source: wgpu::ShaderSource::Wgsl(TRANSVOXEL_EMIT_WGSL.into()),
        });
        let scan_cells_pipeline = compute_pipeline(device, &shader, "scan_regular_cells");
        let scan_blocks_pipeline = compute_pipeline(device, &shader, "scan_regular_blocks");
        let emit_pipeline = compute_pipeline(device, &shader, "emit_regular_cells");
        let scan_cells_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Planetary Transvoxel Cell Scan Bind Group"),
            layout: &scan_cells_pipeline.get_bind_group_layout(0),
            entries: &[
                entry(0, classifier.dispatch_buffer()),
                entry(2, classifier.output_buffer()),
                entry(3, &offsets_buffer),
                entry(4, &blocks_buffer),
            ],
        });
        let scan_blocks_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Planetary Transvoxel Block Scan Bind Group"),
            layout: &scan_blocks_pipeline.get_bind_group_layout(0),
            entries: &[
                entry(0, classifier.dispatch_buffer()),
                entry(4, &blocks_buffer),
                entry(8, &counters_buffer),
            ],
        });
        let emit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Planetary Transvoxel Emission Bind Group"),
            layout: &emit_pipeline.get_bind_group_layout(0),
            entries: &[
                entry(0, classifier.dispatch_buffer()),
                entry(1, classifier.sample_buffer()),
                entry(2, classifier.output_buffer()),
                entry(3, &offsets_buffer),
                entry(4, &blocks_buffer),
                entry(5, &tables_buffer),
                entry(6, &vertices_buffer),
                entry(7, &indices_buffer),
                entry(8, &counters_buffer),
            ],
        });
        let classifier_stats = classifier.resource_stats();
        let resource_stats = TransvoxelExtractorResourceStats {
            buffers: classifier_stats.buffers + 6,
            allocated_bytes: classifier_stats.allocated_bytes
                + offsets_buffer_bytes()
                + blocks_buffer_bytes()
                + tables_buffer_bytes()
                + vertices_buffer_bytes(config)
                + indices_buffer_bytes(config)
                + counters_buffer_bytes(),
        };
        Ok(Self {
            classifier,
            scan_cells_pipeline,
            scan_blocks_pipeline,
            emit_pipeline,
            scan_cells_bind_group,
            scan_blocks_bind_group,
            emit_bind_group,
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
        samples: &[CellWord],
        generation: u64,
        dirty_microbricks: u64,
    ) -> Result<wgpu::SubmissionIndex, TransvoxelGpuError> {
        self.prepare(queue, samples, generation, dirty_microbricks)?;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Planetary Transvoxel Extraction Encoder"),
        });
        self.encode(&mut encoder);
        Ok(queue.submit([encoder.finish()]))
    }

    /// Uploads one page and its dispatch metadata without submitting work.
    /// Callers that need graph integration or timestamp queries can pair this
    /// with [`Self::encode`] in their own command encoder.
    pub fn prepare(
        &self,
        queue: &wgpu::Queue,
        samples: &[CellWord],
        generation: u64,
        dirty_microbricks: u64,
    ) -> Result<(), TransvoxelGpuError> {
        self.classifier.prepare(
            queue,
            samples,
            GpuTransvoxelDispatch::with_limits(
                generation,
                dirty_microbricks,
                self.config.max_vertices,
                self.config.max_indices,
            ),
        )
    }

    /// Encodes the complete regular-cell extraction into a caller-owned
    /// command encoder. [`Self::prepare`] must be called first.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        self.classifier.encode(encoder);
        encoder.clear_buffer(&self.counters_buffer, 0, None);
        dispatch(
            encoder,
            &self.scan_cells_pipeline,
            &self.scan_cells_bind_group,
            TRANSVOXEL_SCAN_BLOCKS,
            "Planetary Transvoxel Cell Scan",
        );
        dispatch(
            encoder,
            &self.scan_blocks_pipeline,
            &self.scan_blocks_bind_group,
            1,
            "Planetary Transvoxel Block Scan",
        );
        dispatch(
            encoder,
            &self.emit_pipeline,
            &self.emit_bind_group,
            TRANSVOXEL_CLASSIFY_WORKGROUPS,
            "Planetary Transvoxel Emission",
        );
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

    pub fn offsets_buffer(&self) -> &wgpu::Buffer {
        &self.offsets_buffer
    }

    pub fn blocks_buffer(&self) -> &wgpu::Buffer {
        &self.blocks_buffer
    }

    pub const fn config(&self) -> TransvoxelGpuExtractorConfig {
        self.config
    }

    pub const fn resource_stats(&self) -> TransvoxelExtractorResourceStats {
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

fn dispatch(
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

fn packed_regular_tables() -> Vec<u32> {
    let mut tables = Vec::with_capacity(REGULAR_TABLE_VALUES);
    for row in generated::REGULAR_VERTEX_DATA {
        tables.extend(row.map(u32::from));
    }
    for row in generated::REGULAR_CELL_VERTEX_INDEX {
        tables.extend(row.map(u32::from));
    }
    tables
}

fn validate_extractor_limits(
    limits: &wgpu::Limits,
    config: TransvoxelGpuExtractorConfig,
) -> Result<(), TransvoxelGpuError> {
    TransvoxelGpuExtractorConfig::new(config.max_vertices, config.max_indices)?;
    if limits.max_storage_buffers_per_shader_stage < 8 {
        return Err(TransvoxelGpuError::DeviceLimit {
            name: "storage buffers per shader stage",
            required: 8,
            available: u64::from(limits.max_storage_buffers_per_shader_stage),
        });
    }
    let compute_width = limits
        .max_compute_invocations_per_workgroup
        .min(limits.max_compute_workgroup_size_x);
    if compute_width < TRANSVOXEL_SCAN_WORKGROUP_SIZE {
        return Err(TransvoxelGpuError::DeviceLimit {
            name: "compute workgroup width",
            required: u64::from(TRANSVOXEL_SCAN_WORKGROUP_SIZE),
            available: u64::from(compute_width),
        });
    }
    let available = limits
        .max_buffer_size
        .min(limits.max_storage_buffer_binding_size);
    for (name, required) in [
        ("cell offsets", offsets_buffer_bytes()),
        ("scan blocks", blocks_buffer_bytes()),
        ("emission tables", tables_buffer_bytes()),
        ("terrain vertices", vertices_buffer_bytes(config)),
        ("terrain indices", indices_buffer_bytes(config)),
        ("emission counters", counters_buffer_bytes()),
    ] {
        if required > available {
            return Err(TransvoxelGpuError::DeviceLimit {
                name,
                required,
                available,
            });
        }
    }
    Ok(())
}

const fn offsets_buffer_bytes() -> u64 {
    (PAGE_CELL_COUNT * core::mem::size_of::<GpuTransvoxelCellOffset>()) as u64
}

const fn blocks_buffer_bytes() -> u64 {
    (TRANSVOXEL_SCAN_BLOCKS as usize * core::mem::size_of::<GpuTransvoxelScanBlock>()) as u64
}

const fn tables_buffer_bytes() -> u64 {
    (REGULAR_TABLE_VALUES * core::mem::size_of::<u32>()) as u64
}

const fn vertices_buffer_bytes(config: TransvoxelGpuExtractorConfig) -> u64 {
    config.max_vertices as u64 * core::mem::size_of::<GpuTerrainVertex>() as u64
}

const fn indices_buffer_bytes(config: TransvoxelGpuExtractorConfig) -> u64 {
    config.max_indices as u64 * core::mem::size_of::<u32>() as u64
}

const fn counters_buffer_bytes() -> u64 {
    core::mem::size_of::<GpuTransvoxelEmissionCounters>() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worst_case_regular_page_budget_is_exact() {
        let config = TransvoxelGpuExtractorConfig::default();
        assert_eq!(offsets_buffer_bytes(), 524_288);
        assert_eq!(blocks_buffer_bytes(), 2_048);
        assert_eq!(tables_buffer_bytes(), 13_248);
        assert_eq!(vertices_buffer_bytes(config), 12_582_912);
        assert_eq!(indices_buffer_bytes(config), 1_966_080);
        assert_eq!(counters_buffer_bytes(), 32);
    }

    #[test]
    fn zero_capacity_is_rejected() {
        assert!(matches!(
            TransvoxelGpuExtractorConfig::new(0, 1),
            Err(TransvoxelGpuError::InvalidExtractionCapacity { .. })
        ));
        assert!(matches!(
            TransvoxelGpuExtractorConfig::new(1, 0),
            Err(TransvoxelGpuError::InvalidExtractionCapacity { .. })
        ));
    }
}
