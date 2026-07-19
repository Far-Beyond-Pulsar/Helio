use crate::{
    manifold_dc_cell_topology, manifold_dc_face_pairing, GpuTerrainVertex, GpuTransvoxelCellOffset,
    GpuTransvoxelScanBlock, EXTRACTION_SAMPLE_COUNT, MANIFOLD_DC_INDICES_PER_QUAD,
    MANIFOLD_DC_MAX_GPU_VERTICES, MANIFOLD_DC_MAX_INDICES, MANIFOLD_DUAL_CONTOURING_GPU_WGSL,
};
use bytemuck::{Pod, Zeroable};
use helio_planet_voxel_core::CellWord;
use wgpu::util::DeviceExt;

pub const MANIFOLD_DC_GPU_CELL_COUNT: u32 = 35_937;
pub const MANIFOLD_DC_GPU_OWNER_COUNT: u32 = 32 * 32 * 32;
pub const MANIFOLD_DC_GPU_WORKGROUP_SIZE: u32 = 64;
pub const MANIFOLD_DC_GPU_CELL_WORKGROUPS: u32 =
    MANIFOLD_DC_GPU_CELL_COUNT.div_ceil(MANIFOLD_DC_GPU_WORKGROUP_SIZE);
pub const MANIFOLD_DC_GPU_OWNER_WORKGROUPS: u32 =
    MANIFOLD_DC_GPU_OWNER_COUNT.div_ceil(MANIFOLD_DC_GPU_WORKGROUP_SIZE);
pub const MANIFOLD_DC_GPU_SCAN_WORKGROUP_SIZE: u32 = 256;
pub const MANIFOLD_DC_GPU_CELL_SCAN_BLOCKS: u32 =
    MANIFOLD_DC_GPU_CELL_COUNT.div_ceil(MANIFOLD_DC_GPU_SCAN_WORKGROUP_SIZE);
pub const MANIFOLD_DC_GPU_OWNER_SCAN_BLOCKS: u32 =
    MANIFOLD_DC_GPU_OWNER_COUNT.div_ceil(MANIFOLD_DC_GPU_SCAN_WORKGROUP_SIZE);
pub const MANIFOLD_DC_GPU_MAX_QUADS: u32 =
    MANIFOLD_DC_MAX_INDICES as u32 / MANIFOLD_DC_INDICES_PER_QUAD as u32;

const TABLE_WORDS_PER_CASE: usize = 2;
const FACE_PAIRING_CASES: usize = 16;
const TABLE_WORD_COUNT: usize = 256 * TABLE_WORDS_PER_CASE + FACE_PAIRING_CASES;

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuManifoldDcDispatch {
    pub generation_low: u32,
    pub generation_high: u32,
    pub cell_count: u32,
    pub owner_count: u32,
    pub max_vertices: u32,
    pub max_indices: u32,
    pub cell_scan_blocks: u32,
    pub owner_scan_blocks: u32,
    pub position_quantization_steps: u32,
    pub _pad: [u32; 7],
}

impl GpuManifoldDcDispatch {
    pub const fn new(generation: u64, config: ManifoldDcGpuExtractorConfig) -> Self {
        Self {
            generation_low: generation as u32,
            generation_high: (generation >> 32) as u32,
            cell_count: MANIFOLD_DC_GPU_CELL_COUNT,
            owner_count: MANIFOLD_DC_GPU_OWNER_COUNT,
            max_vertices: config.max_vertices,
            max_indices: config.max_indices,
            cell_scan_blocks: MANIFOLD_DC_GPU_CELL_SCAN_BLOCKS,
            owner_scan_blocks: MANIFOLD_DC_GPU_OWNER_SCAN_BLOCKS,
            position_quantization_steps: 65_536,
            _pad: [0; 7],
        }
    }

    pub const fn generation(self) -> u64 {
        self.generation_low as u64 | ((self.generation_high as u64) << 32)
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuManifoldDcCell {
    pub packed_case_component_fan_vertex_counts: u32,
    pub edge_fans_low: u32,
    pub edge_fans_high: u32,
    pub fan_material_counts_low: u32,
    pub fan_material_counts_high: u32,
    pub generation_low: u32,
    pub generation_high: u32,
    pub _pad: u32,
}

impl GpuManifoldDcCell {
    const VALID: u32 = 1 << 31;

    pub const fn fixture_case(self) -> u8 {
        (self.packed_case_component_fan_vertex_counts & 0xff) as u8
    }

    pub const fn component_count(self) -> u8 {
        ((self.packed_case_component_fan_vertex_counts >> 8) & 0x07) as u8
    }

    pub const fn fan_count(self) -> u8 {
        ((self.packed_case_component_fan_vertex_counts >> 11) & 0x0f) as u8
    }

    pub const fn vertex_count(self) -> u8 {
        ((self.packed_case_component_fan_vertex_counts >> 15) & 0x1f) as u8
    }

    pub const fn edge_fan(self, edge: usize) -> Option<u8> {
        if edge >= 12 {
            return None;
        }
        let encoded = if edge < 8 {
            (self.edge_fans_low >> (edge * 4)) & 0x0f
        } else {
            (self.edge_fans_high >> ((edge - 8) * 4)) & 0x0f
        } as u8;
        if encoded == 0x0f {
            None
        } else {
            Some(encoded)
        }
    }

    pub const fn fan_material_count(self, fan: usize) -> u8 {
        if fan >= 12 {
            return 0;
        }
        if fan < 8 {
            ((self.fan_material_counts_low >> (fan * 4)) & 0x0f) as u8
        } else {
            ((self.fan_material_counts_high >> ((fan - 8) * 4)) & 0x0f) as u8
        }
    }

    pub const fn generation(self) -> u64 {
        self.generation_low as u64 | ((self.generation_high as u64) << 32)
    }

    pub const fn is_valid_for(self, generation: u64) -> bool {
        self.packed_case_component_fan_vertex_counts & Self::VALID != 0
            && self.generation() == generation
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuManifoldDcOwner {
    pub packed_active_materials: u32,
    pub topology_vertex_count: u32,
    pub generation_low: u32,
    pub generation_high: u32,
    pub _pad: [u32; 4],
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuManifoldDcQuad {
    pub qef_vertices: [u32; 4],
    pub midpoint_vertices: [u32; 4],
    pub center_vertex: u32,
    pub material: u32,
    pub generation_low: u32,
    pub generation_high: u32,
}

impl GpuManifoldDcQuad {
    pub const fn generation(self) -> u64 {
        self.generation_low as u64 | ((self.generation_high as u64) << 32)
    }

    pub const fn is_valid_for(self, generation: u64) -> bool {
        self.generation() == generation
    }
}

impl GpuManifoldDcOwner {
    pub const fn active_mask(self) -> u8 {
        (self.packed_active_materials & 0x07) as u8
    }

    pub const fn material(self, axis: usize) -> Option<u8> {
        if axis < 3 && self.active_mask() & (1 << axis) != 0 {
            Some(((self.packed_active_materials >> (8 + axis * 8)) & 0xff) as u8)
        } else {
            None
        }
    }

    pub const fn generation(self) -> u64 {
        self.generation_low as u64 | ((self.generation_high as u64) << 32)
    }

    pub const fn is_valid_for(self, generation: u64) -> bool {
        self._pad[0] != 0 && self.generation() == generation
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuManifoldDcCounters {
    pub active_cells: u32,
    pub active_components: u32,
    pub active_edges: u32,
    pub required_qef_vertices: u32,
    pub required_topology_vertices: u32,
    pub required_vertices: u32,
    pub required_indices: u32,
    pub emitted_vertices: u32,
    pub emitted_indices: u32,
    pub vertex_overflow: u32,
    pub index_overflow: u32,
    pub completed: u32,
    pub topology_error: u32,
    pub _pad: [u32; 3],
}

impl GpuManifoldDcCounters {
    pub const fn overflowed(self) -> bool {
        self.vertex_overflow != 0 || self.index_overflow != 0
    }

    pub const fn succeeded(self) -> bool {
        self.completed != 0 && !self.overflowed() && self.topology_error == 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ManifoldDcGpuExtractorConfig {
    pub max_vertices: u32,
    pub max_indices: u32,
}

impl ManifoldDcGpuExtractorConfig {
    pub fn new(max_vertices: u32, max_indices: u32) -> Result<Self, ManifoldDcGpuError> {
        if max_vertices == 0 || max_indices == 0 {
            return Err(ManifoldDcGpuError::InvalidExtractionCapacity {
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

impl Default for ManifoldDcGpuExtractorConfig {
    fn default() -> Self {
        Self {
            max_vertices: MANIFOLD_DC_MAX_GPU_VERTICES as u32,
            max_indices: MANIFOLD_DC_MAX_INDICES as u32,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ManifoldDcGpuResourceStats {
    pub buffers: u32,
    pub allocated_bytes: u64,
}

/// Fixed-capacity GPU Hermite Dual Marching Cubes classifier, bounded QEF
/// solver, compactor, and indexed topology emitter. Every dispatch processes
/// one 32^3 page plus its negative cell halo; capacity overflow suppresses the
/// complete result rather than publishing a partial surface.
pub struct ManifoldDcGpuExtractor {
    classify_cells_pipeline: wgpu::ComputePipeline,
    classify_owners_pipeline: wgpu::ComputePipeline,
    scan_cells_pipeline: wgpu::ComputePipeline,
    scan_cell_blocks_pipeline: wgpu::ComputePipeline,
    scan_owners_pipeline: wgpu::ComputePipeline,
    scan_owner_blocks_pipeline: wgpu::ComputePipeline,
    finalize_offsets_pipeline: wgpu::ComputePipeline,
    emit_vertices_pipeline: wgpu::ComputePipeline,
    emit_topology_vertices_pipeline: wgpu::ComputePipeline,
    emit_quads_pipeline: wgpu::ComputePipeline,
    emit_indices_pipeline: wgpu::ComputePipeline,
    classify_cells_bind_group: wgpu::BindGroup,
    classify_owners_bind_group: wgpu::BindGroup,
    scan_cells_bind_group: wgpu::BindGroup,
    scan_cell_blocks_bind_group: wgpu::BindGroup,
    scan_owners_bind_group: wgpu::BindGroup,
    scan_owner_blocks_bind_group: wgpu::BindGroup,
    finalize_offsets_bind_group: wgpu::BindGroup,
    emit_vertices_bind_group: wgpu::BindGroup,
    emit_topology_vertices_bind_group: wgpu::BindGroup,
    emit_quads_bind_group: wgpu::BindGroup,
    emit_indices_bind_group: wgpu::BindGroup,
    dispatch_buffer: wgpu::Buffer,
    sample_buffer: wgpu::Buffer,
    cells_buffer: wgpu::Buffer,
    cell_offsets_buffer: wgpu::Buffer,
    cell_blocks_buffer: wgpu::Buffer,
    owners_buffer: wgpu::Buffer,
    owner_offsets_buffer: wgpu::Buffer,
    owner_blocks_buffer: wgpu::Buffer,
    quads_buffer: wgpu::Buffer,
    vertices_buffer: wgpu::Buffer,
    indices_buffer: wgpu::Buffer,
    counters_buffer: wgpu::Buffer,
    config: ManifoldDcGpuExtractorConfig,
    resource_stats: ManifoldDcGpuResourceStats,
}

impl ManifoldDcGpuExtractor {
    pub fn new(
        device: &wgpu::Device,
        config: ManifoldDcGpuExtractorConfig,
    ) -> Result<Self, ManifoldDcGpuError> {
        validate_limits(&device.limits(), config)?;
        let dispatch_buffer = create_buffer(
            device,
            "Planetary Manifold DC Dispatch",
            dispatch_buffer_bytes(),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let sample_buffer = create_buffer(
            device,
            "Planetary Manifold DC Scalar Halo",
            sample_buffer_bytes(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let table_words = packed_component_tables();
        let tables_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Planetary Manifold DC Component Tables"),
            contents: bytemuck::cast_slice(&table_words),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let cells_buffer =
            storage_readback_buffer(device, "Planetary Manifold DC Cells", cells_buffer_bytes());
        let cell_offsets_buffer = storage_readback_buffer(
            device,
            "Planetary Manifold DC Cell Offsets",
            cell_offsets_buffer_bytes(),
        );
        let cell_blocks_buffer = storage_readback_buffer(
            device,
            "Planetary Manifold DC Cell Scan Blocks",
            cell_blocks_buffer_bytes(),
        );
        let owners_buffer = storage_readback_buffer(
            device,
            "Planetary Manifold DC Edge Owners",
            owners_buffer_bytes(),
        );
        let owner_offsets_buffer = storage_readback_buffer(
            device,
            "Planetary Manifold DC Owner Offsets",
            owner_offsets_buffer_bytes(),
        );
        let owner_blocks_buffer = storage_readback_buffer(
            device,
            "Planetary Manifold DC Owner Scan Blocks",
            owner_blocks_buffer_bytes(),
        );
        let quads_buffer = storage_readback_buffer(
            device,
            "Planetary Manifold DC Resolved Quads",
            quads_buffer_bytes(config),
        );
        let vertices_buffer = create_buffer(
            device,
            "Planetary Manifold DC Vertices",
            vertices_buffer_bytes(config),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_SRC,
        );
        let indices_buffer = create_buffer(
            device,
            "Planetary Manifold DC Indices",
            indices_buffer_bytes(config),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_SRC,
        );
        let counters_buffer = create_buffer(
            device,
            "Planetary Manifold DC Counters",
            counters_buffer_bytes(),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Planetary Manifold DC GPU Shader"),
            source: wgpu::ShaderSource::Wgsl(MANIFOLD_DUAL_CONTOURING_GPU_WGSL.into()),
        });
        let classify_cells_pipeline = compute_pipeline(device, &shader, "classify_cells");
        let classify_owners_pipeline = compute_pipeline(device, &shader, "classify_owners");
        let scan_cells_pipeline = compute_pipeline(device, &shader, "scan_cells");
        let scan_cell_blocks_pipeline = compute_pipeline(device, &shader, "scan_cell_blocks");
        let scan_owners_pipeline = compute_pipeline(device, &shader, "scan_owners");
        let scan_owner_blocks_pipeline = compute_pipeline(device, &shader, "scan_owner_blocks");
        let finalize_offsets_pipeline = compute_pipeline(device, &shader, "finalize_offsets");
        let emit_vertices_pipeline = compute_pipeline(device, &shader, "emit_vertices");
        let emit_topology_vertices_pipeline =
            compute_pipeline(device, &shader, "emit_topology_vertices");
        let emit_quads_pipeline = compute_pipeline(device, &shader, "emit_quads");
        let emit_indices_pipeline = compute_pipeline(device, &shader, "emit_indices");
        let classify_cells_bind_group = bind_group(
            device,
            &classify_cells_pipeline,
            "Planetary Manifold DC Classify Cells Bind Group",
            &[
                (0, &dispatch_buffer),
                (1, &sample_buffer),
                (2, &tables_buffer),
                (3, &cells_buffer),
                (11, &counters_buffer),
            ],
        );
        let classify_owners_bind_group = bind_group(
            device,
            &classify_owners_pipeline,
            "Planetary Manifold DC Classify Owners Bind Group",
            &[
                (0, &dispatch_buffer),
                (1, &sample_buffer),
                (2, &tables_buffer),
                (6, &owners_buffer),
                (11, &counters_buffer),
            ],
        );
        let scan_cells_bind_group = bind_group(
            device,
            &scan_cells_pipeline,
            "Planetary Manifold DC Scan Cells Bind Group",
            &[
                (0, &dispatch_buffer),
                (3, &cells_buffer),
                (4, &cell_offsets_buffer),
                (5, &cell_blocks_buffer),
            ],
        );
        let scan_cell_blocks_bind_group = bind_group(
            device,
            &scan_cell_blocks_pipeline,
            "Planetary Manifold DC Scan Cell Blocks Bind Group",
            &[
                (0, &dispatch_buffer),
                (5, &cell_blocks_buffer),
                (11, &counters_buffer),
            ],
        );
        let scan_owners_bind_group = bind_group(
            device,
            &scan_owners_pipeline,
            "Planetary Manifold DC Scan Owners Bind Group",
            &[
                (0, &dispatch_buffer),
                (6, &owners_buffer),
                (7, &owner_offsets_buffer),
                (8, &owner_blocks_buffer),
            ],
        );
        let scan_owner_blocks_bind_group = bind_group(
            device,
            &scan_owner_blocks_pipeline,
            "Planetary Manifold DC Scan Owner Blocks Bind Group",
            &[
                (0, &dispatch_buffer),
                (8, &owner_blocks_buffer),
                (11, &counters_buffer),
            ],
        );
        let finalize_offsets_bind_group = bind_group(
            device,
            &finalize_offsets_pipeline,
            "Planetary Manifold DC Finalize Offsets Bind Group",
            &[
                (0, &dispatch_buffer),
                (4, &cell_offsets_buffer),
                (5, &cell_blocks_buffer),
                (7, &owner_offsets_buffer),
                (8, &owner_blocks_buffer),
            ],
        );
        let emit_vertices_bind_group = bind_group(
            device,
            &emit_vertices_pipeline,
            "Planetary Manifold DC Emit Vertices Bind Group",
            &[
                (0, &dispatch_buffer),
                (1, &sample_buffer),
                (2, &tables_buffer),
                (3, &cells_buffer),
                (4, &cell_offsets_buffer),
                (9, &vertices_buffer),
                (11, &counters_buffer),
            ],
        );
        let emit_quads_bind_group = bind_group(
            device,
            &emit_quads_pipeline,
            "Planetary Manifold DC Emit Quads Bind Group",
            &[
                (0, &dispatch_buffer),
                (1, &sample_buffer),
                (2, &tables_buffer),
                (3, &cells_buffer),
                (4, &cell_offsets_buffer),
                (6, &owners_buffer),
                (7, &owner_offsets_buffer),
                (11, &counters_buffer),
                (12, &quads_buffer),
            ],
        );
        let emit_topology_vertices_bind_group = bind_group(
            device,
            &emit_topology_vertices_pipeline,
            "Planetary Manifold DC Emit Topology Vertices Bind Group",
            &[
                (0, &dispatch_buffer),
                (1, &sample_buffer),
                (2, &tables_buffer),
                (3, &cells_buffer),
                (4, &cell_offsets_buffer),
                (6, &owners_buffer),
                (7, &owner_offsets_buffer),
                (9, &vertices_buffer),
                (11, &counters_buffer),
            ],
        );
        let emit_indices_bind_group = bind_group(
            device,
            &emit_indices_pipeline,
            "Planetary Manifold DC Emit Indices Bind Group",
            &[
                (0, &dispatch_buffer),
                (10, &indices_buffer),
                (11, &counters_buffer),
                (12, &quads_buffer),
            ],
        );
        let resource_stats = ManifoldDcGpuResourceStats {
            buffers: 13,
            allocated_bytes: dispatch_buffer_bytes()
                + sample_buffer_bytes()
                + tables_buffer_bytes()
                + cells_buffer_bytes()
                + cell_offsets_buffer_bytes()
                + cell_blocks_buffer_bytes()
                + owners_buffer_bytes()
                + owner_offsets_buffer_bytes()
                + owner_blocks_buffer_bytes()
                + quads_buffer_bytes(config)
                + vertices_buffer_bytes(config)
                + indices_buffer_bytes(config)
                + counters_buffer_bytes(),
        };
        Ok(Self {
            classify_cells_pipeline,
            classify_owners_pipeline,
            scan_cells_pipeline,
            scan_cell_blocks_pipeline,
            scan_owners_pipeline,
            scan_owner_blocks_pipeline,
            finalize_offsets_pipeline,
            emit_vertices_pipeline,
            emit_topology_vertices_pipeline,
            emit_quads_pipeline,
            emit_indices_pipeline,
            classify_cells_bind_group,
            classify_owners_bind_group,
            scan_cells_bind_group,
            scan_cell_blocks_bind_group,
            scan_owners_bind_group,
            scan_owner_blocks_bind_group,
            finalize_offsets_bind_group,
            emit_vertices_bind_group,
            emit_topology_vertices_bind_group,
            emit_quads_bind_group,
            emit_indices_bind_group,
            dispatch_buffer,
            sample_buffer,
            cells_buffer,
            cell_offsets_buffer,
            cell_blocks_buffer,
            owners_buffer,
            owner_offsets_buffer,
            owner_blocks_buffer,
            quads_buffer,
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
    ) -> Result<wgpu::SubmissionIndex, ManifoldDcGpuError> {
        if samples.len() != EXTRACTION_SAMPLE_COUNT {
            return Err(ManifoldDcGpuError::SampleCount {
                actual: samples.len(),
                expected: EXTRACTION_SAMPLE_COUNT,
            });
        }
        let dispatch = GpuManifoldDcDispatch::new(generation, self.config);
        queue.write_buffer(&self.sample_buffer, 0, bytemuck::cast_slice(samples));
        queue.write_buffer(&self.dispatch_buffer, 0, bytemuck::bytes_of(&dispatch));
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Planetary Manifold DC Extraction Encoder"),
        });
        encoder.clear_buffer(&self.counters_buffer, 0, None);
        encode_dispatch(
            &mut encoder,
            &self.classify_cells_pipeline,
            &self.classify_cells_bind_group,
            MANIFOLD_DC_GPU_CELL_WORKGROUPS,
            "Planetary Manifold DC Cell Classification",
        );
        encode_dispatch(
            &mut encoder,
            &self.classify_owners_pipeline,
            &self.classify_owners_bind_group,
            MANIFOLD_DC_GPU_OWNER_WORKGROUPS,
            "Planetary Manifold DC Edge Classification",
        );
        encode_dispatch(
            &mut encoder,
            &self.scan_cells_pipeline,
            &self.scan_cells_bind_group,
            MANIFOLD_DC_GPU_CELL_SCAN_BLOCKS,
            "Planetary Manifold DC Cell Scan",
        );
        encode_dispatch(
            &mut encoder,
            &self.scan_cell_blocks_pipeline,
            &self.scan_cell_blocks_bind_group,
            1,
            "Planetary Manifold DC Cell Block Scan",
        );
        encode_dispatch(
            &mut encoder,
            &self.scan_owners_pipeline,
            &self.scan_owners_bind_group,
            MANIFOLD_DC_GPU_OWNER_SCAN_BLOCKS,
            "Planetary Manifold DC Owner Scan",
        );
        encode_dispatch(
            &mut encoder,
            &self.scan_owner_blocks_pipeline,
            &self.scan_owner_blocks_bind_group,
            1,
            "Planetary Manifold DC Owner Block Scan",
        );
        encode_dispatch(
            &mut encoder,
            &self.finalize_offsets_pipeline,
            &self.finalize_offsets_bind_group,
            MANIFOLD_DC_GPU_CELL_WORKGROUPS,
            "Planetary Manifold DC Offset Finalization",
        );
        encode_dispatch(
            &mut encoder,
            &self.emit_vertices_pipeline,
            &self.emit_vertices_bind_group,
            MANIFOLD_DC_GPU_CELL_WORKGROUPS,
            "Planetary Manifold DC Vertex Emission",
        );
        encode_dispatch(
            &mut encoder,
            &self.emit_topology_vertices_pipeline,
            &self.emit_topology_vertices_bind_group,
            MANIFOLD_DC_GPU_OWNER_WORKGROUPS,
            "Planetary Manifold DC Topology Vertex Emission",
        );
        encode_dispatch(
            &mut encoder,
            &self.emit_quads_pipeline,
            &self.emit_quads_bind_group,
            MANIFOLD_DC_GPU_OWNER_WORKGROUPS,
            "Planetary Manifold DC Quad Resolution",
        );
        encode_dispatch(
            &mut encoder,
            &self.emit_indices_pipeline,
            &self.emit_indices_bind_group,
            quad_workgroups(self.config),
            "Planetary Manifold DC Index Emission",
        );
        Ok(queue.submit([encoder.finish()]))
    }

    pub fn cells_buffer(&self) -> &wgpu::Buffer {
        &self.cells_buffer
    }

    pub fn cell_offsets_buffer(&self) -> &wgpu::Buffer {
        &self.cell_offsets_buffer
    }

    pub fn cell_blocks_buffer(&self) -> &wgpu::Buffer {
        &self.cell_blocks_buffer
    }

    pub fn owners_buffer(&self) -> &wgpu::Buffer {
        &self.owners_buffer
    }

    pub fn owner_offsets_buffer(&self) -> &wgpu::Buffer {
        &self.owner_offsets_buffer
    }

    pub fn owner_blocks_buffer(&self) -> &wgpu::Buffer {
        &self.owner_blocks_buffer
    }

    pub fn quads_buffer(&self) -> &wgpu::Buffer {
        &self.quads_buffer
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

    pub const fn config(&self) -> ManifoldDcGpuExtractorConfig {
        self.config
    }

    pub const fn resource_stats(&self) -> ManifoldDcGpuResourceStats {
        self.resource_stats
    }

    pub fn resize(&mut self, _width: u32, _height: u32) {}
}

fn packed_component_tables() -> Vec<u32> {
    let mut words = Vec::with_capacity(TABLE_WORD_COUNT);
    for fixture_case in 0..=u8::MAX {
        let topology = manifold_dc_cell_topology(fixture_case);
        let edges = topology.edge_components();
        let mut first = u32::from(topology.component_count());
        for (edge, component) in edges[..9].iter().copied().enumerate() {
            let encoded = if component == u8::MAX {
                7
            } else {
                u32::from(component)
            };
            first |= encoded << (3 + edge * 3);
        }
        let mut second = 0_u32;
        for (edge, component) in edges[9..].iter().copied().enumerate() {
            let encoded = if component == u8::MAX {
                7
            } else {
                u32::from(component)
            };
            second |= encoded << (edge * 3);
        }
        words.extend([first, second]);
    }
    for pattern in 0..FACE_PAIRING_CASES as u8 {
        let mut packed = 0x0fff_u32;
        for pair in manifold_dc_face_pairing(pattern) {
            if pair[0] == u8::MAX {
                continue;
            }
            for [edge, paired] in [pair, [pair[1], pair[0]]] {
                let shift = u32::from(edge) * 3;
                packed &= !(0x07 << shift);
                packed |= u32::from(paired) << shift;
            }
        }
        words.push(packed);
    }
    words
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

fn bind_group(
    device: &wgpu::Device,
    pipeline: &wgpu::ComputePipeline,
    label: &'static str,
    buffers: &[(u32, &wgpu::Buffer)],
) -> wgpu::BindGroup {
    let entries: Vec<_> = buffers
        .iter()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding: *binding,
            resource: buffer.as_entire_binding(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    })
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

fn create_buffer(
    device: &wgpu::Device,
    label: &'static str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}

fn storage_readback_buffer(device: &wgpu::Device, label: &'static str, size: u64) -> wgpu::Buffer {
    create_buffer(
        device,
        label,
        size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    )
}

fn validate_limits(
    limits: &wgpu::Limits,
    config: ManifoldDcGpuExtractorConfig,
) -> Result<(), ManifoldDcGpuError> {
    ManifoldDcGpuExtractorConfig::new(config.max_vertices, config.max_indices)?;
    if limits.max_storage_buffers_per_shader_stage < 8 {
        return Err(ManifoldDcGpuError::DeviceLimit {
            name: "storage buffers per shader stage",
            required: 8,
            available: u64::from(limits.max_storage_buffers_per_shader_stage),
        });
    }
    if limits.max_uniform_buffers_per_shader_stage < 1 {
        return Err(ManifoldDcGpuError::DeviceLimit {
            name: "uniform buffers per shader stage",
            required: 1,
            available: u64::from(limits.max_uniform_buffers_per_shader_stage),
        });
    }
    let compute_width = limits
        .max_compute_invocations_per_workgroup
        .min(limits.max_compute_workgroup_size_x);
    if compute_width < MANIFOLD_DC_GPU_SCAN_WORKGROUP_SIZE {
        return Err(ManifoldDcGpuError::DeviceLimit {
            name: "compute workgroup width",
            required: u64::from(MANIFOLD_DC_GPU_SCAN_WORKGROUP_SIZE),
            available: u64::from(compute_width),
        });
    }
    let required_workgroups = MANIFOLD_DC_GPU_CELL_WORKGROUPS.max(quad_workgroups(config));
    if limits.max_compute_workgroups_per_dimension < required_workgroups {
        return Err(ManifoldDcGpuError::DeviceLimit {
            name: "compute workgroups per dimension",
            required: u64::from(required_workgroups),
            available: u64::from(limits.max_compute_workgroups_per_dimension),
        });
    }
    let available = limits
        .max_buffer_size
        .min(limits.max_storage_buffer_binding_size);
    for (name, required) in [
        ("manifold scalar halo", sample_buffer_bytes()),
        ("manifold component tables", tables_buffer_bytes()),
        ("manifold cells", cells_buffer_bytes()),
        ("manifold cell offsets", cell_offsets_buffer_bytes()),
        ("manifold cell scan blocks", cell_blocks_buffer_bytes()),
        ("manifold edge owners", owners_buffer_bytes()),
        ("manifold owner offsets", owner_offsets_buffer_bytes()),
        ("manifold owner scan blocks", owner_blocks_buffer_bytes()),
        ("manifold resolved quads", quads_buffer_bytes(config)),
        ("manifold vertices", vertices_buffer_bytes(config)),
        ("manifold indices", indices_buffer_bytes(config)),
        ("manifold counters", counters_buffer_bytes()),
    ] {
        if required > available {
            return Err(ManifoldDcGpuError::DeviceLimit {
                name,
                required,
                available,
            });
        }
    }
    if dispatch_buffer_bytes() > limits.max_uniform_buffer_binding_size {
        return Err(ManifoldDcGpuError::DeviceLimit {
            name: "manifold dispatch uniform",
            required: dispatch_buffer_bytes(),
            available: limits.max_uniform_buffer_binding_size,
        });
    }
    Ok(())
}

const fn dispatch_buffer_bytes() -> u64 {
    core::mem::size_of::<GpuManifoldDcDispatch>() as u64
}

const fn sample_buffer_bytes() -> u64 {
    (EXTRACTION_SAMPLE_COUNT * core::mem::size_of::<CellWord>()) as u64
}

const fn tables_buffer_bytes() -> u64 {
    (TABLE_WORD_COUNT * core::mem::size_of::<u32>()) as u64
}

const fn cells_buffer_bytes() -> u64 {
    MANIFOLD_DC_GPU_CELL_COUNT as u64 * core::mem::size_of::<GpuManifoldDcCell>() as u64
}

const fn cell_offsets_buffer_bytes() -> u64 {
    MANIFOLD_DC_GPU_CELL_COUNT as u64 * core::mem::size_of::<GpuTransvoxelCellOffset>() as u64
}

const fn cell_blocks_buffer_bytes() -> u64 {
    MANIFOLD_DC_GPU_CELL_SCAN_BLOCKS as u64 * core::mem::size_of::<GpuTransvoxelScanBlock>() as u64
}

const fn owners_buffer_bytes() -> u64 {
    MANIFOLD_DC_GPU_OWNER_COUNT as u64 * core::mem::size_of::<GpuManifoldDcOwner>() as u64
}

const fn owner_offsets_buffer_bytes() -> u64 {
    MANIFOLD_DC_GPU_OWNER_COUNT as u64 * core::mem::size_of::<GpuTransvoxelCellOffset>() as u64
}

const fn owner_blocks_buffer_bytes() -> u64 {
    MANIFOLD_DC_GPU_OWNER_SCAN_BLOCKS as u64 * core::mem::size_of::<GpuTransvoxelScanBlock>() as u64
}

const fn quad_capacity(config: ManifoldDcGpuExtractorConfig) -> u32 {
    let capacity = config.max_indices / MANIFOLD_DC_INDICES_PER_QUAD as u32;
    if capacity == 0 {
        1
    } else {
        capacity
    }
}

const fn quad_workgroups(config: ManifoldDcGpuExtractorConfig) -> u32 {
    quad_capacity(config).div_ceil(MANIFOLD_DC_GPU_WORKGROUP_SIZE)
}

const fn quads_buffer_bytes(config: ManifoldDcGpuExtractorConfig) -> u64 {
    quad_capacity(config) as u64 * core::mem::size_of::<GpuManifoldDcQuad>() as u64
}

const fn vertices_buffer_bytes(config: ManifoldDcGpuExtractorConfig) -> u64 {
    config.max_vertices as u64 * core::mem::size_of::<GpuTerrainVertex>() as u64
}

const fn indices_buffer_bytes(config: ManifoldDcGpuExtractorConfig) -> u64 {
    config.max_indices as u64 * core::mem::size_of::<u32>() as u64
}

const fn counters_buffer_bytes() -> u64 {
    core::mem::size_of::<GpuManifoldDcCounters>() as u64
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ManifoldDcGpuError {
    #[error("manifold GPU extraction received {actual} scalar samples; expected {expected}")]
    SampleCount { actual: usize, expected: usize },
    #[error("manifold GPU extraction capacities must be nonzero (vertices={max_vertices}, indices={max_indices})")]
    InvalidExtractionCapacity { max_vertices: u32, max_indices: u32 },
    #[error(
        "manifold GPU extraction requires {required} {name}, but the device exposes {available}"
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
    fn packed_component_table_round_trips_every_case() {
        let tables = packed_component_tables();
        assert_eq!(tables.len(), TABLE_WORD_COUNT);
        for fixture_case in 0..=u8::MAX {
            let topology = manifold_dc_cell_topology(fixture_case);
            let first = tables[usize::from(fixture_case) * 2];
            let second = tables[usize::from(fixture_case) * 2 + 1];
            assert_eq!((first & 0x07) as u8, topology.component_count());
            for edge in 0..12 {
                let encoded = if edge < 9 {
                    (first >> (3 + edge * 3)) & 0x07
                } else {
                    (second >> ((edge - 9) * 3)) & 0x07
                };
                assert_eq!(
                    (encoded != 7).then_some(encoded as u8),
                    topology.component_for_edge(edge),
                    "case {fixture_case} edge {edge}"
                );
            }
        }
        for pattern in 0..FACE_PAIRING_CASES as u8 {
            let packed = tables[256 * TABLE_WORDS_PER_CASE + usize::from(pattern)];
            let expected = manifold_dc_face_pairing(pattern);
            for edge in 0..4_u8 {
                let encoded = ((packed >> (u32::from(edge) * 3)) & 0x07) as u8;
                let paired = expected
                    .iter()
                    .copied()
                    .find_map(|pair| {
                        if pair[0] == edge {
                            Some(pair[1])
                        } else if pair[1] == edge {
                            Some(pair[0])
                        } else {
                            None
                        }
                    })
                    .unwrap_or(7);
                assert_eq!(encoded, paired, "face pattern {pattern} edge {edge}");
            }
        }
    }

    #[test]
    fn worst_case_gpu_budget_is_fixed_and_exact() {
        let config = ManifoldDcGpuExtractorConfig::default();
        assert_eq!(MANIFOLD_DC_GPU_CELL_WORKGROUPS, 562);
        assert_eq!(MANIFOLD_DC_GPU_OWNER_WORKGROUPS, 512);
        assert_eq!(MANIFOLD_DC_GPU_CELL_SCAN_BLOCKS, 141);
        assert_eq!(MANIFOLD_DC_GPU_OWNER_SCAN_BLOCKS, 128);
        assert_eq!(MANIFOLD_DC_GPU_MAX_QUADS, 98_304);
        assert_eq!(dispatch_buffer_bytes(), 64);
        assert_eq!(sample_buffer_bytes(), 157_216);
        assert_eq!(tables_buffer_bytes(), 2_112);
        assert_eq!(cells_buffer_bytes(), 1_149_984);
        assert_eq!(cell_offsets_buffer_bytes(), 574_992);
        assert_eq!(cell_blocks_buffer_bytes(), 2_256);
        assert_eq!(owners_buffer_bytes(), 1_048_576);
        assert_eq!(owner_offsets_buffer_bytes(), 524_288);
        assert_eq!(owner_blocks_buffer_bytes(), 2_048);
        assert_eq!(quads_buffer_bytes(config), 4_718_592);
        assert_eq!(vertices_buffer_bytes(config), 28_311_552);
        assert_eq!(indices_buffer_bytes(config), 9_437_184);
        assert_eq!(counters_buffer_bytes(), 64);
        assert_eq!(quad_workgroups(config), 1_536);
        assert_eq!(
            dispatch_buffer_bytes()
                + sample_buffer_bytes()
                + tables_buffer_bytes()
                + cells_buffer_bytes()
                + cell_offsets_buffer_bytes()
                + cell_blocks_buffer_bytes()
                + owners_buffer_bytes()
                + owner_offsets_buffer_bytes()
                + owner_blocks_buffer_bytes()
                + quads_buffer_bytes(config)
                + vertices_buffer_bytes(config)
                + indices_buffer_bytes(config)
                + counters_buffer_bytes(),
            45_928_928
        );
    }

    #[test]
    fn zero_capacity_is_rejected() {
        assert!(matches!(
            ManifoldDcGpuExtractorConfig::new(0, 1),
            Err(ManifoldDcGpuError::InvalidExtractionCapacity { .. })
        ));
        assert!(matches!(
            ManifoldDcGpuExtractorConfig::new(1, 0),
            Err(ManifoldDcGpuError::InvalidExtractionCapacity { .. })
        ));
    }
}
