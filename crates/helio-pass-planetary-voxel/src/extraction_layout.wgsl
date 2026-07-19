// Shared bounded production extraction layouts. GPU Transvoxel is the selected
// extractor; requests intentionally carry no runtime algorithm selector.

struct GpuExtractionRequest {
    page_slot: u32,
    generation_low: u32,
    generation_high: u32,
    transition_mask: u32,
    dirty_microbricks_low: u32,
    dirty_microbricks_high: u32,
    _pad: vec2<u32>,
};

struct GpuExtractionRange {
    first_vertex: u32,
    vertex_count: u32,
    first_index: u32,
    index_count: u32,
    first_meshlet: u32,
    meshlet_count: u32,
    generation_low: u32,
    generation_high: u32,
};

struct GpuTerrainVertex {
    position: vec3<f32>,
    material: u32,
    normal: vec3<f32>,
    flags: u32,
};

struct GpuTerrainMeshlet {
    first_index: u32,
    index_count: u32,
    first_vertex: u32,
    vertex_count: u32,
    bounds_offset: u32,
    generation_low: u32,
    generation_high: u32,
    _pad: u32,
};

struct GpuExtractionCounters {
    requests: atomic<u32>,
    active_cells: atomic<u32>,
    vertices: atomic<u32>,
    indices: atomic<u32>,
    meshlets: atomic<u32>,
    completed: atomic<u32>,
    stale_rejected: atomic<u32>,
    overflowed: atomic<u32>,
    vertex_overflow: atomic<u32>,
    index_overflow: atomic<u32>,
    meshlet_overflow: atomic<u32>,
    _pad: u32,
};
