struct GpuSurfaceJob {
    slot: u32,
    transition_mask: u32,
    generation_low: u32,
    generation_high: u32,
    regular_max_vertices: u32,
    regular_max_indices: u32,
    transition_max_vertices: u32,
    transition_max_indices: u32,
}

struct GpuPageMeta {
    relative_lod0_cell_min: vec3<i32>,
    lod: u32,
    slot: u32,
    generation_low: u32,
    generation_high: u32,
    transition_mask: u32,
}

struct GpuEmissionCounters {
    required_vertices: u32,
    required_indices: u32,
    emitted_vertices: u32,
    emitted_indices: u32,
    vertex_overflow: u32,
    index_overflow: u32,
    completed: u32,
    _pad: u32,
}

struct GpuTransitionCounters {
    active_cells: u32,
    active_faces: u32,
    required_vertices: u32,
    required_indices: u32,
    emitted_vertices: u32,
    emitted_indices: u32,
    vertex_overflow: u32,
    index_overflow: u32,
    completed: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct GpuTerrainVertex {
    position: vec3<f32>,
    material: u32,
    normal: vec3<f32>,
    flags: u32,
}

struct GpuSurfaceState {
    generation_low: u32,
    generation_high: u32,
    active_bank: u32,
    valid: u32,
    regular_vertex_count: u32,
    regular_index_count: u32,
    transition_vertex_count: u32,
    transition_index_count: u32,
}

struct GpuDrawPage {
    relative_lod0_cell_min: vec3<i32>,
    lod: u32,
    camera_relative_m: vec3<f32>,
    lod0_cell_size_m: f32,
    generation_low: u32,
    generation_high: u32,
    transition_mask: u32,
    visible: u32,
}

struct GpuSurfaceFeedback {
    submitted_jobs: u32,
    published_jobs: u32,
    stale_rejections: u32,
    overflow_rejections: u32,
    incomplete_rejections: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct DrawIndexedIndirectArgs {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

@group(0) @binding(0) var<uniform> job: GpuSurfaceJob;
@group(0) @binding(1) var<storage, read> page_metadata: array<GpuPageMeta>;

fn metadata_is_current() -> bool {
    let page_meta = page_metadata[job.slot];
    return page_meta.slot == job.slot &&
        page_meta.generation_low == job.generation_low &&
        page_meta.generation_high == job.generation_high;
}

@group(0) @binding(2) var<storage, read> regular_counters: GpuEmissionCounters;
@group(0) @binding(3) var<storage, read> regular_source_vertices: array<GpuTerrainVertex>;
@group(0) @binding(4) var<storage, read> regular_source_indices: array<u32>;
@group(0) @binding(5) var<storage, read_write> surface_states: array<GpuSurfaceState>;
@group(0) @binding(6) var<storage, read_write> regular_vertices: array<GpuTerrainVertex>;
@group(0) @binding(7) var<storage, read_write> regular_indices: array<u32>;

fn regular_succeeded() -> bool {
    return regular_counters.completed != 0u &&
        regular_counters.vertex_overflow == 0u &&
        regular_counters.index_overflow == 0u && metadata_is_current();
}

@compute @workgroup_size(64)
fn copy_regular_surface(@builtin(global_invocation_id) id: vec3<u32>) {
    if !regular_succeeded() { return; }
    let next_bank = 1u - min(surface_states[job.slot].active_bank, 1u);
    let bank = job.slot * 2u + next_bank;
    if id.x < regular_counters.emitted_vertices {
        regular_vertices[bank * job.regular_max_vertices + id.x] = regular_source_vertices[id.x];
    }
    if id.x < regular_counters.emitted_indices {
        regular_indices[bank * job.regular_max_indices + id.x] = regular_source_indices[id.x];
    }
}

@group(0) @binding(8) var<storage, read> transition_counters: GpuTransitionCounters;
@group(0) @binding(9) var<storage, read> transition_source_vertices: array<GpuTerrainVertex>;
@group(0) @binding(10) var<storage, read> transition_source_indices: array<u32>;
@group(0) @binding(11) var<storage, read_write> transition_vertices: array<GpuTerrainVertex>;
@group(0) @binding(12) var<storage, read_write> transition_indices: array<u32>;

fn transition_succeeded() -> bool {
    return transition_counters.completed != 0u &&
        transition_counters.vertex_overflow == 0u &&
        transition_counters.index_overflow == 0u && metadata_is_current();
}

@compute @workgroup_size(64)
fn copy_transition_surface(@builtin(global_invocation_id) id: vec3<u32>) {
    if !transition_succeeded() { return; }
    let next_bank = 1u - min(surface_states[job.slot].active_bank, 1u);
    let bank = job.slot * 2u + next_bank;
    if id.x < transition_counters.emitted_vertices {
        transition_vertices[bank * job.transition_max_vertices + id.x] = transition_source_vertices[id.x];
    }
    if id.x < transition_counters.emitted_indices {
        transition_indices[bank * job.transition_max_indices + id.x] = transition_source_indices[id.x];
    }
}

@group(0) @binding(13) var<storage, read> draw_pages: array<GpuDrawPage>;
@group(0) @binding(14) var<storage, read_write> regular_draws: array<DrawIndexedIndirectArgs>;
@group(0) @binding(15) var<storage, read_write> transition_draws: array<DrawIndexedIndirectArgs>;
@group(0) @binding(16) var<storage, read_write> feedback: GpuSurfaceFeedback;

@compute @workgroup_size(1)
fn publish_surface() {
    feedback.submitted_jobs += 1u;
    if !metadata_is_current() {
        feedback.stale_rejections += 1u;
        return;
    }
    if regular_counters.completed == 0u || transition_counters.completed == 0u {
        feedback.incomplete_rejections += 1u;
        return;
    }
    if regular_counters.vertex_overflow != 0u ||
        regular_counters.index_overflow != 0u ||
        transition_counters.vertex_overflow != 0u ||
        transition_counters.index_overflow != 0u {
        feedback.overflow_rejections += 1u;
        return;
    }
    let old_state = surface_states[job.slot];
    let next_bank = 1u - min(old_state.active_bank, 1u);
    surface_states[job.slot] = GpuSurfaceState(
        job.generation_low,
        job.generation_high,
        next_bank,
        1u,
        regular_counters.emitted_vertices,
        regular_counters.emitted_indices,
        transition_counters.emitted_vertices,
        transition_counters.emitted_indices,
    );
    let regular_bank = job.slot * 2u + next_bank;
    regular_draws[job.slot] = DrawIndexedIndirectArgs(
        regular_counters.emitted_indices,
        0u,
        regular_bank * job.regular_max_indices,
        i32(regular_bank * job.regular_max_vertices),
        job.slot,
    );
    let transition_bank = job.slot * 2u + next_bank;
    transition_draws[job.slot] = DrawIndexedIndirectArgs(
        transition_counters.emitted_indices,
        0u,
        transition_bank * job.transition_max_indices,
        i32(transition_bank * job.transition_max_vertices),
        job.slot,
    );
    feedback.published_jobs += 1u;
}

@compute @workgroup_size(64)
fn refresh_visibility(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= arrayLength(&surface_states) { return; }
    let state = surface_states[id.x];
    let visible = select(0u, 1u, state.valid != 0u && draw_pages[id.x].visible != 0u);
    regular_draws[id.x].instance_count = visible;
    transition_draws[id.x].instance_count = visible;
}
