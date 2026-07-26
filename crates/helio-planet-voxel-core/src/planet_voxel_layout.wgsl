// Shared storage/uniform declarations for the planetary voxel pass.
// CellWord: density i16 in bits 0..15, material u8 in 16..23, flags u8 in 24..31.

alias CellWord = u32;

struct PlanetFrameUniform {
    planet_id: vec4<u32>,
    origin_x: vec2<u32>,
    origin_y: vec2<u32>,
    origin_z: vec2<u32>,
    frame_index: vec2<u32>,
    camera_relative_m: vec3<f32>,
    lod0_cell_size_m: f32,
    page_edge_cells: u32,
    _pad: array<u32, 3>,
}

struct GpuPageMeta {
    relative_lod0_cell_min: vec3<i32>,
    lod: u32,
    slot: u32,
    generation_low: u32,
    generation_high: u32,
    transition_mask: u32,
}

struct GpuVoxelMaterial {
    base_color_roughness: vec4<f32>,
    emissive_metalness: vec4<f32>,
}

fn cell_density(cell: CellWord) -> i32 {
    return bitcast<i32>(cell << 16u) >> 16u;
}

fn cell_material(cell: CellWord) -> u32 {
    return (cell >> 16u) & 0xffu;
}

fn cell_flags(cell: CellWord) -> u32 {
    return cell >> 24u;
}

// Reconstruct a bounded camera-local position. Absolute planet coordinates
// remain split integers on the CPU; shaders only convert the checked i32 page
// delta and page-local coordinate to f32.
fn planet_camera_local_position_m(
    frame: PlanetFrameUniform,
    page: GpuPageMeta,
    local_lod0_cell: vec3<f32>,
) -> vec3<f32> {
    return (vec3<f32>(page.relative_lod0_cell_min) + local_lod0_cell)
        * frame.lod0_cell_size_m
        - frame.camera_relative_m;
}
