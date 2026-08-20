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
    lod0_cell_min_x: vec2<u32>,
    lod0_cell_min_y: vec2<u32>,
    lod0_cell_min_z: vec2<u32>,
    lod: u32,
    slot: u32,
    generation_low: u32,
    generation_high: u32,
    transition_mask: u32,
    _pad: u32,
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

fn subtract_i64_words(left: vec2<u32>, right: vec2<u32>) -> vec2<u32> {
    let low = left.x - right.x;
    let borrow = select(0u, 1u, left.x < right.x);
    return vec2<u32>(low, left.y - right.y - borrow);
}

fn add_i64_words(left: vec2<u32>, right: vec2<u32>) -> vec2<u32> {
    let low = left.x + right.x;
    let carry = select(0u, 1u, low < left.x);
    return vec2<u32>(low, left.y + right.y + carry);
}

fn i32_to_i64_words(value: i32) -> vec2<u32> {
    return vec2<u32>(bitcast<u32>(value), select(0u, 0xffffffffu, value < 0));
}

fn i64_words_to_f32(value: vec2<u32>) -> f32 {
    let negative = (value.y & 0x80000000u) != 0u;
    if !negative {
        return f32(value.y) * 4294967296.0 + f32(value.x);
    }
    let magnitude_low = ~value.x + 1u;
    let carry = select(0u, 1u, magnitude_low == 0u);
    let magnitude_high = ~value.y + carry;
    return -(f32(magnitude_high) * 4294967296.0 + f32(magnitude_low));
}

// Subtract in canonical split-integer space before converting to f32. This
// keeps near-surface precision while allowing the same resident planet to be
// rendered from astronomical camera positions.
fn planet_camera_local_position_m(
    frame: PlanetFrameUniform,
    page: GpuPageMeta,
    local_lod0_cell: vec3<f32>,
) -> vec3<f32> {
    // Form the shared integer vertex address before converting to f32. If the
    // page origin is rounded first, two adjacent pages can map their identical
    // boundary to different floats once the camera is millions of cells away.
    let local_integer = vec3<i32>(floor(local_lod0_cell));
    let local_fraction = local_lod0_cell - vec3<f32>(local_integer);
    let relative = vec3<f32>(
        i64_words_to_f32(subtract_i64_words(
            add_i64_words(page.lod0_cell_min_x, i32_to_i64_words(local_integer.x)),
            frame.origin_x,
        )),
        i64_words_to_f32(subtract_i64_words(
            add_i64_words(page.lod0_cell_min_y, i32_to_i64_words(local_integer.y)),
            frame.origin_y,
        )),
        i64_words_to_f32(subtract_i64_words(
            add_i64_words(page.lod0_cell_min_z, i32_to_i64_words(local_integer.z)),
            frame.origin_z,
        )),
    );
    return (relative + local_fraction)
        * frame.lod0_cell_size_m
        - frame.camera_relative_m;
}
