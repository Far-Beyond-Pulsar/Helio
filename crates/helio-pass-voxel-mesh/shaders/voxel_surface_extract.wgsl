// GPU surface extraction for a single brick using Marching Cubes.
// One workgroup (64 threads) per brick.
// Indexed by workgroup_id.x into the dirty_bricks list.

const MAX_VERTS: u32 = 256u;
const MAX_INDICES: u32 = 768u;
const CELLS_PER_DIM: u32 = 7u; // (8-1) cells per dimension
const TOTAL_CELLS: u32 = 343u; // 7×7×7
const WG_SIZE: u32 = 64u;
const CELLS_PER_THREAD: u32 = (TOTAL_CELLS + WG_SIZE - 1u) / WG_SIZE;

struct GpuBrickMeta {
    data_offset: u32,
    occupancy: u32,
}

struct GpuBrickMeshlet {
    vertex_offset: u32,
    index_offset: u32,
    vertex_count: u32,
    index_count: u32,
    brick_index: u32,
    volume_id: u32,
    _pad: array<u32, 2>,
}

struct DrawIndexedIndirect {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

struct DirtyBrick {
    brick_slot: u32,
    volume_id: u32,
    origin_size: vec4<f32>, // xyz = world origin, w = voxel_size
}

@group(0) @binding(0) var<storage, read> brick_meta: array<GpuBrickMeta>;
@group(0) @binding(1) var<storage, read> voxel_data: array<u32>;
@group(0) @binding(2) var<storage, read_write> vertex_buf: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> index_buf: array<u32>;
@group(0) @binding(4) var<storage, read_write> descriptors: array<GpuBrickMeshlet>;
@group(0) @binding(5) var<storage, read_write> indirect_draws: array<DrawIndexedIndirect>;
@group(0) @binding(6) var<storage, read> dirty_bricks: array<DirtyBrick>;
@group(0) @binding(7) var<storage, read_write> normal_buf: array<vec4<f32>>;

var<workgroup> wg_vertex_count: atomic<u32>;
var<workgroup> wg_index_count: atomic<u32>;

const EDGE_TABLE: array<u32, 256> = array<u32, 256>(
    0x000u, 0x109u, 0x203u, 0x30au, 0x406u, 0x50fu, 0x605u, 0x70cu,
    0x80cu, 0x905u, 0xa0fu, 0xb06u, 0xc0au, 0xd03u, 0xe09u, 0xf00u,
    0x190u, 0x099u, 0x393u, 0x29au, 0x596u, 0x49fu, 0x795u, 0x69cu,
    0x99cu, 0x895u, 0xb9fu, 0xa96u, 0xd9au, 0xc93u, 0xf99u, 0xe90u,
    0x230u, 0x339u, 0x033u, 0x13au, 0x636u, 0x73fu, 0x435u, 0x53cu,
    0xa3cu, 0xb35u, 0x83fu, 0x936u, 0xe3au, 0xf33u, 0xc39u, 0xd30u,
    0x3a0u, 0x2a9u, 0x1a3u, 0x0aau, 0x7a6u, 0x6afu, 0x5a5u, 0x4acu,
    0xbacu, 0xaa5u, 0x9afu, 0x8a6u, 0xfaau, 0xea3u, 0xda9u, 0xca0u,
    0x460u, 0x569u, 0x663u, 0x76au, 0x066u, 0x16fu, 0x265u, 0x36cu,
    0xc6cu, 0xd65u, 0xe6fu, 0xf66u, 0x86au, 0x963u, 0xa69u, 0xb60u,
    0x5f0u, 0x4f9u, 0x7f3u, 0x6fau, 0x1f6u, 0x0ffu, 0x3f5u, 0x2fcu,
    0xdfcu, 0xcf5u, 0xfffu, 0xef6u, 0x9fau, 0x8f3u, 0xbf9u, 0xaf0u,
    0x650u, 0x759u, 0x453u, 0x55au, 0x256u, 0x35fu, 0x055u, 0x15cu,
    0xe5cu, 0xf55u, 0xc5fu, 0xd56u, 0xa5au, 0xb53u, 0x859u, 0x950u,
    0x7c0u, 0x6c9u, 0x5c3u, 0x4cau, 0x3c6u, 0x2cfu, 0x1c5u, 0x0ccu,
    0xfccu, 0xec5u, 0xdcfu, 0xcc6u, 0xbcau, 0xac3u, 0x9c9u, 0x8c0u,
    0x8c0u, 0x9c9u, 0xac3u, 0xbcau, 0xcc6u, 0xdcfu, 0xec5u, 0xfccu,
    0x0ccu, 0x1c5u, 0x2cfu, 0x3c6u, 0x4cau, 0x5c3u, 0x6c9u, 0x7c0u,
    0x950u, 0x859u, 0xb53u, 0xa5au, 0xd56u, 0xc5fu, 0xf55u, 0xe5cu,
    0x15cu, 0x055u, 0x35fu, 0x256u, 0x55au, 0x453u, 0x759u, 0x650u,
    0xaf0u, 0xbf9u, 0x8f3u, 0x9fau, 0xef6u, 0xfffu, 0xcf5u, 0xdfcu,
    0x2fcu, 0x3f5u, 0x0ffu, 0x1f6u, 0x6fau, 0x7f3u, 0x4f9u, 0x5f0u,
    0xb60u, 0xa69u, 0x963u, 0x86au, 0xf66u, 0xe6fu, 0xd65u, 0xc6cu,
    0x36cu, 0x265u, 0x16fu, 0x066u, 0x76au, 0x663u, 0x569u, 0x460u,
    0xca0u, 0xda9u, 0xea3u, 0xfaau, 0x8a6u, 0x9afu, 0xaa5u, 0xbacu,
    0x4acu, 0x5a5u, 0x6afu, 0x7a6u, 0x0aau, 0x1a3u, 0x2a9u, 0x3a0u,
    0xd30u, 0xc39u, 0xf33u, 0xe3au, 0x936u, 0x83fu, 0xb35u, 0xa3cu,
    0x53cu, 0x435u, 0x73fu, 0x636u, 0x13au, 0x033u, 0x339u, 0x230u,
    0xe90u, 0xf99u, 0xc93u, 0xd9au, 0xa96u, 0xb9fu, 0x895u, 0x99cu,
    0x69cu, 0x795u, 0x49fu, 0x596u, 0x29au, 0x393u, 0x099u, 0x190u,
    0xf00u, 0xe09u, 0xd03u, 0xc0au, 0xb06u, 0xa0fu, 0x905u, 0x80cu,
    0x70cu, 0x605u, 0x50fu, 0x406u, 0x30au, 0x203u, 0x109u, 0x000u,
);

const TRI_TABLE: array<array<i32, 16>, 256> = array<array<i32, 16>, 256>(
    array<i32, 16>(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1),
    array<i32, 16>(8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1),
    array<i32, 16>(3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1),
    array<i32, 16>(4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1),
    array<i32, 16>(4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1),
    array<i32, 16>(9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1),
    array<i32, 16>(10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1),
    array<i32, 16>(5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1),
    array<i32, 16>(5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1),
    array<i32, 16>(8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1),
    array<i32, 16>(2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1),
    array<i32, 16>(2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1),
    array<i32, 16>(11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1),
    array<i32, 16>(5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1),
    array<i32, 16>(11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1),
    array<i32, 16>(11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1),
    array<i32, 16>(2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1),
    array<i32, 16>(6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1),
    array<i32, 16>(3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1),
    array<i32, 16>(6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1),
    array<i32, 16>(6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1),
    array<i32, 16>(8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1),
    array<i32, 16>(7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1),
    array<i32, 16>(3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1),
    array<i32, 16>(0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1),
    array<i32, 16>(9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1),
    array<i32, 16>(8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1),
    array<i32, 16>(5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1),
    array<i32, 16>(0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1),
    array<i32, 16>(6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1),
    array<i32, 16>(10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1),
    array<i32, 16>(1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1),
    array<i32, 16>(0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1),
    array<i32, 16>(3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1),
    array<i32, 16>(6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1),
    array<i32, 16>(9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1),
    array<i32, 16>(8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1),
    array<i32, 16>(3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1),
    array<i32, 16>(10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1),
    array<i32, 16>(10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1),
    array<i32, 16>(2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1),
    array<i32, 16>(7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1),
    array<i32, 16>(2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1),
    array<i32, 16>(1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1),
    array<i32, 16>(11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1),
    array<i32, 16>(8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1),
    array<i32, 16>(0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1),
    array<i32, 16>(7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1),
    array<i32, 16>(7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1),
    array<i32, 16>(10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1),
    array<i32, 16>(0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1),
    array<i32, 16>(7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1),
    array<i32, 16>(6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1),
    array<i32, 16>(4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1),
    array<i32, 16>(10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1),
    array<i32, 16>(8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1),
    array<i32, 16>(1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1),
    array<i32, 16>(10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1),
    array<i32, 16>(10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1),
    array<i32, 16>(9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1),
    array<i32, 16>(7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1),
    array<i32, 16>(3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1),
    array<i32, 16>(7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1),
    array<i32, 16>(3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1),
    array<i32, 16>(6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1),
    array<i32, 16>(9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1),
    array<i32, 16>(1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1),
    array<i32, 16>(4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1),
    array<i32, 16>(7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1),
    array<i32, 16>(6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1),
    array<i32, 16>(0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1),
    array<i32, 16>(6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1),
    array<i32, 16>(0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1),
    array<i32, 16>(11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1),
    array<i32, 16>(6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1),
    array<i32, 16>(5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1),
    array<i32, 16>(9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1),
    array<i32, 16>(1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1),
    array<i32, 16>(10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1),
    array<i32, 16>(0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1),
    array<i32, 16>(11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1),
    array<i32, 16>(9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1),
    array<i32, 16>(7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1),
    array<i32, 16>(2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1),
    array<i32, 16>(9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1),
    array<i32, 16>(9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1),
    array<i32, 16>(1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1),
    array<i32, 16>(0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1),
    array<i32, 16>(10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1),
    array<i32, 16>(2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1),
    array<i32, 16>(0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1),
    array<i32, 16>(0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1),
    array<i32, 16>(9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1),
    array<i32, 16>(5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1),
    array<i32, 16>(5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1),
    array<i32, 16>(8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1),
    array<i32, 16>(9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1),
    array<i32, 16>(1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1),
    array<i32, 16>(3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1),
    array<i32, 16>(4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1),
    array<i32, 16>(9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1),
    array<i32, 16>(11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1),
    array<i32, 16>(2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1),
    array<i32, 16>(9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1),
    array<i32, 16>(3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1),
    array<i32, 16>(1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1),
    array<i32, 16>(4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1),
    array<i32, 16>(0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1),
    array<i32, 16>(1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    array<i32, 16>(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
);

fn edge_vertex(edge: u32) -> vec3<f32> {
    let edge_mid = array<vec3<f32>, 12>(
        vec3<f32>(0.5, 0.0, 0.0),
        vec3<f32>(1.0, 0.5, 0.0),
        vec3<f32>(0.5, 1.0, 0.0),
        vec3<f32>(0.0, 0.5, 0.0),
        vec3<f32>(0.5, 0.0, 1.0),
        vec3<f32>(1.0, 0.5, 1.0),
        vec3<f32>(0.5, 1.0, 1.0),
        vec3<f32>(0.0, 0.5, 1.0),
        vec3<f32>(0.0, 0.0, 0.5),
        vec3<f32>(1.0, 0.0, 0.5),
        vec3<f32>(1.0, 1.0, 0.5),
        vec3<f32>(0.0, 1.0, 0.5),
    );
    return edge_mid[edge];
}

fn read_voxel(data_offset: u32, x: u32, y: u32, z: u32) -> u32 {
    let linear = z * 64u + y * 8u + x;
    let word_idx = data_offset + linear / 4u;
    let byte_in_word = linear % 4u;
    return (voxel_data[word_idx] >> (byte_in_word * 8u)) & 0xFFu;
}

fn read_voxel_f32(data_offset: u32, x: u32, y: u32, z: u32) -> f32 {
    return select(-1.0, 1.0, read_voxel(data_offset, x, y, z) > 0u);
}

// Clamps a central-difference sample coordinate to the brick's valid [0,7]
// range — without this, sampling at voxel 0 or 7 underflows/overflows the u32
// coordinate and reads garbage (wrapped-around or adjacent-brick) data.
fn clamped_voxel(v: i32) -> u32 {
    return u32(clamp(v, 0, 7));
}

fn compute_normal(data_offset: u32, cx: u32, cy: u32, cz: u32, vs: f32) -> vec3<f32> {
    let icx = i32(cx);
    let icy = i32(cy);
    let icz = i32(cz);
    let sx = read_voxel_f32(data_offset, clamped_voxel(icx + 1), cy, cz) - read_voxel_f32(data_offset, clamped_voxel(icx - 1), cy, cz);
    let sy = read_voxel_f32(data_offset, cx, clamped_voxel(icy + 1), cz) - read_voxel_f32(data_offset, cx, clamped_voxel(icy - 1), cz);
    let sz = read_voxel_f32(data_offset, cx, cy, clamped_voxel(icz + 1)) - read_voxel_f32(data_offset, cx, cy, clamped_voxel(icz - 1));
    let n = vec3<f32>(sx, sy, sz);
    if length(n) < 0.001 {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    return normalize(n);
}

@compute @workgroup_size(WG_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let brick = dirty_bricks[wg_id.x];
    let brick_slot = brick.brick_slot;
    let volume_id = brick.volume_id;
    let origin = brick.origin_size.xyz;
    let vs = brick.origin_size.w;

    if lid.x == 0u {
        atomicStore(&wg_vertex_count, 0u);
        atomicStore(&wg_index_count, 0u);
    }
    workgroupBarrier();

    let brick_meta_entry = brick_meta[brick_slot];
    let data_offset = brick_meta_entry.data_offset;

    let thread_first = lid.x * CELLS_PER_THREAD;
    let thread_last = min(thread_first + CELLS_PER_THREAD, TOTAL_CELLS);

    for (var cell_linear = thread_first; cell_linear < thread_last; cell_linear++) {
        let cz = cell_linear / (CELLS_PER_DIM * CELLS_PER_DIM);
        let cy = (cell_linear / CELLS_PER_DIM) % CELLS_PER_DIM;
        let cx = cell_linear % CELLS_PER_DIM;

        var corner: array<u32, 8>;
        corner[0] = read_voxel(data_offset, cx,     cy,     cz);
        corner[1] = read_voxel(data_offset, cx + 1, cy,     cz);
        corner[2] = read_voxel(data_offset, cx + 1, cy + 1, cz);
        corner[3] = read_voxel(data_offset, cx,     cy + 1, cz);
        corner[4] = read_voxel(data_offset, cx,     cy,     cz + 1);
        corner[5] = read_voxel(data_offset, cx + 1, cy,     cz + 1);
        corner[6] = read_voxel(data_offset, cx + 1, cy + 1, cz + 1);
        corner[7] = read_voxel(data_offset, cx,     cy + 1, cz + 1);

        var cube_index: u32 = 0u;
        for (var i = 0u; i < 8u; i++) {
            if corner[i] != 0u {
                cube_index |= 1u << i;
            }
        }

        if cube_index == 0u || cube_index == 0xFFu {
            continue;
        }

        let edge_mask = EDGE_TABLE[cube_index];
        if edge_mask == 0u {
            continue;
        }

        let tri_entry = TRI_TABLE[cube_index];
        var tri_count: u32 = 0u;
        var edge_indices: array<i32, 15>;
        for (var t = 0u; t < 15u; t++) {
            let e = tri_entry[t];
            if e == -1 {
                break;
            }
            edge_indices[t] = e;
            tri_count++;
        }

        let num_indices = tri_count;
        if num_indices == 0u {
            continue;
        }

        let vert_base = atomicAdd(&wg_vertex_count, num_indices);
        let index_base = atomicAdd(&wg_index_count, num_indices);

        if vert_base + num_indices > MAX_VERTS || index_base + num_indices > MAX_INDICES {
            continue;
        }

        let brick_vert_offset = brick_slot * MAX_VERTS;
        let brick_idx_offset = brick_slot * MAX_INDICES;
        let cell_world = vec3<f32>(f32(cx), f32(cy), f32(cz)) * vs + origin;

        var material: u32 = 0u;
        for (var i = 0u; i < 8u; i++) {
            if corner[i] != 0u {
                material = corner[i];
                break;
            }
        }

        for (var i = 0u; i < tri_count; i++) {
            let edge_idx = edge_indices[i];
            if edge_idx < 0 || edge_idx >= 12 {
                continue;
            }
            let local_pos = edge_vertex(u32(edge_idx));
            let world_pos = cell_world + local_pos * vs;
            let vi = vert_base + i;
            vertex_buf[brick_vert_offset + vi] = vec4<f32>(world_pos, f32(material));
            // Compute normal via central differences on the material field.
            // local_pos components are in {0.0, 0.5, 1.0} (edge_vertex midpoints);
            // round to the nearest corner voxel (0 or 1) — NOT scaled by the
            // brick size, which would read voxels far outside this brick.
            let nx = u32(round(local_pos.x));
            let ny = u32(round(local_pos.y));
            let nz = u32(round(local_pos.z));
            let n = compute_normal(data_offset, cx + nx, cy + ny, cz + nz, vs);
            normal_buf[brick_vert_offset + vi] = vec4<f32>(n, 0.0);
        }

        for (var i = 0u; i < tri_count; i++) {
            let ii = index_base + i;
            index_buf[brick_idx_offset + ii] = vert_base + i;
        }
    }

    workgroupBarrier();

    if lid.x == 0u {
        let vc = min(atomicLoad(&wg_vertex_count), MAX_VERTS);
        let ic = min(atomicLoad(&wg_index_count), MAX_INDICES);

        let brick_vert_offset = brick_slot * MAX_VERTS;
        let brick_idx_offset = brick_slot * MAX_INDICES;

        descriptors[brick_slot] = GpuBrickMeshlet(
            brick_vert_offset,
            brick_idx_offset,
            vc,
            ic,
            brick_slot,
            volume_id,
            array<u32, 2>(0u, 0u),
        );

        let has_geom = select(0u, 1u, vc > 0u && ic > 0u);
        indirect_draws[brick_slot] = DrawIndexedIndirect(
            ic,
            has_geom,
            brick_idx_offset,
            i32(brick_vert_offset),
            0u,
        );
    }
}
