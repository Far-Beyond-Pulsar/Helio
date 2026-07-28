enable wgpu_int16;

// Software rasterize compute shader (Phase 5).
//
// One workgroup per 8×8 pixel tile.  Each thread handles one pixel and
// iterates over all meshlets + triangles in the tile, keeping the nearest
// depth.  Writes three visibility buffers:
//   - visibility_depth:   f32 bits of the nearest depth (non-atomic)
//   - visibility_data:    meshlet_id|(triangle_id<<22) with bit-31 valid flag
//   - visibility_instance: instance_index of the winning meshlet

struct Camera {
    view:           mat4x4<f32>,
    proj:           mat4x4<f32>,
    view_proj:      mat4x4<f32>,
    view_proj_inv:  mat4x4<f32>,
    position_near:  vec4<f32>,
    forward_far:    vec4<f32>,
    jitter_frame:   vec4<f32>,
    prev_view_proj: mat4x4<f32>,
}

struct MeshletEntry {
    center:               vec3<f32>,
    radius:               f32,
    cone_apex:            vec3<f32>,
    cone_cutoff:          f32,
    cone_axis:            vec3<f32>,
    lod_error:            f32,
    packed_counts:        u32,
    meshlet_index_offset: u32,
    meshlet_vertex_offset: u32,
    parent_cluster_id:    u32,
}

struct CullUniforms {
    object_count:           u32,
    screen_width:           u32,
    screen_height:          u32,
    hiz_mip_count:          u32,
    draw_capacity:          u32,
    lod_error_threshold_px: f32,
    object_dispatch_width:  u32,
    work_item_count:        u32,
    work_dispatch_width:    u32,
    hiz_valid:              u32,
    _pad1:                  u32,
    _pad2:                  u32,
}

struct InstanceData {
    transform:    mat4x4<f32>,
    normal_mat_0: vec4<f32>,
    normal_mat_1: vec4<f32>,
    normal_mat_2: vec4<f32>,
    bounds:       vec4<f32>,
    mesh_id:      u32,
    material_id:  u32,
    flags:        u32,
    _pad:         u32,
}

struct GpuMeshletVertex {
    position:       vec3<f32>,
    bitangent_sign: f32,
    tex_coords0:    vec2<f32>,
    tex_coords1:    vec2<f32>,
    normal:         u32,
    tangent:        u32,
}

@group(0) @binding(0)  var<storage, read>     tile_counts:        array<u32>;
@group(0) @binding(1)  var<storage, read>     tile_meshlet_ids:   array<u32>;
@group(0) @binding(2)  var<storage, read>     tile_instance_ids:  array<u32>;
@group(0) @binding(3)  var<storage, read>     meshlets:           array<MeshletEntry>;
@group(0) @binding(4)  var<storage, read>     meshlet_vertices:   array<GpuMeshletVertex>;
@group(0) @binding(5)  var<storage, read>     meshlet_indices:    array<u16>;
@group(0) @binding(6)  var<storage, read_write> visibility_depth:  array<u32>;
@group(0) @binding(7)  var<storage, read_write> visibility_data:   array<u32>;
@group(0) @binding(8)  var<storage, read_write> visibility_instance: array<u32>;
@group(0) @binding(9)  var<uniform>           camera:             Camera;
@group(0) @binding(10) var<uniform>           cull_uni:           CullUniforms;
@group(0) @binding(11) var<storage, read>     instances:          array<InstanceData>;

const TILE_SIZE_X: u32 = 8u;
const TILE_SIZE_Y: u32 = 8u;
const MAX_MESHLETS_PER_TILE: u32 = 64u;
const VIS_VALID_BIT: u32 = 2147483648u;

fn edge_function(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> f32 {
    return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}

@compute @workgroup_size(8, 8, 1)
fn cs_rasterize(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let tile_grid_x = (cull_uni.screen_width + TILE_SIZE_X - 1u) / TILE_SIZE_X;
    let tile_grid_y = (cull_uni.screen_height + TILE_SIZE_Y - 1u) / TILE_SIZE_Y;

    if wg_id.x >= tile_grid_x || wg_id.y >= tile_grid_y { return; }

    let tile_x = wg_id.x;
    let tile_y = wg_id.y;
    let tile_idx = tile_y * tile_grid_x + tile_x;

    let pixel_x = tile_x * TILE_SIZE_X + local_id.x;
    let pixel_y = tile_y * TILE_SIZE_Y + local_id.y;

    if pixel_x >= cull_uni.screen_width || pixel_y >= cull_uni.screen_height { return; }

    let pixel_idx = pixel_y * cull_uni.screen_width + pixel_x;
    let p = vec2<f32>(f32(pixel_x) + 0.5, f32(pixel_y) + 0.5);

    var best_depth_bits = 0x7F7FFFFFu;
    var best_data: u32 = 0u;
    var best_instance: u32 = 0u;

    let meshlet_count = tile_counts[tile_idx];
    let tile_base = tile_idx * MAX_MESHLETS_PER_TILE;

    for (var m = 0u; m < min(meshlet_count, MAX_MESHLETS_PER_TILE); m++) {
        let meshlet_id = tile_meshlet_ids[tile_base + m];
        if meshlet_id >= arrayLength(&meshlets) { continue; }

        let instance_id = tile_instance_ids[tile_base + m];
        if instance_id >= arrayLength(&instances) { continue; }

        let meshlet = meshlets[meshlet_id];
        let inst = instances[instance_id];

        let tri_count = meshlet.packed_counts >> 16u;
        let vert_offset = meshlet.meshlet_vertex_offset;
        let idx_offset = meshlet.meshlet_index_offset;

        for (var t = 0u; t < tri_count; t++) {
            let idx_base = idx_offset + t * 3u;
            let i0 = u32(meshlet_indices[idx_base]);
            let i1 = u32(meshlet_indices[idx_base + 1u]);
            let i2 = u32(meshlet_indices[idx_base + 2u]);

            let v0 = meshlet_vertices[vert_offset + i0];
            let v1 = meshlet_vertices[vert_offset + i1];
            let v2 = meshlet_vertices[vert_offset + i2];

            let clip0 = camera.view_proj * (inst.transform * vec4<f32>(v0.position, 1.0));
            let clip1 = camera.view_proj * (inst.transform * vec4<f32>(v1.position, 1.0));
            let clip2 = camera.view_proj * (inst.transform * vec4<f32>(v2.position, 1.0));

            if clip0.w <= 0.0 || clip1.w <= 0.0 || clip2.w <= 0.0 { continue; }

            let ndc0 = clip0.xyz / clip0.w;
            let ndc1 = clip1.xyz / clip1.w;
            let ndc2 = clip2.xyz / clip2.w;

            if (ndc0.x < -1.0 && ndc1.x < -1.0 && ndc2.x < -1.0) ||
               (ndc0.x > 1.0 && ndc1.x > 1.0 && ndc2.x > 1.0) ||
               (ndc0.y < -1.0 && ndc1.y < -1.0 && ndc2.y < -1.0) ||
               (ndc0.y > 1.0 && ndc1.y > 1.0 && ndc2.y > 1.0) { continue; }

            let s0 = vec2<f32>((ndc0.x * 0.5 + 0.5) * f32(cull_uni.screen_width),
                               (ndc0.y * -0.5 + 0.5) * f32(cull_uni.screen_height));
            let s1 = vec2<f32>((ndc1.x * 0.5 + 0.5) * f32(cull_uni.screen_width),
                               (ndc1.y * -0.5 + 0.5) * f32(cull_uni.screen_height));
            let s2 = vec2<f32>((ndc2.x * 0.5 + 0.5) * f32(cull_uni.screen_width),
                               (ndc2.y * -0.5 + 0.5) * f32(cull_uni.screen_height));

            let ew0 = edge_function(s1, s2, p);
            let ew1 = edge_function(s2, s0, p);
            let ew2 = edge_function(s0, s1, p);

            if ew0 < 0.0 || ew1 < 0.0 || ew2 < 0.0 { continue; }

            let area = ew0 + ew1 + ew2;
            if area <= 0.0 { continue; }

            let bary_u = ew0 / area;
            let bary_v = ew1 / area;
            let bary_w = ew2 / area;

            let depth = ndc0.z * bary_u + ndc1.z * bary_v + ndc2.z * bary_w;
            if depth < 0.0 || depth > 1.0 { continue; }

            let depth_bits = bitcast<u32>(depth);
            if depth_bits < best_depth_bits {
                best_depth_bits = depth_bits;
                best_data = meshlet_id | (t << 22u) | VIS_VALID_BIT;
                best_instance = instance_id;
            }
        }
    }

    if best_data != 0u {
        visibility_depth[pixel_idx] = best_depth_bits;
        visibility_data[pixel_idx] = best_data;
        visibility_instance[pixel_idx] = best_instance;
    }
}
