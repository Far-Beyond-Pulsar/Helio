// Tile binning compute shader (Phase 5: Software Rasterization).
//
// Reads the compact visible meshlet list produced by the cull shader and
// distributes each meshlet into the screen-space tiles it overlaps.
// Each thread handles one visible meshlet.

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

@group(0) @binding(0) var<storage, read> visible_meshlet_ids:   array<u32>;
@group(0) @binding(1) var<storage, read> visible_instance_ids:  array<u32>;
@group(0) @binding(2) var<storage, read> instances:             array<InstanceData>;
@group(0) @binding(3) var<storage, read> meshlets:              array<MeshletEntry>;
@group(0) @binding(4) var<uniform>       camera:                Camera;
@group(0) @binding(5) var<uniform>       cull_uni:              CullUniforms;
@group(0) @binding(6) var<storage, read_write> tile_counts:     array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> tile_meshlet_ids:    array<u32>;
@group(0) @binding(8) var<storage, read_write> tile_instance_ids:   array<u32>;

const TILE_SIZE_X: u32 = 8u;
const TILE_SIZE_Y: u32 = 8u;
const MAX_MESHLETS_PER_TILE: u32 = 64u;

@compute @workgroup_size(64)
fn cs_binning(@builtin(global_invocation_id) id: vec3<u32>) {
    let vis_count = cull_uni.draw_capacity;
    if id.x >= vis_count { return; }

    let meshlet_id = visible_meshlet_ids[id.x];
    if meshlet_id >= arrayLength(&meshlets) { return; }

    let instance_id = visible_instance_ids[id.x];
    if instance_id >= arrayLength(&instances) { return; }

    let meshlet = meshlets[meshlet_id];
    let inst = instances[instance_id];

    // Transform bounding sphere center to world space
    let center_ws = (inst.transform * vec4<f32>(meshlet.center, 1.0)).xyz;
    let world_radius = max(meshlet.radius * 1.0, 0.0);

    // Approximate world radius from the instance transform scale
    let s0 = length(inst.transform[0].xyz);
    let s1 = length(inst.transform[1].xyz);
    let s2 = length(inst.transform[2].xyz);
    let max_scale = max(max(s0, s1), s2);
    let world_radius_scaled = world_radius * max_scale;

    // Project center to clip space
    let clip_center = camera.view_proj * vec4<f32>(center_ws, 1.0);
    if clip_center.w <= 0.0 { return; }

    // NDC to screen
    let w_inv = 1.0 / clip_center.w;
    let ndc = clip_center.xyz * w_inv;
    if abs(ndc.x) > 1.0 || abs(ndc.y) > 1.0 || ndc.z < 0.0 || ndc.z > 1.0 {
        // Check if bounding sphere can still be visible even if center is outside
        // Full sphere-cull is complex; for now just check if sphere touches the frustum
    }

    // Projected sphere radius in screen pixels
    let proj_radius_x = abs(world_radius_scaled * camera.proj[0][0] * w_inv);
    let proj_radius_y = abs(world_radius_scaled * camera.proj[1][1] * w_inv);
    let screen_radius_x = proj_radius_x * f32(cull_uni.screen_width) * 0.5;
    let screen_radius_y = proj_radius_y * f32(cull_uni.screen_height) * 0.5;

    // Screen-space center
    let screen_cx = (ndc.x * 0.5 + 0.5) * f32(cull_uni.screen_width);
    let screen_cy = (ndc.y * -0.5 + 0.5) * f32(cull_uni.screen_height);

    // Compute tile bounding box
    let min_px = i32(floor(screen_cx - screen_radius_x));
    let min_py = i32(floor(screen_cy - screen_radius_y));
    let max_px = i32(ceil(screen_cx + screen_radius_x));
    let max_py = i32(ceil(screen_cy + screen_radius_y));

    let tile_x_start = max(0, min_px / i32(TILE_SIZE_X));
    let tile_y_start = max(0, min_py / i32(TILE_SIZE_Y));
    let tile_x_end = min(i32((cull_uni.screen_width + TILE_SIZE_X - 1u) / TILE_SIZE_X), (max_px + i32(TILE_SIZE_X) - 1) / i32(TILE_SIZE_X));
    let tile_y_end = min(i32((cull_uni.screen_height + TILE_SIZE_Y - 1u) / TILE_SIZE_Y), (max_py + i32(TILE_SIZE_Y) - 1) / i32(TILE_SIZE_Y));

    let tile_grid_x = (cull_uni.screen_width + TILE_SIZE_X - 1u) / TILE_SIZE_X;

    var ty = tile_y_start;
    while ty < tile_y_end {
        var tx = tile_x_start;
        while tx < tile_x_end {
            let tile_idx = u32(ty) * tile_grid_x + u32(tx);
            let slot = atomicAdd(&tile_counts[tile_idx], 1u);
            if slot < MAX_MESHLETS_PER_TILE {
                let base = tile_idx * MAX_MESHLETS_PER_TILE;
                tile_meshlet_ids[base + slot] = meshlet_id;
                tile_instance_ids[base + slot] = instance_id;
            }
            tx++;
        }
        ty++;
    }
}
