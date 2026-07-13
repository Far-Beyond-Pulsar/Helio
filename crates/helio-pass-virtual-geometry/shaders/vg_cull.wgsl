// Virtual geometry culling compute shader.
//
// Stage one assigns one lane per object for conservative object culling and a
// whole-object LOD decision. Stage two assigns one 64-lane workgroup to each
// immutable meshlet span, so very large objects scale across the GPU instead of
// serialising all their meshlets through one workgroup.

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

/// Mirrors GpuMeshletEntry (Rust, 64 bytes).
struct MeshletEntry {
    center:         vec3<f32>,
    radius:         f32,
    cone_apex:      vec3<f32>,
    cone_cutoff:    f32,
    cone_axis:      vec3<f32>,
    lod_error:      f32,
    first_index:    u32,
    index_count:    u32,
    vertex_offset:  i32,
    instance_index: u32,
}

/// Mirrors GpuVgObject (Rust, 128 bytes).
struct VgObjectData {
    instance_index:       u32,
    lod_count:            u32,
    max_meshlet_count:    u32,
    selected_lod_plus_one: u32,
    local_bounds:         vec4<f32>,
    lod_errors:           array<f32, 8>,
    lod_first_meshlets:   array<u32, 8>,
    lod_meshlet_counts:   array<u32, 8>,
}

/// Mirrors GpuInstanceData (Rust, 144 bytes).
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

struct InstanceCullData {
    max_scale:          f32,
    min_scale:          f32,
    cone_cull_enabled:  u32,
    valid_transform:    u32,
}

/// Mirrors wgpu::util::DrawIndexedIndirectArgs (20 bytes).
struct DrawIndexedIndirect {
    index_count:    u32,
    instance_count: u32,
    first_index:    u32,
    base_vertex:    i32,
    first_instance: u32,
}

/// Mirrors GpuVgDraw (Rust, 16 bytes).
struct VgDrawMetadata {
    instance_index: u32,
    meshlet_index:  u32,
    lod_level:      u32,
    reserved:       u32,
}

struct VgWorkItem {
    object_index:       u32,
    local_meshlet_base: u32,
}

/// Mirrors CullUniforms (Rust, 48 bytes).
struct CullUniforms {
    object_count:          u32,
    screen_width:          u32,
    screen_height:         u32,
    hiz_mip_count:         u32,
    draw_capacity:         u32,
    lod_error_threshold_px: f32,
    object_dispatch_width: u32,
    work_item_count:       u32,
    work_dispatch_width:   u32,
    _pad0:                 u32,
    _pad1:                 u32,
    _pad2:                 u32,
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> cull_uni: CullUniforms;
@group(0) @binding(2) var<storage, read> meshlets: array<MeshletEntry>;
@group(0) @binding(3) var<storage, read_write> objects: array<VgObjectData>;
@group(0) @binding(4) var<storage, read> instances: array<InstanceData>;
@group(0) @binding(5) var<storage, read_write> indirect: array<DrawIndexedIndirect>;
@group(0) @binding(6) var<storage, read_write> draw_metadata: array<VgDrawMetadata>;
// Counter 0 is the attempted visible-draw count consumed by indirect-count
// rendering. Counter 1 records attempts rejected by the bounded output arrays.
@group(0) @binding(7) var<storage, read_write> draw_counters: array<atomic<u32>>;
@group(0) @binding(8) var hiz_tex: texture_2d<f32>;
@group(0) @binding(9) var hiz_samp: sampler;
@group(0) @binding(10) var<storage, read> instance_cull: array<InstanceCullData>;
@group(0) @binding(11) var<storage, read> work_items: array<VgWorkItem>;

var<workgroup> wg_planes: array<vec4<f32>, 6>;
var<workgroup> wg_first_meshlet: u32;
var<workgroup> wg_meshlet_count: u32;
var<workgroup> wg_instance_index: u32;
var<workgroup> wg_selected_lod: u32;
var<workgroup> wg_object_visible: u32;
var<workgroup> wg_local_meshlet_base: u32;

fn sphere_visible(center: vec3<f32>, radius: f32) -> bool {
    return (dot(wg_planes[0].xyz, center) + wg_planes[0].w >= -radius)
        && (dot(wg_planes[1].xyz, center) + wg_planes[1].w >= -radius)
        && (dot(wg_planes[2].xyz, center) + wg_planes[2].w >= -radius)
        && (dot(wg_planes[3].xyz, center) + wg_planes[3].w >= -radius)
        && (dot(wg_planes[4].xyz, center) + wg_planes[4].w >= -radius)
        && (dot(wg_planes[5].xyz, center) + wg_planes[5].w >= -radius);
}

fn cull_meshlet(meshlet_index: u32, instance_index: u32, lod_level: u32) {
    if meshlet_index >= arrayLength(&meshlets)
        || instance_index >= arrayLength(&instances)
        || instance_index >= arrayLength(&instance_cull)
    {
        return;
    }

    let m = meshlets[meshlet_index];
    let inst = instances[instance_index];
    let inst_cull = instance_cull[instance_index];
    if inst_cull.valid_transform == 0u {
        return;
    }

    let model = inst.transform;
    let center_ws = (model * vec4<f32>(m.center, 1.0)).xyz;
    let world_radius = max(m.radius * inst_cull.max_scale, 0.0);
    let cam_to_center = center_ws - camera.position_near.xyz;

    if !sphere_visible(center_ws, world_radius) {
        return;
    }

    // Exact meshoptimizer perspective-cone test. The normal matrix is safe
    // only for angle-preserving, non-reflected transforms; CPU precomputes the
    // conservative enable bit once per changed instance.
    let cam_dist_sq = dot(cam_to_center, cam_to_center);
    let guard_radius = world_radius * 1.5;
    if cam_dist_sq > guard_radius * guard_radius
        && m.cone_cutoff <= 1.0
        && inst_cull.cone_cull_enabled != 0u
    {
        let normal_mat = mat3x3<f32>(
            inst.normal_mat_0.xyz,
            inst.normal_mat_1.xyz,
            inst.normal_mat_2.xyz,
        );
        let cone_axis_ws = normalize(normal_mat * m.cone_axis);
        let cone_apex_ws = (model * vec4<f32>(m.cone_apex, 1.0)).xyz;
        let camera_to_apex = cone_apex_ws - camera.position_near.xyz;
        let apex_distance_sq = dot(camera_to_apex, camera_to_apex);
        if apex_distance_sq > 1.0e-12
            && dot(camera_to_apex / sqrt(apex_distance_sq), cone_axis_ws) >= m.cone_cutoff
        {
            return;
        }
    }

    // Conservative max-depth Hi-Z test. Reject only when the complete
    // projected sphere is on-screen and all four footprint corners are behind
    // existing depth at the chosen mip.
    let cull_clip = camera.view_proj * vec4<f32>(center_ws, 1.0);
    if cull_clip.w > 0.0 {
        let cull_ndc = cull_clip.xyz / cull_clip.w;
        let cull_uv = vec2<f32>(cull_ndc.x * 0.5 + 0.5, cull_ndc.y * -0.5 + 0.5);
        let nearest_view_depth = cull_clip.w - world_radius;
        if nearest_view_depth > camera.position_near.w {
            let ndc_r = max(
                abs(world_radius * camera.proj[0][0] / nearest_view_depth),
                abs(world_radius * camera.proj[1][1] / nearest_view_depth),
            );
            let uv_radius = ndc_r * 0.5;
            let uv_min = cull_uv - vec2<f32>(uv_radius);
            let uv_max = cull_uv + vec2<f32>(uv_radius);

            if all(uv_min >= vec2<f32>(0.0)) && all(uv_max <= vec2<f32>(1.0)) {
                let dist_sq = dot(cam_to_center, cam_to_center);
                var near_z = 0.0;
                if dist_sq > world_radius * world_radius {
                    let direction = cam_to_center / sqrt(dist_sq);
                    let near_ws = center_ws - direction * world_radius;
                    let near_clip = camera.view_proj * vec4<f32>(near_ws, 1.0);
                    if near_clip.w > 0.0 {
                        near_z = clamp(near_clip.z / near_clip.w, 0.0, 1.0);
                    }
                }

                let half_height = f32(cull_uni.screen_height) * 0.5;
                let diameter_px = max(ndc_r * half_height * 2.0, 1.0);
                let mip = clamp(
                    u32(ceil(log2(diameter_px))),
                    0u,
                    cull_uni.hiz_mip_count - 1u,
                );
                let hiz_00 = textureSampleLevel(hiz_tex, hiz_samp, uv_min, f32(mip)).r;
                let hiz_01 = textureSampleLevel(
                    hiz_tex,
                    hiz_samp,
                    vec2<f32>(uv_max.x, uv_min.y),
                    f32(mip),
                ).r;
                let hiz_10 = textureSampleLevel(
                    hiz_tex,
                    hiz_samp,
                    vec2<f32>(uv_min.x, uv_max.y),
                    f32(mip),
                ).r;
                let hiz_11 = textureSampleLevel(hiz_tex, hiz_samp, uv_max, f32(mip)).r;
                let hiz_depth = max(max(hiz_00, hiz_01), max(hiz_10, hiz_11));
                if near_z > hiz_depth + 1.0 / 65536.0 {
                    return;
                }
            }
        }
    }

    var command: DrawIndexedIndirect;
    command.index_count = m.index_count;
    command.instance_count = 1u;
    command.first_index = m.first_index;
    command.base_vertex = m.vertex_offset;
    let slot = atomicAdd(&draw_counters[0], 1u);
    let capacity = min(
        cull_uni.draw_capacity,
        min(arrayLength(&indirect), arrayLength(&draw_metadata)),
    );
    if slot < capacity {
        command.first_instance = slot;
        indirect[slot] = command;
        draw_metadata[slot] = VgDrawMetadata(
            instance_index,
            meshlet_index,
            lod_level,
            0u,
        );
    } else {
        atomicAdd(&draw_counters[1], 1u);
    }
}

fn publish_frustum_planes() {
    let vp = camera.view_proj;
    let p0 = vec4<f32>(vp[0][3] + vp[0][0], vp[1][3] + vp[1][0], vp[2][3] + vp[2][0], vp[3][3] + vp[3][0]);
    let p1 = vec4<f32>(vp[0][3] - vp[0][0], vp[1][3] - vp[1][0], vp[2][3] - vp[2][0], vp[3][3] - vp[3][0]);
    let p2 = vec4<f32>(vp[0][3] + vp[0][1], vp[1][3] + vp[1][1], vp[2][3] + vp[2][1], vp[3][3] + vp[3][1]);
    let p3 = vec4<f32>(vp[0][3] - vp[0][1], vp[1][3] - vp[1][1], vp[2][3] - vp[2][1], vp[3][3] - vp[3][1]);
    let p4 = vec4<f32>(vp[0][2], vp[1][2], vp[2][2], vp[3][2]);
    let p5 = vec4<f32>(vp[0][3] - vp[0][2], vp[1][3] - vp[1][2], vp[2][3] - vp[2][2], vp[3][3] - vp[3][2]);
    wg_planes[0] = p0 / length(p0.xyz);
    wg_planes[1] = p1 / length(p1.xyz);
    wg_planes[2] = p2 / length(p2.xyz);
    wg_planes[3] = p3 / length(p3.xyz);
    wg_planes[4] = p4 / length(p4.xyz);
    wg_planes[5] = p5 / length(p5.xyz);
}

@compute @workgroup_size(64)
fn cs_select_objects(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    if lane == 0u {
        publish_frustum_planes();
    }
    workgroupBarrier();

    let group_index = workgroup_id.x
        + workgroup_id.y * cull_uni.object_dispatch_width;
    let object_index = group_index * 64u + lane;
    if object_index >= cull_uni.object_count
        || object_index >= arrayLength(&objects)
    {
        return;
    }

    var selected_lod_plus_one = 0u;
    let object = objects[object_index];
    let lod_count = min(object.lod_count, 8u);
    if lod_count > 0u
        && object.instance_index < arrayLength(&instances)
        && object.instance_index < arrayLength(&instance_cull)
    {
        let inst = instances[object.instance_index];
        let derived = instance_cull[object.instance_index];
        if derived.valid_transform != 0u {
            let center_ws = (
                inst.transform * vec4<f32>(object.local_bounds.xyz, 1.0)
            ).xyz;
            let world_radius = max(object.local_bounds.w * derived.max_scale, 0.0);
            if sphere_visible(center_ws, world_radius) {
                let camera_distance = length(center_ws - camera.position_near.xyz);
                let closest_distance = max(
                    camera_distance - world_radius,
                    max(camera.position_near.w, 1.0e-4),
                );
                let focal_pixels = abs(camera.proj[1][1])
                    * f32(cull_uni.screen_height) * 0.5;
                var selected_lod = 0u;
                var level = 1u;
                loop {
                    if level >= lod_count {
                        break;
                    }
                    let error_px = object.lod_errors[level]
                        * derived.max_scale * focal_pixels / closest_distance;
                    if error_px <= cull_uni.lod_error_threshold_px {
                        selected_lod = level;
                        level += 1u;
                    } else {
                        break;
                    }
                }
                selected_lod_plus_one = selected_lod + 1u;
            }
        }
    }
    objects[object_index].selected_lod_plus_one = selected_lod_plus_one;
}

@compute @workgroup_size(64)
fn cs_cull_meshlets(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let work_index = workgroup_id.x
        + workgroup_id.y * cull_uni.work_dispatch_width;

    if lane == 0u {
        publish_frustum_planes();
        wg_first_meshlet = 0u;
        wg_meshlet_count = 0u;
        wg_instance_index = 0u;
        wg_selected_lod = 0u;
        wg_object_visible = 0u;
        wg_local_meshlet_base = 0u;

        if work_index < cull_uni.work_item_count
            && work_index < arrayLength(&work_items)
        {
            let item = work_items[work_index];
            if item.object_index < arrayLength(&objects) {
                let object = objects[item.object_index];
                if object.selected_lod_plus_one != 0u {
                    let selected_lod = object.selected_lod_plus_one - 1u;
                    let selected_count = object.lod_meshlet_counts[selected_lod];
                    if item.local_meshlet_base < selected_count {
                        wg_first_meshlet = object.lod_first_meshlets[selected_lod];
                        wg_meshlet_count = selected_count;
                        wg_instance_index = object.instance_index;
                        wg_selected_lod = selected_lod;
                        wg_local_meshlet_base = item.local_meshlet_base;
                        wg_object_visible = 1u;
                    }
                }
            }
        }
    }
    workgroupBarrier();

    if wg_object_visible == 0u {
        return;
    }
    let local_meshlet = wg_local_meshlet_base + lane;
    if local_meshlet < wg_meshlet_count {
        cull_meshlet(
            wg_first_meshlet + local_meshlet,
            wg_instance_index,
            wg_selected_lod,
        );
    }
}
