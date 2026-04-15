//! Hi-Z occlusion culling — fully GPU-driven, O(1) CPU.
//!
//! Each thread evaluates one draw slot: project the bounding sphere into
//! screen-space, sample the Hi-Z pyramid at the appropriate mip level, and
//! zero-out instance_count in the indirect buffer for occluded draws.
//!
//! Uses TEMPORAL Hi-Z: the pyramid was built from the PREVIOUS frame's depth,
//! so the OcclusionCullPass runs BEFORE DepthPrepass each frame.
//! Frame 0 is skipped entirely (no pyramid yet).

// ──────────────────────────────────────────────────────────────────────────────
// Bind group 0
// ──────────────────────────────────────────────────────────────────────────────

struct Camera {
    view:          mat4x4<f32>,   // bytes  0 – 63
    proj:          mat4x4<f32>,   // bytes 64 – 127
    view_proj:     mat4x4<f32>,   // bytes 128 – 191
    inv_view_proj: mat4x4<f32>,   // bytes 192 – 255
    position_near: vec4<f32>,     // bytes 256 – 271
    direction_far: vec4<f32>,     // bytes 272 – 287
}
@group(0) @binding(0) var<uniform> camera: Camera;

struct CullParams {
    screen_width:  u32,
    screen_height: u32,
    total_slots:   u32,   // == draw_count / instance_count
    hiz_mip_count: u32,
}
@group(0) @binding(1) var<uniform> params: CullParams;

// GpuInstanceData: 144 bytes, must match libhelio/src/instance.rs exactly.
struct GpuInstanceData {
    model_col0:  vec4<f32>,  //   0 – 15
    model_col1:  vec4<f32>,  //  16 – 31
    model_col2:  vec4<f32>,  //  32 – 47
    model_col3:  vec4<f32>,  //  48 – 63
    normal_col0: vec4<f32>,  //  64 – 79   (w = padding)
    normal_col1: vec4<f32>,  //  80 – 95
    normal_col2: vec4<f32>,  //  96 – 111
    bounds:      vec4<f32>,  // 112 – 127  (xyz = world-space sphere center, w = radius)
    mesh_id:     u32,        // 128
    material_id: u32,        // 132
    flags:       u32,        // 136
    _pad:        u32,        // 140
}
@group(0) @binding(2) var<storage, read> instances: array<GpuInstanceData>;

@group(0) @binding(3) var hiz_tex:  texture_2d<f32>;
@group(0) @binding(4) var hiz_samp: sampler;

// Indirect draw buffer as raw u32 array.
// DrawIndexedIndirect stride = 20 bytes = 5 × u32:
//   [i*5 + 0] index_count
//   [i*5 + 1] instance_count  ← we write 0 (occluded) or 1 (visible)
//   [i*5 + 2] first_index
//   [i*5 + 3] base_vertex     (i32 reinterpreted as u32 for array access)
//   [i*5 + 4] first_instance
@group(0) @binding(5) var<storage, read_write> indirect: array<u32>;

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Project NDC xy to texture UV.
/// wgpu NDC: x∈[-1,+1] left→right, y∈[-1,+1] bottom→top.
/// UV:       u∈[0,1]   left→right, v∈[0,1]   top→bottom.
fn ndc_to_uv(ndc_xy: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        ndc_xy.x *  0.5 + 0.5,
        ndc_xy.y * -0.5 + 0.5,
    );
}

/// Estimate screen-space radius (in pixels) of a sphere.
/// proj[1][1] = cot(fovY/2) = 2n/h for a standard perspective matrix.
fn screen_radius_px(world_radius: f32, clip_w: f32) -> f32 {
    let half_h = f32(params.screen_height) * 0.5;
    return abs(world_radius / clip_w * camera.proj[1][1] * half_h);
}

/// Select HiZ mip level for a sphere footprint of `r_px` pixels.
fn pick_mip(r_px: f32) -> u32 {
    let diameter = max(r_px * 2.0, 1.0);
    let mip = u32(ceil(log2(diameter)));
    return clamp(mip, 0u, params.hiz_mip_count - 1u);
}

// ──────────────────────────────────────────────────────────────────────────────
// Main kernel  (64 threads × 1 × 1 workgroup)
// ──────────────────────────────────────────────────────────────────────────────

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.total_slots {
        return;
    }

    let inst   = instances[idx];
    let center = inst.bounds.xyz;
    let radius = inst.bounds.w;

    // Transform sphere center to clip space.
    let clip = camera.view_proj * vec4<f32>(center, 1.0);

    // Objects behind or at the near plane cannot be occluded — keep visible.
    // (clip.w ≤ 0 means the center is behind the viewer.)
    if clip.w <= 0.0 {
        indirect[idx * 5u + 1u] = 1u;
        return;
    }

    let ndc = clip.xyz / clip.w;

    // UV in [0,1]² for texture sampling.
    let uv = ndc_to_uv(ndc.xy);

    // Frustum culling by projecting the sphere center into clip space and
    // conservatively considering a radius margin.
    let clip_center = camera.view_proj * vec4<f32>(center, 1.0);

    if clip_center.w <= 0.0 {
        indirect[idx * 5u + 1u] = 1u;
        return;
    }

    let ndc_center = clip_center.xyz / clip_center.w;
    // Rough sphere radius in NDC units. Keep conservative by taking max of x/y.
    let ndc_radius = max(
        abs(radius * camera.proj[0][0] / clip_center.w),
        abs(radius * camera.proj[1][1] / clip_center.w),
    );

    if ndc_center.x < -1.0 - ndc_radius || ndc_center.x > 1.0 + ndc_radius ||
       ndc_center.y < -1.0 - ndc_radius || ndc_center.y > 1.0 + ndc_radius ||
       ndc_center.z < 0.0 - ndc_radius || ndc_center.z > 1.0 + ndc_radius {
        indirect[idx * 5u + 1u] = 0u; // outside frustum → cull
        return;
    }

    let sample_uv = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0));

    // Conservative sphere near depth in NDC:
    // Bring the closest face of the sphere toward the camera.
    // clip.w ≈ -view_z (positive for objects in front of camera).
    // Subtracting radius from clip.w gives the near face's clip.w.
    // Then re-derive NDC.z ≈ ndc.z scaled by (clip.w / near_w).
    let near_w  = max(clip.w - radius, 0.001);
    let near_z  = clip.z / clip.w * (near_w / clip.w); // approximate
    let min_depth = clamp(near_z, 0.0, 1.0);

    // Choose mip based on screen footprint.
    let r_px = screen_radius_px(radius, clip.w);
    let mip  = pick_mip(r_px);

    // Sample Hi-Z (MAX pyramid): the stored value is the MAXIMUM (farthest) depth
    // in the footprint at this mip level.
    let hiz_depth = textureSampleLevel(hiz_tex, hiz_samp, sample_uv, f32(mip)).r;

    // Occluded: every point of the sphere is farther than the closest known occluder.
    // In [0,1] depth where 0=near, 1=far: occluded iff min_depth > hiz_depth.
    if min_depth > hiz_depth {
        indirect[idx * 5u + 1u] = 0u;
    } else {
        indirect[idx * 5u + 1u] = 1u;
    }
}
