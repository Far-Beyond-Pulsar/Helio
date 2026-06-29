//! Hi-Z occlusion culling — fully GPU-driven, O(1) CPU.
//!
//! Each thread evaluates one DRAW CALL slot by testing the bounding sphere
//! of that draw call's first (representative) instance against the Hi-Z pyramid.
//! Occluded draws get instance_count=0 in the indirect buffer.
//!
//! IMPORTANT: this pass runs AFTER IndirectDispatchPass (frustum cull). It does
//! NOT re-do frustum culling — only tests occlusion. The indirect buffer is
//! shared: frustum cull writes initial instance_count, then we may zero it.
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
    draw_count:    u32,   // number of draw calls (== indirect buffer entries)
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

/// GpuDrawCall: 20 bytes, matches DrawCall in indirect_dispatch.wgsl.
struct GpuDrawCall {
    index_count:    u32,
    first_index:    u32,
    vertex_offset:  i32,
    first_instance: u32,  // base index into instances[] for this batch
    instance_count: u32,  // number of consecutive instances in this draw
}
@group(0) @binding(3) var<storage, read> draw_calls: array<GpuDrawCall>;

@group(0) @binding(4) var hiz_tex:  texture_2d<f32>;
@group(0) @binding(5) var hiz_samp: sampler;

// Indirect draw buffer as raw u32 array.
// DrawIndexedIndirect stride = 20 bytes = 5 × u32:
//   [i*5 + 0] index_count
//   [i*5 + 1] instance_count  ← we write 0 (occluded) or keep original value
//   [i*5 + 2] first_index
//   [i*5 + 3] base_vertex     (i32 reinterpreted as u32 for array access)
//   [i*5 + 4] first_instance
@group(0) @binding(6) var<storage, read_write> indirect: array<u32>;

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

/// Conservative sphere near depth in NDC [0,1].
/// Projects the point on the sphere nearest to the camera into NDC depth.
fn sphere_near_depth(center: vec3<f32>, radius: f32) -> f32 {
    let cam_pos = camera.position_near.xyz;
    let to_center = center - cam_pos;
    let dist_sq = dot(to_center, to_center);
    if dist_sq <= radius * radius {
        // Camera inside sphere — near depth is 0 (on the near plane)
        return 0.0;
    }
    let dir = to_center * (1.0 / sqrt(dist_sq));
    let near_ws = center - dir * radius;
    let near_clip = camera.view_proj * vec4<f32>(near_ws, 1.0);
    // Protect against near_clip.w <= 0 (shouldn't happen since camera is outside)
    if near_clip.w <= 0.0 {
        return 0.0;
    }
    return clamp(near_clip.z / near_clip.w, 0.0, 1.0);
}

// ──────────────────────────────────────────────────────────────────────────────
// Main kernel  (64 threads × 1 × 1 workgroup)
// ──────────────────────────────────────────────────────────────────────────────

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.draw_count {
        return;
    }

    // Get representative instance for this draw call (same as frustum cull pass).
    let dc = draw_calls[idx];
    let inst = instances[dc.first_instance];
    let center = inst.bounds.xyz;
    let radius = inst.bounds.w;
    if radius <= 0.0 {
        // No bounds — likely a placeholder; skip (keep whatever frustum cull wrote).
        return;
    }

    // ── Compute screen-space bounds ──────────────────────────────────────────
    let clip = camera.view_proj * vec4<f32>(center, 1.0);
    if clip.w <= 0.0 {
        // Center behind camera — cannot occlude (leave existing instance_count alone).
        // The frustum cull pass already set this correctly.
        return;
    }

    let ndc = clip.xyz / clip.w;
    let uv = ndc_to_uv(ndc.xy);

    // Check that the sphere's screen-space footprint is within the viewport.
    // If the entire screen-space footprint is outside the viewport, the object
    // was frustum-culled and we shouldn't touch it.
    let ndc_r = max(
        abs(radius * camera.proj[0][0] / clip.w),
        abs(radius * camera.proj[1][1] / clip.w),
    );
    if ndc.x + ndc_r < -1.0 || ndc.x - ndc_r > 1.0 ||
       ndc.y + ndc_r < -1.0 || ndc.y - ndc_r > 1.0 {
        // Entirely outside viewport (should have been frustum-culled). Skip.
        return;
    }

    // ── Conservative near depth ──────────────────────────────────────────────
    let near_z = sphere_near_depth(center, radius);

    // ── Choose Hi-Z mip level ────────────────────────────────────────────────
    let r_px = screen_radius_px(radius, clip.w);
    let mip  = pick_mip(r_px);

    // ── Conservative 4-tap Hi-Z occlusion test ──────────────────────────────
    // Sample Hi-Z (MAX pyramid) at 4 corners of sphere's bounding rectangle.
    // Take the farthest of the 4 samples — most conservative.
    let uv_half = ndc_r * 0.5;
    let uv_00 = clamp(uv - vec2<f32>(uv_half, uv_half), vec2<f32>(0.0), vec2<f32>(1.0));
    let uv_11 = clamp(uv + vec2<f32>(uv_half, uv_half), vec2<f32>(0.0), vec2<f32>(1.0));

    let hiz_00 = textureSampleLevel(hiz_tex, hiz_samp, uv_00, f32(mip)).r;
    let hiz_01 = textureSampleLevel(hiz_tex, hiz_samp, vec2<f32>(uv_11.x, uv_00.y), f32(mip)).r;
    let hiz_10 = textureSampleLevel(hiz_tex, hiz_samp, vec2<f32>(uv_00.x, uv_11.y), f32(mip)).r;
    let hiz_11 = textureSampleLevel(hiz_tex, hiz_samp, uv_11, f32(mip)).r;
    let hiz_depth = max(max(hiz_00, hiz_01), max(hiz_10, hiz_11));

    let depth_bias = 1.0 / 65536.0;
    if near_z > hiz_depth + depth_bias {
        indirect[idx * 5u + 1u] = 0u;
    }
    // Otherwise keep the existing instance_count (set by frustum cull pass).
}
