// sdf_ray_march_clipmap.wgsl
// Fullscreen SDF ray march through the sparse clipmap brick atlas.
//
// Vertex shader: generates a fullscreen triangle (no vertex buffer).
// Fragment shader: unprojects each NDC pixel into a world-space ray, then
//   sphere-traces through the clipmap levels using DDA brick acceleration.
//
// Bindings:
//   Group 0, binding 0:  camera (GpuCameraUniforms)
//   Group 1, binding 0:  clip_params (SdfClipMapParams)
//   Group 1, binding 1..8: atlas_0..atlas_7 (array<u32>, packed u8 SDF)
//   Group 1, binding 9:  all_brick_indices (array<u32>)

// ---------------------------------------------------------------------------
// Structs - Group 0
// ---------------------------------------------------------------------------

// Matches helio-v3 GpuCameraUniforms (368 bytes).
struct Camera {
    view:          mat4x4<f32>,        // offset   0
    proj:          mat4x4<f32>,        // offset  64
    view_proj:     mat4x4<f32>,        // offset 128
    inv_view_proj: mat4x4<f32>,        // offset 192
    position_near: vec4<f32>,          // offset 256  (.xyz = world position)
    forward_far:   vec4<f32>,          // offset 272  (.xyz = forward, .w = far plane)
    jitter_frame:  vec4<f32>,          // offset 288
    prev_view_proj:mat4x4<f32>,        // offset 304
}

// ---------------------------------------------------------------------------
// Structs - Group 1
// ---------------------------------------------------------------------------

struct ClipLevel {
    world_min:          vec3<f32>,
    voxel_size:         f32,
    grid_dim:           u32,
    brick_dim:          u32,
    brick_index_offset: u32,
    active_brick_count: u32,
    toroidal_origin:    vec3<i32>,
    _pad0:              u32,
    atlas_dim:          vec3<u32>,
    _pad1:              u32,
}

struct ClipMapParams {
    level_count: u32,
    _pad:        vec3<u32>,
    levels:      array<ClipLevel, 8>,
}

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> camera: Camera;

@group(1) @binding(0) var<uniform>      clip_params:       ClipMapParams;
@group(1) @binding(1) var<storage, read> atlas_0:          array<u32>;
@group(1) @binding(2) var<storage, read> atlas_1:          array<u32>;
@group(1) @binding(3) var<storage, read> atlas_2:          array<u32>;
@group(1) @binding(4) var<storage, read> atlas_3:          array<u32>;
@group(1) @binding(5) var<storage, read> atlas_4:          array<u32>;
@group(1) @binding(6) var<storage, read> atlas_5:          array<u32>;
@group(1) @binding(7) var<storage, read> atlas_6:          array<u32>;
@group(1) @binding(8) var<storage, read> atlas_7:          array<u32>;
@group(1) @binding(9) var<storage, read> all_brick_indices: array<u32>;

// ---------------------------------------------------------------------------
// Atlas sampling
// ---------------------------------------------------------------------------

fn atlas_sample(level_idx: u32, atlas_idx: u32, local: vec3<u32>, brick_dim: u32, atlas_dim: vec3<u32>) -> f32 {
    let ax = atlas_idx % atlas_dim.x;
    let ay = (atlas_idx / atlas_dim.x) % atlas_dim.y;
    let az = atlas_idx / (atlas_dim.x * atlas_dim.y);
    let vx = ax * brick_dim + local.x;
    let vy = ay * brick_dim + local.y;
    let vz = az * brick_dim + local.z;
    let sx = atlas_dim.x * brick_dim;
    let sy = atlas_dim.y * brick_dim;
    let flat = vz * sy * sx + vy * sx + vx;
    let word = flat / 4u;
    let shift = (flat % 4u) * 8u;

    var packed = 0u;
    if level_idx == 0u { packed = atlas_0[word]; }
    else if level_idx == 1u { packed = atlas_1[word]; }
    else if level_idx == 2u { packed = atlas_2[word]; }
    else if level_idx == 3u { packed = atlas_3[word]; }
    else if level_idx == 4u { packed = atlas_4[word]; }
    else if level_idx == 5u { packed = atlas_5[word]; }
    else if level_idx == 6u { packed = atlas_6[word]; }
    else { packed = atlas_7[word]; }

    let byte = (packed >> shift) & 0xFFu;
    // Decode: [0,255] -> [-1,1] in normalised units, then scale to voxel_size * 4.
    let level = clip_params.levels[level_idx];
    let max_d = level.voxel_size * 4.0;
    return (f32(byte) / 255.0 * 2.0 - 1.0) * max_d;
}

// Trilinear sample of the SDF atlas at a world position for a given level.
fn sample_level(level_idx: u32, world_pos: vec3<f32>) -> f32 {
    let lvl = clip_params.levels[level_idx];
    let vs  = lvl.voxel_size;
    let bd  = f32(lvl.brick_dim);
    let gd  = f32(lvl.grid_dim);

    // Convert world pos to voxel coords within this level.
    let voxel = (world_pos - lvl.world_min) / vs;
    let voxel_int = vec3<i32>(floor(voxel));

    // Find which brick.
    let brick_coord = vec3<i32>(voxel_int) / vec3<i32>(i32(lvl.brick_dim));
    let gdi = vec3<i32>(i32(lvl.grid_dim));
    if any(brick_coord < vec3<i32>(0)) || any(brick_coord >= gdi) {
        return 1e9;
    }
    let brick_flat = u32(brick_coord.z * gdi.x * gdi.y + brick_coord.y * gdi.x + brick_coord.x);
    // Look up atlas index.  all_brick_indices is flat; need per-level offset.
    // We just read the atlas directly — atlas only has valid data for active bricks.
    // For simplicity, perform a direct sample at the voxel centre.
    let local_voxel = vec3<u32>(vec3<i32>(voxel_int) - brick_coord * vec3<i32>(i32(lvl.brick_dim)));
    // We don't have a brick_index lookup here (that's a per-level buffer in compute),
    // so sample by reading atlas at brick offset.
    // Since we don't have the atlas_idx→brick mapping in the render shader,
    // we use the all_brick_indices list to find the atlas slot for brick_flat.
    let base_offset = lvl.brick_index_offset;
    // Search active bricks list for this grid_flat (linear scan per ray step — coarse).
    // Better: store a brick_index_map per level in group 1.
    // For now, use brute-force lookup for correctness (can be optimised).
    // TODO: add per-level brick_index_map storage bindings in a future pass.
    // As a placeholder, return a smooth approximation from the terrain formula.
    return sample_atlas_direct(level_idx, brick_flat, local_voxel, lvl.brick_dim, lvl.atlas_dim);
}

// Direct atlas lookup using a flat brick_index map.
// Currently we use a packed atlas directly for the active slot.
// The atlas stores bricks sequentially: atlas slot = active-list position.
fn sample_atlas_direct(
    level_idx: u32,
    brick_flat: u32,
    local: vec3<u32>,
    brick_dim: u32,
    atlas_dim: vec3<u32>,
) -> f32 {
    // Determine atlas_idx by scanning all_brick_indices for our brick_flat.
    // This is O(active_bricks) per lookup — acceptable for a first-pass port;
    // replaced with a per-level brick_index buffer in the next optimisation pass.
    let lvl = clip_params.levels[level_idx];
    let base = lvl.brick_index_offset;
    let count = lvl.active_brick_count;
    for (var k = 0u; k < count; k++) {
        if all_brick_indices[base + k] == brick_flat {
            return atlas_sample(level_idx, k, local, brick_dim, atlas_dim);
        }
    }
    return 1e9; // Empty brick.
}

// ---------------------------------------------------------------------------
// SDF query: finds the best (finest) level covering a world point.
// ---------------------------------------------------------------------------

fn sdf_query(world_pos: vec3<f32>) -> f32 {
    for (var li = 0u; li < clip_params.level_count; li++) {
        let lvl = clip_params.levels[li];
        let vs  = lvl.voxel_size;
        let gd  = f32(lvl.grid_dim) * f32(lvl.brick_dim) * vs;
        let lo  = lvl.world_min;
        let hi  = lo + vec3<f32>(gd);
        if all(world_pos > lo + vs) && all(world_pos < hi - vs) {
            return sample_level(li, world_pos);
        }
    }
    return 1e9;
}

// ---------------------------------------------------------------------------
// Ray march
// ---------------------------------------------------------------------------

const MAX_STEPS: u32 = 256u;
const HIT_DIST: f32  = 0.01;
const MISS_DIST: f32 = 8000.0;

fn ray_march(ro: vec3<f32>, rd: vec3<f32>) -> f32 {
    var t = 0.0;
    for (var i = 0u; i < MAX_STEPS; i++) {
        let p = ro + rd * t;
        let d = sdf_query(p);
        if d < HIT_DIST { return t; }
        if t > MISS_DIST { return -1.0; }
        t += max(d * 0.8, HIT_DIST); // Relaxed sphere trace.
    }
    return -1.0;
}

// Tetrahedron normal estimation (4 samples, avoids central differences).
fn calc_normal(p: vec3<f32>) -> vec3<f32> {
    let e = vec2<f32>(1.0, -1.0) * 0.001;
    return normalize(
        e.xyy * sdf_query(p + e.xyy) +
        e.yyx * sdf_query(p + e.yyx) +
        e.yxy * sdf_query(p + e.yxy) +
        e.xxx * sdf_query(p + e.xxx)
    );
}

// ---------------------------------------------------------------------------
// Vertex shader — fullscreen triangle (no vertex buffer)
// ---------------------------------------------------------------------------

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0)       uv:       vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOut {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    let p = pos[vid];
    var out: VertexOut;
    out.clip_pos = vec4<f32>(p, 0.0, 1.0);
    out.uv = p * 0.5 + 0.5;
    return out;
}

// ---------------------------------------------------------------------------
// Fragment shader
// ---------------------------------------------------------------------------

struct FragOut {
    @location(0)             color: vec4<f32>,
    @builtin(frag_depth)     depth: f32,
}

@fragment
fn fs_main(in: VertexOut) -> FragOut {
    // Unproject NDC to world-space ray.
    let ndc = vec4<f32>(in.clip_pos.xy / in.clip_pos.w * vec2<f32>(1.0, -1.0), 1.0, 1.0);
    let world_far = camera.inv_view_proj * vec4<f32>(ndc.xy, 1.0, 1.0);
    let ro = camera.position_near.xyz;
    let rd = normalize(world_far.xyz / world_far.w - ro);

    let t = ray_march(ro, rd);

    var out: FragOut;
    if t < 0.0 {
        // Miss — sky gradient.
        let sky_t = clamp(dot(rd, vec3<f32>(0.0, 1.0, 0.0)) * 0.5 + 0.5, 0.0, 1.0);
        out.color = vec4<f32>(mix(vec3<f32>(0.4, 0.6, 0.9), vec3<f32>(0.1, 0.2, 0.5), sky_t), 1.0);
        out.depth = 1.0;
    } else {
        let hit_pos = ro + rd * t;
        let n = calc_normal(hit_pos);

        // Simple sun + ambient lighting.
        let sun_dir    = normalize(vec3<f32>(0.6, 0.8, 0.3));
        let sun_color  = vec3<f32>(1.0, 0.9, 0.7);
        let ambient    = vec3<f32>(0.1, 0.12, 0.18);
        let albedo     = vec3<f32>(0.55, 0.48, 0.35); // Rock/earth color.

        let diff = max(dot(n, sun_dir), 0.0);
        let lit  = albedo * (ambient + diff * sun_color);

        out.color = vec4<f32>(lit, 1.0);

        // Linearise depth for the hardware depth buffer.
        let clip_p = camera.view_proj * vec4<f32>(hit_pos, 1.0);
        out.depth  = clip_p.z / clip_p.w;
    }
    return out;
}
