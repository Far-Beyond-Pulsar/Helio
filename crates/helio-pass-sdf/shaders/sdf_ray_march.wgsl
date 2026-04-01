// SDF Ray March — fullscreen triangle fragment shader
//
// Sphere-traces through the clip-map SDF volume, reading trilinearly-
// interpolated cached distances from per-level atlas buffers. Outputs
// colour and depth so the SDF integrates with the deferred pipeline.
//
// Bind group 0:
//   b0:  uniform  camera (CameraUniform)
//   b1:  uniform  clip map params (SdfClipMapParams)
//   b2-b9:  storage read  atlas buffer per level (8 levels)
//   b10: storage read  all_brick_indices (concatenated per-level)

// ── Camera uniform ──────────────────────────────────────────────────────────

// Must match libhelio::camera::GpuCameraUniforms exactly.
struct CameraUniform {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    position_near: vec4<f32>,   // xyz = pos, w = near
    forward_far: vec4<f32>,     // xyz = forward, w = far
    jitter_frame: vec4<f32>,    // xy = jitter, z = frame, w = pad
    prev_view_proj: mat4x4<f32>,
};

// ── Clip map params ─────────────────────────────────────────────────────────

struct ClipLevel {
    volume_min: vec3<f32>,
    _pad0: f32,
    volume_max: vec3<f32>,
    _pad1: f32,
    grid_origin: vec3<f32>,
    _pad2: f32,
    voxel_size: f32,
    brick_size: u32,
    brick_grid_dim: u32,
    atlas_bricks_per_axis: u32,
};

struct SdfClipMapParams {
    level_count: u32,
    grid_dim: u32,
    max_march_dist: f32,
    debug_flags: u32,
    levels: array<ClipLevel, 8>,
};

// ── Bindings ────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var<uniform> clip_params: SdfClipMapParams;

@group(0) @binding(2)  var<storage, read> atlas_0: array<u32>;
@group(0) @binding(3)  var<storage, read> atlas_1: array<u32>;
@group(0) @binding(4)  var<storage, read> atlas_2: array<u32>;
@group(0) @binding(5)  var<storage, read> atlas_3: array<u32>;
@group(0) @binding(6)  var<storage, read> atlas_4: array<u32>;
@group(0) @binding(7)  var<storage, read> atlas_5: array<u32>;
@group(0) @binding(8)  var<storage, read> atlas_6: array<u32>;
@group(0) @binding(9)  var<storage, read> atlas_7: array<u32>;
@group(0) @binding(10) var<storage, read> all_brick_indices: array<u32>;

// ── Vertex shader (fullscreen triangle) ─────────────────────────────────────

struct VsOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOutput {
    var out: VsOutput;
    // Single oversized triangle covering the full screen
    let x = f32(i32(vid & 1u)) * 4.0 - 1.0;
    let y = f32(i32(vid >> 1u)) * 4.0 - 1.0;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x * 0.5 + 0.5, 0.5 - y * 0.5);
    return out;
}

// ── Atlas sampling helpers ──────────────────────────────────────────────────

const EMPTY_BRICK: u32 = 0xFFFFFFFFu;

fn read_atlas_byte(level_idx: u32, byte_idx: u32) -> f32 {
    let word = byte_idx / 4u;
    let shift = (byte_idx % 4u) * 8u;
    var raw: u32 = 0u;

    // Unrolled level selection (WGSL doesn't allow indexing into bindings)
    if level_idx == 0u { raw = atlas_0[word]; }
    else if level_idx == 1u { raw = atlas_1[word]; }
    else if level_idx == 2u { raw = atlas_2[word]; }
    else if level_idx == 3u { raw = atlas_3[word]; }
    else if level_idx == 4u { raw = atlas_4[word]; }
    else if level_idx == 5u { raw = atlas_5[word]; }
    else if level_idx == 6u { raw = atlas_6[word]; }
    else { raw = atlas_7[word]; }

    return f32((raw >> shift) & 0xFFu) / 255.0;
}

fn dequantize(t: f32, vs: f32) -> f32 {
    let range = 4.0 * vs;
    return t * 2.0 * range - range;
}

fn atlas_byte_index(atlas_id: u32, local: vec3<u32>, level: ClipLevel) -> u32 {
    let aba = level.atlas_bricks_per_axis;
    let ps = level.brick_size + 1u;
    let bx = atlas_id % aba;
    let by = (atlas_id / aba) % aba;
    let bz = atlas_id / (aba * aba);
    let gx = bx * ps + local.x;
    let gy = by * ps + local.y;
    let gz = bz * ps + local.z;
    let dim = aba * ps;
    return gz * dim * dim + gy * dim + gx;
}

fn brick_index_offset(level_idx: u32) -> u32 {
    // Each level has brick_grid_dim^3 entries
    let bgd = clip_params.levels[0].brick_grid_dim;
    return level_idx * bgd * bgd * bgd;
}

fn sample_level_trilinear(level_idx: u32, world_pos: vec3<f32>) -> f32 {
    let level = clip_params.levels[level_idx];
    let vs = level.voxel_size;
    let bs = level.brick_size;
    let bgd = level.brick_grid_dim;
    let ibs = i32(bs);
    let ibgd = i32(bgd);

    // Absolute world voxel coordinate (NOT relative to volume_min).
    // This is critical: with toroidal addressing, volume_min-relative offsets
    // do NOT correspond to toroidal grid indices.
    let world_voxel = world_pos / vs;
    let grid_pos = world_voxel - 0.5; // half-voxel offset for trilinear center

    let ix = i32(floor(grid_pos.x));
    let iy = i32(floor(grid_pos.y));
    let iz = i32(floor(grid_pos.z));
    let fx = fract(grid_pos.x);
    let fy = fract(grid_pos.y);
    let fz = fract(grid_pos.z);

    var result = 0.0;

    // 8-point trilinear interpolation
    for (var dz = 0; dz < 2; dz++) {
        for (var dy = 0; dy < 2; dy++) {
            for (var dx = 0; dx < 2; dx++) {
                let sx = ix + dx;
                let sy = iy + dy;
                let sz = iz + dz;

                let w = select(1.0 - fx, fx, dx == 1)
                      * select(1.0 - fy, fy, dy == 1)
                      * select(1.0 - fz, fz, dz == 1);

                // World brick coordinate via floor-division
                // (WGSL i32 division truncates; floor(float) rounds towards -inf)
                let bx = i32(floor(f32(sx) / f32(ibs)));
                let by = i32(floor(f32(sy) / f32(ibs)));
                let bz = i32(floor(f32(sz) / f32(ibs)));

                // Toroidal grid coordinate — same modular wrap as CPU world_to_grid
                let gx = ((bx % ibgd) + ibgd) % ibgd;
                let gy = ((by % ibgd) + ibgd) % ibgd;
                let gz = ((bz % ibgd) + ibgd) % ibgd;
                let brick_flat = u32(gz * ibgd * ibgd + gy * ibgd + gx);

                let offset = brick_index_offset(level_idx);
                let atlas_id = all_brick_indices[offset + brick_flat];

                if atlas_id == EMPTY_BRICK {
                    result += w * (4.0 * vs);
                    continue;
                }

                // Local voxel within brick [0, bs-1]
                let lx = u32(sx - bx * ibs);
                let ly = u32(sy - by * ibs);
                let lz = u32(sz - bz * ibs);

                let bi = atlas_byte_index(atlas_id, vec3<u32>(lx, ly, lz), level);
                let t = read_atlas_byte(level_idx, bi);
                result += w * dequantize(t, vs);
            }
        }
    }

    return result;
}

fn point_in_level(level_idx: u32, pos: vec3<f32>) -> bool {
    let level = clip_params.levels[level_idx];
    return all(pos >= level.volume_min) && all(pos <= level.volume_max);
}

// Compute blend factor between clipmap levels for smooth LOD transitions.
// Returns alpha in [0, 1] where 0 = full fine level, 1 = full coarse level.
fn clipmap_blend_alpha(level_idx: u32, pos: vec3<f32>) -> f32 {
    let level = clip_params.levels[level_idx];
    let center = (level.volume_min + level.volume_max) * 0.5;
    let extent = level.volume_max - level.volume_min;

    // Distance from center in each axis, normalized to [0, 1] at the boundary
    let dist = abs(pos - center) / (extent * 0.5);

    // Max distance ratio (use the most constraining axis)
    let d_max = max(max(dist.x, dist.y), dist.z);

    // Transition region: start blending at 70% of the way to edge, fully blend at 95%
    let start_blend = 0.7;
    let end_blend = 0.95;

    return smoothstep(start_blend, end_blend, d_max);
}

// Query SDF with smooth LOD blending between clipmap levels.
// When near the edge of a level, smoothly blends to the next coarser level.
fn sdf_query(world_pos: vec3<f32>) -> f32 {
    // Find the finest level that contains the point
    var fine_level = clip_params.level_count;
    for (var i = 0u; i < clip_params.level_count; i++) {
        if point_in_level(i, world_pos) {
            fine_level = i;
            break;
        }
    }

    // If point is outside all levels, return large distance
    if fine_level >= clip_params.level_count {
        return 1e10;
    }

    // Sample from the finest containing level
    let fine_dist = sample_level_trilinear(fine_level, world_pos);

    // Check if we should blend to coarser level
    let alpha = clipmap_blend_alpha(fine_level, world_pos);
    if alpha > 0.001 && fine_level < clip_params.level_count - 1u {
        // Sample from next coarser level and blend
        let coarse_dist = sample_level_trilinear(fine_level + 1u, world_pos);
        return mix(fine_dist, coarse_dist, alpha);
    }

    return fine_dist;
}

// Query which clip level a point falls in (for consistent normal estimation)
fn sdf_query_level(world_pos: vec3<f32>) -> u32 {
    for (var i = 0u; i < clip_params.level_count; i++) {
        if point_in_level(i, world_pos) {
            return i;
        }
    }
    return clip_params.level_count;
}

// Normal estimation using the blended SDF field for smooth LOD transitions.
// This ensures normals match the geometry that was actually hit by the ray marcher.
fn estimate_normal(p: vec3<f32>, eps: f32) -> vec3<f32> {
    // Use tetrahedron technique (4 samples instead of 6) for smoother normals
    // All samples use sdf_query which blends between clipmap levels smoothly
    let k = vec2<f32>(1.0, -1.0);
    let n = k.xyy * sdf_query(p + k.xyy * eps) +
    k.yyx * sdf_query(p + k.yyx * eps) +
    k.yxy * sdf_query(p + k.yxy * eps) +
    k.xxx * sdf_query(p + k.xxx * eps);
    return normalize(n);
}

// ── Terrain material blending ────────────────────────────────────────────────
//
// Returns a surface colour for a terrain hit point using layered height-based
// and slope-based material transitions. All thresholds are in world-units (y).
//
// Height zones (approximate):
//   y < -2   : deep bedrock / dark wet earth
//   -2 -> 0  : dirt (transition into surface vegetation)
//   0  -> 2  : low grass (valley floors)
//   2  -> 6  : lighter grass (gentle hills)
//   5  -> 10 : rock (high ridges where grass thins)
//   10 -> 15 : snow (mountain caps)
//
// Slope modifier (applied on top of height colour):
//   normal.y > 0.8 : nearly flat  — keep height-based colour
//   normal.y 0.4-0.8 : moderate slope — blend in exposed rock
//   normal.y < 0.4 : steep cliff  — full exposed rock

// Material palette constants
const MAT_DIRT:       vec3<f32> = vec3<f32>(0.38, 0.28, 0.18);  // dark brown soil
const MAT_GRASS_LOW:  vec3<f32> = vec3<f32>(0.28, 0.42, 0.15);  // deep valley green
const MAT_GRASS_HIGH: vec3<f32> = vec3<f32>(0.42, 0.58, 0.22);  // sunlit hill green
const MAT_ROCK:       vec3<f32> = vec3<f32>(0.45, 0.40, 0.35);  // warm grey rock
const MAT_SNOW:       vec3<f32> = vec3<f32>(0.92, 0.93, 0.96);  // cool near-white snow

fn get_terrain_material(pos: vec3<f32>, normal: vec3<f32>, slope: f32) -> vec3<f32> {
    let height = pos.y;

    // Height-based base colour — layered smoothstep transitions
    // Below sea level (-2): dark dirt/bedrock fading in from below
    var base = mix(MAT_DIRT, MAT_GRASS_LOW, smoothstep(-2.0, 0.0, height));

    // Low elevation: valley grass transitions to sunlit hill grass (y 2->6)
    base = mix(base, MAT_GRASS_HIGH, smoothstep(2.0, 6.0, height));

    // Mid elevation: grass gives way to exposed rock on higher ridges (y 5->10)
    base = mix(base, MAT_ROCK, smoothstep(5.0, 10.0, height));

    // High elevation: rock fades into snow cap (y 10->15)
    base = mix(base, MAT_SNOW, smoothstep(10.0, 15.0, height));

    // Slope-based override: steep surfaces expose bare rock regardless of height.
    // smoothstep(0.7, 0.4, normal.y) yields 0 when nearly flat (normal.y~0.7+)
    // and 1 when very steep (normal.y~0.4 or below).
    let slope_factor = smoothstep(0.7, 0.4, normal.y);
    base = mix(base, MAT_ROCK, slope_factor);

    return base;
}

// ── Fragment shader ─────────────────────────────────────────────────────────

struct FsOutput {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

// Sky color constant (matches clear color)
const SKY_COLOR: vec3<f32> = vec3<f32>(0.53, 0.72, 0.90);

// ── Ray march tuning constants ────────────────────────────────────────────
//
// FOG_DENSITY: controls how quickly `fog_accum` grows.  Larger = earlier exit
//   for distant rays.  Keep proportional to 1/max_march_dist.
const FOG_DENSITY: f32 = 0.003;

// FOG_EXIT_THRESHOLD: accumulated fog value above which remaining steps are
//   invisible (optical depth saturation).  3.0 ≈ e^-3 ≈ 5 % transmission.
const FOG_EXIT_THRESHOLD: f32 = 3.0;

@fragment
fn fs_main(in: VsOutput) -> FsOutput {
    var out: FsOutput;

    // Reconstruct ray from camera
    let ndc = vec2<f32>(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
    let near_h = camera.inv_view_proj * vec4<f32>(ndc, 0.0, 1.0);
    let far_h = camera.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let near_w = near_h.xyz / near_h.w;
    let far_w = far_h.xyz / far_h.w;
    let ray_dir = normalize(far_w - near_w);
    let ray_origin = camera.position_near.xyz;

    let max_dist = clip_params.max_march_dist;

    // ── Step budget ───────────────────────────────────────────────────────
    // 256 steps for better quality - prevents banding and missed surfaces
    let max_steps = 256u;

    // Finest-level voxel size — drives the adaptive threshold and min step floor
    let vs0 = clip_params.levels[0].voxel_size;

    // Add ray dither offset to break up banding patterns (IQ technique)
    // Uses fragment position (pixel coordinate) for stable inter-pixel variation
    let dither = fract(sin(dot(in.position.xy, 
                               vec2<f32>(12.9898, 78.233))) * 43758.5453);
    var t = vs0 * 0.5 * dither; // Start with sub-voxel jitter
    var hit = false;
    var hit_pos = vec3<f32>(0.0);

    // Accumulated fog optical depth used exclusively for early exit.
    var fog_accum = 0.0;
    var prev_t = t; // Track previous t to compute step delta for fog

    for (var step = 0u; step < max_steps; step++) {
        let p = ray_origin + ray_dir * t;
        let d = sdf_query(p);

        // ── Adaptive hit threshold ────────────────────────────────────────
        // Slightly conservative for u8-quantised cached SDF: vs0 * 0.05
        // base prevents tunneling through thin terrain features, while the
        // distance-scaled term (t * 0.0005) accounts for accumulated
        // floating-point error on long rays.
        let threshold = max(vs0 * 0.05, t * 0.0005);
        if d < threshold {
            hit = true;
            hit_pos = p;
            break;
        }

        // ── Adaptive step size (IQ relaxed sphere tracing) ─────────────
        // Base multiplier 0.75 is conservative for quantised atlas SDF.
        // Linear ramp (t * 0.0005, capped at 0.15) widens towards 0.90
        // for distant open-space rays, reducing iterations by ~30-40%
        // without risking surface over-stepping.  vs0 * 0.2 floor guards
        // against stalling in flat empty regions.
        let step_mult = 0.75 + min(t * 0.0005, 0.15);
        let step_size = max(d * step_mult, vs0 * 0.2);
        prev_t = t;
        t += step_size;

        if t > max_dist {
            break;
        }

        // ── Fog-based early exit ──────────────────────────────────────────
        // Accumulate fog proportional to the actual step taken (t - prev_t),
        // not total ray distance. Using total `t` caused quadratic blow-up,
        // prematurely discarding distant terrain rays.
        fog_accum += (t - prev_t) * FOG_DENSITY;
        if fog_accum > FOG_EXIT_THRESHOLD {
            break;
        }
    }

    if !hit {
        discard;
    }

    // ── Normal estimation ────────────────────────────────────────────────
    let cam_dist = length(hit_pos - ray_origin);
    // Use hit-point's clip level voxel size for epsilon — ensures normal
    // samples span at least one voxel at the appropriate LOD, avoiding
    // quantisation artifacts without over-smoothing fine detail.
    let hit_level = sdf_query_level(hit_pos);
    let eps_vs = select(vs0, clip_params.levels[hit_level].voxel_size, hit_level < clip_params.level_count);
    let eps = eps_vs * 0.5;
    let normal = estimate_normal(hit_pos, eps);

    // ── Terrain shading ──────────────────────────────────────────────────
    let light_dir = normalize(vec3<f32>(0.4, 0.8, 0.3));
    let ndotl = max(dot(normal, light_dir), 0.0);
    let ambient = 0.2;
    let diffuse = ndotl * 0.8;

    // Slope: 0 = flat ground, 1 = vertical cliff
    let slope = 1.0 - normal.y;

    let base_color = get_terrain_material(hit_pos, normal, slope);
    let color = base_color * (ambient + diffuse);

    // ── Distance fog ─────────────────────────────────────────────────────
    let fog_start = max_dist * 0.3;
    let fog_end = max_dist * 0.85;
    let fog = clamp((cam_dist - fog_start) / (fog_end - fog_start), 0.0, 1.0);
    let final_color = mix(color, SKY_COLOR, fog);

    // ── Debug mode ───────────────────────────────────────────────────────
    if clip_params.debug_flags != 0u {
        // Find which clip level the hit is in
        var hit_level = 0u;
        for (var i = 0u; i < clip_params.level_count; i++) {
            if point_in_level(i, hit_pos) {
                hit_level = i;
                break;
            }
        }
        let level = clip_params.levels[hit_level];
        let bvs = level.voxel_size;
        let bbs = f32(level.brick_size);
        let brick_world = bvs * bbs;

        // Clip level color: green (fine) → red (coarse)
        let t_col = f32(hit_level) / max(f32(clip_params.level_count - 1u), 1.0);
        let level_color = mix(vec3<f32>(0.1, 0.8, 0.2), vec3<f32>(0.9, 0.15, 0.1), t_col);

        // Brick grid outline (thin lines at brick boundaries)
        let brick_frac = fract(hit_pos / brick_world);
        let edge_dist = min(
            min(min(brick_frac.x, 1.0 - brick_frac.x), min(brick_frac.y, 1.0 - brick_frac.y)),
            min(brick_frac.z, 1.0 - brick_frac.z)
        );
        let edge_width = clamp(cam_dist * 0.0004, 0.01, 0.15);
        let edge = 1.0 - smoothstep(0.0, edge_width, edge_dist);

        let lit = ambient + diffuse;
        let debug_color = mix(level_color * lit, vec3<f32>(1.0, 1.0, 1.0), edge * 0.8);
        out.color = vec4<f32>(mix(debug_color, SKY_COLOR, fog), 1.0);
    } else {
        out.color = vec4<f32>(final_color, 1.0);
    }

    // ── Depth output ─────────────────────────────────────────────────────
    let clip_pos = camera.view_proj * vec4<f32>(hit_pos, 1.0);
    out.depth = clip_pos.z / clip_pos.w;

    return out;
}
