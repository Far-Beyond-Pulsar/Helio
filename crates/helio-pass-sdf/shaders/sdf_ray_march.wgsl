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

// ── Clip config (static, matches sdf_scroll.wgsl ClipConfig exactly) ───────

struct ClipConfig {
    level_count:           u32,
    grid_dim:              u32,
    brick_size:            u32,
    brick_grid_dim:        u32,
    bricks_per_level:      u32,
    atlas_bricks_per_axis: u32,
    base_voxel_size:       f32,
    edit_count:            u32,
    bvh_node_count:        u32,
    terrain_enabled:       u32,
    terrain_y_min:         f32,
    terrain_y_max:         f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
    voxel_sizes_lo: vec4<f32>,   // levels 0–3
    voxel_sizes_hi: vec4<f32>,   // levels 4–7
};

// ── Scroll state (GPU-written each frame by sdf_scroll pass) ────────────────

struct ScrollState {
    snap_origins:  array<vec4<i32>, 8>,  // brick-coord snap per level
    edit_gen:      u32,
    prev_edit_gen: u32,
    _pad0:         u32,
    _pad1:         u32,
};

// ── Bindings ────────────────────────────────────────────────────────────────

@group(0) @binding(0)  var<uniform>         camera:           CameraUniform;
@group(0) @binding(1)  var<uniform>         clip_config:      ClipConfig;
@group(0) @binding(2)  var<storage, read>   scroll_state:     ScrollState;

@group(0) @binding(3)  var<storage, read> atlas_0: array<u32>;
@group(0) @binding(4)  var<storage, read> atlas_1: array<u32>;
@group(0) @binding(5)  var<storage, read> atlas_2: array<u32>;
@group(0) @binding(6)  var<storage, read> atlas_3: array<u32>;
@group(0) @binding(7)  var<storage, read> atlas_4: array<u32>;
@group(0) @binding(8)  var<storage, read> atlas_5: array<u32>;
@group(0) @binding(9)  var<storage, read> atlas_6: array<u32>;
@group(0) @binding(10) var<storage, read> atlas_7: array<u32>;
@group(0) @binding(11) var<storage, read> all_brick_indices: array<u32>;

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

// ── Per-level helpers ───────────────────────────────────────────────────────

// Returns the voxel size at the given clip level (powers of two from base).
fn level_voxel_size(level_idx: u32) -> f32 {
    let lo = clip_config.voxel_sizes_lo;
    let hi = clip_config.voxel_sizes_hi;
    if level_idx == 0u { return lo.x; }
    if level_idx == 1u { return lo.y; }
    if level_idx == 2u { return lo.z; }
    if level_idx == 3u { return lo.w; }
    if level_idx == 4u { return hi.x; }
    if level_idx == 5u { return hi.y; }
    if level_idx == 6u { return hi.z; }
    return hi.w;
}

// Returns the world-space volume_min for the given level,
// derived from the GPU scroll_state without any CPU involvement.
fn level_world_min(level_idx: u32) -> vec3<f32> {
    let vs        = level_voxel_size(level_idx);
    let brick_step = vs * f32(clip_config.brick_size);
    let half_bgd  = i32(clip_config.brick_grid_dim) / 2;
    let snap      = scroll_state.snap_origins[level_idx].xyz;
    return vec3<f32>(
        f32(snap.x - half_bgd) * brick_step,
        f32(snap.y - half_bgd) * brick_step,
        f32(snap.z - half_bgd) * brick_step,
    );
}

fn atlas_byte_index(atlas_id: u32, local: vec3<u32>, level_idx: u32) -> u32 {
    let aba = clip_config.atlas_bricks_per_axis;
    let ps = clip_config.brick_size + 1u;
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
    let bgd = clip_config.brick_grid_dim;
    return level_idx * bgd * bgd * bgd;
}

fn sample_level_trilinear(level_idx: u32, world_pos: vec3<f32>) -> f32 {
    let vs  = level_voxel_size(level_idx);
    let bs  = clip_config.brick_size;
    let bgd = clip_config.brick_grid_dim;
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

    // Hoist loop-invariant values: replaces 8× float division + 8× function call.
    let inv_bs   = 1.0 / f32(ibs);
    let base_off = brick_index_offset(level_idx);
    let dq_range = 4.0 * vs;  // dequantize range = 2*range in [-range, +range]

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

                // World brick coordinate — multiply by inv_bs beats 3 float divides
                let bx = i32(floor(f32(sx) * inv_bs));
                let by = i32(floor(f32(sy) * inv_bs));
                let bz = i32(floor(f32(sz) * inv_bs));

                // Toroidal grid coordinate — same modular wrap as scroll shader
                let gx = ((bx % ibgd) + ibgd) % ibgd;
                let gy = ((by % ibgd) + ibgd) % ibgd;
                let gz = ((bz % ibgd) + ibgd) % ibgd;
                let brick_flat = u32(gz * ibgd * ibgd + gy * ibgd + gx);

                let atlas_id = all_brick_indices[base_off + brick_flat];

                if atlas_id == EMPTY_BRICK {
                    result += w * dq_range;
                    continue;
                }

                // Local voxel within brick [0, bs-1]
                let lx = u32(sx - bx * ibs);
                let ly = u32(sy - by * ibs);
                let lz = u32(sz - bz * ibs);

                let bi  = atlas_byte_index(atlas_id, vec3<u32>(lx, ly, lz), level_idx);
                let qd  = read_atlas_byte(level_idx, bi);
                result += w * (qd * 2.0 * dq_range - dq_range);
            }
        }
    }

    return result;
}

fn point_in_level(level_idx: u32, pos: vec3<f32>) -> bool {
    let vs       = level_voxel_size(level_idx);
    let vol_min  = level_world_min(level_idx);
    let grid_sz  = f32(clip_config.grid_dim) * vs;
    let vol_max  = vol_min + vec3<f32>(grid_sz);
    return all(pos >= vol_min) && all(pos <= vol_max);
}

// Compute blend factor between clipmap levels for smooth LOD transitions.
// Returns alpha in [0, 1] where 0 = full fine level, 1 = full coarse level.
fn clipmap_blend_alpha(level_idx: u32, pos: vec3<f32>) -> f32 {
    let vs      = level_voxel_size(level_idx);
    let vol_min = level_world_min(level_idx);
    let vol_max = vol_min + vec3<f32>(f32(clip_config.grid_dim) * vs);
    let center  = (vol_min + vol_max) * 0.5;
    let extent  = vol_max - vol_min;

    // Distance from center in each axis, normalized to [0, 1] at the boundary
    let dist = abs(pos - center) / (extent * 0.5);

    // Max distance ratio (use the most constraining axis)
    let d_max = max(max(dist.x, dist.y), dist.z);

    // Transition region: start blending at 70% of the way to edge, fully blend at 95%
    let start_blend = 0.7;
    let end_blend = 0.95;

    return smoothstep(start_blend, end_blend, d_max);
}

// ── Blended SDF query — shading/normals only, never called from the march loop ─
//
// Level search is skipped because the caller already knows which level the
// hit point is in (the march loop tracks it in `cur_level`).  The inter-level
// blend is kept so normals are smooth across clip-map boundaries.
fn sdf_query_hinted(world_pos: vec3<f32>, level: u32) -> f32 {
    if level >= clip_config.level_count { return 1e10; }
    let fine_dist = sample_level_trilinear(level, world_pos);
    let alpha     = clipmap_blend_alpha(level, world_pos);
    if alpha > 0.001 && level < clip_config.level_count - 1u {
        let coarse_dist = sample_level_trilinear(level + 1u, world_pos);
        return mix(fine_dist, coarse_dist, alpha);
    }
    return fine_dist;
}

// Normal estimation using the SDF field at a consistently-chosen clip level.
//
// The previous implementation (sdf_query_hinted) blended fine and coarse levels
// using a per-sample blend alpha: each tetrahedron point is at a slightly
// different position, so each picked a different blend weight.  That cross-level
// inconsistency produced noisy, flicker-prone normals near clip-map boundaries.
//
// Fix: evaluate the blend alpha ONCE at the hit point (p), pick either the fine
// or coarse level uniformly for all 4 samples, and sample directly without any
// further blending.  This makes the gradient consistent and eliminates the noise.
fn estimate_normal(p: vec3<f32>, eps: f32, level: u32) -> vec3<f32> {
    let alpha       = clipmap_blend_alpha(level, p);
    let query_level = select(level,
                             level + 1u,
                             alpha > 0.5 && level + 1u < clip_config.level_count);
    let k = vec2<f32>(1.0, -1.0);
    let n = k.xyy * sample_level_trilinear(query_level, p + k.xyy * eps) +
            k.yyx * sample_level_trilinear(query_level, p + k.yyx * eps) +
            k.yxy * sample_level_trilinear(query_level, p + k.yxy * eps) +
            k.xxx * sample_level_trilinear(query_level, p + k.xxx * eps);
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

    // ── Ray–AABB clip against the coarsest clip level ─────────────────────
    //
    // (a) Rays that completely miss the volume discard immediately — zero steps.
    // (b) Rays entering from outside advance t to the entry face — no wasted
    //     steps before the terrain starts.
    // (c) `t_vol_exit` is used inside the loop to break as soon as the ray
    //     leaves the back face — no wasted steps after terrain ends.
    //
    // IEEE 754: dividing by zero gives ±inf, which is handled correctly by
    // min/max comparisons, so no guard for axis-aligned rays is needed.
    let aabb_idx   = clip_config.level_count - 1u;
    let aabb_vs    = level_voxel_size(aabb_idx);
    let aabb_min   = level_world_min(aabb_idx);
    let aabb_max   = aabb_min + vec3<f32>(f32(clip_config.grid_dim) * aabb_vs);
    let inv_rd     = 1.0 / ray_dir;
    let t0_aabb    = (aabb_min - ray_origin) * inv_rd;
    let t1_aabb    = (aabb_max - ray_origin) * inv_rd;
    let tmin_aabb  = min(t0_aabb, t1_aabb);
    let tmax_aabb  = max(t0_aabb, t1_aabb);
    let t_enter    = max(max(tmin_aabb.x, tmin_aabb.y), tmin_aabb.z);
    let t_vol_exit = min(min(tmax_aabb.x, tmax_aabb.y), tmax_aabb.z);

    // Ray misses or is entirely behind the camera.
    if t_vol_exit < 0.0 || t_enter > t_vol_exit {
        discard;
    }

    // max_march_dist derived from coarsest level (fully GPU — no CPU upload)
    let coarsest_vs  = level_voxel_size(clip_config.level_count - 1u);
    // (used below for fog distance in the shading block)
    let max_dist     = f32(clip_config.grid_dim) * coarsest_vs * 2.0;

    // ── Step budget ───────────────────────────────────────────────────────
    // 1024 steps: sdf_march_step is ~4x cheaper than the old sdf_query
    // (O(1) level lookup + no inter-level blend), so 1024 steps costs roughly
    // the same wall-time as 256 old steps while allowing near-terrain horizon
    // rays to correctly reach the fog exit distance.
    let max_steps = 1024u;

    // Finest-level voxel size — drives the adaptive threshold and min step floor
    let vs0 = level_voxel_size(0u);

    // Interleaved gradient noise (Jimenez, SIGGRAPH 2014) — better spatial
    // uniformity than the sin hash and no GPU sin-precision issues (which can
    // produce visible banding on some AMD hardware).  The golden-ratio increment
    // on the frame counter (jitter_frame.z) gives near-optimal temporal
    // convergence when TAA is active.
    let dither = fract(52.9829189 * fract(dot(in.position.xy, vec2<f32>(0.06711056, 0.00583715))
                       + camera.jitter_frame.z * 0.61803398875));

    // Start at the volume entry face (clamp to 0 if camera is already inside),
    // backed off by one fine voxel to avoid missing the entry surface, then
    // dithered by a sub-voxel amount to reduce banding.
    // ── Level-bound cache ─────────────────────────────────────────────────
    // Hot path: 6 float comparisons (already-cached vol_min/max).
    // Cold path: only on level transitions — O(level_count) point_in_level.
    // Degenerate initial box forces a cache fill on the very first step.
    var cur_level   = clip_config.level_count; // sentinel = "invalid"
    var cur_vs      = vs0;
    var cur_vol_min = vec3<f32>( 1e10);
    var cur_vol_max = vec3<f32>(-1e10);

    var t         = max(t_enter - vs0, 0.0) + vs0 * 0.5 * dither;
    var hit       = false;
    var hit_pos   = vec3<f32>(0.0);
    var fog_accum = 0.0;

    for (var step = 0u; step < max_steps; step++) {
        if t >= t_vol_exit { break; }

        let p = ray_origin + ray_dir * t;

        // ── Level cache check ─────────────────────────────────────────────
        // Hot path (same level as previous step): just 6 float compares.
        // Cold path (level transition, ~1% of steps): O(level_count) search
        // + one scroll_state buffer read + cache fill.
        if !(all(p >= cur_vol_min) && all(p <= cur_vol_max)) {
            cur_level = clip_config.level_count;
            for (var i = 0u; i < clip_config.level_count; i++) {
                if point_in_level(i, p) { cur_level = i; break; }
            }
            if cur_level < clip_config.level_count {
                cur_vs         = level_voxel_size(cur_level);
                let brick_step = cur_vs * f32(clip_config.brick_size);
                let half_bgd   = i32(clip_config.brick_grid_dim) / 2;
                let snap       = scroll_state.snap_origins[cur_level].xyz;
                cur_vol_min    = vec3<f32>(f32(snap.x - half_bgd),
                                          f32(snap.y - half_bgd),
                                          f32(snap.z - half_bgd)) * brick_step;
                cur_vol_max    = cur_vol_min + vec3<f32>(f32(clip_config.grid_dim) * cur_vs);
            } else {
                // Outside all levels — degenerate cache, take a huge step.
                cur_vs      = vs0;
                cur_vol_min = vec3<f32>( 1e10);
                cur_vol_max = vec3<f32>(-1e10);
            }
        }

        // ── SDF sample ───────────────────────────────────────────────────
        var d = 1e10;
        if cur_level < clip_config.level_count {
            d = sample_level_trilinear(cur_level, p);
        }

        // ── Adaptive hit threshold ────────────────────────────────────────
        // cur_vs * 0.05: scaled to the current LOD so coarse levels don't
        // false-hit sub-voxel noise; t * 0.0005: accounts for fp error on long rays.
        let threshold = max(cur_vs * 0.05, t * 0.0005);
        if d < threshold {
            hit     = true;
            hit_pos = p;
            break;
        }

        // ── Step size — multiplier AND floor scale with cur_vs ────────────
        // The SDF range at each level is 4 * cur_vs.  Using vs0 as the floor
        // caused grazing rays in coarse levels to stall with tiny steps
        // (vs0 vs cur_vs can differ by 128×).  Now both scale with the level.
        // Multiplier ramps from 0.85 (finest, most conservative) to 0.92
        // (coarsest, smoother / less-quantised SDF).
        let step_mult = 0.85 + f32(cur_level) * 0.01; // 0.85 @ L0 → 0.92 @ L7
        let step_size = max(d * step_mult, cur_vs * 0.2);
        t += step_size;

        fog_accum += step_size * FOG_DENSITY;
        if fog_accum > FOG_EXIT_THRESHOLD { break; }
    }

    if !hit {
        discard;
    }

    // ── Normal estimation ────────────────────────────────────────────────
    let cam_dist = length(hit_pos - ray_origin);
    // cur_level/cur_vs are valid at hit: the loop only reaches `hit = true`
    // via the cache-filled path, so cur_level is the finest level at hit_pos.
    // This eliminates sdf_query_level (O(level_count) search) and the 4×
    // redundant level searches that estimate_normal previously triggered.
    let eps    = cur_vs * 0.5;
    let normal = estimate_normal(hit_pos, eps, cur_level);

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
    if clip_config.bvh_node_count == 0xFFFFFFFFu { // debug mode sentinel (unused in normal operation)
        // Find which clip level the hit is in
        var hit_level = 0u;
        for (var i = 0u; i < clip_config.level_count; i++) {
            if point_in_level(i, hit_pos) {
                hit_level = i;
                break;
            }
        }
        let bvs = level_voxel_size(hit_level);
        let bbs = f32(clip_config.brick_size);
        let brick_world = bvs * bbs;

        // Clip level color: green (fine) → red (coarse)
        let t_col = f32(hit_level) / max(f32(clip_config.level_count - 1u), 1.0);
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
