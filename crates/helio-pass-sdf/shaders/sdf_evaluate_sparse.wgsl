// sdf_evaluate_sparse.wgsl
// Compute shader: bake SDF edits + terrain into the sparse brick atlas.
//
// One workgroup invocation per active brick. The shader:
//   1. Reads the brick world AABB from grid params + active_bricks list.
//   2. Evaluates each voxel inside the brick against the per-brick edit list.
//   3. Writes quantised u8 SDF values into the atlas (4 voxels per u32 word).
//
// Bindings (group 0):
//   0  params             SdfGridParams    (uniform)
//   1  edits              array<SdfEdit>   (storage read)
//   2  atlas              array<atomic<u32>> (storage read_write)
//   3  active_bricks      array<u32>       (storage read)
//   4  brick_index        array<u32>       (storage read)
//   5  terrain_params     TerrainParams    (uniform)
//   6  edit_list_offsets  array<u32>       (storage read)
//   7  edit_list_data     array<u32>       (storage read)

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------

struct SdfGridParams {
    world_min:          vec3<f32>,
    voxel_size:         f32,
    grid_dim:           vec3<u32>,
    brick_size:         u32,
    active_brick_count: u32,
    atlas_capacity:     u32,
    edit_count:         u32,
    terrain_enabled:    u32,
    atlas_dim:          vec3<u32>,
    _pad:               u32,
}

struct SdfEdit {
    transform:    mat4x4<f32>,   // world-to-local
    shape_type:   u32,
    boolean_op:   u32,
    blend_radius: f32,
    _pad0:        u32,
    param0:       f32,
    param1:       f32,
    param2:       f32,
    param3:       f32,
}

struct TerrainParams {
    style:       u32,
    height:      f32,
    amplitude:   f32,
    frequency:   f32,
    octaves:     u32,
    lacunarity:  f32,
    persistence: f32,
    _pad:        u32,
}

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform>  params:            SdfGridParams;
@group(0) @binding(1) var<storage, read> edits:        array<SdfEdit>;
@group(0) @binding(2) var<storage, read_write> atlas:  array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> active_bricks: array<u32>;
@group(0) @binding(4) var<storage, read> brick_index:  array<u32>;
@group(0) @binding(5) var<uniform>  terrain_params:    TerrainParams;
@group(0) @binding(6) var<storage, read> edit_list_offsets: array<u32>;
@group(0) @binding(7) var<storage, read> edit_list_data:    array<u32>;

// ---------------------------------------------------------------------------
// IQ noise helpers
// ---------------------------------------------------------------------------

fn hash3(p: vec3<f32>) -> f32 {
    var q = fract(p * 0.3183099 + vec3<f32>(0.1)) * 17.0;
    return fract(q.x * q.y * q.z * (q.x + q.y + q.z));
}

fn noise3(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    let a = hash3(i + vec3<f32>(0.0, 0.0, 0.0));
    let b = hash3(i + vec3<f32>(1.0, 0.0, 0.0));
    let c = hash3(i + vec3<f32>(0.0, 1.0, 0.0));
    let d = hash3(i + vec3<f32>(1.0, 1.0, 0.0));
    let e = hash3(i + vec3<f32>(0.0, 0.0, 1.0));
    let fv = hash3(i + vec3<f32>(1.0, 0.0, 1.0));
    let g_ = hash3(i + vec3<f32>(0.0, 1.0, 1.0));
    let h_ = hash3(i + vec3<f32>(1.0, 1.0, 1.0));

    return mix(mix(mix(a, b, u.x), mix(c, d, u.x), u.y),
               mix(mix(e, fv, u.x), mix(g_, h_, u.x), u.y), u.z) * 2.0 - 1.0;
}

// IQ domain-rotation matrix for FBM.
fn fbm_rotate(p: vec3<f32>, lac: f32) -> vec3<f32> {
    let m = mat3x3<f32>(
        vec3<f32>( 0.00,  0.80,  0.60),
        vec3<f32>(-0.80,  0.36, -0.48),
        vec3<f32>(-0.60, -0.48,  0.64)
    );
    return lac * (m * p);
}

fn terrain_fbm2(xz: vec2<f32>) -> f32 {
    var value     = 0.0;
    var amplitude = 1.0;
    var max_amp   = 0.0;
    var p = vec3<f32>(xz.x, 0.0, xz.y);
    let oct = terrain_params.octaves;
    let lac = terrain_params.lacunarity;
    let per = terrain_params.persistence;
    for (var i = 0u; i < oct; i++) {
        value   += amplitude * noise3(p);
        max_amp += amplitude;
        amplitude *= per;
        p = fbm_rotate(p, lac);
    }
    if max_amp > 0.0 { return value / max_amp; }
    return 0.0;
}

fn terrain_sdf(pos: vec3<f32>) -> f32 {
    let n = terrain_fbm2(pos.xz * terrain_params.frequency);
    return pos.y - (terrain_params.height + n * terrain_params.amplitude);
}

// ---------------------------------------------------------------------------
// SDF primitives (local space)
// ---------------------------------------------------------------------------

fn sd_sphere(p: vec3<f32>, r: f32) -> f32 {
    return length(p) - r;
}

fn sd_box(p: vec3<f32>, b: vec3<f32>) -> f32 {
    let q = abs(p) - b;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn sd_capsule(p: vec3<f32>, r: f32, hh: f32) -> f32 {
    var q = p;
    q.y = q.y - clamp(q.y, -hh, hh);
    return length(q) - r;
}

fn sd_torus(p: vec3<f32>, R: f32, r: f32) -> f32 {
    let q = vec2<f32>(length(p.xz) - R, p.y);
    return length(q) - r;
}

fn sd_cylinder(p: vec3<f32>, r: f32, hh: f32) -> f32 {
    let d = abs(vec2<f32>(length(p.xz), p.y)) - vec2<f32>(r, hh);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0)));
}

fn sdf_eval_shape(p_local: vec3<f32>, edit: SdfEdit) -> f32 {
    let st = edit.shape_type;
    if st == 0u { return sd_sphere(p_local, edit.param0); }
    if st == 1u { return sd_box(p_local, vec3<f32>(edit.param0, edit.param1, edit.param2)); }
    if st == 2u { return sd_capsule(p_local, edit.param0, edit.param1); }
    if st == 3u { return sd_torus(p_local, edit.param0, edit.param1); }
    return sd_cylinder(p_local, edit.param0, edit.param1);
}

// Smooth boolean combinators.
fn sdf_union(a: f32, b: f32, k: f32) -> f32 {
    if k <= 0.0 { return min(a, b); }
    let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

fn sdf_subtraction(a: f32, b: f32, k: f32) -> f32 {
    if k <= 0.0 { return max(-b, a); }
    let h = clamp(0.5 - 0.5 * (a + b) / k, 0.0, 1.0);
    return mix(a, -b, h) + k * h * (1.0 - h);
}

fn sdf_intersection(a: f32, b: f32, k: f32) -> f32 {
    if k <= 0.0 { return max(a, b); }
    let h = clamp(0.5 - 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) + k * h * (1.0 - h);
}

// ---------------------------------------------------------------------------
// Full SDF evaluation at world pos (brick-local edit list)
// ---------------------------------------------------------------------------

fn eval_sdf(world_pos: vec3<f32>, brick_flat: u32) -> f32 {
    var d = 1e9;

    // Terrain.
    if params.terrain_enabled != 0u {
        d = terrain_sdf(world_pos);
    }

    // Per-brick edit list.
    let offset_start = edit_list_offsets[brick_flat];
    let offset_end   = edit_list_offsets[brick_flat + 1u];
    for (var k = offset_start; k < offset_end; k++) {
        let ei = edit_list_data[k];
        if ei >= params.edit_count { continue; }
        let edit = edits[ei];
        let p_local = (edit.transform * vec4<f32>(world_pos, 1.0)).xyz;
        let sd = sdf_eval_shape(p_local, edit);
        let bl = edit.blend_radius;
        let op = edit.boolean_op;
        if op == 0u { d = sdf_union(d, sd, bl); }
        else if op == 1u { d = sdf_subtraction(d, sd, bl); }
        else { d = sdf_intersection(d, sd, bl); }
    }
    return d;
}

// ---------------------------------------------------------------------------
// Atlas packing helpers (4 × u8 per u32, big-endian byte order within word)
// ---------------------------------------------------------------------------

fn pack_u8(value: f32, voxel_size: f32) -> u32 {
    // Map [-max_dist, +max_dist] → [0, 255].
    let max_d = voxel_size * 4.0;
    let normalised = clamp((value / max_d) * 0.5 + 0.5, 0.0, 1.0);
    return u32(normalised * 255.0);
}

fn atlas_base_index(atlas_idx: u32, local: vec3<u32>) -> u32 {
    let bs  = params.brick_size;
    let adim = params.atlas_dim;
    // Atlas brick position.
    let ax = atlas_idx % adim.x;
    let ay = (atlas_idx / adim.x) % adim.y;
    let az = atlas_idx / (adim.x * adim.y);
    let vx = ax * bs + local.x;
    let vy = ay * bs + local.y;
    let vz = az * bs + local.z;
    let stride_x = adim.x * bs;
    let stride_y = adim.y * bs;
    // Flat voxel index.
    let flat_voxel = vz * stride_y * stride_x + vy * stride_x + vx;
    return flat_voxel / 4u;
}

fn atlas_byte_shift(flat_voxel: u32) -> u32 {
    return (flat_voxel % 4u) * 8u;
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

@compute @workgroup_size(256, 1, 1)
fn cs_evaluate_sparse(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let brick_list_idx = global_id.x;
    if brick_list_idx >= params.active_brick_count { return; }

    // Look up which grid cell this is.
    let grid_flat = active_bricks[brick_list_idx];
    let atlas_idx = brick_index[grid_flat];
    if atlas_idx == 0xFFFFFFFFu { return; }

    // Decode grid brick position.
    let gd = params.grid_dim;
    let bx = grid_flat % gd.x;
    let by = (grid_flat / gd.x) % gd.y;
    let bz = grid_flat / (gd.x * gd.y);

    let bs = params.brick_size;
    let vs = params.voxel_size;
    let brick_world_min = params.world_min + vec3<f32>(f32(bx) * f32(bs), f32(by) * f32(bs), f32(bz) * f32(bs)) * vs;

    // Evaluate each voxel inside the brick.
    // Each invocation handles all bs^3 voxels (typically 8^3 = 512).
    // Single-invocation bricks is fine for small brick sizes.
    for (var lz = 0u; lz < bs; lz++) {
        for (var ly = 0u; ly < bs; ly++) {
            for (var lx = 0u; lx < bs; lx++) {
                let world_pos = brick_world_min
                    + vec3<f32>(f32(lx) + 0.5, f32(ly) + 0.5, f32(lz) + 0.5) * vs;
                let d = eval_sdf(world_pos, grid_flat);
                let packed_byte = pack_u8(d, vs);

                let flat_voxel = (atlas_idx * bs * bs * bs) +
                                  lz * bs * bs + ly * bs + lx;
                let word_idx  = flat_voxel / 4u;
                let shift     = (flat_voxel % 4u) * 8u;
                let mask      = 0xFFu << shift;
                let val       = packed_byte << shift;

                // Atomic CAS loop to pack 4 u8 values into a single u32 word.
                var old = atomicLoad(&atlas[word_idx]);
                loop {
                    let new_val = (old & ~mask) | val;
                    let result = atomicCompareExchangeWeak(&atlas[word_idx], old, new_val);
                    if result.exchanged { break; }
                    old = result.old_value;
                }
            }
        }
    }
}
