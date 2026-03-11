// SDF Sparse Brick Evaluation Compute Shader
//
// Evaluates the SDF edit list for active bricks only.
// One workgroup per active brick, @workgroup_size(256, 1, 1).
// 256 threads share 729 (9^3) voxels via strided iteration.

struct SdfGridParams {
    volume_min:           vec3<f32>,
    _pad0:                f32,
    volume_max:           vec3<f32>,
    _pad1:                f32,
    grid_dim:             u32,
    edit_count:           u32,
    voxel_size:           f32,
    max_march_dist:       f32,
    brick_size:           u32,
    brick_grid_dim:       u32,
    active_brick_count:   u32,
    atlas_bricks_per_axis: u32,
    grid_origin:          vec3<f32>,
    debug_flags:          u32,
}

struct SdfEdit {
    transform:    mat4x4<f32>,   // world-to-local
    shape_type:   u32,
    boolean_op:   u32,
    blend_radius: f32,
    _pad:         u32,
    params:       vec4<f32>,     // shape-specific parameters
}

struct TerrainParams {
    enabled:     u32,     // 0 = off, 1 = on
    style:       u32,     // 0=Flat, 1=Rolling, 2=Mountains, 3=Caves, 4=Islands
    height:      f32,     // y-offset of ground plane
    amplitude:   f32,     // height variation scale
    frequency:   f32,     // base noise frequency
    octaves:     u32,     // FBM octaves
    lacunarity:  f32,     // frequency multiplier per octave
    persistence: f32,     // amplitude multiplier per octave
}

@group(0) @binding(0) var<uniform> grid_params: SdfGridParams;
@group(0) @binding(1) var<storage, read> edits: array<SdfEdit>;
@group(0) @binding(2) var<storage, read_write> atlas: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> active_bricks: array<u32>;
@group(0) @binding(4) var<storage, read> brick_index: array<u32>;
@group(0) @binding(5) var<uniform> terrain_params: TerrainParams;
@group(0) @binding(6) var<storage, read> edit_list_offsets: array<u32>;
@group(0) @binding(7) var<storage, read> edit_list_data: array<u32>;

// ─── 3D Value Noise (IQ-style) ───────────────────────────────────────────
// Reference: Inigo Quilez, https://iquilezles.org/articles/

// Hash function: maps integer lattice coordinate to pseudo-random value in [0,1].
// Based on IQ's common hash pattern using sin+dot.
fn hash3(p: vec3<f32>) -> f32 {
    var q = fract(p * 0.3183099 + vec3(0.1, 0.1, 0.1));
    q *= 17.0;
    return fract(q.x * q.y * q.z * (q.x + q.y + q.z));
}

// 3D value noise with quintic interpolation. Returns value in [-1, 1].
fn noise3(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);

    // Quintic interpolant (IQ's preferred smoothstep: 6t^5 - 15t^4 + 10t^3)
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    // Sample 8 corners of the lattice cell
    let a = hash3(i + vec3(0.0, 0.0, 0.0));
    let b = hash3(i + vec3(1.0, 0.0, 0.0));
    let c = hash3(i + vec3(0.0, 1.0, 0.0));
    let d = hash3(i + vec3(1.0, 1.0, 0.0));
    let e = hash3(i + vec3(0.0, 0.0, 1.0));
    let ff = hash3(i + vec3(1.0, 0.0, 1.0));
    let g = hash3(i + vec3(0.0, 1.0, 1.0));
    let h = hash3(i + vec3(1.0, 1.0, 1.0));

    // Trilinear interpolation
    let val = mix(mix(mix(a, b, u.x), mix(c, d, u.x), u.y),
                  mix(mix(e, ff, u.x), mix(g, h, u.x), u.y), u.z);

    // Remap from [0,1] to [-1,1]
    return val * 2.0 - 1.0;
}

// ─── FBM (Fractional Brownian Motion) ─────────────────────────────────────

// Domain rotation matrix: breaks axis-aligned artifacts between octaves.
// Base rotation (unit scale). We multiply by lacunarity/2.0 to support
// arbitrary lacunarity values while keeping the rotation.
// Uses IQ's rational-entry matrix divided by 2 (pure rotation component).
const FBM_ROT_0: vec3<f32> = vec3<f32>( 0.00, -0.80, -0.60);
const FBM_ROT_1: vec3<f32> = vec3<f32>( 0.80,  0.36, -0.48);
const FBM_ROT_2: vec3<f32> = vec3<f32>( 0.60, -0.48,  0.64);

fn fbm_rotate(p: vec3<f32>, lac: f32) -> vec3<f32> {
    return lac * vec3<f32>(
        dot(p, vec3<f32>(FBM_ROT_0.x, FBM_ROT_1.x, FBM_ROT_2.x)),
        dot(p, vec3<f32>(FBM_ROT_0.y, FBM_ROT_1.y, FBM_ROT_2.y)),
        dot(p, vec3<f32>(FBM_ROT_0.z, FBM_ROT_1.z, FBM_ROT_2.z)),
    );
}

fn terrain_fbm2(x: f32, z: f32) -> f32 {
    var value = 0.0;
    var amplitude = 1.0;
    var max_amp = 0.0;
    let oct = terrain_params.octaves;
    // LOD-based octave reduction: skip octaves whose feature size is below voxel resolution.
    // At each octave, the feature wavelength halves. When it drops below ~2 voxels the detail
    // is lost in quantization anyway, so we save GPU cycles on coarser clip levels.
    let base_wavelength = 1.0 / terrain_params.frequency;
    let min_wavelength = grid_params.voxel_size * 2.0;
    var sample_pos = vec3(x, 0.0, z);
    var current_wavelength = base_wavelength;

    for (var i = 0u; i < oct; i++) {
        if current_wavelength < min_wavelength { break; }
        value += amplitude * noise3(sample_pos);
        max_amp += amplitude;
        amplitude *= terrain_params.persistence;
        current_wavelength /= terrain_params.lacunarity;
        sample_pos = fbm_rotate(sample_pos, terrain_params.lacunarity);
    }

    if max_amp > 0.0 {
        return value / max_amp;
    }
    return 0.0;
}

// ─── Terrain SDF ──────────────────────────────────────────────────────────

fn terrain_sdf(world_pos: vec3<f32>) -> f32 {
    if terrain_params.enabled == 0u { return 1e10; }

    let freq = terrain_params.frequency;
    let amp = terrain_params.amplitude;
    let h = terrain_params.height;

    let n = terrain_fbm2(world_pos.x * freq, world_pos.z * freq);
    return world_pos.y - (h + n * amp);
}

// ─── SDF Primitives ────────────────────────────────────────────────────────

fn sdf_sphere(p: vec3<f32>, radius: f32) -> f32 {
    return length(p) - radius;
}

fn sdf_cube(p: vec3<f32>, half_extents: vec3<f32>) -> f32 {
    let d = abs(p) - half_extents;
    return length(max(d, vec3<f32>(0.0))) + min(max(d.x, max(d.y, d.z)), 0.0);
}

fn sdf_capsule(p: vec3<f32>, radius: f32, half_height: f32) -> f32 {
    var q = p;
    q.y -= clamp(q.y, -half_height, half_height);
    return length(q) - radius;
}

fn sdf_torus(p: vec3<f32>, major_r: f32, minor_r: f32) -> f32 {
    let q = vec2<f32>(length(p.xz) - major_r, p.y);
    return length(q) - minor_r;
}

fn sdf_cylinder(p: vec3<f32>, radius: f32, half_height: f32) -> f32 {
    let d = abs(vec2<f32>(length(p.xz), p.y)) - vec2<f32>(radius, half_height);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0)));
}

// ─── Boolean Operations ────────────────────────────────────────────────────

fn op_union(d1: f32, d2: f32) -> f32 { return min(d1, d2); }
fn op_subtract(d1: f32, d2: f32) -> f32 { return max(d1, -d2); }
fn op_intersect(d1: f32, d2: f32) -> f32 { return max(d1, d2); }

fn op_smooth_union(d1: f32, d2: f32, k: f32) -> f32 {
    let h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) - k * h * (1.0 - h);
}

fn op_smooth_subtract(d1: f32, d2: f32, k: f32) -> f32 {
    let h = clamp(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0);
    return mix(d1, -d2, h) + k * h * (1.0 - h);
}

fn op_smooth_intersect(d1: f32, d2: f32, k: f32) -> f32 {
    let h = clamp(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) + k * h * (1.0 - h);
}

// ─── Scene Evaluation ──────────────────────────────────────────────────────

fn evaluate_edit(local_pos: vec3<f32>, edit: SdfEdit) -> f32 {
    switch edit.shape_type {
        case 0u: { return sdf_sphere(local_pos, edit.params.x); }
        case 1u: { return sdf_cube(local_pos, edit.params.xyz); }
        case 2u: { return sdf_capsule(local_pos, edit.params.x, edit.params.y); }
        case 3u: { return sdf_torus(local_pos, edit.params.x, edit.params.y); }
        case 4u: { return sdf_cylinder(local_pos, edit.params.x, edit.params.y); }
        default: { return 1e10; }
    }
}

fn apply_boolean(accumulated: f32, shape_dist: f32, op: u32, k: f32) -> f32 {
    let use_blend = k > 0.001;
    switch op {
        case 0u: {
            if use_blend { return op_smooth_union(accumulated, shape_dist, k); }
            return op_union(accumulated, shape_dist);
        }
        case 1u: {
            if use_blend { return op_smooth_subtract(accumulated, shape_dist, k); }
            return op_subtract(accumulated, shape_dist);
        }
        case 2u: {
            if use_blend { return op_smooth_intersect(accumulated, shape_dist, k); }
            return op_intersect(accumulated, shape_dist);
        }
        default: { return accumulated; }
    }
}

fn evaluate_sdf(world_pos: vec3<f32>, brick_idx: u32) -> f32 {
    // Start from terrain surface (or empty space if terrain is disabled)
    var dist = terrain_sdf(world_pos);

    // Apply only the edits that overlap this brick (per-brick edit list culling)
    let offset = edit_list_offsets[brick_idx];
    let count = edit_list_data[offset];
    for (var j = 0u; j < count; j++) {
        let i = edit_list_data[offset + 1u + j];
        let edit = edits[i];
        let local_pos = (edit.transform * vec4<f32>(world_pos, 1.0)).xyz;
        let d = evaluate_edit(local_pos, edit);
        dist = apply_boolean(dist, d, edit.boolean_op, edit.blend_radius);
    }
    return dist;
}

// ─── Atlas quantization helpers ─────────────────────────────────────────
// Quantize SDF distance to u8 [0..255]. Range is [-half_diag, +half_diag]
// where half_diag = voxel_size * brick_size * sqrt(3) / 2.
// stored = clamp((d / half_diag) * 0.5 + 0.5, 0, 1) * 255

fn atlas_half_diag() -> f32 {
    return grid_params.voxel_size * f32(grid_params.brick_size) * 0.866025403; // sqrt(3)/2
}

fn quantize_distance(d: f32) -> u32 {
    let hd = atlas_half_diag();
    let normalized = clamp(d / hd * 0.5 + 0.5, 0.0, 1.0);
    return u32(normalized * 255.0 + 0.5);
}

fn atlas_linear_index(coord: vec3<u32>) -> u32 {
    let dim = grid_params.atlas_bricks_per_axis * PADDED_SIZE;
    return coord.x + coord.y * dim + coord.z * dim * dim;
}

// ─── Compute Entry Point ───────────────────────────────────────────────────

const PADDED_SIZE: u32 = 9u;
const VOXELS_PER_BRICK: u32 = 729u; // 9 * 9 * 9

@compute @workgroup_size(256, 1, 1)
fn cs_evaluate_sparse(
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    // wid.x = index into active_bricks array
    if wid.x >= grid_params.active_brick_count { return; }

    let brick_linear = active_bricks[wid.x];
    let atlas_slot = brick_index[brick_linear];

    // Decode brick grid position from linear index
    let bgd = grid_params.brick_grid_dim;
    let brick_coord = vec3<u32>(
        brick_linear % bgd,
        (brick_linear / bgd) % bgd,
        brick_linear / (bgd * bgd),
    );

    // Compute atlas write origin from atlas_slot (using padded stride)
    let bpa = grid_params.atlas_bricks_per_axis;
    let atlas_brick = vec3<u32>(
        atlas_slot % bpa,
        (atlas_slot / bpa) % bpa,
        atlas_slot / (bpa * bpa),
    );
    let bs = grid_params.brick_size; // 8 (logical)
    let atlas_origin = atlas_brick * PADDED_SIZE;

    // Toroidal unwrapping constants.
    // grid_origin holds floor(volume_min / voxel_size) — the absolute
    // world-voxel coordinate of the volume's min corner.
    let dim = i32(grid_params.grid_dim);
    let world_voxel_origin = vec3<i32>(grid_params.grid_origin);
    let wrapped_origin = ((world_voxel_origin % dim) + dim) % dim;

    // Compute this brick's world-space voxel offset once.
    let brick_gv = vec3<i32>(brick_coord * bs);
    let brick_offset = ((brick_gv - wrapped_origin + dim) % dim);
    let brick_world_base = world_voxel_origin + brick_offset;

    // 256 threads share 729 voxels. Each thread processes ~3 voxels
    // via a strided loop for better occupancy and memory access patterns.
    for (var v = lid; v < VOXELS_PER_BRICK; v += 256u) {
        let local = vec3<u32>(
            v % PADDED_SIZE,
            (v / PADDED_SIZE) % PADDED_SIZE,
            v / (PADDED_SIZE * PADDED_SIZE),
        );
        let world_voxel_abs = brick_world_base + vec3<i32>(local);
        let world_pos = vec3<f32>(world_voxel_abs) * grid_params.voxel_size;
        let dist = evaluate_sdf(world_pos, wid.x);

        let atlas_coord = atlas_origin + local;
        let linear = atlas_linear_index(atlas_coord);
        let word_idx = linear / 4u;
        let byte_idx = linear % 4u;
        let q = quantize_distance(dist);
        // Pack u8 into the correct byte of the u32 word via atomic CAS loop.
        // A two-step atomicAnd+atomicOr would race if two workgroups write
        // different bytes of the same u32 simultaneously.  CAS retries on
        // contention so every byte is written correctly.
        let mask = 0xFFu << (byte_idx * 8u);
        let shifted = q << (byte_idx * 8u);
        var old = atomicLoad(&atlas[word_idx]);
        loop {
            let desired = (old & ~mask) | shifted;
            let result = atomicCompareExchangeWeak(&atlas[word_idx], old, desired);
            if result.exchanged { break; }
            old = result.old_value;
        }
    }
}
