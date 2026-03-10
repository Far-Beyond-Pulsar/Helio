// SDF Dense Grid Evaluation Compute Shader
//
// Evaluates the SDF edit list at every voxel of a 3D grid
// and stores distances in a 3D storage texture.

struct SdfGridParams {
    volume_min:     vec3<f32>,
    _pad0:          f32,
    volume_max:     vec3<f32>,
    _pad1:          f32,
    grid_dim:       u32,
    edit_count:     u32,
    voxel_size:     f32,
    max_march_dist: f32,
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
    enabled:     u32,
    style:       u32,
    height:      f32,
    amplitude:   f32,
    frequency:   f32,
    octaves:     u32,
    lacunarity:  f32,
    persistence: f32,
}

@group(0) @binding(0) var<uniform> grid_params: SdfGridParams;
@group(0) @binding(1) var<storage, read> edits: array<SdfEdit>;
@group(0) @binding(2) var volume: texture_storage_3d<rgba16float, write>;
@group(0) @binding(3) var<uniform> terrain_params: TerrainParams;

// ─── 3D Value Noise (IQ-style) ───────────────────────────────────────────
// Reference: Inigo Quilez, https://iquilezles.org/articles/

fn hash3(p: vec3<f32>) -> f32 {
    var q = fract(p * 0.3183099 + vec3(0.1, 0.1, 0.1));
    q *= 17.0;
    return fract(q.x * q.y * q.z * (q.x + q.y + q.z));
}

fn noise3(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);

    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    let a = hash3(i + vec3(0.0, 0.0, 0.0));
    let b = hash3(i + vec3(1.0, 0.0, 0.0));
    let c = hash3(i + vec3(0.0, 1.0, 0.0));
    let d = hash3(i + vec3(1.0, 1.0, 0.0));
    let e = hash3(i + vec3(0.0, 0.0, 1.0));
    let ff = hash3(i + vec3(1.0, 0.0, 1.0));
    let g = hash3(i + vec3(0.0, 1.0, 1.0));
    let h = hash3(i + vec3(1.0, 1.0, 1.0));

    let val = mix(mix(mix(a, b, u.x), mix(c, d, u.x), u.y),
                  mix(mix(e, ff, u.x), mix(g, h, u.x), u.y), u.z);

    return val * 2.0 - 1.0;
}

fn terrain_fbm3(p: vec3<f32>) -> f32 {
    var value = 0.0;
    var amplitude = 1.0;
    var freq = 1.0;
    var max_amp = 0.0;

    for (var i = 0u; i < terrain_params.octaves; i++) {
        value += amplitude * noise3(p * freq);
        max_amp += amplitude;
        freq *= terrain_params.lacunarity;
        amplitude *= terrain_params.persistence;
    }
    return value / max_amp;
}

fn terrain_fbm2(x: f32, z: f32) -> f32 {
    return terrain_fbm3(vec3(x, 0.0, z));
}

fn terrain_sdf(world_pos: vec3<f32>) -> f32 {
    if terrain_params.enabled == 0u { return 1e10; }

    let freq = terrain_params.frequency;
    let amp = terrain_params.amplitude;
    let h = terrain_params.height;

    switch terrain_params.style {
        case 0u: {
            let n = terrain_fbm2(world_pos.x * freq, world_pos.z * freq);
            return world_pos.y - (h + n * amp);
        }
        case 1u: {
            let n = terrain_fbm2(world_pos.x * freq, world_pos.z * freq);
            return world_pos.y - (h + n * amp);
        }
        case 2u: {
            let n = terrain_fbm2(world_pos.x * freq, world_pos.z * freq);
            let ridge = 1.0 - abs(n);
            return world_pos.y - (h + ridge * amp);
        }
        case 3u: {
            let density = terrain_fbm3(world_pos * freq);
            let height_bias = (world_pos.y - h) / amp;
            return height_bias - density;
        }
        case 4u: {
            let n = terrain_fbm2(world_pos.x * freq, world_pos.z * freq);
            let dist_xz = length(world_pos.xz);
            let falloff = max(1.0 - min(dist_xz * 0.02, 1.0), 0.0);
            return world_pos.y - (h + n * amp * falloff);
        }
        default: { return 1e10; }
    }
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

fn evaluate_sdf(world_pos: vec3<f32>) -> f32 {
    var dist = terrain_sdf(world_pos);
    for (var i = 0u; i < grid_params.edit_count; i++) {
        let edit = edits[i];
        let local_pos = (edit.transform * vec4<f32>(world_pos, 1.0)).xyz;
        let d = evaluate_edit(local_pos, edit);
        dist = apply_boolean(dist, d, edit.boolean_op, edit.blend_radius);
    }
    return dist;
}

// ─── Compute Entry Point ───────────────────────────────────────────────────

@compute @workgroup_size(4, 4, 4)
fn cs_evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = grid_params.grid_dim;
    if any(gid >= vec3<u32>(dim, dim, dim)) { return; }

    let uvw = (vec3<f32>(gid) + 0.5) / f32(dim);
    let world_pos = mix(grid_params.volume_min, grid_params.volume_max, uvw);
    let dist = evaluate_sdf(world_pos);

    textureStore(volume, gid, vec4<f32>(dist, 0.0, 0.0, 0.0));
}
