// Radiance Probe Update Compute Shader
// Updates spherical harmonics coefficients for each probe

struct RadianceProbe {
    position: vec3<f32>,
    _pad0: f32,
    sh_coefficients: array<vec4<f32>, 9>, // RGB + padding for each SH coefficient
}

// Probe grid configuration
struct ProbeGridConfig {
    dimensions: vec3<u32>,
    spacing: f32,
    origin: vec3<f32>,
    _pad: f32,
}

@group(0) @binding(0)
var<storage, read_write> probes: array<RadianceProbe>;

@group(0) @binding(1)
var<uniform> grid_config: ProbeGridConfig;

@group(0) @binding(2)
var surface_position: texture_2d<f32>;

@group(0) @binding(3)
var surface_normal: texture_2d<f32>;

@group(0) @binding(4)
var surface_albedo: texture_2d<f32>;

// Spherical harmonics basis functions (L=2, 9 coefficients)
fn sh_basis(direction: vec3<f32>) -> array<f32, 9> {
    var basis: array<f32, 9>;

    // L=0
    basis[0] = 0.282095; // Y(0,0)

    // L=1
    basis[1] = 0.488603 * direction.y; // Y(1,-1)
    basis[2] = 0.488603 * direction.z; // Y(1,0)
    basis[3] = 0.488603 * direction.x; // Y(1,1)

    // L=2
    basis[4] = 1.092548 * direction.x * direction.y; // Y(2,-2)
    basis[5] = 1.092548 * direction.y * direction.z; // Y(2,-1)
    basis[6] = 0.315392 * (3.0 * direction.z * direction.z - 1.0); // Y(2,0)
    basis[7] = 1.092548 * direction.x * direction.z; // Y(2,1)
    basis[8] = 0.546274 * (direction.x * direction.x - direction.y * direction.y); // Y(2,2)

    return basis;
}

// Sample irradiance from surface cache
fn sample_surface_cache(probe_pos: vec3<f32>) -> vec3<f32> {
    // For now, return a simple ambient term
    // In production, this would sample the surface cache texture
    // and accumulate lighting from visible surfaces

    let sky_color = vec3<f32>(0.5, 0.7, 1.0);
    let ground_color = vec3<f32>(0.3, 0.25, 0.2);

    // Simple gradient based on height
    let height_factor = clamp((probe_pos.y + 2.0) / 8.0, 0.0, 1.0);
    return mix(ground_color, sky_color, height_factor);
}

// Update a single probe
fn update_probe(probe_index: u32) {
    let grid_dims = grid_config.dimensions;
    let probe_z = probe_index / (grid_dims.x * grid_dims.y);
    let probe_y = (probe_index % (grid_dims.x * grid_dims.y)) / grid_dims.x;
    let probe_x = probe_index % grid_dims.x;

    // Calculate probe world position
    let probe_offset = vec3<f32>(f32(probe_x), f32(probe_y), f32(probe_z)) * grid_config.spacing;
    let probe_pos = grid_config.origin + probe_offset;

    // Update probe position
    probes[probe_index].position = probe_pos;

    // Clear SH coefficients
    for (var i = 0u; i < 9u; i++) {
        probes[probe_index].sh_coefficients[i] = vec4<f32>(0.0);
    }

    // Sample directions around the probe
    let num_samples = 64u;
    let golden_ratio = 1.618033988749895;

    for (var i = 0u; i < num_samples; i++) {
        let fi = f32(i);
        let fn = f32(num_samples);

        // Fibonacci sphere sampling
        let theta = 2.0 * 3.14159265359 * fi / golden_ratio;
        let phi = acos(1.0 - 2.0 * (fi + 0.5) / fn);

        let direction = vec3<f32>(
            sin(phi) * cos(theta),
            sin(phi) * sin(theta),
            cos(phi)
        );

        // Sample radiance in this direction
        let radiance = sample_surface_cache(probe_pos + direction * 0.5);

        // Project into SH
        let sh = sh_basis(direction);
        let weight = 4.0 * 3.14159265359 / fn;

        for (var j = 0u; j < 9u; j++) {
            probes[probe_index].sh_coefficients[j] += vec4<f32>(radiance * sh[j] * weight, 0.0);
        }
    }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let grid_dims = grid_config.dimensions;
    let total_probes = grid_dims.x * grid_dims.y * grid_dims.z;

    let probe_index = global_id.x + global_id.y * grid_dims.x;

    if (probe_index >= total_probes) {
        return;
    }

    update_probe(probe_index);
}
