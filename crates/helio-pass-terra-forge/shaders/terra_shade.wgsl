// Terra Forge v2 — Fullscreen shading pass.
// Reads material + normal from the compute ray march, applies lighting, writes to screen.

struct Camera {
    view:           mat4x4<f32>,
    proj:           mat4x4<f32>,
    view_proj:      mat4x4<f32>,
    inv_view_proj:  mat4x4<f32>,
    position_near:  vec4<f32>,
    forward_far:    vec4<f32>,
    jitter_frame:   vec4<f32>,
    prev_view_proj: mat4x4<f32>,
}

struct GpuMaterial {
    color:     vec3<f32>,
    roughness: f32,
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var in_material: texture_2d<u32>;
@group(0) @binding(2) var in_normal:   texture_2d<f32>;
@group(0) @binding(3) var<storage, read> palette: array<GpuMaterial>;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    // Fullscreen triangle.
    let x = f32(i32(vid & 1u) * 4 - 1);
    let y = f32(i32(vid >> 1u) * 4 - 1);
    var out: VsOut;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.uv  = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

@fragment
fn fs_main(inp: VsOut) -> @location(0) vec4<f32> {
    let coord = vec2<i32>(inp.pos.xy);

    // Read from full-res textures directly.
    let sample_coord = vec2<i32>(clamp(coord, vec2<i32>(0),
        vec2<i32>(textureDimensions(in_material)) - vec2<i32>(1)));

    let mat_raw = textureLoad(in_material, sample_coord, 0).r;
    if mat_raw == 0u {
        // Sky gradient.
        let up_frac = inp.uv.y;
        let sky_top = vec3<f32>(0.35, 0.55, 0.9);
        let sky_bot = vec3<f32>(0.65, 0.75, 0.9);
        let sky = mix(sky_top, sky_bot, up_frac);
        return vec4<f32>(sky, 1.0);
    }

    let normal_depth = textureLoad(in_normal, sample_coord, 0);
    let N = normalize(normal_depth.xyz);

    // Material from palette. Clamp to palette size.
    let mat_index = min(mat_raw, 255u);
    let mat = palette[mat_index];

    // Simple directional lighting.
    let sun_dir = normalize(vec3<f32>(0.6, 1.0, 0.4));
    let ndotl = max(dot(N, sun_dir), 0.0);
    let ambient = 0.25;
    let diffuse = ndotl * 0.75;
    let color = mat.color * (ambient + diffuse);

    return vec4<f32>(color, 1.0);
}
