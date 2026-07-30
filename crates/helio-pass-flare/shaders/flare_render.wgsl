@group(0) @binding(0) var<storage, read> flare_queries: array<GpuFlareQuery>;
@group(0) @binding(1) var<storage, read> flare_count: array<u32>;
@group(0) @binding(2) var flare_atlas: texture_2d<f32>;
@group(0) @binding(3) var flare_sampler: sampler;
@group(0) @binding(4) var<uniform> flare_uniforms: FlareUniforms;

struct GpuFlareQuery {
    screen_pos:      vec2<f32>,
    screen_depth:    f32,
    light_intensity: f32,
    light_color:     vec3<f32>,
    light_index:     u32,
};

struct FlareUniforms {
    light_count:   u32,
    max_flares:    u32,
    screen_width:  f32,
    screen_height: f32,
};

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

const ATLAS_CELLS: u32 = 4u;
const CS: f32 = 1.0 / f32(ATLAS_CELLS);

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    let uv = vec2<f32>(x, y);
    return VertexOutput(vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0), uv);
}

fn cell_uv(idx: u32, offset: vec2<f32>) -> vec2<f32> {
    let col = idx % ATLAS_CELLS;
    let row = idx / ATLAS_CELLS;
    return vec2<f32>(f32(col), f32(row)) * CS + offset * CS + 0.5 * CS;
}

fn sample_atlas(idx: u32, offset: vec2<f32>) -> vec3<f32> {
    let uv = cell_uv(idx, offset);
    return textureSampleLevel(flare_atlas, flare_sampler, uv, 0.0).rgb;
}

fn sample_atlas_chromatic(idx: u32, offset: vec2<f32>, chroma_shift: f32) -> vec3<f32> {
    let base = cell_uv(idx, offset);
    let shift = vec2<f32>(chroma_shift, 0.0);
    let r = textureSampleLevel(flare_atlas, flare_sampler, base + shift, 0.0).r;
    let g = textureSampleLevel(flare_atlas, flare_sampler, base, 0.0).g;
    let b = textureSampleLevel(flare_atlas, flare_sampler, base - shift, 0.0).b;
    return vec3<f32>(r, g, b);
}

fn anamorphic_streak(input_uv: vec2<f32>, flare_uv: vec2<f32>, light_col: vec3<f32>,
                     screen_dims: vec2<f32>, dir: vec2<f32>) -> vec3<f32> {
    let delta = input_uv - flare_uv;
    let parallel = dot(delta, dir);
    let perp_vec = delta - dir * parallel;
    let perp_len = length(perp_vec * screen_dims);
    let streak = exp(-perp_len * perp_len * 0.003) * 0.008;
    return light_col * streak;
}

fn diffraction_spike(input_uv: vec2<f32>, flare_uv: vec2<f32>, light_col: vec3<f32>,
                     screen_dims: vec2<f32>, spike_count: u32) -> vec3<f32> {
    let delta = input_uv - flare_uv;
    let dist = length(delta * screen_dims);
    if dist < 1.0 { return vec3<f32>(0.0); }
    let angle = atan2(delta.y, delta.x);
    let spike_width = 0.04;
    var spike = 0.0;
    for (var i = 0u; i < spike_count; i++) {
        let a = f32(i) * 6.2832 / f32(spike_count) + 1.5708;
        let d = abs(angle - a);
        let norm = d % 6.2832;
        let wrapped = min(norm, 6.2832 - norm);
        if wrapped < spike_width {
            spike = max(spike, 1.0 - wrapped / spike_width);
        }
    }
    let falloff = exp(-dist * 0.004);
    return light_col * spike * falloff * 0.004;
}

@fragment
fn fs_flare(input: VertexOutput) -> @location(0) vec4<f32> {
    let num_flares = flare_count[0];
    if num_flares == 0u { return vec4<f32>(0.0); }

    let screen_dims = vec2<f32>(flare_uniforms.screen_width, flare_uniforms.screen_height);
    if screen_dims.x < 1.0 || screen_dims.y < 1.0 { return vec4<f32>(0.0); }

    let centre_uv = vec2<f32>(0.5, 0.5);
    var result = vec3<f32>(0.0);

    for (var fi = 0u; fi < num_flares; fi++) {
        let flare = flare_queries[fi];
        let intensity = flare.light_intensity;
        if intensity < 0.01 { continue; }

        let sp = flare.screen_pos;
        let flare_uv = sp / screen_dims;
        let edge_dist = min(min(flare_uv.x, 1.0 - flare_uv.x), min(flare_uv.y, 1.0 - flare_uv.y));
        let edge_fade = smoothstep(0.0, 0.25, edge_dist);
        let light_col = flare.light_color * intensity * edge_fade;
        let to_centre = centre_uv - flare_uv;
        let dist_px = length(to_centre * screen_dims);
        let dir = select(normalize(to_centre), vec2<f32>(1.0, 0.0), dist_px < 1.0);
        let dist_norm = dist_px / length(screen_dims);

        // Streaks
        result += anamorphic_streak(input.uv, flare_uv, light_col, screen_dims, vec2<f32>(1.0, 0.0));
        result += anamorphic_streak(input.uv, flare_uv, light_col * 0.5, screen_dims, vec2<f32>(0.0, 1.0));

        // Diffraction spikes
        result += diffraction_spike(input.uv, flare_uv, light_col, screen_dims, 6u);

        // Ghost reflections (use atlas cells 0-5)
        for (var gi = 0u; gi < 6u; gi++) {
            let t = (f32(gi) + 1.0) / 7.0;
            let ghost_dist = dist_px * t;
            let chroma = 0.002 + dist_norm * t * 0.008;
            let ghost_uv = flare_uv + dir * ghost_dist / screen_dims;
            let dx = input.uv - ghost_uv;
            let ghost_size = (20.0 + t * 40.0) * (1.0 + dist_norm * 0.5);
            let ghost_r = length(dx * screen_dims) / ghost_size;

            if ghost_r < 1.0 {
                let atlas_offset = dx / vec2<f32>(ghost_size) * 1.2;
                let atlas_col = sample_atlas_chromatic(gi, atlas_offset, chroma * 5.0);
                let alpha = (1.0 - ghost_r) * (1.0 - t) * 0.2;
                let occlusion = 1.0 - dist_norm * 0.3;
                result += light_col * atlas_col * alpha * occlusion;
            }
        }

        // Central halo (atlas cell 6)
        let dh = input.uv - flare_uv;
        let hr = length(dh * screen_dims);
        let halo_alpha = exp(-hr * hr * 0.0003) * 0.3;
        let chroma_h = 0.003 + dist_norm * 0.005;
        let halo_uv = cell_uv(6u, dh * 0.3);
        let halo_r = textureSampleLevel(flare_atlas, flare_sampler, halo_uv + vec2<f32>(chroma_h, 0.0), 0.0).r;
        let halo_g = textureSampleLevel(flare_atlas, flare_sampler, halo_uv, 0.0).g;
        let halo_b = textureSampleLevel(flare_atlas, flare_sampler, halo_uv - vec2<f32>(chroma_h, 0.0), 0.0).b;
        result += light_col * vec3<f32>(halo_r, halo_g, halo_b) * halo_alpha;
    }

    return vec4<f32>(result, 1.0);
}
