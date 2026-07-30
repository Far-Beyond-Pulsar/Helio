struct Camera {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    position_near: vec4<f32>,
    forward_far: vec4<f32>,
    jitter_frame: vec4<f32>,
    prev_view_proj: mat4x4<f32>,
}

struct GpuDrawPage {
    relative_lod0_cell_min: vec3<i32>,
    lod: u32,
    camera_relative_m: vec3<f32>,
    lod0_cell_size_m: f32,
    generation_low: u32,
    generation_high: u32,
    transition_mask: u32,
    visible: u32,
}

struct GpuTerrainDraw {
    page_slot: u32,
    meshlet_index: u32,
    surface_kind: u32,
    lod: u32,
}

struct GpuTerrainDebugUniform {
    mode: u32,
    draw_path: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> pages: array<GpuDrawPage>;
@group(0) @binding(2) var<storage, read> draws: array<GpuTerrainDraw>;
@group(0) @binding(3) var<uniform> debug: GpuTerrainDebugUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) material: u32,
    @location(2) normal: vec3<f32>,
    @location(3) flags: u32,
    @builtin(instance_index) draw_index: u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) world_position: vec3<f32>,
    @location(2) @interpolate(flat) material: u32,
    @location(3) @interpolate(flat) lod: u32,
    @location(4) @interpolate(flat) page_slot: u32,
    @location(5) @interpolate(flat) meshlet_index: u32,
    @location(6) @interpolate(flat) surface_kind: u32,
    @location(7) @interpolate(flat) transition_mask: u32,
}

fn transform_vertex(
    input: VertexInput,
    page_slot: u32,
    meshlet_index: u32,
    surface_kind: u32,
) -> VertexOutput {
    let page = pages[page_slot];
    let lod_scale = exp2(f32(page.lod));
    let relative_cell =
        vec3<f32>(page.relative_lod0_cell_min) + input.position * lod_scale;
    let world = relative_cell * page.lod0_cell_size_m - page.camera_relative_m;
    var output: VertexOutput;
    // Keep view and projection split. This is the stable D3D12 contract used
    // by the retained page baseline as well as the compact meshlet path.
    output.clip_position = camera.proj * (camera.view * vec4<f32>(world, 1.0));
    output.normal = normalize(input.normal);
    output.world_position = world;
    output.material = input.material;
    output.lod = page.lod;
    output.page_slot = page_slot;
    output.meshlet_index = meshlet_index;
    output.surface_kind = surface_kind;
    output.transition_mask = page.transition_mask;
    return output;
}

@vertex
fn vs_page(input: VertexInput) -> VertexOutput {
    return transform_vertex(input, input.draw_index, 0u, 0u);
}

@vertex
fn vs_meshlet(input: VertexInput) -> VertexOutput {
    let draw = draws[input.draw_index];
    return transform_vertex(
        input,
        draw.page_slot,
        draw.meshlet_index,
        draw.surface_kind,
    );
}

fn material_color(material: u32) -> vec3<f32> {
    switch material {
        case 1u: { return vec3<f32>(0.18, 0.48, 0.24); }
        case 2u: { return vec3<f32>(0.55, 0.34, 0.16); }
        case 3u: { return vec3<f32>(0.42, 0.46, 0.52); }
        case 4u: { return vec3<f32>(0.72, 0.63, 0.25); }
        default: {
            let hue = f32((material * 37u) & 255u) / 255.0;
            return vec3<f32>(0.25 + hue * 0.45, 0.32 + hue * 0.25, 0.38 + hue * 0.18);
        }
    }
}

fn hash_color(value: u32) -> vec3<f32> {
    var x = value * 747796405u + 2891336453u;
    x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    x = (x >> 22u) ^ x;
    return vec3<f32>(
        f32(x & 255u),
        f32((x >> 8u) & 255u),
        f32((x >> 16u) & 255u),
    ) / 255.0 * 0.75 + vec3<f32>(0.2);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(input.normal);
    var color: vec3<f32>;
    switch debug.mode {
        case 1u: {
            // This is keyed by the actual compact draw's meshlet descriptor.
            color = hash_color(input.meshlet_index ^ (input.surface_kind * 0x9e3779b9u));
        }
        case 2u: {
            color = hash_color(input.page_slot);
        }
        case 3u: {
            color = hash_color(input.lod * 131u + 17u);
        }
        case 4u: {
            color = select(
                vec3<f32>(0.12, 0.15, 0.18),
                vec3<f32>(1.0, 0.12, 0.7),
                input.surface_kind != 0u && input.transition_mask != 0u,
            );
        }
        case 5u: {
            color = normal * 0.5 + vec3<f32>(0.5);
        }
        default: {
            let sun = normalize(vec3<f32>(0.35, 0.82, 0.44));
            let diffuse = max(dot(normal, sun), 0.0);
            let lod_tint = vec3<f32>(0.02, 0.015, 0.04) * f32(input.lod);
            color = material_color(input.material) * (0.16 + diffuse * 0.84) + lod_tint;
        }
    }
    return vec4<f32>(color, 1.0);
}
