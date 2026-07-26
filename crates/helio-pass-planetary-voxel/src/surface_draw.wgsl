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

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> pages: array<GpuDrawPage>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) material: u32,
    @location(2) normal: vec3<f32>,
    @location(3) flags: u32,
    @builtin(instance_index) page_slot: u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) world_position: vec3<f32>,
    @location(2) @interpolate(flat) material: u32,
    @location(3) @interpolate(flat) lod: u32,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    let page = pages[input.page_slot];
    let lod_scale = exp2(f32(page.lod));
    let local_lod0_cell = input.position * lod_scale;
    let relative_cell = vec3<f32>(page.relative_lod0_cell_min) + local_lod0_cell;
    let world = relative_cell * page.lod0_cell_size_m - page.camera_relative_m;
    var output: VertexOutput;
    // Keep the view and projection transforms split. The indirect multi-draw
    // path clipped the same vertices on the precombined field on D3D12, while
    // the uploaded matrices and this split transform pass the GPU contract.
    output.clip_position = camera.proj * (camera.view * vec4<f32>(world, 1.0));
    output.normal = normalize(input.normal);
    output.world_position = world;
    output.material = input.material;
    output.lod = page.lod;
    return output;
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

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let sun = normalize(vec3<f32>(0.35, 0.82, 0.44));
    let diffuse = max(dot(normalize(input.normal), sun), 0.0);
    let ambient = 0.16;
    let lod_tint = vec3<f32>(0.02, 0.015, 0.04) * f32(input.lod);
    let color = material_color(input.material) * (ambient + diffuse * 0.84) + lod_tint;
    return vec4<f32>(color, 1.0);
}
