// Browser WebGPU render half of the Corona particle system.
//
// Compute uses read-write storage bindings, which WebGPU forbids from being
// visible to vertex stages. Keeping render in a separate module lets the same
// buffers be declared read-only for vertex processing.

const ATLAS_COLS: u32 = 4u;

struct GpuCoronaUniforms {
    delta_time:      f32,
    total_particles: u32,
    emitter_count:   u32,
    frame_count:     u32,
    sort_k:          u32,
    sort_j:          u32,
    sort_lo:         u32,
    sort_n:          u32,
}

struct Particle {
    pos_and_alive:     vec4<f32>,
    velocity:          vec4<f32>,
    color:             vec4<f32>,
    size_lifetime_age: vec4<f32>,
}

struct EmitterDef {
    transform:          mat4x4<f32>,
    emit_params:        vec4<f32>,
    size_params:        vec4<f32>,
    start_color:        vec4<f32>,
    end_color:          vec4<f32>,
    velocity:           vec4<f32>,
    velocity_variation: vec4<f32>,
    extras:             vec4<f32>,
    texture_index:      i32,
    particle_offset:    u32,
    particle_count:     u32,
    spawn_cursor:       u32,
    _pad:               array<f32, 12>,
}

struct CameraUniforms {
    view:           mat4x4<f32>,
    proj:           mat4x4<f32>,
    view_proj:      mat4x4<f32>,
    inv_view_proj:  mat4x4<f32>,
    position_near:  vec4<f32>,
    forward_far:    vec4<f32>,
    jitter_frame:   vec4<f32>,
    prev_view_proj: mat4x4<f32>,
}

@group(0) @binding(0)  var<uniform>       uniforms:         GpuCoronaUniforms;
@group(0) @binding(1)  var<storage, read> particles:        array<Particle>;
@group(0) @binding(2)  var<storage, read> emitters:         array<EmitterDef>;
@group(0) @binding(3)  var<storage, read> compact_buf:      array<u32>;
@group(0) @binding(6)  var<uniform>       camera:           CameraUniforms;
@group(0) @binding(10) var                particle_tex:     texture_2d<f32>;
@group(0) @binding(11) var                particle_sampler: sampler;

struct VOut {
    @builtin(position)              pos:          vec4<f32>,
    @location(0)                    uv:           vec2<f32>,
    @location(1)                    color:        vec4<f32>,
    @location(2) @interpolate(flat) sprite_index: u32,
}

fn quad_corner(idx: u32) -> vec2<f32> {
    let corners = array<vec2<f32>, 6>(
        vec2<f32>(-0.5, -0.5), vec2<f32>(0.5, -0.5), vec2<f32>(-0.5,  0.5),
        vec2<f32>(-0.5,  0.5), vec2<f32>(0.5, -0.5), vec2<f32>( 0.5,  0.5),
    );
    return corners[idx];
}

fn quad_uv(idx: u32) -> vec2<f32> {
    let coordinates = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
    );
    return coordinates[idx];
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32, @builtin(instance_index) instance_index: u32) -> VOut {
    let particle_index = compact_buf[instance_index];
    let particle = particles[particle_index];
    let right = vec3<f32>(camera.view[0].x, camera.view[1].x, camera.view[2].x);
    let up = vec3<f32>(camera.view[0].y, camera.view[1].y, camera.view[2].y);
    let size = max(particle.size_lifetime_age.x, 0.001);
    let corner = quad_corner(vertex_index);
    let world_pos = particle.pos_and_alive.xyz + right * corner.x * size + up * corner.y * size;
    let emitter_index = min(u32(particle.velocity.w + 0.5), uniforms.emitter_count - 1u);
    let texture_index = emitters[emitter_index].texture_index;

    var out: VOut;
    out.pos = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.uv = quad_uv(vertex_index);
    out.color = particle.color;
    out.sprite_index = select(u32(texture_index) % 16u, 0u, texture_index < 0);
    return out;
}

@fragment
fn fs_main(input: VOut) -> @location(0) vec4<f32> {
    let column = input.sprite_index % ATLAS_COLS;
    let row = input.sprite_index / ATLAS_COLS;
    let atlas_uv = vec2<f32>(
        (f32(column) + input.uv.x) * 0.25,
        (f32(row) + input.uv.y) * 0.25,
    );
    let texel = textureSample(particle_tex, particle_sampler, atlas_uv);
    let alpha = texel.a * input.color.a;
    if alpha < 0.005 {
        discard;
        return vec4<f32>(0.0);
    }
    return vec4<f32>(input.color.rgb * texel.rgb, alpha);
}
