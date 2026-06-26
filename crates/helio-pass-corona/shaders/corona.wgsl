// ── Corona GPU Particle System ─────────────────────────────────────────────
// Entry points:
//   cs_simulate    — advance ages, update physics, kill expired
//   cs_emit        — spawn new particles via ring-buffer cursor
//   cs_count_alive — atomic count of living particles
//   cs_build_indirect — write alive-count into DrawIndirect buffer
//   vs_main / fs_main — camera-facing billboard rendering

const PI: f32 = 3.14159265359;
const INV_MAX_U32: f32 = 1.0 / 4294967295.0;

// ── Bindings ────────────────────────────────────────────────────────────────

struct GpuCoronaUniforms {
    delta_time: f32,
    total_particles: u32,
    emitter_count: u32,
    frame_count: u32,
}
@group(0) @binding(0) var<uniform> uniforms: GpuCoronaUniforms;

@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> emitters: array<EmitterDef>;
@group(0) @binding(3) var<storage, read_write> live_counter: atomic<u32>;
@group(0) @binding(4) var<storage, read_write> indirect_buf: DrawIndirect;

struct Particle {
    pos_and_alive: vec4<f32>,
    velocity: vec4<f32>,
    color: vec4<f32>,
    size_lifetime_age: vec4<f32>,
}

struct EmitterDef {
    transform: mat4x4<f32>,
    emit_params: vec4<f32>,
    size_params: vec4<f32>,
    start_color: vec4<f32>,
    end_color: vec4<f32>,
    velocity: vec4<f32>,
    velocity_variation: vec4<f32>,
    extras: vec4<f32>,
    texture_index: i32,
    particle_offset: u32,
    particle_count: u32,
    spawn_cursor: u32,
    _pad: array<f32, 12>,
}

struct DrawIndirect {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
}

// ── Hash-based RNG ───────────────────────────────────────────────────────────

fn hash(x: u32) -> u32 {
    var h = x;
    h = (h ^ (h >> 16)) * 0x85ebca6b;
    h = (h ^ (h >> 13)) * 0xc2b2ae35;
    h = h ^ (h >> 16);
    return h;
}

fn rng_f32(seed: u32) -> f32 {
    return f32(hash(seed)) * INV_MAX_U32;
}

fn rng_range(seed: u32, lo: f32, hi: f32) -> f32 {
    return lo + rng_f32(seed) * (hi - lo);
}

// ── Simulate (update living particles) ──────────────────────────────────────

@compute @workgroup_size(256)
fn cs_simulate(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= uniforms.total_particles { return; }

    var p = particles[idx];
    let alive = p.pos_and_alive.w;
    if alive < 0.5 { return; }

    p.size_lifetime_age.w += uniforms.delta_time;

    // Kill if expired
    if p.size_lifetime_age.w >= p.size_lifetime_age.z {
        p.pos_and_alive.w = 0.0;
        particles[idx] = p;
        return;
    }

    // Physics
    let px = p.pos_and_alive.x + p.velocity.x * uniforms.delta_time;
    let py = p.pos_and_alive.y + p.velocity.y * uniforms.delta_time;
    let pz = p.pos_and_alive.z + p.velocity.z * uniforms.delta_time;
    p.pos_and_alive = vec4<f32>(px, py, pz, p.pos_and_alive.w);
    // Find emitter for gravity / colour interpolation
    var grav = -9.8;
    var start_col = vec4<f32>(1.0);
    var end_col = vec4<f32>(1.0, 1.0, 1.0, 0.0);
    var start_sz = vec2<f32>(0.5);
    var end_sz = vec2<f32>(0.1);

    for (var e = 0u; e < uniforms.emitter_count; e = e + 1u) {
        let em = emitters[e];
        if idx >= em.particle_offset && idx < em.particle_offset + em.particle_count {
            grav = em.emit_params.w;
            start_col = em.start_color;
            end_col = em.end_color;
            start_sz = em.size_params.xy;
            end_sz = em.size_params.zw;
            break;
        }
    }

    p.velocity.y += grav * uniforms.delta_time;

    let t = p.size_lifetime_age.w / max(p.size_lifetime_age.z, 0.001);
    p.color = mix(start_col, end_col, vec4<f32>(t));
    let sz = mix(start_sz, end_sz, vec2<f32>(t));
    p.size_lifetime_age.x = sz.x;
    p.size_lifetime_age.y = sz.y;

    particles[idx] = p;
}

// ── Emit (spawn new particles per emitter) ──────────────────────────────────

@compute @workgroup_size(1)
fn cs_emit(@builtin(workgroup_id) id: vec3<u32>) {
    let eidx = id.x;
    if eidx >= uniforms.emitter_count { return; }

    var em = emitters[eidx];
    if em.extras.w < 0.5 { return; }

    let count = u32(em.emit_params.x * uniforms.delta_time);
    if count == 0u { return; }

    let base = em.particle_offset;
    let range = max(em.particle_count, 1u);
    let origin = em.transform[3].xyz;
    let etype = u32(em.extras.x);
    let radius = em.extras.y;
    let seed = eidx * 997u + uniforms.frame_count * 7919u;

    for (var i = 0u; i < count; i = i + 1u) {
        let cursor = em.spawn_cursor;
        em.spawn_cursor = (cursor + 1u) % range;

        let pidx = base + cursor;
        let s = seed + i * 1013u;

        var spawn_pos: vec3<f32>;
        if etype >= 1u {
            let theta = rng_f32(s + 1u) * 2.0 * PI;
            let phi = rng_f32(s + 3u) * PI;
            let r = rng_f32(s + 5u) * radius;
            spawn_pos = origin + vec3<f32>(
                r * sin(phi) * cos(theta),
                r * cos(phi),
                r * sin(phi) * sin(theta),
            );
        } else {
            spawn_pos = origin;
        }

        let vv = em.velocity_variation.xyz;
        let vel = em.velocity.xyz + vec3<f32>(
            rng_range(s + 7u, -vv.x, vv.x),
            rng_range(s + 11u, -vv.y, vv.y),
            rng_range(s + 13u, -vv.z, vv.z),
        );
        let life = em.emit_params.y + rng_range(s + 17u, -em.emit_params.z, em.emit_params.z);

        var p: Particle;
        p.pos_and_alive = vec4<f32>(spawn_pos, 1.0);
        p.velocity = vec4<f32>(vel, 0.0);
        p.color = em.start_color;
        p.size_lifetime_age = vec4<f32>(em.size_params.x, em.size_params.y, max(life, 0.01), 0.0);
        particles[pidx] = p;
    }

    emitters[eidx].spawn_cursor = em.spawn_cursor;
}

// ── Reset live counter (single thread, run as a separate dispatch before count) ─

@compute @workgroup_size(1)
fn cs_reset_counter() {
    atomicStore(&live_counter, 0u);
}

// ── Count alive (atomic into live_counter) ──────────────────────────────────

@compute @workgroup_size(256)
fn cs_count_alive(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= uniforms.total_particles { return; }
    if particles[idx].pos_and_alive.w > 0.5 {
        atomicAdd(&live_counter, 1u);
    }
}

// ── Build indirect draw args ─────────────────────────────────────────────────
// We always draw total_particles instances; dead particles are culled in
// vs_main by outputting a degenerate position outside the NDC cube.

@compute @workgroup_size(1)
fn cs_build_indirect() {
    indirect_buf.vertex_count = 6u;
    indirect_buf.instance_count = uniforms.total_particles;
    indirect_buf.first_vertex = 0u;
    indirect_buf.first_instance = 0u;
}

// ── Vertex shader (camera-facing billboard) ─────────────────────────────────

struct CameraUniforms {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    position_near: vec4<f32>,
    forward_far: vec4<f32>,
    jitter_frame: vec4<f32>,
    prev_view_proj: mat4x4<f32>,
}

@group(0) @binding(5) var<uniform> camera: CameraUniforms;
@group(0) @binding(6) var particle_tex: texture_2d<f32>;
@group(0) @binding(7) var particle_sampler: sampler;

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
}

fn quad_corner(idx: u32) -> vec2<f32> {
    let c = array<vec2<f32>, 6>(
        vec2<f32>(-0.5, -0.5), vec2<f32>(0.5, -0.5), vec2<f32>(-0.5, 0.5),
        vec2<f32>(-0.5, 0.5), vec2<f32>(0.5, -0.5), vec2<f32>(0.5, 0.5),
    );
    return c[idx];
}

fn quad_uv(idx: u32) -> vec2<f32> {
    let u = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
    );
    return u[idx];
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
    let p = particles[ii];

    var out: VOut;
    // Dead particles get a degenerate position outside NDC — GPU clips the
    // whole triangle without invoking the fragment shader.
    if p.pos_and_alive.w < 0.5 {
        out.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.uv = vec2<f32>(0.0);
        out.color = vec4<f32>(0.0);
        return out;
    }

    let corner = quad_corner(vi);
    let right = vec3<f32>(camera.view[0].x, camera.view[1].x, camera.view[2].x);
    let up    = vec3<f32>(camera.view[0].y, camera.view[1].y, camera.view[2].y);
    let size = max(p.size_lifetime_age.x, 0.001);

    let world_pos = p.pos_and_alive.xyz
        + right * corner.x * size
        + up    * corner.y * size;

    out.pos = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.uv = quad_uv(vi);
    out.color = p.color;
    return out;
}

// ── Fragment shader ─────────────────────────────────────────────────────────

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    let tex = textureSample(particle_tex, particle_sampler, in.uv);
    let alpha = tex.a * in.color.a;
    if alpha < 0.005 { discard; }
    return vec4<f32>(in.color.rgb * tex.rgb, alpha);
}
