//! Billboard shader - camera-facing instanced quads
//!
//! Vertex inputs:
//!   Slot 0 (per-vertex):   position (vec2), uv (vec2)
//!   Slot 1 (per-instance): world_pos_pad (vec4), scale_flags (vec4), color (vec4)

// Group 0: global uniforms (camera + globals)
struct Camera {
    view_proj: mat4x4<f32>,
    position: vec3<f32>,
    time: f32,
}
struct Globals {
    frame: u32,
    delta_time: f32,
    ambient_intensity: f32,
    _padding: f32,
}
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> globals: Globals;

// Group 1: sprite texture
@group(1) @binding(0) var sprite_tex:     texture_2d<f32>;
@group(1) @binding(1) var sprite_sampler: sampler;

// ── Vertex inputs ──────────────────────────────────────────────────────────

struct QuadVertex {
    @location(0) position: vec2<f32>,
    @location(1) uv:       vec2<f32>,
}

struct BillboardInstance {
    // world position (xyz) + unused pad (w)
    @location(2) world_pos_pad: vec4<f32>,
    // scale (xy), screen_scale flag as f32 (z), unused (w)
    @location(3) scale_flags:   vec4<f32>,
    // RGBA tint color
    @location(4) color:         vec4<f32>,
}

// ── Vertex output ───────────────────────────────────────────────────────────

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0)       uv:       vec2<f32>,
    @location(1)       color:    vec4<f32>,
}

// ── Vertex shader ───────────────────────────────────────────────────────────

@vertex
fn vs_main(quad: QuadVertex, inst: BillboardInstance) -> VertexOut {
    let world_pos    = inst.world_pos_pad.xyz;
    let scale        = inst.scale_flags.xy;
    let screen_scale = inst.scale_flags.z > 0.5;

    // Build camera-facing (billboard) basis vectors
    let cam_pos = camera.position;
    let to_cam  = normalize(cam_pos - world_pos);

    // Right and up vectors perpendicular to the view direction
    let world_up = vec3<f32>(0.0, 1.0, 0.0);
    let right    = normalize(cross(world_up, to_cam));
    let up       = cross(to_cam, right);

    // Offset in world space using the quad's local position
    var offset = right * quad.position.x * scale.x
               + up    * quad.position.y * scale.y;

    // Optional: constant screen-space scaling
    if screen_scale {
        let dist   = length(cam_pos - world_pos);
        offset    *= dist;
    }

    let final_pos = world_pos + offset;

    var out: VertexOut;
    out.clip_pos = camera.view_proj * vec4<f32>(final_pos, 1.0);
    out.uv       = quad.uv;
    out.color    = inst.color;
    return out;
}

// ── Fragment shader ─────────────────────────────────────────────────────────

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let tex_color = textureSample(sprite_tex, sprite_sampler, in.uv);
    // Tint the sprite by the per-instance color; use texture alpha for transparency
    let rgb   = tex_color.rgb * in.color.rgb;
    let alpha = tex_color.a   * in.color.a;
    if alpha < 0.01 { discard; }
    return vec4<f32>(rgb, alpha);
}
