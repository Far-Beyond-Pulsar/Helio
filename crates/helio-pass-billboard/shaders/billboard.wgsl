//! Billboard shader - camera-facing instanced quads
//!
//! Vertex inputs:
//!   Slot 0 (per-vertex):   position (vec2), uv (vec2)
//!   Slot 1 (per-instance): world_pos_pad (vec4), scale_flags (vec4), color (vec4)

// Group 0: global uniforms (camera + globals)
// Layout must match GpuCameraUniforms in libhelio/src/camera.rs (368 bytes).
struct Camera {
    view:          mat4x4<f32>,   // offset   0 — world→view
    proj:          mat4x4<f32>,   // offset  64 — view→clip
    view_proj:     mat4x4<f32>,   // offset 128 — combined VP
    inv_view_proj: mat4x4<f32>,   // offset 192 — clip→world
    position_near: vec4<f32>,     // offset 256 — xyz=world pos, w=near
    forward_far:   vec4<f32>,     // offset 272 — xyz=forward, w=far
    jitter_frame:  vec4<f32>,     // offset 288
    prev_view_proj: mat4x4<f32>,  // offset 304
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
    let cam_pos = camera.position_near.xyz;
    let to_cam  = normalize(cam_pos - world_pos);

    // Right and up vectors perpendicular to the view direction
    let world_up = vec3<f32>(0.0, 1.0, 0.0);
    let right    = normalize(cross(world_up, to_cam));
    let up       = cross(to_cam, right);

    // Offset in world space using the quad's local position
    var offset = right * quad.position.x * scale.x
               + up    * quad.position.y * scale.y;

    // Optional: constant screen-space scaling.
    // We must scale by VIEW-AXIS depth (dot with forward), NOT Euclidean distance.
    // Perspective division uses view-depth, so: world_size = scale * view_depth
    // → projected NDC size = scale * view_depth / view_depth = scale (constant).
    // Using Euclidean distance instead causes off-axis billboards to appear larger.
    if screen_scale {
        let view_depth = max(dot(camera.forward_far.xyz, world_pos - cam_pos), 0.001);
        offset        *= view_depth;
    }

    let final_pos = world_pos + offset;

    var out: VertexOut;
    out.clip_pos = camera.view_proj * vec4<f32>(final_pos, 1.0); // view_proj at offset 128
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
