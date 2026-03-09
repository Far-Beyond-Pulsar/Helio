// billboard.wgsl — GPU instance expansion: each instance expands to a camera-facing quad.
// One vertex-buffer entry per billboard. Vertex index 0..6 per instance (2 triangles).

struct Camera {
    view_proj:     mat4x4<f32>,
    position:      vec3<f32>,
    time:          f32,
    view_proj_inv: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform>  camera:    Camera;
@group(0) @binding(1) var           atlas:     texture_2d<f32>;
@group(0) @binding(2) var           atlas_smp: sampler;

struct BillboardInstance {
    @location(0) world_pos:  vec3<f32>,
    @location(1) size:       f32,
    @location(2) uv_offset:  vec2<f32>,
    @location(3) uv_scale:   vec2<f32>,
    @location(4) color:      vec4<f32>,
};

struct VertOut {
    @builtin(position) clip:  vec4<f32>,
    @location(0)       uv:    vec2<f32>,
    @location(1)       color: vec4<f32>,
};

// Unit quad corners in [-0.5, 0.5] space (tristrip: 0,1,2, 0,2,3).
fn quad_corner(vi: u32) -> vec2<f32> {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-0.5, -0.5),
        vec2<f32>( 0.5, -0.5),
        vec2<f32>(-0.5,  0.5),
        vec2<f32>(-0.5,  0.5),
        vec2<f32>( 0.5, -0.5),
        vec2<f32>( 0.5,  0.5),
    );
    return corners[vi % 6u];
}

fn quad_uv(vi: u32) -> vec2<f32> {
    var uvs = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(1.0, 0.0),
    );
    return uvs[vi % 6u];
}

@vertex
fn vs_main(
    @builtin(vertex_index)   vi:   u32,
    inst:                    BillboardInstance,
) -> VertOut {
    let corner = quad_corner(vi);

    // Camera-facing axes.
    let right = normalize(vec3<f32>(camera.view_proj[0][0], camera.view_proj[1][0], camera.view_proj[2][0]));
    let up    = normalize(vec3<f32>(camera.view_proj[0][1], camera.view_proj[1][1], camera.view_proj[2][1]));

    let world_pos = inst.world_pos
        + right * corner.x * inst.size
        + up    * corner.y * inst.size;

    var out: VertOut;
    out.clip  = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.uv    = quad_uv(vi) * inst.uv_scale + inst.uv_offset;
    out.color = inst.color;
    return out;
}

@fragment
fn fs_main(in: VertOut) -> @location(0) vec4<f32> {
    let tex = textureSample(atlas, atlas_smp, in.uv) * in.color;
    if tex.a < 0.01 { discard; }
    return tex;
}
