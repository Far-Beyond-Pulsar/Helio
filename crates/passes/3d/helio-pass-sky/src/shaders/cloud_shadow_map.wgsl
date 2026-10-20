//!use atmosphere
//!use atmosphere_clouds
//!use cloud_shadow
// The clouds' shadow map (`cloud_shadow.wgsl`), built each frame after the
// atmosphere resolves: `cloud_shadow_resolve` places the map around the
// camera (its texels on a lattice fixed to the planet, so shadows hold
// still while the camera moves), `cloud_shadow_kernel` marches each
// texel's line through the layer towards the sun.

@group(0) @binding(0) var<storage, read> frame: AtmosphereFrame;
@group(0) @binding(1) var cloud_weather: texture_3d<f32>;
@group(0) @binding(2) var cloud_shape: texture_3d<f32>;
@group(0) @binding(3) var cloud_detail: texture_3d<f32>;
@group(0) @binding(4) var cloud_noise_sampler: sampler;
@group(0) @binding(5) var shadow_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var<storage, read_write> cloud_shadow_frame: CloudShadowFrame;
// Declared for the shared lookup, which this module never calls.
@group(0) @binding(7) var cloud_shadow_map: texture_2d<f32>;

// Half the map's extent (km), at most the layer's radius.
const CLOUD_SHADOW_HALF_KM: f32 = 16.0;
// The longest path through the layer a texel marches (km): a low sun's
// grazing lines are cut to it.
const CLOUD_SHADOW_MAX_PATH_KM: f32 = 40.0;
const CLOUD_SHADOW_STEPS: u32 = 40u;
// Stored where a line meets no cloud: its clouds begin past any point.
const CLOUD_SHADOW_NONE: f32 = 60000.0;

@compute @workgroup_size(1)
fn cloud_shadow_resolve() {
    if !atmosphere_has_clouds(frame) {
        cloud_shadow_frame.right = vec4<f32>(0.0);
        return;
    }
    let c = cloud_layer_params(frame);
    let forward = -frame.sun.xyz;
    let reference = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(forward.z) > 0.9);
    let right = normalize(cross(reference, forward));
    let up = cross(forward, right);
    let side = f32(textureDimensions(shadow_out).x);
    let half = min(CLOUD_SHADOW_HALF_KM, c.outer);
    let texel = 2.0 * half / side;
    // The camera's place on the lattice: the map's centre is the texel
    // corner nearest it.
    let eye = frame.eye.xyz;
    let px = dot(eye, right);
    let py = dot(eye, up);
    let centre = right * (round(px / texel) * texel - px) + up * (round(py / texel) * texel - py);
    cloud_shadow_frame.right = vec4<f32>(right, 1.0);
    cloud_shadow_frame.up = vec4<f32>(up, half);
    cloud_shadow_frame.forward = vec4<f32>(forward, side);
    cloud_shadow_frame.centre = vec4<f32>(centre, 0.0);
}

// Whether the line through `o` along `d` meets the sphere of radius `r`.
fn line_meets(o: vec3<f32>, d: vec3<f32>, r: f32) -> bool {
    let b = dot(o, d);
    return dot(o, o) - b * b < r * r;
}

@compute @workgroup_size(8, 8)
fn cloud_shadow_kernel(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(shadow_out);
    if any(id.xy >= size) || cloud_shadow_frame.right.w < 0.5 { return; }
    let c = cloud_layer_params(frame);
    let half = cloud_shadow_frame.up.w;
    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(size);
    let q = cloud_shadow_frame.centre.xyz
        + cloud_shadow_frame.right.xyz * ((uv.x * 2.0 - 1.0) * half)
        + cloud_shadow_frame.up.xyz * ((1.0 - uv.y * 2.0) * half);
    let o = frame.eye.xyz + q;
    let sun = frame.sun.xyz;
    let none = vec4<f32>(CLOUD_SHADOW_NONE, 0.0, 0.0, 0.0);
    if !line_meets(o, sun, c.outer) {
        textureStore(shadow_out, id.xy, none);
        return;
    }
    // Distances along the line towards the sun from `q`: the layer's span
    // between the sun and the near side of the planet.
    let outer = ray_sphere(o, sun, c.outer);
    var lo = outer.x;
    let hi = outer.y;
    if line_meets(o, sun, c.inner) { lo = max(lo, ray_sphere(o, sun, c.inner).y); }
    lo = max(lo, hi - CLOUD_SHADOW_MAX_PATH_KM);
    if hi <= lo {
        textureStore(shadow_out, id.xy, none);
        return;
    }
    let dt = (hi - lo) / f32(CLOUD_SHADOW_STEPS);
    let lod = max(log2(2.0 * half / f32(size.x) * c.inv_shape * CLOUD_SHAPE_TEXELS), 0.0);
    var od = 0.0;
    var found = false;
    var first = 0.0;
    var last = 0.0;
    for (var i = 0u; i < CLOUD_SHADOW_STEPS; i++) {
        let s = hi - (f32(i) + 0.5) * dt;
        // The wisps too: without their erosion the shadows outgrow the clouds.
        let sigma = cloud_density(c, o + sun * s, lod, true).x * c.extinction;
        if sigma > 0.0 {
            if !found { first = s + 0.5 * dt; found = true; }
            last = s - 0.5 * dt;
            od += sigma * dt;
        }
    }
    if !found {
        textureStore(shadow_out, id.xy, none);
        return;
    }
    // Along the light, from the map's plane: z = -s.
    let front = -first;
    let back = -last;
    textureStore(shadow_out, id.xy, vec4<f32>(front, od / max(back - front, dt), od, 1.0));
}
