/// GPU shadow matrix computation
///
/// Computes shadow light-space matrices entirely on GPU to eliminate CPU overhead.
/// One thread per light; each light can output 1-6 matrices depending on type:
///   - Point lights: 6 cube-face matrices (±X, ±Y, ±Z)
///   - Directional lights: 4 CSM cascade matrices
///   - Spot lights: 1 perspective matrix
///
/// Integrates with GPU indirect dispatch system - runs before shadow pass.

// ── Constants matching shadow_math.rs ─────────────────────────────────────────

const FACES_PER_LIGHT: u32 = 6u;
const CSM_SPLITS: vec4f = vec4f(16.0, 80.0, 300.0, 1400.0);
const SCENE_DEPTH: f32 = 4000.0;

// Light types (must match LightType in libhelio/src/light.rs)
const LIGHT_TYPE_DIRECTIONAL: u32 = 0u;
const LIGHT_TYPE_POINT: u32 = 1u;
const LIGHT_TYPE_SPOT: u32 = 2u;

// ── Input/Output structs ──────────────────────────────────────────────────────

/// Must match GpuLight in libhelio/src/light.rs (64 bytes)
struct GpuLight {
    position_range:   vec4f,  // xyz = position, w = range
    direction_outer:  vec4f,  // xyz = direction, w = cos(outer_angle)
    color_intensity:  vec4f,  // xyz = color, w = intensity
    shadow_index:     u32,    // u32::MAX = no shadow, otherwise shadow matrix base index
    light_type:       u32,    // 0=Directional, 1=Point, 2=Spot
    inner_angle:      f32,    // cos(inner_angle) for spot lights
    _pad:             u32,
    // Light shafts — consumed by helio-pass-volumetric-fog. Padding is three
    // scalars, not vec3<u32>, to keep the struct at 96 bytes (vec3 aligns to 16).
    god_rays_enabled:  u32,
    god_rays_density:  f32,
    god_rays_weight:   f32,
    god_rays_decay:    f32,
    god_rays_exposure: f32,
    flare_enabled:      u32,
    flare_type:         u32,
    flare_intensity:    f32,
    flare_scale:        f32,
    flare_tint_r:       f32,
    flare_tint_g:       f32,
    flare_tint_b:       f32,
    ies_profile_index:    i32,
    light_function_index: i32,
    ies_angle_scale:      f32,
    ies_angle_offset:     f32,
}

/// Must match GpuShadowMatrix in uniforms.rs (64 bytes)
struct GpuShadowMatrix {
    mat: mat4x4f,
}

/// Camera data for CSM cascade computation.
/// Layout must match GpuCameraUniforms in libhelio/src/camera.rs (256 bytes).
struct CameraUniforms {
    view:           mat4x4f,   // offset   0
    proj:           mat4x4f,   // offset  64
    view_proj:      mat4x4f,   // offset 128
    inv_view_proj:  mat4x4f,   // offset 192
    position_near:  vec4f,     // offset 256 — xyz = world pos, w = near plane
    forward_far:    vec4f,     // offset 272
    jitter_frame:   vec4f,     // offset 288
    prev_view_proj: mat4x4f,   // offset 304
}

struct ShadowMatrixParams {
    light_count: u32,
    shadow_atlas_size: u32,
    _pad0: u32,
    _pad1: u32,
}

// ── Bindings ──────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<storage, read>       lights:         array<GpuLight>;
@group(0) @binding(1) var<storage, read_write> shadow_mats:    array<GpuShadowMatrix>;
@group(0) @binding(2) var<storage, read> cameras: array<CameraUniforms, 2>;
@group(0) @binding(3) var<uniform>             params:         ShadowMatrixParams;
@group(0) @binding(4) var<storage, read_write> shadow_dirty:   array<atomic<u32>>;  // Atomic dirty flags per caster slot
@group(0) @binding(5) var<storage, read_write> shadow_hashes:  array<u32>;  // FNV hashes to detect changes

// ── Matrix math helpers ───────────────────────────────────────────────────────

const PI: f32 = 3.14159265359;
const FRAC_PI_2: f32 = 1.57079632679;

/// Build perspective projection matrix (RH, depth [0,1])
fn mat4_perspective_rh(fovy: f32, aspect: f32, near: f32, far: f32) -> mat4x4f {
    let f = 1.0 / tan(fovy * 0.5);
    let nf = 1.0 / (near - far);
    return mat4x4f(
        vec4f(f / aspect, 0.0, 0.0, 0.0),
        vec4f(0.0, f, 0.0, 0.0),
        vec4f(0.0, 0.0, far * nf, -1.0),
        vec4f(0.0, 0.0, near * far * nf, 0.0),
    );
}

/// Build orthographic projection matrix (RH, depth [0,1])
fn mat4_orthographic_rh(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> mat4x4f {
    let rml = 1.0 / (right - left);
    let tmb = 1.0 / (top - bottom);
    let fmn = 1.0 / (far - near);
    return mat4x4f(
        vec4f(2.0 * rml, 0.0, 0.0, 0.0),
        vec4f(0.0, 2.0 * tmb, 0.0, 0.0),
        vec4f(0.0, 0.0, -fmn, 0.0),
        vec4f(-(right + left) * rml, -(top + bottom) * tmb, -near * fmn, 1.0),
    );
}

/// Build look-at view matrix (RH)
fn mat4_look_at_rh(eye: vec3f, center: vec3f, up: vec3f) -> mat4x4f {
    let f = normalize(center - eye);
    let s = normalize(cross(f, up));
    let u = cross(s, f);
    return mat4x4f(
        vec4f(s.x, u.x, -f.x, 0.0),
        vec4f(s.y, u.y, -f.y, 0.0),
        vec4f(s.z, u.z, -f.z, 0.0),
        vec4f(-dot(s, eye), -dot(u, eye), dot(f, eye), 1.0),
    );
}

// ── Point light matrices (6 cube faces) ───────────────────────────────────────

fn compute_point_light_matrices(light_idx: u32, position: vec3f, range: f32) {
    let base = lights[light_idx].shadow_index;
    // Extend far plane to ensure full spherical coverage
    // With 90° FOV, worst case is corners at sqrt(3) * range from light center
    let far_plane = max(range, 0.1) * 2.5;  // 2.5x provides full coverage with margin
    let proj = mat4_perspective_rh(FRAC_PI_2, 1.0, 0.05, far_plane);

    let views = array<mat4x4f, 6>(
        mat4_look_at_rh(position, position + vec3f(1.0, 0.0, 0.0),  vec3f(0.0, -1.0, 0.0)),  // +X
        mat4_look_at_rh(position, position + vec3f(-1.0, 0.0, 0.0), vec3f(0.0, -1.0, 0.0)),  // -X
        mat4_look_at_rh(position, position + vec3f(0.0, 1.0, 0.0),  vec3f(0.0, 0.0, 1.0)),   // +Y
        mat4_look_at_rh(position, position + vec3f(0.0, -1.0, 0.0), vec3f(0.0, 0.0, -1.0)),  // -Y
        mat4_look_at_rh(position, position + vec3f(0.0, 0.0, 1.0),  vec3f(0.0, -1.0, 0.0)),  // +Z
        mat4_look_at_rh(position, position + vec3f(0.0, 0.0, -1.0), vec3f(0.0, -1.0, 0.0)),  // -Z
    );

    for (var i = 0u; i < 6u; i++) {
        shadow_mats[base + i].mat = proj * views[i];
    }
}

// ── Spot light matrix (single perspective) ────────────────────────────────────

fn compute_spot_matrix(light_idx: u32, position: vec3f, direction: vec3f, range: f32, cos_outer: f32) {
    let base = lights[light_idx].shadow_index;
    let dir = normalize(direction);

    // Outer angle from cos(outer) → fov = 2 * acos(cos_outer), clamped to [45°, 179°]
    let outer_angle = acos(cos_outer);
    let fov = clamp(outer_angle * 2.0, PI * 0.25, PI - 0.01);

    let up = select(vec3f(0.0, 0.0, 1.0), vec3f(0.0, 1.0, 0.0), abs(dot(dir, vec3f(0.0, 1.0, 0.0))) < 0.99);
    let view = mat4_look_at_rh(position, position + dir, up);
    let proj = mat4_perspective_rh(fov, 1.0, 0.05, max(range, 0.1));

    shadow_mats[base].mat = proj * view;
}

// ── Directional light cascades (CSM with sphere-fit + texel snap) ─────────────

fn compute_directional_cascades(light_idx: u32, direction: vec3f) {
    let base = lights[light_idx].shadow_index;
    let dir = normalize(direction);
    let up = select(vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(dot(dir, vec3f(0.0, 1.0, 0.0))) > 0.99);

    // Build unjittered frustum rays from rotation and projection scale. Avoid
    // far-plane inverse projection: it amplifies f32 error at large far/near
    // ratios and makes stationary shadows depend on temporal AA jitter.
    let camera=cameras[0];
    let rotation=transpose(mat3x3f(camera.view[0].xyz,camera.view[1].xyz,camera.view[2].xyz));
    var rays:array<vec3f,4>;
    var ortho_offsets:array<vec3f,4>;
    for(var j=0u;j<4u;j++) {
        let xy=vec2f(select(-1.0,1.0,(j&1u)!=0u),select(-1.0,1.0,(j&2u)!=0u));
        let asymmetric=xy+camera.proj[2].xy+camera.jitter_frame.xy;
        rays[j]=rotation*normalize(vec3f(asymmetric.x/camera.proj[0][0],asymmetric.y/camera.proj[1][1],-1.0));
        let ortho_xy=xy-camera.proj[3].xy+camera.jitter_frame.xy;
        ortho_offsets[j]=rotation*vec3f(ortho_xy.x/camera.proj[0][0],ortho_xy.y/camera.proj[1][1],0.0);
    }
    let prev_d = array<f32,4>(0.0,CSM_SPLITS.x,CSM_SPLITS.y,CSM_SPLITS.z);
    for(var cascade_idx=0u;cascade_idx<4u;cascade_idx++) {
        // Adjacent slices cover the lighting pass's five-percent blend margins.
        let d0=max(camera.position_near.w,prev_d[cascade_idx]*0.95);
        let d1=CSM_SPLITS[cascade_idx]*1.05;
        var cc:array<vec3f,8>;
        for(var j=0u;j<4u;j++) {
            cc[j*2u]=rays[j]*d0;
            cc[j*2u+1u]=rays[j]*d1;
            if camera.proj[3].w!=0.0 {
                cc[j*2u]=ortho_offsets[j]-rotation[2]*d0;
                cc[j*2u+1u]=ortho_offsets[j]-rotation[2]*d1;
            }
        }
        // Sphere fit: centroid + radius
        var centroid = vec3f(0.0);
        for (var i = 0u; i < 8u; i++) {
            centroid += cc[i];
        }
        centroid /= 8.0;

        var radius = 0.0;
        for (var i = 0u; i < 8u; i++) {
            radius = max(radius, length(cc[i] - centroid));
        }

        // Quantize the rotation-invariant sphere radius, then reserve one texel
        // of padding for snapping. The old radius / texel_size was always N/2
        // and did not stabilize the radius.
        let resolution=f32(max(params.shadow_atlas_size,4u));
        let radius_snap=ceil(radius*16.0)/16.0*resolution/(resolution-2.0);
        let texel_size=2.0*radius_snap/resolution;
        let right_ws=normalize(cross(dir,up));
        let up_ws=cross(right_ws,dir);
        let world_centroid=camera.position_near.xyz+centroid;
        // Snap in a fixed light basis. Transforming the centroid by a view
        // centred on itself yields zero and cannot anchor texels to the world.
        let cx=round(dot(right_ws,world_centroid)/texel_size)*texel_size;
        let cy=round(dot(up_ws,world_centroid)/texel_size)*texel_size;
        let cz=dot(dir,world_centroid);
        let light_view=mat4x4f(
            vec4f(right_ws.x,up_ws.x,-dir.x,0.0),
            vec4f(right_ws.y,up_ws.y,-dir.y,0.0),
            vec4f(right_ws.z,up_ws.z,-dir.z,0.0),
            vec4f(-cx,-cy,cz-SCENE_DEPTH,1.0),
        );
        let proj = mat4_orthographic_rh(-radius_snap, radius_snap, -radius_snap, radius_snap, 0.1, SCENE_DEPTH * 2.0);

        shadow_mats[base + cascade_idx].mat = proj * light_view;
    }

    // Fill slots 4-5 with identity (point light faces 4-5 unused for directional)
    for (var i = 4u; i < 6u; i++) {
        shadow_mats[base + i].mat = mat4x4f(
            vec4f(1.0, 0.0, 0.0, 0.0),
            vec4f(0.0, 1.0, 0.0, 0.0),
            vec4f(0.0, 0.0, 1.0, 0.0),
            vec4f(0.0, 0.0, 0.0, 1.0),
        );
    }
}

// ── FNV-1a hash for matrix change detection ───────────────────────────────────

fn fnv_hash_mat(m: mat4x4f) -> u32 {
    var hash: u32 = 2166136261u;
    for (var col = 0u; col < 4u; col++) {
        for (var row = 0u; row < 4u; row++) {
            let bits = bitcast<u32>(m[col][row]);
            hash ^= (bits & 0xFFu);
            hash = hash * 16777619u;
            hash ^= ((bits >> 8u) & 0xFFu);
            hash = hash * 16777619u;
            hash ^= ((bits >> 16u) & 0xFFu);
            hash = hash * 16777619u;
            hash ^= ((bits >> 24u) & 0xFFu);
            hash = hash * 16777619u;
        }
    }
    return hash;
}

fn fnv_hash_mats_6(base_idx: u32) -> u32 {
    var hash: u32 = 2166136261u;
    for (var i = 0u; i < 6u; i++) {
        let mat_hash = fnv_hash_mat(shadow_mats[base_idx + i].mat);
        hash ^= (mat_hash & 0xFFu);
        hash = hash * 16777619u;
        hash ^= ((mat_hash >> 8u) & 0xFFu);
        hash = hash * 16777619u;
        hash ^= ((mat_hash >> 16u) & 0xFFu);
        hash = hash * 16777619u;
        hash ^= ((mat_hash >> 24u) & 0xFFu);
        hash = hash * 16777619u;
    }
    return hash;
}

// ── Main compute entry point ──────────────────────────────────────────────────

@compute @workgroup_size(64)
fn compute_shadow_matrices(@builtin(global_invocation_id) gid: vec3u) {
    let light_idx = gid.x;
    if light_idx >= params.light_count { return; }

    let light = lights[light_idx];

    // Skip shadow computation if light doesn't cast shadows
    if light.shadow_index == 0xFFFFFFFFu { return; }

    // Compute matrices based on light type
    if light.light_type == LIGHT_TYPE_POINT {
        compute_point_light_matrices(light_idx, light.position_range.xyz, light.position_range.w);
    } else if light.light_type == LIGHT_TYPE_DIRECTIONAL {
        compute_directional_cascades(light_idx, light.direction_outer.xyz);
    } else if light.light_type == LIGHT_TYPE_SPOT {
        compute_spot_matrix(light_idx, light.position_range.xyz, light.direction_outer.xyz, light.position_range.w, light.direction_outer.w);
    }

    // Hash the computed matrices to detect changes
    // This enables shadow atlas caching for static geometry
    let base_idx = light.shadow_index;
    let caster_slot = base_idx / FACES_PER_LIGHT;
    let new_hash = fnv_hash_mats_6(base_idx);
    let old_hash = shadow_hashes[caster_slot];

    // Update hash and mark dirty if changed
    if new_hash != old_hash {
        shadow_hashes[caster_slot] = new_hash;
        // Atomically set dirty flag for this light
        atomicStore(&shadow_dirty[caster_slot], 1u);
    }
}
