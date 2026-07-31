// hitbox.frag.wgsl — AABB-based water displacement (replaces sphere.frag).
//
// For each hitbox we compute how much water the AABB *was* displacing (old bounds)
// and how much it *now* displaces (new bounds).  The difference drives a height
// change:  rise where the box vacated, fall where it now sits.
//
// Texture layout (Rgba16Float):
//   R = height  (read-write)
//   G = velocity (read-only, pass through)
//   B = normal.x (read-only, pass through)
//   A = normal.z (read-only, pass through)

@group(0) @binding(0) var water_texture: texture_2d<f32>;
@group(0) @binding(1) var water_sampler: sampler;

/// One AABB hitbox (80 bytes = 5 × vec4<f32>)
struct GpuWaterHitbox {
    old_min:  vec4<f32>,   // xyz = old AABB min
    old_max:  vec4<f32>,   // xyz = old AABB max
    new_min:  vec4<f32>,   // xyz = new AABB min
    new_max:  vec4<f32>,   // xyz = new AABB max
    params:   vec4<f32>,   // x = edge_softness, y = strength
}

struct HitboxUniforms {
    /// Number of active hitboxes
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}
@group(0) @binding(2) var<uniform> u: HitboxUniforms;

@group(0) @binding(3) var<storage, read> hitboxes: array<GpuWaterHitbox>;

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Smooth 3-D Gaussian falloff inside an AABB.
///
/// Returns a value in [0, 1]:
///   1.0 at the box centre, smoothly tapering to 0 at and beyond the edges.
/// `softness` scales how quickly the falloff extends outside the box interior.
fn volume_in_box(box_min: vec3<f32>, box_max: vec3<f32>, uv: vec2<f32>, softness: f32) -> f32 {
    let box_center  = (box_min + box_max) * 0.5;
    let box_half    = (box_max - box_min) * 0.5;

    // Map the water-surface point (in [-1,1] water-sim space) to world XZ
    // The sim texture covers [-1,1]² in X and Z.
    let world_xz = uv * 2.0 - 1.0;          // remap [0,1] UV → [-1,1] world
    let world_y  = 0.0;                       // we probe at the water-surface level

    let p = vec3<f32>(world_xz.x, world_y, world_xz.y);

    // Per-axis distance from box surface (negative inside, positive outside)
    let d = abs(p - box_center) - box_half;

    // Smooth falloff: exp(-clamp(d/softness, 0, 4)^2) per axis, multiplied together
    let soft = max(d, vec3<f32>(0.0)) / max(softness, 0.001);
    let weight = exp(-dot(soft * soft, vec3<f32>(1.0)));

    // Only count displacement where the box is (partially) submerged.
    // The box must reach down from box_max.y to at least the surface (y = 0).
    let submerged_depth = clamp(box_max.y - box_min.y, 0.0, box_max.y - world_y);

    return weight * submerged_depth * 0.1;
}

// ── Entry point ──────────────────────────────────────────────────────────────

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    var info = textureSample(water_texture, water_sampler, uv);

    for (var i: u32 = 0u; i < u.count; i = i + 1u) {
        let hb = hitboxes[i];
        let softness = hb.params.x;
        let strength = hb.params.y;

        // Water rises where the box *was* (old position)
        info.r += volume_in_box(hb.old_min.xyz, hb.old_max.xyz, uv, softness) * strength;
        // Water falls where the box *is* (new position)
        info.r -= volume_in_box(hb.new_min.xyz, hb.new_max.xyz, uv, softness) * strength;
    }

    return info;
}
