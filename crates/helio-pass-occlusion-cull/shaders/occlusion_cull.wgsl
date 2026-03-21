// Hi-Z occlusion culling compute shader.
//
// For every active instance slot, projects its world-space AABB to screen
// space, samples the Hi-Z pyramid at the appropriate mip level, and marks
// the slot visible or occluded in a bitmask.
//
// The bitmask is written as u32 words: bit (slot % 32) of word (slot / 32).
// Bit set = potentially visible; bit clear = occluded.
//
// Temporal reprojection note
// ──────────────────────────
// This pass READS the Hi-Z built from the PREVIOUS frame's depth buffer and
// WRITES the visibility for the CURRENT frame.  Newly visible objects receive
// a free pass on their first frame (all bits default to 1 after a clear) and
// will be culled correctly starting from the second frame they appear.  This
// eliminates the need for GPU readback and keeps the pipeline fully async.
//
// Depth convention: 0 = near, 1 = far (standard wgpu / Vulkan).
// Hi-Z stores MAX depth per region (farthest visible surface).
// Cull condition: object's minimum projected depth > Hi-Z sample.

struct OcclusionCullInput {
    view_proj:      mat4x4f,
    view_proj_inv:  mat4x4f,
    camera_pos:     vec3f,
    _pad0:          u32,
    // Screen dimensions (pixels)
    screen_width:   u32,
    screen_height:  u32,
    total_slots:    u32,  // high-water-mark of the slot allocator
    hiz_mip_count:  u32,
}

struct GpuInstanceAabb {
    aabb_min: vec3f,
    _pad0:    f32,
    aabb_max: vec3f,
    _pad1:    f32,
}

@group(0) @binding(0) var<uniform>            cull_input:   OcclusionCullInput;
@group(0) @binding(1) var<storage, read>       aabb_buf:     array<GpuInstanceAabb>;
@group(0) @binding(2) var                      hiz_tex:      texture_2d<f32>;
@group(0) @binding(3) var                      hiz_sampler:  sampler;
@group(0) @binding(4) var<storage, read_write> visibility:   array<atomic<u32>>;

// Project a world-space point to NDC.  Returns (ndc_x, ndc_y, ndc_z, valid).
// valid = 0 if w <= 0 (behind camera or at infinity).
fn project(world_pos: vec3f) -> vec4f {
    let clip = cull_input.view_proj * vec4f(world_pos, 1.0);
    if clip.w <= 0.0 { return vec4f(0.0, 0.0, 0.0, 0.0); }
    let ndc = clip.xyz / clip.w;
    return vec4f(ndc, 1.0);  // w=1 means valid
}

@compute @workgroup_size(64)
fn occlusion_cull(
    @builtin(global_invocation_id) gid: vec3u,
) {
    let slot = gid.x;
    if slot >= cull_input.total_slots { return; }

    let inst = aabb_buf[slot];

    // Zero AABB = empty slot.  Treat as not visible (no draw emitted).
    if all(inst.aabb_min == vec3f(0.0)) && all(inst.aabb_max == vec3f(0.0)) {
        // Clear the visibility bit for this slot.
        let word = slot / 32u;
        let bit  = slot % 32u;
        atomicAnd(&visibility[word], ~(1u << bit));
        return;
    }

    let lo = inst.aabb_min;
    let hi = inst.aabb_max;

    // Project all 8 AABB corners to NDC.
    let corners = array<vec3f, 8>(
        vec3f(lo.x, lo.y, lo.z),
        vec3f(hi.x, lo.y, lo.z),
        vec3f(lo.x, hi.y, lo.z),
        vec3f(hi.x, hi.y, lo.z),
        vec3f(lo.x, lo.y, hi.z),
        vec3f(hi.x, lo.y, hi.z),
        vec3f(lo.x, hi.y, hi.z),
        vec3f(hi.x, hi.y, hi.z),
    );

    var ndc_min = vec3f( 1e9,  1e9,  1e9);
    var ndc_max = vec3f(-1e9, -1e9, -1e9);
    var any_valid = false;

    for (var i = 0u; i < 8u; i++) {
        let p = project(corners[i]);
        if p.w < 0.5 { continue; }  // behind camera
        any_valid = true;
        ndc_min = min(ndc_min, p.xyz);
        ndc_max = max(ndc_max, p.xyz);
    }

    if !any_valid {
        // All corners behind camera — assume visible (conservative).
        let word = slot / 32u;
        let bit  = slot % 32u;
        atomicOr(&visibility[word], 1u << bit);
        return;
    }

    // Clamp to NDC [-1,1] × [-1,1].
    ndc_min = clamp(ndc_min, vec3f(-1.0), vec3f(1.0));
    ndc_max = clamp(ndc_max, vec3f(-1.0), vec3f(1.0));

    // Minimum depth of the AABB (closest corner — the hardest to occlude).
    let obj_min_depth = max(ndc_min.z, 0.0);

    // Convert NDC xy to UV [0,1] for texture sampling.
    // NDC (-1,-1) = top-left in wgpu/Vulkan (y-down).
    let uv_min = vec2f( ndc_min.x * 0.5 + 0.5,  0.5 - ndc_max.y * 0.5);
    let uv_max = vec2f( ndc_max.x * 0.5 + 0.5,  0.5 - ndc_min.y * 0.5);
    let uv_center = (uv_min + uv_max) * 0.5;

    // Screen-space footprint size → choose Hi-Z mip level.
    let sw = f32(cull_input.screen_width);
    let sh = f32(cull_input.screen_height);
    let footprint_px = max((uv_max.x - uv_min.x) * sw,
                           (uv_max.y - uv_min.y) * sh);

    // log2(footprint) → mip where one texel covers the footprint.
    // Clamp to [0, hiz_mip_count-1].
    let mip_f = log2(max(footprint_px, 1.0));
    let mip   = clamp(u32(mip_f), 0u, cull_input.hiz_mip_count - 1u);

    // Sample Hi-Z (MAX depth in the region at this mip level).
    let hiz_val = textureSampleLevel(hiz_tex, hiz_sampler, uv_center, f32(mip)).r;

    // Occlusion test: if the MINIMUM depth of the object (its closest point)
    // is greater than the MAX depth stored in the Hi-Z region, the entire
    // object is behind all visible geometry → occluded.
    let occluded = (obj_min_depth > hiz_val);

    let word = slot / 32u;
    let bit  = slot % 32u;
    if occluded {
        atomicAnd(&visibility[word], ~(1u << bit));
    } else {
        atomicOr(&visibility[word], 1u << bit);
    }
}
