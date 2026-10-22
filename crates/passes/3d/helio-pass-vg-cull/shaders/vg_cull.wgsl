//!use helio_prelude
// Virtual geometry cull: instance BVH -> per-asset cluster-group hierarchy ->
// cluster cut, with two-phase Hi-Z occlusion and tier binning.
//
// Composed in Rust (see `shader_source()`): helio prelude, generated `VGC_*`
// constants, vg_format.wgsl, helio-vg-stream bindings + helpers (binding 4..11),
// helio-pass-vg-bvh `NODES_WGSL`, then this
// file. Entry points: cs_seed, cs_round_prepare, cs_traverse, cs_finalize.
//
// Job = vec2<u32>(x, y), tag = y >> 30:
//   0  GCBVH node          x = 0xffffffff, y = node index
//   1  asset hierarchy item x = instance,   y = TAG | (parent_node << 2 | slot)
//   2  cluster item         x = instance,   y = TAG | cluster_slot (pool_slot << 11 | local)
//   3  instance             x = instance,   y = TAG
// Every job tests only its own bound, so deferring/dropping a job is exact.

struct CullParams {
    screen_width: u32,
    screen_height: u32,
    hiz_mip_count: u32,
    mode: u32,
    tau_px: f32,
    sw_edge_px: f32,
    instance_count: u32,
    bin_capacity: u32,
    queue_capacity: u32,
    history_bits_mask: u32,
    hist_read_base: u32,
    hist_write_base: u32,
    flags: u32,
    dense_tiles_x: u32,
    dense_tiles_y: u32,
    max_groups: u32,
}

struct RoundParams {
    src: u32,
    dst: u32,
    phase: u32,
    hiz_valid: u32,
}

// GpuStaticObject row as published by the scene join (see
// helio-pass-object-batch `StaticObjectRow`); only `transform` is read.
struct VgObjectRow {
    mesh_slot: u32,
    mesh_generation: u32,
    material_slot: u32,
    material_generation: u32,
    transform: array<array<f32, 4>, 4>,
    prev_transform: array<array<f32, 4>, 4>,
    normal_mat: array<array<f32, 4>, 3>,
    bounds: array<f32, 4>,
    index_count: u32,
    first_index: u32,
    vertex_offset: i32,
    material_class: u32,
    graph_hash_lo: u32,
    graph_hash_hi: u32,
    flags: u32,
    skin_vertex_offset: u32,
    bone_palette_offset: u32,
    bone_palette_count: u32,
    _skin_pad: u32,
}

@group(0) @binding(0) var<storage, read> cameras: array<Camera, 2>;
@group(0) @binding(1) var<uniform> params: CullParams;
@group(0) @binding(2) var<storage, read> vg_instances: array<VgInstance>;
@group(0) @binding(3) var<storage, read> objects: array<VgObjectRow>;
// Bindings 4..=11 (page pool, residency, nodes, group refs, asset headers,
// bases, quant, feedback) are declared by `helio_vg_stream::bindings_wgsl(0, 4, true)`.
@group(0) @binding(12) var<storage, read> vg_gcbvh_nodes: array<VgBvhNode>;
@group(0) @binding(13) var<storage, read> vg_gcbvh_header: VgBvhHeader;
@group(0) @binding(14) var<storage, read> vg_gcbvh_items: array<u32>;
@group(0) @binding(15) var<storage, read> dense_mask: array<u32>;
@group(0) @binding(16) var hiz_tex: texture_2d<f32>;
@group(0) @binding(17) var hiz_samp: sampler;
@group(0) @binding(18) var<storage, read_write> state: array<atomic<u32>>;
@group(0) @binding(19) var<storage, read_write> queues: array<vec2<u32>>;
@group(0) @binding(20) var<storage, read_write> visible: array<vec2<u32>>;
@group(0) @binding(21) var<storage, read_write> history: array<atomic<u32>>;

@group(1) @binding(0) var<uniform> rp: RoundParams;
@group(2) @binding(0) var<storage, read_write> dispatch_args: array<u32>;

const WG: u32 = 64u;
const STACK_CAP: u32 = 1024u;
const INF_ERR: f32 = 1.0e30;
const HUGE_PX: f32 = 3.0e38;
const ROOT_ONLY_TAU: f32 = 1.0e30;

var<workgroup> wg_stack: array<vec2<u32>, 1024>;
var<workgroup> wg_sp: atomic<u32>;
var<workgroup> wg_mode: u32;
var<workgroup> wg_n: u32;
var<workgroup> wg_base: u32;
var<workgroup> wg_planes: array<vec4<f32>, 5>;

// ── Small helpers ──────────────────────────────────────────────────────────

fn stat_add(i: u32) {
    if (params.flags & VGC_FLAG_STATS) != 0u {
        atomicAdd(&state[VGC_ST_STATS + i], 1u);
    }
}

fn vg_hash(a: u32, b: u32) -> u32 {
    var h = (a * 0x9e3779b1u) ^ ((b + 0x7f4a7c15u) * 0x85ebca6bu);
    h ^= h >> 15u;
    h *= 0x2c1b3c6du;
    h ^= h >> 12u;
    h *= 0x297a2d39u;
    h ^= h >> 15u;
    return h;
}

fn job_key(job: vec2<u32>) -> u32 { return vg_hash(job.x, job.y); }

fn history_test(key: u32) -> bool {
    let bit = key & params.history_bits_mask;
    let w = atomicLoad(&history[params.hist_read_base + (bit >> 5u)]);
    return ((w >> (bit & 31u)) & 1u) != 0u;
}

fn history_mark(key: u32) {
    if (params.flags & VGC_FLAG_TWO_PHASE) == 0u { return; }
    let bit = key & params.history_bits_mask;
    atomicOr(&history[params.hist_write_base + (bit >> 5u)], 1u << (bit & 31u));
}

fn vp_row(r: u32) -> vec4<f32> {
    let m = cameras[0].view_proj;
    return vec4<f32>(m[0][r], m[1][r], m[2][r], m[3][r]);
}

// Normalised frustum planes, inside when dot(n, p) + d >= 0. No far plane
// (infinite reversed-Z projection); mirrors helio_core::depth::frustum_planes.
fn frustum_plane(i: u32) -> vec4<f32> {
    let r3 = vp_row(3u);
    var p: vec4<f32>;
    switch i {
        case 0u: { p = r3 + vp_row(0u); }
        case 1u: { p = r3 - vp_row(0u); }
        case 2u: { p = r3 + vp_row(1u); }
        case 3u: { p = r3 - vp_row(1u); }
        default: { p = r3 - vp_row(2u); }
    }
    return p / length(p.xyz);
}

fn sphere_in_frustum(c: vec3<f32>, r: f32) -> bool {
    for (var i = 0u; i < 5u; i++) {
        let p = wg_planes[i];
        if dot(p.xyz, c) + p.w < -r { return false; }
    }
    return true;
}

fn aabb_in_frustum(mn: vec3<f32>, mx: vec3<f32>) -> bool {
    for (var i = 0u; i < 5u; i++) {
        let p = wg_planes[i];
        let v = select(mn, mx, p.xyz > vec3<f32>(0.0));
        if dot(p.xyz, v) + p.w < 0.0 { return false; }
    }
    return true;
}

fn view_depth_of(c: vec3<f32>) -> f32 {
    return -(cameras[0].view * vec4<f32>(c, 1.0)).z;
}

// Projected size in pixels of an object-space error `err` (already scaled by
// instance scale) at a bounding sphere. Uses the sphere's nearest view depth,
// clamped to the near plane, so a containing sphere never projects smaller
// than a contained one (monotone => the cut is hole-free). Mirrors
// `reference::project_error`.
fn proj_err(err: f32, c: vec3<f32>, r: f32) -> f32 {
    if err >= INF_ERR { return HUGE_PX; }
    let d = max(view_depth_of(c) - r, cameras[0].position_near.w);
    return err * cameras[0].proj[1][1] * 0.5 * f32(params.screen_height) / d;
}

struct InstCtx {
    m: mat4x4<f32>,
    scale: f32,
    err_scale: f32,
    tau: f32,
    flags: u32,
    asset: u32,
}

fn load_inst(id: u32) -> InstCtx {
    let inst = vg_instances[id];
    let t = objects[inst.object_row].transform;
    let m = mat4x4<f32>(
        vec4<f32>(t[0][0], t[0][1], t[0][2], t[0][3]),
        vec4<f32>(t[1][0], t[1][1], t[1][2], t[1][3]),
        vec4<f32>(t[2][0], t[2][1], t[2][2], t[2][3]),
        vec4<f32>(t[3][0], t[3][1], t[3][2], t[3][3]),
    );
    var c: InstCtx;
    c.m = m;
    c.scale = max(length(m[0].xyz), max(length(m[1].xyz), length(m[2].xyz)));
    c.err_scale = max(inst.lod_error_scale, 1.0e-4);
    c.flags = inst.flags;
    c.asset = inst.asset_slot;
    c.tau = select(params.tau_px, ROOT_ONLY_TAU, (inst.flags & VG_INST_FORCE_ROOT_ONLY) != 0u);
    return c;
}

// ── Hi-Z (mirrors helio-pass-occlusion-cull, parameterised on the matrix) ─

fn sphere_near_depth(center: vec3<f32>, radius: f32, vp: mat4x4<f32>) -> f32 {
    let to_center = center - cameras[0].position_near.xyz;
    let dist_sq = dot(to_center, to_center);
    if dist_sq <= radius * radius { return 1.0; }
    let dir = to_center * (1.0 / sqrt(dist_sq));
    let near_clip = vp * vec4<f32>(center - dir * radius, 1.0);
    if near_clip.w <= 0.0 { return 1.0; }
    return clamp(near_clip.z / near_clip.w, 0.0, 1.0);
}

fn hiz_sphere_occluded(center: vec3<f32>, radius: f32, vp: mat4x4<f32>) -> bool {
    if radius <= 0.0 { return false; }
    let clip = vp * vec4<f32>(center, 1.0);
    if clip.w <= 0.0 { return false; }
    let ndc_r = max(
        abs(radius * cameras[0].proj[0][0] / clip.w),
        abs(radius * cameras[0].proj[1][1] / clip.w),
    );
    let ndc = clip.xyz / clip.w;
    if ndc.x + ndc_r < -1.0 || ndc.x - ndc_r > 1.0 || ndc.y + ndc_r < -1.0 || ndc.y - ndc_r > 1.0 {
        return false;
    }
    let near_z = sphere_near_depth(center, radius, vp);
    let r_px = abs(radius / clip.w * cameras[0].proj[1][1] * f32(params.screen_height) * 0.5);
    let mip = clamp(u32(ceil(log2(max(r_px * 2.0, 1.0)))), 0u, params.hiz_mip_count - 1u);
    let uv = helio_ndc_to_uv(ndc.xy);
    let uv_half = ndc_r * 0.5;
    let uv_00 = clamp(uv - vec2<f32>(uv_half), vec2<f32>(0.0), vec2<f32>(1.0));
    let uv_11 = clamp(uv + vec2<f32>(uv_half), vec2<f32>(0.0), vec2<f32>(1.0));
    let h00 = textureSampleLevel(hiz_tex, hiz_samp, uv_00, f32(mip)).r;
    let h01 = textureSampleLevel(hiz_tex, hiz_samp, vec2<f32>(uv_11.x, uv_00.y), f32(mip)).r;
    let h10 = textureSampleLevel(hiz_tex, hiz_samp, vec2<f32>(uv_00.x, uv_11.y), f32(mip)).r;
    let h11 = textureSampleLevel(hiz_tex, hiz_samp, uv_11, f32(mip)).r;
    let farthest = helio_depth_farthest(helio_depth_farthest(h00, h01), helio_depth_farthest(h10, h11));
    return helio_hiz_occludes(farthest, near_z);
}

const POLICY_PROCESS: u32 = 0u;
const POLICY_DEFER: u32 = 1u;
const POLICY_DROP: u32 = 2u;

// Two-phase occlusion policy for one job bound.
//   phase 0: only work that was visible last frame is tried, against last
//            frame's Hi-Z (old view-projection); everything else is deferred.
//            Wrong answers here are safe both ways: a false "visible" is just
//            an extra draw, a false "occluded" is retested in phase 1.
//   phase 1: retest deferred work against the Hi-Z built from phase-0 depth.
fn occlusion_policy(job: vec2<u32>, c: vec3<f32>, r: f32) -> u32 {
    if (params.flags & VGC_FLAG_TWO_PHASE) == 0u { return POLICY_PROCESS; }
    if rp.phase == 0u {
        if !history_test(job_key(job)) { return POLICY_DEFER; }
        if rp.hiz_valid != 0u && hiz_sphere_occluded(c, r * 1.02, cameras[0].prev_view_proj) {
            return POLICY_DEFER;
        }
        return POLICY_PROCESS;
    }
    if rp.hiz_valid != 0u && hiz_sphere_occluded(c, r, cameras[0].view_proj) {
        return POLICY_DROP;
    }
    return POLICY_PROCESS;
}

// ── Queues ──────────────────────────────────────────────────────────────────

fn push_job(job: vec2<u32>) {
    let i = atomicAdd(&wg_sp, 1u);
    if i < STACK_CAP {
        wg_stack[i] = job;
        return;
    }
    let t = atomicAdd(&state[VGC_ST_Q_TAIL + rp.dst], 1u);
    // dst 3 is the sink of a phase's last round: nothing is stored, finalize
    // reports the count as overflow.
    if rp.dst == 3u { return; }
    if t < params.queue_capacity {
        queues[rp.dst * params.queue_capacity + t] = job;
        stat_add(VGC_STAT_SPILLED);
        atomicMax(&state[VGC_ST_STATS + VGC_STAT_PEAK_QUEUE], t + 1u);
    } else {
        stat_add(VGC_STAT_JOB_OVERFLOW);
    }
}

fn defer_job(job: vec2<u32>) {
    let t = atomicAdd(&state[VGC_ST_Q_TAIL + 2u], 1u);
    if t < params.queue_capacity {
        queues[2u * params.queue_capacity + t] = job;
        stat_add(VGC_STAT_DEFERRED);
        atomicMax(&state[VGC_ST_STATS + VGC_STAT_PEAK_QUEUE], t + 1u);
    } else {
        stat_add(VGC_STAT_JOB_OVERFLOW);
    }
}

// ── Binning ────────────────────────────────────────────────────────────────

fn tri_class(tris: u32) -> u32 {
    if tris <= 32u { return 0u; }
    if tris <= 64u { return 1u; }
    if tris <= 96u { return 2u; }
    return 3u;
}

fn dense_overlap(cw: vec3<f32>, r_px: f32) -> bool {
    let clip = cameras[0].view_proj * vec4<f32>(cw, 1.0);
    if clip.w <= 0.0 { return false; }
    let uv = helio_ndc_to_uv(clip.xy / clip.w);
    let px = uv * vec2<f32>(f32(params.screen_width), f32(params.screen_height));
    let tmax = vec2<i32>(i32(params.dense_tiles_x) - 1, i32(params.dense_tiles_y) - 1);
    let lo = clamp(vec2<i32>(floor((px - vec2<f32>(r_px)) / f32(VGC_TILE))), vec2<i32>(0), tmax);
    let hi = min(clamp(vec2<i32>(floor((px + vec2<f32>(r_px)) / f32(VGC_TILE))), vec2<i32>(0), tmax), lo + vec2<i32>(3));
    for (var y = lo.y; y <= hi.y; y++) {
        for (var x = lo.x; x <= hi.x; x++) {
            let idx = u32(y) * params.dense_tiles_x + u32(x);
            if idx < arrayLength(&dense_mask) && dense_mask[idx] != 0u { return true; }
        }
    }
    return false;
}

fn choose_bin(tris: u32, ic: InstCtx, cw: vec3<f32>, rw: f32) -> u32 {
    let hw_bin = tri_class(tris);
    if params.mode == VGC_MODE_FORCE_HW { return hw_bin; }
    let lock = (ic.flags >> VG_INST_TIER_SHIFT) & VG_INST_TIER_MASK;
    if lock == 1u { return hw_bin; }
    if lock == 2u { return VGC_BIN_SW; }
    if lock == 3u { return VGC_BIN_DENSE; }
    let d = max(view_depth_of(cw), cameras[0].position_near.w);
    let r_px = rw * cameras[0].proj[1][1] * 0.5 * f32(params.screen_height) / d;
    let edge = VGC_EDGE_FACTOR * r_px / sqrt(f32(max(tris, 1u)));
    if edge <= params.sw_edge_px {
        if (params.flags & VGC_FLAG_DENSE_MASK) != 0u && dense_overlap(cw, r_px) {
            return VGC_BIN_DENSE;
        }
        return VGC_BIN_SW;
    }
    return hw_bin;
}

fn emit_cluster(inst: u32, slot: u32, bin: u32, triangles: u32) {
    let k = atomicAdd(&state[VGC_ST_BINS + bin], 1u);
    if k >= params.bin_capacity {
        stat_add(VGC_STAT_VISIBLE_OVERFLOW);
        return;
    }
    visible[bin * params.bin_capacity + k] = vec2<u32>(inst, slot);
    stat_add(VGC_STAT_EMITTED);
    if (params.flags & VGC_FLAG_STATS) != 0u {
        atomicAdd(&state[VGC_ST_STATS + VGC_STAT_TRIANGLES], triangles);
    }
}

// ── Cluster test (cut rule, cone, frustum, occlusion, binning) ─────────────

fn try_cluster(inst: u32, cs: u32, ic: InstCtx) {
    stat_add(VGC_STAT_CLUSTERS_TESTED);
    let c = vg_load_cluster(cs >> VG_PAGE_CLUSTER_SHIFT, cs & VG_PAGE_CLUSTER_MASK);
    let cw = (ic.m * vec4<f32>(c.sphere.xyz, 1.0)).xyz;
    let rw = c.sphere.w * ic.scale;
    let pw = (ic.m * vec4<f32>(c.parent_sphere.xyz, 1.0)).xyz;
    let prw = c.parent_sphere.w * ic.scale;
    let k = ic.err_scale * ic.scale;
    let s_err = proj_err(c.self_error * k, cw, rw);
    let p_err = proj_err(c.parent_error * k, pw, prw);

    // Residency: if the clusters this one was simplified from (the members of
    // its producer group) are not resident, this cluster stands in for them
    // (it is the resident parent). `group_id` is asset-local.
    var kids_resident = true;
    if (c.flags & VGC_CLUSTER_FLAG_LEAF) == 0u && c.group_id != VG_NODE_INVALID {
        let gref = vg_group_refs[vg_group_ref_base(ic.asset) + c.group_id];
        let child_page = vg_global_page(ic.asset, gref);
        kids_resident = vg_page_resident(child_page);
        if !kids_resident && s_err > ic.tau {
            vg_request_page(child_page);
        }
    }
    let draw = p_err > ic.tau && (s_err <= ic.tau || !kids_resident);
    if !draw { return; }

    if (ic.flags & VG_INST_CONE_CULL) != 0u {
        let cone = unpack4x8snorm(c.cone);
        // cone.w = cos(half-angle of the normal cone); 1.0 disables.
        if cone.w < 0.9999 {
            let axis = normalize((ic.m * vec4<f32>(cone.xyz, 0.0)).xyz);
            let dv = cw - cameras[0].position_near.xyz;
            let sin_a = sqrt(max(1.0 - cone.w * cone.w, 0.0));
            if dot(dv, axis) >= sin_a * length(dv) + rw {
                stat_add(VGC_STAT_CULLED_CONE);
                return;
            }
        }
    }
    if !sphere_in_frustum(cw, rw) {
        stat_add(VGC_STAT_CULLED_FRUSTUM);
        return;
    }
    let job = vec2<u32>(inst, (2u << 30u) | cs);
    let pol = occlusion_policy(job, cw, rw);
    if pol == POLICY_DEFER { defer_job(job); return; }
    if pol == POLICY_DROP { stat_add(VGC_STAT_CULLED_OCCLUSION); return; }
    history_mark(job_key(job));
    let triangles = c.counts_material & 0xffu;
    emit_cluster(inst, cs, choose_bin(triangles, ic, cw, rw), triangles);
}

// `gidx` is an asset-local group-ref index.
fn do_group(inst: u32, gidx: u32, ic: InstCtx) {
    let g = vg_group_refs[vg_group_ref_base(ic.asset) + gidx];
    let slot = vg_use_or_request_page(vg_global_page(ic.asset, g));
    if slot == VG_PAGE_NOT_RESIDENT { return; }
    let first = g.first_cluster_count & 0xffffu;
    let count = g.first_cluster_count >> 16u;
    for (var i = 0u; i < count; i++) {
        try_cluster(inst, (slot << VG_PAGE_CLUSTER_SHIFT) | (first + i), ic);
    }
}

// ── Job kinds ──────────────────────────────────────────────────────────────

fn job_gcbvh(job: vec2<u32>) {
    let idx = job.y & 0x3fffffffu;
    if idx >= vg_gcbvh_header.node_count { return; }
    let n = vg_gcbvh_nodes[idx];
    if !aabb_in_frustum(n.bmin, n.bmax) {
        stat_add(VGC_STAT_CULLED_FRUSTUM);
        return;
    }
    let c = (n.bmin + n.bmax) * 0.5;
    let r = length(n.bmax - n.bmin) * 0.5;
    let pol = occlusion_policy(job, c, r);
    if pol == POLICY_DEFER { defer_job(job); return; }
    if pol == POLICY_DROP { stat_add(VGC_STAT_CULLED_OCCLUSION); return; }
    history_mark(job_key(job));
    if vg_bvh_is_leaf(n) {
        let count = vg_bvh_leaf_count(n);
        let first = vg_bvh_leaf_first(n);
        for (var i = 0u; i < count; i++) {
            // Leaf items are rows of `vg_instances` through the items table.
            let inst = vg_gcbvh_items[first + i];
            if inst < params.instance_count {
                push_job(vec2<u32>(inst, 3u << 30u));
            }
        }
    } else {
        push_job(vec2<u32>(0xffffffffu, vg_bvh_child(n, 0u)));
        push_job(vec2<u32>(0xffffffffu, vg_bvh_child(n, 1u)));
    }
}

fn job_instance(job: vec2<u32>) {
    if job.x >= params.instance_count { return; }
    let inst = vg_instances[job.x];
    if (inst.flags & VG_INST_ENABLED) == 0u { return; }
    if inst.asset_slot >= arrayLength(&vg_asset_headers) { return; }
    // Not VG-active until the root pages are resident: draw nothing (the
    // derivation keeps such instances on the classic path).
    if !vg_asset_ready(inst.asset_slot) { return; }
    let hdr = vg_asset_headers[inst.asset_slot];
    if hdr.node_count == 0u { return; }
    let root = vg_node_base(inst.asset_slot) + hdr.root_node;
    let children = vg_nodes[root].children;
    for (var s = 0u; s < 4u; s++) {
        if children[s] != VG_NODE_INVALID {
            push_job(vec2<u32>(job.x, (1u << 30u) | (root << 2u) | s));
        }
    }
}

fn job_asset_item(job: vec2<u32>) {
    let ic = load_inst(job.x);
    let payload = job.y & 0x3fffffffu;
    let parent = payload >> 2u;     // global node index
    let slot = payload & 3u;
    let child = vg_nodes[parent].children[slot];
    if child == VG_NODE_INVALID { return; }
    let sph = vg_nodes[parent].spheres[slot];
    let cw = (ic.m * vec4<f32>(sph.xyz, 1.0)).xyz;
    let rw = sph.w * ic.scale;
    // Enter only while something below may still need drawing.
    let mpe = vg_nodes[parent].max_parent_error[slot];
    if proj_err(mpe * ic.err_scale * ic.scale, cw, rw) <= ic.tau {
        stat_add(VGC_STAT_CULLED_SSE);
        return;
    }
    if !sphere_in_frustum(cw, rw) {
        stat_add(VGC_STAT_CULLED_FRUSTUM);
        return;
    }
    let pol = occlusion_policy(job, cw, rw);
    if pol == POLICY_DEFER { defer_job(job); return; }
    if pol == POLICY_DROP { stat_add(VGC_STAT_CULLED_OCCLUSION); return; }
    history_mark(job_key(job));
    if (child & VG_NODE_LEAF_BIT) != 0u {
        do_group(job.x, child & ~VG_NODE_LEAF_BIT, ic);
    } else {
        let gchild = vg_node_base(ic.asset) + child;
        let kids = vg_nodes[gchild].children;
        for (var s = 0u; s < 4u; s++) {
            if kids[s] != VG_NODE_INVALID {
                push_job(vec2<u32>(job.x, (1u << 30u) | (gchild << 2u) | s));
            }
        }
    }
}

fn job_cluster(job: vec2<u32>) {
    try_cluster(job.x, job.y & 0x3fffffffu, load_inst(job.x));
}

fn process_job(job: vec2<u32>) {
    stat_add(VGC_STAT_JOBS);
    switch job.y >> 30u {
        case 0u: { job_gcbvh(job); }
        case 1u: { job_asset_item(job); }
        case 2u: { job_cluster(job); }
        default: { job_instance(job); }
    }
}

// ── Entry points ───────────────────────────────────────────────────────────

// One thread, frame start: seed the traversal with the GCBVH root.
@compute @workgroup_size(1)
fn cs_seed() {
    if params.mode == VGC_MODE_OFF { return; }
    if vg_gcbvh_header.node_count == 0u || vg_gcbvh_header.root_node == VG_BVH_INVALID || params.instance_count == 0u { return; }
    queues[0] = vec2<u32>(0xffffffffu, vg_gcbvh_header.root_node);
    atomicStore(&state[VGC_ST_Q_TAIL], 1u);
}

// One thread per round: reset queue cursors and size the persistent dispatch.
@compute @workgroup_size(1)
fn cs_round_prepare() {
    let n = min(atomicLoad(&state[VGC_ST_Q_TAIL + rp.src]), params.queue_capacity);
    atomicStore(&state[VGC_ST_Q_HEAD + rp.src], 0u);
    atomicStore(&state[VGC_ST_Q_TAIL + rp.dst], 0u);
    atomicStore(&state[VGC_ST_Q_HEAD + rp.dst], 0u);
    dispatch_args[0] = min((n + WG - 1u) / WG, params.max_groups);
    dispatch_args[1] = 1u;
    dispatch_args[2] = 1u;
}

// Persistent workgroups: pull 64-job chunks from the source queue, traverse
// depth-first through a workgroup-local stack, spill to the next round's queue
// only when that stack is full.
@compute @workgroup_size(64)
fn cs_traverse(@builtin(local_invocation_index) lid: u32) {
    if lid < 5u { wg_planes[lid] = frustum_plane(lid); }
    if lid == 0u { atomicStore(&wg_sp, 0u); }
    let src = rp.src;
    let src_n = min(atomicLoad(&state[VGC_ST_Q_TAIL + src]), params.queue_capacity);
    workgroupBarrier();
    loop {
        if lid == 0u {
            let sp = min(atomicLoad(&wg_sp), STACK_CAP);
            if sp > 0u {
                let take = min(sp, WG);
                atomicStore(&wg_sp, sp - take);
                wg_n = take;
                wg_base = sp - take;
                wg_mode = 1u;
            } else {
                let base = atomicAdd(&state[VGC_ST_Q_HEAD + src], WG);
                if base >= src_n {
                    wg_mode = 0u;
                    wg_n = 0u;
                    wg_base = 0u;
                } else {
                    wg_mode = 2u;
                    wg_n = min(WG, src_n - base);
                    wg_base = base;
                }
            }
        }
        workgroupBarrier();
        let mode = workgroupUniformLoad(&wg_mode);
        if mode == 0u { break; }
        let n = workgroupUniformLoad(&wg_n);
        let base = workgroupUniformLoad(&wg_base);
        var job = vec2<u32>(0u);
        let have = lid < n;
        if have {
            if mode == 1u {
                job = wg_stack[base + lid];
            } else {
                job = queues[src * params.queue_capacity + base + lid];
            }
        }
        // Pops are read before any push may reuse the slots.
        workgroupBarrier();
        if have { process_job(job); }
        workgroupBarrier();
    }
}

// One thread per phase end: clamp counts, publish ranges and indirect args.
@compute @workgroup_size(1)
fn cs_finalize() {
    let cap = params.bin_capacity;
    let phase = rp.phase;
    let left = atomicLoad(&state[VGC_ST_Q_TAIL + 3u]);
    if left > 0u {
        atomicAdd(&state[VGC_ST_STATS + VGC_STAT_JOB_OVERFLOW], left);
        atomicStore(&state[VGC_ST_Q_TAIL + 3u], 0u);
    }
    let args = VGC_ST_ARGS + phase * VGC_ARGS_PHASE_WORDS;
    let rng = VGC_ST_RANGES + phase * VGC_BIN_COUNT * 2u;
    for (var b = 0u; b < VGC_BIN_COUNT; b++) {
        let raw = min(atomicLoad(&state[VGC_ST_BINS + b]), cap);
        var skip = 0u;
        var c = raw;
        if phase == 0u {
            atomicStore(&state[VGC_ST_PHASE1 + b], raw);
        } else {
            skip = min(atomicLoad(&state[VGC_ST_PHASE1 + b]), raw);
            c = raw - skip;
        }
        let first = b * cap + skip;
        atomicStore(&state[rng + b * 2u], first);
        atomicStore(&state[rng + b * 2u + 1u], c);
        if b < 4u {
            atomicStore(&state[args + b * 4u], 96u * (b + 1u));
            atomicStore(&state[args + b * 4u + 1u], c);
            atomicStore(&state[args + b * 4u + 2u], 0u);
            atomicStore(&state[args + b * 4u + 3u], first);
        } else {
            let o = args + 16u + (b - 4u) * 4u;
            atomicStore(&state[o], min(c, VGC_SW_GROUP_WIDTH));
            atomicStore(&state[o + 1u], (c + VGC_SW_GROUP_WIDTH - 1u) / VGC_SW_GROUP_WIDTH);
            atomicStore(&state[o + 2u], 1u);
            atomicStore(&state[o + 3u], 0u);
        }
    }
}
