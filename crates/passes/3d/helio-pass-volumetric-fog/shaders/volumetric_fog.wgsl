//!use helio_prelude
//
// Volumetric fog — froxel grid (Hillaire, "Physically Based and Unified Volumetric
// Rendering in Frostbite", SIGGRAPH 2015).
//
// Two compute passes over a view-space 3D grid, rather than a raymarch per pixel:
//
//   cs_inject     — one thread per froxel: density, one shadow tap per light,
//                   temporally blended against the reprojected previous frame.
//   cs_integrate  — one thread per (x,y) column: marches z once, turning the
//                   per-froxel scattering/extinction into accumulated
//                   in-scattering + transmittance.
//
// The composite in postprocess.wgsl is then a single trilinear 3D fetch at the
// pixel's depth:
//
//     color = color * fog.a + fog.rgb
//
// Why this shape:
//   - Cost is decoupled from screen resolution. 192x108x128 = ~2.65M froxels lit
//     once each, against 1280x720x64 = ~59M samples for the per-pixel march.
//   - The trilinear fetch filters in x, y *and* depth, so there is no blocky
//     upsample and no bilateral filter to write.
//   - Temporal reprojection carries samples across frames, which is what makes
//     one shadow tap per froxel enough. Without it this would need many more.

// ── Fog config ──────────────────────────────────────────────────────────────
// Byte-identical to GpuFogUniforms (64 bytes at PP offset 304).
// The legacy block supplies a compatibility medium and the resolved view range.

struct FogUniforms {
    fog_enabled:               u32,
    fog_mode:                  u32,
    fog_density:               f32,
    fog_height_falloff:        f32,
    fog_start_distance:        f32,
    fog_max_distance:          f32,
    fog_height:                f32,
    fog_scattering_anisotropy: f32,
    fog_color:                 vec3<f32>,
    _pad_fog_color:            f32,
    fog_emissive:              vec3<f32>,
    _pad_fog_emissive:         f32,
}

const FOG_MODE_UNIFORM: u32 = 0u;
const FOG_MODE_HEIGHT: u32  = 1u;

// ── Lights ──────────────────────────────────────────────────────────────────
// Mirror of GpuLight (128 bytes). See that struct's doc comment for the
// full list of shaders that must be edited together.

struct GpuLight {
    position_range:  vec4<f32>,
    direction_outer: vec4<f32>,
    color_intensity: vec4<f32>,
    shadow_index:    u32,
    light_type:      u32,
    inner_angle:     f32,
    _pad:            u32,
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

struct LightMatrix { mat: mat4x4<f32> }

struct FogGlobals {
    /// Globals.csm_splits — cascade boundaries for directional shadow lookup.
    csm_splits:  vec4<f32>,
    _reserved: u32,
    frame:       u32,
    /// 0 on the first frame or after a camera cut: ignore history.
    history_valid: u32,
    /// Weight of the current frame in the temporal blend. Lower = steadier, but
    /// slower to react to lights and shadows moving.
    temporal_blend: f32,
    time: f32,
    /// Last frame's camera jitter (NDC), to reproject into its unjittered grid.
    prev_jitter_x: f32,
    prev_jitter_y: f32,
    _pad2: f32,
    grid: vec3<u32>,
    enabled: u32,
}

const LIGHT_DIRECTIONAL: u32 = 0u;
const LIGHT_POINT:       u32 = 1u;
const LIGHT_SPOT:        u32 = 2u;

const NO_SHADOW: u32 = 4294967295u;

@group(0) @binding(0) var<storage, read> cameras: array<Camera, 2>;
@group(0) @binding(1) var<uniform>       fog:             FogUniforms;
@group(0) @binding(2) var<uniform>       fog_globals:     FogGlobals;
@group(0) @binding(3) var<storage, read> lights:          array<GpuLight>;
@group(0) @binding(4) var<storage, read> shadow_matrices: array<LightMatrix>;
@group(0) @binding(5) var                shadow_atlas:    texture_depth_2d_array;
@group(0) @binding(6) var                shadow_samp:     sampler_comparison;
@group(0) @binding(20) var               static_shadow_atlas: texture_depth_2d_array;
/// Translucent casters (stained glass): rgb = 1 - T, a = 1 - nearest pane depth.
@group(0) @binding(21) var               shadow_transmittance: texture_2d_array<f32>;
/// Previous frame's scattering grid, for temporal reprojection.
@group(0) @binding(7) var                scatter_history: texture_3d<f32>;
@group(0) @binding(8) var                linear_samp:     sampler;
/// rgb = in-scattered radiance * density, a = extinction.
@group(0) @binding(9) var                scatter_out:     texture_storage_3d<rgba16float, write>;

// SceneDB PP row. Only its fog block is interpreted; shader_source() derives
// the opaque tail from the current Rust PP ABI (including appended lens data).
struct FogVolume {
    bounds_min: vec4<f32>,
    bounds_max: vec4<f32>,
    priority: f32,
    blend_radius: f32,
    blend_weight: f32,
    unbound: u32,
    _pad: vec4<f32>,
    before_fog: array<vec4<u32>, 19>,
    medium: FogUniforms,
    after_fog: array<vec4<u32>, __PP_TAIL_VEC4__>,
}
struct WorldMedium {
    enabled: u32, mode: u32, extinction: f32, height_falloff: f32,
    height: f32, anisotropy: f32, _pad: vec2<f32>,
    albedo: vec3<f32>, _pad_albedo: f32,
    emission: vec3<f32>, _pad_emission: f32,
}
struct LocalMedium {
    bounds_min: vec4<f32>, bounds_max: vec4<f32>, medium: WorldMedium,
    edge_fade: f32, _pad0: f32, _pad1: f32, _pad2: f32,
}
struct RenderSettings {
    occupied: u32, view_id: u32, quality: u32, enabled: u32,
    max_distance: f32, light_max_distance: f32, temporal_blend: f32, history_rejection: f32,
    light_samples: u32, history_epoch: u32, _pad: vec2<u32>,
}
struct ActiveMedia {
    volume_count: u32,
    light_count: u32,
    previous_range: f32,
    history_compatible: u32,
    global_count: u32,
    local_count: u32,
    has_medium: u32,
    previous_quality: u32,
    grid: vec3<u32>,
    render_enabled: u32,
    light_max_distance: f32,
    temporal_blend: f32,
    history_rejection: f32,
    light_samples: u32,
    previous_epoch: u32,
    epoch: u32,
    previous_light_hash: u32,
    last_active: u32,
    _reserved: array<u32, 12>,
    volumes: array<u32, 64>,
    lights: array<u32, 256>,
    globals: array<u32, 64>,
    locals: array<u32, 64>,
}
@group(0) @binding(11) var<storage, read> volumes: array<FogVolume>;
@group(0) @binding(12) var<storage, read_write> media_list: ActiveMedia;
// Distinct layout for classification: this buffer is never bound while used
// as an indirect-dispatch argument (STORAGE_WRITE and INDIRECT conflict).
// [0..3] inject, [3..6] cull, [6..9] integrate (columns of the allocated grid).
@group(0) @binding(13) var<storage, read_write> dispatch: array<u32, 9>;
@group(0) @binding(14) var<storage, read> global_media: array<WorldMedium>;
@group(0) @binding(15) var<storage, read> local_media: array<LocalMedium>;
@group(0) @binding(16) var<storage, read> render_settings: array<RenderSettings>;
@group(0) @binding(17) var<storage, read> legacy_media: array<FogUniforms>;
@group(0) @binding(18) var<storage, read_write> resolved_fog: FogUniforms;
struct Cluster { count: u32, lights: array<u32, 64> }
@group(0) @binding(19) var<storage, read_write> clusters: array<Cluster>;

fn finite_clamp(v: f32, lo: f32, hi: f32) -> f32 {
    return select(lo, min(v, hi), v >= lo);
}

// SceneDB rows are indexed by entity, so every scanned array spans the whole
// entity range and is mostly holes. Scans use one 256-thread workgroup and
// read only row headers; lowest-row selection keeps results deterministic.
const SCAN_THREADS: u32 = 256u;
const NONE_ROW: u32 = 0xffffffffu;
var<workgroup> wg_legacy_row: atomic<u32>;
var<workgroup> wg_exact_row: atomic<u32>;
var<workgroup> wg_fallback_row: atomic<u32>;

@compute @workgroup_size(256)
fn cs_resolve(@builtin(local_invocation_index) lid: u32) {
    if lid == 0u {
        atomicStore(&wg_legacy_row, NONE_ROW);
        atomicStore(&wg_exact_row, NONE_ROW);
        atomicStore(&wg_fallback_row, NONE_ROW);
    }
    workgroupBarrier();
    let view_id = bitcast<u32>(cameras[0].jitter_frame.w);
    for (var i = lid; i < arrayLength(&legacy_media); i += SCAN_THREADS) {
        if legacy_media[i].fog_enabled != 0u { atomicMin(&wg_legacy_row, i); }
    }
    for (var i = lid; i < arrayLength(&render_settings); i += SCAN_THREADS) {
        if render_settings[i].occupied == 0u { continue; }
        let row_view = render_settings[i].view_id;
        if row_view == view_id { atomicMin(&wg_exact_row, i); }
        else if row_view == 0xffffffffu { atomicMin(&wg_fallback_row, i); }
    }
    workgroupBarrier();
    if lid != 0u { return; }

    var config = fog;
    let legacy_row = atomicLoad(&wg_legacy_row);
    if legacy_row != NONE_ROW { config = legacy_media[legacy_row]; }
    // Exact view matches beat the all-view fallback; lowest row wins ties.
    var selected = atomicLoad(&wg_exact_row);
    if selected == NONE_ROW { selected = atomicLoad(&wg_fallback_row); }
    var quality = 0u;
    var epoch = 0u;
    var range = select(1000.0, config.fog_max_distance, config.fog_max_distance > HELIO_FROXEL_NEAR);
    media_list.render_enabled = fog_globals.enabled;
    media_list.temporal_blend = finite_clamp(fog_globals.temporal_blend, 0.01, 1.0);
    media_list.history_rejection = 0.8;
    media_list.light_max_distance = range;
    media_list.light_samples = 4u;
    if selected != 0xffffffffu {
        let s = render_settings[selected];
        quality = min(s.quality, 1u);
        epoch = s.history_epoch;
        range = s.max_distance;
        media_list.render_enabled &= s.enabled;
        media_list.temporal_blend = finite_clamp(s.temporal_blend, 0.01, 1.0);
        media_list.history_rejection = finite_clamp(s.history_rejection, 0.01, 1.0);
        media_list.light_max_distance = finite_clamp(s.light_max_distance, 0.1, 100000.0);
        media_list.light_samples = select(clamp(s.light_samples, 1u, 32u), select(4u, 12u, quality == 1u), s.light_samples == 0u);
    }
    range = finite_clamp(range, HELIO_FROXEL_NEAR + 0.01, 100000.0);
    media_list.history_compatible = select(0u, 1u,
        abs(media_list.previous_range - range) < 0.001 && media_list.previous_epoch == epoch && media_list.previous_quality == quality);
    media_list.previous_range = range;
    media_list.previous_epoch = epoch;
    media_list.previous_quality = quality;
    media_list.epoch = epoch;
    media_list.grid = max(fog_globals.grid / select(2u, 1u, quality == 1u), vec3<u32>(1));
    config.fog_max_distance = range;
    if media_list.render_enabled == 0u { config.fog_enabled = 0u; }
    resolved_fog = config;

}

fn world_medium_active(m: WorldMedium) -> bool {
    return m.enabled != 0u && (m.extinction > 0.0 || any(m.emission > vec3<f32>(0.0)));
}
fn legacy_volume_active(v: FogVolume) -> bool {
    return v.unbound == 0u && v.blend_weight > 0.0 && v.medium.fog_enabled != 0u
        && v.medium.fog_density > 0.0 && all(v.bounds_max.xyz > v.bounds_min.xyz);
}
fn local_medium_active(v: LocalMedium) -> bool {
    return world_medium_active(v.medium) && all(v.bounds_max.xyz > v.bounds_min.xyz);
}
fn light_active(l: GpuLight) -> bool {
    return l.god_rays_enabled != 0u && l.color_intensity.w > 0.0
        && l.god_rays_weight > 0.0 && l.god_rays_density > 0.0 && l.god_rays_exposure > 0.0;
}
fn hash_word(h: u32, word: u32) -> u32 { return (h ^ word) * 16777619u; }
fn hash_vector(h: u32, v: vec4<f32>) -> u32 {
    return hash_word(hash_word(hash_word(hash_word(h, bitcast<u32>(v.x)), bitcast<u32>(v.y)), bitcast<u32>(v.z)), bitcast<u32>(v.w));
}

var<workgroup> wg_volume_rows: array<u32, 64>;
var<workgroup> wg_global_rows: array<u32, 64>;
var<workgroup> wg_local_rows: array<u32, 64>;
var<workgroup> wg_light_rows: array<u32, 256>;
var<workgroup> wg_volume_count: atomic<u32>;
var<workgroup> wg_global_count: atomic<u32>;
var<workgroup> wg_local_count: atomic<u32>;
var<workgroup> wg_light_count: atomic<u32>;
var<workgroup> wg_light_hash: atomic<u32>;

// Structural identity only: which rows participate, their type and shadow slot.
// Continuous changes (flicker, colour, gains, a moving or rotating light) must
// not discard history: every froxel would then show its raw jittered sample,
// i.e. per-frame noise. The exponential blend follows them within a few
// frames, as it does for drifting smoke.
fn light_hash_of(i: u32, light: GpuLight) -> u32 {
    var h = hash_word(2166136261u, i);
    h = hash_word(h, light.shadow_index);
    return hash_word(h, light.light_type);
}

// Authored media, by value: an edit (density, colour, emission, bounds) is a
// discrete change that must show this frame, not fade in over the blend.
// Animated smoke is time-driven and leaves the rows, so this hash, unchanged.
fn medium_hash_of(seed: u32, m: WorldMedium) -> u32 {
    var h = hash_word(hash_word(seed, m.enabled), m.mode);
    h = hash_vector(h, vec4<f32>(m.extinction, m.height_falloff, m.height, m.anisotropy));
    h = hash_vector(h, vec4<f32>(m.albedo, 0.0));
    return hash_vector(h, vec4<f32>(m.emission, 0.0));
}

// Parallel compaction of live rows, then a rank sort so the compact lists are
// in row order regardless of atomic ordering. Counts past a list's capacity
// are still counted; consumers then scan every row instead of losing any.
@compute @workgroup_size(256)
fn cs_classify(@builtin(local_invocation_index) lid: u32) {
    if lid == 0u {
        atomicStore(&wg_volume_count, 0u);
        atomicStore(&wg_global_count, 0u);
        atomicStore(&wg_local_count, 0u);
        atomicStore(&wg_light_count, 0u);
        atomicStore(&wg_light_hash, 0u);
    }
    workgroupBarrier();
    for (var i = lid; i < arrayLength(&volumes); i += SCAN_THREADS) {
        if volumes[i].blend_weight > 0.0 && legacy_volume_active(volumes[i]) {
            let slot = atomicAdd(&wg_volume_count, 1u);
            if slot < 64u { wg_volume_rows[slot] = i; }
        }
    }
    for (var i = lid; i < arrayLength(&global_media); i += SCAN_THREADS) {
        if global_media[i].enabled != 0u && world_medium_active(global_media[i]) {
            atomicAdd(&wg_light_hash, medium_hash_of(hash_word(0x9e3779b9u, i), global_media[i]));
            let slot = atomicAdd(&wg_global_count, 1u);
            if slot < 64u { wg_global_rows[slot] = i; }
        }
    }
    for (var i = lid; i < arrayLength(&local_media); i += SCAN_THREADS) {
        if local_media[i].medium.enabled != 0u && local_medium_active(local_media[i]) {
            let bounds = hash_vector(hash_vector(hash_word(0x85ebca6bu, i), local_media[i].bounds_min),
                local_media[i].bounds_max);
            atomicAdd(&wg_light_hash, medium_hash_of(hash_word(bounds, bitcast<u32>(local_media[i].edge_fade)), local_media[i].medium));
            let slot = atomicAdd(&wg_local_count, 1u);
            if slot < 64u { wg_local_rows[slot] = i; }
        }
    }
    for (var i = lid; i < arrayLength(&lights); i += SCAN_THREADS) {
        if lights[i].god_rays_enabled != 0u && light_active(lights[i]) {
            // Commutative combination: independent of scan order.
            atomicAdd(&wg_light_hash, light_hash_of(i, lights[i]));
            let slot = atomicAdd(&wg_light_count, 1u);
            if slot < 256u { wg_light_rows[slot] = i; }
        }
    }
    workgroupBarrier();
    let volume_total = atomicLoad(&wg_volume_count);
    let global_total = atomicLoad(&wg_global_count);
    let local_total = atomicLoad(&wg_local_count);
    let light_total = atomicLoad(&wg_light_count);
    // Rows are unique, so each element's rank is the count of smaller rows.
    if lid < min(volume_total, 64u) {
        let row = wg_volume_rows[lid];
        var rank = 0u;
        for (var j = 0u; j < min(volume_total, 64u); j++) { rank += select(0u, 1u, wg_volume_rows[j] < row); }
        media_list.volumes[rank] = row;
    }
    if lid < min(global_total, 64u) {
        let row = wg_global_rows[lid];
        var rank = 0u;
        for (var j = 0u; j < min(global_total, 64u); j++) { rank += select(0u, 1u, wg_global_rows[j] < row); }
        media_list.globals[rank] = row;
    }
    if lid < min(local_total, 64u) {
        let row = wg_local_rows[lid];
        var rank = 0u;
        for (var j = 0u; j < min(local_total, 64u); j++) { rank += select(0u, 1u, wg_local_rows[j] < row); }
        media_list.locals[rank] = row;
    }
    if lid < min(light_total, 256u) {
        let row = wg_light_rows[lid];
        var rank = 0u;
        for (var j = 0u; j < min(light_total, 256u); j++) { rank += select(0u, 1u, wg_light_rows[j] < row); }
        media_list.lights[rank] = row;
    }
    if lid != 0u { return; }

    media_list.volume_count = volume_total;
    media_list.global_count = global_total;
    media_list.local_count = local_total;
    media_list.light_count = light_total;
    // Lights (structural) and media (by value) share one commutative hash.
    let light_hash = atomicLoad(&wg_light_hash) ^ light_total;
    if media_list.previous_light_hash != light_hash { media_list.history_compatible = 0u; }
    media_list.previous_light_hash = light_hash;
    media_list.has_medium = select(0u, 1u, media_list.render_enabled != 0u &&
        ((fog.fog_enabled != 0u && fog.fog_density > 0.0) || media_list.volume_count + media_list.global_count + media_list.local_count > 0u));
    let was_active = media_list.last_active;
    if media_list.has_medium != media_list.last_active { media_list.history_compatible = 0u; }
    media_list.last_active = media_list.has_medium;
    let tiles = (media_list.grid.xy + vec2<u32>(7)) / 8u;
    dispatch[0] = tiles.x * media_list.has_medium;
    dispatch[1] = tiles.y;
    dispatch[2] = media_list.grid.z;
    dispatch[3] = tiles.x * media_list.has_medium;
    dispatch[4] = tiles.y;
    dispatch[5] = (media_list.grid.z + 3u) / 4u;
    // Integration rewrites the whole grid. With no medium this frame or last,
    // the grid already holds the neutral (0,0,0,1) and needs no work at all.
    let integrate = select(0u, 1u, media_list.has_medium != 0u || was_active != 0u);
    dispatch[6] = ((fog_globals.grid.x + 7u) / 8u) * integrate;
    dispatch[7] = (fog_globals.grid.y + 7u) / 8u;
    dispatch[8] = 1u;
}

// An overflow scans all rows instead of silently losing light or extinction.
fn volume_count() -> u32 { return select(media_list.volume_count, arrayLength(&volumes), media_list.volume_count > 64u); }
fn global_count() -> u32 { return select(media_list.global_count, arrayLength(&global_media), media_list.global_count > 64u); }
fn local_count() -> u32 { return select(media_list.local_count, arrayLength(&local_media), media_list.local_count > 64u); }
fn light_count() -> u32 { return select(media_list.light_count, arrayLength(&lights), media_list.light_count > 256u); }
fn volume_index(i: u32) -> u32 { if media_list.volume_count > 64u { return i; } return media_list.volumes[i]; }
fn global_index(i: u32) -> u32 { if media_list.global_count > 64u { return i; } return media_list.globals[i]; }
fn local_index(i: u32) -> u32 { if media_list.local_count > 64u { return i; } return media_list.locals[i]; }
fn light_index(i: u32) -> u32 { if media_list.light_count > 256u { return i; } return media_list.lights[i]; }

fn cluster_index(tile: vec3<u32>) -> u32 {
    let tiles = (media_list.grid.xy + vec2<u32>(7)) / 8u;
    return (tile.z * tiles.y + tile.y) * tiles.x + tile.x;
}

@compute @workgroup_size(1)
fn cs_cull(@builtin(global_invocation_id) tile: vec3<u32>) {
    let cell_min = tile * vec3<u32>(8, 8, 4);
    let cell_max = min(cell_min + vec3<u32>(8, 8, 4), media_list.grid);
    var lo = vec3<f32>(1e30);
    var hi = vec3<f32>(-1e30);
    // The frustum cell's eight corners conservatively bound every jittered sample.
    for (var corner = 0u; corner < 8u; corner++) {
        let c = vec3<u32>(select(cell_min.x, cell_max.x, (corner & 1u) != 0u),
            select(cell_min.y, cell_max.y, (corner & 2u) != 0u),
            select(cell_min.z, cell_max.z, (corner & 4u) != 0u));
        let uvw = vec3<f32>(c) / vec3<f32>(media_list.grid);
        let p = froxel_world_pos(uvw.xy, uvw.z);
        lo = min(lo, p); hi = max(hi, p);
    }
    let id = cluster_index(tile);
    var count = 0u;
    for (var i = 0u; i < light_count(); i++) {
        let index = light_index(i);
        let l = lights[index];
        if !light_active(l) { continue; }
        let delta = l.position_range.xyz - clamp(l.position_range.xyz, lo, hi);
        if l.light_type != LIGHT_DIRECTIONAL && dot(delta, delta) > l.position_range.w * l.position_range.w { continue; }
        if count < 64u { clusters[id].lights[count] = index; }
        count++;
    }
    clusters[id].count = count;
}

// Integration reads the scattering grid written above and writes the accumulated
// result. Separate group so the two dispatches can swap only what differs.
@group(1) @binding(0) var scatter_in:     texture_3d<f32>;
/// rgb = accumulated in-scattering, a = transmittance.
@group(1) @binding(1) var integrated_out: texture_storage_3d<rgba16float, write>;

// ── Density ─────────────────────────────────────────────────────────────────

fn density_at(p: vec3<f32>, config: FogUniforms) -> f32 {
    var density = max(config.fog_density, 0.0);
    if config.fog_mode != FOG_MODE_UNIFORM {
        // Full density at or below fog_height, decaying exponentially above it.
        // max(h, 0) rather than h: without the clamp a point far below the base
        // gets exp(+large) and the grid fills with an opaque wall.
        let h = p.y - config.fog_height;
        density *= exp(-max(h, 0.0) * max(config.fog_height_falloff, 0.0));
    }
    if config.fog_mode == 2u {
        let advected = p * 0.24 - vec3<f32>(0.10, 0.24, 0.055) * fog_globals.time;
        let coarse = smoke_noise(advected);
        let detail = smoke_noise(advected * 2.07 + vec3<f32>(coarse * 1.7));
        let billow = smoothstep(0.25, 0.72, coarse * 0.7 + detail * 0.3);
        density *= billow * billow * 2.8;
    }
    return density;
}

fn smoke_hash(p: vec3<f32>) -> f32 {
    var q = fract(p * 0.1031);
    q += dot(q, q.yzx + vec3<f32>(33.33));
    return fract((q.x + q.y) * q.z);
}

fn smoke_noise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(mix(smoke_hash(i), smoke_hash(i + vec3<f32>(1,0,0)), u.x),
            mix(smoke_hash(i + vec3<f32>(0,1,0)), smoke_hash(i + vec3<f32>(1,1,0)), u.x), u.y),
        mix(mix(smoke_hash(i + vec3<f32>(0,0,1)), smoke_hash(i + vec3<f32>(1,0,1)), u.x),
            mix(smoke_hash(i + vec3<f32>(0,1,1)), smoke_hash(i + vec3<f32>(1,1,1)), u.x), u.y), u.z);
}

struct Medium {
    extinction: f32,
    albedo: vec3<f32>,
    emissive: vec3<f32>,
    anisotropy: f32,
    scattering_weight: f32,
}

fn world_config(m: WorldMedium) -> FogUniforms {
    var f: FogUniforms;
    f.fog_enabled = m.enabled;
    f.fog_mode = m.mode;
    f.fog_density = finite_clamp(m.extinction, 0.0, 10000.0);
    f.fog_height_falloff = finite_clamp(m.height_falloff, 0.0, 10000.0);
    f.fog_height = m.height;
    return f;
}

fn local_shape(p: vec3<f32>, v: LocalMedium) -> f32 {
    if any(p < v.bounds_min.xyz) || any(p > v.bounds_max.xyz) { return 0.0; }
    if v.edge_fade <= 0.0 { return 1.0; }
    let edge = min(p - v.bounds_min.xyz, v.bounds_max.xyz - p);
    return smoothstep(0.0, v.edge_fade, min(edge.x, min(edge.y, edge.z)));
}

fn add_world_medium(accum: Medium, m: WorldMedium, p: vec3<f32>, shape: f32) -> Medium {
    var result = accum;
    var config = world_config(m);
    config.fog_density = 1.0;
    let profile = density_at(p, config) * shape;
    let d = finite_clamp(m.extinction, 0.0, 10000.0) * profile;
    let scattering = clamp(m.albedo, vec3<f32>(0), vec3<f32>(1)) * d;
    result.extinction += d;
    result.albedo += scattering;
    result.emissive += max(m.emission, vec3<f32>(0)) * profile;
    // Scattering-weighted phase, not absorption-weighted phase.
    let weight = dot(scattering, vec3<f32>(1.0 / 3.0));
    result.anisotropy += finite_clamp(m.anisotropy, -0.95, 0.95) * weight;
    result.scattering_weight += weight;
    return result;
}

fn medium_at(p: vec3<f32>, view_depth: f32) -> Medium {
    var m: Medium;
    if fog.fog_enabled != 0u && view_depth >= fog.fog_start_distance {
        m.extinction = density_at(p, fog);
        m.albedo = clamp(fog.fog_color, vec3<f32>(0.0), vec3<f32>(1.0)) * m.extinction;
        m.emissive = fog.fog_emissive * m.extinction;
        m.scattering_weight = dot(m.albedo, vec3<f32>(1.0 / 3.0));
        m.anisotropy = fog.fog_scattering_anisotropy * m.scattering_weight;
    }
    for (var i = 0u; i < volume_count(); i++) {
        let v = volumes[volume_index(i)];
        if !legacy_volume_active(v) { continue; }
        if any(p < v.bounds_min.xyz) || any(p > v.bounds_max.xyz) { continue; }
        if view_depth < v.medium.fog_start_distance || view_depth > v.medium.fog_max_distance { continue; }
        let edge = min(p - v.bounds_min.xyz, v.bounds_max.xyz - p);
        let fade = smoothstep(0.0, max(v.blend_radius, 0.001), min(edge.x, min(edge.y, edge.z)));
        var shape = fade;
        if v.medium.fog_mode == 2u {
            // A billowing pocket tapers before the AABB corners; the box is
            // still its authoritative extent, not an opaque rectangular slab.
            let q = (2.0 * p - v.bounds_min.xyz - v.bounds_max.xyz) / (v.bounds_max.xyz - v.bounds_min.xyz);
            shape *= 1.0 - smoothstep(0.35, 1.0, dot(q, q));
        }
        let d = density_at(p, v.medium) * clamp(v.blend_weight, 0.0, 1.0) * shape;
        // Overlapping media add extinction/scattering, not camera-weighted screen effects.
        m.extinction += d;
        m.albedo += clamp(v.medium.fog_color, vec3<f32>(0.0), vec3<f32>(1.0)) * d;
        m.emissive += v.medium.fog_emissive * d;
        let weight = dot(clamp(v.medium.fog_color, vec3<f32>(0), vec3<f32>(1)) * d, vec3<f32>(1.0 / 3.0));
        m.anisotropy += v.medium.fog_scattering_anisotropy * weight;
        m.scattering_weight += weight;
    }
    for (var i = 0u; i < global_count(); i++) {
        let config = global_media[global_index(i)];
        if world_medium_active(config) { m = add_world_medium(m, config, p, 1.0); }
    }
    for (var i = 0u; i < local_count(); i++) {
        let v = local_media[local_index(i)];
        if local_medium_active(v) { m = add_world_medium(m, v.medium, p, local_shape(p, v)); }
    }
    if m.extinction > 0.0 { m.albedo /= m.extinction; }
    if m.scattering_weight > 0.0 { m.anisotropy /= m.scattering_weight; }
    return m;
}

// Clip the light segment to each local volume before quadrature; a small dense
// cloud must not disappear between samples of a much longer light ray.
fn ray_box(p: vec3<f32>, dir: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>, distance: f32) -> vec2<f32> {
    var interval = vec2<f32>(0.0, distance);
    for (var axis = 0u; axis < 3u; axis++) {
        if abs(dir[axis]) < 1e-8 {
            if p[axis] < lo[axis] || p[axis] > hi[axis] { return vec2<f32>(0); }
        } else {
            let a = (lo[axis] - p[axis]) / dir[axis];
            let b = (hi[axis] - p[axis]) / dir[axis];
            interval.x = max(interval.x, min(a, b));
            interval.y = min(interval.y, max(a, b));
        }
    }
    return vec2<f32>(interval.x, max(interval.x, interval.y));
}

fn global_optical_depth(config: FogUniforms, p: vec3<f32>, dir: vec3<f32>, distance: f32) -> f32 {
    if config.fog_mode == 0u { return max(config.fog_density, 0.0) * distance; }
    if config.fog_mode != 2u {
        // Integrate the exponential height profile analytically. Order the
        // endpoints by height so the exponent never grows or overflows.
        let slope = abs(dir.y);
        if slope < 1e-8 { return density_at(p, config) * distance; }
        let low = min(p.y, p.y + dir.y * distance);
        let below = clamp((config.fog_height - low) / slope, 0.0, distance);
        let falloff = max(config.fog_height_falloff, 0.0);
        let start = exp(-max(low + below * slope - config.fog_height, 0.0) * falloff);
        return max(config.fog_density, 0.0) * (below + start * segment_integral(falloff * slope, distance - below));
    }
    let count = max(media_list.light_samples, 1u);
    let step = distance / f32(count);
    var tau = 0.0;
    for (var i = 0u; i < count; i++) {
        tau += density_at(p + dir * ((f32(i) + 0.5) * step), config) * step;
    }
    return tau;
}

fn medium_transmittance(p: vec3<f32>, dir: vec3<f32>, distance: f32) -> f32 {
    var tau = 0.0;
    if fog.fog_enabled != 0u { tau += global_optical_depth(fog, p, dir, distance); }
    for (var i = 0u; i < global_count(); i++) {
        let m = global_media[global_index(i)];
        if world_medium_active(m) { tau += global_optical_depth(world_config(m), p, dir, distance); }
    }
    let count = max(media_list.light_samples, 1u);
    for (var i = 0u; i < local_count(); i++) {
        let v = local_media[local_index(i)];
        if !local_medium_active(v) { continue; }
        let segment = ray_box(p, dir, v.bounds_min.xyz, v.bounds_max.xyz, distance);
        let length = segment.y - segment.x;
        if length <= 0.0 { continue; }
        let config = world_config(v.medium);
        if v.edge_fade <= 0.0 {
            tau += global_optical_depth(config, p + dir * segment.x, dir, length);
        } else {
            let step = length / f32(count);
            for (var j = 0u; j < count; j++) {
                let q = p + dir * (segment.x + (f32(j) + 0.5) * step);
                tau += density_at(q, config) * local_shape(q, v) * step;
            }
        }
    }
    for (var i = 0u; i < volume_count(); i++) {
        let v = volumes[volume_index(i)];
        if !legacy_volume_active(v) { continue; }
        let segment = ray_box(p, dir, v.bounds_min.xyz, v.bounds_max.xyz, distance);
        let step = (segment.y - segment.x) / f32(count);
        for (var j = 0u; j < count; j++) {
            let q = p + dir * (segment.x + (f32(j) + 0.5) * step);
            let edge = min(q - v.bounds_min.xyz, v.bounds_max.xyz - q);
            var shape = smoothstep(0.0, max(v.blend_radius, 0.001), min(edge.x, min(edge.y, edge.z)));
            if v.medium.fog_mode == 2u {
                let relative = (2.0 * q - v.bounds_min.xyz - v.bounds_max.xyz) / (v.bounds_max.xyz - v.bounds_min.xyz);
                shape *= 1.0 - smoothstep(0.35, 1.0, dot(relative, relative));
            }
            tau += density_at(q, v.medium) * clamp(v.blend_weight, 0.0, 1.0) * shape * step;
        }
    }
    return exp(-min(tau, 80.0));
}

// ── Froxel <-> world ────────────────────────────────────────────────────────

/// World position at the centre of a froxel, given normalized grid coords.
///
/// `slice_norm` maps through the prelude's exponential distribution, so this is
/// the exact inverse of what the composite does to find a slice from a depth.
fn froxel_world_pos(uv: vec2<f32>, slice_norm: f32) -> vec3<f32> {
    // The grid is anchored to the UNJITTERED frustum. With TSR the camera
    // matrices carry a sub-pixel jitter that changes every frame; a grid that
    // moves with it is resampled at the jitter delta each frame, and steep
    // gradients (a lamp's glow, shaft edges) then oscillate: shimmer that grows
    // with resolution. Adding the jitter back unprojects through the jittered
    // inverse to the unjittered ray.
    let ndc = helio_uv_to_ndc(uv) + cameras[0].jitter_frame.xy;

    // Ray through this pixel: unproject the near and far plane points. Cheaper
    // schemes exist, but this one cannot disagree with the depth reconstruction
    // the rest of the engine does.
    let p_near = cameras[0].view_proj_inv * vec4<f32>(ndc, 0.0, 1.0);
    let p_far  = cameras[0].view_proj_inv * vec4<f32>(ndc, 1.0, 1.0);
    let wn = p_near.xyz / p_near.w;
    let wf = p_far.xyz / p_far.w;
    let dir = normalize(wf - wn);

    let view_depth = helio_froxel_view_depth_from_slice(slice_norm, fog.fog_max_distance);

    // Slices are planes of constant *view depth*, not spheres of constant radius,
    // so the radial distance along this ray is view_depth / cos(angle to forward).
    // Skipping this bows the grid toward the camera at the screen edges.
    let fwd = normalize(cameras[0].forward_far.xyz);
    let cos_a = max(dot(dir, fwd), 1e-4);
    return cameras[0].position_near.xyz + dir * (view_depth / cos_a);
}

// ── Shadowing ───────────────────────────────────────────────────────────────

// Same face order and tie policy as deferred_lighting.wgsl / HLFS and the
// shadow-matrix producer: +X, -X, +Y, -Y, +Z, -Z. Projection uses the shared
// helio_shadow_project helper, including atlas Y flip and validity checks.
fn point_light_face(dir: vec3<f32>) -> u32 {
    let a = abs(dir);
    if a.x >= a.y && a.x >= a.z { return select(0u, 1u, dir.x < 0.0); }
    if a.y >= a.x && a.y >= a.z { return select(2u, 3u, dir.y < 0.0); }
    return select(4u, 5u, dir.z < 0.0);
}

/// Fraction of `light_idx` reaching `p`. 1.0 = fully lit.
///
/// One comparison tap, not the PCF/PCSS kernel deferred lighting uses. That is
/// affordable because it runs once per froxel rather than once per pixel per
/// step, and the temporal blend averages the result across frames.
fn shaft_visibility(light_idx: u32, p: vec3<f32>) -> vec3<f32> {
    if textureDimensions(shadow_atlas).x <= 1u { return vec3<f32>(1.0); }
    let light = lights[light_idx];
    if light.shadow_index == NO_SHADOW { return vec3<f32>(1.0); }

    var layer = light.shadow_index;

    if light.light_type == LIGHT_DIRECTIONAL {
        let dist = length(p - cameras[0].position_near.xyz);
        let sel = helio_csm_select(dist, fog_globals.csm_splits);
        layer = light.shadow_index + sel.cascade_a;
    } else if light.light_type == LIGHT_POINT {
        layer = light.shadow_index + point_light_face(p - light.position_range.xyz);
    }

    if layer >= arrayLength(&shadow_matrices) || layer >= textureNumLayers(shadow_atlas) { return vec3<f32>(1.0); }
    let proj = helio_shadow_project(shadow_matrices[layer].mat, p);
    // Outside the map or behind the light: lit, not shadowed. Returning 0.0 would
    // ring the fog with a black shell wherever the cascade ends.
    if !proj.valid { return vec3<f32>(1.0); }

    // Dynamic (movable) and cached static casters, as deferred lighting does.
    let lit = min(textureSampleCompareLevel(shadow_atlas, shadow_samp, proj.uv, layer, proj.depth),
                  textureSampleCompareLevel(static_shadow_atlas, shadow_samp, proj.uv, layer, proj.depth));
    if lit <= 0.0 || layer >= textureNumLayers(shadow_transmittance) { return vec3<f32>(lit); }
    // Light that crossed stained glass arrives coloured: the shafts take the
    // panes' tint, not just their outline.
    let glass = textureSampleLevel(shadow_transmittance, linear_samp, proj.uv, layer, 0.0);
    let tint = select(vec3<f32>(1.0), 1.0 - glass.rgb, 1.0 - proj.depth < glass.a);
    return lit * tint;
}

// ── Light evaluation ────────────────────────────────────────────────────────

/// In-scattered radiance from one light at `p`, for a view ray `ray_dir`.
///
/// `god_rays_weight` / `god_rays_exposure` / `god_rays_density` come from the
/// radial-blur god-ray technique and have no physical meaning here; they are kept
/// as artistic multipliers so existing light setups author the same way.
/// `god_rays_decay` is the migrated volumetric geometric-shadow strength.
/// Editor-native lights use weight as their single artistic gain, density and
/// exposure equal to one, and enabled as their participation flag.
/// The froxel one injected sample stands for: its view ray segment [t0, t1]
/// (radial distances from the camera) and lateral half-width. Unset (t1 <= t0)
/// outside cs_inject, e.g. for point probes, which then evaluate at `p`.
var<private> sample_t0: f32 = 0.0;
var<private> sample_t1: f32 = 0.0;
var<private> sample_lateral: f32 = 0.0;
/// The froxel's centre ray (unjittered), along which local lights are evaluated.
var<private> sample_center_dir: vec3<f32> = vec3<f32>(0.0);

/// Mean of 1/r^2 from a point light over the froxel's ray segment, in closed
/// form: with h the light's distance from the ray and tc its projection,
/// int dt / (h^2 + (t - tc)^2) = atan((t - tc) / h) / h. Point-sampling the
/// jittered depth instead gave samples near a lamp wildly different radiance
/// (a froxel column is metres deep), which history cannot average out in
/// time: visible shimmer around every lamp in the medium. The lateral extent
/// softens h (mean of 1/r^2 over a disc), so the lamp centre stays finite.
fn segment_inverse_square(light_pos: vec3<f32>) -> f32 {
    let ray_dir = sample_center_dir;
    let origin = cameras[0].position_near.xyz;
    let to_light = light_pos - origin;
    let tc = dot(to_light, ray_dir);
    let h2 = max(dot(to_light, to_light) - tc * tc, 0.0) + sample_lateral * sample_lateral / 3.0;
    let h = sqrt(max(h2, 1e-8));
    let span = max(sample_t1 - sample_t0, 1e-6);
    return (atan((sample_t1 - tc) / h) - atan((sample_t0 - tc) / h)) / (h * span);
}

fn inscatter_from_light(light_idx: u32, p: vec3<f32>, ray_dir: vec3<f32>, anisotropy: f32) -> vec3<f32> {
    let light = lights[light_idx];

    // Opt-in per light: each one costs a shadow tap per froxel.
    if !light_active(light) { return vec3<f32>(0.0); }

    var to_light: vec3<f32>;
    var atten = 1.0;
    var shadow_distance = media_list.light_max_distance;
    var view_dir = ray_dir;

    if light.light_type == LIGHT_DIRECTIONAL {
        if dot(light.direction_outer.xyz, light.direction_outer.xyz) < 1e-12 { return vec3<f32>(0); }
        to_light = normalize(-light.direction_outer.xyz);
    } else {
        // Local lights are evaluated deterministically on the froxel's centre
        // ray: 1/r^2 is integrated over the segment in closed form, and the
        // direction (phase, cone, range window) is taken at the segment point
        // nearest the light, which dominates that integral. Near a lamp the
        // direction to it swings through large angles inside one froxel, so a
        // jittered point made the forward-peaked phase differ tenfold between
        // frames: shimmer in every lamp's glow. Shadow visibility still uses
        // the jittered p, which antialiases shadow edges over time.
        var q = p;
        if sample_t1 > sample_t0 {
            let origin = cameras[0].position_near.xyz;
            let t = clamp(dot(light.position_range.xyz - origin, sample_center_dir), sample_t0, sample_t1);
            q = origin + sample_center_dir * t;
            view_dir = sample_center_dir;
        }
        let delta = light.position_range.xyz - q;
        let dist = length(delta);
        shadow_distance = length(light.position_range.xyz - p);
        let range = max(light.position_range.w, 1e-4);
        if dist > range { return vec3<f32>(0.0); }
        to_light = delta / max(dist, 1e-6);

        // Inverse-square with a windowed cutoff, so the contribution reaches zero
        // exactly at the range boundary instead of popping. Inside cs_inject the
        // froxel stores the segment mean of 1/r^2 (see segment_inverse_square).
        let window = clamp(1.0 - pow(dist / range, 4.0), 0.0, 1.0);
        var inverse_square = 1.0 / max(dist * dist, 1e-4);
        if sample_t1 > sample_t0 {
            inverse_square = segment_inverse_square(light.position_range.xyz);
        }
        atten = window * window * inverse_square;

        if light.light_type == LIGHT_SPOT {
            if dot(light.direction_outer.xyz, light.direction_outer.xyz) < 1e-12 { return vec3<f32>(0); }
            let cd = dot(-to_light, normalize(light.direction_outer.xyz));
            let outer = light.direction_outer.w;
            let inner = light.inner_angle;
            let spot = clamp((cd - outer) / max(inner - outer, 1e-4), 0.0, 1.0);
            atten *= spot * spot;
        }
    }

    // cos = 1 looking straight at the light, so g > 0 peaks into the sun.
    let phase = helio_hg_phase(dot(view_dir, to_light), clamp(anisotropy, -0.95, 0.95));
    // ABI adapter: decay was unused; it now controls geometric fog-shadow
    // strength. Medium absorption remains physical even with geometry opt-out.
    let shadow_strength = finite_clamp(light.god_rays_decay, 0.0, 1.0);
    var vis = vec3<f32>(1.0);
    if shadow_strength > 0.0 { vis = mix(vec3<f32>(1.0), shaft_visibility(light_idx, p), shadow_strength); }
    if all(vis <= vec3<f32>(0.0)) || atten <= 0.0 { return vec3<f32>(0); }
    vis *= medium_transmittance(p, to_light, shadow_distance);

    let radiance = light.color_intensity.rgb * light.color_intensity.w;
    return radiance * atten * phase * vis
        * light.god_rays_weight * light.god_rays_exposure * light.god_rays_density;
}

// ── Temporal reprojection ───────────────────────────────────────────────────

/// Previous frame's scattering at world position `p`, or `none` if it reprojects
/// off-grid.
///
/// Returns w < 0 to signal "no history" — a froxel that was off-screen or behind
/// the camera last frame has nothing to blend with, and reusing a clamped edge
/// sample there smears fog across the screen edges as the camera turns.
fn sample_history(p: vec3<f32>) -> vec4<f32> {
    let prev_clip = cameras[0].prev_view_proj * vec4<f32>(p, 1.0);
    // For the engine's perspective matrix, clip.w is the positive view depth.
    if prev_clip.w <= HELIO_FROXEL_NEAR { return vec4<f32>(0.0, 0.0, 0.0, -1.0); }

    // prev_view_proj is last frame's jittered matrix: remove its jitter to land
    // in last frame's (unjittered) grid.
    let prev_ndc = prev_clip.xyz / prev_clip.w;
    let prev_uv = helio_ndc_to_uv(prev_ndc.xy - vec2<f32>(fog_globals.prev_jitter_x, fog_globals.prev_jitter_y));
    if any(prev_uv < vec2<f32>(0.0)) || any(prev_uv > vec2<f32>(1.0)) {
        return vec4<f32>(0.0, 0.0, 0.0, -1.0);
    }

    let prev_slice = helio_froxel_slice_from_view_depth(prev_clip.w, fog.fog_max_distance);
    if prev_slice < 0.0 || prev_slice > 1.0 {
        return vec4<f32>(0.0, 0.0, 0.0, -1.0);
    }

    return textureSampleLevel(scatter_history, linear_samp,
        scatter_coordinates(vec3<f32>(prev_uv, prev_slice), textureDimensions(scatter_history)), 0.0);
}

fn scatter_coordinates(uvw: vec3<f32>, physical: vec3<u32>) -> vec3<f32> {
    let logical = vec3<f32>(media_list.grid);
    return clamp(uvw * logical, vec3<f32>(0.5), logical - 0.5) / vec3<f32>(physical);
}

fn temporal_result(current: vec4<f32>, history: vec4<f32>, blend: f32, rejection: f32) -> vec4<f32> {
    // Never resurrect removed media, vanished lights, or residual emission.
    if all(current == vec4<f32>(0)) || all(current.rgb == vec3<f32>(0)) { return current; }
    // Only a medium appearing or vanishing rejects history. Lighting is never
    // compared: each sample is jittered inside its froxel, so across a shaft
    // edge successive samples legitimately differ by 100%, and rejecting on
    // that is exactly what left the raw per-frame noise on screen.
    let density_change = abs(history.a - current.a) / max(max(history.a, current.a), 1e-8);
    let weight = max(blend, smoothstep(rejection, min(rejection + 0.15, 1.0), density_change));
    return mix(history, current, clamp(weight, 0.0, 1.0));
}

// ── Injection ───────────────────────────────────────────────────────────────

/// Interleaved-gradient noise, used to jitter the sample point within its froxel.
/// Combined with the temporal blend this turns slice banding into noise that
/// averages out across frames.
fn ign(pixel: vec2<f32>, frame: u32) -> f32 {
    let f = pixel + 5.588238 * f32(frame % 64u);
    return fract(52.9829189 * fract(dot(f, vec2<f32>(0.06711056, 0.00583715))));
}

@compute @workgroup_size(8, 8, 1)
fn cs_inject(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = media_list.grid;
    if any(gid >= dims) { return; }

    let coord = vec3<i32>(gid);

    if media_list.has_medium == 0u {
        textureStore(scatter_out, coord, vec4<f32>(0.0));
        return;
    }

    // Jitter within the froxel in all three axes, varying per frame: the
    // temporal blend then integrates the whole cell (shadow edges, lancet
    // patterns, smoke detail) instead of point-sampling its centre, which
    // aliases and crawls whenever the camera moves. Depth uses IGN; X/Y use
    // the R2 sequence with an independent per-cell (Cranley-Patterson) offset.
    // A shared offset moves every cell in lockstep, so the residual error forms
    // a coherent pattern that crawls across the medium each frame (shimmer);
    // decorrelated, it is fine grain the reconstruction and TSR filter away.
    let j = ign(vec2<f32>(gid.xy), fog_globals.frame);
    let cell = hash_word(hash_word(hash_word(2166136261u, gid.x), gid.y), gid.z);
    let rotation = vec2<f32>(f32(cell & 0xffffu), f32(cell >> 16u)) / 65536.0;
    let jxy = fract(vec2<f32>(0.7548776662, 0.5698402910) * f32(fog_globals.frame % 4096u) + rotation);
    let uv = (vec2<f32>(gid.xy) + jxy) / vec2<f32>(dims.xy);
    let slice_norm = (f32(gid.z) + j) / f32(dims.z);

    let p = froxel_world_pos(uv, slice_norm);
    let ray_dir = normalize(p - cameras[0].position_near.xyz);

    let view_depth = helio_froxel_view_depth_from_slice(slice_norm, fog.fog_max_distance);
    let medium = medium_at(p, view_depth);
    // This froxel's ray segment (radial = view depth / cos) and half-width.
    let radial_scale = distance(p, cameras[0].position_near.xyz) / max(view_depth, 1e-6);
    sample_t0 = helio_froxel_view_depth_from_slice(f32(gid.z) / f32(dims.z), fog.fog_max_distance) * radial_scale;
    sample_t1 = helio_froxel_view_depth_from_slice(f32(gid.z + 1u) / f32(dims.z), fog.fog_max_distance) * radial_scale;
    sample_lateral = 0.5 * distance(froxel_world_pos(uv + vec2<f32>(1.0 / f32(dims.x), 0.0), slice_norm), p);
    sample_center_dir = normalize(froxel_world_pos((vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims.xy), slice_norm)
        - cameras[0].position_near.xyz);
    let density = medium.extinction;
    // Integrate the complete medium. The full-resolution composite stops at
    // each pixel's surface depth; coarse depth rejection leaks at silhouettes.

    var scattering = vec3<f32>(0.0);
    if density > 0.0 {
        var radiance = vec3<f32>(0.0);
        let cluster = cluster_index(gid / vec3<u32>(8, 8, 4));
        let count = clusters[cluster].count;
        let overflow = count > 64u;
        let iterations = select(count, light_count(), overflow);
        for (var li = 0u; li < iterations; li++) {
            var index: u32;
            if overflow { index = light_index(li); } else { index = clusters[cluster].lights[li]; }
            radiance += inscatter_from_light(index, p, ray_dir, medium.anisotropy);
        }
        // With no actual illumination, scattering is zero. Sky/indirect
        // illumination needs an explicit radiance source, never a constant.
        scattering = radiance * medium.albedo * density;
    }
    scattering += medium.emissive;

    var result = vec4<f32>(scattering, density);

    if fog_globals.history_valid != 0u && media_list.history_compatible != 0u {
        let history_pos = froxel_world_pos(uv, (f32(gid.z) + 0.5) / f32(dims.z));
        let hist = sample_history(history_pos);
        if hist.w >= 0.0 {
            // Reject history at moving smoke edges and after density changes.
            // Stable interiors retain the low-noise lighting history.
            result = temporal_result(result, hist, media_list.temporal_blend, media_list.history_rejection);
        }
    }

    textureStore(scatter_out, coord, clamp(result, vec4<f32>(0), vec4<f32>(65504)));
}

// Stable (1-exp(-tau))/sigma_t, including the exact vacuum limit. The series
// avoids cancellation for thin media without suppressing their scattering.
fn segment_integral(sigma_t: f32, length: f32) -> f32 {
    let tau = sigma_t * length;
    if tau < 0.01 { return length * (1.0 - tau * 0.5 + tau * tau / 6.0 - tau * tau * tau / 24.0); }
    return (1.0 - exp(-tau)) / sigma_t;
}

// ── Integration ─────────────────────────────────────────────────────────────

@compute @workgroup_size(8, 8, 1)
fn cs_integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(integrated_out);
    if gid.x >= dims.x || gid.y >= dims.y { return; }
    if media_list.has_medium == 0u {
        for (var z = 0u; z < dims.z; z++) {
            textureStore(integrated_out, vec3<i32>(vec2<i32>(gid.xy), i32(z)), vec4<f32>(0, 0, 0, 1));
        }
        return;
    }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims.xy);

    // Slice planes are constant view depth, so a ray at the screen edge travels
    // further between two slices than one down the centre. Without this the fog
    // thins toward the corners.
    let ndc = helio_uv_to_ndc(uv) + cameras[0].jitter_frame.xy;
    let p_near = cameras[0].view_proj_inv * vec4<f32>(ndc, 0.0, 1.0);
    let p_far  = cameras[0].view_proj_inv * vec4<f32>(ndc, 1.0, 1.0);
    let dir = normalize(p_far.xyz / p_far.w - p_near.xyz / p_near.w);
    let cos_a = max(dot(dir, normalize(cameras[0].forward_far.xyz)), 1e-4);

    var accum = vec3<f32>(0.0);
    var transmittance = 1.0;
    var prev_depth = HELIO_FROXEL_NEAR;

    for (var z = 0u; z < dims.z; z++) {
        let slice_norm = (f32(z) + 1.0) / f32(dims.z);
        let depth = helio_froxel_view_depth_from_slice(slice_norm, fog.fog_max_distance);
        let step_len = max(depth - prev_depth, 0.0) / cos_a;
        prev_depth = depth;

        let uvw = vec3<f32>(uv, (f32(z) + 0.5) / f32(dims.z));
        let s = textureSampleLevel(scatter_in, linear_samp, scatter_coordinates(uvw, textureDimensions(scatter_in)), 0.0);
        let scattering = s.rgb;
        let sigma_t = max(s.a, 0.0);

        let step_transmittance = exp(-sigma_t * step_len);

        // Hillaire's analytic slice integral: the exact integral of scattering
        // over a segment of constant density, rather than a point sample scaled
        // by step length. Point-sampling over-brightens as density rises, because
        // it has no saturation term.
        let s_int = scattering * segment_integral(sigma_t, step_len);

        accum += transmittance * s_int;
        transmittance *= step_transmittance;

        textureStore(
            integrated_out,
            vec3<i32>(i32(gid.x), i32(gid.y), i32(z)),
            vec4<f32>(min(accum, vec3<f32>(65504)), transmittance),
        );
    }
}
