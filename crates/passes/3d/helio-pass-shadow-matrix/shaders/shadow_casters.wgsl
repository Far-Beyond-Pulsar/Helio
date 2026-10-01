/// GPU shadow-caster allocation (Helio#246, view-aware budget selection).
///
/// Authors request a shadow map by writing any `shadow_index` other than
/// `u32::MAX` (0 by convention). This kernel turns requests into atlas slots:
/// it ranks requesting lights by projected coverage, distance attenuation,
/// and a small incumbent-slot hysteresis bonus, keeps the top
/// `caster_capacity`, and writes
/// `shadow_index = 6 * slot` back into the same `"scene_lights"` rows every
/// pass already reads. Losers get `u32::MAX`. Slots follow row order, so a
/// winning set that does not change keeps its slots.
///
/// The request survives the overwrite in `_pad`: `ALLOCATED` marks a row this
/// kernel has rewritten and `WANTS_SHADOW_MAP` holds the original request.
/// SceneDB re-uploads a row whenever it is edited, which clears both bits, so
/// the next run reads the author's value again. The pass dispatches this only
/// when the light rows change; between edits the written slots stay valid.
///
/// One workgroup. Each thread owns a contiguous chunk of rows, so prefix sums
/// over chunk counts give row-ordered results: a four-digit radix select finds
/// the `caster_capacity`-th largest key, ties go to the lowest rows, and slots
/// are assigned to winners in row order.

struct GpuLight {
    position_range:   vec4f,
    direction_outer:  vec4f,
    color_intensity:  vec4f,
    shadow_index:     u32,
    light_type:       u32,
    inner_angle:      f32,
    _pad:             u32,
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

struct CasterParams {
    row_count: u32,
    caster_capacity: u32,
    nonce: u32,
    _pad1: u32,
    view_proj: mat4x4<f32>,
    camera_position: vec4f,
}

@group(0) @binding(0) var<storage, read_write> lights: array<GpuLight>;
@group(0) @binding(1) var<uniform> params: CasterParams;
/// The allocation for the CPU: `[0]` = casters assigned, `[1 + slot]` =
/// that slot's `light_type`, last word = `params.nonce`. Read back by ShadowMatrixPass so ShadowPass
/// renders only the faces each caster uses.
@group(0) @binding(2) var<storage, read_write> caster_table: array<u32>;

const THREADS: u32 = 256u;
const NO_SHADOW: u32 = 0xFFFFFFFFu;
const FACES_PER_CASTER: u32 = 6u;
const LIGHT_TYPE_DIRECTIONAL: u32 = 0u;
// `_pad` bits. 0-1 are GpuLight::set_ray_traced_shadows's explicit intent.
const RT_EXPLICIT: u32 = 1u;
const RT_ENABLED: u32 = 2u;
const WANTS_SHADOW_MAP: u32 = 4u;
const ALLOCATED: u32 = 8u;

var<workgroup> histogram: array<atomic<u32>, 256>;
var<workgroup> offsets: array<u32, 256>;
var<workgroup> threshold: u32;
var<workgroup> threshold_mask: u32;
var<workgroup> needed: u32;
var<workgroup> take_all: u32;

fn is_live(light: GpuLight) -> bool {
    // Matches ShadowMatrixPass: vacant SceneDB rows are zeroed.
    return light.color_intensity.w > 0.0;
}

fn requests_shadow_map(light: GpuLight) -> bool {
    if (light._pad & ALLOCATED) != 0u { return (light._pad & WANTS_SHADOW_MAP) != 0u; }
    return light.shadow_index != NO_SHADOW;
}

fn projected_score(light: GpuLight) -> f32 {
    if light.light_type == LIGHT_TYPE_DIRECTIONAL {
        return 1.0e20;
    }
    let radius = max(light.position_range.w, 0.0);
    let delta = light.position_range.xyz - params.camera_position.xyz;
    let distance_sq = max(dot(delta, delta), 1.0);
    let clip = params.view_proj * vec4f(light.position_range.xyz, 1.0);
    // A sphere bound is conservative: lights whose influence does not touch
    // the view are excluded, while intersecting edge lights remain eligible.
    let clip_radius = radius * max(abs(params.view_proj[0][0]), abs(params.view_proj[1][1]));
    if clip.w + clip_radius <= 0.0 { return 0.0; }
    if abs(clip.x) > clip.w + clip_radius || abs(clip.y) > clip.w + clip_radius { return 0.0; }
    // Projected area approximates screen coverage. Radiometric falloff keeps
    // nearby useful lights ahead of equally sized lights at range.
    let projected_radius = radius / max(abs(clip.w), 0.01);
    let coverage = min(projected_radius * projected_radius, 4.0);
    let falloff = 1.0 / (1.0 + distance_sq);
    let score = max(light.color_intensity.w, 0.0) * coverage * falloff;
    // Retaining an existing allocation within a narrow score margin prevents
    // slot churn when two lights have nearly equal importance.
    let incumbent = (light._pad & ALLOCATED) != 0u && light.shadow_index != NO_SHADOW;
    return score * select(1.0, 1.15, incumbent);
}

fn is_candidate(light: GpuLight) -> bool {
    return is_live(light) && requests_shadow_map(light) && projected_score(light) > 0.0;
}

/// Larger is more important. Non-negative floats order like their bits.
fn importance_key(light: GpuLight) -> u32 {
    let score = projected_score(light);
    // NaN compares false: it ranks last instead of poisoning the order.
    return bitcast<u32>(select(0.0, min(score, 3.0e38), score > 0.0));
}

/// Workgroup-wide exclusive prefix sum of `offsets`; returns this thread's
/// offset. Every thread writes its own count to `offsets[lid]` first.
fn exclusive_scan(lid: u32) -> u32 {
    workgroupBarrier();
    if lid == 0u {
        var total = 0u;
        for (var j = 0u; j < THREADS; j++) {
            let count = offsets[j];
            offsets[j] = total;
            total += count;
        }
    }
    workgroupBarrier();
    return offsets[lid];
}

@compute @workgroup_size(256)
fn assign_shadow_casters(@builtin(local_invocation_index) lid: u32) {
    let n = min(params.row_count, arrayLength(&lights));
    let chunk = (n + THREADS - 1u) / THREADS;
    let begin = min(lid * chunk, n);
    let end = min(begin + chunk, n);

    if lid == 0u {
        threshold = 0u;
        threshold_mask = 0u;
        needed = params.caster_capacity;
        take_all = 0u;
    }

    // ── Radix select: the caster_capacity-th largest key, one byte at a time.
    for (var digit = 0u; digit < 4u; digit++) {
        let shift = 24u - 8u * digit;
        atomicStore(&histogram[lid], 0u);
        workgroupBarrier();
        let prefix = threshold;
        let mask = threshold_mask;
        for (var i = begin; i < end; i++) {
            let light = lights[i];
            if !is_candidate(light) { continue; }
            let key = importance_key(light);
            if (key & mask) == prefix {
                atomicAdd(&histogram[(key >> shift) & 0xFFu], 1u);
            }
        }
        workgroupBarrier();
        if lid == 0u && take_all == 0u {
            if digit == 0u {
                var total = 0u;
                for (var b = 0u; b < 256u; b++) { total += atomicLoad(&histogram[b]); }
                // Every request fits: no selection needed.
                if total <= needed { take_all = 1u; }
            }
            if take_all == 0u {
                // Walk from the largest byte down to the one holding the
                // needed-th key; everything in larger bytes already wins.
                var above = 0u;
                var bucket = 255u;
                loop {
                    let count = atomicLoad(&histogram[bucket]);
                    if above + count >= needed || bucket == 0u { break; }
                    above += count;
                    bucket -= 1u;
                }
                needed -= above;
                threshold |= bucket << shift;
                threshold_mask |= 0xFFu << shift;
            }
        }
        workgroupBarrier();
    }
    let all = take_all != 0u;
    let key_threshold = threshold;
    let ties_needed = needed;

    // ── Ties at the threshold go to the lowest rows.
    var ties = 0u;
    for (var i = begin; i < end; i++) {
        let light = lights[i];
        if is_candidate(light) && importance_key(light) == key_threshold { ties++; }
    }
    offsets[lid] = ties;
    let tie_base = exclusive_scan(lid);

    // ── Winners in row order take consecutive slots.
    var winners = 0u;
    var tie_rank = tie_base;
    for (var i = begin; i < end; i++) {
        let light = lights[i];
        if !is_candidate(light) { continue; }
        let key = importance_key(light);
        var wins = all || key > key_threshold;
        if !wins && key == key_threshold {
            wins = tie_rank < ties_needed;
            tie_rank++;
        }
        if wins { winners++; }
    }
    workgroupBarrier();
    offsets[lid] = winners;
    var slot = exclusive_scan(lid);

    tie_rank = tie_base;
    for (var i = begin; i < end; i++) {
        let light = lights[i];
        if !is_live(light) { continue; }
        let requested = requests_shadow_map(light);
        var wins = false;
        if requested {
            let key = importance_key(light);
            wins = all || key > key_threshold;
            if !wins && key == key_threshold {
                wins = tie_rank < ties_needed;
                tie_rank++;
            }
        }
        var pad = (light._pad & ~WANTS_SHADOW_MAP) | ALLOCATED | select(0u, WANTS_SHADOW_MAP, requested);
        // Ray-traced shadows used to follow `shadow_index != u32::MAX` when no
        // explicit intent was set. Pin that intent before overwriting the slot.
        if (light._pad & RT_EXPLICIT) == 0u {
            pad |= RT_EXPLICIT | select(0u, RT_ENABLED, requested);
        }
        lights[i]._pad = pad;
        if wins {
            lights[i].shadow_index = slot * FACES_PER_CASTER;
            if 1u + slot < arrayLength(&caster_table) - 1u {
                caster_table[1u + slot] = light.light_type;
            }
            slot++;
        } else {
            lights[i].shadow_index = NO_SHADOW;
        }
    }
    // Slots are handed out in thread order, so the last thread ends on the
    // total.
    if lid == THREADS - 1u {
        caster_table[0] = slot;
        caster_table[arrayLength(&caster_table) - 1u] = params.nonce;
    }
}
