// Directional sky bound for rays that start at the eye.
//
// Every point of a ray from the eye lies in one plane through the planet
// centre, so seen from the eye it keeps a single azimuth, and its angular
// distance from the eye grows monotonically. Each frame the resident
// summary blocks are binned by azimuth sector and by the farthest angular
// distance they reach; a suffix maximum over distance then bounds every
// terrain cell such a ray can still meet. A rising ray above that bound
// leaves the planet without another hit.

const SECTORS: u32 = 256u;
const BUCKETS: u32 = 32u;
const HORIZON_NONE: i32 = -0x3fffffff;
const TAU: f32 = 6.283185307;

// Accumulated maxima: [bucket][sector], then one all-sector row per bucket.
@group(0) @binding(17) var<storage, read_write> horizon_acc: array<atomic<i32>>;
// Suffix maxima over distance, dilated by one sector, same layout.
@group(0) @binding(18) var<storage, read_write> horizon: array<i32>;

const FACE_N: array<vec3<f32>, 6> = array<vec3<f32>, 6>(
    vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(-1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0),
    vec3<f32>(0.0, -1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 0.0, -1.0));
const FACE_A: array<vec3<f32>, 6> = array<vec3<f32>, 6>(
    vec3<f32>(0.0, 0.0, -1.0), vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0),
    vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(-1.0, 0.0, 0.0));
const FACE_B: array<vec3<f32>, 6> = array<vec3<f32>, 6>(
    vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, -1.0),
    vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 1.0, 0.0));

// Tangent basis at the eye (identical for building and querying).
fn eye_t1() -> vec3<f32> {
    let u = frame.eye.xyz;
    let a = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(u.x) > 0.7);
    return normalize(cross(u, a));
}

fn sector_of(azimuth: f32) -> i32 {
    return clamp(i32(floor((azimuth + 3.14159265) * f32(SECTORS) / TAU)), 0, i32(SECTORS) - 1);
}

fn angle_between(a: vec3<f32>, b: vec3<f32>) -> f32 {
    return atan2(length(cross(a, b)), dot(a, b));
}

fn phi_bucket(phi: f32) -> u32 {
    let phi0 = frame.lod.x / frame.eye.w * 0.5;
    if phi < phi0 { return 0u; }
    return min(u32(floor(log2(phi / phi0))) + 1u, BUCKETS - 1u);
}

fn face_dir(face: u32, i: f32, j: f32) -> vec3<f32> {
    let a = -0.785398163 + i * frame.layer.z;
    let b = -0.785398163 + j * frame.layer.z;
    return normalize(FACE_N[face] + FACE_A[face] * tan(a) + FACE_B[face] * tan(b));
}

@compute @workgroup_size(64)
fn horizon_clear(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < (SECTORS + 1u) * BUCKETS {
        atomicStore(&horizon_acc[id.x], HORIZON_NONE);
    }
}

// One thread per tier-1 summary block (4 x 4 columns) of every level and
// face: small enough that the coarse blocks around the eye fall inside their
// level's ring and drop out.
@compute @workgroup_size(64)
fn horizon_blocks(@builtin(global_invocation_id) id: vec3<u32>) {
    let levels = u32(frame.layer_i.z);
    if id.x >= (1u << 14u) || id.y >= levels * 6u { return; }
    let level = id.y / 6u;
    let face = id.y % 6u;
    let slot = ((level * 6u + face) * frame.extra.y + id.x) * 4u;
    if block_state[slot + 3u] <= 0 { return; }
    let bi = block_state[slot];
    let bj = block_state[slot + 1u];
    let top = block_state[slot + 2u] << level;
    // Base index footprint of the block (32 level cells per side).
    let span = f32(32 << level);
    let i0 = f32(bi) * span;
    let j0 = f32(bj) * span;
    let c = face_dir(face, i0 + span * 0.5, j0 + span * 0.5);
    // Cell boundaries are great circles, so the corners bound the block's cap.
    var rho = angle_between(c, face_dir(face, i0, j0));
    rho = max(rho, angle_between(c, face_dir(face, i0 + span, j0)));
    rho = max(rho, angle_between(c, face_dir(face, i0, j0 + span)));
    rho = max(rho, angle_between(c, face_dir(face, i0 + span, j0 + span)));
    rho = rho * 1.001 + 2e-6;
    let u = frame.eye.xyz;
    let theta = angle_between(u, c);
    let reach = (theta + rho) * 1.001 + 1e-6;
    // Rays can only use this level beyond its ring: nearer blocks never
    // serve a sky-bounded ray.
    if reach < frame.ring[level >> 2u][level & 3u] { return; }
    let bucket = phi_bucket(reach);
    atomicMax(&horizon_acc[SECTORS * BUCKETS + bucket], top);
    var lo = 0;
    var count = i32(SECTORS);
    if theta > rho + 2e-5 && theta + rho < 3.1 {
        let half = asin(min(sin(rho) / sin(theta), 1.0));
        if half < 1.2 {
            let t1 = eye_t1();
            let t2 = cross(u, t1);
            // Centre sector +- the cap's half width (+1 for rounding); the
            // modulo below wraps ranges across -pi / pi.
            let w = i32(ceil(half * f32(SECTORS) / TAU)) + 1;
            lo = sector_of(atan2(dot(c, t2), dot(c, t1))) - w;
            count = min(2 * w + 1, i32(SECTORS));
        }
    }
    for (var s = 0; s < count; s++) {
        let sector = u32((lo + s + i32(SECTORS) * 2) % i32(SECTORS));
        atomicMax(&horizon_acc[bucket * SECTORS + sector], top);
    }
}

// Suffix maximum over distance buckets, dilated by one sector per side.
@compute @workgroup_size(64)
fn horizon_suffix(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x > SECTORS { return; }
    var running = HORIZON_NONE;
    for (var b = i32(BUCKETS) - 1; b >= 0; b--) {
        let row = u32(b) * SECTORS;
        if id.x == SECTORS {
            running = max(running, atomicLoad(&horizon_acc[SECTORS * BUCKETS + u32(b)]));
            horizon[SECTORS * BUCKETS + u32(b)] = running;
        } else {
            let s = id.x;
            let prev = (s + SECTORS - 1u) % SECTORS;
            let next = (s + 1u) % SECTORS;
            running = max(running, max(atomicLoad(&horizon_acc[row + s]),
                max(atomicLoad(&horizon_acc[row + prev]), atomicLoad(&horizon_acc[row + next]))));
            horizon[row + s] = running;
        }
    }
}

// Sector of an eye ray (-1: use the all-sector bound) and its sin/cos of
// elevation terms for the angular distance.
struct SkyRay {
    sector: i32,
    lt: f32,   // |tangential part of l|
    lu: f32,   // l · eye direction
}

fn sky_ray(l: vec3<f32>, spread: f32) -> SkyRay {
    let u = frame.eye.xyz;
    var s: SkyRay;
    s.lu = dot(l, u);
    let lt = l - s.lu * u;
    s.lt = length(lt);
    s.sector = -1;
    // The dilated table covers azimuth errors up to one sector width.
    if s.lt > 1e-5 && (spread + 2e-6) / s.lt < TAU / f32(SECTORS) {
        let t1 = eye_t1();
        s.sector = sector_of(atan2(dot(lt, cross(u, t1)), dot(lt, t1)));
    }
    return s;
}

fn no_sky() -> SkyRay {
    var s: SkyRay;
    s.sector = -2;
    return s;
}

// Sky bound for a camera ray; the camera must sit at the eye (world origin).
fn eye_sky(l: vec3<f32>, spread: f32) -> SkyRay {
    if (u32(frame.screen.w) & 2u) == 0u || dot(camera.position_near.xyz, camera.position_near.xyz) > 1e-6 { return no_sky(); }
    return sky_ray(l, spread);
}

// Highest base layer any terrain reachable after angular distance `phi` can
// occupy along this ray's azimuth.
fn horizon_layer(s: SkyRay, phi: f32) -> i32 {
    let b = phi_bucket(phi * 0.999);
    if s.sector < 0 { return horizon[SECTORS * BUCKETS + b]; }
    return horizon[b * SECTORS + u32(s.sector)];
}

// Angular distance from the eye of the eye-ray point at `t`.
fn eye_phi(s: SkyRay, t: f32) -> f32 {
    return atan2(t * s.lt, frame.eye.w + t * s.lu);
}
