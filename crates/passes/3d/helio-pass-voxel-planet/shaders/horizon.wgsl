// Directional sky bound for rays that start at the eye.
//
// On a plane the same construction uses horizontal distance from the eye in
// place of angular distance (metres instead of radians).
//
// Every point of a ray from the eye lies in one plane through the planet
// centre, so seen from the eye it keeps a single azimuth, and its angular
// distance from the eye grows monotonically. Each frame the resident
// summary blocks are binned by azimuth sector and by the range of angular
// distances they cover. Per bucket, that bounds every terrain cell a ray
// can meet there. A ray's height at any angular distance grows with its
// elevation, so each bucket stores the lowest elevation that clears it; a
// primary ray starts at the nearest bucket it does not clear and ends after
// the farthest one (`sky_span`).

const SECTORS: u32 = 256u;
const BUCKETS: u32 = 32u;
// Coarse sector groups: blocks spanning many sectors write whole groups.
const GROUPS: u32 = 16u;
const GROUP_SECTORS: i32 = 16;
const HORIZON_NONE: i32 = -0x3fffffff;
const TAU: f32 = 6.283185307;

// Accumulated maxima: [bucket][sector], then [bucket][group].
@group(0) @binding(17) var<storage, read_write> horizon_acc: array<atomic<i32>>;
// Clearing elevations (f32 bits), dilated by one sector: [bucket][sector],
// then one all-sector row.
@group(0) @binding(18) var<storage, read_write> horizon: array<f32>;
// Live tier-1 summary block slots (maintained by the CPU residency).
@group(0) @binding(19) var<storage, read> live_blocks: array<u32>;
const H_ALL: u32 = SECTORS * BUCKETS;

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

fn phi0() -> f32 {
    if is_plane() { return frame.lod.x * 0.5; }
    return frame.lod.x / frame.eye.w * 0.5;
}

// Bucket 0 is [0, phi0); bucket b > 0 is [phi0 2^(b-1), phi0 2^b).
fn phi_bucket(phi: f32) -> u32 {
    if phi < phi0() { return 0u; }
    return min(u32(floor(log2(phi / phi0()))) + 1u, BUCKETS - 1u);
}

fn bucket_start(b: u32) -> f32 {
    if b == 0u { return 0.0; }
    return phi0() * exp2(f32(b - 1u));
}

fn face_dir(face: u32, i: f32, j: f32) -> vec3<f32> {
    let a = -0.785398163 + i * frame.layer.z;
    let b = -0.785398163 + j * frame.layer.z;
    return normalize(FACE_N[face] + FACE_A[face] * tan(a) + FACE_B[face] * tan(b));
}

@compute @workgroup_size(64)
fn horizon_clear(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < (SECTORS + GROUPS) * BUCKETS {
        atomicStore(&horizon_acc[id.x], HORIZON_NONE);
    }
}

// One thread per live tier-1 summary block (4 x 4 columns): small enough
// that the coarse blocks around the eye fall inside their level's ring and
// drop out.
@compute @workgroup_size(64)
fn horizon_blocks(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= frame.extra.w { return; }
    let entry = live_blocks[id.x];
    let region = entry / frame.extra.y;
    let level = region / 6u;
    let face = region % 6u;
    let e = block_state[entry];
    if e.w <= 0 { return; }
    let bi = e.x;
    let bj = e.y;
    let top = e.z << level;
    if is_plane() {
        plane_block(e, level);
        return;
    }
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
    // serve a sky-bounded ray, and nearer parts of a block neither.
    let ring = frame.ring[level >> 2u][level & 3u];
    if reach < ring { return; }
    let b_lo = phi_bucket(max(theta - rho, ring) * 0.999 - 1e-6);
    let b_hi = phi_bucket(reach);
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
    // Narrow spans write sectors, wide ones the groups that cover them.
    var base = 0u;
    var stride = SECTORS;
    var modulus = i32(SECTORS);
    if count > 2 * GROUP_SECTORS {
        let first = (lo + i32(SECTORS) * 2) / GROUP_SECTORS;
        let last = (lo + count - 1 + i32(SECTORS) * 2) / GROUP_SECTORS;
        lo = first;
        count = min(last - first + 1, i32(GROUPS));
        base = SECTORS * BUCKETS;
        stride = GROUPS;
        modulus = i32(GROUPS);
    }
    for (var s = 0; s < count; s++) {
        let cell = u32((lo + s + modulus * 4) % modulus);
        for (var b = b_lo; b <= b_hi; b++) {
            let at = base + b * stride + cell;
            // Most cells already hold a higher top: skip the contended write.
            if atomicLoad(&horizon_acc[at]) < top {
                atomicMax(&horizon_acc[at], top);
            }
        }
    }
}

// Record a plane summary block: horizontal distance range and azimuth span
// of its footprint around the eye.
fn plane_block(e: vec4<i32>, level: u32) {
    let f = frame.faces[PLANE_FACE];
    let span = f32(32 << level);
    // Block centre relative to the eye, in metres along the face axes.
    let du = (f32(e.x * (32 << level) - f.index.x) - f.q_a.w + span * 0.5) * frame.layer.z;
    let dv = (f32(e.y * (32 << level) - f.index.y) - f.q_b.w + span * 0.5) * frame.layer.z;
    let w = f.m_a.xyz * du + f.m_b.xyz * dv;
    let theta = length(vec2<f32>(du, dv));
    let rho = span * frame.layer.z * 0.7072 + 1e-3;
    let reach = theta + rho;
    let ring = frame.ring[level >> 2u][level & 3u];
    if reach < ring { return; }
    let b_lo = phi_bucket(max(theta - rho, ring) * 0.999 - 1e-3);
    let b_hi = phi_bucket(reach * 1.001);
    var lo = 0;
    var count = i32(SECTORS);
    if theta > rho * 1.0001 {
        let half = asin(min(rho / theta, 1.0));
        if half < 1.2 {
            let t1 = eye_t1();
            let t2 = cross(frame.eye.xyz, t1);
            let wdth = i32(ceil(half * f32(SECTORS) / TAU)) + 1;
            lo = sector_of(atan2(dot(w, t2), dot(w, t1))) - wdth;
            count = min(2 * wdth + 1, i32(SECTORS));
        }
    }
    record_block(e.z << level, lo, count, b_lo, b_hi);
}

// Accumulate a block's top into its sectors (or sector groups) and buckets.
fn record_block(top: i32, lo_in: i32, count_in: i32, b_lo: u32, b_hi: u32) {
    var lo = lo_in;
    var count = count_in;
    var base = 0u;
    var stride = SECTORS;
    var modulus = i32(SECTORS);
    if count > 2 * GROUP_SECTORS {
        let first = (lo + i32(SECTORS) * 2) / GROUP_SECTORS;
        let last = (lo + count - 1 + i32(SECTORS) * 2) / GROUP_SECTORS;
        lo = first;
        count = min(last - first + 1, i32(GROUPS));
        base = SECTORS * BUCKETS;
        stride = GROUPS;
        modulus = i32(GROUPS);
    }
    for (var s = 0; s < count; s++) {
        let cell = u32((lo + s + modulus * 4) % modulus);
        for (var b = b_lo; b <= b_hi; b++) {
            let at = base + b * stride + cell;
            if atomicLoad(&horizon_acc[at]) < top {
                atomicMax(&horizon_acc[at], top);
            }
        }
    }
}

// Lowest eye-ray elevation whose every point in angular distances [pa, pb]
// is higher than `top` (and the cut height). Passing above height H at
// angle phi needs tan e > (k cos phi - 1) / (k sin phi), k = 1 + H / rho;
// that bound falls with phi once cos phi < k, so the maximum over the
// bucket is at pa, or at acos k when the top is below the eye.
fn clearing_elevation(top: i32, b: u32) -> f32 {
    let h = max(layer_height(top), frame.lod.z);
    if is_plane() {
        // A ray at elevation e is tan(e) d above the eye at distance d: it
        // must clear h at the bucket's near edge (h above the eye) or its
        // far edge (h below).
        let pa = bucket_start(b);
        if h >= 0.0 {
            if pa <= 0.0 { return 1.5707964; }
            return atan2(h, pa) + 2e-6;
        }
        if b + 1u >= BUCKETS { return 2e-6; }
        return atan2(h, bucket_start(b + 1u)) + 2e-6;
    }
    let rho = frame.eye.w;
    let k = 1.0 + h / rho;
    let pa = bucket_start(b);
    var pb = 3.14159265;
    if b + 1u < BUCKETS { pb = bucket_start(b + 1u); }
    var phi = pa;
    if h < 0.0 { phi = clamp(2.0 * asin(sqrt(-h / (2.0 * rho))), pa, pb); }
    if phi <= 0.0 { return 1.5707964; }
    let s = sin(phi * 0.5);
    return atan2(h / rho - 2.0 * k * s * s, k * sin(phi)) + 2e-6;
}

var<workgroup> all_sectors: array<atomic<i32>, 32>;

// Clearing elevations per distance bucket, dilated by one sector per side;
// one workgroup, one thread per sector. The all-sector row reduces through
// workgroup memory.
@compute @workgroup_size(256)
fn horizon_suffix(@builtin(local_invocation_index) s: u32) {
    if s < BUCKETS { atomicStore(&all_sectors[s], HORIZON_NONE); }
    workgroupBarrier();
    let prev = (s + SECTORS - 1u) % SECTORS;
    let next = (s + 1u) % SECTORS;
    let g = s / u32(GROUP_SECTORS);
    let g_prev = prev / u32(GROUP_SECTORS);
    let g_next = next / u32(GROUP_SECTORS);
    for (var b = i32(BUCKETS) - 1; b >= 0; b--) {
        let row = u32(b) * SECTORS;
        let groups = SECTORS * BUCKETS + u32(b) * GROUPS;
        let own = max(atomicLoad(&horizon_acc[row + s]), atomicLoad(&horizon_acc[groups + g]));
        atomicMax(&all_sectors[b], own);
        let v = max(own, max(
            max(atomicLoad(&horizon_acc[row + prev]), atomicLoad(&horizon_acc[groups + g_prev])),
            max(atomicLoad(&horizon_acc[row + next]), atomicLoad(&horizon_acc[groups + g_next]))));
        horizon[row + s] = clearing_elevation(v, u32(b));
    }
    workgroupBarrier();
    if s < BUCKETS {
        horizon[H_ALL + s] = clearing_elevation(atomicLoad(&all_sectors[s]), s);
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

// Distance along an eye ray of elevation e to angular distance phi (3e38
// beyond the ray's reach).
fn eye_ray_distance(e: f32, phi: f32) -> f32 {
    if is_plane() {
        let c = cos(e);
        if c <= 1e-6 { return 3.0e38; }
        return phi / c;
    }
    let c = cos(e + phi);
    if c <= 1e-6 { return 3.0e38; }
    return frame.eye.w * sin(phi) / c;
}

// Ray distances between which an eye ray can meet resident terrain: from
// the start of the nearest distance bucket whose clearing elevation the ray
// does not exceed to the end of the farthest one. Everything outside is
// provably empty. (0, 0) when the ray clears every bucket; (0, 3e38) when
// the table cannot tell.
fn sky_span(s: SkyRay) -> vec2<f32> {
    if s.sector < -1 { return vec2<f32>(0.0, 3.0e38); }
    let e = atan2(s.lu, s.lt);
    var row = H_ALL;
    var stride = 1u;
    if s.sector >= 0 {
        row = u32(s.sector);
        stride = SECTORS;
    }
    var last = i32(BUCKETS) - 1;
    while last >= 0 && e > horizon[row + u32(last) * stride] { last -= 1; }
    if last < 0 { return vec2<f32>(0.0); }
    var first = 0;
    while first < last && e > horizon[row + u32(first) * stride] { first += 1; }
    var span = vec2<f32>(0.0, 3.0e38);
    if first > 0 {
        let t = eye_ray_distance(e, bucket_start(u32(first)));
        if t < 3.0e38 { span.x = max(t * 0.9999 - 1e-3, 0.0); }
    }
    if last + 1 < i32(BUCKETS) {
        let t = eye_ray_distance(e, bucket_start(u32(last + 1)));
        if t < 3.0e38 { span.y = t * 1.0001 + 1e-3; }
    }
    return span;
}
