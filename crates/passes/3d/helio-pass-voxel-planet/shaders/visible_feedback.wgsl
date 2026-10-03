// Primary rays request complete missing 4x4-column blocks. The CPU still
// owns admission and publication. A new salt retries hash-bit collisions.
override VISIBLE_FEEDBACK: bool = false;
override PRIMARY_SAMPLES: bool = false;
const VISIBLE_REQUEST_CAPACITY: u32 = 256u;
struct VisibleRequests {
    count: atomic<u32>,
    overflow: atomic<u32>,
    attempts: atomic<u32>,
    sampled: atomic<u32>,
    keys: array<vec2<u32>, 256>,
    seen: array<atomic<u32>, 4096>,
}
@group(0) @binding(21) var<storage, read_write> visible_requests: VisibleRequests;
var<private> visible_request_attempts: u32 = 0u;
var<private> visible_feedback_sample: bool = false;

fn begin_visible_feedback_pixel(xy: vec2<u32>) {
    let phase = frame.hints.w >> 8u;
    visible_feedback_sample = VISIBLE_FEEDBACK && (xy.x & 7u) == (phase & 7u)
        && (xy.y & 7u) == ((phase >> 3u) & 7u);
}

fn request_visible_block(face: u32, level: u32, ci: i32, cj: i32) {
    if !VISIBLE_FEEDBACK || !visible_feedback_sample || (frame.hints.w & 16u) == 0u || visible_request_attempts >= 1u { return; }
    visible_request_attempts += 1u;
    atomicAdd(&visible_requests.attempts, 1u);
    if atomicLoad(&visible_requests.count) >= VISIBLE_REQUEST_CAPACITY {
        atomicStore(&visible_requests.overflow, 1u);
        return;
    }
    let key0 = column_key0(face, level, ci & ~3);
    let key1 = bitcast<u32>(cj & ~3);
    let salt = frame.hints.w >> 8u;
    let hash = column_slot(key0, key1 ^ (salt * 0x9e3779b9u)) & 131071u;
    let bit = 1u << (hash & 31u);
    if (atomicOr(&visible_requests.seen[hash >> 5u], bit) & bit) != 0u { return; }
    let at = atomicAdd(&visible_requests.count, 1u);
    if at < VISIBLE_REQUEST_CAPACITY {
        visible_requests.keys[at] = vec2<u32>(key0, key1);
    } else {
        atomicStore(&visible_requests.overflow, 1u);
    }
}

// Four independent 8-bit counters fit the unused header word. The CPU
// chooses a power-of-two stride whose entire sample grid has <=255 rays,
// so adding packed increments cannot carry into a neighbouring counter.
// This runs once for the final primary Hit, including the sky-bound retry.
fn sample_primary_hit(xy: vec2<u32>, hit: Hit) {
    if !PRIMARY_SAMPLES || (u32(frame.screen.w) & 8u) == 0u { return; }
    let exponent = (u32(frame.screen.w) >> 8u) & 31u;
    let mask = (1u << exponent) - 1u;
    let phase = frame.hints.w >> 8u;
    if (xy.x & mask) != (phase & mask)
        || (xy.y & mask) != ((phase >> exponent) & mask) { return; }
    let status = hit.info & 3u;
    if status == ST_LOADING || status == ST_EXHAUSTED {
        atomicAdd(&visible_requests.sampled, 1u << 24u);
    } else if status == ST_HIT {
        let level = (hit.info >> 5u) & 31u;
        var increment = 1u;
        if level > 0u {
            let pixel = 2.0 * max(hit.t, 0.05) / (abs(camera.proj[1][1]) * frame.screen.y);
            let width = frame.layer.y * f32(1u << level) / pixel;
            increment += select(0u, 1u << 8u, width > 2.0)
                + select(0u, 1u << 16u, width > 4.0);
        }
        atomicAdd(&visible_requests.sampled, increment);
    }
}
