// Primary rays request complete missing 4x4-column blocks. The CPU still
// owns admission and publication. A new salt retries hash-bit collisions.
override VISIBLE_FEEDBACK: bool = false;
const VISIBLE_REQUEST_CAPACITY: u32 = 256u;
struct VisibleRequests {
    count: atomic<u32>,
    overflow: atomic<u32>,
    attempts: atomic<u32>,
    pad: u32,
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
