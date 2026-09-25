// Sampling revision 1. Upload a validated LandformSnapshot's dense height atlas.
// Cell centers are odd integers in 0.05 m units. Floating point is used only
// to estimate division; wide-integer correction certifies every quotient.
struct LFSettings { resolution: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<uniform> lf_settings: LFSettings;
@group(0) @binding(1) var<storage, read> lf_heights: array<i32>;

// Packed words: face, u, v, fraction_u, fraction_v, signed height bits,
// occupancy, validity. Invalid cells use face=0xffffffff and seven zeroes.
struct LFCellSample { address: vec4<u32>, value: vec4<u32> }

fn lf_mul(a: u32, b: u32) -> vec2<u32> {
    let a0 = a & 65535u;
    let a1 = a >> 16u;
    let b0 = b & 65535u;
    let b1 = b >> 16u;
    let p0 = a0 * b0;
    let p1 = a1 * b0 + (p0 >> 16u);
    let p2 = a0 * b1 + (p1 & 65535u);
    return vec2<u32>((p2 << 16u) | (p0 & 65535u),
        a1 * b1 + (p1 >> 16u) + (p2 >> 16u));
}

fn lf_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let low = a.x + b.x;
    return vec2<u32>(low, a.y + b.y + select(0u, 1u, low < a.x));
}

fn lf_sub(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(a.x - b.x, a.y - b.y - select(0u, 1u, a.x < b.x));
}

fn lf_greater(a: vec2<u32>, b: vec2<u32>) -> bool {
    return a.y > b.y || (a.y == b.y && a.x > b.x);
}

// denominator > 0, true quotient <= maximum. Correctness does not depend on
// an assumed rounding mode or a fixed number of float-estimate corrections.
fn lf_div(numerator: vec2<u32>, denominator: u32, maximum: u32) -> u32 {
    let estimate = (f32(numerator.y) * 4294967296.0 + f32(numerator.x)) / f32(denominator);
    var quotient = u32(clamp(floor(estimate), 0.0, f32(maximum)));
    loop {
        let product = lf_mul(quotient, denominator);
        if lf_greater(product, numerator) {
            quotient -= 1u;
            continue;
        }
        let remainder = lf_sub(numerator, product);
        if remainder.y != 0u || remainder.x >= denominator {
            quotient += 1u;
            continue;
        }
        break;
    }
    return quotient;
}

fn lf_project(component: i32, dominant: u32, resolution: u32) -> vec2<u32> {
    let denominator = dominant * 2u;
    let numerator = lf_mul(u32(component + i32(dominant)), resolution);
    let whole = min(lf_div(numerator, denominator, resolution), resolution - 1u);
    // After clamping at the positive face edge, remainder may equal denominator.
    let remainder = lf_sub(numerator, lf_mul(whole, denominator)).x;
    let fraction = lf_div(lf_mul(remainder, 65536u), denominator, 65536u);
    return vec2<u32>(whole, fraction);
}

fn lf_lerp(a: i32, b: i32, fraction: u32) -> i32 {
    let delta = b - a;
    let product = lf_mul(u32(abs(delta)), fraction);
    let quotient = (product.y << 16u) | (product.x >> 16u);
    if delta < 0 {
        // Division must floor toward negative infinity, including tiny deltas.
        return a - i32(quotient + select(0u, 1u, (product.x & 65535u) != 0u));
    }
    return a + i32(quotient);
}

fn lf_sample_cell(cell: vec3<i32>) -> LFCellSample {
    let n = lf_settings.resolution;
    if any(cell < vec3<i32>(-100000000)) || any(cell > vec3<i32>(100000000))
        || n == 0u || n > 256u || (n & (n - 1u)) != 0u {
        return LFCellSample(vec4<u32>(0xffffffffu, 0u, 0u, 0u), vec4<u32>(0u));
    }
    let center = cell * 2 + vec3<i32>(1);
    let magnitude = vec3<u32>(abs(center));
    var axis = 0u;
    if magnitude.y > magnitude[axis] { axis = 1u; }
    if magnitude.z > magnitude[axis] { axis = 2u; }
    let face = axis * 2u + select(0u, 1u, center[axis] < 0);
    var other = center.yz;
    if axis == 1u { other = center.xz; }
    if axis == 2u { other = center.xy; }
    let u = lf_project(other.x, magnitude[axis], n);
    let v = lf_project(other.y, magnitude[axis], n);
    let side = n + 1u;
    let index = face * side * side + v.x * side + u.x;
    let row0 = lf_lerp(lf_heights[index], lf_heights[index + 1u], u.y);
    let row1 = lf_lerp(lf_heights[index + side], lf_heights[index + side + 1u], u.y);
    let height = lf_lerp(row0, row1, v.y);
    let radius = u32(127420000 + height);
    let squared = lf_add(lf_add(lf_mul(magnitude.x, magnitude.x),
        lf_mul(magnitude.y, magnitude.y)), lf_mul(magnitude.z, magnitude.z));
    let solid = !lf_greater(squared, lf_mul(radius, radius));
    return LFCellSample(vec4<u32>(face, u.x, v.x, u.y),
        vec4<u32>(v.y, bitcast<u32>(height), select(0u, 1u, solid), 1u));
}
