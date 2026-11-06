// Every TLAS instance's thin-sheet transmission
// (`helio_frame_api::rays::RAY_TRANSMISSION`): a header (flags, row count,
// zero, zero) and one row per instance (linear RGB, then padding; zero RGB
// is opaque). Bind it as `ray_transmission`.
struct RayTransmissionData { header: vec4<u32>, rows: array<vec4<f32>> };

/// Header flag: the BLAS triangle opacity matches the rows (zero RGB opaque,
/// every other row non-opaque), so opaque hits need no row lookup.
const RAY_TRANSMISSION_OPACITY_MATCHES: u32 = 1u;

/// Instances the bound rows describe; a hit past them is opaque.
fn ray_transmission_count() -> u32 {
    return min(ray_transmission.header.y, arrayLength(&ray_transmission.rows));
}

/// What instance `instance` lets through (zero: opaque).
fn ray_transmission_of(instance: u32) -> vec3<f32> {
    if instance >= ray_transmission_count() {
        return vec3<f32>(0.0);
    }
    return ray_transmission.rows[instance].rgb;
}
