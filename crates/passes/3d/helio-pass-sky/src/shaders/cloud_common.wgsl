// =============================================================================
// Cloud Common — Shared Volumetric Cloud Utilities
// =============================================================================
// Contains height gradient, noise helpers, dual HG phase, Beer-Powder,
// multi-scattering octaves, and ambient ground albedo. Included by raymarch
// and procedural paths. No entry point — pure functions.
// =============================================================================

const PI_COMMON: f32 = 3.141592653589793;

// Dual-Henyey-Greenstein: d1*HG(g1) + (1-d1)*HG(g2), g1~0.8 forward, g2~-0.3 back
fn hg_phase(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / (4.0 * PI_COMMON * pow(max(1.0 + g2 - 2.0 * g * cos_theta, 0.0001), 1.5));
}
fn dual_hg(cos_theta: f32, g1: f32, g2: f32, blend: f32) -> f32 {
    return blend * hg_phase(cos_theta, g1) + (1.0 - blend) * hg_phase(cos_theta, g2);
}

// Beer-Powder: Approx internal multi-scattering, dark bellies, edge highlights
// Light Attenuation = exp(-tau*d) * (1 - exp(-2*tau*d))
fn beer_powder(tau_d: f32) -> f32 {
    return exp(-tau_d) * (1.0 - exp(-2.0 * tau_d));
}

// Multi-scattering octaves: 2-3 octaves with exponentially decreasing density
// and increasing phase isotropy, no secondary rays
fn ms_octaves(sun_vis: f32, density: f32) -> f32 {
    let o0 = sun_vis;
    let o1 = sqrt(max(sun_vis, 0.0)) * 0.28 * exp(-density * 0.5);
    let o2 = pow(max(sun_vis, 0.0), 0.25) * 0.10 * exp(-density * 0.25);
    return (o0 + o1 + o2) / 1.38;
}

// Height-gradient ambient: dark blue/grey at bottom → sky ambient at top
fn ambient_height_gradient(h: f32, bottom: vec3<f32>, top: vec3<f32>) -> vec3<f32> {
    return mix(bottom, top, smoothstep(0.0, 1.0, h));
}

// Cloud type height profiles
fn cloud_height_profile(h: f32, cloud_type: u32) -> f32 {
    if (cloud_type == 0u) { // Cumulus
        return smoothstep(0.0, 0.15, h) * (1.0 - smoothstep(0.55, 0.95, h));
    } else if (cloud_type == 1u) { // Stratocumulus
        return smoothstep(0.0, 0.10, h) * (1.0 - smoothstep(0.45, 0.75, h)) * 1.1;
    } else { // Cumulonimbus
        let tower = smoothstep(0.0, 0.08, h) * (1.0 - smoothstep(0.85, 1.0, h));
        let anvil = smoothstep(0.70, 0.82, h) * 0.35;
        return tower + anvil;
    }
}
