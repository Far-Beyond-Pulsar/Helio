# PBR Correctness Audit — Issues & Fixes

## Issue 1: Environment Specular BRDF Fallback

**File:** `crates/helio-pass-deferred-light/shaders/deferred_lighting.wgsl`
**Lines:** 834–839

**Current code:**
```wgsl
let env_lod      = roughness * 8.0;
let env_sample   = textureSampleLevel(env_cube, env_sampler, R, env_lod).rgb;
let spec_scale   = 1.0 - roughness * roughness;
let spec_ind     = F_ibl * env_sample * spec_scale;
```

**Problem:**
`spec_scale = 1.0 - roughness²` is an ad-hoc hack. It doesn't conserve energy: smooth surfaces get too little environment reflection, rough surfaces get too much. The Fresnel term `F_ibl` also uses an IBL variant of Schlick that doesn't interact correctly with the `spec_scale` multiplier.

**Fix:**
Replace with the split-sum IBL approximation using an analytical BRDF integration. The correct form is:

```
// Analytical approximation of the split-sum BRDF LUT (Karis 2013 / UE4).
// Returns (scale, bias) such that the environment specular contribution is:
//   env_sample * (F0 * scale + bias)
fn env_brdf_approx(NdotV: f32, roughness: f32) -> vec2<f32> {
    let a    = roughness * roughness;
    let a2   = a * a;
    let ndv  = NdotV;
    let bias = 1.0 - a2 * 0.5 / (a2 * 0.5 + 0.33) * ndv;
    let scale = 1.0 - bias;
    return vec2<f32>(scale, bias);
}
```

Usage:
```wgsl
let env_brdf   = env_brdf_approx(NdV, roughness);
let spec_ind   = env_sample * (F0 * env_brdf.x + env_brdf.y);
```

Remove `F_ibl` from the environment path (it was the IBL Fresnel that is now handled by the BRDF LUT). Keep `F_ibl` only if used elsewhere.

---

## Issue 2: HLFS Shader Non-Physical Shading

**File:** `crates/helio-pass-hlfs/shaders/hlfs_shade.wgsl`
**Lines:** 299–328 (evaluate_light), 389–417 (fragment)

**Problem:**
The HLFS path uses a simple Lambertian diffuse + Phong specular hack. There is no GGX distribution, no Smith geometry, no Fresnel, and no energy conservation. Metals and dielectrics are both rendered with a flat 0.04 specular color, so metallic surfaces look like plastic.

**Fix:**
Replace `evaluate_light` and the fragment shading with the same Cook-Torrance BRDF used in `deferred_lighting.wgsl`:

1. Copy these functions from deferred_lighting.wgsl (lines 500–531):
   - `distribution_ggx`
   - `geometry_schlick_ggx`
   - `geometry_smith`
   - `fresnel_schlick`

2. Rewrite `evaluate_light` to accept (light, world_pos, normal, V, F0, albedo, roughness, metallic) and return the Cook-Torrance radiance (same as `pbr_direct_light` in the deferred shader).

3. In the fragment shader:
   - Reconstruct V (view direction) from camera and world_pos
   - Read F0 from the GBuffer albedo and ORM textures (`F0 = mix(0.04, albedo, metallic)`)
   - Call the new evaluate_light for each light
   - Remove the Phong specular hack
   - For indirect lighting, use the existing field-based indirect but apply proper Fresnel and energy conservation

**Requirements:**
- The HLFS shader needs V (view direction). Currently it doesn't compute it. Add:
  ```wgsl
  let V = normalize(camera.position_near.xyz - world_pos);
  ```
- The HLFS shader needs F0. Currently it doesn't decode it. Add:
  ```wgsl
  let F0 = mix(vec3<f32>(0.04), albedo, metallic);
  ```
- The HLFS shader needs to read lightmap_uv from GBuffer for VG detection (location 4 in gbuffer).
  Add to group 1: `@group(1) @binding(5) var gbuf_lightmap_uv: texture_2d<f32>;`
  The HLFS pass Rust code would need the lightmap_uv texture added to its bind group.

---

## Issue 3: Point/Spot Light Falloff

**File:** `crates/helio-pass-deferred-light/shaders/deferred_lighting.wgsl`
**Lines:** 557–559

**Current code:**
```wgsl
let ratio   = dist / light.position_range.w;
let falloff = max(0.0, 1.0 - ratio * ratio);
var atten   = falloff * falloff;
```

**Problem:**
`(1 - (d/r)²)²` is artist-friendly but not physically correct. Correct point light attenuation follows the inverse-square law: `1 / d²`. The smooth falloff makes lights seem dimmer at close range and brighter at range vs the physical law.

**Fix:**
Replace with physically-based inverse-square attenuation:
```wgsl
var atten = 1.0 / (dist * dist + 0.0001);
// Optional: apply range fade near the cutoff distance to avoid hard clipping
let normalized_dist = dist / light.position_range.w;
atten *= max(0.0, 1.0 - normalized_dist * normalized_dist * normalized_dist * normalized_dist);
```

The range fade is a smooth fade to zero at the cutoff radius (Unity HDRP approach). The `+0.0001` prevents division by zero.
