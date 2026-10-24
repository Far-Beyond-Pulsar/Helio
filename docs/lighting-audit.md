# Helio lighting pipeline audit (voxel planets, ground to orbit)

Date: 2026-10-14. Branch `wip/restore-visual-fixes`. Read-only audit; nothing was built or run.
Target: AAA look for procedural voxel planets (Earth-like, airless moons, deserts) from 2 m to orbit,
RTX 3060, 1440p output with TSR (internal ~1.6-2.2 Mpx), terrain GPU <= 6 ms.

Cost figures are estimates for that target, not measurements. Paths are relative to the Helio repo
unless prefixed `Pulsar:`. Line references are as of the audit and have since drifted.

**Radiance cascades (2026-10-22):** the 3D radiance cascades and every consumer of them were deleted; world-space GI is helio-pass-gi's probe clipmap. Items below that mention RC are historical.

**Status (2026-10-15).** Done: #1 (per-surface sky irradiance table, `atmosphere_sky_table`),
#8 (grading in log space and after the tone map, Bradford white balance), the metering half of #3
(64-bin histogram without sky or space; no EV100, physical camera or pre-exposure yet) and the
world-description half of #2 (`VoxelTerrainComponent::air` derives the sky, `None` is airless; the
planet frame and sun disk without air are not done). Open: #4, #5, #6, #7, #9, #10 and the rest of
#2 and #3.

---

## 0. Summary: top findings

| # | Pri | Finding | Visible consequence | Fix | Est. cost |
|---|-----|---------|---------------------|-----|-----------|
| 1 | P0 | Sky ambient is one L2 SH probe integrated from the sky-view LUT **at the eye** (`atmosphere_luts.wgsl:239-285`), applied to every pixel (`deferred_lighting.wgsl:1324-1325`). | From orbit, up-facing ground gets ~0 sky fill (shadows go black, sunlit ground loses ~10-15 % luminance and ~25 % of its blue); terrain past the terminator is lit by the eye's daylight probe (and vice versa); distant terrain at sunset takes the eye's sky. | Per-pixel sky irradiance table E(r, mu_s) as local-frame SH (Bruneton 2008 / 2017 style), stored in the atmosphere **uniform** (deferred is at its 16-texture limit). | Build ~0.1 ms (only when the atmosphere changes); lookup ~0.03-0.05 ms. |
| 2 | P0 | Atmosphere presence is a level object, not part of the world description. Random moons keep Earth air; `AtmosphereComponent.planet_radius_km` duplicates `VoxelTerrainComponent.planet_radius`; without an atmosphere there is no planet frame, no sun disk and the ambient hemisphere uses world +Y. | Airless worlds get a blue sky, blue haze and a reddened, ~25-40 % dimmer sun; a radius mismatch fills the planet interior with sea-level air; airless worlds show no sun at all. | Planet body owns `environment` (atmosphere preset/none, gravity, star); the atmosphere row is derived; publish a planet frame independent of air; space sky path (sun disk, stars). | ~0 ms; mostly data plumbing. |
| 3 | P0 | Exposure is fixed (Manual, EV 0, `postprocess.wgsl:612-620`); the "auto" mode is a plain mean of log luminance over all pixels including black space (`postprocess.wgsl:521-587`); no EV100/physical camera; no pre-exposure. | Orbit/moon views are dark and blue-grey (Earth air on the moon accounts for ~35-45 % loss + blue path radiance; Lambert regolith darkens toward the limb); turning auto on would push +4 EV (clamp) and blow out the planet. | Physical camera (EV100, Lagarde & de Rousiers 2014), pre-exposed lighting, 64-bin histogram metering with low/high percentile rejection and sky/space mask, EV100 limits. | Histogram ~0.05 ms; pre-exposure ~0 ms. |
| 4 | P1 | Voxel sun visibility (`directional_visibility`) is applied to every pixel whose light direction matches, including meshes drawn in front of terrain (`deferred_lighting.wgsl:1261-1267`). Meshes are never shadowed by terrain. | Props/foliage take the shadow state of the terrain behind them; objects under a mountain's shadow are sunlit. | Gate by the voxel tag; walk the chunk trees from non-voxel gbuffer pixels too (reuse `shadow_fallback`). | 0.1-0.5 ms with many mesh pixels. |
| 5 | P1 | No sky specular: the environment cube is the baked probe or a black 1x1 cube (`helio-pass-deferred-light/src/lib.rs:1023-1028`); SSR is off by default and fades out above roughness 0.4-0.7. | Water, ice, wet/snowy ground show no Fresnel sky reflection; oceans are dark at grazing angles and from orbit. | Runtime sky capture from the atmosphere into env layer 0 (GGX-prefiltered), captured at the eye inside the air and at the ground under the view when in space. | ~0.1-0.2 ms amortised. |
| 6 | P1 | Voxel sun walks are binary point-sun rays (`chunk_draw.wgsl:1918-1922`); the 0.267 deg solar disk is ignored. | Razor-hard shadows at kilometre range (real penumbra of a 1 km occluder is ~9 m). | Jitter the walk direction inside the solar cone per pixel per frame; TSR integrates. | ~0 ms. |
| 7 | P1 | No bounce light at all on planets: radiance cascades are not in the default graph (`helio-default-passes/src/lib.rs:921`), no SSAO/SSGI, Pulsar sets ambient to 0. | Lunar shadows are pitch black (real ones are filled by sunlit crater walls); Earth valleys miss warm ground bounce. | Analytic terrain-bounce term from the existing sky-visibility complement; later one-bounce diffuse rays on the chunk trees. | ~0 ms now; ~0.5-1 ms for traced bounce. |
| 8 | P1 | Colour grading bugs: `color_saturation` and `color_contrast` are plain multipliers (`postprocess.wgsl:819-824`); the LUT path clamps scene-linear HDR to [0,1] before the LUT (`:797-800`); white balance gives r x1.54, b x0.65 at its default 6500 K (`:829-837`). | The level's "saturation 0.85 / contrast 1.15" just scales brightness by 0.98; any LUT clips highlights before tone mapping; enabling WB tints the frame orange. | Luminance-preserving saturation, log-space contrast around 0.18, grade after tone mapping (or in a log/ACEScct shaper), Bradford/CAT02 white balance from CCT. | ~0 ms. |
| 9 | P1 | Regolith is Lambertian with an authored albedo of ~0.27 linear; no Lunar-Lambert/Hapke, no opposition surge, no sub-pixel relief roughness/micro-shadowing. | Moons look like matte plaster balls with limb darkening; terminator too soft and too bright on unresolved relief. | Material flag for regolith BRDF (McEwen 1991 Lunar-Lambert + Hapke opposition), slope-variance roughness and micro-shadowing from the generator's octave table. | ~0.02 ms ALU. |
| 10 | P1 | From space the atmosphere is marched per pixel with 32 **uniform** steps along the whole shell chord (`atmosphere_composite.wgsl:25,48-66`). | ~70 km steps across a 2,000+ km limb chord vs. 8 km / 1.2 km scale heights: limb glow and grazing haze band and misestimate; visible pop when crossing `top_radius`. | Tangent-point-centred step distribution (or Bruneton 4D table for space), blend band around `top_radius`. | Same or lower cost (~0.3-0.6 ms). |

Everything else (CSM details, PCSS formula, specular occlusion, multi-scatter GGX, tone curve choice,
aerial RGB transmittance, LUT caching, multi-body atmospheres, night sky) is P2 and listed per area below.

### What is already good

- The Hillaire 2020 core is faithful: LUT sizes (256x64 transmittance with 40 steps, 32x32
  multi-scattering with 64 directions x 20 steps, 192x108 sky-view, 32^3 aerial), Bruneton's
  transmittance parameterisation, the psi_ms = L2/(1 - f_ms) multiple-scattering transfer including
  the ground bounce, planet shadowing in single scattering (`hits_ground`), Cornette-Shanks Mie,
  a horizon-concentrated sky-view mapping with a sqrt azimuth mapping (an improvement over the paper),
  sun limb-occultation fade on surfaces (`atmosphere_common.wgsl:201-204`), and a per-pixel space
  path. Camera-relative precision is handled carefully (`resolve`, `atmosphere_view_ray`).
- Sun walks through the chunk trees reach any distance (`walk(..., 1e30, ...)`), which beats any CSM
  at planetary scale; amortised sampling (2x2 sun, 4x4 sky blocks, TSR-gathered) is the right pattern.
- Far-field albedo is prefiltered by pixel footprint (`chunk_draw.wgsl:1445-1460`), voxel AO/edges fade
  out below a few pixels (`:1505-1531`), slope normals replace sub-pixel step faces (`hit_face`).
- Post: frame-rate-independent adaptation, Jimenez 13-tap bloom downsample, TSR accumulates in a
  reversible tone-mapped space (`tsr_main.wgsl:102-106`).

---

## 1. Units, exposure, tone mapping (issue 3)

### 1.1 What exists

- Sun: Pulsar's directional `intensity` is passed through as lux whatever the unit selector says
  (`crates/helio-component/src/components/light_component/mapping.rs:81-95`). The shipped level uses
  **3.0** (`Pulsar:assets/examples/voxel_planet.level`, `voxel_sun.intensity = 3.0`, unit label
  "Lumens"); `voxel_flight` also uses 3.0 (`crates/helio-default-passes/examples/voxel_flight/worlds.rs`).
  The atmosphere treats it as top-of-atmosphere illuminance (`atmosphere_luts.wgsl:58-70`), and sky
  radiance, aerial in-scatter and the SH ambient all scale with it, so sun and sky are mutually
  consistent.
- Local lights: lumens are converted to candela physically (`mapping.rs:88-94`).
- Exposure: `exposure_scale()` = 2^(compensation [+ auto]) (`postprocess.wgsl:612-620`); defaults
  Manual, EV 0, tone map None (`helio-pass-postprocess/src/gpu_types.rs:295-325`). The level overrides
  ACES (Narkowicz fit) with `tonemap_exposure 1.1`, `color_saturation 0.85`, `color_contrast 1.15`.
- Auto mode: geometric mean of luminance from every 4th pixel, all pixels equal weight
  (`postprocess.wgsl:521-587`), exponential adaptation (`:595-607`), target 0.18, clamp [-4, +4] EV
  around EV 0. The Rust comment calls it "histogram-based" (`gpu_types.rs:49`); it is not.
- Lighting target is RGBA16F (`FOGGED_HDR_FORMAT`), storing absolute (un-pre-exposed) radiance.
- Bloom: threshold 1.0 applied to **pre-exposure** radiance (`postprocess.wgsl:647-660`); off by default.

### 1.2 Quantification

- Equivalent physical calibration. Under Lagarde & de Rousiers' camera model,
  exposure = 1 / (1.2 * 2^EV100). The current exposure of 1.0 is EV100 = log2(1/1.2) = -0.26. A sun of
  3.0 at EV100 -0.26 renders exactly like a sun of 3.0 * 2^15.26 = **~118 klux at EV100 15** ("sunny 16").
  So **today's daylight calibration is already physically plausible**; what is missing is the
  physical *camera*, so nothing else (local lights, emissives, night, space) shares that scale.
- Local lights are therefore ~2^15.3 = ~39,000x (15.3 EV) too bright relative to the sun if authored in
  real lumens/candela (an 800 lm bulb = 64 cd lights a surface 2 m away at 16 lux, about 5x the
  "sun"). AAA engines put both on one scale (Frostbite: sun ~100 klux, EV100 ~15; UE5 physical
  lighting guidance does the same).
- The dark lunar disk (moon world on Earth's air, sun 45 deg, regolith albedo ~0.27 linear from
  sRGB (142,140,135) via `engine.rs:91` and `terrain.rs:361`):
  - Airless Lambert: L = 0.27 * 3 * 0.707 / pi = 0.18 -> exactly middle grey -> ACES ~0.27 -> sRGB ~0.56.
  - With Earth air: sun transmittance at the surface for mu_s = 0.7 is ~0.80 (G) / ~0.6 (B), view
    transmittance from orbit ~0.85 (mean, grey in the composite), plus Rayleigh path radiance of roughly
    (0.02, 0.03, 0.07) in sun units. Result ~(0.13, 0.145, 0.18) -> sRGB ~(0.42, 0.44, 0.49):
    **dark blue-grey**, matching the report. The Earth air alone costs ~35-45 % and adds the blue cast.
  - Lambert makes it worse away from the sub-solar point (cos falloff to the terminator and limb);
    the real Moon at full phase is nearly uniformly bright (Lommel-Seeliger behaviour).
  - Photographers expose the Moon ~1.4 EV brighter than daylight Earth ("Looney 11": f/11, 1/ISO,
    EV100 ~13.6 vs sunny-16 EV100 15). A metering that ignores black space lands there naturally.
- Mean-log auto exposure in orbit: half the frame black (log2(1e-4) = -13.3) and half planet at
  log2(0.15) = -2.7 gives a mean of -8.0 and an auto EV of log2(0.18) + 8 = +5.5, clamped to +4:
  the planet would read 0.15 * 16 = 2.4 -> nearly white. Auto mode is unusable for space views.
- Physical-unit overflow: the sun disk radiance is E / Omega with Omega = 2 pi (1 - cos 0.004675) =
  6.87e-5 sr. At E = 3 that is 43,700 (already 2/3 of FP16 max 65,504); at 118 klux it is 1.7e9 and
  overflows to +inf in the RGBA16F target. Physical units therefore **require pre-exposure**.

### 1.3 Gaps vs AAA and fixes

1. **Physical camera (P1, ~0 ms).** Add `ev100` (manual) plus auto in EV100 space with `min_ev100`/
   `max_ev100` (e.g. -2..17) instead of relative EV. exposure = 1/(1.2 * 2^EV100) (Lagarde & de Rousiers
   2014, section 4.1). Migration with an identical image: multiply directional intensities by 2^15.26
   (sun 3.0 -> 118,000 lux), set manual EV100 15. Fix the Pulsar unit label: directional lights in lux.
2. **Pre-exposure (P1, required by 1).** Multiply every radiance written to the scene-linear targets by
   last frame's exposure (deferred direct and indirect, emissive, atmosphere composite and sun disk,
   forward/transparent, fog), kept in a GPU buffer (no readback); the post pass applies
   exposure / pre_exposure. This is how UE ("PreExposure") and Frostbite keep FP16 and TSR history
   in range; it also makes the TSR Reinhard weighting and the deferred `radiance < 0.002` early-out
   (`deferred_lighting.wgsl:873`) behave at every EV.
3. **Histogram metering (P0 for orbit, ~0.05 ms).** 64-bin log-luminance histogram on the
   1/4-res image, UE-style low/high percentile window (e.g. 10 %/90 %), centre weighting, and **exclude
   sky/space pixels** (depth == sky) or give them a small weight; meter the planet, not the void.
   Target "key" 0.18 with an exposure-compensation curve vs. scene EV (UE's "exposure compensation
   curve") so bright sunlit planets on black are exposed ~+1 EV over sunny 16. Keep the existing
   adaptation filter.
4. **Bloom in exposed units (P2).** Apply threshold/knee after exposure (or use thresholdless,
   energy-conserving bloom as UE5 does by default) so bloom does not change with EV. Enable a small
   default intensity: the sun disk and specular glints from orbit need glare to read as bright.
5. **Grading fixes (P1, ~0 ms).** `color_grade` (`postprocess.wgsl:814-825`): saturation =
   lerp(luma, c, s); contrast in log2 around 0.18 (UE: (log2 c - log2 0.18) * k + log2 0.18). Run LUTs
   on tone-mapped (display-referred) values or through a log shaper (ACEScct-like), never by clamping
   scene-linear HDR (`sample_lut` `:797-800`). White balance: Bradford/CAT02 in LMS from the CCT
   white point to D65; identity at 6500 K.
6. **Tone curve (P2).** The Narkowicz ACES fit is per-channel RGB with no gamut handling: skies drift
   cyan/purple, sunsets go yellow/"notorious six". AgX (Blender 4.x), ACES 2.0 output transform, or
   Khronos PBR Neutral (for material fidelity) are the current choices; keep ACES fit as an option.

---

## 2. Atmosphere (Hillaire 2020)

### 2.1 Feature check

| Feature | Status | Where | Notes |
|---|---|---|---|
| Transmittance LUT | OK | `atmosphere_luts.wgsl:107-124`, `atmosphere_common.wgsl:95-116` | 256x64, 40 steps, Bruneton mapping. |
| Multiple scattering (psi_ms) | OK | `atmosphere_luts.wgsl:128-179` | 32x32, 64 Fibonacci directions x 20 steps, ground bounce in L2. Matches the paper. |
| Sky-view LUT | OK | `:181-206` | 192x108, 32 quadratic steps, analytic ground below horizon. |
| Aerial perspective | OK with caveats | `:210-237`, composite `:111-119` | 32^3, range = horizon + shell clamped 32-4,000 km (~1,136 km at the ground: slice centres 0.28, 2.5, 6.9, 13.6 km...). Grey (mean) transmittance. |
| Planet shadow on the air | OK | `atmosphere_sun_transmittance_lut` via `hits_ground` | Earth shadow / twilight wedge correct. Terrain shadowing of the air (mountain shafts) missing (P2). |
| Sun disk + limb darkening | Mostly | `atmosphere_composite.wgsl:95-103` | Linear law u = 0.6 is right, but not normalised: disk mean is 0.8, so the disk carries 80 % of E. Divide by (1 - u/3). FP16 overflow with physical units (see 1.2). |
| Ground albedo bounce | OK | multi-scattering kernel `:167-172` | Constant 0.3 (`AtmosphereComponent::earth`, `atmosphere.rs:141`); should come from the planet's material mean (P2). |
| Space views | Per-pixel march | composite `:48-66`, `SPACE_STEPS = 32` | Correct choice (UE also marches outside the shell), but uniform steps (P1, see 2.3). |
| Update frequency | Every frame, all LUTs | `atmosphere.rs:496-500` | Transmittance and multi-scattering depend only on the medium; cache on change (P2, ~0.1-0.2 ms). |
| Sky irradiance | Eye SH probe | `atmosphere_luts.wgsl:239-285` | Issue 1, P0 (section 3). |
| Surfaces' sun transmittance | 12-step march per pixel | `atmosphere_common.wgsl:192-221`, used `deferred_lighting.wgsl:811-813` | Works; a uniform-buffer LUT (section 3.3) is cheaper and exact. |
| Forward/transparent/fog | Not connected | only `deferred_lighting.wgsl` uses atmosphere functions | Translucents and volumetric fog see an unattenuated white sun and no sky ambient (P1 consistency). |
| Multiple bodies | First enabled row, first directional light | `atmosphere_luts.wgsl:49-72` | Planet + moon scenes get one atmosphere (P2). |
| Medium below the datum | Clamped to sea-level density | `medium_at` `atmosphere_common.wgsl:60` | Fine for valleys; catastrophic if the terrain radius is smaller than `bottom_radius` (section 4). |

### 2.2 Aerial perspective details (P2)

- Grey transmittance (`aerial_kernel` stores the RGB mean, composite blends dst * alpha) loses
  chromatic extinction: distant terrain at sunset does not redden. Store RGB transmittance in a second
  32^3 volume (256 KB) and apply it with dual-source blending, or apply aerial perspective inside
  deferred lighting. ~0.05 ms.
- The froxel volume is built from the jittered projection: harmless, but use the unjittered one.
- The sky-view and aerial LUTs switch to per-pixel marching exactly at `top_radius`
  (`atmosphere_composite.wgsl:80`): blend the two over e.g. 0.9-1.1 x shell height to avoid a pop.

### 2.3 Space marching (P1)

32 uniform steps between the shell entry and exit (`atmosphere_integrate(..., quadratic = false)`).
A limb ray's chord through a 100 km shell is ~2 x sqrt(6460^2 - 6360^2) = 2,262 km, so ~70 km per
step against an 8 km Rayleigh and 1.2 km Mie scale height; the densest region (the tangent point) is
sampled once or twice. Expect banding and a wrong-brightness limb glow, and under-sampled haze on
terrain seen at grazing angles from orbit.

Fix (same cost): split the ray at the closest approach to the centre and distribute steps on each half
with spacing growing away from the tangent point (or uniform in altitude); use 8-24 steps chosen by
the optical path length (UE uses a variable sample count). For a full AAA space look, precompute
Bruneton's 4D scattering table once per atmosphere (Bruneton 2017 reference implementation, sun
independent, ~8 MB at 256x128x32 RGBA16F) and evaluate space and high-altitude views in O(1)
(S(x) - T(x, x_end) S(x_end)); it also yields the ground irradiance table of section 3 for free.

---

## 3. Sky ambient: the eye probe (issue 1)

### 3.1 What exists

`irradiance_kernel` projects 1,024 directions of the **sky-view LUT at the eye** onto L2 SH, convolves
with the cosine lobe and multiplies by the sun (`atmosphere_luts.wgsl:243-284`). Deferred lighting uses
`atmosphere_sky_irradiance(atmosphere, N) * albedo` for every pixel (`deferred_lighting.wgsl:1324-1325`),
times material AO times voxel sky visibility. The SH math (basis, Ramamoorthi-Hanrahan bands, the /pi)
is correct.

### 3.2 Why it is wrong and how much

Sky irradiance depends on the receiver's altitude r and sun cosine mu_s (and its normal), not the eye's.

- Ground, sun at 45 deg, Hillaire's Earth medium: a numerical single-scattering integration gives sky
  irradiance on a horizontal surface = **7 % of the direct sun's luminance (17 % in blue)**; multiple
  scattering and ground bounce raise that to roughly 10-15 % (25 %+ in blue). In shadow the sky is
  100 % of the light.
- **Orbit (400 km):** the probe's upper hemisphere is black, so up-facing terrain below loses all of
  that: sunlit ground -10-15 % and less blue; every shadow (crater floors, cliffs, valleys) goes black.
  The probe's lower hemisphere sees the sunlit analytic planet, so terrain near the limb and beyond the
  terminator whose normals tilt toward that disk picks up a few percent of daylight as false fill:
  a faint glow on the night side next to the terminator.
- **Ground at dusk:** the planet's sun angle changes ~1 deg per 111 km; terrain 200-400 km away at the
  terminator takes the eye's sky (daylight fill on night terrain, or twilight fill on sunlit peaks).
- Overhang ceilings and caves take the eye probe's ground radiance (analytic albedo 0.3), not their own.

### 3.3 Fix (P0)

Precompute **sky irradiance per (r, mu_s) as SH in the local frame** (z = local up, x = sun azimuth):

- By the sky's symmetry about the sun's vertical plane the y-odd coefficients vanish: L1 needs 3 RGB
  coefficients (DC, up, sun-azimuth), symmetric L2 needs 6 (adds Y20, Y21(xz), Y22). Include the lower
  hemisphere as the analytic ground (albedo x (sun + sky) / pi), as Bruneton's irradiance texture does.
- Table: 32 mu_s (non-linear, dense near the horizon for twilight) x 8 r (R-1 km to R+20 km for terrain;
  clamp above) x 6 vec4 = **24.6 KB**. Put it **in the AtmosphereFrame uniform** (or a second uniform):
  the deferred fragment stage is at the WebGPU 16 sampled-texture and storage-buffer limits
  (comments at `deferred_lighting.wgsl:147-152`, `:1309-1310`). A 64x16 RGB transmittance table
  (16 KB) can join it and replace the 12-step per-pixel march in `atmosphere_transmittance_towards`.
- Build: one compute kernel, per entry 256-512 sphere directions x 16-24 steps using the existing
  transmittance and psi_ms LUTs: ~2-3 M steps, ~0.1 ms. It depends only on the medium (not the sun or
  camera), so rebuild only when the AtmosphereComponent changes.
- Per pixel: up = normalize(P - C), r, mu_s, local frame from (up, sun), rotate N, manual bilinear over
  4 entries (24 uniform loads, ~50 ALU): ~0.03-0.05 ms at 1440p internal. Use the same function in
  forward, transparent and fog so all lighting agrees.
- Keep the eye SH only as a debug comparison (it must match the table at the eye's (r, mu_s)).
- Reference: Bruneton & Neyret 2008 (irradiance texture E(r, mu_s)), Bruneton 2017 implementation;
  Hillaire 2020 section 5 notes the per-view probe is a simplification for local scenes.

---

## 4. Airless worlds and world-driven atmosphere (issue 2)

### 4.1 What exists

- Helio: "Without a row there is no atmosphere (space, a moon)" (`helio-pass-sky/src/atmosphere.rs:7-8`).
  With no row: composite inactive, background stays black (`helio-default-passes/src/background.rs:1-10`),
  so **no sun disk**; deferred falls back to `globals.ambient_color` as a hemisphere around **world +Y**
  (`deferred_lighting.wgsl:1327-1330`) - wrong anywhere on a sphere but the north pole. Pulsar sets
  that ambient to 0 (`Pulsar:crates/core/engine_backend/src/subsystems/render/helio_renderer/renderer.rs:1240`).
- Pulsar: the level carries one `AtmosphereComponent` (Earth air, `planet_radius_km 6371`,
  placement PlanetAtOwner) beside `VoxelTerrainComponent.planet_radius 6371000`. The random-world
  blueprint (`Pulsar:assets/examples/random_world.blueprint.json`) rolls Lunar/Rules/Earthlike material
  styles and layers but never touches the atmosphere: moons keep Earth air.
- `AtmosphereComponent::around_planet` keeps Earth's 100 km thickness and 8 km/1.2 km scale heights for
  any radius (`atmosphere.rs:170-177`).

### 4.2 Consequences

- Moons and dry worlds get a blue sky, blue aerial haze and a sun reddened and dimmed by Earth air
  (~20-45 % depending on sun height), see 1.2.
- Two independent radii: if the terrain radius is ever smaller than `bottom_radius` (e.g. the
  blueprint test's 1,000 km planet with the default 6,371 km air), every point below `bottom_radius`
  is integrated at sea-level density (`medium_at` clamps h < 0 to 0): thousands of km of air between
  the eye and anything, i.e. the whole planet fogged to sky colour.
- An airless planet has no planet frame for lighting at all (no up for AO/ambient, no limb occultation
  of the sun, no sun disk).

### 4.3 Recommendation (P0)

Make the **planet body the single source of truth** and derive rendering rows from it:

1. A body description owned by the terrain/world (e.g. `VoxelTerrainComponent` or a new `PlanetBody`):
   `radius`, `centre` (owner), `surface_gravity`, `star` (illuminance at the body, angular radius,
   colour temperature) and `atmosphere: None | Preset(Earth, Mars, ThinDust, Dense) | Custom(params)`.
2. The environment join derives the sky pass row from it: `bottom_radius` = terrain radius (never authored
   twice), thickness ~ 12 x the Rayleigh scale height, H = R T / (M g) scaled by gravity, Rayleigh
   coefficients proportional to surface pressure / scale height, Mie from an aerosol amount, ozone only
   for oxygen worlds, `ground_albedo` = coverage-weighted mean of the material palette. A level-authored
   `AtmosphereComponent` becomes an override, not a second source.
3. The random-world blueprint picks the atmosphere with the material style (Lunar -> None, Rules/desert
   -> thin dusty or Mars-like, Earthlike -> Earth with randomised turbidity), through the same script
   natives it already uses for layers.
4. Split "planet exists" from "air exists" in `AtmosphereFrame` (`planet.w` currently means both):
   publish centre, radius and sun for every body; `has_air` gates transmittance (1 when false), sky
   radiance (black) and aerial perspective. Airless worlds then still get the sun disk, limb occultation,
   a planet-relative up, and the terrain-bounce ambient of section 6.
5. Space sky path in the composite for airless scenes: sun disk with limb darkening (+ bloom), star field
   (P2), Earthshine-like fill from another lit body (P2).
6. Multiple bodies (P2): keep one row per body, select the body whose shell contains the camera for the
   LUTs, and march other in-view shells per pixel.

---

## 5. Direct lighting and shadows

### 5.1 BRDF and lights

- Cook-Torrance GGX + Schlick-GGX (k = (r+1)^2/8) + Schlick Fresnel + Lambert
  (`crates/helio-mats/shaders/pbr_eval.wgsl:12-66`, `deferred_lighting.wgsl:870-911`). Correct UE4-style
  baseline (Karis 2013).
- AAA gaps (P2): height-correlated Smith (Heitz 2014; Frostbite), multi-scatter energy compensation
  (Kulla & Conty 2017 / Fdez-Aguera 2019) - small for F0 0.03 terrain, visible on rough metals;
  Burley or Oren-Nayar diffuse for rough soils/vegetation (retro-reflective hot spot with the sun
  behind the viewer).
- Sun: no area-light specular treatment of the solar disk (Karis "representative point" or a roughness
  floor of ~0.004675 rad) - water/ice glints are point-light sharp (P2).
- The sun is attenuated by the atmosphere per pixel (`deferred_lighting.wgsl:808-813`): correct
  Hillaire usage; replace the 12-step march with the uniform LUT (section 3.3).

### 5.2 Voxel sun walks (terrain)

- Binary visibility from a point sun (`chunk_draw.wgsl:1918-1922`), sampled at one pixel of each 2x2
  block, shared within 4 cells (`:2060-2105`), with listed fallbacks. Range unlimited: correct choice
  for planets.
- **Soft shadows (P1, ~0 ms):** jitter `params.sun` per sample inside the solar cone (0.267 deg,
  `sun_angular_radius` is already in the atmosphere row) with a blue-noise sequence per frame; TSR
  accumulates a physically sized penumbra (occluder distance x 0.0093). Same rays, no extra cost.
  Reference: cone-sampled ray-traced shadows (Ray Tracing Gems ch. 13; UE5 VSM SMRT).
- **Mesh pixels (P1 bug):** `directional_visibility` holds the voxel hit behind each pixel and is
  applied without checking that the gbuffer pixel *is* that voxel (`deferred_lighting.wgsl:1261-1267`
  versus the `is_stored_voxel` tag at `:1113-1114`). Gate it on the tag, and for non-voxel pixels walk
  from the reconstructed position (an extra indirect list like `shadow_fallback`, `chunk_draw.wgsl:2117`)
  so props and foliage are shadowed by terrain.
- Debug leftover: `if voxel_visibility.x < 0.0 { return vec4(4.0, 0.0, 2.6, 1.0); }`
  (`deferred_lighting.wgsl:1264`) returns magenta from production lighting (P2: move behind a debug mode).

### 5.3 Cascaded shadow maps (meshes)

- Fixed splits 16/80/300/1,400 m (`helio-pass-shadow-matrix/src/gpu_types.rs:32`), 1,024^2 per cascade
  (texel ~3 cm to ~2.7 m), nothing beyond 1.4 km. Acceptable if CSM only serves meshes and terrain uses
  walks; the level has `cast_shadows = false` on the sun, so meshes currently cast no shadows on terrain.
- Normal offset is a constant 0.01 m (`deferred_lighting.wgsl:311`): not proportional to the cascade
  texel size, so acne/peter-panning vary per cascade (P2: scale by texel world size, UE/HDRP do).
- PCSS uses the perspective formula (d_r - d_b) / d_b * size (`:407-416`), wrong for an orthographic
  sun: the penumbra is (d_r - d_b) * tan(sun angular radius) in world units, then converted to texels
  (P2).
- Contact shadows: 16 steps over 0.5 m (`:1528-1551`): fine for props; irrelevant at terrain scale.

---

## 6. Ambient, AO, GI

### 6.1 What exists

- Indirect diffuse = SH sky (section 3) x albedo x (material AO x voxel sky visibility). SSAO is skipped
  for voxel pixels (`deferred_lighting.wgsl:1113-1117`) and is not in the default graph at all.
- Voxel sky visibility: 4 cosine-weighted rays about the planet's up per 4x4 block per frame, range
  48 **cells** (`chunk_draw.wgsl:1778-1779, 1928-1951`), times corner AO (`:1505-1531`).
- Radiance cascades: not in the default graph (`helio-default-passes/src/lib.rs:921`); the deferred
  sampler is a 16^3 camera-centred probe grid (`deferred_lighting.wgsl:916-1009`), meaningless at
  planetary scale. SSR off by default (`lib.rs:944`). Pulsar ambient 0.

### 6.2 Gaps and fixes

1. **Terrain bounce (P1, ~0 ms).** Use the complement of sky visibility as the view factor to the
   surrounding terrain: E_bounce ~ (1 - V_sky) x rho_mean x (E_sun T_sun max(mu_s, 0) + E_sky), with
   rho_mean the local material albedo (or the body's mean). This is the dominant fill on moons (sunlit
   crater walls lighting shadowed floors) and adds warm bounce in Earth valleys. Clamp to avoid
   double counting where the sun walk says the neighbourhood is shadowed.
2. **LOD-dependent AO radius (P1).** `range = SKY_RANGE_CELLS * f.cell` makes the occlusion radius grow
   with distance (4.8 m at 0.1 m cells, ~480 m where cells are 10 m): a valley darkens as you fly away
   and brightens as you approach ("breathing"). Use two fixed world-space scales: a near radius in
   metres for the rays, plus a **large-scale terrain visibility** stored per coarse column at generation
   (horizon angles or a sky-visibility scalar per tile, as Far Cry 5 / Frostbite heightfield AO do) and
   combine them (min or product).
3. **Bent normal (P2).** The visibility is a scalar computed about the planet's up, multiplied with
   SH(N). Accumulate the mean open direction of the 4 rays (bent normal) and evaluate SH with it
   (Jimenez et al. 2016, GTAO); it also feeds specular occlusion.
4. **Traced one-bounce diffuse (P1 later, ~0.5-1 ms).** The walk infrastructure already shoots sky rays:
   on a hit, read the hit's albedo and its sun visibility (the 2x2 sun samples) to return one bounce;
   TSR accumulates. This is the Lumen-like path for planets using the chunk trees as the scene
   representation.
5. **Unit-dependent RC blend (P2 bug when RC is on).** `rc_weight = clamp(length(rc_irr) * 4, 0, 1)`
   (`deferred_lighting.wgsl:1334`): dark (occluded) RC falls back to the sky hemisphere, leaking light
   into occluded areas, and the threshold depends on the light units. Use RC volume coverage, not
   brightness.
6. Ambient for metals ignores metallic and Fresnel (`:1325`, `hemi * albedo`) (P2):
   multiply by (1 - metallic) x (1 - F_ibl) like the RC path does (`:1286-1288`).

---

## 7. Reflections

### 7.1 What exists

`fs_reflection` (`deferred_lighting.wgsl:1375-1451`): env cube (split-sum with Lazarov's analytic DFG,
`pbr_eval.wgsl:70-76`), SSR (roughness fade 0.4-0.7, `helio-pass-ssr/shaders/ssr_trace.wgsl:62`),
planar, RC rough fallback, all multiplied by AO. The env cube is the baked probe array or a black
fallback (`helio-pass-deferred-light/src/lib.rs:1023-1028`); nothing captures the atmosphere.

### 7.2 Gaps and fixes

1. **Sky reflection probe (P1, ~0.1-0.2 ms amortised).** Render the atmosphere (sky-view LUT + sun
   disk as a clamped lobe) into a 128^2 cube each frame (or one face per frame), GGX-prefilter the mips
   (importance-sampled, Karis 2013), and bind it as layer 0 when no bake exists. Capture point: the eye
   when inside the shell; when in space, the ground point under the view centre (otherwise oceans seen
   from orbit reflect black space). UE's real-time sky light capture is the model.
2. **Specular occlusion (P2, ~0 ms).** Replace `spec * ao` (`:1449`) with Lagarde's
   SO = saturate((NdV + ao)^exp2(-16 r - 1) - 1 + ao) (Lagarde & de Rousiers 2014), or GTSO from the
   bent normal and visibility cone (Jimenez et al. 2016). Add horizon occlusion for normal-mapped
   reflections (fade R below the geometric surface).
3. **Consistent SSR weighting (P2).** SSR is blended as `ssr * F_ibl` (`:1432`) while the env term uses
   the split-sum `F0 * A + B`; use the same DFG term for both.
4. For rough terrain (roughness 0.84-0.97) specular IBL is a small term; the visible cases are water
   (roughness 0.35), snow/ice and wet ground, so the sky probe is the priority, not SSR.

---

## 8. Terrain shading specifics

1. **Regolith BRDF (P1).** Lambert regolith (`chunk_draw.wgsl:1588-1595`, f0 0.03) cannot produce the
   flat full Moon or its opposition surge. Add `SURFACE_FLAG_REGOLITH` (gbuffer extra.a already carries
   flags) and evaluate Lunar-Lambert (McEwen 1991: rho/pi x [2 L(alpha) mu0/(mu0 + mu) + (1 - L(alpha)) mu0])
   with the Hapke opposition term B(alpha) = B0 / (1 + tan(alpha/2) / h). ~20 ALU per lit pixel.
   With it, use real albedos (lunar highlands ~0.12-0.2, mare ~0.07, linear); today's 0.27 regolith
   partly compensates the missing exposure.
2. **Unresolved relief (P1).** Slope normals for sub-pixel steps exist (`hit_face`), but the lost
   micro-relief is not turned into roughness or micro-shadowing, so the terminator and low-sun terrain
   look flat and too bright at distance (and differ from near terrain). The generator knows its octave
   table: sigma^2 = sum over unresolved octaves k of (m_k c A_k / lambda_k)^2 (m_k the local amplitude
   modulation: region masks, erosion fade; c the noise's RMS-slope constant); add it to alpha^2 and apply a Smith-G micro-shadowing factor to the diffuse term
   (LEAN/LEADR, Dupuy et al. 2013; Bruneton & Neyret 2010 for the geometry-to-BRDF hand-off).
3. **Albedo prefiltering (P2).** Noise contributions fade with footprint (good) but toward `t = 0.52`
   through a piecewise palette mapping, so the far colour is f(E[t]) rather than E[f(t)]; precompute the
   mean palette colour per material and fade toward it. sRGB->linear uses pow 2.2
   (`engine.rs:91`); use the exact sRGB curve (dark-material error up to ~10 %).
4. **Materials coverage (P2).** The atmosphere's `ground_albedo` (0.3) should equal the planet's mean
   albedo so the multi-scattering ground bounce and the analytic ground seen from orbit match the
   rendered terrain.

---

## 9. Prioritised plan

| Step | Pri | Work | Files | Cost |
|---|---|---|---|---|
| 1 | P0 | Planet body owns atmosphere and radius; derived sky row; `has_air` split; space sun disk | Pulsar voxel/atmosphere components, `helio-component/.../atmosphere_component.rs`, `atmosphere_luts.wgsl:44-91`, composite | ~0 ms |
| 2 | P0 | Sky irradiance table E(r, mu_s) (symmetric L2) + transmittance table in a uniform; use in deferred, forward, transparent, fog | `atmosphere_luts.wgsl`, `atmosphere_common.wgsl`, `deferred_lighting.wgsl:1324` | 0.1 ms build on change, ~0.05 ms/frame |
| 3 | P0 | Physical camera EV100 + pre-exposure + histogram metering with sky/space mask | `postprocess.wgsl`, `gpu_types.rs`, all lighting writers, Pulsar light units | ~0.05 ms |
| 4 | P1 | Gate voxel visibility to voxel pixels; walks for mesh pixels | `deferred_lighting.wgsl:1261`, `chunk_draw.wgsl` | 0.1-0.5 ms |
| 5 | P1 | Grading fixes (saturation, contrast, LUT on display-referred, Bradford WB) | `postprocess.wgsl:797-837` | ~0 ms |
| 6 | P1 | Soft sun via cone jitter in walks | `chunk_draw.wgsl:1918-1922, 2023-2057` | ~0 ms |
| 7 | P1 | Terrain bounce term; fixed-radius sky visibility + coarse terrain visibility | `deferred_lighting.wgsl`, `chunk_draw.wgsl:1928-1951`, generation | ~0-0.1 ms |
| 8 | P1 | Runtime sky reflection cube | new capture kernel in helio-pass-sky, deferred binding | 0.1-0.2 ms |
| 9 | P1 | Regolith BRDF + unresolved-relief roughness/micro-shadowing | `chunk_draw.wgsl` publish, `pbr_eval.wgsl`, deferred | ~0.02 ms |
| 10 | P1 | Tangent-centred space march; LUT/march blend band | `atmosphere_composite.wgsl` | 0 to -0.2 ms |
| 11 | P2 | Rest: RGB aerial transmittance, LUT caching, Lagarde SO, bent normals, height-correlated Smith, multi-scatter GGX, AgX/ACES 2, thresholdless bloom, CSM texel-scaled bias and directional PCSS, multi-body atmospheres, stars/night, traced one-bounce GI | various | see sections |

Validation (cheap, worth doing first): a false-colour EV100 debug view; fixed test views (ground noon,
sunset, 400 km orbit day side, terminator, moon from orbit) with expected luminances, e.g. sunlit
lunar highlands (albedo 0.15, sun 45 deg, 120 klux): L ~ 0.15 x 120,000 x 0.707 / pi ~ 4,000 cd/m^2
(EV100 ~ 15); clear zenith sky at mid sun ~ 2,000-8,000 cd/m^2; and a check that the irradiance table
equals the eye probe at the eye's (r, mu_s).

---

## 10. References (summarised in my own words)

- S. Hillaire, "A Scalable and Production Ready Sky and Atmosphere Rendering Technique", EGSR 2020 /
  CGF 39(4): transmittance, multiple-scattering (psi_ms), sky-view and aerial LUTs; per-pixel marching
  outside the atmosphere; sun illuminance defined above the air.
- E. Bruneton, F. Neyret, "Precomputed Atmospheric Scattering", EGSR 2008; E. Bruneton, "A Qualitative
  and Quantitative Evaluation of 8 Clear Sky Models" (2016) and the 2017 open-source reimplementation:
  4D scattering table usable from any altitude, ground irradiance texture E(r, mu_s).
- S. Lagarde, C. de Rousiers, "Moving Frostbite to Physically Based Rendering", SIGGRAPH 2014 course:
  photometric units, EV100 camera model (exposure = 1/(1.2 x 2^EV100)), specular occlusion from AO,
  sun as an area light.
- B. Karis, "Real Shading in Unreal Engine 4", SIGGRAPH 2013: split-sum IBL, Schlick-GGX k remap,
  representative-point area lights.
- Unreal Engine 5 documentation: Auto Exposure (histogram, low/high percent, EV100 min/max,
  exposure compensation curve), Physical Lighting units, Sky Atmosphere and real-time Sky Light capture,
  pre-exposure.
- J. Jimenez, X. Wu, A. Pesce, A. Jarabo, "Practical Real-Time Strategies for Accurate Indirect
  Occlusion", SIGGRAPH 2016: GTAO, bent normals, GTSO specular occlusion.
- E. Heitz, "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs", JCGT 2014;
  C. Kulla, A. Conty, "Revisiting Physically Based Shading at Imageworks", 2017 (energy compensation).
- A. McEwen, "Photometric functions for photoclinometry and other applications", Icarus 1991
  (Lunar-Lambert); B. Hapke, "Theory of Reflectance and Emittance Spectroscopy" (opposition effect).
- M. Olano, D. Baker, "LEAN Mapping", I3D 2010; J. Dupuy et al., "Linear Efficient Antialiased
  Displacement and Reflectance Mapping", SIGGRAPH Asia 2013; E. Bruneton, F. Neyret, N. Holzschuch,
  "Real-time Realistic Ocean Lighting using Seamless Transitions from Geometry to BRDF", CGF 2010.
- J. Jimenez, "Next Generation Post Processing in Call of Duty: Advanced Warfare", SIGGRAPH 2014 (bloom).
- AgX (T. Sobotka; Blender 4.0 default view transform); Khronos PBR Neutral tone mapper (2024).
