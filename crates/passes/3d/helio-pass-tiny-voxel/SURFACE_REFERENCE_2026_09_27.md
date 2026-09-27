# Canonical surface appearance reference

> Correction: these historical captures used a compute stream submitted before
> graphics. Their lighting/albedo capture was therefore not paired with the
> current primary hits. Do not use their appearance errors or joint
> lighting/geometry checks as acceptance evidence. Primary hit-only checks and
> standalone traversal timings retain their stated scope. See
> [the corrected appearance experiment](APPEARANCE_2026_09_27.md).

The research audit's next step is implemented as an opt-in, offline reference.
It integrates separately lit canonical voxel samples in linear HDR through
Helio's deferred graph. It does not introduce a production surface cache or
change the default far representation. The planet quality/performance goal
remains open.

## What runs

`helio-default-graphs/voxel-reference` enables canonical far occupancy plus a
separate ray-preparation dispatch. The flight example captures pre-AA lighting,
GBuffer albedo, directional visibility, canonical hits and the actual GPU camera.
It keeps six axis-face contributions and joint face/material coverage rather
than shading an averaged normal. The existing engine material model still
applies its own footprint filtering; these are references for that model,
not an independent physical-lighting oracle.

The 32 cases comprise eight fixtures, two sun directions and two camera
positions separated by 2.5 cm. Fixtures include a slope, horizon, carved
opening, thin voxel shell and destroyed shell, plus close views of the last
three. The shell is one authored voxel thick. The retained runs use 10 cm
voxels and 96x54 pixels. Larger grids and complete flights remain separate gates.

The graph supports final resource consumers through factories retained across
resize. Reading the GBuffer group and pre-AA output extends their lifetimes
through capture. A regression checks the actual texture size and that the
consumer survives resize at the graph's internal resolution.

## Corrections found while building the reference

- HDR exposed texture aliasing that reused a larger texture for a smaller
  attachment. The texture pool and alias planner now require matching extents,
  layer counts and mip counts. A GPU test reproduces the old color/depth size
  mismatch and checks that compatible same-size reuse still works. This fix is
  published separately as `2c649561`; it is a correctness fix, not a memory saving.
- Canonical rendering no longer uses the sampled far field's zero material-depth
  shortcut or quarter-pixel sunlight offset. The diagnostic sunlight replay
  uses the same offset rule.
- Combined ray generation/traversal produced a few-bit variation in directions
  with identical camera input. Actual GPU camera readback equalled the CPU
  upload. Some edge rays then selected another face. Separating ray preparation
  into a materialized buffer value removed the observed differences. The exact
  compiler/hardware cause is not established. This extra dispatch is restricted
  to the offline `surface-reference` feature. A CPU no-inline trial did not
  eliminate the issue and was removed.

## Retained evidence

Release build, Vulkan, NVIDIA RTX 3060, driver 616.64. These are diagnostic
captures, not uncontended performance measurements.

| Spatial samples per pixel | Cases | Rendered frames | Sampled CPU center rays |
| --- | ---: | ---: | ---: |
| 16 | 32 | 636 | 2,304 |
| 64 | 32 | 2,172 | 2,304 |
| 256 | 32 | 8,316 | 2,304 |

Every run completed GPU validation, settled-residency checks, repeated-frame
equality, GPU/CPU camera equality, and the sampled CPU coverage/first-cell/
material audit. Center hits and lighting are byte-identical between all three
sampling grids. Across the three runs, 55,738,368 subpixel samples had valid
primary results, with valid sunlight values on surface hits. The CPU rays repeat the same pose set;
they are not 6,912 independent camera configurations.

Linear RGB RMSE, first sun and first camera position:

| Fixture | One sample versus 256 | 16 versus 64 | 64 versus 256 |
| --- | ---: | ---: | ---: |
| Slope | 0.060086 | 0.014982 | 0.006494 |
| Horizon | 0.052304 | 0.013931 | 0.006893 |
| Carved opening | 0.045462 | 0.011189 | 0.005185 |
| Thin shell | 0.040405 | 0.011427 | 0.006295 |
| Destroyed shell | 0.040604 | 0.011478 | 0.006350 |
| Opening, close | 0.017872 | 0.004485 | 0.002410 |
| Thin shell, close | 0.013064 | 0.003357 | 0.001886 |
| Destroyed shell, close | 0.025327 | 0.006109 | 0.003169 |

Across all cases, the latter sampling difference is 1.78-2.44 times smaller
than the former, with a median factor of 1.93. This supports convergence but
does not prove that 256 samples are sufficient: maximum per-pixel discrepancies
remain much larger than the frame averages, especially at edges.

At 256 samples, 5,015 of 5,184 slope pixels mix multiple face directions.
The close thin shell mixes faces in only 49 pixels. Visual inspection of the
enlarged 64-sample contact sheet shows reduced slope stippling and preserved
resolved shell edges. Opening interiors still show voxel rows; these low-
resolution patches do not qualify the game's final appearance.

For the slope's 2.5 cm translation, raw RGB image delta RMSE is 0.044299 with
one sample and 0.002182 with the 256-sample mean. This includes real parallax
and is not a motion-compensated flicker measurement or a production speedup.

Focused checks: two texture-alias GPU tests, 27 default engine unit tests,
28 reference-feature unit tests, and the deferred graph/resize GPU test passed.
Each unit suite leaves the existing CPU selection benchmark ignored.

## Reproduce and continue

Use the commands in the [README](README.md), then repeat with
`HELIO_VOXEL_SURFACE_REFERENCE=16` and compare the 64- and 256-sample outputs.
All generated data remains local and ignored:

- `target/voxel-goal/surface-patches-v6-{4,8,16}/`
- `target/voxel-goal/surface-patches-v6-8/reference-contact-sheet.png`
- `target/voxel-goal/surface-{alias,reference,default,graph}-tests.log`
- Reference binary SHA-256:
  `7bd6b4757a8471a8cc4cae2dfa4536011cfa80ec6a356535e58b0b708cb02bba`.

The next implementation must approximate these lit face/coverage distributions
with bounded work. Start with a small exact surface-brick cache and compare a
derived face/material mixture on these fixtures, refining whenever projected
geometry error is too large. Keep conservative bounds separate from the
sampled depth intervals exported here. Measure construction, local edits,
residency memory and shading cost before extending it to the planet.

No cached representation, conservative silhouette guarantee, full movement
sequence, populated editor scene, arrival latency or performance target is
qualified by this checkpoint.
