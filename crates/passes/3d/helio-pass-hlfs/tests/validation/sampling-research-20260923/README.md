# Dense-light sampling investigation, 2026-09-23

The parked technology gallery remains the light-count diagnostic: 1,024 **static** independently shadowed point lights and 38,400 triangles. The existing [full-graph capture](../static-gallery-fullgraph-20260923/README.md) places most native-1440p HLFS cost in the sampling stage. Its unshadowed control left that cost nearly unchanged, so reducing shadow rays or geometry alone does not address the measured bottleneck. That gallery has camera motion, but its lights do not move; it cannot fulfill the dynamic-light acceptance gate.

## Rejected GPU experiments

I prototyped an opt-in same-frame spatial reservoir pass entirely on the GPU. It read current light choices from neighboring shading pixels, checked normal and depth compatibility, re-evaluated the selected lights at the receiver, and traced final visibility. The prototype compiled and passed an existing Vulkan RT smoke test. With the fixed moving-camera, moving-light, blocker, and key-switch fixture (seed 11, 1 sample, 8 candidates, half-size HLFS shading), static display-change RMS increased from 0.0117 to 0.0223 against the 0.0100 gate. The recorded frames show more colored noise. Both configurations remain red; the prototype was removed. Its extra pass also added work, so no speed claim follows from it. See the two motion CSVs and frames 63, 64, 80, and 95 in this folder. The complete 96-frame sequences and quality/grain CSVs remain under `target/validation/moving-{temporal-only,spatial-ris}-seed11/` on the validation host.

A second experiment widened the reactive temporal luminance clip from 5% to 10% in `temporal.wgsl`. Four development seeds passed the unchanged quality, motion, and grain gates. On the separate held-out seeds, seed 211 missed the motion gate (0.01057). This was also reverted, without tuning to the held-out result. The fixed seed lists and thresholds are in `gpu_hlfs_rt.rs`; development and held-out motion CSVs are preserved here. This is evidence that half-size sampling is close on the small planar fixture, **not** a visual acceptance of the full-resolution gallery or cathedral.

No renderer or test code from those two rejected experiments was retained. PR #248 remains a draft.

## Small retained spatial-stage improvement

The existing GPU spatial filter evaluates `exp(-distance²/radius²)` at every accepted neighbor. Its radius is 1 or 2, and its offsets are integer pixels, so the set of possible factors is finite. The shader now uses f32-rounded constant factors for those distances. It keeps the same filter footprint and geometry checks. This does not change the dense-light sampling algorithm.

The same pre-change release binary and rebuilt candidate were captured in control/candidate/candidate/control order on the idle RTX 3060. Both used the full 1,024-static-light gallery, native 2560x1440 output and HLFS shading, 1 sample, 8 candidates, native TSR, and 100 camera-moving frames. The first 16 HLFS rows were excluded as warmup. These are GPU timestamps, not presentation FPS.

| Run | Spatial p50 | Full graph p50 |
| --- | ---: | ---: |
| Control 1 | 1.683 ms | 23.729 ms |
| Candidate 1 | 1.638 ms | 23.631 ms |
| Candidate 2 | 1.642 ms | 23.690 ms |
| Control 2 | 1.684 ms | 23.846 ms |

Raw stage and graph CSVs are included here. Captured frame 0 was byte-identical between control and candidate; frame 99 and selected moving-camera checkpoints were visually inspected without finding a new shape, shadow, or reflection artifact. The pre-existing colored mottling remains. The reduced-resolution hardware-visibility and colored-transmission GPU regression tests pass. On the 96-frame dynamic-light fixture the pre-existing flicker gate remains red: control 0.01171, candidate 0.01179 against 0.01000. This change is a small stage-level speedup, not completion of the visual or overall performance target.

## Next implementation target

The next candidate should address the expensive light scoring and random light-data access in initial sampling, while keeping the light count, geometry, RT visibility, and output resolution fixed. NVIDIA's [RTXDI integration](https://github.com/NVIDIA-RTX/RTXDI/blob/main/Doc/Integration.md) describes a GPU RIS buffer that may also carry compact light data for locality, and separate initial, temporal, and spatial resampling stages. Its [application bridge](https://github.com/NVIDIA-RTX/RTXDI/blob/main/Doc/RtxdiApplicationBridge.md) requires consistent target weights, correct current/previous light identity, and visibility policy; its [noise and bias guide](https://github.com/NVIDIA-RTX/RTXDI/blob/main/Doc/NoiseAndBias.md) explains why a naive spatial merge can add noise or bias. These are design references, not measured Helio improvements.

First profile a compact GPU light-record path or a specialized initial sampler against the unchanged same-binary control. Keep all light types and colored transmission correct. Then require a whole-graph gain in the 1,024-light gallery, passing dynamic-light/camera visual sequences, and full-size inspection before promoting it. Nsight Systems GPU counter collection on this RTX 3060 reported insufficient privilege, so the available evidence is per-stage GPU timestamps rather than occupancy or cache counters.

The user's GitHub access also allowed inspection of Unreal Engine's private [MegaLights sampling shader](https://github.com/EpicGames/UnrealEngine/blob/396c9f059903aed5fec78ecd3d437a40c6415368/Engine/Shaders/Private/MegaLights/MegaLightsSampling.usf), [shading shader](https://github.com/EpicGames/UnrealEngine/blob/396c9f059903aed5fec78ecd3d437a40c6415368/Engine/Shaders/Private/MegaLights/MegaLightsShading.usf), and [temporal denoiser](https://github.com/EpicGames/UnrealEngine/blob/396c9f059903aed5fec78ecd3d437a40c6415368/Engine/Shaders/Private/MegaLights/MegaLightsDenoiserTemporal.usf). MegaLights writes packed light choices and ray descriptions from sampling, then shades from those results in a separate pass; it also tracks visible and hidden light history and adjusts temporal accumulation by confidence. This motivated the split experiment below. No Unreal code was copied into Helio, and this source review is not performance evidence for Helio.

## Split-pipeline falsification

An opt-in selection / visibility split was implemented for the existing half-resolution, one-sample temporal RIS mode. The first GPU pass selected and packed a light choice; a second GPU pass traced its current-frame visibility and shaded it. Exact small sets and glossy pixels kept the existing path. The prototype compiled, ran on the RTX 3060, and preserved the fixed seed-11 moving fixture's final-output and grain gates. Its pre-existing static flicker gate stayed red (0.01179 versus the control's 0.01171, threshold 0.01000). Frame 0 of the full graph capture was byte-identical; inspected frame 99 had the same visible colored mottling.

The 1,024-**static**-light gallery was captured at 2560x1440 output with 1280x720 HLFS shading, native TSR, one sample, eight candidates, and unchanged scene geometry. The A/B/B/A runs show no repeatable whole-graph or HLFS win:

| Run | Sampling GPU p50 | HLFS GPU p50 | Full graph GPU p50 |
| --- | ---: | ---: | ---: |
| Control 1 | 3.357 ms | 5.993 ms | 12.551 ms |
| Split 1 | 3.318 ms | 6.209 ms | 12.394 ms |
| Split 2 | 3.432 ms | 6.041 ms | 12.576 ms |
| Control 2 | 3.320 ms | 6.238 ms | 12.376 ms |

Raw GPU timestamp CSVs and the motion result are included here; the full image sequences remain under `target/validation/split-*-half-1440/` and `target/validation/moving-split-seed11/` on the validation host. The split implementation was removed. A broader, more specialized selection and traversal architecture may still help, but the simple extra-pass version did not. This experiment cannot support a claim about native-resolution performance or 1,000 dynamic lights.

## Rejected deterministic prefix lookup

The coarse GPU light proposal uses atomic appends while building its alias table. That makes its slot order a possible source of run-to-run changes in the noisy moving fixture. I replaced the alias construction with an eight-step parallel prefix table and selected strata by binary search. This was a GPU-only prototype with the same proposal weights and uniform discovery component. It made the coarse stage faster but added random proposal-table reads to every dense-light sample.

In the 1,024-**static**-light gallery at native 2560 × 1440 output/shading, one sample, eight candidates, and the unchanged moving camera, the control/candidate GPU medians were coarse 0.252/0.213 ms, sampling 12.108/13.001 ms, HLFS 17.565/18.366 ms, and full graph 23.788/24.700 ms. These are one release-binary pair after 16 warmup frames, with raw CSVs here. Frame 99 was inspected and retained the visible colored mottling. Since the whole graph regressed by about 0.9 ms, the code was removed without spending more runs on the prototype. This does not prove atomic order is the only cause of the moving fixture's run-to-run variance.
