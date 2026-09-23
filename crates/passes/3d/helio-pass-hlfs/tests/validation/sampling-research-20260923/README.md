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
