# Native HLFS pixel-coordinate fast path, 2026-09-23

At `sample_scale == 1`, the four-rooks offset in `sample_pixel` is always zero: `offset % 1 == 0`, so the old function returned `min(p, screen_size - 1)`. The shader now returns that value before computing a phase or modulo. Half-size shading retains the old path. This is a GPU shader change, with no CPU image/light processing, geometry reduction, candidate reduction, or visibility change.

The local NVIDIA GeForce RTX 3060 used Vulkan GPU timestamps. These are medians after 16 warmup frames; they are not presented FPS. The gallery and cathedral captures move the camera along their fixed paths. A/B/B/A pairs used fixed release executables with identical capture settings. The cathedral and dynamic-probe control executables precede a trivial Rust field refactor in the compact-light commit; their sampling shaders and GPU work were unchanged.

## Native 1440p full graph

Both output and HLFS shading are 2560 × 1440, one sample, eight candidates, native TSR, and hardware RT. The gallery has 1,024 **static** lights and 38,400 triangles. The large cathedral has about 415,300 triangles, a sun plus 12 interior lights, and stained-glass colored transmission.

| Scene and run | Sampling | Temporal | HLFS | Full graph |
| --- | ---: | ---: | ---: | ---: |
| Gallery control A | 10.118 ms | 2.602 ms | 15.657 ms | 21.934 ms |
| Gallery fast path A | 9.595 ms | 2.517 ms | 14.886 ms | 21.244 ms |
| Gallery fast path B | 9.610 ms | 2.526 ms | 14.956 ms | 21.353 ms |
| Gallery control B | 10.242 ms | 2.625 ms | 15.663 ms | 22.074 ms |
| Cathedral control A | 2.053 ms | 1.182 ms | 6.378 ms | 14.491 ms |
| Cathedral fast path A | 2.051 ms | 1.085 ms | 5.969 ms | 14.338 ms |
| Cathedral fast path B | 2.052 ms | 1.087 ms | 6.161 ms | 14.359 ms |
| Cathedral control B | 2.062 ms | 1.188 ms | 6.352 ms | 14.455 ms |

All four large-cathedral 1440p checkpoints matched byte-for-byte between the control and candidate pair, including moving-camera frames 31, 63, and 99. The gallery's frame 0 matched; later noisy frames differed even between control runs. I inspected full-size frames and found the same scene structure with existing colored mottling. [Gallery frame 99](gallery-candidate-f099.png) and [cathedral frame 99](cathedral-candidate-f099.png) are retained for inspection.

## 4K, screen-space, and moving-light controls

At native 3840 × 2160 cathedral output and HLFS shading, one 50-frame pair measured control/candidate HLFS 14.311/13.968 ms and full graph 30.768/30.444 ms. Its three checkpoints matched byte-for-byte; [frame 49](cathedral-4k-candidate-f049.png) is retained. At 1440p output with half-size screen-space HLFS shading, control/candidate HLFS was 3.040/3.047 ms and full graph 11.239/11.141 ms. Its four checkpoints also matched. The 4K and screen-space pairs are single pairs, so their small timing differences should not be overinterpreted.

The independent native-1440p RT probe has 1,024 **moving** lights, 10,000 moving instances of a 100-triangle mesh, 120 warmup and 600 measured frames. It excludes the full renderer and reports TLAS separately. Its A/B/B/A medians were:

| Run | TLAS + HLFS | HLFS only |
| --- | ---: | ---: |
| Control A | 10.470 ms | 10.286 ms |
| Fast path A | 10.372 ms | 10.190 ms |
| Fast path B | 10.294 ms | 10.110 ms |
| Control B | 10.494 ms | 10.311 ms |

The 96-frame native-shading seed-11 moving-camera/light/blocker fixture remains visually unacceptable. Its unchanged static display-change gate is `< 0.010`; control RMS was 0.03497 and candidate RMS 0.03507. The grain gate also failed in both. Inspected frames [63](seed11-spp1-c8-f63-sampled.png), [64](seed11-spp1-c8-f64-sampled.png), [80](seed11-spp1-c8-f80-sampled.png), and [81](seed11-spp1-c8-f81-sampled.png) visibly flicker. The test process returned a failure after writing its metrics and frames, as intended. Full sequences remain under `target/validation/sample-pixel-motion-native-{control,candidate}/` on the validation host. The moving-light RT probe's near-uniform pictures are weak scene-level visual evidence; its GPU timing scope is the useful result.

## Diagnostic sweep and verification

One same-binary moving-camera gallery sweep at native 1440p isolated the sampling cost. With two, four, eight, and sixteen candidates, sampling medians were 9.339, 9.768, 10.415, and 11.324 ms. Removing shadow policy at two and eight candidates measured 8.825 and 9.961 ms. Thus the large residual sampling cost is not explained by candidate count or shadow traversal alone. These are diagnostic single runs; they do not justify reducing the accepted candidate or shadow settings. Raw CSVs and capture configurations are included.

The current source passed 17 release Vulkan hardware RT tests and 14 release screen-space GPU tests. The native gallery improvement repeats at full-graph scope. The retained change does **not** meet the 3–4 ms native-1440p HLFS goal, does not fix moving-light flicker, and does not make PR #248 ready for review.

## Rejected exact-coordinate follow-ups

I tried keeping the temporal filter's halo positions in view space to avoid transforming every neighbor to world space. The native 1440p gallery A/B/B/A full-graph medians were control 21.206, candidate 21.251, candidate 21.251, control 21.207 ms. Temporal-stage medians changed by only about 0.02 ms. No whole-graph win appeared, so the change was removed.

I also tested an explicit native branch in `sample_position_from_uv` that skips division by one. Gallery full-graph medians were control 21.351, candidate 21.260, candidate 21.255, control 21.371 ms, but the large cathedral was control 14.403, candidate 14.379, candidate 14.452, control 14.428 ms, with candidate HLFS medians about 0.25 ms higher. That is not a convincing cross-scene gain; this branch was removed too. The raw GPU CSVs and configurations for both experiments are in this folder. Neither rejected shader change is in the PR.
