# Native 1440p large cathedral investigation, 2026-09-23

The release capture used an RTX 3060 (Vulkan), the 415,300-triangle large cathedral, stained-glass RGB transmission, a daylight sun plus seven chandelier and five candle lights, native 2560 × 1440 output **and** HLFS shading, native TSR, two samples, two candidates, and 100 camera-moving frames. SSR was off. These are GPU timestamp medians after frame 16; the graph number covers the render graph, not presentation. The full configuration is in `capture-config.json`. This scene tests complex geometry and transmission, not 1,000 dynamic lights.

| Run | Sampling | Composite | HLFS total | Render graph |
| --- | ---: | ---: | ---: | ---: |
| Control 1 | 2.064 ms | 1.954 ms | 6.184 ms | 14.461 ms |
| Unshadowed diagnostic | 1.031 ms | 0.868 ms | 4.256 ms | 12.367 ms |
| Exact-tile fast path 1 | 2.012 ms | 1.949 ms | 6.143 ms | 14.575 ms |
| Exact-tile fast path 2 | 2.019 ms | 1.959 ms | 6.391 ms | 14.515 ms |
| Control 2 | 2.053 ms | 1.957 ms | 6.210 ms | 14.687 ms |

The unshadowed run changed only the lights' RT-shadow flag and is a cost diagnostic, not a valid visual alternative. It places roughly 1 ms of the sampling stage and 1 ms of the full-size composite in current-frame visibility. The native cathedral differs from the dense static gallery: this small light set already evaluates its local lights exactly, while the key daylight sun is traced at full size in composite. Simply reducing the reservoir setup cannot remove the visibility cost.

The fast-path prototype was an early GPU branch for transmitting native-resolution tiles with at most 32 lights. It skipped workgroup reservoir initialization, barriers, and feedback sorting while doing the same exact ray/lighting loop. In the captures, all seven stored candidate checkpoints matched control 1 byte for byte. Control 2 differed from control 1 by at most five color-channel values of one 8-bit level in the final three checkpoints. The candidate saved approximately 0.04–0.05 ms in sampling but gave **no repeatable HLFS or render-graph gain**. Its shader code was removed. Raw CSVs for every run are in this folder.

The three full-resolution images here show camera movement through the nave at frames 31, 63, and 99. I inspected the sunlit floor, pew silhouettes, glazing, and pillar boundaries at these checkpoints. The compared candidate frames are byte-identical to the corresponding control 1 frames, so this rejected change adds no visible difference there. Checkpoints and hashes cannot certify smooth motion between them; the pre-existing moving-light fixture's flicker gate also remains red. The local validation host retains all captured checkpoints and logs under `target/validation/large-cathedral-native-1440-*`.

The baseline serialized capture median (CPU setup, renderer call, and GPU wait, excluding PNG readback) was 21.207 ms. That is about 47.1 serialized capture frames/s. The graph's 14.461 ms corresponds to about 69.2 GPU-bound frames/s in isolation; neither is a measured presentation FPS. The target 3–4 ms native HLFS and smooth 1,000-dynamic-light interaction are still unmet.

Reproduce with `cargo build --release -p examples --bin indoor_cathedral_hlfs`, then set `HLFS_RT=1`, `HLFS_PRESAMPLED=1`, `HLFS_TSR_NATIVE=1`, `HLFS_SAMPLE_SCALE=1`, `HLFS_SAMPLE_COUNT=2`, `HLFS_CANDIDATE_COUNT=2`, `HLFS_RESOLUTION=1440p`, `HLFS_CAPTURE_FRAMES=100`, `HLFS_CAPTURE_TAIL=4`, `HLFS_CAPTURE_TIMINGS=1`, and `HLFS_GRAPH_TIMINGS=1`. Run `target/release/indoor_cathedral_hlfs.exe --capture-large <directory>`. Add `HLFS_UNSHADOWED=1` only for the shadow-cost diagnostic. Run captures in isolation after building.

## Retained full-size key-light calculation

The composite shader previously calculated the same key-light incident twice: once to check if it could illuminate the receiver and once during BRDF evaluation. It now passes the calculated incident to BRDF evaluation, leaving ray visibility, the light set, and the illumination check unchanged. The A/B/B/A release captures used the same large-cathedral settings and GPU:

| Run | Composite | HLFS total | Render graph |
| --- | ---: | ---: | ---: |
| Control 1 | 1.968 ms | 6.468 ms | 14.632 ms |
| Candidate 1 | 1.891 ms | 6.384 ms | 14.602 ms |
| Candidate 2 | 1.884 ms | 6.108 ms | 14.437 ms |
| Control 2 | 1.965 ms | 6.417 ms | 14.553 ms |

The composite stage improves by about 0.08 ms in both repetitions; the full render-graph gain is small relative to run variation. Seven stored moving-camera cathedral checkpoints in candidate 1 are byte-identical to control 1. Candidate 2 matches through frame 63; its last four checkpoints differ in at most 46 channel values, by no more than two 8-bit levels. I inspected the full-size control frames 31, 63, and 99. This is a narrow, retained GPU shader improvement, not a solution to the visibility cost or target frame time.

All 17 non-benchmark hardware-RT regression tests passed with `cargo test --release -p helio-pass-hlfs --test gpu_hlfs_rt -- --include-ignored --skip benchmark_ --test-threads=1`. The fixed seed-11 half-size moving-light/camera/blocker fixture is **still red**. Its motion RMS was 0.03511 for the candidate and 0.03485/0.03459 for two control runs, against a 0.01000 gate. Identical-source control runs also differed broadly at pixel level in saved noisy frames. These runs therefore do not establish either a new dynamic-light visual regression or a visual pass. The three motion CSVs and frame-63 images are preserved here; the full 96-frame sequences remain on the validation host under `target/validation/moving-key-*-seed11/`. Investigating that run-to-run variation and the flicker gate is still required before moving the PR out of draft.

A separate 50-frame native 3840 × 2160 large-cathedral stress pair used the same settings and 16-frame warmup. Control/candidate composite medians were 4.628/4.500 ms; HLFS totals 14.345/14.194 ms; render graph 30.788/30.510 ms. This is one pair, so it checks for a large regression rather than proving a stable 4K speedup. All six captured 4K checkpoints were byte-identical. I inspected frame 49 at full source resolution for geometry, floor tint, and shadow placement. Its local capture and the other checkpoints remain under `target/validation/key-4k-{control,candidate}/`; the 4K configuration and raw timing CSVs are included here.
