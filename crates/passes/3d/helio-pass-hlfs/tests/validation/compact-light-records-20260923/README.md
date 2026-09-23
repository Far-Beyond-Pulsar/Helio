# GPU compact light records, 2026-09-23

The RTX 3060 profiling pointed to repeated random light-record reads in dense-light sampling. SceneDB keeps its authoritative 128-byte rows. HLFS now copies their first 64 bytes into a derived GPU storage buffer once per frame and reads that buffer in its light-selection and shading passes. The copy is GPU compute; there is no CPU light list or reduced light count. The removed fields are not read by HLFS. Sparse SceneDB rows remain in their original positions, preserving light IDs and temporal lookup.

The measured candidate preceded a small Rust refactor that stores the current light count in a separate field; the buffer layout, copy dispatch, and shader stayed the same. The source after that refactor was compiled again. The A/B/B/A figures below compare fixed release executables from immediately before the refactor. All GPU results are Vulkan timestamps on the local NVIDIA GeForce RTX 3060; they are not presentation FPS. Full graph rows include more than HLFS, while the dynamic RT probe is an isolated TLAS plus HLFS fixture.

## Native 1440p, 1,024 static lights

The technology gallery has 38,400 triangles and a moving camera. Both 2560 × 1440 output and HLFS shading were native resolution, with one sample, eight candidates, native TSR, and 100 frames. GPU medians exclude the first 16 warmup frames.

| Fixed binary run | Sampling | HLFS | Full graph |
| --- | ---: | ---: | ---: |
| Control A | 12.544 ms | 17.974 ms | 24.399 ms |
| Compact A | 10.218 ms | 15.655 ms | 22.171 ms |
| Compact B | 10.239 ms | 15.692 ms | 22.074 ms |
| Control B | 12.503 ms | 18.080 ms | 24.390 ms |

The improvement repeats in the full graph. This scene's lights are static; camera motion alone does not meet the dynamic-light gate. Frame 0 was byte-identical. Later noisy frames are not bitwise deterministic. I inspected frames 31 and 99 at full size and found the same scene structure with the existing colored mottling; `gallery-candidate-f099.png` is a checkpoint, not a visual acceptance claim.

## Moving lights and instanced geometry

The independent native-1440p RT probe updated 1,024 lights and 10,000 instances of a 100-triangle mesh every frame: one million instanced triangles. It used one sample, eight candidates, presampling, reactive history, 120 warmup and 600 measured frames. The CSVs record each measured frame and distinguish TLAS time, HLFS time, and serialized wall time. This fixture excludes the full renderer.

| Fixed binary run | TLAS + HLFS median | HLFS median | TLAS + HLFS p95 | Serialized wall median |
| --- | ---: | ---: | ---: | ---: |
| Control A | 11.199 ms | 11.016 ms | 11.632 ms | 18.527 ms |
| Compact A | 10.706 ms | 10.521 ms | 11.211 ms | 17.569 ms |
| Compact B | 10.844 ms | 10.658 ms | 11.739 ms | 17.632 ms |
| Control B | 11.304 ms | 11.120 ms | 12.401 ms | 18.034 ms |

The 96-frame seed-11 moving camera/light/blocker/key-switch fixture still fails its unchanged static display-change gate: candidate RMS 0.03480 against the 0.01000 threshold. Two control runs measured 0.03485 and 0.03459. Inspected frames 63, 64, 80, and 81 show colored flicker; sampled checkpoints and metrics are included here. The full sequence and RT-probe screenshots remain under `target/validation/moving-compact-seed11/` and `target/validation/compact-dynamic-*/` on the validation host. The RT probe screenshots are nearly uniform surfaces, so its timing is useful but its pictures are weak evidence of scene-level visual quality.

## Cathedral and 4K controls

The large cathedral has about 415,300 triangles, stained-glass transmission, daylight sun, and 12 interior lights. It has moving-camera checkpoints. At native 1440p output and shading, 100 frames with 16 warmup frames excluded:

| Fixed binary run | HLFS median | Full graph median |
| --- | ---: | ---: |
| Control A | 6.109 ms | 14.423 ms |
| Compact A | 6.413 ms | 14.534 ms |
| Compact B | 6.334 ms | 14.477 ms |
| Control B | 6.130 ms | 14.505 ms |

The extra copy costs a few tenths of a millisecond inside HLFS for a small light set; whole-graph differences here are within run noise. At native 3840 × 2160, one 50-frame pair measured control/candidate HLFS 14.102/14.160 ms and full graph 30.314/30.373 ms. Six 4K image checkpoints were byte-identical. In the cathedral's 1440p output, half-size screen-space fallback, a pair measured HLFS 3.359/3.367 ms and full graph 11.665/11.536 ms, with four byte-identical checkpoints. The native cathedral's frames 0, 31, 63, 96, and 97 were byte-identical; frames 98 and 99 differed slightly. `cathedral-candidate-f099.png` shows the retained candidate image. Colored transmission and structure are visible, but the full motion/flicker acceptance gate remains open.

## Verification and limits

- Release Vulkan hardware RT suite: 17 passed (`gpu_hlfs_rt`, benchmark tests excluded).
- Release screen-space GPU suite: 14 passed (`gpu_hlfs`, benchmark tests excluded).
- All run CSVs and capture configurations named in the tables are included in this folder. The `compact-dynamic-*.log` files preserve the probe setup and printed summary.
- The dense-gallery gain does not reach the native-1440p HLFS 3–4 ms target. The moving-light probe is still above 10 ms HLFS, the cathedral has no repeatable full-graph speedup, and the motion gate is red. This optimization is retained as a dense-light improvement, with no claim that PR #248 is ready for review.

## Rejected follow-up

I also tried skipping unused workgroup scratch initialization in the tile-presampled sampling shader. This keeps the sampling logic unchanged, but four alternating runs showed no repeatable HLFS gain:

| Run | Sampling median | HLFS median | Full graph median |
| --- | ---: | ---: | ---: |
| Control A | 10.230 ms | 15.639 ms | 22.042 ms |
| Candidate A | 10.014 ms | 15.612 ms | 21.924 ms |
| Candidate B | 10.181 ms | 15.675 ms | 21.981 ms |
| Control B | 10.191 ms | 15.745 ms | 22.035 ms |

The same native 1440p static gallery and moving camera were used. Frame 0 matched across all four runs; later frames varied even between control runs. The raw timestamp CSVs are included. The shader experiment was reverted.
