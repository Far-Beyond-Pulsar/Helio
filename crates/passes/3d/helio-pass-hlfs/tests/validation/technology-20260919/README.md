# Technology gallery: rejected visual baseline

This new workload contains 1,024 independently shadowed local lights in a 48 x 68 m hall, with matching emissive fixtures, metallic display cylinders of differing roughness, structural bays and colored accents. The point sources sit below the opaque fixtures; the initial inside-fixture placement was corrected after a black lighting capture. Geometry is batched into eight materials. This is a diagnostic development scene, not a completed reflection showcase.

1440p, RTX 3060 / Vulkan, presampled RT, FXAA, 100 moving-camera frames, 16 warmup and 84 measured. Single runs. HLFS-only timing excludes TLAS, fog, other passes and CPU.

| Path | HLFS median ms | P95 ms | Display RGB NRMSE frames 31 / 63 / 99 |
| --- | ---: | ---: | --- |
| Presampled | 3.174 | 4.024 | 13.87 / 13.54 / 13.03% |
| All-light reference | 116.021 | 119.334 | reference |

**Visual gate fails:** severe colored sampling variance on walls and metal. Reference renders smooth direct-light gradients. The performance figure does not make the noisy result acceptable. Metrics use post-tonemap RGB, not linear energy. The current HLFS graph still lacks reflection tracing/composition; black polished cylinders are not a verified reflection result. No realistic texture assets are present. The next work is variance control and reflection integration, followed by motion/disocclusion and 4K validation.

Build: `cargo build --release -p examples --bin technology_gallery_hlfs`.
Capture: set `HLFS_RT=1`, `HLFS_PRESAMPLED=1`, `HLFS_RESOLUTION=1440p`, `HLFS_FXAA=1`, `HLFS_CAPTURE_TIMINGS=1`, then run `target/release/technology_gallery_hlfs.exe --capture target/technology`. Add `HLFS_REFERENCE=1` for the all-light reference. Both 100-frame captures complete; release build passes. No interactive-viewer acceptance yet.

![Rejected sampled result](sampled.png)
![Direct-light reference](reference.png)

## Variance experiments

Added `HLFS_CANDIDATE_COUNT` to the capture harness to vary candidate scoring independently of shadow rays. The default remains unchanged. Same scene, camera, reference and capture protocol as above:

| Experiment | HLFS median ms | P95 ms | Display RGB NRMSE 31 / 63 / 99 |
| --- | ---: | ---: | --- |
| 2 rays, 8 candidates | 3.652 | 4.352 | 11.53 / 11.45 / 11.04% |
| 2 rays, 16 candidates | 4.028 | 4.631 | 11.35 / 11.16 / 10.69% |
| 4 rays, 8 candidates | 4.533 | 5.022 | 10.04 / 9.79 / 9.34% |
| Wider temporal bounds, 2 rays / 2 candidates | 3.175 | 3.770 | 14.65 / 14.49 / 13.54% |

The wider-history experiment removed the fixed 5% luminance bound and used sample standard deviation instead of standard error for clipping. It worsened error and was reverted. Increasing candidates helps modestly, but four rays exceed the budget while retaining obvious mottling. None passes the visual gate, and no production defaults changed. This rules out solving the current scene through these simple parameter increases alone; sampling/reuse needs further work.

## Stationary-camera control

`HLFS_FIXED_CAMERA=1` holds the final camera pose from frame zero, and the capture harness now also saves its final frame for runs longer than 100 frames. A 400-frame default two-ray run remains at display-RGB NRMSE 13.05 / 12.97 / 12.99 / 12.91% at frames 31 / 63 / 99 / 399 against the prior final-pose reference. Camera motion alone therefore does not explain the variance floor. Fog temporal history is not identical between the moving and fixed reference paths, so these are diagnostic image differences, not a strict isolated lighting-energy test.

A temporary composite visualization displayed stored history age divided by the configured maximum while leaving temporal processing in normal mode. Broad surfaces reached the configured age; low-age pixels clustered around edges. The diagnostic shader was removed and the normal binary rebuilt. This rules against a wholesale history reset as the main cause; it does not prove every reprojection or lighting-change case correct.

![Stationary frame 399, still rejected](static-399.png)
![History age diagnostic, white indicates mature history](history-age.png)


## History clipping isolation and rejected variance bounds

A diagnostic replaced only `history_color` in `filtered_history` with `max(history.rgb, vec3<f32>(0.0))`, retaining the existing age/blend and geometry rejection. With the stationary camera it reduced frame-399 RGB NRMSE from 12.91% to 6.01%. This implicates clipping/reset behavior as a substantial contributor to mottling; it does not prove the sampler unbiased or justify disabling rejection in production.

A second experiment replaced the fixed 5% luminance cap with a conservative scalar floor for all YCoCg bounds: `3 * sqrt(max(history_second_moment - history_luminance^2, 0) / max(age, 1))`. The exact rejected patch is preserved alongside these results. Both changes were reverted.

| Diagnostic | Final display RGB NRMSE | HLFS median / p95 ms | Outcome |
| --- | ---: | ---: | --- |
| Unclipped, stationary 400 frames | 6.01% | 3.095 / 3.860 | Diagnostic only; rejection disabled |
| Variance bounds, stationary 400 frames | 6.22% | 3.112 / 5.547 | Rejected; visible noise remains |
| Variance bounds, moving 100 frames | 9.32% | 3.176 / 3.760 | Rejected; lighting-change regression |

Single RTX 3060 runs at 1440p with FXAA, two rays/two candidates, 16 warmup frames. HLFS-only excludes TLAS and other passes. Stationary comparisons retain the earlier moving-reference fog-history limitation. The static p95 spike is retained, not discarded as an outlier. Full numbers are in `history-experiments.json`; CSVs preserve timings and all quality rows.

The 14 hardware-RT regression tests passed with variance bounds. However, the existing four-seed quality frontier (seeds 11/29/47/71, setting 2:2, presampling, reactive history, discovery=1) failed **45 of 90 final-output rows**, primarily after blockers appeared/disappeared. Restoring the original shader passed all 90 rows with the same settings. This is a measured regression, not an accepted noise/performance tradeoff. Reproduce with `cargo test --release -p helio-pass-hlfs --test gpu_hlfs_rt benchmark_rt_quality_frontier -- --ignored --test-threads=1`, setting `HLFS_RT_QUALITY_OUTPUT`, `HLFS_RT_QUALITY_SETTING=2:2`, `HLFS_RT_QUALITY_PRESAMPLE=1`, `HLFS_RT_QUALITY_REACTIVE=1`, and `HLFS_RT_QUALITY_DISCOVERY=1`.

Next: distinguish a spatially coherent lighting change from stochastic single-pixel disagreement before clipping temporal history. Test both stationary variance and the unchanged blocker-motion gate; a smoother still image alone is insufficient. No production defaults changed and the visual gate remains failed.

![Unclipped stationary diagnostic](history-unclipped.png)
![Rejected variance bounds during camera motion](history-variance-aware-moving.png)
