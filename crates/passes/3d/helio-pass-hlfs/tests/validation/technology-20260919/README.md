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
