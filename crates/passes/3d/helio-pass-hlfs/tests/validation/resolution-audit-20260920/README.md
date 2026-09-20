# Capture resolution correction

The old shared architectural capture harness inherited RendererConfig's render_scale=0.75 except when HLFS_TSR_NATIVE selected 1.0. Thus FXAA and no-AA cathedral, monument and technology captures labeled 1440p had 2560x1440 output but 1920x1080 internal rendering. Presampled HLFS used 960x540 samples. Corresponding 4K output captures had 2880x1620 internal rendering and 1440x810 HLFS samples. Reference mode forced HLFS sample_scale=1, not renderer scale=1. Standalone GPU fixtures that configure their own target dimensions are not covered by this correction. Native TSR captures remain native.

The harness now defaults explicitly to native rendering. HLFS_RENDER_SCALE permits a labeled scaled control, but cannot contradict HLFS_TSR_NATIVE. Each capture writes capture-config.json after rendering, verifies the actual HLFS output texture against configured dimensions, and records HLFS sampling dimensions from its effective configuration.

Release cathedral build passed. Three sequential 100-frame RTX 3060 Vulkan captures use RT, presampling, colored glass, fixed 60 Hz camera path, no SSR, and discard the first 16 frames. The CSV timestamps cover only HLFS, excluding AA, other passes and TLAS. These are single runs, not a sustained target pass.

| AA | Output | Internal | HLFS samples | HLFS median / p95 ms |
|---|---|---|---|---|
| FXAA | 2560x1440 | 2560x1440 | 1280x720 | 9.558 / 10.307 |
| FXAA scaled control | 2560x1440 | 1920x1080 | 960x540 | 5.631 / 6.214 |
| TSR Native | 2560x1440 | 2560x1440 | 1280x720 | 9.656 / 10.081 |

At matched resolution there is no evidence here that camera jitter doubles HLFS cost. The scaled control is slower than earlier historical runs, so these measurements do not explain all between-run variance. Native FXAA frame 99 was visually inspected: colored transmission is visible but shadow boundaries still alias; this is not artifact-free acceptance. No new 4K or full-frame GPU result is claimed. The 3-4 ms target remains unmet.

Reproduce with cargo build --release -p examples --bin indoor_cathedral_hlfs, then set HLFS_RT=1, HLFS_PRESAMPLED=1, HLFS_RESOLUTION=1440p, HLFS_CAPTURE_TIMINGS=1, HLFS_CAPTURE_FRAMES=100 and exactly one of HLFS_FXAA=1 or HLFS_TSR_NATIVE=1. Run target/release/indoor_cathedral_hlfs.exe --capture <directory>. Only the scaled FXAA control adds HLFS_RENDER_SCALE=0.75.
