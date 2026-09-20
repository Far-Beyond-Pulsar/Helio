# Transparency coverage for TSR

Development evidence, not final visual or performance acceptance.

The HLFS+TSR graph now writes a separate internal-resolution R8 coverage attachment during transparent rendering. The existing HDR color format and color blend stay unchanged. Coverage uses alpha-over accumulation and clears to zero every frame. TSR samples it at the same jitter-corrected position as current color, retains current coverage in history alpha, and uses the maximum of current and reprojected previous coverage to reduce history. Zero coverage preserves opaque RGB; full coverage uses the current-only resolve. This is raster transparency metadata, independent of the RGB RT transmission coefficients.

Coverage is opt-in for other graphs. Legacy custom transparent templates remain compatible but do not write coverage unless they implement the documented location-1 output and marker. This does not add transparent-object motion vectors, refraction, caustics, thickness absorption, or coverage for reflected/refracted glass. Reprojection still follows the opaque depth behind a pane. Thus it reduces inappropriate history reuse without establishing complete transparent-motion correctness.

## Validation

`cargo test --release -p helio-pass-tsr -p helio-pass-transparent -- --test-threads=1` passes (see tests.log). The production TSR draw/readback checks current and previous full coverage, departed coverage clearing, partial coverage, reset metadata, disabled coverage, unchanged opaque RGB, depth rejection, and full global reactivity at several frame rates. Other TSR tests cover camera jitter and resized history publication. Transparent GPU tests compile all four combinations of default/legacy template and coverage enabled/disabled, and render an empty pass at two sizes to prove graph-routed coverage is attached and cleared. This is not an interactive resize test or a numerical stacked-pane blend test.

An initial development capture used the wrong resource-registry lookup and is excluded from this report. The corrected graph attachment lookup is covered by the empty-pass GPU regression. Only `transparency-tsr-final-*` captures are preserved here.

All captures: RTX 3060, Vulkan, native internal/output resolution, 100 frames, fixed 60 Hz simulation, RT+presampled lighting, stone textures, native TSR, no SSR. `HLFS_NO_TRANSPARENCY_REACTIVITY=1` disables the TSR consumer for the control while retaining the attachment, so this comparison does not measure the total cost of producing coverage. Frame 0 RGB is exactly equal on/off. Later differences do not by themselves establish improved image quality. Frame 99 at 1440p and 4K was visually inspected; jagged trim, faceting and sampled-lighting floor-edge artifacts remain.

| Capture | HLFS-only median / p95 ms | Serialized CPU+GPU frame median / p95 ms |
|---|---:|---:|
| Native 1440p, coverage enabled | 7.873 / 8.456 | 32.341 / 35.375 |
| Native 1440p, coverage disabled | 7.653 / 8.301 | 31.979 / 35.602 |
| Native 4K, coverage enabled | 16.557 / 17.196 | 66.614 / 71.033 |

Single sequential runs; first 16 frames excluded. HLFS GPU queries exclude TSR, transparency, TLAS and other passes. Serialized frame latency excludes capture readback and is not full-frame GPU timing. No speedup is claimed. The 3–4 ms target and artifact-free acceptance remain unmet.

Reproduce with `HLFS_RT=1 HLFS_PRESAMPLED=1 HLFS_RESOLUTION=1440p HLFS_TSR_NATIVE=1 HLFS_CAPTURE_TIMINGS=1 HLFS_CAPTURE_FRAMES=100` and `target/release/indoor_cathedral_hlfs.exe --capture <directory>` (set environment variables using the host shell). Add `HLFS_NO_TRANSPARENCY_REACTIVITY=1` for the control; use `HLFS_RESOLUTION=4k` for stress. Run `analyze.py` to reproduce image differences and HLFS statistics from preserved files.
