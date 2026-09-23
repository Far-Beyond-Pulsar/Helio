# Rejected glossy tile-proposal shortcut, 2026-09-23

The native 2560 × 1440 technology gallery overflows its light grid, leaving
glossy pixels to score global light IDs. I tested reusing the existing GPU
coarse-tile alias proposal for glossy pixels, with the gallery's 1,024 static
lights, moving camera, one configured sample, eight configured candidates,
native TSR and hardware RT. The control shader raises glossy pixels to four
samples and 16 global candidates per sample. The first variant changed the
glossy minimum to eight and used the tile proposal for both candidate scans.
The second variant kept 16 candidates and changed only their proposal source.
All three variants preserve a uniform proposal component and the selected
light's BRDF/visibility evaluation.

The control and eight-candidate variant were captured as separate release
binaries on the RTX 3060. GPU timestamp medians exclude frames 0–15:

| Native gallery, single pair | Sampling | HLFS | Render graph |
| --- | ---: | ---: | ---: |
| Control, global glossy proposal | 9.325 ms | 14.778 ms | 21.019 ms |
| Tile proposal, eight glossy candidates | 7.994 ms | 13.358 ms | 19.620 ms |

This is a screening speed result in a **static-light** gallery, not an
accepted optimization. The [raw GPU timestamps](control-hlfs-gpu-timings.csv)
and [candidate timestamps](alias8-hlfs-gpu-timings.csv), graph CSVs and capture
configurations are retained here.

The fixed seed-11 glossy fixture then moved 1,024 colored lights, the camera,
and the blocker/key-light transition for 96 frames at native HLFS shading.
All three runs used `HLFS_RT_QUALITY_PRESAMPLE=1`,
`HLFS_RT_QUALITY_REACTIVE=1`, `HLFS_RT_QUALITY_GLOSSY_MOTION=1`,
`HLFS_RT_QUALITY_CAPTURE_MOTION=1`, sample scale 1, and setting `1:8`.
The unchanged control passed its final-output, motion and grain gates.
Both tile-proposal variants passed the final-output and grain gates but failed
the unchanged static-frame motion RMS limit of 0.010:

| Moving fixture | Motion RMS | Motion gate |
| --- | ---: | --- |
| Control, global glossy proposal | 0.008400 | pass |
| Tile proposal, eight glossy candidates | 0.012068 | fail |
| Tile proposal, original 16 glossy candidates | 0.011061 | fail |

The eight-candidate version has more visible colored speckle in the inspected
[frame 64](alias8-f64.png) and [frame 80](alias8-f80.png) than the paired
[control frame 64](control-f64.png) and [control frame 80](control-f80.png).
The frames straddle the light/camera/blocker transitions; the motion metric
uses the captured consecutive sequence. Raw quality, motion and grain CSVs
and frames 63/64/80/81 for control and candidate are included. The 16-candidate
variant's motion/quality/grain CSVs isolate the proposal source, but it was
stopped before a full gallery timing capture after failing motion.

An initial fixture invocation omitted `HLFS_RT_QUALITY_REACTIVE`; both the
unchanged control and variants failed under that mismatched setup. Those runs
are excluded from the table above. The source was restored after the matched
comparison. PR #248 retains the original glossy proposal path. A future GPU
proposal needs better support for changing glossy contributions before
candidate work can safely be reduced.
