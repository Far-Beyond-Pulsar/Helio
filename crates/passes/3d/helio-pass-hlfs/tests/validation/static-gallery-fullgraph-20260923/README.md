# Static light gallery: idle GPU full graph diagnostic

The previously parked `technology_gallery_hlfs` example was reused to isolate light-count cost. It contains 1,024 independently shadowed **static** colored lights and 38,400 triangles. The capture moves the camera over 100 frames, but does not animate the lights or geometry. This is a 37-pass renderer capture, not the synthetic HLFS-only probe.

Both captures use 2560x1440 native internal/output resolution, native TSR, ray-traced shadows, no SSR, one HLFS sample per shading pixel, eight candidates, and tile presampling. The only changed setting is `HLFS_SAMPLE_SCALE`: 2 for `half-*` and 1 for `native-*`. `HLFS_SAMPLE_SCALE` is a capture-only override in `crates/examples/hlfs_capture.rs`. Frame 99 PNGs and raw timing CSVs are included. The initial run occurred while the GPU was busy and was discarded; these files are from the later idle reruns.

| HLFS shading scale | Graph GPU p50 / p95 | HLFS p50 / p95 | Sampling p50 / p95 | Approx. GPU FPS at graph p50 |
| --- | ---: | ---: | ---: | ---: |
| 2 (1280x720) | 12.442 / 13.500 ms | 5.947 / 6.399 ms | 3.099 / 3.705 ms | 80.4 |
| 1 (2560x1440) | 24.701 / 26.201 ms | 18.252 / 19.229 ms | 12.425 / 13.138 ms | 40.5 |

Graph times exclude readback and are GPU timestamps; approximate GPU FPS is `1000 / graph_gpu_ms`, not measured display presentation rate. HLFS timings include its stages only, so do not add them to graph times. The first 16 HLFS rows are excluded as warmup; the graph CSV has 83 usable timing rows after the query pipeline fills.

Visual inspection of both full-size frame 99 images shows prominent colored mottling on the ceiling and walls. Native sampling reduces some of it but is still not an acceptable result. The faster scale-2 path must not be presented as a successful performance setting until moving-image quality passes separately. The moving-light and moving-blocker visual gate in `../moving-native-rt-20260923/` remains red.

The native sampling stage is the main cost in this light-dense scene. A diagnostic with unshadowed lights left sampling nearly unchanged, so shadow ray traversal alone does not explain the cost. Reducing candidates from eight to two gave only a modest timing improvement and loses sample quality; it is not a proposed fix. A native-resolution `sample_pixel` fast path did not produce a stable whole-frame gain in alternating control/candidate runs and was reverted. No light count or geometry was removed.
An early return before BRDF evaluation for zero-contribution lights also failed
to lower the sampling median in the native gallery and was reverted.

The composite shader now takes a direct filtered-lighting path when the native
shading texel matches the receiver geometry. It falls back to the prior
reconstruction path on mismatches. Matching **release** binary A/B captures
with the same scene and settings are saved as `composite-scale{1,2}-*` CSVs.
At scale 1, graph p50 was 24.093 ms control / 23.895 ms candidate and composite
p50 was 0.889 / 0.599 ms. At scale 2, graph p50 was 12.244 / 12.255 ms and
composite p50 was 1.060 / 1.071 ms, within run variation. Frame 0 PNG hashes
matched exactly in both pairs. Later frames vary even between unchanged control
runs because this renderer capture is not bitwise deterministic; visual
inspection found no new artifact beyond the pre-existing mottling. These A/B
captures are evidence of a small native composite improvement, not a quality
pass for the gallery.

To repeat a capture, set `HLFS_RT=1`, `HLFS_PRESAMPLED=1`, `HLFS_SAMPLE_COUNT=1`, `HLFS_CANDIDATE_COUNT=8`, `HLFS_SAMPLE_SCALE=1` or `2`, `HLFS_RESOLUTION=1440p`, `HLFS_TSR_NATIVE=1`, `HLFS_GRAPH_TIMINGS=1`, `HLFS_CAPTURE_TIMINGS=1`, `HLFS_CAPTURE_FRAMES=100`, and `HLFS_CAPTURE_TAIL=4`; then run `cargo run --release -p examples --bin technology_gallery_hlfs -- --capture <output-directory>`. Keep other capture settings and the build profile equal when comparing runs.
