# Connected cathedral masonry and daylight

The cathedral now uses connected smooth tubes for pointed arches, rather than separate capped cylinders at every segment. Main shafts, clustered shafts and capitals have smoother normals and more radial segments. Column cores use the moulding material; the flat-wall photographic UV projection is no longer applied to curved column cores. Photographic stone remains on structural walls. Removing redundant arch caps lowers the complete scene from 251,076 to 242,180 triangles, with the same 14 material batches (six glass).

RT mode defaults to one warm directional daylight source plus the three chandeliers and five candle clusters. This gives a coherent exterior light direction. `HLFS_LEGACY_CATHEDRAL_LIGHTS=1` retains the earlier nine white exterior point emitters plus the eight interior lights as a transmission stress control. Both the cathedral and drone examples share these geometry/lighting changes. This is a different scene workload, not an equivalent-work renderer optimization. The 1,024-light technology target remains separate and unresolved. The cathedral's extent is still approximately 56x22x20 metres, not a 1:1 model of a named cathedral.

## Checks and captures

Release builds of both examples passed. Both example test binaries pass the connected-arch test, which checks finite, nondegenerate triangles, consistent normal/winding orientation and a closed two-face-per-edge manifold after welding UV/cap-normal seams. Existing UV rectangle coverage tests also pass. Interactive drone input/resize was not tested.

Captures use native internal/output resolution, fixed 60 Hz simulation, a moving camera, 100 frames, RT presampling, photographic walls and native TSR with transparency coverage. The clear-glass control changes only the RT sheet transmission to white; visible pane materials retain their tint. Its neutral projected light versus the colored-glass capture confirms that transmitted tint comes from the RT coefficients, not colored exterior emitters. This is thin-sheet attenuation, not refractive/caustic glass.

The exhaustive full-resolution daylight reference gives final-image RGB NRMSE of 2.5352%, 2.3218%, and 1.9033% at frames 31, 63 and 99 respectively. This aggregate metric does not certify individual edges. The frame-99 daylight, clear, legacy-emitter, 4K and SSR captures were visually inspected. Columns have smoother shading; floor/trim aliasing and incomplete or unstable reflections still need work. No artifact-free claim is made.

| Scene | HLFS-only median / p95 ms | Serialized CPU+GPU frame median / p95 ms |
|---|---:|---:|
| Daylight, native 1440p | 3.848 / 4.813 | 26.295 / 29.081 |
| Daylight, clear transmission | 3.898 / 4.323 | 25.951 / 30.560 |
| Legacy window emitters, new smooth geometry | 7.172 / 7.928 | 29.978 / 33.774 |
| Daylight, native 4K | 8.693 / 11.205 | 55.313 / 63.285 |

Single sequential runs, first 16 frames excluded. The 1440p daylight median falls in the requested band, but its p95 exceeds 4 ms. Changing 17 lights to nine does not satisfy the many-light performance requirement. HLFS queries exclude transparency, TSR, SSR, TLAS and other passes; serialized latency is not full-frame GPU timing. `results.json` also records the reference and SSR run statistics. The SSR run enables `HLFS_SSR=1`; all other captures here disable SSR. This capture schema does not yet record SSR/clear-glass flags, so the directory labels and this report specify those controls explicitly.

Reproduce using `HLFS_RT=1 HLFS_PRESAMPLED=1 HLFS_RESOLUTION=1440p HLFS_TSR_NATIVE=1 HLFS_CAPTURE_FRAMES=100 HLFS_CAPTURE_TIMINGS=1` with the cathedral binary's `--capture <directory>` option. Add `HLFS_REFERENCE=1`, `HLFS_CLEAR_GLASS=1`, `HLFS_LEGACY_CATHEDRAL_LIGHTS=1`, or `HLFS_SSR=1` separately for the respective controls; use `HLFS_RESOLUTION=4k` for stress. `analyze.py` requires Pillow/NumPy and reproduces statistics from the preserved evidence.
