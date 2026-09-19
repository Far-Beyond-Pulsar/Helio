# Artifact repair checkpoint (not final acceptance)

The earlier review image had localized bright edge spikes and self-shadow stripes that aggregate image-error gates did not reject. PR #248 is back in draft; the tracker is In Progress.

## Changes and falsification

- Thin surfaces missing half-resolution coverage were shaded after denoising with a random estimator. Small light sets (<=64 rows) now use exact repair in the presampled tier. The new one-pixel, 17-light regression checks 24 uncovered phases against all-lights reference. The 1,024-light path still requires further repair work.
- A 4x depth-ULP shadow bias removed stripes but erased a 2 cm blocker at 100 m. Rejected. A 1.5x bias also erased contacts at 150 m. Rejected; these were not accepted optimizations.
- Separating view and projection in depth/G-buffer rasterization removes precombined-matrix rounding error and visibly improves the cathedral. A new real raster-depth regression exposed a remaining self-hit at 150 m, missed by the older ideal CPU-depth fixture.
- G-buffer motion is now RGBA16F: XY retain motion in pixels, Z stores the small signed view-depth reconstruction residual, and W=2 validates it. Corrected ray origins use the residual quantization/transform error rather than a depth-buffer ULP. Producers without this residual explicitly write W=0. Portal/foliage attachment contracts were updated. G-buffer bandwidth increases from 48 to 52 bytes/sample; fresh performance validation is required.

10 RT GPU regressions and 14 screen-space GPU regressions pass. The affected G-buffer, depth, transparency, portal and foliage package tests pass (170 unit/layout checks). These do not prove scene-wide visual completion.

Matched 100-frame 1440p captures were rendered and inspected. The sampled image still has blotchy lighting and jagged edges. Native TSR was also captured and inspected: tracery edges improve, but lighting blotchiness remains. The original capture did not enable temporal AA. Native TSR adds cost: 32.625 ms median / 53.135 ms p95 serialized latency in this run, not an accepted GPU-budget result. The serialized sampled capture measured 23.388 ms median / 27.483 ms p95 including CPU preparation and frame work, excluding image readback. This is not the HLFS-only GPU timing or an acceptance result.

![Current reference, frame 99](reference.png)

![Current sampled, frame 99](sampled.png)

## Outstanding goal

Fix remaining visible artifacts and revalidate quality and RTX 3060 1440p performance, with 4K stress. Add useful engine RT support including coloured transmission and validate glass stacking, opaque occlusion and dynamic changes. Build three substantial showcase scenes: a cathedral grounded in real dimensions, a monumental stone arch, and a technology scene exercising varied reflections and many lights. Use realistic licensed PBR textures at appropriate physical scale, and validate normal/roughness texture loading. Audit and integrate existing fog and other appropriate engine passes instead of assuming that a named pass is complete. Keep the same PR draft until the whole requested state is verified; publish final images/numbers only when ready for review.

The current engine audit found fog in the HLFS graph but no SSR/planar-reflection composition there. The SSR package has a hybrid RT path to investigate. The glass-specific transparent material template still contains a debug-green return; it is not the material path used by these cathedral panes.

Research references for subsequent scene work:

- [Epic MegaLights documentation](https://dev.epicgames.com/documentation/unreal-engine/megalights-in-unreal-engine): stochastic light sampling, complexity diagnostics, area lights, fog/translucency and limitations.
- [Cologne Cathedral official dimensions](https://www.koelner-dom.de/erleben/der-dom-in-zahlen): exterior length 144.58 m, width 86.25 m, nave interior width 45.19 m, nave height 43.35 m, side aisle height 19.80 m. These support a real-scale scene, not a claim of an exact architectural replica without detailed plans.
- [Arc de Triomphe educational material](https://www.paris-arc-de-triomphe.fr/var/cmn_inter/storage/original/application/96c69db2ffcd9842fbd27e3e029883b5.pdf): dimensional drawing to inspect before implementing the arch.
