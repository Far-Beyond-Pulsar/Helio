# Sparse SceneDB light slots at thin geometry

The full-resolution repair path previously selected exhaustive versus stochastic shading using `globals.light_count`, the light-buffer allocation length. A small active light population in a large sparse SceneDB allocation could therefore use a noisy estimator over mostly empty slots. It now consumes the existing fine tile grid and exactly evaluates nonoverflowing local lists (at most 64 lights), skipping the extracted key as before. Overflowing lists retain the bounded stochastic fallback. This fixes sparse-slot correctness; it is not a solution for every visible edge artifact.

The one-pixel hardware RT regression now runs with both 17 dense lights and the same 17 lights at slots 2000+37*i in a 4096-slot allocation. With the old shader, frame 1 returned RGB `[0.22717285, 0.15136719, 0.03668213]` against reference `[16.40625, 11.046875, 2.4902344]`. The corrected shader passes the reference tolerance across all tested unsampled phases. `before.txt` preserves the failing run; `after.txt` the initial corrected sparse-only run; `suite.txt` includes the final dense+sparse test. All 2 unit, 14 screen-space GPU and 16 nonbenchmark RT tests pass.

Matched cathedral controls use native 2560x1440 internal/output resolution, fixed camera 0.5, FXAA, RT presampling, photographic stone and 100 frames. Reference uses exhaustive full-resolution lighting. Frames 96–99 are RGB-identical before/after the sparse repair. Their reference NRMSE remains 5.2557%, bottom-440-row NRMSE 3.7936%, and mean temporal RGB standard deviation 0.98744/255. Thus the visible floor pattern remains unresolved. Frame 99 was visually inspected against the full reference.

| Single run | HLFS-only median / p95 ms | Composite median / p95 ms |
|---|---:|---:|
| Baseline | 7.7133 / 8.5840 | 2.5426 / 3.1843 |
| Local-list repair | 7.5233 / 7.7411 | 2.0593 / 2.4440 |

First 16 frames excluded. The captures are sequential single runs, not proof of a sustained speedup. These GPU timings exclude TLAS and all non-HLFS passes; the 3–4 ms target remains unmet. The expanded exact fallback can cost more in scenes containing many thin pixels with nonoverflowing local lists. Full-scene performance acceptance remains open.

Reproduce the fixture with `cargo test --release -p helio-pass-hlfs --test gpu_hlfs_rt uncovered_thin_edges -- --include-ignored --test-threads=1`. The full suite uses `cargo test --release -p helio-pass-hlfs -- --include-ignored --skip benchmark_ --test-threads=1`. Run `analyze.py` (Pillow/NumPy) against the preserved images and CSVs.


The 1024-light technology diagnostic was rebuilt and visually checked at native 640x360, fixed camera 0.5, FXAA, 100 frames. Final-frame RGB NRMSE against the existing fixed-camera exhaustive frame-16 reference is 22.6414%, versus 22.6149% for the earlier control. Mean brightness ratios are 0.93742 and 0.93741. Lighting is static in this shared offscreen capture loop. The current HLFS median is 0.8643 ms at this diagnostic resolution; this is not the 1440p performance target. The large error and structured floor noise remain. These historical-control comparisons do not establish a speedup or complete many-light image quality; the new GPU regression establishes the sparse thin-edge fix.
