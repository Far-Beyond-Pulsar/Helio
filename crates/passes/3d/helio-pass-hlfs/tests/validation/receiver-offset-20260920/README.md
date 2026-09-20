# Shared conditional receiver offset

Opaque and RGB-transmission queries now use one receiver-offset implementation. The precision-corrected G-buffer branch skips the legacy adjacent-depth reconstruction; the offset formula, floors and legacy fallback are unchanged. Opaque exact-light loops can reuse their prepared receiver across lights instead of the previous no-op wrapper recalculating it per ray. The edit does not increase bias, change normal-map behavior or reduce the light/ray budget.

Release cathedral build passed. All 16 non-benchmark RT hardware regressions pass, including rasterized corrected receivers and legacy depth receivers at long distances with nearby contact blockers, colored sheets and opaque blockers. Command: cargo test --release -p helio-pass-hlfs --test gpu_hlfs_rt -- --include-ignored --skip benchmark_ --test-threads=1.

Two alternating baseline/candidate pairs: RTX 3060 Vulkan, native 2560x1440 output/internal, 1280x720 HLFS samples, FXAA, colored transmission, photographic stone with normal maps, no SSR, 100 fixed-clock moving-camera frames each. First 16 frames excluded. Control and candidate images are exactly RGBA-identical at frames 0/31/63/99 in both pairs. Timings cover only HLFS GPU stages, excluding TLAS and other passes.

| Pair | Control HLFS median ms | Candidate HLFS median ms | Control composite ms | Candidate composite ms |
|---|---|---|---|---|
| 1 | 7.784 | 7.621 | 2.566 | 2.457 |
| 2 | 7.786 | 7.726 | 2.538 | 2.499 |

The improvement is small and variable; this does not meet the 3-4 ms gate. Raw per-stage CSV and median/p95 JSON are retained. Pixel identity preserves known scene artifacts rather than proving their removal. The shader consolidation reduces duplicated receiver logic, but no new claim of normal-map/contact correctness beyond the listed tests is made.
