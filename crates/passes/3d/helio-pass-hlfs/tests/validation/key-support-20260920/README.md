# Extracted key support rejection

The full-resolution composite now checks can_illuminate before tracing the extracted key light, as the exact local-light loop already does. Receivers outside the light support or facing away cannot contribute direct energy. The check avoids their visibility query without changing the light budget or transmission semantics.

Expanded the existing directional-key GPU regression to include front-facing, back-facing and tangent directions for one, two and seventeen lights across four frames each, compared with the unextracted reference. All sixteen non-benchmark RT regressions pass on RTX 3060 Vulkan: cargo test --release -p helio-pass-hlfs --test gpu_hlfs_rt -- --include-ignored --skip benchmark_ --test-threads=1. Release cathedral builds passed.

Native 2560x1440 internal/output, 1280x720 HLFS samples, FXAA, RT presampling, colored panes, no SSR, fixed 60 Hz moving camera, 100 frames, first 16 excluded. Sequential freshly built control and candidate runs had exactly identical RGBA images at frames 0, 31, 63 and 99. This preserves existing visual output, including its known artifacts; it does not establish artifact-free visuals.

| Variant | Sampling median ms | Composite median ms | HLFS median ms |
|---|---|---|---|
| Control | 3.979 | 2.487 | 7.726 |
| Support rejection | 4.007 | 2.506 | 7.766 |

No cathedral speedup is established. Earlier same-day runs were about 9.6 ms, including substantially slower sampling even though this edit only changes composite. That between-run variation must not be attributed to this edit. Retain this as a supported zero-contribution rejection, not a claimed step toward the 3-4 ms target. GPU timings exclude all other passes, AA, and TLAS. No new 4K or whole-frame result is claimed.
