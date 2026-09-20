> Resolution correction (2026-09-20): architectural scene captures made by the shared harness without native TSR used renderer scale 0.75. A 1440p output was internally 1920x1080; 4K output was internally 2880x1620. References used that same internal size with full HLFS shading. Native-resolution wording for those captures is superseded; standalone GPU fixtures are unaffected. See the HLFS validation report `resolution-audit-20260920/README.md` for corrected, matched measurements.

# Rejected phase-aware reconstruction trials

Base: f6cb94d2, RTX 3060 Vulkan, cathedral, 1440p, colored RT transmission, FXAA, no SSR, 100 moving-camera frames. HLFS_RT=1 HLFS_PRESAMPLED=1 HLFS_RESOLUTION=1440p HLFS_FXAA=1 HLFS_CAPTURE_TIMINGS=1; reference adds HLFS_REFERENCE=1. The reference uses full-resolution shading; candidate shading is half resolution. First 16 frames excluded from timings. CSV p95 is linearly interpolated. Errors are post-tonemap display-RGB NRMSE, not linear radiometric correctness.

The compositor normally weights samples at block centers, although each sample is placed at a phase-varying pixel within the block. The strict trial instead used separable tent weights at actual sample locations, retaining the existing four-block neighborhood. This can leave zero-weight gaps even when a geometry-valid sample exists, so it invokes more full-resolution repair rays. Error fell modestly but composition cost rose from about 1.56 to 2.35 ms, taking total HLFS from 4.45 to 5.28 ms. Rejected.

The support trial floored the spatial weight at 0.0001 after geometry rejection to prevent these gaps. Total HLFS returned to 4.57 ms, but errors at frames 31/63/99 became 4.660/5.076/4.921%, versus baseline 4.655/4.915/4.809%. Rejected. This does not establish that all phase-aware reconstruction is wrong; this four-neighbor weighting change fails the combined quality/performance requirement.

Both patches are retained solely as rejected experiments (apply with git apply --unidiff-zero for reproduction). Production composite.wgsl was restored. The samples around shadow boundaries need a better reconstruction strategy; the modest strict-trial gain does not justify increased repair-ray cost. Timing is HLFS-only and excludes FXAA, TLAS and the rest of the renderer. Neither trial meets the 3-4 ms goal. No final visual acceptance or review promotion.

Restoration validation: rebuilt the release cathedral binary and repeated the 100-frame path; frames 0/31/63/99 are RGBA pixel-identical to baseline. CI for base commit f6cb94d2 passed.
