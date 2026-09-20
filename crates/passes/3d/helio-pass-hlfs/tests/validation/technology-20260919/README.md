# Technology gallery: rejected visual baseline

This new workload contains 1,024 independently shadowed local lights in a 48 x 68 m hall, with matching emissive fixtures, metallic display cylinders of differing roughness, structural bays and colored accents. The point sources sit below the opaque fixtures; the initial inside-fixture placement was corrected after a black lighting capture. Geometry is batched into eight materials. This is a diagnostic development scene, not a completed reflection showcase.

1440p, RTX 3060 / Vulkan, presampled RT, FXAA, 100 moving-camera frames, 16 warmup and 84 measured. Single runs. HLFS-only timing excludes TLAS, fog, other passes and CPU.

| Path | HLFS median ms | P95 ms | Display RGB NRMSE frames 31 / 63 / 99 |
| --- | ---: | ---: | --- |
| Presampled | 3.174 | 4.024 | 13.87 / 13.54 / 13.03% |
| All-light reference | 116.021 | 119.334 | reference |

**Visual gate fails:** severe colored sampling variance on walls and metal. Reference renders smooth direct-light gradients. The performance figure does not make the noisy result acceptable. Metrics use post-tonemap RGB, not linear energy. The current HLFS graph still lacks reflection tracing/composition; black polished cylinders are not a verified reflection result. No realistic texture assets are present. The next work is variance control and reflection integration, followed by motion/disocclusion and 4K validation.

Build: `cargo build --release -p examples --bin technology_gallery_hlfs`.
Capture: set `HLFS_RT=1`, `HLFS_PRESAMPLED=1`, `HLFS_RESOLUTION=1440p`, `HLFS_FXAA=1`, `HLFS_CAPTURE_TIMINGS=1`, then run `target/release/technology_gallery_hlfs.exe --capture target/technology`. Add `HLFS_REFERENCE=1` for the all-light reference. Both 100-frame captures complete; release build passes. No interactive-viewer acceptance yet.

![Rejected sampled result](sampled.png)
![Direct-light reference](reference.png)
