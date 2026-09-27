# Canonical ray audit and brick-entry correction

Generated captures, logs, CSVs, manifests and archives are local-only in the ignored `validation/` directory. They are not included in the branch; paths below identify local artifacts.

This checkpoint fixes a ray displacement and measures the remaining difference between the stored far field and the editable voxel volume. It does not qualify the current far representation, visual quality or performance.

## Defect and correction

The far tracer clamped its integer entry anchor to the selected brick, then replaced an out-of-brick fractional coordinate with `0.5`. An entry on a brick's high face belongs to its last cell with fraction `1`, not `0.5`. Recentering that coordinate moved the ray; returning the correct voxel did not guarantee correct depth.

The tracer now retains the original fractional coordinate plus the integer anchor correction. Keeping that correction separate also avoids losing a one-cell offset to orbital floating-point spacing. A GPU regression enters a known planar field from outside the brick and checks the analytically known first voxel and entry distance across 64 rays at each of three levels. Before the fix it failed at level 1: **4.4870005 m instead of 4.537 m**. After the fix it passes alongside all 25 retained terrain tests; one CPU benchmark remains ignored.

## Exact audit

`HELIO_VOXEL_FLIGHT_CANONICAL=1` adds a 16×9 grid of exact CPU rays to six settled flight captures. It consumes the actual primary-ray direction bits, the uploaded split camera origin, the authored base grid, and the same immutable world. It records first cell, material, depth, whether a reported GPU cell is canonically occupied, and oracle cost. Internal render dimensions are used before upscaling. Pending residency is rejected.

These are **single-ray probes**, not supersampled pixel coverage or silhouette acceptance. A difference from the exact first cell alone does not quantify a visible error when many cells fit in a pixel. `depth_error_footprints` scales along-ray depth error by the pixel's world-space footprint; it is not a projected silhouette-error measurement. No timing acceptance is claimed for oracle or replay runs.

At 200 m in the 720p captures, all 144 ray origins/directions match between baseline and corrected runs:

| Metric | Before | Corrected |
| --- | ---: | ---: |
| Correct first cell | 58 / 144 | 88 / 144 |
| Correct-cell hits with depth error >1 mm | 11 | 0 |
| Maximum depth error among correct-cell hits | 0.098066 m | 0.000169 m |
| GPU surface cells in canonical air | 36 | 36 |

The unchanged last row isolates a separate limitation: interpolation of sparse densities still creates a different occupied volume. At 1 km the corrected probe has 18 exact first-cell matches out of 144; 70 reported GPU cells are canonically air. At orbit, the depth-error p95 is about 1.196 km, or 0.756 of the sampled pixel footprint. Those figures do not establish a coverage-error bound. A faithful filtered representation still needs reference images, material/coverage integration and edited-feature tests.

Only 65 rays match between the two 1 km captures, and none match in the ground/edit/1 m capture pairs, because frame-count-dependent camera jitter changes during asynchronous settling. Do not interpret those pairs as identical-ray image differences. Each remains individually compared with its own exact CPU rays. The orbital pair has 144 matched rays.

## Validation and provenance

Parent: Helio `24e522966c164bf0f82e8fbbe35d6b9e21f7d0b0`. Platform: Windows, Vulkan on RTX 3060, Ryzen 5 3400G, Rust 1.98.1. The baseline adds only the optional oracle and hit-buffer extent accessor to that parent. The corrected executable adds the entry fix and GPU regression. Binary hashes and raw CSV/log evidence are in validation/2026-09-26-canonical (local `validation/2026-09-26-canonical`).

- Deliberate pre-fix regression: failed by 5 cm. The first command used an incorrect test filter and ran zero tests; that log is retained and is **not** regression evidence. The corrected filter is `far_traversal_matches`.
- Corrected release terrain suite: **25 passed, 1 ignored**.
- Corrected 720p native full flight: **1,155 frames completed**.
- Corrected 1080p Quality flight with sunlight, stage profiling, exact primary replay and the canonical oracle: **1,175 frames completed**. All 21 full captures passed primary replay and sunlight validity checks, including resize, remote editing and 1 m grid replacement.
- Each of the three flights audits 864 canonical rays. All sampled hit/miss classifications agree with the CPU oracle. That sparse observation is not whole-image coverage acceptance.

The corrected captures retain the dark contour bands and aliasing documented earlier. This change does not provide a new accepted movement recording or a faster-renderer claim. Far truth, regional publication, arrival latency, materials/lighting, populated-editor and simulation gates remain open in [the full goal](PLANET_TERRAIN_GOAL.md).

The next architectural experiment is tighter conservative bounds for the canonical noise field. Current region selection uses global slope bounds even when a box lies within one interpolation cell. Interval evaluation of that cell's actual coefficients could certify more empty/solid regions before allocating and generating exact bricks. This must preserve the source volume and pass CPU/GPU conservativeness checks before any performance comparison.
