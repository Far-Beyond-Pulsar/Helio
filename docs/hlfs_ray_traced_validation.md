# Ray-traced HLFS validation protocol

This is a proposed protocol for the [RT implementation plan](hlfs_ray_traced_plan.md), not a record of executed RT tests. Freeze fixture assets, camera paths, random seeds, reference outputs and thresholds before selecting an optimized variant. Keep failed variants and raw measurements. New evidence may change the implementation choice; it must not silently change the acceptance test.

## Resolutions and measurement boundaries

Primary: 2560×1440 GBuffer and output, RTX 3060, Vulkan, 1,024 dynamic local lights. Measure native shading (`sample_scale=1`) and reconstructed shading (`sample_scale=2`) separately. Candidate counts: 4/8/16. Samples per shading site: 1/2/4. Current compact ScreenSpace at scale 2, two samples and eight candidates is the control. Use 1920×1080 to connect with historical measurements; use 3840×2160, both scales, for stress results. Also test odd 2561×1441 and 3839×2159 dimensions for correctness.

The candidate selected for the primary tier must meet the quality gates and an inclusive incremental direct-lighting GPU median of ≤4 ms, with ≤3 ms a stretch target and p95 ≤5 ms. Include proposal construction, every visibility/reuse/repair operation, denoising, composition and new RT scene/dependency work. Publish lighting-only and acceleration-structure timings as separate columns as well. If another effect shares the TLAS, report both standalone cost and incremental shared cost. A shared-cost pass cannot establish the standalone claim.

4K results have no predetermined 4 ms acceptance threshold. Report the achieved time, quality, memory and failure envelope. Do not silently lower GBuffer resolution, samples, light count, exposure, light bounds, or geometry coverage between an RT result and its baseline. If full-resolution lighting misses the budget and reconstructed lighting passes, name the successful configuration explicitly.

## Deterministic scene manifest

Implement and retain each fixture with at least four fixed seed variants. Record all vertex/index counts, instance counts, light types/ranges/intensities and material flags rather than describing a fixture only as “complex.” The following counts are proposed starting definitions, to be frozen at Phase 0 before optimization:

| Fixture | Geometry and light setup | Purpose |
| --- | --- | --- |
| Analytic oracle | Plane, boxes and known ray segments; 1/8/65 lights | Isolate transforms, ray ranges, origin bias and exact energy |
| Sparse rooms | 1 million opaque triangles, 10,000 rigid instances, 1,024 finite-range lights distributed over separated rooms | Primary tier; conservative locality and cold-cache discovery |
| Dense overlap | Same geometry/counts, all 1,024 lights affect the same receivers | Primary tier; variance, overlap scoring and overflow pressure |
| Dynamic doors/lights | Primary scene, 10% rigid transforms change each frame; all lights translate or change intensity on deterministic paths | Primary tier; moving casters, hidden-light revelation and history correctness |
| Fine geometry | Fences, wires and isolated one-pixel surfaces; matching opaque and alpha-cutout fixtures | Repair coverage and material correctness |
| Deformation | 100,000 deforming triangles added to the primary scene, fixed topology; separate topology-replacement sequence | BLAS update/rebuild costs and stale geometry |
| Large world | Translated and rebased copies at small/large coordinate scales, distant directional illumination | Bias, coordinate epoch and far-caster coverage |
| Capacity stress | 16,384 lights; 65,535/65,536 light identity boundary; 65,536/65,537 instances | Explicit overflow, growth, no missing lights/casters |
| Integration | Existing cathedral camera path, ported with identical geometry/lights; actual editor camera and resize interactions | Visual regression and graph/resource lifetime |

The three primary room/overlap/dynamic fixtures must each pass the 1440p target; do not average a failed dense scene into a successful sparse one. Masked/deforming fixtures define additional published coverage tiers and are required before claiming general material/animated-scene readiness. Stress tiers may miss the timing target but must not crash, truncate work silently, retain removed geometry, or return false successful measurements.

## Reference construction

Use two independent checks. First, analytic CPU ray/triangle or known geometric segment tests validate GPU visibility for small scenes, including offscreen blockers and transformed instances. Second, a full-light direct-lighting oracle uses the same BRDF and light definitions with current complete geometry, no denoising or temporal reuse. It must itself be validated against the small analytic scenes so that the optimized path and oracle do not share an undetected visibility bug.

For punctual lights, evaluate every light directly. If area lights are later introduced, use enough independent samples to establish a confidence interval tighter than the optimization error threshold and record the convergence study. A ScreenSpace reference cannot judge offscreen RT shadows. Preserve the old reference for ScreenSpace regression, alongside the new world-visibility reference.

Capture linear HDR diffuse/specular/raw/final buffers, output display images, depth/normal/motion, selected-light IDs, effective mode and current scene generation. Compare identical camera/light frames. Keep zero-reference masks explicit so normalized errors do not divide by a near-zero value and hide dark-region leaks.

## Correctness and visual gates

Preserve the existing mean-energy error <8% and normalized image RMSE <20% tests, including finite-rank noise, light extinction, packed-light overflow and odd-resolution reprojection. Report raw and filtered errors independently; passing average final energy is insufficient. For new fixtures use linear luminance `Y`, mean error `abs(mean(Y)-mean(Yref))/max(mean(Yref),1e-6)` and NRMSE `sqrt(mean((Y-Yref)^2))/max(sqrt(mean(Yref^2)),1e-6)`. Preserve existing tests' metric implementations rather than replacing them with these new definitions.

New acceptance gates, fixed before optimization:

- Static analytic visibility: zero incorrect hit/miss classifications for non-boundary test rays. Explicit grazing/bias tolerances use a documented geometric epsilon and separate near-boundary mask.
- Light removal/extinction and same-count replacement: no stale selected identity on the next frame; raw contribution from removed lights must be zero. Filtered residual after eight frames must be <1% of the pre-change reference peak in the changed region.
- Newly revealed dominant light: raw sampling must retain nonzero support immediately; across fixed-seed runs, at least 95% of affected pixels must select it at least once within eight frames when that light contributes ≥50% of local reference luminance. Publish discovery latency distribution and affected-region mean error at frames 1/2/4/8/16. If unmet, this gate fails rather than asserting a statistical guarantee from an exploration fraction alone.
- Disocclusion and motion: evaluate both the full image and disoccluded/moving-shadow/glossy masks at frames 1/2/4/8/16; each must satisfy mean error <8% and NRMSE <20% for the declared production tier. Retain curves so short-lived failures are visible.
- Empty/missing/stale TLAS: empty-ready scenes render correctly; unavailable or stale acceleration data yields the documented mode/error behavior. Device capability alone must never count as successful RT execution.
- Thin and masked geometry: no systematic missing surface or solid alpha-cutout shadow. Check every sampling phase and inspect frame sequences at native output scale. Reconstruction repair overflow uses correct work or an explicit unsupported state, never silent clipping.
- Mode and resource changes: ScreenSpace→RT→ScreenSpace, resize, odd sizes, packed-format fallback, exposure change, camera cut, external light-buffer replacement, geometry-pool growth and scene rebase must be coherent on the first usable frame.

Visual review is mandatory even when numerical gates pass. Inspect fixed-camera crops, moving-camera sequences, glossy highlights, small shadows, dark interiors, offscreen occlusion and light-to-dark transitions. Record noise, boiling, ghosting, blur and leaks separately. Display metrics are supplemental and must not replace HDR energy checks. A visibly wrong implementation is rejected.

## Performance protocol

For each comparison, build first and run with no competing benchmark/build processes. Record commit, binary/shader hashes, GPU model/VRAM, driver, OS, backend, adapter features, clocks/power/thermal state when available, resolution, formats, scene counts, quality settings and effective mode. Use deterministic frame paths and seeds. Collect at least 120 warmup frames and 600 measured frames per case, in serial ABBA order with three independent ABBA blocks for finalists. Capture cold construction separately; do not hide it in warmup.

Publish per-run medians, p95, p99 and the raw per-frame series; do not pool away a slow run. Primary acceptance requires every finalist block's measured run to meet the thresholds. Use paired block differences and confidence intervals when claiming an improvement. If a proposed optimization's benefit is within repeat-to-repeat noise, call it inconclusive and retain the simpler implementation unless another measured advantage justifies it.

Separate timestamp regions for geometry deformation dependencies, BLAS builds/updates, TLAS, grid/proposals, candidate selection, temporal/spatial reservoir work, visibility, repair classification/trace, radiance temporal/spatial filters and composition. Also report an encompassing GPU interval, CPU extraction/upload/submit time and whole-frame timing. Timestamp-only tests do not establish latency under normal asynchronous scheduling. Profile isolated work for diagnosis and the normal graph for the final critical-path result.

Required counters: total candidate evaluations; valid/invalid history; local/global proposal use; grid overflow; selected and unique traced lights; primary/correction/repair/contact rays; query-active fraction; queue occupancy/overflow; repair pixel coverage; current BLAS/TLAS generations; built/updated triangles/instances; cache hits; peak requested allocation bytes and driver-reported memory where available. Do not infer actual hardware traversal steps from ray count; use an appropriate profiler where supported and label estimated quantities.

Measure light counts 1/8/64/256/1,024/4,096/16,384 with both distributed and overlapping influence. The 65,536 boundary test may be a small-resolution correctness case. Plot GPU time and error against both light count and overlapping-light count. A bound on queries is useful evidence, but “flat scaling” requires showing the observed range and separately accounting for preparation and scene costs.

## Controlled ablations

| Variant | Change from its immediate control | Question |
| --- | --- | --- |
| A | ScreenSpace at current main | Reproduce historical behavior at new resolutions |
| B | Pure RT visibility, existing guided sampler, complete RT repair | What does hardware visibility alone cost and fix? |
| C | B plus split/compacted query dispatch | Does coherence/inactive-work reduction exceed bandwidth and dispatch cost? |
| D | Best valid B/C plus tile presampling, same candidate/ray budget | Can shared proposals reduce selection work without losing discovery? |
| E | D plus limited temporal/spatial ReSTIR, explicit correction accounting | Does persistent reuse beat guided sampling at equal quality or equal time? |
| F | Best valid variant plus optional short screen contacts | Does the hybrid improve missing-detail quality enough to justify its cost/errors? |
| G | Best valid variant plus optional world-space proposal grid | Do separated rooms justify ReGIR-like maintenance and memory? |

Do not enable E/F/G together before measuring them separately. Include full precision versus packed storage and one/two/four samples in a bounded parameter sweep. Select on development fixtures, then run all frozen seed variants and withheld camera paths without retuning. Keep a quality-versus-time frontier rather than ranking variants by milliseconds alone.

## Fair engine comparisons

The research report's engine table compares mechanisms only. A later measured comparison needs matched scene assets, light bounds/intensities, material models, camera paths, geometry coverage, GBuffer/output resolution, exposure, sample counts, denoising warmup and shadow semantics. Disable GI/reflections/volumetrics or report them separately. For engines without equivalent many-light RT, state that the control is clustered/raster direct lighting; do not present it as an RT algorithm contest.

Use Unreal GPU/graph profiling with asynchronous overlap handled explicitly, then measure normal scheduling as well. Include its RT scene cost and any shadow-map passes still required by selected lights. Godot/Unity/Bevy results must name exact engine revisions and effective rendering settings. Do not compare screenshots with different shadow coverage, or infer performance from unmatched published FPS. A full external-engine benchmark suite is a later validation task, not evidence already established by this planning PR.

## Delivery and issue workflow

The planning PR contains research and this protocol only. It references the implementation issue without a closing keyword so that merging documentation cannot close unimplemented work. The planning PR may move to In Review when the document, source links and required CI pass. The implementation issue remains Todo until implementation starts, with the design available for review.

Implementation PRs must include the exact tested revision, scenes, raw results, visual captures, supported geometry/material matrix and known limits. Mark ready and In Review only when their declared scope passes. If the 4 ms goal fails, report the failure rather than relabeling partial functionality as completion. Leave merge decisions to the maintainer.
