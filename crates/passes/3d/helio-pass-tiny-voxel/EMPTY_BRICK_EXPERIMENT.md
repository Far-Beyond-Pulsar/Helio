# Exact empty brick certificate experiment

Pre-comparison protocol, 26 September 2026. Parent runtime: `156f6a7b`; instrumentation is the current uncommitted goal work. Candidate changes no voxel values, selection policy, generation batch budget, far representation or visible resolution.

The baseline work diagnostic measured 551.6 exact-cell loop iterations per primary ray at the returned ground view, and 65.0 visited leaves. The candidate reduces all 2,048 material words in each generated exact brick to a nonempty certificate. Traversal skips the entire brick only when that certificate is zero. The certificate is generated on the same encoder after its payload, and slots retain the existing immutable publication lifetime. This costs 256 KiB at the maximum 65,536-slot pool. Finer occupancy masks are deferred until measurement justifies them.

Required checks before retention:

- GPU certificate agrees with complete payloads, including one occupied cell in the last word and clearing a reused slot. Existing exact CPU/GPU material and far traversal tests pass.
- Full flight passes hit/coverage, composition, resize and remote-edit checks. Compare all matching settled PNGs to the unmodified baseline, inspecting any differences. Dynamic refinements can have different schedules and are reported separately.
- The diagnostic specialization agrees byte-for-byte on status, direction and distance with the actual primary ray buffer; counts are measured outside frame timing.
- Two alternating baseline/candidate flights at 1280x720 native. No other GPU workload or compiler. Retain only if walking primary GPU p95 improves at least 20%, and walking/descent full-graph and synchronized p95 do not regress by more than 10% in the paired runs. Record the complete stages, worst frames, allocation delta and failed trials. This small experiment does not establish overall performance acceptance.
- If retained, check a 1920x1080 Quality flight and inspect movement captures. Further scene, far-fidelity, arrival and engine integration gates in `PLANET_TERRAIN_GOAL.md` remain open.

## Initial correctness finding and control adjustment

The first diagnostic reduced the ground view's mean exact-cell checks from 551.6 to 120.7, but the unaccelerated replay disagreed on a far hit by 0.052 m. That trial failed correctness and is retained under `target/voxel-goal/empty-brick-work` with its executable. It is not a qualified optimization.

The old fine DDA accumulated its next-crossing distances. At brick exit, `max(box.far, t)` retained positive rounding residue, so a later sampled-density brick could start with a different fractional anchor when preceding empty bricks were skipped. Both paths now reset to the shared box exit plane. The reference diagnostic records the complete hit separately from work counts, and compares all 32 hit bytes against the actual primary result.

Before any performance comparison, the control will be rebuilt from the same source as the candidate with only the `STORED_SKIP_EMPTY` default disabled. Both include the boundary fix, certificates and instrumentation. This isolates traversal skipping; the original renderer remains a historical comparison. Capture-time reference dispatches and recording are disabled during paired timing runs. No paired performance result is recorded yet.

## First paired result: failed tail-time gate

The exact replay passed all captured flight stages once it reused actual primary direction bits (independent shader normalization differed by two ulps in the earlier diagnostic). Complete hits matched with skipping enabled and disabled. The returned-ground diagnostic used 120.573 exact steps per ray versus 551.597 without skipping, with identical leaf/far iteration counts.

Two alternating 720p pairs finished without hit/coverage failures. Walking primary GPU p50 fell from 12.926/14.068 ms to 6.399/6.304 ms. However, candidate p95 was 18.022/18.561 ms against control 13.964/19.712 ms: the required 20% tail improvement failed. Candidate spikes cluster at walking steps 102-109 in both runs, after generation has completed. This is not an accepted optimization. Raw runs are `target/voxel-goal/paired-{control,candidate}-{1,2}`; `paired-results.csv` records frame and graph timings. Investigation will audit step 106 on the same path before changing the implementation or protocol.

## Idle repeat and spike investigation

The video-paused repeats did not qualify the candidate. Walking primary p95 was 17.678/19.752 ms for controls and 18.114/7.471 ms for candidates; candidate 1 descent synchronized p95 rose from 17.572 to 29.017 ms. Preserve `idle-{control,candidate}-{1,2}` as failed trials. Idle preflight alone is insufficient to attribute these spikes to external load.

Auditing walking steps 106 and 119 preserved all hit bytes. Step 119, despite an 18.397 ms primary dispatch, used only 96.101 exact checks per ray (reference 545.018), at most 326 leaf visits and 459 far steps. Holding that pose for 300 frames let the transient subside. NVIDIA clocks stayed near 1.95 GHz; this does not establish the cause. A per-process GPU-engine monitor perturbed timing and produced no slow walking samples. Nsight Systems Vulkan capture succeeded; hardware counters were unavailable due to driver permissions. These are diagnostic runs, not acceptance measurements.

## Parent-link traversal follow-up protocol

Before comparison: test an additional representation change replacing the dynamically indexed, 28-entry private ancestor array with immutable parent links in the shared GPU tree. The child/parent layout adds 1 MiB at maximum tree capacity and removes the production specialization's per-ray stack. This is a candidate for lower local-memory pressure, not a diagnosed explanation of the transient. NVIDIA's [shader profiler guidance](https://docs.nvidia.com/nsight-graphics/UserGuide/shader-profiler.html) identifies dynamically indexed private arrays as a potential local-memory cost; actual benefit must be measured here.

Both new control and candidate use the identical interleaved node layout, occupancy generation, leaf-boundary fix and instrumentation. Control disables both parent links and empty skipping; candidate enables both. The unaccelerated capture replay disables both and compares every hit byte to the production output. Require the existing full-flight and GPU tests, including deep real planetary cuts, and the same two-pair 720p p95 gates stated above; preserve all failures. Also compare against the saved skip-only executable to distinguish the incremental traversal result. If qualified, run 1080p Quality and movement/sunlight checks. Do not relax any whole-goal requirement.

## Parent-link result and disposition

Two alternating pairs passed full-flight correctness but failed performance qualification. Walking primary p95 changed from 14.000 to 7.713 ms in pair 1 (-44.9%) and 18.587 to 16.879 ms in pair 2 (-9.2%). The second pair missed the 20% requirement. Walking full-graph/synchronized p95 improved in both pairs; descent full-graph changes were -4.6%/+1.8% and synchronized changes +7.3%/-8.7%, within the separate 10% bounds. Raw results remain under `target/voxel-goal/parent-{control,candidate}-{1,2}` and `parent-paired-results.csv`.

The parent-link layout and traversal are reverted. Their complete source snapshots, patch, shaders and executable hashes are preserved under `target/voxel-goal/parent-source`, `parent-links-source.patch` and the named binaries. The follow-up does not support attributing the transient to a private ancestor stack.

Empty-brick skipping remains experimental and **disabled by default** (`STORED_SKIP_EMPTY=false`). Capture-time replay can exercise it against the actual unaccelerated output on the identical resident cut. Its exact certificate and regression checks remain available for further work. The 256 KiB certificate allocation and generation reduction remain a measured diagnostic cost; no shipped frame-rate improvement is claimed. The shared leaf-exit boundary correction is retained as a correctness fix. Far reconstruction, arrival delay, shading/aliasing and complete-engine performance remain unresolved.
