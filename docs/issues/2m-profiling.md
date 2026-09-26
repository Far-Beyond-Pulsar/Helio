# GPU pass attribution omitted work recorded on the graphics encoder

## Resolution

Fixed in [18d2458b93509595e61d6101e62bb5030149f15e](https://github.com/Far-Beyond-Pulsar/Helio/commit/18d2458b93509595e61d6101e62bb5030149f15e).
This is a retrospective issue documenting a verified fix, not an unresolved
request. The implementation and tests are already on Helio's main branch.

## Problem and exact cause

The two-million-object scene ran at roughly 1–2 FPS, while the original pass
breakdown failed to explain the GPU frame duration. It was unsafe to conclude
that the missing time belonged to SceneDB/CDB, presentation, or fog.

The graph placed pass markers on `compute_encoder`, but passes including
LightCull invoke `begin_compute_pass` through `ctx.encoder_ptr`, which points
to the graphics encoder. A shader being compute work does not imply it is
recorded in the command buffer named Compute Graph. The measurement therefore
bracketed an almost-empty command stream instead of the expensive dispatch.
Sharing a query set across these sequential encoders was not established as
the cause; the placement of its markers was the demonstrated bug.

Additional accounting problems: CPU timing covered prepare without execute;
repeated names could overwrite CPU samples or export a GPU aggregate several
times; summing graph envelopes and their child passes double-counted GPU work.

## Fix

- Bracket each serial pass, including render bundles, on both encoders.
- Close equal-name nested scopes in reverse order so each start/end pair uses
  the same command stream; aggregate the durations by logical pass label.
- Use separate compute and graphics graph envelopes, resolved after both
  streams' final timestamps. Preserve the renderer's target-clear scope.
- Disable render-pass fusion while profiling so individual scopes remain
  attributable. Normal non-profiled chaining remains available.
- Accumulate CPU prepare and execute times, including repeated labels.
- Export one row per aggregate label; use graph envelopes for total GPU time
  rather than adding parent and child durations together.
- Preserve frame indices, lag, availability, readback drops and overflow counts.

## Measurements and interpretation

RTX 4060 Laptop GPU, Vulkan, 800×600, profiling enabled. With corrected markers
but original shaders, the graph measured about **648.6 ms**: LightCull
**562.667 ms**, frustum culling **42.509 ms**, occlusion **36.438 ms**, and
ShadowMatrix **4.622 ms**. Those measurements identified the actual fixes.

After the rendering fixes, the final graph is **4.0665 ms**, while actual
present-to-present cadence is **5.2349 ms / 191.02 FPS**. These are different
metrics; reciprocal GPU duration is not actual FPS. Per-pass sums are
3.9187 ms; the 0.1478 ms envelope difference is retained as boundary time,
not falsely assigned to a shader. Target clear is outside the graph at
0.0049 ms. No readback-slot drops or query overflows; delivered lag ≤2 frames.

The final window has 120 cadence samples and 109 distinct GPU source frames.
Async latest-snapshot delivery can repeat or supersede frames even with no
readback-slot drops. Queries must use the epoch-aware deduplicated views.

## Reproduction and validation

Run `cargo run --release -p examples --bin two_million_dynamic`, then query
`gpu_pass_samples` by source checkpoint. CPU samples remain attached to their
own `profiler_frames.frame_id`, not the GPU readback's source frame.

`cargo test --release -p helio-core --features profiling --lib profiling::`
passed four tests: CPU accumulation, duplicate GPU labels, saturating duration
aggregation, and an actual GPU dual-encoder regression. The latter verifies
one exported aggregate, source-frame/lag metadata, nonzero measured work,
zero overflow, and agreement between envelope, snapshot and exported totals.

See [the complete evidence report](https://github.com/Far-Beyond-Pulsar/Helio/blob/18d2458b93509595e61d6101e62bb5030149f15e/docs/two_million_diagnosis.md).

## Before/after code

The GitHub issue includes the actual implementation-commit diff for the graph,
CPU profiler, snapshot/export code and renderer instrumentation below.
