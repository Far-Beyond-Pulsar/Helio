# Volumetric fog dispatched froxel work when neither camera fog nor volumes were active

## Resolution

Fixed in [18d2458b93509595e61d6101e62bb5030149f15e](https://github.com/Far-Beyond-Pulsar/Helio/commit/18d2458b93509595e61d6101e62bb5030149f15e).

## Problem

The default graph retains a volumetric-fog pass for stable resource wiring.
Merely having the pass in the graph does not mean that the current scene
contains participating media. The two-million-cube benchmark has no relevant
postprocess volumes and does not enable camera fog, yet the previous execute
path had no renderer-driven inactive gate before its expensive froxel work.

This issue records a real inactive-pass optimization, **not** the cause of the
roughly 648 ms frame. Initial attribution was misleading because other passes'
GPU markers were on the wrong encoder. The subsequent corrected measurements
identified sparse LightCull and culling parallelism as the main bottlenecks.
No reliable isolated fog-before timing is available; one must not invent a
speedup or assign the missing frame time to fog.

## Fix and behavior

- Add `active` state to VolumetricFogPass, defaulting to true for existing
  standalone users of the pass.
- Expose `set_active`; Renderer updates it from
  `has_pp_volumes || camera.postprocess_settings.fog_enabled`.
- Return at the beginning of execute when inactive, before froxel dispatches.
- Keep the pass/resources in the graph instead of rebuilding graph topology.

The volume condition is deliberately conservative: the presence of PP volumes
keeps fog eligible because a volume can enable it even when the camera default
is off. It does not prove every such volume actually contributes fog, nor does
it optimize every possible empty or out-of-range volume configuration.

## Evidence and validation limits

In the completed final sweep on RTX 4060 Laptop GPU/Vulkan at 800×600, the
inactive `VolumetricFogPass` scope averaged **0.00298 ms**. That includes scope
overhead; it is not an active volumetric rendering cost. Whole-graph time was
4.0665 ms and actual final cadence was 5.2349 ms / 191.02 FPS after the other
fixes as well.

The default-graph benchmark compiled and completed 300 initial frames plus
20 dynamic checkpoints with this gate enabled. Source inspection verifies
that camera fog or PP volumes enable the pass and that the default remains
active for callers that do not use the new setter. A dedicated active-volume
visual regression was **not** performed in this task; the benchmark only
exercises the inactive scene. The other GPU tests must not be presented as
coverage of active volumetric appearance.

## Reproduction

Run `cargo run --release -p examples --bin two_million_dynamic` without camera
fog or PP volumes and query `gpu_pass_samples` for `VolumetricFogPass`. Inspect
the execute early-return and renderer activation condition in the diff below.

## Before/after code

The actual pass-state, execute guard and renderer integration diff is appended
to the issue. The renderer diff also contains target-clear profiling added in
the same fixing commit; that instrumentation is documented separately.
