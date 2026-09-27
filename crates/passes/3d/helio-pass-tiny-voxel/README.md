# Tiny voxel terrain pass

One editable voxel world and one stored-terrain Helio backend. Enable the `engine` feature and register `engine::LazyEngineVoxelPass` through `helio_default_graphs::build_default_graph_external_with_voxel_passes`. The graph inserts registered voxel passes after opaque geometry and before decals and deferred lighting. The lazy pass allocates residency buffers only after a matching source publishes a frame.

Publish an `EngineVoxelFrame` through `SharedVoxelFrame`: high-precision integer cell origin and fraction, camera basis, `Arc<World>` and the sunlight setting. Helio supplies the actual jittered camera and internal render size. The pass writes the existing eight GBuffer attachments, depth and directional visibility on the graphics command-encoder timeline.

Call `ready()` to distinguish initial loading from a publishable terrain cut. It stays true during movement and teleports: the previous complete cut remains visible while the new view refines. `stats().refining`, `planning`, and `pending` describe refinement, and `needs_frame()` keeps an idle editor ticking until it finishes. This does not certify that the arrival already has exact nearby detail. Complete cuts publish after generation; missing data is not rendered as air. Graph rebuilds on the same device retain residency when the source mailbox is unchanged. `set_stage_profiling(true)` exposes generation, primary, GBuffer and sunlight timing spans.

`World::set_voxel_size` selects an authored base size from 0.1 m to 1 m in 0.1 m increments. This changes the actual base grid, not a visual LOD. Integer addresses and edit radii retain canonical decimetre/half-decimetre units, so changing the grid does not rescale the planet. Queries and generation use the same integer sample representative, including negative coordinates and cubes crossing storage bricks. Existing recipes default to 0.1 m.

`World::apply_edit` updates exact occupancy and spatial indexing. `journal::JournalWriter` persists snapshots asynchronously and drains on drop. CPU picking/collision remain exact on the selected base grid; distant GPU occupancy is reconstructed from sampled density and is approximate. The recipe and journal format belong to this backend; SceneDB's generic `VoxelTerrainComponent` selects it by an opaque renderer ID. Generic live payload chunk dimensions do not dictate this backend's private acceleration bricks.

```powershell
cargo test -p helio-pass-tiny-voxel --features engine --release --lib -- --test-threads=1
```

GPU tests require an adapter. They check exact generated materials and actual graph hits/resize/AA composition; passing them is not a visual or frame-rate qualification.

The `helio-default-graphs` example `voxel_flight` exercises the full deferred graph, SceneDB sunlight, temporal resolve, walking, orbital descent and resize. It writes captures and a CSV of synchronized CPU submission plus GPU completion times (excluding readback and presentation):

```powershell
cargo run -p helio-default-graphs --release --example voxel_flight -- target/voxel-flight/run 1280 720
cargo run -p helio-default-graphs --release --example voxel_flight -- target/voxel-flight/quality 1920 1080 quality
```

Set `HELIO_VOXEL_FLIGHT_RECORD=1` for a separate visual run. Open its `movement.html` to play or scrub every walking and descent frame. Playback speed is illustrative, not measured FPS. Readback changes worker scheduling, so do not compare its residency timing with an unrecorded run. Each capture includes a hit-status audit; the example rejects traversal exhaustion and missing terrain after initial loading. Its first-frame magenta sentinel checks final graph composition.

`HELIO_VOXEL_FLIGHT_BASE_METRES=0.3` selects a 30 cm authored grid for the initial flight; supported values are 0.1 through 1.0 in 0.1 m increments. The final grid-replacement stage still selects 1 m. The default initial grid remains 10 cm. Generated diagnostics belong under the ignored `target/` or local `validation/` directories and must not be committed.

`HELIO_VOXEL_FLIGHT_PROFILE=1` also writes `gpu.csv` and `memory.csv`. GPU samples identify the frame that produced them, including the graph clock restarting after resize. The whole graph scope is not a sum of child passes. Logical terrain allocations exclude driver/allocator overhead, shared graph attachments, diagnostic allocations and CPU caches. `frames.csv` separates CPU submission from the synchronized GPU wait. The last pending readbacks and those discarded with a rebuilt graph may be absent; `examples/voxel_flight/analyze.py` reports those gaps rather than relabeling old results.

`HELIO_VOXEL_FLIGHT_TRACE_WORK=1` replays captured primary rays outside frame timing and records leaf/exact/far iteration counts. It checks the full hit against both accelerated and unaccelerated traversal of the same resident cut using the original ray direction bits. This diagnoses empty-space work and protects traversal changes; it does not establish that the sampled far field matches canonical terrain.

`HELIO_VOXEL_FLIGHT_CANONICAL=1` compares 144 actual primary rays with the exact editable volume at each of six settled flight poses. The [canonical fidelity audit](CANONICAL_FIDELITY_2026_09_26.md) records the brick-entry correction and the remaining far-field differences. Sparse ray agreement does not establish pixel coverage or silhouette fidelity.

The opt-in `canonical-far-experiment` feature evaluates canonical occupancy inside far leaves. The [reference experiment](CANONICAL_REFERENCE_2026_09_26.md) records precision regressions, source-snapshot isolation and successful sampled comparisons. Direct recipe evaluation is much too expensive for normal rendering; the feature stays disabled by default and does not qualify a replacement far representation.

The `helio-default-graphs/voxel-reference` feature and
`HELIO_VOXEL_SURFACE_REFERENCE=N` run spatial reference patches through the
engine's linear HDR lighting. N is a power of two from 2 through 16, giving
N*N samples per pixel. Use native resolution, at most 65,536 pixels. This mode
enables canonical far occupancy, materializes camera rays before traversal,
disables temporal AA, automatic jitter and reflections, and captures lighting before
AA/postprocessing. It retains separate face/material coverage and lit face
contributions. It is an offline reference for the current material and lighting
model, not a fast terrain path.

```powershell
cargo +1.98 build --release -p helio-default-graphs --features voxel-reference --example voxel_flight
$env:HELIO_VOXEL_SURFACE_REFERENCE = '4'
.\target\release\examples\voxel_flight.exe target/voxel-goal/surface-4 96 54 native
$env:HELIO_VOXEL_SURFACE_REFERENCE = '8'
.\target\release\examples\voxel_flight.exe target/voxel-goal/surface-8 96 54 native
python crates/helio-default-graphs/examples/voxel_flight/analyze_surface.py target/voxel-goal/surface-4 target/voxel-goal/surface-8
Remove-Item Env:HELIO_VOXEL_SURFACE_REFERENCE
```

The default 32 cases cover slopes, a horizon, carved openings, thin shells and
destroyed shells, including close views, two sun directions and 2.5 cm camera
translations. `HELIO_VOXEL_REFERENCE_CASES` can select comma-separated fixture
names, for example `slope,ridge`. Repeated frames must match, the GPU camera
must equal the CPU camera, and sampled center rays must agree with the CPU
source. The comparison script requires identical center rays/lighting across
the two sampling runs. CSV depth ranges are sampled estimates, not conservative
bounds. Finite sampling, unregistered image differences during translation and
small diagnostic crops do not establish convergence, motion quality or speed.
All output stays local under ignored `target/`; do not commit it.

`HELIO_VOXEL_FLIGHT_SUN_WORK=1` separately replays sunlight rays from primary terrain hits. Use with `HELIO_VOXEL_FLIGHT_SUN=1` outside timing runs. It saves traversal maxima and the ray origin/direction for exhausted traces; it does not establish shadow fidelity or account for later mesh coverage.

The current far representation is not a filtered reduction of exact edited leaves. That fidelity gap, refinement latency and visual aliasing must be measured independently of the exact CPU query and generation tests. This backend is not yet qualified for AAA quality or seamless exact-detail arrival.

See the [26 September validation report](VALIDATION_2026_09_26.md) for current source provenance, measurements with the separate GPU job paused, movement captures and the rejected larger-batch experiment. The [preceding report](VALIDATION_2026_09_25.md) records the implementation checks and earlier measurements.

The active [planet terrain goal](PLANET_TERRAIN_GOAL.md) records required behavior and acceptance targets. The [empty-brick experiment](EMPTY_BRICK_EXPERIMENT.md) records the current optimization protocol and its correctness findings.

The [region-bounds and corner-traversal checkpoint](BOUNDS_AND_TRAVERSAL_2026_09_26.md) records tighter canonical certificates, fewer requested bricks, the sunlight boundary-cycle regression, and full-flight validation. Far fidelity and performance remain unqualified.

`HELIO_VOXEL_FLIGHT_SUN=1` enables directional terrain visibility and rejects exhausted/invalid sunlight values in each capture. `HELIO_VOXEL_FLIGHT_AUDIT_WALK=N` audits one chosen walking frame; `HELIO_VOXEL_FLIGHT_HOLD_WALK=1` holds the last pose for a diagnostic repeat. These options perturb the workload and must be reported with measurements. Empty-brick skipping remains experimental and disabled by default; see the [profiling checkpoint](VALIDATION_2026_09_26_GOAL.md) for failed optimization trials and the retained boundary correction.
