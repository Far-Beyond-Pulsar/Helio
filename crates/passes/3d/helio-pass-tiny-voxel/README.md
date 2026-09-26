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
```

The current far representation is not a filtered reduction of exact edited leaves. That fidelity gap, refinement latency and visual aliasing must be measured independently of the exact CPU query and generation tests. This backend is not yet qualified for AAA quality or seamless exact-detail arrival.
