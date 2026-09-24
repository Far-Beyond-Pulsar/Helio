# Tiny voxel terrain pass

One editable voxel world and one stored-terrain Helio backend. Enable the `engine` feature and register `engine::LazyEngineVoxelPass` through `helio_default_graphs::build_default_graph_external_with_voxel_passes`. The graph inserts registered voxel passes after opaque geometry and before decals and deferred lighting. The lazy pass allocates residency buffers only after a matching source publishes a frame.

Publish an `EngineVoxelFrame` through `SharedVoxelFrame`: high-precision integer cell origin and fraction, camera basis, `Arc<World>` and the sunlight setting. Helio supplies the actual jittered camera and internal render size. The pass writes the existing eight GBuffer attachments, depth and directional visibility on the graphics command-encoder timeline.

Call `ready()` to distinguish arrival loading from a publishable terrain cut. Ordinary movement can continue while `terrain.stats()` reports pending work. Complete cuts publish after generation; missing data is not rendered as air. `set_stage_profiling(true)` exposes generation, primary, GBuffer and sunlight timing spans.

`World::apply_edit` updates exact occupancy and spatial indexing. `journal::JournalWriter` persists snapshots asynchronously and drains on drop. CPU picking/collision remain exact at 10 cm; distant GPU occupancy is reconstructed from sampled density and is approximate. The recipe and journal format belong to this backend; SceneDB's generic `VoxelTerrainComponent` selects it by an opaque renderer ID.

```powershell
cargo test -p helio-pass-tiny-voxel --features engine --release --lib -- --test-threads=1
```

GPU tests require an adapter. They check exact generated materials and actual graph hits/resize/AA composition; passing them is not a visual or frame-rate qualification.
