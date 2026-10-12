# helio-pass-voxel-planet

Destructible voxel worlds for Helio: Earth-sized cube-sphere planets, finite
planes and effectively infinite planes, built from exact voxels of 0.1 m to
1 m, editable everywhere and at any scale, and drawn from **chunk trees**:
64³-cell chunks generated on the GPU, streamed as a screen-space cut of a
quadtree per cube face, and rasterized as boxes that rays enter only inside
one chunk ([`docs/chunk-trees.md`](docs/chunk-trees.md)).

This document maps the system for people who will work on it: what each part
does, which invariants hold it together, how to measure it and how to extend
it. The Pulsar editor integration is documented in Pulsar-Native's
`docs/voxel-system.md`.

## Contents

1. [Goals](#goals)
2. [Crate map](#crate-map)
3. [Data model](#data-model)
4. [Chunk trees](#chunk-trees)
5. [Invariants and decisions](#invariants-and-decisions)
6. [Measuring](#measuring)
7. [Tests](#tests)
8. [Extending](#extending)
9. [Known limits and open work](#known-limits-and-open-work)

## Goals

- **No visible refinement.** Moving at any speed, from the ground to orbit,
  the player never sees detail arrive or levels switch: finer chunks fade in
  over coarser ones, and residency prefetches where the eye is going.
- **Dig anything, anywhere.** Brushes from one voxel to the whole planet,
  generated caves and overhangs. An edit shows the next frame; a dug region
  renders and streams like untouched ground, however many edits made it.
- **One world, two consumers.** The CPU (collision, ray casts, edits,
  gameplay queries) and the GPU (generation, drawing) evaluate the same
  integer field and the same edits, to the bit.
- **Budget.** Terrain under 6 ms GPU at 1440p (TSR at 0.75) on an RTX 3060.

## Crate map

| Path | Role |
|---|---|
| `src/grid.rs` | The canonical grid: equal-angle cube sphere (or plane), cells, levels, faces, exact cell walking maths. |
| `src/noise.rs`, `shaders/noise.wgsl`, `shaders/wide64.wgsl` | Bit-exact integer noise (Q16 and fine Q24), with hardware 64-bit products where the device has them. The only noise terrain may use. |
| `src/terrain.rs` | Pluggable generators: `TerrainGenerator` -> `TerrainField` (CPU) + `TerrainProgram` (WGSL). Registry, material ids, `check_field`, `lean_height`. |
| `src/layers.rs` | The built-in generator `helio.terrain`: ordered layer stacks (`TerrainLayers`), presets (Earth, moon, desert, flat), validation and compilation. |
| `src/landform.rs`, `shaders/landform.wgsl`, `src/ridge_envelope.rs` | The stack interpreter (CPU and WGSL): octaves, layer composition, craters, erosion, caves, overhangs and material styles; coarse levels' ridge means. |
| `shaders/world.wgsl`, `shaders/material_state.wgsl` | Grid mapping, layer quantization and material ids shared by generation and terrain programs. |
| `src/edits.rs` | Brushes (sphere/cube, remove/add/paint), per-face integer resolution (`FaceBrush`), the ordered `EditLog` and its spatial index. |
| `src/edit_store.rs` | Baked edits: what sealed brushes did to each cell, in sparse 8³ bricks with majority-built coarser levels. |
| `src/planet.rs` | `PlanetRecipe` + `Planet`: canonical queries (`kind`, `material`, `solid`), exact ray casts, `surface_point`, `air_clearance`, `ground_height`; `Edits` (large, baked and recent brushes, sealing, undo). |
| `src/journal.rs`, `src/snapshot.rs` | Saving edits: an append-only brush journal, and snapshots of what the edits left. |
| `src/chunks/mod.rs` | `ChunkWorld`: chunk generation jobs, GPU storage (slots, leaf pool, materials), edit sync, the draw (seed raster, trace, sun and sky light, gbuffer publish), picks, timings. |
| `src/chunks/residency.rs` | The planner (on a worker thread): which tiles are drawn, generated next and evicted for a moving eye. |
| `src/chunks/drawn.rs` | The drawn cut: drawn chunks, column tops and level fades, uploaded as the changes of each frame. |
| `src/chunks/pass.rs` | `ChunkPass`, the render-graph pass hosts add: reads a `PlanetFrame`, writes Helio's gbuffer and `directional_visibility`. |
| `src/engine.rs` | What hosts share with the pass (`PlanetFrame`, picks, appearance) and the terrain half of the shaders (`terrain_source`), checked by `verify_field`. |
| `shaders/chunk_generate.wgsl` | Tile columns, chunk columns (field, volume terms, lean, edits, materials), 64-tree build, eviction. |
| `shaders/chunk_draw.wgsl` | The cut walk, chunk tracing, fades, sun shadows, sky visibility, appearance and gbuffer publish. |
| `shaders/chunk_table.wgsl` | Scatters a frame's chunk-table changes. |
| `tests/chunks.rs` | GPU correctness against the CPU planet (see [Tests](#tests)). |
| `../../../helio-default-graphs/examples/voxel_flight.rs` | The flight harness: held views, flights, digs, picks, captures. |

## Data model

### Grid

A planet is a cube sphere with an **equal-angle** mapping: on each of the six
faces, cell boundaries are planes through the planet centre at equal angle
steps, and radial layers are concentric spheres. Every cell is therefore
aligned with gravity: flat ground is flat everywhere, and a straight ray
crosses each boundary family in closed form (planes: a linear equation in the
face's angle frame; spheres: a stable quadratic).

- **Cells** are addressed `(face, i, j, k)`: `i, j` horizontal indices on the
  face, `k` the radial layer (height above the datum in layers; negative below
  it). Layer thickness equals the voxel size, quantized to whole millimetres.
- **Voxel size** is authored per world (0.1 m to 1 m). The *shape* of the
  terrain does not depend on it: generators sample a fixed **reference
  lattice** of 0.1 m (`REFERENCE_VOXEL`), and a world of 0.3 m voxels samples
  that lattice at its own cell centres.
- **Levels**: level `L` cells are `2^L` base cells wide and tall. A **tile**
  is 64x64 columns of one level; a **chunk** is 64³ cells of one level, a
  tile's layer `cz`.
- **Planes** use the +Y face basis with axis-aligned cells and horizontal
  layers; an infinite plane is 2^27 reference cells (~13 400 km) across.

### Terrain field

A generator is a pure function of a column: `height(p, level)` in integer
millimetres and `ground_material(p, top_height, depth, slope, layer)`.
It exists twice, in Rust (`TerrainField`) and WGSL (`TerrainProgram`), and the
two must return equal integers for every input. That is the contract that lets
the CPU ray cast what the GPU draws.

- All arithmetic is wrapping integer maths on the shared noise library.
  `noise_fine` / `mul_fine` (Q24, exact wide products) exist because Q16 noise at
  continent wavelengths is constant over metres and steps by one unit; scaled
  by kilometres of relief that became long straight terraces.
- The noise domain is a sphere at the planet's radius (planes: the
  horizontal position), in 1.25 cm units: domain distance is physical
  distance, the field is smooth across cube edges, and tangent directions
  (slopes, gullies) are defined everywhere. Integer `sphere_point`
  straightens the equal-angle cube coordinates with a tan polynomial and
  normalizes them (Q30 Newton reciprocal square root, exact 64-bit
  products).
- 32x32 -> 64-bit products (`mul_wide`, `mul_shr`) are 16-bit limb
  sequences in noise.wgsl between `// wide:begin` and `// wide:end`; on
  devices with `SHADER_INT64` that region is replaced by wide64.wgsl's
  native `u64` products, with the same bits (`tests/wide_products.rs`).
- Every world is an ordered layer stack (`layers.rs`): Warp, Continents,
  Mountains, Hills, Roughness, Erosion, Craters, Basins, Plateau, each with
  a mask (everywhere, land, above deep sea), plus caves, overhangs and a
  material style (Earthlike, Lunar, Layered, Rules). The stack is data,
  compiled to one octave table sorted coarse to fine; one interpreter runs
  it on CPU and GPU, so changing layers never recompiles shaders. Bounds
  (`bound_margins`, `height_range`) derive from the layers, so any stack
  keeps `check_field`.
- Craters: one candidate per cell of a 3D lattice per size, seamless across
  cube edges; parabolic bowl, smooth-min rim, ejecta to twice the radius.
- Erosion octaves lay stripes across the downhill direction of the coarser
  terrain (analytic gradients through the domain warp); their gradient
  steers the finer octaves, so gullies branch.
- Detail finer than a level's footprint is omitted at that level: coarse
  levels are band-limited samples of the same field. Their display height
  keeps the conditional mean of the first Mountains layer's unresolved
  ridges (`ridge_envelope.rs`), so distant ranges keep their mass; level 0
  and canonical queries are unchanged.
- Heights are relative to the datum (the planet radius or the plane's y = 0)
  and may be negative. Nothing may clamp heights to the datum.

### Edits

A `Brush` (centre in planet metres, radius, sphere or cube, remove / add /
paint, material) is resolved per cube face into integer half-cell coordinates
(`FaceBrush`), so CPU and GPU apply the same integer containment test. Later
brushes override earlier ones; at level `L` a brush smaller than half a level
cell is omitted.

A planet's `Edits` hold every brush in three layers, applied in order:
**large** brushes (too big to bake, kept analytic), the **baked** store
(`EditStore`: each cell's whole history collapsed into one `CellEdit`, in
8³ bricks per column, coarser levels by majority) and the **recent** brushes
(analytic, undoable one by one). Recent brushes are sealed into the store in
batches; memory follows what the edits left, not how many there were, and the
latest seals stay undoable. A history hash identifies every state, so a
`ChunkWorld` syncs by difference: changed brushes regenerate the tiles they
touch first (drawn ones, nearest first, within the frame's budget: a small
dig shows the frame it is made, a blast's far and hidden tiles follow over
the next frames), changed baked columns regenerate as budget allows, and an
unrelated history rebuilds the world.

The `EditLog` keeps an adaptive octree per face whose leaves list, in order,
only the brushes that can still change a cell inside them: CPU queries (ray
casts, `surface_point`) cost what is exposed at a point, however long the
history.

### Planet

`Planet` = recipe + field + edits. It answers the canonical questions:
`kind(cell)`, `material(cell)`, `solid(cell)`, exact `raycast` (a cell walk
clipped to the world shell or plane), `surface_point(p, clearance)`,
`air_clearance(eye)` (conservative distance to any possible solid cell, used
for near planes) and `ground_height(eye)` (height above the ground below,
used for camera speed). Gameplay, physics and the editor use these; they
never touch the GPU.

## Chunk trees

[`docs/chunk-trees.md`](docs/chunk-trees.md) has the design and every
phase's measurements. In short, a frame of `ChunkPass`:

1. **Residency** (`ChunkWorld::update`). The planner thread keeps a
   quadtree of tiles per face from every coarsest tile down, refined while
   a tile's cells would project larger than `lod_pixels` from the eye (or
   from where it will be in `prefetch_seconds`), and sends the drawn set's
   changes and the most wanted tiles. The render thread generates tiles
   within a GPU budget (`generate_ms`, against the measured cost of a tile's
   columns and of a chunk, so a tall edited tile costs its chunks): tile
   columns (field height,
   surface word, volume extent), then each chunk's columns as 64-bit masks
   (volume terms, lean, large brushes, baked cells, recent brushes,
   materials), then the chunk's 64-tree into the leaf pool. Results come
   back a frame later: empty chunks are freed, a tile whose measured layers
   need more chunks is generated again in place.
2. **The drawn cut** (`drawn.rs`). The chunk table holds drawn chunks and
   the cut's columns with their tops (a max pyramid over ancestors);
   switches between levels fade over `FADE_FRAMES` frames with a per-pixel
   dither. Only a frame's changes are uploaded.
3. **Draw** (`ChunkWorld::draw`). Chunk boxes are rasterized (seed); each
   pixel's ray enters only the boxes it hits and walks the cut
   (`walk()`): inside a chunk the 64-tree, between chunks the column tops,
   across faces the neighbour's frame. Rays that reach below a tile's
   generated layers report it, and the tile is generated through its
   whole volume (caves).
4. **Light.** Sun shadows are walks toward the sun from quarter-resolution
   samples with a compacted fallback for hits no sample serves; sky
   visibility is four short rays per 4x4 block. Both use the same walk.
5. **Publish.** Material colour (patches, pigment, edge darkening), corner
   AO and the shading normal go to Helio's gbuffer; Helio's deferred
   lighting, atmosphere and TSR do the rest.

## Invariants and decisions

- **CPU/GPU bit identity.** Terrain programs, noise, the lean lattice and
  edit containment are integer and mirrored; `engine::verify_field` and
  `tests/chunks.rs` enforce it. Never introduce floats into a field.
- **Camera-relative frames.** The host's world origin is the eye; chunk
  frames are affine maps rebased on the GPU from f64 origins, so every GPU
  position is small at any planet size.
- **Every coarsest tile is a root.** Residency does not depend on where
  the eye started. A sphere's tiles below the eye's horizon over its
  lowest ground are not wanted (from orbit the whole visible disc is
  drawn; a plane stops at `max_distance`), and a generated root ages out
  like any tile.
- **Edits are part of a tile's extent.** A tile's ranges include the
  layers its brushes and baked bricks change, so the floor of a deep dig
  is near an eye standing on it and its chunks are generated.
- **Brushes are exact.** A ball's crossing of a column is solved in f32
  with a bound on its error; cells within that bound of either end are
  tested with exact 64-bit squares (`FaceBrush::contains`).
- **Settled means settled for this eye.** A plan counts only for the eye
  and view it was made for, so hosts that render until settled keep
  rendering after the camera moves.
- **A frame costs what changed.** Plans are diffs from the worker; the
  chunk table is patched on the GPU; edits regenerate the tiles they touch.
- **Overhangs lean the heightfield.** A cell is solid where it lies below
  the field's height at a horizontally displaced point,
  `z < H(x + W(x, z))`: `W` (`terrain_lean_offset`) varies with height at
  the ledge spacing and eight times more slowly across the ground, so the
  solid is a continuous deformation of the heightfield's (no floating rock,
  no tears). Generation evaluates the field on a global lattice per level
  (`terrain_lean`: nodes every `spacing` cells within `reach`; a brick
  column's 64 lanes evaluate its 8x8 nodes once) and passes the bilinear
  height at the displaced point plus the column's own detail off the
  lattice (`terrain::lean_height`).
- **Caves** are a network under a rock cover: tunnels narrow to nothing
  towards the cover and the cave depth, caverns raise their threshold as
  they close, and the cover opens only in entrance zones.
- **Brushes.** Cubes are tested in each face's half-cell index space (a
  one-block cube is exactly one cell); spheres are balls in the seamless
  volume space (`Grid::volume_point`), round at any size and depth, the
  planet's centre included. Both use exact 64-bit squares.

## Measuring

Everything is measured in Helio's graph or on `ChunkWorld` directly with
`helio-default-graphs`' `voxel_flight` example (its module docs list every
variable):

```
cargo run -p helio-default-graphs --release --example voxel_flight -- OUT [W H [native|quality]]
```

| Variable | Effect |
|---|---|
| `HELIO_VOXEL_FLIGHT_HOLD=<view>` | Settles `ground`, `mountain_slope`, `tunnel`, `face_edge`, `cave` or `eye` (`HELIO_VOXEL_FLIGHT_EYE=x,y,z,fx,fy,fz`) and holds it: `CHUNKHOLD` prints the primary, seed and shadow GPU times. |
| `HELIO_VOXEL_CHUNKS_ENGINE=1` | The held view drawn by the renderer's `ChunkPass` (`CHUNKENGINE`), as a host draws it; saves `chunks_<view>.png`. |
| `HELIO_VOXEL_CHUNKS_DIG=<n>` | Digs `n` balls across the held view and undoes them: generation per edit, when it shows, and whether the undone frame matches. |
| `HELIO_VOXEL_CHUNKS_BLAST=<r>` (with `_DIG`) | One ball `r` m across, half sunk beyond the view centre, then undone: frames to settle, tiles, the worst frame's GPU and CPU, against the CPU raycast. |
| `HELIO_VOXEL_CHUNKS_FLY=descent\|cruise` | Streams from 400 km down to 2 m, or at 300 m/s 40 m over the ground: detail lag (`coarse`), primary and CPU update times. |
| `HELIO_VOXEL_CHUNKS_PROBE=x,y;...`, `HELIO_VOXEL_PICKS=x,y;...` | Pixels' drawn hits (or picks) against CPU ray casts. |
| `HELIO_VOXEL_CHUNKS_COUNT=1` (with `HELIO_VOXEL_CHUNKS_HEAT=1`, `HELIO_VOXEL_CHUNKS_PNG=<path>`) | A counting build: steps, boxes and hits per pixel (`CHUNKCOUNT`); with `HEAT` the colour target shows each pixel's primary steps (red to 64, yellow to 128, white to 192). |
| `HELIO_VOXEL_CHUNKS_BUDGET`, `HELIO_VOXEL_CHUNKS_GENERATE_MS`, `HELIO_VOXEL_LOD_PIXELS` | Tiles a frame, GPU generation a frame, cells per pixel of the cut. |

Measuring pitfalls: the GPU's clock drops whenever it idles (synchronous
readbacks, frames waited for one by one), so compare interleaved A/B runs and
their minima, not runs minutes apart; other desktop applications share the
GPU.

## Tests

`cargo test --release -p helio-pass-voxel-planet -- --test-threads=1` (CPU
unit tests, `tests/chunks.rs`, `tests/wide_products.rs`) and
`cargo test --release -p helio-default-graphs --test voxel_pass_graph --
--test-threads=1` (the pass inside the deferred graph).

- Field: `terrain_programs_are_bit_identical_to_cpu` (every generator and
  shape), the field's own unit tests (no steps between columns, bounds,
  caves and overhangs without floating rock).
- Drawn hits near the eye match CPU ray casts: ground, across a face edge,
  natural caves (shallow and 800 m deep), overhangs, planes.
- Edits: brushes, flat-topped boxes, a 600 m shaft at its bottom and half
  way up, a 3 km crater's floor, 18 000 block edits, and sealed edits with
  undo under a running world.
- Solid gaps: a hall dug 300 m under the ground and a world's hollowed core,
  seen from inside, with no chunks in the rock over them.
- Coverage from orbit; picks report the drawn hit.
- Residency: the planner (roots wherever the eye starts, parents drawn
  until all children exist, eviction), the drawn set and its fades.
- Graph: the pass builds, settles from any first eye, holds still views
  settled, drops its world with the source and resizes; appearance edits
  show in one frame; the atmosphere follows the eye and sun.

## Extending

**A terrain generator.** Implement `TerrainGenerator` (id, version, info with
an optional settings-component name) returning a `TerrainField` and its
`TerrainProgram` (WGSL defining `TerrainConstants`, `terrain_height`,
`ground_material`, plus the constants' bytes). A program may also define
`terrain_surface(p, level, height) -> u32`, an 8-bit surface word per column
passed to `ground_material` (what materials need besides height: the
erosion term, fresh ejecta). Volumetric generators also define
`terrain_extent` and `terrain_density` (and `TerrainField::extent`,
`density`, `volume_bounds`), leaning ones `terrain_lean` and
`terrain_lean_offset` (`TerrainField::lean`, `lean_offset`), and optionally
`terrain_clearance` (cells a column's scan may step over below the lean
band). A density is the signed distance from the cell centre to the surface
(256 per level cell, positive inside solid): combine terms as constructive
solid geometry on distances, and keep the extent tight, since every cell in
it is evaluated when a chunk is generated. Return an empty extent at levels
that cannot show a feature. Use only the integer noise library. Register it
with `terrain::register`; bump its version when a released generator changes
its output (saved edits record it). Add a test calling
`engine::verify_field` for every shape and `terrain::check_field`.

**Materials.** Drawing knows a material only through its
`MaterialAppearance` (16 per world): colour and roughness, optional
world-space patch colours, a lip material, flecks and the host a speck
blends into. No material id is special to the renderer. A stack picks
materials by style, or by up to 16 ordered `MaterialRule`s (height, slope,
depth, moisture, erosion, noise patches, strata, specks), so a game's biomes
change without recompiling shaders.

**Appearance.** `ChunkPass::set_appearance` updates the material table and
detail weights without regenerating terrain. RGB is sRGB; roughness is
linear. When it returns `true`, reset temporal history to show the change
in an idle viewport.

**A brush shape.** Extend `BrushShape`, its per-face resolution (culling
extent, height bounds) and `FaceBrush::contains` in `edits.rs`, and
`brush_cells` in `chunk_generate.wgsl`.

## Known limits and open work

- Generation's cost is mostly the terrain program per column.
- A tile marks one solid gap between its chunks; a second gap of rock
  keeps a chunk per layer.
