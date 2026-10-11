# Chunk trees: the voxel terrain architecture

Status: decided 2026-10-11; replaces span columns (`span-columns.md`) and
the column tracer. `aaa-architecture.md` keeps the measurements that led
here.

## Goal

Terrain made of voxels of 0.1 to 1.0 m, on planets, planes and infinite
planes:

- **No visible refinement.** Moving at any speed, from the ground to orbit,
  the player never sees detail arrive, levels switch or coarse blocks
  replaced by fine ones.
- **Dig anything, anywhere.** Brushes from one voxel to the whole planet,
  caves and overhangs. An edit shows the next frame. A dug region renders
  and streams like untouched ground, however many edits made it.
- **AAA look and speed.** Exact voxels near the eye, filtered appearance
  where voxels shrink below a pixel, sun shadows, sky occlusion, AO.
  Terrain under 6 ms GPU at 1440p on an RTX 3060.

## Why the span-column design cannot get there

Each problem below is structural; tuning inside the design measured 0-14%
per attempt (`aaa-architecture.md`).

1. **A step costs ~700 lane-cycles.** The research tracers spend 30-60
   instructions a step. Ours walks the curved cube-sphere grid directly
   (shell and plane solves every box exit), looks up a column per 8x8 cells
   through a hash table, decodes spans, lane words, relief cuts and level
   transitions, and only skips air vertically. Tunnels and overhangs cost
   44 steps a ray.
2. **Refinement is visible because detail is late or coarse.** Levels are
   independent windows of procedural evaluation. The pool fills before
   cells reach a pixel at every ring (`lod_pixels` is capacity-bound:
   pressure moved level 0 in 1.3-2.8x), and moving generates rings that
   arrive frames after they are needed. Nothing hides a switch: the coarse
   surface is simply replaced.
3. **Edits are history.** A dug column is regenerated from every brush that
   touches it, at every level, every time it becomes resident. Cost grows
   with the number of edits, not with what they left.
4. **Shading recomputes the world.** Natural ground stores no material, so
   every pixel runs the material rules, noise and slope stencils
   (3-5 ms).

## The design

### 1. Data: a planet-wide octree of chunk trees

- The world is the existing integer cube-sphere grid (`grid.rs`): face,
  cell indices, level. Voxels stay upright everywhere and keep their exact
  integer identity; collision, picks, saves and the CPU mirror address
  them as today.
- Space is divided into **chunks** of 64^3 voxels of one level (6.4 m at
  0.1 m voxels). A level-L chunk has 8 children at level L-1: the chunks
  form an octree over each cube face (planes: one face; infinite planes: a
  root grid).
- A chunk's contents are a **64-tree**: three levels of 4^3 nodes, each a
  64-bit child mask plus the index of its first child (children packed,
  addressed by popcount rank). Leaves are 4^3 voxels: a 64-bit occupancy
  mask. Uniform nodes (all air or all solid) are a flag, not storage.
- **Attributes per solid surface voxel** (solid with an air face): an 8-bit
  material at the finest generated level; at coarser levels also an
  averaged colour (RGB565) and normal (octahedral, 16 bits) of the voxels
  it covers. Buried voxels keep only a material run per leaf, enough to
  expose them when dug.
- Budget at 1440p (90 degree view, voxels at most a pixel wide where each
  level ends): level 0 reaches ~130 m, every level holds 2-4 M surface
  voxels, ~20 levels hold 40-80 M. At 2-5 bytes a surface voxel (leaf masks
  shared by ~16 surface voxels, a material byte, coarse attributes) that is
  200-400 MB of a 12 GB card: the pool is never what sets the LOD.

### 2. Generation: chunks produced once, on the GPU

- A chunk job evaluates the existing terrain contract (`terrain_height`,
  `terrain_density`, `terrain_extent`, `ground_material`,
  `terrain_surface`): heights for its 64^2 columns, density only where the
  interval bounds say a voxel can be mixed (caves, overhangs), as today.
- The job writes the 64-tree directly, with materials evaluated once per
  surface voxel (the rules no longer run per pixel) and, at coarse levels,
  the averaged colour and the exact field normal (no staircase normals from
  quantized heights).
- The field's octave rules already keep a level within its parent's voxel
  (erosion steered only by coarser octaves). Chunk generation keeps that
  invariant and a test enforces it: a chunk's surface lies within one
  parent voxel of its parent's surface.

### 3. Edits: data is the truth once a chunk exists

- An edit is a GPU pass over every resident chunk the brush touches, at
  every level: fine chunks voxel-exact, coarse chunks by brush coverage per
  voxel. Parents of edited chunks are re-derived by downsampling (edited
  octants) and keep their generated values elsewhere. The edit shows the
  next frame.
- Edited finest-level chunks are read back asynchronously, compressed and
  kept on the CPU and in the save. Evicting and reloading them costs a
  copy, not a regeneration.
- The brush log remains only for space no resident chunk covered when the
  brush was made (planet-scale brushes): the edit BVH culls it at chunk
  generation, as today. Brushes smaller than half a voxel of a level are
  skipped at that level, which downsampled data already represents.
- Cost follows the surface the edits leave: an edited chunk is a chunk.

### 4. Residency and LOD: a screen-space cut of the octree

- Each frame the CPU (on the residency worker) selects the octree cut whose
  voxels project to at most one pixel (the `lod_pixels` target), with
  hysteresis, the way Nanite and GigaVoxels select clusters and bricks.
- Children replace a parent only when all of their non-empty siblings are
  resident: no partial holes, no fallbacks inside a frame.
- **Prefetch:** the cut is also computed for the eye 0.5-1 s ahead along its
  velocity, and those chunks are requested first. Priority is the screen
  error a missing chunk would cause, so fast flight streams what it is
  about to need.
- **Invisible switches:** a region changing level is drawn by both levels
  for ~8 frames with complementary dither masks that move from one to the
  other; TSR resolves them into a crossfade. Combined with sub-pixel voxels
  at the switch, no transition is visible.

### 5. Rendering: rasterized chunk boxes, rays only inside one chunk

- **Culling:** a compute pass culls the cut against the frustum and the
  previous frame's Hi-Z and writes an indirect draw list, front to back.
- **Primary visibility:** each visible chunk draws its box, tight to its
  occupied range. The fragment shader transforms the ray once into the
  chunk's frame and walks the 64-tree with an integer DDA (64-bit masks,
  popcount, ~30-60 instructions a step), the method of Teardown, Aokana and
  sparse 64-tree tracers. It writes a visibility buffer (chunk, voxel,
  face) and conservative depth (early-Z survives), so hidden boxes behind
  a hit are rejected before they run. The hardware rasterizer does what the
  sky bound, summary blocks, air blocks and the tile pre-pass tried to do.
- **The chunk's frame is affine.** Within a chunk the cube-sphere mapping is
  linear to far below a voxel: the deviation over a 64-voxel chunk is
  512 x voxel / radius of a voxel, 1e-5 at level 0 on Earth. Levels where it would
  pass 5% of a voxel (from about level 13 on Earth, where voxels are
  already sub-pixel) use 32^3 or 16^3 chunks. Boxes overlap by one voxel so
  neighbouring frames leave no cracks.
- **Depth is shared.** The terrain writes the same depth buffer as Helio's
  meshes, so props, characters and editor geometry compose with it.
- **Shading** resolves the visibility buffer in compute: stored material and
  attributes, per-voxel hash variation, filtered appearance from the stored
  coarse colour and normal, AO from neighbour masks in the same tree. No
  generator code runs per pixel.
- **Sun and sky:** compute rays from the hit through the same chunk trees
  (a per-level chunk table finds the chunk, the DDA walks it), climbing to
  coarser levels with distance; temporal accumulation through TSR.

### What survives and what goes

Survives: the integer field, landform interpreter and terrain contract with
its CPU mirror and bit-identity tests; the noise library (with hardware
wide products); the cube-sphere grid; the edit log, BVH and journal (for
unresident space and saves); the Pulsar component, tools and picks; TSR and
the sky.

Goes, once the new path passes its gates: span columns, lane words, column
records and their hash table, summary blocks, air blocks, the sky bound and
horizon pass, the column tracer and its fallbacks, relief fractions and
surface bytes, the per-pixel material rules and slope stencils, content
windows, the column residency planner.

## Phases and gates

Every phase is measured in hold mode against the current renderer on the
same views (ground, mountain_slope, tunnel), at 1080p and 1440p, with
interleaved runs. A phase that misses its gate stops the plan for review.

1. **Chunk trees and primary visibility.** GPU generation of chunk trees
   from the field (no edits, fixed cut by distance), raster boxes and the
   chunk DDA, a plain shade. Gate: primary at most half of today's on
   mountain_slope and tunnel, ground no slower; resident memory and
   generation time per surface voxel reported.
2. **Cut selection, streaming and transitions.** Screen-space cut,
   prefetch, crossfades. Gates: no level switch with voxels over 1 px;
   descent from orbit and fast flight with no visible refinement (frame
   differences at transitions below the TSR noise floor); arrival settled
   within 250 ms.
3. **Edits on data.** GPU brush pass, downsampled parents, persisted
   chunks, brush log for unresident space. Gates: edit visible next frame;
   a 10,000-brush dig renders and streams at the cost of undug ground.
4. **Appearance and lighting.** Stored materials, coarse attributes, AO,
   sun and sky rays, look parity with today in the editor. Gate: terrain
   under 6 ms at 1440p in every view; then Pulsar switches over and the old
   path is deleted.

## Phase 1 results (2026-10-11)

`src/chunks.rs`, `shaders/chunk_generate.wgsl`, `shaders/chunk_draw.wgsl`;
measured in the flight harness's hold mode (`HELIO_VOXEL_CHUNKS=1`), 1080p,
RTX 3060, interleaved with the span-column renderer in the same binary:

| View | Span columns | Chunk trees |
|---|---|---|
| ground | 3.41 ms | 4.30 ms |
| mountain_slope | 12.57 ms | 5.74 ms |
| tunnel | 13.78 ms | 4.61 ms |

- The cut at 1 px holds ~4,100 tiles per level (a ring of
  [D, 2D] where D = voxel / pixel angle, 3 pi (D / 64 voxels)^2 tiles), 12
  levels in view, 58-63 k chunks, 180-215 MB (leaves 9 bytes, chunk nodes
  1.6 KB). Generation of the whole cut takes 3.4-4 s on the GPU (tiles,
  column masks and trees: ~6 ns a column).
- Caves under the surface are content no eye ray reaches from above; the
  whole cave band at every level overflowed 131 k chunks on a mountain.
  Phase 1 generates it only within four chunks of the eye; phase 2's
  residency follows visibility (rays request what they reach).
- Depth: chunk boxes do not overlap, so a box's front face orders the hits
  correctly; writing it instead of the hit's depth keeps early depth tests
  (conservative depth writes did not keep them on this GPU). With the
  depth test off the ground view costs 15.9 ms: early rejection works.
- Rejected: a box per 16x16-column group tight to its layers (16 boxes a
  chunk): ground 3.75 -> 4.59 ms, mountain 5.6 -> 7.0 (16x the triangles,
  and every group face a ray crosses starts another fragment).
- Ground was the open gate. Starting from the previous frame's exact
  depth (perfect occlusion culling) changed nothing (3.7 vs 3.8 ms): hidden
  boxes cost nothing. Letting each pixel run only the box holding its hit
  cost 3.3-3.4 ms (fragment), the same rays traced in compute 1.5-2.0 ms:
  fragment shaders run 2x2 quads, and every box edge on screen splits quads
  between boxes.

**Primary visibility, revised: seed by raster, trace in compute.** The
boxes are rasterized only to seed each pixel with the nearest box its ray
enters (a fragment writes the box's index; 0.2-0.3 ms). A compute pass
traces each ray in its seed chunk and continues across chunk faces: the
chunk beyond (`chunk_table`: same level, else coarser or finer), and space
with no chunk (air) crossed as empty chunks of the level in the last
chunk's frame. One round resolves every pixel with terrain (ground 90%,
mountain_slope 61%, tunnel 100% of pixels; the rest is sky); rays crossing a
cube face edge are left to a second round (the next face's neighbours are
phase 2). Raster peeling instead of continuation (re-seeding beyond each
miss) needed 4-8 rounds at 0.3+ ms each: rejected.

| View | Span columns | Fragment walk | Seed + compute |
|---|---|---|---|
| ground | 3.41 ms | 4.30 ms | 3.5-3.9 ms |
| mountain_slope | 12.57 ms | 5.74 ms | 4.6-5.4 ms |
| tunnel | 13.78 ms | 4.61 ms | 4.3-4.5 ms |

Rays cross 1.65 chunks on the ground (5.4 mountain, 4.4 tunnel) and take
16.6 tree steps (39, 34.5): the walk inside chunks is the cost, about 560
M rays/s on the ground. The 64-tree does not remove steps near grazing
surfaces (cell steps through mixed leaves); the old tracer took 11-44
steps at about 700 lane-cycles, this one about 400. Next for primary: steps
near surfaces (leaf-level distance, a per-column top in leaves), the start
of each ray from the previous frame's hit chunk.

## Phase 2a results (2026-10-11): streaming

`src/chunks/residency.rs` (the planner, on its own thread),
`src/chunks/drawn.rs` (the drawn set and chunk table, updated by what
changed), `ChunkWorld::update`; the flight harness's
`HELIO_VOXEL_CHUNKS_FLY=descent|cruise` (60 simulated frames a second,
the spawn loaded first).

- A refined tile stays drawn until all four children are generated, and
  children are requested only once their parent's layers are measured (2
  frames): a child's chunks then span only layers that can hold a surface
  (~2 chunks a tile; estimates from unmeasured parents made 14, mostly
  empty). A settled mountain loads in 6.2 s instead of 123.
- Leaves live in blocks of 9 size classes with free stacks on the GPU;
  evicted chunks return their blocks (`evict` pass); slots are recycled on
  the CPU. Generation results come back without waiting (a readback ring).
- Chunk frames are written once per slot (origin as a high and a low f32,
  rebased to the eye on the GPU); a frame uploads only the list of drawn
  slots (draw CPU 42 -> 1.6 ms). The drawn set and the chunk table change
  by tiles (table 12 -> 0.7 ms a frame), the plan runs on a worker thread
  (update CPU 40 -> 2 ms p50).
- Tiles in view are generated first; the budget is GPU time (3 ms a frame
  at 50 us a tile: ~55 tiles).

| Flight (1080p) | coarse in view, mean | primary p50 | generation p50 | update CPU p50 |
|---|---|---|---|---|
| cruise, 300 m/s at 34 m | 6.5% | 2.5 ms | 2.7 ms | 2.0 ms |
| descent, 400 km to 2 m in 12 s | 6.9% | 1.3 ms | 2.8 ms | 2.3 ms |

`coarse` is the share of drawn tiles in view whose cells project larger
than 1.5 px: detail still to arrive. Cruise at 300 m/s and 34 m needs about
4,000 new tiles a second at the finest two levels; generation at 50 us a
tile spends 3 ms a frame on it.

## Phase 2b results (2026-10-11): the chunk pass in the render graph

`src/chunks/pass.rs` (`ChunkPass`): residency, primary visibility and
Helio's gbuffer from the same frame source as the span-column pass, so
Helio's lighting, sky and TSR apply unchanged. The harness's
`HELIO_VOXEL_CHUNKS_ENGINE=1` installs it in place of `PlanetPass`.

- A ray that leaves its chunk walks the drawn cut. The chunk table also
  holds each drawn tile's column (its chunk layers) and, for it and every
  ancestor, the top solid cell under it: a max pyramid over the cut. At
  the cell past the face it left by, the ray finds the drawn tile over
  that column (the same level, coarser where the level has none, finer
  under an ancestor), then the chunk at its layer, an empty layer, or the
  air over the tile, grown to the coarsest column whose top lies below the
  ray; air is crossed in one step. Solid chunks are kept (dropped as
  "uniform" they left holes inside slopes).
- This replaced cubes of air that grew blindly every few steps: they
  jumped over drawn finer chunks inside them, so rays hit the side of a
  tile beyond the surface (grey slabs at the valley edge, ~5 m behind the
  exact hit). Tops in whole layers made poor bounds (rays that miss
  crossed 117 boxes); in cells, 27.
- GPU times come back through a ring of timestamp slots without waiting.
- `ChunkWorld` records through `helio_core::CommandRecorder`, so the pass
  records on the graph's stream.

| View | span columns | chunk trees, hold (1080p) | chunk pass in the graph (TSR) |
|---|---|---|---|
| ground | 3.41 ms | 1.79 ms | 1.9 ms |
| mountain slope | 12.57 ms | 4.06 ms | 4.8 ms |
| tunnel | 13.78 ms | 2.71 ms | |

Interleaved with the growing cubes: ground 1.88 -> 1.79 ms, mountain 4.40
-> 4.06, tunnel 2.75 -> 2.71, cruise primary p50 2.69 -> 2.58; every probe
now matches the exact CPU raycast.

A plan used to cost the main thread a pass over every drawn tile and a
buffer write per run of changed table entries (update p95 14 ms in the
cruise, a hitch per plan). The worker now sends what changed since its
last plan, and the changed entries go up in one buffer that a compute pass
scatters into the table (`chunk_table.wgsl`):

| Flight (1080p) | drawn set, mean | table, mean | update CPU p50 | p95 |
|---|---|---|---|---|
| cruise, before | 2.35 ms | 1.10 ms | 2.05 ms | 14.4 ms |
| cruise | 0.24 ms | 0.08 ms | 1.83 ms | 3.9 ms |
| descent | 0.19 ms | 0.07 ms | 1.40 ms | 3.4 ms |

The descent's coarse share fell from 6.9% to 5.1%.

### Crossfades between levels

A tile replaced by finer or coarser ones no longer pops: for 16 frames
both levels are drawn and each pixel shows one, by a dither that moves
each frame, so TSR blends them (`drawn.rs`). A fading tile's state in the
chunk table holds its first frame; the trace works out the fade's progress
q from its own frame, so a fade costs nothing while it runs. Incoming
tiles show where the dither is under q, outgoing ones where it is not, so
every point is shown once (a test checks this for every frame and dither
over random cuts, with plans that land mid-fade). A pixel at a tile it
does not show looks finer under the coarse side and coarser over the fine
side; air in a finer tile is crossed in that tile's own frame (in the
coarser frame its top rounded and rays stalled). A plan that changes a
switch under way ends it at once, with every switch it overlaps.

| Cruise (1080p), interleaved | without fades | with fades |
|---|---|---|
| fading tiles, mean | 0 | 2,450 (5% of drawn) |
| hits / seeded rays | 0.998 | 0.998 |
| primary p50 | 2.72, 2.85 ms | 2.98, 3.27 ms |
| update CPU p50 | 1.92, 1.86 ms | 2.37, 2.59 ms |

Settled views pay about 0.1 ms (mountain hold, fastest of four
interleaved runs: 4.07 -> 4.18 ms; runs vary 4.1-5.3 ms with the GPU's
clocks).

### Across cube faces

A ray leaving a face continues on the face across the edge
(`across_edge`): the faces are equal-angle, so across an edge an index
runs on mirrored and along it cells meet exactly on the edge, an integer
map from the face bases. Air on the new face is crossed in the frame of
one of its chunks. Boxes stop at the face's edge (a coarse tile can reach
past it, its columns there empty, and a ray crossing them skipped what lay
across). Rays used to end at the edge: 40 m from it the terrain beyond was
sky. The harness's `face_edge` view looks across the +Y face's edge; its
probes on the next face match the exact CPU raycast (52.68 m), and the
holds are unchanged (mountain 4.17 -> 4.19 ms, ground 1.83 -> 1.85,
fastest of three interleaved).

### Where generation goes

Generation is timed by kernel (`FrameTimings::stages`). In the cruise, a
tile costs 47 us in tile columns (the terrain program, once per column:
warp, erosion gradients, ridges), 1.9 us in chunk columns and 4 us in
chunk builds. Streaming throughput is the terrain program's: the cruise at
300 m/s, 34 m up keeps 5-7% of the drawn tiles in view coarser than
wanted (the crossfades hide their switches). The program already skips
octaves finer than a level's cells (`landform_resolved`).

Left for phase 2: residency by ray reach (caves); the terrain program's
cost per column.

## References

Laine and Karras, *Efficient Sparse Voxel Octrees* (2010); Crassin et al.,
*GigaVoxels* (2009); Careil, Billeter and Eisemann, *Interactive Editing of
Large Sparse Voxel DAGs* (HashDAG, 2020); Fang et al., *Aokana: a GPU-driven
voxel rendering framework for open world games* (2025); Teardown's renderer
(Gustafsson, 2020-22); sparse 64-tree tracing (dubiousconst282, 2024);
Karis, *Nanite* (2021) for the screen-space cut; *Filtered Appearance for
Voxels* (HPG 2023); Distant Horizons for LOD crossfades.
