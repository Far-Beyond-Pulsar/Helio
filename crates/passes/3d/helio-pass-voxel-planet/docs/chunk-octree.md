# Chunk octree: 3D residency, drawing and refinement

Status: decided 2026-10-15, replacing the tile quadtree of `chunk-trees.md`
(phase 2). The chunk format (64³ cells as a 64-tree), generation kernels,
the in-chunk walk, exact cells, lighting and edits' integer maths stay.

## Why

The terrain is volumetric (caves, overhangs, whole-planet destruction), but
residency was a quadtree of 64×64-column tiles with a stack of chunk layers
each:

- A tile refined its whole stack as one: a cliff, a cave 800 m down or a
  dug shaft refined with the surface above it, and an eye in a cave refined
  the ground far over its head.
- A tile held one surface band. Layers under it were assumed rock (`BELOW`)
  until a ray reached them, then the whole volume was generated: caves
  popped open. One solid gap per tile (`Gap`); a second kept a chunk per
  layer. Floating rock, hollow worlds or stacked caves did not fit.
- Refinement was by distance only, all around the eye: occluded ground and
  ground behind the camera cost nearly what visible ground did (one level
  coarser), and tiles were evicted 2 s after their last visit, so turning
  back generated them again.

## The structure

- A **node** is a chunk: `(face, level, cx, cy, cz)`, 64³ cells of its
  level. Its 8 children are `(level - 1, 2cx + a, 2cy + b, 2cz + c)` (cells
  halve in all three axes, so layers nest).
- A **footprint** `(face, level, cx, cy)` is the 64×64 columns over a stack
  of nodes. It stays the unit of column generation (heights, surface words,
  volume extent: `tile_columns_main`) and of classification: its measured
  ranges and its edits decide what each node in it is.
- **Roots** are every node of the coarsest level from the planet's centre
  (or a plane's lowest layer) to the ceiling.
- A node is one of:
  - **Mixed**: may hold a surface; generated into a slot.
  - **Empty** or **Solid**: uniform from its footprint's ranges and edits;
    drawn without data (air to cross, rock to hit).
  - **Buried**: within the field's volume (caves may lie there) but under
    the surface band and away from the eye; drawn as rock until a ray hits
    it, then generated (it was the `BELOW` path of a whole tile).

## Residency: a cut of the octree

The planner (worker thread) visits the octree from the roots each plan:

- A Mixed node is refined while its cells project larger than `lod_pixels`
  from the eye or the prefetch eye (3D distance to its band's box), with
  the `coarsen` hysteresis, motion allowance and the horizon test as
  before. Empty, Solid and Buried nodes are never refined.
- **A band switches as one.** A node's band is its footprint's contiguous
  Mixed layers; every test (distance, view, rays, horizon, readiness) is
  the band's, so the surface band of a footprint is at one level, as a
  tile's column was. A coarse cell's ground, rounded, may lie one level
  finer under the node above or below it: nodes of one band refined apart
  left the surface between them (holes through a fade's dither, seen in
  descents). Separate bands (caves, digs) still switch apart.
- **Rays decide where detail goes.** The trace samples one pixel an 8×8
  block each frame (jittered over 64 frames) and reports the node at the
  level its hit wants (cells of a pixel, two levels of margin). The
  planner marks that node and its ancestors seen. A node not seen within
  `seen_frames` refines only to `unseen_pixels` times `lod_pixels`
  (occluded ground, ground behind the camera): a level coarser. Hits on
  Buried nodes request their generation.
- A refined node is drawn as its children once every Mixed child is
  generated; until then it is drawn itself. Children are classified from
  the parent footprint's measured ranges, widened by a parent cell (as
  child ranges were); a footprint measured later reclassifies its nodes.
- **Eviction is by pressure, least recently visited first.** Generated
  nodes stay resident until chunk slots or leaf pages run short; then the
  nodes the plan has not visited the longest go. Revisits cost nothing.

## Drawing: one table entry per node of the cut

The GPU chunk table holds, for every node of the cut and every ancestor of
one, `(face | level << 8, cx, cy, cz) -> (slot, state, kind, content)`:

- `state` as before (`choice`): shown, look finer, or fading with its
  start frame.
- `kind`: mixed, empty, solid, buried.
- `content`: the node's occupied box in its own cells (6 bytes): a mixed
  chunk's occupied layers over its whole footprint; an ancestor's, the
  union of its children's, rounded out. It generalizes the old column tops
  (a max pyramid of heights) to 3D.

`place_at` finds the node holding a point (coarser while none, finer
through refined ancestors), then:

- a shown Mixed node: its slot;
- Solid or Buried: a hit where the ray enters (Buried reports itself);
- Empty, or outside an ancestor's content box: air, grown to the largest
  slab of the coarsest ancestor whose content box the point is outside of
  (above, below or beside it), so rays cross open air and dug halls in a
  few steps as they did over column tops.

## Generation

A frame's batch groups the wanted nodes by footprint, then a chunk job per
node. A footprint's columns (heights, surface words, volume extents) and
lean halos stay on the GPU in a **footprint pool** (2048 slots, 70 KB each,
reused least recently first, never one the batch uses): they depend only on
the terrain, so a footprint is computed once while it stays in use,
whichever of its nodes are generated when, edits included. The batch
lists the footprints it computes first (the column kernels run over them;
their two-rate nodes are per-batch scratch), then those it reads from the
pool. Results read back as before: computed footprints' ranges (reclassify
their nodes), chunk occupancy (empty chunks free their slot).

Edits mark the nodes their bands cross at every level: drawn and visited
ones regenerate (urgent, as before); cached ones not in the cut are
evicted (they regenerate if wanted again).

## What goes

`Gap`, `BELOW`/`ROCK` via gaps, the `Reach` buffer and deep tiles,
`layers_needed`'s surface band, tile-level `Layers`/bands and their
regeneration on measurement, `COLUMN`/`GAP` table entries, column tops,
age eviction of generated tiles, the tile quadtree planner and its tests.

## Gates

Measured with `voxel_flight`'s stress flights in the editor's world
(`HELIO_VOXEL_FLIGHT_SEED=75 HELIO_VOXEL_STRESS_AT=0,1,0
HELIO_VOXEL_FLIGHT_SUN_ELEVATION=32`, 2560×1440, real time) and the
refinement gate (`CHUNKGATE`: 8×8 blocks of a flown frame that differ from
the settled view past two settled captures' noise):

- the gate's mean and worst share fall in cruise, descent and glide;
- level switches over 3 px fall;
- `mountain_slope` hold primary and sun/sky times within noise of the
  quadtree's (the walk's loop must not grow);
- `tests/chunks.rs` passes; no lost rays (`HELIO_VOXEL_CHUNKS_COUNT`).

## Results (2026-10-15)

Built as designed, with the invariants below found on the way. `tests/chunks.rs`
(19 GPU tests: ground, face edges, caves, deep caves, overhangs, planes,
brushes, a 600 m shaft, a 3 km crater, a hollowed core, an underground hall,
18 000 block edits, sealed edits, orbit coverage, picks) match CPU ray casts
exactly; the graph tests pass.

### Invariants: a node answers for its descendants

A coarse node's own cells are a band-limited sample; classifying it by them
alone hid what finer levels hold, and the cut never refined to it:

- **Rounding.** Coarse cells round their tops down: ground 10 m over the
  datum lies in layer -1 from level 16 to 5 and in layer 0 at level 4. The
  field's band is widened by `DESCENT` (2) cells, and a Mixed node whose
  chunk comes back empty is still refined (drawn as air).
- **Caves.** `terrain_extent` is empty where a level cannot show a cave.
  The Buried band reaches `TerrainField::volume_bounds` under the surface at
  every level.
- **Small edits.** A brush under half a cell is inactive at a level, and
  coarse baked bricks are built by majority (a 4 m shaft vanishes at 6.4 m
  cells). Brushes classify at any size; `BrickIndex` marks every brick's
  node and all its ancestors.
- **Buried nodes never hold a parent back.** They are rock until generated:
  only those holding the eye (`near_cells`) or met by rays are generated.
  A four-chunk radius generated hundreds of invisible chunks a level.
- **Only Mixed children gate a switch.** A refined node that drew a
  missing child as air flashed holes through the fade's dither.
- **Rays only keep nodes from refining.** Coarsening a refined node no ray
  met oscillated at occlusion edges: its coarser cells poked out, were met,
  and refined it again. Ray reports are not structural events: they come
  every frame, and counting them kept the world from settling.
- **Settled waits for the rays to sweep the view.** A block's report
  samples one of its 64 pixels a frame, so a plan settled before 64 frames
  of reports since the last change may yet refine where a pixel not
  sampled meets a node its distance does not ask for (seen: a still view
  settled at frame 59, then asked for 8 chunks). Hosts render about 64
  frames past the last change before going idle.
- **The lookup** climbs past bare ancestors over the hidden fine side of a
  switch, and looks above an ancestor it climbed to for a coarser node
  fading in (only while something fades: sun and sky walks call it
  constantly).

### Measured against the tile quadtree

Editor world (seed 75, pole, 32 degree sun), 2560x1440, fixed-step flights
(`HELIO_VOXEL_STRESS_STEP=1`: every run sees the same eyes), interleaved;
`mountain_slope` hold at 1920x1080:

| | quadtree | octree |
|---|---|---|
| Descent gate mean / worst | 0.03 / 0.13 % | 0.03 / 0.08 % |
| Cruise gate mean / worst | 0.99 / 3.77 % | 1.02 / 4.02 % |
| Glide gate | 0 | 0 |
| Cruise frame interval p50 | 19.0 ms | 21.3 ms |
| Cruise generation | 89 k tiles, 3.13 s GPU | 128 k chunks, 3.31 s GPU |
| Hold primary | 2.85 ms | 2.84 ms |
| Hold sun and sky | 1.58 ms | 1.72 ms |

Visible refinement is at parity in all three flights; the structure is now
volumetric (caves, shafts and halls refine where the eye is; no gap or band
limits; eviction by pressure). It costs 0.14 ms of sun and sky (the
lookup's probes) and about 0.8 ms of update CPU (classification per node).
Rays saved little in these flights: open terrain is visible, and refined
nodes stay refined. Per chunk, a footprint's columns are recomputed for
each batch that wants one of its nodes (16-20 us of a chunk's 26).

### The footprint pool (2026-10-15)

Fixed-step flights against the octree before it (interleaved):

| | before | pool |
|---|---|---|
| Cruise: us a chunk (footprint columns) | 27.1 (18.1) | 21.8 (13.1) |
| Descent | 30.0 (21.7) | 21.9 (14.3) |
| Glide | 26.6 (19.8) | 18.7 (12.5) |
| Cruise gate mean | 1.06 % | 0.98 % |

About a fifth of the footprints a flight wants come from the pool: most are
new ground. The rest of a footprint's cost is its first computation.

An intermittent corrupt capture (a sample in a few fixed-step descents
whose 8x8 and 4x8 pixel blocks, scattered over about a sixth of the frame,
show the background's haze) is not the octree's: main's chunk trees show
it too. The voxel publish runs for every pixel and finds a hit for every
pixel, yet after the frame those blocks of every G-buffer target and of
depth still hold the G-buffer pass's earlier clear, so lighting takes them
for sky. Only NVIDIA's Vulkan driver shows it (616.64, RTX 3060): D3D12
never does, Khronos synchronization validation finds no hazard, and no
barrier, fence, single command buffer or recording-cache setting changes
it. It needs the frame to be recorded while the GPU still runs the frame
before, with generation in it; waiting for the GPU before the voxel pass
records removes it (at +44% frame time). The gate's outliers above 5% are
that. See [Known driver issue](#known-driver-issue) below.

### Known driver issue

Lost G-buffer writes on NVIDIA Vulkan, 2026-10-16. Measured with captures
every 0.25 s (65 a descent; a descent shows it about one time in two):

| Change | Descents with lost blocks |
|---|---|
| None (Vulkan) | about 55 % |
| D3D12 | 0 of 6 |
| GPU idle before each frame is recorded | 0 of 6 |
| GPU idle before the voxel pass records | 0 of 8 (frame p50 18 to 26 ms) |
| GPU idle before submit (recorded while busy) | hit |
| Recording cache off; serial `finish`; one command buffer | hit |
| Frame fences (colour output; every stage) | hit |
| G-buffer cleared by a draw, or by the publish's own pass | hit (blocks hold that clear) |
| G-buffer not cleared, publish writes every pixel | 0 of 8 |
| Generation inputs in persistent buffers | hit |
| Tile ranges reset on the GPU instead of by queue writes | 2 of 7 |

A forum report with the same profile (same driver, Ampere, writes lost
intermittently, fixed only by waiting for the GPU, barriers no help,
validation clean): https://forums.developer.nvidia.com/t/382912.

Reported to NVIDIA on 2026-10-16 as bug **6939141**, with repro steps from
this repository and an affected frame emailed to NVSDKIssues. Next: try a
newer driver, a standalone repro if NVIDIA asks for one, and a cheap
mitigation.

### Per-voxel materials (2026-10-16)

A leaf (4³ cells) draws two materials: its surface's (a vote of the top
materials of its columns whose top cell is in it), shown by cells under
air, and its subsurface's (soil or rock by the depth of its centre). Where
edits set materials (add or paint, including baked cells), each cell now
keeps its own:

- A chunk job whose brushes or baked bricks set any material gets a
  `cell_materials` slot (32 a batch; a job past them waits for the next
  batch). `chunk_columns_main` writes each solid cell's edited material
  there (0: the terrain's).
- `chunk_build_main` gives every leaf page holding a leaf with such cells a
  **detail page** (`detail_pages[leaf page]`, 4 KiB: a byte a cell, 16 words
  a leaf by its place in the page), off a free stack (16384 pages by
  default, `HELIO_VOXEL_DETAIL_PAGES`); eviction returns them. Such leaves
  are never implicit (full under their mid's material). Out of pages, the
  chunk keeps its leaves' pairs (flag 16; `UpdateStats::out_of_details`).
- Drawing reads the cell's byte first (0: the leaf's pair), at the hit
  and in the top face's jittered material sample; `slot_details` skips the
  lookup in chunks without detail pages. `PickHit::material` reports what
  is drawn.

The format holds any cell's material, so natural leaves can use it later.

The render thread finds the brushes over a footprint through a
`BrushIndex` (a loose grid per face, each brush in the smallest level where
it spans at most two buckets a side) instead of scanning every brush per
chunk.

Measured:

- `edited_cells_draw_their_own_materials`: a wall of single-cell blocks in
  eight materials and a checkerboard of painted ground cells. All 16306
  pixels on edited cells draw the edit's material (5 detail pages). With
  the lookup disabled, 14610 of them do not.
- Natural ground against `Planet::material`
  (`natural_cells_against_the_leaf_rule`, ignored: a report). 906k level-0
  pixels at eight spots picked for relief and mixed top materials, from
  1.7 m and 6 m, and with a pit dug in front:
  - 40.5 % of pixels draw another material than the canonical one; flat
    grassland 0 %.
  - Most of that is not the leaf rule. The generation kernel evaluates
    `ground_material` at the column's domain point; `Planet::material` at
    the cell's volume point. The snowline's wander and the rock patches
    are noise of that point, and on a mountain 2 km up the two points'
    noise is unrelated: 38.8 % of pixels differ between the canonical
    material and the same rule at the column's point. Whole views flip
    (snow drawn as stone, dark stone as grass).
  - Drawn against the rule at the column's point (the leaf rule and the
    kernel's slope estimate together): 10.7 % of pixels, up to 22 % on
    steep slopes and in pit walls.
  - Fixed since: see [The material contract](#the-material-contract-2026-10-16).
- Cost on natural ground (no detail pages), interleaved against the
  octree before it:
  - `mountain_slope` hold: primary rays 2.861 ms both (four rounds each).
    Without the `slot_details` flag, the lookup at every hit cost +0.9 %
    (2.859 to 2.885 ms).
  - Cruise flight: chunk columns 5.6 to 5.5 us a chunk, chunk builds 2.8 to
    2.9, draw p50 3.79 to 3.80 ms; gate mean 1.00 to 1.03 %.

### The material contract (2026-10-16)

`Planet::material` is the canonical material of a base cell (picks, a dig's
yield, gameplay). The generation kernel now evaluates `ground_material` with
the same inputs for the cells it stands for:

- **Point.** Each evaluation (a column's top cell, its soil and rock
  samples) uses its cell's volume point, not the column's domain point.
- **Slope.** `Planet::material_slope`: level-4 heights two cells each way,
  block slopes interpolated in 32nds, one scale whatever level draws it.
  For each computed footprint of levels 0 to 3, three dispatches evaluate
  the level-4 heights its blocks read (10x10 at level 0, 38x38 at level
  3): a ring of shared nodes for the macro octaves (3x3 to 5x5), the
  heights 64 a workgroup, then the block slopes, kept with its columns in
  the footprint pool (4.6 KB a slot).
  A column of level L reads the base column at its centre. Coarser levels'
  cells span many base columns and keep the gradient over 64 base cells in
  their tile.
- **Surface word.** Its byte, as the CPU masks it.

Measured:

- `uniform_leaves_draw_canonical_materials` (a gate): the top faces of
  leaves whose 16 columns all top in them with one material draw
  `Planet::material` exactly. 31710 pixels at three varied spots, none
  wrong; 779 wrong with the kernel before.
- The natural-cells report again, the same 906k pixels: 9.35 % wrong
  (40.5 % before). By what decides the drawn material:
  - Top faces of uniform leaves: 0 of 190908.
  - Other cells under air (a leaf's surface vote over mixed columns, side
    faces): 8.8 % of 627729.
  - Cells under ground (pit walls: a leaf's one subsurface byte, by its
    centre's depth): 33.5 % of 87379. The commonest: dirt drawn as dark
    stone (soil a cell deep drawn as rock), gravel as stone.
  That is the leaf rule alone, what natural per-voxel leaves would recover.
- Cost, cruise flights interleaved against the commit before:
  - Generation 19.8 to 20.5 us a chunk before, 22.4 to 22.8 after; the
    footprint columns 11.5 to 11.7 before, 13.6 to 13.9 after (+2 us,
    +10 % of generation).
  - All of it is levels 1 to 3's lattices: with level 0's alone, 11.6.
  - One workgroup a footprint, its lattice in workgroup memory, cost
    15.8 us (a level-3 lattice is 23 heights a lane in series); spread
    over workgroups, 13.6.
  - `mountain_slope` hold: primary rays 2.866 to 2.863 ms (drawing is
    unchanged).

### Dug walls (2026-10-16)

A leaf's subsurface byte follows its centre's depth, so the cells of a
dug wall drew rock where they are soil a cell deep (a third of a pit's
wall pixels). Every chunk under edits now gets a `cell_materials` slot (128
a batch, 272 KiB each), whatever the edits set:

- `chunk_columns_main` writes each cell's edit material, then the column's
  natural top: over the chunk where the volume puts solid cells there
  (scanned as `terrain::generated_top` does), else its highest natural
  cell's.
- `chunk_build_main` (`leaf_cells`), leaf by leaf with its pair: each
  exposed solid cell (air on a side in the chunk's masks, or at the
  chunk's faces) without an edit material gets `Planet::material`'s
  natural one (`ground_material` at its volume point, depth from that
  top). A cell its pair draws as it is (the surface under air, the
  subsurface under ground) is cleared, except in a leaf that would be
  implicit (drawn with its mid's material). Leaves with cells left get
  detail pages as before.

`dug_walls_draw_canonical_materials` (a gate): pits 5 m across at three
varied spots; all 57050 pixels of their walls and floors draw
`Planet::material` (about a third wrong before). A pit took about 170
detail pages (0.7 MB). Before both fixes: 117 pixels where the volume
carved the column's top cell (the natural top over the chunk), 66 in full
leaves cleared against their own pair while drawn with their mid's.

The natural-cells report, the same 906k pixels: 3.56 % wrong (9.35 %
before). Under ground (the pits' walls): 0 of 87379. Top faces of uniform
leaves: 0 of 190908. Other cells under air: 5.1 % of 627729 (8.8 %: the
chunks a pit's edit covers now draw their surface cells exactly too).
What is left is unedited leaves' surface vote over mixed columns; the
commonest, dirt drawn as dark stone (3.1 % of pixels).

Only levels 0 and 1 (`DETAIL_LEVELS`) get these: at every level, ten digs
8 m across filled a test world's 2048 detail pages; at two, 368. Coarser
leaves keep their pairs (an edit's top materials vote in them). Out of
pages, a chunk keeps its pairs (`out_of_details`): 100 large digs around
the eye fill 2048, a game's world holds 16384.

### Edits against history (2026-10-16)

What one more edit costs a running world after `n` edits
(`edit_cost_against_history`, ignored: a report; ground view, 10 edits
each, CPU of `sync_edits` and of the frame that regenerates, GPU waits
included):

| History | small digs (0.4 m: baked) sync / frame | large digs (8 m: analytic) sync / frame |
|---|---|---|
| 0 | 0.24 / 3.3 ms | 0.28 / 3.9 ms |
| 100 | 0.24 / 3.1 ms | 0.29 / 2.8 ms |
| 1000 | 0.26 / 4.7 ms | 0.48 / 4.2 ms |
| 10000 | 0.28 / 3.5 ms | 3.16 / 13.1 ms |

Small brushes are sealed into the baked store and cost what they leave.
Brushes over `BAKE_RADIUS_HALF` (3.2 m) stay analytic forever: every edit
re-hashes, re-indexes and re-uploads them all (`sync_edits`), and every
chunk under them evaluates each in its column kernel. Explosions are
large brushes, so a long game slows every edit and every chunk.

**Edits on data**, in steps, each with its gate:

1. **Large brushes into data** (done, below). Gate: the table's 10000
   large digs at the cost of 100.
2. **Edited chunks kept** (done, below). Edited level-0 chunks are read back,
   compressed and kept on the CPU (and in the save), so evicting one costs
   a copy, not a regeneration from the history; parents re-derived by
   downsampling.
3. **Incremental sync** (done with step 1).

### Large brushes into data (2026-10-16)

Where the cost went at 10000 large digs: `sync_edits` 3.8 ms (hashing
both brush lists 1.2-2.2, a new 480 KB brush buffer 1.4, the list, index
and view rebuilt 1); the regenerated chunks 4.35 ms of GPU (700 us each,
coarse ones evaluating up to a thousand brushes a column).

- **Large brushes bake at their bake level** (`planet::bake_level`: the
  first level whose cells a brush spans at most 32 of, level 2 for a dig
  8 m across). Sealing applies one over the baked base cells (as before)
  and bakes it at that level (`EditStore::bake_at`); coarser levels follow
  by majority, and a coarse cell whose children are mostly unchanged keeps
  its own edit. Below its bake level a large brush stays analytic. Every
  cell still reads `recent(baked(large(terrain)))` in the brushes' order:
  at the bake level, cell for cell as the analytic brush read
  (`sealed_large_brushes_read_the_same_at_their_bake_level`). Snapshots
  are version 3; older ones bake their large brushes on load, under the
  cells baked after them (what analytic-first applied read).
- **Brushes are indexed by the levels they apply at** (`BrushIndex` parts:
  below their size's, and a large brush below its bake level), so a coarse
  footprint's query skips the baked ones.
- **Incremental sync.** The view keeps the large layer's faces and index
  while seals only append to it (the large log's prefix hash); an edit
  rebuilds the recent brushes' index alone, diffs the lists past their
  common start, and writes the brush buffer's tail in place; a seal
  extends the brick index by its columns.
- **A brick cache.** Baked bricks stay on the GPU across batches
  (`BrickCache`, 64 MB by default, least recently used reused): a
  regenerated chunk sends its table, not its bricks again (a seal's
  regenerated chunks sent 5.5 MB a frame, an edit's 0.5 MB); a brick of one
  edit is its table entry. Runs of new bricks go in one write each.

One more edit after `n` large digs 8 m across (ground view; the frame's
wall time is too noisy in this harness, which waits for the GPU every
frame and lets it downclock):

| History | sync | update CPU | generation GPU |
|---|---|---|---|
| 100 | 0.34 ms | 0.30 ms | 0.20 ms |
| 10000, before | 3.8 ms | 0.96 ms | 4.35 ms |
| 10000 | 0.51 ms | 0.48 ms | 0.37 ms |

What is left grows with the baked data under a heavily dug region (10000
digs 8 m across lower 720 m of ground by some 40 m), not with the history.

### Edited chunks kept (2026-10-16)

An edited chunk (generated under brushes or baked bricks) the plan evicts
is packed on the GPU before its pages go back (`pack_main`): its meta,
mids, mid leaves and materials, then each leaf page's leaves, materials
and normals, then its detail pages, 12 KB on average. The 64-tree is the
compression: a chunk's leaves are a few hundred of its 4096 columns' 64
cells. Packs come back with the frame's results (64 a frame at most, in
2 MB) and are kept on the CPU (`Capacity::kept_bytes`, 256 MB by default,
least recently used dropped). An edited chunk that comes out empty is kept
as such (no pack).

A kept node the plan wants again is restored, not generated
(`restore_main`, after the builds): new leaf and detail pages off the free
stacks, its words copied in, its result written as a build's. It needs no
footprint, no terrain program and no brushes. An empty one is reported
empty at once (the plan, a few frames behind its results, asked for dug
chunks it had just found empty, and each was generated again from the
field and every brush over it). An edit over a kept node (a brush that came
or went over it, a seal's baked columns under it) drops it, and a node
generated again drops what was kept of it.

`edited_chunks_come_back_by_copy` (a gate): a site under 2000 digs 8 m
across, settled, left for a far view (a world of 9000 chunk slots evicts
it), seen again:

| | edited chunks on return: restored / generated | generation GPU on return | pixels as before |
|---|---|---|---|
| keeping nothing | 0 / 534 | 308 ms | all |
| kept (2.4 MB) | 1041 / 0 | 58 ms | all |

Then a brick mound added over the restored chunks draws exactly (against
CPU ray casts), and left and seen again, the chunks it changed come back
with it (940 restored, exact).

### Kept chunks in the save (2026-10-16)

Before saving, `ChunkWorld::keep_resident` packs the resident edited
chunks too (as packs fit a frame; `keeping` counts those left), and
`kept_chunks` takes every kept chunk (sharing the packs: cheap on the
render thread). `KeptChunks::to_bytes` (on the saving thread) writes a
header (the fingerprint of the generation shaders, terrain constants and
world, so other generation code never loads them, and the history's
length and hash), then the chunks deflated; `KeptChunks::from_bytes`
reads them back and `ChunkWorld::load_kept` keeps them before the world's
first update, for exactly that history.

They are saved beside the edit snapshot, not in it: a host folds only its
oldest edits into a snapshot (Pulsar lists its newest 4096), and the
chunks are of every edit. Pulsar's `VoxelEditJournal` keeps them
(`VoxelKeptChunks`: the bytes and the journal length and hash they are
for), saved as `kept` with the level and dropped by any change of the
edits.

Hosts reach this through the pass: a `PlanetFrame`'s `kept` chunks go into
the world built for its planet (`ChunkPass::kept_loaded` says how many, or
why they were refused), and its `export` (`KeptExport`) asks for them: the
pass packs the resident edited chunks over the next frames (asking for
frames meanwhile) and answers with the planet shown and its kept chunks.

`a_loaded_world_restores_its_edited_chunks` (a gate): the dug site saved
(313 chunks, 316 KiB besides the snapshot's 260; 3.3 MiB undeflated) and
loaded into a new world: 1172 edited chunks restored and none generated
(313 without them), 90 ms of generation GPU against 234, every pixel as
the saved world drew it. Kept chunks of another history, or damaged, are
refused.

### Next

- Natural per-voxel materials where unedited leaves mix materials (5 % of
  their surface pixels).
- Occlusion-limited refinement where it pays (valleys, caves, cities).
- A footprint's first computation: progressive (parent-seeded) synthesis.
