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
    steep slopes and in pit walls. That much natural per-voxel leaves
    would recover.
- Cost on natural ground (no detail pages), interleaved against the
  octree before it:
  - `mountain_slope` hold: primary rays 2.861 ms both (four rounds each).
    Without the `slot_details` flag, the lookup at every hit cost +0.9 %
    (2.859 to 2.885 ms).
  - Cruise flight: chunk columns 5.6 to 5.5 us a chunk, chunk builds 2.8 to
    2.9, draw p50 3.79 to 3.80 ms; gate mean 1.00 to 1.03 %.

### Next

- The material contract: the kernel and `Planet::material` evaluate
  `ground_material` at different points (and estimate slope differently),
  so a pick, a dig's yield and the drawn colour disagree on 39 % of
  mountain pixels. Choose one (the cell's volume point is the canonical)
  and make the other match it bit for bit, then measure the leaf rule
  alone.
- Natural per-voxel materials in detail pages where leaves mix materials.
- Occlusion-limited refinement where it pays (valleys, caves, cities).
- A footprint's first computation: progressive (parent-seeded) synthesis.
