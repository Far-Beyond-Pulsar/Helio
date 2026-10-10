# helio-pass-voxel-planet

Destructible voxel worlds for Helio: Earth-sized cube-sphere planets, finite
planes and effectively infinite planes, built from exact voxels of 0.1 m to
1 m, fully editable, and rendered by tracing every pixel through a GPU-driven
clipmap. Near geometry and gameplay use the authored voxel grid. Distant
columns without geometry edits retain fractional radial height and filtered
slope lighting so sub-pixel terrain keeps its relief without tracing every
tiny voxel.

This document maps the system for people who will work on it: what each part
does, how a frame flows, which invariants hold it together, why things are
the way they are, how to measure it and how to extend it. The Pulsar editor
integration is documented in Pulsar-Native's `docs/voxel-system.md`.

## Contents

1. [Goals and non-goals](#goals-and-non-goals)
2. [Crate map](#crate-map)
3. [Data model](#data-model)
4. [A frame](#a-frame)
5. [Invariants and decisions](#invariants-and-decisions)
6. [Measuring](#measuring)
7. [Tests](#tests)
8. [Extending](#extending)
9. [Known limits and open work](#known-limits-and-open-work)
10. [Measured dead ends](#measured-dead-ends)

## Goals and non-goals

- Crisp near voxels and stable distant relief: filter sub-pixel detail while
  preserving visible landforms, materials and edits, without visible LOD steps.
- One world, two consumers: the CPU (collision, ray casts, edits, gameplay
  queries) and the GPU (streaming, rendering) evaluate the same integer field
  and the same edits. Field evaluations agree to the bit; distant rendering
  filters their appearance without changing the authored world.
- Space to ground in seconds: a camera can fall from orbit to walking height
  at the editor's altitude-proportional speed while residency keeps up.
- Volumetric worlds: generated caves and overhangs, not only heightfields.
  Heightmaps are one input among others. A terrain program adds 3D terms
  around its surface (`terrain_extent`, `terrain_density`); the built-in
  generator carves a cave network (tunnels and caverns under a rock cover,
  open to the surface at entrances) and leans steep ground over into
  overhangs (a continuous deformation of the heightfield: nothing floats),
  and erosion octaves carve branching gullies down its slopes. Caves may
  reach any depth (see vertical windows below).
- Destruction at any scale, up to the entire planet: digs of any depth,
  sphere brushes hundreds of kilometres wide, a hollowed core. Tens of
  thousands of edits stay exact and cheap; every regenerated column replays
  its whole brush list.
- Budgets (RTX 3060, 1080p Quality, i.e. 1440x810 internal): terrain GPU
  p95 <= 5 ms, no CPU frame stalls, terrain GPU memory <= 1 GiB.
- Non-goals (for now): translucent water, meshes inside the voxel pass,
  multiple worlds per pass.

## Crate map

| Path | Role |
|---|---|
| `src/grid.rs` | The canonical grid: equal-angle cube sphere (or plane), cells, levels, faces, exact cell walking maths. |
| `src/noise.rs`, `shaders/noise.wgsl` | Bit-exact integer noise (Q16 and fine Q24). The only noise terrain may use. |
| `src/terrain.rs` | Pluggable generators: `TerrainGenerator` -> `TerrainField` (CPU) + `TerrainProgram` (WGSL). Registry, material ids, `check_field`. |
| `src/layers.rs` | The built-in generator `helio.terrain`: ordered layer stacks (`TerrainLayers`), presets (Earth, moon, flat), validation and compilation. |
| `src/landform.rs`, `shaders/landform.wgsl` | The stack interpreter (CPU and WGSL): octaves, layer composition, craters, erosion, caves, overhangs and material styles. |
| `src/edits.rs` | Brushes (sphere/cube, remove/add/paint), per-face integer resolution, the shared `EditLog` and its edit tree. |
| `src/journal.rs` | Binary append-only edit journal (save/replay with recipe fingerprint and checksums). |
| `src/planet.rs` | `PlanetRecipe` + `Planet`: canonical queries (`kind`, `material`, `solid`), exact ray casts, `surface_point`, `air_clearance`, `ground_height`. |
| `src/windows.rs` | Level windows: which columns each clipmap level wants; runs on a worker and emits add/remove diffs. |
| `src/residency.rs` | CPU side of the clipmap: applies window diffs under a CPU budget, admits columns (records, edit lists, summary blocks), syncs edits. |
| `src/column_index.rs` | Resident columns: the CPU mirror of the GPU hash table plus per-record data. No rehashing, backward-shift deletion. |
| `src/engine.rs` | The Helio pass: `PlanetPass` (graph pass), `PlanetRenderer` (GPU buffers, pipelines, per-frame work), settings, stats, timing. |
| `shaders/world.wgsl` | Grid mapping, layer quantization, material ids shared by the engine and terrain programs. |
| `shaders/common.wgsl` | GPU residency structures: column records, hash lookup, summary blocks. |
| `shaders/generate.wgsl` | GPU column generation, brick-run allocation and publication (evict -> generate -> count -> refill -> allocate -> fixup -> publish). |
| `shaders/horizon.wgsl` | Directional sky bound: per azimuth sector and distance bucket, the elevation that clears all terrain. |
| `shaders/trace.wgsl` | Hierarchical grid traversal with authored radial tops for unedited distant columns. |
| `shaders/surface.wgsl` | Primary rays, shading (materials, filtered appearance, AO), traced sunlight. |
| `shaders/gbuffer.wgsl`, `shaders/view.wgsl` | GBuffer publication (depth-tested against meshes) and shared view helpers. |
| `tests/gpu.rs` | GPU correctness tests (see [Tests](#tests)). |
| `../../../helio-default-graphs/examples/voxel_flight.rs` | The flight harness: gates, editor trips, probes, audits. |

## Data model

### Grid

A planet is a cube sphere with an **equal-angle** mapping: on each of the six
faces, cell boundaries are planes through the planet centre at equal angle
steps, and radial layers are concentric spheres. Every cell is therefore
aligned with gravity: flat ground is flat everywhere, and a straight ray
crosses each boundary family in closed form (planes: a linear equation in the
face's angle frame; spheres: a stable quadratic), which is what lets the GPU
walk the exact grid.

- **Cells** are addressed `(face, i, j, k)`: `i, j` horizontal indices on the
  face, `k` the radial layer (height above the datum in layers; negative below
  it). Layer thickness equals the voxel size, quantized to whole millimetres.
- **Voxel size** is authored per world (0.1 m to 1 m). The *shape* of the
  terrain does not depend on it: generators sample a fixed **reference
  lattice** of 0.1 m (`REFERENCE_VOXEL`), and a world of 0.3 m voxels samples
  that lattice at its own cell centres.
- **Levels**: level `L` cells are `2^L` base cells wide and tall. A **column**
  is an 8x8 footprint of cells at one level (`BRICK = 8`), the unit of
  residency, with its vertical content as **spans** over a window around the
  eye (see [Span columns](#span-columns)).
- **Keys**: a column key packs `(face, level, ci)` into `key0` (24 bits of
  column index, 3 of face, 5 of level) and `cj` into `key1`. Column indices are
  never negative; decode them unsigned (2^24 columns cover a 0.1 m Earth face).
- **Planes** use the +Y face basis with axis-aligned cells and horizontal
  layers; an infinite plane is 2^27 reference cells (~13 400 km) across.

### Terrain field

A generator is a pure function of a column: `height(p, level)` in integer
millimetres and `ground_material(p, top_height, depth, slope, layer)`.
It exists twice, in Rust (`TerrainField`) and WGSL (`TerrainProgram`), and the
two must return equal integers for every input. That is the contract that lets
the CPU raycast what the GPU draws.

- All arithmetic is wrapping integer maths on the shared noise library.
  `noise_fine` / `mul_fine` (Q24 with 12-bit limbs) exist because Q16 noise at
  continent wavelengths is constant over metres and steps by one unit; scaled
  by kilometres of relief that became long straight terraces.
- The noise domain is a sphere at the planet's radius (planes: the
  horizontal position), in 1.25 cm units: domain distance is physical
  distance, the field is smooth across cube edges, and tangent directions
  (slopes, gullies, later flow) are defined everywhere. Integer
  `sphere_point` straightens the equal-angle cube coordinates with a tan
  polynomial and normalizes them (Q30 Newton reciprocal square root, exact
  64-bit products); the eighth-cell unit keeps its rounding far below a
  layer of height on steep slopes.
- Every world is an ordered layer stack (`layers.rs`): Warp, Continents,
  Mountains, Hills, Roughness, Erosion, Craters, Basins, Plateau, each with
  a mask (everywhere, land, above deep sea), plus caves, overhangs and a
  material style (Earthlike, Lunar, Layered). Planet, moon or plane is a
  game decision: the stack is data (presets `earth`, `moon`, `flat`, or one
  a game builds from a seed), compiled to one octave table sorted coarse to
  fine with each octave tagged with its layer. One interpreter runs it on
  CPU and GPU, so changing layers never recompiles shaders. Octaves
  accumulate per layer; the layers then compose in stack order (Basins
  flatten what precedes them). Bounds (`bound_margins`, `height_range`)
  derive from the layers, so any stack keeps `check_field`. A layer added
  or switched to another kind starts from that kind's defaults
  (`Layer::new`; the presets are made of them). The octave table holds 64
  (6 for the warp); cost follows the octaves used.
- Earth: continents, mountain ranges with erosion, hills from 140 m over
  9 km down to ~30 m knolls over a kilometre (five octaves at persistence
  0.6: the land near the eye has shape) and metre-scale roughness. Low basins get mud patches of a few metres (single-cell mud
  and sand specks read as noise that hid the ground's shape). The snowline
  wanders a sixth of its height over ~1.6 km above a rock band a sixth as
  tall, so meadows climb and no ruler-straight snow edge runs along a range.
- Craters: one candidate per cell of a 3D lattice per size, kept with the
  layer's density when within half a cell of the surface, so they are
  seamless across cube edges; parabolic bowl, smooth-min rim, ejecta to twice
  the radius. Large craters keep full-precision offsets (exact 64-bit
  squares): rounding them moved their steep walls by centimetres between
  columns.
- The stack carries analytic gradients (`noise_fine_grad`, chain rule through
  the domain warp) when an erosion octave is resolved. Each erosion octave
  lays stripes across the downhill direction of the coarser terrain on its
  own 3D lattice (random phase per corner, trilinear fade); its gradient
  steers the finer octaves, so gullies branch. Only strictly coarser octaves
  steer it, so every level that resolves it computes it alike. The phase
  turns up to 2 sqrt(3) STRIPES times across a cell, so the steering field
  is continuous (crest sign flips and clamp edges softened) and kept in
  Q30 unit vectors: a 1e-5 direction error is centimetres of height.
- Detail finer than a level's footprint is omitted at that level: coarse
  levels are band-limited point samples of the same field. Display
  generation retains the conditional mean of the display layer's (the first
  Mountains layer's) unresolved ridges, rather than dropping their mountain
  height. This lookup is baked once per stack;
  canonical field queries and level 0 remain unchanged. Levels 1 and above
  retain fractional radial tops unless Add/Remove edits change their geometry.
  Short low-level spans store exact base-layer tops in the existing byte header.
  Paint retains that relief. Slope lighting and the material height come from
  the stored exact surface (`ground_field`), never from tracing finer cells
  or running the generator per pixel.
- Heights are relative to the datum (the planet radius or the plane's y = 0)
  and may be negative: lowland and ocean basins sit below it. Nothing in the
  pipeline may clamp heights to the datum (see the tops invariant below).

### Edits

A `Brush` (centre in planet metres, radius, sphere or cube, remove / add /
paint, material) is resolved per cube face into integer half-cell coordinates
(`FaceBrush`), so CPU and GPU apply the same integer containment test. Later
brushes override earlier ones. At level `L`, a brush smaller than half a level
cell is omitted (smaller than the point sample).

`EditLog` is ordered and shared between world copies: brushes live in
`Arc` chunks of 1024 and every brush carries a prefix hash of the log up to
it, so the renderer finds the common prefix of its synced log and the
current one by binary search on prefix hashes (O(1) when unchanged).

**Edit tree.** The log's spatial index is, per face, an adaptive octree over
base cells (`EditTree`) whose leaves keep, in order, only the brushes that
can still change a cell inside them (the hierarchical edit culling of
Dreams, with HashDAG's full nodes):

- A Remove or Add holding a node's whole box replaces the brushes before it
  there (but larger ones, which still apply at the levels too coarse for
  it), and collapses the subtree under it.
- A Remove or Paint inside a box an earlier, at least as large Remove
  emptied, with no Add since, is left out: the dug surfaces under the air
  of overlapping strokes leave the lists.
- A leaf splits past 16 brushes, down to 16 cells or a 64th of its smallest
  brush's radius (surfaces dragged over each other cannot split nodes
  along a whole sphere).

Containment is proven, not sampled: boxes in half cells, balls in an f64
copy of the volume map (`Grid::volume_map`, within 16 units of the integer
points) whose corners hold a box's cells in their hull, give or take the
map's bound on bending. A query returns the leaves a box overlaps: a column
or a cell gets the surfaces exposed there, however long the history. Four
passes of a 2 km dig (400 brushes) leave about 20 brushes per column where
the old tile index gave every column under it every brush over it (about
110), filling the GPU's edit words until admission stopped. Columns with the
same list share one edit block, and a job's predicted cost counts its
brushes (`UNITS_PER_BRUSH`). Nodes are shared copy-on-write: copying a log
copies roots. Inserting a 2 km ball into that dig costs about 2 ms.

Edit cost does not grow with the brushes piled on one spot (sculpting):

- Generation culls a column's list per brick: the workgroup loads it in
  chunks of 64, keeps the brushes whose box reaches the brick (in list order,
  by ballot and rank) and applies only those to the brick's cells.
- Shading never replays the list for occupancy (the bricks have it). A hit
  in a column with Add or Paint brushes (`INFO_EDIT_MATERIALS`) takes its
  material from the latest Add or Paint containing the cell, scanning from
  the end; a later Remove containing it would have left air.
- `Planet::surface_point` walks the column under the point down from its
  highest possibly solid layer (generated top, Add brush tops) with the
  column's brushes queried once, instead of a ray from the outer radius
  through the edit index cell by cell (426 ms at 2000 brushes before).

**Natural surface of edited columns.** Every column keeps its natural
per-cell tops (the generated ground's, whatever edits did there) in its
header, over a column base, in bytes or 16 bits when they spread further
(`INFO_TOPS_WIDE`): material depth counts from them at any depth, so the
floor of a dig 400 m deep is rock at every level. They used to be stored
relative to the column's band and could not reach a deep floor: near the eye
it read depth 0 and turned to grass, while coarse levels showed rock.
Edited columns keep no relief, so their surface offsets count level cells
(`column_surface_offset`). In edited and generated columns the natural
ground is the top cell, the risers of steps down to neighbours (air side
above the neighbour's top) and ledge lips within two cells of the top; cave
walls, ceilings and dug faces are not.

**Picks.** A tool asks the pass for the terrain hit under a view point
(`PlanetFrame::picks`): the pass copies that pixel's primary hit to a small
readback ring, and the answer (distance along the pixel's ray and the size of
the cell that drew it) arrives a few frames later. The editor's brush walks
the exact base grid only a few cells around it: a CPU walk from the eye
through 0.1 m cells took seconds to reach a mountain 20 km away and tens of
seconds near the horizon. Ray walks look a column's top and brushes up once
per column.

Sculpting stress (`HELIO_VOXEL_FLIGHT_SCULPT=1`, three stamps a frame on one
ring): brush CPU per frame 397 / 590 / 1704 ms -> 0.4 / 1.8 / 4.8 ms (dig r1,
dig r4, build r1), terrain GPU 49 / 76 / 134 ms at 720p -> 13 / 16 / 21 ms at
1440p.

### Span columns

A column's vertical content is an ordered list of spans
([`docs/span-columns.md`](docs/span-columns.md)): everything below the first
is solid, everything from the column's `top` up is air. Each span is

| Kind | Meaning | Payload |
|---|---|---|
| `AIR` | every cell empty (dug out, a cave's hall) | none |
| `SOLID` | every cell full | none |
| `LANES` | each of the 64 lanes uniform over the span (a crater's wall) | 2 words |
| `TOPS` | each lane solid below its own top (a dug floor) | 16 words |
| `NATURAL` | each lane solid below its natural top (open ground) | none |
| `BRICKS` | arbitrary occupancy (caves, sculpting) | mixed/solid bits per brick, one unit per mixed brick |

A column that is only `NATURAL` (`INFO_HEIGHTFIELD`) stores no span table:
open terrain costs what it did as a heightfield column.

Generation evaluates cells only in **candidate intervals** where some
lane's occupancy can change: around each lane's terrain top (and the
volume the terrain program evaluates), where each brush's surface crosses
each lane, and at each baked brick. Between them no lane changes state, so
one exact evaluation per lane decides each gap (`AIR`, `SOLID` or `LANES`).
Cost follows the surfaces in a column, not its height: the interior of a dig
2 km wide evaluates its floor, not 4000 cells of air above it, and a wall
however tall is two words. A column has at most 7 intervals after merging
(the closest are merged), so at most 15 spans.

Rays cross an `AIR` span as one box (the whole footprint, however tall) and
a lane span (`LANES`, `TOPS`, `NATURAL`) lane by lane: the next event is
leaving the lane, leaving the span or descending onto the lane's top, one
solve however many cells the lane spans. Only mixed bricks of `BRICKS` spans
run a cell DDA. `column_view::ColumnView` decodes records and pool words as
the shaders read them, for tests and diagnostics.

### Planet

`Planet` = recipe + field + edit log. It answers the canonical questions:
`kind(cell)`, `material(cell)`, `solid(cell)`, exact `raycast` (a cell walk
clipped to the world shell or plane), `surface_point(p, clearance)`,
`air_clearance(eye)` (conservative distance to any possible solid cell, used
for near planes) and `ground_height(eye)` (height above the ground directly
below, used for camera speed). These are what gameplay, physics and the
editor use; they never touch the GPU.

## A frame

The pass is a GBuffer-stage pass in Helio's deferred graph. The frontend
publishes a `PlanetFrame` (eye in f64 world metres, planet, sun) into a shared
mailbox; frames are camera-relative (the renderer's world origin is the eye),
so all GPU positions are small.

Helio's sky pass accepts a `PlanetarySky` with the f64 eye, planet radius
and sun direction. Its lookup follows the radial horizon, including from
orbit; an authored scene sky takes precedence. `ambient_radiance` supplies
dim diffuse light on the night side; set it to zero for solar-only lighting.

### CPU: `Residency::plan` (residency worker thread)

The residency lives on its own thread (`ResidencyWorker`). Each frame the
render thread takes the finished plan, uploads its work and submits the
next request (eye, level-0 distance, job budget, CPU budget, job failures
read back). The worker plans while the frame is encoded and executed. At
most one plan is in flight and each result is uploaded once, in order, so
the GPU sees exactly the sequence of `plan` calls. A result that is not
ready when the next frame starts is uploaded a frame later (`late_plans`).
With the result come the stats, the
summary block list and the `Coverage` that `fallback_distances` needs, as
they are once its work is on the GPU. Steps:

1. **Edit sync.** New or undone brushes since the last frame are found by
   prefix hash; new face brushes are uploaded; resident columns they touch
   are queued as urgent regenerations.
2. **Windows.** When the eye moved and the previous plan has been applied,
   a `WindowRequest` goes to the window worker thread, which computes each level's wanted disc of columns and
   returns add/remove diffs. Level 0 covers the level-0 distance (cells about
   a pixel wide at its edge), each coarser level twice the distance. A level
   is on only if terrain within its reach can be nearer than that distance:
   the worker bounds the terrain around the eye per level
   (`Planet::local_outer_radius`), so over a meadow 1 km below the fine
   levels are off instead of streaming columns under a tenth of a pixel.
3. **Diff application.** Diffs are queued and applied in order within the
   plan's CPU budget (60 % of the frame interval, at least 1.5 ms moving or 4 ms
   still, at most 12 ms): removes evict residents,
   a switched-off level clears its queue, adds become pending in a priority
   bucket. Diffs get at most 60 % of the budget while columns wait. A level
   with unapplied diffs is *catching up*: its `fallback_distances` entry is 0
   (no guaranteed coverage).
4. **Re-ranking.** Priorities are distances from the eye when the window
   was planned. Once the eye has moved 1/16 of a level's radius from where
   its queue was ranked, the queued columns are re-bucketed by distance from
   the current eye, up to 32k per plan, farthest-ranked first.
5. **Admission.** Pending columns are issued nearest-first (the coarsest level
   always first, for global coverage) until the GPU job budget (from the
   measured GPU cost per job) or the CPU budget runs out: allocate a record,
   build its edit-reference list, reference its summary blocks, insert it in
   the hash table. Everything issued by one plan is one GPU patch, with each
   table and summary block slot once, at its final value.

### GPU

1. **Upload** of the patch: table writes, jobs, evictions, edit lists, block
   inits.
2. **Generate** (`generate.wgsl`): per job, evaluate the terrain program for
   the 8x8 columns, find where any lane's occupancy can change, evaluate
   cells only there, describe the column as spans (see
   [Span columns](#span-columns)), allocate a run of the right size class,
   publish the record, and raise the column's summary-block tops and the
   level's top. Evicted runs return to free lists. Failed jobs (scratch or
   pool full) append their keys to a failure list that the CPU reads back
   and retries. Cells within the program's `terrain_extent` of the
   heightfield top are evaluated in 3D (the sign of `terrain_density` at the
   seamless `volume_point`) in two passes: the first finds each lane's runs
   of cells that differ from the heightfield, then those and one more on each
   side are evaluated. Under the band the surface may lean through, only
   caves change cells, and the first pass steps over the rock they cannot
   reach (`terrain_clearance`: a tunnel needs both of its noises within the
   tunnel width, a cavern its noise over its threshold, and no noise moves
   faster than `noise::NOISE_SLOPE` per lattice unit), instead of evaluating
   every cell down to the cave depth (1,200 at 0.1 m: a mountain's columns
   cost 100 heightfield columns and loaded in 17 s; now 3x faster, and
   exactly the cells the dense scan found). Only a column with such cells is generated volume, marked
   `INFO_GENERATED` (`INFO_TOPOLOGY` is only for edit cuts: a generated
   column keeps its natural surface, relief and materials); a column whose
   cells all keep the heightfield's kinds (most of a cave region's rock,
   ground too flat to lean) stays a heightfield column
   (`generated_volume_is_stored_only_where_cells_change`). The column and its
   lean lattice nodes share one `generation_column` call site (compilers
   inline every call). Densities are signed distances to the field height
   itself (mm, passed to `terrain_density`), not to the floor of the level's
   cell, so a coarse level folds the same surface the base level does. Lanes
   the overhangs fold take every cell and their relief from the density: the
   top cell's fraction is the zero crossing between the highest solid cell's
   centre and the air cell above, and a crossing in the upper half of that
   air cell makes it the solid partial top cell, as in a heightfield.
   Elsewhere a cell the volume leaves as the heightfield has it keeps the
   heightfield's kind and relief. Before, lanes whose surface the volume
   changed lost their relief: overhang regions (about a third of Earth's
   land) showed whole-cell ledges at every coarse level, drawn as grey and
   brown patches that became grass on approach. The natural tops of a
   generated column are its generated tops (first air above the highest
   generated solid cell), so material depth counts from the real surface:
   overhang lips are turf, cave walls, floors and ceilings are rock. Side
   faces measure from the air-side cell's top (a cave wall lies far below
   it, a natural riser does not).
3. **Horizon** (`horizon.wgsl`): the directional sky bound. Resident summary
   blocks are binned by azimuth sector and distance bucket around the eye;
   each bucket stores the lowest elevation that clears it.
4. **Primary** (`surface.wgsl` / `trace.wgsl`): each pixel ray starts at the
   nearest bucket it does not clear and ends after the farthest (a ray the
   bound ended that still descends into the terrain shell continues without
   it: a fail-safe that costs nothing in consistent frames); it chooses a
   level from distance (with a per-column stable dither band) and repeatedly
   exits the largest provably empty box: a summary block whose top it is
   above, a column above its top, an air brick; only mixed bricks run a cell
   DDA. A column that is not resident falls back to a coarser level.
5. **Shade**: material from the generator (or edit), macro normals and
   filtered appearance for cells about a pixel wide, AO from neighbour
   occupancy.
6. **Sunlight**: one traced shadow ray per 2x2 block, shared with the other
   pixels of the block when they lie on the representative's surface.
7. **GBuffer**: albedo/normal/ORM/velocity and depth, depth-tested against
   meshes; deferred lighting multiplies the sun by the traced visibility.

## Invariants and decisions

- **CPU/GPU bit identity.** Terrain programs, noise and edit containment are
  integer and mirrored; `engine::verify_field` and the GPU tests enforce it.
  Never introduce floats into a field.
- **Tops bound occupancy, tightly.** Column tops, summary-block tops and level
  tops must bound every solid cell and should be tight. The generator's column
  top once clamped to the datum (`max(top, 0)`): below sea level every column
  claimed ~30 m of air, rays stepped cell by cell through it (137 steps per
  ray instead of 6) and columns stored empty bricks. Generation publishes
  the exact top (`resident_columns_hold_the_canonical_cells` finds a solid
  cell just below every top).
- **Demand stays inside capacity; nothing stalls at a limit.** Resident
  columns grow with the pixel count: at 1440p a ground view holds ~1.7M
  columns and 80% of the pool, the editor viewport ~2.8M (the record cap was
  3M). Once records or pool units ran out, admission stopped and the view
  stayed coarse. Admission is CPU-bound (~2 us per column) while churn grows with resolution squared times speed: at 1440p flying 90 m/s
  near the ground the windows churn ~100k columns/s, and at 25 km altitude
  and 25 km/s about as many. Above 85% of records or pool, or with over 10%
  of the wanted columns outstanding while moving (pending or to be added by
  queued diffs; a still camera is loading, not churning), the
  level-0 distance shrinks in 10-25% steps (`lod_pressure` in the stats;
  churn falls with its square, cells get slightly wider on screen); below
  65% and 2% it recovers in 5% steps.
- **Residency plans run one frame ahead, off the render thread.**
  Admission on the render thread took 1.5-4 ms of every frame and still
  managed only ~1500-3000 columns per frame. `ResidencyWorker` keeps one
  plan in flight: frame N uploads the plan requested in frame N-1. Ordering
  invariants carry over unchanged because results are uploaded in request
  order, once each: records evicted by one plan are reused only by the next,
  and table patches are final values computed on the worker. Everything the
  render thread reads about residency (stats, `Coverage`, `blocks_exact`,
  live blocks, diagnostics' table copy) comes with the result, so it matches
  the GPU state, not the worker's newer state. The plan's job readback is
  reserved when it is requested. `freeze_residency` neither takes nor
  requests a plan. A still, idle view requests no plans, so `settled()`
  becomes true.
- **Pending priorities follow the eye.** A window's adds are ranked by
  distance from where it was planned. With admission lagging at speed, the
  eye flew over columns queued as far while columns it had left were issued
  first. A level whose queue was ranked 1/16 of its radius from the current
  eye is re-ranked, in bounded chunks.
- **Window plans are coalesced.** A new plan is requested only after the
  previous one has been applied; the worker diffs against the last window it
  sent, so one diff spans all motion since. A request per moved frame queued
  every intermediate window: climbing at 1440p the queue reached 3000+ level
  diffs, eviction and admission fell further behind every frame and the view
  stayed coarse (100% of terrain pixels enlarged, up to 465 px).
- **Every job outcome reaches the CPU.** Failed jobs are appended to a GPU
  list that is copied to a readback with the allocator counters; with no free
  readback the frame issues no jobs. Results used to be copied only when a
  readback slot was free and otherwise assumed successful, so a failed column
  stayed resident on the CPU, unpublished on the GPU, and was never retried.
- **Pool pages return to the free stack.** Size classes take whole pages; a
  per-page free-run count lets `allocator_recycle.wgsl` (run under pool
  pressure, at most every 60 frames) give wholly free pages back, so terrain
  that changes character no longer strands pages in classes it stopped using.
- **Exact traversal, conservative accelerations.** The sky bound, summary
  blocks and residency hints may only skip space proven empty; tests compare
  renders with and without each acceleration.
- **Hash table.** 8M slots (at 4M slots and 2.5M+ columns probe runs passed
  the GPU limit). A column is only inserted within the GPU's probe limit of
  its home slot (`ColumnIndex::can_insert`). Linear probing, `slot_hash(key0, key1)`, at most
  `MAX_PROBES = 64` probes on the GPU. The CPU uses the same table for its own
  lookups (`ColumnIndex`); deletion is backward shift (no tombstones), so
  probe runs never degrade and the table never needs a rehash. Every slot
  write goes into the frame's table patch.
- **Table patches carry final values.** The GPU applies a frame's patches in
  parallel, in no order, and backward shift rewrites a slot several times in
  one frame. Each slot is sent once with its final CPU value. Sending the
  raw write list let an earlier value win: an empty slot inside a probe run
  hid every column past it, and a coarse level drew over the fine terrain in
  patches (the editor at 8 m). `gpu_column_table_matches_cpu_while_moving`
  compares the GPU table with the CPU table every frame. Any other buffer
  patched in parallel (summary blocks too) needs the same rule.
- **The generation budget is measured, always.** Jobs per frame = a GPU time
  target (1.5 ms moving, 6 ms still) / the measured cost per column, from
  stage timestamps. The stage profiler exists whenever the device supports
  encoder timestamps; it is not a diagnostic toggle. When it was, the editor
  (profiling off) kept the conservative default and streamed 3x slower than
  every harness run (profiling on): enlarged blocks while descending into
  new terrain and level transitions visibly catching up. Timestamps arrive
  frames late and repeat until a newer sample completes, so each sample is
  used once with the job count of the frame it measured (dividing by the
  last frame's jobs overestimated the cost 2-7x). Measured: ~0.22 us per
  column on an RTX 3060, ~6500 jobs per moving frame. The residency
  worker's CPU budget is 60% of the frame interval (1.5-12 ms).
- **Pending queues are exact.** Each level's pending columns sit in
  priority buckets with a position index, so a window moving at speed
  removes columns in O(1). A lazy heap kept millions of stale entries and
  admission spent tens of seconds popping them after the camera stopped.
- **Level windows follow local terrain, not the highest peak.** Using the
  planet's peak kept every level on below ~5 km, so cruising at 1.2 km and
  590 m/s streamed ~180k columns/s of 10 cm terrain nobody could see and
  starved everything else. The local bound uses the field's margins capped at
  4 cells (sampled rises never exceed 2; the certified margins, 13-28 cells,
  stay in the GPU bounds where correctness depends on them). Being
  optimistic here only makes a coarser level draw that terrain.
- **No frame does unbounded CPU work.** Window diffs and admission are
  time-budgeted on the residency worker (a late plan costs a frame without
  uploads); nothing rehashes or reallocates in bulk there or on the render
  thread (a 1M-entry `HashMap` doubling cost 70 ms; a table rehash 70-90 ms).
  If you add per-column CPU work, keep it inside the budgeted loops.
- **Stable LOD dither.** The level-transition threshold is hashed per column,
  not per frame, so a moving camera sees each column change level once
  instead of flickering between two levels while TAA history is rejected.
- **A level switch is not a surface.** When a ray changes level, the new
  level may already be solid at the cursor although the ray has only crossed
  air at the old one. `trace.wgsl` keeps walking the current level there
  (`level_contains_solid`) instead of publishing an interior hit with the
  last, unrelated face normal. Without that check, moving cameras saw grey
  patches sweep across the terrain in waves along transition rings.
- **Camera-relative frames.** All GPU positions are relative to the eye.
  Anything defined in world space (overlays, billboards, SceneDB rows) must
  be rebased by `PrepareContext::world_origin`. Rays toward the far plane must
  be built as `far.xyz - far.w * eye` (homogeneous): with near 5 cm and far
  40 000 km the far plane is at f32 infinity and dividing by `w` gives NaN.
- **Filtered appearance** (after "Filtered appearance for voxels", HPG 2023):
  cells about a pixel wide are shaded with the ground's smooth normal and show
  surface material on risers, so distant terrain has no contour lines; cells
  several pixels wide keep crisp faces. The blend follows the pixel
  footprint, so level changes show no seam. Base voxel steps fade from 2
  pixels down by the size their faces project to (`step_filter_weight`: a
  riser seen from above is a fraction of a voxel tall on screen).
- **Smooth surface model.** Every column stores, per cell, the exact surface
  below voxel precision: a surface offset (one byte, 1/128 of a base cell,
  `column_surface_offset`) over its stored height (level-0 top or relief
  base-cell top), from the generator's exact height or, at level 0, a
  density surface's zero crossing; coarser density surfaces keep it in their
  Q16 relief fractions. Occupancy, relief and materials never read it. The
  smooth normal (`relief_field_gradient`) is its central differences one
  cell each way at the four cell centres around the pixel's base cell,
  interpolated bilinearly across cells and columns, at every level. The
  per-column stencils it replaced lit a voxel staircase: one secant per 8x8
  column at level 0 (0.8 m tiles under a low sun) and in-column differences
  of base-quantized heights at coarser levels, zero on treads metres long
  and spiking at risers (dark worms along contours and column borders).
  It also gives the material height (the exact height at the pixel), so
  the screen-space climate pass and its far-relief normal (a second,
  per-pixel-stencil normal with a known 0.58 rad defect at 0.3 m voxels)
  are gone. Cost: +0.34 ms shade at 1196x729 (`surface_offsets_reconstruct_the_field_height`,
  `stored_sphere_normals_match_authored_macro_slopes_and_ignore_reuse_hint`).
  Natural ground at any size is lit
  partly with its slope's normal (step softness, appearance `detail.w`,
  0.7 on Earth), casts no step shadows, keeps AO soft and keeps turf on its
  risers: a staircase standing for a slope reads as voxel texture instead
  of black contour lines and brown soil dashes on every step.
- **Voxel mosaic at every distance** (Lay of the Land look). Each drawn
  cell, a base voxel near the eye and an LOD cell beyond (1-2 pixels wide
  at every level), takes its own brightness and its own place on its
  material's patch ramp from a hash of its volume point (blades of
  different hue; weathered and fresh stone), down to about half a pixel
  (`mosaic_weight`); temporal reconstruction averages smaller cells. Keyed
  to the base voxel it faded out as voxels shrank below a pixel, before the
  coarser levels took over: a smooth band between voxel ground near the eye
  and stepped ground far away. The pattern changes with the level (LOD
  voxels are coarser); lighting stays the smooth ground's, so steps draw no
  contour lines. The natural top
  of generated cave and overhang columns filters too; cave walls and edit
  cuts stay crisp (`natural_surface_hit`).
- **Material slope at one scale.** Materials (rock, scree, snow, grass)
  classify a slope measured the same way whatever level draws the pixel:
  central differences of relief heights 3.2 m each way (two level-4 cells,
  one level-5 cell), interpolated between cell centres; levels 4 and finer
  read level-4 columns, level 5 its own, coarser levels their own cells
  (`material_slope` in surface.wgsl and planet.rs). Coarse levels used to
  mix in a two-cell local derivative and a screen-space gradient, steeper
  over roughness: hillsides turned to rock and scree far away. Each level used to
  measure across its own 8-cell block (0.7 m at level 0, 11 m at level 4):
  rock and snow changed as the camera approached, and the 0.1 m steps of
  level 0 flickered across the thresholds (a rock riser on every step of a
  snowfield). Forced-coarser views of one flank now keep their rock share
  within 18-24 % (it was 34 % at the finest levels, 19 % at the coarsest).
- **Overhang amplitude** grows from 0 at an overhang region's edge (less the
  two level cells a level cannot resolve). It used to jump from 0 to two
  cells there, a step seam along every region border.
- **Overhangs lean the heightfield.** A cell is solid where it lies below
  the field's height at a horizontally displaced point,
  `z < H(x + W(x, z))`: `W` (`terrain_lean_offset`) varies with height at
  the ledge spacing and eight times more slowly across the ground, so for
  each height the map `x -> x + W` stays invertible and the solid is a
  continuous deformation of the heightfield's (no floating rock, no tears).
  Steep ground bends over into ledges; flat ground is unchanged. The engine
  evaluates the field on a global lattice per level (`terrain_lean`: nodes
  every `spacing` cells within `reach` of the column; generation's 64 lanes
  evaluate the 8x8 nodes once per column) and passes the bilinear height at
  the displaced point plus the column's own detail off the lattice
  (`terrain::lean_height`); CPU and GPU agree to the bit
  (`overhang_view_matches_canonical_cpu_ray_casts`,
  `overhangs_lean_steep_ground_without_floating_rock`). The earlier overhangs
  added a 3D noise to the height and left floating pieces and tears.
- **Caves** are a network under a rock cover: tunnels narrow to nothing over
  two radii towards the cover and the cave depth, caverns raise their
  threshold as they close, and the cover opens only in entrance zones
  (`caves_open_to_the_surface_only_at_entrances`,
  `caves_leave_no_floating_rock`).
- **Vertical windows.** A column describes a window of 2048 of its level's
  cells. Generation first measures the range of the column's candidates:
  when they fit, the window covers them all wherever the eye is (open
  ground, a dig, a crater wall: nothing clipped, nothing regenerated as the
  eye climbs or descends); a taller column (a crater kilometres deep at a
  fine level) describes the cells around the eye's layer within that range.
  Spans past the window are not stored and the record is flagged clipped
  below and/or above (`INFO_CLIP_*`): rays beyond a clipped side continue at
  the next coarser level, whose window reaches twice as far (never refining
  into it), and summary and level tops keep the unclipped bound. Generation
  reports clipped columns with their window's bottom and clipped sides
  (`clipped_sides`); residency regenerates one when the eye comes within a
  quarter window of a side it clips (`follow_clipped`). Windows around the
  eye's layer for every column left the ground of columns under a high eye
  clipped below: descending from orbit, rays fell back level by level (50
  ms of primary rays) and every level regenerated each time the eye moved a
  quarter window. The coarsest level, the
  coverage every ray falls back to, is never clipped: its columns describe
  their whole radial line (a few dozen bricks), so a bite thousands of
  kilometres deep shows from orbit instead of loading. A column begins at
  the planet's centre, where its lane's volume points reach zero; below it
  the line runs out through the antipode, and a gap reaching there was
  classified by a cell on the other side of the planet (a planet removed
  entirely still drew rock).
- **Brushes.** Cubes are tested in each face's half-cell index space (a
  one-block cube is exactly one cell, aligned with the ground); spheres are
  balls in the seamless volume space (`Grid::volume_point`), round at any
  size and depth, the planet's centre included (a ball around the core
  resolves onto every face). Both use exact 64-bit squares, up to 2^29 half
  cells of radius. A face brush carries its horizontal culling extent and
  the half-cell heights it can touch. Generation finds where a brush's
  surface crosses each lane (a box: its faces; a ball: an interval of the
  lane, which is linear in volume space, solved in f32 with a margin for its
  error) and evaluates cells only there. Jobs that outgrow the generation
  scratch are retried with a smaller job budget (`scratch_retries`).

## Measuring

Everything is measured in the full engine graph with
`helio-default-graphs`' `voxel_flight` example:

```
cargo run -p helio-default-graphs --release --example voxel_flight -- OUT [W H [native|quality]]
```

The default run is the gate flight (ground spawn, walk/run/vehicle speeds,
orbit and back, altitude reversals, teleport, mountain flyover, digs and
builds, resize, other voxel sizes) and writes `frames.csv`, captures and a
report of gates. Frames are pipelined two deep like a game loop; GPU stage
times come from timestamps.

| Variable | Effect |
|---|---|
| `HELIO_VOXEL_FLIGHT_TRIP=<deg>` | Replays the editor trip at `deg` from the pole: settle, climb, orbit, descend, cruise at 30 m, at the editor's speed (10 m/s x height/20 m). |
| `HELIO_VOXEL_FLIGHT_TRIP_CLIMB=<s>` | Climb/descend seconds (14: ~9 km, 30: ~5000 km). |
| `HELIO_VOXEL_FLIGHT_TRIP_LOW=<m>`, `_TRIP_END=<s>` | Cruise height of the low phase (30 m); stop the trip at a time. |
| `HELIO_VOXEL_FLIGHT_PROBE=1` | Reads every frame's rays back and reports holes (misses that pass below the deepest terrain, loading, exhausted), capturing those frames. |
| `HELIO_VOXEL_FLIGHT_AUDIT_AT=t1,t2` | Traversal audit (steps, lookups, block skips, CPU/GPU agreement) at trip times; with `HELIO_VOXEL_FLIGHT_HEAT=1` also a step heatmap. |
| `HELIO_VOXEL_FLIGHT_SKIM=1` | An editor camera pressed against the ground with editor overlays on (camera-relative overlay regressions). |
| `HELIO_VOXEL_FLIGHT_EDITOR_PATH=<deg>` | Descent from 300 km then cruise, with residency logs. |
| `HELIO_VOXEL_FLIGHT_BLOCKY=1` | Logs the share of terrain pixels drawn by a coarser-than-base level with cells wider than 2 and 4 px (by design at most ~2.2 px; wider means a finer level is still loading). Cruise and replay always log it. |
| `HELIO_VOXEL_FLIGHT_TRIP_FROM=<km>`, `_TRIP_FPS=<n>`, `_TRIP_EVERY=<frames>` | Trip start on the flank of the nearest summit (rock, scree, snow); flight frames per second (120; the editor runs near 60); capture cadence. |
| `HELIO_VOXEL_FLIGHT_LODCMP=<km>` | One view over a flank rendered with levels forced progressively coarser: surface colour shares must not change (they stay within 3 %). |
| `HELIO_VOXEL_FLIGHT_LONG=<s>` | Sustained heavy travel (dives from 25 km to 30 m and climbs, turning, at 2-3x editor speed), then a stop: logs pool, table probe runs, queued diffs, `lod_pressure` and enlarged blocks. Run it at the editor's resolution (e.g. `2560 1440`); at 720p demand is 4x lower and capacity limits never show. |
| `HELIO_VOXEL_FLIGHT_CRUISE=<m>`, `_CRUISE_SECS=<s>` | Level flight at that height at the editor's speed for 20 s, then a stop: residency lag while moving and time to converge. |
| `HELIO_VOXEL_FLIGHT_REPLAY=<engine.log>`, `_REPLAY_FROM/_TO=<s of day>`, `_REPLAY_DEG` | Replays the altitude timeline of a Pulsar editor session logged with `PULSAR_VOXEL_STATS=1`. |
| `HELIO_VOXEL_FLIGHT_SUN=x,y,z` | Sun direction (the editor's default Sun is straight up). |
| `HELIO_VOXEL_FLIGHT_VIEWS=h:pitch,...` | Settles and captures views `h` metres above the ground site (`view_<h>_<pitch>.png`); with `_VIEWS_CAVES=1` above the nearest cave or overhang region, with `_VIEWS_OVERHANGS=1` above the nearest hillside of an overhang region. |
| `HELIO_VOXEL_FLIGHT_SCULPT=1` | Sculpting stress: dig r1, dig r4 and build r1 strokes stamped three (two) times a frame on one ring; logs brush CPU, frame time and generation cost per stroke. |
| `HELIO_VOXEL_FLIGHT_HEIGHTFIELD=1` | The Earth stack without caves and overhangs. |
| `HELIO_VOXEL_FLIGHT_SEED=<n>` | The Earth stack with another seed (7). |
| `HELIO_VOXEL_FLIGHT_PRESET=earth\|moon\|desert`, `_NO_CAVES`, `_NO_OVERHANGS`, `_GRAIN=scale_m,octaves,ratio` | The layer stack; without caves or overhangs; an extra roughness layer. |
| `HELIO_VOXEL_FLIGHT_VIEWS_AHEAD=<m>,<right m>`, `_LOD_PIXELS=<f>` | `VIEWS` from a site moved along the view heading; the renderer's `lod_pixels` (2: every level one step finer). `VIEWS` and `LODCMP` (with `_VIEWS_POLE`: the pole's hills at 60 and 300 m) skip the ground audits. |
| `HELIO_VOXEL_FLIGHT_VIEWS_POLE=<rad>`, `_VIEWS_MOUNTAIN=<km>` | `VIEWS` from the north pole (Pulsar's example spawn) along a bearing, or from the flank of the nearest summit facing it. |
| `HELIO_VOXEL_FLIGHT_LOOK=exposure,contrast,saturation` | A camera post-process with an outdoor look (ACES tone map and a grade), as the Pulsar example level has. |
| `HELIO_VOXEL_DEBUG=<n>` | Debug shading: 1 level colours (brighter where filtered), 2 the same lit from the vertical (only sun shadows stay dark), 3 the level the distance asks for, 4 column kinds (generated volume red, edit topology orange, relief green, plain blue), 5 faces and burial (red sides, blue undersides, green where material depth > 0). |
| `HELIO_VOXEL_FLIGHT_QUICK=1`, `_GROUND_ONLY=1`, `_CPU_PROBE=1` | Short timing probe, ground audits only, CPU per pass. |
| `HELIO_VOXEL_PLAN_TRACE=<ms>` | Logs residency plan phases of frames taking over `<ms>` (10 if not a number). |
| `HELIO_VOXEL_LOD_DITHER`, `HELIO_VOXEL_NO_HORIZON`, `HELIO_VOXEL_NO_FAILSAFE` | Override the dither width; disable the sky bound; disable its fail-safe (A/B timing). |
| `HELIO_VOXEL_COARSE_RELIEF=0`, `HELIO_VOXEL_RIDGE_DISPLAY=0` | Disable fractional radial tops or ridge-envelope display heights for A/B comparisons. Set before loading terrain. |

Measuring pitfalls: synchronous readbacks (audits, probes, captures) idle
the GPU and the driver drops its clock (frames right after them show 210 MHz
in `frames.csv`); compare interleaved A/B runs, not runs minutes apart; other
desktop applications share the GPU.

Latest editor trip (30 deg from the pole, 1080p Quality, RTX 3060, sun
overhead): terrain GPU p50 / p95 settle 4.0 / 5.2 ms (while loading), climb
3.2 / 4.2, orbit 2.8 / 3.6, descend 3.4 / 4.3, low flight 3.8 / 4.6; worst
residency CPU per moving frame 1.7 ms (5.7 on the ground while loading); frame
p95 13.5-14.5 ms. Cruise at 1175 m and 588 m/s keeps pending near 0 and has
nothing left to load when it stops (before: 480k pending, over 30 s).

## Tests

`cargo test --release -p helio-pass-voxel-planet` (CPU unit tests and
`tests/gpu.rs`), and `cargo test --release -p helio-default-graphs --test
voxel_pass_graph` (the pass inside the deferred graph, editor overlays).

- Field: `terrain_programs_are_bit_identical_to_cpu` (every generator and
  shape), `the_field_has_no_steps_between_neighbouring_columns`, bounds.
- Traversal: ground, plane and far-face-edge views match canonical CPU ray
  casts; orbital coverage; published tops bound occupancy.
- Accelerations: sky bound conservative (static, moving, dithered, planes);
  accelerations change nothing while streaming.
- Edits: edits reach GPU generation; thousands of block edits render exactly.
- Residency: windows bounded and complete; budgeted planning converges to the
  same residency and table as unbounded planning; `ColumnIndex` matches a map
  under churn with an exact GPU mirror; table lookups reach every resident;
  the GPU table equals the CPU table every frame while moving; the local
  terrain bound holds for sampled columns and is local over lowland.
- Graph: settles, resizes and drops the source in the deferred graph; editor
  overlays stay in world space in camera-relative frames; the planetary sky
  follows the eye and sun, has no wedges from orbit and keeps a dim shadowed
  hemisphere; appearance edits show in one frame without rebuilding residency.
- Appearance and relief (one file each): `grey_patch` (level transitions
  enter the surface, not subsoil; alpine features survive distance),
  `coarse_relief*`, `column_relief`, `raw_sphere_relief`, `far_relief`,
  `mixed_brick_range` (fractional tops, heightfield columns, paint and
  edits), `ridge_envelope` (display generation keeps ridge mass; canonical
  queries stay exact), `coherent_relief`, `material_filter`, `face_local_uv`,
  `shadow_receiver` (filtered materials, slopes, soil lips, grazing faces and
  shadow origins). Windows want complete 4x4 blocks
  (`windows::tests`). Two far-relief checks are `#[ignore]`d as known defects
  present before this layout too: one L12 pixel's relief normal at 0.3 m, and
  one pixel's material id under raw sphere relief.

## Extending

**A terrain generator.** Implement `TerrainGenerator` (id, version, info with
an optional settings-component name) returning a `TerrainField` and its
`TerrainProgram` (WGSL defining `TerrainConstants`, `terrain_height`,
`ground_material`, plus the constants' bytes). A program may also define
`terrain_surface(p, level, height) -> u32`: an 8-bit surface word per column
cell, computed once when the column is generated and stored with it (one
pool unit per column, only for programs that define it), passed to
`ground_material`. It carries what materials need besides height (Earthlike
style: the erosion term, so gully floors fill with gravel and rock shows on
the ribs; Lunar: fresh ejecta and basins); shading never runs the generator
per pixel. Volumetric generators also
define `terrain_extent` and `terrain_density` (and `TerrainField::extent`,
`density`, `volume_bounds`), and leaning ones `terrain_lean` and
`terrain_lean_offset` (`TerrainField::lean`, `lean_offset`; the engine passes
the leaning height to `terrain_density` as `lean_height`), and optionally
`terrain_clearance` (cells a lane's scan may step over below the lean band;
without it every cell of the extent is evaluated). A density is the signed distance from the cell
centre to the surface (256 per level cell, positive inside solid; the cell is
solid where it is positive): combine terms as constructive solid geometry on
distances (intersection: minimum, union: maximum), and make every cut a
term (the built-in caves' region edge, depth floor and cavern cover are),
so the field is a distance on both sides of every surface. Smooth surfaces
interpolate it between cell centres (`densities_are_signed_distances_to_the_surface`
holds the built-in field within 1.6 cells across every crossing). Keep the extent tight, since every cell in it is
evaluated per job and a column evaluates at most the 256 bricks of its
window. Return an empty
extent at levels that cannot show a feature (the stack resolves tunnels while
their radius spans a cell, covered caverns while a cell fits in the cover):
volumetric columns lose relief and filtered shading. Use only the integer noise
library. One version of each generator is registered; saved edits record
it, so bump it when a released generator changes its output (in-development
changes replace the output in place). Register it with `terrain::register`. Add a test calling
`engine::verify_field` for every shape and `terrain::check_field`. Changing
settings rebuilds the world without recompiling shaders; pipelines are keyed
by program. A new program compiles on a worker thread (`PlanetPass`; the
large pipelines in parallel, ~12 s in a row cold) while the previous terrain
stays on screen, so neither a host's start nor a terrain change freezes it.

**Materials.** Shading knows a material only through its
`MaterialAppearance` (16 per world): colour and roughness, optional
world-space patch colours (turf), a lip material on the sides of its surface
cells (soil under turf), the material and share of its single-voxel flecks,
and the host a filtered speck blends into. No material id is special to the
renderer. A terrain program may report display-only blends for the shaded
cell through the appearance channel (`common.wgsl`): a coverage between two
materials, a mix of four, and the material whose flecks are averaged.
The Earthlike style uses them for snow edges and stone bands; canonical ids
never change.

**Material rules.** A stack picks materials by style: Earthlike (meadows,
dry lands, outcrops, strata, snow, with display filtering), Lunar, Layered,
or Rules: an ordered list of up to 16 `MaterialRule`s, each a material and
conditions that must all hold (column height, slope, depth below the top,
moisture, the erosion surface word, noise patches with a size and share,
strata bands, single-cell specks); the first that holds wins, else `rock`.
Rules are uniform data (`PackedRule`), so a game's biomes change without
recompiling shaders (`TerrainLayers::desert` is built from rules only).

**Appearance.** `PlanetPass::set_appearance` updates the material table
and detail without rebuilding terrain. RGB is sRGB; roughness is linear.
When it returns `true`, reset temporal colour history to show the change in
an idle viewport.
Unedited Earthlike rock has a world-space weathered surface coating;
canonical material ids, underlying strata and explicit paint are unchanged.

**A brush shape.** Extend `BrushShape`, its per-face resolution (culling
extent, height bounds) in `edits.rs`, and the containment test in both
`FaceBrush::contains` and `brush_contains` (`common.wgsl`).

**GPU work per column.** Put it in `generate.wgsl`; keep CPU admission cheap
and inside the budgeted loop in `Residency::plan`.

## Known limits and open work

- The directional sky bound occasionally (a few frames per orbit trip, during
  very fast altitude changes) ends limb rays before their terrain; the
  primary-ray fail-safe hides it completely (`HELIO_VOXEL_FLIGHT_PROBE` finds
  no holes), but the table inconsistency behind it is not yet understood.
  Rings and fallback distances are identical in the bad frame and the one
  before; suspect block publication vs. the live-block list.
- The fallback sky (Helio's sky pass) assumes world +Y up and a fixed sun;
  away from the pole and from orbit the sky is wrong. Planned: atmosphere
  around the planet from engine sky systems.
- Dense construction stores edit references per column (4M-word pool);
  sparse block-override bricks would scale building further.
- Edit ids (`next_brush`) grow monotonically after undo.
- Distant grass colour variation reads as blotches from kilometres up.
- `tests/gpu.rs` once hung when run in parallel with other GPU work; it has
  not reproduced since.

## Measured dead ends

Kept out after measurement: a beam or depth prepass (no faster than tracing
once made rigorous), wavefront ray compaction, sunlight temporal reuse, hash
fingerprints, a sun-ray escape bound, and tilted plane-fitted summary-block
bounds (rays converge on slopes, so only 1 in 6-13 attempts succeeded).
