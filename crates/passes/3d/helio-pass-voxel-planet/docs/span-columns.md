# Span columns: a fully destructible planet, always visible

Status: implemented 2026-10-09 (phases 0-5), edit history 2026-10-10 (below); the single-band column is
gone. Measured in the editor harness: phase 6.

## Why

The goal is a planet that can be destroyed entirely, shown from orbit to the
ground at AAA quality and frame time. Measured in the Pulsar editor inside a
2 km dig (flamegraph, RTX 3060, 1617x926 internal):

| Stage | Inside the dig | On a mountain (reference) |
|---|---|---|
| Primary trace | 13.3 ms (max 56) | 8-11 ms |
| Generation | 7.7 ms (max 36), 12.2 us/unit | ~0.5 us/unit |
| Sunlight | 5.0 ms (max 18) | ~2 ms |
| Pending columns | 930 000 at 0.1 m | ~0 when still |

Plus: the floor of the dig turns to grass near the eye, and the camera crawls.

Every one of these comes from one assumption of the column format: a column
(an 8x8 footprint of one level) stores **one band** of bricks, everything
below it solid and everything above it air.

1. **Generation cost grows with the band, not the surface.** A dig makes the
   band span from its floor to the original surface: hundreds of bricks,
   every cell of every one evaluated (field, then every brush). Carved air is
   evaluated cell by cell.
2. **Rays only skip space above a column.** Inside a band, air bricks are
   crossed one by one; carved volume costs like terrain.
3. **Per-cell tops count from the band.** Material depth reads the natural
   top as a byte above the band base (or below its top). A floor 4000 cells
   below the original surface has no representable top: depth reads 0, and
   rock becomes grass. Coarse levels fit it, so the error changes with
   distance.
4. **Level windows follow the generated terrain.** `local_outer_radius`
   bounds terrain from the field and Add brushes; Remove brushes are
   ignored. Inside a dig the eye is "underground", every fine level turns on
   with a full window, and kilometres of air are requested at 0.1 m.
5. **One band per column caps depth.** Deep features are clipped to a window
   of 256 bricks around the eye's layer; a column with a cave below a dug
   floor below the surface keeps only one of them.

## Design

The address and residency unit stays the 2D column `(face, level, ci, cj)`:
the hash table, summary blocks, horizon, relief, appearance and the
residency machinery keep working. What changes is the column's vertical
content: an ordered list of **spans** covering a vertical window, each span
one of

| Kind | Meaning | Storage |
|---|---|---|
| `AIR` | every cell empty | none |
| `SOLID` | every cell full | none |
| `LANES` | each of the 64 lanes uniform over the span (a vertical wall) | 2 words |
| `TOPS` | each lane solid below its own top, air above (ground, a dig floor) | 16 words (byte tops over the span base) |
| `BRICKS` | arbitrary occupancy | air/solid/mixed bits per brick + one unit per mixed brick |

A natural column is `SOLID, TOPS, AIR`: today's heightfield column. A dug
column is `SOLID, TOPS(floor), AIR(carved), ...`. A vertical cliff or the
wall of a crater is `LANES` however tall. Caves, overhangs and
arbitrary sculpting are `BRICKS`. The planet's core can be `AIR`.

### Generation: evaluate only where a lane can change

A workgroup still generates one column (64 lanes). It finds the **candidate
intervals** where any lane's occupancy can change, evaluates cells only
there, and classifies everything between from one representative cell per
lane:

- the terrain surface: each lane's field top, widened by the program's
  `terrain_extent` (caves, overhangs) and one cell;
- each brush active at the level: where its boundary crosses each lane. A
  ball is an interval along a lane (the lane's volume points are linear in
  the layer): solved in float around the lane's exact integer volume point,
  widened by a margin that covers f32 error, and only ever used to choose
  which cells to evaluate exactly; a box has its two faces;
- each baked brick.

Between candidate intervals no lane changes state, so each lane is uniform
there: one exact evaluation per lane decides `AIR`, `SOLID` or `LANES`.
Inside candidates, cells are evaluated exactly as today (per-brick brush
culling, latest edit wins), and a candidate whose lanes are monotone becomes
`TOPS` instead of bricks. Cost follows the surfaces in a column, not its
height: a 2 km dig's interior column evaluates its floor, not 4000 cells of
air.

### The natural surface is not a span

Material depth, relief, the exact surface and the smooth normal belong to
the natural (generated) surface, wherever the spans are. The header stores
each lane's natural top relative to a column base in its lane word (with
the exact surface over it; README "Lane words"). Depth = natural top -
cell, exactly as the CPU computes it (`Planet::material`). The dig floor is
rock at every level.

### Vertical windows: the clipmap is 3D

Each level describes its columns over a vertical window of 2048 cells.
Spans outside it are not stored; rays leaving a column's window continue at
the coarser level, whose window is twice as tall. *Revised 2026-10-10:* the
window covers a column's whole candidate range when it fits (nearly every
column), wherever the eye is; only taller columns take the cells around the
eye's layer within their range and are regenerated when the eye nears a
side they clip. Windows centred on the eye for every column made fast
vertical movement regenerate every level and left columns under a high eye
without their ground (`docs/aaa-architecture.md`).

### Level selection sees edits

`Planet::local_outer_radius` (the bound of solid terrain around the eye that
turns levels on) accounts for Remove brushes: a brush that covers a region
of columns lowers its bound to the brush's floor there. The eye in a dig
then has its altitude above the dig floor, and fine levels are on only when
that floor is within their reach. Baked edits stay conservative (they never
lower the bound).

### Traversal

A column visit finds the span holding the cursor's layer (the first span is
in the record; others in the pool, searched in order, at most 16).

- `AIR`: exit the box `column footprint x span` in one step. Carved volume
  is crossed like sky.
- `SOLID`: hit.
- `LANES`: a 2D cell walk across the footprint, vertical exits at the span's
  bounds.
- `TOPS`: today's heightfield and relief path.
- `BRICKS`: today's brick path.

Summary blocks, level tops and the horizon keep using the highest solid
layer, which generation already computes exactly.

## Phases

Each phase leaves the editor working and the GPU exactness tests passing.

0. **Merge Helio main** into the branch (pass command recording, cached bind
   groups) before the rework, and pin Pulsar to it.
1. **Edit-aware level windows** (CPU only). `local_outer_radius` with Remove
   brushes; tests: an eye in a dug pit turns fine levels off; an eye 2 m
   over the pit floor keeps level 0.
2. **Natural surface header.** Natural tops relative to a column base;
   materials, relief, offsets and normals read it. Removes
   `INFO_TOPS_DOWN` and the band-relative tops. Test: deep dig floor
   materials equal `Planet::material` at every level.
3. **Span columns.** Format, generation (candidate intervals, lane
   classification, `TOPS`/`LANES`/`BRICKS`), publication, traversal, picks,
   summary tops. Replaces the band, `MAX_BAND`, `INFO_HEIGHTFIELD` and the
   clip flags. A CPU mirror (`Planet::column_spans`) for tests. Tests: all
   of `tests/gpu.rs` (ground, caves, overhangs, deep shaft, planet-scale
   crater, hollow core, thousands of block edits) plus span-specific ones:
   a 2 km dig interior column stores no bricks; a crater wall is `LANES`.
4. **3D windows.** Per-level vertical windows around the eye; vertical
   following; residency regenerates columns the eye leaves vertically.
5. **Planet-scale destruction.** Brushes larger than the planet, a hollowed
   world, everything removed; coarse levels, sky bound and horizon show it
   from orbit. Tests: remove a hemisphere and render it from orbit; remove
   everything and render sky.
6. **Measure and tune** through the editor harness (`voxel_editor_harness`)
   at editor resolution: ground, orbit, inside a 2 km dig, a tunnel network,
   mass destruction. Budget (README): terrain GPU p95 <= 5 ms at 1080p
   Quality, no CPU frame stalls.
7. **Clean up and document**: README data model and invariants, Pulsar
   `docs/voxel-system.md`, delete what the spans replaced, PR descriptions.

## What stays

The cube-sphere grid and its exact walking, the bit-exact integer field and
generator contract, layer stacks, the edit log, store, journal and
snapshots, residency worker, window worker, hash table, summary blocks,
horizon, picks, shading and appearance, and every Pulsar tool.

## Edit history (after the first editor sessions)

Span columns made a column's *content* independent of how deep edits go,
but every column still received every brush whose footprint covered it. In
an editor session of ~420 strokes of 0.3-2 km digs near a cube edge,
columns under the dig listed hundreds of brushes: generation cost ~40x a
column's usual (the adaptive budget fell to 60 units a frame), the GPU's
edit words filled and admission stopped (80k columns pending for minutes),
and sealing large brushes over baked bricks replayed every cell (seconds on
the render thread). Production engines keep edit cost proportional to the
surface left exposed, never to the history (Dreams' hierarchical edit
culling with full/empty cells, HashDAG's full nodes, Space Engineers' edit
octree; No Man's Sky replays its list and caps it at ~15k edits).

Done:

- **Edit tree** (`edits.rs`, README "Edit tree"): per-face octree whose
  leaves keep only the brushes that can still change a cell in them;
  containment proven with an f64 copy of the volume map and its movement
  and bend bounds. ~20 brushes per column under 400 overlapping 2 km digs.
- **Shared edit blocks** and **brush-aware job cost**.
- **Sealing by bricks**: a large brush sealed over baked bricks fills or
  skips whole bricks it holds or misses.
- **Picks**: a stamp whose exact search misses lands at the renderer's hit.

Next, in order:

1. **3D level selection**: levels from the distance to the nearest
   non-air span in 3D, not the column's highest solid (caverns, a
   hollowed core), and ray feedback requesting refinement where rays end
   on a level coarser than their footprint (GigaVoxels, Voxy).
2. **Occupancy masks in bricks**: 4^3 64-bit masks and a 2^3 mask per
   brick, so close sculpted rock skips empty space inside mixed bricks
   instead of stepping cell by cell; a beam pre-pass over 8x8 tiles.
3. **Asynchronous world updates**: the editor applies a stroke to the next
   world snapshot on a worker and publishes it, never on the render thread.
