# Voxel planet: architecture review against the AAA bar

Status: review 2026-10-10; phases below in progress.

## Goals

- **World-scale edits.** Brushes from 0.1 m to planet size, any number of
  them, at any depth; the whole planet destructible. Cost follows the
  surface edits leave, never their history.
- **Fast movement.** Orbit to ground and flight at any speed without holes,
  blur or stalls: streaming keeps up within a fixed per-frame budget.
- **Tiny voxels, invisible LOD.** 0.1 m voxels near the eye; level changes
  never visible (cells at most about a pixel where a level ends, filtered
  appearance, stable dithered transitions).
- **AAA quality.** Exact voxels, sun shadows, sky occlusion, materials and
  ambient occlusion that hold from 0.1 m to orbit.
- **Performance.** Terrain under ~6 ms GPU at 1440p output on an RTX 3060
  (TSR from ~0.75 scale), no CPU frame stall over a few ms.

## Measured (editor, RTX 3060, 1796x1235 at 0.75)

From the editor flamegraph of 2026-10-09 (inside a 400-brush dig):

| Stage | Open terrain | In sculpted digs / close rock |
|---|---|---|
| primary | 1.5 ms | 6-10 ms |
| shade | 3.5-4 ms | 3.5-4 ms |
| sunlight | 2-3 ms | 4-5 ms |
| skylight | 0.5-0.7 ms | 1.5-2 ms |
| TSR | 1.1 ms | 1.1 ms |
| residency / generate | 0 (still) | 7-18 ms spikes (edit lists) |

Gate flight (`voxel_flight`, 1920x1080, RTX 3060), 2026-10-10 baseline,
mean GPU ms per stage:

| Scenario | primary | shade | sunlight | terrain GPU |
|---|---|---|---|---|
| ground, warm | 3.2 | 4.3 | 1.4 | 10.3 |
| walk / run | 3.3-3.8 | 3.7-3.9 | 1.1-1.3 | 9.7-10.3 |
| vehicle (fast, low) | 7.9 | 3.4 | 1.1 | 14.8 |
| ascent to orbit | 1.4-5 | 4.4 | 0.3-1.5 | 8.8 |
| **descent from orbit** | **54** | 4.0 | **19** | **82** |
| **arrival (settled)** | **50** | 3.9 | **17** | **84** |
| altitude reversals | 7.8 | 4.9 | 8.6 | 23 |
| mountain walk | 12 | 2.9 | 1.6 | 23 |
| dig | 3.2 | 5.1 | 1.7 | 11.4 |

Gates failing: terrain GPU p95 <= 5 ms (60), warm sync p95 (84 ms),
movement sync p99 (86 ms), arrival settles <= 250 ms (1116 ms), near-field
exact cell agreement (4 of 294,608 cells).

After content windows (`f7103991`): descent primary 5.6 ms, sunlight 1.2
ms, terrain 14.6 ms; arrival settles in 5 ms; warm sync p95 15.4 ms (gate
passes). Still failing: terrain GPU p95 21.9 ms, movement sync p99 26.9 ms,
near-field cell agreement (4 cells).

**Descent and arrival.** At the same place and height, rays cost 3-4 ms
going up and 50 ms after coming down: the arrival audit has 93 column
lookups and 120 steps per ray (2-5 and 10-15 elsewhere). Column windows
were centred on the eye's layer when generated: under an eye far above,
a column's ground lay below its window (clipped below, no heightfield
fast path), rays fell back level by level, and every level regenerated each
time the eye moved a quarter window. Fixed by anchoring windows on the
column's content (below).

## Subsystems: what we have, what the best known systems do, decision

References: research notes of 2026-10-10 (Dreams, Claybook, HashDAG,
Space Engineers, OpenVDB, Teardown, Lumen, Aokana, ESVO, sparse 64-trees,
GigaVoxels, kajiya, FidelityFX).

### 1. World data: span columns in a 3D clipmap

*Have:* per level, 8x8 columns of ordered vertical spans (air, solid, lane
tops, natural, mixed 8^3 bricks) in a vertical window around the eye;
residency by level windows; a GPU hash table and summary blocks.
*Best known:* Distant Horizons stores columns as vertical runs (the same
idea); Aokana/64-trees use shallow sparse trees of bitmask nodes; Teardown
and Claybook keep mips of occupancy.
*Decision:* **keep** the span columns (open ground is a header, a dig a few
spans: far denser than any octree for a planet's surface), **replace the
inside of mixed bricks** with a two-level occupancy bitmask (a 2^3 mask of
4^3 sub-blocks and 64-bit masks), shared by primary, sun and sky rays.
**Windows follow content, not the eye** (done): generation measures a
column's candidate range first; when it fits 2048 cells (nearly always) the
window covers it wherever the eye is; only taller columns take the cells
around the eye within their range, and regenerate when the eye nears a side
they clip.

### 2. Generation

*Have:* GPU column jobs from the integer-exact field, caves and edit lists;
CPU mirror for exactness; adaptive unit budget.
*Best known:* produce once, cache, budget by time (GigaVoxels, Lumen);
everything a pixel needs per frame is produced with the data (Teardown's
palette, per-vertex AO of meshed voxel engines, Lumen's surface cache).
*Measured:* a mountain view loaded for ~1,000 frames (17 s) where the same
view without caves took 125: each lane evaluated three cave noises at
every cell down to the cave depth (1,200 cells at 0.1 m).
*Decision:* **keep** exact GPU generation. **Interval evaluation of the
volume** (done, the same idea as Keeter's interval arithmetic and Dreams'
culling): the scan steps over rock caves cannot reach by a tested slope
bound of the noise, and keeps changed cells as runs; exact (no mismatch in
cave and overhang audits); mountain load 994 -> 335 frames. Next: the same
for the overhang band. **Move shading inputs into generation** where the
shading measurements below justify it. Budget generation by measured GPU
time.

### 3. Edits

*Have (done 2026-10-10):* edit tree with proven containment (Dreams-style
culling with HashDAG-style full nodes), baked bricks for small brushes,
shared edit blocks, brush-aware job cost, async world builds in Pulsar,
picks from the renderer's hit.
*Decision:* done; next only a brick-level containment cut in generation if
profiles show list scans.

### 4. Primary visibility

*Have:* a full-length ray per pixel from the eye, cut by the directional sky
bound; spans skip air; mixed bricks stepped cell by cell.
*Best known:* a conservative coarse pre-pass over 8x8 tiles starting full
rays at the tile's nearest surface (Claybook 0.2 ms, ESVO beams +27%),
bitmask traversal (2x on deep trees), visibility buffer then shading
(Aokana).
*Decision:* **add a tile pre-pass** (coarse levels and air spans only,
min t per 8x8 tile) and **bitmask bricks** (1). Hits already form a
visibility buffer.
*Measured (2026-10-10, QUICK `mountain` mode, loading and settled frames
timed apart):* the mountain views' 12-14 ms were mostly frames still
loading. Settled: mountain_air 6.0 ms (15 steps a ray), mountain_slope 11.1
ms (31 steps, 38 for a tile's slowest ray), ground 3.2 ms (11 steps).
Loading: mountain_air 12.1 ms (44 steps, 8.4 column lookups a ray; sky rays
take 30-40 steps while the view loads, none once settled: open).
Every step costs about 700 lane-cycles even in a coherent straight-down
view. The driver (`examples/shader_stats.rs`) reports `primary` at 64
registers, no spills and a 74 KB binary (sunlight 154 KB): the loop
inlines three column lookups, two occupancy tests and four exit
computations. One probe for both transition directions and one box exit
for every path (summary block, column above its top, air span or brick,
lane, relief cut cell) made it 62 KB (sunlight 129 KB) but 3-14% slower (a
16-byte spill, a layer solve every step): code size is not the cost
(reverted). Not the cost either: Morton-ordered summary slots, folding the
relief path into the lane path (no change each), ridge display. Cold
pipeline compiles: generate 20 s, shade 4.5 s, sunlight and skylight 1.5 s
each (a new terrain program in the editor waits on them).
Upper bound of a tile pre-pass: starting every ray at its tile's nearest
hit of the previous frame saves 25-31% (slope 11.6 -> 8.0 ms, ground 3.6
-> 2.7). Finer-level probes inside the dither band (a ray may step back to
a finer level per column) cost about 8%.
*Loading (2026-10-10):* a mountain view reached by teleport waits 15-19
frames for its first plan, then generates 30-200 thousand columns at 4-5
thousand a frame (1.2-1.8 us a unit, the 10 ms budget); 860-930 thousand
stay resident. That count is the pool's capacity, not the LOD density:
`lod_pixels` 1, 1.5 and 2 all settle there because pool pressure divides
the level-0 range by 1.3, 2.0 and 2.8 (cells are 1-2 pixels across a
level's range at 1; larger is finer). While loading, a ray hops to a
coarser level about 10 times (missing columns) and looks up 8 columns
instead of 3; summary skips need complete blocks, so the sky bound saves
2 of 42 steps instead of 17 of 32 once settled. Walking and flying load
only at the rings' edges (ground_load primary 3.9 ms vs 3.3 settled).
A finer sky bound (1024 sectors, four buckets an octave instead of 256 and
one) made rays take more steps (ground 11.2 -> 14.8, straight down 6.0 ->
8.5) and cost 0.55 ms to build: rejected.
*Interleaved A/B (2026-10-10, two runs each, noise about 10%):* a tunnel
view (150 balls of 3-7 m from the surface, eye inside looking along it)
costs 14.6-16.3 ms primary at 44 steps a ray: the edit views' cost. Bitmask
bricks (the 64-tree layout inside mixed bricks: one 64-bit load per 4^3
sub-block, empty ones crossed in one step) changed nothing for primary and
made tunnel sunlight slower (3.5 vs 2.9 ms; each sub-block skip relocates
the cursor): rejected. Doubling every box exit's geometry (plane, shell
and relocation solves) adds about 20% (ground, mountains): geometry is a
fifth of a step, not its bulk. One more dependent summary load per column
entry adds about 8%.
*Nsight Graphics GPU Trace (2026-10-10, tunnel view, real-time shader
profiler):* primary issues at 56-68% of SM peak with the ALU (integer and
logic) pipe the busiest (43-52%), FMA 27-33%, transcendentals 12-14%; L2
and DRAM traffic about 2%; 25 of 32 lanes active; 20-24 warps a SM; a
third of the time the SM holds no active warp (the tails of long rays).
Its top stall reasons are "selected" and "not selected": the traversal is
bound by the instructions it issues, not by memory. The per-line profile is
flat: the plane solve's polynomial 4.6%, the loop head 2.4%, the sky bound's
two `atan2` per pixel 3.6%, the column probe 3.5%, the span walk 1.4%; no
line above 3%. What remains is fewer steps (3D empty space under the
surface, the tile pre-pass) and the long-ray tail (compaction), not a hot
spot. Sunlight and skylight kept 11 and 14 of 32 lanes busy: the pixels no
representative serves traced inside each thread's pixel loop. They now go
to a workgroup queue traced one per thread: sunlight 10-15% faster in
every view (ground 1.41 -> 1.28 ms, tunnel 2.92 -> 2.52), skylight 12% on
the ground. Sky rays two levels coarser (Teardown's mips, Lumen) were
faster only in the tunnel (2.03 -> 1.67 ms) and 10-20% slower on open
terrain (probing coarser levels): rejected.
*Long rays and underground air (2026-10-10):* capping rays at 64 steps cut
primary in proportion to the steps it removed (tunnel 16.4 -> 11.6 ms with
21% of its rays capped; ground 3.45 -> 3.24 ms with 0.4%): long rays cost
their own steps, not idle neighbours, so compacting them would recover
little. In the tunnel a ray takes 10.5 air-span column boxes, 15.8 lane
steps and 19.3 cells inside mixed bricks. Air blocks (per tier-1 summary
block, a 64-bit mask of the 8-layer bricks that are air in all 16 columns,
rebuilt after each publication; `air_blocks_build`, `air_run`) let an eye
ray cross four columns of a tunnel's air a step: 21% fewer steps, exact
(the CPU comparison is unchanged), primary 7% faster in the tunnel. Each
such box costs the relocation solve a lane step avoids; sun and sky rays
(which leave a surface within a few cells) do not use them (skylight was
12% slower).
*Where the traversal stands:* every step-saving structure measured here
(summary tiers, the tile start's upper bound, bitmask bricks, coarser sky
rays, air blocks) trades fewer steps for dearer ones and nets 0-10%. The
step itself is the cost: about 700 lane-cycles of integer and logic work
(column lookup, summary tiers, span and lane decoding, level transitions,
relief) plus a fifth in cube-sphere geometry. The research tracers
(64-trees and Aokana, Teardown, ESVO) spend 30-60 instructions a step: a
64-bit node mask, a popcount and a linear integer DDA. *Decision:* keep span
columns as the storage and generation format, and give traversal its own:
per resident column a compact 64-tree of its occupancy, walked inside the
column by a linear DDA in the column's own index frame (one ray transform
per column; within a column's footprint the cube-sphere's curvature is far
below a cell, and the hit is checked against the exact geometry). Column
entry (lookup, summaries, transitions) stays as it is.
*Measured (hold mode, 1,600-3,600 settled frames a median, interleaved):*
a column walk that modelled the ray once per column (linear lane
crossings, exact shell crossings for floors and ceilings, a quadratic
height for the layer, air crossed to the column's side in one step) ran
the tunnel at 13.4-14.4 ms against 14.0-14.2 and the ground at 3.65 ms
against 3.39-3.46: no better. The model's setup at every column entry and
the vertical solves cost what the outer loop's prologue did. Reverted;
per-column 64-trees inherit the same entry cost and are not pursued.

### 5. Shading

*Have:* 4.3 ms per frame at 1080p at any view: per pixel the ground field
(12 heights of up to 4 columns), the material slope (16 heights of a coarser
level's columns), procedural material rules and 8 occupancy lookups for
corner AO.
*Measured (2026-10-10, ground view, 4.26 ms):* without the material rules
-1.6 ms, without the ground field -1.1, without the material slope -0.85,
without the AO lookups -0.35. Removing the neighbour-column hash lookups
alone changes nothing: the cost is decoding heights (top byte, relief
fraction, surface offset: three loads and branches each, ~28 per pixel).
Computing gradients and slopes at generation needs a ring of field
evaluations around each column: measured +55% generation cost (mountain
load 335 -> 486 frames), rejected.
*Best known:* read cached per-texel attributes (Unreal's runtime virtual
textures for landscapes, Lumen's surface cache 2.4 vs 11.5 ms evaluating
materials per hit).
*Decision:* **one 32-bit word per lane** for the surface (exact height,
fraction and surface word) replacing the tops, relief, surface and offset
units at about the same memory; shading and traversal read one load per
lane. Then cache view-independent material inputs per lane if the material
rules still dominate.
*Done (2026-10-10):* lane words (15-bit natural top, Q16 exact surface,
17 bits; surface words in their own byte unit, read once per pixel), five
header units for every column. Shade at QUICK 1080p: ground_warm 4.26 ->
3.62 ms, hover_330 4.38 -> 3.83, hover_55k 4.39 -> 3.36, orbit 4.25 ->
3.35, moon_orbit 3.09 -> 2.39, plane_ground 3.56 -> 2.98; primary unchanged
(traversal steps equal or lower). A 12-bit top with 1024ths of a cell was
tried first and failed two tests: coarse relief floored up to 2^(level-10)
layers under the surface (subsoil at level transitions) and a surface in a
level-17 cell's lowest 1024th made the whole cell solid (hits past the sky
bound). Material rules (1.6 ms) are next.

### 6. Sunlight

*Have:* one ray per 2x2 block at a rotating pixel, neighbours reuse it on
the same surface (kajiya-like); rays use the full traversal.
*Best known:* half resolution with rotation (have), tile classification so
only penumbra tiles trace (FidelityFX), coarse occupancy for long rays.
*Decision:* **keep**, gain from bitmask bricks (1); add tile classification
if still above ~1 ms.

### 7. Skylight

*Have:* 6 rays per 4x4 block, 48 cells of the hit's level.
*Best known:* super-sparse rays on coarse mips (Teardown), radiance caches at
1/16 resolution (Lumen).
*Decision:* **trace sky rays two levels coarser** (occlusion at terrain
scale; cells four times larger see the same valleys) and keep the block
reuse.

### 8. LOD selection and transitions

*Have:* pixel-footprint level selection with a stable per-column dither,
appearance filtering of sub-pixel voxels, LOD comparison gate.
*Decision:* **keep**; verify with captures that no transition shows.

### 9. Streaming

*Have:* distance windows per level planned on a worker; GPU jobs with an
adaptive unit budget; coarsest level always resident.
*Decision:* **keep**; cap generation by measured GPU time per frame (~1 ms
while moving), priority by screen error.

### 10. Upscaling

*Have:* TSR 1.1 ms. *Decision:* keep; DLSS through `dlss_wgpu` is possible on
Vulkan later.

## Phases

1. Edits (done).
2. Shading inputs from generation (lane gradients and materials, brick face
   occlusion); shade reads the home column.
3. Bitmask bricks for primary, sun and sky rays.
4. Tile pre-pass for primary.
5. Coarse skylight; generation time budget.
6. Measure everything through the flight harness and the editor harness;
   update README and remove what each phase replaced.
