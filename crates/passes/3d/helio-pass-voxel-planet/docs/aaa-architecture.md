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
*Decision:* **keep** exact GPU generation; **move shading inputs into
generation**: per lane the ground gradient and surface material, per brick
face the corner occlusion, so shading reads one column instead of up to 20
hash lookups and noise per pixel. Budget generation by measured GPU time.

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

### 5. Shading

*Have:* 3.5-4 ms per frame at any view: per pixel the ground field (4
column lookups, 12 heights), material slope, procedural material noise and
8 occupancy lookups for corner AO.
*Best known:* read cached per-voxel attributes (Teardown palette; Lumen
surface cache 2.4 vs 11.5 ms evaluating per hit).
*Decision:* **replace** per-pixel reconstruction with generated attributes
(2); keep procedural albedo detail (cheap hashes) and filtering.

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
