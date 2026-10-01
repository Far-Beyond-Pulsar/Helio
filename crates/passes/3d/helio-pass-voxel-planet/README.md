# helio-pass-voxel-planet

Destructible voxel worlds for Helio: Earth-sized cube-sphere planets, finite
planes and effectively infinite planes, built from exact voxels of 0.1 m to
1 m, fully editable, and rendered by tracing every pixel through a GPU-driven
clipmap. Terrain is traced rather than smoothly meshed. The current distance
field uses larger, band-limited cells; preserving the base voxel geometry
and small edits at every distance remains an open requirement.

This document maps the system for people who will work on it: what each part
does, how a frame flows, which invariants hold it together, why things are
the way they are, how to measure it and how to extend it. The Pulsar editor
integration is documented in Pulsar-Native's `docs/voxel-system.md`.

## Contents

### Flight and appearance corrections (draft)

Level selection now defers a switch when the destination cell is already
solid. Switching resolution is not a surface crossing: returning an interior
hit used to retain an unrelated top-face normal and shade grass as subsoil.
`tests/grey_patch.rs` checks radial surface entries at 1/4/16/64 km and snow
visibility at 64/300/1,000 km.

Climate and filtered grass appearance use the ray's base-grid footprint.
Fine hits reuse the resident canonical column top at base-cell precision;
distant 2x2 pixel footprints share an unrounded canonical height query,
with individual queries at distant face/depth discontinuities. This
prevents high-altitude cells rounding snowy
mountains to sea-level grass. This is appearance filtering; the existing
band-limited distance geometry remains a separate, unfinished fidelity gate.
Small brushes are still omitted by coarse levels. Exact filtered canonical
geometry and edit coverage at every distance are **not qualified** by this
change, nor by the near-field CPU/GPU tests.

All distant height queries, including discontinuity fallbacks, run in the
climate prepass. The material shader only reads the resolved height, so the
terrain generator's register pressure does not affect fine material/AO work.
The screen cache stores one height per pixel (4 bytes); fine hits do not read
it. At 1440x810 this adds 3.34 MiB relative to the former 2x2-only cache.

Motion forecasts prefetch up to 350 ms ahead, bounded by ground clearance and
the window capacity. Moving generation/admission budgets grow with backlog.
The offscreen flight is still not a substitute for a quiet native fast-flight
test. New Pulsar logs include eye/forward/up/viewport; replay preserves those
poses when available, and identifies legacy altitude-only logs explicitly.

`Settings::appearance` / `PlanetPass::set_appearance` control the sRGB palette,
per-material roughness, three grass colours and detail contrast without
rebuilding residency or changing recipes/edits. Defaults use muted meadow
greens, cool rock, warmer soil and less reflective snow. Pulsar persists the
same settings in `VoxelTerrainComponent.appearance_parameters` (JSON, empty
means defaults). For example `{"detail":[0.6,0.12,0.04,0.0]}` reduces grass
patch contrast, per-voxel pigment and edge occlusion; omitted fields retain
defaults. RGB and detail values are clamped to [0,1] before GPU upload.
RGB values are converted to linear colour on the CPU once per frame upload;
roughness stays linear. Shading does not repeat palette gamma conversion.

Hosts without an authored sky can call `Renderer::set_planetary_sky` with
`PlanetarySky::earth_like(eye_m, radius_m, sun_direction)`. It shares the
planet-centred eye and sun with the sky LUT and a bounded aerial-perspective
approximation after deferred lighting. Authored skies retain precedence.
Planetary sky is HDR and passes through the graph's normal tone mapping.
Atmosphere settings survive graph resize; the graph regression checks the
same local sky at two rotated poles. This is single scattering, not a claim
of volumetric weather, vegetation, water or a completed AAA art pipeline.

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

- Crisp voxels at every distance: near cells are cubes; far cells are the
  same field sampled at a coarser level, shaded with filtered appearance so
  sub-pixel cells do not alias. No meshes, no smooth LOD surface.
- One world, two consumers: the CPU (collision, ray casts, edits, gameplay
  queries) and the GPU (streaming, rendering) evaluate the same integer field
  and the same edits and agree to the bit.
- Space to ground in seconds: a camera can fall from orbit to walking height
  at the editor's altitude-proportional speed while residency keeps up.
- Destruction at scale: tens of thousands of edits stay exact and cheap.
- Budgets (RTX 3060, 1080p Quality, i.e. 1440x810 internal): terrain GPU
  p95 <= 5 ms, no CPU frame stalls, terrain GPU memory <= 1 GiB.
- Non-goals (for now): caves in generated terrain (edits make caves),
  translucent water, meshes inside the voxel pass, multiple worlds per pass.

## Crate map

| Path | Role |
|---|---|
| `src/grid.rs` | The canonical grid: equal-angle cube sphere (or plane), cells, levels, faces, exact cell walking maths. |
| `src/noise.rs`, `shaders/noise.wgsl` | Bit-exact integer noise (Q16 and fine Q24). The only noise terrain may use. |
| `src/terrain.rs` | Pluggable generators: `TerrainGenerator` -> `TerrainField` (CPU) + `TerrainProgram` (WGSL). Registry, material ids, `check_field`. |
| `src/landform.rs`, `shaders/landform.wgsl`, `shaders/flat.wgsl` | Built-in generators `helio.landform` and `helio.flat`. |
| `src/edits.rs` | Brushes (sphere/cube, remove/add/paint), per-face integer resolution, the shared `EditLog` and its tile index. |
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
| `shaders/trace.wgsl` | Exact hierarchical traversal of the canonical grid. |
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
  is an 8x8 footprint of cells at one level (`BRICK = 8`), with its whole
  occupied vertical extent; it is the unit of residency.
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
- Detail finer than a level's footprint is omitted at that level: coarse
  levels are band-limited point samples of the same field, not a separate
  smooth approximation. That is why LOD transitions never change the shape of
  the land, only its resolution.
- Heights are relative to the datum (the planet radius or the plane's y = 0)
  and may be negative: lowland and ocean basins sit below it. Nothing in the
  pipeline may clamp heights to the datum (see the band-top invariant below).

### Edits

A `Brush` (centre in planet metres, radius, sphere or cube, remove / add /
paint, material) is resolved per cube face into integer half-cell coordinates
(`FaceBrush`), so CPU and GPU apply the same integer containment test. Later
brushes override earlier ones. At level `L`, a brush smaller than half a level
cell is omitted (smaller than the point sample).

`EditLog` is ordered and shared between world copies: brushes live in
`Arc` chunks of 1024, the spatial index is a sealed shared map of tiles plus
recent entries, and every brush carries a prefix hash of the log up to it.
Copying a world with 50k edits costs about 0.1 ms, and the renderer finds the
common prefix of its synced log and the current one by binary search on
prefix hashes (O(1) when unchanged). The finest index tile is one column (8
cells), which keeps dense block edits (buildings) cheap to query.

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

### CPU: `Residency::plan` (render thread)

1. **Edit sync.** New or undone brushes since the last frame are found by
   prefix hash; new face brushes are uploaded; resident columns they touch
   are queued as urgent regenerations.
2. **Windows.** When the eye moved, a `WindowRequest` goes to the window
   worker thread, which computes each level's wanted disc of columns and
   returns add/remove diffs. Level 0 covers the level-0 distance (cells about
   a pixel wide at its edge), each coarser level twice the distance. A level
   is on only if terrain within its reach can be nearer than that distance:
   the worker bounds the terrain around the eye per level
   (`Planet::local_outer_radius`), so over a meadow 1 km below the fine
   levels are off instead of streaming columns under a tenth of a pixel.
3. **Diff application.** Diffs are queued and applied in order within the
   frame's CPU budget (1.5-3 ms moving by backlog, 4 ms still): removes evict residents,
   a switched-off level clears its queue, adds become pending in a priority
   bucket. Diffs get at most 60 % of the budget while columns wait. A level
   with unapplied diffs is *catching up*: its `fallback_distances` entry is 0
   (no guaranteed coverage).
4. **Admission.** Pending columns are issued nearest-first (the coarsest level
   always first, for global coverage) until the GPU job budget (from the
   measured GPU cost per job) or the CPU budget runs out: allocate a record,
   build its edit-reference list, reference its summary blocks, insert it in
   the hash table. Everything issued this frame is one GPU patch.

### GPU

1. **Upload** of the patch: table writes, jobs, evictions, edit lists, block
   inits.
2. **Generate** (`generate.wgsl`): per job, evaluate the terrain program for
   the 8x8 columns, apply edits, find the occupied band (solid below, air
   above, mixed bricks in between), allocate a brick run of the right size
   class, write mixed bricks, publish the record, and raise the column's
   summary-block tops and the level's top. Evicted runs return to free lists.
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
  tops must bound every solid cell and should be tight. The generator's band
  top once clamped to the datum (`max(top, 0)`): below sea level every column
  claimed ~30 m of air, rays stepped cell by cell through it (137 steps per
  ray instead of 6) and columns stored empty bricks. Tops are allowed to be
  loose only by the 3-bit `gap` (<= 7 cells).
- **Exact traversal, conservative accelerations.** The sky bound, summary
  blocks and residency hints may only skip space proven empty; tests compare
  renders with and without each acceleration.
- **Hash table.** Linear probing, `slot_hash(key0, key1)`, at most
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
  column on an RTX 3060, ~6500 jobs per moving frame. The CPU budget for
  diffs and admission grows from 1.5 to 3 ms while moving as the backlog
  reaches 20k columns.
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
  time-budgeted; nothing rehashes or reallocates in bulk on the render
  thread (a 1M-entry `HashMap` doubling cost 70 ms; a table rehash 70-90 ms).
  If you add per-column CPU work, keep it inside the budgeted loops.
- **Stable LOD dither.** The level-transition threshold is hashed per column,
  not per frame, so a moving camera sees each column change level once
  instead of flickering between two levels while TAA history is rejected.
- **Camera-relative frames.** All GPU positions are relative to the eye.
  Anything defined in world space (overlays, billboards, SceneDB rows) must
  be rebased by `PrepareContext::world_origin`. Rays toward the far plane must
  be built as `far.xyz - far.w * eye` (homogeneous): with near 5 cm and far
  40 000 km the far plane is at f32 infinity and dividing by `w` gives NaN.
- **Filtered appearance** (after "Filtered appearance for voxels", HPG 2023):
  cells about a pixel wide are shaded with the column's macro normal and show
  surface material on risers, so distant terrain has no contour lines; cells
  several pixels wide keep crisp faces. The blend follows the pixel
  footprint, so level changes show no seam.
- **Band overflow.** A column taller than `MAX_BAND` bricks is not published;
  coarser levels cover it.

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
| `HELIO_VOXEL_FLIGHT_CRUISE=<m>`, `_CRUISE_SECS=<s>` | Level flight at that height at the editor's speed for 20 s, then a stop: residency lag while moving and time to converge. |
| `HELIO_VOXEL_FLIGHT_REPLAY=<engine.log>`, `_REPLAY_FROM/_TO=<s of day>`, `_REPLAY_DEG` | Replays the altitude timeline of a Pulsar editor session logged with `PULSAR_VOXEL_STATS=1`. |
| `HELIO_VOXEL_FLIGHT_SUN=x,y,z` | Sun direction (the editor's default Sun is straight up). |
| `HELIO_VOXEL_FLIGHT_QUICK=1`, `_GROUND_ONLY=1`, `_CPU_PROBE=1` | Short timing probe, ground audits only, CPU per pass. |
| `HELIO_VOXEL_PLAN_TRACE=<ms>` | Logs residency plan phases of frames taking over `<ms>` (10 if not a number). |
| `HELIO_VOXEL_LOD_DITHER`, `HELIO_VOXEL_NO_HORIZON`, `HELIO_VOXEL_NO_FAILSAFE` | Override the dither width; disable the sky bound; disable its fail-safe (A/B timing). |

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
  overlays stay in world space in camera-relative frames.

## Extending

**A terrain generator.** Implement `TerrainGenerator` (id, version, info with
an optional settings-component name) returning a `TerrainField` and its
`TerrainProgram` (WGSL defining `TerrainConstants`, `terrain_height`,
`ground_material`, plus the constants' bytes). Use only the integer noise
library. Register it with `terrain::register`. Add a test calling
`engine::verify_field` for every shape and `terrain::check_field`. Changing
settings rebuilds the world without recompiling shaders; pipelines are keyed
by program.

**A material.** Add the id to `terrain::material` and `world.wgsl`, its
colour to `palette` in `surface.wgsl`, and to editor-facing enums (Pulsar's
`VoxelTerrainMaterial`).

**A brush shape.** Extend `BrushShape`, its per-face resolution in
`edits.rs`, the containment test in both `edits.rs` and `generate.wgsl`
(`apply_edits`), and the band bounds from brushes in `generate.wgsl`.

**GPU work per column.** Put it in `generate.wgsl`; keep CPU admission cheap
and inside the budgeted loop in `Residency::plan`.

## Known limits and open work

- The directional sky bound occasionally (a few frames per orbit trip, during
  very fast altitude changes) ends limb rays before their terrain; the
  primary-ray fail-safe hides it completely (`HELIO_VOXEL_FLIGHT_PROBE` finds
  no holes), but the table inconsistency behind it is not yet understood.
  Rings and fallback distances are identical in the bad frame and the one
  before; suspect block publication vs. the live-block list.
- Hosts must supply `PlanetarySky` each frame for the planetary fallback;
  otherwise the outdoor fallback keeps its fixed world +Y convention.
- Dense construction stores edit references per column (4M-word pool);
  sparse block-override bricks would scale building further.
- Edit ids (`next_brush`) grow monotonically after undo.
- Appearance contrast can be tuned, but existing coarse geometry transitions
  and distant small-edit fidelity are still open.
- `tests/gpu.rs` once hung when run in parallel with other GPU work; it has
  not reproduced since.

## Measured dead ends

Kept out after measurement: a beam or depth prepass (no faster than tracing
once made rigorous), wavefront ray compaction, sunlight temporal reuse, hash
fingerprints, a sun-ray escape bound, and tilted plane-fitted summary-block
bounds (rays converge on slopes, so only 1 in 6-13 attempts succeeded).
