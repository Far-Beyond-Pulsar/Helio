# Rendering: from hybrid baseline to path-traced lighting

Status: started 2026-11-17, after the layered rewrite (`layered-rewrite.md`,
`plugin-migration.md`). The architecture those documents describe stays: Helio
knows no object kind, kinds are plugins, kinds meet at frame contracts. This
document is the order the rendering work follows on top of it, and what each
phase built.

Goal: lighting at the level of the current path-traced and ray-traced
engines (UE 5.8 Lumen and MegaLights, RTXPT 1.9, the 2026 NVIDIA
path-tracing guidance), on every vendor's GPU, with a tier that needs no
hardware ray tracing.

Two rules hold in every phase:

- **Rays are budgeted per signal.** Direct light, diffuse indirect, specular
  indirect, cache updates and volumetrics each have a ray count the frame can
  report and a consumer can cap.
- **Caches, denoisers and upscalers are part of the renderer.** A technique
  is judged with them, and against a brute-force reference path tracer, not
  by eye on one scene.

## 1. Where it started

Measured on Helio `main` c5c35846 and Pulsar-Native `main` 61f26b106.

| Tier (NVIDIA path-tracing guidance, 2026) | State |
|---|---|
| Baseline hybrid: raster G-buffer, ray queries, visibility rays, temporal denoise | There, with ray queries off by default in the editor |
| Shipping: ReSTIR direct and indirect, AS compaction and refit, material LOD by ray class, radiance cache, shared denoiser | Missing |
| RTX: shader execution reordering, opacity micromaps, ray reconstruction | Not reachable through wgpu |
| Clustered geometry, partitioned TLAS | Not reachable; NVIDIA-only on Vulkan, cross-vendor only in the DX12 preview |

The gaps were structural:

- Only meshes were in the TLAS. Voxel terrain traced through its own walk,
  and only for GI's probes; foliage, skinned meshes and glass were not
  traced.
- The one cross-kind ray path (rays, tracers, hits) belonged to GI, and the
  meshes' tracer lived in the GI plugin.
- A hit was shaded with one colour per instance and Lambert.
- Diffuse indirect was a probe clipmap only; specular indirect had no
  dynamic source by default; no ambient occlusion publisher for meshes.
- TSR, the light field and the voxel shadows each had a private denoiser.
- No area or emissive lights.
- The wgpu fork had no acceleration-structure refit, and compaction
  deadlocked against present.

## 2. The order

Each phase is a branch `render/r<N>-<slug>`, the same name in every
repository it touches, one PR per repository.

| Phase | What | Why here |
|---|---|---|
| r1 | Ray streams: one way to trace every kind. AS refit and compaction in the fork. Skinned meshes and foliage traced. Ray queries on by default, AS memory and rays-per-signal statistics. | Everything after traces rays; nothing after should know a kind. |
| r2 | The surface at a hit at full material quality, by ray class, through the materials' evaluator. A reference path tracer. | Reflections and a cache need real surfaces; every later technique is checked against the reference. |
| r3 | One denoiser contract with history confidence. A world-space radiance cache. Per-pixel diffuse indirect; the probe clipmap stays as the tier without ray tracing and for media. | The largest visible gain; the cache also makes r4's secondary bounces cheap. |
| r4 | Traced reflections with lit hits. | Needs r2's surfaces and r3's cache and denoiser. |
| r5 | Direct light: many lights sampled with traced visibility, area and emissive lights, one shadow path. | Replaces the atlas as the primary path; the atlas stays for the tier without ray tracing. |
| r6 | AS scale: opacity micromaps, cluster BLAS for virtual geometry, partitioned TLAS, where a backend offers them. | Vendor features; the engine must be right without them first. |
| r7 | Vendor reconstruction and upscalers as optional backends; dynamic resolution. | They replace r3's denoiser where present. |

Geometry work (virtual-geometry shadows, texture streaming) is a separate
track.

## 3. r1: ray streams

### The problem

A consumer of rays (GI's probes, reflections, a cache's update, shadow
rays) must see every kind of geometry, and Helio must know none. Triangles
live in a TLAS. A voxel terrain is no BLAS: it is traced by a compute walk
over its own chunk trees.

### How others do it

- **Unreal, Frostbite, RTXPT:** one ray-tracing scene the renderer builds,
  because the renderer knows every primitive type. Hits run hit groups from
  a shader table (ray-tracing pipelines) or one inline megakernel. Path
  tracers on pipelines need shader execution reordering to get material
  coherence back.
- **Wavefront path tracers** (Laine, Karras and Aila, HPG 2013; Cycles and
  the production GPU renderers): rays, hits and shading requests are rows in
  buffers, and each stage is a pass over a batch. Coherence comes from
  sorting rows.
- **Call of Duty (SIGGRAPH 2026):** a constant ray count per frame,
  allocated across pixels.

wgpu has ray queries only: no pipelines, no shader table. An inline
megakernel would put every kind's intersection code, material code and
bindings into every consumer's shader, and a plugin that is absent would be
a shader permutation.

### The design

A **ray stream** is a frame contract (`helio_frame_api::rays::stream`):

- A consumer declares a `Stream` (a name, a class, a capacity), owns its
  buffers (`StreamBuffers`) and writes rays into them: `SceneRay`, 32 bytes,
  frame space.
- Each kind registers one `Tracer`. For a stream, the tracer's pass reads
  the rays and writes, into each ray's `SceneHit` (16 bytes), the nearer of
  what is there and its own hit. For a nearest-hit stream the tracer that
  owns a hit then writes its `SceneSurface` (20 bytes): normals, base
  colour, metalness, roughness, transmission, emission. For an occlusion
  stream the row accumulates what gets through.
- `stream::trace(build, &stream)` adds every tracer's passes at that point
  of the graph. The consumer names no kind; a kind names no consumer.

The stream is the shader table as data: the tracer id in a hit is the hit
group, and a pass can bin hits by tracer and material.

What it costs: 68 bytes a ray across the three rows, written and read once
each. What it scales with: the rays a consumer asks for, never the scene's
size.

The TLAS stays the frame's (`rays::SCENE_RAYS`), built by the meshes'
projection; the meshes' tracer queries it. A consumer that needs one
visibility test per pixel against triangles alone may still query it inline,
and then sees no other kind.

### As built

#### The contract

`helio_frame_api::rays::stream` and its snippet (`//!use frame_ray_stream`,
`shaders/ray_stream.wgsl`).

- **Rows.** `StreamHeader` (16 bytes: indirect dispatch groups of
  `GROUP` = 64 rays, and the count), `SceneRay` (32), `SceneHit` (16),
  `SceneSurface` (20). Layout tests cover Rust against WGSL.
- **A ray's cone** (`SceneRay::cone`, `scene_ray_cone`): the spread a metre
  (half float), the caster mask (`MASK_STATIC`, `MASK_MOVABLE`), and a
  starting width (8 bits, a micrometre to kilometres in steps of 9%), so a
  ray that leaves a hit starts as wide as the footprint that found it.
  `scene_ray_footprint(ray, t)` is what a tracer reads textures and tests
  cut-outs at.
- **A tracer traces from the ray's origin**, from distance 0. The consumer
  starts its rays off the surface they leave; a tracer adds no bias of its
  own.
- **A nearest hit** (`scene_nearest`): the distance, the tracer's id (4
  bits, `MAX_TRACERS` = 15), `INSIDE` (the origin is inside something
  solid), and 27 + 32 + 32 bits that are the tracer's own (it reads them
  back in its surface pass). **An occlusion row** (`scene_occlusion`): what
  gets through, as a colour, so cut-outs and tinted glass share one row.
- **A surface** has what a consumer may light without knowing the kind:
  shading and face normals (octahedral), base colour, metalness, roughness,
  transmission, emission (RGB9E5).
- **Registration.** A kind's feature calls
  `RegisterRayTracer::ray_tracer(Tracer { name, supported, trace, surface })`.
  `supported(env)` says whether it traces on this device and graph (the
  meshes' tracer needs ray queries; the terrain's needs none), so
  `stream::traced(env)` is how many kinds a consumer's rays would see, and a
  consumer that sees none schedules nothing.
- **Statistics.** `Stream::stat` reports the rays of each stream a frame
  under `rays.<stream>`: the per-signal ray count the rules above ask for.
- **Runs from one origin** (`Stream::with_run`, `Stream::run`, 1 = none):
  the consumer states that its rays come in runs of N consecutive rows
  from row 0 sharing their origin and caster mask (a probe's rays, a hit's
  shadow rays toward several lights, a pixel's samples). A tracer whose
  cost has a part that depends on the origin alone pays it once a run; one
  without (a TLAS query) ignores it. Wavefront tracers recover this
  coherence by sorting rays; here the consumer that made the rays that way
  says so.

#### Tracers

- **Meshes** (`helio-pass-mesh-rays`, the meshes plugin): one compute pass
  a stream over the frame's TLAS with a ray query. Cut-outs are tested at
  the ray's footprint (the alpha mip that footprint covers; stochastic past
  mip 0). The surface is written in the same pass: only the committed ray
  query has the instance's transform. r2 splits a surface pass off once the
  projections publish a transform a row.
- **Voxel terrain** (the voxel plugin's `chunks::rays`): the chunk-tree walk
  the camera uses, entered from any origin. A nearest stream is an entry,
  which places each ray in its chunk, then the walk, then a surface pass. A
  stream in runs (`Stream::run`) places each run's origin once, a thread a
  run, where the origin is in the volume (where a ray starts depends on its
  direction only from outside, and those runs enter ray by ray). A ray whose
  origin is in rock, in a solid cell or on a solid face reports `INSIDE`
  with no walk. One dispatch a drawn volume: cost is rays times volumes,
  which a ray-to-volume binning pass would remove when a scene has many
  volumes. The tracer walks the cut the camera's draw published, so a
  consumer schedules after the G-buffer phase to see this frame's terrain.
- **Not built: occlusion at the ray's footprint.** Answering the
  terrain's part of an occlusion ray along the sun from its sun occlusion
  pages at the level whose texel is the ray's width (geometry level of
  detail by ray footprint: Lumen lighting its surface cache from shadow
  maps and distance fields) cut GI's sun rays on voxel_craft from 0.41 to
  0.07-0.09 ms, but brightened voxel_craft's shade by 3-8 % with every
  filter tried, where the same build with the pages off matched the walk
  inside run-to-run noise (the voxel plugin's `chunk-octree.md`, "Ray
  streams"). Coarse pages under canopies let more sun through than the
  walk; until that is understood the terrain walks occlusion rays.

#### GI on ray streams

GI (the GI plugin's `helio-pass-gi`) is a consumer of two streams and knows
no kind:

- `gi_probe` (nearest, 64 rays a probe updated this frame, up to 174,080,
  each probe's a run from the probe itself: `Stream::with_run(64)`) and
  `gi_shadow` (occlusion: a sun ray and one local light's a probe ray's
  hit, packed from row 0, the count written on the GPU). A shadow ray
  starts as wide as the probe ray's footprint at its hit
  (`GI_RAY_SPREAD * t` from the probe) and keeps that width, so a tracer
  may answer it at that level of detail (the meshes test cut-outs at it).
- One node at `Slot::Geometry` 20, `once_per_frame`, after the G-buffer
  phase (so the terrain's tracer walks this frame's cut): **GI** (the sun,
  probes that came into the clipmap reset, the probe rays) → each kind's
  trace and surface passes → **GILight** (a hit's light from its
  `SceneSurface`: the sun's and one local light's, each pending on a
  shadow ray, and the rest: the probes' irradiance at the hit, through it,
  what it emits; a miss takes the sky) → each kind's trace of the shadow
  stream → **GIIntegrate**. The probes take this frame's rays in this frame;
  the resolve (`GIResolve`, `Slot::LightingInputs`) reads them for the
  lighting.
- A miss takes the sky through the atmosphere's published sky frame and
  table (`lighting::sky`), the sun its illuminance through the air; without
  an atmosphere, the lights' first directional light and the renderer's
  ambient.
- A tracer that knows geometry it could not trace this frame says so on
  the stream (`stream::Answers::short`: the terrain while its ray kernels
  compile or before its first draw), and GIIntegrate holds the probes that
  frame. Without it, a scene whose meshes answered before the terrain
  taught the probes a world without ground, which their history kept for
  hundreds of frames (the probes over open ground held 0.18 of light from
  below where their rays found 0.26).
- **The indirect diffuse's occlusion distance**
  (`lighting::indirect_occlusion`): the smallest distance at which the
  graph's indirect diffuse resolves occlusion. A kind that lights its own
  surfaces adds the occlusion nearer than that itself and leaves the rest
  to the indirect diffuse. GI offers twice its finest probe spacing, through
  a preparer evaluated when a consumer asks (the terrain's draw builds
  before GI's node), and offers nothing where it will not build (nothing
  traces, a forward lighting model, its node absent or bypassed:
  `BuildEnv::plans`). The terrain's per-pixel sky rays stop there; with no
  offer they keep their own reach (48 cells). The setting that held this
  distance per host (`voxels.sky_reach`) is gone. r3's traced per-pixel
  indirect diffuse offers a few centimetres, and the terrain's own term
  shrinks to contact occlusion with no change to either side.

Measured on an RTX 3060, median GPU ms over pipelined frames (voxel_craft:
median of six 240-frame windows, one pose; voxel_flight: the ground holds'
p50 over 25 s; r0 and the closing build alternated, two runs each,
2026-11-18; voxel_flight's r0 GI rays in the draw are its VoxelChunks less
the closing build's, the rest of that pass being the same work):

| Scene | r0 | r1 as first built | r1 closing |
|---|---|---|---|
| voxel_craft | rays in the draw 1.78-1.82, GI 0.47, GIMeshRays 0.07: 2.31-2.36 | 2.94 | GI 0.06, VoxelRays gi_probe 1.59, surfaces 0.23, MeshRays gi_probe 0.05, GILight 0.37, VoxelRays gi_shadow 0.40, MeshRays gi_shadow 0.03, GIIntegrate 0.13: 2.87 |
| voxel_flight Earth | rays in the draw 1.08, GI 0.29: 1.37 | 2.53 | GI 0.07, VoxelRays gi_probe 1.50, surfaces 0.16, GILight 0.26, VoxelRays gi_shadow 0.34, GIIntegrate 0.13: 2.46 |
| voxel_flight Moon | rays in the draw 0.94, GI 0.25: 1.19 | 2.52 | GI 0.06, VoxelRays gi_probe 1.54, surfaces 0.13, GILight 0.22, VoxelRays gi_shadow 0.46, GIIntegrate 0.13: 2.54 |

Where the first build's cost goes, split by pass (a profiling build timed
the entry and the walk apart): the probe walk's entry 0.085 ms on
voxel_craft and 0.11-0.12 on the planets, its walk 1.53 and 1.44; the
sun's shadow rays 0.36-0.45 (an entry 0.08 and a walk 0.28-0.37 on the
planets). What changed here: a probe's 64 rays are a run
(`Stream::with_run`), its origin placed once, and an origin in a solid
cell is an inside hit with no walk (on GI's 174,080 rays in isolation,
`stream_rays_cost`: entry 0.166 -> 0.130 ms, walk with entry 1.324 ->
1.303). The sun's shadow rays answered from the terrain's sun occlusion
pages at their footprint cost 0.07-0.09 instead of 0.41 but brightened
voxel_craft's shade by 3-8 % and are not built (Tracers, above).

What remains, measured, against r0: the probe walk itself, 1.44-1.53 ms
in the frame against r0's whole in-draw GI kernel at 1.05 on the planets
and 1.78 on voxel_craft. The kernels are the same walk: on identical rays
in isolation (`stream_rays_cost`, 174,080 probe rays about a ball on an
Earth) the stream walk with its entry costs 1.30 ms against the old
kernel's 1.12 with its shading and sun lookup. In the frame the old
kernel ran inside the chunk pass's own compute pass, beside its shadow and
denoiser dispatches with no barrier between them, where a stream's passes
are the GI node's, between barriers on the stream's rows; that the planets'
gap is lost overlap and not work is the likely reading and is not proven
(an r0 build dispatching its kernel twice measured the second at 0.8 ms
with the caches warm, which does not separate the two). The sun's shadow
rays walked as a stream (0.33-0.45) where r0 read the pages in its kernel;
the surfaces as a pass of their own (0.13-0.25: the hit shaded as the
G-buffer shades it at the ray's footprint, where r0 wrote a flat colour
and face normal); GILight and GIIntegrate with the shadow stream between
them (0.44-0.58 against r0's one GI pass at 0.25-0.47: a `GiPending` row a
ray). r2 shades hits by ray class; a ray-to-volume binning and r3's
radiance cache take probe rays off the walk; the sun's visibility of a
hit at its footprint waits on the canopy question above.

Final images (RGB 0-1, mean |Δ| of 8-frame burst means; voxel_craft four
poses, no clouds; voxel_flight ground holds):

| | r0-r0 | r1-r1 | r0-r1 (closing build) |
|---|---|---|---|
| voxel_craft, 4 poses | 0.00004-0.00046 (p99 0.0015-0.0039) | 0.0006-0.0017 | 0.0010-0.0025 (p99 0.008-0.020); mean luminance r1/r0 -0.2 % on the forward pose, +0.3 to +0.6 % on the others; temporal std 0.00007-0.00027 in both |
| voxel_flight Earth hold | 0.0006-0.0008 | 0.0008 (closing against first build) | 0.0009-0.0012; luminance 0.4615 / 0.4614 |
| voxel_flight Moon hold | 0.0008-0.0010 | 0.0011 | 0.0015-0.0016; luminance 0.5264 / 0.5270 |

The r0-r1 image difference on voxel_craft is not explained. Proven here:
the stream changes of this round (runs, inside origins, probe rays from
the probe itself) do not move the image (a build with them and without
the page answers against the build before them: 0.0006-0.0013, inside
r1-r1 noise); answering the sun's visibility from the sun occlusion pages
(the candidate: r0 read them) moves voxel_craft further from r0, brighter
by 3-8 %, not toward it; an instrumented r0 with the terrain's sun
visibility of GI hits set to 1 rendered differently from r1 with the
terrain's occlusion rays off by 0.005-0.022, its tree trunks and held mesh
losing the green bounce both other builds show, so that bisection is
itself unreliable. Earth and Moon holds stay at 1-2x run-to-run noise.

#### Hosts

Ray tracing is the default tier. The engine's device
(`request_engine_device` in Pulsar: editor window, game window, headless)
asks for ray queries where the adapter runs them, unless the user turned
`editor.renderer.hardware_ray_queries` off; the tier without ray tracing is
what that setting, or an adapter without them, gets.

#### Acceleration structures

`helio_core::acceleration` (`BlasManager`, `TlasManager`) is the build
policy every ray-traced kind uses; a kind's projection keys its BLASes,
says what each one's geometry does (`Movability`) and hands over complete
instance lists. Helio pins the wgpu fork branch `helio/as-refit`: refit
(`PreferUpdate`, `request_rebuild`), compaction safe against present, and
`Blas::sizes` / `Tlas::sizes` for memory.

- **Static geometry** (`!movability.can_deform()`): `PREFER_FAST_TRACE |
  ALLOW_COMPACTION`, `Build`. `BlasManager::maintain(queue)`, once a frame
  before the frame's builds, prepares the BLASes built two calls ago and
  compacts those the device reported ready (on its own submits and polls:
  no wait, no poll of the manager's), oldest first, up to
  `BlasPolicy::compaction_bytes_per_frame` (32 MiB; at least one a call).
  The compacted handle replaces the built one and `revision()` changes; the
  next TLAS build rewrites only the slots whose handle changed, and the
  built structure stays alive while earlier TLAS builds hold it. A change
  of static geometry is a new structure (a compacted one cannot be built).
- **Deforming geometry** (`Movability::Dynamic`): `PREFER_FAST_BUILD |
  ALLOW_UPDATE`, `PreferUpdate`. `BlasGeometry::positions_revision`
  (positions written in place) refits; `topology_revision`, counts, index
  buffer, first index, opacity or a movability change make a new structure.
  Every `BlasPolicy::refits_per_rebuild` (16) refits the policy rebuilds it,
  the k-th deforming BLAS first rebuilding after `N - k % (N + 1)` refits so
  rebuilds spread over frames; `request_rebuild(id)` rebuilds on the next
  change (a large deformation the owner knows of). COPY_SRC-only index pools
  are copied once and the copy kept, since a refit needs the same index
  buffer. A refit or rebuild changes the BLAS's generation, so the next TLAS
  build refits the instances holding it (wgpu rejects a TLAS older than its
  BLASes).
- **TLAS**: `PREFER_FAST_TRACE | ALLOW_UPDATE`, `PreferUpdate`, always fast
  trace: every ray of the frame starts in it and its build is linear in
  instances, so build-speed flags would trade every ray's cost for a small
  saving. Only changed slots are rewritten. A build is a refit unless the
  count changed, a slot holds another instance (its BLAS id or custom data
  changed), `request_rebuild()` was called, or `TlasPolicy` says so: after
  `max_refits` (8) refits, or once the instances moved since the last full
  build reach `moved_fraction` (0.5) of the count. The last adapts to the
  churn the manager observes: a scene where most things move every frame
  rebuilds every frame; one with a few movers refits.
- **Memory and stats**: `BlasManager::memory()` (`count`, `bytes`,
  `bytes_before_compaction`), `TlasManager::memory()` (structure and
  instance buffer), `counters()` on both (builds, refits, policy rebuilds,
  compactions, scratch bytes). `report(&FrameStats)` once a frame adds them
  under `acceleration::stat`: `rays.blas_bytes`,
  `rays.blas_bytes_before_compaction`, `rays.tlas_bytes`,
  `rays.as_scratch_bytes`, `rays.blas_builds`, `rays.blas_refits`,
  `rays.blas_compactions`, `rays.tlas_builds`, `rays.tlas_refits`. Every
  owner adds into the same names, so the totals cover every kind.
- **Ordering**: encoders the managers record into are submitted in the
  order recorded (a refit submitted after a different build of the same
  structure is rejected at submit).

How it compares: Unreal's ray-tracing geometry keeps static geometry
fast-trace and compacted through a queue after its build, and updates
dynamic geometry (skinned, deforming) with fast-build refits and periodic
full builds (`EAccelerationStructureBuildMode::Update` against `Build`),
its dynamic-geometry scratch shared across the frame's updates; the 2026
NVIDIA path-tracing guidance is the same split (static: fast trace,
compacted; dynamic: refit, rebuild periodically), with update priority by
hit counts under instance and geometry budgets. What breaks each: a refit
under large deformation (an explosion, a ragdoll) keeps a tree whose boxes
overlap, so trace cost grows until the rebuild; compaction holds both
copies until the frames using the built one are done (bounded here by the
byte budget a frame); a TLAS rebuild is linear in instances, which a
partitioned TLAS (r6) removes. Not built yet: priorities by hit counts and
geometry budgets (the callers' projections budget their own builds), and
a scratch pool shared across builds (wgpu allocates scratch per build
call).
