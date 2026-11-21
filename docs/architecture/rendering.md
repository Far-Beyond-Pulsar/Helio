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
  `SceneSurface` (20; 32 since r2, the frame's `SurfaceData`, section 4).
  Layout tests cover Rust against WGSL.
- **A ray's cone** (`SceneRay::cone`, `scene_ray_cone`): the spread a metre
  (half float; 11 bits since r2), the caster mask (`MASK_STATIC`,
  `MASK_MOVABLE`; 4 bits since r2, beside the ray's coordinate space,
  section 4), and a starting width (8 bits, a micrometre to kilometres in
  steps of 9%), so a ray that leaves a hit starts as wide as the footprint
  that found it.
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
  from row 0 sharing their origin and caster mask (r2's camera rays first,
  a hit's shadow rays toward several lights, a pixel's samples). A tracer
  whose cost has a part that depends on the origin alone pays it once a
  run; one without (a TLAS query) ignores it. Wavefront tracers recover
  this coherence by sorting rays; here the consumer that made the rays that
  way says so. Runs serve only rays that really share an origin: GI's probe
  rays do not (below).

#### Tracers

- **Meshes** (`helio-pass-mesh-rays`, the meshes plugin): one compute pass
  a stream over the frame's TLAS with a ray query. Cut-outs are tested at
  the ray's footprint (the alpha mip that footprint covers; stochastic past
  mip 0). The surface was written in the same pass (only the committed ray
  query had the instance's transform); r2 split a surface pass off on the
  instances' transforms the projections now publish (section 4).
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
  each starting `gi_ray_offset` behind its probe along its own direction,
  and GILight taking a hit within that distance as inside) and
  `gi_shadow` (occlusion: a sun ray and one local light's a probe ray's
  hit, packed from row 0, the count written on the GPU). A shadow ray
  starts as wide as the probe ray's footprint at its hit
  (`GI_RAY_SPREAD * t` from the probe) and keeps that width, so a tracer
  may answer it at that level of detail (the meshes test cut-outs at it).
- **Why a probe's rays are not a run.** Started at the probe itself (a run
  of 64 from one origin, which the terrain's tracer places once), a probe
  lying on a surface met it only with the rays toward it: the others left
  from the surface's far side, so the probe stayed live and carried light
  through the surface (a solid cut-out card let 1 % through,
  `vg_ray_proxies`). Started `gi_ray_offset` behind the probe along its own
  direction, every ray of such a probe crosses the surface within that
  distance and the probe is dropped, whichever side f32 rounding put it.
  The rays then share no origin, so the stream is not in runs and the
  terrain enters them ray by ray.
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
planets). What changed here: an origin in a solid cell is an inside hit
with no walk, and a stream in runs places each run's origin once (on GI's
174,080 rays in isolation as runs of 64 from the probe, `stream_rays_cost`:
entry 0.166 -> 0.130 ms, walk with entry 1.324 -> 1.303). The closing
build's figures above were taken with GI's probe rays in runs; they are no
longer (GI on ray streams, above), so GI pays the terrain's entry ray by ray
again, about 0.04 ms of that isolated entry, not re-measured in the frame.
Runs stay for streams whose rays share an origin, r2's camera rays first.
The sun's shadow rays answered from the terrain's sun occlusion
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
the probe itself; GI's probes have since gone back to rays from behind the
probe, not in runs) do not move the image (a build with them and without
the page answers against the build before them: 0.0006-0.0013, inside
r1-r1 noise); answering the sun's visibility from the sun occlusion pages
(the candidate: r0 read them) moves voxel_craft further from r0, brighter
by 3-8 %, not toward it; an instrumented r0 with the terrain's sun
visibility of GI hits set to 1 rendered differently from r1 with the
terrain's occlusion rays off by 0.005-0.022, its tree trunks and held mesh
losing the green bounce both other builds show, so that bisection is
itself unreliable. Earth and Moon holds stay at 1-2x run-to-run noise.

Carried into r2, which owns each of them:

- **The terrain GI cost** (+0.5 ms on voxel_craft, +1.1 to +1.35 ms on the
  planets against r0). r2 decides how a hit is shaded by ray class, so the
  terrain's separate surfaces pass, and the walk's cost in the frame against
  the same walk in isolation, are measured and settled there. r1 merged with
  this cost, stated, instead of holding six repositories' branches open.
  r2 (section 4, "The terrain's surfaces"): the surfaces pass stays a pass
  by measurement, the sun's rays are answered at their footprint, and the
  walk's in-frame excess is the frame's rays' work, not overlap or cache.
- **The sun's visibility of a hit at the ray's footprint.** It waits on why
  coarse sun-occlusion pages let more light through under canopies than the
  walk does. That is a fault of the pages, which every reader of them sees,
  and is to be found before anything reads them at a footprint.
  r2 (section 4, "The terrain's surfaces"): found. The pages hold the
  walk's occluders; their read's bias of a texel hid every occluder within a
  texel of a receiver finer than its footprint. The sun's rays now walk
  that band and read the pages past it.
- **The r0-r1 image difference on voxel_craft.** r0 is not ground truth:
  both builds approximate. r2's reference path tracer is; the difference is
  judged against it, on the same poses.

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

#### Skinned meshes, foliage and culling in the TLAS

**One owner of skinned positions: the skin cache** (the meshes plugin,
`meshes_component::skin_cache`). One compute pass blends each posed skinned
instance's palette into its vertices once a frame (`helio_mesh_api::SKIN_CACHE`,
`SkinnedVertex`: position, normal and tangent octahedral at 16 bits an axis
(the frame's `scene_pack_unit`, the tangent's handedness the vertex's
bitangent sign), previous position; 32 bytes, the position first so a BLAS
builds from it directly) and an instance
table by entity row (`SKIN_CACHE_ROWS`, the instance rows' `mesh_id`). Every
drawer that drew a skinned mesh blended the palette in its vertex shader (five
copies: G-buffer, shadow casters, shadow transmittance, forward, transparent;
the selection mask through a sixth function); they all read the cache now
(`//!use mesh_skinning`), so what is rasterised is what is traced, and the
G-buffer's previous positions are the cache's. The pass blends the current
palette and the palette of the pose shown before it (the component mirrors
both) in one go, so a previous position costs no read of the cache. The
instance rows lost their skin
fields (`skin_vertex_offset`, `bone_palette_offset`, `bone_palette_count`):
a skinned instance is `INSTANCE_FLAG_SKINNED`, set by its draw rows, and
the scene join reads no skin or palette handles (`StaticObjectRow` 252 to
240 bytes, `GpuInstanceData` 224 to 208).

- **Cost**: the journal of `SkeletalMeshComponent` says which instances
  changed; an instance whose palette is bitwise the one it was posed with costs
  nothing (an animator showing a paused pose each frame). A posed instance is
  skinned once; on the frame after its last pose its row is marked settled
  (`SkinCacheRow::settled`: previous = current, a row write, no pass), so
  motion vectors stop with the pose, which the vertex-shader path got wrong
  (a pose shown every other frame kept its motion on the frame in between).
  The row's `pose` changes with every pose: the shadow pages key a skinned
  caster on it (`helio-pass-shadow-dirty`), since its transform alone does
  not change when it moves (the pages re-render with the pose, not because
  something else happened to invalidate them). No buffer or bind group is made unless the cache, its table or
  a pool grows.
- **What is skinned**: only what the frame draws or traces
  (`skin_cache::SkinView`, from the frame's camera, which host parts now get
  in `RenderHostPart::publish_frame`): an instance whose bounding sphere is in
  the camera frustum's side planes, or that the ray-tracing culling keeps
  (within `culling_radius`, or spanning `culling_angle`; that sphere around
  the eye covers the shadow casters near it). A pose that changes while
  nothing wants the instance is kept pending and skinned the frame the
  instance is wanted, with no motion that frame (its previous pose is
  stale); a posed instance off-screen and outside the traced range costs a
  sphere test. An instance unwanted for 300 frames gives its range back; one
  that is drawn while not resident (a shadow from outside both) draws its
  rest pose.
- **Memory**: 32 bytes a vertex of the resident instances plus 16 bytes an
  entity row up to the highest skinned one. Fifty 30,000-vertex characters in view are 48 MB. The
  cache doubles when an allocation does not fit (the old contents copied on
  the GPU) and repacks into a buffer of twice its use once less than a
  quarter is used (never below 16,384 vertices); a repack moves every range,
  so the ray-traced scene re-evaluates those instances.
- **Why raster moved onto it**: Unreal's GPU skin cache, which its ray tracing
  requires, is the same split (compute writes positions and tangents and keeps
  the previous positions; the raster and the ray-tracing geometry read them);
  Frostbite and Decima pre-skin in compute. Unreal budgets the cache's memory
  and falls back to vertex-shader skinning past it, a second skinning path;
  here every resident instance is in the cache, so there is one path, and the
  lever on memory is a skinned level of detail (open: skeletal meshes load LOD
  0 only). The vertex shaders lose four palette loads and a blend a vertex a
  view, which shadows paid per cascade and face.

**Skinned meshes in the TLAS** (the engine world's projection,
`meshes_component::ray_scene`): a BLAS per instance (they deform
independently), `Movability::Dynamic`, its segments the mesh's sections from
the cache (stride 32) and the mesh's index pool. `positions_revision` is the
cache's revision of the instance, so a pose refits, the acceleration policy
rebuilds after `refits_per_rebuild` refits, and an unchanged pose records
nothing. Refits are budgeted in triangles a frame (`ray_tracing`'s
`skinned_refit_triangles`, 262,144 by default: about 0.5 ms of refits on an RTX 3060, below), nearest first with each
frame waited counting as nearer (none waits for ever); one over the budget
keeps last frame's BLAS, a frame stale. Its rows carry `RAY_INSTANCE_SKINNED`
and the cache's first vertex (`RayInstance::skinned_vertex`, where the
transmission's fourth float was): the meshes' tracer reads a skinned triangle's
corners and normals from the cache (`RAY_SKINNED_VERTICES`, published beside
the TLAS) and its UVs from the mesh pool, so a hit shades from the skinned
normal.

**Culling by distance and angle**: both projections leave out an instance
whose bounding sphere is past `ray_tracing.culling_radius` (300 m) and spans
less than `culling_angle` (1 degree) from the eye, as Unreal's ray-tracing
culling does (`r.RayTracing.Culling.Radius` / `.Angle`); a traced instance
leaves only past 1.1 times both, so one at a threshold does not flicker. The
traced instances are tested again when the eye has moved a twentieth of the
radius (and an instance when it moves): never every instance every frame. A
culled skinned instance drops its BLAS. The standalone projection culls from
the eye its host gives (`SceneDbRayTracing::set_view`) and traces everything
without one.

**Foliage**: the foliage plugin places no mesh instances. Its kinds are
procedural blades and cards and voxel blocks expanded on the GPU each frame
(`FoliageKind::Mesh` is declared and drawn by nothing), and its feature
declares its layers untraced (`untraced_geometry`). Trees and bushes placed as
meshes are mesh instances, traced like any mesh, their leaves non-opaque
cut-outs alpha-tested at the ray's footprint (mesh rays' `cut_out_surface`).
Procedural grass is not traced, as in Unreal (landscape grass is out of its
ray-tracing scene) and the other engines: its blades are sub-pixel occluders
whose shadowing is a contact term: the deferred lighting's screen-space
contact visibility (`contact_visibility`, half a metre) and the blades' own
occlusion from their height (`GBUFFER_TAG_OWN_OCCLUSION`); their bounce is the
ground's. Wind and vertex animation:
traced geometry is the rest pose (a mesh's BLAS is built from its pool
vertices; foliage wind is a vertex-shader offset), the common choice, since a
refit a frame for every swaying tree would cost the TLAS's whole budget for
centimetres of motion GI cannot resolve.

**One TLAS owner per scene model**: `meshes_component::ray_scene` projects the
engine world (mesh components placed by their owners' hierarchy, material
instances, skinned meshes); `helio_mesh_host::ray_tracing` projects the
standalone hosts' `StaticObjectComponent` rows (Helio's examples, the plugins'
GPU tests). They read different data; each gets the culling and the
statistics; only the engine model has skinned meshes.

**Statistics**: both projections call `BlasManager::report` and
`TlasManager::report` once a frame (the engine world into its renderer's
`frame_stats()`; the standalone one once its host hands them over,
`set_stats`), so `rays.blas_bytes`, `rays.tlas_bytes`, builds, refits and
compactions are in the frame's stats, with the skin cache's
(`meshes.skin_cache_bytes`, `_instances`, `_updated`, `_vertices`) and the
TLAS's instances (`meshes.ray_instances`, `meshes.ray_instances_culled`).

Measured on an RTX 3060 (`meshes_component`'s `skin_cache_cost`, a release
build, median GPU ms over pipelined frames, three in flight; a 10,000-vertex,
19,602-triangle mesh (or a 29,000-vertex one) on two joints, every instance
posed each frame; the CPU column is the recording of the cache's update and
the refits). The skin pass column before is the pass that read the previous
position back from the cache; after, it blends the previous palette:

| Instances | Skin pass before | Skin pass | BLAS refits (and the policy's rebuilds) | TLAS | CPU |
|---|---|---|---|---|---|
| 1 x 10k posed | 0.010 | 0.011 | 0.023 | 0.033 | 0.08 |
| 16 x 10k posed | 0.048 | 0.043 | 0.63 | 0.049 | 0.24 |
| 50 x 29k posed | 0.606 | 0.529 | 2.68 | 0.057 | 0.58 |
| 256 x 10k posed | 0.602 | 0.452 | 9.24 | 0.074 | 2.95 |
| 256 x 10k, pose unchanged | 0.001 | 0.001 | 0.001 | 0.001 | 0.36 |
| 256 x 10k posed, off-screen and untraced | 0.002 | 0.001 | 0.001 | 0.001 | 0.21 |

Cost follows the posed, wanted instances' vertices and triangles; an
unchanged pose, or a posed instance nothing draws or traces, costs nothing
on the GPU. The skin pass runs at about 5.7 billion vertices a second
(256 x 10k): one mesh shared by every instance keeps its reads in L2, so
it is the 32-byte write of each vertex, about 180 GB/s, half this GPU's
bandwidth. What goes faster next is the input, not the kernel: skin weights
as four 8-bit joints and four 16-bit weights (12 bytes a vertex for the
current 32) once the importer writes them (open). The refits dominate
(about 2 ns a triangle), which is why they are budgeted: the default budget
of 262,144 triangles is about 0.5 ms of them on this GPU.

Frames (`installed_plugins`' `skinned_frames`: the engine path, a two-joint
skinned tube, a tree, ground, a sun, GI, ray-traced shadows, TSR at native
size, 640x360) against the meshes before the skin cache (kept executable):
lit frames at rest, bent and mid-animation differ by a mean of 0.22 to 0.24
(8-bit, max channel), the differing pixels on the axis gizmo's line and
silhouette edges (TSR history); the indirect view by 0.34 to 0.41, the most
of it the tube's bounce on the ground around its foot, which the earlier
TLAS did not hold. The bent pose's shadow on a renderer that saw the rest
pose first matches one that never did (mean 0.19, max 17). Culling the tube,
then the tree, out of the TLAS by `culling_radius` changes the indirect view
by 0.48 and 0.38 (max 25 and 15) where their occlusion and bounce were.

Open: real frames with a skinned character and a tree under GI and
ray-traced shadows, and the G-buffer A/B of a skinned mesh before and after
the cache (positions are the vertex shaders' formula, checked in
`skin_cache.rs`'s test to 1e-5; normals are snorm8 as the mesh's own), were
not captured: no example or level holds a skinned mesh, and the engine's
render acceptance suite was not run here.

**A second skinning path, recorded**: the skeletal mesh editor's preview
(`plugins/vendor/skm_editor`, `editor/src/rendering/viewport/skin_gpu.rs`
and its skinning WGSL) draws through a renderer of its own with its own
`SkinVertex` and vertex-shader blend. It is the animation plugin's, and goes
away when the preview draws through an engine-owned Helio runtime (Pulsar's
`render_runtimes`, which plugins get handles to and never build Helio
themselves); until then it is a known copy, not the engine's skinning.

## 4. r2: the surface at a hit, and the reference

### The problem

r1 gave every consumer the same hits. What a consumer gets at a hit is still
thin: the meshes' tracer writes the instance's row, one colour and a vertex
normal; no texture is read past a cut-out's alpha, no material graph runs.
Reflections (r4) and a radiance cache (r3) need the surface a pixel has. And
nothing in the engine says what the right image is: r1's voxel_craft frames
differ from the build before it by a fraction of a percent, and neither is
ground truth.

### How others do it

- **Unreal**: a hit runs the material's closest-hit shader from the
  ray-tracing pipeline's shader table, with cheaper variants by ray type
  (ray-tracing material LOD, "simplified" materials for shadow and GI rays,
  `r.RayTracing.*.EnableMaterials`); Lumen lights reflection hits either from
  its surface cache (cheap, coarse) or with full hit lighting (material and
  direct light at the hit) by roughness, its hit lighting one shading model
  shared with the deferred pass. Its path tracer is the in-editor reference,
  sharing the material graph through the same hit shaders. What it costs: a
  shader table permutation per material and ray type, and hit-shader
  divergence that SER (NVIDIA only) recovers.
- **RTXPT and the 2026 NVIDIA guidance**: material LOD by ray class
  (primary, specular bounce, diffuse bounce, shadow), texture LOD from ray
  cones, SER to bring hits of one material together; next-event estimation
  through one light-sampling interface (ReSTIR DI, a light BVH or power
  sampling) that returns a sample and its pdf for MIS against the BSDF
  sample; the reference path tracer and the real-time path share the
  material model and the light sampler.
- **Wavefront renderers** (Laine, Karras and Aila, HPG 2013; Cycles X;
  the production GPU renderers): rays, hits, shading requests and shadow
  queries are rows; hits are sorted (or binned with an atomic counter a
  bin) by material and each material's kernel runs over its own hits; a
  light-sampling stage writes shadow rays that the same traversal answers.
- **Coordinate spaces in traversal**: hardware ray tracing filters
  instances by an 8-bit instance mask against a ray cull mask (DXR
  `InstanceMask`, Vulkan `mask`), with no other per-instance test before
  any-hit; engines that need more groups than eight bits force candidates
  to any-hit for the rays that need it (Unreal's per-instance visibility
  for shadow-only and hidden-in-game primitives uses masks first).
- **G-buffer and hit payload**: Unreal's `FGBufferData` is decoded once
  from the targets and the ray-tracing material payload is encoded and
  decoded to the same struct, so lighting code takes one input; Frostbite
  and Decima pack a surface the same way for their hit lighting.

wgpu has ray queries only: no shader table, no SER. A stream's hits are
rows a pass bins by tracer and material, which is the wavefront answer.

### The design

#### A stream states the surface detail it needs

`Stream::detail` (`SurfaceDetail`): **`Diffuse`** (base colour, emission,
the shading normal, the share let through from behind, at the ray's
footprint; roughness and metalness as the material's constants; what a
probe, a radiance cache's update or a diffuse bounce lights) and **`Full`**
(the material evaluated as the G-buffer's drawers evaluate it: its template
or graph, normal maps, every `SurfaceData` field, at the ray's footprint;
reflections, a specular bounce, the reference). A tracer answers at least
the level asked and may use a cheaper kernel for a lower one; every level
writes a whole row, a lower level leaving what it does not evaluate at
`surface_data`'s neutral values. The default is `Full`: a consumer that does
not say gets the most, never less. GI's probe stream asks `Diffuse`.

Detail is the stream's, not the ray's. A tracer then picks its kernel when
the graph is built: no per-ray branch over material code (which would keep
the most expensive kernel's registers for every ray of a wave), no bit in a
row that is full. A consumer with rays of several classes writes a stream a
class (a path tracer's camera rays and its diffuse bounces), which is how
ray classes are budgeted (section 1) anyway.

#### One surface, two encodings

The one owner of "the surface a lighting model takes" was nobody: the
materials plugin's `SurfaceData` was what a template returned, each of four
G-buffer drawers (meshes, foliage, the voxel terrain's publish, virtual
geometry's shade) wrote the targets field by field from its own copy of the
layout, the deferred lighting read them back field by field, and the ray
stream's 20-byte `SceneSurface` was a third, thinner description. The
G-buffer is the frame's (Helio core owns its layout), the lighting models and
the streams' consumers read it with no material, and a kind with no material
table (a plain-coloured voxel) writes one: the frame owns it.

- **`SurfaceData`** (`helio_frame_api::gbuffer::SURFACE_SNIPPET`,
  `//!use frame_surface`) moved from the materials plugin to the frame, its
  fields and `SURFACE_FLAG_*` unchanged, so every template and graph keeps
  compiling. A template returns it; a kind without a table builds one with
  `surface_data(albedo, normal, roughness, metallic, emissive)`, a thin
  two-sided surface with `surface_with_transmission`.
- **The G-buffer's encoding**: `gbuffer_surface_texels` (encode; drawers call
  `gbuffer_output(surface, tag, motion)` or `gbuffer_depth_output`) and
  `gbuffer_surface_at` (decode; the deferred lighting's `fs_main`).
- **The ray's encoding**: `SceneSurface`, 32 bytes, `scene_surface_pack`
  (a tracer) and `scene_surface_unpack` (a consumer), every G-buffer field
  plus the face's normal a hit needs to leave its surface.

So a pixel and a hit carry the same fields and the lighting that takes a
`SurfaceData` lights either. What the G-buffer does not hold (a lighting
model a material class would need beyond these fields) a hit does not hold
either, until the G-buffer does.

#### A ray carries its coordinate space

A ray's space is a camera's encoding of the authored space slot
(`view_info.w`: the slot plus one, or 0 for every space). A ray serving a
view takes the view's (GI: `cameras[0].view_info.w`, so its probes see what
the camera draws directly); a ray leaving a hit keeps its parent's, or the
content space of a remapped pixel (`VIEW_REMAP_SPACES`). A tracer answers only
with geometry of the ray's space (`scene_ray_sees_space`).

Where the space goes was decided by measurement. The row was full (32
bytes: origin, `t_max`, direction, the cone word).

- **Directions as two 16-bit octahedral words** would free 8 bytes, but
  their worst angular error is 6.4e-5 rad (400,000 directions): 0.12 px at
  1080p and 60 degrees, 1.4 px at 4K and 10 degrees, 5.6 px at 8K and 5
  degrees, and 6.4 cm at a kilometre (a 10 cm voxel world seen from 1 km).
  The reference path tracer's camera rays need well under a pixel, and a
  long sun ray must stay in its cell. Rejected.
- **A 48-byte row** (WGSL rounds a struct holding a `vec3` to 16 bytes) was
  built and measured: GI's ray generator went from 0.114 to 0.153 ms in
  `indoor_cathedral_gi` (a 48-byte stride touches about 2.25 times the cache
  lines of a 32-byte one: three 16-byte stores a row over a wider span), the
  terrain's entry from 0.127 to 0.134 ms on 174,080 rays. Writing the
  padding as well, so each row is written whole, was slower still (0.21 ms).
  Rejected.
- **The cone word holds it**: the spread in 11 bits instead of a half
  float's 16 (its 5 exponent bits and 6 of its 10 mantissa bits: 0.8%
  relative at worst, about 0.01 of a mip in a texture or cut-out lookup,
  where the cone itself is an estimate), the caster mask in 4 bits (two
  classes are used), the space in 9 (slot 255 plus one), the width in 8 as
  before. The row stays 32 bytes, and costs what r1's did (measured below).

**The meshes** filter through the TLAS mask: an instance's mask is one bit,
its caster class (static, movable) in its space's group, `1 << (2 * group +
class)`; four groups of two classes fill eight bits. Group 0 is the world's
alone (its bits are the frame's `MASK_STATIC` / `MASK_MOVABLE`, so every
inline query of the TLAS keeps its meaning for world geometry); up to three
other traced spaces take a group each; a ray restricted to one space is
traced with that group's bits alone, culled by the hardware with no
per-candidate work. Past three other traced spaces they share groups
(`RAY_SPACE_SHARED`); a ray of a shared group's space is traced with every
candidate forced to the any-hit loop (`FORCE_NON_OPAQUE`), where a row of
another space is passed and an opaque row of its own confirmed. That is
correct at any count of spaces, and costs only those rays the any-hit loop,
never the world's or an unshared group's. Instance custom data (24 bits)
stays the row index: stealing 8 bits for the space would cap rows at 65,536.

**The voxel terrain** filters by the space each volume is authored in
(`VoxelVolume::space`, from `pulsar_scene_model::frames::space_of`; its ray
parameters carry it): a volume of another space is not walked.

**Crossing an aperture (the portals' seam)**, not built here: the portals
register a tracer whose trace pass answers a ray's nearest aperture of its
space as a hit (its own tracer id, the chain in its payload) and whose
surface pass, instead of a surface, re-emits the ray past the aperture in
place: origin and direction through the chain's drawn-to-content matrix,
the space the destination's (`scene_cone_in_space`), `t_max` what is left,
the hit row cleared, and the distance travelled to the aperture and the
chain kept in a buffer of the portals' own, a row a ray, for a consumer that
wants its hit back in its own space (the frame's remap contract,
`VIEW_REMAPS`, names the chain's matrix). `stream::trace` then needs one more thing from the frame: a stream
re-traced while any of its rays was re-emitted (an indirect count the
portals' surface pass writes, a few rounds at most), so a consumer gets the
hit beyond the aperture without knowing portals. Who owns what: the frame
owns the re-trace rounds; the portals own the crossing and the remap.

#### Direct light at a point

`helio_frame_api::rays::light_samples`, beside the streams. A consumer
declares a `LightBatch` (points: `LightQuery` rows) and a kind of light
registers a `LightSampler`; `light_samples::sample(build, &batch)` adds every
sampler's pass at that point of the graph. Each sampler offers one sample of
its own lights at each point (it chooses among them uniformly or by power;
r5 replaces the choice behind the same contract) and merges it into the row
by weighted reservoir sampling over the samplers' offers (weight: the offer's
luminance), so the row is one unbiased sample of every registered kind's
light. The consumer turns it into an occlusion-stream ray and weights what
gets through by its own BSDF: `bsdf * cos * light_sample_radiance(s) *
visibility`; `light_sample_pdf(s)` is the direction's solid-angle density,
the choices included, for MIS against its BSDF sample, 0 for a delta light.
Emissive surfaces stay emission at hits (a BSDF sample finds them; they are
no sampler yet), so nothing is counted twice; an area or emissive light kind
adds a sampler and the pdf query its BSDF hits need (r5).

#### Materials at hits: what a binned surface pass needs

r1's meshes' tracer wrote a hit's surface inside the trace pass, because
only the committed ray query held the instance's transform. Now each
`RayInstance` names its TLAS instance (`instance`) and both projections
publish every instance's object-to-frame transform beside the TLAS
(`RAY_INSTANCE_TRANSFORMS`, a `RayTransform` an instance, 48 bytes), so a
surface pass needs no ray query: the meshes' surface is a pass of its own
(`mesh_rays_surface`, the tracer's `surface` build), writing only the hits
that stayed the meshes'. A hit's payload is its row (27 bits), triangle and
barycentrics; the row gives the material (`RayInstance::material`), mesh
pools, flags and instance.

The binning key is the material's **program**: a table-driven material
class (its `material_class`, `MATERIAL_CLASS_*`, one template) or a material
graph (`MATERIAL_FLAG_HAS_CUSTOM_SHADER`, its graph's hash, the key the
drawers compile a program by: `RadiantShaderKey { template_id, graph_hash }`).
The surface pass becomes: a binning pass counting the stream's meshes hits a
program (`atomicAdd` into a bin's count, a list of row indices), an indirect
dispatch a bin running that program's kernel (the template composed into a
hit program by the materials' `RadiantTemplate::compose`, as into a drawer's),
texture LOD from the ray's cone instead of derivatives.

#### The reference path tracer

A consumer of streams and of light samples with no cache and no denoiser:
camera rays in a run from the eye (`Stream::with_run`, `SurfaceDetail::Full`,
in the camera's space), bounces as streams of their own (diffuse bounces
`Diffuse` where the reference allows, `Full` by default), next-event
estimation through `light_samples` and occlusion streams, MIS against the
BSDF sample, Russian roulette after a few bounces, accumulated over frames
while the camera holds still. A technique is judged by its error against the
reference on fixed poses.

#### What scales with what

- Rays: 32 bytes a ray, 16 a hit, 32 a surface (80 a nearest ray, written
  and read once each); cost follows the rays a consumer asks for, never the
  scene.
- Surfaces: the surface pass runs over the stream's rows once a tracer,
  writing only its own hits; the binned material passes will scale with hits
  a program, never with programs that have none.
- Spaces: the space groups cost nothing per ray up to three traced spaces
  besides the world; past that, only rays of a shared group's space pay the
  any-hit loop. Building the groups is linear in traced instances, and runs
  only when the traced set changes.
- Transforms: 48 bytes a TLAS instance, written for the instances that moved
  (all of them only when the rows are laid out again or the origin moves).
- Light samples: points times samplers (one in practice), each sampler's
  choice its own cost.

### As built

#### The contract

**Rows** (`helio_frame_api`; layout tests `tests/layouts.rs`, Rust against
WGSL, and a GPU round trip of both surface encodings, `tests/surface_gpu.rs`):

| Row | Bytes | Fields |
|---|---|---|
| `SceneRay` | 32 | `origin: vec3<f32>`, `t_max: f32`, `dir: vec3<f32>`, `cone: u32`: bits 0..11 the spread (a half float's top 11 bits), 11..15 the caster mask (`RAY_MASK_STATIC` 1, `RAY_MASK_MOVABLE` 2, `RAY_MASK_ALL` 0xf), 15..24 the space (`SCENE_SPACE_ANY` 0, or slot + 1), 24..32 the width (0 none, else `2^((bits - 160) / 8)` m) |
| `SceneHit` | 16 | unchanged |
| `SceneSurface` | 32 | `normal` (shading, `scene_pack_unit`), `face` (same), `base_color` (RGBA8: albedo, metallic), `params` (RGBA8: roughness, ao, subsurface radius as r / (1 + r), flags), `emissive` (RGB9E5), `specular` (F0 10:10:10 unorm), `subsurface` (RGB9E5: subsurface colour, or the share let through from behind with `SURFACE_FLAG_TWO_SIDED`), `anisotropy` (aniso x 8 bits, y 8, rotation 16 as a fraction of a turn); unpacks with alpha 1 |
| `LightQuery` | 32 | `position: vec3<f32>`, `width: f32` (footprint), `normal: u32` (packed), `random: u32` (two 16-bit uniforms: where on a light), `pick: u32` (uniform: the choices), `space: u32` (bits 0..16 the ray encoding, bit 31 `LIGHT_QUERY_ANY_SIDE`) |
| `LightSample` | 32 | `radiance: vec3<f32>` (incident radiance over its probability within its sampler), `distance: f32` (`LIGHT_SAMPLE_FAR` for a directional light), `dir: u32` (packed, toward the light), `pdf: f32` (solid angle within its sampler, 0 delta), `weight: f32` (sum of offers' weights, 0 no sample), `light: u32` (sampler id 28..32, the sampler's light id below) |

**Snippets** (each a `ShaderSnippet`; a shader opts in with the marker and
lists the snippet):

| Marker | Const | What |
|---|---|---|
| `//!use frame_surface` | `gbuffer::SURFACE_SNIPPET` | `SurfaceData`, `SURFACE_FLAG_*` (1 subsurface, 2 anisotropic, 4 low specular, 8 two-sided), `surface_data`, `surface_with_transmission`, `surface_transmission`, `GBufferSurfaceTexels`, `gbuffer_surface_texels`, `gbuffer_surface_at` |
| `//!use gbuffer_output` | `gbuffer::OUTPUT_SNIPPET` | adds `gbuffer_output(surface, tag, motion)` and `gbuffer_depth_output(.., depth)` (with `frame_surface`) |
| `//!use frame_ray_stream` | `rays::stream::SNIPPET` | the rays, hits, packings; `scene_ray_cone(width, spread, mask)` (every space), `scene_cone_in_space(cone, space)`, `scene_ray_space(ray)`, `SCENE_SPACE_ANY`, `scene_space_of_slot`, `scene_ray_sees_space(ray, slot)`; needs nothing else |
| `//!use frame_ray_surface` | `rays::stream::SURFACE_SNIPPET` | `SceneSurface`, `scene_surface_pack(surface, face)`, `scene_surface_unpack(row)`, `scene_surface_face(row)` (with `frame_ray_stream` and `frame_surface`) |
| `//!use frame_light_samples` | `rays::light_samples::SNIPPET` | the rows, `light_query_point`, `light_query_uniform(q, k)`, `light_query_position_uniforms`, `light_query_space`, `light_query_sees_space`, `light_query_faces`, `light_sample_of`, `light_sample_offer(row, offer, q, sampler)`, `light_sample_found`, `light_sample_dir`, `light_sample_distance`, `light_sample_radiance`, `light_sample_pdf` (with `frame_ray_stream`) |

**Rust**: `SceneRay::new(origin, t_max, dir, cone)`, `SceneRay::cone(width,
spread, mask)` (every space), `.in_space(space)`, `.space()`,
`.sees_space(slot)`, `SPACE_ANY`, `space_of_slot(slot)`, `MASK_ALL` (0x0f);
`SceneSurface` accessors (`base_color`, `roughness`, `ao`, `flags`,
`specular_f0`, `transmission`) and `surface_flags`; `Stream::with_detail`,
`SurfaceDetail::{Diffuse, Full}` (default `Full`). Light samples:
`LightBatch::new(name, capacity)` (resources `<name>_light_header`,
`_light_queries`, `_light_samples`; the header is a `StreamHeader`),
`LightBatchBuffers` (`new`, `clear`, `publish`), `LightSampler { name,
supported, sample }`, `RegisterLightSampler::light_sampler` (a feature's
prepare hook, as `RegisterRayTracer`), `offer` (a host without features),
`sampled(env)`, `sampler_names(env)`, `sample(build, &batch)`, `SamplerId`
(1-based, registration order, `MAX_SAMPLERS` 15), `LightSample::{offer,
contribution, pdf, sampler}` (the WGSL's CPU mirror, for tests).

#### Kinds as built

- **G-buffer drawers** (meshes, foliage, the voxel publish, virtual
  geometry's shade) write through `gbuffer_output` /
  `gbuffer_depth_output` / `gbuffer_surface_texels`; the deferred lighting's
  `fs_main` decodes with `gbuffer_surface_at`. Its `fs_reflection` (a second
  entry kept under the portable 16-texture limit) still reads only the
  normal, ORM and emissive targets for the specular composite, by the same
  table; HLFS reads the channels its lighting needs. The materials plugin's
  `MATERIAL_SNIPPETS` lead with the frame's surface snippet, and every
  material program opts in with `//!use frame_surface`.
- **Meshes** (`helio-mesh-api`): `RayInstance` keeps 64 bytes (`emissive`
  lost its unused fourth float for `instance: u32`); its `flags` carry the
  space slot at `RAY_INSTANCE_SPACE_SHIFT` (8, 8 bits), `in_space` /
  `space()`; `RAY_INSTANCE_TRANSFORMS` (`RayTransform { rows: [[f32; 4];
  3] }`, by TLAS instance); `RAY_SPACE_GROUPS` (256 `u32`: group bits 0..2,
  `RAY_SPACE_SHARED` 4, `RAY_SPACE_PRESENT` 8; absent: the world's alone),
  `ray_instance_mask(group, movable)`, `ray_space_groups(spaces)`; WGSL
  `ray_instance_space`, `RayTransform`, `ray_transform_point` / `_vector` /
  `_normal` (inverse transpose), `ray_space_cull(classes, space, entry)`.
  The engine world's projection (`meshes_component::ray_scene`) reads each
  owner's space, follows `SpaceSlot`'s journal, lays the groups and masks
  out with the rows, and writes the transforms of the instances that moved;
  the standalone projection (`helio_mesh_host::ray_tracing`) traces the
  world's space only and publishes the transforms. The tracer
  (`helio-pass-mesh-rays`): `mesh_rays_nearest` and `mesh_rays_occlusion`
  cull by the space groups, `mesh_rays_surface` writes the meshes' hits'
  surfaces from the transform table (`MeshRayTracer::surfaces()`, a
  `MeshRaySurfaces <stream>` pass). It still writes the row's flat colour
  and the material's constants: below `Diffuse` (no texture at the
  footprint) until the materials' evaluator runs at hits.
- **Voxel terrain**: `VoxelVolume::space` / `in_space`, `ChunkWorld::set_space`,
  the ray parameters' `volume.w` bits 8..16 (`RAY_VOLUME_SPACE_SHIFT`);
  `stream_sees` tests `scene_ray_sees_space`; `ray_surface` packs the
  publish's surface (dielectric F0 0.03, not metallic, a leaf's share of
  its colour let through). It answers `Diffuse` (the publish's shading at the
  footprint) for either level until the voxel agent's `Full` kernel.
- **GI**: the probe stream asks `SurfaceDetail::Diffuse`; probe and shadow
  rays carry the camera's space (`GiTrace.info.y`); `GILight` unpacks hits
  with `scene_surface_unpack` and lights the far side by
  `surface_transmission` (the hit's share times its colour, as before).
  `local_lights` (`gi_local_lights.wgsl`, `gi_gather_lights.wgsl`) still read
  the lights' rows: they go when the lights' sampler lands (below).

#### The three steps built against this

The briefs as written at the contract step; what each built, measured and
left open is in its own subsection below (materials at mesh hits, the
terrain's surfaces, direct light samples and the reference).

1. **Materials at mesh hits** (materials and meshes plugins): replace
   `mesh_rays_surface` with the binned material passes: bin the stream's
   meshes hits by program (`material_class`, or the graph hash for
   `MATERIAL_FLAG_HAS_CUSTOM_SHADER`, the drawers' `RadiantShaderKey`),
   compose each template or graph into a hit program over the row, the mesh
   pools (UV0, tangents) and `RAY_INSTANCE_TRANSFORMS`, sample textures at
   the ray cone's footprint (`scene_ray_footprint`), and write
   `scene_surface_pack(radiant_eval_surface(..), face)`. `Diffuse` streams
   may take the table-driven default kernel (base colour texture at the
   footprint, emission, normal); `Full` streams run every program. The
   `RayInstance` colour fields (`base_color`, `emissive`, the texture means)
   go with the flat kernel. Open for them: the foliage template writes
   `subsurface_color = albedo` (the raster lights a leaf's back with its full
   colour) where hits carry `albedo * material_transmission` (GI's far side):
   the share belongs in the template, which changes the raster image.
2. **The voxel terrain's surfaces** (voxel plugin): a `Full` kernel of
   `ray_surface` that shades a hit as the publish does (scene materials,
   textured normals, the shore's swash, the cell's occlusion), keeping the
   current one as `Diffuse`, chosen by `stream.detail`; and the carried r1
   items (the surface pass and walk cost, the sun pages under canopies).
3. **The lights' sampler and the reference path tracer** (lights plugin and
   a new consumer): register a `LightSampler` from the lights' feature
   (`RegisterLightSampler::light_sampler`) whose pass reads
   `helio_light_api::SCENE_LIGHTS` and, for each query, picks a point, spot
   or directional light that `light_query_sees_space` and
   `light_query_faces` allow (uniformly or by power), evaluates its
   radiance as the deferred lighting's `light_incidence` does (range
   falloff, spot cone, IES and light functions, the atmosphere's
   transmittance for a directional light), and merges
   `light_sample_of(radiance / choice, dir, distance, 0.0, id, light)` with
   `light_sample_offer`. Then delete GI's `local_lights.rs`,
   `gi_local_lights.wgsl` and `gi_gather_lights.wgsl` and move GILight onto a
   `LightBatch` (its sun stays the atmosphere's until the sampler takes
   directional lights through the sky frame). The reference path tracer
   consumes streams and batches as described above.

#### Measured

On an RTX 3060, the build before this change (r1 merged) against this one,
interleaved, two or three runs each; GPU medians in milliseconds.

| | r1 | r2 |
|---|---|---|
| Terrain's streams in isolation (`stream_rays_cost`, 174,080 GI-shaped rays): nearest walk | 1.279-1.292 | 1.284-1.295 |
| surfaces (20-byte rows, then 32) | 0.091-0.092 | 0.094-0.095 |
| entry | 0.126-0.127 | 0.127 |
| occlusion | 0.249-0.250 | 0.250-0.252 |
| `indoor_cathedral_gi` (180 frames to the capture): GI | 0.107-0.118 | 0.114-0.115 |
| MeshRays gi_probe | 0.110-0.117 | 0.113-0.114 |
| MeshRaySurfaces gi_probe (new) | | 0.071-0.072 |
| GILight / MeshRays gi_shadow / GIIntegrate | 0.357-0.411 / 0.044-0.050 / 0.226-0.248 | 0.377-0.393 / 0.044-0.050 / 0.243-0.245 |
| `voxel_craft` (one pose, the last six 240-frame windows' medians): GI's passes in all | 2.904-2.908 | 2.926-2.937 |
| of which VoxelRaySurfaces gi_probe / MeshRaySurfaces gi_probe | 0.234-0.235 / none | 0.241-0.242 / 0.022-0.023 |

What it costs: the meshes' surface pass of its own (0.022 ms on
voxel_craft's few mesh hits, 0.071 in the cathedral's all-mesh scene),
which the trace's latency used to hide; and 3-4% on the terrain's surface
pass for the 32-byte surface rows. The ray row costs nothing (it stayed 32
bytes). The surface pass is the one the materials agent replaces with the
binned material passes, which need it.

Images (RGB 0-1, mean |difference|): `indoor_cathedral_gi` frame 240, three
runs each: r1-r1 0.00005-0.00023 (0.00067 in an earlier pair), r2-r2
0.00004-0.00108, r1-r2 0.00008-0.00108 (the means of three against each
other 0.00040), luminance within -0.33% to +0.10%. `voxel_craft` one pose,
8-frame burst means, no clouds: r1-r1 0.00156, r2-r2 0.00159, r1-r2
0.00096-0.00165, luminance within 0.05%. Both inside run-to-run noise: this
step changes encodings, not shading.

#### Open

- **Carried from r1** (section 3), each to its owner in this phase: the
  terrain's surface pass and walk cost and the sun pages' leak under
  canopies (the voxel agent), the voxel_craft r0-r1 image difference judged
  against the reference path tracer, and an acceptance suite on a ray-query
  device (the reference's agent).
- **A pre-existing red met here, not this change's**: the transparent
  drawer's `production_pipelines_accept_coverage_and_legacy_templates` (its
  "legacy" string replacement targets a `TransparentOutput` of two fields;
  the shader returns three). The code path is not touched here. The
  deferred lighting's `each_entry_point_fits_the_portable_texture_limit`,
  red at this step, passes since the lights' step (its count is 14).
- **The raster and the hits disagree on a leaf's back**, unchanged by this
  step (images held): the foliage template's `subsurface_color` is the full
  albedo, a hit's is `albedo * material_transmission`. The materials agent
  settles it in the template (it changes the raster image).
- **Not verified**: a scene with authored spaces traced through the
  engine (no example or level has a rigid frame or portal pocket with
  traced meshes; the engine projection's groups and `SpaceSlot` journal are
  covered by compilation and the tracer's own GPU test, not by frames).
  The Earth and Moon holds and the engine's render acceptance suite, not
  run at this step, are in "Integration" below.

#### Materials at mesh hits

A mesh hit's surface is its material, evaluated at the hit as the G-buffer's
drawers evaluate it. `mesh_rays_surface` (the row's flat colour and the
material's constants) and the projections' texture means are deleted.

**The row and the programs** (`helio-mesh-api`). `RayInstance` lost
`base_color` and `emissive` (64 to 48 bytes) and names the program its
material is evaluated by (`program`): an index of the frame's `RayPrograms`,
a `PayloadContract` (`RAY_PROGRAMS`) both projections publish with the TLAS.
A program is the drawers' material key without raster flags
(`RayProgram { template, graph_hash }`); entry 0 is always the table-driven
evaluation (`RayProgram::TABLE`: the default class, no graph). The engine
projection reads each instance's resolved row and key through
`helio_mats::instance::resolved_row` / `resolved_key` (the mirror's CPU
shadow as an explicit API; it never resolves under the scene lock); the
standalone one from its object rows, and now traces graph materials.
`mesh_ray_rows` (the rows, transforms and space groups) was split from
`mesh_ray_surface` (the pools, triangles, `ray_triangle_tangent`,
`mesh_ray_normals`) so a pass reading rows alone binds no pool.

**One evaluator, lowered for compute** (`helio-mats`). A hit program is the
meshes' `HIT_PROGRAM` (`mesh_hit_surface.wgsl`) with the class's template and
the graph composed in by `RadiantTemplate::compose`, the drawers' own
composition, then lowered by `helio_mats::shader::explicit_gradients`: every
texture read of the evaluator at the gradients the stage sets
(`//!use material_gradients`, through each slot's UV transform), a graph's
own samples at the UV gradients, specular anti-aliasing's normal derivatives
as the stage's, other derivatives zero. Virtual geometry's visibility-buffer
shade had a private copy of this lowering (`gradient_sampling`); it now uses
the materials' one. A material is thus read alike on the raster, in a
visibility buffer and at a hit.

**Texture detail from the ray's cone.** The cone at the hit
(`scene_ray_footprint`) meets the triangle's plane in an ellipse; its two
axes, along the slope toward the ray and across it, each spanning the
footprint's width seen along the ray (Akenine-Moller et al., "Improved
Shader and Texture Level of Detail Using Ray Cones", JCGT 2021, the
anisotropic formulation Ray Tracing Gems II ch. 7 builds on), give the
barycentrics of their ends, hence UV0 gradients and the shading normal's
turn across the footprint. `textureSampleGrad` then picks each texture's
level by its own size and UV transform, anisotropically where the sampler
allows; a head-on cone one, four and sixteen texels wide reads mips 0, 2
and 4, and a slanted one a coarser level along its slope
(`textures_are_read_at_the_ray_cone_s_footprint`). The axes are the full
width (a pixel's derivatives are a pixel apart), so a camera ray as wide as
its pixel reads the level the raster read there.

**Detail.** `default_pbr_surface` reads the pipeline constant
`MATERIAL_SURFACE_DIFFUSE` (false for every drawer): set, it returns base
colour and emission at the footprint, the vertex normal, and the material's
constant roughness and metalness, with no normal, ORM, occlusion or specular
texture read and no specular anti-aliasing. One module a program, two
pipelines (one a `SurfaceDetail`); the compiler drops what the diffuse one
does not read. A program still runs at `Diffuse`, so a graph's base colour
is the graph's in GI.

**Binning** (`helio_pass_mesh_rays::surfaces`). Where the frame's only
program is the table's (no other present, or none compiled yet) one dispatch
over the stream's rows shades every meshes hit, the shape of the deleted
pass. Otherwise three passes bin the hits (`mesh_hit_bins.wgsl`): a count
(each hit takes its bin, `remap[program]`, and a rank, from a workgroup-local
histogram added to the bin once a workgroup, not an atomic a ray), a prefix
over the bins in one workgroup (an invocation a bin: offsets, indirect
dispatch arguments, counters cleared for the next frame), a scatter into
the list (a buffer of its own: the arguments buffer holds arguments only);
then
one indirect dispatch a bin of its program over its own hits (empty bins
dispatch nothing). This is the wavefront renderers' sort by material (Laine,
Karras and Aila 2013; Cycles X sorts by shader) and what SER does in
hardware, as bins: a hit's program is known before shading, the bins are few
and bounded, so a counting pass beats a radix sort. **Bound and fallback**:
at most `MAX_HIT_PROGRAMS` (64) bins a frame, bin 0 the table's; a program
compiling (`helio_core::compile::Compiling`, started the first frame a row
names it, never on the frame), one whose graph is not registered yet, one
that failed, and programs past the bound are evaluated by the table-driven
program (the row's factors and textures): a defined surface, never a hole.
The table-driven program is built with the graph, as the trace kernels are.

**Vertex attributes at a hit**: UV0 and the tangent frame (the vertex
tangents and the bitangent's sign, turned with a two-sided back face as the
G-buffer's fragment turns them; a skinned instance's normals and tangents
from the skin cache; a proxy's tangent from its triangle's UV directions).
Meshes carry no vertex colour (`PackedVertex` has none, and no drawer reads
one), so neither does a hit.

**Virtual-geometry proxies** (Pulsar-Native#1744). A proxy LOD keeps its
triangles grouped by material section (`helio_vg_stream::ProxySection`: the
clusters' `material_slot`, the mesh's section ordinal); the engine
projection makes each section a BLAS segment with its own row and material,
the standalone one traces the section its object row draws. A hit on a proxy
is shaded by its section's material with the proxy's UVs (textures and
normal maps included).

**The leaf's back.** What leaves the back of a thin two-sided surface lit
from the front is what it transmits: its transmission share of its colour
(`helio_mats` `material_transmitted`, `surface_with_transmission`), not its
reflectance. A leaf whose back gave off its full albedo while its front
reflects the same albedo returns up to twice the light that reaches it. The
foliage template now describes its leaf with `material_transmitted`, the
description a hit already carried; the raster and a hit agree. No shipped
scene draws a mesh with the foliage class: voxel_craft's leaves are the voxel
publish's, which describes them the same way (`chunk_draw.wgsl`,
`surface_with_transmission(surface, material_transmitted(row, albedo))`, in
the raster and at hits), its grass the foliage plugin's procedural blades
(no transmission).

**Measured** (RTX 3060, the GPU shared with two other agents' runs).

`indoor_cathedral_gi` (all-mesh, flat materials, every hit the table-driven
program, one dispatch, `Diffuse`), the build before this step (kept
executable) against this one, interleaved, three runs each, median of the
frames from 60 to the capture:

| | before (flat colour) | after (material at the hit) |
|---|---|---|
| MeshRaySurfaces / MeshHitSurfaces gi_probe | 0.0696-0.0727 | 0.0778-0.0819 |
| GI / MeshRays gi_probe / GILight / GIIntegrate | 0.115 / 0.115 / 0.383 / 0.246 | 0.118 / 0.114 / 0.378 / 0.246 |
| Image at frame 240 (RGB 0-1, mean abs difference) | base-base 0.0005-0.0010 | new-new 0.0003-0.0004; base-new 0.0002-0.0010 |
| Mean luminance | 0.1790-0.1800 | 0.1790-0.1792 |

The material evaluation costs about 0.009 ms (12 %) over the flat row on
the cathedral's 174,080 probe rays. The image does not change beyond
run-to-run noise: the cathedral's materials are untextured, so a hit's
material evaluated is the row's colour the flat path wrote; where a
texture speaks the bounce takes it (the acceptance suite's textured wall).
Images: `C:/t/r2m/cathedral_{base,new}_{1,2,3}.png`.

The binned path (`mesh_rays_gpu`'s `hit_surface_cost`, release build: 174,080
incoherent rays onto 81 textured, normal-mapped quads, one 256-square mipped
texture as base colour, normal map and ORM; the surface passes alone, 24
pipelined frames of 16 runs each, the last two runs of each frame stamped at
every pass end; a stage is the time between pass ends, since a beginning
stamp may be written before the work ahead drains; the median frame; two
interleaved runs a build, ranges across them; ms):

| Programs present (bins) | Count | Prefix | Scatter | Binning | Shading, Diffuse | Shading, Full |
|---|---|---|---|---|---|---|
| table alone (1) | - | - | - | - | 0.084-0.090 | 0.119-0.130 |
| table, an unused program present: before (2) | 0.030 | 0.010 | 0.020 | 0.060 | 0.096 | 0.127 |
| the same, after (1) | - | - | - | - | 0.084-0.085 | 0.119-0.120 |
| 64 present, 2 with hits, after (2) | 0.030 | 0.011 | 0.021 | 0.061 | 0.130 | 0.162 |
| 2 programs (2) | 0.030 | 0.010 | 0.020 | 0.060 | 0.130 | 0.162 |
| 8 programs (8) | 0.032-0.042 | 0.012-0.015 | 0.025-0.031 | 0.069-0.088 | 0.30-0.42 | 0.47-0.61 |
| 64 programs (64) | 0.19-0.20 | 0.034-0.038 | 0.10-0.11 | 0.33-0.35 | 3.58-3.70 | 3.81-4.04 |

The figures first published here (2 to 8.7 ms) came from a debug test
binary with the GPU idle between frames: they were the harness's.

**What the time is.** Binning is 0.06 ms for up to a few programs and
grows to about 0.34 ms at 64 bins (the scatter's writes spread over 64
lists; the count's rise is not contention on the bin counters: 32
counters a bin changed nothing and were dropped). The shading is not:
each bin with hits costs about the same whatever its size. Timed in a pass
of its own, a bin takes 74-95 us (`Diffuse`) and 79-107 us (`Full`) with
20,000 hits (8 programs) and with 4,000 (64): a dispatch's time here is its
kernel's latency, not its work (the whole stream, 174,080 hits, is
0.085 ms in one dispatch). Bin dispatches cannot overlap: each writes the
stream's surfaces (read-write storage), and wgpu puts a barrier between
two dispatches that use one buffer read-write, a full compute drain on
Vulkan; and each indirect dispatch is preceded by wgpu's indirect-call
validation (a dispatch of its own and two barriers), which alone is about a
quarter of it (64 programs with the validation off: 2.8 ms `Diffuse`).
Distinct pipelines cost nothing measurable (64 bins on one pipeline: the
same 3.6 ms). So the cost per extra program with hits is a fixed
0.03-0.06 ms on this GPU, about one kernel latency, and the total follows
the programs with hits, not the hits.

**What changed.** A program whose hits were counted zero takes no bin: the
bins' counts are read back without waiting (three slots in flight) and
read at the next `set_programs`, so a program present in the frame but
hit by none of the stream's rays costs nothing after a frame or two
(before, 0.06 ms of binning and 0.012 ms of the binned list's indirection
for the table's hits: the second and third rows), and 64 programs present with 2 hit cost what 2
programs do. The binning pipelines skip workgroup zeroing (every variable
is written before it is read; not measurable here). **Diffuse against
Full** on these textured, normal-mapped materials: 0.084-0.090 against
0.119-0.130 ms for the table's evaluation over the whole stream (about 30 %
less), 0.130 against 0.162 for 2 programs; past 8 programs both are the
bins' latencies.

**The fixed cost's structural fixes**, which are not this step's: bins
that overlap. The wavefront renderers' per-material dispatches run back to
back without barriers between them (on D3D12 and Vulkan an application
omits the barrier between dispatches writing disjoint rows); wgpu cannot
say that. Either the wgpu fork gains a pass-level way to issue dispatches
writing disjoint ranges of one buffer without barriers and to skip
indirect validation for arguments a GPU pass computed (an engine-wide
decision: every Helio pass would gain), or the programs present compile
into one pipeline that switches on the bin (Falcor's material dispatch:
one dispatch, the cost follows the hits, but a module recompiled whenever
the set of programs changes and registers at the widest program's).

**Acceptance on the engine path** (`installed_plugins`' `ray_tier_acceptance`,
the editor's headless renderer, the shipped deferred graph, a ray-query
device; skipped without one): GI's bounce on the ground before a wall drawn
with a graph sampling a red texture is red (0.193, 0.010, 0.0), before a
procedural graph's green wall green (0.049, 0.233, 0.023), before a grey
wall grey (0.115, 0.143, 0.111), with three programs in three bins; a
cut-out card shadows the ground by its leaves (0.05) and not its holes
(0.42). The engine's graph materials carry their own class
(`MATERIAL_CLASS_CUSTOM`), which no template registry holds: a graph program
of that class composes over the default class's template, as the drawers
do (before that, a graph's hits took the table's evaluation of its fallback
row, which the first runs of this suite caught). `skinned_frames` now
asserts and runs in local gates (rest against bent 1.84, bent after rest
against bent fresh 0.19, tracing the tube 0.48 and the tree 0.38, 8-bit).
The one-surface test is the meshes' `a_hit_s_surface_is_the_g_buffer_s_at_the_same_pixel`
(a standalone Helio host: the deferred graph's G-buffer decoded with
`gbuffer_surface_at` against camera rays through the same pixels, a `Full`
stream, two programs binned): base colour, metalness and roughness within
2-3/255, normals within 0.01 rad, emission within 2 %, over 82 floor, 58
normal-mapped metallic wall and 9 emissive samples. It runs on the meshes'
standalone host because the engine registers no test feature in its graph.

**Open**

- **The binned path's fixed cost per program** (above): understood, 0.03-0.06
  ms a program with hits on an RTX 3060, from barriers that serialise the
  bins; the fix is bins that overlap (a wgpu fork change or one pipeline
  over the programs present), a decision beyond this step. A frame pays it
  only for programs its rays hit; the table-only frame is one dispatch.
- **Deforming static meshes in the engine TLAS** (Pulsar-Native#1744's last
  item) are not part of this step.

#### The terrain's surfaces

The voxel plugin's tracer (`chunks::rays`, chunk_draw.wgsl 8, the plugin's
`docs/chunk-octree.md`, "Ray streams") answers both `SurfaceDetail` levels,
picked per stream when the graph builds (`StreamTracer::with_detail`), and
the sun's occlusion rays at their footprint.

- **One shading of a face** (`voxel_surface`). The G-buffer's publish
  calls it at a pixel's footprint and multiplies in the screen's sky
  samples; a `Full` stream's surface pass (`ray_surface_full`) calls it at
  the ray's footprint, the material boundaries' jitter keyed by the ray's
  row as a pixel's is by the pixel. Materials and their patches, per-voxel
  pigment, corner occlusion and edges, the host's scene materials, the
  shore's swash, emission, a leaf's share let through: there is no second
  copy of the terrain's material code. `full_stream_surfaces_are_the_gbuffer_texels`
  (as the contract's test does for meshes): every second pixel of a view of
  the ground and a glowing, translucent brick ball, traced as a ray from
  the eye with the pixel's footprint, decodes to the surface the publish
  wrote there in every field within the encodings' precision: 0 of 4,649
  differ, worst normal cosine 0.9999999; with a block of leaves added, 0 of
  4,746, its 97 leaf texels and 801 two-sided ones letting through their
  transmission share of their colour (`material_transmission` times the
  albedo, the materials' rule), never their whole colour. Two of the publish's inputs a hit
  did not have moved into the surface: a natural step lights with its own
  stored slope before the one a neighbouring shading sample lends (34 of
  the 4,649 normals were up to 4 degrees off; `face_rule`'s comment said
  stored first), and a leaf's share let through is in its surface
  (`subsurface_color = albedo * transmission`) instead of in the shading
  pass's sun visibility of a back-lit leaf: the same sun light, and local
  lights and GI behind a leaf now take the share too.
- **`Diffuse`** (`ray_surface`, GI's probes): the material's colour,
  patches and clumps at the footprint from the cell's centre, no pigment,
  corner occlusion or textures (at GI's footprints the publish fades those
  out too), the face rule's normal, emission and the share let through.
  Inline or a pass, by measurement: written by the walk as it takes each
  hit (the contract allows it; its wording now says so, below), the walk's
  kernel took 1.553 ms on GI's 174,080 rays in isolation against 1.327 +
  0.094 for walk and pass, and 2.01-2.06 in voxel_craft against 1.61-1.62
  + 0.24: the shading's registers slow every ray of the walk. It stays a
  pass. `Full`'s pass costs 0.176 ms on the same rays.
- **The sun's rays at their footprint** (`paged_occlusion`).
  `sun_pages_under_a_canopy` (a forest planet, 2,573 receivers shaped as
  GI's probe-ray hits, each against the walk along the epoch's sun, every
  level 0..12 about the eye) found the cause of r1's 3-8 % brightening:
  the pages hold the right occluder (each receiver's nearest texel's centre
  ray, walked, meets what the texel stores at 99.8-100 % of texels on every
  level), and every receiver the pages light that the walk shades lies
  within the read's bias of its occluder: a texel, plus the receiver's
  slope across the 2x2 filter. The bias keeps a surface from shadowing
  itself and fits a receiver whose own surface is the texel's size (the
  shading pass reads the pages for cells under a pixel and a half, at the
  pixel's footprint); from a GI hit, finer than its footprint, it hides
  every leaf less than a texel over it: the pages lit 0.75 (level 3), 0.80
  (4), 0.89 (5) of the receivers where the walk lit 0.61. The pages were
  not wrong; reading them from the surface at a texel's bias was. An
  occlusion ray along the sun at least a finest texel wide, seeing every
  caster and every volume the pages hold, now walks the first two texels of
  the finest level resident at its origin up to the one whose texel is half
  its width (so the read's 2x2 filter spans the ray; section 4) in each
  volume and takes what the pages let through from a texel up the sun on:
  the band the bias hides was walked, and the receiver's own surface
  (slopes to 1.3 over the filter) stays its own. On the same receivers it
  lets through 0.600 where the walk does 0.607 (59 of 2,573 differ, every
  level within 0.02). Decided by the ray's width and page residency, never
  by the consumer. Rays ask for no pages (asked for at GI's footprints the
  page tracer cost 0.19 ms a frame on voxel_craft, where it costs 0.01);
  the pixels' pages and the always-resident ones about the eye answer
  them.
- **The walk's cost in the frame.** In voxel_craft the probe walk with its
  entry takes 1.61-1.64 ms, as r1's (1.61-1.72 in the same runs); in
  isolation (a ball on bare ground) 1.33. Not lost overlap: a stream's pass
  runs between barriers, so its own timestamps time its work alone, and
  overlap lost with neighbours would show in the frame's total, not in the
  pass. Not cache state after the G-buffer or a first dispatch's cost: the
  same rays walked a second time right after (a measurement build, rows
  already holding their hits, so the same cells walked to the same hit)
  took 1.64-1.75 against 1.65-1.68 for the first, and the isolated walk
  costs the same recorded before or after the draw (1.327 / 1.327). The
  remainder is the work of the frame's rays: voxel_craft's probes sit among
  trees and walls whose cells the bare-ground benchmark does not have. r0's
  in-draw kernel on voxel_craft took 1.78-1.82 ms; the stream walk takes
  less there. Open: the planets' (below).
- **Coordinate spaces**: verified against the skill's rules.
  `stream_rays_see_only_volumes_of_their_space`: two blocks at the same
  place in authored spaces 2 and 3 over the world's ground; the world's
  rays meet only the ground (through both blocks), each space's only its
  own block, and a direct ray (every space) the nearest of all three, each
  against CPU casts over the volumes it may see. The paged answer serves a
  ray only where the pages hold volumes of its space alone (or it sees
  every space). Each volume's dispatch still runs for a stream of another
  space (its rays leave at the space test; a nearest stream's entry writes
  a miss row a ray, 64 bytes): no walk, but rays x volumes threads. Open:
  the stream's space known on the CPU would let a volume of another space
  skip its dispatch (the consumer knows it, GI's `GiTrace.info.y`; the
  contract has no CPU-side per-frame field for it).

Costs, RTX 3060, GPU medians over pipelined frames (voxel_craft: one pose,
the last six 240-frame windows with the terrain drawn; runs alternated with
r1's final exe):

| | r1 final | r2 |
|---|---|---|
| voxel_craft, GI's passes in all | 2.92-2.96 | 2.73-2.77 |
| of which VoxelRays gi_probe / VoxelRaySurfaces gi_probe | 1.61-1.64 / 0.233-0.237 | 1.61-1.62 / 0.239-0.246 |
| VoxelRays gi_shadow | 0.409-0.422 | 0.202-0.225 |
| VoxelChunks (its sun occlusion pages) | 4.20-4.26 (0.013) | 4.22-4.36 (0.02) |
| `stream_rays_cost`: nearest with entry / Diffuse / Full surfaces | 1.33 / 0.094 / none | 1.327 / 0.095 / 0.176 |
| its sun rays (45,466 from the probe rays' hits): walked / at the footprint | 0.60 / none | 0.60 / 0.14-0.15 |

voxel_flight ground holds (p50 over the hold; r0's GI rays in the draw are
its VoxelChunks less r2's, the rest of that pass being the same work):

| | r0 | r1 closing | r2 |
|---|---|---|---|
| Earth (r0 three runs, r2 three) | rays in the draw 1.06-1.10, GI 0.28-0.29: 1.35-1.39 | 2.46 | GI 0.06-0.07, VoxelRays gi_probe 1.56-1.58, surfaces 0.17, GILight 0.25-0.26, VoxelRays gi_shadow 0.21-0.22, GIIntegrate 0.13: 2.38-2.43 |
| Moon (r0 three runs, r2 three) | rays in the draw 0.84-0.96, GI 0.26: 1.10-1.22 | 2.54 | GI 0.06-0.07, VoxelRays gi_probe 1.57-1.63, surfaces 0.13-0.14, GILight 0.22-0.23, VoxelRays gi_shadow 0.23-0.24, GIIntegrate 0.13: 2.34-2.41 |

On voxel_craft the terrain's GI is 0.15-0.2 ms under r1's, all of it the
sun's rays (0.41 -> 0.21); on the planets 0.03-0.2 under r1, and still
about 1.0-1.2 ms over r0, the probe walk (1.56-1.63 against r0's in-draw
kernel's 0.84-1.10 with its shading) being the remainder (open, below).
The holds' images were not compared.

Images (voxel_craft, four poses, 8-frame burst means, no clouds, RGB 0-1
mean |difference|): r1-r1 0.0003-0.0012; r1-r2 0.0004-0.0007 and
0.0007-0.0015 against the two r1 runs, luminance -0.05 to -0.16 %: inside
run-to-run noise, slightly darker (the footprint answer keeps the shade the
walk gives; the leaf share now in the surface). r2-r2 and the Earth and
Moon hold images were not compared (the shared GPU was held by the other
sessions' runs).

The **contract** changed in one doc comment (`Tracer::surface`): a tracer
may write a hit's surface in its trace pass as it takes the hit, adding no
surface pass; a tracer that takes the row later writes its own.

Open:
- The planets' probe walk against r0's in-draw kernel (r1: 1.44-1.53 ms
  against 1.05): measured here only on voxel_craft, where the stream walk
  is cheaper than r0's. The planets' probe walk is 1.57-1.58 ms in the frame where
  r0's in-draw kernel cost 0.95-1.21 with its shading and sun lookup; the
  same proof as on voxel_craft (pass timestamps between barriers, a second
  identical walk costing the same) was run on voxel_craft only. r0's kernel
  walked from cells its CPU placed and shared the draw's registers'
  footprint; the stream walk pays a placed entry a ray (0.19 ms on GI's
  rays in isolation) and a start row's read. A ray-to-volume binning and r3's
  radiance cache take probe rays off the walk.
- Scene materials at `Full` hits: `voxel_surface` reads them, but the ray
  kernels are compiled from the draw source without the scene-material
  splice (no host maps voxel materials to scene materials today:
  `ChunkPass::set_scene_materials` has no caller); a host that does gets
  the appearance table's colours at hits until the ray kernels take the
  scene variant and its groups 2 and 3.
- A volume of another space still dispatches over a stream (above).
- **Startup** (closed, not this change's): the time to a settled test view
  (`publish_shows_hits_only_in_front_of_meshes`, wall time, holding the
  build slot and the GPU lock) is the NVIDIA driver's shader cache, keyed by
  the executable: the contract build's test exe at its old path 4.5 s then
  2.7 s; the same file copied to a new path 72.6, 63.7 and 31.0 s over three
  runs; this build's 58.0, 33.7 and 45.9 s. The warm runs' slow pipelines
  include generation kernels this change does not touch
  (`feature_block_candidates_main` 17.5 s, `chunk_columns_main` 4.7 s), and
  the draw module grew 6 % (`chunk_draw.wgsl` 239,327 to 253,795 bytes, 38
  to 39 entry points, the composed module 327,840 bytes; one module a form,
  as before). The cache (1.1 GB, at its limit with several sessions
  compiling) keeps a new executable's pipelines only partly from run to
  run.
- Fog reads the pages with a texel's bias from points in the air
  (`sun_occlusion_point`, the lights plugin's): under a canopy it takes
  sun within a texel of leaves; a point in the air could read with none.
  The fog's and the lights' owners decide.

#### Direct light samples and the reference

**The lights' sampler** (the lights plugin, `helio-pass-light-sampler`,
registered with `RegisterLightSampler::light_sampler` from the lights'
feature set). How others choose a light at a point: Unreal's path tracer
keeps a world-space light grid of per-cell light lists and picks a light in
the cell by its estimated contribution; RTXDI resamples candidates from a
world-space grid (ReGIR) or a power-proportional alias table; PBRT-v4 and
Cycles walk a light BVH with a bounded-importance traversal (Conty and
Kulla 2018). The screen-tile lists the lights already build (the deferred
lighting's binning, HLFS's grids) cover what the camera sees and nothing
else: a bounce, a probe or a reflection's hit is off the screen as often as
on it, so the sampler builds its own world-space structure before the first
batch a graph samples, again only when the light list's contents or size,
the eye's finest cell or the frame's origin change (indoor_cathedral_gi with
its lights untouched: 0.003 ms, against 0.06 a rebuild):

- **The grid**: six nested levels of 32^3 cubic cells about the eye, level
  `l`'s cells `2^l` m across (32 m to 1 km wide). A local light lives in the
  finest level whose cells are at least its range wide and whose box holds
  its sphere (at most 27 cells, so the entries are bounded by 27 a light
  row and never overflow); directional lights and local lights no level
  holds are the globals. Three dispatches: count, allocate (an atomic
  offset a cell), fill. 1.5 MiB of cells plus 108 bytes a light row.
- **The choice**: a point weighs the globals and the lights of the cell
  holding it at each level (each light once), by `light_arrival` (the
  falloff and cone) times the cosine to its normal, a light its space does
  not see weighing 0, and keeps one by weighted reservoir sampling. The
  kept light is then evaluated in full by `light_incidence` (its IES
  profile and light function, a sun's air), and a sun by the media's
  transmittance (`sun_transmittance`, the clouds'), and offered as
  `radiance / (weight / total)`. The bound used for the choice is at least
  the full evaluation (profiles, light functions, air and media only take
  light away), so the estimate is unbiased; every light here is a delta
  (point, spot, directional), so the density is 0. The choice is one
  kernel (`light_sample`), which r5's resampled choice replaces behind the
  same rows.
- **One owner of a light's evaluation**: `light_arrival` (in `light_list`)
  and `light_incidence` (`//!use light_incidence`) moved out of the
  deferred lighting; the deferred lighting, the forward function
  (`light_forward`, which had its own copy of the falloff) and the sampler
  call them. The IES profiles and their stand-in moved to the lights' api
  (`IesProfiles`, `ies_stand_in`). Spaces match as the deferred lighting
  matches them: a light of slot `s` reaches a query whose space is
  `SCENE_SPACE_ANY` or `s + 1` (`light_query_sees_space`).
- **What scales with what**: the build is linear in the light list's rows
  and the grid's cells, once however many batches ask (and a member only
  while one of them can be: a batch added under a member gate leaves the
  grid with it); a point
  costs the lights that can reach it (the globals and one cell a level),
  not every light. What the grid does not bound: many lights whose ranges
  exceed a kilometre, or lights far from the eye, which are globals every
  point weighs.

**GI on the contract.** GI's light pass writes a light query a hit into its
batch (`light_batch()`, `gi_probe`), the samplers answer, and a shadow pass
(`GIShadow`) turns the samples into the shadow stream; `local_lights.rs`,
`gi_local_lights.wgsl`, `gi_gather_lights.wgsl`, `gi_sun` and GI's reading of
`scene_lights` and `directional_lights` are gone: GI depends on no light
kind. The sun is one more answer of the sampler. Why: GI traced a sun
shadow ray at every hit and a second toward one local light; one sample
chosen by contribution traces the sun where it dominates and a lamp where it
does, one shadow ray a hit (`gi_shadow` holds `MAX_RAYS` rows, half its
size), and the sun is lit exactly as the deferred lighting lights it (its
air at the hit, not at the eye).

**The BSDF as a contract.** How a surface returns light belongs with the
surface (`helio_frame_api::gbuffer::BSDF_SNIPPET`, `//!use frame_bsdf`,
beside `SurfaceData`): a `SurfaceData`'s fields mean something only under one
reflectance model, and every reader of it (the lighting models over the
G-buffer, the ray streams' consumers, the reference, the materials'
shading-cost probe) lights it with that model. A lighting model owns how
light arrives (its lights, shadows, indirect inputs), not how the surface
returns it, so two models over one G-buffer cannot disagree about a surface
and the reference judges the real-time image with the same reflectance.
`bsdf_eval(s, wo, wi)` (the BSDF times the cosine), `bsdf_pdf`,
`bsdf_sample(s, wo, u)`, `bsdf_prepare` / `bsdf_eval_prepared` (the
surface's frame made once where many lights light it), `bsdf_facing` (a
two-sided surface turned toward the side a hit sees, as a drawer turns it by
`front_facing`), and the split sum an environment is weighted by
(`bsdf_environment_split_sum`, `bsdf_fresnel_roughness`).

- **The model is the one the engine lit with**: `bsdf_eval` is the deleted
  `pbr_eval.wgsl` and the deferred lighting's `pbr_direct_light`, term for
  term: GGX with alpha = roughness^2 and its `+1e-4`, Smith-Schlick masking
  with Karis's remapping `k = (roughness + 1)^2 / 8`, `D G F / (4 n.wo n.wi +
  1e-4)`, the anisotropic distribution and masking of
  `roughness_aniso_x/_y`, Schlick's Fresnel at `wo.h`, Lambert weighted by
  `(1 - F)(1 - metallic)`, the wrapped subsurface lobe mixed with
  `subsurface_color / pi`, a thin two-sided surface's back passing
  `subsurface_color / pi`. A first version of this contract replaced the
  model (Smith's height-correlated G2, an F90 rule, a flip of two-sided
  normals); moving the evaluation must not change it, so it was put back
  (the table below measures each term).
- **Changed only where it gained energy** (the furnace, below): (1) the
  grazing reflectance is 0 under 2% F0 (Unreal's `F_Schlick`; F0 0 gained
  4.8% from a grazing specular lobe on a full diffuse one); (2) the diffuse
  lobe is scaled by `1 - bsdf_diffuse_loss`, a bound measured by quadrature
  of this model on what `1 - F(wo.h)` over-counts (1.8% gained at F0 4%,
  3.4% at 2%, roughness 0.3, n.wo 0.2; zero at normal views, smooth and
  rough surfaces and F0 past 7%); (3) the wrapped lobe is normalised (it
  returned `(1 + 2w) / (1 + w)`, up to 37% more), its specular part keeps
  its own cosine and the scattered light is what the specular lobe leaves.
  The principled replacement of (2) is a multiple-scattering GGX with its
  directional albedo tabulated (Kulla and Conty 2017; Filament's energy
  compensation): a change of the engine's look, not of where it is
  evaluated, so not here.
- **Sampling**: the diffuse and through lobes by the cosine, the specular
  lobe by GGX's visible normals (Dupuy and Benyoub 2023, alpha floored at
  0.045^2), lobes drawn in proportion to an estimate of what each reflects,
  every density the mixture's and every weight `bsdf_eval / bsdf_pdf`: an
  estimator is unbiased for whatever `f` is evaluated, so the sampler need
  not be the model.

How each term changed voxel_craft's image against r1's merged build (four
poses, a burst of four frames each, clouds off; mean |difference| of RGB 0-1
and luminance ratio; terms put back one at a time from the first version,
from disk with `HELIO_SHADER_FROM_DISK`):

| step | mean \|d\| | luminance |
|---|---|---|
| r1 against r1 (noise) | 0.0003 | 1.0003 |
| r2's base build against r1 | 0.0007 | 1.0000 |
| first version (exact Smith G2, F90 rule, two-sided flip, no faint-light cut) | 0.0022 | 1.0028 |
| + the old Fresnel (F90 1) | 0.0020 | 1.0032 |
| + the old specular and diffuse terms (Smith-Schlick, the epsilons) | 0.0003 | 1.0004 |
| + no two-sided flip | 0.0008 | 1.0004 |
| + the faint-light cut (all old terms) | 0.0006 | 1.0004 |
| the three energy fixes on the old terms (from disk) | 0.0003 | 1.0002 |
| as built (release) | 0.0009 | 1.0000 |

So the masking term carried the change (the height-correlated G2 against the
remapped Smith-Schlick), the F90 rule none (voxel_craft has no F0 under 2%),
the flip and the cut nothing measurable. The +10.7% reported before compared
a kept r1 capture with a fresh run of another setup; interleaved and alike,
the first version was +0.28%. indoor_cathedral_gi (frame 240, two runs
each): before-before 0.0005, after-after 0.0007, before-after 0.0008-0.0009,
luminance +0.3% (the chandeliers flicker by wall time).

The cost: `DeferredLight` took 0.58-0.62 ms in indoor_cathedral_gi against
0.50 before. Not the math (the old shader run from disk in this build: 0.49-
0.52; the old formula inline in the new shader: 0.51) but the anisotropic
frame `bsdf_eval` built a light (with the anisotropic branch removed: 0.50);
`bsdf_prepare` builds it once a pixel: 0.50-0.52 ms as built, against
0.49-0.53 for r2's base build.

**The reference path tracer** (`helio-pass-reference`, a Helio pass crate in
the feature layer: it needs only frame contracts, so it lives below every
plugin and every host building Helio's presets has it). A wavefront path
tracer over the streams and light batches, knowing no kind:

- **A frame** traces a path for each pixel of a moving window of the image
  (`ReferenceSettings::paths_per_frame`, 2^19 by default), as unrolled
  stages (`max_depth`, 6 by default): camera rays from the eye as one run of
  the depth-0 stream (`Stream::with_run(capacity)`); at each depth the
  tracers' nearest hits and `Full` surfaces, `ReferenceShade` (emission, or
  the sky on a miss; a light query a hit; the BSDF's sample, Russian
  roulette from depth 3 by the throughput, survivors compacted into the
  next depth's stream), the samplers, `ReferenceShadow` (the sample times
  `bsdf_eval`, weighted against `bsdf_pdf` by the power heuristic, a delta
  light's weight 1; a shadow ray), the tracers' occlusion, and
  `ReferenceResolve`. Two nearest streams, one batch and one shadow stream
  serve every depth (the stages alternate them). `ReferenceAccumulate` sums
  each pixel's paths in 32-bit floats (a path that came back not finite is
  counted as no light); no cache, no denoiser, no clamp. A frame any tracer
  answered short of (`stream::Answers`: geometry it could not trace yet) is
  traced but not summed.
- **The sky** on a miss is the atmosphere's sky table back to radiance
  (GI's probe rays take it the same way); the sun's disc is not in it, the
  sun being a delta light the sampler samples, so it is counted once.
  Emissive surfaces are emission at hits (no sampler offers them), weight 1.
- **Ray cones**: a camera ray starts with no width and a pixel's spread; a
  ray leaving a hit is as wide as the footprint that found it (the stream
  contract's), a bounce spreading by its lobe, twice the GGX alpha for a
  specular sample and a 90-degree cone for a diffuse one (Akenine-Moller et
  al., RTG II, 2021), a shadow ray with no spread.
- **The black terrain was not the rays' width.** A first version took
  voxel_craft's terrain going black under wide shadow rays for a tracer
  fault and gave the reference's rays zero width. It was the reference
  summing frames the terrain had not answered: the terrain's draw and ray
  kernels compile off the frame (`helio_core::compile`, which now logs each
  bundle's wait and build time: on this machine with a cold driver cache
  the chunk draw pipelines take 64 s and the ray stream pipelines 38 s, so
  about 3,800 frames of voxel_craft), and until they have, the terrain
  answers no stream (`Answers::short`). The first version had no answers
  check, so at 1,500 frames most of its sum was frames without the
  terrain. The single-change runs that seemed to prove the width (zero
  width, a 0.25 m offset, foliage off: lit) ran from shader sources on
  disk after earlier runs, most likely on a warm driver cache with the
  kernels ready within their 300 frames (the same executable run fresh for
  300 frames shows no terrain light at all; not proven further). Proof:
  the same pre-merge executable, unchanged (footprint-wide shadow rays,
  foliage on), summed for 6,000 frames lights the terrain with the trees'
  shadows; the merged build with footprint-wide rays does too; and the
  terrain's tracer answers wide occlusion rays leaving the ground as the
  exact walk does (`wide_occlusion_rays_leave_the_ground` in the voxel
  plugin's `tests/chunks.rs`). The reference now skips frames a tracer
  answered short of, and its rays are back on the contract's widths.
- **What a width does in the tracers**: the meshes' tracer tests cut-outs
  at the footprint's mip; the terrain's answers an occlusion ray at least a
  sun-page texel wide from the sun's pages past a short walked band
  (`paged_occlusion`). The test above (a forest, the sun 50 degrees up, a
  solid 6 m roof added; ground points under open sky and under cover, each
  judged by nine zero-width walks within half a metre; rays of 0, 1 cm,
  10 cm and 1 m) finds no width fault: walked widths answer exactly, and
  the pages' answers for wider rays are off at 1 of 164 open and 3 of 159
  covered points at 1 m (a page texel is a point sample; the test allows
  one in fifty). One improvement there: the pages were read at the level
  whose texel is the ray's width, so the read's 2x2 filter spanned twice
  the ray; read at half the width (`chunk_draw.wgsl`, `paged_occlusion`)
  it spans the ray, and 10 cm rays went from 1 open point stopped and 1
  covered let through to none (1 m unchanged). Judged by five walks (the
  canopies' gappy cover counted as cover), metre-wide rays came back lit at
  9 of 87 covered points: a metre-wide ray under a canopy with gaps is
  partly lit, and a point-sampled answer reads one side or the other. GI's
  shadow rays carry the probe ray's footprint (0.443 times the hit's
  distance); voxel_craft's GI indirect view (23, four poses) before and
  after the half-width reads: mean |difference| 0.0003, luminance 0.9999
  (run-to-run noise); GI's shadow rays of zero width against the
  footprint's: 0.0005, luminance +0.45%. GI's light did not change.
- **Ray offsets**: a reference ray starts `0.002 + 4e-6 |p| + footprint /
  100` off the surface it leaves, along the hit's face (the terrain takes an
  origin within 1/1024 of a cell of a solid face for an inside hit).
- **Restart**: the sum restarts when the camera (the renderer's
  `camera_generation`, which the per-frame jitter does not move; a key of
  the unjittered matrices rounded differently every frame and restarted
  voxel_craft's sum each frame), the frame's origin, the image size or the
  scene's buffers (`SceneBufferProjection::content_signature`) change. A
  host that rewrites rows unchanged every frame restarts it too (a write
  marks a row changed even when equal): the voxel examples now leave an
  unchanged light, object and interactor alone.
  There is no frame-level history-invalidation seam in Helio; this is what
  the frame offers, and it does not see a kind's state outside SceneDB (a
  voxel edit, a TLAS payload by itself).
- **Only while shown**: `RenderGraph::set_member_gate` (helio-core) puts
  the passes added under a gate out of the schedule while the gate is
  closed. The reference's node adds its passes, its streams' tracer passes
  (`stream::trace`) and its batches' sampler passes
  (`light_samples::sample`) under one gate, open while debug view 62
  (`REFERENCE_VIEW`, the image) or 63 (`REFERENCE_DIFFUSE_VIEW`, the light
  that reached the camera's hit by a first bounce off a diffuse lobe) is
  shown; its buffers are freed when it leaves. The light grid, shared by
  every batch, is a member while any of its batches can be. indoor_cathedral_gi
  with the reference off: 35 passes scheduled and 55 dormant, against 80
  scheduled in the first version; the 45 that left are the reference's own
  39 (each idle at the timer's 0.003 ms floor before) and its batches' 6
  sampler passes; the GI passes' and `DeferredLight`'s timings
  are those of r2's base build (table below). `ReferenceDisplay::image`
  gives a host or test the sums (`vec4` a pixel, rgb and count) and, once
  asked for (`request_snapshot`), the real-time scene colour at the same
  pixels.
- **Settings**: `ReferenceSettings` (resolution scale, paths a frame,
  bounces), described for hosts as `helio_pass_reference::REFERENCE`.
- **Runs**: the camera stream is the first real stream in runs. The
  terrain's run entry places a run's origin with one thread and reads a
  start row a run (`rows / run` rows), nothing sized by the run's length, so
  a run of 2^19 rays costs one thread; nothing needed changing.

**Verified** (RTX 3060):

- The BSDF (`helio-frame-api` `tests/bsdf_gpu.rs`, 2M directions a case,
  twelve surfaces at n.wo 1, 0.5, 0.2 and 0.05, against the function the
  deferred lighting calls): the evaluated integral and the sampled estimate
  agree within five standard errors everywhere; a white Lambertian surface
  (F0 0) reflects 1.0000; white dielectrics at F0 4% and 2% and roughness
  0.3 and 0.45 (where the old model gained most) 0.971 to 0.998, a white
  rough dielectric 0.978 to 0.982, a white subsurface surface 0.977 to
  0.995, white metals 0.16 to 0.85 (single-scattering GGX with Smith-Schlick
  loses energy); `bsdf_pdf` integrates to 1 within its standard error. The
  old model, by quadrature (`C:/t/r2d/ab/need.py`): a white dielectric of F0
  4% at roughness 0.3 and n.wo 0.2 returned 1.018, of F0 0 up to 1.048, the
  wrapped lobe up to 1.375.
- The sampler (`tests/sampler_gpu.rs`): over 48 point lights of ranges
  0.6 m to 70 m, a spot, a global 900 m light, two suns and dead rows, at
  points inside the grid, outside its finest levels and outside every
  level, 8192 samples a point: the mean of `light_sample_radiance *
  cos` is the brute-force sum of `light_incidence * cos` over every light
  within five standard errors (plus 1e-4 for f32 and the row's packing);
  every density 0; a light behind a one-sided point never chosen; a point
  out of every light's range gets no sample; a point of one coordinate
  space lit only by its space's lights, a point of every space by all.
- The reference (`tests/reference_gpu.rs`, the tests' own analytic tracer
  and sampler, 32x32): the furnace (a closed room of albedo 0.502, every
  wall emitting 1) at depths 1, 2, 3: 1, 1.50195, 1.75391 against
  1, 1.50196, 1.75393; at depth 12 with Russian roulette 2.0065 against
  2.0074 (256 paths a pixel); direct light from a point light and a sun on
  Lambertian ground, one surface a path: mean relative error 0.05%, worst
  pixel 0.23% (512 paths a pixel); the furnace's RMS error at 16 and 64
  paths a pixel 0.507 and 0.261, ratio 1.94 (1/sqrt(N): 2). Two faults the
  tests found and this fixes: a WGSL `vec3` field padded the owners' row to
  48 bytes in a 32-byte buffer (a third of the light samples were written
  out of bounds), and a stage bound its count buffer for writing while
  dispatching indirectly from it, which wgpu rejects (a stage now
  dispatches over the frame's paths, its lanes returning past its count).
- Member gates (`helio-core` `tests/scene_membership.rs`): gated passes are
  neither scheduled nor prepared while closed, join in build order when
  open, and leave (told once) when it closes again.
- Wide occlusion rays (the voxel plugin's `tests/chunks.rs`
  `wide_occlusion_rays_leave_the_ground`): above; and
  `sun_pages_under_a_canopy` with the half-width reads: 58 of 2,573
  footprint answers differ from the walk (59 before), lit 0.600 against
  0.607.
- On the merged tree: the frame-api suites (BSDF, layouts, surface), the
  reference's (unit and GPU), member gates, the deferred lighting's, HLFS's,
  the sampler's (unit and GPU) and GI's (lib, `probes`, `mesh_gi_graph`,
  `vg_ray_proxies`) pass.
- GI's suites (lib, `probes`, `mesh_gi_graph`, `vg_ray_proxies`) pass on the
  lights' sampler; the sampler's, the deferred lighting's (the texture
  count test included: 14), HLFS's and the materials' unit tests pass;
  Helio's layer gate, fmt, the frame-api tests; `one_owner`.

**indoor_cathedral_gi** (960x540 internal, its capture at frame 240; r2's
contract build `C:/t/int/indoor_cathedral_gi_r2.exe` interleaved with this
one, two runs each; GPU medians in ms):

| | before | after |
|---|---|---|
| GI (probe rays) | 0.109-0.118 | 0.098-0.106 |
| GILight | 0.385-0.407 | 0.307-0.337 |
| LightSampleGrid (fixed, a rebuild) | | 0.057-0.063 |
| LightSamples (a query a probe-ray hit) | | 0.057-0.063 |
| GIShadow (a sample into a shadow ray) | | 0.057-0.064 |
| GIIntegrate | 0.230-0.248 | 0.229-0.249 |
| GI's passes in all | 0.72-0.77 | 0.81-0.88 |
| DeferredLight | 0.49-0.53 | 0.50-0.52 |

Where GI's 0.1 ms goes: GILight lost its loop over its 256 gathered lights
(-0.08) and gained a pass a stage of the contract: the grid (fixed a
rebuild: the cathedral's chandeliers flicker, so its light rows change every
frame; untouched lights cost 0.003, the timer's floor), the batch (one
query a probe-ray hit) and GIShadow (one shadow ray a sample, a pass of its
own because the samples arrive from another pass). The grid is rebuilt
only when lights change, and leaves the schedule with its last user; the
other two are the price of a shared sampler, a dispatch each.

The reference against the real-time image (the merged tree; the scene held
still, `HELIO_STILL`; linear, at the lighting's output before fog,
transparency and the tone map; 3,940 paths a pixel, six bounces): real-time
luminance 0.737 of the reference's, mean |error| 0.0066 (p99 0.045), mean
relative error 24%. The real-time image is darker by a quarter: the
reference's interreflections between the stone walls (albedo near 0.5)
carry more light than the probes' irradiance gives the deferred lighting.
The images through the post (both raw: a debug view's scene colour goes
through without exposure or tone curve, so the real-time image is captured
in debug mode 250, which no pass draws): the whole frame 0.746 of the
reference's, mean |error| 0.0062; the indirect diffuse alone (the deferred
lighting's view 23 against the reference's 63) 0.805, mean |error| 0.0039.
(An earlier "final images 1.21" compared the tone-mapped real-time image
with the raw reference.) The reference still shows fireflies (paths that
find a candle through a bounce); its p99 errors are theirs.

A reference frame (518,400 paths, six stages, the cathedral's meshes
traced through the TLAS): about 7.5 ms on the RTX 3060, by stage: camera
rays 0.41; each depth's trace, surfaces, shade, samples, shadow rays,
shadow trace and resolve about 1.5 ms at depths 0 to 2 falling to 0.4 ms at
depth 5 as Russian roulette and misses end paths; the sums 0.18; the
display 0.06.

**voxel_craft** (the merged tree; the tour's first pose, the grass off as
no tracer traces it (`NO_FOLIAGE=1`: with it the meadow covers 26% of the
image by more than 0.05, where the reference shows bare turf); r0's
baseline, r1's merged build and this one, real-time bursts of eight frames
after 5,000; the reference summed over 7,300 paths a pixel;
`scripts/r2/rerun_reference.sh PART=craft NO_FOLIAGE=1`, results in
`C:/t/r2d/final/craft`).

Two faults in the comparisons before this one, both corrected: the
reference summed frames the terrain had not answered (above), and the
real-time "final" images were tone-mapped while the reference's view went
through the post as a debug view, which shows the scene colour raw (no
exposure, no tone curve: `helio-pass-postprocess`'s `DEBUG_VIEW`), so the
earlier "real-time a third brighter" was the tone curve. The real-time
image is now captured raw too (debug mode 250, which no pass draws), and
this build's real-time scene colour is snapshotted linear with the
reference (`HELIO_REFERENCE_DUMP`, `ReferenceDisplay::keep_tracing`).

The hit surfaces first: the reference's first-hit albedo (from the
tracers' `Full` surfaces) against the G-buffer's albedo view (4), linear,
over the pixels that hit something: 0.1497 0.2097 0.0967 against 0.1495
0.2096 0.0967 (ground 0.0907 0.1515 0.0562 against 0.0905 0.1511 0.0562,
canopy 0.2624 0.3209 0.1788 against 0.2619 0.3203 0.1789): equal.

| against the reference (raw images, through the same post) | whole frame: hits mean \|err\| | luminance | sky mean \|err\| | luminance | indirect (23 vs 63) on hits mean \|err\| | luminance |
|---|---|---|---|---|---|---|
| r0 | 0.00705 | 1.003 | 0.0263 | 0.777 | 0.00310 | 1.054 |
| r1 | 0.00691 | 1.009 | 0.0263 | 0.778 | 0.00303 | 1.083 |
| r2 (this build) | 0.00693 | 1.007 | 0.0263 | 0.777 | 0.00302 | 1.078 |

(Whole images: r0 0.01015 mean |err|, p99 0.072, luminance 0.930; r1
0.01004, 0.072, 0.934; r2 0.01006, 0.072, 0.933.) r0 against r1: r1 has
the lower error everywhere, r0 the luminance nearer 1 on hits and in the
indirect diffuse, each by about 1% of the error, inside what one pose and
eight frames resolve (r2 against r1 on the same runs: 0.0019 mean). They
are alike at this pose; neither is nearer truth by a margin this
measurement can stand on.

Where this build's gap lies (linear, the lighting's output before fog and
the tone map; real-time luminance 0.894 of the reference's over the whole
image), by the reference's own parts (`reference_parts.py`) against the
real-time image's views:

| linear mean luminance | reference | real-time |
|---|---|---|
| sky pixels (16% of the image) | 0.0775 | 0.0423 (0.546) |
| hits (84%): all light | 0.0439 | 0.0445 (1.013) |
| hits: direct (the lights at the first hit; view 22) | 0.0355 | 0.0377 (1.06) |
| hits: indirect (sky 0.0026 + first bounce 0.0049 + later 0.0009; view 23) | 0.0084 | 0.0071 (0.84) |
| canopy hits: direct / indirect | 0.0175 / 0.0129 | 0.0192 / 0.0106 |
| ground hits: direct / indirect | 0.0445 / 0.0058 | 0.0459 / 0.0053 |

So r2's real-time image is not brighter than the reference: over what the
camera hits it is within 1.3%, with direct light 6% above the reference's
and indirect light 16% below it (the first bounce off the canopy carries
most of what GI's probes miss); the whole-image gap is the sky the camera
sees, which the reference takes from the atmosphere's low-order sky table
(`sky_table_eval`, the irradiance harmonics back to radiance, as GI's probe
rays do) at 1.8 times the atmosphere's rendered sky. That is where r3
starts: the reference's (and GI's) sky radiance against the sky pass's, then
the real-time direct light's 6% (shadow softness and the sun's
transmittance), then the indirect light's missing first bounce.
Every comparison here is to be rerun at integration with
`scripts/r2/rerun_reference.sh`.

**Open**:

- The reference's sky on a miss is the atmosphere's low-order sky table
  (as GI's probe rays take it): 1.8 times the sky pass's radiance where
  voxel_craft's camera sees the sky. The first item of r3.
- Grass is not traced: the reference and GI see bare turf under it.
- The terrain answers no ray stream until its draw and ray kernels have
  compiled: 64 s and 38 s on this machine with a cold driver cache (about
  3,800 frames of voxel_craft), during which GI holds its probes and the
  reference sums nothing. A host keeps its pipeline cache
  (`PULSAR_PIPELINE_CACHE`); compiling the ray kernels a stream's detail
  needs first (the trace and the `Diffuse` surfaces before `Full`'s) would
  let GI start sooner. The terrain's owners'.
- `HELIO_STILL=2` (rows written once) culls the cathedral's objects: Hi-Z
  history faults when nothing writes a row; the comparisons use
  `HELIO_STILL=1` (rows rewritten unchanged).
- Emissive surfaces are not light samples: paths find them only by
  bouncing, the fireflies.
- HLFS still evaluates its lights itself (its compacted light rows, its own
  BRDF, no IES or light-function bindings), a second copy of
  `light_incidence`'s and `bsdf_eval`'s jobs. Moving it onto them changes
  its bindings and its image, so it is not a local change; owner: the
  lights plugin, the phase that reworks HLFS.
- The wrapped (subsurface) lobe's weights reach 90 near the horizon (the
  cosine sampling's density goes to zero where the wrap does not): fireflies
  in the reference on subsurface surfaces.
- The reference restarts on SceneDB content and the camera; state outside
  SceneDB (a voxel edit, a GI setting) does not restart it.

#### Integration

The steps merged (Helio 299854c5 and the plugins' `render/r2-hit-surfaces`),
with Pulsar-Native `main` 54ae9db58 merged in, on an RTX 3060.

**Faults found here and fixed**, all of one kind: r2's ray kernels take
longer to compile than r1's, and three places counted frames or assumed the
terrain could already answer.

- *A ground batch was marked answered when no probe ran.* The chunk pass
  answered the foliage's `GroundLink` batch whenever it called
  `probe_ground`, which returns without dispatching while the world cannot
  draw (pipelines compiling, no primary volume). The placement then read
  hits nothing wrote and kept no blade. `probe_ground` now says whether it
  answered, and the pass answers the batch only then; the placement asks an
  unanswered batch again.
- *The pass stopped asking for frames while its ray kernels compiled.*
  `ChunkPass::needs_frame` went false once the residency settled, though
  `settled()` also requires the rays' link complete. A host drawing on
  demand (the editor) would go idle with the streams unanswered. The pass
  asks for frames until `ray_pipelines_ready`.
- *Tests waited a number of frames.* `foliage_on_voxels` ran 600 frames and
  then counted blades; it now waits for `ChunkPass::settled()` under a
  wall-clock guard. A test waits on a readiness signal, never a count.
- The shader validation test's snippet list lacked the frame's surface,
  BSDF, ray stream, ray surface, light sample and heatmap snippets, so the
  reference's shader did not resolve there.

**Gates.**

- Helio: `cargo test --workspace --no-fail-fast`, 108 suites, none failing;
  the layer gate, 0 violations; fmt.
- The engine's installed suites, each exe on its own under the GPU lock:
  `render_acceptance` 40 of 40 (a process a test), `ray_tier_acceptance` 3
  of 3, `default_pipelines`, `plugin_pass_lists`, `runtime_render_parity`,
  `voxel_render_parity`, `skinned_frames`, `level_editor`, `lifecycle`,
  `memory_over_ticks`, `gameplay_defaults` and the rest of the list pass;
  `one_owner` passes.
- The plugins' own: GI (lib 10, `mesh_gi_graph` 2, `probes` 15,
  `vg_ray_proxies` 3), meshes (rays lib 4, `mesh_rays_gpu` 15, component
  50, `skin_cache` 2, `wgsl_rows` 6), materials 13, lights (`sampler_gpu`
  3, deferred lib 3, globals 21, layouts 18), voxel in release (lib 35,
  `chunks` 55, `clouds` 6, `presets` 1, `volumes` 8, `voxel_pass_graph` 13,
  `wgsl` 2, `wide_products` 1, `foliage_on_voxels` 1).
- Red as on `main`, with the same messages and numbers as `main`'s logs:
  `catalog` and `traits` (trait definitions the blueprint and trait editor
  plugins do not declare), `contracts`
  (`descriptors_require_what_their_crates_link`), the virtual-geometry
  acceptance (4,801 depth texels against the classic render) and its dense
  suite (coverage 0 against 4,835).

**Frames.** Mean absolute difference of the images, on a scale of 0 to 1;
"same build" is two runs of one build.

| | same build | r2 against the baseline | luminance |
|---|---|---|---|
| voxel_craft, 4 poses, 8-frame bursts, no clouds (against r1) | 0.0001-0.0015 | 0.0005-0.0016 (p99 at most 0.016) | -0.03% to -0.15% |
| Earth ground hold, 1600x900 (against r0) | 0.0002-0.0005 | 0.0010-0.0011 | 0.3575 against 0.3577 |
| Moon ground hold, 1600x900 (against r0) | 0.0002-0.0009 | 0.0014-0.0017 (p99 0.0235) | 0.5226 against 0.5220 |

The planets differ from r0 as r1 did (section 3's table); r2 changes the
encodings of a hit and the pages' read, not the image.

**The editor** started on a fresh copy of the `voxel_planet` example
project and ran 240 s with no validation error or panic; its graph schedules
the voxel and mesh tracers for GI's probe and shadow streams, the terrain's
and the meshes' surface passes and `GIShadow`.

**Not verified**: a scene with authored spaces traced through the engine
(above); the reference against real time through the editor (the
comparisons are the examples').

**Carried into r3**, in order:

1. The sky a miss returns (the atmosphere's sky table) at 1.8 times the sky
   pass's radiance.
2. Real time against the reference: direct light 6% above and indirect 16%
   below on voxel_craft, the cathedral at 0.74.
3. The binned material passes' fixed 0.03 to 0.06 ms a program with hits:
   a fork change (dispatches that write disjoint ranges of one buffer
   without a barrier and an indirect validation between them), or one
   kernel switching on the program.
4. The planets' probe walk, 1.0 to 1.2 ms above the kernel r1 replaced.
5. The terrain's kernels compiled in the order the streams need them
   (trace and `Diffuse` before `Full`).
6. Grass traced; emissive surfaces as light samples (r5 owns the lights'
   side); HLFS on `light_incidence` and the frame's BSDF; the reference
   restarting on state outside SceneDB; deforming static meshes in the
   engine TLAS (Pulsar-Native#1744); two TLAS projections
   (Pulsar-Native#1739).

## 5. r3: one denoiser, a radiance cache, indirect diffuse a pixel

Branch `render/r3-radiance-cache`.

### Carried in

From r2's "Integration", in order: the sky a miss returns against the sky
pass, real time against the reference (direct and indirect), the binned
material passes' fixed cost a program, the planets' probe walk, the order
the terrain's kernels compile in, and the list of its item 6.

### The problem

Read on r2's tree.

- **Indirect diffuse is a lattice of probes read at a pixel.** `GIResolve`
  interpolates eight probes 0.5 m apart (finest level) into
  `INDIRECT_DIFFUSE` with no temporal or spatial filter. Nothing nearer a
  surface than about two spacings is in it (it offers 1 m as
  `indirect_occlusion`), so indirect light has no contact detail, leaks
  through walls thinner than a spacing, and the reference finds it 16% dark
  on voxel_craft and the cathedral's indirect diffuse at 0.81.
- **A hit is lit where it is read.** GI lights each probe hit itself (a
  light query and a shadow ray a hit, 174,080 a frame) and reads the probes
  back for the bounces after. A second reader of lit hits (the pixels'
  rays here, r4's reflections) would repeat that work for the same surfaces.
- **No one owns a pixel's history.** TSR, the light field and the terrain's
  shadows each keep a previous depth and normal, each reproject and each
  decide disocclusion with its own test; the f64 composition of
  clip-to-previous-clip exists only inside TSR. The frame publishes motion
  vectors and nothing else of the previous frame.
- **Three private denoisers** (TSR's accumulation, the light field's
  temporal and spatial stages, the terrain's shadow stages), none of which
  another signal can use.
- **Meshes have no short-range occlusion.** The deferred lighting reads an
  `"ssao"` texture nothing writes; the terrain adds its own short sky rays.

### How others do it

- **Lumen** (UE5; Wright, SIGGRAPH 2021/2022). Lit surfaces live in a
  surface cache of cards a mesh, final gather through screen probes (an
  octahedral probe each 16 pixels, importance-sampled from last frame's
  lighting, filtered in probe space, interpolated a pixel), a world radiance
  cache of probes for distant light, and a full-resolution screen-space
  bent normal for contact detail. The surface cache is cheap to read and is
  a parametrisation a mesh: it has no answer for terrain that is not a
  mesh, for foliage, or for anything its cards do not cover.
- **AMD GI-1.0** (Boissé et al. 2023). Screen probes (8x8 texels a tile)
  lit from a world-space hash grid of radiance at hit points, the probes
  spawned and reused through reservoirs. The hash grid needs only a hit's
  position, so it serves any geometry.
- **NVIDIA's path-tracing guidance** (SHaRC, ReSTIR GI, NRD). A ray a
  pixel, paths ended in a spatial hash of radiance keyed by position at a
  level set by distance to the camera (40 bytes an entry, 2^22 entries, an
  update pass over a few percent of the pixels), samples reused across
  pixels and frames by reservoirs (Ouyang et al. 2021), and one denoiser
  library for every signal taking motion vectors, normal, roughness, depth,
  hit distance and an optional history confidence from the producer
  (ReBLUR, ReLAX, SIGMA; a directional mode for signals traced below full
  resolution).
- **Kajiya** (Stachowiak 2022). An irradiance cache of entries allocated
  where rays hit, each tracing its own rays a frame, with ReSTIR GI at half
  resolution reading it.
- **DDGI** (Majercik et al. 2019), what r1 kept: probes on a lattice. Its
  cost does not depend on the screen and it lights points in the air; it
  cannot resolve what is finer than its spacing.

What they agree on: lit surfaces are cached in world space once and read by
every ray; the final gather reuses samples in screen space; history is
judged once, by geometry and by whether the light changed. Where they
differ is the cache's parametrisation, and only the hash of hit points
needs nothing from the kind that was hit.

### The design

Four things, each with one owner. Hits are still read only through ray
streams, light only through light samples: nothing here names a kind.

#### The frame owns a pixel's history

`helio_frame_api::history`, published by the spine for a view whenever a
reader asks:

- **The previous guides**: last frame's view depth, normal and roughness at
  the internal resolution (the G-buffer's targets are transient; the spine
  keeps this copy).
- **The reprojection**: the f64 composition from this frame's clip space to
  the previous one's, rebased across an origin change (what TSR computes
  for itself today), as a uniform a view.
- **Where a pixel was**: for each pixel, its previous position (from the
  velocity where the surface moves by itself, from depth and the
  reprojection elsewhere) and whether the previous frame held the same
  surface there (depth along the previous view within the pixel's slope,
  the normals agreeing). One pass a frame; every temporal pass reads it.

TSR takes the reprojection from here in this phase. The light field's and
the terrain's shadow stages keep their own histories until r5 replaces the
signals they filter (moving a filter under a signal about to be replaced is
the work twice); r5 moves them.

#### One denoiser contract

`helio_frame_api::denoise`. A producer describes a **signal**: its kind
(irradiance as first-order harmonics of luminance with two chroma terms;
radiance; a visibility), the resolution it was traced at against the
internal one, its hit distance, and optionally a **confidence** in [0, 1]
that its history still describes the light (the producer knows: a
reservoir's revalidated sample, a cache entry that changed). It gets back
the filtered signal at the internal resolution. `denoise::filter(build,
&signal)` adds the registered denoiser's passes after the producer's, as
`stream::trace` adds the tracers'. With none registered the signal passes
through, so a producer needs no denoiser to build.

The implementation is `helio-pass-denoise`, a pass crate that needs only
frame contracts: reprojection through the frame's history, accumulation
whose length follows geometric validity times the producer's confidence, a
wide reconstruction where history is short, and a variance-guided spatial
stage stopped by depth, normal and hit distance (near occluders keep
contact detail). A vendor's reconstruction (r7) registers as another
denoiser under the same contract.

Its first signal is the pixels' indirect diffuse; r4 brings reflections and
r5 the shadows.

#### The radiance cache: where a surface is lit for indirect light

A world-space hash of **irradiance arriving at a surface cell**, direct and
indirect apart, read by every ray that needs a lit hit. A reader multiplies
by its own hit's albedo (so the cache holds no texture detail and a cell
may be coarse), and one that lights its hit itself (a mirror reflection in
r4) takes the indirect part alone.

- **Key**: the hit's cell in the world (an integer cell, not a frame-space
  one), the cell's level, the dominant side of the normal, the ray's
  coordinate space. Cell sizes are powers of two in metres, so the frame's
  origin (a multiple of 1024 m) is a whole number of cells at every level
  and an origin change moves no entry; cells are counted in 64 bits, so a
  planet's far side is a cell like any other.
- **Level**: from the footprint of the ray that reads it (its cone's width
  at the hit) and the distance to the eye, whichever is coarser. A wide
  probe ray reads a coarse cell; a pixel's ray near the eye a fine one. A
  cell with no value answers from the next coarser level that has one.
- **Life**: a read marks its cell used. A cell first used takes a point and
  normal from the hit that asked. Each frame the cache spends a ray budget
  on the cells in use, new ones first and then the longest unlit: it traces
  rays from the cell's point through a stream, lights their hits with one
  light sample and one shadow ray each, adds the sky on a miss and the
  cache's own irradiance at the hit for the bounces after, and folds the
  result into the cell with a weight that shortens when its estimate
  moves. A cell unused for a while is evicted.
- **Who reads**: the probes' rays (GI's light and shadow passes for probe
  hits go: a probe hit is `albedo * cache + emissive`), the pixels' rays
  below, r4's reflections. Light at a surface for indirect purposes is
  computed in one place.

The contract (keys, the entry's rows, `//!use frame_radiance_cache` with
the read and the mark) is the frame's; the cache that publishes it is the
GI plugin's. Absent, a read finds nothing and the reader falls back (the
probes to their own previous state).

Against the alternatives: a surface cache needs a parametrisation a kind
(none for voxels at 10 cm or for foliage); probes cannot hold a wall's two
sides apart; a neural cache needs inference this API has no fast path for.
A hash of hit points serves every tracer, costs memory with the surfaces
rays reach and not with the world, and its error is bounded by a cell that
shrinks towards the eye.

#### Indirect diffuse at a pixel

On a budget grid: one ray to each block of pixels (a setting: 16, 4 or 1
pixels a ray, the traced pixel rotating in its block), from the G-buffer's
surface, its hit lit from the cache. Samples are kept in a reservoir a
traced pixel and reused across frames (through the frame's history) and
across neighbours (ReSTIR GI), with a share of the kept samples retraced
each frame: a retraced sample that changed lowers the confidence the
denoiser is given, and that is how a light switched on does not ghost.
The reservoirs resolve to irradiance as first-order harmonics, so a pixel
between traced ones takes its light by its own normal; the denoiser filters
that, and the result is `INDIRECT_DIFFUSE` in its present encoding.

Occlusion nearer than a block's footprint is not in a ray a block: a
full-resolution screen-space horizon term over that distance and no farther
(the distance the gather offers as `indirect_occlusion`) restores it, in
the publisher of `INDIRECT_DIFFUSE` and nowhere else. The deferred
lighting's unwritten `"ssao"` input goes.

The probes stay: they light points in the air (media, particles), answer a
cell the cache does not hold yet, and are the cheaper setting (no pixels'
rays; the resolve as today, its probes lit from the cache). All three
(probes, cache, pixels) read streams, so each works wherever any tracer
answers, with or without the ray query.

Against screen probes (Lumen, GI-1.0): the same ray budget kept as a
hemisphere a tile filters well and stops there; reservoirs a traced pixel
are the same idea at one ray a pixel, converge to the reference as the
budget rises, and are the machinery r5's direct light reuses. The cost is
the correlation resampling leaves, which is the denoiser's to remove and
the reference's to measure.

#### What scales with what

| | scales with | not with |
|---|---|---|
| History pass | internal pixels | scene |
| Denoiser | internal pixels a signal | scene, rays |
| Cache memory | cells rays reached lately (capacity a setting) | world size, objects |
| Cache update | its ray budget (a setting) | cells, screen |
| Pixels' gather | pixels over the block size | scene, lights |
| Probes | probes updated a frame | screen |

Every budget is a setting, none a constant fitted to one GPU: a faster one
takes a ray a pixel and a larger cache.

#### Steps

1. The sky a miss returns and direct light against the reference (carried).
2. The frame's history and the denoiser.
3. The radiance cache, the probes lit from it.
4. The pixels' gather and the near occlusion (needs 2 and 3).
5. The binned material passes' cost a program, the terrain's kernel order
   and its probe walk (carried; cost, not image).

### As built

#### The sky a miss returns, and direct light against the reference

**The sky a ray takes.** A ray that meets nothing (a reference path's
miss, a GI probe ray's) took the sky from the atmosphere's sky table: per
altitude and sun cosine, six L2 harmonics of the sky's irradiance / pi,
turned back into radiance band by band (`sky_table_eval` with bands 1, 3/2,
4). Harmonics of second order hold a cosine-weighted integral well (the
irradiance a Lambertian surface takes: Ramamoorthi and Hanrahan's 3%) and
the sky along a direction badly: the sky has a bright band at the horizon,
the sun's aureole (Mie, g 0.8) and the lit ground below a sharp horizon,
and nine coefficients smear each of them over the sphere. Measured on the
sphere against the sky the sky pass draws (`sky_radiance_gpu`, below): the
table's radiance integrated to the drawn sky's within 0.2% (the band-0
coefficient is exact) and was wrong direction by direction by a mean 16% to
85% of the sky (sun 60 degrees: 16%; sun 2 degrees: 72%; 80 km up: 51%),
with whole bands off: 1.11 to 1.37 times the drawn sky 10 to 30 degrees up
and 0.38 to 0.70 of it past 60 degrees, depending on the sun. The drawn sky
itself projected on nine harmonics in any frame is as far off (0.13 to 0.82
mean error). No exposure, factor of pi or unit was wrong: the
representation was. The 1.8 at voxel_craft's sky pixels is that error over
the part of the sky its camera sees (canopy gaps 20 to 60 degrees up away
from the sun), with the r2 table's coarser sun columns.

How others give a ray the sky: Unreal's Lumen and path tracer take the
sky light's cubemap, captured from one point (the camera, or a placed
capture) each frame or on demand; Hillaire 2020's sky-view LUT is the sky
from the camera only; Bruneton and Neyret 2008 (and Bruneton 2017) keep a
4D scattering table over altitude, view zenith, sun zenith and the
view-sun angle, valid anywhere in the atmosphere, for any number of suns
by linearity. A capture from one point is wrong for what is far from that
point at planet scale: a probe on the night side of a planet the camera
orbits, a hit kilometres up a mountain, the sky seen from above the air.
The sky-view LUT is the camera's alone.

**As built** (frame-api `lighting::sky`, `frame_sky.wgsl`,
`sky_radiance.wgsl`; the sky plugin's `atmosphere_luts.wgsl`): the sky
table is two tables over one set of entries, both published by the sky's
atmosphere pass and rebuilt only when its rows change:

- `SKY_RADIANCE` (`sky_radiance_kernel`), the radiance table: a 3D texture,
  one slice an entry (16 altitudes, quadratic in height from 1 m to the
  top of the air; 40 sun cosines, quadratic away from the horizon on both
  sides, 16 of them below it down to -0.4), each slice a 32 x 64 map of the
  sky's radiance for a unit sun in the entry's frame (x towards the sun):
  azimuth 0 to pi from the sun (an entry holds one sun, about whose plane
  its sky is symmetric), columns concentrated near the sun, rows at the
  horizon as the sky-view LUT's. Each texel is the sky-view LUT's own
  integral (`sky_along`: `sky_integrate`, 32 steps, single and multiple
  scattering, the lit ground below the horizon through the air), the
  function the camera's LUT now calls too, so the two cannot drift. 10.5
  MiB (rgba16f); rebuilt in 3 to 5 ms when the medium changes (the tests'
  wall time, all parameter tables), never a frame otherwise.
- `SKY_TABLE`, the irradiance table: its six harmonics now integrated from
  the radiance table's entry (256 directions of it), so a surface's sky
  irradiance is the integral of the radiance a ray takes. Irradiance only:
  `sky_table_eval` lost its `bands` argument and every caller passed 1
  (the deferred lighting's ambient, the clouds' sky light).
- `sky_radiance_eval(from_camera, d)` (`//!use sky_radiance`, bindings
  `sky_frame`, `sky_radiance`, `sky_radiance_sampler`): the four entries
  about the point's altitude and each sun's cosine, each read at the
  coordinate of `d` at the point's actual radius (each row's map is laid
  out about its own horizon, so the horizon lines up between rows), mixed
  geometrically (the sky falls exponentially with altitude and with the
  sun's depth below the horizon; a linear mix was 25% high at 80 km and
  10% in twilight), summed over the frame's suns by their radiance. From
  above the air, the sky where `d` enters it; nothing where it misses.
  Every reader moved onto it: the reference (`sky_radiance_at`) and GI's
  probe rays (`gi_light.wgsl`); water bound the sky table and read
  nothing from it (binding deleted).
- **The sky-view LUT was symmetric about the first sun**: its azimuth was
  `|atan2|`, so with two suns (a sun and a moon) the second one's glow was
  mirrored across the first one's vertical plane. It now covers the full
  circle (256 x 108, columns concentrated on both sides of the first
  sun); `sky_view_uv` and `sky_view_direction_angles` changed for every
  reader (the composite, water, the eye's irradiance).

**The contract, tested** (`plugins/vendor/sky/feature/tests/sky_radiance_gpu.rs`):
the atmosphere pass in a render graph on the device, a probe reading, for
16,384 directions of a Fibonacci sphere from the eye, the sky-view LUT as
the composite samples it and `sky_radiance_eval`; Earth with the eye at 2
m and suns at 60, 31.5, 25, 8, 2 and -3 degrees, at 300 m, 3 km, 20 km and
80 km, two suns, Mars. Over the sphere the miss's sky integrates to the
drawn sky's within 2%, its mean absolute error is within 2% of the sky's
mean, and 99% of directions are within 10%:

| pose | integral | mean \|err\| | p99 |
|---|---|---|---|
| ground, sun 60 / 31.5 / 25 | 0.999 / 0.999 / 1.000 | 0.2% | 1.3% to 2.6% |
| ground, sun 8 / 2 | 0.997 / 1.000 | 0.5% / 0.4% | 3.2% / 5.8% |
| ground, sun -3 (twilight) | 0.983 | 1.8% | 3.1% |
| 300 m / 3 km / 20 km / 80 km | 0.999 / 0.996 / 0.995 / 1.000 | 0.2% to 0.5% | 1.4% to 8.6% |
| two suns / Mars | 1.000 / 0.996 | 0.2% / 0.5% | 1.1% / 2.1% |

The last percent of directions is the aureole of a sun on the horizon, the
limb seen from high up and the zenith there (radiance near zero: above 80 km
it is 0.3 of the drawn sky's). Two faults of the frame's sky the test found
on the way, both fixed at their owner (`frame_sky.wgsl`): `ray_sphere` took
the near root as the difference of two near-equal distances, so a point
metres above an Earth-sized ground (radii in km) lost its sign and saw
through the ground (the drawn sky was black in places 53 degrees down from
2 m under a 2-degree sun; the table's lowest row had to sit at 10 m); it
now takes the root nearer zero from the roots' product, and the lowest row
is 1 m. And the table's rows put row 0 at 10 m while its lookup put row 0 at
the ground, so an eye between the lowest rows read them at the wrong
weights; one mapping now serves both (`sky_table_row_radius`,
`sky_table_row`).

**Direct light.** r2 measured real-time direct light (the deferred
lighting's view 22) 6% above the reference's lights at the first hit on
voxel_craft (grass off, clouds off). Measured here per cause, on the
tour's first pose, mean luminance of the direct light over the pixels the
reference's first hit is a surface (its albedo dump), every real-time view a
linear snapshot (`HELIO_REFERENCE_DUMP`), every reference over 2,400 summed
frames past the terrain's readiness (a fresh delta-sun reference of this
build matches r2's to 0.00006 mean difference, luminance 1.0000):

| step | real time / reference |
|---|---|
| r2's build and this one, as they were | 1.074 / 1.078 |
| the reference's sun a disk (its shadow rays aim at a point of the atmosphere's sun disc, as the terrain's walks do) instead of a delta | the reference +0.6% |
| the terrain's sun by exact walks over the disc at every pixel (steady shadows) instead of its block samples, pages and denoiser | real time -1.4% |
| the reference's shadow rays zero-wide (exact walks) instead of the footprint (pages past a walked band) | no change (1.0001) |
| the deferred lighting's 30% light past a subsurface surface's shadow removed | -0.3% (noise) |
| the sun's air: real time and the sampler both take `light_incidence` (`sky_transmittance_towards`) | same function |
| steady shadows with the natural steps' skip removed / with every skip removed | 1.056 / 1.034 against the disc reference |
| the steps' shadows put back (`step_shadowing`, below) before the denoiser | 1.014 against the disc reference, 1.020 against the delta one (not landed) |
| **as built: none of the above changed; real time as r2** | **1.072 (the rerun below; canopy 1.10, ground 1.07)** |

**The cause** is the terrain's prefiltered steps. A voxel slope is lit as
the smooth surface its steps stand for (natural steps by their stored
slope's normal, smoothed surfaces by theirs, cells of a few pixels or less
by the slope around them: `face_rule`, `face_lit`), and its sun rays skip
the steps (`skip_along`, up to 12 cells) so the steps' shadows never show as
stripes. Both together light every facet of the staircase with the plane's
cosine and none of its own shade: a staircase standing for a plane of
normal n has |n|_1 of facet area per unit of the plane, and the light the
plane takes ((n . l) per unit) cannot light more than that, so a sunlit
slope comes out up to 1 / (1 - shaded) too bright (a 10-degree slope under
voxel_craft's 31-degree sun from up-slope: 1.65). The reference traces the
steps and sees their shadows: the stripes and triangle lattices of the
difference image at mid distance, where cells are a few pixels wide.

**The fix, designed and measured, not landed.** It belongs to the owner,
the terrain's sun visibility (the voxel plugin's `chunk_draw.wgsl`): where
a ray skips the steps (`Face::skip`), multiply the sun by the share of the
staircase it lights as the pixel shows it. Per cell axis i the staircase has
|n_i| of facets facing sign(n_i); those facing away from the sun are dark;
the plane's light falls on those facing it, which would take sum |n_i l_i|
if none shaded another, so each keeps (n . l) / that of its area; the pixel
shows the facets facing the eye, weighted by |n_i v_i|. It is exact for a
staircase whose steps shade only its own facets, 1 on a level surface and on
a slope lit from down-slope, mixed in by the skip's weight. How others do
it: the facets' shadowing of a microsurface (the V-cavity masking-shadowing
of Torrance and Sparrow, Heitz 2014), with the facets the voxel axes rather
than a distribution of normals; filtering the shadows away (what is there)
is the stripe-free look without the energy. Three placements were tried,
and each failed a terrain gate, so the voxel plugin is unchanged:

- multiplied into the walks' samples before the denoiser: 1.078 to 1.020;
  the denoiser's history lagged a sun turned 20 degrees, 1.37% of pixels
  off by more than half at 4 frames against the walks' own 0.92%
  (`terrain_shadows_match_steady_shadows`);
- in the resolve after the filters, from the hit's face alone (no ground
  hint there): almost no effect (1.073);
- carried in the sun word's terrain byte: the flicker gate failed (0.07%
  of pixels against 0.055%) and the walked metric no longer matched.

The patch of the resolve placement is kept outside the repositories
(`C:/t/r3s/step_shadowing_resolve.patch`). Landing it needs the share in the
terrain's own visibility data, so the denoiser filters a quantity whose
history follows the sun (the share is analytic per pixel; the history lag is
the walks', not the share's) — the voxel plugin's decision, carried to r4.

**What else is left** besides the steps, by region (measured with the steps' shadows put back, 1.4% against the disc reference):

- *Canopies, 1.11*: two-sided leaves lit from behind. Real time takes the
  sun through the cell (`shadow`: the sun's pages at the cell's far side,
  the "thin two-sided surface" the material stands for); the terrain's
  tracer answers a ray that starts on a solid cell's face, into it, as
  blocked by that cell (`stream_enter`, `START_FACE`), so the reference
  (and GI's shadow rays from leaves) never pass light through a leaf. The
  tracer's to change (a ray leaving a cell that lets light through starts
  past it, as the real-time walk's `through` does); not changed here.
- *Distant ground near the horizon (image rows above 250), 1.10*: there the
  real-time shading normal (the screen's slope, `screen_ground`) gives a
  sun cosine 0.71 of the tracer's surface (the cell's stored slope), and its
  far-field visibility (coarse pages, the canopies' cut) is higher than the
  reference's; the two partly cancel. Over all hits the real-time cosine is
  0.957 of the reference's (`max(N . L, 0)` written by both, from disk).
  The terrain's two descriptions of a hit's slope (screen-space for the
  raster, the stored slope for rays) are its own to reconcile.
- The steps' sky light (`sky_visibility` also skips the steps) and local
  lights' terrain shadows (`light_skip`) take the same prefilter; their
  share of the staircase is not put back here (indirect and lamps are not
  this step).

**The rerun** (`scripts/r2/rerun_reference.sh` steps, this build against
r2's results in `C:/t/r2d/final/craft`; voxel_craft's first pose, grass and
clouds off, the reference over 6,230 paths a pixel; linear, the lighting's
output before fog and the tone map; the same pixel masks for both, the
reference's first-hit albedo; results in `C:/t/r3s/compare/craft`):

| linear mean luminance: reference \| real-time (ratio) | r2 | r3 (this build) |
|---|---|---|
| sky pixels (14% of the image) | 0.0810 \| 0.0415 (0.512), mean \|err\| 0.0398 | 0.0413 \| 0.0415 (1.005), mean \|err\| 0.0002 |
| hits (86%): all light | 0.0443 \| 0.0445 (1.005) | 0.0437 \| 0.0443 (1.014) |
| hits: direct (view 22) | 0.0350 \| 0.0376 (1.074) | 0.0350 \| 0.0375 (1.072) |
| hits: indirect (view 23) | 0.0083 \| 0.0075 (0.898) | 0.0081 \| 0.0073 (0.899) |
| of which the reference's sky / first bounce / later | 0.0026 / 0.0049 / 0.0008 | 0.0024 / 0.0049 / 0.0008 |
| canopy hits: direct / indirect | 1.180 / 0.751 | 1.104 / 0.763 |
| ground hits: direct / indirect | 1.067 / 1.022 | 1.070 / 1.008 |
| whole image | 0.0493 \| 0.0441 (0.894) | 0.0434 \| 0.0440 (1.013) |

The sky on a miss now matches: the camera's sky pixels from 0.51 to 1.005
of the reference, the whole image from 0.894 to 1.013, and the reference's
sky light on hits fell by 8% (its sky was the 1.8x table). The real-time
image did not move (0.0415 sky, 0.0440 whole), as the real-time sky was
already the sky pass's. Direct light is r2's (the steps, above, are not
landed); on the canopy, real-time view 22 is 6% below r2's on the same pose (ground unchanged), which this session did not isolate: the voxel plugin is unchanged, and the canopy is where one pose's eight-frame burst varies most. What is left of the indirect gap is the
canopy's: 0.76 of the reference there (ground 1.01), the first bounce off and
through the leaves that GI's probes miss, r3's radiance cache and per-pixel
indirect diffuse.

**indoor_cathedral_gi** (the same steps as r2's above: `HELIO_STILL=1`
real time at frame 240 against the reference at frame 2,400, 2,340 paths a
pixel; results in `C:/t/r3s/compare/cathedral`): real-time luminance 0.737
of the reference's, mean |error| 0.0067, p99 0.058 (r2: 0.737, 0.0066); the
indirect diffuse alone (view 23 against the reference's 63, through the
post) 0.817, mean |error| 0.0038 (r2: 0.805, 0.0039). The cathedral is
indoors: no camera pixel and few probe rays see the sky, so the sky's change
leaves it where r2 left it, and its indirect gap (the stone walls'
interreflections the probes' irradiance misses) is r3's radiance cache's.

**Frames against r2's merged build** (tone-mapped captures, RGB 0-1;
`C:/t/r3s/frames`): voxel_craft's tour, four poses of eight-frame bursts
after 5,000 frames (clouds off, grass on), r2 against this build: mean
|difference| 0.0018 (0.0014 to 0.0025 an image), luminance -0.34%; this
build against itself: 0.0002. The change is the sky light: GI's probe rays now take
the drawn sky rather than the 1.8x table, and the irradiance table the
deferred lighting reads is integrated from the radiance table (the reference's
indirect sky on hits fell by the same 8%). voxel_flight's Earth ground hold
(25 s, clouds off): r2 against this build @@EARTH@@.
