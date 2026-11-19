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

#### What the next three agents build against this

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
- **Pre-existing reds met here, not this change's**: the deferred lighting's
  `each_entry_point_fits_the_portable_texture_limit` (its documented count
  15 against its layouts' 14) and the transparent drawer's
  `production_pipelines_accept_coverage_and_legacy_templates` (its "legacy"
  string replacement targets a `TransparentOutput` of two fields; the shader
  returns three). Neither code path is touched here.
- **The raster and the hits disagree on a leaf's back**, unchanged by this
  step (images held): the foliage template's `subsurface_color` is the full
  albedo, a hit's is `albedo * material_transmission`. The materials agent
  settles it in the template (it changes the raster image).
- **Not verified here**: a scene with authored spaces traced through the
  engine (no example or level has a rigid frame or portal pocket with
  traced meshes; the engine projection's groups and `SpaceSlot` journal are
  covered by compilation and the tracer's own GPU test, not by frames);
  Earth and Moon holds (voxel_flight), whose terrain code changed only in
  its rows' packing; the engine's render acceptance suite.

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
  differ, worst normal cosine 0.9999999. Two of the publish's inputs a hit
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
  the finest level resident at its origin up to its footprint's in each
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
| Earth | rays in the draw 1.09-1.21, GI 0.28-0.30: 1.37-1.51 | 2.46 | GI 0.06, VoxelRays gi_probe 1.58, surfaces 0.17, GILight 0.25, VoxelRays gi_shadow 0.21, GIIntegrate 0.13: 2.40 |
| Moon | rays in the draw 0.95, GI 0.26: 1.21 | 2.54 | GI 0.06, VoxelRays gi_probe 1.57, surfaces 0.13, GILight 0.22, VoxelRays gi_shadow 0.23, GIIntegrate 0.13: 2.34 |

On voxel_craft the terrain's GI is 0.15-0.2 ms under r1's, all of it the
sun's rays (0.41 -> 0.21); on the planets 0.06-0.2 under r1, and still
about 1 ms over r0, the probe walk (1.57-1.58 against r0's in-draw kernel's
0.95-1.21 with its shading) being the remainder (open, below).

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
- **Startup**: the terrain's draw pipelines took 41-74 s to come up in this
  build's test and voxel_craft runs (every run: no driver cache hit), the
  r2 contract build's 52 s once and 9 s after (`publish_shows_hits_only_in_front_of_meshes`
  wall time, same machine; the NVIDIA shader cache at its 1.1 GB size with
  three sessions compiling). Not bisected: which of this change's draw
  pipelines misses the cache, or whether it is the cache's size.
- Fog reads the pages with a texel's bias from points in the air
  (`sun_occlusion_point`, the lights plugin's): under a canopy it takes
  sun within a texel of leaves; a point in the air could read with none.
  The fog's and the lights' owners decide.
