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
wide reconstruction where history is short, and a spatial stage stopped by
geometry alone (depth, normal, an occluder the depth shows between two
receivers, and the hit distance, so near occluders keep contact detail).
No weight depends on the light being filtered, so a signal's mean is kept
whatever its noise (as built: "The denoiser keeps the mean"). A vendor's reconstruction (r7) registers as another
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

#### The frame's history and the denoiser

Step 2. Helio `render/r3-denoise`.

**The history** (`helio_frame_api::history`, snippet `//!use frame_history`,
`shaders/history.wgsl`; published by `helio-pass-history`, a spine-layer
crate the spine registers through `RegisterHistory`).

- **Asking.** A reader calls `history::request(build, Need)` while its graph
  builds, before adding its own passes. The first reader of a graph adds the
  publisher's passes at that point (after the G-buffer, which everything
  that reads a history follows); a later one finds them. `Need::Reprojection`
  adds `HistoryReprojection` (a 128-byte upload, no GPU work);
  `Need::Pixels` adds `HistoryPixels` too. A graph whose nodes ask for
  neither schedules neither: today the TSR node asks for the reprojection,
  the denoiser for the pixels. Without a publisher (a host with no spine)
  `request` answers `false`: TSR and the denoiser then refuse to build
  (the denoiser reads its guides from the history). The api
  crate holds no state: the publisher and what was added live in the build
  environment's shared state, as the ray tracers do.
- **Per view.** The passes key their state by the camera's view id
  (`jitter_frame[3]`), as the renderer keys `prev_view_proj`: mono or each
  XR eye; a capture is a graph of its own, so its own history. Eight views
  at most (least recently rendered first out).
- **`REPROJECTION`** (uniform, `HistoryReprojection`, 128 bytes): TSR's
  former `Reprojection` moved here unchanged (UE's ClipToPrevClip composed
  in f64 from `view_proj`, `prev_view_proj` and the previous view rebased
  across an origin change by `history::rebase_previous_view`) plus `info.x`,
  whether the view has a previous frame, and `info.y`, the frames it
  rendered. `history_reproject(h, raster_uv, depth)` returns the previous
  raster uv (jittered), in-front, and the previous view depth (0 sky; sky
  reprojects as a direction). Rasters are as drawn, each frame with its own
  jitter: a reader on another grid (TSR's unjittered output) removes the
  previous jitter itself.
- **`PIXELS`** (`Rgba16Float`, internal resolution): xy the surface's
  previous position minus the pixel's centre in pixels; z a 4-bit mask of
  the 2x2 previous texels around it that held the same surface; w the flags
  (`MOVES`: from the G-buffer's velocity; `HAS_PREVIOUS`). `history_pixel`
  decodes it, `history_taps` gives the bilinear footprint's weights with the
  rejected taps zeroed (a reader renormalises by `history_taps_weight`, or
  keeps less history where it is small). The previous position is the
  velocity's where bit 0 of its w says the surface moves by itself, the
  reprojection's elsewhere. A tap holds the same surface when its previous
  view depth is the expected one within what the surface's slope moves
  depth by over the tap's offset, and its normal faces the same side
  (n . n' > 0: depth cannot tell the two sides of a leaf or a thin wall
  apart; a stricter normal test rejects a normal-mapped surface's own
  detail). The slope: the four neighbours, reprojected too (once a texel
  into workgroup memory), give the local map from a step of one current
  pixel to a previous-raster displacement and a previous-depth change; the
  tap's offset solved in those steps times the larger of each axis's two
  one-pixel changes bounds a plane's depth there (depth along a plane is
  monotone and convex in screen position, so over part of a pixel on either
  side it changes by at most that part of the larger side's change). The
  neighbours are the same surface's where any are (a normal turning by more
  than 60 degrees is a crease: the ground's depth at a wall's foot says
  nothing of the wall's slope), and the larger side is held to twice the
  smaller (along a plane the two differences stay within a factor of two
  unless its horizon is within three pixels; a depth step before a parallel
  surface is far more); one side alone counts twice (along a plane the
  slope at the pixel exceeds the chord to a neighbour a step away by
  1 + step / distance to the horizon). A surface that moves by itself uses its current
  depth changes and may change its depth by 1/32 a frame (20 m/s at 10 m
  and 60 Hz). Floor: 1 mm plus 2^-12 of the depth (an order above the f32
  rounding of depth, positions and the reprojection). Earlier versions of
  this test (the smaller side's slope alone, a slope per previous pixel
  along each axis, one side at its own value) rejected valid taps on
  grazing ground and at a wall's foot in the moving-camera test.
- **`DEPTH`** and **`NORMAL`** (added with the denoiser's cost rework):
  this frame's guides, the same textures the pixels pass already wrote for
  the next frame, published under their own names, so every reader (the
  denoiser's signals, any temporal pass) takes the view's one copy; next
  frame the slots swap and they are the previous guides. The denoiser's own
  packed guide copy (16 bytes a pixel a signal) went with this.
- **`PREVIOUS_DEPTH`** (`R32Float` view depth, 0 sky) and
  **`PREVIOUS_NORMAL`** (`Rgba16Float`: shading normal, roughness): last
  frame's guides, ping-ponged by the pixels pass, valid only when the view's
  previous frame wrote them (a frame the pass was gated out of, a resize or
  a view's first frame invalidate them).

TSR takes the reprojection from `REPROJECTION` (bound at 21 beside its own
uniform, which lost the 112 bytes). Its `Reprojection`, `rebase_previous_view`
and previous-view and origin tracking are deleted; its tests drive the
history's reprojection pass beside TSR as the spine does. The light field
and the terrain's shadow stages keep their histories until r5, as the
design says.

**The denoiser contract** (`helio_frame_api::denoise`, snippet
`//!use frame_denoise_irradiance`, `shaders/denoise_irradiance.wgsl`).

- A producer declares `Signal::irradiance(name, TraceBlock)` (`One`,
  `Four`, `Sixteen`: one traced pixel each 1, 2x2 or 4x4 internal pixels),
  optionally `.with_confidence()`; writes `<name>_signal0` and
  `<name>_signal1` (`Rgba16Float`, `Signal::traced_size`, one texel a block)
  every frame; then calls `denoise::filter(build, &signal)` after the passes
  that write them, and reads what `Filtered` names: the denoiser's
  `<name>_denoised0`/`1` at internal resolution, or with no denoiser
  registered its own inputs (`block` says which), read with
  `denoise_irradiance_input`.
- The traced pixel: `traced_pixel(block, frame, side, size)` /
  `denoise_traced_pixel`, frame the graph's frame number truncated: each
  block visits its pixels in Bayer order, every pixel once in side^2
  frames, the blocks of a side x side group out of phase.
- Inputs: 0 = (luminance, Co, Cg of the irradiance estimate at the traced
  normal, hit distance in metres; past the half float's range: none); 1 =
  (the luminance-weighted arrival direction, confidence). A producer sums
  its samples with `denoise_irradiance_add(input, L cos / pdf, dir)`. The
  denoiser reads a sample as the linear irradiance `a + b . n` exact at the
  traced normal, `b` = 3/4 of the luminance-weighted direction (over a
  uniformly lit hemisphere the weighted mean direction is 2/3 of the normal
  and the first-order harmonics' cosine convolution gives a gradient of
  half the value along it), so first-order harmonics of luminance and two
  chroma terms, and samples traced at other normals average.
- Outputs: 0 = (a, Co/Y, Cg/Y, the luminance's standard error; -1: no light
  yet), 1 = (b, 1/hit distance, harmonic mean). `denoise_irradiance(out0,
  out1, n)` is the RGB irradiance at a pixel's own normal.
- Registration: `RegisterDenoiser::denoiser(Denoiser { name, supports,
  filter })` (a feature's prepare hook), `denoise::offer` for a host or test;
  the first registered that supports a signal filters it. A later signal
  kind (r4's radiance, r5's visibility) is a `SignalKind` variant with its
  two tables and its snippet; `Signal`, `filter` and the registration stay.

**A producer, as step 4's gather uses it.** In its node's build: `let
signal = Signal::irradiance("pixel_indirect", TraceBlock::of_pixels(n))
.with_confidence();` then its own passes (they write `signal.inputs` at
`signal.traced_size(internal)`, `Rgba16Float`, every frame, and read the
frame's history themselves with `history::request(build, Need::Pixels)?`
before adding them: their reservoirs reproject through `PIXELS` with
`//!use frame_history`, `history_pixel(texel, pixel)`, `history_taps`,
`history_tap`, `history_taps_weight`), then `let filtered =
denoise::filter(build, &signal)?;`, then the pass that resolves
`filtered.textures` into `INDIRECT_DIFFUSE` (`//!use
frame_denoise_irradiance`: `denoise_irradiance(out0, out1, normal)`, or
`denoise_irradiance_input(in0, in1, traced_normal, normal)` with the traced
pixel `denoise_traced_pixel(block, frame, side, size)` when
`filtered.denoiser` is `None`). Traced texel `b` (one thread a texel of
`traced_size`) traces internal pixel `denoise_traced_pixel(b, frame_num,
side, internal size)`, `frame_num` the graph's frame number truncated; its
sample is the sum of
`denoise_irradiance_add(input, L * cos / pdf, dir)` over its samples, its
hit distance the nearest hit's (or none), its confidence what the
reservoir's revalidation says.

**The denoiser** (`helio-pass-denoise`, feature `helio.denoise` in Helio's
default features; feature layer, group `denoise`; frame contracts only: the
signal and the frame's history, its guides included). One pass a signal,
its stages dispatches under GPU scopes. The rule its cost follows: light is
spread where the information is, once a traced sample; only what needs a
pixel's own geometry (reprojecting its history, reconstructing by its own
depth and normal) runs once a pixel.

1. **Temporal** (`Denoise: temporal`, a pixel). A pixel's new sample is the
   traced pixels of its four nearest blocks, each weighted by a tent one
   block wide over its distance (a partition of unity in expectation: one
   sample a block's area) times the geometry weight's surface and occluder
   stops (no hit-distance radius here: below); their sum is how much this
   frame says about the pixel. Its history is the bilinear read where
   `PIXELS` says it was, from the taps that held the surface, renormalised.
   The history's length times the producer's confidence times the
   footprint's valid share, plus this frame's weight, capped at
   `max_history`, sets the blend (weight over length).
2. **Records** (`Denoise: records`, a block, for blocks of 4 and 16): the
   temporal result at the block's representative pixel (its centre). Traced
   at every pixel, the pixels are the records and this stage is skipped.
3. **Spatial** (`Denoise: spatial`, a block), twice: an a-trous filter of
   the linear B-spline's 3x3 over the records at steps of s and 2s blocks,
   `s = sqrt(16 / length)` within 1 to 4: the footprint adds about s^2
   records to a record's `length` frames, so every estimate rests on about
   16 frames' worth (wide where history is short, which is the history fix
   ReLAX runs as a separate pass; one block from 16 frames on). Each tap is
   weighted by the geometry weight alone, its radius one for all taps: the
   harmonic mean hit distance of the footprint's records counted alike.
   Each record carries the samples' worth it holds (a weighted mean of
   estimates of n_q samples holds (sum w)^2 / sum (w^2 / n_q)) and its
   standard error (propagated as sum w^2 s^2 / (sum w)^2).
4. **Reconstruction** (`Denoise: reconstruct`, a pixel): the filtered
   records of the nearest 2x2 blocks (a tent of the block times the
   geometry weight; the 4x4 around them where none holds the pixel's
   surface, a feature narrower than a block), at the pixel's own normal,
   combined with the pixel's own temporal result by the samples each rests
   on: the pixel's own its length, the records' what the spatial stage left
   them times the share of the pixel's tent their geometry lets them serve
   (1 on an open surface, toward 0 at a contact, where the pixel keeps its
   own). The result is the output and the next frame's history: the
   spatial stage is recurrent (its work accumulates over frames, as
   ReBLUR's post-blur history does), while the moments stay the raw
   samples' (the temporal result's moments are the next frame's history
   moments, swapped, not copied), so the variance never sees the filter;
   the standard error is reported in the output, and no weight uses it.

The geometry weight, one function in every stage: irradiance caching's
error bound (Ward, Rubinstein and Clear 1988; Tabellion and Lamorlette
2004): a record holds over its harmonic mean hit distance (the error of
using it at distance d is d / R), never below its pixel's diagonal, so
near occluders keep contact detail and open ground filters wide; times a
normal stop (a sample's irradiance holds the hemisphere about its traced
normal; a normal tilted by t sees a share (1 - cos t)/2 of its
cosine-weighted hemisphere never sampled; weight 0 at 1/8, t = 41 degrees)
and a plane stop (two points of a surface whose normals differ by those 41
degrees leave each other's tangent plane by tan(20.7) = 0.378 of their
separation; a step by a whole one), and an occluder stop: no pixel on the
screen's segment between the two may hold a point standing out of both
receivers' tangent planes, toward their side, by more than the plane stop
accepts of a neighbour (0.378 of its distance and a pixel's footprint).
Positions are view space from the frame's guide depth through the inverse
projection (the raw depth recovered from the projection's z and w rows),
so the frame origin's size never enters; the guides' normals are frame
space, turned into view space by the view's rotation (a uniform) for the
plane and occluder stops.

What is stored and read back on the next frame (the temporal result, the
records, the history) is rounded to half floats stochastically (up with
the probability of the distance from the half below, so the stored mean is
the value whatever rounding the hardware's conversion uses), one hash a
texel, a byte of it a channel: this GPU's storage conversion truncates,
which a running mean amplifies by its length (the first convergence run
read 1.03% dark at 32 frames). The luminance's two moments are f32: with
the mean taken from the half-float state, the half floats' rounding alone
read as a 0.6% standard error on converged pixels.

`DenoiseSettings::max_history` (host setting `denoise.max_history`) is 32:
the running mean's noise falls as its square root (5.7 times; NRD's default
is 30) and light that changes without the producer's confidence lags by it
(half a second at 60 Hz). Memory a signal and view: 64 bytes a pixel
(history 16, the outputs; moments 2 x 16; the temporal result 16) and 24
bytes a record in two sets (three for blocks of 4 and 16; 20 before the
records carried their samples' worth): 232 MB at
1920x1080 traced at every pixel, 170 MB at a quarter, 142 MB at a
sixteenth (216, 164 and 140 at 20 bytes) (the first version: 166 MB at any block).

**What the per-block design replaced, and what it loses.** The first version
ran a 5x5 fix and four 3x3 a-trous iterations at steps of 1, 2, 4 and 8
blocks at every internal pixel: at a sixteenth it filtered the same
block-spaced lattice sixteen times a block, its cost following pixels while
its information follows traced samples, and it kept a packed depth and
normal copy (16 bytes a pixel a signal) beside the history's own. Against
it the new path loses spatial reach a frame (two iterations, three blocks at
the long-history step, against four reaching fifteen) and makes it up over
frames through the recurrence; on the tests' noisy wall the converged RMS
error at a sixteenth rose from 0.0119 to 0.0240 (bound 0.1016) and at a
quarter is 0.0089; on the bumpy surface and behind the moving box (the
tests that would show lost per-pixel detail or leaking light) nothing
changed: worst 0.0008 and 0.0000.

What it costs (RTX 3060, 1920x1080 internal, measured against the first
version under **Verified**): 1.59-1.76 ms at a sixteenth (was 4.20-4.33),
2.87-3.38 ms at a quarter (was 4.31-5.14; about 1.9-2.3 ms once the runs'
lower clocks are taken out), and 2.5 ms traced at every pixel in the one
undisturbed run (was 4.7). The spatial stage, which was three quarters of
the first version's cost, now follows the traced samples (0.33-0.37 ms at a
sixteenth against 3.05-3.18); what is left follows pixels: the temporal
stage and the reconstruction, each one pixel's own taps.

**Against NRD, SVGF and UE.** SVGF (Schied et al. 2017) is the same three
stages on colour: temporal accumulation with moments, a 7x7 spatial
variance estimate where history is short, five a-trous iterations stopped by
luminance (sigma_l = 4 of one sample's deviation), depth and normal (power
128); A-SVGF (Schied et al. 2018) adds temporal gradients from re-shaded
samples to shorten history where light changes. ReLAX (NRD) is SVGF's
stages with a history fix of the disoccluded pixels, an anti-firefly pass and
the input's confidence (`IN_DIFF_CONFIDENCE`) scaling accumulated frames;
ReBLUR (NRD) is a blur whose radius follows the hit distance and the
accumulated frames, with the same confidence input; both have an SH mode
(luminance first-order harmonics plus YCoCg chroma, resolved by the pixel's
normal) for signals traced below full resolution. UE5's Lumen screen-probe
gather filters in probe space and then temporally, and its denoisers
(`ScreenSpaceDenoise`) are per signal and private. Ours: the producer's
confidence plays A-SVGF's gradients and NRD's confidence input (the
producer knows, from its reservoirs' revalidation, more than a gradient a
denoiser re-traces can tell, and it costs the denoiser nothing); the
history fix is ReBLUR's (a spatial footprint that widens as the history
shortens) rather than ReLAX's separate pass; the spatial stage is recurrent
as ReBLUR's is, with the variance kept on the unfiltered samples as SVGF
keeps it; there is no luminance stop (SVGF's and ReLAX's), which weighs a
heavy-tailed signal's bright samples down ("The denoiser keeps the mean");
the hit distance enters as irradiance caching's record radius, which is
ReBLUR's hit-distance radius with a bound it derives from, pooled over the
footprint so that a record's own ray lengths do not weigh its light;
the SH representation is NRD's, made exact at each sample's traced normal so
samples from neighbours average. History is judged once, by the frame's
history, for every temporal pass. What it lacks against NRD: an
anti-firefly pass and history clamping on fast lights (the producer's
confidence is the answer here; a producer without one lags by max_history),
and a separate specular path (r4's addition).

Their costs, as published (NRD v4.18.0's README, "Performance": RTX 4080 at
2560x1440 native, a diffuse and a specular signal together): ReBLUR 2.55 ms
(3.40 in its SH mode), ReLAX 3.25 ms (4.80 SH), SIGMA's shadow 0.40 ms. NRD
publishes no diffuse-only or below-full-resolution figure, and an RTX 4080 is
about 2.5 to 3 times an RTX 3060, so the comparison is of order only:
ReBLUR's SH mode spends 0.92 ns a pixel for two signals on a 4080, about
2.5 ns on a 3060 by that ratio, 5.2 ms at 1920x1080 (half of it, roughly,
one signal's); ours is under **Verified**.

**Verified** (RTX 3060):

- **Where a pixel was** (`helio-pass-history` `tests/pixels_gpu.rs`: an
  analytic scene ray-cast on the CPU in f64 into the G-buffer, cameras
  uploaded as the renderer does, 96x64): every pixel's previous position
  against the f64 projection through the previous camera, and every 2x2
  that lies on one quad found whole (all four taps) or, where it was hidden,
  found empty. Worst position error: still camera (jitter moving) 0.0005
  px, moving camera (translated and turned) 0.017 px, an object moving by
  itself (its motion vectors) 0.0004 px with 85 uncovered pixels finding no
  tap, a frame-origin jump of (1024, 0, -2048) m 6,371 km from the world's
  origin 0.0023 px; bound 0.02 px (a half-float offset's step under 16 px
  of motion plus f32 evaluation). A view's first frame has no history.
- **The denoiser** (`helio-pass-denoise` `tests/denoise_gpu.rs`, the same
  fixture, 160x112, the tests the producer; every test at blocks of 1, 4
  and 16, bounds unchanged by the per-block rework): a signal with 50%
  relative noise a sample over a lit wall (a gradient and a normal term)
  converges to the field with a relative RMS error of 0.0056 / 0.0089 /
  0.0240 at blocks of 1 / 4 / 16 (the first version: 0.0055 / - / 0.0119;
  bound 0.1016: sigma / sqrt(32) plus 15%; the bound is the temporal mean's
  alone) and a mean of +0.0000 / +0.0018 / +0.0028 (first version +0.0001 /
  - / +0.0019; bound 0.0102); a wall behind a box that moves by itself for
  three frames, then a camera that pans and turns for four: worst error
  0.0000 at every block (first version 0.0005; bound 0.02); confidence 0
  after a change: 2.0000 at every block for a field stepped from 1 to 2 and
  a history of length 1, against 1.0312 without a confidence (1 + 1/32);
  over a bumpy surface: worst error 0.0008 at each pixel's own normal at
  every block, where one value a block would err by 0.165; two runs, one
  moving the frame's origin mid-run: worst difference 0.0009, mean 0.00001
  at every block (bound five half-float steps, 0.0024). Readiness is the
  history's length read back under a wall-clock guard. (After the mean
  fix, bounds unchanged: 0.0054 / 0.0090 / 0.0244 and +0.0002 / +0.0017 /
  +0.0026; the moving box and camera 0.0000; the bumpy surface 0.0008; the
  origin change 0.0008. Its own tests: "The denoiser keeps the mean".) `pixels_gpu` (5
  tests, with the history now publishing this frame's guides) passes.
- **TSR**: its suite (14 tests, the reprojection and origin cases driven
  through the history's pass) passes; its image on real frames: below.
- **Cost** at 1920x1080 internal, pipelined (two frames in flight), the
  graph's GPU timings, medians of 200 frames after 100, a signal whose noise
  is redrawn every frame (four half-float draws uploaded in turn); the first
  version's test exe kept and run interleaved with the new one under the GPU
  lock (base, new, base, new; each run all three blocks). Clocks sampled
  through each run: idle-low (210-495 MHz core) in most samples and 1777-1935
  MHz in 2 to 3 of them for every run alike; the runs at a quarter came out
  at lower clocks for the new build (`HistoryPixels`, the same code in both,
  0.75 and 0.84 ms against 0.51 and 0.57), so its stages there read about
  1.5 times high. Medians, run 1 / run 2, ms:

  | stage | first version, 1/16 | new, 1/16 | first version, 1/4 | new, 1/4 | scales with (new) |
  |---|---|---|---|---|---|
  | HistoryReprojection | 0.002 / 0.002 | 0.002 / 0.002 | 0.003 / 0.002 | 0.003 / 0.003 | views |
  | HistoryPixels | 0.51 / 0.51 | 0.53 / 0.51 | 0.51 / 0.57 | 0.76 / 0.85 | internal pixels |
  | Denoise: guides | 0.19 / 0.20 | deleted | 0.19 / 0.20 | deleted | |
  | Denoise: temporal | 0.68 / 0.68 | 0.79 / 0.72 | 0.72 / 0.85 | 1.13 / 1.38 | internal pixels (4 sample, 4 history taps; writes the moments) |
  | Denoise: fix | 0.28 / 0.28 | deleted | 0.30 / 0.30 | deleted | |
  | Denoise: records | | 0.063 / 0.063 | | 0.20 / 0.20 | traced samples |
  | Denoise: spatial | 3.18 / 3.05 | 0.37 / 0.33 | 3.10 / 3.78 | 0.72 / 0.88 | traced samples x 2 iterations x 9 taps |
  | Denoise: reconstruct | | 0.52 / 0.47 | | 0.75 / 0.90 | internal pixels (4 record taps) |
  | Denoise in all | 4.33 / 4.20 | 1.76 / 1.59 | 4.31 / 5.14 | 2.87 / 3.38 | |

  Traced at every pixel the test uploads 16 MB of signal a frame and three
  of the four runs were disturbed (`HistoryPixels` at 5.3-5.5 ms); the
  undisturbed tenth percentiles: the first version 4.70 ms (temporal 0.63,
  guides 0.19, fix 0.30, spatial 3.58), the new one 2.55 ms (temporal 0.57,
  spatial 1.58, reconstruct 0.39). Convergence, old against new: above
  (relative RMS 0.0055 / 0.0119 at blocks of 1 / 16 before, 0.0056 / 0.0089
  / 0.0240 at 1 / 4 / 16 now, every bound unchanged). Nothing depends on the
  scene.
- **Gates**: Helio's layer gate (0 violations), `cargo test -p helio-core
  --test wgsl_validation`, `helio-frame-api` `tests/layouts.rs` (the
  reprojection's 128 bytes against WGSL, every new snippet function
  validated), the crates' unit tests, clippy (no new warnings), per-package
  fmt, Pulsar `cargo test -p one_owner` (its layout heuristic flagged
  `HistoryReprojection`'s first fields as the portals' `GpuPortalStage`; the
  struct leads with its `info` header instead).

**TSR on the history's reprojection, real frames** (RTX 3060; r2's merged
builds `C:/t/int2/voxel_craft_r2_merged.exe` and `voxel_flight_r2_merged.exe`
against this branch's; a warm-up run of the new exe first; mean |difference|
of the images, 0 to 1):

| | same build | this against r2 | luminance |
|---|---|---|---|
| voxel_craft, 4 poses, 8-frame bursts, no clouds | r2 twice: 0.00004-0.00056; this twice: 0.0006-0.0015 (p99 0.013) | 0.0006-0.0018 (p99 0.016) | -0.06% to +0.01% |
| Earth ground hold, 1600x900 | this twice: 0.00075 | 0.00088-0.00108 (p99 0.0078) | 0.3575-0.3576 against 0.3578 |

The new build's own two voxel_craft runs differ as much from each other as
from r2: its terrain kernels compiled cold on both runs (49.5 and 65.7 s for
the generation and draw pipelines, shots at frames 6093 and 7060), where
r2's binary found them in the driver's cache (shots at frame 1755), and the
game's clock follows real frame time, so the bursts are taken at different
times of day. The reprojection itself is TSR's former composition, moved;
TSR's suite, which checks the uniform's values, the origin cases and the
resolve's image, passes. A second round (r2 and this alternated again) was
stopped while the GPU queue was held for hours by other work.

**Open**:

- No producer yet: the denoiser runs on real frames from step 4 (built:
  "Indirect diffuse at a pixel", below, its first producer). The
  reference path tracer's one-frame first-bounce estimate as a real signal
  (against its converged image, the cathedral) was not built: it needs the
  reference to export each path's first-bounce direction and incident
  radiance, which is step 4's producer's own data path; step 4 builds it
  with its producer.
- The previous position is a half float offset: 0.5 px steps past 1024 px
  of motion a frame (a fast turn), enough for a filter, not for TSR (which
  keeps its own).
- 80 bytes a pixel a signal: the guides could be shared by every signal of
  a view (one texture a view).
- Captures build the denoiser into their graphs like the primary view when
  a producer is in them; nothing does today.

#### The denoiser keeps the mean

**The fault.** The first version weighed what it averaged by the light it
carried. The spatial stage's luminance stop (SVGF's edge-stopping weight
`exp(-|y_p - y_q| / (3 sqrt(s_p^2 + s_q^2)))`, with the standard errors
estimated from each pixel's own moments), the reconstruction's
inverse-variance mix of a pixel's own estimate with the records (with a
bias term from their difference), and the skip of a record whose standard
error was under the half floats' rounding all depend on the samples' own
values. For a positive heavy-tailed signal (a few rays a block that find a
bright source among many that find little) the estimates holding a bright
sample are exactly those with a high estimated variance, so they count
less: the mean is biased down, and the recurrence (the output is the next
frame's history) carries the bias into the history. Three more terms
showed once those were found: chroma ratios were averaged with the
samples' weights rather than their luminance (a sample of no light pulled
the colour toward grey); the temporal stage held each sample over the
radius of its own hit distance, and a ray's length goes with what it
found (a near lamp is short), so a bright sample's own short radius
counted it less; and the plane stop dotted frame-space normals with
view-space offsets, which only agree when the camera is not turned (no
bias, a lost neighbour on a tilted view's floor).

**The rule.** A weight that decides how much a sample counts must not
depend on that sample's light. If every weight of a normalised filter
`out_i = sum_j w_ij y_j / sum_j w_ij` is independent of the values, then
for a stationary signal of mean `mu` its expectation is `sum_j w_ij mu /
sum_j w_ij = mu` at every pixel, whatever the noise's distribution; a
convex mix of two such estimates with weights independent of both keeps it
too, and so does the recurrence. Chroma is a ratio: its luminance-weighted
mean `sum w Y c / sum w Y` is the ratio of the means of `Y c` and `Y`, so
colour is kept the same way.

**What the stages weigh by now.**

- Temporal: the tent of the block, the surface stops (normal and plane, in
  view space) and the occluder stop; no hit-distance radius (contact
  detail finer than a block is the producer's, the gather's near field,
  which is also why the gather floors its hit distance at a block's
  footprint). The blend by weight over history length is unchanged.
- Spatial: the B-spline taps, the surface and occluder stops and the
  distance against one radius for the whole footprint: the harmonic mean
  hit distance of its records counted alike, so one record's ray lengths
  move it by a ninth rather than set its own weight.
- Reconstruction: tents, geometry and the occluder stop, mixed with the
  pixel's own estimate by the samples each rests on (the pixel's length;
  the records' samples' worth carried through the spatial stage, times the
  share of the tent their geometry serves).

The standard error is still computed from the raw samples' moments and
reported in the output (output 0's w); nothing weighs by it.

**The occluder's foot.** Two points of one floor either side of a thin
wall are close, on one plane, with one normal: depth, normal and plane
weights pass them. The hit distance cannot tell them apart either: their
hemispheres meet the wall alike, mirror images (it does narrow the filter
near any occluder, both sides equally). What distinguishes them is the
wall standing between, which the frame's depth shows: the occluder stop
walks the screen's segment between the two pixels and fails where a
pixel's point stands out of both receivers' tangent planes, toward their
side, by more than the plane stop accepts of a neighbour (0.378 of the
chord and a pixel's footprint). It depends on geometry alone, so it keeps
the mean; it is symmetric in the two pixels. The gather's spatial reuse
makes the same test (`seen_between` in `gi_gather.wgsl`) on its own
positions. The first version's luminance stop also separated the two sides
in the synthetic test, but only by their light, which is the bias above.

**How others do it.** SVGF (Schied et al. 2017) and ReLAX (NRD) stop the
a-trous filter by luminance against the variance estimated from the
samples' moments: the weight falls with the sample's own estimated
variance, biased for heavy tails exactly as above (SVGF's paper accepts
it; ReLAX adds an anti-firefly clamp of the centre to its neighbourhood's
range, a bias by design, and clamps history to the neighbourhood after a
disocclusion). ReBLUR (NRD) has no luminance stop in its blur: its radius
follows the hit distance and the accumulated frames, its weights are
geometry, normal and hit distance; its hit-distance weight compares the
centre's and the tap's normalised hit distances, which correlates with
light where ray length does; its anti-firefly and history clamping are
optional and biased. Ours is ReBLUR's choice made exact: geometry,
counts and a hit distance pooled so that no sample's own ray sets its
weight, and the occluder taken from the depth rather than from the light.

**What it costs and what it leaves.**

- *Tests* (`denoise_gpu.rs`, 160x112, every block; each fails on the first
  version):
  - *A heavy-tailed signal keeps its mean*: a uniformly lit wall, one block
    in 4 and in 16 carrying all the light (relative deviations 1.7 and
    3.9), log-normal of log-deviation 0.75 and 1.5, and one in 16 whose
    bright samples' rays are short (5 cm against 5 m), each 4x4 group's
    factors averaging 1 every frame so the input's image mean is the
    field's; a 32-frame mean of the output's image mean once the history
    has settled, bound 2%, chroma 0.01. Now 0.998-1.001 everywhere (the
    short rays 1.0001 / 1.0002 / 1.0009 at blocks of 1 / 4 / 16), chroma
    0.0000-0.0002. The first version: one in 4 0.940 / 0.992 / 0.992, one
    in 16 0.578 / 0.914 / 0.928, log-normal 1.5 0.957 / 0.988 / 0.986, the
    short rays 0.571 / 0.329 / 0.037, chroma off by 0.12-0.15. With each
    sample held over its own hit distance (this design before the pooled
    radius) the short rays read 0.974 / 0.982 / 0.967.
  - *After a disocclusion*: the wall jumps between 4 and 6 m every 12
    frames (history rejected everywhere, checked), 32 jumps, the output
    over the input on each of the four frames after a jump, bound 3%: one
    in 16 0.989-1.000, log-normal 0.992-1.003 (the jumps' spread 0.02-0.05
    at a sixteenth). The first version: 0.13-0.61 and 0.82-0.96.
  - *An occluder's foot*: a wall 1 m tall and 2 cm thick seen from above,
    the floor lit before it (1) and dark behind (0.2), samples traced into
    the scene for their hit distance and floored at a block's footprint as
    the gather does, 50% noise; light carried behind within 10 cm and dark
    carried before, each as a share of the far floors' difference, bound
    2%: now -0.0006 / +0.0007 / +0.0000 and -0.0014 / +0.0020 / -0.0008;
    the far floors' relative RMS 0.022 / 0.042 / 0.044 (the first version
    0.029 / 0.046 / 0.052; bound 0.10). Without the occluder stop this
    design (as it stood before the temporal and spatial radius changes
    below) carries 4.7% behind and 3.4% before at a sixteenth. The first
    version passes it (-0.0006 / +0.0044 at worst): its luminance stop
    told the sides apart by their light.
  - The existing tests pass with their bounds unchanged (above).
- *GI's `pixel_gather`* (6 tests pass): the occluder test's output over
  input, still camera / swaying, plain rays 0.848 / 0.767 before,
  0.998-1.011 / 0.988-0.993 now (two runs); the resampled signal 0.951 /
  0.969 before, 0.996-1.011 / 0.980-0.986 now (asserted within 3%). Found on the way: held over its own hit distance
  in the temporal stage the plain rays read 0.962 / 0.957; over the
  pixel's history's, 0.985 / 0.981; with no radius there, 0.995 / 0.991.
- *Cost* (`indoor_cathedral_gi`, the default quality, a sixteenth; frames
  60 to 1500 pipelined, `HELIO_CAPTURE_TIMINGS`, medians; the first
  version's and this exe alternated, the denoiser's shader from disk in
  both): Denoise 0.706 / 0.705 ms before, 1.004 / 0.962 now (temporal
  0.29 to 0.38, spatial 0.17 to 0.22, reconstruct 0.20 to 0.33-0.34,
  records 0.036 alike). The 1920x1080 stage test (`the_stages_cost`, the
  GPU at the low clocks other sessions' runs left it in: the first version
  3.36-3.42 ms at a sixteenth, 3.89-3.92 at a quarter) reads 5.29-5.44 and
  5.36-5.44, and 3.56 and 4.40 without the occluder stop: the stop is
  almost all of the extra (the samples' worth, the pooled radius, the
  luminance-weighted chroma and the view-space normals are +6% and +13%).
  What the extra buys is the stop the mean needs: with no luminance stop,
  only it keeps a thin occluder's two sides apart (above). Three cheaper
  forms were measured and not kept: the tile's depth in workgroup memory
  (5.60), a tile-wide one-surface test that skips the walks (6.90) and the
  walks gathered before the blend's state is live (6.64); the walk's memory
  traffic is not what costs, and the next step is a profiler capture of the
  stages (not done here).
- *Noise on real frames* (the cathedral, view 23, 64 frames from frame
  3000, still camera; reference-free): mean luminance 0.01542 before,
  0.01694 now (+9.9%, the light the first version lost; against the path
  tracer's reference 0.954 before, 1.048 now); the per-pixel temporal
  standard deviation over the mean 0.0116 before, 0.0140 now over 16
  frames (0.0194 and 0.0236 over 64), the frame mean's 0.0023 and 0.0012;
  boiling (a 40x40 tile 30% over its burst mean in a frame) none in either.
  Per pixel the noise is about a fifth higher: the first version's weights
  damped exactly the bright samples, which was its bias; on the synthetic
  tests (no heavy tail) the noise is unchanged on the front-facing wall
  and lower on a tilted floor (the view-space plane stop).
- *Not verified*: voxel_craft and the planets with this denoiser (the
  example and the setups that use it all run the same pass); an XR pair of
  views. The gather keeps its own copy of the denoiser's surface stops and
  its own `seen_between` (`gi_gather.wgsl`, frame space); one function for
  both belongs in a frame snippet, which the merged gather would have to
  take (not done here).

#### The radiance cache

Helio `helio_frame_api::rays::radiance_cache` (the contract, beside the ray
streams and the light samples), the GI plugin's `cache.rs` and
`gi_cache.wgsl` (the publisher), GI's light passes (the probes' reader and
the cache's own lighting). Branch `render/r3-cache`.

**The contract.** Keys and rows (Rust and WGSL, layout tests
`tests/layouts.rs`; the WGSL against its Rust mirror on a GPU,
`tests/radiance_cache_gpu.rs`):

| Row | Bytes | Fields |
|---|---|---|
| `RadianceCacheFrame` (`radiance_cache_frame`, a uniform) | 64 | the frame's origin in whole 1024 m (64-bit two's complement an axis), the slots (0: no cache), the frame's number, the origin's rest below 1024 m (0 for an engine frame), the eye angle, the eye in the frame's space |
| `RadianceCacheSlot` (`radiance_cache_slots`) | 8 | `state` (the key's checksum, 0 empty), `used` (the frame a reader last marked it) |
| `RadianceCacheKey` | 28 | `tag` (level 5 bits, side 3, space 9), the world cell: x, y, z, 64 bits each |
| `RadianceCacheEntry` (`radiance_cache_entries`, a slot each) | 64 | key, point (15 bits an axis in the cell: its top 10:10:10 here, its low 5:5:5 in `history` from bit 16), normal (`scene_pack_unit`), history (bits 0..16: updates folded, direct 8 bits, indirect 8; 0 no value), direct irradiance (3 f32, W/m^2), indirect irradiance (3 f32) |

`//!use frame_radiance_cache` (with `frame_ray_stream`), for a shader that
binds the three under those names:

- `radiance_cache_read(p, n, footprint, space, reach)`: what the cache
  holds for a hit at frame point `p`, the face's normal turned toward the
  ray `n`, the ray's footprint there and its space: `direct`, `indirect`,
  the `level` it came from, the cell's `history` (updates folded a part:
  how long its value has held still, a reader's confidence), `found`
  (false: nothing at the hit's level or up to `reach` levels coarser).
- `radiance_cache_mark(p, n, footprint, space)`: the hit's cell in use this
  frame, claimed with the hit's point and normal where no slot holds it,
  and the cell `ANCESTOR` (3) levels coarser.
- For a publisher: `radiance_cache_key`, `_find`, `_level`, `_size`,
  `_cells`, `_entry_point`, `_side`. Rust: `RadianceCacheFrame::{new, level,
  key, origin_cell}`, `RadianceCacheKey::{new, cell, bucket, checksum}`,
  `cell_size`, `stand_in(device)` (a frame without slots: a read finds
  nothing, a mark does nothing).

The api crate holds no `static`, `thread_local!` or `inventory::collect!`.

**The hash and its levels.**

- **A cell**: level `l` is `2^(l - 6)` m wide, 17 levels from 1/64 m to
  1024 m. The coarsest is the frame origin's step, so the origin is a whole
  number of cells at every level: the cell of a frame point is the origin's
  cell (whole 1024 m shifted by `16 - l` bits, 64-bit) plus
  `floor((p + rest) / size)`, an origin change moves no cell, and a point an
  Earth's radius or 10^12 m from the world's zero keys as one near it (its
  high words carry). An origin off the 1024 m step (a test's, a host's own)
  keeps world cells too, its rest carried in the frame. The finest: a sixth
  of the 10 cm voxel, below what any reader's indirect light resolves.
- **The level** of a read: `round(log2(max(footprint, a * d)))`, the
  coarser of the ray's footprint at the hit and the eye angle `a` at the
  hit's distance `d` from the eye. The eye angle is the publisher's, from
  its capacity: cells at least `a d` wide hold at most `4 pi / a^2` cells a
  level for surfaces all round the eye, over 17 levels half the slots: `a
  = sqrt(4 pi 17 / (slots / 2))`, 0.040 rad at the default 2^18 slots (a
  10 cm cell 2.5 m away, a metre 25 m away). A probe's ray (0.443 a metre
  of footprint) reads cells about half its distance wide; a pixel's ray
  near the eye the eye angle's.
- **The side**: the dominant axis of the face's normal turned toward the
  ray, and its sign. A thin wall's two sides are two keys whatever the
  cell's size; a floor and a wall in one cell are two cells.
- **The space**: the ray's (`SceneRay::space`), so two coordinate spaces
  at one place are two cells, and a cell's rays travel in its space.
- **The hash**: buckets of 16 slots (the bucket a 32-bit hash of the key
  names, a second hash its checksum); a key sits in its bucket's first slot
  that holds it or is empty. A read compares the checksum and then the
  whole key, so a checksum collision costs a miss, never a wrong value. A
  claim is a compare-exchange of the slot's state; a claim racing another
  of the same key finds its checksum and marks; one racing a different key
  moves on. A full bucket keeps a cell out (a read answers from a coarser
  one): at half load, 0.4% of keys (Poisson's tail past 16 at 8 a bucket).
- **A cell's point**: the first marking hit's, 15 bits an axis in the cell
  (a 1024 m cell's within 3 cm, where 10 bits put it up to a metre off its
  hit, outside a room near its corner); one mark in
  8 of a cell in use moves it to the marking hit (a uniform pick among the
  hits that read it, over time), so a coarse cell's light is gathered over
  the places its readers meet it rather than at one of them. Its rays start
  off it along its normal by f32's rounding there and by the point's own
  precision (`sqrt(3) / 65536` of the cell).
- **A mark's ancestor**: the cell 3 levels coarser is marked with it, so a
  new fine cell's read answers from a cell 8 times wider while it has no
  value of its own (1/64th as many cells of a surface), for a reader whose
  reach allows it.

**The update** (GI's node, after the probes' rays; costs below):

1. `RadianceCache`: `cache_scan` over the slots (each cell's frames since
   it was marked, from its slot, and since it was lit, from its life: the
   publisher's 32 bytes a slot beside the contract's rows, the frame it was
   lit plus one and the fold's change tests; the two passes over the slots
   read a slot's 8 bytes and its life's first 4, never an entry's line), into histograms of
   powers of two),
   `cache_plan` (one invocation: what goes, what is lit), `cache_select`
   over the slots (evicts, lists), `cache_rays` (4 rays a listed cell into
   GI's stream after the probes', cosine distributed about its normal in a
   2x2 stratified set (the frame BSDF's frame and cosine direction), from no
   width spreading by `sqrt(2 pi / 128)`, the share of the hemisphere one
   ray of the cell's history stands for), `cache_header` (the stream's
   count, the statistics). A first version started the rays as wide as the
   cell (the stream contract's convention for a ray leaving a hit): their
   hits then keyed cells as coarse as the cell, and a floor's cell under a
   5 cm slab took the slab's lit top into it (a 1 m cell under it read 2.5%
   of the top's light); from no width their hits key cells at the detail the
   cell's estimate resolves, or the eye angle's.
   - Evicted: a cell no reader marked for 128 frames (twice the longest a
     probe of GI's coarser levels goes between updates, 41 frames, a power
     of two above), and past half the slots the least recently used, whole
     histogram bins, never a cell marked this frame or the last.
   - Lit: `rays / 4` cells at most, never-lit cells first, then by how long
     ago they were lit.
2. The stream's tracers, once for the probes' rays and the cache's: one
   stream, so the per-stream fixed costs section 4 measured (a dispatch a
   volume on the terrain, a binned material pass a program on meshes) are
   paid once. Not in runs: a probe's rays share no origin (section 3), and a
   stream's runs are the whole stream's.
3. `GILight`: each cache ray's hit lit by the cache's indirect part there
   (for the bounces after), its emission, the light through it; the sky on a
   miss (`sky_radiance`, the one call GI's rays make); a light query at the
   hit and one at the cell's point. The probes' rays are lit here too
   (below). Every hit marks its cell.
4. The light samplers, `GIShadow` (the hit's sample off its surface, the
   cell's as irradiance there, a shadow ray each, two a cache ray at most,
   packed), the shadow stream's tracers.
5. `RadianceCacheFold`, a lane a listed cell: the indirect part `pi` times
   its rays' mean radiance, the direct part its light samples' mean, each
   the plain mean of its updates since it last restarted, 32 at most. A
   change test restarts a part at its **next** update: a fast mean (4
   updates) and a slow one (the history's) of each update's luminance,
   clamped from above at the fast mean plus 3 mean absolute deviations, more
   than 1.2 deviations apart (3 standard deviations of their difference:
   `sqrt(1/7 + 1/63 - 2c)` of an update's, `c` the two exponential means'
   covariance) and 5% of the slow mean, once 8 updates measured them. A
   first version shortened the history on the samples themselves (a jump: an
   update far off the fast mean restarted the part at it; a lag: the value
   moved to the fast mean, ReBLUR's anti-lag, Zhdan 2021). Weights chosen by
   the samples they weigh bias the mean: a history restarted at a bright
   update keeps it and weighs it more. On the heavy-tailed updates of one
   light sample among the cathedral's many lamps that put its indirect light
   at twice the reference (simulated: x2.0 where 2% of samples are 50 times
   the rest). The test now sees an update only through the clamp and decides
   only about updates it has not seen: simulated within 1.2% of the plain
   mean on Gaussian, shadowed (Bernoulli) and those heavy-tailed updates, a
   light switched off restarting its parts within about 3 updates. On a
   frame a tracer answered short, or no ray met anything, the cells hold, as
   the probes do.

Settings (`GiSettings`, the `gi` host group): `radiance_cache_slots` (a
power of two, 2^12 to 2^24; 2^18 by default) and `radiance_cache_rays`
(1024 up; 32,768 a frame by default: at the load the eviction keeps, half
the default slots' cells, every cell lit at least once in the time the probes
that read it take to come round, the finest levels' 21 frames: `2^17 * 4 /
21` = 24,966 rays, the power of two above). Statistics:
`rays.radiance_cache`, `radiance_cache.cells` (in use), `.new`, `.evicted`,
`.lit`; GI's stream counts both kinds of ray under `rays.gi`. The debug
view `Radiance Cache` (39, GI's resolve): the cell a pixel-wide read finds
at each pixel, its irradiance / pi, cell edges darker, dark purple where
none holds a value.

**The probes read the cache.** A probe ray's hit is `albedo * (direct +
indirect) / pi + emission`, plus its transmission share of the far side's
cell (read with the face turned away), at the probe ray's footprint (0.443 a
metre), with a reach of one level (a cell twice as wide). GI's light queries
and shadow rays a probe hit are gone: the light batch and the shadow stream
are the cache's alone (65,536 rows each at most against 174,080 before).
Where the cache has no value within that reach (a cell new this frame or the
last), the probes' own irradiance at the hit stands in, as a bounce did
before; a probe more than 1/32 of whose reflecting hits had none holds that
update (its irradiance off by at most that share otherwise) rather than
learn a world lit by its own previous state. Why the reach is one level: a
first version let probes read a new cell's ancestor (8 times wider); their
first updates took it into their histories (alpha 1), and the ancestor of a
floor cell by a thin wall spans both sides: the VG suite's solid cut-out
card let 11% of the open room's light into the dark half (`far < 0.001`
before), a black roof's shade under it 0.112 of the open ground against a
true 0.021. Read one level up, a probe whose cells are new holds an update
cycle instead (about 20 frames on the finest levels); at a scene's start
GI's probes come in after their first cycle. The cache rays' hits read with
the same reach.

**Against the alternatives** (section 5, "How others do it"):

- **SHaRC** (NVIDIA RTXGI 2.x): a spatial hash of radiance keyed by a
  quantised position at a level from the distance to the camera, filled by
  a few percent of the pixels' paths and read where paths end. Its key is
  a packed position of bounded range (positions far apart alias), cells
  hold radiance and are lit only by the camera's paths, so what the camera
  does not see goes stale. Here cells are exact 64-bit world cells (an
  origin change and an Earth's far side key alike), sides and spaces are
  keys, and the cache lights its cells with its own budget wherever readers
  marked them (probes off screen, reflections).
- **AMD GI-1.0's hash grid cache**: the closest design (cells at hit
  points, lit by rays of their own, read by screen probes); it serves one
  world space and keys cells by position and distance from the camera.
- **Kajiya's irradiance cache**: entries in camera-centred cascades, each
  tracing its own rays, read by ReSTIR GI; bounded to the cascades' reach,
  entries on a grid about the camera rather than at surfaces' sides.
- **Lumen**: the surface cache is cards a mesh, cheap to read and exact in
  texture space, with no answer for 10 cm voxels, foliage or anything
  without cards; its world radiance cache is probes, which cannot hold a
  wall's two sides apart. Here every tracer's hits are cached alike.
- **Neural radiance caching** (Müller et al. 2021): an MLP trained on path
  suffixes each frame; better generalisation between cells, but it needs
  fused tensor-core inference and training wgpu has no fast path for.

What it costs and scales with: memory 104 bytes a slot (64 the entry, 8 the
slot, 32 the publisher's life), 26 MiB at 2^18 slots, with the slots (a
setting), never the world; a frame's update two passes over the slots
(their 8 bytes and the lives' first 4: 6 MiB read at 2^18), the rays, light
queries and shadow rays of the budget (a setting), never the screen; a
read one bucket of 16 slots a level it walks (one, mostly), a mark two.
Cells in use follow the surfaces readers reach and their distance from the
eye (at most `4 pi / a^2` a level), not the world's size: an Earth-sized
planet costs what a room does.

**Verified** (RTX 3060; every test waits on a signal under a wall-clock
guard, never a count of frames):

- The contract (`helio-frame-api`): layout tests of the four rows; on a GPU
  (`radiance_cache_gpu`), keys, levels, buckets and checksums from the WGSL
  equal to the Rust mirror's bit for bit over 4,096 random hits (footprints
  a millimetre to tens of metres, every side, four spaces) at frame origins
  at the world's zero, an Earth's radius in every axis's sign, 10^12 m (the
  high words carrying) and off the 1024 m step, each hit inside its world
  cell (f64); marks claim each cell once with the hit's point (back within
  1/32768 of its cell) and normal, and its ancestor; a read finds nothing
  before a value, the publisher's value after, the ancestor's where its own
  cell has none, nothing on the other side or in another space, nothing in
  the stand-in. Unit tests: an origin change moves no cell at any level (the
  zero, an Earth's radius, 10^12 m), any origin keys world cells, the level
  rule, sides, spaces and levels apart, buckets spread.
- The cache (GI's `tests/radiance_cache.rs`, over the analytic tracer and
  the lights' sampler, read and marked by a pass outside GI through the
  contract):
  - **A closed box with one point light** (6 x 4 x 6 m, albedo 128/255)
    against a radiosity solution on the CPU (patches of a third of a metre,
    Jacobi to convergence; the solver itself checked by the energy it must
    conserve: the walls' direct flux 1.0008 of the light's `4 pi I`, their
    absorbed flux 1.007): direct within 0.2% at four wall points (one delta
    light, every sample exact), indirect -6.8%, -6.4%, -7.9% and -5.0%
    (bound 10%: the estimate's noise over 128 frames, about 2%, and the
    coarser cells its rays read and the probes' irradiance standing in where
    a hit's cell is new). All four low: the mean over the 128 frames after
    the cells settle (94 frames) still holds the later bounces' rise, and a
    coarse cell's one point; the biased first fold read -0.9% to -5.6%
    (its upward bias hid part of it). Not separated further.
  - **A thin plate's two sides** (5 cm, read with a metre's footprint):
    the top 2.000 (the sun's 2), the underside exactly 0.
  - **A light switched off**: the four points under 5% of their light after
    103 frames (24,087 cells in use, 8,192 lit a frame: a cell updated every
    3 frames, so about 34 updates, the bounces falling one after another);
    bound 285 frames, what a part whose history never shortened would take
    (95 updates).
  - **An origin change** (the eye at a 1024 m boundary, the origin moving
    1024 m): every settled cell keeps its slot, key, history and values bit
    for bit, and the reads at the same world points their values.
  - **Earth scale**: a glowing furnace of albedo 1/2 converges to `2 pi`
    within 1% in 311 frames at the world's zero, 316 at an Earth's radius in
    x, 313 at an Earth's radius in -y and -z, each read at the same level.
  - **Eviction and re-use** (2^13 slots, 503 probe cells): a set of 2,880
    keys (96 spaces of 30 points) lit, then a second as large: lit 4 frames
    after, 6,318 cells evicted meanwhile, the first set's reads find nothing,
    and read again it comes back.
  - **Two spaces at one place**: a wall point in each space reads its own
    room's light, `pi` and `3 pi`.
- GI's suites on the cache: `probes` 15 of 15 (they wait for the probes
  they check: a probe whose hits' cells are new holds), `mesh_gi_graph` 2 of
  2, `vg_ray_proxies` 3 of 3 (the solid cut-out card's far half 0.000000 and
  0.000122 of the open room's 2.044, as before), lib 13 (constants against
  their derivations, the shaders' copies, settings).
- Gates: Helio's layer gate (0), `helio-core`'s `wgsl_validation` (the
  snippet in its list), the frame-api suites (lib 30, layouts 5,
  `radiance_cache_gpu` 2, `surface_gpu`, `bsdf_gpu`), clippy on
  `helio-frame-api`, `helio-pass-gi`, `helio-gi-api` and `gi-examples` with
  no warnings of theirs, `cargo fmt -p` on every package touched,
  `one_owner`, `helio-pass-voxel-planet`'s tests compile.

**Measured** (RTX 3060, release builds; `indoor_cathedral_gi` held still,
against r2's reference images `scripts/r2/reference_compare.py`; the timing
A/Bs interleaved, two runs a build, after a warm-up of the new exe's name):

| | base (r2 merged) | the cache |
|---|---|---|
| Cathedral indirect diffuse / reference, frame 240 | 0.805 | 0.801 |
| ... frame 1200 (settled) | 0.884 | 0.994 |
| Cathedral whole frame / reference, frame 240 | 0.746 | 0.801 |
| ... frame 1200 | 0.797 | 0.852 |
| Cathedral indirect mean relative error, frame 1200 | 0.231 | 0.269 |
| Cathedral GI passes (resolve aside) | 1.039, 1.057 ms | 1.232, 1.228 ms |
| voxel_craft GI passes (four poses, median of 240-frame windows) | 2.92, 2.83 ms | 3.16, 3.06 ms |
| voxel_flight Earth ground hold, GI passes | 2.89 ms | 3.42 ms |
| voxel_flight Moon ground hold, GI passes | 2.77 ms | 3.39 ms |

The cathedral's indirect light was 12% short of the reference with the
probes' own light queries (one sample a probe hit, its shadow ray, its
history the probe's); read from the cache it is within 1% once the cache
settles. Per pixel it is noisier (the mean relative error 0.269 against
0.231): a coarse cell's one light sample an update among many lamps
shows as blotches cell by cell until its 32 updates average it, and two
runs of the cache differ more than two of the base (0.0041 mean against
0.00075 at frame 240: the order of claims and of the lit cells' list).
The cache's passes: `RadianceCache` 0.16 ms, `RadianceCacheFold` 0.05 ms in
the cathedral (64k cells, 8,192 lit a frame, 32,768 rays); GI's `GILight`
fell 0.33 → 0.22 ms (no light queries a probe hit); the stream's tracers
take the cache's rays beside the probes'.

voxel_craft (four poses, eight-frame bursts, no clouds): the cache's frames
0.4% to 1.7% darker than r2's (the shade under trees and overhangs), mean
difference 0.004 to 0.008 against 0.0026 to 0.0040 between two of its own
runs (r2's own: 0.0002); the first pose without foliage against r2's
reference: the whole frame 0.925 of the reference (r2 0.933), indirect
diffuse in the same post 1.31 (r2 1.29, both images over r2's reference's
post). The cache's passes there: `RadianceCache` 0.09 ms, `RadianceCacheFold`
0.03 ms; the stream's voxel walk takes the 32,768 cache rays beside the
probes' (`VoxelRays gi` 1.66 → 1.96 ms), which is most of the cost.
voxel_flight's ground holds (1600x900, 25 s, no clouds; one interleaved
pair a planet, the p50 of the hold, against r2's merged build; an earlier
build of the cache with the first fold measured the same, Earth 2.42 →
2.98 ms, Moon 2.36 → 3.00 ms): the images 0.0014 (Earth) and 0.0013 (Moon)
mean apart, 0.2% darker; `VoxelRays gi` 1.54 → 2.00 ms on the Earth.

**Open** (for step 4 and after):

- **Thin occluders and coarse cells.** A cell's rays start at its one
  point; a cell straddling a thin wall (wider than the wall, both faces
  turned one way: a floor under a slab) lights one place for the whole
  cell. The slab-box test reads 0.67% of the lit top under it. Step 4's
  pixel reads go finer near the eye (the eye angle), where it matters most. Step 4
  measured one more case: a VG proxy wall with the pixels' gather on (below,
  "Indirect diffuse at a pixel", Open).
- **Cell noise near many lights.** One light sample a cell an update: with
  the cathedral's many lamps a cell's direct part is a mean of heavy-tailed
  samples, visible as blotches cell by cell in view 39 until 32 updates
  average it. ReSTIR-style reuse of samples between neighbouring cells, or
  a light cache per cell, would cut it.
- **Start-up.** Probes whose hits' cells are new hold an update cycle;
  a scene's first GI comes about 20 frames later than before.
- **Cost.** The default budget (32,768 rays) adds about 0.2 ms to GI's
  passes in the cathedral and in voxel_craft, 0.5-0.6 ms on the voxel
  planets, where the walk of the cache's rays dominates (8-12 ns a ray).
  A lower default, or a budget that follows the cells in use, is a
  coordinator decision.
- **The scan** reads every slot each frame (O(slots), 6 MiB at the
  default); a cell list kept by the marks would make it O(cells in use).

#### Indirect diffuse at a pixel

Step 4. GI plugin `render/r3-gather` (`gather.rs`, `gi_gather.wgsl`, the
shared hit lighting `gi_hit.wgsl`, the resolve's `gi_resolve_pixels`),
lights `render/r3-gather` (the deferred lighting's `"ssao"` input deleted;
indirect diffuse by the diffuse lobe's reflectance), voxel
`render/r3-gather` (the occlusion offer's test, a no-ray-query test, the
examples' A/B switch), Helio `render/r3-gather` (the frame BSDF's
`bsdf_diffuse_reflectance`, the reference's diffuse split, the lighting
contract's docs, this section).

**Where it lives.** GI publishes `INDIRECT_DIFFUSE`, so the gather is GI's:
the resolve node (`helio.pass.gi_resolve`, `Slot::LightingInputs`) builds it
before the resolve when `GiSettings::quality` asks, in the primary view only
(captures and other views resolve the probes, as before: the reservoirs and
the denoiser's history are a view's, and a capture would pay the gather
again; per-view state is keyed by the camera's view id, as the denoiser's
is, so XR eyes in one graph each keep their own). It reads frame contracts
only: the G-buffer, a ray stream (`gi_pixels`, `SurfaceDetail::Diffuse`:
every registered tracer answers, the terrain's walk on any device, the
meshes' TLAS with ray queries), the radiance cache, the frame's history
(`Need::Pixels`), the denoiser (`denoise::filter`), and for the hits' own
light the frame's light samplers (`light_samples`) and two occlusion
streams (`gi_pixels_shadow`, `gi_pixels_visibility`). It names no kind. How
a hit is lit (the cache's irradiance times the albedo, the transmission
share of the far side's cell, the emission; the probes standing in where
the cache has no cell; the sky on a miss; a hit's direct light from a light
sample and its shadow ray) moved out of `gi_light.wgsl` into `gi_hit.wgsl`,
one copy for the probes', the cache's and the pixels' rays.

**The passes** (one ray a block of `side x side` internal pixels; `side`
1, 2, 4 for `pixels_1`, `pixels_4`, `pixels_16`):

1. `GIGather`. *Near field* (a pixel): the frame points of a workgroup's
   8x8 pixels and 4 around them read once into workgroup memory; the
   pixel's geometric normal from its neighbours (the nearer one-sided
   difference an axis); along 4 screen directions and both their sides,
   every pixel out to `side`, the highest occluder within the block's
   footprint (`side` pixels' width at the pixel's depth) and no farther
   closes the elevations below it, `sin^2` of its elevation of that
   azimuth's cosine-weighted measure; `V` is the mean open share over the 8
   azimuths. A surface whose occlusion is its own (`gbuffer_own_occlusion`:
   the terrain's short sky rays, a meadow's) takes its material occlusion
   as `V`. The pixel's factor is `V` over the mean `V` of its block's
   pixels (workgroup memory again: a block's side divides the workgroup's
   8), over the material occlusion where that is the surface's own (the
   lighting multiplies it in itself). No floor, no division of the signal.
   *Rays* (a traced texel): the traced pixel is the denoiser's
   (`denoise_traced_pixel`, the graph's frame number), its ray
   cosine-distributed about the G-buffer normal by Roberts' R2 over frames
   shifted by a hash of the texel (32-bit fixed point: exact at any frame
   number), from `gi_ray_offset` off the surface, as wide as the block's
   footprint there (the patch the estimate stands for) and spreading by
   `GI_RAY_SPREAD` (`sqrt(2 pi / 32)`: a 32-frame history's share of the
   hemisphere), in the camera's coordinate space; and for one texel in
   `RETRACE_PERIOD` (4), a second ray from its previous reservoir's
   receiver along its kept sample, after every texel's row.
2. The tracers (`stream::trace`).
3. `GIGatherHits` (a texel, 64 a workgroup): each hit's light. A cell of
   the radiance cache no wider than the ray is long (`radiance_cache_size`
   of the level the hit's footprint reads, against the hit distance) stands
   for the hit's light: `gi_hit_light`, the cell claimed. A wider cell
   averages light the hit does not see from where it is (the floor past a
   contact, the lit side of a thin wall: SHaRC's rule that a short path may
   not end in a coarse cell), so there the hit takes the cell's indirect
   part alone, read without claiming the cell, and one light query of its
   own (`light_query_point` at the hit, as wide as the ray's footprint, two
   sided where light passes through), compacted a workgroup at a time into
   the light batch `gi_pixels`; where the cache has no cell the probes'
   whole irradiance stands in and no light of its own is added. The
   luminance of the cell's whole light rides along (what a re-trace
   compares). The frame's light samplers answer (`light_samples::sample`).
4. `GIGatherShadow`: each sample's light at its hit (`gi_hit_direct`, the
   same Lambertian and transmission terms as the cache rays' hits, now one
   function in `gi_hit.wgsl`), waiting on a shadow ray from the light's
   side (`gi_shadow_ray`, likewise shared), compacted into the occlusion
   stream `gi_pixels_shadow`. The tracers answer.
5. `GIGatherShade`, three dispatches over the traced texels:
   - *validate*: a re-traced sample's cache light now against the luminance
     kept with it. The same point (the hit within `2 gi_ray_offset + t /
     256` of where the sample was) keeps the sample, its light scaled by
     the change; the relative change beyond the cache's own noise
     (`1 / sqrt(32)`; 0 for the sky) lowers the confidence written for the
     denoiser (`1 - (change - noise) / (1 - noise)`) and the reservoir's M by
     the same factor; anything else drops the reservoir, confidence 0.
   - *temporal*: this frame's hit as a reservoir (M 1, W = pi / G: the
     cosine density in the sample's measure) combined by generalised RIS
     with the reservoir of the previous block the traced pixel's surface was
     in (the history's tap of largest weight, its receiver within a block's
     diagonal and a pixel, the denoiser's normal and plane stops, and no
     surface the depth buffer shows between the two receivers: a block
     across an occluder's foot is the reprojection's rejection), its M
     capped at `M_CAP` (8); balance-heuristic weights over both receivers'
     target functions (luminance times the geometry term). The previous
     sample was traced from this surface point: no visibility ray, its
     validity over time the re-trace's (as ReSTIR GI). Only these
     reservoirs carry over.
   - *spatial*: combined with up to 4 neighbours' (a disc of
     `SPATIAL_RADIUS` = 3 texels turned each frame; the same stops, and no
     surface the depth buffer shows nearer the eye than the straight line
     between the two receivers: a neighbour on the same plane across an
     occluder is not taken), balance heuristic over their receivers
     (geometry terms), the integrand at the receiver's exact surface (the
     G-buffer's position and normal: W was taken with the exact cosines).
     Where the sample chosen is a neighbour's, a visibility ray from this
     receiver to it, into the occlusion stream `gi_pixels_visibility`.
   Reservoirs are 32 bytes: the receiver as a distance and a 16:16
   octahedral direction from the eye (the previous frame's eye moved into
   this frame's space carries an origin change), its normal 8:8 (the
   resampling's weights only) and the kept cache luminance a half; the
   sample as a direction and distance from the receiver (below 0: the sky),
   its normal 12:12 (8 bits an axis put a grazing sample's light where its
   sampling put none) and M in 8 bits (16ths); the light
   (`scene_pack_radiance`) and W.
6. The tracers answer the visibility stream.
7. `GIGatherSignal`: the chosen sample's estimate `L G W` times what its
   visibility ray let through; where nothing got through, this frame's own
   sample's plain estimate (`pi L`, traced from here); where the texel has
   no sample, the probes' irradiance at its pixel (confidence 1: a sample of
   no light would be taken as dark). Written undivided as the signal
   (`denoise_irradiance_add`, the direction toward the sample), its hit
   distance this frame's ray's, no nearer than the block's footprint; its
   confidence the reused reservoir's validation.
8. `Denoise` (`Signal::irradiance("gi_pixels", block).with_confidence()`).
9. `GIResolve` (`gi_resolve_pixels`): the filtered irradiance at the pixel's
   normal times its near-field factor, divided by pi, into
   `INDIRECT_DIFFUSE`; the probes' where the gather has no light yet or a
   remap put the pixel's content in another space. Debug view 37 shows the
   factor.

The deferred lighting multiplies the frame's indirect diffuse (and the
sky's ambient) by `bsdf_diffuse_reflectance` at the view's angle (Helio's
frame BSDF: albedo, 1 - metallic, the energy fix's loss, and `1 - F(wo.h)`
averaged over light from every direction above, a fit within 0.001 of its
quadrature, `bsdf_gpu`), no longer by base colour.

Short frames (a tracer answered a stream short, `Answers`; one word a
stream in `gather_answer`, written by the first pass that reads it): no new
sample is taken where the rays or shadow rays were short (the reservoirs
hold), and where the visibility rays were short the texel takes the probes'
irradiance with its history kept. An origin change: the reservoirs hold
their receivers from the eye, and the previous frame's eye is moved into
this frame's space by the previous origin minus this one (whole 1024 m
steps, exact in f32).

**The near field and `indirect_occlusion`.** One ray a block resolves the
light a block receives; occlusion whose scale is under a block's footprint
varies between its pixels, which the denoiser cannot tell apart in motion.
The near field moves the block's light between its pixels by their `V`
over the block's mean `V`, so a block's mean is the rays' estimate whatever
`V` is. The first version divided the signal by the traced pixel's `V` and
multiplied by each pixel's: exact at the traced pixel only if `V` were all
the ray sees, but a ray that meets a near occluder brings back that
occluder's light, which the division counted again (and the denoiser spread
between texels: at a block of one, where the two cancel exactly, it read
0.927 of the reference against 0.868 with no near field). At a block of 1
the factor is 1 by construction. A surface whose occlusion is its own (its
material occlusion is its `V`) is moved the same way and the lighting's own
multiply is divided out, so the terrain's short sky rays never darken what
the rays found, and GI offers `indirect_occlusion` as where its rays start,
`OCCLUSION_START` (2 mm, `gi_ray_offset`'s constant part); the terrain
clamps that to its own minimum of four cells.

**What was deleted.** The deferred lighting's `"ssao"` input (binding 5
and its sampler 6 in both groups, the white fallback texture, the read of a
key nothing wrote) and its debug view 37 (now GI's near field); the
lighting's base-colour multiply of indirect diffuse and the sky's ambient.
The copies of the sky's radiance, a hit's cache read, a hit's direct light
and a shadow ray toward a light sample that `gi_light.wgsl` held privately
(now `gi_hit.wgsl`'s, shared with the gather). The near field's floor and
its division of the signal. The 48-byte reservoir. The probes-only pin in
`vg_ray_proxies`.

**The constants and where they come from** (`gather.rs`'s docs; its unit
test checks each derivation):

| Constant | Value | Derivation |
|---|---|---|
| `M_CAP` | 8 | a kept sample stays about M + 1 frames, so the denoiser's 32-frame history rests on at least 32 / 9 = 3.6 independent samples, while resampling still chooses among 8 samples' worth |
| `RETRACE_PERIOD` | 4 | a light switched on or off reaches every reservoir's confidence within 4 frames (67 ms at 60 Hz), for a quarter more rays |
| `SPATIAL_RADIUS` | 3 texels | the denoiser's reach at full history (its a-trous steps of 1 and 2 blocks): reuse correlates texels no farther than the denoiser averages anyway |
| neighbours | 4 | 4 x `M_CAP` covers the radius's 28 texels over the frames a sample stays |
| `CACHE_NOISE` | 1 / sqrt(32) | a settled cell's mean of 32 updates of about their own size each |
| `GI_RAY_SPREAD` | sqrt(2 pi / 32) | a 32-frame history's share of the hemisphere |
| ray width | the block's footprint | the patch one ray's estimate stands for |
| `OCCLUSION_START` | 2 mm | `gi_ray_offset`'s constant part |
| `NEAR_SLICES` | 4 (8 azimuths) | every pixel within the block's side along them: 8 x side taps, deterministic, no history of its own |
| `RESERVOIR_BYTES` | 32 | the layout above |

**The visibility decision** (measured, `pixel_gather`'s two occluder
tests: a room halved by a wall to half its height, a glowing panel on one
side, the floor on both sides in view, the camera still and swaying across
the wall; the gather's own estimate before the denoiser, a block of 4, the
far side's pixels within two blocks of the near side's against one plain
ray a pixel; bound 1.15):

| | camera still | camera swaying |
|---|---|---|
| the gather (a neighbour's sample gets a visibility ray, an occluded one none; the temporal stage rejects a receiver across the wall) | 0.92-1.05 | 0.98-1.00 |
| no visibility ray | 1.49-1.53 | |
| the temporal stage's rejection off | 1.00 | 1.18, 1.03 (two runs) |

The near side's light agrees within 3.3% in every row. The swaying test does
not reliably fail with the rejection off: the sway carries few blocks' history
across the wall within the far side's band, so one run of two stayed under the
bound; a sharper case (an occluder that moves across a lit receiver) is open. A temporal sample
needs no visibility ray once the reprojection rejects a different surface
point (ReSTIR GI's rule); a spatial one does. Everything after the
denoiser moves more and is the denoiser's (below).

**Against the others** (section 5, "How others do it"):

- **Lumen's final gather** (UE5): an octahedral screen probe each 16x16
  pixels (8x8 directions, importance-sampled from last frame's lighting and
  the BRDF), filtered in probe space, interpolated to pixels, plus a
  full-resolution short-range AO / bent normal within the probe spacing;
  hits lit from the surface cache (cards) or the world radiance cache.
  Published cost (Wright, "Lumen: Real-time Global Illumination in Unreal
  Engine 5", SIGGRAPH 2022): the screen-probe gather about 2 ms at 1080p
  on a PS5/RTX 2080-class GPU, "High" scalability. Ours keeps the same
  ray budget a pixel at `pixels_16` (one ray a 4x4 block, as one probe
  texel a 16x16 tile's 64 rays) but as reservoirs a traced pixel, whose
  samples converge to the reference as the budget rises (`pixels_1` is a
  ray a pixel) instead of a fixed hemisphere resolution, and the same
  short-range term, as a near field that moves the rays' estimate between
  a block's pixels rather than an AO multiplied on top. Hits are lit from a hash of hit
  points that serves every tracer (no cards: voxels and foliage alike).
- **AMD GI-1.0** (Boissé et al. 2023): screen probes spawned and reused by
  reservoirs, hits lit from a world hash grid; about 2.5-3 ms at 1080p on an
  RX 7900 XTX (its paper's own timings, "GI-1.0: A Fast Scalable Two-Level
  Radiance Caching Scheme"). The closest design: ours resamples per traced
  pixel rather than per probe and shares its cache with the probes and
  every other reader.
- **ReSTIR GI** (Ouyang et al. 2021; Wyman et al., the ReSTIR course
  2023): a ray a pixel at half resolution, temporal then spatial
  resampling, visibility rays on spatial reuse in the unbiased variant;
  about 8-9 ms at 1080p on an RTX 3090 including path tracing of the
  hits. Ours lights hits from the cache (no secondary path), feeds back
  only the temporal reservoirs (the spatial ones only resolve, which the
  course notes give as the remedy for boiling), caps M at 8 from the
  denoiser's history, keeps a neighbour or a previous receiver across an
  occluder the depth buffer shows out of the reuse and traces a visibility
  ray to a neighbour's sample only, discarding an occluded one (ReSTIR GI's
  rule and its biased form; measured above). Reuse is in area measure: a
  sample is a point with its normal, the target the light times the
  geometry term at each receiver (both cosines and the distance), so the
  reconnection Jacobian is in the target itself and in the balance
  heuristic's weights over every receiver, and needs no clamp.
- **Kajiya** (Stachowiak 2022): ReSTIR GI at half resolution over an
  irradiance cache, its reservoirs validated by re-tracing in turn (one in
  4 frames there too) and invalidated on a radiance change. The same
  structure; its cache is camera-centred cascades, ours a hash of world
  cells.
- **RTXGI / SHaRC + NRD** (NVIDIA): a ray a pixel, paths ended in the hash
  cache, the denoiser taking a confidence from the producer, and SHaRC's
  rule that a path ends in the cache only where the cell is small against
  the path's length. Ours is that at `pixels_1` (the same rule at every
  setting, with the hit's own light sampled where the cell is too coarse),
  with the denoiser's contract built for it.

What scales with what: the near field with internal pixels times the
block's side (8 x side taps from workgroup memory); the rays, their
shading and the reservoirs with internal pixels / side^2 (plus a quarter
for the re-traced samples), the shadow and visibility rays with the texels
whose hit needs its own light and whose sample was reused; the denoiser
with internal pixels (its records with traced texels); memory a traced
texel 388 bytes (32 a reservoir twice, 16 each of texel state, signal, the
chosen estimate and the hit's light, 80 a row of the pixels' stream at 1.25
rows, 48 a row of the shadow and of the visibility streams, 64 a light
query and its sample) and 8 a pixel for the near field: at 1920x1080,
67 MB at a sixteenth, 218 MB at a quarter, 821 MB at every pixel, plus the
denoiser's (142-232 MB). Nothing scales with the scene, the lights or the
world's size: an Earth costs what a room does.

**Verified** (RTX 3060):

- **The indirect channel against the reference.** The first measurement
  read the gather's indirect diffuse at 1.21-1.35 of the reference's view
  63 on the cathedral. Three things made that number, none of them the
  gather's rays:
  1. *The reference's split.* View 63 counted the paths whose first bounce
     was drawn from the diffuse lobe, each weighted by the whole BSDF over
     the mixture's density: where a rough specular lobe overlaps the
     diffuse one (the floor at grazing views) the diffuse lobe's share of
     the specular draws was lost. Each first bounce now weights what follows
     by the diffuse lobe's share of the BSDF in its direction
     (`helio-pass-reference`): the converged view 63 rises by 17.7% (by
     44% on the floor's rows), the same scene and frames with the old split
     against the new. The reference's own noise, from a half-length render
     against the full one: 1.8% in the mean, 0.062 per pixel after a 5-pixel
     blur.
  2. *Unconverged captures.* At frame 1200 the probes and the cache were
     still rising (the probes 0.747 at 1200, 0.815 at 3000; a quarter 0.932
     at 1200, 1.001 at 3000). Every row below is at frame 3000.
  3. *The lighting's multiply.* Indirect diffuse was multiplied by base
     colour, not by the diffuse lobe's reflectance (metals took light they
     have no lobe for; dielectrics the specular lobe's share): now
     `bsdf_diffuse_reflectance`, checked against the lobe's integral within
     0.002 (`bsdf_gpu`). Worth 2% on the cathedral's probes (0.815 against
     0.832 with base colour) and inside the run-to-run spread on the gather.

  The cathedral at frame 3000 against the new reference (mean of linear
  luminance over the pixels the reference lights, and the mean relative
  error per pixel after a 5-pixel blur of both; each row its own run,
  shaders from disk; the reference's noise 1.8% / 0.062):

  | Row | a quarter | every pixel | a sixteenth |
  |---|---|---|---|
  | probes | 0.815 / 0.191 | | |
  | one plain ray, no reuse, near field (block mean) | 0.984 / 0.115 | 0.925 / 0.092 | |
  | temporal reuse only | 1.004 / 0.112 | | |
  | temporal + spatial (the gather, block-mean near field) | 1.014 / 0.120 | 0.953 / 0.126 | 0.932 / 0.101 |
  | ... with the first near field (divide at the traced pixel) | 1.001 / 0.107 | 0.927 / 0.129 | |
  | ... no near field | 0.995 / 0.128 | 0.868 / 0.133 | |
  | ... no visibility ray | 0.940 / 0.092 | | |
  | ... every cell trusted (no light of the hit's own) | 1.026 / 0.126 | | |
  | ... indirect times base colour | 0.958 / 0.091 | | |

  Run to run, rows that are the same estimator differ by up to 4% at a
  quarter and up to 9% at every pixel (a block of one has no near field to
  move, so its three forms should agree: 0.868-0.953): the cathedral's
  cache, lit by many lamps, settles differently with when the pipelines
  compiled. Within that, a quarter reads the reference (0.98-1.03 over its
  forms); every pixel and a sixteenth read 0.93-0.95; the probes 0.815.
  **Means over 16 consecutive frames** (the final build: visibility rays
  for a neighbour's sample only, the temporal stage's rejection, the
  integrand at the receiver's exact surface; frames 3000-3015; per-frame
  noise is the spread of each frame's ratio, a separate number):

  | Setting | run a | run b | per-frame sd | frames of 16 with a 40-pixel tile 30% over its burst mean |
  |---|---|---|---|---|
  | probes | 0.830 / 0.183 (a first run drew no scene: discarded) | 0.838 / 0.178 | 0.0001-0.0003 | 0 |
  | a sixteenth | 0.991 / 0.088 | 1.029 / 0.107 | 0.0005-0.0018 | 0 |
  | a quarter | 1.173 / 0.226 | 1.064 / 0.131 | 0.0005 | 0 |
  | every pixel | 0.978 / 0.089 | 0.922 / 0.100 | 0.0013-0.0023 | 0 |

  Per frame the gather moves by under 0.3% of its mean and no frame of 64
  (a quarter, the build before this one: 0/64 in four variants) shows a
  block-sized bright region over its burst's mean: the excess of a quarter's
  run is not a bright sample shared between neighbours for a few frames but
  a state a run settles into and keeps (the left wall at 1.2-1.4 of the
  reference in every quarter's run, the floor at 1.0 in one run and 1.35 in
  another; not at a sixteenth or every pixel). Checked against the
  causes named for it:
  - *The reconnection Jacobian.* Reuse is in area measure: a sample is a
    point with its normal, W the inverse of its density over surface area
    (pi / G for the cosine-sampled ray), the target the light times the
    geometry term at the receiver (both cosines over the squared
    distance). Moving a sample to another receiver changes only the target,
    which carries the Jacobian itself; the balance heuristic's weights are
    the targets at every receiver (the receiver's cosine included), the
    same functions on both sides (the luminance cancels in them). No clamp
    is needed and none is taken.
  - *The integrand at a reused sample.* It used the reservoir's 8-bit
    normals: at a grazing sample W (from the exact cosines) is large, and
    an 8-bit cosine put light where the sampling put none. Now the
    receiver's exact surface (the G-buffer's) and the sample's normal to
    12 bits. Measured on the occluder test: no change in its means.
  - *The denoiser.* The gather's own estimate agrees with one plain ray a
    pixel before the denoiser (the occluder tests: 0.97-1.01 on the lit
    side); after it the two part by 13-21%. Over a whole image the
    denoiser's output keeps 0.95 of the resampled signal's mean and 0.77 of
    one plain ray a block's (`pixel_gather`'s occluder test: output over
    input, 1 where the mean is kept): it weighed records by their variance,
    and with the variance estimated from the samples a noisier signal's
    bright records counted less. Fixed in the denoiser ("The denoiser keeps
    the mean", above): the same test now reads 0.998-1.011 / 0.988-0.993
    for the plain rays (still camera / swaying, two runs; were 0.848 /
    0.767 in this build's baseline) and 0.996-1.011 / 0.980-0.986 for the
    resampled signal (were 0.951 / 0.969), and asserts both within 3%; the
    two sides' reuse over plain after the denoiser 0.95-1.00 behind the
    wall and 0.98-1.03 before it (were 0.69-1.29 and 1.13-1.22).
  - *Bright specks on voxel_craft* (its first pose, 16 frames, view 23):
    443 pixels a frame of block-sized green specks that come and go
    (3,394 pixels specked in some frame, 90 in half of them; the probes
    none). With every
    cache cell trusted (no light sample of the hit's own) 148 a frame and
    1,013: two thirds are the hits' own direct light, one light sample and
    one shadow ray a hit (the sun on a leaf a short ray met), which
    resampling then favours. Their number falls with the hits that must
    sample their own light: the cache rework's per-cell clear distances
    let a short ray trust more cells (the trust rule moves behind
    `radiance_cache_read` there).
  The remaining shortfall is shared by every row that lights hits from the
  cache, the probes most: the cache's closed box reads 4-8% low (the
  furnace below; its own test's bound), and a hit's light reflects again
  in a room. That is the cache's (its owner's), not the gather's.
- **Tests** (`plugins/vendor/gi/feature/tests/pixel_gather.rs`, the
  meshes' tracer with ray queries, a 64x64 renderer at its internal 48x48;
  every wait a signal under a wall-clock guard):
  - *A furnace* (albedo 1/2 glowing with radiance 1; the indirect diffuse
    is 2 everywhere), over blocks the near field leaves whole: probes 0.922
    (mean relative error 0.078), a sixteenth 0.924 (0.076), a quarter 0.954
    (0.046), every pixel 0.958 (0.045); bound 10% (the cache's closed box
    reads 5-8% low). Blocks with a closed pixel: 0.913 / 0.959 / 0.961.
  - *The reservoirs against one plain ray a pixel* (a lit room with a
    pillar, 128 frames a ray a pixel): 0.9996-1.0023, worst 8x8 tile 5.1%.
  - *Reuse across an occluder*, the camera still and swaying (above): the
    far side beside the wall 0.92-1.05 of the plain rays before the
    denoiser (1.49-1.53 with no visibility ray; 1.18 and 1.03 in two
    swaying runs with the temporal rejection off), the near side 0.97-1.01.
  - *A light switched off*: a confidence under 1/2 within 4-5 frames; the
    indirect diffuse under 10% of its light after 177-184 frames (the
    cache's own decay).
  - *The near field*: 10.7% / 3.3% / 0% of the room's pixels moved darker at
    blocks of 16 / 4 / 1; every block's factors average 1 within half
    floats' rounding.
  - *Without ray queries* (`helio-pass-voxel-planet`): a device without `EXPERIMENTAL_RAY_QUERY`, a flat 10 cm voxel world
    under a sun: the gather builds with the terrain's tracer alone and its
    rays light 2,040 of 2,040 traced texels, the indirect diffuse 32,400 of
    32,400 pixels; the occlusion offer's test (2 mm with the gather) and the
    probes over terrain pass.
- **Planets at the default** (voxel_flight's ground holds, 1600x900, 25 s,
  no clouds; the graph's GPU p50):

  | ms | probes | a sixteenth | a quarter |
  |---|---|---|---|
  | Earth | 13.26 | 19.40 | 29.48 |
  | Moon | 11.44 | 16.07 | 22.28 |

  At a sixteenth on the Earth: the terrain's walk of the pixels' rays 1.91,
  of the visibility rays 1.26, of the shadow rays 0.12; the denoiser 1.66;
  the shading 0.62 (Moon 1.51 / 0.59 / 0.12 / 1.24 / 0.62). The images'
  mean |difference| from the probes' 0.0032 (Earth) and 0.0027 (Moon),
  luminance within 0.1%. The rays are as long as before (`GI_RAY_REACH`
  everywhere) and as wide as the estimator's sample: the block's footprint
  at the origin, spreading by `GI_RAY_SPREAD`, so a hit's cone is the patch
  the estimate stands for. The stream contract carries that cone
  (`scene_ray_cone(width, spread)`, `scene_ray_footprint(ray, t) = width +
  spread t`): nothing is missing in it. The terrain's nearest-hit walk does
  not use it: it walks the finest voxels to a hit or to the shell's top
  however wide the ray is there (only its sun-page occlusion reads the
  width). A walk that stops at the clipmap level whose cell the footprint
  covers is the voxel tracer's work, and is where the planets' cost is.
- **Cost** (voxel_craft 1920x1080, four poses, two runs a setting
  interleaved, the graph's GPU medians over 240-frame windows; `GILight`,
  the same pass in every run, 0.36-0.37 ms):

  | ms | probes | a sixteenth | a quarter | every pixel |
  |---|---|---|---|---|
  | whole frame (GPU), runs a / b | 14.79 / 14.80 | 18.59 / 18.49 | 23.65 / 23.49 | 45.55 / 45.46 |
  | GIGather (near field, rays) | | 0.21 (0.17, 0.04) | 0.23 (0.12, 0.08) | 0.39 (0.11, 0.25) |
  | the terrain's walk of the rays, their surfaces | | 1.10 + 0.16 | 3.36 + 0.47 | 12.31 + 1.66 |
  | GIGatherHits | | 0.20 | 0.54 | 1.81-1.93 |
  | shadow rays' walk, their pass | | 0.07 + 0.03 | 0.08 + 0.05 | 0.16 + 0.13 |
  | GIGatherShade (validate, temporal, spatial) | | 0.38 (0.04, 0.08, 0.25) | 1.05 (0.09, 0.21, 0.73) | 3.37-3.45 (0.29, 0.73, 2.31-2.36) |
  | the terrain's walk of the visibility rays | | 0.68-0.71 | 2.25 | 9.19-9.26 |
  | GIGatherSignal | | 0.04 | 0.11 | 0.34-0.35 |
  | Denoise | | 0.90-0.91 | 0.87 | 0.98-0.99 |
  | HistoryPixels | | 0.18 | 0.18 | 0.18 |
  | GIResolve | 0.28 | 0.11 | 0.11 | 0.10 |

  The default, a sixteenth, costs 3.7-3.8 ms over the probes at 1080p on
  voxel_craft (2.6-2.8 before the visibility, shadow and hit passes); a
  quarter 8.8 ms; every pixel 30.7 ms, of which the terrain's walk of the
  two ray sets is 21.5 (the walk's footprint, Open).

  With visibility rays for a neighbour's sample only (one run each, this
  run's clock 2% slower: `GILight` 0.36-0.38, the probes 15.10): the whole
  frame 19.16 / 24.26 / 45.74 ms at a sixteenth / a quarter / every pixel
  (4.1 / 9.2 / 30.6 over the probes); the terrain's walk of the visibility
  rays 0.60-0.62 / 2.03 / 8.00 ms (0.68-0.71 / 2.25 / 9.19-9.26 before),
  the shading 0.39 / 1.14 / 3.49. A reused sample is a neighbour's most of
  the time: the temporal reservoir's own samples were a tenth of the
  visibility rays.

  *Packing the reservoir to 32 bytes*: the shading before it (one pass that
  also lit the hits and wrote the signal) took 1.01 ms at a quarter and
  3.29-3.76 ms at every pixel; its share now (validate, temporal, spatial)
  1.05 and 3.37-3.45, with the hits' light (0.54, 1.81-1.93) and the
  signal (0.11, 0.34) in passes of their own and the spatial stage's depth test between
  receivers added. The reservoirs' memory halves (96 to 64 bytes a texel);
  the spatial stage is what costs at every pixel (five domains unpacked a
  lane).
- **Temporal**: voxel_craft's sideways flight past glowstone
  (`VOXEL_CRAFT_FLIGHT=120`, `tools/ghosting.py`, each moving frame against
  the same camera held still until converged): the probes |dL| 0.0098, 0.39%
  of pixels trailing brighter by over 0.1, 0.30% darker; a sixteenth
  0.0104, 0.39% and 0.53%. A sixteenth now trails no more than the probes
  it replaces (the first build trailed 0.49% against their 0.31%; between
  the two, what a block's history holds changed: it held the signal
  divided by its traced pixels' `V`, which in motion carried a near
  occluder's light, divided again, to the pixels its records were
  reprojected to, and now holds the rays' estimate itself; the two builds
  differ in the visibility rays too, and the flight was not run with each
  change alone). Darker than the still frame it lags a
  little more (0.53% against 0.30%): at a sixteenth a pixel's own samples
  come once in 16 frames and disocclusion leans on the block's records.
  Still bursts (8 frames, four poses): frame-to-frame |difference|
  0.46-1.67e-4 at a sixteenth, 0.44-1.50e-4 at a quarter, 0.43-1.46e-4
  every pixel, the probes 0.31-1.24e-4 (an 8-bit step is 3.9e-3): no
  boiling.
- **Gates**: GI's suites (`probes` 15, `radiance_cache` 8, `mesh_gi_graph`
  2, `pixel_gather` 5, lib 18) pass; `vg_ray_proxies` fails 1 of 3 (below);
  the lights' deferred suites (3 + 21 + 18); Helio's `bsdf_gpu`,
  `wgsl_validation`, `denoise_gpu` (5, 1 ignored as before) and
  `pixels_gpu` (5) on this tree; the layer gate (0 violations); Pulsar's
  `one_owner` (no exception added); clippy with no warnings in the crates
  changed; `cargo fmt -p` on every package touched.

**Open**:

- **The VG proxy wall** (`vg_ray_proxies`'s
  `a_vg_wall_blocks_probe_and_shadow_rays_through_its_proxy`, now at GI's
  default): the far half's probes read 0.0019 of light against the classic
  wall's 0.00007 (bound twice that plus 1e-4). With the gather's hits
  marking no cell (shaders from disk) both walls leak alike, 0.0015 and
  0.0017: the cache's cells that straddle a thin wall. The gather's fine
  marks at the classic wall's face take that away; at the VG wall they do
  not, its G-buffer surface (the source triangles) and its ray proxy (a
  1%-radius simplification) being apart by more than the rays' 2 mm start,
  so the pixels' rays leave from behind or inside the proxy's face. The
  gather reads and marks only where its rule allows (a cell no wider than
  the ray is long); what remains is the cache's thin-wall item and the
  proxy's offset from its surface.
- **The visibility rays on terrain** cost as much as a second set of
  pixels' rays (their sky samples walk the shell's height): the terrain
  walk's footprint, above.
- **The cache's level** (the furnace's 4-8%): the cache owner's.
- **The cathedral against its reference**: with the denoiser keeping the
  mean, a sixteenth's indirect view reads 1.048 of the reference (was
  0.954; "The denoiser keeps the mean"), so what the denoiser's bias hid
  is the gather's and the cache's excess (a quarter's runs settled at
  1.06-1.17 before it, the left wall at 1.2-1.4): not resolved here.
- **The material occlusion path** (voxel_craft's terrain, 16 frames): with
  it switched off (own-occlusion surfaces take the horizon like any
  other) the indirect diffuse of the terrain's half of the image reads
  0.94 of the path's, the whole image 0.98.
- **Captures and other views** resolve the probes. Two primary views (XR's
  eyes) each build the gather in the view's graph and keep their own
  reservoirs, near field and denoiser history by view id; not run on an XR
  device.
- **Remapped pixels** (portals' content) take the probes' irradiance.
- **Blue noise**: the directions are R2 over frames with a hash per texel,
  one function (`texel_sample`); a spatiotemporal blue-noise texture is
  r4's first step.
