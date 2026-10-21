# helio-pass-gi

World-space diffuse global illumination: a camera-centred clipmap of
irradiance probes kept up to date by rays every frame, read by deferred
lighting at every pixel (meshes and voxel terrain alike). It gives bounce
light from sunlit ground into shadow, sky light that does not leak into caves
or under overhangs, and light that changes as the world is dug and built.

## Probes

- **Clipmap.** `LEVELS` (6) cubic grids of `SIDE`^3 (24^3) probes, each
  twice as far apart as the one inside it (1, 2, 4 .. 32 m): half-extents
  of 12 to 384 m around the eye. Probes sit at whole multiples of their
  spacing in world coordinates, and are stored toroidally: storage
  coordinates are world coordinates modulo `SIDE`. A probe keeps its record
  while it stays in the clipmap, and a grid that scrolls resets only the
  slabs that came in. The clipmap is a function of the eye's world position
  alone (`Clipmap::around(eye(ctx))`), so a tracer keeping state per probe
  follows it by computing it too.
- **A probe** (`gi_probes`, 72 words) holds:
  - its irradiance / pi as L1 spherical harmonics (RGB, half floats);
  - its state (active, updates taken);
  - its visibility as an 8x8 octahedral map of ray distance moments (mean,
    mean^2; half floats), as in DDGI.

  A probe whose rays mostly meet back faces is inside the ground: inactive,
  so it takes no weight.
- **Updates.** Each frame updates `LEVEL_UPDATES` probes per level (680
  on the two finest, 340 on the coarser, whose far surroundings change
  slowly and whose rays walk farthest): those that came in first, oldest
  first, then the others in turn. Each
  update traces 64 rays along a spherical Fibonacci set turned by a random
  rotation each frame. Results blend into the probe with hysteresis (1/n
  for its first updates, then 3% for irradiance and 5% for moments).

## The frame

`GiPass` runs before the tracers:

1. **`gi_sun_main`:** the sun. With an atmosphere, its direction and its
   illuminance through the air to the eye; without one, the scene's first
   directional light.
2. **`gi_integrate`:** last frame's rays, a workgroup per probe, a lane per
   ray. A ray that hit lights its probe with radiance `albedo * (sun *
   visible * cos / pi + probe irradiance at the hit)`; the probes' own
   irradiance at the hit gives every further bounce. A back face gives
   nothing. A miss sees the sky: the atmosphere's sky table at the probe,
   cosine-convolved L2 harmonics turned back into radiance band by band.
   Without an atmosphere, the flat ambient above the horizon and 0.15 of it
   below.
3. **`gi_reset`:** the probes that came in.
4. **`gi_rays_main`:** this frame's rays (`gi_rays`).

It then clears `gi_hits` and publishes `gi_frame`, `gi_probes`, `gi_rays`,
`gi_hits` and `gi_sun`.

**Tracers** read every ray and write the hits of their own geometry
(`GiHit`: distance, packed normal, albedo RGB8 with the sun's visibility in
alpha, and the tracer). Tracers today:

- **Voxel terrain** (`helio-pass-voxel-planet`, `gi_trace`): each probe's
  base cell is located on the CPU in f64 when the probe comes in
  (`ChunkWorld::follow_gi`). A ray walks the drawn cut from there: its
  chunk; the air box it is in, in the frame of a drawn chunk beside it; or
  rock (a back face at once). Its hit walks to the sun from half a cell off
  the face. Probe rays see the cut as the camera drew it: coarser where the
  camera needs less detail. The test `terrain_traces_gi_probe_rays` checks
  them against CPU rays.
- **Meshes** (hardware ray queries against the scene's TLAS) are next.
  Until then, meshes receive GI but neither occlude nor bounce it.

`GiResolvePass` runs after the tracers and before deferred lighting. For
each pixel it writes `gi_irradiance` (`gi_sample.wgsl`): irradiance / pi
and how much the probes hold it. It reads the finest level whose grid holds
the point. The 8 probes around the point are weighted by:

- trilinear position;
- facing;
- a Chebyshev test of the point's distance against each probe's moments
  toward it.

Near its grid's edge, the level blends into the next one.

## Costs (targets, 1080p, RTX 3060 class)

About 4k probe updates a frame (261k rays, each with its sun walk): under
2 ms with the terrain tracer. Memory: 6 x 24^3 probes x 288 bytes (24 MB),
plus rays and hits (12.5 MB).
