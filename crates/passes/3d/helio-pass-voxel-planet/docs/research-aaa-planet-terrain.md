# Research: AAA planet terrain generation and far-field appearance

Status: research notes, 2026-10-14. No code changed. Companion to
`chunk-trees.md` (the architecture) and `two-rate-field.md` (macro octaves
at nodes).

Sources marked **[read]** were fetched and read for this note; sources
marked **[recall]** are from prior knowledge and were not re-verified
(treat their numbers as approximate).

## The problem in numbers

- Descent 400 km to 2 m is log2(200,000) = 17.6 halvings of altitude. In 6 s
  that is **2.9 halvings a second**. Measured: ~64,000 tiles, i.e. ~3,600
  tiles per halving, at ~31 us a tile = ~2 s of GPU.
- Every halving needs a screen's worth of new detail at twice the
  resolution. Looking straight down with a 90 degree horizontal FOV, the
  visible ground spans distances h to ~1.5 h (corner rays at ~49 degrees),
  less than one level. A 2560x1440 screen at one column per pixel is
  3.7 M columns = 900 tiles of 64x64; ring cells between 0.5 and 1 px and
  quadtree granularity make that ~1,800-2,700 tiles. **The floor is
  ~2,000 tiles per halving; we generate ~3,600.**
- The budget the planner uses (3 ms a frame, 180 ms/s) at 31 us buys
  ~5,800 tiles/s = ~2,000 tiles per halving at 2.9 halvings/s. So the
  descent needs roughly **2x fewer tiles or 2x cheaper tiles to keep pace,
  and ~4-6x to also build a lead** (prefetch margin, arrival burst, edits).
  It does not need the 10-20x of a full rewrite.
- Shipped games bound this rate by flight model (see section 3): Elite's
  glide and supercruise near bodies keep d(log h)/dt around 0.1 halvings/s
  [recall]. Our test is ~30x harsher than that.

---

## 1. Hierarchical / progressive terrain synthesis

### Proland (Bruneton & Neyret, INRIA) [read]

Docs: https://proland.inrialpes.fr/doc/proland-4.0/terrain/html/index.html
and https://proland.inrialpes.fr/doc/proland-4.0/core/html/index.html.
Papers list: https://proland.inrialpes.fr/publications.html ("Real-time
rendering and editing of vector-based terrains", CGF 27(2) 2008).

- **Upsample + residual ("multiresolution subdivision surface").** Every
  tile except the root = upsampled quarter of its parent + a residual tile.
  Residuals are precomputed (from DEMs, stored as 16-bit TIFF pyramids with
  a per-file scale) or procedural (pseudo-random residuals with one
  amplitude per quadtree level, e.g. `noise="100,50,25,12,6,3,2,1"`).
- **Filter: interpolating 4-point (Dyn-Levin-Gregory / Kobbelt).**
  Even-even child samples copy the parent; edge midpoints use
  (-1/16, 9/16, 9/16, -1/16) along their row or column; cell centres use the
  16-tap tensor product (1/256, -9/256, 81/256, ...). The limit surface is
  smooth, so missing scales can be left null (smooth) or filled with noise.
- **Borders.** One upsampling step needs a 1-sample parent border;
  recursion needs 2, and **2 suffices for any number of steps**. A tile for
  an n x n quad mesh is (n+5) x (n+5) samples. This is the bounded support
  that makes per-tile generation from the parent possible.
- **Amplitude modulation.** The "upsample and add" shader may scale the
  residual by local altitude, slope or curvature (more noise in mountains
  than plains); examples terrain1/terrain2.
- **Outputs (zf, zc, zm).** zf: fine elevation; zc: the parent *mesh's*
  elevation at the same point, linearly interpolated from the parent's
  modified elevations zm (a planar-triangle parent, aware of the render
  grid size and diagonal orientation); zm: zf after layers. Normals: (nx,
  ny, ncx, ncy), the coarse normal for the same morph.
- **Morph (continuous LOD).** A quad of size l is split when its
  L-infinity distance d < k l (split factor k > 1 keeps the quadtree
  restricted). Blend = clamp((d/l - k - 1)/(k - 1), 0, 1); value = blend
  parent + (1 - blend) child. The transition is (k - 1) l wide. Distance
  includes height above ground: d = max(dx, dy, z - ground_z). Pixel size
  bound: W / (2 k T tan(fov/2)) for T x T textures.
- **Non-linear features are layers, not residuals.** Roads, rivers, lakes
  are vector data rasterized into each tile at its resolution (modifying
  zm). Editing invalidates the producing tasks; the Ork task graph
  recomputes every dependent tile.
- **Normal resolution > geometry.** Elevation tiles of (192+5)^2 drawn
  with a 24x24 grid but a 193^2 normal map: 8x more normal resolution than
  geometry.
- **Asynchronous mode.** Tile tasks need not finish in the frame; the
  first ready ancestor is drawn instead (`setStoreParent`,
  `setAsynchronous`, a prefetching scheduler). The docs say plainly that
  this pops when data arrives and can break the restricted quadtree.

### Geometry clipmaps (Losasso & Hoppe 2004; Asirvatham & Hoppe, GPU Gems 2 ch. 2) [read]

https://developer.nvidia.com/gpugems/gpugems2/part-i-geometric-complexity/chapter-2-terrain-rendering-using-gpu-based-geometry

- Each level's elevation is **predicted by upsampling the coarser level**
  with the tensor 4-point interpolant (C1), then a residual is added:
  decompressed (PTC coder) or synthesized as uncorrelated Gaussian noise
  read from a small wrapped texture with a slight magnification to break
  periodicity (Fournier et al. 1982 style).
- zf and zd = zc - zf are packed in one float (integer and fraction parts);
  zc is the coarse level's elevation linearly interpolated along the coarse
  edge. Vertex shader: alpha = clamp((|x - viewer| - offset) / width),
  z = zf + alpha zd.
- Normal map at **2x geometry resolution**; it packs (Nx, Ny, Ncx, Ncy) so
  the pixel shader blends to the coarser normal with the same alpha.
- Costs (GeForce 6800 GT, 2005, a whole 255x255 level): upsampling 1.0 ms,
  synthesis ~0 ms, normal map 0.6 ms. That is ~65 k samples/ms on a 2005
  GPU. A modern GPU is two orders faster: **a 64x64 upsample-and-add costs
  well under 1 us, against our 31 us a tile.** That gap is the whole
  attraction of hierarchical synthesis.

### Outerra (Brano Kemen) [read]

- https://outerra.blogspot.com/2008/07/intro.html (search snippet):
  wavelet-compressed elevation, decompressed on GPU; finer tiles produced by
  a fractal algorithm on the GPU.
- https://outerra.blogspot.com/2009/02/procedural-terrain-algorithm.html:
  ~150 m source data; bicubic subdivision alone gives "molten ice cream"
  hills; homogeneous wavelet noise looks uniform; **noise amplitude
  modulated by slope** (flat areas get less); below the angle of repose
  the noise is suppressed (sediment); since slope-only modulation leaves
  peaks smooth, **amplitude also rises with positive curvature and with
  elevation**. Detail is mostly in **normal maps generated from finer
  fractal subdivision** than the mesh.
- https://outerra.blogspot.com/search/label/fractal%20terrain: four
  independent fractal channels per quadtree node plus a low-pass filtered
  slope in u and v; horizontal displacement along the filtered slope
  direction (bias, wavelength 19-76 m, amplitude) to make rock bulge or
  carve, normals from the Jacobian of the displacement.
- https://outerra.blogspot.com/search/label/craters (2013): craters are a
  **vector overlay stage** re-applied when tiles are generated: 64 bits per
  crater, created in under 0.5 s, up to ~1 km, shape by surface type and
  explosion depth, **immediately reflected in collision data**; they cost
  only when tiles are regenerated (moving), not per frame.

Take-away: Outerra and Proland keep the *smooth* fractal hierarchical and
put every sharp or edited feature in an explicit layer applied at each
tile's resolution.

### SpaceEngine (Vladimir Romanyuk) [read]

- https://spaceengine.org/news/blog171120/: six quadtrees on a cube-sphere;
  node textures 256x256, geometry 33x33; Earth limited to level 12 (~9.5
  m/px) plus 3 "virtual" levels that reuse level-12 heights. Because
  geometry is 8x coarser than textures, **heights are generated only every
  3rd level** at first (2^3 = 8): silhouettes appear in full detail almost
  instantly while textures are blurry; skipped levels are filled afterwards,
  only for nodes actually drawn. **Normal maps are generated before
  colour** (they matter more perceptually). Priority: finer level first,
  then distance. The split metric is distance to the node edge divided by
  node size in unwarped cube coordinates.
- https://spaceengine.org/news/blog190328/: static texture arrays (2048
  per type) made generation ~10x faster; a "loading speed" slider trades
  tiles per frame for frame rate. **Full terrain load takes ~5 s (LOD 0)
  to ~30 s (LOD 1) on an RTX 2080 at 4K**; at speed 20, 12 s at 18 fps.
  Even a dedicated planet renderer shows refinement; it hides it with
  ancestor textures and priorities.

### Coarse/fine consistency, tile borders, exact CPU queries

- **Consistency.** All hierarchical systems morph a sample between its
  own value and the parent surface's interpolated value (zc), so at the
  transition boundary both levels show the *same* surface. Our voxel
  analog is the 16-frame dithered crossfade (`chunk-trees.md`); a
  geometric morph does not apply to voxels, but see option 3.
- **Borders.** With an interpolating 4-point filter a tile needs a
  2-sample border of its parent and is otherwise independent of its
  neighbours; seams cannot appear because border samples are themselves
  defined by the same recursion (Proland generates noise "without seams
  between cube faces" with a face id).
- **Exact queries.** In a hierarchical definition, the value at one point
  of level L depends on a 4x4 window of level L+1, which depends on a
  window converging to ~5x5 (w_next = ceil(w/2) + 3) at every ancestor up
  to the root. A cold point query therefore costs ~levels x 25-36 samples
  (x the octaves entering at each level): for ~30 levels, 750-1,100
  sample evaluations against ~50 octaves today, **20-40x**. Shipped engines
  sidestep this by deriving collision from the generated tiles: Elite
  generates separate uniform physics patches through the same GPU pipeline
  [read, below]; Outerra's craters show up in collision immediately.
  Helio's CPU mirror is stronger (exact without readback) and would have
  to cache ancestor windows to survive a hierarchical definition.

---

## 2. Inigo Quilez: techniques that cut cost or improve the far field [read]

Index: https://iquilezles.org/articles/

- **Value noise with analytic derivatives**
  (https://iquilezles.org/articles/morenoise/). Expand the trilinear blend
  as n = k0 + k1 u + k2 v + k3 w + k4 uv + k5 vw + k6 wu + k7 uvw (k1 = b - a,
  k4 = a - b - c + d, k7 = -a + b + c - d + e - f - g + h, ...), then
  dn/dx = (k1 + k4 v + k6 w + k7 vw) u'(x), with quintic u = 6x^5 - 15x^4 +
  10x^3, u' = 30 x^2 (x - 1)^2. Value and gradient share the k terms; he
  measures central differences as 5x slower. Value noise has one hash per
  corner and no gradient dot products (cheaper than gradient noise, at the
  price of a lumpier, more grid-aligned spectrum).
- **Derivative-damped fbm ("erosion-like")**: a += b n / (1 + dot(d, d))
  with d the running sum of octave gradients, octaves rotated by
  [[0.8, -0.6], [0.6, 0.8]] and scaled by 2. Steep places stop receiving
  detail: flat valleys, rough slopes. One division per octave on top of
  the gradient we already compute. Needs quintic interpolation (cubic
  shows C2 creases).
- **fbm spectra** (https://iquilezles.org/articles/fbm/). Gain G = 2^-H.
  G = 0.5 (H = 1) gives a -9 dB/octave spectrum that matches measured
  mountain profiles; it is isotropically self-similar, so **every octave
  adds the same RMS slope**. 24 octaves cover the Earth down to ~2 m.
  Lacunarity 1.98-2.01 and per-octave rotation stop peaks of different
  octaves from aligning. Consequence for us (section 4): dropping the
  sub-pixel octaves at distance removes most of the surface's slope
  variance, which is why far terrain shades flat.
- **Band-limiting** (https://iquilezles.org/articles/bandlimiting/).
  Replace an octave by noise(x) * smoothstep(1.0, 0.5, w), w the filter
  width in that octave's lattice units, propagated by the lacunarity
  (w *= 2.01). Box-filtered cosine = cos(x) sin(w/2)/(w/2). Our coarse
  levels already drop unresolved octaves by level; the fade per octave
  avoids a hard band edge (useful for albedo, ejecta and roughness, which
  are point-sampled per cell today).
- **Terrain marching** (https://iquilezles.org/articles/terrainmarching/):
  step dt = 0.01 t (error proportional to distance), linear interpolation
  of the crossing between the last two samples, soft shadows from how deep
  the shadow ray went (min of k h / t). Relevant to far sun rays over the
  coarse cut: penumbra size should scale with distance, not cell count.
- **FBM detail in SDFs** (https://iquilezles.org/articles/fbmsdf/): add
  octaves as a grid of random spheres combined with smin, clipped by an
  smax against the inflated host, each octave half the size. Keeps a valid
  distance bound (unlike additive noise), which is what overhang and cave
  detail needs when a ray tracer relies on distance or occupancy bounds.
- Others worth reading for us: "Voronoise" (one function spanning cell
  noise, value noise and Voronoi via jitter u and smoothness v: a cheap base
  for crater placement), "Directional derivative" (diffuse lighting from
  one extra sample along the light instead of a full normal),
  "Multires ambient occlusion".

---

## 3. Shipped planet engines

| Engine | Where generation runs | What is precomputed | Hiding refinement | Notes for us |
|---|---|---|---|---|
| **Elite Dangerous** (Stellar Forge) [read: Doc Ross interview, https://80.lv/articles/generating-the-universe-in-elite-dangerous] | GPU compute: cube faces as quadtrees of uniform patches, spherified, then "noise graphs" fed with planet id and simulated geology parameters (tectonics, volcanism, crater/basin/mountain statistics) | Only the galaxy distribution and star catalogues; planets are parameters | Top LODs are flat geometry with generated height/normal textures; **each patch also generates lighting (cast-shadow shapes, brightening) used by it and its neighbours**, because the flat far LODs looked too flat; near the ground ordinary shadows take over | Noise in double or double-float emulation for mm precision; **physics patches from the same pipeline**, always one available per entity. Supercruise/glide limit descent speed [recall] |
| **SpaceEngine** [read, above] | GPU per node (256^2 textures, 33^2 grid) | Real-body textures; procedural otherwise | Ancestor textures; generate heights every 3rd level first; normals before colour; priority level then distance | 5-30 s for a full load on an RTX 2080: refinement is visible, managed |
| **No Man's Sky** [secondary: AI summary of the GDC 2017 talk, https://github.com/lazyxeon/AstraWeave/blob/main/docs/audits/nms_streaming_architecture_summary_2026-04-24.md; talk: https://www.gdcvault.com/play/1024265/Continuous-World-Generation-in-No] | CPU jobs: 2D height blocking, then 3D density in ~32 m voxel regions, dual contouring, physics mesh, then scatter | Nothing per planet | **Dithered fade-in** for terrain, creatures and LOD transitions; octree LOD | Stateless voxels (no neighbour knowledge), erosion as an end state of noise transforms. Descent speed is limited by the flight model near planets [recall] |
| **Star Citizen** (Planet Tech v4/v5) [read: https://starcitizen.tools/Planet_Tech_v4, https://megagames.com/news/ten-years-in-star-citizen-is-still-rebuilding-how-its-planets-work] | v4: authored + procedural composition (PlanEd painting; temperature/humidity maps choose biomes); v5 (2026 roadmap, tentative) moves generation heavily to the GPU | Planet-scale painted maps and biome assets [recall: ecosystem heightmap tiles] | Not documented publicly in technical detail | Hand-authored macro shape is how they get character from orbit |
| **Space Engineers** [recall] | CPU voxel generation from per-face authored heightmaps and material maps plus noise; small planets (tens of km) | Heightmaps per cube face | Voxel LOD meshes, visible pop at speed | Edits stored as voxel deltas over the procedural base |
| **Dual Universe** [read: https://dualuniverse.fandom.com/wiki/Procedural_Generation] | Procedural on demand; a "hexoctree" voxel grid for planets [devblog title] | Nothing; only player modifications are stored server-side | n/a | Edits as deltas over a regenerable base, as in Helio |
| **KSP / Parallax** [recall] | CPU PQS quad-sphere with PQSMods (heightmaps, simplex, VoronoiCraters); Parallax adds GPU tessellation and scatter | Heightmaps for stock bodies | Visible LOD pop | Craters as a Voronoi mod: a classic cheap crater field |
| **Astroneer, Outer Wilds** [recall] | Tiny planets (Outer Wilds hand-authored) | Everything / authored | Scale avoids the problem | Not comparable |
| **Minecraft** (not a planet, but the closest shipped precedent to two-rate) [read: https://minecraft.wiki/w/Density_function] | CPU | Nothing | n/a | 3D density is evaluated per **cell of 4x8x4 blocks and trilinearly interpolated** (`interpolated`), with per-column caches (`flat_cache`, `cache_2d`): exactly the two-rate idea, in 3D, including caves |

Patterns:

1. **Everyone hides refinement rather than eliminating it**: ancestor
   fallback (Proland async, SpaceEngine), morph zc/zf (Proland, clipmaps),
   dithered fades (NMS, Helio), normal-first ordering (SpaceEngine),
   coarse-level skipping (SpaceEngine).
2. **Rate limiting by design**: flight models cap d(log altitude)/dt.
3. **Character from orbit comes from macro data** (authored or simulated
   parameters) and from **lighting generated beyond the geometry**
   (Elite's per-patch shadows, normal maps finer than meshes).
4. **Edits are deltas over a regenerable base** (Outerra craters, Dual
   Universe, Space Engineers, Helio).

---

## 4. Far-field appearance

### Why a planet looks flat from orbit

- **Relief is small against the radius.** The Moon's relief is ~10-20 km
  on a 1,737 km radius (~1%); geometry barely moves the limb. What reads
  is shading and albedo.
- **With H ~ 1 every octave carries the same slope variance** (section 2).
  A coarse level resolves only the octaves above its cell; the 15-25
  sub-pixel octaves that would roughen the shading are gone, so the
  surface lights like a smooth ball. This is the dominant cause in a band-
  limited point-sampled field.
- **Photometry.** Airless regolith is not Lambertian. The Lunar-Lambert
  function (McEwen 1991, Icarus 92; https://ui.adsabs.harvard.edu/abs/1991Icar...92..298M/abstract,
  ISIS `LunarLambertMcEwen`) is
  `I/F = A [ 2 L(g) mu0 / (mu0 + mu) + (1 - L(g)) mu0 ]`,
  mu0 = cos incidence, mu = cos emission, L(g) ~1 at small phase falling
  toward 0 at large phase [paper located; formula and L(g) from recall]. The
  Lommel-Seeliger term makes the **full Moon a uniformly bright disc with no
  limb darkening and almost no relief**, while near the terminator
  brightness goes as mu0/mu and relief is maximal. A Lambert Moon looks
  like a plaster ball; a correct one looks flat at full phase and
  dramatically sculpted at quarter phase, as we see it from Earth.
- **Shadows at the terminator.** Shadow length = height / tan(sun
  elevation). Tycho (~85 km wide, ~4.8 km deep) at 5 degrees elevation
  casts a ~55 km shadow: tens of pixels from Earth with a small telescope,
  and hundreds of pixels from orbit. Craters read from far away mostly
  because of these cast shadows near the terminator.
- **Albedo is the far-field signal.** Maria (albedo ~0.07) versus highlands
  (~0.12-0.18) and **fresh craters' bright ejecta and rays** (Tycho's rays
  run >1,500 km, ~18 diameters) are what the naked eye sees of the Moon
  [recall]. They are independent of geometric resolution if prefiltered.
- **Crater shape statistics** [recall, Pike 1977]: simple lunar craters
  depth d ~ 0.196 D^1.01 (D < ~15 km), complex craters d ~ 1.044 D^0.301 km;
  rim height a few percent of D. Cumulative size-frequency ~ D^-2 for
  saturated surfaces: equal areal coverage per diameter octave, i.e.
  craters behave like an H ~ 1 fractal too and lose resolved slope the
  same way.

### Techniques

- **Normals finer than geometry** (Proland 8x, clipmaps 2x, Outerra "normal
  maps from finer fractal subdivision"). For Helio at 1 px cells this means
  the shading normal of a coarse cell should be the **analytic field
  gradient** (we have it per column) rather than voxel faces, possibly with
  the next one or two finer octaves' gradients added (one or two more
  gradient evaluations per surface column, not per voxel).
- **Prefiltered roughness: Toksvig, LEAN, LEADR** [recall]. Toksvig
  (2005): shorten the averaged normal and widen the specular lobe by its
  length. LEAN mapping (Olano & Baker, I3D 2010): store first and second
  moments of slope (E[s], E[s s^T]), filter them linearly, read a Beckmann
  lobe whose covariance is the slope variance. LEADR (Dupuy et al., SIGGRAPH
  Asia 2013) extends this to displacement with masking-shadowing; Bruneton
  & Neyret's ocean paper (CGF 2010, "seamless transitions from geometry to
  BRDF", listed on the Proland page) does the same hand-off for waves:
  resolved octaves are geometry, unresolved octaves become BRDF slope
  variance. **For a procedural field the second moment needs no sampling:**
  the unresolved octaves' slope variance is known from the octave table,
  sigma^2 = sum over unresolved octaves k of (m_k c A_k / lambda_k)^2, with
  m_k the local amplitude modulation (region masks, erosion fade) and c the
  noise's RMS-slope constant. Feed it as roughness (alpha^2 += 2 sigma^2
  for Beckmann/GGX) and into a microshadowing term (Smith G with that
  roughness), which darkens grazing light: exactly the terminator look.
  For airless bodies, use the same sigma as Hapke's macroscopic roughness
  angle theta-bar (lunar fits ~20-25 degrees) [recall].
- **Far shadows.** Elite precomputes per-patch cast-shadow shapes for far
  LODs; horizon maps (Max 1988; Sloan & Cohen 2000) [recall] store the
  horizon elevation per direction. Helio already traces sun rays through
  the cut; what matters is that rays over coarse tiles are not range-
  limited, use a bias proportional to the cell size, and use IQ-style
  distance-scaled penumbrae.
- **Albedo prefiltering.** Ejecta rays and blankets are thin at coarse
  levels; a point sample per coarse cell either misses them or aliases.
  Evaluate their albedo as a box-filtered (area-averaged) function over the
  cell (radial falloffs and streak densities integrate analytically or with
  2x2 sub-samples at coarse levels only).
- **Exaggeration.** Vertical exaggeration (common in visualization tools)
  conflicts with continuous zoom, collision and edits; if ever used, only
  on impostors of distant bodies seen in the sky.

---

## 5. GPU generation efficiency

### What a sample costs today

`noise.wgsl` evaluates 3D Perlin gradient noise in Q24 with i64 products
(`wide64.wgsl` when `SHADER_INT64` is present, 16-bit limbs otherwise).
`noise_fine_grad` per octave: 8 corner hashes (~10 integer ops each), 32
`grad` selects (value + 3 gradient components per corner), 3 quintic fades
(3 `mul_fine` each), 3 `dw` (4 `mul_fine` each), 7 value lerps + 21
gradient lerps + 7 "along" lerps (each a 64-bit multiply, add, shift).
Roughly **600-700 instructions per octave with gradient**. GENBENCH:
~23 us a tile = ~36,000 lane-cycles per column on an RTX 3060 (3584 lanes
at ~1.8 GHz), consistent with ~20 band octaves plus erosion, ridges, warps
and craters.

Facts that shape the options [recall unless noted]:

- **Integer throughput is half of FP32 on Ampere**: a GA10x SM has 128 FP32
  lanes, of which 64 can issue INT32. A fixed-point field runs at half the
  machine's ALU peak by construction. 32x32->64 multiplies map to IMAD.WIDE
  (one or two instructions) when the compiler sees sign-extended 32-bit
  operands; the 16-bit limb fallback costs ~10.
- **Float determinism** would need: no FMA contraction (SPIR-V
  `NoContraction`, HLSL `precise`; WGSL exposes neither), no
  transcendental builtins (vendor-specific), identical denormal handling,
  and identical evaluation order. Not reachable from WGSL today. **Keep
  integers for anything collision sees**; floats are fine for display-only
  terms (shading roughness, albedo detail).
- **Noise kinds** (Gustavson 2005 "Simplex noise demystified"; McEwan et
  al. 2012 JGT "Efficient computational noise in GLSL"): 3D value noise =
  8 hashes, trilerp; 3D gradient = 8 hashes + 8 dot products + trilerp; 3D
  simplex = 4 corners, each a radial kernel (0.6 - r^2)^4 times a dot
  product, no lerp chain, cheap analytic derivatives; roughly **half the
  work of 3D Perlin with gradients** in 3D. The simplex skew (F3 = 1/3,
  G3 = 1/6) is exact in integers. Changes the look of every layer.
- **Hashes**: Jarzynski & Olano 2020, "Hash Functions for GPU Rendering"
  (JCGT 9(3)) rank pcg3d / xxhash32 as good quality per cost. Our
  `hash3` multiplies each axis by a constant then finalizes; since
  (x + 1) K = x K + K (mod 2^32), the 8 corners can share the 6 axis
  products (3 multiplies + 3 adds instead of 24 multiplies), bit-identical.
  The finalizer (2 multiplies, 3 xor-shifts) still runs per corner.
- **Lattice caching**: an 8x8 workgroup at one level covers a small 3D
  box; for any octave whose lattice spacing is >= 8 columns the box
  touches at most 3 lattice points per axis (<= 27 corners). Hashing and
  expanding those 27 gradients once per workgroup (27 x 4 words = 432 B of
  workgroup memory per octave, barrier per octave, uniform control flow
  because every lane runs the same compiled program) removes ~40% of an
  octave's per-lane work. Different from the failed node-in-workgroup
  attempt (12.5 KB per group).
- **Lower-rate octaves with interpolation (two-rate)**: the error of
  interpolating an octave of amplitude A and wavelength lambda from nodes
  spaced s:
  - bilinear: |e| <= s^2/8 max|f''| ~ 4.9 A (s/lambda)^2;
  - cubic Hermite with exact values and derivatives: |e| <= s^4/384
    max|f''''| ~ 4.1 A (s/lambda)^4;
  - 4-point interpolating (Kobbelt) at midpoints: 1 - (3/128)(omega s)^4
    transfer, ~36 A (s/lambda)^4.
  Hermite / bilinear ~ 0.8 (s/lambda)^2. With A proportional to lambda
  (G = 0.5), the bilinear error goes as s^2/lambda: our measured 0.13
  cells at level 0 with s = 16 columns and lambda >= 4,096 columns
  (`MACRO_GAP` 15) would become ~1 cell at lambda >= 512 and ~2 cells at
  256 with bilinear blending, but ~0.001 and ~0.007 cells with Hermite,
  and ~0.05 cells at lambda >= 128. Minecraft lives with trilinear
  4x8x4 cells because blocks are 1 m and density is thresholded.
  Limits: interpolation blunts kinks (ridge crests, gully floors, crater
  rims, clamps), so those stay per column, as two-rate already decided;
  cubic interpolation can overshoot where a node quantity is clamped
  (region masks), so clamp after blending.

---

## What this means for Helio

### Options ranked by expected gain per cost and risk

**1. Multi-rate field with Hermite node blending (keep generating from the
field definition).** *Gain: high (column work -40 to -65%, tile ~1.6-2.5x
cheaper). Cost: 1-2 weeks. Risk: low.*

- Nodes already carry value **and gradient** for every macro quantity
  (value, region, ridge noise, warp with Jacobian rows). Replace the
  bilinear `lerp4` by a tensor cubic Hermite blend. With t = d/16 and d the
  integer offset, the basis scaled by 16^3 is exact: H00 = (16 + 2d)(16 -
  d)^2, H01 = d^2 (48 - 2d), H10 = d (16 - d)^2, H11 = -d^2 (16 - d) (the
  derivative terms also take the node spacing). Project each node's 3D
  domain gradient onto its face tangents dp/di, dp/dj once in the node
  pass.
- The cross term (twist f_ij) is needed for fourth-order accuracy. A
  zero-twist patch is only ~5x better than bilinear. Compute twists as
  central differences of node gradients, which needs a 9x9 node grid per
  tile instead of 7x7 (+65% of a 1.4 us pass), **never one-sided
  differences at the tile ring**: a node's derived quantities must depend
  only on the node, or neighbouring tiles disagree.
- Lower the gap from wavelength >= 4,096 columns to >= 256 (or 128):
  expected error <= 0.01-0.05 cells (test
  `two_rate_field_stays_within_a_fraction_of_a_cell` decides). That moves
  4-5 octave scales of every smooth layer from columns to nodes.
- Add a **meso rate**: nodes every 4 columns (17x17 + ring per tile) for
  wavelengths 32-256 columns, Hermite blended. Per column then remain only
  octaves under ~32 columns (~4-5 scales) plus the kinks (erosion gullies,
  crater rims, ridge application). Effective per-column octave count goes
  from ~15-20 to ~5-6 + kinks.
- Exact CPU queries stay O(1) deep: a column needs its 4 macro and 4 meso
  nodes plus its own octaves (~3x today's single-rate work for an isolated
  query, nothing for coherent ones that share nodes).
- Same 3D idea for **caves** (Minecraft's 4x8x4 interpolated density):
  evaluate cave noise on a coarse 3D lattice per chunk and interpolate,
  with exact per-cell evaluation only where the interpolated density is
  within a bound of the threshold (the noise slope bound `NOISE_SLOPE`
  gives that bound). This addresses the measured cave cost (three cave
  noises at every cell down to the cave depth).

**2. Generate only what pixels need, sooner: demand-driven cut,
motion-aware pixel target, no parent round trip.** *Gain: high (tiles per
descent 64 k -> ~15-25 k; ~0.5 s less latency). Cost: 1-2 weeks. Risk:
medium (perceptual tuning).*

- **Feedback from rays** (virtual-texturing style): the primary pass
  already knows the tile and cell size at each hit; write the wanted level
  per hit tile into a small feedback buffer and request children only for
  tiles whose hits want finer cells. This drops tiles hidden behind ridges
  and the ring overlap; the floor is ~2,000 tiles per halving versus
  ~3,600 measured.
- **Motion-aware `lod_pixels`.** In a descent the image expands radially:
  at 2.9 halvings/s and 60 fps the scale changes 3.4% a frame, so the
  screen edge moves ~44 px a frame at 1440p while the centre barely moves.
  TSR rejects most history at such flow and the eye cannot resolve 1 px
  detail there. Use a per-tile target p = clamp(1 + flow_px_per_frame / 16,
  1, 3): tiles scale as 1/p^2, so the periphery of a fast descent needs
  4-9x fewer tiles while the centre and any slow motion keep 1 px. This is
  SpaceEngine's level skipping, made continuous.
- **Do not wait two frames per level for measured layers.** 17.6 levels x
  2 frames is ~0.6 s of pure latency in a 6 s descent. Request children
  with ranges from the field's slope bound (as `Ranges::child` already
  estimates, margin doubling per guessed level) and batch a tile with its
  next 1-2 descendants along the predicted trajectory in one submission;
  the field definition does not depend on the parent, only the chunk
  layer ranges do.
- **Order like SpaceEngine**: tiles under the screen centre and the
  velocity's vanishing point first, then by level, then by distance.

**3. Preview tiles from the parent (display only, not the definition).**
*Gain: medium-high for descents (transient tiles 5-10x cheaper). Cost:
2-3 weeks (a second generation path). Risk: medium.*

- For a tile the eye will pass through in under ~0.5 s and that no physics
  body is near, build its chunks from the parent's heights and gradient
  (2-sample border, Kobbelt 4-point upsample: 16 taps, exact in
  integers with weights (-1, 9, 9, -1)/16) plus the exact evaluation of the
  2-3 octaves of its band, with their non-linear ops (erosion steered by
  the upsampled gradient, crater rims, ridge steps). No caves.
- The upsampled part is the parent's resolved octaves (wavelength >= ~8
  child cells), whose 4-point interpolation error is under ~1% of their
  amplitude; region-mask thresholds, steering and ridged weights add
  differences that must be measured. Replace previews with exact tiles
  through the existing crossfade when the eye lingers, edits arrive or a
  physics query lands in them.
- This is Proland's synthesis used only where it is cheap and harmless: it
  never feeds collision, saves or edits.

**4. Far-field readability package.** *Gain: high for orbit and moons.
Cost: 1-2 weeks. Risk: low.*

- Shading normal of coarse cells from the analytic field gradient (plus
  optionally one finer octave's gradient), not voxel faces.
- Unresolved slope variance from the octave table (formula in section 4),
  into roughness and a Smith microshadowing term. Zero extra noise
  evaluations; a few multiplies per column using masks the column already
  has.
- Airless bodies: Lunar-Lambert (or Hapke with theta-bar ~20-25 degrees)
  instead of Lambert. Earth-like bodies keep their BRDF plus the variance.
- Crater layer emits albedo: dark maria (basins), bright fresh ejecta
  blankets (~1 radius past the rim) and rays (to ~10+ radii) decaying with
  crater age; the stack already has a share of young craters with bright
  ejecta. Box-filter these per cell at coarse levels (or 2x2 sub-samples
  only above a level).
- Sun rays over the coarse cut: no range cap, bias ~ cell size,
  distance-scaled penumbra; check that terminator crater shadows survive
  at 400 km.
- Optional later: per-tile horizon or shadow summaries (Elite) if the
  traced far shadows cost too much.

**5. Fully hierarchical definition (Proland-style tile = upsample(parent)
+ residual).** *Gain: largest per tile on paper (~10x). Cost: months.
Risk: high. Not recommended.*

- Each sample would pay only the octaves entering at its level, but:
  CPU exact queries become 20-40x costlier cold (ancestor windows to the
  root) and need ancestor caches; every non-linear layer (ridged weights,
  region masks, warps, erosion, craters) must be redefined as per-level
  state carried in channels. Our node state is 116 words: at 68x68 samples
  per tile that is ~2 MB of state per tile, so it could only be kept for
  the frontier tiles (~130 per level) and regenerated otherwise.
- The smooth octaves are where hierarchy pays, and option 1 captures most
  of that (per-column ~5-6 octave scales against ~1-2 for a full
  hierarchy) without the dependency chain. The kinks cost the same in
  both schemes. Caves stay 3D and field-defined either way.

### (a) Tile from its parent or from the field definition?

**From the field definition, with the hierarchy inside the tile (option
1), and parent upsampling only as a display-only preview (option 3).**

- Bit-exact CPU queries: a field-defined column depends on a fixed number
  of nodes; a parent-defined column depends on ~30 ancestor windows.
- Edits and digging are CSG deltas over the field either way (as in
  Outerra, Dual Universe, Space Engineers); neither scheme helps or hurts
  them, but a hierarchical definition makes the base that edits sit on
  depend on generation order and caching.
- Caves and overhangs are 3D and stay field-defined; their speedup is the
  3D two-rate lattice (Minecraft precedent).
- Non-linear layers: both Proland and Outerra treat sharp features as
  explicit layers applied at full resolution per tile. Helio already does
  the equivalent (erosion, craters and ridge application per column); the
  remaining win is making the smooth part cheaper, which nodes do.

### (b) Fewer tiles for a descent

From ~3,600 to ~2,000 tiles per halving with feedback-driven requests,
then 2-4x fewer in the periphery with a motion-aware pixel target;
combined with option 1's ~2x cheaper tiles, the 6 s descent needs
~0.3-0.5 s of GPU instead of ~2 s, under the 3 ms/frame budget with a
lead. Remove the two-frame wait per level. If the design target can be
expressed as a maximum d(log altitude)/dt for the flight model (as
shipped games do), state it: the generation rate scales with it linearly.

### (c) Craters and relief from far away

Readability is shading and albedo, not geometry: analytic normals,
unresolved slope variance as roughness and microshadowing, correct
airless photometry (flat at full phase, sculpted near the terminator),
far sun shadows that reach across the coarse cut, and a prefiltered
albedo channel for maria, ejecta and rays. Leave relief unexaggerated.

### Suggested order

1. Option 4's cheap parts (variance roughness, Lunar-Lambert, analytic
   normals): days, and immediately visible from orbit.
2. Option 1 (Hermite, lower gap, then meso rate); measure with GENBENCH
   and the two-rate accuracy test.
3. Option 2 (feedback, motion-aware target, no layer round trip); measure
   with `HELIO_VOXEL_CHUNKS_FLY=descent` at 6 s: coarse share and largest
   visible cell.
4. Option 3 only if 1+2 leave the 6 s descent short.
