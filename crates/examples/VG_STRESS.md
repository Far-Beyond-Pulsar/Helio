# VG museum example

From `crates/renderer/helio`, run:

```powershell
cargo vg-stress
```

`cargo vg-stress` is an alias (`.cargo/config.toml`) for
`cargo run -p examples --features vg-stress --bin vg_stress --release`. The
`vg_stress` binary requires the `vg-stress` feature; other examples and default
builds never download or compile any of the provisioning dependencies
(`ureq`, `sha2`). Close a running copy before rebuilding on Windows.

## First run: self-provisioning

The six Stanford scans (about 570 MB of downloads, about 1.2 GB of prepared
meshes), plus the Poly Haven marble and trees described below, are fetched and
prepared by pure Rust, with no Python, Blender, curl or
tar involved (`vg_stress_prep/`, shared by `build.rs` and the example):

1. **Build time.** With the `vg-stress` feature in a *release* build,
   `build.rs` checks `assets/vg_stress/meshes/prepared.json`. If any scan is
   missing it runs the provisioning (progress appears as `cargo:warning=` lines
   when the script finishes). Failure never breaks the build.
2. **Run time fallback.** If the meshes are still missing when the demo starts
   (prebuilt binary, debug build, failed download), the loader thread runs the
   same code first; progress is shown in the window title and console.
3. Downloads resume (`.part` files, HTTP Range), retry with back-off, try
   mirrors, and are checked against size and SHA-256. Archives are streamed
   through gzip/tar and PLY parsers (ascii, binary little/big endian). Each scan
   is stood up, centred, grounded and scaled exactly as the old Blender script
   did (no decimation), given smooth normals and written as
   `assets/vg_stress/meshes/<name>.glb`. Prepared outputs, sources and caches
   are git-ignored. The Stanford licence (non-commercial, credit Stanford
   Computer Graphics Laboratory) is printed when downloading and kept in
   `assets/vg_stress/CREDITS.md`.

Set `HELIO_VG_STRESS_SKIP_PROVISION=1` to stop the build script provisioning
(for example for `cargo check`). Delete `assets/vg_stress/meshes/prepared.json`
to force re-conversion; delete `assets/vg_stress/source` to force re-download.

### Your own meshes

The stress-grid helpers (`vg_stress_data/layout.rs`: `scan_mesh_folder`,
`stress_grid_layout`) take every `.fbx`, `.obj`, `.gltf` or `.glb` placed
directly in a folder (sorted by name; names starting with `_` or `.` are
skipped) and spread them over a seeded grid. Drop your files into
`assets/vg_stress/meshes/` (git-ignored) or any other folder you pass in.

The architecture kit (`kit/meshes/*.fbx`, `kit_manifest.json`) is committed
(original geometry). Marble and trees are real CC0 assets from Poly Haven,
provisioned on first run (next section); the procedural marble generator is
gone. This standalone example assembles the museum shell, columns, pedestals,
courtyard, props, marble materials, lights, the scan exhibits and the trees in
memory. It never creates a `.level`. Shared dimensions are in
`assets/vg_stress/museum_dims.json`.

## Poly Haven assets (all CC0)

Provisioned by the same pure-Rust code as the scans (`vg_stress_prep/polyhaven.rs`):
the Poly Haven API (`https://api.polyhaven.com/files/<id>` and `/info/<id>`, sent
with a `PulsarEngine-Helio-vg_stress/1.0` User-Agent as the API terms ask)
lists each asset's files, sizes and MD5s; glTF packages keep their directory
structure (`trees/<id>/<id>_2k.gltf`, `<id>.bin`, `textures/*.jpg`); texture sets go
to `marble/<id>/`. Downloads resume, retry, and are checked against size and MD5.
`vg_stress_prep/polyhaven_pins.json` pins the sizes and MD5s captured on
2026-10-22 (a mismatch with the live API only warns; the API stays authoritative).
Each asset gets a `polyhaven.json` manifest (resolution, files, authors) and a
`prepared.json` entry `ph:<id>`; bumping `PH_RECIPE_VERSION` (or the scan
`RECIPE_VERSION`, now 2) makes existing installs re-provision. The credits file
`assets/vg_stress/CREDITS_POLYHAVEN.md` (authors from the API) is regenerated
whenever something is downloaded. Everything downloaded is git-ignored.

| Asset | Use | Download | Notes |
|---|---|---|---|
| `grey_cartago_01` | main floor, all exhibit pedestals, impluvium | 78.5 MB | 8k jpg diffuse + nor_gl + arm (16k exists, not used) |
| `grey_cartago_03` | Ionic columns, benches, steles, secondary floor slabs, plinths | 84.9 MB | 8k |
| `marble_01` | Doric columns, entablature, cella walls, pediments, steps | 25.7 MB | 8k |
| `marble_mosaic_tiles` | floor inlay border (`marble_green` slot), plaza | 149.5 MB | 8k |
| `pine_tree_01` | hero and forest conifers | 982 MB (948,849,556 B .bin) | 2k, 3 variants, 17.18M tris |
| `fir_tree_01` | conifers | 510 MB | 2k, 3 variants, 6.98M tris |
| `island_tree_01/02/03` | courtyard accents, forest | 79.8 / 59.2 / 97.6 MB | 1.60M / 1.07M / 2.09M tris |
| `tree_small_02` | courtyard accents, forest | 115.3 MB | 2.06M tris |

If 8k is not offered the set falls back to 4k then 2k (consistently for all
three maps); the optional displacement maps are only fetched with
`VG_STRESS_DISPLACEMENT=1` (the renderer does not use them yet). Marble uses
the ARM map as the roughness/occlusion/metalness texture. Whichever file fails to
download, the material keeps a flat colour and the scene still opens.

### Choosing the tree set, density and texture budget

| Setting | CLI | Environment | Values |
|---|---|---|---|
| Trees | `--trees <v>` / `--small-trees` | `VG_STRESS_TREES` | `full` (default, about 1.5 GB), `small` (island trees + tree_small_02, about 0.35 GB), `none`, or a list of ids |
| Density | `--density <v>` | `VG_STRESS_DENSITY` | `low` 0.4, `normal` 1 (default), `high` 3, `insane` 8, or a number |
| Textures | `--textures <v>` | `VG_STRESS_TEXTURES` | `low`, `balanced` (default), `max` |

Pass CLI flags after `--`: `cargo vg-stress -- --trees small --density high`. The
build script only sees the environment, so `VG_STRESS_TREES=small` also limits the
build-time download. The scene is seeded and deterministic; the startup log
prints the instance counts, the expected source triangles, GPU texture memory and
download size.

| Preset | Tree instances | Statue instances | Source triangles |
|---|---|---|---|
| `small` trees, density 1 | 162 | 72 | 695 M |
| default (full, density 1) | 162 (8 hero, 24 courtyard, 130 forest) | 72 (18 exhibits, 24 hall copies, 30 courtyard crowd) | 777 M (kit 1.7 M, statues 408 M, trees 368 M) |
| `high` (3) | 470 | 156 | 2.07 G |
| `insane` (8) | 1240 | 173 | 3.9 G |

Instances of one asset share the same GPU geometry (one mesh, one VG asset;
only an object row per instance). Hero trees stand along the approach, forests
fill the grounds beyond the plaza, the hall copies line the ambulatories and the
crowd stands on the plaza with seeded yaw and scale.

GPU texture budget at `balanced`: marble diffuse 8k, normal/ARM 4k with full mip
chains (CPU generated, sRGB-aware, normals renormalised), trees 2k diffuse and 1k
normal/ARM; about 2.6 GiB with all six trees (printed at startup). `low` halves
that; `max` uploads everything at its downloaded size (marble 8k throughout is
about 4 GiB). The kit's UVs repeat every 2 m, so the marble texel density is far
above the photographed scale (the tiles are 0.75 to 2 m): veins look larger than
life, the price of keeping the kit untouched.

### Alpha-masked tree parts

Virtual geometry draws alpha-tested materials but not blended ones, and
`VirtualGeometryHost` keeps blended surfaces classic. The tree
glTFs mark these materials `alphaMode: BLEND`:

| Tree | Masked slot | Masked triangles | Opaque (VG-eligible) |
|---|---|---|---|
| `pine_tree_01` | none (needles are opaque geometry) | 0 | all 17.18M: trunk, bark, dead branches, twigs/needles |
| `fir_tree_01` | `fir_tree_01_twig` | 6.72M of 6.98M | bark, trunk, dead branches |
| `island_tree_01/02/03` | `*_leaves` | 1.06M / 0.71M / 1.41M | trunk and branches |
| `tree_small_02` | `tree_small_02_leaves` | 1.94M of 2.06M | trunk and branches |
Each tree variant stays one glTF mesh per material slot (parts). The glTFs mark
the leaf/twig materials `alphaMode: BLEND` (never VG-eligible), so the museum
authors them as `FLAG_ALPHA_TEST | FLAG_DOUBLE_SIDED` (cutoff 0.5 via the
single-row fallback) and `MASKED_PARTS_USE_VG` is `true`: masked parts are
cooked and drawn through the masked VG raster like every other part, with no
classic stand-in. The leaf colour JPEGs carry no alpha; the separate
`*_alpha_*.png` shipped with the package is combined with the albedo into one
RGBA upload at load time (`decode_with_opacity`), so the alpha test samples the
albedo texture's alpha. Setting the flag to `false` hides masked parts (there is
no classic path). `KHR_texture_transform` UV scale/offset (bark, branches) is
baked into the vertices at import.

## Loading and scale (cook-first)

Virtual-geometry assets (scans, trees) are **never shown in a stand-in form**.
The window opens at once and renders the architecture kit (the only
always-classic geometry); the loader thread then works in this order: kit,
textures, assets whose caches are already valid, then the cook phase. An asset's
instances are created only after its `.vgeo` exists, and they draw only after the
streaming runtime reports its root pages resident (`select_visible` in
`vg_stress.rs`, unit tested). If a cook fails, one error is logged and nothing is
shown for that asset; there is no classic fallback. A device without the VG
passes shows no scans or trees (one warning).

* **Persistent and resumable.** `.vgeo` files in `assets/vg_stress/cache` are
  keyed by content id and validated against the format version and cook
  revision in their header; a valid file is never cooked again. Writes are
  atomic (unique temp file, fsync, rename), so a killed run leaves at most a
  stale `*.tmp` (removed on the next start) and never a corrupt cache. A small
  per-asset part record (`parts-v6-*.bin`: transform, slot, radius, content id,
  records) makes a warm start skip source parsing entirely.
* **Memory-bounded.** Source files are cooked as jobs (a file's variants share
  one import). Each job's peak is estimated from triangle counts (96 B/triangle
  imported + 200 B/triangle of the largest part). A gate admits jobs while the
  sum of estimates stays within the budget (60% of free RAM, clamped to
  2-24 GiB, `VG_STRESS_COOK_MEM_GB=<n>` overrides), at most one 10M+ triangle
  file (pine, Thai, Lucy) at a time, at most `cpus/4` (1-3) files at once, smallest
  first so cheap assets appear first. A job larger than the whole budget runs alone.
  `VG_STRESS_COOK_THREADS` sets the cook pool (default `cpus - 2`, 2-8).
* **Progress.** The title bar and the console show
  `Cooking VG k/N assets (done of total tris) | now: <asset (tris, time)> | elapsed | ETA`;
  per-asset start/finish lines and a 15 s heartbeat go to the console.
* **Headless prepare.** `cargo vg-stress -- --prepare` (or `VG_STRESS_PREPARE=1`)
  provisions the sources and cooks every cache with no window or GPU, then exits
  (non-zero on errors). After that the demo starts with everything ready.
  `build.rs` cannot cook: the build script has no VG builder/importer
  dependencies, so provisioning (download/convert) stays there and cooking runs in
* **Alpha-masked parts** are cooked like every other part (they are the
  largest by triangle count, e.g. fir variant a is 4.1M of which 4.07M masked;
  the per-file memory estimate already counts all triangles of the file).

### Instancing

One asset key is one mesh (kit) or one VG asset (one placeholder
mesh row, one `.vgeo`, one asset slot) however many placements use it; a
placement adds only an object row. The default scene (565 placements) has 31
distinct kit meshes for 331 kit instances and 16 distinct VG assets for 234 VG
instances. VG parts own no classic geometry: their mesh row is a one-triangle
zero-area placeholder whose interned vertex range is the key of the VG asset.
The kit's instances go through the object-batch pass (per-mesh batches). The VG
derivation re-publishes its buffers only when an instance is added or an asset's
readiness changes, not every frame. The startup log prints the live counts
(`[Museum instancing]`).

### Triangle selection and tau

`VG_STRESS_TAU=<px>` sets the VG error tolerance (default 1 px); `-` and `=`
lower and raise it by 20% / 25% live (0.1-64). The title shows tau, scene
triangles and VG selected triangles. Once the first pine (or first VG asset)
is ready the console prints the CPU reference cut at 20/50/100 m
(`[Museum LOD] ...`).

Architectural dimensions are calibrated against the kit manifest. Each scan is
displayed two metres tall; importer node transforms are included in the
calibration so unit conversion does not multiply the scene scale.

## Controls

WASD moves, Space/Shift moves vertically, left click grabs the mouse, and
Escape releases it (or exits when already released).

| Key | Debug view/action |
|---|---|
| `F3` | VG triangle ID |
| `F4` | VG cluster ID |
| `F5` | VG LOD level (cluster coarseness heat ramp) |
| `F6` | VG projected triangle size (pixels per triangle) |
| `F7` | VG raster tier (HW / micro-SW / dense / ray traced) |
| `F8` | VG page residency and streaming pressure |
| `F9` | VG projected error versus tau |
| `1`–`9` | Albedo, normals, roughness, metallic, AO, emissive, shadows, direct lighting, indirect lighting |
| `[` / `]`, Page Up / Page Down | Cycle every registered debug view |
| `0` | Normal rendering |
| `F1` | Print all available views to the console |

Pressing the active view's shortcut again returns to normal rendering. The
catalogue comes from the active render graph, including device-dependent
passes. The remaining Virtual Geometry views (detail vs root, software overdraw,
dense-tile overlap and mask, culling survival, instance ID, barycentrics, vis
normal, material bin, classic-vs-VG mask) are reached with `[` / `]`; the whole
category is registered by `helio-pass-vg-shade` under "Virtual Geometry".
VG views colour VG surfaces (the tile views cover the whole frame); classic
architecture continues through its regular debug path. `VG_DEBUG_VIEW=<id or
key>` (for example `lod`, `tier`, `101`) selects a view at startup; see
`helio_vg_format::debug_view` for the ids and keys. Views cost nothing while
unselected.

The title shows the active view, loading status, source scene triangle total,
and **VG selected** triangles. The scene total includes all loaded instances
and stays constant when moving. VG selected is a delayed asynchronous GPU
sample of emitted triangles across both culling phases. It excludes classic
architecture/preview triangles and shadow work; it is not a whole-frame count.

The title also carries the cull's stats line (selected triangles, emitted/tested
clusters, HW/SW/dense split, frustum/occlusion/backface/error culls and overflow
counters), read from the same asynchronous readback, so it adds no GPU stall.

## Validation

```powershell
cargo test --release -p helio-vg-build --test build distant_sphere_has_bounded_density_and_sampled_surface_error -- --nocapture
cargo test --release -p helio-pass-vg-cull -p helio-pass-vg-shade
cargo test --release -p examples --features vg-stress --bin vg_stress
# Checks the cooked caches (run `cargo vg-stress -- --prepare` first):
cargo test --release -p examples --features vg-stress --bin vg_stress cached_museum_scans_reduce_triangles_with_distance -- --ignored --nocapture
```

The builder test samples actual simplified triangles against an analytic sphere
from three camera axes and requires at most 1 pixel of projected surface error
at 720p. It also checks monotonic triangle reductions and caps the 96m sphere at
1,024 triangles. GPU tests cover decreasing counts, restoring near detail,
instance-scale equivalence, both temporal phases, counter accuracy, and
off-screen rejection. Built-mesh tests check one cluster per leaf-to-root path.

The museum cache test measures the actual six Stanford scans with the demo's
corrected transforms, 45-degree field of view, 720p, and 1px error. It checks
monotonic distance reduction, a 25,000-triangle per-scan ceiling at 96m, a
7,000-triangle root ceiling, and eventual selection of the actual roots. It
also compares each cut with cone culling enabled and disabled, requiring the
cone-cull result to remain nonempty and never add triangles. These tests
validate CPU selection and projected error; the live GPU counters are still
needed to measure whole-scene frame time and total draw cost.

The same test also reports selected counts when the two-metre scan fills 18,
100, and 1,000 vertical pixels, for 1/2/4/8px geometric-error tolerances. The
current 1px results are still too dense for the requested pixel-area budget:

| Scan | 18px | 100px | 1,000px |
|---|---:|---:|---:|
| Dragon | 9,322 | 23,744 | 233,340 |
| Happy Buddha | 6,536 | 25,706 | 144,518 |
| Manuscript | 6,276 | 6,276 | 79,268 |
| XYZRGB dragon | 9,719 | 43,383 | 1,180,339 |
| Thai statue | 6,432 | 37,796 | 313,766 |
| Lucy | 1,184 | 31,732 | 235,776 |

Those counts confirm the distinction between error tolerance and a triangle
budget: the selector only tests projected geometric deviation. `sw_edge_px`
classifies a selected cluster for rasterization; it does not stop traversal or
reduce the cut. The current cached scans also show cone culling removes only
about 0–2% from these views, so backface culling does not explain or solve the
main excess. The next VG selection change needs an explicit projected-density
budget and tests for it; raising the global error tolerance would visibly
loosen geometry and is not a substitute.

The texture timing printed at runtime spans the whole bounded producer phase,
including time the worker waits for the renderer to accept texture uploads; it
is not decode-only CPU time. Cold tree VG cache misses are prepared after
previews and texture events, so the scene is visible while this work continues,
but the loader can remain active well after first frame.

### Distance and backface fixes

The old caches used weighted quadric costs accumulated up the hierarchy. The
builder now simplifies with meshoptimizer while preserving material sections
and group boundaries, measures deviation against the source surface, carries
the maximum child error instead of summing it, and continues building while the
triangle count falls even if the cluster count stays level. It multiplies the
measured error by a safety margin. The cache cook revision invalidates output
from the older builder; cache replacement uses `std::fs::rename`, which replaces an existing destination on Windows and Unix.

The demo now enables conservative normal-cone backface culling on its opaque VG
instances. Alpha-tested tree cards remain classic meshes because VG does not
yet preserve their alpha mode; their sampled classic proxies are a remaining
source of distant geometry. Opaque tree trunks/branches and statue meshes use
their real VG geometry once the current cache is ready. A fresh run rebuilds
stale VG files automatically; the explicit scan cooker and distance test above
can be used to refresh and inspect the six museum caches offline.

For a hidden startup smoke run:

```powershell
$env:HELIO_VG_SMOKE_SECONDS = '18'
cargo vg-stress
Remove-Item Env:HELIO_VG_SMOKE_SECONDS
```

The prep tests (PLY ascii/binary parsing, normalisation, tar/gzip handling, GLB
writer, MD5, and a local HTTP fixture covering resume, retry, mirror fall-through
and hash mismatch) run with the same `cargo test` command and need no network.
The Poly Haven tests plan packages and texture sets from captured real API
responses (`vg_stress_prep/fixtures/`) and provision them end to end against a
local API stand-in (directory structure, manifests, credits, hash mismatch,
resolution fallback, recipe bump).
Changing the scan preparation recipe or the Stanford URLs lives in
`vg_stress_prep/scans.rs`; bump `RECIPE_VERSION` to regenerate GLBs.

## Foliage correctness and profiling

Leaf colour JPEGs require their separate linear PNG opacity maps. Provisioning
recipe 2 downloads these maps; the loader combines them before resizing and
preserves binary alpha coverage in the mip chain at the demo's 0.5 cutoff.
Missing opacity is a load error. Leaves use double-sided cutout materials, with
matching alpha tests in G-buffer, depth prepass, and cached shadow rendering.
Texture publication invalidates cached shadow faces. Opaque and two-sided
materials occupy separate GPU draw segments.

The classic Hi-Z test now projects conservative sphere intervals and covers
every queried mip texel. Camera, scene, draw-segment or resolution changes bypass
stale history until a matching depth frame exists. This single-phase fallback
can increase draws during motion; a current-depth second phase is still needed
for efficient occlusion during motion.

VG traversal redistributes its frontier after bounded local work, then drains
the last round. Exact-set GPU tests check 128 instances over 2/4/8 rounds and
1/256 workgroups, with no dropped or duplicate clusters. This changes work
scheduling, not the geometric error tolerance or selected triangles.

Set `HELIO_VG_PROFILE=1` to log interval frame percentiles, GPU pass timings and
readback age. With `HELIO_VG_SMOKE_SECONDS`, profiling keeps a fixed debug view
instead of cycling views. Run only one demo when comparing GPU times. The
corrected pixel-height table above uses `tan(FOV/2)`; the previous report used
`tan(FOV)` and mislabeled object sizes.

Remaining work: constant sampled leaf proxies are not canopy LODs, VG scan
density still exceeds the desired pixel budget, and texture cooking is not yet
persistently cached. Passing these regressions does not establish full-scene
visual stability or a production performance target.
