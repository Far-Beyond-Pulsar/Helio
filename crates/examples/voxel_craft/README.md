# Voxel Craft

A sandbox on a Helio voxel planet of 10 cm voxels: walk, jump, fly, dig and
build on a 16 km cube-sphere planet, among generated forests, bushes,
boulders, deserts, mountains and caves, with cottages, towers, ruins, wells
and pyramids to find, and a blocky character you can see in third person.
Every voxel's colour is procedural: nothing is textured.

```
cargo run -p examples --release --bin voxel_craft
```

| Input | Action |
|---|---|
| WASD, mouse | move, look |
| Space | jump (tap twice, or `F`, to fly; Space/Shift go up/down) |
| Ctrl | sprint / fast flight |
| LMB / RMB / MMB | dig a ball / build a half-metre block / pick the block under the crosshair |
| wheel, 1-9 | hotbar |
| V or F5 | first / third person |
| T | speed time up |
| R | respawn |
| Esc | release the mouse (Esc again quits) |

Environment: `VOXEL_CRAFT_QUALITY=performance|balanced|quality|native`,
`VOXEL_CRAFT_LOD` (cells per pixel, default 0.6), `VOXEL_CRAFT_DISTANCE`
(view distance in metres, default 12 000), `VOXEL_CRAFT_NO_CLOUDS=1` (a
clear sky).

### Screenshot tour

`VOXEL_CRAFT_SHOTS=<dir>` is the example's render check: with nobody at the
keyboard, it holds the time of day at mid-morning, poses the camera at the
spawn in nine views (looking ahead, to the side, up and down; two third-person
views; from 45 m up looking down; behind; and from 400 m up looking out to
the horizon), saves each as `<dir>/shot_NN.png` and quits after the last.
A pose is shot once the planet pass reports the view settled and at least
`VOXEL_CRAFT_WAIT` frames have passed (default 120, for the temporal
anti-aliasing to converge), or after 900 frames regardless.

```
VOXEL_CRAFT_SHOTS=shots cargo run -p examples --release --bin voxel_craft
```

* `VOXEL_CRAFT_POSES=<n>`: shoot only the first `n` poses.
* `VOXEL_CRAFT_WAIT=<frames>`: the minimum frames before each shot.
* `VOXEL_CRAFT_BURST=<n>`: save `n` consecutive frames a pose
  (`shot_PP_FFF.png`), the camera still: flicker shows as frame-to-frame
  differences. `python tools/flicker.py <dir>` measures it per pose (mean
  luminance change a frame, pixels changing by more than 2%, and each
  pixel's deviation over the burst, saved as `std_PP.npy`).
* `VOXEL_CRAFT_TRUNK=1`: at frame 700, move the camera 4 m in front of the
  nearest tree log (within 15 m) and log that log and the voxels in
  front of it, so the next shot looks at a trunk (pair it with
  `VOXEL_CRAFT_POSES=1 VOXEL_CRAFT_WAIT=800`).

## Layout

| Path | What |
|---|---|
| `main.rs`, `app.rs` | entry point; the winit window and its events |
| `game.rs` | game state and one frame (input, player, world, draw, title HUD) |
| `gfx.rs` | device, surface, scene (sun, atmosphere, post), the voxel planet pass, frame drawing |
| `config.rs` | tunables |
| `blocks.rs` | the blocks: voxel material ids, colours and patches, light through and given off |
| `palette.rs` | the planet's appearance from the blocks, and the block-in-hand materials |
| `terrain/` | the planet recipe (layer stack and material rules), biomes, hash noise |
| `world/` | the edit worker thread, snapshots, block queries; `decorate.rs` is the decoration: tree, cactus and structure stamps and the spawns that place them |
| `trees/` | growth of shapes in metres: oak, dark oak, birch, spruce, pine, jungle giant, bushes, boulders, cacti |
| `structures/` | blueprints (stamps of this game's blocks, a block a metre): cottage, tower, ruins, well, pyramid |
| `player/` | movement and collision on a sphere (stepping up ledges), digging and building, the character |
| `input.rs` | keyboard and mouse state |

## How it fits together

* **Terrain** is the voxel planet's layered generator (`helio.terrain`), set
  up as data in `terrain/recipe.rs`. Block ids are the planet's material ids:
  the generator's own (grass, dirt, stone, sand, snow, gravel, ...) and ids 22
  to 31 for this game's (birch, spruce and jungle wood and leaves, cactus,
  glowstone, stone bricks, dark oak planks).
* **Colour is procedural.** Each block is a voxel material with a colour,
  and the natural ones dry, middle and lush patch colours
  (`Block::look`); the planet pass varies each voxel's pigment and each
  region's patches (`TerrainAppearance::detail`), so a meadow, a bark or a
  rock face reads as voxels of near colours, as in Teardown. Leaves pass a
  third of the light through for global illumination; glowstone lights its
  surroundings.
* **Edits** (the player's) are brushes applied by a worker thread to the
  authoritative planet, which publishes immutable snapshots
  (`world/worker.rs`): a dig takes a 35 cm ball out, a build lays a
  half-metre cube on the block lattice. The renderer draws the latest
  snapshot; physics reads it plus the player's pending edits, so a dig or a
  build is felt at once.
* **Decoration** is the voxel planet's *features*
  (`helio_pass_voxel_planet::features`, `world/decorate.rs`): a library of
  stamps of shapes (tree variants grown per species as tapered capsules,
  sparse leaf balls and cones; bushes, boulders, cacti; structures with
  their foundations) and spawns placing them by biome, height, flatness and
  forest fields. They are part of the planet's recipe and evaluated like its
  terrain, on the GPU for drawing and on the CPU for physics and picks:
  nothing streams around the player, memory does not grow with the area,
  forests show to the horizon as their canopies, and what the player digs
  or builds applies over them.
