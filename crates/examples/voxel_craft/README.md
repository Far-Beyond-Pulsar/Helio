# Voxel Craft

A Minecraft-style sandbox on a Helio voxel planet: walk, jump, fly, break and
place blocks on a 16 km cube-sphere planet of 1 m blocks, among generated
forests, deserts, mountains and caves, with cottages, towers, ruins, wells and
pyramids to find, and a blocky character you can see in third person.

```
cargo run -p examples --release --bin voxel_craft
```

| Input | Action |
|---|---|
| WASD, mouse | move, look |
| Space | jump (tap twice, or `F`, to fly; Space/Shift go up/down) |
| Ctrl | sprint / fast flight |
| LMB / RMB / MMB | break / place / pick the block under the crosshair |
| wheel, 1-9 | hotbar |
| V or F5 | first / third person |
| T | speed time up |
| R | respawn |
| Esc | release the mouse (Esc again quits) |

Environment: `VOXEL_CRAFT_QUALITY=performance|balanced|quality|native`,
`VOXEL_CRAFT_LOD` (cells per pixel, default 1.2), `VOXEL_CRAFT_DISTANCE`
(view distance in metres, default 12 000), `VOXEL_CRAFT_PIXELATED=1` (crisp
block pixels instead of smooth magnification).

## Layout

| Path | What |
|---|---|
| `main.rs`, `app.rs` | entry point; the winit window and its events |
| `game.rs` | game state and one frame (input, player, world, draw, title HUD) |
| `gfx.rs` | device, surface, scene (sun, atmosphere, post), the voxel planet pass, frame drawing |
| `config.rs` | tunables |
| `blocks.rs` | the block palette: voxel material ids, faces, emission |
| `terrain/` | the planet recipe (layer stack and material rules), biomes, hash noise |
| `world/` | the edit worker thread, snapshots, block queries; `decorate.rs` decides each region's trees and structures |
| `trees/` | tree generators: oak, dark oak, birch, spruce, pine, jungle giant (as voxel stamps) |
| `structures/` | blueprints (voxel stamps of this game's blocks): cottage, tower, ruins, well, pyramid |
| `textures/` | block tiles to Helio scene materials; composes grass and leaves |
| `player/` | movement and collision on a sphere, block interaction, the character |
| `input.rs` | keyboard and mouse state |

## How it fits together

* **Terrain** is the voxel planet's layered generator (`helio.terrain`), set
  up as data in `terrain/recipe.rs`. Block ids are the planet's material ids:
  the generator's own (grass, dirt, stone, sand, snow, gravel, ...) and ids 22
  to 31 for this game's (birch, spruce and jungle wood and leaves, cactus,
  glowstone, stone bricks, dark oak planks).
* **Textures go through Helio's material system.** Each block face is a
  scene `MaterialComponent` with a base colour texture in the SceneDB texture
  store, like any mesh's material. `ChunkPass::set_scene_materials` maps each
  voxel material to the scene material rows of its top, side and bottom faces,
  and the pass's publish stage samples them through the renderer's material
  table (`helio_default_graphs::voxel_scene_materials`). The same table feeds
  the character's boxes and the block in hand, so any block can later be
  swapped for a graph-compiled material. Devices without descriptor indexing
  fall back to the averaged flat colours.
* **Edits** (the player's) are cube brushes applied by a worker thread to the
  authoritative planet, which publishes immutable snapshots
  (`world/worker.rs`): the renderer draws the latest, physics reads it plus
  the player's pending edits so a break or placement is felt at once.
* **Decoration** is the voxel planet's *features*
  (`helio_pass_voxel_planet::features`): each 32 x 32 column region's trees
  and structures are decided from the world seed, the region and the terrain
  alone, placed as voxel stamps (`helio_pass_voxel_planet::stamp`), loaded
  nearest the player first and unloaded far behind them. They are not edits:
  saves and undo never hold them, memory stays bounded however far the player
  travels, and a region comes back the same when they return, with whatever
  they dug or built over it.

## Credits and licenses

* Block textures: *Modernized 128x* by Sir E of Cali,
  [kallyluv/Modernized-128x](https://github.com/kallyluv/Modernized-128x),
  MIT (`assets/textures/LICENSE-Modernized-128x`). Only the tiles listed in
  `textures/mod.rs` are bundled.
* Tree generators: ports of the oak, birch, spruce/pine and kapok workers of
  [NGNT/treegen-pinegen](https://github.com/NGNT/treegen-pinegen), MIT
  (`assets/trees/LICENSE-treegen-pinegen`).
