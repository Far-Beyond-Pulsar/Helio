//! Voxel Craft: a Minecraft-style sandbox on a voxel planet.
//!
//! Walk, jump, fly, break and place blocks on a 16 km cube-sphere planet of
//! 1 m blocks drawn by Helio's voxel planet pass (`VoxelTerrainComponent`'s
//! renderer), with a blocky character, procedurally grown trees (oak, birch,
//! spruce, pine and jungle giants), cottages, towers, ruins, wells and
//! pyramids, a day-night cycle, and block textures from the Modernized 128x
//! pack (MIT) applied through ordinary Helio scene materials.
//!
//! ```text
//! cargo run -p examples --release --bin voxel_craft
//! ```
//!
//! Controls: WASD move, Space jump (tap twice, or F, to fly), Ctrl sprint,
//! left mouse break, right mouse place, middle mouse pick, mouse wheel or 1-9
//! pick from the hotbar, V third-person camera, T speed up time, R respawn,
//! Esc release the mouse.
//!
//! Environment: `VOXEL_CRAFT_QUALITY` (performance|balanced|quality|native),
//! `VOXEL_CRAFT_LOD` (cells per pixel, 0.6), `VOXEL_CRAFT_DISTANCE` (view
//! distance in metres), `VOXEL_CRAFT_PIXELATED=1` (crisp block pixels);
//! `VOXEL_CRAFT_SHOTS=<dir>` runs the screenshot tour (see the README).
//!
//! Layout: `terrain/` the planet's recipe and biomes, `world/` edits, the
//! worker thread and decoration, `trees/` and `structures/` the generators,
//! `textures/` the block materials, `player/` movement, interaction and the
//! character, `gfx.rs` the renderer, `game.rs` one frame.

#[path = "../v3_demo_common.rs"]
#[allow(dead_code)]
mod v3_demo_common;

mod app;
mod blocks;
mod config;
mod game;
mod gfx;
mod input;
mod player;
mod structures;
mod terrain;
mod textures;
mod trees;
mod world;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn,voxel_craft=info")).init();
    let event_loop = winit::event_loop::EventLoop::new().expect("event loop");
    let mut app = app::App::default();
    event_loop.run_app(&mut app).expect("run");
}
