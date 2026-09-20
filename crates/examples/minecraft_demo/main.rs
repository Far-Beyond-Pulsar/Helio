//! Dedicated Minecraft-style voxel demo entry point.
//!
//! The renderer shell is shared with the mesh voxel sample while the
//! Minecraft-specific block registry, generation settings, and SceneDB world
//! assembly live in this folder so the demo can grow independently.

mod blocks;
mod generation;
mod world;

#[path = "../voxel/mesh_demo.rs"]
mod renderer_shell;

fn main() {
    renderer_shell::run_with_world_setup(world::spawn_default_world);
}
