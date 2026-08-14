//! Proves the "zero-copy" claim in [`helio_scenedb::MaterialSlot`]'s doc:
//! a plain `World::insert`/`get_mut` write to a `MaterialSlot` component
//! lands, byte-exact, in the SAME physical `wgpu::Buffer` that
//! `GpuSystemContext::bind::<SdbGpuMaterial>(BufferKey::of("builtin_material"))`
//! resolves -- the buffer a real Helio bind group would read, not a copy of
//! it. Does not touch `helio::Scene`/`HelioRenderSubsystem` at all -- this
//! is purely about the SceneDB-side buffer-sharing mechanism (see
//! `subsystem.rs`'s module doc for what's NOT yet wired end-to-end through
//! `Scene::insert_object`).

#[path = "support/mod.rs"]
mod support;

use std::sync::Arc;

use helio_scenedb::{MaterialSlot, SdbGpuMaterial};
use libhelio::material::GpuMaterial;
use pulsar_scenedb::gpu::{readback_row, BufferKey, GpuMirrorHandle, GpuSystemContext, SceneGpuStore};
use pulsar_scenedb::World;

fn sample_material(seed: f32) -> GpuMaterial {
    GpuMaterial {
        base_color: [seed, seed * 2.0, seed * 3.0, 1.0],
        emissive: [0.0, 0.0, 0.0, 0.0],
        roughness_metallic: [0.5, 0.0, 1.5, 0.0],
        tex_base_color: u32::MAX,
        tex_normal: u32::MAX,
        tex_roughness: u32::MAX,
        tex_emissive: u32::MAX,
        tex_occlusion: u32::MAX,
        workflow: 0,
        flags: 0,
        material_class: 0,
        class_params: [0.0; 4],
    }
}

#[test]
fn render_material_write_is_byte_exact_in_the_buffer_gpu_system_context_resolves() {
    let ctx = support::test_context();
    let store = Arc::new(SceneGpuStore::new(&ctx, support::scene_cfg()));
    let mirror = GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue()));
    let mut world = World::new();
    world.attach_gpu_mirror(mirror);

    let e1 = world.spawn();
    world.insert(e1, MaterialSlot::new(sample_material(1.0)));
    let e2 = world.spawn();
    world.insert(e2, MaterialSlot::new(sample_material(2.0)));

    // Zero manual registration call anywhere above -- auto-registered on
    // first insert, per the crate's own "zero manual steps" design.
    world.flush_gpu_mirror(ctx.queue()).expect("mirror attached");

    let sys = GpuSystemContext::new(store.buffer_registry());
    let binding = sys
        .bind::<SdbGpuMaterial>(BufferKey::of("builtin_material"), false)
        .expect("builtin_material must be resolvable by key after MaterialSlot's first insert");

    // Read back row = Entity::index() for each entity, from the buffer
    // GpuSystemContext resolved -- NOT from anything World/SceneGpuStore
    // exposes directly -- and check it's exactly what was inserted.
    let got1: SdbGpuMaterial = readback_row(ctx.device(), ctx.queue(), binding.buffer(), e1.index());
    let got2: SdbGpuMaterial = readback_row(ctx.device(), ctx.queue(), binding.buffer(), e2.index());

    assert_eq!(bytemuck_bits(got1.0), bytemuck_bits(sample_material(1.0)));
    assert_eq!(bytemuck_bits(got2.0), bytemuck_bits(sample_material(2.0)));

    // A get_mut mutation reaches the SAME buffer too (Mut<T>'s GPU
    // write-through hook, not just insert's).
    world.get_mut::<MaterialSlot>(e1).unwrap().data = SdbGpuMaterial(sample_material(9.0));
    world.flush_gpu_mirror(ctx.queue()).expect("mirror attached");
    let got1_after: SdbGpuMaterial = readback_row(ctx.device(), ctx.queue(), binding.buffer(), e1.index());
    assert_eq!(bytemuck_bits(got1_after.0), bytemuck_bits(sample_material(9.0)));
    // e2's row must be untouched by e1's update.
    let got2_after: SdbGpuMaterial = readback_row(ctx.device(), ctx.queue(), binding.buffer(), e2.index());
    assert_eq!(bytemuck_bits(got2_after.0), bytemuck_bits(sample_material(2.0)));
}

/// Byte-for-byte comparison helper -- `GpuMaterial` isn't `PartialEq`, and
/// this test cares about the literal bytes that made the GPU round trip,
/// not a field-by-field semantic comparison.
fn bytemuck_bits(m: GpuMaterial) -> Vec<u8> {
    bytemuck::bytes_of(&m).to_vec()
}
