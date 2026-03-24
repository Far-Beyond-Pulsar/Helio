use glam::Vec3;
use helio_pass_sdf::{
    brick::BrickState,
    clip_map::{ClipLevel, SdfClipMap, ATLAS_DIM, BRICK_SIZE, GRID_DIM, LEVEL_COUNT},
    uniforms::{GpuClipLevel, SdfClipMapParams},
};
use std::mem;

// ──────────────────────── Constants ──────────────────────────────────────────

#[test]
fn level_count_is_8() {
    assert_eq!(LEVEL_COUNT, 8);
}

#[test]
fn grid_dim_is_16() {
    assert_eq!(GRID_DIM, 16);
}

#[test]
fn brick_size_is_8() {
    assert_eq!(BRICK_SIZE, 8);
}

#[test]
fn atlas_dim_is_8() {
    assert_eq!(ATLAS_DIM, 8);
}

#[test]
fn total_bricks_per_level_is_4096() {
    let n = (GRID_DIM as usize).pow(3);
    assert_eq!(n, 4096);
}

#[test]
fn atlas_capacity_is_512() {
    let cap = (ATLAS_DIM as usize).pow(3);
    assert_eq!(cap, 512);
}

#[test]
fn voxels_per_brick_is_512() {
    let v = (BRICK_SIZE as usize).pow(3);
    assert_eq!(v, 512);
}

#[test]
fn total_voxels_per_level_axis_is_128() {
    // 16 bricks × 8 voxels/brick = 128 voxels per axis per level.
    let v = GRID_DIM * BRICK_SIZE;
    assert_eq!(v, 128);
}

// ──────────────────────── ClipLevel construction ─────────────────────────────

#[test]
fn clip_level_new_stores_level_idx() {
    let level = ClipLevel::new(3, 1.0, Vec3::ZERO);
    assert_eq!(level.level_idx, 3);
}

#[test]
fn clip_level_new_stores_center() {
    let c = Vec3::new(10.0, 0.0, -5.0);
    let level = ClipLevel::new(0, 1.0, c);
    assert_eq!(level.center, c);
}

#[test]
fn clip_level_0_voxel_size_equals_base() {
    let level = ClipLevel::new(0, 2.0, Vec3::ZERO);
    assert_eq!(level.brick_map.voxel_size, 2.0);
}

#[test]
fn clip_level_1_voxel_size_doubles() {
    let level = ClipLevel::new(1, 1.0, Vec3::ZERO);
    assert_eq!(level.brick_map.voxel_size, 2.0);
}

#[test]
fn clip_level_7_voxel_size_is_128x_base() {
    let level = ClipLevel::new(7, 1.0, Vec3::ZERO);
    assert_eq!(level.brick_map.voxel_size, 128.0);
}

#[test]
fn clip_level_brick_map_grid_dim() {
    let level = ClipLevel::new(0, 1.0, Vec3::ZERO);
    assert_eq!(level.brick_map.grid_dim, GRID_DIM);
}

#[test]
fn clip_level_brick_map_brick_size() {
    let level = ClipLevel::new(0, 1.0, Vec3::ZERO);
    assert_eq!(level.brick_map.brick_size, BRICK_SIZE);
}

#[test]
fn clip_level_brick_map_atlas_dim() {
    let level = ClipLevel::new(0, 1.0, Vec3::ZERO);
    assert_eq!(level.brick_map.atlas_dim, ATLAS_DIM);
}

#[test]
fn clip_level_states_count() {
    let level = ClipLevel::new(0, 1.0, Vec3::ZERO);
    assert_eq!(level.brick_map.states.len(), (GRID_DIM as usize).pow(3));
}

#[test]
fn clip_level_states_all_empty_initially() {
    let level = ClipLevel::new(0, 1.0, Vec3::ZERO);
    assert!(level.brick_map.states.iter().all(|s| *s == BrickState::Empty));
}

#[test]
fn clip_level_dirty_flags_count() {
    let level = ClipLevel::new(0, 1.0, Vec3::ZERO);
    assert_eq!(level.brick_map.dirty.len(), (GRID_DIM as usize).pow(3));
}

#[test]
fn clip_level_dirty_all_false_initially() {
    let level = ClipLevel::new(0, 1.0, Vec3::ZERO);
    assert!(level.brick_map.dirty.iter().all(|d| !d));
}

#[test]
fn clip_level_world_min_uses_center() {
    // With center=ZERO, base_voxel_size=1.0, level 0:
    // bs = BRICK_SIZE * voxel_size = 8 * 1 = 8
    // half = GRID_DIM * bs * 0.5 = 16 * 8 * 0.5 = 64
    // snapped center at 0 → world_min should be around -64 on each axis.
    let level = ClipLevel::new(0, 1.0, Vec3::ZERO);
    let expected = -64.0;
    assert!((level.brick_map.world_min.x - expected).abs() < 1e-4);
    assert!((level.brick_map.world_min.y - expected).abs() < 1e-4);
    assert!((level.brick_map.world_min.z - expected).abs() < 1e-4);
}

// ──────────────────────── SdfClipMap construction ────────────────────────────

#[test]
fn sdf_clip_map_has_correct_level_count() {
    let cm = SdfClipMap::new(1.0, Vec3::ZERO);
    assert_eq!(cm.levels.len(), LEVEL_COUNT);
}

#[test]
fn sdf_clip_map_level_indices_correct() {
    let cm = SdfClipMap::new(1.0, Vec3::ZERO);
    for (i, level) in cm.levels.iter().enumerate() {
        assert_eq!(level.level_idx, i);
    }
}

#[test]
fn sdf_clip_map_voxel_sizes_double_each_level() {
    let base = 0.5;
    let cm = SdfClipMap::new(base, Vec3::ZERO);
    for (i, level) in cm.levels.iter().enumerate() {
        let expected = base * (1u32 << i) as f32;
        assert!(
            (level.brick_map.voxel_size - expected).abs() < 1e-5,
            "level {i}: expected {expected}, got {}",
            level.brick_map.voxel_size
        );
    }
}

// ──────────────────────── GPU struct sizes ───────────────────────────────────

#[test]
fn gpu_clip_level_size_is_64() {
    assert_eq!(mem::size_of::<GpuClipLevel>(), 64);
}

#[test]
fn sdf_clip_map_params_size_is_528() {
    // 16 bytes header + 8 * 64 bytes = 528 bytes.
    assert_eq!(mem::size_of::<SdfClipMapParams>(), 528);
}

#[test]
fn sdf_clip_map_params_level_array_size() {
    assert_eq!(mem::size_of::<[GpuClipLevel; 8]>(), 512);
}
