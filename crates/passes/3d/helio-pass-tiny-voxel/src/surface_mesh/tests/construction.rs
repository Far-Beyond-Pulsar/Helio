use super::*;
use crate::surface_cache::Cache;
use std::{sync::Arc, time::Instant};

#[test]
#[ignore = "local source/mesh construction, memory and edit-reuse diagnostic"]
fn measure_canonical_tile_mesh_construction_and_edit_reuse() {
    for metres in [0.1, 0.3, 1.0] {
        let mut world = World::default();
        world.set_voxel_size(metres).unwrap();
        let ground = world.ground_spawn(-0.3, -0.3, 0.0);
        let cell = crate::world::cell_of(ground);
        let key = Key::containing(cell, world.voxel_step());
        world.chunks = Default::default();
        let world = Arc::new(world);
        let mut cache = Cache::new(8 * 1024 * 1024, 64);
        cache.set_world(world.clone());
        let start = Instant::now();
        let mut records = Vec::new();
        for z in -1..=1 {
            for x in -1..=1 {
                let key = Key([key.0[0] + x, key.0[1], key.0[2] + z]);
                let brick = cache.get(key).unwrap();
                let mesh = Mesh::from_world(&world, key, &brick);
                assert_eq!(mesh.exposed_faces as u32, brick.summary.face_count());
                records.push((key, brick, mesh));
            }
        }
        let cold_ms = start.elapsed().as_secs_f64() * 1000.0;
        let material_bytes: usize = records.iter().map(|r| r.1.material_bytes()).sum();
        let quad_bytes: usize = records.iter().map(|r| r.2.logical_bytes()).sum();
        let triangles: u32 = records
            .iter()
            .flat_map(|r| &r.2.quads)
            .map(|q| q.triangles())
            .sum();
        let padded_triangles: u32 = records
            .iter()
            .flat_map(|r| &r.2.quads)
            .map(|q| q.triangles().next_power_of_two())
            .sum();
        let mut changed = (*world).clone();
        changed
            .apply_edit(crate::world::Edit {
                cell,
                radius: 0.61,
                material: 0,
            })
            .unwrap();
        let changed = Arc::new(changed);
        let start = Instant::now();
        cache.set_world(changed.clone());
        let mut rebuilt = 0;
        for (key, brick, mesh) in &mut records {
            let replacement = cache.get(*key).unwrap();
            if !Arc::ptr_eq(brick, &replacement) {
                *mesh = Mesh::from_world(&changed, *key, &replacement);
                *brick = replacement;
                rebuilt += 1;
                assert_eq!(mesh.exposed_faces as u32, brick.summary.face_count());
            }
        }
        eprintln!("SURFACE_MESH_WORLD metres={metres} tiles=9 cold_ms={cold_ms:.4} edit_ms={:.4} rebuilt={rebuilt} material_bytes={material_bytes} quad_bytes={quad_bytes} triangles={triangles} padded_triangles={padded_triangles}",start.elapsed().as_secs_f64()*1000.0);
    }
}
