use super::*;
use std::collections::BTreeMap;

fn verify(sample: impl Fn([i32; 3]) -> u32) -> Mesh {
    let mesh = Mesh::from_samples(&sample);
    let mut faces = BTreeMap::new();
    for quad in &mesh.quads {
        let face = quad.face() as usize;
        let axis = face / 2;
        let u = (axis + 1) % 3;
        let v = (axis + 2) % 3;
        let mut base = quad.origin().map(|i| i as i32);
        base[axis] -= i32::from(face % 2 == 0);
        let [width, height] = quad.extent();
        for y in 0..height as i32 {
            for x in 0..width as i32 {
                let mut cell = base;
                cell[u] += x;
                cell[v] += y;
                assert!(cell.iter().all(|v| (0..32).contains(v)));
                assert!(faces.insert((cell, face), quad.material()).is_none());
            }
        }
    }
    let mut expected = BTreeMap::new();
    for z in 0..32 {
        for y in 0..32 {
            for x in 0..32 {
                let cell = [x, y, z];
                let material = sample(cell);
                if material == 0 {
                    continue;
                }
                for face in 0..6 {
                    let mut neighbour = cell;
                    neighbour[face / 2] += if face % 2 == 0 { 1 } else { -1 };
                    if sample(neighbour) == 0 {
                        expected.insert((cell, face), material);
                    }
                }
            }
        }
    }
    assert_eq!(faces, expected);
    assert_eq!(mesh.exposed_faces, expected.len());
    mesh
}

#[test]
fn rectangles_preserve_every_exposed_face_material_and_halo_boundary() {
    assert!(verify(|_| 0).quads.is_empty());
    assert!(verify(|_| 2).quads.is_empty());
    let plane = verify(|q| if q[1] < 17 { 2 } else { 0 });
    assert_eq!(plane.quads.len(), 1);
    assert_eq!(plane.exposed_faces, 1024);
    assert_eq!(plane.logical_bytes(), 8);
    let box_mesh = verify(|q| u32::from(q.iter().all(|&v| (3..29).contains(&v))));
    assert_eq!(box_mesh.quads.len(), 6);
    verify(|q| {
        if q[1] < 4 + (q[0] + q[2]).rem_euclid(24) {
            1 + (q[0].div_euclid(5).rem_euclid(3)) as u32
        } else {
            0
        }
    });
    verify(|q| {
        if q.iter().all(|&v| (1..31).contains(&v)) && q.iter().any(|&v| v == 1 || v == 30) {
            3
        } else {
            0
        }
    });
    // Alternating cells defeat merging and expose all three material slots.
    verify(|q| ((q[0] + q[1] + q[2]).rem_euclid(4)) as u32);
}

#[test]
fn canonical_edit_and_undo_meshes_keep_all_authored_grids_and_negative_tiles() {
    for metres in [0.1, 0.3, 1.0] {
        let mut world = World::default();
        world.set_voxel_size(metres).unwrap();
        let step = world.voxel_step() as i32;
        let key = Key([-1, 70_000_000 / (32 * step), -1]);
        let low = key.low(world.voxel_step());
        world
            .apply_edit(crate::world::Edit {
                cell: low.map(|v| v + 16 * step),
                radius: (9.0 * metres) as f32,
                material: 3,
            })
            .unwrap();
        let old = world.clone();
        for undo in [false, true] {
            if undo {
                world = old.clone();
            } else {
                world
                    .apply_edit(crate::world::Edit {
                        cell: [low[0] + 23 * step, low[1] + 16 * step, low[2] + 16 * step],
                        radius: (5.0 * metres) as f32,
                        material: 0,
                    })
                    .unwrap();
            }
            let brick = Brick::from_world(&world, key);
            let mesh = Mesh::from_world(&world, key, &brick);
            let expected =
                verify(|q| world.material(std::array::from_fn(|a| low[a] + q[a] * step)));
            assert_eq!(mesh.exposed_faces, expected.exposed_faces);
            assert_eq!(
                bytemuck::cast_slice::<_, u32>(&mesh.quads),
                bytemuck::cast_slice::<_, u32>(&expected.quads)
            );
        }
    }
}

mod construction;
mod gpu;
