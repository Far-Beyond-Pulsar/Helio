//! Shared spherical topology, immutable landform data and the voxel recipe.
//! This graph is generation data, never replacement surface geometry. The
//! engine adapter now uses this data with generator-5 voxel occupancy and edits.
use glam::DVec3;
use std::collections::{BTreeMap, BTreeSet};

mod snapshot;
pub use snapshot::{LandformRecipe, LandformSnapshot, SnapshotId, MAX_HEIGHT_UNITS};
mod sampling;
pub use sampling::{CellSample, FaceAddress, SAMPLING_SHADER};
mod bounds;
pub use bounds::{BoundedLandforms, RegionBounds, RegionClass, BOUNDS_SHADER};
mod volume;
pub use volume::{VoxelEdit, VoxelField, VoxelSample, VOLUME_REVISION, VOLUME_SHADER};
mod defaults;
pub use defaults::{default_clearance, default_field, DEFAULT_LANDFORM_ID};

/// Primitive rational direction; shared by faces and nested resolutions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct NodeKey(pub [i32; 3]);

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuNode {
    pub neighbors: [u32; 8],
    pub reverse: [u32; 8],
    pub inverse_distance: [f32; 8],
    pub rain: f32,
    pub degree: u32,
    pub padding: [u32; 2],
}

pub struct GlobalTopology {
    resolution: u32,
    keys: Vec<NodeKey>,
    directions: Vec<DVec3>,
    face_nodes: Vec<u32>,
    nodes: Vec<GpuNode>,
    areas: Vec<f64>,
}

impl NodeKey {
    fn new(mut xyz: [i32; 3]) -> Self {
        let gcd = |mut a: i32, mut b: i32| {
            while b != 0 {
                let next = a % b;
                a = b;
                b = next;
            }
            a
        };
        let divisor = gcd(gcd(xyz[0].abs(), xyz[1].abs()), xyz[2].abs());
        for v in &mut xyz {
            *v /= divisor;
        }
        Self(xyz)
    }
}

fn face_key(n: u32, face: u32, u: u32, v: u32) -> NodeKey {
    let side = if face % 2 == 0 { n as i32 } else { -(n as i32) };
    let a = u as i32 * 2 - n as i32;
    let b = v as i32 * 2 - n as i32;
    NodeKey::new(match face / 2 {
        0 => [side, a, b],
        1 => [a, side, b],
        2 => [a, b, side],
        _ => unreachable!(),
    })
}

fn triangle_area(a: DVec3, b: DVec3, c: DVec3) -> f64 {
    2.0 * a
        .dot(b.cross(c))
        .abs()
        .atan2(1.0 + a.dot(b) + b.dot(c) + c.dot(a))
}

impl GlobalTopology {
    pub fn new(resolution: u32) -> Result<Self, String> {
        Self::with_face_order(resolution, [0, 1, 2, 3, 4, 5])
    }

    fn with_face_order(n: u32, order: [u32; 6]) -> Result<Self, String> {
        if !n.is_power_of_two() || n > 256 {
            return Err("Landform face resolution must be a power of two in 1..=256".into());
        }
        let mut all_keys = BTreeSet::new();
        for face in order {
            for v in 0..=n {
                for u in 0..=n {
                    all_keys.insert(face_key(n, face, u, v));
                }
            }
        }
        let keys: Vec<_> = all_keys.into_iter().collect();
        let ids: BTreeMap<_, _> = keys
            .iter()
            .enumerate()
            .map(|(i, k)| (*k, i as u32))
            .collect();
        let directions: Vec<_> = keys
            .iter()
            .map(|k| DVec3::from_array(k.0.map(f64::from)).normalize())
            .collect();
        let side = (n + 1) as usize;
        let face_index =
            |f: u32, u: u32, v: u32| f as usize * side * side + v as usize * side + u as usize;
        let mut face_nodes = vec![0; 6 * side * side];
        for face in 0..6 {
            for v in 0..=n {
                for u in 0..=n {
                    face_nodes[face_index(face, u, v)] = ids[&face_key(n, face, u, v)];
                }
            }
        }
        let mut neighbors = vec![Vec::new(); keys.len()];
        let mut areas = vec![0.0; keys.len()];
        for face in 0..6 {
            for v in 0..=n {
                for u in 0..=n {
                    let id = face_nodes[face_index(face, u, v)] as usize;
                    for dv in -1..=1 {
                        for du in -1..=1 {
                            if du == 0 && dv == 0 {
                                continue;
                            }
                            let x = u as i32 + du;
                            let y = v as i32 + dv;
                            if x >= 0 && y >= 0 && x <= n as i32 && y <= n as i32 {
                                neighbors[id]
                                    .push(face_nodes[face_index(face, x as u32, y as u32)]);
                            }
                        }
                    }
                }
            }
            // Canonical face/row order makes the area sum independent of the
            // order in which node keys were discovered.
            for v in 0..n {
                for u in 0..n {
                    let quad = [
                        face_nodes[face_index(face, u, v)],
                        face_nodes[face_index(face, u + 1, v)],
                        face_nodes[face_index(face, u + 1, v + 1)],
                        face_nodes[face_index(face, u, v + 1)],
                    ];
                    let [a, b, c, d] = quad.map(|i| directions[i as usize]);
                    let area = (triangle_area(a, b, c) + triangle_area(a, c, d))
                        * crate::world::RADIUS.powi(2)
                        * 0.25;
                    for id in quad {
                        areas[id as usize] += area;
                    }
                }
            }
        }
        for adjacent in &mut neighbors {
            adjacent.sort_unstable();
            adjacent.dedup();
        }
        let mut nodes = Vec::with_capacity(keys.len());
        for (i, adjacent) in neighbors.iter().enumerate() {
            assert!(adjacent.len() <= 8);
            let mut node = GpuNode {
                neighbors: [i as u32; 8],
                reverse: [0; 8],
                inverse_distance: [0.0; 8],
                rain: (2.0 * areas[i]).sqrt() as f32,
                degree: adjacent.len() as u32,
                padding: [0; 2],
            };
            for (slot, j) in adjacent.iter().enumerate() {
                node.neighbors[slot] = *j;
                node.reverse[slot] =
                    neighbors[*j as usize].binary_search(&(i as u32)).unwrap() as u32;
                node.inverse_distance[slot] = (1.0
                    / ((directions[i] - directions[*j as usize]).length() * crate::world::RADIUS))
                    as f32;
            }
            nodes.push(node);
        }
        Ok(Self {
            resolution: n,
            keys,
            directions,
            face_nodes,
            nodes,
            areas,
        })
    }

    pub fn resolution(&self) -> u32 {
        self.resolution
    }
    pub fn keys(&self) -> &[NodeKey] {
        &self.keys
    }
    pub fn directions(&self) -> &[DVec3] {
        &self.directions
    }
    pub fn nodes(&self) -> &[GpuNode] {
        &self.nodes
    }
    pub fn face_nodes(&self) -> &[u32] {
        &self.face_nodes
    }
    pub fn areas(&self) -> &[f64] {
        &self.areas
    }
    pub fn face_node(&self, face: u32, u: u32, v: u32) -> Option<u32> {
        if face >= 6 || u > self.resolution || v > self.resolution {
            return None;
        }
        let side = (self.resolution + 1) as usize;
        Some(self.face_nodes[face as usize * side * side + v as usize * side + u as usize])
    }
}

#[cfg(test)]
mod tests;

mod clearance;
pub use clearance::FieldClearance;
