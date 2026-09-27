//! Experimental exact surface quads derived from canonical authored cells.
//!
//! Rectangles merge coplanar faces of one material, never occupied volumes.
//! The one-cell halo suppresses faces shared by adjacent tiles. Quads retain
//! integer coordinates; their enclosing tile is not replacement geometry.
use crate::surface_cache::{Brick, Key, SIDE};
use crate::World;

pub mod raster;

/// Two GPU words per rectangle: local vertex origin (three six-bit integers),
/// face (+X,-X,+Y,-Y,+Z,-Z), material, then two six-bit positive side lengths.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Quad {
    pub origin_face_material: u32,
    pub extent: u32,
}
impl Quad {
    fn new(origin: [u32; 3], face: u32, material: u32, extent: [u32; 2]) -> Self {
        assert!(origin.iter().all(|&v| v <= 32));
        assert!(face < 6 && (1..4).contains(&material));
        assert!(extent.iter().all(|&v| (1..=32).contains(&v)));
        Self {
            origin_face_material: origin[0]
                | (origin[1] << 6)
                | (origin[2] << 12)
                | (face << 18)
                | (material << 21),
            extent: extent[0] | (extent[1] << 6),
        }
    }
    pub fn origin(self) -> [u32; 3] {
        [0, 6, 12].map(|shift| (self.origin_face_material >> shift) & 63)
    }
    pub fn face(self) -> u32 {
        (self.origin_face_material >> 18) & 7
    }
    pub fn material(self) -> u32 {
        (self.origin_face_material >> 21) & 3
    }
    pub fn extent(self) -> [u32; 2] {
        [self.extent & 63, (self.extent >> 6) & 63]
    }
    /// Conforming unit-spaced perimeter vertices avoid greedy T-junctions.
    /// A single cell uses two triangles; larger rectangles use a center fan.
    pub fn triangles(self) -> u32 {
        let [width, height] = self.extent();
        if width == 1 && height == 1 {
            2
        } else {
            2 * (width + height)
        }
    }
}

#[derive(Debug, Default)]
pub struct Mesh {
    pub quads: Vec<Quad>,
    pub exposed_faces: usize,
}
impl Mesh {
    /// Reuses an exact cached payload, sampling only boundary neighbours from
    /// its immutable source. Caller must provide the matching source and key.
    pub fn from_world(world: &World, key: Key, brick: &Brick) -> Self {
        if brick.summary.face_count() == 0 {
            return Self::default();
        }
        let step = world.voxel_step() as i32;
        let low = key.low(world.voxel_step());
        let cell = |q: [i32; 3]| {
            std::array::from_fn(|a| low[a].checked_add(q[a] * step).expect("mesh halo address"))
        };
        let (lo, hi) = (cell([-1; 3]), cell([32; 3]));
        let class = world.classify_region(lo, hi);
        let edits = world.region_edits(lo, hi);
        Self::from_samples(|q| {
            if q.iter().all(|&v| (0..32).contains(&v)) {
                brick.material(q.map(|v| v as u32))
            } else {
                world.material_in_region(cell(q), class, &edits)
            }
        })
    }

    /// Samples the payload and six face-adjacent halo slabs once. No corners or
    /// edges of the halo are needed for opaque face exposure.
    pub fn from_samples(mut sample: impl FnMut([i32; 3]) -> u32) -> Self {
        let index = |q: [i32; 3]| ((q[0] + 1) + 34 * (q[1] + 1) + 34 * 34 * (q[2] + 1)) as usize;
        let mut dense = vec![0u8; 34 * 34 * 34];
        for z in -1..=SIDE {
            for y in -1..=SIDE {
                for x in -1..=SIDE {
                    let q = [x, y, z];
                    if q.iter().filter(|&&v| v == -1 || v == SIDE).count() > 1 {
                        continue;
                    }
                    let material = sample(q);
                    assert!(material < 4, "tiny terrain material domain");
                    dense[index(q)] = material as u8;
                }
            }
        }
        let mut mesh = Self::default();
        for face in 0..6u32 {
            let axis = face as usize / 2;
            let u = (axis + 1) % 3;
            let v = (axis + 2) % 3;
            for slice in 0..32 {
                let mut mask = [0u8; 32 * 32];
                for y in 0..32 {
                    for x in 0..32 {
                        let mut cell = [0; 3];
                        cell[axis] = slice;
                        cell[u] = x;
                        cell[v] = y;
                        let material = dense[index(cell)];
                        if material == 0 {
                            continue;
                        }
                        let mut neighbour = cell;
                        neighbour[axis] += if face % 2 == 0 { 1 } else { -1 };
                        if dense[index(neighbour)] == 0 {
                            mask[(x + 32 * y) as usize] = material;
                            mesh.exposed_faces += 1;
                        }
                    }
                }
                for y in 0..32usize {
                    for x in 0..32usize {
                        let material = mask[x + 32 * y];
                        if material == 0 {
                            continue;
                        }
                        let mut width = 1;
                        while x + width < 32 && mask[x + width + 32 * y] == material {
                            width += 1;
                        }
                        let mut height = 1;
                        while y + height < 32
                            && (x..x + width).all(|i| mask[i + 32 * (y + height)] == material)
                        {
                            height += 1;
                        }
                        for row in y..y + height {
                            mask[row * 32 + x..row * 32 + x + width].fill(0);
                        }
                        let mut origin = [0; 3];
                        origin[axis] = slice as u32 + u32::from(face % 2 == 0);
                        origin[u] = x as u32;
                        origin[v] = y as u32;
                        mesh.quads.push(Quad::new(
                            origin,
                            face,
                            u32::from(material),
                            [width as u32, height as u32],
                        ));
                    }
                }
            }
        }
        mesh
    }

    pub fn logical_bytes(&self) -> usize {
        self.quads.len() * std::mem::size_of::<Quad>()
    }
}

#[cfg(test)]
mod tests;
