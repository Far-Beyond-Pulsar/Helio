//! Exact, derived surface bricks for the tiny backend.
//!
//! A brick spans 32 authored cells, not 32 fixed decimetres. Uniform 4^3
//! microbricks need one descriptor; mixed microbricks use two 64-bit material
//! bitplanes. Identical microbricks share a payload within a brick. This layout
//! does not prescribe the generic component's source or payload format.
//!
//! Surface summaries count exposed canonical faces with a one-cell halo.
//! They describe geometry, not visibility: projected area alone is NOT opacity
//! or a correct filtered pixel in the presence of occlusion. Bounds enclose
//! actual exposed faces, never a replacement block rendered as terrain.
use crate::world::{Edit, World};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

pub const SIDE: i32 = 32;
const MICRO_SIDE: i32 = 4;
const MICRO_COUNT: usize = 8 * 8 * 8;
const DICTIONARY: u32 = 4;
const HALO_SIDE: usize = 34;
pub const GPU_SHADER: &str = include_str!("surface_cache.wgsl");

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Key(pub [i32; 3]);

impl Key {
    pub fn containing(cell: [i32; 3], step: u32) -> Self {
        assert!((1..=10).contains(&step));
        Self(cell.map(|v| v.div_euclid(SIDE * step as i32)))
    }

    pub fn low(self, step: u32) -> [i32; 3] {
        assert!((1..=10).contains(&step));
        self.0.map(|v| {
            i32::try_from(i64::from(v) * i64::from(SIDE) * i64::from(step))
                .expect("surface brick outside canonical address range")
        })
    }

    fn affected(self, step: u32, edit: Edit) -> bool {
        // Include neighbouring authored cells: edits just outside the payload
        // can expose or cover its boundary faces without changing its materials.
        let low = self.low(step);
        let radius = i64::from(edit.radius_units().div_ceil(2));
        let margin = i64::from(step);
        (0..3).all(|a| {
            let lo = i64::from(low[a]) - margin;
            let hi = i64::from(low[a]) + i64::from(SIDE) * margin + margin - 1;
            lo <= i64::from(edit.cell[a]) + radius && hi >= i64::from(edit.cell[a]) - radius
        })
    }
}

/// Face order +X, -X, +Y, -Y, +Z, -Z; material zero is air.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Summary {
    pub faces: [[u32; 4]; 6],
    /// Inclusive vertex bounds in authored-cell coordinates; None has no faces.
    pub bounds: Option<[[u8; 3]; 2]>,
    pub occupied: u32,
}

impl Summary {
    pub fn face_count(&self) -> u32 {
        self.faces.iter().flatten().sum()
    }

    /// Unoccluded projected-area hypothesis, useful for falsification only.
    /// Facing points toward the viewer. Returns zero when no faces face it.
    pub fn projected_area_weights(&self, facing: [f64; 3]) -> [[f64; 4]; 6] {
        assert!(facing.iter().all(|v| v.is_finite()));
        let mut weights = [[0.0; 4]; 6];
        let mut total = 0.0;
        for f in 0..6 {
            let cosine = (facing[f / 2] * if f % 2 == 0 { 1.0 } else { -1.0 }).max(0.0);
            for m in 1..4 {
                weights[f][m] = f64::from(self.faces[f][m]) * cosine;
                total += weights[f][m];
            }
        }
        if total > 0.0 {
            for w in weights.iter_mut().flatten() {
                *w /= total;
            }
        }
        weights
    }

    fn add_face(&mut self, q: [i32; 3], face: usize, material: usize) {
        self.faces[face][material] += 1;
        let mut low = q.map(|v| v as u8);
        let mut high = q.map(|v| v as u8 + 1);
        let axis = face / 2;
        let plane = low[axis] + u8::from(face % 2 == 0);
        low[axis] = plane;
        high[axis] = plane;
        if let Some(bounds) = &mut self.bounds {
            for a in 0..3 {
                bounds[0][a] = bounds[0][a].min(low[a]);
                bounds[1][a] = bounds[1][a].max(high[a]);
            }
        } else {
            self.bounds = Some([low, high]);
        }
    }
}

#[derive(Debug)]
pub struct Brick {
    /// GPU layout: word 0 < 4 is a uniform material. Otherwise it is 4,
    /// followed by 512 micro descriptors. A descriptor < 4 is uniform;
    /// otherwise payload index = descriptor - 4. Four words per payload:
    /// material bit 0 low/high, then material bit 1 low/high.
    /// Micro and cell indices are x-major. This is derived data, not a file codec.
    words: Box<[u32]>,
    pub summary: Summary,
}

impl Brick {
    fn uniform(material: u32) -> Self {
        Self {
            words: vec![material].into_boxed_slice(),
            summary: Summary {
                occupied: if material == 0 {
                    0
                } else {
                    (SIDE * SIDE * SIDE) as u32
                },
                ..Default::default()
            },
        }
    }

    pub fn words(&self) -> &[u32] {
        &self.words
    }
    pub fn material_bytes(&self) -> usize {
        self.words.len() * 4
    }
    pub fn mixed_patterns(&self) -> usize {
        if self.words[0] < DICTIONARY {
            0
        } else {
            (self.words.len() - 1 - MICRO_COUNT) / 4
        }
    }

    pub fn material(&self, q: [u32; 3]) -> u32 {
        assert!(q.iter().all(|v| *v < SIDE as u32));
        if self.words[0] < DICTIONARY {
            return self.words[0];
        }
        let micro = q.map(|v| v / MICRO_SIDE as u32);
        let descriptor = self.words[1 + (micro[0] + 8 * micro[1] + 64 * micro[2]) as usize];
        if descriptor < DICTIONARY {
            return descriptor;
        }
        let local = q.map(|v| v & 3);
        let bit = local[0] + 4 * local[1] + 16 * local[2];
        let offset = 1 + MICRO_COUNT + (descriptor - DICTIONARY) as usize * 4;
        ((self.words[offset + (bit / 32) as usize] >> (bit & 31)) & 1)
            | (((self.words[offset + 2 + (bit / 32) as usize] >> (bit & 31)) & 1) << 1)
    }

    pub fn from_world(world: &World, key: Key) -> Self {
        let low = key.low(world.voxel_step());
        let step = world.voxel_step() as i32;
        let cell = |q: [i32; 3]| {
            std::array::from_fn(|a| {
                low[a]
                    .checked_add(q[a] * step)
                    .expect("surface halo outside canonical address range")
            })
        };
        let (halo_low, halo_high) = (cell([-1; 3]), cell([32; 3]));
        let edits = world.region_edits(halo_low, halo_high);
        // Only a certificate covering the complete halo may make a uniform
        // surface summary implicit. A uniform payload can still expose faces.
        if let Some(&i) = edits.last() {
            let edit = world.edits[i];
            let lo = world.sample_cell(halo_low);
            let hi = world.sample_cell(halo_high);
            let far = std::array::from_fn(|a| {
                if (i64::from(lo[a]) - i64::from(edit.cell[a])).abs()
                    > (i64::from(hi[a]) - i64::from(edit.cell[a])).abs()
                {
                    lo[a]
                } else {
                    hi[a]
                }
            });
            if edit.contains(far) {
                return Self::uniform(edit.material);
            }
        } else {
            use crate::landforms::RegionClass;
            match world.classify_region(halo_low, halo_high) {
                RegionClass::AllAir => return Self::uniform(0),
                RegionClass::AllSolid => return Self::uniform(1),
                RegionClass::Mixed => {}
            }
        }
        // Classify 8-authored-cell regions including the halo. Sampling the
        // fixed-decimetre ChunkStore here would generate up to 1,000 repeated
        // storage cells for each 1 m authored voxel, plus cache-lock traffic.
        let mut regions = Vec::with_capacity(125);
        for z in 0..5 {
            for y in 0..5 {
                for x in 0..5 {
                    let lo = [x * 8 - 1, y * 8 - 1, z * 8 - 1];
                    let hi = lo.map(|v| (v + 7).min(32));
                    let (lo, hi) = (cell(lo), cell(hi));
                    regions.push((world.classify_region(lo, hi), world.region_edits(lo, hi)));
                }
            }
        }
        Self::from_samples(|q| {
            let r = q.map(|v| ((v + 1) / 8) as usize);
            let (class, edits) = &regions[r[0] + r[1] * 5 + r[2] * 25];
            world.material_in_region(cell(q), *class, edits)
        })
    }

    /// Calls the sampler for the complete [-1, 32]^3 halo, exactly once per
    /// sample. The sampler supplies canonical material, including ordered edits.
    pub fn from_samples(mut sample: impl FnMut([i32; 3]) -> u32) -> Self {
        let index = |q: [i32; 3]| -> usize {
            ((q[0] + 1)
                + HALO_SIDE as i32 * (q[1] + 1)
                + (HALO_SIDE * HALO_SIDE) as i32 * (q[2] + 1)) as usize
        };
        let mut dense = vec![0u8; HALO_SIDE * HALO_SIDE * HALO_SIDE];
        for z in -1..=SIDE {
            for y in -1..=SIDE {
                for x in -1..=SIDE {
                    let value = sample([x, y, z]);
                    assert!(value < 4, "tiny backend supports four material slots");
                    dense[index([x, y, z])] = value as u8;
                }
            }
        }
        let mut summary = Summary::default();
        let mut descriptors = Vec::with_capacity(MICRO_COUNT);
        let mut dictionary: HashMap<[u32; 4], u32> = HashMap::new();
        let mut payloads: Vec<[u32; 4]> = Vec::new();
        for micro in 0..MICRO_COUNT {
            let base = [
                (micro % 8) as i32 * 4,
                (micro / 8 % 8) as i32 * 4,
                (micro / 64) as i32 * 4,
            ];
            let mut planes = [0u32; 4];
            let first = u32::from(dense[index(base)]);
            let mut uniform = true;
            for bit in 0..64u32 {
                let q = [
                    base[0] + (bit % 4) as i32,
                    base[1] + (bit / 4 % 4) as i32,
                    base[2] + (bit / 16) as i32,
                ];
                let value = u32::from(dense[index(q)]);
                uniform &= value == first;
                planes[(bit / 32) as usize] |= (value & 1) << (bit & 31);
                planes[2 + (bit / 32) as usize] |= ((value >> 1) & 1) << (bit & 31);
                if value == 0 {
                    continue;
                }
                summary.occupied += 1;
                for face in 0..6 {
                    let mut neighbour = q;
                    neighbour[face / 2] += if face % 2 == 0 { 1 } else { -1 };
                    if dense[index(neighbour)] == 0 {
                        summary.add_face(q, face, value as usize);
                    }
                }
            }
            let descriptor = if uniform {
                first
            } else {
                *dictionary.entry(planes).or_insert_with(|| {
                    let id = DICTIONARY + payloads.len() as u32;
                    payloads.push(planes);
                    id
                })
            };
            descriptors.push(descriptor);
        }
        let first = descriptors[0];
        let words = if first < DICTIONARY && descriptors.iter().all(|v| *v == first) {
            vec![first]
        } else {
            let mut words = Vec::with_capacity(1 + MICRO_COUNT + payloads.len() * 4);
            words.push(DICTIONARY);
            words.extend(descriptors);
            words.extend(payloads.into_iter().flatten());
            words
        };
        Self {
            words: words.into_boxed_slice(),
            summary,
        }
    }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Stats {
    pub built: u64,
    pub reused: u64,
    pub invalidated: u64,
    pub evicted: u64,
    pub resident: usize,
    /// Packed material words plus explicitly accounted summary data (128 B).
    /// Excludes allocator/hash-table overhead; this is not process RAM or VRAM.
    pub logical_bytes: usize,
}

/// Bounded, synchronous prototype cache. Build it off the render thread.
/// It owns immutable derived bricks; callers holding a predecessor Arc retain
/// a coherent old snapshot after edits or eviction. Admission never blocks on
/// an old reader. GPU residency/publication is a separate implementation step.
pub struct Cache {
    entries: BTreeMap<Key, (Arc<Brick>, u64)>,
    world: Option<Arc<World>>,
    byte_budget: usize,
    brick_budget: usize,
    clock: u64,
    stats: Stats,
}

impl Cache {
    pub fn new(byte_budget: usize, brick_budget: usize) -> Self {
        Self {
            entries: BTreeMap::new(),
            world: None,
            byte_budget,
            brick_budget,
            clock: 0,
            stats: Stats::default(),
        }
    }

    pub fn stats(&self) -> Stats {
        self.stats
    }

    pub fn set_world(&mut self, world: Arc<World>) {
        if self
            .world
            .as_ref()
            .is_some_and(|old| Arc::ptr_eq(old, &world))
        {
            return;
        }
        if let Some(old) = &self.world {
            let same_domain = old.voxel_step() == world.voxel_step()
                && old.generator_revision == world.generator_revision
                && old.landform_id == world.landform_id;
            let common = old
                .edits
                .iter()
                .zip(&world.edits)
                .take_while(|(a, b)| a == b)
                .count();
            let edits: Vec<_> = old.edits[common..]
                .iter()
                .chain(&world.edits[common..])
                .copied()
                .collect();
            let before = self.entries.len();
            self.entries.retain(|key, _| {
                same_domain && !edits.iter().any(|e| key.affected(world.voxel_step(), *e))
            });
            self.stats.invalidated += (before - self.entries.len()) as u64;
        }
        self.world = Some(world);
        self.recount();
    }

    fn recount(&mut self) {
        self.stats.resident = self.entries.len();
        self.stats.logical_bytes = self
            .entries
            .values()
            .map(|(b, _)| b.material_bytes() + 128)
            .sum();
    }

    /// Returns None when the brick cannot fit. A miss is never interpreted as
    /// empty terrain. This prototype returns data only after a complete build.
    pub fn get(&mut self, key: Key) -> Option<Arc<Brick>> {
        self.clock += 1;
        if let Some((brick, age)) = self.entries.get_mut(&key) {
            *age = self.clock;
            self.stats.reused += 1;
            return Some(brick.clone());
        }
        if self.brick_budget == 0 || self.byte_budget < 132 {
            return None;
        }
        let brick = Arc::new(Brick::from_world(
            self.world.as_ref().expect("set a source snapshot first"),
            key,
        ));
        self.stats.built += 1;
        let bytes = brick.material_bytes() + 128;
        if bytes > self.byte_budget {
            return None;
        }
        while self.entries.len() >= self.brick_budget
            || self.stats.logical_bytes + bytes > self.byte_budget
        {
            let victim = self
                .entries
                .iter()
                .min_by_key(|(_, (_, age))| *age)
                .map(|(key, _)| *key)
                .unwrap();
            self.entries.remove(&victim);
            self.stats.evicted += 1;
            self.recount();
        }
        self.entries.insert(key, (brick.clone(), self.clock));
        self.recount();
        Some(brick)
    }
}

#[cfg(test)]
mod tests;
