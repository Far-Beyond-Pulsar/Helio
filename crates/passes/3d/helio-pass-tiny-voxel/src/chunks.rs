//! Canonical 32-cubed chunks for gameplay, using two-bit voxel materials.
//! The owning World supplies planet identity. Eviction never discards saved edits.
use crate::{
    landforms::RegionClass,
    world::{Edit, World},
};
use std::sync::{Arc, Mutex};

pub const CAPACITY: usize = 8_192;
const WAYS: usize = 4;
pub const SIDE: i32 = 32;
pub const CELLS: usize = (SIDE * SIDE * SIDE) as usize;
#[cfg(test)]
mod tests;

pub fn key(cell: [i32; 3]) -> [i32; 3] {
    cell.map(|v| v.div_euclid(SIDE))
}

pub fn slot(key: [i32; 3]) -> usize {
    let [x, y, z] = key.map(|v| v as u32);
    let mut h =
        x.wrapping_mul(0x8da6b343) ^ y.wrapping_mul(0xd8163841) ^ z.wrapping_mul(0xcb1ab31f);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb352d);
    h ^= h >> 15;
    ((h as usize) & (CAPACITY / WAYS - 1)) * WAYS
}

pub struct Chunk {
    pub key: [i32; 3],
    // Uniform chunks need no per-voxel allocation. Mixed chunks retain materials
    // for CPU gameplay, in the same two-bit format as GPU terrain bricks.
    uniform: u32,
    data: Option<Box<ChunkData>>,
}
struct ChunkData {
    materials: [u32; CELLS / 16],
}
impl Chunk {
    pub fn material(&self, cell: [i32; 3]) -> u32 {
        let q = cell.map(|v| (v & (SIDE - 1)) as usize);
        let i = q[0] + q[1] * SIDE as usize + q[2] * (SIDE * SIDE) as usize;
        self.data.as_ref().map_or(self.uniform, |d| {
            (d.materials[i / 16] >> ((i % 16) * 2)) & 3
        })
    }

    fn intersects(&self, edit: Edit) -> bool {
        let low = self.key.map(|v| i64::from(v) * i64::from(SIDE));
        let r = i64::from(edit.radius_units().div_ceil(2));
        (0..3).all(|a| {
            low[a] <= i64::from(edit.cell[a]) + r
                && low[a] + i64::from(SIDE - 1) >= i64::from(edit.cell[a]) - r
        })
    }
    fn generate(world: &World, key: [i32; 3]) -> Self {
        let low = key.map(|v| v * SIDE);
        let high = low.map(|v| v.saturating_add(SIDE - 1));
        let edits = world.region_edits(low, high);
        // A final edit covering this complete chunk replaces its prior contents.
        if let Some(&i) = edits.last() {
            let e = world.edits[i];
            let far = std::array::from_fn(|a| {
                if (i64::from(high[a]) - i64::from(e.cell[a])).abs()
                    > (i64::from(low[a]) - i64::from(e.cell[a])).abs()
                {
                    high[a]
                } else {
                    low[a]
                }
            });
            if e.contains(far) {
                return Self {
                    key,
                    uniform: e.material,
                    data: None,
                };
            }
        }
        let class = crate::landforms::default_field()
            .classify(low, high)
            .map_or(RegionClass::Mixed, |v| v.classification);
        if edits.is_empty() && class != RegionClass::Mixed {
            return Self {
                key,
                uniform: u32::from(class == RegionClass::AllSolid),
                data: None,
            };
        }
        let mut classes = [class; 64];
        if class == RegionClass::Mixed {
            for (i, class) in classes.iter_mut().enumerate() {
                let lo = [
                    low[0] + (i & 3) as i32 * 8,
                    low[1] + ((i >> 2) & 3) as i32 * 8,
                    low[2] + (i >> 4) as i32 * 8,
                ];
                *class = crate::landforms::default_field()
                    .classify(lo, lo.map(|v| v + 7))
                    .map_or(RegionClass::Mixed, |b| b.classification);
            }
        }
        let mut values = vec![0u32; CELLS];
        for (i, value) in values.iter_mut().enumerate() {
            let x = i % 32;
            let y = (i / 32) % 32;
            let z = i / 1024;
            let cell = [low[0] + x as i32, low[1] + y as i32, low[2] + z as i32];
            *value = edits
                .iter()
                .rev()
                .find_map(|&i| {
                    let e = world.edits[i];
                    e.contains(cell).then_some(e.material)
                })
                .unwrap_or_else(|| match classes[x / 8 + (y / 8) * 4 + (z / 8) * 16] {
                    RegionClass::AllAir => 0,
                    RegionClass::AllSolid => 1,
                    RegionClass::Mixed => crate::world::base_material(cell),
                });
        }
        if values.iter().all(|&v| v == values[0]) {
            return Self {
                key,
                uniform: values[0],
                data: None,
            };
        }
        let mut data = Box::new(ChunkData {
            materials: [0; CELLS / 16],
        });
        for (i, v) in values.into_iter().enumerate() {
            data.materials[i / 16] |= v << ((i % 16) * 2);
        }
        Self {
            key,
            uniform: 0,
            data: Some(data),
        }
    }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Stats {
    pub generated: u64,
    pub hits: u64,
    pub invalidated: u64,
    pub resident: usize,
}
#[derive(Clone)]
struct State {
    entries: Vec<Option<Arc<Chunk>>>,
    edits: Vec<Edit>,
    stats: Stats,
    clock: u64,
    last_used: Vec<u64>,
}
impl Default for State {
    fn default() -> Self {
        Self {
            entries: vec![None; CAPACITY],
            edits: Vec::new(),
            stats: Stats::default(),
            clock: 0,
            last_used: vec![0; CAPACITY],
        }
    }
}
impl State {
    fn find(&self, key: [i32; 3]) -> Option<usize> {
        let start = slot(key);
        (start..start + WAYS).find(|&i| self.entries[i].as_ref().is_some_and(|c| c.key == key))
    }
    fn touch(&mut self, index: usize) {
        self.clock += 1;
        self.last_used[index] = self.clock;
    }
}
pub struct ChunkStore {
    state: Mutex<State>,
}
impl Default for ChunkStore {
    fn default() -> Self {
        Self {
            state: Mutex::new(State::default()),
        }
    }
}
impl Clone for ChunkStore {
    // An edited World can share immutable chunk payloads with its previous
    // snapshot, but must have its own residency/invalidation table.
    fn clone(&self) -> Self {
        Self {
            state: Mutex::new(self.state.lock().unwrap().clone()),
        }
    }
}
impl ChunkStore {
    #[cfg(test)]
    pub(crate) fn contains(&self, key: [i32; 3]) -> bool {
        let mut state = self.state.lock().unwrap();
        if let Some(i) = state.find(key) {
            state.touch(i);
            true
        } else {
            false
        }
    }
    pub fn material(&self, world: &World, cell: [i32; 3]) -> u32 {
        let key = key(cell);
        {
            let mut state = self.state.lock().unwrap();
            if let Some(index) = state.find(key) {
                let value = state.entries[index].as_ref().unwrap().material(cell);
                state.stats.hits += 1;
                state.touch(index);
                return value;
            }
        }
        let chunk = Arc::new(Chunk::generate(world, key));
        let value = chunk.material(cell);
        let mut state = self.state.lock().unwrap();
        state.stats.generated += 1;
        if let Some(index) = state.find(key) {
            state.touch(index);
            return value;
        }
        let start = slot(key);
        let index = (start..start + WAYS)
            .find(|&i| state.entries[i].is_none())
            .unwrap_or_else(|| {
                (start..start + WAYS)
                    .min_by_key(|&i| state.last_used[i])
                    .unwrap()
            });
        state.stats.resident += usize::from(state.entries[index].is_none());
        state.touch(index);
        state.entries[index] = Some(chunk);
        value
    }
    pub(crate) fn invalidate_edit(&self, edit: Edit) {
        let mut state = self.state.lock().unwrap();
        let mut removed = 0;
        for entry in &mut state.entries {
            if entry.as_ref().is_some_and(|c| c.intersects(edit)) {
                *entry = None;
                removed += 1;
            }
        }
        state.stats.invalidated += removed as u64;
        state.stats.resident -= removed;
        state.edits.push(edit);
    }
    pub(crate) fn reconcile(&self, edits: &[Edit]) {
        let mut state = self.state.lock().unwrap();
        let common = state
            .edits
            .iter()
            .zip(edits)
            .take_while(|(a, b)| a == b)
            .count();
        if common == edits.len() && common == state.edits.len() {
            return;
        }
        let changed: Vec<_> = state.edits[common..]
            .iter()
            .chain(&edits[common..])
            .copied()
            .collect();
        let mut removed = 0;
        for entry in &mut state.entries {
            if entry
                .as_ref()
                .is_some_and(|chunk| changed.iter().any(|&e| chunk.intersects(e)))
            {
                *entry = None;
                removed += 1;
            }
        }
        state.stats.invalidated += removed as u64;
        state.stats.resident -= removed;
        state.edits = edits.to_vec();
    }
    pub fn stats(&self) -> Stats {
        self.state.lock().unwrap().stats
    }
}
