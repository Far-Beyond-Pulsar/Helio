use super::{BoundedLandforms, RegionBounds, RegionClass};
use std::sync::Arc;

pub const VOLUME_REVISION: u32 = 1;
pub const VOLUME_SHADER: &str = include_str!("volume.wgsl");
const LIMIT: i32 = 100_000_000;

/// An exact voxel mask. Radius is in 0.05m units, fixed once on the host.
/// CPU/GPU never round the same floating brush size independently.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VoxelEdit {
    cell: [i32; 3],
    radius_units: u32,
    material: u32,
    padding: [u32; 3],
}
impl VoxelEdit {
    pub fn new(cell: [i32; 3], radius_units: u32, material: u32) -> Result<Self, String> {
        if cell.iter().any(|c| !(-LIMIT..=LIMIT).contains(c))
            || !(1..=254_840_000).contains(&radius_units)
            || material > 3
        {
            return Err("Invalid canonical voxel edit".into());
        }
        Ok(Self {
            cell,
            radius_units,
            material,
            padding: [0; 3],
        })
    }
    pub fn cell(&self) -> [i32; 3] {
        self.cell
    }
    pub fn radius_units(&self) -> u32 {
        self.radius_units
    }
    pub fn material(&self) -> u32 {
        self.material
    }
    pub fn contains(&self, c: [i32; 3]) -> bool {
        let delta = std::array::from_fn::<_, 3, _>(|a| {
            (i64::from(c[a]) - i64::from(self.cell[a])).unsigned_abs()
        });
        within(delta, self.radius_units)
    }
    fn relation(&self, low: [i32; 3], high: [i32; 3]) -> (bool, bool) {
        let near = std::array::from_fn::<_, 3, _>(|a| {
            (i64::from(low[a]) - i64::from(self.cell[a]))
                .max(i64::from(self.cell[a]) - i64::from(high[a]))
                .max(0) as u64
        });
        if !within(near, self.radius_units) {
            return (false, false);
        }
        let far = std::array::from_fn::<_, 3, _>(|a| {
            (i64::from(low[a]) - i64::from(self.cell[a]))
                .unsigned_abs()
                .max((i64::from(high[a]) - i64::from(self.cell[a])).unsigned_abs())
        });
        (true, within(far, self.radius_units))
    }
}
fn within(delta: [u64; 3], radius: u32) -> bool {
    // Early per-axis rejection also bounds squaring for arbitrary caller cells.
    delta.iter().all(|d| *d <= u64::from(radius) / 2)
        && delta.iter().map(|d| d * d).sum::<u64>() * 4 <= u64::from(radius).pow(2)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VoxelSample {
    pub material: u32,
    pub edit_source: u32,
}

/// Immutable landforms plus a new volumetric detail recipe and ordered edits.
/// Source index is one-based; zero denotes the unedited procedural field.
#[derive(Clone)]
pub struct VoxelField {
    landforms: Arc<BoundedLandforms>,
    seed: u32,
    edits: Vec<VoxelEdit>,
}
impl VoxelField {
    pub fn new(landforms: Arc<BoundedLandforms>, seed: u32) -> Self {
        Self {
            landforms,
            seed,
            edits: Vec::new(),
        }
    }
    pub fn landforms(&self) -> &BoundedLandforms {
        &self.landforms
    }
    pub fn detail_seed(&self) -> u32 {
        self.seed
    }
    pub fn edits(&self) -> &[VoxelEdit] {
        &self.edits
    }
    pub fn gpu_settings(&self) -> [u32; 4] {
        [
            self.landforms.snapshot().resolution(),
            self.seed,
            self.edits.len() as u32,
            VOLUME_REVISION,
        ]
    }
    pub fn push_edit(&mut self, edit: VoxelEdit) -> Result<(), String> {
        if self.edits.len() >= crate::world::MAX_EDITS {
            return Err("Voxel edit capacity reached; no edit was discarded".into());
        }
        // Revalidate because POD values can be constructed from external bytes.
        self.edits
            .push(VoxelEdit::new(edit.cell, edit.radius_units, edit.material)?);
        Ok(())
    }
    pub fn height_units(&self, c: [i32; 3]) -> Option<i32> {
        let base = self.landforms.snapshot().sample_cell(c)?.height_units;
        Some(base + detail(c, self.seed))
    }
    pub fn sample_cell(&self, c: [i32; 3]) -> Option<VoxelSample> {
        if c.iter().any(|c| !(-LIMIT..=LIMIT).contains(c)) {
            return None;
        }
        for (i, e) in self.edits.iter().enumerate().rev() {
            if e.contains(c) {
                return Some(VoxelSample {
                    material: e.material,
                    edit_source: i as u32 + 1,
                });
            }
        }
        let radius = 127_420_000 + i64::from(self.height_units(c)?);
        let squared = c
            .iter()
            .map(|v| (i64::from(*v) * 2 + 1).pow(2))
            .sum::<i64>();
        Some(VoxelSample {
            material: u32::from(squared <= radius * radius),
            edit_source: 0,
        })
    }
    /// Height bounds describe the procedural field even when an edit replaces
    /// occupancy. Classification includes every edit in chronological order.
    pub fn classify(&self, low: [i32; 3], high: [i32; 3]) -> Option<RegionBounds> {
        let base = self.landforms.classify(low, high)?;
        let d = detail_range(low, high, self.seed);
        let height = [base.height_units[0] + d[0], base.height_units[1] + d[1]];
        let near = std::array::from_fn::<_, 3, _>(|a| {
            let x = i64::from(low[a]) * 2 + 1;
            let y = i64::from(high[a]) * 2 + 1;
            if x < 0 && y > 0 {
                1
            } else {
                x.abs().min(y.abs())
            }
        });
        let far = std::array::from_fn::<_, 3, _>(|a| {
            (i64::from(low[a]) * 2 + 1)
                .abs()
                .max((i64::from(high[a]) * 2 + 1).abs())
        });
        let inner = 127_420_000 + i64::from(height[0]);
        let outer = 127_420_000 + i64::from(height[1]);
        let mut class = if near.iter().map(|x| x * x).sum::<i64>() > outer * outer {
            RegionClass::AllAir
        } else if far.iter().map(|x| x * x).sum::<i64>() <= inner * inner {
            RegionClass::AllSolid
        } else {
            RegionClass::Mixed
        };
        for e in &self.edits {
            let (overlap, covered) = e.relation(low, high);
            if !overlap {
                continue;
            }
            let target = if e.material == 0 {
                RegionClass::AllAir
            } else {
                RegionClass::AllSolid
            };
            if covered {
                class = target;
            } else if class != target {
                class = RegionClass::Mixed;
            }
        }
        Some(RegionBounds {
            height_units: height,
            classification: class,
        })
    }
}

fn hash(c: [i32; 3], seed: u32) -> u32 {
    let mut h = (c[0] as u32).wrapping_mul(0x8da6b343)
        ^ (c[1] as u32).wrapping_mul(0xd8163841)
        ^ (c[2] as u32).wrapping_mul(0xcb1ab31f)
        ^ seed;
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846ca68b);
    h ^ (h >> 16)
}
fn lerp(a: i32, b: i32, t: u32) -> i32 {
    a + (((i64::from(b) - i64::from(a)) * i64::from(t)) >> 16) as i32
}
fn noise(c: [i32; 3], shift: u32, seed: u32) -> i32 {
    let size = 1i32 << shift;
    let offsets = [
        seed.wrapping_mul(0x9e3779b9) ^ 0xa341316c,
        seed.wrapping_mul(0x85ebca6b) ^ 0xc8013ea4,
        seed.wrapping_mul(0xc2b2ae35) ^ 0xad90777d,
    ]
    .map(|v| (v & (size as u32 - 1)) as i32);
    let p = std::array::from_fn::<_, 3, _>(|a| c[a] + offsets[a]);
    let base = p.map(|v| v.div_euclid(size));
    let t = p.map(|v| {
        let t = ((v.rem_euclid(size) as u64 * 2 + 1) << (15 - shift)) as u64;
        ((t * t * (3 * 65536 - 2 * t)) >> 32) as u32
    });
    let v = std::array::from_fn::<_, 8, _>(|i| {
        (hash(
            [
                base[0] + (i & 1) as i32,
                base[1] + ((i >> 1) & 1) as i32,
                base[2] + ((i >> 2) & 1) as i32,
            ],
            seed,
        ) & 65535) as i32
    });
    lerp(
        lerp(lerp(v[0], v[1], t[0]), lerp(v[2], v[3], t[0]), t[1]),
        lerp(lerp(v[4], v[5], t[0]), lerp(v[6], v[7], t[0]), t[1]),
        t[2],
    )
}
fn scale(v: i32, amplitude: i32) -> i32 {
    ((i64::from(v) * i64::from(amplitude)) >> 15) as i32
}
fn detail(c: [i32; 3], seed: u32) -> i32 {
    scale(noise(c, 14, seed ^ 73) - 32768, 8000)
        + scale(16384 - (noise(c, 10, seed ^ 191) - 32768).abs(), 480)
        + scale(noise(c, 7, seed ^ 311) - 32768, 40)
}
fn noise_range(low: [i32; 3], high: [i32; 3], shift: u32, seed: u32) -> [i32; 2] {
    let middle = std::array::from_fn::<_, 3, _>(|a| low[a] + (high[a] - low[a]) / 2);
    let displacement = (0..3)
        .map(|a| (middle[a] - low[a]).max(high[a] - middle[a]) as u32)
        .sum::<u32>();
    let size = 1u32 << shift;
    if 3 * displacement >= 2 * size {
        return [0, 65535];
    }
    let value = noise(middle, shift, seed);
    if displacement == 0 {
        return [value, value];
    }
    // The early full-range case bounds this numerator below 2^31 for the
    // largest supported layer (shift 14); the divisor is a power of two.
    let error = ((3 * 65535 * displacement + 2 * size - 1) >> (shift + 1)) + 12;
    [
        (value - error as i32).max(0),
        (value + error as i32).min(65535),
    ]
}
fn detail_range(low: [i32; 3], high: [i32; 3], seed: u32) -> [i32; 2] {
    let a = noise_range(low, high, 14, seed ^ 73).map(|v| scale(v - 32768, 8000));
    let ridge = noise_range(low, high, 10, seed ^ 191).map(|v| v - 32768);
    let nearest = if ridge[0] <= 0 && ridge[1] >= 0 {
        0
    } else {
        ridge[0].abs().min(ridge[1].abs())
    };
    let farthest = ridge[0].abs().max(ridge[1].abs());
    let b = [scale(16384 - farthest, 480), scale(16384 - nearest, 480)];
    let c = noise_range(low, high, 7, seed ^ 311).map(|v| scale(v - 32768, 40));
    [a[0] + b[0] + c[0], a[1] + b[1] + c[1]]
}
