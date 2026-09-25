use super::{LandformSnapshot, MAX_HEIGHT_UNITS};

pub const BOUNDS_SHADER: &str = include_str!("bounds.wgsl");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum RegionClass {
    Mixed = 0,
    AllAir = 1,
    AllSolid = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegionBounds {
    pub height_units: [i32; 2],
    pub classification: RegionClass,
}

/// A validated immutable snapshot and its conservative patch hierarchy.
/// This owns the snapshot so a hierarchy cannot be queried with another field.
/// Certificates describe this radial base field, before fine detail or edits.
pub struct BoundedLandforms {
    snapshot: LandformSnapshot,
    hierarchy: Vec<[i32; 2]>,
}

impl BoundedLandforms {
    pub fn new(snapshot: LandformSnapshot) -> Self {
        let n = snapshot.resolution() as usize;
        let side = n + 1;
        let stride = (4 * n * n - 1) / 3;
        let h = snapshot.height_atlas();
        let mut hierarchy = Vec::<[i32; 2]>::with_capacity(6 * stride);
        for face in 0..6 {
            for v in 0..n {
                for u in 0..n {
                    let i = face * side * side + v * side + u;
                    let corners = [h[i], h[i + 1], h[i + side], h[i + side + 1]];
                    hierarchy.push([
                        *corners.iter().min().unwrap(),
                        *corners.iter().max().unwrap(),
                    ]);
                }
            }
            let mut width = n;
            let mut offset = face * stride;
            while width > 1 {
                for v in 0..width / 2 {
                    for u in 0..width / 2 {
                        let i = offset + v * 2 * width + u * 2;
                        let children = [
                            hierarchy[i],
                            hierarchy[i + 1],
                            hierarchy[i + width],
                            hierarchy[i + width + 1],
                        ];
                        hierarchy.push([
                            children.iter().map(|c| c[0]).min().unwrap(),
                            children.iter().map(|c| c[1]).max().unwrap(),
                        ]);
                    }
                }
                offset += width * width;
                width /= 2;
            }
        }
        Self {
            snapshot,
            hierarchy,
        }
    }

    pub fn snapshot(&self) -> &LandformSnapshot {
        &self.snapshot
    }
    pub fn hierarchy(&self) -> &[[i32; 2]] {
        &self.hierarchy
    }

    // Exact sampling-revision-1 interpolation at a patch-local Q16 address.
    fn patch_sample(&self, face: u32, u: u32, v: u32, t: [u32; 2]) -> i32 {
        let side = self.snapshot.resolution() as usize + 1;
        let i = face as usize * side * side + v as usize * side + u as usize;
        let h = self.snapshot.height_atlas();
        let lerp = |a: i32, b: i32, t: u32| {
            a + (((i64::from(b) - i64::from(a)) * i64::from(t)) >> 16) as i32
        };
        lerp(
            lerp(h[i], h[i + 1], t[0]),
            lerp(h[i + side], h[i + side + 1], t[0]),
            t[1],
        )
    }

    fn rectangle(&self, face: u32, low: [u32; 2], high: [u32; 2]) -> [i32; 2] {
        let n = self.snapshot.resolution();
        let first = low.map(|q| (q >> 16).min(n - 1));
        let last = high.map(|q| (q >> 16).min(n - 1));
        let mut result = [MAX_HEIGHT_UNITS, -MAX_HEIGHT_UNITS];
        if last[0] - first[0] <= 1 && last[1] - first[1] <= 1 {
            for v in first[1]..=last[1] {
                for u in first[0]..=last[0] {
                    let origin = [u * 65536, v * 65536];
                    let a = std::array::from_fn::<_, 2, _>(|i| {
                        low[i].saturating_sub(origin[i]).min(65536)
                    });
                    let b = std::array::from_fn::<_, 2, _>(|i| {
                        high[i].saturating_sub(origin[i]).min(65536)
                    });
                    for t in [[a[0], a[1]], [b[0], a[1]], [a[0], b[1]], [b[0], b[1]]] {
                        let value = self.patch_sample(face, u, v, t);
                        // Real bilinear extrema occur at rectangle corners. Two
                        // staged signed floors differ from that real value by <2.
                        result[0] = result[0].min(value - 2);
                        result[1] = result[1].max(value + 2);
                    }
                }
            }
            result[0] = result[0].max(-MAX_HEIGHT_UNITS);
            result[1] = result[1].min(MAX_HEIGHT_UNITS);
        } else {
            let span = (last[0] - first[0] + 1).max(last[1] - first[1] + 1);
            let level = span.next_power_of_two().trailing_zeros();
            let width = n >> level;
            let stride = (4 * n * n - 1) / 3;
            let offset = face * stride + (4 * n * n - 4 * width * width) / 3;
            // A rectangle no wider than one tile spans at most two tiles per
            // axis. Including their extra patches only makes the bound looser.
            for v in first[1] >> level..=last[1] >> level {
                for u in first[0] >> level..=last[0] >> level {
                    let range = self.hierarchy[(offset + v * width + u) as usize];
                    result[0] = result[0].min(range[0]);
                    result[1] = result[1].max(range[1]);
                }
            }
        }
        result
    }

    /// Classify every cell in an inclusive AABB. Invalid/domain-crossing boxes
    /// return None, never an empty-space certificate. Cell-boundary traversal
    /// can skip an AllAir box's full voxel cubes, not just their centers.
    pub fn classify(&self, minimum: [i32; 3], maximum: [i32; 3]) -> Option<RegionBounds> {
        if (0..3).any(|a| {
            minimum[a] > maximum[a] || minimum[a] < -100_000_000 || maximum[a] > 100_000_000
        }) {
            return None;
        }
        let lo = minimum.map(|v| i64::from(v) * 2 + 1);
        let hi = maximum.map(|v| i64::from(v) * 2 + 1);
        let closest = std::array::from_fn::<_, 3, _>(|a| {
            if lo[a] < 0 && hi[a] > 0 {
                1
            } else {
                lo[a].abs().min(hi[a].abs())
            }
        });
        let farthest = std::array::from_fn::<_, 3, _>(|a| lo[a].abs().max(hi[a].abs()));
        let n = i64::from(self.snapshot.resolution());
        let mut height = [MAX_HEIGHT_UNITS, -MAX_HEIGHT_UNITS];
        for face in 0..6u32 {
            let axis = face as usize / 2;
            let other = match axis {
                0 => [1, 2],
                1 => [0, 2],
                _ => [0, 1],
            };
            let (d0, d1) = if face % 2 == 0 {
                (lo[axis].max(1), hi[axis])
            } else {
                ((-hi[axis]).max(1), -lo[axis])
            };
            let d0 = d0.max(closest[other[0]]).max(closest[other[1]]);
            if d1 < d0 {
                continue;
            }
            let projected = other.map(|a| {
                let mut range = [u32::MAX, 0];
                for d in [d0, d1] {
                    for c in [lo[a], hi[a]] {
                        let c = c.clamp(-d, d);
                        let q = (((c + d) * n * 65536) / (2 * d)) as u32;
                        range[0] = range[0].min(q);
                        range[1] = range[1].max(q);
                    }
                }
                range
            });
            let range = self.rectangle(
                face,
                [projected[0][0], projected[1][0]],
                [projected[0][1], projected[1][1]],
            );
            height[0] = height[0].min(range[0]);
            height[1] = height[1].max(range[1]);
        }
        let inner = 127_420_000 + height[0] as i64;
        let outer = 127_420_000 + height[1] as i64;
        let near_squared = closest.iter().map(|a| a * a).sum::<i64>();
        let far_squared = farthest.iter().map(|a| a * a).sum::<i64>();
        let classification = if near_squared > outer * outer {
            RegionClass::AllAir
        } else if far_squared <= inner * inner {
            RegionClass::AllSolid
        } else {
            RegionClass::Mixed
        };
        Some(RegionBounds {
            height_units: height,
            classification,
        })
    }
}
