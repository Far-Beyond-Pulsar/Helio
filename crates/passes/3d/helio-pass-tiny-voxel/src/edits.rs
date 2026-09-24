//! Incremental spatial bins for chronological voxel edits. An ordinary append
//! touches at most eight bins, independent of the total world edit count.
use crate::world::Edit;
use std::collections::HashMap;

#[derive(Clone, Default)]
pub struct EditIndex {
    levels: Vec<HashMap<[i32; 3], Vec<usize>>>,
    indexed: Vec<Edit>,
    pub bounds: Option<([i32; 3], [i32; 3])>,
}
impl EditIndex {
    pub fn len(&self) -> usize {
        self.indexed.len()
    }
    pub fn reconcile(&mut self, edits: &[Edit]) {
        let common = self
            .indexed
            .iter()
            .zip(edits)
            .take_while(|(a, b)| a == b)
            .count();
        if common < self.indexed.len() {
            *self = Self::default();
        }
        for &edit in &edits[self.indexed.len()..] {
            self.append(edit);
        }
    }
    pub fn append(&mut self, edit: Edit) {
        let index = self.indexed.len();
        let radius = edit.radius_units().div_ceil(2) as i32;
        let low = edit.cell.map(|v| v.saturating_sub(radius));
        let high = edit.cell.map(|v| v.saturating_add(radius));
        let side = (radius as u32 * 2 + 1).max(32).next_power_of_two();
        let level = side.trailing_zeros() as usize - 5;
        self.levels
            .resize_with(self.levels.len().max(level + 1), HashMap::new);
        let a = low.map(|v| v.div_euclid(side as i32));
        let b = high.map(|v| v.div_euclid(side as i32));
        for z in a[2]..=b[2] {
            for y in a[1]..=b[1] {
                for x in a[0]..=b[0] {
                    self.levels[level].entry([x, y, z]).or_default().push(index);
                }
            }
        }
        self.bounds = Some(self.bounds.map_or((low, high), |(a, b)| {
            (
                std::array::from_fn(|i| a[i].min(low[i])),
                std::array::from_fn(|i| b[i].max(high[i])),
            )
        }));
        self.indexed.push(edit);
    }
    pub fn latest(&self, cell: [i32; 3]) -> Option<usize> {
        let mut latest = None;
        for (level, bins) in self.levels.iter().enumerate() {
            let key = cell.map(|v| v.div_euclid(32i32 << level));
            if let Some(indices) = bins.get(&key) {
                for &i in indices.iter().rev() {
                    if latest.is_some_and(|n| n >= i) {
                        break;
                    }
                    if self.indexed[i].contains(cell) {
                        latest = Some(i);
                        break;
                    }
                }
            }
        }
        latest
    }
    pub fn region(&self, low: [i32; 3], high: [i32; 3]) -> Vec<usize> {
        let mut result = Vec::new();
        for (level, bins) in self.levels.iter().enumerate() {
            let size = 32i32 << level;
            let a = low.map(|v| v.div_euclid(size));
            let b = high.map(|v| v.div_euclid(size));
            let count = (0..3).fold(1u64, |n, i| {
                n.saturating_mul((i64::from(b[i]) - i64::from(a[i]) + 1) as u64)
            });
            if count > bins.len() as u64 {
                for (key, values) in bins {
                    if (0..3).all(|i| key[i] >= a[i] && key[i] <= b[i]) {
                        result.extend_from_slice(values);
                    }
                }
            } else {
                for z in a[2]..=b[2] {
                    for y in a[1]..=b[1] {
                        for x in a[0]..=b[0] {
                            if let Some(values) = bins.get(&[x, y, z]) {
                                result.extend_from_slice(values);
                            }
                        }
                    }
                }
            }
        }
        result.sort_unstable();
        result.dedup();
        result.retain(|&i| {
            let e = self.indexed[i];
            let r = i64::from(e.radius_units().div_ceil(2));
            (0..3).all(|a| {
                i64::from(low[a]) <= i64::from(e.cell[a]) + r
                    && i64::from(high[a]) >= i64::from(e.cell[a]) - r
            })
        });
        result
    }
}
