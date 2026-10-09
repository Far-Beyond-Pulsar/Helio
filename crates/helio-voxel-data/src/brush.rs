//! Shape edits of a voxel world: the generic, persistent destruction and
//! construction journal every terrain generator understands.
//!
//! Brushes are ordered: a later brush wins where they overlap. Coordinates
//! are world metres relative to the terrain's origin, so the same journal
//! applies at every voxel size. Per-sample edits live in the payload chunks
//! instead (see `VoxelSampleEdit`).

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VoxelBrushShape {
    Sphere,
    /// An axis-aligned cube in the world's own cell axes.
    Cube,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VoxelBrushOp {
    /// Carve cells to air.
    Remove,
    /// Fill cells with `material`.
    Add,
    /// Recolour existing solid cells with `material`.
    Paint,
}

/// One journal entry.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct VoxelBrushEdit {
    pub center: [f64; 3],
    pub radius: f64,
    pub shape: VoxelBrushShape,
    pub op: VoxelBrushOp,
    /// Terrain material for `Add` and `Paint` (the generator's material
    /// table); ignored by `Remove`.
    #[serde(default)]
    pub material: u32,
}

impl VoxelBrushEdit {
    /// Finite centre, positive finite radius.
    pub fn validate(&self) -> Result<(), String> {
        if !self.center.iter().all(|v| v.is_finite()) || !(self.radius.is_finite() && self.radius > 0.0) {
            return Err("brush centre and radius must be finite, radius positive".into());
        }
        Ok(())
    }
}

/// Edits per shared chunk of a [`VoxelEditJournal`].
const JOURNAL_CHUNK: usize = 1024;

/// The ordered edit journal of a voxel world.
///
/// Copies share structure (edits live in shared chunks), and every edit
/// records a hash of the journal up to it, so the scene projection can hand
/// the journal to renderers every frame and they can tell in O(1) whether
/// it changed or only grew.
///
/// Edits belong to the terrain they were made on: `terrain` is that
/// terrain's fingerprint (form, voxel size, generator, seed and settings;
/// 0 before any). Another terrain starts with no edits ([`Self::made_on`]).
/// Serialized as `{"terrain": .., "edits": [..]}`.
#[derive(Clone, Default)]
pub struct VoxelEditJournal {
    chunks: Vec<std::sync::Arc<Vec<(VoxelBrushEdit, u64)>>>,
    len: usize,
    terrain: u64,
}

/// FNV-1a over an edit, continuing `seed`.
fn edit_hash(seed: u64, edit: &VoxelBrushEdit) -> u64 {
    let mut h = seed ^ 0xcbf2_9ce4_8422_2325;
    let mut eat = |v: u64| {
        for byte in v.to_le_bytes() {
            h = (h ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3);
        }
    };
    for c in edit.center {
        eat(c.to_bits());
    }
    eat(edit.radius.to_bits());
    eat(edit.shape as u64);
    eat(edit.op as u64);
    eat(u64::from(edit.material));
    h
}

impl VoxelEditJournal {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn get(&self, index: usize) -> Option<&VoxelBrushEdit> {
        (index < self.len).then(|| &self.chunks[index / JOURNAL_CHUNK][index % JOURNAL_CHUNK].0)
    }
    /// Hash of the first `count` edits (0 for none).
    pub fn prefix_hash(&self, count: usize) -> u64 {
        match count {
            0 => 0,
            n => self.chunks[(n - 1) / JOURNAL_CHUNK][(n - 1) % JOURNAL_CHUNK].1,
        }
    }
    pub fn push(&mut self, edit: VoxelBrushEdit) {
        let hash = edit_hash(self.prefix_hash(self.len), &edit);
        if self.len % JOURNAL_CHUNK == 0 {
            self.chunks.push(std::sync::Arc::new(Vec::with_capacity(JOURNAL_CHUNK)));
        }
        // Copies only this chunk when another journal still shares it.
        std::sync::Arc::make_mut(self.chunks.last_mut().expect("pushed above")).push((edit, hash));
        self.len += 1;
    }
    /// Remove the newest edit.
    pub fn pop(&mut self) -> Option<VoxelBrushEdit> {
        let chunk = self.chunks.last_mut()?;
        let (edit, _) = std::sync::Arc::make_mut(chunk).pop()?;
        if chunk.is_empty() {
            self.chunks.pop();
        }
        self.len -= 1;
        Some(edit)
    }
    pub fn iter(&self) -> impl Iterator<Item = &VoxelBrushEdit> {
        self.chunks.iter().flat_map(|chunk| chunk.iter()).map(|(edit, _)| edit)
    }
    /// Edits from `start` on.
    pub fn iter_from(&self, start: usize) -> impl Iterator<Item = &VoxelBrushEdit> {
        self.iter().skip(start)
    }
    /// Whether `prefix` is this journal's beginning (O(1), by hash).
    pub fn starts_with(&self, prefix: &VoxelEditJournal) -> bool {
        prefix.terrain == self.terrain && prefix.len <= self.len && self.prefix_hash(prefix.len) == prefix.prefix_hash(prefix.len)
    }
    /// Fingerprint of the terrain the edits were made on (0: none yet).
    pub fn terrain(&self) -> u64 {
        self.terrain
    }
    /// Make the journal belong to `terrain`: edits made on another terrain
    /// are dropped (they would land on ground that is not there); a journal
    /// with no terrain yet adopts it. Returns how many edits were dropped.
    pub fn made_on(&mut self, terrain: u64) -> usize {
        if self.terrain == terrain {
            return 0;
        }
        if self.terrain == 0 {
            self.terrain = terrain;
            return 0;
        }
        let dropped = self.len;
        *self = Self { terrain, ..Self::default() };
        dropped
    }
}

impl PartialEq for VoxelEditJournal {
    /// Equal length and equal prefix hash (O(1)).
    fn eq(&self, other: &Self) -> bool {
        self.terrain == other.terrain && self.len == other.len && self.prefix_hash(self.len) == other.prefix_hash(other.len)
    }
}

impl std::fmt::Debug for VoxelEditJournal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VoxelEditJournal").field("terrain", &self.terrain).field("len", &self.len).finish()
    }
}

impl FromIterator<VoxelBrushEdit> for VoxelEditJournal {
    fn from_iter<I: IntoIterator<Item = VoxelBrushEdit>>(edits: I) -> Self {
        let mut journal = Self::default();
        edits.into_iter().for_each(|edit| journal.push(edit));
        journal
    }
}

impl Extend<VoxelBrushEdit> for VoxelEditJournal {
    fn extend<I: IntoIterator<Item = VoxelBrushEdit>>(&mut self, edits: I) {
        edits.into_iter().for_each(|edit| self.push(edit));
    }
}

/// The saved form of a [`VoxelEditJournal`].
#[derive(Serialize, Deserialize)]
struct SavedJournal<E> {
    #[serde(default)]
    terrain: u64,
    #[serde(default)]
    edits: E,
}

impl Serialize for VoxelEditJournal {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let edits: Vec<&VoxelBrushEdit> = self.iter().collect();
        SavedJournal { terrain: self.terrain, edits }.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for VoxelEditJournal {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let saved = SavedJournal::<Vec<VoxelBrushEdit>>::deserialize(deserializer)?;
        let mut journal: Self = saved.edits.into_iter().collect();
        journal.terrain = saved.terrain;
        Ok(journal)
    }
}

#[cfg(test)]
mod journal_tests {
    use super::*;

    fn edit(x: f64) -> VoxelBrushEdit {
        VoxelBrushEdit { center: [x, 0.0, 0.0], radius: 1.0, shape: VoxelBrushShape::Sphere, op: VoxelBrushOp::Remove, material: 0 }
    }

    #[test]
    fn journals_compare_by_prefix_and_serialize_with_their_terrain() {
        let a: VoxelEditJournal = (0..3000).map(|k| edit(k as f64)).collect();
        let mut b = a.clone();
        b.push(edit(-1.0));
        assert!(b.starts_with(&a) && !a.starts_with(&b) && a != b);
        assert_eq!(b.pop(), Some(edit(-1.0)));
        assert_eq!(a, b);
        b.pop();
        b.push(edit(12345.0));
        assert!(!b.starts_with(&a), "a changed edit is no longer a prefix");
        assert_eq!(a.get(2999), Some(&edit(2999.0)));
        assert_eq!(a.iter_from(2998).count(), 2);

        let mut a = a;
        assert_eq!(a.made_on(6), 0, "a journal without a terrain adopts the first");
        assert_eq!((a.terrain(), a.len()), (6, 3000));
        assert_eq!(a.made_on(7), 3000, "edits made on another terrain are dropped");
        assert!(a.is_empty() && a.terrain() == 7);
        a.extend((0..3).map(|k| edit(k as f64)));
        assert_eq!(a.made_on(7), 0);
        let json = serde_json::to_value(&a).unwrap();
        assert_eq!(json["terrain"], 7);
        assert_eq!(json["edits"].as_array().unwrap().len(), 3);
        let back: VoxelEditJournal = serde_json::from_value(json).unwrap();
        assert_eq!(back, a);
        let mut other = back.clone();
        other.made_on(8);
        assert!(!other.starts_with(&back) && other != back);
    }
}
