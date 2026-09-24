use super::{face_key, GlobalTopology};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

const MAGIC: &[u8; 8] = b"HLFM0001";
const FORMAT: u32 = 1;
const SAMPLING: u32 = 1;
const MAX_BYTES: usize = 80 + 6 * 257 * 257 * 8;
pub const MAX_HEIGHT_UNITS: i32 = 40_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SnapshotId(pub [u8; 32]);
impl std::fmt::Display for SnapshotId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for b in self.0 {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

/// Provenance is included in the identity; it never substitutes for the actual
/// published values. Backend changes cannot silently regenerate this snapshot.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LandformRecipe {
    pub revision: u32,
    pub seed: u32,
    pub iterations: u32,
    pub backend_revision: u32,
    pub rain_scale: f32,
    pub erosion_rate: f32,
}
impl LandformRecipe {
    fn normalized(mut self) -> Result<Self, String> {
        if self.revision == 0
            || self.backend_revision == 0
            || !self.rain_scale.is_finite()
            || self.rain_scale < 0.0
            || !self.erosion_rate.is_finite()
            || self.erosion_rate < 0.0
        {
            return Err("Invalid landform recipe metadata".into());
        }
        if self.rain_scale == 0.0 {
            self.rain_scale = 0.0;
        }
        if self.erosion_rate == 0.0 {
            self.erosion_rate = 0.0;
        }
        Ok(self)
    }
}

/// Immutable face atlas. Duplicate edge/corner records are validated against
/// one rational node identity; rendering can upload only `height_atlas()`.
pub struct LandformSnapshot {
    pub(super) resolution: u32,
    recipe: LandformRecipe,
    pub(super) heights: Vec<i32>,
    streams: Vec<u32>,
    id: SnapshotId,
}

impl LandformSnapshot {
    pub fn from_state(
        topology: &GlobalTopology,
        recipe: LandformRecipe,
        state: &[[f32; 2]],
    ) -> Result<Self, String> {
        if state.len() != topology.nodes().len()
            || state
                .iter()
                .any(|s| !s[0].is_finite() || !s[1].is_finite() || s[1] < 0.0)
        {
            return Err("Invalid landform generation state".into());
        }
        // f64 exactly represents the f32 source and this multiplication by 20;
        // round once, halfway away from zero, into 0.05m integer height units.
        let units: Vec<_> = state
            .iter()
            .map(|s| (f64::from(s[0]) * 20.0).round())
            .collect();
        if units.iter().any(|h| h.abs() > f64::from(MAX_HEIGHT_UNITS)) {
            return Err("Landform height exceeds supported radius range".into());
        }
        let units: Vec<_> = units.into_iter().map(|h| h as i32).collect();
        let streams: Vec<_> = state.iter().map(|s| s[1]).collect();
        Self::from_units(topology, recipe, &units, &streams)
    }

    pub fn from_units(
        topology: &GlobalTopology,
        recipe: LandformRecipe,
        units: &[i32],
        streams: &[f32],
    ) -> Result<Self, String> {
        let recipe = recipe.normalized()?;
        if units.len() != topology.nodes().len()
            || streams.len() != units.len()
            || units
                .iter()
                .any(|h| !(-MAX_HEIGHT_UNITS..=MAX_HEIGHT_UNITS).contains(h))
            || streams.iter().any(|s| !s.is_finite() || *s < 0.0)
        {
            return Err("Invalid published landform records".into());
        }
        let heights = topology
            .face_nodes()
            .iter()
            .map(|i| units[*i as usize])
            .collect();
        let streams = topology
            .face_nodes()
            .iter()
            .map(|i| {
                let s = streams[*i as usize];
                if s == 0.0 {
                    0
                } else {
                    s.to_bits()
                }
            })
            .collect();
        let mut snapshot = Self {
            resolution: topology.resolution(),
            recipe,
            heights,
            streams,
            id: SnapshotId([0; 32]),
        };
        snapshot.id = SnapshotId(Sha256::digest(snapshot.payload()).into());
        Ok(snapshot)
    }

    pub fn id(&self) -> SnapshotId {
        self.id
    }
    pub fn resolution(&self) -> u32 {
        self.resolution
    }
    pub fn recipe(&self) -> LandformRecipe {
        self.recipe
    }
    pub fn height_atlas(&self) -> &[i32] {
        &self.heights
    }
    pub fn stream_bits_atlas(&self) -> &[u32] {
        &self.streams
    }

    fn payload(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(48 + self.heights.len() * 8);
        bytes.extend_from_slice(MAGIC);
        for word in [
            FORMAT,
            SAMPLING,
            self.resolution,
            self.recipe.revision,
            self.recipe.seed,
            self.recipe.iterations,
            self.recipe.backend_revision,
            self.recipe.rain_scale.to_bits(),
            self.recipe.erosion_rate.to_bits(),
            self.heights.len() as u32,
        ] {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        for value in &self.heights {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in &self.streams {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = self.payload();
        bytes.extend_from_slice(&self.id.0);
        bytes
    }

    pub fn decode(bytes: &[u8], expected: SnapshotId) -> Result<Self, String> {
        if bytes.len() < 80 || bytes.len() > MAX_BYTES || &bytes[..8] != MAGIC {
            return Err("Invalid landform snapshot header/size".into());
        }
        let payload = &bytes[..bytes.len() - 32];
        let digest: [u8; 32] = Sha256::digest(payload).into();
        if digest != expected.0 || bytes[bytes.len() - 32..] != digest {
            return Err("Landform snapshot identity mismatch".into());
        }
        let word = |index: usize| {
            u32::from_le_bytes(bytes[8 + index * 4..12 + index * 4].try_into().unwrap())
        };
        if word(0) != FORMAT || word(1) != SAMPLING {
            return Err("Unsupported landform format or sampling revision".into());
        }
        let n = word(2);
        if !n.is_power_of_two() || n > 256 {
            return Err("Invalid landform face resolution".into());
        }
        let count = 6 * (n as usize + 1).pow(2);
        if word(9) as usize != count || bytes.len() != 80 + count * 8 {
            return Err("Invalid landform record count".into());
        }
        let recipe = LandformRecipe {
            revision: word(3),
            seed: word(4),
            iterations: word(5),
            backend_revision: word(6),
            rain_scale: f32::from_bits(word(7)),
            erosion_rate: f32::from_bits(word(8)),
        }
        .normalized()?;
        if recipe.rain_scale.to_bits() != word(7) || recipe.erosion_rate.to_bits() != word(8) {
            return Err("Noncanonical recipe zero".into());
        }
        let heights: Vec<_> = bytes[48..48 + count * 4]
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let streams: Vec<_> = bytes[48 + count * 4..48 + count * 8]
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        if heights
            .iter()
            .any(|h| !(-MAX_HEIGHT_UNITS..=MAX_HEIGHT_UNITS).contains(h))
            || streams.iter().any(|s| {
                let value = f32::from_bits(*s);
                !value.is_finite() || value < 0.0 || *s == 0x80000000
            })
        {
            return Err("Invalid landform height or stream data".into());
        }
        let mut boundary = BTreeMap::new();
        let side = n as usize + 1;
        for face in 0..6 {
            for v in 0..=n {
                for u in 0..=n {
                    if u != 0 && v != 0 && u != n && v != n {
                        continue;
                    }
                    let i = face as usize * side * side + v as usize * side + u as usize;
                    let record = (heights[i], streams[i]);
                    if let Some(previous) = boundary.insert(face_key(n, face, u, v), record) {
                        if previous != record {
                            return Err("Inconsistent shared landform face boundary".into());
                        }
                    }
                }
            }
        }
        Ok(Self {
            resolution: n,
            recipe,
            heights,
            streams,
            id: expected,
        })
    }

    pub fn load(path: &Path, expected: SnapshotId) -> Result<Self, String> {
        let mut bytes = Vec::new();
        fs::File::open(path)
            .map_err(|e| e.to_string())?
            .take((MAX_BYTES + 1) as u64)
            .read_to_end(&mut bytes)
            .map_err(|e| e.to_string())?;
        Self::decode(&bytes, expected)
    }

    /// Publish via a flushed temporary file and an atomic, non-replacing hard
    /// link on the same filesystem. Unsupported filesystems return an error.
    /// This guarantees visibility of complete bytes, not power-loss durability
    /// of the directory entry. Existing identifiers are verified, never replaced.
    pub fn publish(&self, directory: &Path) -> Result<PathBuf, String> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        fs::create_dir_all(directory).map_err(|e| e.to_string())?;
        let destination = directory.join(format!("{}.hlfm", self.id));
        let bytes = self.encode();
        let (temporary, mut file) = loop {
            let nonce = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = directory.join(format!(".{}.{}-{nonce}.tmp", self.id, std::process::id()));
            match fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(file) => break (path, file),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e.to_string()),
            }
        };
        let outcome = (|| {
            file.write_all(&bytes).map_err(|e| e.to_string())?;
            file.sync_all().map_err(|e| e.to_string())?;
            match fs::hard_link(&temporary, &destination) {
                Ok(()) => Ok(destination.clone()),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    let existing = Self::load(&destination, self.id)?;
                    if existing.encode() != bytes {
                        return Err("Existing landform identity has different bytes".into());
                    }
                    Ok(destination.clone())
                }
                Err(e) => Err(e.to_string()),
            }
        })();
        drop(file);
        // This path was exclusively created by this invocation.
        let cleanup = fs::remove_file(&temporary).map_err(|e| e.to_string());
        match outcome {
            Ok(path) => {
                cleanup?;
                Ok(path)
            }
            Err(e) => Err(e),
        }
    }
}

#[cfg(test)]
mod tests;
