//! Binary edit journal: the persistent, append-only record of a planet's
//! destruction and construction.
//!
//! The terrain itself is procedural, so a planet is fully described by its
//! [`PlanetRecipe`] plus the ordered brushes applied to it. The journal
//! stores exactly that, little-endian:
//!
//! * a 16-byte header: magic `HVPJ`, format version, and a fingerprint of
//!   the recipe (a journal never replays onto a different planet);
//! * fixed 48-byte records, each a brush or an undo, carrying its sequence
//!   number and a checksum.
//!
//! Records are only ever appended, so a writer can flush each edit as it
//! happens (autosave, network replication). A crash mid-append leaves a
//! partial last record, which reading drops; a corrupted or reordered
//! record is an error rather than a silently different planet.
use crate::edits::{Brush, BrushOp, BrushShape};
use crate::planet::{Planet, PlanetRecipe};

const MAGIC: [u8; 4] = *b"HVPJ";
const VERSION: u16 = 1;
pub const HEADER_BYTES: usize = 16;
pub const RECORD_BYTES: usize = 48;

const KIND_BRUSH: u8 = 1;
const KIND_UNDO: u8 = 2;

/// One journal record.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Entry {
    Brush(Brush),
    /// Undo the most recent remaining brush.
    Undo,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JournalError {
    /// Not a journal, or shorter than its header.
    NotAJournal,
    UnsupportedVersion(u16),
    /// Written for a different planet recipe.
    RecipeMismatch { journal: u64, planet: u64 },
    /// Record `index` failed its checksum, sequence or field validation.
    Corrupt { index: usize, reason: &'static str },
    /// Record `index` could not be applied to the planet.
    Rejected { index: usize, reason: String },
}

impl std::fmt::Display for JournalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotAJournal => write!(f, "not a planet edit journal"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported journal version {v}"),
            Self::RecipeMismatch { journal, planet } => {
                write!(f, "journal recipe {journal:016x} does not match planet recipe {planet:016x}")
            }
            Self::Corrupt { index, reason } => write!(f, "journal record {index} is corrupt: {reason}"),
            Self::Rejected { index, reason } => write!(f, "journal record {index} was rejected: {reason}"),
        }
    }
}

impl std::error::Error for JournalError {}

/// Stable fingerprint of a recipe (FNV-1a over its canonical JSON).
pub fn recipe_fingerprint(recipe: &PlanetRecipe) -> u64 {
    fnv64(recipe.to_json().as_bytes())
}

fn fnv64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ u64::from(*b)).wrapping_mul(0x0000_0100_0000_01b3))
}

fn fnv32(bytes: &[u8]) -> u32 {
    bytes.iter().fold(0x811c_9dc5u32, |h, b| (h ^ u32::from(*b)).wrapping_mul(0x0100_0193))
}

/// Journal header for a planet recipe.
pub fn header(recipe: &PlanetRecipe) -> [u8; HEADER_BYTES] {
    let mut out = [0u8; HEADER_BYTES];
    out[..4].copy_from_slice(&MAGIC);
    out[4..6].copy_from_slice(&VERSION.to_le_bytes());
    out[8..16].copy_from_slice(&recipe_fingerprint(recipe).to_le_bytes());
    out
}

/// Encode record `seq` (0-based position in the journal).
pub fn encode(seq: u32, entry: &Entry) -> [u8; RECORD_BYTES] {
    let mut out = [0u8; RECORD_BYTES];
    match entry {
        Entry::Brush(b) => {
            out[0] = KIND_BRUSH;
            out[1] = match b.shape {
                BrushShape::Sphere => 0,
                BrushShape::Cube => 1,
            };
            out[2] = match b.op {
                BrushOp::Remove => 0,
                BrushOp::Add => 1,
                BrushOp::Paint => 2,
            };
            out[4..8].copy_from_slice(&b.material.to_le_bytes());
            for (axis, v) in b.center.iter().enumerate() {
                out[8 + axis * 8..16 + axis * 8].copy_from_slice(&v.to_le_bytes());
            }
            out[32..40].copy_from_slice(&b.radius.to_le_bytes());
        }
        Entry::Undo => out[0] = KIND_UNDO,
    }
    out[40..44].copy_from_slice(&seq.to_le_bytes());
    let sum = fnv32(&out[..44]);
    out[44..48].copy_from_slice(&sum.to_le_bytes());
    out
}

fn decode(index: usize, r: &[u8]) -> Result<Entry, JournalError> {
    let corrupt = |reason| JournalError::Corrupt { index, reason };
    let u32_at = |at: usize| u32::from_le_bytes(r[at..at + 4].try_into().unwrap());
    let f64_at = |at: usize| f64::from_le_bytes(r[at..at + 8].try_into().unwrap());
    if u32_at(44) != fnv32(&r[..44]) {
        return Err(corrupt("checksum"));
    }
    if u32_at(40) as usize != index {
        return Err(corrupt("sequence"));
    }
    match r[0] {
        KIND_UNDO => Ok(Entry::Undo),
        KIND_BRUSH => {
            let shape = match r[1] {
                0 => BrushShape::Sphere,
                1 => BrushShape::Cube,
                _ => return Err(corrupt("brush shape")),
            };
            let op = match r[2] {
                0 => BrushOp::Remove,
                1 => BrushOp::Add,
                2 => BrushOp::Paint,
                _ => return Err(corrupt("brush operation")),
            };
            let center = [f64_at(8), f64_at(16), f64_at(24)];
            let radius = f64_at(32);
            if !center.iter().all(|v| v.is_finite()) || !(radius.is_finite() && radius > 0.0) {
                return Err(corrupt("brush geometry"));
            }
            Ok(Entry::Brush(Brush { center, radius, shape, op, material: u32_at(4) }))
        }
        _ => Err(corrupt("record kind")),
    }
}

/// Entries read from a journal.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Contents {
    pub entries: Vec<Entry>,
    /// Bytes of an incomplete last record (an interrupted append), dropped.
    pub torn_bytes: usize,
}

/// Read a journal written for `recipe`.
pub fn read(bytes: &[u8], recipe: &PlanetRecipe) -> Result<Contents, JournalError> {
    if bytes.len() < HEADER_BYTES || bytes[..4] != MAGIC {
        return Err(JournalError::NotAJournal);
    }
    let version = u16::from_le_bytes([bytes[4], bytes[5]]);
    if version != VERSION {
        return Err(JournalError::UnsupportedVersion(version));
    }
    let journal = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
    let planet = recipe_fingerprint(recipe);
    if journal != planet {
        return Err(JournalError::RecipeMismatch { journal, planet });
    }
    let records = bytes[HEADER_BYTES..].chunks_exact(RECORD_BYTES);
    let torn_bytes = records.remainder().len();
    let entries = records.enumerate().map(|(index, r)| decode(index, r)).collect::<Result<_, _>>()?;
    Ok(Contents { entries, torn_bytes })
}

/// Appends records for one planet; `bytes` is always a valid journal.
#[derive(Clone, Debug)]
pub struct Writer {
    bytes: Vec<u8>,
    next: u32,
}

impl Writer {
    pub fn new(recipe: &PlanetRecipe) -> Self {
        Self { bytes: header(recipe).to_vec(), next: 0 }
    }
    /// Continue a journal already read with [`read`] (its torn tail dropped).
    pub fn resume(bytes: &[u8], contents: &Contents) -> Self {
        let len = bytes.len() - contents.torn_bytes;
        Self { bytes: bytes[..len].to_vec(), next: contents.entries.len() as u32 }
    }
    /// Append a record and return its bytes (for flushing to a file or peer).
    pub fn push(&mut self, entry: &Entry) -> [u8; RECORD_BYTES] {
        let record = encode(self.next, entry);
        self.next += 1;
        self.bytes.extend_from_slice(&record);
        record
    }
    pub fn len(&self) -> usize {
        self.next as usize
    }
    pub fn is_empty(&self) -> bool {
        self.next == 0
    }
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

impl Planet {
    /// Compact journal of the current edit state: one record per remaining
    /// brush, in order (undone brushes are gone).
    pub fn journal(&self) -> Vec<u8> {
        let mut writer = Writer::new(self.recipe());
        for brush in self.edits().brushes() {
            writer.push(&Entry::Brush(*brush));
        }
        writer.into_bytes()
    }

    /// Apply journal entries in order. On a rejected entry the planet keeps
    /// the entries before it.
    pub fn replay(&mut self, entries: &[Entry]) -> Result<(), JournalError> {
        for (index, entry) in entries.iter().enumerate() {
            match entry {
                Entry::Brush(brush) => {
                    self.apply(*brush).map_err(|reason| JournalError::Rejected { index, reason })?;
                }
                Entry::Undo => {
                    if self.undo().is_none() {
                        return Err(JournalError::Rejected { index, reason: "undo with no edits".into() });
                    }
                }
            }
        }
        Ok(())
    }

    /// A planet from its recipe and journal.
    pub fn from_journal(recipe: PlanetRecipe, bytes: &[u8]) -> Result<Self, JournalError> {
        // Build first: the planet names the concrete generator version (a
        // recipe may ask for the latest), which the journal was written with.
        let mut planet = Planet::new(recipe).map_err(|reason| JournalError::Rejected { index: 0, reason })?;
        let contents = read(bytes, planet.recipe())?;
        planet.replay(&contents.entries)?;
        Ok(planet)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;

    fn planet() -> Planet {
        Planet::new(PlanetRecipe::default()).unwrap()
    }

    fn brushes(p: &Planet) -> Vec<Brush> {
        let ground = p.surface_point(DVec3::new(0.3, 1.0, 0.2), 0.0);
        (0..12)
            .map(|n| Brush {
                center: (ground + DVec3::new(f64::from(n) * 0.7, 0.0, 0.0)).to_array(),
                radius: 0.4 + 0.1 * f64::from(n % 5),
                shape: if n % 3 == 0 { BrushShape::Cube } else { BrushShape::Sphere },
                op: [BrushOp::Remove, BrushOp::Add, BrushOp::Paint][n as usize % 3],
                material: n % 7,
            })
            .collect()
    }

    #[test]
    fn journal_round_trips_edits_and_undo() {
        let mut a = planet();
        let mut writer = Writer::new(a.recipe());
        for (n, brush) in brushes(&a).into_iter().enumerate() {
            a.apply(brush).unwrap();
            writer.push(&Entry::Brush(brush));
            if n % 4 == 3 {
                a.undo().unwrap();
                writer.push(&Entry::Undo);
            }
        }
        let b = Planet::from_journal(PlanetRecipe::default(), writer.bytes()).unwrap();
        assert_eq!(a.edits().brushes().collect::<Vec<_>>(), b.edits().brushes().collect::<Vec<_>>());
        // The compact journal holds the same remaining brushes.
        let compact = Planet::from_journal(PlanetRecipe::default(), &a.journal()).unwrap();
        assert_eq!(a.edits().brushes().collect::<Vec<_>>(), compact.edits().brushes().collect::<Vec<_>>());
        assert_eq!(a.journal().len(), HEADER_BYTES + RECORD_BYTES * a.edits().len());
    }

    #[test]
    fn torn_tail_is_dropped_and_writing_resumes() {
        let p = planet();
        let mut writer = Writer::new(p.recipe());
        let list = brushes(&p);
        for brush in &list[..5] {
            writer.push(&Entry::Brush(*brush));
        }
        let mut bytes = writer.into_bytes();
        bytes.extend_from_slice(&encode(5, &Entry::Brush(list[5]))[..20]);
        let contents = read(&bytes, p.recipe()).unwrap();
        assert_eq!(contents.entries.len(), 5);
        assert_eq!(contents.torn_bytes, 20);
        let mut resumed = Writer::resume(&bytes, &contents);
        resumed.push(&Entry::Brush(list[5]));
        let again = read(resumed.bytes(), p.recipe()).unwrap();
        assert_eq!(again.entries.len(), 6);
        assert_eq!(again.torn_bytes, 0);
    }

    #[test]
    fn corruption_reordering_and_foreign_recipes_are_rejected() {
        let p = planet();
        let mut writer = Writer::new(p.recipe());
        for brush in brushes(&p) {
            writer.push(&Entry::Brush(brush));
        }
        let bytes = writer.into_bytes();

        let mut flipped = bytes.clone();
        flipped[HEADER_BYTES + RECORD_BYTES * 3 + 10] ^= 0x40;
        assert_eq!(read(&flipped, p.recipe()), Err(JournalError::Corrupt { index: 3, reason: "checksum" }));

        let mut swapped = bytes.clone();
        let (a, b) = (HEADER_BYTES + RECORD_BYTES, HEADER_BYTES + RECORD_BYTES * 2);
        let first = swapped[a..a + RECORD_BYTES].to_vec();
        swapped.copy_within(b..b + RECORD_BYTES, a);
        swapped[b..b + RECORD_BYTES].copy_from_slice(&first);
        assert_eq!(read(&swapped, p.recipe()), Err(JournalError::Corrupt { index: 1, reason: "sequence" }));

        let other = PlanetRecipe { voxel_size_m: 0.3, ..PlanetRecipe::default() };
        assert!(matches!(read(&bytes, &other), Err(JournalError::RecipeMismatch { .. })));
        assert_eq!(read(b"not a journal at all", p.recipe()), Err(JournalError::NotAJournal));
    }

    #[test]
    fn undo_past_the_start_is_rejected() {
        let mut p = planet();
        assert!(matches!(p.replay(&[Entry::Undo]), Err(JournalError::Rejected { index: 0, .. })));
    }
}
