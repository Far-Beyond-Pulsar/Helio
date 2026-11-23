//! The radiance cache: where a surface is lit for indirect light, once, for
//! every ray that needs a lit hit (`docs/architecture/rendering.md`, section
//! 5, "The radiance cache").
//!
//! A world-space hash of **irradiance arriving at surface cells**, the light
//! from the lights (`direct`) and from everything else (`indirect`: other
//! surfaces, the sky) apart. A reader with a hit (a GI probe's ray, a pixel's
//! ray, a reflection's) multiplies by its own hit's albedo, so the cache holds
//! no texture detail and a cell may be coarse; one that lights its hit itself
//! with light samples (a mirror's hit) takes the indirect part alone.
//!
//! - **The key** of a hit: the world cell holding it (64-bit cell counts an
//!   axis, so a planet's far side is a cell like any other), the cell's
//!   level, the dominant side of the hit's normal (the face turned toward
//!   the ray: a thin wall's two sides are two cells) and the ray's coordinate
//!   space (two spaces at one place are two cells). Cells are powers of two
//!   in metres, [`LEVELS`] of them from 1/64 m to 1024 m: the frame's origin
//!   is a multiple of 1024 m, so it is a whole number of cells at every
//!   level, a frame point's cell is the origin's plus the point's, and an
//!   origin change moves no entry.
//! - **The level** of a read: the cell at least as wide as the ray's
//!   footprint at the hit and as the publisher's eye angle at the hit's
//!   distance from the eye, whichever is coarser (a probe's wide ray reads
//!   a coarse cell, a pixel's ray near the eye a fine one). A cell without a
//!   value answers from the next coarser one that has one, as many levels up
//!   as the reader allows (its `reach`: a coarser cell spans more, and
//!   across a thin occluder holds both sides' light; a probe that keeps
//!   what it reads in a history allows one, a pixel's ray that a denoiser
//!   filters may allow all).
//! - **The mark**: a reader marks the cell it read as used this frame (and
//!   the one [`ANCESTOR`] levels coarser, so a new fine cell has something to
//!   answer from); a cell first used takes its point and normal from the
//!   hit that marked it. What the publisher lights, how often and what it
//!   evicts is its own.
//!
//! The WGSL ([`SNIPPET`], `//!use frame_radiance_cache`) holds the rows, the
//! read and the mark; the Rust here mirrors the key's arithmetic for tests
//! and hosts. A publisher writes [`RadianceCacheFrame`] each frame and
//! publishes its buffers under [`FRAME`], [`SLOTS`] and [`ENTRIES`]; a reader
//! binds them, or [`stand_in`] where nothing publishes, and then finds
//! nothing.
//!
//! Why a hash of hit points (against cards a mesh, probes, a network): it
//! needs nothing from the kind that was hit (voxels at 10 cm, foliage, any
//! tracer), it holds a wall's two sides apart, its memory follows the
//! surfaces rays reached lately and not the world, and its cells shrink
//! toward the eye.

use bytemuck::{Pod, Zeroable};
use helio_core::shader::ShaderSnippet;

/// Marker opting a shader into [`SNIPPET`] (with `frame_ray_stream`, whose
/// packings it uses).
pub const MARKER: &str = "//!use frame_radiance_cache";
/// The cache's WGSL (`shaders/radiance_cache.wgsl`): the rows, the key's
/// arithmetic, `radiance_cache_read`, `radiance_cache_mark`, and what a
/// publisher decodes (`radiance_cache_entry_point`, `radiance_cache_size`).
pub const SNIPPET: ShaderSnippet =
    helio_core::wgsl_snippet!(MARKER, "../shaders/radiance_cache.wgsl");

/// Graph key of the frame's [`RadianceCacheFrame`] (a uniform).
pub const FRAME: &str = "radiance_cache_frame";
/// Graph key of the slots ([`RadianceCacheSlot`] rows, bound read-write:
/// a read loads them atomically, a mark writes them).
pub const SLOTS: &str = "radiance_cache_slots";
/// Graph key of the entries ([`RadianceCacheEntry`] rows, one a slot, bound
/// read-write: a mark writes a cell it claims).
pub const ENTRIES: &str = "radiance_cache_entries";

/// Levels of cells: level `l` is `2^(l + FINEST)` m wide.
pub const LEVELS: u32 = 17;
/// The finest level's width, `2^FINEST` m (1/64 m: a sixth of the finest
/// voxel the engine draws, under what any reader's indirect light resolves).
pub const FINEST: i32 = -6;
/// The coarsest level's width (m): the frame origin's step.
pub const COARSEST: f64 = 1024.0;
const _: () = assert!(LEVELS as i32 + FINEST - 1 == 10);
/// Slots a key may sit in.
pub const BUCKET: u32 = 16;
/// Levels a mark marks above the read's.
pub const ANCESTOR: u32 = 3;
/// The new cells' queue holds a place for every this many slots
/// (`RADIANCE_CACHE_QUEUE_SHARE`): it follows the hash's slots in
/// [`SLOTS`]'s buffer, its count first.
pub const QUEUE_SHARE: u32 = 4;
/// The directions along a cell's surface its clear distances are kept in
/// (`RADIANCE_CACHE_WAYS`).
pub const WAYS: u32 = 8;
/// [`RadianceCacheEntry::point`] of a cell its publisher unseated
/// (`RADIANCE_CACHE_UNSEATED`): its rays found its point inside a solid (the
/// surface it was taken on is gone), so it holds no value and the next mark
/// seats it at that mark's hit.
pub const UNSEATED: u32 = u32::MAX;

/// Slot rows [`SLOTS`]'s buffer holds for a cache of `slots` slots: the
/// hash's, the new cells' queue's count and its places.
pub fn slot_rows(slots: u32) -> u64 {
    u64::from(slots) + 1 + u64::from(slots / QUEUE_SHARE)
}

/// The width (m) of level `level`'s cells.
pub fn cell_size(level: u32) -> f32 {
    ((level as i32 + FINEST) as f32).exp2()
}

/// `RadianceCacheFrame` (see the WGSL), 64 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct RadianceCacheFrame {
    /// The origin's whole 1024 m, x and y (lo, hi each).
    pub origin_xy: [u32; 4],
    /// z (lo, hi), the slots (0: no cache), the frame's number.
    pub origin_z: [u32; 4],
    /// The origin's rest below 1024 m; w the eye angle (radians).
    pub offset: [f32; 4],
    /// The eye in the frame's space.
    pub eye: [f32; 4],
}

impl RadianceCacheFrame {
    pub const BYTES: u64 = std::mem::size_of::<Self>() as u64;

    /// The frame whose origin is world point `origin` (f64), its eye at
    /// frame point `eye`, a cache of `slots` slots (a power of two, at
    /// least [`BUCKET`]; 0: none) whose cells are at least `eye_angle` wide
    /// seen from the eye, on frame `frame`.
    pub fn new(origin: [f64; 3], eye: [f32; 3], eye_angle: f32, slots: u32, frame: u32) -> Self {
        debug_assert!(slots == 0 || (slots.is_power_of_two() && slots >= BUCKET));
        let whole = origin.map(|c| (c / COARSEST).floor());
        let rest: [f32; 3] = std::array::from_fn(|k| (origin[k] - whole[k] * COARSEST) as f32);
        let w = whole.map(|c| c as i64 as u64);
        let (lo, hi) = (|v: u64| v as u32, |v: u64| (v >> 32) as u32);
        Self {
            origin_xy: [lo(w[0]), hi(w[0]), lo(w[1]), hi(w[1])],
            origin_z: [lo(w[2]), hi(w[2]), slots, frame],
            offset: [rest[0], rest[1], rest[2], eye_angle],
            eye: [eye[0], eye[1], eye[2], 0.0],
        }
    }

    pub fn slots(&self) -> u32 {
        self.origin_z[2]
    }

    /// `radiance_cache_level`.
    pub fn level(&self, p: [f32; 3], footprint: f32) -> u32 {
        let d = (0..3)
            .map(|k| (p[k] - self.eye[k]).powi(2))
            .sum::<f32>()
            .sqrt();
        let size = footprint.max(self.offset[3] * d);
        let l = size.max(1e-30).log2().round() - FINEST as f32;
        l.clamp(0.0, (LEVELS - 1) as f32) as u32
    }

    /// `radiance_cache_origin_cell`: the origin's cell at `level`, an axis.
    pub fn origin_cell(&self, level: u32) -> [i64; 3] {
        let whole = |lo: u32, hi: u32| ((u64::from(hi) << 32) | u64::from(lo)) as i64;
        let s = LEVELS - 1 - level;
        [
            whole(self.origin_xy[0], self.origin_xy[1]),
            whole(self.origin_xy[2], self.origin_xy[3]),
            whole(self.origin_z[0], self.origin_z[1]),
        ]
        .map(|c| c.wrapping_shl(s))
    }

    /// `radiance_cache_key`: the cell of frame point `p` at `level`, on the
    /// side of unit `n`, in coordinate space `space`.
    pub fn key(&self, p: [f32; 3], n: [f32; 3], level: u32, space: u32) -> RadianceCacheKey {
        let inv = (-((level as i32 + FINEST) as f32)).exp2();
        let o = self.origin_cell(level);
        let c: [i64; 3] = std::array::from_fn(|k| {
            o[k].wrapping_add(((p[k] + self.offset[k]) * inv).floor() as i32 as i64)
        });
        RadianceCacheKey::new(c, level, side(n), space)
    }
}

/// `radiance_cache_side`: the dominant side of unit `n`.
pub fn side(n: [f32; 3]) -> u32 {
    let a = n.map(f32::abs);
    if a[0] >= a[1] && a[0] >= a[2] {
        u32::from(n[0] < 0.0)
    } else if a[1] >= a[2] {
        2 + u32::from(n[1] < 0.0)
    } else {
        4 + u32::from(n[2] < 0.0)
    }
}

/// `RadianceCacheSlot` (see the WGSL), 8 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct RadianceCacheSlot {
    /// The key's checksum; 0 empty.
    pub state: u32,
    /// The frame a reader last marked it.
    pub used: u32,
}

impl RadianceCacheSlot {
    pub const BYTES: u64 = std::mem::size_of::<Self>() as u64;
}

/// `RadianceCacheKey` (see the WGSL), 28 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Pod, Zeroable)]
pub struct RadianceCacheKey {
    /// Level (bits 0..5), side (5..8), space (8..17).
    pub tag: u32,
    pub x_lo: u32,
    pub x_hi: u32,
    pub y_lo: u32,
    pub y_hi: u32,
    pub z_lo: u32,
    pub z_hi: u32,
}

fn mix(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

impl RadianceCacheKey {
    /// The key of world cell `cell` (in level `level`'s cells), side `side`,
    /// coordinate space `space`.
    pub fn new(cell: [i64; 3], level: u32, side: u32, space: u32) -> Self {
        let (lo, hi) = (|v: i64| v as u64 as u32, |v: i64| ((v as u64) >> 32) as u32);
        Self {
            tag: level | (side << 5) | ((space & 0x1ff) << 8),
            x_lo: lo(cell[0]),
            x_hi: hi(cell[0]),
            y_lo: lo(cell[1]),
            y_hi: hi(cell[1]),
            z_lo: lo(cell[2]),
            z_hi: hi(cell[2]),
        }
    }

    pub fn level(&self) -> u32 {
        self.tag & 31
    }

    pub fn side(&self) -> u32 {
        (self.tag >> 5) & 7
    }

    pub fn space(&self) -> u32 {
        (self.tag >> 8) & 0x1ff
    }

    /// The world cell, an axis.
    pub fn cell(&self) -> [i64; 3] {
        let whole = |lo: u32, hi: u32| ((u64::from(hi) << 32) | u64::from(lo)) as i64;
        [
            whole(self.x_lo, self.x_hi),
            whole(self.y_lo, self.y_hi),
            whole(self.z_lo, self.z_hi),
        ]
    }

    fn hash(&self, seed: u32) -> u32 {
        let mut h = mix(seed ^ self.tag);
        for w in [
            self.x_lo, self.x_hi, self.y_lo, self.y_hi, self.z_lo, self.z_hi,
        ] {
            h = mix(h ^ w);
        }
        h
    }

    /// `radiance_cache_bucket`: the first slot of the key's bucket in a
    /// cache of `slots` slots.
    pub fn bucket(&self, slots: u32) -> u32 {
        (self.hash(0x9e37_79b9) & (slots / BUCKET - 1)) * BUCKET
    }

    /// `radiance_cache_checksum`: a slot's state for the key (never 0).
    pub fn checksum(&self) -> u32 {
        self.hash(0x85eb_ca6b).max(1)
    }
}

/// `RadianceCacheEntry` (see the WGSL), 96 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct RadianceCacheEntry {
    pub key: RadianceCacheKey,
    /// Its point in the cell, 15 bits an axis: here the top 10 (10:10:10),
    /// the low 5 in [`Self::history`] from bit 16 (5:5:5); [`UNSEATED`] while
    /// it has none.
    pub point: u32,
    /// Its side's normal (`pack_unit`).
    pub normal: u32,
    /// Updates folded: direct (bits 0..8), indirect (8..16); bits 16..31 the
    /// point's low bits.
    pub history: u32,
    /// Irradiance from the lights (W / m^2).
    pub direct: [f32; 3],
    /// Irradiance from what the cell's rays meet as they meet it: what
    /// surfaces emit and reflect of the lights' light, the sky.
    pub indirect: [f32; 3],
    /// Irradiance from the light those surfaces reflect of the cache's own
    /// (the later bounces); a read's indirect light is `indirect + bounced`.
    pub bounced: [f32; 3],
    /// Updates folded into `bounced` (bits 0..8).
    pub bounced_history: u32,
    /// How far from the cell's point, along its surface, its rays found
    /// nothing in the way, in [`WAYS`] directions (f16 pairs, metres;
    /// `radiance_cache_holds`: a reader past that reads a finer cell).
    pub clear: [u32; 4],
}

impl RadianceCacheEntry {
    pub const BYTES: u64 = std::mem::size_of::<Self>() as u64;

    /// `radiance_cache_has_value`.
    pub fn has_value(&self) -> bool {
        self.history & 0xffff != 0
    }

    /// The point's place in its cell, each axis in [0, 1).
    pub fn point_in_cell(&self) -> [f32; 3] {
        let fine = self.history >> 16;
        [0, 1, 2].map(|k| {
            let high = (self.point >> (10 * k)) & 1023;
            let low = (fine >> (5 * k)) & 31;
            (((high << 5) | low) as f32 + 0.5) / 32768.0
        })
    }
}

/// A cache that holds nothing: a frame without slots, one slot and one
/// entry (the bindings' minimum sizes), for a reader in a graph where
/// nothing publishes the cache.
pub struct StandIn {
    pub frame: wgpu::Buffer,
    pub slots: wgpu::Buffer,
    pub entries: wgpu::Buffer,
}

/// The stand-in's buffers ([`StandIn`]); zeroed at creation, so a read finds
/// nothing and a mark does nothing.
pub fn stand_in(device: &wgpu::Device) -> StandIn {
    let buffer = |label: &str, size: u64, usage: wgpu::BufferUsages| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    };
    StandIn {
        frame: buffer(
            "radiance cache stand-in frame",
            RadianceCacheFrame::BYTES,
            wgpu::BufferUsages::UNIFORM,
        ),
        slots: buffer(
            "radiance cache stand-in slots",
            RadianceCacheSlot::BYTES,
            wgpu::BufferUsages::STORAGE,
        ),
        entries: buffer(
            "radiance cache stand-in entries",
            RadianceCacheEntry::BYTES,
            wgpu::BufferUsages::STORAGE,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The WGSL's copies of the contract's constants.
    #[test]
    fn the_snippet_s_constants_are_the_contract_s() {
        for (name, value) in [
            ("RADIANCE_CACHE_LEVELS", LEVELS),
            ("RADIANCE_CACHE_BUCKET", BUCKET),
            ("RADIANCE_CACHE_ANCESTOR", ANCESTOR),
            ("RADIANCE_CACHE_QUEUE_SHARE", QUEUE_SHARE),
            ("RADIANCE_CACHE_WAYS", WAYS),
        ] {
            let line = format!("const {name}: u32 = {value}u;");
            assert!(SNIPPET.source.contains(&line), "{line}");
        }
        let line = format!("const RADIANCE_CACHE_UNSEATED: u32 = {UNSEATED:#x}u;");
        assert!(SNIPPET.source.contains(&line), "{line}");
    }

    #[test]
    fn rows_have_their_contract_sizes() {
        assert_eq!(RadianceCacheFrame::BYTES, 64);
        assert_eq!(RadianceCacheSlot::BYTES, 8);
        assert_eq!(std::mem::size_of::<RadianceCacheKey>(), 28);
        assert_eq!(RadianceCacheEntry::BYTES, 96);
    }

    #[test]
    fn levels_run_from_a_64th_of_a_metre_to_the_origin_step() {
        assert_eq!(cell_size(0), 1.0 / 64.0);
        assert_eq!(f64::from(cell_size(LEVELS - 1)), COARSEST);
    }

    /// A world point keys the same cell whatever frame origin (a multiple
    /// of 1024 m) it is seen from, at every level, near the world's zero, at
    /// an Earth's radius and past 2^32 cells (the high words carry).
    #[test]
    fn an_origin_change_moves_no_cell() {
        let n = [0.0, 1.0, 0.0];
        for world in [
            [3.3f64, -0.7, 12.9],
            [6_371_000.4, 1.2, -6_371_000.9],
            [-1.0e12 + 0.3, 7.0, 4.0e11 + 0.6],
        ] {
            for level in 0..LEVELS {
                let keys: Vec<_> = [0.0, 1024.0, -2048.0, 3.0 * 1024.0]
                    .iter()
                    .map(|&step| {
                        let origin = [
                            (world[0] / 1024.0).floor() * 1024.0 + step,
                            (world[1] / 1024.0).floor() * 1024.0,
                            (world[2] / 1024.0).floor() * 1024.0 - step,
                        ];
                        let frame = RadianceCacheFrame::new(origin, [0.0; 3], 0.0, 1 << 16, 1);
                        let p: [f32; 3] = std::array::from_fn(|k| (world[k] - origin[k]) as f32);
                        frame.key(p, n, level, 1)
                    })
                    .collect();
                assert!(
                    keys.windows(2).all(|w| w[0] == w[1]),
                    "{world:?} {level}: {keys:?}"
                );
                let size = f64::from(cell_size(level));
                let want = world.map(|c| (c / size).floor() as i64);
                assert_eq!(keys[0].cell(), want, "{world:?} at {level}");
                assert_eq!(keys[0].level(), level);
            }
        }
    }

    /// An origin off the 1024 m step (a test's or a host's own) keys the
    /// same world cells, its rest carried in `offset`.
    #[test]
    fn any_origin_keys_world_cells() {
        let world = [1000.3f64, 2.7, -500.2];
        let a = RadianceCacheFrame::new([1000.0, 2.0, -500.0], [0.0; 3], 0.0, 1 << 16, 1);
        let b = RadianceCacheFrame::new([0.0, 0.0, 0.0], [0.0; 3], 0.0, 1 << 16, 1);
        for level in [0, 4, 6, 10, 16] {
            let at = |f: &RadianceCacheFrame, o: [f64; 3]| {
                f.key(
                    std::array::from_fn(|k| (world[k] - o[k]) as f32),
                    [1.0, 0.0, 0.0],
                    level,
                    0,
                )
            };
            assert_eq!(at(&a, [1000.0, 2.0, -500.0]), at(&b, [0.0; 3]), "{level}");
        }
    }

    /// The level: at least the footprint and the eye angle at the distance,
    /// the nearest power of two, within the levels.
    #[test]
    fn a_read_keys_the_coarser_of_its_footprint_and_the_eye_angle() {
        let frame = RadianceCacheFrame::new([0.0; 3], [0.0; 3], 1.0 / 32.0, 1 << 16, 1);
        // Near the eye a 1 cm footprint: 1/64 m.
        assert_eq!(frame.level([0.1, 0.0, 0.0], 0.01), 0);
        // 32 m away, 1 m by the angle; a 4 m footprint wins.
        assert_eq!(frame.level([32.0, 0.0, 0.0], 0.01), 6);
        assert_eq!(frame.level([32.0, 0.0, 0.0], 4.0), 8);
        // Past the coarsest: the coarsest.
        assert_eq!(frame.level([1e7, 0.0, 0.0], 0.0), LEVELS - 1);
    }

    #[test]
    fn sides_spaces_and_levels_are_apart() {
        let frame = RadianceCacheFrame::new([0.0; 3], [0.0; 3], 0.0, 1 << 16, 1);
        let p = [0.5, 0.5, 0.5];
        let up = frame.key(p, [0.0, 1.0, 0.0], 4, 1);
        let down = frame.key(p, [0.0, -1.0, 0.0], 4, 1);
        let other_space = frame.key(p, [0.0, 1.0, 0.0], 4, 2);
        let coarser = frame.key(p, [0.0, 1.0, 0.0], 5, 1);
        for other in [down, other_space, coarser] {
            assert_ne!(up, other);
            assert_ne!(up.checksum(), other.checksum());
        }
        assert_eq!((up.side(), down.side()), (2, 3));
        assert_eq!(other_space.space(), 2);
        assert!(up.bucket(1 << 16).is_multiple_of(BUCKET) && up.bucket(1 << 16) < 1 << 16);
    }

    /// Buckets spread keys: a grid of neighbouring cells fills the buckets
    /// of a cache twice its size about evenly (a full bucket would turn a
    /// cell away).
    #[test]
    fn neighbouring_cells_spread_over_the_buckets() {
        let slots = 1u32 << 14;
        let mut counts = vec![0u32; (slots / BUCKET) as usize];
        let mut n = 0;
        for x in 0..32i64 {
            for z in 0..32i64 {
                for y in 0..8i64 {
                    let key = RadianceCacheKey::new([x, y, z], 3, 2, 1);
                    counts[(key.bucket(slots) / BUCKET) as usize] += 1;
                    n += 1;
                }
            }
        }
        // 8192 keys in 1024 buckets: 8 a bucket on average (half full);
        // Poisson's tail past 16 is under a percent of buckets.
        let full = counts.iter().filter(|&&c| c > BUCKET).count();
        assert!(
            full * 100 < counts.len(),
            "{full} full buckets of {}",
            counts.len()
        );
        assert_eq!(counts.iter().sum::<u32>(), n);
    }
}
