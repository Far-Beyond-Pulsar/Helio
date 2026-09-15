//! Shared-memory ring-buffer framing.
//!
//! Hand-copied (not crate-shared) between this in-process agent and the
//! standalone `scenedb_inspector` egui client -- see the crate-level docs
//! for why. Keep both copies in sync if this changes; bump [`VERSION`] on
//! any breaking layout change so a stale client fails loudly (a version
//! mismatch) instead of misreading garbage.
//!
//! Single-writer / single-reader (SPSC). Each slot is guarded by its own
//! seqlock (Linux-kernel style): `seq` is odd while a write is in progress
//! and even once stable; a reader that observes the same even `seq` before
//! and after copying the payload out knows the copy was torn-free, and
//! retries otherwise. `latest_slot` points readers at the most recently
//! published slot so they don't have to poll every slot every time.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

pub const MAGIC: u32 = 0x5343_4442; // "SCDB", read as a plain sanity marker
pub const VERSION: u32 = 2;
pub const DEFAULT_SLOT_COUNT: u32 = 3;
pub const DEFAULT_SLOT_CAPACITY: u32 = 8 * 1024 * 1024; // 8 MiB

// magic, version, slot_count, slot_capacity, request_latest,
// response_latest: 6 x u32 = 24, + 8 bytes padding so slot 0 (and every slot after it, since
// SLOT_HEADER_LEN + slot_capacity is always a multiple of 8) starts 8-byte
// aligned for its `seq: AtomicU64`.
const HEADER_LEN: usize = 32;
// seq: u64 (8) + len: u32 (4) + 4 bytes padding.
const SLOT_HEADER_LEN: usize = 16;

/// Total shared-memory region size needed for `slot_count` slots of
/// `slot_capacity` bytes each.
pub fn region_size(slot_count: u32, slot_capacity: u32) -> usize {
    HEADER_LEN + 2 * (slot_count as usize) * (SLOT_HEADER_LEN + slot_capacity as usize)
}

/// Thin view over a mapped shared-memory region implementing the framing
/// above. `base` must point at a region at least
/// `region_size(slot_count, slot_capacity)` bytes long, valid for the
/// lifetime of this handle (in practice: for as long as the backing
/// `Shmem` mapping stays alive).
pub struct RingView {
    base: *mut u8,
    slot_count: u32,
    slot_capacity: u32,
}

// SAFETY: every access to `base` goes through atomics at fixed, in-bounds
// offsets; the raw pointer itself is never read or written non-atomically
// after construction, so sharing/moving this handle across threads is sound
// exactly as it would be for the atomics it wraps.
unsafe impl Send for RingView {}
unsafe impl Sync for RingView {}

impl RingView {
    /// Initialize a freshly-mapped region's header (writer side, once).
    ///
    /// # Safety
    /// `base` must point at a valid region of at least
    /// `region_size(slot_count, slot_capacity)` bytes, exclusively owned by
    /// the caller for the duration of this call.
    pub unsafe fn init(base: *mut u8, slot_count: u32, slot_capacity: u32) -> Self {
        let view = Self {
            base,
            slot_count,
            slot_capacity,
        };
        view.magic_atomic().store(MAGIC, Ordering::Relaxed);
        view.version_atomic().store(VERSION, Ordering::Relaxed);
        view.slot_count_atomic().store(slot_count, Ordering::Relaxed);
        view.slot_capacity_atomic().store(slot_capacity, Ordering::Relaxed);
        for direction in [Direction::Request, Direction::Response] {
            for i in 0..slot_count {
                view.slot_seq_atomic(direction, i).store(0, Ordering::Relaxed);
                view.slot_len_atomic(direction, i).store(0, Ordering::Relaxed);
            }
        }
        // Published last, after every slot is zeroed, so a reader that
        // sees a non-sentinel `latest_slot` can trust that slot's header
        // atomics are already initialized.
        view.latest_slot_atomic(Direction::Request).store(u32::MAX, Ordering::Release);
        view.latest_slot_atomic(Direction::Response).store(u32::MAX, Ordering::Release);
        view
    }

    /// Attach to an already-initialized region (reader side). Validates
    /// magic/version and reads back the writer-chosen slot layout.
    ///
    /// # Safety
    /// `base` must point at a region a writer has already `init`'d (or is
    /// concurrently `init`ing -- magic/version are written first, so a
    /// racing attach either sees the old contents of a reused region, none
    /// of which this call trusts until magic/version check out, or the new
    /// ones), at least large enough for whatever layout the writer chose.
    pub unsafe fn attach(base: *mut u8) -> Result<Self, String> {
        let probe = Self {
            base,
            slot_count: 0,
            slot_capacity: 0,
        };
        let magic = probe.magic_atomic().load(Ordering::Relaxed);
        if magic != MAGIC {
            return Err(format!(
                "scenedb-inspector: bad shared-memory magic (expected {MAGIC:#x}, got {magic:#x}) -- \
                 is this segment actually a scenedb-inspector-agent region?"
            ));
        }
        let version = probe.version_atomic().load(Ordering::Relaxed);
        if version != VERSION {
            return Err(format!(
                "scenedb-inspector: protocol version mismatch (client expects {VERSION}, agent wrote {version}) -- \
                 rebuild the inspector client and/or the target app against matching ring.rs copies"
            ));
        }
        let slot_count = probe.slot_count_atomic().load(Ordering::Relaxed);
        let slot_capacity = probe.slot_capacity_atomic().load(Ordering::Relaxed);
        Ok(Self {
            base,
            slot_count,
            slot_capacity,
        })
    }

    pub fn slot_count(&self) -> u32 {
        self.slot_count
    }

    pub fn slot_capacity(&self) -> u32 {
        self.slot_capacity
    }

    // -- header atomics --

    unsafe fn atomic_u32_at(&self, offset: usize) -> &AtomicU32 {
        &*(self.base.add(offset) as *const AtomicU32)
    }
    unsafe fn atomic_u64_at(&self, offset: usize) -> &AtomicU64 {
        &*(self.base.add(offset) as *const AtomicU64)
    }

    fn magic_atomic(&self) -> &AtomicU32 {
        unsafe { self.atomic_u32_at(0) }
    }
    fn version_atomic(&self) -> &AtomicU32 {
        unsafe { self.atomic_u32_at(4) }
    }
    fn slot_count_atomic(&self) -> &AtomicU32 {
        unsafe { self.atomic_u32_at(8) }
    }
    fn slot_capacity_atomic(&self) -> &AtomicU32 {
        unsafe { self.atomic_u32_at(12) }
    }
    fn latest_slot_atomic(&self, direction: Direction) -> &AtomicU32 {
        unsafe { self.atomic_u32_at(match direction { Direction::Request => 16, Direction::Response => 20 }) }
    }

    fn slot_offset(&self, direction: Direction, slot: u32) -> usize {
        let direction_offset = match direction {
            Direction::Request => 0,
            Direction::Response => self.slot_count as usize,
        };
        HEADER_LEN + (direction_offset + slot as usize) * (SLOT_HEADER_LEN + self.slot_capacity as usize)
    }
    fn slot_seq_atomic(&self, direction: Direction, slot: u32) -> &AtomicU64 {
        unsafe { self.atomic_u64_at(self.slot_offset(direction, slot)) }
    }
    fn slot_len_atomic(&self, direction: Direction, slot: u32) -> &AtomicU32 {
        unsafe { self.atomic_u32_at(self.slot_offset(direction, slot) + 8) }
    }
    fn slot_data_ptr(&self, direction: Direction, slot: u32) -> *mut u8 {
        unsafe { self.base.add(self.slot_offset(direction, slot) + SLOT_HEADER_LEN) }
    }

    /// Writer: publish `payload` into `slot`, then point `latest_slot` at
    /// it. Returns `false` without writing anything if `payload` doesn't
    /// fit in a slot -- the caller decides how to handle/log that.
    fn publish_direction(&self, direction: Direction, slot: u32, payload: &[u8]) -> bool {
        if payload.len() as u32 > self.slot_capacity {
            return false;
        }
        let seq = self.slot_seq_atomic(direction, slot);
        seq.fetch_add(1, Ordering::AcqRel); // now odd: mid-write
        unsafe {
            std::ptr::copy_nonoverlapping(payload.as_ptr(), self.slot_data_ptr(direction, slot), payload.len());
        }
        self.slot_len_atomic(direction, slot)
            .store(payload.len() as u32, Ordering::Release);
        seq.fetch_add(1, Ordering::AcqRel); // now even: stable, with new content
        self.latest_slot_atomic(direction).store(slot, Ordering::Release);
        true
    }

    pub fn publish(&self, slot: u32, payload: &[u8]) -> bool {
        self.publish_direction(Direction::Response, slot, payload)
    }

    pub fn publish_request(&self, slot: u32, payload: &[u8]) -> bool {
        self.publish_direction(Direction::Request, slot, payload)
    }

    /// Reader: copy out the latest published payload, retrying up to
    /// `max_attempts` times if a write is caught mid-flight. Returns `None`
    /// if nothing has ever been published, or every attempt raced a write
    /// (vanishingly unlikely at inspector polling rates -- the caller
    /// should just try again next tick rather than busy-loop here).
    fn read_direction(&self, direction: Direction, max_attempts: u32) -> Option<Vec<u8>> {
        for _ in 0..max_attempts.max(1) {
            let slot = self.latest_slot_atomic(direction).load(Ordering::Acquire);
            if slot == u32::MAX || slot >= self.slot_count {
                return None;
            }
            let seq0 = self.slot_seq_atomic(direction, slot).load(Ordering::Acquire);
            if seq0 & 1 != 0 {
                continue; // mid-write, retry
            }
            let len = self.slot_len_atomic(direction, slot).load(Ordering::Acquire) as usize;
            if len > self.slot_capacity as usize { continue; }
            let mut buf = vec![0u8; len];
            unsafe {
                std::ptr::copy_nonoverlapping(self.slot_data_ptr(direction, slot), buf.as_mut_ptr(), len);
            }
            let seq1 = self.slot_seq_atomic(direction, slot).load(Ordering::Acquire);
            if seq0 == seq1 {
                return Some(buf);
            }
            // Torn read -- a write landed mid-copy. Retry.
        }
        None
    }

    pub fn read_latest(&self, max_attempts: u32) -> Option<Vec<u8>> {
        self.read_direction(Direction::Response, max_attempts)
    }

    pub fn read_request(&self, max_attempts: u32) -> Option<Vec<u8>> {
        self.read_direction(Direction::Request, max_attempts)
    }
}

#[derive(Clone, Copy)]
enum Direction { Request, Response }
