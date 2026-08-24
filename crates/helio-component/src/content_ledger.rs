//! The production [`HandleLedger`] for content-addressed assets: a sharded,
//! allocation-free-on-the-hot-path strong-count table keyed by
//! `pulsar_scenedb::HandleId`, with a registrable exactly-once drop callback.
//!
//! # Layering (deliberately strict)
//!
//! This crate NEVER touches wgpu/GPU types here. The ledger counts
//! references and reports 1→0 transitions; what a transition MEANS is the
//! callback's business. Today's callback evicts the CPU parse-cache entry
//! (`crate::subsystems::SharedMesh`) -- the GPU-residency hookpoint gets
//! consumed by the streaming scheduler in a later milestone, by simply
//! registering a callback that also frees GPU state. Nothing here changes
//! when that happens; that's the point of the seam.
//!
//! # Locking shape
//!
//! 64 independent shards (`parking_lot::Mutex<HashMap<HandleId, Entry>>`
//! each), selected by the id's HIGH bits: content ids are hash digests, so
//! their high bits are as good as random, which makes shard assignment
//! effectively uniform -- two unrelated assets colliding on one shard's
//! lock has probability ~1/64 and no pathological clustering exists to
//! engineer. All mutations of ONE id serialize on its shard lock, which is
//! the entire ordering guarantee the counts need:
//!
//! - Every acquire/release of an id observes a linearized count sequence.
//! - The 1→0 transition removes the entry UNDER the lock; the drop callback
//!   fires AFTER the lock is released (never under any lock: callbacks may
//!   take other locks, re-enter the parse cache, or allocate).
//!
//! Exactly-once follows from removal-under-lock: only the thread whose
//! decrement observed 1→0 removes the entry, and it does so once; every
//! later release of that id finds NO entry and takes the unknown-release
//! path (tolerated, logged). A concurrent re-acquire racing the eviction
//! creates a fresh entry with count 1 -- meaning the callback's eviction of
//! cached state must be RESURRECTION-SAFE (a subsequent load re-parses and
//! re-registers). That TOCTOU is inherent to refcounted caches and is why
//! the callback contract below says "eviction is advisory", not "the asset
//! is gone".
//!
//! # Panic safety
//!
//! The ledger state transition (count→0, entry removed) is complete BEFORE
//! the callback runs, and the call is wrapped in `catch_unwind`: a panicking
//! callback cannot corrupt counts, cannot lose or duplicate the transition,
//! and cannot poison shard locks (parking_lot poisons nothing anyway, but
//! the boundary is explicit so a panicking eviction surfaces as a logged
//! error while the process keeps running -- an editor dropping one asset
//! must not take down the session over its cleanup path).
//!
//! # Hot-path cost
//!
//! Steady-state acquire/release on a live id: one hash + one shard-mutex
//! acquisition + an integer add/sub. ZERO allocation (entry lives inline in
//! the map node; no Vec growth, no boxing). The despawn fast path
//! (`release_row`) is a loop over the batch -- still O(n) total, still zero
//! allocations, and still ONE logical operation per dying entity from the
//! caller's perspective.

use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, OnceLock};

use parking_lot::Mutex;
use pulsar_scenedb::handle_ledger::{HandleId, HandleLedger};

use crate::content_id::ContentId;

/// Shard count -- power of two so shard selection is a shift+mask. 64 gives
/// ~uncontended behavior up to heavy multithreaded hydrate loads while
/// keeping total idle memory negligible (one empty HashMap per shard).
const SHARDS: usize = 64;
const SHARD_SHIFT: u32 = 128 - 6; // log2(64)

/// What fires when an asset's LAST reference dies. Receives the id, not the
/// payload: resolving id→state is the registrant's own concern (it owns the
/// caches the id keys into).
///
/// Contract:
/// - Called EXACTLY ONCE per 1→0 transition, never at 2→1 or on unknown ids.
/// - May be called from ANY thread that performed the final release.
/// - Must be resurrection-safe: another thread may have already re-acquired
///   the same id and re-populated shared state; eviction is advisory.
/// - Must not panic (a panic is caught and logged -- see module doc) and
///   must not block indefinitely (it runs inline on the releasing thread).
pub type DropCallback = Box<dyn Fn(ContentId) + Send + Sync>;

struct Entry {
    strong: i64,
}

struct Shard {
    entries: Mutex<HashMap<HandleId, Entry>>,
}

/// See the module doc. Cheap to clone (`Arc`-free but all state behind
/// shard mutexes; share one instance process-wide via
/// [`shared_content_ledger`]).
pub struct ContentLedger {
    shards: Box<[Shard]>,
    /// Registered exactly once per process (first registration wins --
    /// there is exactly one consumer layer per milestone; re-registration
    /// attempts are logged and rejected rather than silently replacing a
    /// live hookpoint mid-session).
    drop_callback: OnceLock<DropCallback>,
}

impl ContentLedger {
    pub fn new() -> Self {
        Self {
            shards: (0..SHARDS)
                .map(|_| Shard {
                    entries: Mutex::new(HashMap::new()),
                })
                .collect(),
            drop_callback: OnceLock::new(),
        }
    }

    /// Registers THE drop callback. First registration wins for the
    /// process-lifetime of this instance (see field doc); returns `false`
    /// if one was already installed.
    pub fn set_drop_callback(&self, cb: DropCallback) -> bool {
        self.drop_callback.set(cb).is_ok()
    }

    #[inline]
    fn shard(&self, id: HandleId) -> &Shard {
        &self.shards[((id.0 >> SHARD_SHIFT) as usize) & (SHARDS - 1)]
    }

    /// Current strong count (test/audit tooling -- NOT a hot-path API; do
    /// not build game logic on polling this).
    pub fn strong_count(&self, id: HandleId) -> i64 {
        self.shard(id)
            .entries
            .lock()
            .get(&id)
            .map(|e| e.strong)
            .unwrap_or(0)
    }

    /// Reconciles ledger truth against an externally computed multiset
    /// `(id, expected_strong_count)` -- the debug-build / test assertion
    /// half of the capability. Returns every mismatch found (ledger vs
    /// expectation), capped at 32 entries so a broken batch doesn't produce
    /// a gigabyte of error text. Ids absent from `live` but present in the
    /// ledger ARE mismatches; vice versa too. Zero-count expectations are
    /// treated as "should be absent".
    ///
    /// Deliberately cheap enough for debug builds (one lock pass per shard,
    /// no allocation beyond the error report) but NOT free -- call at
    /// frame/level boundaries, not per entity.
    pub fn audit(
        &self,
        live: impl IntoIterator<Item = (HandleId, u32)>,
    ) -> Result<(), Vec<AuditMismatch>> {
        let mut expected: HashMap<HandleId, i64> = HashMap::new();
        for (id, count) in live {
            if count > 0 {
                expected.insert(id, count as i64);
            }
        }

        let mut mismatches = Vec::new();
        for shard in &self.shards {
            let entries = shard.entries.lock();
            for (&id, entry) in entries.iter() {
                match expected.remove(&id) {
                    Some(exp) if exp == entry.strong => {}
                    Some(exp) => mismatches.push(AuditMismatch {
                        id,
                        ledger: entry.strong,
                        expected: exp,
                    }),
                    None => mismatches.push(AuditMismatch {
                        id,
                        ledger: entry.strong,
                        expected: 0,
                    }),
                }
            }
        }
        // Whatever survives in `expected` was never counted (or already at
        // zero where nonzero was required).
        for (id, exp) in expected {
            mismatches.push(AuditMismatch {
                id,
                ledger: 0,
                expected: exp,
            });
        }
        if mismatches.is_empty() {
            Ok(())
        } else {
            mismatches.truncate(32);
            Err(mismatches)
        }
    }

    /// Release-side core shared by [`HandleLedger::release`] and
    /// [`HandleLedger::release_row`]: decrements under the shard lock,
    /// removes at 1→0, invokes the callback outside the lock with panic
    /// containment. Returns the id whose callback should fire, if any --
    /// returned rather than fired internally so `release_row` can fire ALL
    /// its transitions after ITS lock work completes (still outside any
    /// lock; batching the invocations keeps one dying entity's evictions
    /// adjacent).
    fn release_inner(&self, id: HandleId, out_dropped: &mut Vec<ContentId>) {
        if id.is_zero() {
            return; // zero convention: unreachable via World events, kept for direct callers
        }
        let shard = self.shard(id);
        let dropped = {
            let mut entries = shard.entries.lock();
            match entries.get_mut(&id) {
                Some(entry) => {
                    entry.strong -= 1;
                    if entry.strong > 0 {
                        None
                    } else {
                        debug_assert_eq!(
                            entry.strong,
                            0,
                            "strong count went negative for id {} -- double-release past zero",
                            crate::content_id::ContentId(id.0)
                        );
                        entries.remove(&id);
                        Some(ContentId(id.0))
                    }
                }
                None => {
                    // Unknown release: tolerated by contract (World can emit
                    // these after into_inner escape-hatch writes, or a
                    // caller bug double-releasing). Never wrap, never UB --
                    // surface loudly in debug, warn in release.
                    debug_assert!(
                        false,
                        "release of untracked handle id {}",
                        crate::content_id::ContentId(id.0)
                    );
                    tracing::warn!(
                        "ContentLedger: release of untracked content id {} (double-release or \
                         pre-ledger insert); ignored",
                        ContentId(id.0)
                    );
                    None
                }
            }
        };
        if let Some(content) = dropped {
            out_dropped.push(content);
        }
    }

    fn fire_drop_callbacks(&self, dropped: &[ContentId]) {
        if dropped.is_empty() {
            return;
        }
        let Some(cb) = self.drop_callback.get() else {
            // No callback registered: transitions still happen correctly;
            // consumers just aren't listening (counts-only usage, tests).
            return;
        };
        for &content in dropped {
            let result = catch_unwind(AssertUnwindSafe(|| cb(content)));
            if result.is_err() {
                // State is ALREADY consistent (removal preceded this call);
                // contain the panic, keep the process alive, make it loud.
                tracing::error!(
                    "ContentLedger drop callback PANICKED for content id {} -- eviction skipped; \
                     ledger state remains consistent",
                    content
                );
            }
        }
    }
}

impl Default for ContentLedger {
    fn default() -> Self {
        Self::new()
    }
}

impl HandleLedger for ContentLedger {
    fn acquire(&self, id: HandleId) {
        if id.is_zero() {
            return;
        }
        let mut entries = self.shard(id).entries.lock();
        let entry = entries.entry(id).or_insert(Entry { strong: 0 });
        entry.strong += 1;
    }

    fn release(&self, id: HandleId) {
        // `Vec::new()` allocates nothing; the single-element push happens
        // only on a 1→0 transition (rare by definition -- an asset died).
        // Live-id inc/dec, the actual steady state, stays allocation-free.
        let mut dropped = Vec::new();
        self.release_inner(id, &mut dropped);
        self.fire_drop_callbacks(&dropped);
    }

    fn release_row(&self, row: &[HandleId]) {
        // One pass over the batch, collecting transitions per shard as we
        // go; each id's mutation serializes on its own shard exactly like
        // singular `release`. Callbacks fire afterwards, outside every
        // lock, exactly-once each.
        let mut dropped = Vec::new();
        for &id in row {
            self.release_inner(id, &mut dropped);
        }
        self.fire_drop_callbacks(&dropped);
    }
}

/// One divergence found by [`ContentLedger::audit`].
#[derive(Clone, Copy, Debug)]
pub struct AuditMismatch {
    pub id: HandleId,
    pub ledger: i64,
    pub expected: i64,
}

static SHARED_LEDGER: OnceLock<Arc<ContentLedger>> = OnceLock::new();

/// The process-global ledger every `StaticMeshComponent` hydration shares.
/// Global is idiomatic HERE precisely because hydrate's fixed signature
/// (`&mut World, Entity, &Value`) has no context object to thread a ledger
/// through -- the same reason `engine_state::get_project_path()` is global.
pub fn shared_content_ledger() -> Arc<ContentLedger> {
    Arc::clone(SHARED_LEDGER.get_or_init(|| Arc::new(ContentLedger::new())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn h(v: u128) -> HandleId {
        HandleId(v)
    }

    #[test]
    fn inc_dec_and_swap_math() {
        let ledger = ContentLedger::new();
        ledger.acquire(h(1));
        ledger.acquire(h(1));
        assert_eq!(ledger.strong_count(h(1)), 2);

        ledger.release(h(1));
        assert_eq!(ledger.strong_count(h(1)), 1);
        ledger.release(h(1));
        assert_eq!(ledger.strong_count(h(1)), 0, "entry removed at zero");
        assert_eq!(ledger.strong_count(h(1)), 0);
    }

    #[test]
    fn drop_callback_fires_exactly_once_at_transition_and_never_at_two_to_one() {
        let ledger = ContentLedger::new();
        static FIRES: AtomicUsize = AtomicUsize::new(0);
        static LAST: Mutex<Option<u128>> = Mutex::new(None);
        assert!(ledger.set_drop_callback(Box::new(|c| {
            FIRES.fetch_add(1, Ordering::SeqCst);
            *LAST.lock() = Some(c.0);
        })));

        ledger.acquire(h(42));
        ledger.acquire(h(42));
        ledger.release(h(42)); // 2→1: MUST NOT fire
        assert_eq!(FIRES.load(Ordering::SeqCst), 0);

        ledger.release(h(42)); // 1→0: fires ONCE
        assert_eq!(FIRES.load(Ordering::SeqCst), 1);
        assert_eq!(*LAST.lock(), Some(42));

        // NOTE: no unknown-id release here -- debug builds assert on those
        // (see `release_unknown_is_asserted_in_debug_and_ignored_in_release`);
        // this test pins the exactly-once transition behavior itself.
    }

    #[cfg(debug_assertions)]
    #[test]
    fn release_unknown_is_asserted_in_debug() {
        let ledger = ContentLedger::new();
        let result = catch_unwind(AssertUnwindSafe(|| ledger.release(h(777))));
        assert!(
            result.is_err(),
            "debug builds must loudly assert unknown releases"
        );
    }

    #[cfg(not(debug_assertions))]
    #[test]
    fn release_unknown_is_ignored_in_release() {
        let ledger = ContentLedger::new();
        ledger.release(h(777)); // logged warning, no panic, no wraparound
        assert_eq!(ledger.strong_count(h(777)), 0);
    }

    #[test]
    fn second_callback_registration_is_rejected() {
        let ledger = ContentLedger::new();
        assert!(ledger.set_drop_callback(Box::new(|_| {})));
        assert!(!ledger.set_drop_callback(Box::new(|_| {})));
    }

    #[test]
    fn release_of_zero_is_tolerated_everywhere() {
        let ledger = ContentLedger::new();
        ledger.release(HandleId::ZERO); // zero convention: silent no-op
        ledger.acquire(HandleId::ZERO); // ditto
        assert_eq!(ledger.strong_count(HandleId::ZERO), 0);
    }

    #[test]
    fn release_row_fires_each_true_last_unref_exactly_once() {
        let ledger = ContentLedger::new();
        static FIRES: AtomicUsize = AtomicUsize::new(0);
        assert!(ledger.set_drop_callback(Box::new(|_| {
            FIRES.fetch_add(1, Ordering::SeqCst);
        })));

        // Multiset: [7,7,9] acquired twice+once respectively...
        ledger.acquire(h(7));
        ledger.acquire(h(7));
        ledger.acquire(h(9));
        // ...then the whole batch dies at once.
        ledger.release_row(&[h(7), h(7), h(9)]);
        assert_eq!(ledger.strong_count(h(7)), 0);
        assert_eq!(ledger.strong_count(h(9)), 0);
        assert_eq!(
            FIRES.load(Ordering::SeqCst),
            2,
            "two distinct contents died"
        );
    }

    #[test]
    fn panicking_callback_does_not_corrupt_state_or_escape() {
        let ledger = ContentLedger::new();
        static FIRES: AtomicUsize = AtomicUsize::new(0);
        assert!(ledger.set_drop_callback(Box::new(|c| {
            FIRES.fetch_add(1, Ordering::SeqCst);
            if c.0 == 13 {
                panic!("deliberate test panic inside eviction");
            }
        })));

        ledger.acquire(h(13));
        ledger.release(h(13)); // panicking eviction contained
        assert_eq!(FIRES.load(Ordering::SeqCst), 1);
        assert_eq!(
            ledger.strong_count(h(13)),
            0,
            "transition survived the panic"
        );

        // Ledger fully usable afterwards.
        ledger.acquire(h(14));
        ledger.release(h(14));
        assert_eq!(FIRES.load(Ordering::SeqCst), 2);
        assert_eq!(ledger.strong_count(h(14)), 0);

        // And re-acquiring the previously-doomed id works fresh.
        ledger.acquire(h(13));
        assert_eq!(ledger.strong_count(h(13)), 1);
        ledger.release(h(13));
        assert_eq!(FIRES.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn audit_reports_every_mismatch_class() {
        let ledger = ContentLedger::new();
        ledger.acquire(h(100));
        ledger.acquire(h(100));

        // Perfect match.
        assert!(ledger.audit([(h(100), 2)]).is_ok());

        // Under-counted expectation.
        assert!(ledger.audit([(h(100), 1)]).is_err());

        // Ledger holds an id the world forgot about.
        ledger.acquire(h(200));
        let errs = ledger.audit([(h(100), 2)]).unwrap_err();
        assert!(
            errs.iter()
                .any(|m| m.id == h(200) && m.expected == 0 && m.ledger == 1)
        );

        // World claims an id the ledger never saw.
        let errs = ledger.audit([(h(100), 2), (h(300), 5)]).unwrap_err();
        assert!(
            errs.iter()
                .any(|m| m.id == h(300) && m.ledger == 0 && m.expected == 5)
        );
    }

    #[test]
    fn high_bits_select_shards_without_clustering_all_ids_onto_one() {
        // Real content ids are hash digests -- effectively uniform across
        // their whole width, TOP bits included (shard selection reads the
        // top 6). Feed 512 avalanched ids through and demand most shards
        // saw traffic: expected distinct shards ≈ 64·(1−e^−8) ≈ 63, so a
        // degenerate selector can't fake its way past this bound.
        let ledger = ContentLedger::new();
        let mix = |x: u128| -> u128 {
            let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15_9E37_79B9_7F4A_7C15);
            z = (z ^ (z >> 64)).wrapping_mul(0xBF58_476D_1CE4_E5B9_BF58_476D_1CE4_E5B9);
            z ^ (z >> 96)
        };
        for i in 0..512u128 {
            ledger.acquire(HandleId(mix(i)));
        }
        let touched = ledger
            .shards
            .iter()
            .filter(|s| !s.entries.lock().is_empty())
            .count();
        assert!(
            touched >= 32,
            "only {touched}/{SHARDS} shards received traffic"
        );
    }
}
