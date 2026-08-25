//! The texel-streaming demand policy (Helio#238 §2) — a PURE function.
//!
//! # Statelessness rule (read this first)
//!
//! [`evaluate`] is a free function over borrowed snapshots returning owned
//! intents. It reads no clock, no global, no cache; it holds nothing between
//! calls; there is no `static`, no interior mutability, no lazily-populated
//! anything in this module. Two consecutive evaluations of the same input
//! produce identical outputs, byte-for-byte, forever. That is not style — it
//! is the load-bearing half of Helio's frame-statelessness gate (issue Test
//! 4): all mutable streaming state lives in exactly one place, SceneDB's tier
//! substrate, and the renderer crates retain none of it.
//!
//! # Shape
//!
//! Inputs are SNAPSHOTS the caller assembles between frames:
//! - `wanted_mips[slot]` — the feedback compaction's per-slot max wanted mip
//!   (pixels are the only ground truth for perceptual demand).
//! - `residency[slot]` — one [`ResidencyRow`] built FROM `tier_peek`/audit
//!   records (this module never locks a store).
//! - `last_use[slot]` — SceneDB's own monotonic demand stamps.
//!
//! Outputs are INTENTS ([`PolicyIntent`]) the thin impure executor
//! ([`crate::tier_executor`], wherever the scene-store handle lives)
//! translates into `touch_tier`/`release_tier` calls.
//!
//! Decision rules (all pure math over the snapshot):
//! - **Promote** when demand exceeds residency beyond the up-hysteresis band;
//!   touches are ABSOLUTE demands upstream (`max(live, target)` clamp at the
//!   store), so a late/stale touch can never demote anyone.
//! - **Release** only downward, only beyond the down-hysteresis band, and
//!   never below the immortal floor ranks (rank 0..[`FLOOR_RANKS`] — the
//!   coarse-first canonical order makes those the permanent low-detail tail;
//    releasing them would strand sampling with no fallback).
//! - **Budget overflow**: if the projected VRAM footprint exceeds the budget,
//!   victims are chosen oldest-`last_use` first among release candidates —
//!   the same LRU ordering SceneDB itself applies, expressed here so the
//!   policy's DECISION is complete and reviewable before any verb fires.
//!   LRU *stamping* is deliberately absent: the touch verb stamps `last_use`
//!   as a side effect (demand IS a use); re-stamping here would be state.
//!
//! Unit-testable without a GPU: every test in this file runs on host, no
//! device, no store.

use crate::texture_cache::{PAGE_SIZE, MAX_PAYLOAD_PAGES};

/// Bindless-table slot count the feedback/policy arrays are sized for —
/// Helio's 256-wide `scene_textures`/`scene_samplers` binding arrays.
///
/// Mirrors `libhelio::VT_SLOT_COUNT`. Declared locally because this crate's
/// `libhelio` resolves through Pulsar-Native's own workspace pin, not the
/// renderer checkout; the authority test in `tests/tier_policy.rs` pins the
/// two together so they cannot drift.
pub const VT_SLOT_COUNT: usize = 256;

/// Demand must exceed residency by more than this before we promote — one
/// mip of measurement noise (derivative rounding, intra-cell races) must not
/// cause a promotion flight.
pub const HYSTERESIS_UP: u32 = 1;
/// Residency must exceed demand by more than this before we release.
pub const HYSTERESIS_DOWN: u32 = 2;

/// Ranks 0..N that are never release candidates: the coarse-first canonical
/// body order puts the coarsest pages FIRST, so this prefix is the permanent
/// low-detail fallback every shader miss lands on.
pub const FLOOR_RANKS: u32 = 1;

/// One slot's residency snapshot, assembled by the caller FROM
/// `tier_peek`/`tier_audit` rows. Mirrors the audit vocabulary rather than
/// importing store types wholesale, so the policy stays compilable and
/// testable against any substrate revision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidencyRow {
    /// SceneDB `Tier` discriminant: 0 = Disk, 1 = Ram, 2 = Vram.
    /// (Kept as the raw byte — the policy compares it against [`TIER_VRAM`]
    /// and nothing else.)
    pub tier_byte: u8,
    /// Inclusive rank-prefix watermark (SceneDB's `resident_through`).
    pub resident_through_rank: u32,
    /// True when the entry is STATIC/pinned while referenced (interned
    /// content-id payloads are exactly this while any component references
    /// them). Pinned rows are skipped by release candidates entirely.
    pub pinned: bool,
}

impl Default for ResidencyRow {
    fn default() -> Self {
        // Absence of a record READS AS Ram generation 0 with nothing resident
        // (SceneDB's own default-row contract).
        Self { tier_byte: TIER_RAM, resident_through_rank: 0, pinned: false }
    }
}

/// SceneDB `Tier::Vram`'s discriminant.
pub const TIER_VRAM: u8 = 2;
/// SceneDB `Tier::Ram`'s discriminant.
pub const TIER_RAM: u8 = 1;

/// Consumer budgets — the mirror of SceneDB's `TierConfig` pair, kept local so
/// `evaluate` signatures never move when the substrate grows fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TierConfigMirror {
    pub vram_budget_bytes: u64,
    #[allow(dead_code)] // RAM reconciliation happens inside SceneDB at flush;
                        // carried here so callers can log one coherent config.
    pub ram_budget_bytes: u64,
}

/// Everything [`evaluate`] needs, all of it borrowed.
#[derive(Debug, Clone, Copy)]
pub struct PolicyInput<'a> {
    /// Per-slot max wanted mip from this frame's feedback compaction
    /// (`libhelio::unpack_feedback` output, already max-reduced per slot).
    /// Index == bindless slot index. Slots past the slice end read as "no
    /// demand observed".
    pub wanted_mips: &'a [u8],
    /// Per-slot residency snapshots; may be shorter than [`VT_SLOT_COUNT`]
    /// (missing rows default per [`ResidencyRow::default`]).
    pub residency: &'a [ResidencyRow],
    /// SceneDB's per-slot `last_use` stamps (0 = never touched). May be
    /// shorter than [`VT_SLOT_COUNT`].
    pub last_use: &'a [u64],
    /// Consumer budgets (engine settings translated once at configure time).
    pub budgets: TierConfigMirror,
    /// The caller's monotonic stamp for this evaluation instant — used ONLY
    /// for ordering arithmetic (age computations); never stored.
    pub now_stamp: u64,
}

/// One decision. `through_rank`/`above_rank` speak SceneDB's
/// [`TierSpan::ThroughRank`] dialect: inclusive prefix for touch, exclusive
/// remainder for release.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyIntent {
    /// Extend residency through rank `through_rank` (absolute demand up).
    Touch { slot: u32, through_rank: u32 },
    /// Withdraw residency above rank `above_rank` (relative demote down).
    Release { slot: u32, above_rank: u32 },
}

/// Evaluates one snapshot into a batch of intents.
///
/// Deterministic output order (slot ascending within each phase, releases
/// after touches) — the executor relies on nothing about order, but tests
/// assert exact vectors, so the order is part of this function's contract.
pub fn evaluate(input: PolicyInput<'_>) -> Vec<PolicyIntent> {
    let mut intents = Vec::new();

    // No feedback snapshot ⇒ no decisions AT ALL. A frame whose compaction
    // readback was not ready must never be misread as "nothing wants
    // anything"; absence of evidence about demand is not evidence of absence.
    if input.wanted_mips.is_empty() {
        return intents;
    }

    // ── Phase 1: hysteresis-banded demand reconciliation, slot ascending ──
    let mut candidate_releases: Vec<(u32, u32, u64)> = Vec::new(); // (slot, keep_through, last_use)
    let mut vram_bytes: u64 = 0;

    let slot_count = input.wanted_mips.len().min(VT_SLOT_COUNT);
    for slot in 0..slot_count {
        let row = input.residency.get(slot).copied().unwrap_or_default();
        let wanted = u32::from(input.wanted_mips[slot]);

        // Stale-row tolerance: a row claiming ranks past any plausible chain
        // (4096-page ceiling = MAX_PAYLOAD_PAGES) is treated as its clamped
        // self rather than trusted; garbage in must not become verbs out.
        let resident = row.resident_through_rank.min(max_rank_ceiling());

        if wanted > resident.saturating_add(HYSTERESIS_UP) {
            // Demand clears the up-band: promote through the WANTED rank
            // (absolute; the store clamps to max(live, target), so this can
            // never demote anyone regardless of intent ordering).
            intents.push(PolicyIntent::Touch { slot: slot as u32, through_rank: wanted });
        } else if resident > wanted.saturating_add(HYSTERESIS_DOWN) && !row.pinned {
            // Demand fell below the down-band: candidate release keeping the
            // demanded prefix — but never below the immortal floor.
            let keep_through = wanted.max(FLOOR_RANKS.saturating_sub(1));
            if resident > keep_through {
                let age = input.now_stamp.saturating_sub(input.last_use.get(slot).copied().unwrap_or(0));
                candidate_releases.push((slot as u32, keep_through, age));
            }
        }

        // Projected VRAM accounting: ranks are page-granular.
        if row.tier_byte == TIER_VRAM {
            vram_bytes += u64::from(resident + 1) * PAGE_SIZE_U64;
        }
    }

    // ── Phase 2: LRU victim selection under forced budget overflow ──
    // Only if the CURRENT footprint already busts the budget (projected
    // promotions are admitted optimistically — SceneDB re-checks admission at
    // flush and evicts there too; duplicating full admission math would be a
    // second policy drifting against the first).
    if input.budgets.vram_budget_bytes > 0 && vram_bytes > input.budgets.vram_budget_bytes {
        let excess = vram_bytes - input.budgets.vram_budget_bytes;
        // Oldest stamp first (largest age), ties broken by slot ascending —
        // deterministic victim order.
        candidate_releases.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
        let mut reclaimed: u64 = 0;
        let mut taken = vec![false; candidate_releases.len()];
        for (idx, (slot, _keep, _age)) in candidate_releases.iter().enumerate() {
            if reclaimed >= excess {
                break;
            }
            let row = input.residency.get(*slot as usize).copied().unwrap_or_default();
            let resident = row.resident_through_rank.min(max_rank_ceiling());
            // A victim releases EVERYTHING above the immortal floor.
            let keep_through = FLOOR_RANKS.saturating_sub(1);
            if resident <= keep_through || row.pinned {
                continue;
            }
            reclaimed += u64::from(resident - keep_through) * PAGE_SIZE_U64;
            taken[idx] = true;
        }
        // Emit victim releases (slot-ascending for deterministic vectors),
        // then the remaining ordinary band releases.
        let mut victim_slots: Vec<usize> =
            (0..candidate_releases.len()).filter(|&i| taken[i]).collect();
        victim_slots.sort_by_key(|&i| candidate_releases[i].0);
        for i in victim_slots {
            let (slot, _, _) = candidate_releases[i];
            intents.push(PolicyIntent::Release { slot, above_rank: FLOOR_RANKS - 1 });
        }
        for (i, (slot, keep_through, _)) in candidate_releases.iter().enumerate() {
            if taken[i] {
                continue;
            }
            intents.push(PolicyIntent::Release { slot: *slot, above_rank: *keep_through });
        }
    } else {
        for (slot, keep_through, _) in candidate_releases {
            intents.push(PolicyIntent::Release { slot, above_rank: keep_through });
        }
    }

    intents
}

/// The rank ceiling any snapshot row may claim (`MAX_PAYLOAD_PAGES-1`, the
/// texture-payload coverage ceiling — mirrored from [`MAX_PAYLOAD_PAGES`] via
/// this helper so intent stays readable at use sites).
fn max_rank_ceiling() -> u32 {
    MAX_PAYLOAD_PAGES - 1
}

/// `PAGE_SIZE` as u64 for byte arithmetic.
const PAGE_SIZE_U64: u64 = PAGE_SIZE as u64;

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(vram_pages: u64) -> TierConfigMirror {
        TierConfigMirror {
            vram_budget_bytes: vram_pages * PAGE_SIZE_U64,
            ram_budget_bytes: u64::MAX,
        }
    }

    fn row_vram(through: u32) -> ResidencyRow {
        ResidencyRow { tier_byte: TIER_VRAM, resident_through_rank: through, pinned: false }
    }

    #[test]
    fn empty_feedback_is_a_no_op() {
        let input = PolicyInput {
            wanted_mips: &[],
            residency: &[],
            last_use: &[],
            budgets: cfg(1000),
            now_stamp: 42,
        };
        assert!(evaluate(input).is_empty());
    }

    #[test]
    fn hysteresis_band_edges_stay_put() {
        // Resident through 5. Demand 4 (= 5-1): inside the down-band → hold.
        // Demand 6 (= 5+1): NOT above the up-band (needs > 5+1) → hold.
        let residency = [row_vram(5)];
        for wanted in [4u8, 5, 6] {
            let input = PolicyInput {
                wanted_mips: &[wanted],
                residency: &residency,
                last_use: &[10],
                budgets: cfg(1000),
                now_stamp: 20,
            };
            assert!(evaluate(input).is_empty(), "wanted {wanted} must stay put");
        }
        // Demand 7 (> 5+1) promotes; demand 2 (< 5-2) releases.
        let input = PolicyInput {
            wanted_mips: &[7],
            residency: &residency,
            last_use: &[10],
            budgets: cfg(1000),
            now_stamp: 20,
        };
        assert_eq!(evaluate(input), vec![PolicyIntent::Touch { slot: 0, through_rank: 7 }]);
        let input = PolicyInput {
            wanted_mips: &[2],
            residency: &residency,
            last_use: &[10],
            budgets: cfg(1000),
            now_stamp: 20,
        };
        assert_eq!(
            evaluate(input),
            vec![PolicyIntent::Release { slot: 0, above_rank: 2 }]
        );
    }

    #[test]
    fn demand_never_demotes_and_release_never_promotes() {
        // Wanted BELOW resident produces only Release intents…
        let residency = [row_vram(9)];
        let input = PolicyInput {
            wanted_mips: &[0],
            residency: &residency,
            last_use: &[0],
            budgets: cfg(1000),
            now_stamp: 5,
        };
        for intent in evaluate(input) {
            match intent {
                PolicyIntent::Touch { .. } => panic!("touch must not fire below residency"),
                PolicyIntent::Release { above_rank, .. } => assert!(above_rank < 9),
            }
        }
        // …and wanted ABOVE resident produces only Touch intents.
        let input = PolicyInput {
            wanted_mips: &[12],
            residency: &residency,
            last_use: &[0],
            budgets: cfg(1000),
            now_stamp: 5,
        };
        for intent in evaluate(input) {
            match intent {
                PolicyIntent::Touch { through_rank, .. } => assert!(through_rank > 9),
                PolicyIntent::Release { .. } => panic!("release must not fire above residency"),
            }
        }
    }

    #[test]
    fn floor_ranks_are_immortal() {
        // Resident through 3, demand collapses to 0: the release keeps the
        // floor prefix (FLOOR_RANKS-1 = 0), i.e. above_rank 0 — rank 0 (the
        // coarsest page, the permanent fallback) survives.
        let residency = [row_vram(3)];
        let input = PolicyInput {
            wanted_mips: &[0],
            residency: &residency,
            last_use: &[1],
            budgets: cfg(1000),
            now_stamp: 9,
        };
        assert_eq!(
            evaluate(input),
            vec![PolicyIntent::Release { slot: 0, above_rank: 0 }]
        );
        // Even full-budget-pressure victims stop at the floor.
        let residency = [
            row_vram(50), // victim (old)
            row_vram(50), // stays (fresh)
        ];
        let input = PolicyInput {
            wanted_mips: &[0, 0],
            residency: &residency,
            last_use: &[100, 200],
            budgets: cfg(10), // force massive overflow
            now_stamp: 300,
        };
        for intent in evaluate(input) {
            if let PolicyIntent::Release { above_rank, .. } = intent {
                assert_eq!(above_rank, FLOOR_RANKS - 1, "victim kept the floor");
            }
        }
    }

    #[test]
    fn lru_victim_selection_under_forced_overflow() {
        // Three slots resident through 29 (30 pages each = 90 pages total);
        // budget admits 50 → excess 40. One victim reclaims 29 < 40, so a
        // SECOND is taken (58 ≥ 40, stop). All three slots are release
        // candidates (demand collapsed everywhere); the test asserts WHO was
        // picked for the forced eviction: the two OLDEST stamps (slot 2 @age
        // 950, slot 0 @age 900), never the freshest (slot 1 @age 100) — which
        // still releases, but through the ordinary hysteresis path.
        let residency = [
            ResidencyRow { tier_byte: TIER_VRAM, resident_through_rank: 29, pinned: false },
            ResidencyRow { tier_byte: TIER_VRAM, resident_through_rank: 29, pinned: false },
            ResidencyRow { tier_byte: TIER_VRAM, resident_through_rank: 29, pinned: false },
        ];
        // Explicit stamps: slot0 @100 (age 900), slot1 @900 (age 100, the
        // freshest), slot2 @50 (age 950). Oldest-first: slot2, slot0, slot1.
        let input = PolicyInput {
            wanted_mips: &[0, 0, 0],
            residency: &residency,
            last_use: &[100, 900, 50],
            budgets: cfg(50),
            now_stamp: 1000,
        };
        assert_eq!(
            evaluate(input),
            vec![
                PolicyIntent::Release { slot: 0, above_rank: 0 }, // victim (2nd oldest)
                PolicyIntent::Release { slot: 2, above_rank: 0 }, // victim (oldest)
                PolicyIntent::Release { slot: 1, above_rank: 0 }, // ordinary band release
            ]
        );
    }

    #[test]
    fn stale_rows_are_clamped_not_trusted() {
        // A corrupted watermark far past the 4096-rank ceiling must clamp to
        // the ceiling, not propagate nonsense into intents or byte math.
        let residency = [ResidencyRow {
            tier_byte: TIER_VRAM,
            resident_through_rank: u32::MAX,
            pinned: false,
        }];
        let input = PolicyInput {
            wanted_mips: &[0],
            residency: &residency,
            last_use: &[1],
            budgets: cfg(1_000_000),
            now_stamp: 2,
        };
        // Clamped resident = 4095; demand 0 is far below the down-band → one
        // bounded release, nothing more.
        assert_eq!(
            evaluate(input),
            vec![PolicyIntent::Release { slot: 0, above_rank: 0 }]
        );
    }

    #[test]
    fn pinned_rows_are_skipped_by_releases_but_still_touchable() {
        // Pinned + over-resident: no release (static floors are immortal by
        // request), but a genuine demand raise still stamps a touch.
        let residency = [ResidencyRow {
            tier_byte: TIER_VRAM,
            resident_through_rank: 8,
            pinned: true,
        }];
        let input = PolicyInput {
            wanted_mips: &[0],
            residency: &residency,
            last_use: &[1],
            budgets: cfg(1000),
            now_stamp: 2,
        };
        assert!(evaluate(input).is_empty(), "pinned row must not be released");
        let input = PolicyInput {
            wanted_mips: &[11],
            residency: &residency,
            last_use: &[1],
            budgets: cfg(1000),
            now_stamp: 2,
        };
        assert_eq!(
            evaluate(input),
            vec![PolicyIntent::Touch { slot: 0, through_rank: 11 }],
            "promotion of a pinned row is still expressible (upstream clamps)"
        );
    }

    #[test]
    fn short_snapshots_default_missing_rows() {
        // Feedback covering 2 slots, residency covering 1: the missing row
        // defaults (Ram, nothing resident) and demand 0 stays quiet.
        let wanted = [0u8, 0u8];
        let residency = [row_vram(0)];
        let input = PolicyInput {
            wanted_mips: &wanted,
            residency: &residency,
            last_use: &[7],
            budgets: cfg(1000),
            now_stamp: 8,
        };
        assert!(evaluate(input).is_empty());
    }

    #[test]
    fn evaluation_is_deterministic_across_calls() {
        let residency = [row_vram(9), row_vram(1)];
        let make_input = || PolicyInput {
            wanted_mips: &[12, 0],
            residency: &residency,
            last_use: &[3, 4],
            budgets: cfg(1000),
            now_stamp: 77,
        };
        assert_eq!(evaluate(make_input()), evaluate(make_input()));
    }
}

