//! Policy evaluation benchmarks (Helio#238 BM2): `tier_policy::evaluate`
//! cost versus touch count.
//!
//! PURE-CPU by design — that is the point being measured. The policy is a
//! free function over borrowed snapshots (see the module's statelessness
//! rule), so its cost must scale with SNAPSHOT WIDTH alone; nothing here
//! touches a device, a store, or an allocator beyond the output Vec.
//!
//! Measured shapes:
//! - BM2a: full-snapshot sweep — 8/64/256 slots, every slot demanding
//!   promotion (worst case: one Touch intent per slot).
//! - BM2b: release sweep — 256 fully-resident slots whose demand collapsed
//!   to zero (worst case: one Release intent per slot).
//! - BM2c: budget-overflow path — 256 VRAM-resident slots against a tiny
//!   budget forces the LRU victim-selection sort on top of phase 1.
//! - BM2d: quiet frame — 256 resident slots inside the hysteresis band
//!   (the common steady-state shape: zero intents out).
//!
//! Matches `texture_benchmarks.rs`'s `Instant`-timed idiom: each shape runs
//! WARMUP iterations untimed, then TIMES the batch and prints ns/eval plus
//! the intents-produced count so regressions in BOTH cost and decision
//! volume are visible side by side.

use helio_component::tier_policy::{
    evaluate, PolicyInput, ResidencyRow, TierConfigMirror, TIER_VRAM,
};
use std::time::Instant;

const WARMUP: usize = 200;
const BATCH: usize = 2_000;

fn budgets(vram_pages: u64) -> TierConfigMirror {
    TierConfigMirror {
        vram_budget_bytes: vram_pages.saturating_mul(16 * 1024), // PAGE_SIZE mirror
        ram_budget_bytes: u64::MAX,
    }
}

#[test]
fn bm2a_eval_cost_vs_promotion_count() {
    for slots in [8usize, 64, 256] {
        let wanted = vec![9u8; slots]; // demand far above residency 2
        let residency: Vec<ResidencyRow> = (0..slots)
            .map(|_| ResidencyRow { tier_byte: TIER_VRAM, resident_through_rank: 2, pinned: false })
            .collect();
        let last_use: Vec<u64> = vec![7; slots];
        let input = PolicyInput {
            wanted_mips: &wanted,
            residency: &residency,
            last_use: &last_use,
            budgets: budgets(u64::MAX),
            now_stamp: 10_000,
        };
        for _ in 0..WARMUP {
            std::hint::black_box(evaluate(std::hint::black_box(input)));
        }
        let start = Instant::now();
        let mut intents_out = 0usize;
        for _ in 0..BATCH {
            intents_out += evaluate(std::hint::black_box(input)).len();
        }
        let elapsed = start.elapsed();
        println!(
            "BM2a promote: {slots:>3} slots -> {:>8.1} ns/eval, {} intents/eval",
            elapsed.as_nanos() as f64 / BATCH as f64,
            intents_out / BATCH,
        );
        assert_eq!(intents_out / BATCH, slots, "every promoted slot yields one Touch");
    }
}

#[test]
fn bm2b_eval_cost_vs_release_count() {
    let slots = 256usize;
    // Fully resident, demand collapsed to zero → one Release per slot.
    let wanted = vec![0u8; slots];
    let residency: Vec<ResidencyRow> = (0..slots)
        .map(|_| ResidencyRow { tier_byte: TIER_VRAM, resident_through_rank: 30, pinned: false })
        .collect();
    let last_use: Vec<u64> = vec![5; slots];
    let input = PolicyInput {
        wanted_mips: &wanted,
        residency: &residency,
        last_use: &last_use,
        budgets: budgets(u64::MAX),
        now_stamp: 10_000,
    };
    for _ in 0..WARMUP {
        std::hint::black_box(evaluate(std::hint::black_box(input)));
    }
    let start = Instant::now();
    let mut intents_out = 0usize;
    for _ in 0..BATCH {
        intents_out += evaluate(std::hint::black_box(input)).len();
    }
    let elapsed = start.elapsed();
    println!(
        "BM2b release: {slots:>3} slots -> {:>8.1} ns/eval, {} intents/eval",
        elapsed.as_nanos() as f64 / BATCH as f64,
        intents_out / BATCH,
    );
    assert_eq!(intents_out / BATCH, slots, "every collapsed slot yields one Release");
}

#[test]
fn bm2c_eval_cost_under_budget_overflow_lru_victim_sweep() {
    let slots = 256usize;
    // 31 pages each × 256 slots = 7936 pages demanded; budget admits 512 →
    // the LRU victim loop walks candidates oldest-first until reclaimed.
    let wanted = vec![0u8; slots];
    let residency: Vec<ResidencyRow> = (0..slots)
        .map(|_| ResidencyRow { tier_byte: TIER_VRAM, resident_through_rank: 30, pinned: false })
        .collect();
    // Strictly increasing stamps: victim order is fully determined (slot 0
    // oldest) — also proves the sort does not explode the cost.
    let last_use: Vec<u64> = (0..slots).map(|i| i as u64).collect();
    let input = PolicyInput {
        wanted_mips: &wanted,
        residency: &residency,
        last_use: &last_use,
        budgets: budgets(512),
        now_stamp: 100_000,
    };
    for _ in 0..WARMUP {
        std::hint::black_box(evaluate(std::hint::black_box(input)));
    }
    let start = Instant::now();
    let mut victims = 0usize;
    for _ in 0..BATCH {
        let intents = evaluate(std::hint::black_box(input));
        victims += intents
            .iter()
            .filter(|i| matches!(i, helio_component::tier_policy::PolicyIntent::Release { above_rank: 0, .. }))
            .count();
    }
    let elapsed = start.elapsed();
    println!(
        "BM2c overflow: {slots:>3} slots -> {:>8.1} ns/eval, {} floor-keeping victims/eval",
        elapsed.as_nanos() as f64 / BATCH as f64,
        victims / BATCH,
    );
    assert!(victims / BATCH > 0, "forced overflow must evict someone");
}

#[test]
fn bm2d_quiet_frame_is_the_cheapest_shape() {
    let slots = 256usize;
    // Demand exactly at residency: everything inside both hysteresis bands.
    let wanted = vec![4u8; slots];
    let residency: Vec<ResidencyRow> = (0..slots)
        .map(|_| ResidencyRow { tier_byte: TIER_VRAM, resident_through_rank: 4, pinned: false })
        .collect();
    let last_use: Vec<u64> = vec![9; slots];
    let input = PolicyInput {
        wanted_mips: &wanted,
        residency: &residency,
        last_use: &last_use,
        budgets: budgets(100_000),
        now_stamp: 10_000,
    };
    for _ in 0..WARMUP {
        std::hint::black_box(evaluate(std::hint::black_box(input)));
    }
    let start = Instant::now();
    let mut intents_out = 0usize;
    for _ in 0..BATCH {
        intents_out += evaluate(std::hint::black_box(input)).len();
    }
    let elapsed = start.elapsed();
    println!(
        "BM2d quiet:   {slots:>3} slots -> {:>8.1} ns/eval, {} intents/eval",
        elapsed.as_nanos() as f64 / BATCH as f64,
        intents_out / BATCH,
    );
    assert_eq!(intents_out, 0, "steady-state frame decides nothing");
}
