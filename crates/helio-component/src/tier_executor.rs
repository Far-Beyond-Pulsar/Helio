//! The impure half of the texel-streaming policy seam (Helio#238 §2/§5).
//!
//! [`crate::tier_policy::evaluate`] decides; this module ACTS — translating
//! intents into SceneDB tier verbs wherever the caller holds a
//! `SceneGpuStore` handle. Deliberately thin and deliberately the only place
//! store types appear next to policy types: everything above it stays pure,
//! everything below it is SceneDB's own machinery.
//!
//! # Error mapping (the whole reason this shim exists)
//!
//! - [`TierError::Pinned`] — STATIC/interned payloads are pinned while any
//!   component references them; a release attempt is the system working as
//!   designed, not a failure. Logged at debug and skipped.
//! - [`TierError::StaleHandle`] / [`TierError::UnknownResource`] — the slot's
//!   tenant changed between snapshot and verb (content freed/reloaded).
//!   Skipped loudly: next frame's snapshot re-derives truth.
//! - Everything else ([`TierError::NotConfigured`], budget declines surfaced
//!   through audit, …) is SURFACED to the caller — silent degradation of
//!   streaming state would corrupt exactly the invariants the gate tests
//!   pin.
//!
//! Touches are issued with `Tier::Vram` targets: the store clamps to
//! `max(live, target)`, so replaying an intent whose target already moved is
//! always safe.

use pulsar_scenedb::gpu::{Tier, TierError, TierSelector, TierSpan};
use pulsar_scenedb::gpu::SceneGpuStore;

use crate::tier_policy::PolicyIntent;

/// What one slot's residency verbs address. Built once per frame from the
/// same component rows that produced the policy's snapshots — never guessed.
pub type SlotSelectors = [Option<TierSelector>];

/// Applies intents to the store. Returns the number of intents applied
/// cleanly; skipped-as-designed pins/stales are logged, real errors surface.
pub fn apply_intents(
    store: &SceneGpuStore,
    selectors: &SlotSelectors,
    intents: &[PolicyIntent],
) -> Result<usize, TierError> {
    let mut applied = 0usize;
    for intent in intents {
        let Some(selector) = selectors.get(intent_slot(*intent) as usize).and_then(|s| s.as_ref())
        else {
            // No selector for this slot: nothing to address, skip silently —
            // the feedback array covers 256 bindless slots, most scenes use a
            // handful.
            continue;
        };
        let result = match *intent {
            PolicyIntent::Touch { through_rank, .. } => {
                store.touch_tier(*selector, TierSpan::ThroughRank(through_rank), Tier::Vram)
            }
            PolicyIntent::Release { above_rank, .. } => {
                store.release_tier(*selector, TierSpan::ThroughRank(above_rank))
            }
        };
        match result {
            Ok(()) => applied += 1,
            Err(TierError::Pinned) => {
                tracing::debug!(
                    "tier policy: release skipped for a pinned static (slot {}) — expected \
                     while its content id is referenced by visible geometry",
                    intent_slot(*intent)
                );
            }
            Err(TierError::StaleHandle | TierError::UnknownResource) => {
                tracing::debug!(
                    "tier policy: intent skipped for slot {} (resource identity changed under \
                     the snapshot) — next frame's snapshot re-derives",
                    intent_slot(*intent)
                );
            }
            Err(e) => return Err(e),
        }
    }
    Ok(applied)
}

fn intent_slot(intent: PolicyIntent) -> u32 {
    match intent {
        PolicyIntent::Touch { slot, .. } | PolicyIntent::Release { slot, .. } => slot,
    }
}
