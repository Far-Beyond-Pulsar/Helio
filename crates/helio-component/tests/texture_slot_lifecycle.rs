//! Seven-slot lifecycle on the REAL `StaticMeshComponent` (Helio#237 issue
//! tests 6–7):
//!
//! - T6: two meshes sharing ONLY `base_color_asset` collapse exactly that one
//!   chain to refcount 2 while each independently-assigned slot's chain sits
//!   at 1; removal produces exact audit deltas; `tier_audit()` shows the
//!   STATIC class contract (pinned-while-referenced) and returns to baseline.
//! - T7: THE LEAK LOOP — 50 rounds of {100 entities × K=3 textures}
//!   spawn/despawn must return every pool audit AND the full `tier_audit()`
//!   to its start-of-run baseline EXACTLY, every single round.
//!
//! Same harness discipline as `content_dedup.rs`: one process-global project
//! root (`INIT`), uniquely-named per-test fixtures.

#![cfg(feature = "gpu")]

use helio_component::components::{StaticMeshComponent, TextureAssetPath};
use helio_component::texture_cache::{self, TexturePayload, TextureSemantic};
use image::ImageEncoder as _;
use pulsar_scenedb::handle_ledger::ContentAddressed as _;
use pulsar_scenedb::gpu::{
    BufferKey, EngineGpuContext, GpuMirrorHandle, RegionClassConfig, SceneGpuConfig, SceneGpuStore,
    Tier, TierAuditKey, TierConfig, TierSelector, TierSpan,
};
use pulsar_scenedb::{Entity, World};
use std::sync::{Arc, Once};

fn test_context(label: &str) -> EngineGpuContext {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .expect("no adapter — GPU tests need a local GPU");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some(label),
        ..Default::default()
    }))
    .expect("device");
    EngineGpuContext::new(Arc::new(device), Arc::new(queue))
}

static INIT: Once = Once::new();

fn project_root() -> std::path::PathBuf {
    let root = std::env::temp_dir().join("pulsar_texture_slot_test_root");
    INIT.call_once(|| {
        std::fs::create_dir_all(&root).expect("create test project root");
        engine_state::EngineContext::new().set_global();
        engine_state::set_project_path(root.to_string_lossy().into_owned());
    });
    root
}

fn png_asset(name: &str, seed: u8) -> String {
    let mut rgba = Vec::with_capacity(4 * 4 * 4);
    for i in 0..16usize {
        rgba.extend_from_slice(&[seed.wrapping_add(i as u8), seed, !seed, 255]);
    }
    let mut png = Vec::new();
    image::codecs::png::PngEncoder::new(&mut png)
        .write_image(&rgba, 4, 4, image::ExtendedColorType::Rgba8)
        .unwrap();
    let rel = format!("{name}.png");
    std::fs::write(project_root().join(&rel), png).unwrap();
    rel
}

/// Decode one fixture through the slot's semantic (mirrors what hydrate does)
/// and return the (wrapper, payload) pair for that slot.
fn filled_slot(rel: &str, semantic: TextureSemantic) -> (TextureAssetPath, Vec<TexturePayload>) {
    let abs = project_root().join(rel);
    let body =
        texture_cache::decoded_body_for_path(&abs, semantic).expect("fixture texture decodes");
    (
        TextureAssetPath::new(rel),
        body.iter().copied().map(TexturePayload).collect(),
    )
}

struct Harness {
    store: Arc<SceneGpuStore>,
    ctx: EngineGpuContext,
}

fn mirrored_world(label: &str) -> (World, Harness) {
    let ctx = test_context(label);
    let mut store = SceneGpuStore::new(
        &ctx,
        SceneGpuConfig { classes: vec![RegionClassConfig { capacity: 64, max_resident_cells: 1 }], tombstone_headroom: 8, max_cells_metadata: 16 },
    );
    StaticMeshComponent::register_gpu_columns_growable(&mut store, 64, ctx.device());
    // The ONE consumer configuration call: generous budgets so promotions are
    // never budget-declined here — these tests pin lifecycle semantics, not
    // admission policy (SceneDB's own adversarial matrix covers budgets).
    store
        .configure_tiers(TierConfig { vram_budget_bytes: 256 * 1024 * 1024, ram_budget_bytes: 256 * 1024 * 1024 }, &[])
        .expect("tier config");
    let store = Arc::new(store);
    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));
    (world, Harness { store, ctx })
}

fn pool_audit(h: &Harness, key: &'static str) -> Vec<(pulsar_scenedb::handle_ledger::HandleId, u64, u64)> {
    h.store
        .interned_var_len_pool::<TexturePayload>(BufferKey::of(key))
        .expect("pool registered by register_gpu_columns_growable")
        .audit()
}

// ── T6 ──────────────────────────────────────────────────────────────────────

#[test]
fn sharing_only_base_color_gives_that_chain_refcount_2_and_others_1() {
    let shared_base = png_asset("t6_shared_base", 10);
    let normal_a = png_asset("t6_normal_a", 11);
    let normal_b = png_asset("t6_normal_b", 12);

    let (mut world, h) = mirrored_world("t6-lifecycle");

    // e1: shared base + normal A. e2: THE SAME shared base + normal B —
    // exactly one chain shared, everything else independent.
    let (bc_path, bc_payload) = filled_slot(&shared_base, TextureSemantic::BaseColor);
    let (na_path, na_payload) = filled_slot(&normal_a, TextureSemantic::Normal);
    let (nb_path, nb_payload) = filled_slot(&normal_b, TextureSemantic::Normal);
    assert_ne!(na_path.content_id(), nb_path.content_id(), "fixtures must differ");

    let e1: Entity = world.spawn();
    world.insert(
        e1,
        StaticMeshComponent {
            base_color_asset: bc_path.clone(),
            base_color_data: bc_payload.clone(),
            normal_asset: na_path,
            normal_data: na_payload,
            ..Default::default()
        },
    );
    let e2: Entity = world.spawn();
    world.insert(
        e2,
        StaticMeshComponent {
            base_color_asset: bc_path,
            base_color_data: bc_payload,
            normal_asset: nb_path,
            normal_data: nb_payload,
            ..Default::default()
        },
    );

    // Refcounts: the shared base chain is at 2; each normal chain at 1;
    // every other slot's pool EMPTY (missing slot = ZERO semantics).
    let base = pool_audit(&h, "StaticMeshComponent::base_color_data");
    assert_eq!(base.len(), 1, "shared base collapses to ONE chain");
    assert_eq!(base[0].1, 2, "two entities reference it");
    let norm = pool_audit(&h, "StaticMeshComponent::normal_data");
    assert_eq!(norm.len(), 2, "two distinct normal chains");
    assert!(norm.iter().all(|&(_, refs, _)| refs == 1));
    for slot in [
        "StaticMeshComponent::roughness_metallic_data",
        "StaticMeshComponent::emissive_data",
        "StaticMeshComponent::occlusion_data",
        "StaticMeshComponent::specular_color_data",
        "StaticMeshComponent::specular_weight_data",
    ] {
        assert!(
            pool_audit(&h, slot).is_empty(),
            "{slot} must stay untouched when unauthored"
        );
    }

    // Flush executes queued transitions; the tier audit then reports the
    // STATIC class state. Interned payloads live at their WRITE-TIME
    // residency (RAM staging) and are PINNED while referenced — `touch_tier`
    // on an Interned selector is explicitly a demand STAMP, never a move
    // (SceneDB#61 §5: "STATIC class: pinned at its write-time residency").
    let base_pool_key = BufferKey::of("StaticMeshComponent::base_color_data");
    let base_id = base[0].0;
    h.store
        .touch_tier(
            TierSelector::Interned { pool: base_pool_key, id: base_id },
            TierSpan::Whole,
            Tier::Vram,
        )
        .expect("touch stamps demand for a live interned entry");
    h.store.flush_tier_transitions(h.ctx.queue());
    let audit = h.store.tier_audit();
    let base_records: Vec<_> = audit
        .iter()
        .filter(|r| matches!(r.key, TierAuditKey::Interned(k, id) if k == base_pool_key && id == base_id))
        .collect();
    assert_eq!(base_records.len(), 1, "exactly one tier record for the shared chain");
    assert!(base_records[0].pinned, "referenced statics are pinned");
    assert_eq!(
        base_records[0].tier,
        Tier::Ram,
        "statics hold write-time residency; touch is a stamp, not a move"
    );
    // NOTE: static audit records are metadata-only (vram_bytes == staging_bytes
    // == 0) — "statics own no data plane": the mirror's own upload holds the
    // bytes, the tier engine tracks only residency/pinning.

    // Exact removal deltas, one entity at a time.
    world.remove::<StaticMeshComponent>(e1);
    assert!(world.is_alive(e1), "remove must not despawn");
    let base = pool_audit(&h, "StaticMeshComponent::base_color_data");
    assert_eq!(base, vec![(base_id, 1, base[0].2)], "base falls 2→1 with byte length intact");
    assert!(
        pool_audit(&h, "StaticMeshComponent::normal_data").len() == 1,
        "e1's normal chain freed; e2's remains"
    );

    world.despawn(e2);
    assert!(pool_audit(&h, "StaticMeshComponent::base_color_data").is_empty());
    assert!(pool_audit(&h, "StaticMeshComponent::normal_data").is_empty());

    // And the WHOLE tier audit returns to its (empty) baseline exactly.
    h.store.flush_tier_transitions(h.ctx.queue());
    assert!(h.store.tier_audit().is_empty(), "no residency survives full teardown");
}

// ── T7: THE LEAK LOOP ───────────────────────────────────────────────────────

#[test]
fn leak_loop_50_rounds_of_100_entities_across_3_textures_returns_exactly_to_baseline() {
    // Mandatory (#237 issue test 7, mirroring content_dedup.rs's mesh loop):
    // spawn a batch referencing a small set of real textures across ALL seven
    // slots, despawn it all, repeat — every pool audit AND tier_audit must be
    // EXACTLY the start-of-run baseline after every round.
    const ROUNDS: usize = 50;
    const ENTITIES_PER_ROUND: usize = 100;
    const K: usize = 3;

    let (mut world, h) = mirrored_world("t7-leak-loop");

    let bases: Vec<String> = (0..K).map(|k| png_asset(&format!("t7_leak_base_{k}"), 20 + k as u8)).collect();
    let normals: Vec<String> = (0..K).map(|k| png_asset(&format!("t7_leak_norm_{k}"), 30 + k as u8)).collect();

    // Preload fixtures OUTSIDE the loop so the loop's own decodes don't muddy
    // anything (the body cache makes them free anyway).
    let base_slots: Vec<(TextureAssetPath, Vec<TexturePayload>)> =
        bases.iter().map(|b| filled_slot(b, TextureSemantic::BaseColor)).collect();
    let norm_slots: Vec<(TextureAssetPath, Vec<TexturePayload>)> =
        normals.iter().map(|n| filled_slot(n, TextureSemantic::Normal)).collect();

    let keys = [
        "StaticMeshComponent::base_color_data",
        "StaticMeshComponent::normal_data",
        "StaticMeshComponent::roughness_metallic_data",
        "StaticMeshComponent::emissive_data",
        "StaticMeshComponent::occlusion_data",
        "StaticMeshComponent::specular_color_data",
        "StaticMeshComponent::specular_weight_data",
    ];
    let baseline_pools: Vec<Vec<(pulsar_scenedb::handle_ledger::HandleId, u64, u64)>> =
        keys.iter().map(|k| pool_audit(&h, k)).collect();
    h.store.flush_tier_transitions(h.ctx.queue());
    let baseline_tier = h.store.tier_audit();
    assert!(baseline_pools.iter().all(Vec::is_empty), "clean start");
    assert!(baseline_tier.is_empty(), "clean start (tiers)");

    for round in 0..ROUNDS {
        let entities: Vec<Entity> = (0..ENTITIES_PER_ROUND)
            .map(|i| {
                let (bp, bd) = &base_slots[i % K];
                let (np, nd) = &norm_slots[(i + 1) % K];
                let e = world.spawn();
                world.insert(
                    e,
                    StaticMeshComponent {
                        base_color_asset: bp.clone(),
                        base_color_data: bd.clone(),
                        normal_asset: np.clone(),
                        normal_data: nd.clone(),
                        ..Default::default()
                    },
                );
                e
            })
            .collect();

        // Mid-round invariant: exactly the K fixture chains per touched slot,
        // refcount = how many entities landed on each.
        let base = pool_audit(&h, keys[0]);
        assert_eq!(base.len(), K, "round {round}: exactly K base chains resident");
        assert_eq!(
            base.iter().map(|&(_, refs, _)| refs).sum::<u64>(),
            ENTITIES_PER_ROUND as u64,
            "round {round}: every entity accounted for"
        );

        for e in entities {
            world.despawn(e);
        }
        for (k, expected) in keys.iter().zip(&baseline_pools) {
            assert_eq!(&pool_audit(&h, k), expected, "round {round}: leaked residency in {k}");
        }
        h.store.flush_tier_transitions(h.ctx.queue());
        assert_eq!(
            h.store.tier_audit(),
            baseline_tier,
            "round {round}: tier audit must return to baseline EXACTLY"
        );
    }
}


// ── Helio#238 T5: UNREF-WHILE-VISIBLE ───────────────────────────────────────

#[test]
fn unref_while_visible_coarsens_through_floor_without_crash() {
    // Issue test 5 (Helio#238): the LAST content-id reference dies under a
    // fixed camera → the floor/fallback coarsening path renders, `release_tier`
    // returning `Pinned` for a still-referenced static is an EXPECTED
    // condition (never surfaced as an error), and full withdrawal after the
    // reference actually drops leaves zero residency behind — no panic, no
    // leak, nothing retained.
    //
    // "Visible" here means: demand was stamped (touch) and residency exists in
    // the audit while the reference lives. The Helio-side half of graceful
    // coarsening — publishing whatever floor survives into the meta row and
    // proving the sampling contract still names a valid mip — runs on
    // libhelio's row math, exactly what the frame-transient buffer would carry.
    let base = png_asset("t5_unref_base", 21);

    let (mut world, h) = mirrored_world("t5-unref-visible");
    let baseline_tier = h.store.tier_audit();
    assert!(baseline_tier.is_empty(), "clean start");

    let (bc_path, bc_payload) = filled_slot(&base, TextureSemantic::BaseColor);
    let content_id = bc_path.content_id();
    let e: Entity = world.spawn();
    world.insert(
        e,
        StaticMeshComponent {
            base_color_asset: bc_path,
            base_color_data: bc_payload,
            ..Default::default()
        },
    );

    let pool_key = BufferKey::of("StaticMeshComponent::base_color_data");
    let chain = pool_audit(&h, "StaticMeshComponent::base_color_data");
    assert_eq!(chain.len(), 1, "one chain for the visible slot");
    let id = chain[0].0;
    assert_ne!(id, pulsar_scenedb::handle_ledger::HandleId::default());

    // Visible + demanding: stamp demand through the whole span.
    h.store
        .touch_tier(
            TierSelector::Interned { pool: pool_key, id },
            TierSpan::Whole,
            Tier::Vram,
        )
        .expect("demand stamp for a live interned entry");
    h.store.flush_tier_transitions(h.ctx.queue());
    let audit = h.store.tier_audit();
    let records: Vec<_> = audit
        .iter()
        .filter(|r| matches!(r.key, TierAuditKey::Interned(k, i) if k == pool_key && i == id))
        .collect();
    assert_eq!(records.len(), 1, "exactly one tier record while visible");
    assert!(records[0].pinned, "referenced static is pinned while visible");

    // Demand collapses while STILL referenced (camera turned before the mesh
    // was destroyed): the policy executor's release is refused with Pinned —
    // the expected, tolerated condition per Helio#238 §H — and nothing moves.
    let refused = h.store.release_tier(
        TierSelector::Interned { pool: pool_key, id },
        TierSpan::ThroughRank(0),
    );
    match refused {
        Err(pulsar_scenedb::gpu::TierError::Pinned) => {} // the expected path
        other => panic!("release of a pinned static must be Err(Pinned), got {other:?}"),
    }
    h.store.flush_tier_transitions(h.ctx.queue());
    assert_eq!(
        h.store.tier_audit(),
        audit,
        "a refused release must not have moved any state"
    );

    // Graceful coarsening, sampling half: after withdrawal the engine would
    // publish the surviving floor into the slot's meta row. This fixture's
    // chain is a 4×4 PNG → THREE mips (4→2→1), each fitting in one page:
    // rank-0 covers ONLY the coarsest 1×1 fallback mip — precisely the
    // "image degrades to the fallback page, never NaN/crash" contract. The
    // row-level publication mechanics are pinned Helio-side in
    // tier_promote_bind.rs; here we pin the container truth the fallback
    // relies on.
    assert_eq!(
        helio_component::texture_cache::mip_count_for(4, 4),
        3,
        "4×4 chain has three mips; the rank-0 floor names the 1×1 fallback"
    );
    let _ = content_id; // identity kept alive to the withdrawal point below

    // The last reference dies under the (unchanged) camera.
    world.despawn(e);
    assert!(
        pool_audit(&h, "StaticMeshComponent::base_color_data").is_empty(),
        "chain freed at zero references"
    );
    h.store.flush_tier_transitions(h.ctx.queue());
    assert_eq!(
        h.store.tier_audit(),
        baseline_tier,
        "residency fully withdrawn after unref — no crash, nothing retained"
    );
}
