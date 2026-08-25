//! T2 - PROMOTE-BEFORE-BIND ordering (Helio#238 issue test 2).
//!
//! Frame N's policy touches are committed by flush_tier_transitions BEFORE
//! frame N+1 builds bindings: residency observed via tier_peek/tier_audit
//! between frames, then floors published into meta rows whose equality the
//! bind gate checks. Uses a LOCAL payload type whose segment layout mirrors
//! the canonical texture-payload one (rank r == page r, 16 KiB), so this test
//! needs no cross-workspace dependency on helio-component.
//!
//! GPU used only for the store; headless machines skip gracefully.

use std::sync::{Arc, Once};

use libhelio::{finest_fully_resident_mip, GpuVtMetaRow};
use pulsar_scenedb::gpu::{
    register_segment_layout, BufferKey, EngineGpuContext, RegionClassConfig, SceneGpuConfig,
    SceneGpuStore, Segment, Tier, TierConfig, TierPeek, TierSelector, TierSpan,
};
use pulsar_scenedb::gpu::VarLenHandle;

const PAGE_BYTES: u64 = 16 * 1024;
const RANKS: u64 = 4;

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct T2Payload(u8);

// SAFETY: all-zero bytes valid (u8 newtype), no Drop - mirrors the canonical
// TexturePayload contract.
unsafe impl pulsar_scenedb::Pod for T2Payload {}

fn register_layout_once() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let segments: Vec<Segment> = (0..RANKS)
            .map(|rank| Segment::new(rank * PAGE_BYTES, PAGE_BYTES, rank as u32))
            .collect();
        register_segment_layout::<T2Payload>(&segments).expect("fixture layout registers");
    });
}

fn device_or_skip(label: &str) -> Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    })) {
        Ok(a) => a,
        Err(_) => {
            eprintln!("skipping: no GPU adapter (headless)");
            return None;
        }
    };
    match pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some(label),
        ..Default::default()
    })) {
        Ok((d, q)) => Some((Arc::new(d), Arc::new(q))),
        Err(_) => {
            eprintln!("skipping: no device");
            None
        }
    }
}

#[test]
fn touch_from_frame_n_is_resident_before_frame_n_plus_1_binds() {
    let Some((device, queue)) = device_or_skip("T2 promote-before-bind") else {
        return;
    };
    register_layout_once();

    let ctx = EngineGpuContext::new(Arc::clone(&device), Arc::clone(&queue));
    let gpu_cfg = SceneGpuConfig {
        classes: vec![RegionClassConfig { capacity: 16, max_resident_cells: 4 }],
        tombstone_headroom: 2,
        max_cells_metadata: 8,
    };
    let store = SceneGpuStore::new(&ctx, gpu_cfg);
    // THE one consumer configuration call (engine seam equivalent).
    store
        .configure_tiers(TierConfig { vram_budget_bytes: 64 << 20, ram_budget_bytes: 64 << 20 }, &[])
        .expect("configure");

    let pool = store.register_var_len_gpu_pool::<T2Payload>(BufferKey::of("T2TexturePayload"), 64, &device);
    let data: Vec<T2Payload> = vec![T2Payload(0xAB); (RANKS * PAGE_BYTES) as usize];
    let handle: VarLenHandle = pool
        .write_var_row(&queue, VarLenHandle::default(), &data)
        .expect("payload write");
    assert!(handle.count > 0);

    let selector = TierSelector::PoolSlot { pool: BufferKey::of("T2TexturePayload"), handle };

    // Frame N: policy demands rank 2 (absolute demand up)...
    store
        .touch_tier(selector, TierSpan::ThroughRank(2), Tier::Vram)
        .expect("touch queues");
    // ...and THE FLUSH IS THE FRAME BOUNDARY: executed before any bind work.
    let stats = store.flush_tier_transitions(&queue);
    assert!(stats.promoted > 0, "promotion flight executed at flush");

    // Between frames: residency observable without blocking the render path.
    assert!(matches!(
        store.tier_peek(selector).expect("peek"),
        TierPeek::Resident { .. }
    ));

    // Frame N+1 binding CONTENT: publish floors into meta rows; the row must
    // name exactly what rank prefix 2 proves (computed floor, not assumed).
    let mut row = GpuVtMetaRow::for_asset(1024, 1024, 11, 7, false, 16);
    row.set_resident_through(2);
    let expected_floor = finest_fully_resident_mip(&row, 2);
    assert_eq!(row.floor_flags[1], expected_floor);
    assert!(
        row.floor_flags[1] >= 7,
        "rank 2 covers every mip coarser than index 7 (coarse-first closure)"
    );

    // Rebuild determinism: same inputs, byte-equal rows - the gate T4 leans on.
    let mut row_again = GpuVtMetaRow::for_asset(1024, 1024, 11, 7, false, 16);
    row_again.set_resident_through(2);
    assert_eq!(row, row_again);
}