//! Texture benchmarks (Helio#237): import cost per 1K/4K texture; memoized
//! vs cold content-id lookup; hydrate of 100 entities sharing K textures
//! asserting exactly K decodes. Matches `content_dedup_benchmarks.rs`'s
//! `Instant`-timed idiom.
//!
//! Unlike the mesh suite's BM3 (which discloses that per-entity CPU loads are
//! NOT cached), textures DO memoize the decoded body per content id — so BM3'
//! below asserts the stronger property its issue text demands: exactly one
//! real decode per unique asset, no matter how many entities share it.

#![cfg(feature = "gpu")]

use helio_component::components::{StaticMeshComponent, TextureAssetPath};
use helio_component::texture_cache::{self, TexturePayload, TextureSemantic};
use image::ImageEncoder as _;
use pulsar_scenedb::gpu::{
    BufferKey, EngineGpuContext, GpuMirrorHandle, RegionClassConfig, SceneGpuConfig, SceneGpuStore,
};
use pulsar_scenedb::World;
use std::sync::{Arc, Once};
use std::time::Instant;

fn test_context() -> EngineGpuContext {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .expect("no adapter — GPU tests need a local GPU");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("texture-benchmarks"),
        ..Default::default()
    }))
    .expect("device");
    EngineGpuContext::new(Arc::new(device), Arc::new(queue))
}

static INIT: Once = Once::new();
fn project_root() -> std::path::PathBuf {
    let root = std::env::temp_dir().join("pulsar_texture_bench_root");
    INIT.call_once(|| {
        std::fs::create_dir_all(&root).expect("create bench project root");
        engine_state::EngineContext::new().set_global();
        engine_state::set_project_path(root.to_string_lossy().into_owned());
    });
    root
}

fn png_bytes(width: u32, height: u32, seed: u8) -> Vec<u8> {
    let mut rgba = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            rgba.push(((x ^ y).wrapping_add(seed as u32)) as u8);
            rgba.push((y as u8).wrapping_mul(3).wrapping_add(seed));
            rgba.push((x as u8).wrapping_mul(5).wrapping_add(seed));
            rgba.push(255);
        }
    }
    let mut png = Vec::new();
    image::codecs::png::PngEncoder::new(&mut png)
        .write_image(&rgba, width, height, image::ExtendedColorType::Rgba8)
        .unwrap();
    png
}

/// BM1: full pipeline import cost (decode → gamma-correct mipgen → BC7 encode
/// → container write) at AAA-common sizes.
#[test]
fn bm1_import_cost_per_texture() {
    texture_cache::clear_decoded_payload_cache();
    let root = project_root();

    for (label, dim) in [("1K", 1024u32), ("4K", 4096u32)] {
        let png = png_bytes(dim, dim, 7);
        let src = root.join(format!("bm1_source_{label}.png"));
        std::fs::write(&src, &png).unwrap();
        // Warm the page cache once so we measure conversion, not disk weather.
        let _ = std::fs::read(&src).unwrap().len();

        let start = Instant::now();
        let native = texture_cache::import_texture_to_native_default(&src, &root)
            .unwrap_or_else(|e| panic!("import failed: {e}"));
        let elapsed = start.elapsed();
        let native_len = std::fs::metadata(&native).expect("native written").len();
        eprintln!(
            "BM1[{label}]: {dim}x{dim} import {elapsed:?} ({:.1} ms), .ptex size {:.2} MiB",
            elapsed.as_secs_f64() * 1000.0,
            native_len as f64 / (1024.0 * 1024.0)
        );
    }
}

/// BM2: memoized content-id lookup vs a cold hash of the same file.
#[test]
fn bm2_memoized_content_id_lookup_vs_cold_hash() {
    texture_cache::clear_decoded_payload_cache();
    let root = project_root();
    let path = root.join("bm2_asset.png");
    std::fs::write(&path, png_bytes(256, 256, 9)).unwrap();

    let cold_start = Instant::now();
    let id_cold = texture_cache::content_id_for_path(&path).unwrap();
    let cold_elapsed = cold_start.elapsed();

    const WARM_CALLS: u32 = 10_000;
    let warm_start = Instant::now();
    for _ in 0..WARM_CALLS {
        let id_warm = texture_cache::content_id_for_path(&path).unwrap();
        assert_eq!(id_warm, id_cold);
    }
    let warm_elapsed = warm_start.elapsed();
    let warm_ns_per_call = warm_elapsed.as_nanos() as f64 / WARM_CALLS as f64;

    eprintln!(
        "BM2: cold lookup {cold_elapsed:?}; {WARM_CALLS} warm (memoized) lookups {warm_elapsed:?} ({warm_ns_per_call:.0} ns/call)"
    );
    assert!(
        warm_ns_per_call * 10.0 < cold_elapsed.as_nanos() as f64,
        "warm memoized lookups should be ≥10x cheaper than the cold hash (cold {cold_elapsed:?}, warm {warm_ns_per_call:.0}ns)"
    );
}

/// BM3': hydrate-shaped load+insert of 100 entities sharing K=3 textures —
/// exactly K decodes TOTAL (one per unique asset), and exactly K interned
/// chains at refcount 100 afterwards.
#[test]
fn bm3_hydrate_100_entities_sharing_k_textures_assert_exactly_k_decodes() {
    texture_cache::clear_decoded_payload_cache();
    let root = project_root();

    const K: usize = 3;
    let assets: Vec<String> = (0..K)
        .map(|k| {
            let rel = format!("bm3_shared_{k}.png");
            std::fs::write(root.join(&rel), png_bytes(64, 64, 40 + k as u8)).unwrap();
            rel
        })
        .collect();

    let ctx = test_context();
    let mut store = SceneGpuStore::new(
        &ctx,
        SceneGpuConfig { classes: vec![RegionClassConfig { capacity: 64, max_resident_cells: 1 }], tombstone_headroom: 8, max_cells_metadata: 16 },
    );
    StaticMeshComponent::register_gpu_columns_growable(&mut store, 512, ctx.device());
    let store = Arc::new(store);
    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));

    texture_cache::reset_decode_count();
    const N: usize = 100;
    let start = Instant::now();
    for i in 0..N {
        let rel = &assets[i % K];
        // The hydrate shape: resolve + decoded-body fetch + insert.
        let body = texture_cache::decoded_body_for_path(&root.join(rel), TextureSemantic::BaseColor)
            .expect("bench fixture decodes");
        let e = world.spawn();
        world.insert(
            e,
            StaticMeshComponent {
                base_color_asset: TextureAssetPath::new(rel.clone()),
                base_color_data: body.iter().copied().map(TexturePayload).collect(),
                ..Default::default()
            },
        );
    }
    let elapsed = start.elapsed();

    let decodes = texture_cache::decode_count();
    eprintln!(
        "BM3': hydrated+inserted {N} entities sharing {K} textures in {elapsed:?} ({:.1} us/entity); real decodes: {decodes}",
        elapsed.as_micros() as f64 / N as f64
    );
    assert_eq!(decodes, K as u64, "exactly ONE decode per UNIQUE texture, not per entity");

    let pool = store
        .interned_var_len_pool::<TexturePayload>(BufferKey::of("StaticMeshComponent::base_color_data"))
        .unwrap();
    let audit = pool.audit();
    assert_eq!(audit.len(), K, "{K} shared chains total");
    // Round-robin assignment splits 100 entities across 3 chains: each chain
    // holds ⌊N/K⌋ or ⌈N/K⌉ references, and they must sum to N.
    let (lo, hi) = ((N / K) as u64, (N.div_ceil(K)) as u64);
    assert!(
        audit.iter().all(|&(_, refs, _)| (lo..=hi).contains(&refs)),
        "each chain held by its round-robin share of entities"
    );
    assert_eq!(audit.iter().map(|&(_, refs, _)| refs).sum::<u64>(), N as u64);
}

