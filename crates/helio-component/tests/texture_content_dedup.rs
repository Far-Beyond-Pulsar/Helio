//! Texture content-dedup adversarial coverage (Helio#237 issue tests 2 and 3)
//! on the real `TextureAssetPath` production type — the texture twin of
//! `content_dedup.rs`'s `MeshAssetPath` chain:
//!
//! - T2: path aliasing convergence — `t.png` vs a redundant-spelling alias vs
//!   an identical-bytes copy all resolve to ONE content id.
//! - T3: file mutation between hydrates mints a NEW id and the OLD chain's
//!   refcount falls as its last holder goes away.
//!
//! Shares `content_dedup.rs`'s project-root discipline: ONE process-global
//! root set exactly once (`INIT`), each test writing uniquely-named files, so
//! concurrent test threads never race.

#![cfg(feature = "gpu")]

use helio_component::components::{StaticMeshComponent, TextureAssetPath};
use helio_component::texture_cache::{self, TextureSemantic};
use image::ImageEncoder as _;
use pulsar_scenedb::handle_ledger::ContentAddressed as _;
use pulsar_scenedb::gpu::{
    BufferKey, EngineGpuContext, GpuMirrorHandle, RegionClassConfig, SceneGpuConfig, SceneGpuStore,
};
use pulsar_scenedb::{Entity, World};
use std::sync::{Arc, Once};

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
        label: Some("texture-content-dedup"),
        ..Default::default()
    }))
    .expect("device");
    EngineGpuContext::new(Arc::new(device), Arc::new(queue))
}

fn scene_cfg() -> SceneGpuConfig {
    SceneGpuConfig { classes: vec![RegionClassConfig { capacity: 64, max_resident_cells: 1 }], tombstone_headroom: 8, max_cells_metadata: 16 }
}

static INIT: Once = Once::new();

fn project_root() -> std::path::PathBuf {
    let root = std::env::temp_dir().join("pulsar_texture_dedup_test_root");
    INIT.call_once(|| {
        std::fs::create_dir_all(&root).expect("create test project root");
        engine_state::EngineContext::new().set_global();
        engine_state::set_project_path(root.to_string_lossy().into_owned());
    });
    root
}

/// Encode a small deterministic rgba buffer as PNG bytes.
fn png_bytes(width: u32, height: u32, seed: u8) -> Vec<u8> {
    let mut rgba = Vec::with_capacity((width * height * 4) as usize);
    for i in 0..(width * height) as usize {
        rgba.extend_from_slice(&[seed.wrapping_add(i as u8), seed ^ 0x5A, seed.wrapping_mul(7), 255]);
    }
    let mut png = Vec::new();
    image::codecs::png::PngEncoder::new(&mut png)
        .write_image(&rgba, width, height, image::ExtendedColorType::Rgba8)
        .expect("encode fixture png");
    png
}

/// Write a unique source-image asset under the shared root; returns the
/// project-relative path shape the wrapper fields store.
fn write_png_asset(name: &str, seed: u8) -> String {
    let rel = format!("{name}.png");
    std::fs::write(project_root().join(&rel), png_bytes(4, 4, seed)).expect("write fixture");
    rel
}

fn base_color_pool(store: &SceneGpuStore) -> Arc<pulsar_scenedb::gpu::InternedVarLenPool<helio_component::texture_cache::TexturePayload>> {
    store
        .interned_var_len_pool::<helio_component::texture_cache::TexturePayload>(BufferKey::of(
            "StaticMeshComponent::base_color_data",
        ))
        .expect("registered on first insert")
}

fn spawn_with_texture(world: &mut World, rel: &str) -> Entity {
    // Mirror `hydrate_static_mesh_component`: resolve+decode FIRST, insert
    // with the payload populated (an empty Vec wouldn't exercise interning).
    let abs = project_root().join(rel);
    let body = helio_component::texture_cache::decoded_body_for_path(&abs, TextureSemantic::BaseColor)
        .expect("fixture texture decodes");
    let e = world.spawn();
    world.insert(
        e,
        StaticMeshComponent {
            base_color_asset: TextureAssetPath::new(rel),
            base_color_data: body.iter().copied().map(helio_component::texture_cache::TexturePayload).collect(),
            ..Default::default()
        },
    );
    e
}

// ── T2: aliasing convergence ────────────────────────────────────────────────

#[test]
fn path_aliases_and_identical_copies_converge_on_one_id() {
    let root = project_root();
    let rel = write_png_asset("aliasing_target", 1);

    // Redundant spellings of the SAME file canonicalize to one entry.
    let plain = root.join(&rel);
    let redundant = root.join("./sub/..").join(&rel);
    std::fs::create_dir_all(root.join("sub")).expect("subdir for the alias spelling");

    let id_plain = texture_cache::content_id_for_path(&plain).expect("plain resolves");
    let id_redundant = texture_cache::content_id_for_path(&redundant).expect("alias resolves");
    assert_eq!(id_plain, id_redundant, "`./sub/../<name>` must converge on `<name>`");

    // An identical-BYTES copy is the same content by construction.
    std::fs::copy(root.join(&rel), root.join("aliasing_copy.png")).expect("copy fixture");
    let id_copy = texture_cache::content_id_for_path(&root.join("aliasing_copy.png")).unwrap();
    assert_eq!(id_copy, id_plain, "byte-identical copies share one content id");

    // And through the wrapper's own seam — same resolution chain.
    let via_wrapper = TextureAssetPath::new(&rel).content_id();
    let zero = TextureAssetPath::new("").content_id();
    assert_eq!(via_wrapper.0, id_plain);
    assert_eq!(zero, pulsar_scenedb::handle_ledger::HandleId::ZERO, "empty path opts out");
}

// ── T3: mutation → new id, old refcount falls ───────────────────────────────

#[test]
fn file_mutation_between_hydrates_mints_a_new_id_and_the_old_refcount_falls() {
    let root = project_root();
    let rel = "mutating_texture.png";
    std::fs::write(root.join(rel), png_bytes(4, 4, 42)).unwrap();

    let ctx = test_context();
    let mut store = SceneGpuStore::new(&ctx, scene_cfg());
    StaticMeshComponent::register_gpu_columns_growable(&mut store, 16, ctx.device());
    let store = Arc::new(store);
    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));

    let pool = || base_color_pool(&store);

    let e1 = spawn_with_texture(&mut world, rel);
    assert_eq!(pool().audit().len(), 1, "first hydrate interns exactly one chain");
    let old_id = pool().audit()[0].0;

    // Mutate the FILE (different bytes ⇒ different id).
    std::thread::sleep(std::time::Duration::from_millis(10)); // coarse-mtime safety
    std::fs::write(root.join(rel), png_bytes(4, 4, 137)).unwrap();

    // A second entity referencing the same PATH now lands on the NEW id while
    // e1 still holds the OLD chain — both coexist.
    let e2 = spawn_with_texture(&mut world, rel);
    let audit = pool().audit();
    assert_eq!(audit.len(), 2, "old and new chains coexist while both referenced");
    assert!(
        audit.iter().any(|&(id, refs, _)| id == old_id && refs == 1),
        "e1 still holds the OLD id at refcount 1"
    );

    // Removal deltas: dropping e1 frees the OLD chain; dropping e2 empties.
    world.despawn(e1);
    let audit = pool().audit();
    assert_eq!(audit.len(), 1, "old chain freed with its last reference");
    assert_ne!(audit[0].0, old_id, "the survivor is the NEW chain");
    assert_eq!(audit[0].1, 1);

    world.despawn(e2);
    assert!(pool().audit().is_empty(), "no residency survives both despawns");
}

