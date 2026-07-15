use nebula_light::{LightmapConfig, LightmapOutput, AtlasRegion, CHUNK_TAG};
use nebula_core::traits::BakeOutput;

// ── LightmapConfig defaults ───────────────────────────────────────────────────

#[test]
fn lightmap_config_default_resolution_is_1024() {
    assert_eq!(LightmapConfig::default().resolution, 1024);
}

#[test]
fn lightmap_config_default_samples_per_texel_is_64() {
    assert_eq!(LightmapConfig::default().samples_per_texel, 64);
}

#[test]
fn lightmap_config_default_bounce_count_is_2() {
    assert_eq!(LightmapConfig::default().bounce_count, 2);
}

#[test]
fn lightmap_config_default_hdr_output_is_true() {
    assert!(LightmapConfig::default().hdr_output);
}

#[test]
fn lightmap_config_default_denoise_is_true() {
    assert!(LightmapConfig::default().denoise);
}

#[test]
fn lightmap_config_default_area_light_samples_is_16() {
    assert_eq!(LightmapConfig::default().area_light_samples, 16);
}

#[test]
fn lightmap_config_default_debug_normals_is_false() {
    assert!(!LightmapConfig::default().debug_normals);
}

// ── Fast preset ───────────────────────────────────────────────────────────────

#[test]
fn lightmap_config_fast_resolution_is_512() {
    assert_eq!(LightmapConfig::fast().resolution, 512);
}

#[test]
fn lightmap_config_fast_samples_per_texel_is_8() {
    assert_eq!(LightmapConfig::fast().samples_per_texel, 8);
}

#[test]
fn lightmap_config_fast_bounce_count_is_1() {
    assert_eq!(LightmapConfig::fast().bounce_count, 1);
}

#[test]
fn lightmap_config_fast_denoise_is_false() {
    assert!(!LightmapConfig::fast().denoise);
}

// ── Ultra preset ──────────────────────────────────────────────────────────────

#[test]
fn lightmap_config_ultra_resolution_is_4096() {
    assert_eq!(LightmapConfig::ultra().resolution, 4096);
}

#[test]
fn lightmap_config_ultra_samples_is_512() {
    assert_eq!(LightmapConfig::ultra().samples_per_texel, 512);
}

#[test]
fn lightmap_config_ultra_bounce_count_is_4() {
    assert_eq!(LightmapConfig::ultra().bounce_count, 4);
}

// ── Preset ordering ───────────────────────────────────────────────────────────

#[test]
fn lightmap_config_resolution_ordering() {
    assert!(LightmapConfig::fast().resolution < LightmapConfig::default().resolution);
    assert!(LightmapConfig::default().resolution < LightmapConfig::ultra().resolution);
}

#[test]
fn lightmap_config_sample_ordering() {
    assert!(LightmapConfig::fast().samples_per_texel < LightmapConfig::default().samples_per_texel);
    assert!(LightmapConfig::default().samples_per_texel < LightmapConfig::ultra().samples_per_texel);
}

// ── CHUNK_TAG ──────────────────────────────────────────────────────────────────

#[test]
fn lightmap_chunk_tag_is_lmap() {
    assert_eq!(CHUNK_TAG.to_bytes(), *b"LMAP");
}

// ── LightmapOutput trait ──────────────────────────────────────────────────────

#[test]
fn lightmap_output_kind_name_is_lightmap() {
    assert_eq!(LightmapOutput::kind_name(), "lightmap");
}

// ── AtlasRegion ───────────────────────────────────────────────────────────────

#[test]
fn atlas_region_fields_accessible() {
    let region = AtlasRegion {
        mesh_id:   uuid::Uuid::new_v4(),
        uv_offset: [0.0, 0.0],
        uv_scale:  [1.0, 1.0],
    };
    assert_eq!(region.uv_offset, [0.0_f32, 0.0]);
    assert_eq!(region.uv_scale, [1.0_f32, 1.0]);
}

#[test]
fn atlas_region_non_overlapping_tiles() {
    // Two 50% tiles occupying left and right halves of the atlas.
    let left = AtlasRegion {
        mesh_id:   uuid::Uuid::new_v4(),
        uv_offset: [0.0, 0.0],
        uv_scale:  [0.5, 1.0],
    };
    let right = AtlasRegion {
        mesh_id:   uuid::Uuid::new_v4(),
        uv_offset: [0.5, 0.0],
        uv_scale:  [0.5, 1.0],
    };
    // Verify they don't share the same X origin.
    assert_ne!(left.uv_offset[0], right.uv_offset[0]);
    // Together they span the full atlas width.
    assert!((left.uv_scale[0] + right.uv_scale[0] - 1.0).abs() < f32::EPSILON);
}

// ── Serde round-trip ──────────────────────────────────────────────────────────

#[test]
fn lightmap_config_serde_roundtrip() {
    let cfg = LightmapConfig::ultra();
    let json = serde_json::to_string(&cfg).expect("serialize");
    let back: LightmapConfig = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back.resolution, cfg.resolution);
    assert_eq!(back.samples_per_texel, cfg.samples_per_texel);
    assert_eq!(back.bounce_count, cfg.bounce_count);
    assert_eq!(back.hdr_output, cfg.hdr_output);
}

#[test]
fn lightmap_output_serde_roundtrip() {
    let out = LightmapOutput {
        width: 2, height: 2,
        channels: 4,
        is_f32: true,
        texels: vec![0u8; 64],
        atlas_regions: vec![],
        config_json: "{}".to_string(),
    };
    let json = serde_json::to_string(&out).expect("serialize");
    let back: LightmapOutput = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back.width, 2);
    assert_eq!(back.channels, 4);
    assert_eq!(back.is_f32, true);
    assert_eq!(back.texels.len(), 64);
}

// ── Baker (GPU-gated) ─────────────────────────────────────────────────────────

#[test]
#[ignore = "requires GPU adapter"]
fn lightmap_baker_produces_output_for_simple_scene() {
    use nebula_core::{context::BakeContext, progress::NullReporter, scene::SceneGeometry};
    use nebula_light::LightmapBaker;
    use nebula_core::traits::BakePass;

    pollster::block_on(async {
        let ctx = BakeContext::new().await.expect("BakeContext");
        let scene = SceneGeometry::default();
        let cfg = LightmapConfig::fast();
        let out = LightmapBaker.execute(&scene, &cfg, &ctx, &NullReporter).await
            .expect("bake");
        assert_eq!(out.width, cfg.resolution);
        assert_eq!(out.height, cfg.resolution);
    });
}
