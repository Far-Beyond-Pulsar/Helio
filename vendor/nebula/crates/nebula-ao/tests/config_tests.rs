use nebula_ao::{AoConfig, AoOutput};
use nebula_core::traits::BakeOutput;

// ── AoConfig defaults ─────────────────────────────────────────────────────────

#[test]
fn ao_config_default_resolution_is_1024() {
    assert_eq!(AoConfig::default().resolution, 1024);
}

#[test]
fn ao_config_default_ray_count_is_128() {
    assert_eq!(AoConfig::default().ray_count, 128);
}

#[test]
fn ao_config_default_max_distance_is_10() {
    assert!((AoConfig::default().max_distance - 10.0).abs() < f32::EPSILON);
}

#[test]
fn ao_config_default_bias_is_small() {
    let cfg = AoConfig::default();
    assert!(cfg.bias > 0.0 && cfg.bias < 0.01);
}

#[test]
fn ao_config_default_denoise_is_true() {
    assert!(AoConfig::default().denoise);
}

// ── Fast preset ───────────────────────────────────────────────────────────────

#[test]
fn ao_config_fast_resolution_is_512() {
    assert_eq!(AoConfig::fast().resolution, 512);
}

#[test]
fn ao_config_fast_ray_count_is_16() {
    assert_eq!(AoConfig::fast().ray_count, 16);
}

#[test]
fn ao_config_fast_denoise_is_false() {
    assert!(!AoConfig::fast().denoise);
}

// ── Ultra preset ──────────────────────────────────────────────────────────────

#[test]
fn ao_config_ultra_resolution_is_4096() {
    assert_eq!(AoConfig::ultra().resolution, 4096);
}

#[test]
fn ao_config_ultra_ray_count_is_512() {
    assert_eq!(AoConfig::ultra().ray_count, 512);
}

#[test]
fn ao_config_ultra_denoise_is_true() {
    // ultra inherits default denoise = true
    assert!(AoConfig::ultra().denoise);
}

// ── Preset hierarchy ──────────────────────────────────────────────────────────

#[test]
fn ao_config_presets_resolution_ordering() {
    assert!(AoConfig::fast().resolution < AoConfig::default().resolution);
    assert!(AoConfig::default().resolution < AoConfig::ultra().resolution);
}

#[test]
fn ao_config_presets_ray_count_ordering() {
    assert!(AoConfig::fast().ray_count < AoConfig::default().ray_count);
    assert!(AoConfig::default().ray_count < AoConfig::ultra().ray_count);
}

// ── AoOutput ──────────────────────────────────────────────────────────────────

#[test]
fn ao_output_kind_name_is_ao() {
    assert_eq!(AoOutput::kind_name(), "ao");
}

#[test]
fn ao_output_fields_survive_clone() {
    let out = AoOutput {
        width: 4, height: 4,
        texels: vec![0u8; 4 * 4 * 4],
        config_json: "{}".to_string(),
    };
    let cloned = out.clone();
    assert_eq!(cloned.width, 4);
    assert_eq!(cloned.height, 4);
    assert_eq!(cloned.texels.len(), 64);
}

// ── Serde round-trip ──────────────────────────────────────────────────────────

#[test]
fn ao_config_serde_roundtrip() {
    let cfg = AoConfig::ultra();
    let json = serde_json::to_string(&cfg).expect("serialize");
    let back: AoConfig = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back.resolution, cfg.resolution);
    assert_eq!(back.ray_count, cfg.ray_count);
    assert_eq!(back.denoise, cfg.denoise);
}

#[test]
fn ao_output_serde_roundtrip() {
    let out = AoOutput {
        width: 2, height: 2,
        texels: vec![0xFF, 0xFF, 0x00, 0x00],
        config_json: r#"{"resolution":512}"#.to_string(),
    };
    let json = serde_json::to_string(&out).expect("serialize");
    let back: AoOutput = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back.width, 2);
    assert_eq!(back.height, 2);
    assert_eq!(back.texels, out.texels);
    assert_eq!(back.config_json, out.config_json);
}

// ── Baker (GPU-gated) ─────────────────────────────────────────────────────────

#[test]
#[ignore = "requires GPU adapter"]
fn ao_baker_produces_output_for_simple_scene() {
    use nebula_core::{context::BakeContext, progress::NullReporter, scene::SceneGeometry};
    use nebula_ao::AoBaker;
    use nebula_core::traits::BakePass;

    pollster::block_on(async {
        let ctx = BakeContext::new().await.expect("BakeContext");
        let scene = SceneGeometry::default();
        let cfg = AoConfig::fast();
        let out = AoBaker.execute(&scene, &cfg, &ctx, &NullReporter).await
            .expect("bake");
        assert_eq!(out.width, cfg.resolution);
        assert_eq!(out.height, cfg.resolution);
    });
}
