use nebula_audio::{
    AcousticConfig, AcousticOutput, ImpulseResponse, ReverbZone, CHUNK_TAG,
    config::{FREQ_BAND_COUNT, FREQ_BAND_CENTRES},
};
use nebula_core::traits::BakeOutput;

// ── Constants ─────────────────────────────────────────────────────────────────

#[test]
fn freq_band_count_is_8() {
    assert_eq!(FREQ_BAND_COUNT, 8);
}

#[test]
fn freq_band_centres_has_8_entries() {
    assert_eq!(FREQ_BAND_CENTRES.len(), 8);
}

#[test]
fn freq_band_centres_are_monotonically_increasing() {
    for pair in FREQ_BAND_CENTRES.windows(2) {
        assert!(pair[0] < pair[1], "band centres must be strictly increasing");
    }
}

#[test]
fn freq_band_centres_start_at_62_5_hz() {
    assert!((FREQ_BAND_CENTRES[0] - 62.5).abs() < f32::EPSILON);
}

#[test]
fn freq_band_centres_end_at_8000_hz() {
    assert!((FREQ_BAND_CENTRES[7] - 8000.0).abs() < f32::EPSILON);
}

#[test]
fn chunk_tag_is_auir() {
    assert_eq!(CHUNK_TAG.to_bytes(), *b"AUIR");
}

// ── AcousticConfig defaults ───────────────────────────────────────────────────

#[test]
fn acoustic_config_default_diffuse_rays_is_512() {
    assert_eq!(AcousticConfig::default().diffuse_rays, 512);
}

#[test]
fn acoustic_config_default_max_order_is_2() {
    assert_eq!(AcousticConfig::default().max_order, 2);
}

#[test]
fn acoustic_config_default_emit_reverb_zone_is_true() {
    assert!(AcousticConfig::default().emit_reverb_zone);
}

#[test]
fn acoustic_config_default_listener_points_is_empty() {
    assert!(AcousticConfig::default().listener_points.is_empty());
}

#[test]
fn acoustic_config_default_air_absorption_has_8_bands() {
    assert_eq!(AcousticConfig::default().air_absorption.len(), FREQ_BAND_COUNT);
}

#[test]
fn acoustic_config_default_air_absorption_is_positive() {
    for &a in &AcousticConfig::default().air_absorption {
        assert!(a > 0.0, "air absorption must be positive");
    }
}

#[test]
fn acoustic_config_default_air_absorption_increases_with_frequency() {
    let abs = AcousticConfig::default().air_absorption;
    for pair in abs.windows(2) {
        assert!(pair[0] <= pair[1], "air absorption should increase with frequency");
    }
}

// ── Fast preset ───────────────────────────────────────────────────────────────

#[test]
fn acoustic_config_fast_diffuse_rays_is_64() {
    assert_eq!(AcousticConfig::fast().diffuse_rays, 64);
}

#[test]
fn acoustic_config_fast_max_order_is_1() {
    assert_eq!(AcousticConfig::fast().max_order, 1);
}

#[test]
fn acoustic_config_fast_max_duration_is_0_5() {
    assert!((AcousticConfig::fast().max_duration_secs - 0.5).abs() < f32::EPSILON);
}

// ── Ultra preset ──────────────────────────────────────────────────────────────

#[test]
fn acoustic_config_ultra_diffuse_rays_is_8192() {
    assert_eq!(AcousticConfig::ultra().diffuse_rays, 8192);
}

#[test]
fn acoustic_config_ultra_max_order_is_5() {
    assert_eq!(AcousticConfig::ultra().max_order, 5);
}

// ── Preset ordering ───────────────────────────────────────────────────────────

#[test]
fn acoustic_config_preset_ray_count_ordering() {
    assert!(AcousticConfig::fast().diffuse_rays < AcousticConfig::default().diffuse_rays);
    assert!(AcousticConfig::default().diffuse_rays < AcousticConfig::ultra().diffuse_rays);
}

// ── AcousticOutput trait ──────────────────────────────────────────────────────

#[test]
fn acoustic_output_kind_name_is_acoustic() {
    assert_eq!(AcousticOutput::kind_name(), "acoustic");
}

// ── Serialize / deserialize round-trip ───────────────────────────────────────

fn make_minimal_output() -> AcousticOutput {
    AcousticOutput {
        impulse_responses: vec![],
        reverb_zones: vec![],
        config_json: "{}".to_string(),
    }
}

#[test]
fn acoustic_output_serialize_deserialize_empty() {
    let out = make_minimal_output();
    let bytes = out.serialize_to_bytes().expect("serialize");
    let back = AcousticOutput::deserialize_from_bytes(&bytes).expect("deserialize");
    assert_eq!(back.impulse_responses.len(), 0);
    assert_eq!(back.reverb_zones.len(), 0);
    assert_eq!(back.config_json, "{}");
}

#[test]
fn acoustic_output_serialize_produces_non_empty_bytes() {
    let out = make_minimal_output();
    let bytes = out.serialize_to_bytes().expect("serialize");
    assert!(!bytes.is_empty());
}

#[test]
fn acoustic_output_deserialize_corrupt_returns_error() {
    let result = AcousticOutput::deserialize_from_bytes(b"\xFF\xFF\xFF\xFF");
    assert!(result.is_err());
}

#[test]
fn acoustic_output_with_impulse_response_roundtrip() {
    let ir = ImpulseResponse {
        listener_position: [1.0, 2.0, 3.0],
        sample_rate: 44100,
        bands: std::array::from_fn(|i| vec![0.0_f32; i + 1]),
        t60_per_band: [0.5; FREQ_BAND_COUNT],
        broadband_t60: 0.5,
        early_late_split_secs: 0.08,
    };
    let out = AcousticOutput {
        impulse_responses: vec![ir],
        reverb_zones: vec![],
        config_json: "{}".to_string(),
    };
    let bytes = out.serialize_to_bytes().expect("serialize");
    let back = AcousticOutput::deserialize_from_bytes(&bytes).expect("deserialize");
    assert_eq!(back.impulse_responses.len(), 1);
    let back_ir = &back.impulse_responses[0];
    assert_eq!(back_ir.listener_position, [1.0_f32, 2.0, 3.0]);
    assert_eq!(back_ir.sample_rate, 44100);
    assert_eq!(back_ir.broadband_t60, 0.5);
}

#[test]
fn acoustic_output_reverb_zone_roundtrip() {
    let rz = ReverbZone {
        aabb_min: [0.0; 3],
        aabb_max: [10.0; 3],
        t60: 1.2,
        edt: 0.3,
        c80: 3.0,
        d50: 0.6,
        room_gain_db: 2.0,
        drr_db: 5.0,
        absorption: [0.1; FREQ_BAND_COUNT],
    };
    let out = AcousticOutput {
        impulse_responses: vec![],
        reverb_zones: vec![rz],
        config_json: "{}".to_string(),
    };
    let bytes = out.serialize_to_bytes().expect("serialize");
    let back = AcousticOutput::deserialize_from_bytes(&bytes).expect("deserialize");
    assert_eq!(back.reverb_zones.len(), 1);
    assert!((back.reverb_zones[0].t60 - 1.2).abs() < 1e-5);
}

// ── Serde (JSON) round-trip ───────────────────────────────────────────────────

#[test]
fn acoustic_config_json_roundtrip() {
    let cfg = AcousticConfig::ultra();
    let json = serde_json::to_string(&cfg).expect("serialize");
    let back: AcousticConfig = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back.diffuse_rays, cfg.diffuse_rays);
    assert_eq!(back.max_order, cfg.max_order);
}

// ── Baker (GPU-gated) ─────────────────────────────────────────────────────────

#[test]
#[ignore = "requires GPU adapter"]
fn acoustic_baker_produces_output_for_empty_scene() {
    use nebula_core::{context::BakeContext, progress::NullReporter, scene::SceneGeometry};
    use nebula_audio::AcousticBaker;
    use nebula_core::traits::BakePass;

    pollster::block_on(async {
        let ctx = BakeContext::new().await.expect("BakeContext");
        let scene = SceneGeometry::default();
        let cfg = AcousticConfig::fast();
        let out = AcousticBaker.execute(&scene, &cfg, &ctx, &NullReporter).await
            .expect("bake");
        // No listener points configured → impulse_responses should be empty.
        assert!(out.impulse_responses.is_empty());
    });
}
