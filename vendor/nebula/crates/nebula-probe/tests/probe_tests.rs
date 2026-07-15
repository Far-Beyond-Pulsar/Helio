use nebula_probe::{
    ProbeConfig, ReflectionOutput, IrradianceOutput, ShCoeff,
    REFLECTION_CHUNK_TAG, IRRADIANCE_CHUNK_TAG,
};
use nebula_core::traits::BakeOutput;

// ── Chunk tags ────────────────────────────────────────────────────────────────

#[test]
fn reflection_chunk_tag_is_rpro() {
    assert_eq!(REFLECTION_CHUNK_TAG.to_bytes(), *b"RPRO");
}

#[test]
fn irradiance_chunk_tag_is_irsh() {
    assert_eq!(IRRADIANCE_CHUNK_TAG.to_bytes(), *b"IRSH");
}

// ── ProbeConfig defaults ──────────────────────────────────────────────────────

#[test]
fn probe_config_default_face_resolution_is_256() {
    assert_eq!(ProbeConfig::default().face_resolution, 256);
}

#[test]
fn probe_config_default_sh_order_is_3() {
    assert_eq!(ProbeConfig::default().sh_order, 3);
}

#[test]
fn probe_config_default_samples_per_face_is_1024() {
    assert_eq!(ProbeConfig::default().samples_per_face, 1024);
}

#[test]
fn probe_config_default_specular_mip_levels_is_8() {
    assert_eq!(ProbeConfig::default().specular_mip_levels, 8);
}

#[test]
fn probe_config_default_exposure_is_1() {
    assert!((ProbeConfig::default().exposure - 1.0).abs() < f32::EPSILON);
}

#[test]
fn probe_config_default_use_rgbe_is_false() {
    assert!(!ProbeConfig::default().use_rgbe);
}

// ── Fast preset ───────────────────────────────────────────────────────────────

#[test]
fn probe_config_fast_face_resolution_is_64() {
    assert_eq!(ProbeConfig::fast().face_resolution, 64);
}

#[test]
fn probe_config_fast_specular_mip_levels_is_4() {
    assert_eq!(ProbeConfig::fast().specular_mip_levels, 4);
}

#[test]
fn probe_config_fast_samples_per_face_is_128() {
    assert_eq!(ProbeConfig::fast().samples_per_face, 128);
}

// ── Ultra preset ──────────────────────────────────────────────────────────────

#[test]
fn probe_config_ultra_face_resolution_is_512() {
    assert_eq!(ProbeConfig::ultra().face_resolution, 512);
}

#[test]
fn probe_config_ultra_samples_per_face_is_8192() {
    assert_eq!(ProbeConfig::ultra().samples_per_face, 8192);
}

// ── Preset ordering ───────────────────────────────────────────────────────────

#[test]
fn probe_config_face_resolution_ordering() {
    assert!(ProbeConfig::fast().face_resolution < ProbeConfig::default().face_resolution);
    assert!(ProbeConfig::default().face_resolution < ProbeConfig::ultra().face_resolution);
}

#[test]
fn probe_config_samples_ordering() {
    assert!(ProbeConfig::fast().samples_per_face < ProbeConfig::default().samples_per_face);
    assert!(ProbeConfig::default().samples_per_face < ProbeConfig::ultra().samples_per_face);
}

// ── SH coefficient count ──────────────────────────────────────────────────────

#[test]
fn sh_order_3_gives_16_coefficients() {
    // (sh_order + 1)^2 = (3 + 1)^2 = 16
    let order = ProbeConfig::default().sh_order;
    let expected_coeff_count = (order + 1).pow(2) as usize;
    assert_eq!(expected_coeff_count, 16);
}

#[test]
fn sh_order_2_gives_9_coefficients() {
    let order: u32 = 2;
    assert_eq!((order + 1).pow(2), 9);
}

#[test]
fn sh_order_1_gives_4_coefficients() {
    let order: u32 = 1;
    assert_eq!((order + 1).pow(2), 4);
}

// ── BakeOutput trait ──────────────────────────────────────────────────────────

#[test]
fn reflection_output_kind_name() {
    assert_eq!(ReflectionOutput::kind_name(), "reflection_probe");
}

#[test]
fn irradiance_output_kind_name() {
    assert_eq!(IrradianceOutput::kind_name(), "irradiance_probe");
}

// ── ReflectionOutput serialize/deserialize ────────────────────────────────────

fn make_reflection_output() -> ReflectionOutput {
    ReflectionOutput {
        face_resolution: 64,
        mip_levels: 1,
        is_rgbe: false,
        face_data: vec![0u8; 64 * 64 * 6 * 16], // 6 faces × 16 bytes/pixel (RGBA32F)
        config_json: "{}".to_string(),
    }
}

#[test]
fn reflection_output_serialize_deserialize_roundtrip() {
    let out = make_reflection_output();
    let bytes = out.serialize_to_bytes().expect("serialize");
    let back = ReflectionOutput::deserialize_from_bytes(&bytes).expect("deserialize");
    assert_eq!(back.face_resolution, 64);
    assert_eq!(back.mip_levels, 1);
    assert_eq!(back.is_rgbe, false);
    assert_eq!(back.face_data.len(), out.face_data.len());
}

#[test]
fn reflection_output_corrupt_bytes_returns_error() {
    let result = ReflectionOutput::deserialize_from_bytes(b"\xDE\xAD\xBE\xEF");
    assert!(result.is_err());
}

// ── IrradianceOutput serialize/deserialize ────────────────────────────────────

fn make_irradiance_output(order: u32) -> IrradianceOutput {
    let coeff_count = ((order + 1) * (order + 1)) as usize;
    IrradianceOutput {
        sh_order: order,
        coefficients: vec![ShCoeff { r: 0.5, g: 0.5, b: 0.5 }; coeff_count],
        config_json: "{}".to_string(),
    }
}

#[test]
fn irradiance_output_coeff_count_matches_order_3() {
    let out = make_irradiance_output(3);
    assert_eq!(out.coefficients.len(), 16);
}

#[test]
fn irradiance_output_coeff_count_matches_order_2() {
    let out = make_irradiance_output(2);
    assert_eq!(out.coefficients.len(), 9);
}

#[test]
fn irradiance_output_coeff_count_matches_order_1() {
    let out = make_irradiance_output(1);
    assert_eq!(out.coefficients.len(), 4);
}

#[test]
fn irradiance_output_serialize_deserialize_roundtrip() {
    let out = make_irradiance_output(3);
    let bytes = out.serialize_to_bytes().expect("serialize");
    let back = IrradianceOutput::deserialize_from_bytes(&bytes).expect("deserialize");
    assert_eq!(back.sh_order, 3);
    assert_eq!(back.coefficients.len(), 16);
    assert!((back.coefficients[0].r - 0.5).abs() < 1e-6);
}

#[test]
fn irradiance_output_corrupt_bytes_returns_error() {
    let result = IrradianceOutput::deserialize_from_bytes(b"\x00\x01\x02");
    assert!(result.is_err());
}

// ── ShCoeff ───────────────────────────────────────────────────────────────────

#[test]
fn sh_coeff_fields_are_accessible() {
    let c = ShCoeff { r: 0.1, g: 0.2, b: 0.3 };
    assert!((c.r - 0.1).abs() < 1e-7);
    assert!((c.g - 0.2).abs() < 1e-7);
    assert!((c.b - 0.3).abs() < 1e-7);
}

#[test]
fn sh_coeff_clone_is_equal() {
    let c = ShCoeff { r: 1.0, g: 2.0, b: 3.0 };
    let d = c;
    assert!((d.r - 1.0).abs() < 1e-7);
}

// ── Serde (JSON) round-trip ───────────────────────────────────────────────────

#[test]
fn probe_config_json_roundtrip() {
    let cfg = ProbeConfig::ultra();
    let json = serde_json::to_string(&cfg).expect("serialize");
    let back: ProbeConfig = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back.face_resolution, cfg.face_resolution);
    assert_eq!(back.sh_order, cfg.sh_order);
    assert_eq!(back.samples_per_face, cfg.samples_per_face);
}

// ── Baker (GPU-gated) ─────────────────────────────────────────────────────────

#[test]
#[ignore = "requires GPU adapter"]
fn probe_baker_produces_reflection_output() {
    use nebula_core::{context::BakeContext, progress::NullReporter, scene::SceneGeometry};
    use nebula_probe::ProbeBaker;
    use nebula_core::traits::BakePass;

    pollster::block_on(async {
        let ctx = BakeContext::new().await.expect("BakeContext");
        let scene = SceneGeometry::default();
        let cfg = ProbeConfig::fast();
        let out = ProbeBaker.execute(&scene, &cfg, &ctx, &NullReporter).await
            .expect("bake");
        assert_eq!(out.face_resolution, cfg.face_resolution);
    });
}
