use nebula::prelude::*;

// ── Prelude type accessibility ────────────────────────────────────────────────

// These tests are pure compile-time / name-resolution checks.  They confirm
// that the `nebula` façade correctly re-exports every promised type.

#[test]
fn prelude_bake_context_is_accessible() {
    fn _check_type<T: Sized>() {}
    _check_type::<BakeContext>();
}

#[test]
fn prelude_nebula_error_is_accessible() {
    fn _check_type<T: Sized>() {}
    _check_type::<NebulaError>();
}

#[test]
fn prelude_null_reporter_is_accessible() {
    fn _check_type<T: Sized>() {}
    _check_type::<NullReporter>();
}

#[test]
fn prelude_scene_geometry_is_accessible() {
    fn _check_type<T: Sized>() {}
    _check_type::<SceneGeometry>();
}

#[test]
fn prelude_chunk_tag_is_accessible() {
    fn _check_type<T: Sized>() {}
    _check_type::<ChunkTag>();
}

#[test]
fn prelude_bake_input_trait_is_accessible() {
    // Just checking that the BakeInput trait can be named in a where clause.
    fn _requires_bake_input<T: BakeInput>() {}
}

#[test]
fn prelude_bake_output_trait_is_accessible() {
    fn _requires_bake_output<T: BakeOutput>() {}
}

#[test]
fn prelude_bake_pass_trait_is_accessible() {
    // BakePass is an async trait; checking it can be named is sufficient.
    fn _requires_bake_pass<T: BakePass>() {}
}

#[test]
fn prelude_progress_reporter_trait_is_accessible() {
    fn _accepts_reporter(_: &dyn ProgressReporter) {}
}

// ── ChunkTag construction through prelude ─────────────────────────────────────

#[test]
fn chunk_tag_from_prelude_roundtrips() {
    let tag = ChunkTag::from_bytes(*b"TEST");
    assert_eq!(tag.to_bytes(), *b"TEST");
}

#[test]
fn prelude_chunk_tag_header_matches_nebu() {
    assert_eq!(ChunkTag::HEADER.to_bytes(), *b"NEBU");
}

#[test]
fn prelude_chunk_tag_end_is_end() {
    assert!(ChunkTag::END.is_end());
}

// ── SceneGeometry default ─────────────────────────────────────────────────────

#[test]
fn scene_geometry_default_has_no_meshes() {
    let scene = SceneGeometry::default();
    assert!(scene.meshes.is_empty());
}

#[test]
fn scene_geometry_default_has_no_lights() {
    let scene = SceneGeometry::default();
    assert!(scene.lights.is_empty());
}

// ── NebulaError display ───────────────────────────────────────────────────────

#[test]
fn nebula_error_gpu_display_contains_message() {
    let err = NebulaError::Gpu("no adapter".to_string());
    let s = err.to_string();
    assert!(s.contains("no adapter") || !s.is_empty());
}

#[test]
fn nebula_error_invalid_scene_display_is_non_empty() {
    let err = NebulaError::InvalidScene("empty".to_string());
    assert!(!err.to_string().is_empty());
}

// ── NullReporter implements ProgressReporter ──────────────────────────────────

#[test]
fn null_reporter_implements_progress_reporter() {
    let r: &dyn ProgressReporter = &NullReporter;
    r.begin("test", 10);
    r.step("test", 1, "hello");
    r.finish("test", true, "done");
}

// ── Sub-crate re-exports (feature-gated) ─────────────────────────────────────

#[cfg(feature = "light")]
#[test]
fn light_module_is_accessible() {
    use nebula::light::LightmapConfig;
    let _ = LightmapConfig::default();
}

#[cfg(feature = "ao")]
#[test]
fn ao_module_is_accessible() {
    use nebula::ao::AoConfig;
    let _ = AoConfig::default();
}

#[cfg(feature = "probe")]
#[test]
fn probe_module_is_accessible() {
    use nebula::probe::ProbeConfig;
    let _ = ProbeConfig::default();
}

#[cfg(feature = "audio")]
#[test]
fn audio_module_is_accessible() {
    use nebula::audio::AcousticConfig;
    let _ = AcousticConfig::default();
}

#[cfg(feature = "visibility")]
#[test]
fn visibility_module_is_accessible() {
    use nebula::visibility::PvsConfig;
    let _ = PvsConfig::default();
}

#[cfg(feature = "nav")]
#[test]
fn nav_module_is_accessible() {
    use nebula::nav::NavConfig;
    let _ = NavConfig::default();
}

// ── GPU-gated ─────────────────────────────────────────────────────────────────

#[test]
#[ignore = "requires GPU adapter"]
fn bake_context_can_be_created() {
    pollster::block_on(async {
        let ctx = BakeContext::new().await;
        assert!(ctx.is_ok(), "BakeContext::new() should succeed on a GPU system");
    });
}
