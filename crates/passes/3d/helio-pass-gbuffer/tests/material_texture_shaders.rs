#[test]
fn texture_consumers_validate_with_scene_owned_slot_defaults() {
    for (name, source) in [
        ("gbuffer", include_str!("../shaders/gbuffer.wgsl")),
        (
            "forward",
            include_str!("../../helio-pass-forward-lit/shaders/forward_lit.wgsl"),
        ),
        (
            "portal",
            include_str!("../../helio-pass-portal-instances/shaders/gbuffer_portal.wgsl"),
        ),
        (
            "virtual_geometry",
            include_str!("../../helio-pass-virtual-geometry/shaders/vg_gbuffer.wgsl"),
        ),
    ] {
        let resolved = helio_core::shader::resolve_with(source, &[helio_mats::PBR_EVAL_SNIPPET]);
        let module = naga::front::wgsl::parse_str(&resolved)
            .unwrap_or_else(|error| panic!("{name}: {}", error.emit_to_string(&resolved)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|error| panic!("{name}: {error:?}"));
    }
}
