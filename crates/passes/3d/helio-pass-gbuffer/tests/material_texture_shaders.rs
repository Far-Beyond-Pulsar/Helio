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

#[test]
fn depth_cutouts_validate_for_native_and_expanded_texture_bindings() {
    let material = include_str!("../shaders/depth_material.wgsl");
    for geometry in [
        include_str!("../../helio-pass-shadow/shaders/shadow.wgsl"),
        include_str!("../../helio-pass-depth-prepass/shaders/depth_prepass.wgsl"),
    ] {
        let native = format!("{material}\n{geometry}");
        for source in [native.clone(), helio_mats::apply_webgpu_material_bindings(&native, 4)] {
            let module = naga::front::wgsl::parse_str(&source)
                .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module).unwrap_or_else(|e| panic!("{e:?}"));
        }
    }
}
