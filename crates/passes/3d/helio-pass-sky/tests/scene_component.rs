use helio_pass_sky::{ShaderSkyUniforms, SkyComponent};

#[test]
fn sky_component_is_the_shader_row_contract() {
    assert_eq!(
        std::mem::size_of::<SkyComponent>(),
        std::mem::size_of::<ShaderSkyUniforms>()
    );
    assert_eq!(std::mem::align_of::<SkyComponent>(), 4);
    assert_eq!(SkyComponent::default().cloud_mode, 1);
}
