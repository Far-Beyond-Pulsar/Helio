use helio_core::{PipelineFormatKey, PipelineHandle, PipelineRecipeBuilder};

#[test]
fn recipes_are_pass_local_and_format_keyed() {
    let mut recipes = PipelineRecipeBuilder::new();
    let key = PipelineFormatKey::new(
        "test_pass",
        [wgpu::TextureFormat::Rgba8Unorm],
        Some(wgpu::TextureFormat::Depth32Float),
    );
    let handle: PipelineHandle = 7;
    recipes.add(
        handle,
        key.clone(),
        move |_device, resolved, _driver_cache| {
            panic!("recipe is executor-owned and must not run during declaration: {resolved:?}")
        },
    );

    assert_eq!(recipes.len(), 1);
    assert!(!recipes.is_empty());
}
