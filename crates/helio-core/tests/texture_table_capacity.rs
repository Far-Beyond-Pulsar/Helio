//! The texture table's capacity follows the device: binding arrays count
//! against the binding-array limits (thousands of textures on a desktop
//! GPU), one-binding-per-texture tables against the per-stage limits. The
//! table is read through one sampler, so no sampler limit bounds it.

use helio_core::{
    texture_table_capacity, TEXTURE_BINDING_ARRAYS, TEXTURE_TABLE_BINDINGS_LIMIT,
    TEXTURE_TABLE_LIMIT, TEXTURE_TABLE_RESERVE,
};

/// What wgpu reports for a D3D12 resource-binding tier 2+ adapter: a
/// million-descriptor heap, 2048 shader-visible samplers.
fn d3d12_tier2() -> wgpu::Limits {
    wgpu::Limits {
        max_sampled_textures_per_shader_stage: 1_000_000,
        max_samplers_per_shader_stage: 2048,
        max_binding_array_elements_per_shader_stage: 1_000_000,
        max_binding_array_sampler_elements_per_shader_stage: 2048,
        ..wgpu::Limits::default()
    }
}

/// What a desktop Vulkan driver with descriptor indexing reports (NVIDIA,
/// AMD): update-after-bind limits in the hundreds of thousands or more.
fn vulkan_descriptor_indexing() -> wgpu::Limits {
    wgpu::Limits {
        max_sampled_textures_per_shader_stage: 1_048_576,
        max_samplers_per_shader_stage: 1_048_576,
        max_binding_array_elements_per_shader_stage: 500_000,
        max_binding_array_sampler_elements_per_shader_stage: 1_048_576,
        ..wgpu::Limits::default()
    }
}

#[cfg(not(any(
    target_arch = "wasm32",
    target_os = "macos",
    target_os = "ios",
    target_os = "android"
)))]
#[test]
fn desktop_binding_arrays_hold_thousands_of_textures() {
    const { assert!(TEXTURE_TABLE_LIMIT >= 4096) };
    // D3D12: the 2048-sampler heap does not bound the table (one sampler
    // reads every texture).
    assert_eq!(
        texture_table_capacity(TEXTURE_BINDING_ARRAYS, &d3d12_tier2()),
        TEXTURE_TABLE_LIMIT
    );
    // Vulkan: the device offers more than the table's cap.
    assert_eq!(
        texture_table_capacity(TEXTURE_BINDING_ARRAYS, &vulkan_descriptor_indexing()),
        TEXTURE_TABLE_LIMIT
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn binding_arrays_count_only_textures_against_the_element_limit() {
    let limits = wgpu::Limits {
        max_binding_array_elements_per_shader_stage: 600,
        max_binding_array_sampler_elements_per_shader_stage: 0,
        ..wgpu::Limits::default()
    };
    // 600 textures use the 600 elements; no sampler is a binding array.
    assert_eq!(
        texture_table_capacity(TEXTURE_BINDING_ARRAYS, &limits),
        600.min(TEXTURE_TABLE_LIMIT)
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn binding_arrays_ignore_the_per_stage_binding_limits() {
    // wgpu counts binding-array elements only against the binding-array
    // limits, so the WebGPU per-stage defaults (16) do not shrink the table.
    let limits = wgpu::Limits {
        max_sampled_textures_per_shader_stage: 16,
        max_samplers_per_shader_stage: 16,
        ..d3d12_tier2()
    };
    assert_eq!(
        texture_table_capacity(TEXTURE_BINDING_ARRAYS, &limits),
        TEXTURE_TABLE_LIMIT
    );
}

#[test]
fn one_binding_per_texture_stays_under_the_bindings_limit() {
    // A partial binding-array capability is none: one binding per texture,
    // capped however much the device offers.
    for features in [
        wgpu::Features::empty(),
        wgpu::Features::TEXTURE_BINDING_ARRAY,
    ] {
        assert_eq!(
            texture_table_capacity(features, &vulkan_descriptor_indexing()),
            TEXTURE_TABLE_BINDINGS_LIMIT
                .min(wgpu::Limits::default().max_bindings_per_bind_group as usize - 3)
        );
    }
    // The WebGPU baseline: 16 sampled textures, five kept for the reader.
    assert_eq!(
        texture_table_capacity(wgpu::Features::empty(), &wgpu::Limits::default()),
        (16 - TEXTURE_TABLE_RESERVE).min(TEXTURE_TABLE_BINDINGS_LIMIT)
    );
    // The sampler limit does not bound it: one sampler reads every texture.
    let one_sampler = wgpu::Limits {
        max_samplers_per_shader_stage: 2,
        ..wgpu::Limits::default()
    };
    assert_eq!(
        texture_table_capacity(wgpu::Features::empty(), &one_sampler),
        (16 - TEXTURE_TABLE_RESERVE).min(TEXTURE_TABLE_BINDINGS_LIMIT)
    );
}

#[cfg(any(
    target_arch = "wasm32",
    target_os = "macos",
    target_os = "ios",
    target_os = "android"
))]
#[test]
fn mobile_and_browser_keep_the_baseline_table() {
    const { assert!(TEXTURE_TABLE_LIMIT == 16) };
    const { assert!(TEXTURE_TABLE_BINDINGS_LIMIT == 16) };
    assert!(texture_table_capacity(TEXTURE_BINDING_ARRAYS, &d3d12_tier2()) <= 16);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[should_panic(expected = "at least one sampled texture binding")]
fn a_device_with_no_binding_array_elements_has_no_table() {
    let limits = wgpu::Limits {
        max_binding_array_elements_per_shader_stage: 0,
        ..wgpu::Limits::default()
    };
    texture_table_capacity(TEXTURE_BINDING_ARRAYS, &limits);
}
