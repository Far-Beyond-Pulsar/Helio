pub const WEBGPU_MATERIAL_TEXTURES: usize = 16;

const TEXTURE_BINDING_START: u32 = 2;
const SAMPLER_BINDING_START: u32 = TEXTURE_BINDING_START + WEBGPU_MATERIAL_TEXTURES as u32;

pub fn webgpu_material_shader(source: &str) -> String {
    const BINDINGS_MARKER: &str = "// HELIO_WEBGPU_MATERIAL_BINDINGS";
    const SAMPLER_MARKER: &str = "// HELIO_WEBGPU_MATERIAL_SAMPLER";

    assert!(source.contains(BINDINGS_MARKER));
    assert!(source.contains(SAMPLER_MARKER));

    let mut bindings = String::new();
    for slot in 0..WEBGPU_MATERIAL_TEXTURES {
        bindings.push_str(&format!(
            "@group(1) @binding({}) var scene_texture_{slot}: texture_2d<f32>;\n",
            TEXTURE_BINDING_START + slot as u32
        ));
    }
    for slot in 0..WEBGPU_MATERIAL_TEXTURES {
        bindings.push_str(&format!(
            "@group(1) @binding({}) var scene_sampler_{slot}: sampler;\n",
            SAMPLER_BINDING_START + slot as u32
        ));
    }

    let mut sampler = String::from(
        "fn sample_scene_texture(\n\
         \x20   texture_index: u32,\n\
         \x20   uv: vec2<f32>,\n\
         \x20   uv_dx: vec2<f32>,\n\
         \x20   uv_dy: vec2<f32>,\n\
         ) -> vec4<f32> {\n\
         \x20   switch texture_index {\n",
    );
    for slot in 0..WEBGPU_MATERIAL_TEXTURES {
        sampler.push_str(&format!(
            "        case {slot}u: {{ return textureSampleGrad(scene_texture_{slot}, scene_sampler_{slot}, uv, uv_dx, uv_dy); }}\n"
        ));
    }
    sampler.push_str(
        "        default: { return vec4<f32>(1.0); }\n\
         \x20   }\n\
         }",
    );

    source
        .replace(BINDINGS_MARKER, &bindings)
        .replace(SAMPLER_MARKER, &sampler)
}

pub fn webgpu_material_layout_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
    let mut entries = Vec::with_capacity(WEBGPU_MATERIAL_TEXTURES * 2);
    for slot in 0..WEBGPU_MATERIAL_TEXTURES {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: TEXTURE_BINDING_START + slot as u32,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });
    }
    for slot in 0..WEBGPU_MATERIAL_TEXTURES {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: SAMPLER_BINDING_START + slot as u32,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        });
    }
    entries
}

pub fn push_webgpu_material_bindings<'a>(
    entries: &mut Vec<wgpu::BindGroupEntry<'a>>,
    texture_views: &'a [&'a wgpu::TextureView],
    samplers: &'a [&'a wgpu::Sampler],
) {
    assert_eq!(texture_views.len(), WEBGPU_MATERIAL_TEXTURES);
    assert_eq!(samplers.len(), WEBGPU_MATERIAL_TEXTURES);

    for (slot, texture_view) in texture_views.iter().enumerate() {
        entries.push(wgpu::BindGroupEntry {
            binding: TEXTURE_BINDING_START + slot as u32,
            resource: wgpu::BindingResource::TextureView(texture_view),
        });
    }
    for (slot, sampler) in samplers.iter().enumerate() {
        entries.push(wgpu::BindGroupEntry {
            binding: SAMPLER_BINDING_START + slot as u32,
            resource: wgpu::BindingResource::Sampler(sampler),
        });
    }
}
