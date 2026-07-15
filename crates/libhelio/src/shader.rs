/// Replace native material binding arrays with baseline-WebGPU bindings.
///
/// Browser WebGPU does not expose wgpu's `binding_array` WGSL extension. The
/// fixed bindings and switch preserve every material texture slot without
/// requiring a native-only feature.
pub fn apply_webgpu_material_bindings(src: &str, max_textures: usize) -> String {
    let mut declarations = String::new();
    for index in 0..max_textures {
        declarations.push_str(&format!(
            "@group(1) @binding({}) var scene_texture_{index}: texture_2d<f32>;\n",
            2 + index,
        ));
    }
    for index in 0..max_textures {
        declarations.push_str(&format!(
            "@group(1) @binding({}) var scene_sampler_{index}: sampler;\n",
            2 + max_textures + index,
        ));
    }

    let mut source = String::with_capacity(src.len() + declarations.len());
    for line in src.lines() {
        if line.contains("scene_textures:") && line.contains("binding_array<texture_2d") {
            source.push_str(&declarations);
        } else if line.contains("scene_samplers:") && line.contains("binding_array<sampler") {
            // Both binding tables were emitted in place of scene_textures.
        } else {
            source.push_str(line);
            source.push('\n');
        }
    }

    let mut sample_switch = String::from("switch slot.texture_index {\n");
    for index in 0..max_textures {
        sample_switch.push_str(&format!(
            "        case {index}u: {{ return textureSampleLevel(scene_texture_{index}, scene_sampler_{index}, uv, 0.0); }}\n",
        ));
    }
    sample_switch.push_str("        default: { return fallback; }\n    }");

    source.replace(
        "return textureSample(scene_textures[slot.texture_index], scene_samplers[slot.texture_index], uv);",
        &sample_switch,
    )
}

#[cfg(test)]
mod tests {
    use super::apply_webgpu_material_bindings;

    #[test]
    fn expands_material_binding_arrays() {
        let source = r#"
@group(1) @binding(2) var scene_textures: binding_array<texture_2d<f32>, 2>;
@group(1) @binding(3) var scene_samplers: binding_array<sampler, 2>;
fn sample_texture(slot: MaterialTextureSlot, uv: vec2<f32>, fallback: vec4<f32>) -> vec4<f32> {
    return textureSample(scene_textures[slot.texture_index], scene_samplers[slot.texture_index], uv);
}
"#;
        let fixed = apply_webgpu_material_bindings(source, 2);

        assert!(!fixed.contains("binding_array"));
        assert!(fixed.contains("@binding(2) var scene_texture_0"));
        assert!(fixed.contains("@binding(5) var scene_sampler_1"));
        assert!(fixed.contains("case 1u:"));
        assert!(fixed.contains("textureSampleLevel"));
    }
}
