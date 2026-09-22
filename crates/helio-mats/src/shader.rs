//! Shared PBR/BRDF evaluation — `fresnel_schlick`, the distribution and geometry terms.
//!
//! Lives alongside [`crate::material`] and [`crate::radiant`], which already own the
//! CPU side of the material model and consume this text directly (manual
//! concatenation, not `helio-core`'s generic shader-snippet mechanism — see
//! `helio-core/src/shader/mod.rs`'s doc for why that mechanism only owns the
//! generic prelude and pass-declared snippets, never a specific domain's text).
//!
//! Registering a module here is the step that is easy to miss and gives no warning when
//! missed: the `//!use` marker is an ordinary WGSL comment, so an unregistered module
//! leaves the marker inert, the source passes through untouched, and the shader fails at
//! `create_shader_module` with "no definition in scope" for a function that plainly
//! exists. That is exactly how `deferred_lighting.wgsl` and `forward_lit.wgsl` came to be
//! uncompilable — which on this path takes out lighting entirely.
pub const PBR_EVAL: &str = include_str!("../shaders/pbr_eval.wgsl");

/// Marker opting a shader into [`PBR_EVAL`]. Must appear in the source.
pub const PBR_MARKER: &str = "//!use pbr_eval";

/// [`helio_core::shader::ShaderSnippet`] for [`PBR_EVAL`]. Real consumers
/// (gbuffer/forward-lit/deferred-light templates) currently splice
/// [`PBR_EVAL`] in by hand rather than through `helio_core::shader`'s
/// marker mechanism, but this is the snippet form for anything that wants
/// to go through `resolve_with`/`module_with` instead.
pub const PBR_EVAL_SNIPPET: helio_core::shader::ShaderSnippet =
    helio_core::shader::ShaderSnippet::new(PBR_MARKER, PBR_EVAL);

/// Replace native material binding arrays with baseline-WebGPU bindings.
///
/// Browser WebGPU does not expose wgpu's `binding_array` WGSL extension. The
/// fixed bindings and switch preserve every material texture slot without
/// requiring a native-only feature. The native-only extension directive is
/// removed together with the binding-array declarations.
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
        if line.trim() == "enable wgpu_binding_array;" {
            // The rewritten shader no longer uses this native-only extension.
        } else if line.contains("scene_textures:") && line.contains("binding_array<texture_2d") {
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

    // VT fetches must retain the wanted mip and residency lookup when arrays
    // are expanded. Compute derivatives before choosing a concrete texture.
    let mut vt_switch = String::from("let vt_row = vt_meta[slot.texture_index];\n    let vt_wanted = vt_wanted_mip(vt_row, uv);\n    switch slot.texture_index {\n");
    for index in 0..max_textures {
        vt_switch.push_str(&format!(
            "        case {index}u: {{ return vt_sample_level(scene_texture_{index}, scene_sampler_{index}, vt_row, uv, vt_wanted); }}\n",
        ));
    }
    vt_switch.push_str("        default: { return fallback; }\n    }");
    source.replace(
        "return textureSample(scene_textures[slot.texture_index], scene_samplers[slot.texture_index], uv);",
        &sample_switch,
    ).replace(
        "return vt_sample(scene_textures[slot.texture_index], scene_samplers[slot.texture_index], vt_meta[slot.texture_index], uv);",
        &vt_switch,
    )
}

/// Replace the decal pass's scene binding arrays with baseline-WebGPU bindings.
///
/// Same idea as [`apply_webgpu_material_bindings`], but for the decal collect
/// compute shader: it indexes the table with a plain `u32` rather than a
/// material slot, samples with `textureSampleLevel` (compute has no implicit
/// derivatives), and owns group 1 outright so its tables start at binding 0.
pub fn apply_webgpu_decal_bindings(src: &str, max_textures: usize) -> String {
    let mut declarations = String::new();
    for index in 0..max_textures {
        declarations.push_str(&format!(
            "@group(1) @binding({index}) var scene_texture_{index}: texture_2d<f32>;\n",
        ));
    }
    for index in 0..max_textures {
        declarations.push_str(&format!(
            "@group(1) @binding({}) var scene_sampler_{index}: sampler;\n",
            max_textures + index,
        ));
    }

    let mut source = String::with_capacity(src.len() + declarations.len());
    for line in src.lines() {
        if line.contains("scene_textures:") && line.contains("binding_array<texture_2d") {
            source.push_str(&declarations);
        } else if line.contains("scene_samplers:") && line.contains("binding_array<sampler") {
            // Both binding tables were emitted in place of scene_textures.
        } else if line.trim() == "enable wgpu_binding_array;" {
            // wgpu-only extension; baseline WebGPU rejects the directive.
        } else {
            source.push_str(line);
            source.push('\n');
        }
    }

    let mut sample_switch = String::from("switch texture_index {\n");
    for index in 0..max_textures {
        sample_switch.push_str(&format!(
            "        case {index}u: {{ return textureSampleLevel(scene_texture_{index}, scene_sampler_{index}, uv, 0.0); }}\n",
        ));
    }
    sample_switch.push_str("        default: { return vec4<f32>(1.0); }\n    }");

    source.replace(
        "return textureSampleLevel(scene_textures[texture_index], scene_samplers[texture_index], uv, 0.0);",
        &sample_switch,
    )
}

#[cfg(test)]
mod tests {
    use super::{apply_webgpu_decal_bindings, apply_webgpu_material_bindings};

    #[test]
    fn expands_material_binding_arrays() {
        let source = r#"
enable wgpu_binding_array;
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

    #[test]
    fn expands_decal_binding_arrays() {
        let source = r#"
enable wgpu_binding_array;
@group(1) @binding(0) var scene_textures: binding_array<texture_2d<f32>, 256>;
@group(1) @binding(1) var scene_samplers: binding_array<sampler, 256>;
fn sample_decal_texture(texture_index: u32, uv: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(scene_textures[texture_index], scene_samplers[texture_index], uv, 0.0);
}
"#;
        let fixed = apply_webgpu_decal_bindings(source, 2);

        // The extension directive and both tables must be gone — baseline
        // WebGPU rejects all three.
        assert!(!fixed.contains("binding_array"));
        assert!(!fixed.contains("enable wgpu_binding_array"));
        assert!(fixed.contains("@group(1) @binding(0) var scene_texture_0"));
        assert!(fixed.contains("@group(1) @binding(1) var scene_texture_1"));
        // Samplers follow the textures: base = max_textures.
        assert!(fixed.contains("@group(1) @binding(2) var scene_sampler_0"));
        assert!(fixed.contains("@group(1) @binding(3) var scene_sampler_1"));
        assert!(fixed.contains(
            "case 1u: { return textureSampleLevel(scene_texture_1, scene_sampler_1, uv, 0.0); }"
        ));
        assert!(fixed.contains("default: { return vec4<f32>(1.0); }"));
    }
}
