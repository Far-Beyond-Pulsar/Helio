use std::collections::HashMap;

pub struct RadiantTemplate {
    pub name: &'static str,
    /// Base WGSL source with `// RADIANT_OVERRIDE_SURFACE` markers
    pub wgsl_source: &'static str,
}

impl RadiantTemplate {
    /// Build the final WGSL source by optionally injecting a graph snippet.
    /// If `graph_wgsl` is empty, the OVERRIDE markers are replaced with a no-op
    /// passthrough to keep the default PBR evaluation.
    pub fn build_shader_source(&self, graph_wgsl: &str, max_textures: usize) -> String {
        let max_tex_str = max_textures.to_string();
        let mut src = self
            .wgsl_source
            .replace(
                "binding_array<texture_2d<f32>, 256>",
                &format!("binding_array<texture_2d<f32>, {max_tex_str}>"),
            )
            .replace(
                "binding_array<sampler, 256>",
                &format!("binding_array<sampler, {max_tex_str}>"),
            );

        // wgpu's Naga requires `enable wgpu_binding_array;` for binding_array
        // support.  Browser WebGPU (Chrome's Tint) does not recognise this
        // extension — it has binding_array support in core WGSL.  Strip it for
        // wasm builds, keep it for native wgpu.
        #[cfg(target_arch = "wasm32")]
        {
            src = src.replace("enable wgpu_binding_array;\n", "");
            src = src.replace("enable wgpu_binding_array;\r\n", "");
        }

        if graph_wgsl.is_empty() {
            // No graph: remove the override markers, leaving the default code
            src.replace("// RADIANT_OVERRIDE_SURFACE\n", "")
                .replace("// RADIANT_OVERRIDE_END\n", "")
        } else {
            // Graph present: replace everything from OVERRIDE_SURFACE to OVERRIDE_END
            // with the graph's override code
            let override_start = "// RADIANT_OVERRIDE_SURFACE";
            let override_end = "// RADIANT_OVERRIDE_END";
            if let Some(start) = src.find(override_start) {
                if let Some(end) = src.find(override_end) {
                    let before = &src[..start];
                    let after = &src[end + override_end.len()..];
                    format!("{}{}\n{}", before, graph_wgsl, after)
                } else {
                    src
                }
            } else {
                src
            }
        }
    }

    /// Replace native binding arrays with baseline-WebGPU individual bindings.
    ///
    /// WebGPU does not expose wgpu's `binding_array` WGSL extension. Individual
    /// texture/sampler bindings plus an explicit switch retain all material slots
    /// without requiring a native-only feature.
    pub fn apply_webgpu_fixups(src: &str, max_textures: usize) -> String {
        libhelio::shader::apply_webgpu_material_bindings(src, max_textures)
    }
}

/// Built-in templates shipped with the engine.
pub struct RadiantTemplateRegistry {
    templates: HashMap<u32, RadiantTemplate>,
    next_id: u32,
}

impl RadiantTemplateRegistry {
    pub fn new() -> Self {
        let mut reg = Self {
            templates: HashMap::new(),
            next_id: 1,
        };
        reg.templates.insert(
            0,
            RadiantTemplate {
                name: "default_pbr",
                wgsl_source: include_str!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../helio-pass-gbuffer/shaders/gbuffer.wgsl"
                )),
            },
        );
        reg
    }

    pub fn get(&self, class: u32) -> Option<&RadiantTemplate> {
        self.templates.get(&class)
    }

    pub fn register(&mut self, class: u32, template: RadiantTemplate) {
        self.templates.insert(class, template);
    }

    /// Load a template from a WGSL file on disk. The template should contain
    /// `// RADIANT_OVERRIDE_SURFACE` and `// RADIANT_OVERRIDE_END` markers.
    /// Returns the assigned template_id.
    pub fn load_from_file(&mut self, path: &std::path::Path) -> std::io::Result<u32> {
        let source = std::fs::read_to_string(path)?;
        Ok(self.register_str(
            path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown"),
            source,
        ))
    }

    /// Register a template from a string (useful for embedded or generated templates).
    pub fn register_str(&mut self, name: &str, wgsl_source: String) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.templates.insert(
            id,
            RadiantTemplate {
                name: Box::leak(format!("Radiant:{}", name).into_boxed_str()),
                wgsl_source: Box::leak(wgsl_source.into_boxed_str()),
            },
        );
        id
    }

    pub fn len(&self) -> usize {
        self.templates.len()
    }
}
