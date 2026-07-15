use std::collections::HashMap;

/// Key for cached shader modules: (template_id, graph_hash, feature_flag_mask)
#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
pub struct RadiantShaderKey {
    pub template_id: u32,
    pub graph_hash: u64,
    pub feature_flags: u32,
}

/// Cache of compiled shader modules keyed by template variant.
///
/// The same (template, graph, flags) tuple always produces identical WGSL, so
/// we keep the `wgpu::ShaderModule` around to avoid recompilation.
pub struct RadiantShaderCache {
    modules: HashMap<RadiantShaderKey, wgpu::ShaderModule>,
}

impl RadiantShaderCache {
    pub fn new() -> Self {
        Self {
            modules: HashMap::new(),
        }
    }

    pub fn get(&self, key: &RadiantShaderKey) -> Option<&wgpu::ShaderModule> {
        self.modules.get(key)
    }

    pub fn insert(&mut self, key: RadiantShaderKey, module: wgpu::ShaderModule) {
        self.modules.insert(key, module);
    }

    /// Get or compile a shader module for the given key.
    /// If `graph_wgsl` is empty, the default template is used as-is.
    pub fn get_or_compile(
        &mut self,
        device: &wgpu::Device,
        key: RadiantShaderKey,
        template: &super::template::RadiantTemplate,
        graph_wgsl: &str,
        max_textures: usize,
        label: &str,
    ) -> &wgpu::ShaderModule {
        if !self.modules.contains_key(&key) {
            let source = template.build_shader_source(graph_wgsl, max_textures);
            #[cfg(target_arch = "wasm32")]
            let source =
                super::template::RadiantTemplate::apply_webgpu_fixups(&source, max_textures);
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            self.modules.insert(key, module);
        }
        self.modules.get(&key).unwrap()
    }

    pub fn len(&self) -> usize {
        self.modules.len()
    }
}
