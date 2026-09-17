use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use crate::material::{
    MATERIAL_CLASS_ANISOTROPIC, MATERIAL_CLASS_CLEAR_COAT, MATERIAL_CLASS_SKIN,
    MATERIAL_CLASS_SUBSURFACE,
};

pub struct RadiantTemplate {
    pub name: Arc<str>,
    /// Base WGSL source with `// RADIANT_OVERRIDE_SURFACE` markers
    pub wgsl_source: Arc<str>,
}

impl Clone for RadiantTemplate {
    fn clone(&self) -> Self {
        Self {
            // Both fields are immutable for the lifetime of a template.  They
            // are already static (built-ins and registered strings), so a clone
            // can copy the references directly.  Allocating/leaking here used
            // to make every registry clone permanently grow the process heap.
            name: Arc::clone(&self.name),
            wgsl_source: Arc::clone(&self.wgsl_source),
        }
    }
}

impl RadiantTemplate {
    /// Build the final WGSL source by optionally injecting a graph snippet.
    /// If `graph_wgsl` is empty, the OVERRIDE markers are replaced with a no-op
    /// passthrough to keep the default PBR evaluation.
    pub fn build_shader_source(&self, graph_wgsl: &str, max_textures: usize) -> String {
        let max_tex_str = max_textures.to_string();
        let src = self
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
        crate::shader::apply_webgpu_material_bindings(src, max_textures)
    }
}

/// Built-in templates shipped with the engine.
pub struct RadiantTemplateRegistry {
    templates: HashMap<u32, RadiantTemplate>,
    next_id: u32,
    dynamic_ids: VecDeque<u32>,
}

const MAX_DYNAMIC_TEMPLATES: usize = 1024;

/// The base gbuffer.wgsl source, embedded at compile time.
fn base_gbuffer_source() -> &'static str {
    include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../passes/3d/helio-pass-gbuffer/shaders/gbuffer.wgsl"
    ))
}

/// Replace a function in the base WGSL with a custom override.
/// `fn_marker` is the function declaration prefix (e.g. `"fn radiant_eval_surface"`).
/// The override function must have the same signature as the original.
fn compose_fn_override(base: &str, override_fn: &str, fn_marker: &str) -> String {
    if let Some(start) = base.find(fn_marker) {
        if let Some(body_start) = base[start..].find('{') {
            let body_start_abs = start + body_start;
            let mut depth = 1u32;
            let mut i = body_start_abs + 1;
            let bytes = base.as_bytes();
            while i < bytes.len() && depth > 0 {
                match bytes[i] {
                    b'{' => depth += 1,
                    b'}' => depth -= 1,
                    _ => {}
                }
                i += 1;
            }
            let body_end = i;
            let before = &base[..start];
            let after = &base[body_end..];
            return format!("{}{}\n{}", before, override_fn, after);
        }
    }
    override_fn.to_string()
}

impl Clone for RadiantTemplateRegistry {
    fn clone(&self) -> Self {
        Self {
            templates: self.templates.clone(),
            next_id: self.next_id,
            dynamic_ids: self.dynamic_ids.clone(),
        }
    }
}

/// A cheaply-clonable, thread-safe handle to a registry that the renderer
/// and its passes share instead of each keeping their own copy.
///
/// A shared registry avoids copying the template map when it is handed to
/// multiple renderer passes. Template clones themselves only copy static
/// references and do not allocate.
pub type SharedTemplateRegistry = std::sync::Arc<std::sync::RwLock<RadiantTemplateRegistry>>;

impl RadiantTemplateRegistry {
    /// Create an empty registry (no built-in templates).
    /// Used by TransparentPass to avoid inheriting gbuffer templates.
    pub fn new_empty() -> Self {
        Self {
            templates: HashMap::new(),
            next_id: 5,
            dynamic_ids: VecDeque::new(),
        }
    }

    pub fn new() -> Self {
        let mut reg = Self {
            templates: HashMap::new(),
            // Start at 5 to avoid conflicts with built-in templates:
            //   0 = default_pbr, 1 = clear_coat, 2 = subsurface,
            //   3 = anisotropic, 4 = skin
            next_id: 5,
            dynamic_ids: VecDeque::new(),
        };
        reg.templates.insert(
            0,
            RadiantTemplate {
                name: Arc::from("default_pbr"),
                wgsl_source: Arc::from(include_str!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../passes/3d/helio-pass-gbuffer/shaders/gbuffer.wgsl"
                ))),
            },
        );
        reg.register_default_templates();
        reg
    }

    /// Register the built-in tier-2 surface templates shipped with the engine.
    /// Each template is registered with its predefined `MATERIAL_CLASS_*` ID
    /// so users can reference them by the constants from `helio_mats`.
    fn register_default_templates(&mut self) {
        self.register_partial_str_with_id(
            MATERIAL_CLASS_CLEAR_COAT,
            "clear_coat",
            include_str!("../../templates/clear_coat.wgsl").to_string(),
        );
        self.register_partial_str_with_id(
            MATERIAL_CLASS_SUBSURFACE,
            "subsurface",
            include_str!("../../templates/subsurface.wgsl").to_string(),
        );
        self.register_partial_str_with_id(
            MATERIAL_CLASS_ANISOTROPIC,
            "anisotropic",
            include_str!("../../templates/anisotropic.wgsl").to_string(),
        );
        self.register_partial_str_with_id(
            MATERIAL_CLASS_SKIN,
            "skin",
            include_str!("../../templates/skin.wgsl").to_string(),
        );
    }

    pub fn get(&self, class: u32) -> Option<&RadiantTemplate> {
        self.templates.get(&class)
    }

    pub fn keys(&self) -> Vec<u32> {
        self.templates.keys().copied().collect()
    }

    /// Iterate over all (class_id, template) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&u32, &RadiantTemplate)> {
        self.templates.iter()
    }

    pub fn register(&mut self, class: u32, template: RadiantTemplate) {
        self.templates.insert(class, template);
    }

    /// Override an existing class with a new WGSL source (used by TransparentPass
    /// to replace the default gbuffer base with its own transparent base shader).
    pub fn override_class(&mut self, class: u32, name: &str, wgsl_source: &str) {
        self.templates
            .insert(class, RadiantTemplate { name: Arc::from(name), wgsl_source: Arc::from(wgsl_source) });
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
        log::info!("[Radiant] register_str '{}' → class {}", name, id);
        self.templates.insert(
            id,
            RadiantTemplate {
                name: Arc::from(format!("Radiant:{}", name)),
                wgsl_source: Arc::from(wgsl_source),
            },
        );
        self.dynamic_ids.push_back(id);
        while self.dynamic_ids.len() > MAX_DYNAMIC_TEMPLATES {
            if let Some(oldest) = self.dynamic_ids.pop_front() {
                self.templates.remove(&oldest);
            }
        }
        id
    }

    /// Register a partial template — a WGSL snippet containing ONLY the
    /// `radiant_eval_surface()` function body. The snippet is composed with
    /// the base gbuffer.wgsl at registration time.
    pub fn register_partial_str(&mut self, name: &str, override_fn: String) -> u32 {
        let base = base_gbuffer_source();
        let composed = compose_fn_override(base, &override_fn, "fn radiant_eval_surface");
        self.register_str(name, composed)
    }

    /// Compose a transparent override function with the transparent base shader.
    /// Returns the composed WGSL source ready for registration.
    pub fn compose_transparent_override(&self, override_fn: &str) -> String {
        let base = include_str!("../../templates/transparent_base.wgsl");
        compose_fn_override(base, override_fn, "fn radiant_eval_transparent")
    }

    /// Register a template at a specific class ID (instead of auto-assigning).
    pub fn register_str_at(&mut self, class: u32, name: &str, wgsl_source: String) {
        self.templates.insert(
            class,
            RadiantTemplate {
                name: Arc::from(format!("Radiant:{}", name)),
                wgsl_source: Arc::from(wgsl_source),
            },
        );
    }

    /// Register a partial template for the transparent pass.
    /// The `override_fn` should contain a `fn radiant_eval_transparent(...)` function body.
    /// Composed with the transparent base shader at registration time.
    pub fn register_transparent_partial_str(&mut self, name: &str, override_fn: String) -> u32 {
        let base = include_str!("../../templates/transparent_base.wgsl");
        let composed = compose_fn_override(base, &override_fn, "fn radiant_eval_transparent");
        self.register_str(name, composed)
    }

    /// Register a partial template with a specific class ID (instead of auto-assigning).
    /// Used internally by `register_default_templates()` to map templates to the
    /// predefined `MATERIAL_CLASS_*` constants.
    fn register_partial_str_with_id(&mut self, id: u32, name: &str, override_fn: String) {
        let base = base_gbuffer_source();
        let composed = compose_fn_override(base, &override_fn, "fn radiant_eval_surface");
        self.templates.insert(
            id,
            RadiantTemplate {
                name: Arc::from(format!("Radiant:{}", name)),
                wgsl_source: Arc::from(composed),
            },
        );
    }

    pub fn len(&self) -> usize {
        self.templates.len()
    }
}
