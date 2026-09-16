use std::collections::HashMap;

/// Registry of graph-generated WGSL snippets, keyed by hash.
///
/// External tools (graph compilers, editors) call `register()` to make a
/// compiled material graph available to the engine. The GBuffer pass looks
/// up the snippet by hash when it encounters a material with `graph_hash != 0`.
pub struct RadiantGraphRegistry {
    /// graph_hash → WGSL source snippet injected at RADIANT_OVERRIDE_SURFACE
    snippets: HashMap<u64, String>,
}

impl RadiantGraphRegistry {
    pub fn new() -> Self {
        Self {
            snippets: HashMap::new(),
        }
    }

    /// Register a compiled graph snippet. `graph_hash` should be a content-hash
    /// of the graph's serialized form (provided by the external compiler).
    pub fn register(&mut self, graph_hash: u64, wgsl_snippet: String) {
        self.snippets.insert(graph_hash, wgsl_snippet);
    }

    /// Look up a graph snippet by hash. Returns None if not found.
    pub fn get(&self, graph_hash: u64) -> Option<&str> {
        self.snippets.get(&graph_hash).map(|s| s.as_str())
    }

    /// Remove a graph snippet (e.g. when a material is deleted).
    pub fn unregister(&mut self, graph_hash: u64) {
        self.snippets.remove(&graph_hash);
    }

    /// Number of registered graph snippets.
    pub fn len(&self) -> usize {
        self.snippets.len()
    }
}
