use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShaderStage {
    Vertex,
    Fragment,
    Compute,
    Geometry,
    TessellationControl,
    TessellationEvaluation,
    Mesh,
    Task,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShaderSource {
    pub stage: ShaderStage,
    pub source: String,
    pub entry_point: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Shader {
    pub name: String,
    pub sources: Vec<ShaderSource>,
}

impl Shader {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            sources: Vec::new(),
        }
    }
    
    pub fn add_source(mut self, stage: ShaderStage, source: String, entry_point: String) -> Self {
        self.sources.push(ShaderSource {
            stage,
            source,
            entry_point,
        });
        self
    }
}
