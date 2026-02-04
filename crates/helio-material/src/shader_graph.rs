use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeType {
    TextureSample,
    Constant,
    Add,
    Multiply,
    Lerp,
    Normalize,
    DotProduct,
    CrossProduct,
    FresnelSchlick,
    Power,
    VertexColor,
    TexCoord,
    WorldPosition,
    ViewDirection,
    LightDirection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShaderNode {
    pub id: u32,
    pub node_type: NodeType,
    pub inputs: Vec<u32>,
    pub parameters: HashMap<String, f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShaderGraph {
    pub nodes: Vec<ShaderNode>,
    pub output_node: u32,
}

impl ShaderGraph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            output_node: 0,
        }
    }
    
    pub fn add_node(&mut self, node: ShaderNode) -> u32 {
        let id = self.nodes.len() as u32;
        self.nodes.push(node);
        id
    }
    
    pub fn compile(&self) -> String {
        // Would generate WGSL/SPIR-V from graph
        String::from("// Compiled shader")
    }
}

impl Default for ShaderGraph {
    fn default() -> Self {
        Self::new()
    }
}
