use blade_graphics as gpu;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

pub type ResourceId = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceType {
    Buffer,
    Texture,
    RenderTarget,
    ComputeTarget,
}

#[derive(Debug)]
pub struct ResourceNode {
    pub id: ResourceId,
    pub name: String,
    pub resource_type: ResourceType,
    pub dependencies: Vec<ResourceId>,
    pub write_pass: Option<String>,
    pub read_passes: Vec<String>,
}

pub struct FrameGraph {
    nodes: HashMap<ResourceId, ResourceNode>,
    passes: Vec<RenderPassNode>,
    next_id: ResourceId,
    execution_order: Vec<usize>,
}

pub struct RenderPassNode {
    pub name: String,
    pub reads: Vec<ResourceId>,
    pub writes: Vec<ResourceId>,
    pub execute: Box<dyn Fn(&mut gpu::CommandEncoder) + Send + Sync>,
}

impl FrameGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            passes: Vec::new(),
            next_id: 0,
            execution_order: Vec::new(),
        }
    }
    
    pub fn add_resource(&mut self, name: String, resource_type: ResourceType) -> ResourceId {
        let id = self.next_id;
        self.next_id += 1;
        
        self.nodes.insert(
            id,
            ResourceNode {
                id,
                name,
                resource_type,
                dependencies: Vec::new(),
                write_pass: None,
                read_passes: Vec::new(),
            },
        );
        
        id
    }
    
    pub fn add_pass<F>(&mut self, name: String, reads: Vec<ResourceId>, writes: Vec<ResourceId>, execute: F)
    where
        F: Fn(&mut gpu::CommandEncoder) + Send + Sync + 'static,
    {
        for &read_id in &reads {
            if let Some(node) = self.nodes.get_mut(&read_id) {
                node.read_passes.push(name.clone());
            }
        }
        
        for &write_id in &writes {
            if let Some(node) = self.nodes.get_mut(&write_id) {
                node.write_pass = Some(name.clone());
            }
        }
        
        self.passes.push(RenderPassNode {
            name,
            reads,
            writes,
            execute: Box::new(execute),
        });
    }
    
    pub fn compile(&mut self) {
        self.execution_order.clear();
        
        let mut visited = HashSet::new();
        let mut stack = Vec::new();
        
        for i in 0..self.passes.len() {
            if !visited.contains(&i) {
                self.topological_sort(i, &mut visited, &mut stack);
            }
        }
        
        self.execution_order = stack;
    }
    
    fn topological_sort(&self, pass_idx: usize, visited: &mut HashSet<usize>, stack: &mut Vec<usize>) {
        visited.insert(pass_idx);
        
        let pass = &self.passes[pass_idx];
        for &read_id in &pass.reads {
            if let Some(node) = self.nodes.get(&read_id) {
                if let Some(ref write_pass_name) = node.write_pass {
                    if let Some(dep_idx) = self.passes.iter().position(|p| &p.name == write_pass_name) {
                        if !visited.contains(&dep_idx) {
                            self.topological_sort(dep_idx, visited, stack);
                        }
                    }
                }
            }
        }
        
        stack.push(pass_idx);
    }
    
    pub fn execute(&self, encoder: &mut gpu::CommandEncoder) {
        for &pass_idx in &self.execution_order {
            (self.passes[pass_idx].execute)(encoder);
        }
    }
    
    pub fn clear(&mut self) {
        self.passes.clear();
        self.nodes.clear();
        self.execution_order.clear();
    }
}

impl Default for FrameGraph {
    fn default() -> Self {
        Self::new()
    }
}
