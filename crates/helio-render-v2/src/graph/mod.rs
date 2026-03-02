//! Render graph system with automatic dependency resolution
//!
//! The render graph automatically orders passes based on resource dependencies
//! and manages transient resource allocation/deallocation.

mod pass;
mod resource;
mod builder;

pub use pass::{RenderPass, PassContext};
pub use resource::{PassId, ResourceHandle};
pub use builder::GraphBuilder;

use crate::{Result, Error};
use crate::resources::ResourceManager;
use std::collections::{HashMap, VecDeque};

/// Render graph for automatic pass ordering and resource management
pub struct RenderGraph {
    passes: Vec<PassNode>,
    execution_order: Vec<usize>,
    transient_resources: HashMap<ResourceHandle, TransientResource>,
}

struct PassNode {
    pass: Box<dyn RenderPass>,
    reads: Vec<ResourceHandle>,
    writes: Vec<ResourceHandle>,
    creates: Vec<ResourceHandle>,
}

struct TransientResource {
    desc: ResourceDesc,
    first_use: usize,  // First pass that uses this resource
    last_use: usize,   // Last pass that uses this resource
}

/// Resource description for graph allocation
#[derive(Clone, Debug)]
pub enum ResourceDesc {
    Texture {
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
    },
    Buffer {
        size: u64,
        usage: wgpu::BufferUsages,
    },
}

impl RenderGraph {
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),
            execution_order: Vec::new(),
            transient_resources: HashMap::new(),
        }
    }

    /// Add a pass to the graph
    pub fn add_pass(&mut self, pass: impl RenderPass + 'static) -> PassId {
        let id = PassId(self.passes.len());

        // Get resource declarations from the pass
        let mut builder = PassResourceBuilder::new();
        pass.declare_resources(&mut builder);

        let node = PassNode {
            pass: Box::new(pass),
            reads: builder.reads,
            writes: builder.writes,
            creates: builder.creates,
        };

        self.passes.push(node);
        id
    }

    /// Build the graph - resolve dependencies and determine execution order
    pub fn build(&mut self) -> Result<()> {
        log::info!("Building render graph with {} passes", self.passes.len());

        // Build dependency graph using topological sort

        // First pass: collect ALL resource writers before building edges.
        // This makes ordering independent of pass registration order.
        let mut resource_writers: HashMap<ResourceHandle, usize> = HashMap::new();
        for (i, pass) in self.passes.iter().enumerate() {
            for &resource in &pass.writes {
                resource_writers.insert(resource, i);
            }
            for &resource in &pass.creates {
                resource_writers.insert(resource, i);
            }
        }

        // Second pass: build adjacency list based on reads
        let mut in_degree = vec![0; self.passes.len()];
        let mut adj_list: Vec<Vec<usize>> = vec![Vec::new(); self.passes.len()];

        for (i, pass) in self.passes.iter().enumerate() {
            for &resource in &pass.reads {
                if let Some(&writer_idx) = resource_writers.get(&resource) {
                    adj_list[writer_idx].push(i);
                    in_degree[i] += 1;
                }
            }
        }

        // Topological sort (Kahn's algorithm — FIFO to preserve insertion order)
        let mut queue: VecDeque<usize> = (0..self.passes.len())
            .filter(|&i| in_degree[i] == 0)
            .collect();

        let mut order = Vec::new();

        while let Some(node) = queue.pop_front() {
            order.push(node);

            for &neighbor in &adj_list[node] {
                in_degree[neighbor] -= 1;
                if in_degree[neighbor] == 0 {
                    queue.push_back(neighbor);
                }
            }
        }

        // Check for cycles
        if order.len() != self.passes.len() {
            return Err(Error::Graph(
                "Cyclic dependency detected in render graph".to_string()
            ));
        }

        self.execution_order = order;

        // Log the execution order
        for (i, &pass_idx) in self.execution_order.iter().enumerate() {
            log::debug!(
                "  Pass {}: {}",
                i,
                self.passes[pass_idx].pass.name()
            );
        }

        // Compute resource lifetimes for transient resources
        self.compute_resource_lifetimes();

        log::info!("Render graph built successfully");
        Ok(())
    }

    /// Compute lifetimes for transient resources
    fn compute_resource_lifetimes(&mut self) {
        let mut resource_usage: HashMap<ResourceHandle, (usize, usize)> = HashMap::new();

        for (exec_idx, &pass_idx) in self.execution_order.iter().enumerate() {
            let pass = &self.passes[pass_idx];

            // Track all resources used by this pass
            for &res in pass.reads.iter().chain(&pass.writes).chain(&pass.creates) {
                resource_usage
                    .entry(res)
                    .and_modify(|(first, last)| {
                        *first = (*first).min(exec_idx);
                        *last = (*last).max(exec_idx);
                    })
                    .or_insert((exec_idx, exec_idx));
            }
        }

        // Store lifetime info for transient resources
        for (&handle, &(first, last)) in &resource_usage {
            if let Some(resource) = self.transient_resources.get_mut(&handle) {
                resource.first_use = first;
                resource.last_use = last;
            }
        }
    }

    /// Execute the render graph
    pub fn execute(&mut self, ctx: &mut GraphContext) -> Result<()> {
        log::trace!("Executing render graph (frame {})", ctx.frame);

        // Clone execution order to avoid borrow checker issues
        let execution_order = self.execution_order.clone();

        // Execute passes in order
        for (exec_idx, &pass_idx) in execution_order.iter().enumerate() {
            let pass_name = self.passes[pass_idx].pass.name().to_string();
            log::trace!("  Executing pass: {}", pass_name);

            // Allocate transient resources for this pass
            self.allocate_transient_resources(exec_idx, ctx)?;

            // Execute the pass
            let mut pass_ctx = PassContext {
                encoder: ctx.encoder,
                resources: ctx.resources,
                target: ctx.target,
                depth_view: ctx.depth_view,
                global_bind_group: ctx.global_bind_group,
                lighting_bind_group: ctx.lighting_bind_group,
                sky_color: ctx.sky_color,
            };

            self.passes[pass_idx].pass.execute(&mut pass_ctx)?;

            // Release transient resources no longer needed
            self.release_transient_resources(exec_idx, ctx)?;
        }

        Ok(())
    }

    fn allocate_transient_resources(&mut self, exec_idx: usize, _ctx: &mut GraphContext) -> Result<()> {
        for (handle, resource) in &self.transient_resources {
            if resource.first_use == exec_idx {
                // Allocate this resource
                log::trace!("    Allocating transient resource {:?}", handle);
                // TODO: Actually allocate from resource manager pool
            }
        }
        Ok(())
    }

    fn release_transient_resources(&mut self, exec_idx: usize, _ctx: &mut GraphContext) -> Result<()> {
        for (handle, resource) in &self.transient_resources {
            if resource.last_use == exec_idx {
                // Release this resource back to pool
                log::trace!("    Releasing transient resource {:?}", handle);
                // TODO: Actually release to resource manager pool
            }
        }
        Ok(())
    }

    /// Declare a transient resource
    pub fn declare_transient(&mut self, handle: ResourceHandle, desc: ResourceDesc) {
        self.transient_resources.insert(
            handle,
            TransientResource {
                desc,
                first_use: usize::MAX,
                last_use: 0,
            },
        );
    }
}

/// Context for graph execution
pub struct GraphContext<'a> {
    pub encoder: &'a mut wgpu::CommandEncoder,
    pub resources: &'a ResourceManager,
    pub target: &'a wgpu::TextureView,
    /// Depth buffer (Depth32Float)
    pub depth_view: &'a wgpu::TextureView,
    pub frame: u64,
    /// Bind group 0 – camera + globals
    pub global_bind_group: &'a wgpu::BindGroup,
    /// Bind group 2 – lights, shadows, env
    pub lighting_bind_group: &'a wgpu::BindGroup,
    /// Sky / background clear color (linear RGB)
    pub sky_color: [f32; 3],
}

/// Builder for declaring pass resource dependencies
pub struct PassResourceBuilder {
    reads: Vec<ResourceHandle>,
    writes: Vec<ResourceHandle>,
    creates: Vec<ResourceHandle>,
}

impl PassResourceBuilder {
    fn new() -> Self {
        Self {
            reads: Vec::new(),
            writes: Vec::new(),
            creates: Vec::new(),
        }
    }

    /// Declare that this pass reads a resource
    pub fn read(&mut self, resource: ResourceHandle) -> &mut Self {
        self.reads.push(resource);
        self
    }

    /// Declare that this pass writes to a resource
    pub fn write(&mut self, resource: ResourceHandle) -> &mut Self {
        self.writes.push(resource);
        self
    }

    /// Declare that this pass creates a transient resource
    pub fn create(&mut self, resource: ResourceHandle) -> &mut Self {
        self.creates.push(resource);
        self
    }
}

impl Default for RenderGraph {
    fn default() -> Self {
        Self::new()
    }
}
