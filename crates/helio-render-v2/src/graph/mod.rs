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
use crate::profiler::GpuProfiler;
use std::collections::{HashMap, HashSet, VecDeque};

/// Render graph for automatic pass ordering and resource management
pub struct RenderGraph {
    passes: Vec<PassNode>,
    execution_order: Vec<usize>,
    transient_resources: HashMap<ResourceHandle, TransientResource>,
    /// Pass names that should be skipped during execution.
    disabled_passes: HashSet<String>,
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
            disabled_passes: HashSet::new(),
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

        // Build dependency graph using topological sort.
        // Dependencies are constructed in pass registration order.
        // Each pass depends on the LAST writer it sees for each resource it reads.
        // This correctly handles read-modify-write chains (e.g., compositing into color_target).

        let mut in_degree = vec![0; self.passes.len()];
        let mut adj_list: Vec<Vec<usize>> = vec![Vec::new(); self.passes.len()];
        let mut last_writer: HashMap<ResourceHandle, usize> = HashMap::new();

        // Collect all writers first so we can resolve explicit single-writer
        // ordering tokens (e.g. "transparent_done") even when the reader pass
        // is registered before the writer pass.
        let mut all_writers: HashMap<ResourceHandle, Vec<usize>> = HashMap::new();
        for (idx, pass) in self.passes.iter().enumerate() {
            for &resource in pass.writes.iter().chain(&pass.creates) {
                all_writers.entry(resource).or_default().push(idx);
            }
        }

        for (i, pass) in self.passes.iter().enumerate() {
            // Add edges from all writers of resources this pass reads
            for &resource in &pass.reads {
                let mut add_edge = |from: usize, to: usize| {
                    if from != to && !adj_list[from].contains(&to) {
                        adj_list[from].push(to);
                        in_degree[to] += 1;
                    }
                };

                // If a resource has exactly one writer globally, this read should
                // depend on it regardless of registration order.
                if let Some(writers) = all_writers.get(&resource) {
                    if writers.len() == 1 {
                        add_edge(writers[0], i);
                        continue;
                    }
                }

                // Multi-writer resource: preserve existing behavior that depends on
                // the latest writer seen so far in registration order.
                if let Some(&writer_idx) = last_writer.get(&resource) {
                    add_edge(writer_idx, i);
                }
            }

            // Register this pass as the latest writer for all resources it writes/creates
            for &resource in pass.writes.iter().chain(&pass.creates) {
                last_writer.insert(resource, i);
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
    pub fn execute(&mut self, ctx: &mut GraphContext, profiler: Option<&mut GpuProfiler>) -> Result<()> {
        log::trace!("Executing render graph (frame {})", ctx.frame);

        // Convert profiler to raw pointer so we can freely split borrows with
        // ctx.encoder inside the loop.  This is sound: profiler outlives the loop,
        // and it is only ever accessed from this (single) thread.
        let profiler_raw: *mut GpuProfiler =
            profiler.map(|p| p as *mut GpuProfiler).unwrap_or(std::ptr::null_mut());

        // Execute passes in order — iterate by index so we don't hold a borrow
        // on self.execution_order while calling &mut self methods inside the loop.
        let pass_count = self.execution_order.len();
        for exec_idx in 0..pass_count {
            let pass_idx = self.execution_order[exec_idx];
            let pass_name = self.passes[pass_idx].pass.name();
            log::trace!("  Executing pass: {}", pass_name);

            // Allocate query slots and record begin timestamp
            let slot_pair: Option<(u32, u32)> = if !profiler_raw.is_null() {
                // SAFETY: profiler_raw is valid, non-null, and exclusively accessed here.
                let p = unsafe { &mut *profiler_raw };
                // allocate_scope takes &str and stores name.to_string() only when profiling.
                p.allocate_scope(pass_name).map(|(b, e)| {
                    ctx.encoder.write_timestamp(p.query_set(), b);
                    (b, e)
                })
            } else {
                None
            };

            // Skip disabled passes
            if self.disabled_passes.contains(pass_name) {
                log::trace!("  Skipping disabled pass: {}", pass_name);
                continue;
            }

            let cpu_start = crate::time::Instant::now();

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
                has_sky: ctx.has_sky,
                sky_state_changed: ctx.sky_state_changed,
                sky_bind_group: ctx.sky_bind_group,
                profiler: profiler_raw,
                camera_position: ctx.camera_position,
                camera_forward: ctx.camera_forward,
                draw_list_generation: ctx.draw_list_generation,
                transparent_start: ctx.transparent_start,
            };

            self.passes[pass_idx].pass.execute(&mut pass_ctx)?;

            let cpu_ms = cpu_start.elapsed().as_secs_f32() * 1000.0;

            // Record end timestamp and store CPU time
            if !profiler_raw.is_null() {
                // SAFETY: same as above.
                let p = unsafe { &mut *profiler_raw };
                if let Some((_, end)) = slot_pair {
                    ctx.encoder.write_timestamp(p.query_set(), end);
                }
                p.set_last_scope_cpu_ms(cpu_ms);
            }

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

    /// Return current execution order as pass names.
    pub fn execution_pass_names(&self) -> Vec<String> {
        self.execution_order
            .iter()
            .map(|&idx| self.passes[idx].pass.name().to_string())
            .collect()
    }

    /// Toggle a pass on/off by name. Returns the new enabled state.
    pub fn toggle_pass(&mut self, name: &str) -> bool {
        if self.disabled_passes.contains(name) {
            self.disabled_passes.remove(name);
            true
        } else {
            self.disabled_passes.insert(name.to_string());
            false
        }
    }

    /// Set whether a pass is enabled.
    pub fn set_pass_enabled(&mut self, name: &str, enabled: bool) {
        if enabled {
            self.disabled_passes.remove(name);
        } else {
            self.disabled_passes.insert(name.to_string());
        }
    }

    /// Returns true if the named pass is currently enabled.
    pub fn is_pass_enabled(&self, name: &str) -> bool {
        !self.disabled_passes.contains(name)
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
    /// Sky / background clear color (linear RGB) – used when has_sky=false
    pub sky_color: [f32; 3],
    /// True when a SkyAtmosphere is present – SkyPass renders; GeometryPass uses LoadOp::Load
    pub has_sky: bool,
    /// True when sky atmosphere parameters changed since the last frame.
    pub sky_state_changed: bool,
    /// Sky bind group (group 1 in SkyPass pipeline), None when no sky
    pub sky_bind_group: Option<&'a wgpu::BindGroup>,
    /// Camera world-space position for distance-based culling
    pub camera_position: glam::Vec3,
    /// Camera forward direction for view-depth sorting/culling
    pub camera_forward: glam::Vec3,
    /// Monotonically increasing counter from the Renderer.  Passes compare this
    /// against their cached generation to decide whether to rebuild RenderBundles.
    pub draw_list_generation: u64,
    /// Index of the first transparent entry in the draw list.
    /// `draw_list[0..transparent_start]` = opaque; `[transparent_start..]` = transparent.
    /// Transparent draws are already appended after opaque by the GPU scene;
    /// passes use this to skip the O(N_total) scan when looking for transparent items.
    pub transparent_start: usize,
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
