//! Billboard rendering feature

use super::{FeatureContext, PrepareContext};
use crate::features::{Feature, ShaderDefine};
use crate::passes::BillboardPass;
use crate::Result;
use std::collections::HashMap;
use std::sync::Arc;

/// A single billboard to render
#[derive(Clone, Debug)]
pub struct BillboardInstance {
    /// World-space position
    pub position: [f32; 3],
    /// Width and height in world units (or angular units when `screen_scale` is true)
    pub scale: [f32; 2],
    /// RGBA tint (multiplied with billboard texture / color)
    pub color: [f32; 4],
    /// When true the size is kept constant in screen-space
    pub screen_scale: bool,
}

impl BillboardInstance {
    pub fn new(position: [f32; 3], scale: [f32; 2]) -> Self {
        Self {
            position,
            scale,
            color: [1.0, 1.0, 1.0, 1.0],
            screen_scale: false,
        }
    }

    pub fn with_color(mut self, color: [f32; 4]) -> Self {
        self.color = color;
        self
    }

    pub fn with_screen_scale(mut self, enabled: bool) -> Self {
        self.screen_scale = enabled;
        self
    }
}

/// GPU instance data (must match WGSL struct)
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GpuBillboardInstance {
    pub position: [f32; 3],
    pub _pad1: f32,
    pub scale: [f32; 2],
    pub screen_scale: u32,
    pub _pad2: u32,
    pub color: [f32; 4],
}

/// Billboard quad vertex (local-space, unit size)
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct BillboardVertex {
    pub position: [f32; 2],
    pub uv: [f32; 2],
}

/// Billboard rendering feature
///
/// Renders camera-facing quads at arbitrary world positions.  Useful for
/// particles, editor gizmos, sprites, lens flares, etc.
///
/// The feature creates a shared quad mesh and a per-instance VERTEX buffer that
/// it re-uploads every frame via `prepare()`.  A `BillboardPass` (registered
/// during `register()`) reads those buffers and issues one instanced draw call.
pub struct BillboardsFeature {
    enabled: bool,
    instances: Vec<BillboardInstance>,
    max_instances: u32,
    // Shared with BillboardPass via Arc
    vertex_buffer: Option<Arc<wgpu::Buffer>>,
    index_buffer: Option<Arc<wgpu::Buffer>>,
    instance_buffer: Option<Arc<wgpu::Buffer>>,
    instance_count: Arc<std::sync::atomic::AtomicU32>,
}

impl BillboardsFeature {
    pub fn new() -> Self {
        Self {
            enabled: true,
            instances: Vec::new(),
            max_instances: 1024,
            vertex_buffer: None,
            index_buffer: None,
            instance_buffer: None,
            instance_count: Arc::new(std::sync::atomic::AtomicU32::new(0)),
        }
    }

    pub fn with_max_instances(mut self, max: u32) -> Self {
        self.max_instances = max;
        self
    }

    pub fn add_billboard(&mut self, instance: BillboardInstance) {
        self.instances.push(instance);
    }

    pub fn set_billboards(&mut self, instances: Vec<BillboardInstance>) {
        self.instances = instances;
    }

    pub fn clear_billboards(&mut self) {
        self.instances.clear();
    }

    pub fn billboards(&self) -> &[BillboardInstance] {
        &self.instances
    }
}

impl Default for BillboardsFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl Feature for BillboardsFeature {
    fn name(&self) -> &str {
        "billboards"
    }

    fn register(&mut self, ctx: &mut FeatureContext) -> Result<()> {
        use wgpu::util::DeviceExt;

        // Unit quad in local space (-0.5 .. +0.5), Y-up
        let vertices = [
            BillboardVertex { position: [-0.5, -0.5], uv: [0.0, 1.0] },
            BillboardVertex { position: [ 0.5, -0.5], uv: [1.0, 1.0] },
            BillboardVertex { position: [ 0.5,  0.5], uv: [1.0, 0.0] },
            BillboardVertex { position: [-0.5,  0.5], uv: [0.0, 0.0] },
        ];
        let indices: [u32; 6] = [0, 1, 2, 0, 2, 3];

        let vertex_buffer = Arc::new(ctx.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Billboard Quad Vertices"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            },
        ));

        let index_buffer = Arc::new(ctx.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Billboard Quad Indices"),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX,
            },
        ));

        let instance_buffer = Arc::new(ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Billboard Instance Buffer"),
            size: std::mem::size_of::<GpuBillboardInstance>() as u64 * self.max_instances as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // Register the billboard pass with shared buffer refs and the atomic counter
        ctx.graph.add_pass(BillboardPass::new(
            vertex_buffer.clone(),
            index_buffer.clone(),
            instance_buffer.clone(),
            self.instance_count.clone(),
            ctx.surface_format,
        ));

        self.vertex_buffer = Some(vertex_buffer);
        self.index_buffer = Some(index_buffer);
        self.instance_buffer = Some(instance_buffer);

        log::info!(
            "Billboards feature registered: max {} instances",
            self.max_instances
        );
        Ok(())
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> Result<()> {
        let count = self.instances.len().min(self.max_instances as usize);

        if let Some(buf) = &self.instance_buffer {
            if count > 0 {
                let gpu: Vec<GpuBillboardInstance> = self.instances[..count]
                    .iter()
                    .map(|b| GpuBillboardInstance {
                        position: b.position,
                        _pad1: 0.0,
                        scale: b.scale,
                        screen_scale: b.screen_scale as u32,
                        _pad2: 0,
                        color: b.color,
                    })
                    .collect();
                ctx.queue.write_buffer(buf, 0, bytemuck::cast_slice(&gpu));
            }
        }

        self.instance_count
            .store(count as u32, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    fn shader_defines(&self) -> HashMap<String, ShaderDefine> {
        HashMap::new()
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}
