//! Radiance Cascades global illumination feature
//!
//! Surface-based irradiance probes arranged in 5 hierarchical cascade levels.
//! Uses hardware ray tracing (EXPERIMENTAL_RAY_QUERY) to trace one ray per probe
//! direction, then merges cascades from coarse to fine.

use super::{FeatureContext, PrepareContext};
use crate::features::{Feature, ShaderDefine};
use crate::passes::RadianceCascadesPass;
use crate::Result;
use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;
use std::sync::{Arc, atomic::AtomicU32};

pub const CASCADE_COUNT: usize = 4;
/// Probe grid dimension per cascade (cubed = total probes per cascade)
/// dir_dim starts at 4 so cascade 0 has upper+lower hemisphere bins (Y-up oct encoding)
pub const PROBE_DIMS: [u32; CASCADE_COUNT] = [16, 8, 4, 2];
/// Direction bins per atlas axis per cascade (start at 4, not 2)
pub const DIR_DIMS: [u32; CASCADE_COUNT] = [4, 8, 16, 32];
/// Maximum ray distances per cascade (metres)
pub const T_MAXS: [f32; CASCADE_COUNT] = [0.5, 1.0, 2.0, 1000.0];

/// Atlas width = PROBE_DIM * DIR_DIM = 64 for every cascade
pub const ATLAS_W: u32 = 64;
/// Atlas heights per cascade: PROBE_DIM² * DIR_DIM
pub const ATLAS_HEIGHTS: [u32; CASCADE_COUNT] = [1024, 512, 256, 128];

/// GPU-side dynamic RC uniforms uploaded every frame
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct RCDynamic {
    pub world_min:   [f32; 4],
    pub world_max:   [f32; 4],
    pub frame:       u32,
    pub light_count: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

/// GPU-side static per-cascade uniforms (constant after register)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct CascadeStatic {
    pub cascade_index:    u32,
    pub probe_dim:        u32,
    pub dir_dim:          u32,
    pub t_max_bits:       u32,
    pub parent_probe_dim: u32,
    pub parent_dir_dim:   u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

/// Radiance Cascades global illumination feature.
///
/// Call `with_world_bounds()` to set the scene AABB that the probes cover.
pub struct RadianceCascadesFeature {
    enabled: bool,
    world_min: [f32; 3],
    world_max: [f32; 3],
    // Kept alive across frames (GPU objects held here)
    _cascade_textures: Vec<wgpu::Texture>,
    cascade_arc_views: Vec<Arc<wgpu::TextureView>>,
    _dummy_tex: Option<wgpu::Texture>,
    _dummy_view: Option<wgpu::TextureView>,
    _static_bufs: Vec<wgpu::Buffer>,
    rc_dynamic_buf: Option<Arc<wgpu::Buffer>>,
    light_count_arc: Option<Arc<AtomicU32>>,
}

impl RadianceCascadesFeature {
    pub fn new() -> Self {
        Self {
            enabled: true,
            world_min: [-10.0, -1.0, -10.0],
            world_max: [10.0, 10.0, 10.0],
            _cascade_textures: Vec::new(),
            cascade_arc_views: Vec::new(),
            _dummy_tex: None,
            _dummy_view: None,
            _static_bufs: Vec::new(),
            rc_dynamic_buf: None,
            light_count_arc: None,
        }
    }

    /// Set the world-space AABB that the probe grid covers.
    pub fn with_world_bounds(mut self, min: [f32; 3], max: [f32; 3]) -> Self {
        self.world_min = min;
        self.world_max = max;
        self
    }

}

impl Default for RadianceCascadesFeature {
    fn default() -> Self { Self::new() }
}

impl Feature for RadianceCascadesFeature {
    fn name(&self) -> &str { "radiance_cascades" }

    fn register(&mut self, ctx: &mut FeatureContext) -> Result<()> {
        use wgpu::util::DeviceExt;
        let device = ctx.device;

        // ── Cascade textures ────────────────────────────────────────────────
        let mut cascade_textures: Vec<wgpu::Texture> = Vec::with_capacity(CASCADE_COUNT);
        let mut cascade_arc_views: Vec<Arc<wgpu::TextureView>> = Vec::with_capacity(CASCADE_COUNT);
        for c in 0..CASCADE_COUNT {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("RC Cascade {c}")),
                size: wgpu::Extent3d { width: ATLAS_W, height: ATLAS_HEIGHTS[c], depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            });
            let view = Arc::new(tex.create_view(&wgpu::TextureViewDescriptor::default()));
            cascade_textures.push(tex);
            cascade_arc_views.push(view);
        }

        // Dummy parent texture for cascade 4 (no parent)
        let dummy_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("RC Dummy Parent"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let dummy_view = dummy_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // ── Dynamic uniform buffer ───────────────────────────────────────────
        let rc_dyn_buf = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RC Dynamic Uniform"),
            size: std::mem::size_of::<RCDynamic>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // ── Static per-cascade buffers ───────────────────────────────────────
        let mut static_bufs: Vec<wgpu::Buffer> = Vec::with_capacity(CASCADE_COUNT);
        for c in 0..CASCADE_COUNT {
            let (parent_probe, parent_dir) = if c + 1 < CASCADE_COUNT {
                (PROBE_DIMS[c + 1], DIR_DIMS[c + 1])
            } else {
                (0, 0)
            };
            let data = CascadeStatic {
                cascade_index: c as u32,
                probe_dim: PROBE_DIMS[c],
                dir_dim: DIR_DIMS[c],
                t_max_bits: T_MAXS[c].to_bits(),
                parent_probe_dim: parent_probe,
                parent_dir_dim: parent_dir,
                _pad0: 0,
                _pad1: 0,
            };
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("RC Static {c}")),
                contents: bytemuck::bytes_of(&data),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            static_bufs.push(buf);
        }

        // ── TLAS ─────────────────────────────────────────────────────────────
        let tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
            label: Some("RC TLAS"),
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            max_instances: 256,
        });

        // ── Light buffer (may be None if LightingFeature not registered) ─────
        use crate::features::lighting::MAX_LIGHTS;
        use crate::features::lighting::GpuLight;
        let light_buf: Arc<wgpu::Buffer> = ctx.light_buffer.clone().unwrap_or_else(|| {
            Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("RC Null Light Buffer"),
                size: (std::mem::size_of::<GpuLight>() * MAX_LIGHTS as usize) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }))
        });

        // ── Compute pipeline (layout: None = auto from shader) ───────────────
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rc_trace"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/passes/rc_trace.wgsl").into(),
            ),
        });
        let pipeline = Arc::new(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rc_trace"),
            layout: None,
            module: &shader,
            entry_point: Some("cs_trace"),
            compilation_options: Default::default(),
            cache: None,
        }));
        let bg_layout = pipeline.get_bind_group_layout(0);

        // ── Bind groups (one per cascade: C writes cascade[C], reads cascade[C+1]) ─
        let bind_groups: Vec<wgpu::BindGroup> = (0..CASCADE_COUNT)
            .map(|c| {
                let parent_view: &wgpu::TextureView = if c + 1 < CASCADE_COUNT {
                    &cascade_arc_views[c + 1]
                } else {
                    &dummy_view
                };
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("RC BG {c}")),
                    layout: &bg_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&cascade_arc_views[c]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(parent_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: rc_dyn_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: static_bufs[c].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::AccelerationStructure(&tlas),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: light_buf.as_entire_binding(),
                        },
                    ],
                })
            })
            .collect();

        // ── Set context outputs ──────────────────────────────────────────────
        ctx.rc_cascade0_view = Some(cascade_arc_views[0].clone());
        ctx.rc_world_bounds = Some((self.world_min, self.world_max));

        // Keep GPU objects alive in the feature struct
        self.cascade_arc_views = cascade_arc_views;
        self._cascade_textures = cascade_textures;
        self._dummy_view = Some(dummy_view);
        self._dummy_tex = Some(dummy_tex);
        self._static_bufs = static_bufs;
        self.rc_dynamic_buf = Some(rc_dyn_buf.clone());
        self.light_count_arc = Some(ctx.light_count_arc.clone());

        // ── Register the pass ────────────────────────────────────────────────
        ctx.graph.add_pass(RadianceCascadesPass::new(
            ctx.device_arc.clone(),
            ctx.draw_list.clone(),
            pipeline,
            bind_groups,
            rc_dyn_buf,
            tlas,
            self.world_min,
            self.world_max,
        ));

        log::info!(
            "RadianceCascades feature registered: {} cascades, world [{:?} .. {:?}]",
            CASCADE_COUNT, self.world_min, self.world_max,
        );
        Ok(())
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> Result<()> {
        let Some(buf) = &self.rc_dynamic_buf else { return Ok(()); };
        let light_count = self.light_count_arc.as_ref()
            .map(|a| a.load(std::sync::atomic::Ordering::Relaxed))
            .unwrap_or(0);
        let dyn_data = RCDynamic {
            world_min:   [self.world_min[0], self.world_min[1], self.world_min[2], 0.0],
            world_max:   [self.world_max[0], self.world_max[1], self.world_max[2], 0.0],
            frame: ctx.frame as u32,
            light_count,
            _pad0: 0, _pad1: 0,
        };
        ctx.queue.write_buffer(buf, 0, bytemuck::bytes_of(&dyn_data));
        Ok(())
    }

    fn shader_defines(&self) -> HashMap<String, ShaderDefine> {
        HashMap::new()
    }

    fn is_enabled(&self) -> bool { self.enabled }
    fn set_enabled(&mut self, enabled: bool) { self.enabled = enabled; }

    fn cleanup(&mut self, _device: &wgpu::Device) {
        self._cascade_textures.clear();
        self.cascade_arc_views.clear();
        self._dummy_tex = None;
        self._dummy_view = None;
        self._static_bufs.clear();
        self.rc_dynamic_buf = None;
    }
}
