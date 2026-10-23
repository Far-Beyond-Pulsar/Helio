//! The scene's meshes as a tracer of the probes' rays (`gi_mesh_trace.wgsl`,
//! README.md "Tracers"): hardware ray queries against the frame's TLAS
//! (`RenderEnvironment::tlas`), shaded from its instances' surfaces
//! (`helio_core::RayInstance`, `ray_instances`). It runs after the other
//! tracers and before [`crate::GiResolvePass`]; on a frame without a TLAS
//! or its rows it leaves the hits as they are.

use crate::{buffer, storage, uniform, FRAME_BYTES, GI_FRAME, GI_HITS, GI_LIGHT, GI_RAYS, GI_SUN, MAX_RAYS};
use bytemuck::{Pod, Zeroable};
use helio_core::graph::ResourceBuilder;
use helio_core::{PassContext, PrepareContext, RenderPass, ResourceKey, Result as HelioResult};
use helio_pass_shadow_matrix::{SUN_OCCLUSION_FRAME, SUN_OCCLUSION_FRAME_BYTES, SUN_OCCLUSION_MAPS, SUN_OCCLUSION_SNIPPET};

/// The scene's point and spot lights a hit chooses from: the first so many
/// live ones (`gi_gather_lights`).
pub const MAX_LIGHTS: u32 = 256;
/// Bytes of the gathered lights: a count, then 64 bytes a light.
const LOCAL_LIGHTS_BYTES: u64 = 16 + 64 * MAX_LIGHTS as u64;

/// `MeshTrace` (`gi_mesh_trace.wgsl`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuTrace {
    eye: [f32; 4],
    info: [u32; 4],
}

/// The meshes' tracer (module docs).
pub struct GiMeshTracePass {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    gather_layout: wgpu::BindGroupLayout,
    gather: wgpu::ComputePipeline,
    trace: wgpu::Buffer,
    local_lights: wgpu::Buffer,
    /// No terrain: its occlusion frame says no levels.
    no_occlusion: (wgpu::Buffer, wgpu::TextureView),
    no_lights: wgpu::Buffer,
}

impl GiMeshTracePass {
    /// The tracer, on a device with hardware ray queries (none without:
    /// the probes then see only the other tracers' geometry).
    pub fn new(device: &wgpu::Device) -> Option<Self> {
        if !device.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY) || device.limits().max_tlas_instance_count == 0 {
            return None;
        }
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GI mesh rays"),
            entries: &[
                uniform(0, FRAME_BYTES),
                storage(1, true),
                storage(2, false),
                storage(3, true),
                storage(4, false),
                uniform(5, std::mem::size_of::<GpuTrace>() as u64),
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::AccelerationStructure { vertex_return: false },
                    count: None,
                },
                storage(7, true),
                storage(8, true),
                storage(9, true),
                uniform(11, SUN_OCCLUSION_FRAME_BYTES),
                wgpu::BindGroupLayoutEntry {
                    binding: 12,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                storage(13, false),
            ],
        });
        let gather_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GI mesh lights"),
            entries: &[uniform(5, std::mem::size_of::<GpuTrace>() as u64), storage(10, true), storage(13, false)],
        });
        let module = helio_core::shader::module_with(
            device,
            "GI mesh rays",
            helio_core::include_wgsl!("../shaders/gi_mesh_trace.wgsl"),
            &[crate::GI_COMMON_SNIPPET, SUN_OCCLUSION_SNIPPET],
        );
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("GI mesh rays"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = helio_core::shader::create_compute_pipeline(device, wgpu::ComputePipelineDescriptor {
            label: Some("gi_mesh_trace"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("gi_mesh_trace"),
            compilation_options: Default::default(),
            cache: None,
        });
        let gather_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("GI mesh lights"),
            bind_group_layouts: &[Some(&gather_layout)],
            immediate_size: 0,
        });
        let gather = helio_core::shader::create_compute_pipeline(device, wgpu::ComputePipelineDescriptor {
            label: Some("gi_gather_lights"),
            layout: Some(&gather_pipeline_layout),
            module: &module,
            entry_point: Some("gi_gather_lights"),
            compilation_options: Default::default(),
            cache: None,
        });
        let maps = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("GI no sun occlusion"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: helio_pass_shadow_matrix::SUN_OCCLUSION_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let maps = maps.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() });
        let uniform_usage = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
        Some(Self {
            layout,
            pipeline,
            gather_layout,
            gather,
            local_lights: buffer(device, "GI mesh local lights", LOCAL_LIGHTS_BYTES, wgpu::BufferUsages::STORAGE),
            trace: buffer(device, "GI mesh trace", std::mem::size_of::<GpuTrace>() as u64, uniform_usage),
            no_occlusion: (buffer(device, "GI no sun occlusion", SUN_OCCLUSION_FRAME_BYTES, uniform_usage), maps),
            no_lights: buffer(device, "GI mesh no lights", 128, wgpu::BufferUsages::STORAGE),
        })
    }

    /// Records the meshes' hits for this frame's rays: `gi` is GiPass's
    /// frame, rays, hits, sun and hits' light; `eye` the eye in the TLAS's
    /// frame; `instances`, `vertices` and `indices` the TLAS's rows and the
    /// scene's mesh pools; `lights` the scene's (its first `count` rows);
    /// `occlusion` the sun's occlusion maps, if published.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cmds: &mut helio_core::CommandRecorder<'_>,
        [frame, rays, hits, sun, light]: [&wgpu::Buffer; 5],
        tlas: &wgpu::Tlas,
        eye: [f32; 3],
        [instances, vertices, indices]: [&wgpu::Buffer; 3],
        lights: Option<(&wgpu::Buffer, u32)>,
        occlusion: Option<(&wgpu::Buffer, &wgpu::TextureView)>,
        seed: u32,
    ) {
        let (lights, count) = lights.unwrap_or((&self.no_lights, 0));
        let count = count.min((lights.size() / 128) as u32);
        let (occlusion_frame, occlusion_maps) = occlusion.unwrap_or((&self.no_occlusion.0, &self.no_occlusion.1));
        let trace = GpuTrace { eye: [eye[0], eye[1], eye[2], 0.0], info: [count, seed, 0, 0] };
        helio_core::upload::write_buffer(queue, &self.trace, 0, bytemuck::bytes_of(&trace));
        fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry { binding, resource: buffer.as_entire_binding() }
        }
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GI mesh rays"),
            layout: &self.layout,
            entries: &[
                entry(0, frame),
                entry(1, rays),
                entry(2, hits),
                entry(3, sun),
                entry(4, light),
                entry(5, &self.trace),
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::AccelerationStructure(tlas) },
                entry(7, instances),
                entry(8, vertices),
                entry(9, indices),
                entry(11, occlusion_frame),
                wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::TextureView(occlusion_maps) },
                entry(13, &self.local_lights),
            ],
        });
        let gather = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GI mesh lights"),
            layout: &self.gather_layout,
            entries: &[entry(5, &self.trace), entry(10, lights), entry(13, &self.local_lights)],
        });
        let mut pass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("GI mesh rays"), timestamp_writes: None });
        pass.set_bind_group(0, Some(&gather), &[]);
        pass.set_pipeline(&self.gather);
        pass.dispatch_workgroups(1, 1, 1);
        pass.set_bind_group(0, Some(&group), &[]);
        pass.set_pipeline(&self.pipeline);
        // Every ray slot: those past this frame's count return at once.
        pass.dispatch_workgroups(MAX_RAYS.div_ceil(64), 1, 1);
    }
}

impl RenderPass for GiMeshTracePass {
    fn name(&self) -> &'static str {
        "GIMeshRays"
    }

    fn reads(&self) -> &'static [&'static str] {
        &[GI_FRAME, GI_RAYS, GI_HITS, GI_SUN, GI_LIGHT, "render_environment", helio_core::RAY_INSTANCES, SUN_OCCLUSION_MAPS, SUN_OCCLUSION_FRAME]
    }

    fn writes(&self) -> &'static [&'static str] {
        &[GI_HITS, GI_LIGHT]
    }

    fn declare_resources(&self, _builder: &mut ResourceBuilder) {}

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a helio_core::ResourceRegistry<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn prepare(&mut self, _ctx: &PrepareContext) -> HelioResult<()> {
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let binding = |name: &str| ctx.registry.buffer_binding(name);
        let (Some(frame), Some(rays), Some(hits), Some(sun), Some(light)) = (binding(GI_FRAME), binding(GI_RAYS), binding(GI_HITS), binding(GI_SUN), binding(GI_LIGHT)) else {
            return Ok(());
        };
        let Some(tlas) = ctx.registry.get::<helio_core::RenderEnvironment>(helio_core::resource_keys::render_environment()).and_then(|e| e.tlas) else {
            return Ok(());
        };
        let Some(instances) = ctx.registry.get::<&wgpu::Buffer>(ResourceKey::new(helio_core::RAY_INSTANCES)) else {
            return Ok(());
        };
        let pool = |name: &'static str| ctx.scene_buffers.get(helio_core::BufferKey::of(name)).map(|h| h.buffer.clone());
        let (Some(vertices), Some(indices)) = (pool("builtin_mesh_vertex"), pool("builtin_mesh_index")) else {
            return Ok(());
        };
        let lights = pool("scene_lights");
        let occlusion = match (binding(SUN_OCCLUSION_FRAME), ctx.registry.texture_binding(SUN_OCCLUSION_MAPS)) {
            (Some(frame), Some(maps)) => Some((frame, maps)),
            _ => None,
        };
        let camera = ctx.camera_data;
        let eye = [camera.position_near[0], camera.position_near[1], camera.position_near[2]];
        let seed = (ctx.frame_num as u32).wrapping_mul(0x9e37_79b9);
        let mut cmds = ctx.graphics_cmds();
        self.record(ctx.device, ctx.queue, &mut cmds, [frame, rays, hits, sun, light], tlas, eye, [instances, &vertices, &indices], lights.as_ref().map(|l| (l, u32::MAX)), occlusion, seed);
        Ok(())
    }
}
