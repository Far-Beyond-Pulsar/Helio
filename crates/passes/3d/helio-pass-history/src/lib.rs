//! The frame's pixel history (`helio_frame_api::history`), as the spine
//! publishes it: [`HistoryReprojectionPass`] (the reprojection uniform, CPU
//! only) and [`HistoryPixelsPass`] (where each pixel was, and the guides the
//! next frame compares against). [`publisher`] is what the spine registers;
//! a reader asks through `history::request`, and only a graph with a reader
//! schedules either.
//!
//! **Per view.** A graph keeps the history of every view it renders, by the
//! camera's view id (`GpuCameraUniforms::jitter_frame[3]`): mono, or each eye
//! of a sequential XR frame; a capture is a graph of its own. The renderer
//! keeps each view's previous view-projection the same way
//! (`prev_view_proj`); this keeps its previous view matrix, its origin and
//! its guides. A view unseen for [`MAX_VIEWS`] others is dropped.
//!
//! **Cost.** The reprojection is a 128-byte upload a frame. The pixels pass
//! is one dispatch over the internal pixels: a depth, normal, roughness and
//! velocity read, the neighbours' depth, normal and reprojection once a
//! texel into workgroup memory (an 8x8 tile and its ring), four reads of each
//! previous guide, three writes. Memory: 4 + 8 bytes a pixel of guides,
//! twice (this frame's and the previous), and 8 of pixels: 32 bytes a pixel
//! a view (66 MB at 1920x1080).

use std::sync::{Arc, Mutex};

use bytemuck::{Pod, Zeroable};
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};
use helio_feature_api::NodeBuild;
use helio_frame_api::history::{
    self, rebase_previous_view, HistoryPublisher, HistoryReprojection, DEPTH, NORMAL, PIXELS,
    PREVIOUS_DEPTH, PREVIOUS_NORMAL, REPROJECTION,
};

/// Views a graph keeps history for at once (mono, two eyes and a few
/// cameras a host alternates between).
pub const MAX_VIEWS: usize = 8;

/// One view's history.
struct View {
    id: u32,
    /// The frame (`frame_num`) it last rendered.
    used: u64,
    reprojection: wgpu::Buffer,
    /// What it rendered last: its view matrix and the frame origin then.
    previous_view: [f32; 16],
    previous_origin: glam::DVec3,
    /// Frames it has rendered (saturating).
    frames: u32,
    guides: Option<Guides>,
}

/// A view's guides at one size: this frame's and the previous frame's
/// (`latest` the last written), and where its pixels were.
struct Guides {
    size: [u32; 2],
    depth: [wgpu::TextureView; 2],
    normal: [wgpu::TextureView; 2],
    pixels: wgpu::TextureView,
    pixels_texture: wgpu::Texture,
    /// The slot the last frame wrote, and the view's `frames` then.
    latest: usize,
    written_at: Option<u32>,
}

impl Guides {
    fn new(device: &wgpu::Device, size: [u32; 2]) -> Self {
        let make = |label: &str, format: wgpu::TextureFormat| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: size[0].max(1),
                    height: size[1].max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                // Copies: tests read the history back.
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        let texture = |label: &str, format| make(label, format).create_view(&Default::default());
        let pixels_texture = make("History pixels", history::PIXELS_FORMAT.to_wgpu());
        let depth = history::PREVIOUS_DEPTH_FORMAT.to_wgpu();
        let normal = history::PREVIOUS_NORMAL_FORMAT.to_wgpu();
        Self {
            size,
            depth: [
                texture("History depth A", depth),
                texture("History depth B", depth),
            ],
            normal: [
                texture("History normal A", normal),
                texture("History normal B", normal),
            ],
            pixels: pixels_texture.create_view(&Default::default()),
            pixels_texture,
            latest: 0,
            written_at: None,
        }
    }
}

/// What a graph's two history passes share.
#[derive(Default)]
struct Views {
    views: Vec<View>,
    /// The view this execution renders (its index in `views`).
    active: Option<usize>,
}

impl Views {
    fn active(&mut self) -> Option<&mut View> {
        self.active.map(|i| &mut self.views[i])
    }

    /// The view `id`, made on its first frame; the least recently used one
    /// leaves past [`MAX_VIEWS`].
    fn view(&mut self, device: &wgpu::Device, id: u32, frame: u64) -> usize {
        if let Some(i) = self.views.iter().position(|v| v.id == id) {
            self.views[i].used = frame;
            return i;
        }
        if self.views.len() >= MAX_VIEWS {
            let oldest = (0..self.views.len())
                .min_by_key(|&i| self.views[i].used)
                .expect("a full list has a view");
            self.views.swap_remove(oldest);
        }
        self.views.push(View {
            id,
            used: frame,
            reprojection: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("History reprojection"),
                size: HistoryReprojection::BYTES,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            previous_view: [0.0; 16],
            previous_origin: glam::DVec3::ZERO,
            frames: 0,
            guides: None,
        });
        self.views.len() - 1
    }
}

type Shared = Arc<Mutex<Views>>;

/// The views of the graph being built, shared by its two passes.
#[derive(Default)]
struct GraphViews(Option<Shared>);

fn shared(build: &mut NodeBuild<'_, '_>) -> Shared {
    build
        .env
        .shared(GraphViews::default)
        .0
        .get_or_insert_with(Default::default)
        .clone()
}

/// The publisher the spine registers (`history::RegisterHistory`).
pub fn publisher() -> HistoryPublisher {
    HistoryPublisher {
        reprojection: Arc::new(|build| {
            let views = shared(build);
            build
                .graph
                .add_pass(Box::new(HistoryReprojectionPass { views }));
            Ok(())
        }),
        pixels: Arc::new(|build| {
            let views = shared(build);
            let pass = HistoryPixelsPass::new(build.env.device, views);
            build.graph.add_pass(Box::new(pass));
            Ok(())
        }),
    }
}

/// The reprojection of the view a frame renders ([`REPROJECTION`]).
pub struct HistoryReprojectionPass {
    views: Shared,
}

impl HistoryReprojectionPass {
    /// A pass of its own graph (for a host or a test that builds no
    /// feature).
    pub fn standalone() -> Self {
        Self {
            views: Default::default(),
        }
    }

    /// The reprojection uniform of the view the last frame rendered.
    pub fn buffer(&self) -> Option<wgpu::Buffer> {
        let mut views = self.views.lock().unwrap();
        views.active().map(|v| v.reprojection.clone())
    }
}

impl RenderPass for HistoryReprojectionPass {
    fn name(&self) -> &'static str {
        "HistoryReprojection"
    }

    fn writes(&self) -> &'static [&'static str] {
        &[REPROJECTION]
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let camera = ctx.camera_data;
        let id = camera.jitter_frame[3].to_bits();
        let origin = ctx.world_origin;
        let mut views = self.views.lock().unwrap();
        let index = views.view(ctx.device, id, ctx.frame_num);
        views.active = Some(index);
        let view = &mut views.views[index];
        let has_previous = view.frames > 0;
        // The previous view in this frame's coordinates; a view's first
        // frame has none (its previous matrices are this frame's).
        let previous_view = if has_previous {
            let shift = origin - view.previous_origin;
            rebase_previous_view(view.previous_view, shift.to_array())
        } else {
            camera.view
        };
        let reprojection = HistoryReprojection::new(
            camera.view_proj,
            [
                camera.position_near[0],
                camera.position_near[1],
                camera.position_near[2],
            ],
            camera.prev_view_proj,
            previous_view,
        )
        .with_previous(has_previous, view.frames);
        ctx.write_buffer(&view.reprojection, 0, bytemuck::bytes_of(&reprojection));
        // The raw matrix and its origin, not this frame's rebased one, so
        // successive shifts are never applied twice.
        view.previous_view = camera.view;
        view.previous_origin = origin;
        view.frames = view.frames.saturating_add(1);
        Ok(())
    }

    fn execute(&mut self, _ctx: &mut PassContext) -> HelioResult<()> {
        Ok(())
    }

    fn publish<'a>(&self, frame: &mut helio_core::ResourceRegistry<'a>) {
        let mut views = self.views.lock().unwrap();
        if let Some(view) = views.active() {
            frame.write_buffer_binding(REPROJECTION, &view.reprojection, "HistoryReprojection");
        }
    }
}

/// `info` of `history_pixels.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PixelsInfo {
    valid: u32,
    velocity: u32,
    _pad: [u32; 2],
}

/// Where each pixel was ([`PIXELS`]) and the previous guides
/// ([`PREVIOUS_DEPTH`], [`PREVIOUS_NORMAL`]).
pub struct HistoryPixelsPass {
    views: Shared,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    info: wgpu::Buffer,
    no_velocity: wgpu::TextureView,
    cache: helio_core::gpu_cache::GpuCache,
    /// This frame's dispatch ran (a frame without a G-buffer publishes
    /// nothing).
    ran: bool,
}

impl HistoryPixelsPass {
    /// The [`PIXELS`] texture of the view the last frame rendered (for tests
    /// and tools that read it back).
    pub fn pixels_texture(&self) -> Option<wgpu::Texture> {
        let mut views = self.views.lock().unwrap();
        views
            .active()
            .and_then(|v| v.guides.as_ref())
            .map(|g| g.pixels_texture.clone())
    }
}

/// The snippets `history_pixels.wgsl` is built with.
pub const SNIPPETS: [helio_core::shader::ShaderSnippet; 1] = [history::SNIPPET];

impl HistoryPixelsPass {
    fn new(device: &wgpu::Device, views: Shared) -> Self {
        let texture = |binding, sample_type| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let float = wgpu::TextureSampleType::Float { filterable: false };
        let storage = |binding, format| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        };
        let buffer = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("History pixels"),
            entries: &[
                texture(0, wgpu::TextureSampleType::Depth),
                texture(1, float),
                texture(2, float),
                texture(3, float),
                buffer(4, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer(5, wgpu::BufferBindingType::Uniform),
                texture(6, float),
                texture(7, float),
                storage(8, history::PREVIOUS_DEPTH_FORMAT.to_wgpu()),
                storage(9, history::PREVIOUS_NORMAL_FORMAT.to_wgpu()),
                storage(10, history::PIXELS_FORMAT.to_wgpu()),
                buffer(11, wgpu::BufferBindingType::Uniform),
            ],
        });
        let module = helio_core::shader::module_with(
            device,
            "History pixels",
            helio_core::include_wgsl!("../shaders/history_pixels.wgsl"),
            &SNIPPETS,
        );
        let pipeline = helio_core::shader::create_compute_pipeline(
            device,
            wgpu::ComputePipelineDescriptor {
                label: Some("History pixels"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("History pixels"),
                        bind_group_layouts: &[Some(&layout)],
                        immediate_size: 0,
                    }),
                ),
                module: &module,
                entry_point: Some("history_pixels"),
                // Every invocation fills its share of the neighbour tile before
                // the barrier; naga would otherwise zero it from one first.
                compilation_options: wgpu::PipelineCompilationOptions {
                    zero_initialize_workgroup_memory: false,
                    ..Default::default()
                },
                cache: None,
            },
        );
        // No velocity: every pixel moves with the camera (w = 0).
        let no_velocity = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("History no velocity"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: helio_frame_api::gbuffer::VELOCITY_FORMAT.to_wgpu(),
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default());
        Self {
            views,
            pipeline,
            layout,
            info: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("History pixels info"),
                size: std::mem::size_of::<PixelsInfo>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            no_velocity,
            cache: helio_core::gpu_cache::GpuCache::new(),
            ran: false,
        }
    }
}

fn texture_entry(binding: u32, view: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(view),
    }
}

fn buffer_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

impl RenderPass for HistoryPixelsPass {
    fn name(&self) -> &'static str {
        "HistoryPixels"
    }

    fn reads(&self) -> &'static [&'static str] {
        &[REPROJECTION]
    }

    fn writes(&self) -> &'static [&'static str] {
        &[PIXELS, DEPTH, NORMAL, PREVIOUS_DEPTH, PREVIOUS_NORMAL]
    }

    fn declare_resources(&self, builder: &mut helio_core::graph::ResourceBuilder) {
        builder.read(helio_frame_api::gbuffer::GROUP);
        builder.read(helio_frame_api::gbuffer::DEPTH);
        builder.read_optional(helio_frame_api::gbuffer::VELOCITY);
    }

    fn on_resize(&mut self, _device: &wgpu::Device, _size: helio_core::graph::RenderSize) {
        // Guides of the old size describe nothing at the new one.
        for view in &mut self.views.lock().unwrap().views {
            view.guides = None;
        }
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let mut views = self.views.lock().unwrap();
        let Some(view) = views.active() else {
            return Ok(());
        };
        let size = [ctx.width, ctx.height];
        if view.guides.as_ref().is_none_or(|g| g.size != size) {
            view.guides = Some(Guides::new(ctx.device, size));
        }
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        self.ran = false;
        let mut views = self.views.lock().unwrap();
        let Some(view) = views.active() else {
            return Ok(());
        };
        let frames = view.frames;
        let Some(gbuffer) = ctx.registry.view_group::<4>(helio_core::ResourceKey::new(
            helio_frame_api::gbuffer::GROUP,
        )) else {
            return Ok(());
        };
        let velocity = ctx
            .registry
            .texture_view(helio_core::ResourceKey::new(
                helio_frame_api::gbuffer::VELOCITY,
            ))
            .cloned();
        let guides = view.guides.as_mut().expect("prepared");
        // The previous slot holds the view's last frame when it wrote it the
        // frame before this one (the reprojection counted this frame first).
        let valid = guides.written_at == Some(frames.wrapping_sub(2));
        let previous = guides.latest;
        let next = previous ^ 1;
        ctx.write_buffer(
            &self.info,
            0,
            bytemuck::bytes_of(&PixelsInfo {
                valid: u32::from(valid),
                velocity: u32::from(velocity.is_some()),
                _pad: [0; 2],
            }),
        );
        self.cache.end_frame();
        let group = self.cache.bind_group(
            ctx.device,
            &wgpu::BindGroupDescriptor {
                label: Some("History pixels"),
                layout: &self.layout,
                entries: &[
                    texture_entry(0, ctx.depth),
                    texture_entry(1, gbuffer.views[1]),
                    texture_entry(2, gbuffer.views[2]),
                    texture_entry(3, velocity.as_ref().unwrap_or(&self.no_velocity)),
                    buffer_entry(4, ctx.camera),
                    buffer_entry(5, &view.reprojection),
                    texture_entry(6, &guides.depth[previous]),
                    texture_entry(7, &guides.normal[previous]),
                    texture_entry(8, &guides.depth[next]),
                    texture_entry(9, &guides.normal[next]),
                    texture_entry(10, &guides.pixels),
                    buffer_entry(11, &self.info),
                ],
            },
        );
        let mut cmds = ctx.graphics_cmds();
        ctx.begin_gpu_scope(&mut cmds, "History: pixels");
        {
            let mut pass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("History pixels"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, Some(&group), &[]);
            pass.dispatch_workgroups(guides.size[0].div_ceil(8), guides.size[1].div_ceil(8), 1);
        }
        ctx.end_gpu_scope(&mut cmds, "History: pixels");
        guides.latest = next;
        guides.written_at = Some(frames.wrapping_sub(1));
        drop(views);
        self.ran = true;
        Ok(())
    }

    fn publish<'a>(&self, frame: &mut helio_core::ResourceRegistry<'a>) {
        if !self.ran {
            return;
        }
        let mut views = self.views.lock().unwrap();
        let Some(guides) = views.active().and_then(|v| v.guides.as_ref()) else {
            return;
        };
        // This frame wrote `latest`; the other slot is the previous frame's.
        let previous = guides.latest ^ 1;
        frame.write_texture_binding(DEPTH, &guides.depth[guides.latest], "HistoryPixels");
        frame.write_texture_binding(NORMAL, &guides.normal[guides.latest], "HistoryPixels");
        frame.write_texture_binding(PREVIOUS_DEPTH, &guides.depth[previous], "HistoryPixels");
        frame.write_texture_binding(PREVIOUS_NORMAL, &guides.normal[previous], "HistoryPixels");
        frame.write_texture_binding(PIXELS, &guides.pixels, "HistoryPixels");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shader_validates() {
        let text = helio_core::shader::resolve_with(
            include_str!("../shaders/history_pixels.wgsl"),
            &SNIPPETS,
        );
        let module = wgpu::naga::front::wgsl::parse_str(&text)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&text)));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&text)));
    }
}
