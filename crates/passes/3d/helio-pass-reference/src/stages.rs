//! The reference's stages after the camera rays: a depth's three steps, the
//! accumulation and the display (crate docs).

use std::sync::Arc;

use helio_core::{PassContext, RenderPass, Result as HelioResult};

use crate::{entry, Shared, IMAGE, PATHS, REFERENCE_DIFFUSE_VIEW};

/// What a stage does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    /// Emission or the sky, a light query a hit, the next ray.
    Shade,
    /// The light samples' contributions and their shadow rays.
    Shadow,
    /// What the shadow rays let through.
    Resolve,
}

/// One step of one depth.
pub struct ReferenceStage {
    shared: Arc<Shared>,
    depth: u32,
    step: Step,
    /// `ReferenceShade d<depth>` and so on: a timing a stage.
    name: &'static str,
    reads: &'static [&'static str],
    writes: &'static [&'static str],
}

impl ReferenceStage {
    pub(crate) fn new(shared: Arc<Shared>, depth: u32, step: Step) -> Self {
        let s = &shared.streams;
        let d = depth as usize;
        let (mut reads, mut writes) = (vec![PATHS], vec![PATHS]);
        match step {
            Step::Shade => {
                reads.extend([s.nearest[d].header, s.nearest[d].rays, s.nearest[d].hits]);
                reads.extend(s.nearest[d].surfaces);
                writes.extend([
                    s.batches[d].header,
                    s.batches[d].queries,
                    s.batches[d].samples,
                ]);
                if let Some(next) = s.nearest.get(d + 1) {
                    writes.extend([next.header, next.rays, next.hits]);
                }
            }
            Step::Shadow => {
                reads.extend([s.batches[d].header, s.batches[d].samples]);
                writes.extend([s.shadows[d].header, s.shadows[d].rays, s.shadows[d].hits]);
            }
            Step::Resolve => {
                reads.extend([s.shadows[d].header, s.shadows[d].hits]);
            }
        }
        let step_name = match step {
            Step::Shade => "ReferenceShade",
            Step::Shadow => "ReferenceShadow",
            Step::Resolve => "ReferenceResolve",
        };
        Self {
            name: helio_frame_api::static_name(&format!("{step_name} d{depth}")),
            reads: helio_frame_api::static_names(&reads),
            writes: helio_frame_api::static_names(&writes),
            shared,
            depth,
            step,
        }
    }
}

impl RenderPass for ReferenceStage {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reads(&self) -> &'static [&'static str] {
        self.reads
    }

    fn writes(&self) -> &'static [&'static str] {
        self.writes
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let shared = &self.shared;
        let state = shared.state.lock().unwrap();
        let (Some(b), true) = (state.buffers.as_ref(), state.paths > 0) else {
            return Ok(());
        };
        let k = &shared.kernels;
        let group = shared.group(ctx.device, b, self.depth, shared.sky(ctx.registry));
        let cur = (self.depth % 2) as usize;
        let mut cmds = ctx.graphics_cmds();
        match self.step {
            Step::Shade => {
                cmds.clear_buffer(&b.rays[1 - cur].header, 0, None);
                cmds.clear_buffer(&b.lights.header, 0, None);
                b.lights.clear(&mut cmds);
            }
            Step::Shadow => cmds.clear_buffer(&b.shadows.header, 0, None),
            Step::Resolve => {}
        }
        let mut pass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(self.name()),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, Some(&group), &[]);
        let (kernel, header) = match self.step {
            Step::Shade => (&k.shade, Some(&k.shade_headers)),
            Step::Shadow => (&k.shadow, Some(&k.shadow_header)),
            Step::Resolve => (&k.resolve, None),
        };
        // Over the frame's paths, each step's rows (at most one a path)
        // returning past their count: the counts are atomics the same
        // dispatch writes, which an indirect dispatch could not read
        // (a buffer bound for writing is not an indirect argument in the
        // same dispatch).
        pass.set_pipeline(kernel);
        pass.dispatch_workgroups(state.paths.div_ceil(64), 1, 1);
        drop(pass);
        // The counts' dispatches in a pass of their own: they read what
        // every workgroup of the step counted.
        if let Some(header) = header {
            let mut pass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Reference headers"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, Some(&group), &[]);
            pass.set_pipeline(header);
            pass.dispatch_workgroups(1, 1, 1);
        }
        Ok(())
    }
}

/// The frame's paths into the image's sums.
pub struct ReferenceAccumulate {
    pub(crate) shared: Arc<Shared>,
}

impl RenderPass for ReferenceAccumulate {
    fn name(&self) -> &'static str {
        "ReferenceAccumulate"
    }

    fn reads(&self) -> &'static [&'static str] {
        &[PATHS]
    }

    fn writes(&self) -> &'static [&'static str] {
        &[IMAGE]
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let shared = &self.shared;
        let state = shared.state.lock().unwrap();
        let (Some(b), true) = (state.buffers.as_ref(), state.paths > 0) else {
            return Ok(());
        };
        // A frame a tracer answered short of (geometry it could not trace
        // yet) would sum misses where that geometry is: not summed.
        if !shared.answers.iter().all(|a| a.complete()) {
            log::debug!("reference: a frame answered short is not summed");
            return Ok(());
        }
        let group = shared.group(ctx.device, b, 0, shared.sky(ctx.registry));
        let mut cmds = ctx.graphics_cmds();
        let mut pass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Reference accumulate"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, Some(&group), &[]);
        pass.set_pipeline(&shared.kernels.accumulate);
        pass.dispatch_workgroups(state.paths.div_ceil(64), 1, 1);
        Ok(())
    }
}

/// The reference's image as a host reads it ([`ReferenceDisplay::image`]):
/// its size, each pixel's sum of paths (rgb) and their count (a), its
/// indirect diffuse's; and, once asked for, the real-time scene colour at
/// the same pixels ([`ReferenceDisplay::request_snapshot`]). Buffers of
/// `vec4<f32>` a pixel, row-major, `COPY_SRC`.
pub struct ReferenceImage {
    pub size: [u32; 2],
    /// Frames since the sum restarted.
    pub frames: u32,
    pub image: wgpu::Buffer,
    pub diffuse: wgpu::Buffer,
    /// The scene colour, once a snapshot was taken.
    pub snapshot: Option<wgpu::Buffer>,
}

/// The mean over the scene colour where one of the views is shown.
pub struct ReferenceDisplay {
    shared: Arc<Shared>,
    color: &'static str,
    layout: wgpu::BindGroupLayout,
    snapshot_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    snapshot: wgpu::ComputePipeline,
    view: wgpu::Buffer,
    reads: &'static [&'static str],
    writes: &'static [&'static str],
}

impl ReferenceDisplay {
    pub(crate) fn new(
        device: &wgpu::Device,
        shared: Arc<Shared>,
        color: &'static str,
        format: wgpu::TextureFormat,
    ) -> Self {
        let buffer_entry = |binding, visibility, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let fragment = wgpu::ShaderStages::FRAGMENT;
        let compute = wgpu::ShaderStages::COMPUTE;
        let read = wgpu::BufferBindingType::Storage { read_only: true };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Reference display"),
            entries: &[
                buffer_entry(0, fragment, read),
                buffer_entry(1, fragment, read),
                buffer_entry(2, fragment, wgpu::BufferBindingType::Uniform),
            ],
        });
        let snapshot_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Reference snapshot"),
            entries: &[
                buffer_entry(2, compute, wgpu::BufferBindingType::Uniform),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: compute,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                buffer_entry(
                    4,
                    compute,
                    wgpu::BufferBindingType::Storage { read_only: false },
                ),
            ],
        });
        let module = helio_core::shader::module(
            device,
            "Reference display",
            helio_core::include_wgsl!("../shaders/reference_display.wgsl"),
        );
        let pipeline_layout = |label, layout: &wgpu::BindGroupLayout| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(layout)],
                immediate_size: 0,
            })
        };
        let display_layout = pipeline_layout("Reference display", &layout);
        let pipeline = helio_core::shader::create_render_pipeline(
            device,
            wgpu::RenderPipelineDescriptor {
                label: Some("Reference display"),
                layout: Some(&display_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            },
        );
        let snapshot_pipeline_layout = pipeline_layout("Reference snapshot", &snapshot_layout);
        let snapshot = helio_core::shader::create_compute_pipeline(
            device,
            wgpu::ComputePipelineDescriptor {
                label: Some("Reference snapshot"),
                layout: Some(&snapshot_pipeline_layout),
                module: &module,
                entry_point: Some("reference_snapshot"),
                compilation_options: Default::default(),
                cache: None,
            },
        );
        Self {
            shared,
            color,
            layout,
            snapshot_layout,
            pipeline,
            snapshot,
            view: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Reference display view"),
                size: 32,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            reads: helio_frame_api::static_names(&[IMAGE, color]),
            writes: helio_frame_api::static_names(&[color]),
        }
    }

    /// The reference's image while one of its views is shown (none
    /// otherwise), for a host or test that reads it back.
    pub fn image(&self) -> Option<ReferenceImage> {
        let state = self.shared.state.lock().unwrap();
        state.buffers.as_ref().map(|b| ReferenceImage {
            size: b.size,
            frames: state.frames,
            image: b.image.clone(),
            diffuse: b.diffuse.clone(),
            snapshot: state.snapshot_taken.then(|| b.snapshot.clone()),
        })
    }

    /// Asks for the real-time scene colour at the image's pixels on the next
    /// frame, taken before the reference covers it
    /// ([`ReferenceImage::snapshot`]).
    pub fn request_snapshot(&self) {
        let mut state = self.shared.state.lock().unwrap();
        state.snapshot_requested = true;
        state.snapshot_taken = false;
    }

    /// Whether the reference traces and sums while none of its views is
    /// shown: a host measuring the real-time image (any debug view of it,
    /// [`Self::request_snapshot`]) against the reference. Its passes are
    /// members while it does.
    pub fn keep_tracing(&self, measuring: bool) {
        self.shared.state.lock().unwrap().measuring = measuring;
    }
}

impl ReferenceImage {
    /// Writes the sums and the snapshot to `<prefix>_reference.f32`,
    /// `<prefix>_diffuse.f32` and `<prefix>_realtime.f32` (when taken): each
    /// `width`, `height` (u32 little-endian) then RGBA f32 a pixel (a sum and
    /// its count of paths; the snapshot's colour), what
    /// `scripts/r2/reference_compare.py` reads. Waits for the GPU: a host's
    /// measurement, not a frame's work. Returns the files written.
    pub fn write(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        prefix: &str,
    ) -> std::io::Result<Vec<String>> {
        let mut buffers = vec![("reference", &self.image), ("diffuse", &self.diffuse)];
        buffers.extend(self.snapshot.as_ref().map(|s| ("realtime", s)));
        let mut written = Vec::new();
        for (name, buffer) in buffers {
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Reference dump"),
                size: buffer.size(),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
            queue.submit([encoder.finish()]);
            staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, |r| r.expect("reference dump maps"));
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(std::io::Error::other)?;
            let pixels = (self.size[0] * self.size[1]) as usize * 16;
            let mut bytes = Vec::with_capacity(8 + pixels);
            bytes.extend_from_slice(&self.size[0].to_le_bytes());
            bytes.extend_from_slice(&self.size[1].to_le_bytes());
            bytes.extend_from_slice(
                &staging.slice(..).get_mapped_range().expect("mapped")[..pixels],
            );
            let path = format!("{prefix}_{name}.f32");
            std::fs::write(&path, bytes)?;
            written.push(path);
        }
        Ok(written)
    }
}

impl RenderPass for ReferenceDisplay {
    fn name(&self) -> &'static str {
        "ReferenceDisplay"
    }

    fn reads(&self) -> &'static [&'static str] {
        self.reads
    }

    fn writes(&self) -> &'static [&'static str] {
        self.writes
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let mut state = self.shared.state.lock().unwrap();
        let Some(b) = state.buffers.clone().filter(|_| state.paths > 0) else {
            return Ok(());
        };
        let Some(color) = ctx
            .registry
            .texture_view(helio_core::ResourceKey::new(self.color))
        else {
            return Ok(());
        };
        let mode = u32::from(state.mode == REFERENCE_DIFFUSE_VIEW);
        let view = [b.size[0], b.size[1], ctx.width, ctx.height, mode, 0, 0, 0];
        helio_core::upload::write_buffer(ctx.queue, &self.view, 0, bytemuck::cast_slice(&view));
        let cache = &self.shared.cache;
        if std::mem::take(&mut state.snapshot_requested) {
            let group = cache.bind_group(
                ctx.device,
                &wgpu::BindGroupDescriptor {
                    label: Some("Reference snapshot"),
                    layout: &self.snapshot_layout,
                    entries: &[
                        entry(2, &self.view),
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(color),
                        },
                        entry(4, &b.snapshot),
                    ],
                },
            );
            let mut cmds = ctx.graphics_cmds();
            let mut pass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Reference snapshot"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, Some(&group), &[]);
            pass.set_pipeline(&self.snapshot);
            pass.dispatch_workgroups((b.size[0] * b.size[1]).div_ceil(64), 1, 1);
            state.snapshot_taken = true;
        }
        // Measured, not shown: the scene colour stays the real-time image.
        if !state.shown() {
            return Ok(());
        }
        drop(state);
        let group = cache.bind_group(
            ctx.device,
            &wgpu::BindGroupDescriptor {
                label: Some("Reference display"),
                layout: &self.layout,
                entries: &[
                    entry(0, &b.image),
                    entry(1, &b.diffuse),
                    entry(2, &self.view),
                ],
            },
        );
        let attachments = [Some(wgpu::RenderPassColorAttachment {
            view: color,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        })];
        let descriptor = wgpu::RenderPassDescriptor {
            label: Some("Reference display"),
            color_attachments: &attachments,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        };
        let mut pass = ctx.begin_render_pass(&descriptor);
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, Some(&group), &[]);
        pass.draw(0..3, 0..1);
        Ok(())
    }
}
