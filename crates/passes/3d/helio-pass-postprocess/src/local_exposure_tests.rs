//! Local exposure on a GPU: the production grid and blur kernels, and the
//! multiplier fs_uber applies, read back per pixel (an adapter is required;
//! the tests fail rather than skip).

use super::*;
use wgpu::util::DeviceExt;

/// Evaluates fs_uber's local exposure multiplier at every pixel.
const PROBE: &str = "
@group(0) @binding(30) var<storage, read_write> probe_out: array<f32>;
@compute @workgroup_size(8, 8)
fn probe_local_exposure(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(hdr_input);
    if any(id.xy >= dims) { return; }
    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(dims);
    let color = textureLoad(hdr_input, vec2<i32>(id.xy), 0).rgb;
    probe_out[id.y * dims.x + id.x] = local_exposure_scale(color, uv);
}
";

struct Probe {
    device: wgpu::Device,
    queue: wgpu::Queue,
    grid: wgpu::ComputePipeline,
    blur: wgpu::ComputePipeline,
    probe: wgpu::ComputePipeline,
    build_layout: wgpu::BindGroupLayout,
    probe_layout: wgpu::BindGroupLayout,
}

impl Probe {
    fn new() -> Self {
        let (device, queue) = pollster::block_on(async {
            let instance =
                wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
            let adapter = instance
                .request_adapter(&Default::default())
                .await
                .expect("GPU adapter required for local exposure tests");
            adapter.request_device(&Default::default()).await.unwrap()
        });
        let source = PostProcessPass::build_shader_source(&[]) + PROBE;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Local exposure probe"),
            source: wgpu::ShaderSource::Wgsl(helio_core::shader::resolve(&source).into_owned().into()),
        });
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty,
            count: None,
        };
        let buffer = |ty| wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None };
        let storage = buffer(wgpu::BufferBindingType::Storage { read_only: false });
        let texture = |view_dimension| wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension,
            multisampled: false,
        };
        let shared = [
            entry(0, buffer(wgpu::BufferBindingType::Uniform)),
            entry(2, texture(wgpu::TextureViewDimension::D2)),
            entry(4, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)),
            entry(11, storage),
        ];
        let layout = |extra: &[wgpu::BindGroupLayoutEntry]| {
            let entries: Vec<_> = shared.iter().chain(extra).cloned().collect();
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &entries })
        };
        // The pass binds the grid's storage and sampled views in separate
        // groups (compute and render) too: one group may not hold both.
        let build_layout = layout(&[
            entry(23, storage),
            entry(
                24,
                wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D3,
                },
            ),
        ]);
        let probe_layout = layout(&[entry(25, texture(wgpu::TextureViewDimension::D3)), entry(30, storage)]);
        let pipeline = |layout: &wgpu::BindGroupLayout, entry_point| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[Some(layout)],
                    immediate_size: 0,
                })),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Self {
            grid: pipeline(&build_layout, "cs_local_exposure_grid"),
            blur: pipeline(&build_layout, "cs_local_exposure_blur"),
            probe: pipeline(&probe_layout, "probe_local_exposure"),
            build_layout,
            probe_layout,
            device,
            queue,
        }
    }

    /// The local exposure multiplier of each pixel of a grey `width` x
    /// `height` image of luminances `luminance(x, y)` (as half floats), with
    /// auto exposure adapted to `adapted_log2` (that luminance meets the
    /// curve as middle grey) and no tone curve.
    fn scales(
        &self,
        settings: &PostProcessSettings,
        width: u32,
        height: u32,
        adapted_log2: f32,
        luminance: impl Fn(u32, u32) -> f32,
    ) -> Vec<f32> {
        let device = &self.device;
        let mut uniforms = settings.to_gpu();
        uniforms.exposure_mode = ExposureMode::Auto as u32;
        (uniforms.exposure_min, uniforms.exposure_max) = (-16.0, 16.0);
        uniforms.tonemap_operator = TonemapOperator::None as u32;
        let uniforms =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let adaptation = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[adapted_log2, adapted_log2, 0.0f32, 1.0]),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let pixels: Vec<u16> = (0..height)
            .flat_map(|y| (0..width).map(move |x| (x, y)))
            .flat_map(|(x, y)| {
                let l = half(luminance(x, y));
                [l, l, l, half(1.0)]
            })
            .collect();
        let image = device.create_texture_with_data(
            &self.queue,
            &wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            bytemuck::cast_slice(&pixels),
        );
        let image = image.create_view(&Default::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            min_filter: wgpu::FilterMode::Linear,
            mag_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let grid = LocalExposureGrid::new(device, width, height);
        let size = u64::from(width * height) * 4;
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let group = |layout, extra: &[wgpu::BindGroupEntry]| {
            let mut entries = vec![
                wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&image) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&sampler) },
                wgpu::BindGroupEntry { binding: 11, resource: adaptation.as_entire_binding() },
            ];
            entries.extend_from_slice(extra);
            device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout, entries: &entries })
        };
        let build = group(
            &self.build_layout,
            &[
                wgpu::BindGroupEntry { binding: 23, resource: grid.raw.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 24,
                    resource: wgpu::BindingResource::TextureView(&grid.storage_view),
                },
            ],
        );
        let probe = group(
            &self.probe_layout,
            &[
                wgpu::BindGroupEntry {
                    binding: 25,
                    resource: wgpu::BindingResource::TextureView(&grid.sampled_view),
                },
                wgpu::BindGroupEntry { binding: 30, resource: output.as_entire_binding() },
            ],
        );
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, &build, &[]);
            for (pipeline, [x, y, z]) in [&self.grid, &self.blur].into_iter().zip(grid.workgroups()) {
                pass.set_pipeline(pipeline);
                pass.dispatch_workgroups(x, y, z);
            }
        }
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.probe);
            pass.set_bind_group(0, &probe, &[]);
            pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, size);
        self.queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        readback.slice(..).map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let scales = bytemuck::cast_slice(&readback.slice(..).get_mapped_range().unwrap()).to_vec();
        readback.unmap();
        scales
    }
}

const GREY: f32 = 0.18;

/// `v` (positive, in half's normal range) as half-float bits, truncated.
fn half(v: f32) -> u16 {
    let bits = v.to_bits();
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    assert!((1..31).contains(&exponent), "{v} is not a normal half");
    ((exponent as u32) << 10 | (bits >> 13) & 0x3ff) as u16
}

/// A dark half beside a bright half, metered to middle grey between them:
/// each half moves toward middle grey by its own contrast, right up to the
/// edge (no halo), and an image of one luminance is left as it is.
#[test]
fn dark_half_lifts_toward_grey_without_halos_and_uniform_images_hold() {
    let probe = Probe::new();
    let settings = PostProcessSettings::default();
    let (width, height) = (256u32, 144u32);
    let ev = |scale: f32| scale.log2();

    // Shade 4 EV under middle grey, sky 2 EV over it.
    let (dark, bright) = (GREY / 16.0, GREY * 4.0);
    let scales = probe.scales(&settings, width, height, GREY.log2(), |x, _| if x < width / 2 { dark } else { bright });
    let shadow_lift = -4.0 * (settings.local_exposure_shadow_contrast - 1.0);
    let highlight_drop = 2.0 * (settings.local_exposure_highlight_contrast - 1.0);
    assert!(shadow_lift > 1.0, "the default lifts shade by over an EV: {shadow_lift}");
    for y in [0, height / 2, height - 1] {
        for x in 0..width {
            let scale = scales[(y * width + x) as usize];
            let (expected, luminance) =
                if x < width / 2 { (shadow_lift, dark) } else { (highlight_drop, bright) };
            // Every pixel, the ones beside the edge too: a halo would show
            // as the dark side darkening or the bright side lifting there.
            assert!(
                (ev(scale) - expected).abs() < 0.02,
                "pixel ({x}, {y}): {} EV, expected {expected}",
                ev(scale)
            );
            // Toward middle grey, never past it.
            let out = luminance * scale;
            assert!((out - GREY).abs() < (luminance - GREY).abs(), "pixel ({x}, {y}) {luminance} -> {out}");
        }
    }

    // One luminance everywhere, metered as the frame's mean: unchanged.
    for luminance in [GREY / 32.0, 0.5, 40.0] {
        let scales = probe.scales(&settings, width, height, luminance.log2(), |_, _| luminance);
        let worst = scales.iter().map(|s| ev(*s).abs()).fold(0.0f32, f32::max);
        assert!(worst < 0.01, "a uniform image of {luminance} moved by {worst} EV");
    }

    // Contrasts of 1 leave every pixel alone.
    let mut off = settings.clone();
    off.local_exposure_shadow_contrast = 1.0;
    off.local_exposure_highlight_contrast = 1.0;
    off.local_exposure_detail_strength = 1.0;
    let scales = probe.scales(&off, width, height, GREY.log2(), |x, _| if x < width / 2 { dark } else { bright });
    assert!(scales.iter().all(|&s| s == 1.0));
}

/// Texture within a region keeps its contrast at detail strength 1: the
/// region's mean moves, its pixels move together.
#[test]
fn detail_inside_a_region_keeps_its_contrast() {
    let probe = Probe::new();
    let settings = PostProcessSettings::default();
    let (width, height) = (256u32, 144u32);
    // A dark region (-4 EV) with a fine checker of ±0.5 EV.
    let base = GREY / 16.0;
    let checker = |x: u32, y: u32| base * if (x / 2 + y / 2) % 2 == 0 { 2f32.sqrt() } else { 0.5f32.sqrt() };
    let scales = probe.scales(&settings, width, height, GREY.log2(), checker);
    let at = |x: u32, y: u32| checker(x, y) * scales[(y * width + x) as usize];
    for (x, y) in [(64, 64), (128, 72), (200, 100)] {
        let before = (checker(x, y) / checker(x + 2, y)).log2().abs();
        let after = (at(x, y) / at(x + 2, y)).log2().abs();
        assert!(after > 0.8 * before, "texture at ({x}, {y}): {before} EV -> {after} EV");
    }
}
