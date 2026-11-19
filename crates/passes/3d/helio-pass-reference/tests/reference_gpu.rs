//! The reference path tracer against analytic answers, independent of the
//! engine's own lighting: the tests offer their own tracer (a closed room or
//! a ground plane, analytic intersections) and their own light sampler (a
//! point light and a directional light evaluated here), through the frame's
//! contracts, and read the accumulated image back. Skipped without a GPU.
//!
//! - **Furnace**: inside a closed room of albedo `a` whose walls emit `E`,
//!   every path sees `E (1 - a^D) / (1 - a)` with at most `D` surfaces (no
//!   light sample: no light). Checked at depths 1, 2, 3 (energy across bounce
//!   depth, before Russian roulette) and 12 (with it).
//! - **Direct light**: ground of albedo `rho` under a point light and a sun,
//!   one surface a path: each pixel's radiance is
//!   `rho / pi (I cos / d^2 window + E cos_sun)`.
//! - **Convergence**: the furnace's RMS error over the image falls as
//!   1 / sqrt(N): four times the frames, half the error.

use std::sync::Arc;

use bytemuck::Zeroable;
use glam::{DVec3, Mat4, Vec3};
use helio_core::{
    GpuCameraUniforms, PassContext, RenderGraph, RenderPass, Result as HelioResult,
    SceneBufferProjection, SceneInput,
};
use helio_feature_api::{BuildEnv, FeatureSet, FrameDesc, NodeBuild, NodeRegistry};
use helio_frame_api::lighting::LightingSettings;
use helio_frame_api::rays::light_samples::{self, LightBatch, LightSampler};
use helio_frame_api::rays::stream::{self, RayClass, Stream, Tracer, TracerId};
use helio_pass_reference::{ReferenceDisplay, ReferenceFeature, ReferenceSettings, REFERENCE_VIEW};

const SIZE: u32 = 32;

struct Gpu {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

fn gpu() -> Option<&'static Gpu> {
    static GPU: std::sync::OnceLock<Option<Gpu>> = std::sync::OnceLock::new();
    GPU.get_or_init(|| {
        let instance = wgpu::Instance::new(helio_core::wgpu_instance_descriptor());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
        // The adapter's limits, as the engine's device asks for them (the
        // reference binds more storage buffers than WebGPU's baseline).
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: wgpu::Limits {
                max_buffer_size: limits.max_buffer_size.min(u32::MAX as u64),
                ..limits
            },
            ..Default::default()
        }))
        .ok()?;
        Some(Gpu {
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    })
    .as_ref()
}

fn read(gpu: &Gpu, buffer: &wgpu::Buffer) -> Vec<[f32; 4]> {
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: buffer.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
    gpu.queue.submit([encoder.finish()]);
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, |r| r.unwrap());
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    bytemuck::cast_slice(&staging.slice(..).get_mapped_range().unwrap()).to_vec()
}

/// The scene the test's tracer and sampler see (`kind` 0: a room from
/// `lo` to `hi`; 1: ground at `lo.y`), and its lights.
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Scene {
    lo: [f32; 4],
    hi: [f32; 4],
    /// rgb albedo, w the kind.
    albedo: [f32; 4],
    emission: [f32; 4],
    /// A point light: position, intensity (w); its range.
    point: [f32; 4],
    range: [f32; 4],
    /// A directional light: toward it (xyz), illuminance (w).
    sun: [f32; 4],
}

const TRACER: &str = r#"//!use frame_ray_stream
//!use frame_ray_surface
//!use frame_surface
struct Scene { lo: vec4<f32>, hi: vec4<f32>, albedo: vec4<f32>, emission: vec4<f32>, point: vec4<f32>, range: vec4<f32>, sun: vec4<f32> }
@group(0) @binding(0) var<storage, read> header: RayStreamHeader;
@group(0) @binding(1) var<storage, read> rays: array<SceneRay>;
@group(0) @binding(2) var<storage, read_write> hits: array<SceneHit>;
@group(0) @binding(3) var<storage, read_write> surfaces: array<SceneSurface>;
@group(0) @binding(4) var<uniform> scene: Scene;
// x: the tracer's id, y: 1 for an occlusion stream.
@group(0) @binding(5) var<uniform> info: vec4<u32>;

@compute @workgroup_size(64)
fn trace(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= header.count { return; }
    let ray = rays[id.x];
    if ray.t_max <= 0.0 { return; }
    var t = -1.0;
    var n = vec3<f32>(0.0, 1.0, 0.0);
    if scene.albedo.w < 0.5 {
        // The room from inside: where the ray leaves the box.
        let inv = 1.0 / ray.dir;
        let t0 = (scene.lo.xyz - ray.origin) * inv;
        let t1 = (scene.hi.xyz - ray.origin) * inv;
        let far = max(t0, t1);
        t = min(far.x, min(far.y, far.z));
        if t == far.x { n = vec3<f32>(-sign(ray.dir.x), 0.0, 0.0); } else if t == far.y { n = vec3<f32>(0.0, -sign(ray.dir.y), 0.0); } else { n = vec3<f32>(0.0, 0.0, -sign(ray.dir.z)); }
    } else if ray.dir.y < 0.0 && ray.origin.y > scene.lo.y {
        t = (scene.lo.y - ray.origin.y) / ray.dir.y;
    }
    if t <= 0.0 || t >= scene_hit_reach(ray, hits[id.x]) { return; }
    if info.y == 1u {
        hits[id.x] = scene_occlusion(vec3<f32>(0.0));
        return;
    }
    hits[id.x] = scene_nearest(info.x, t, false, 0u, 0u, 0u);
    // Pure Lambert: no specular reflectance at all (F0 0, so F90 0 too).
    var s = surface_data(scene.albedo.rgb, n, 1.0, 0.0, scene.emission.rgb);
    s.specular_f0 = vec3<f32>(0.0);
    surfaces[id.x] = scene_surface_pack(s, n);
}
"#;

const SAMPLER: &str = r#"//!use frame_ray_stream
//!use frame_light_samples
struct Scene { lo: vec4<f32>, hi: vec4<f32>, albedo: vec4<f32>, emission: vec4<f32>, point: vec4<f32>, range: vec4<f32>, sun: vec4<f32> }
@group(0) @binding(0) var<storage, read> header: RayStreamHeader;
@group(0) @binding(1) var<storage, read> queries: array<LightQuery>;
@group(0) @binding(2) var<storage, read_write> samples: array<LightSample>;
@group(0) @binding(3) var<uniform> scene: Scene;
@group(0) @binding(4) var<uniform> info: vec4<u32>;

// A point light (inverse square, windowed by (1 - (d / range)^4)) and a
// directional light, one chosen by its unshadowed light times the cosine.
@compute @workgroup_size(64)
fn sample(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= header.count { return; }
    let q = queries[id.x];
    let n = scene_unpack_unit(q.normal);
    let to_light = scene.point.xyz - q.position;
    let d = length(to_light);
    let l = to_light / d;
    let x = d / scene.range.x;
    let point = scene.point.w * max(0.0, 1.0 - x * x * x * x) / (d * d);
    let wp = point * max(dot(n, l), 0.0);
    let ws = scene.sun.w * max(dot(n, scene.sun.xyz), 0.0);
    let total = wp + ws;
    if total <= 0.0 { return; }
    var offer: LightSample;
    if light_query_uniform(q, 16u) * total < wp {
        offer = light_sample_of(vec3<f32>(point) * total / wp, l, d, 0.0, info.x, 0u);
    } else {
        offer = light_sample_of(vec3<f32>(scene.sun.w) * total / ws, scene.sun.xyz, LIGHT_SAMPLE_FAR, 0.0, info.x, 1u);
    }
    samples[id.x] = light_sample_offer(samples[id.x], offer, q, info.x);
}
"#;

/// A compute pass of the test's (the tracer's or the sampler's): one
/// indirect dispatch over `header`, its buffers by name.
struct TestPass {
    name: &'static str,
    pipeline: wgpu::ComputePipeline,
    names: Vec<Option<&'static str>>,
    scene: wgpu::Buffer,
    info: wgpu::Buffer,
    reads: &'static [&'static str],
    writes: &'static [&'static str],
}

impl TestPass {
    #[allow(clippy::too_many_arguments)]
    fn new(
        device: &wgpu::Device,
        name: &'static str,
        source: &str,
        entry: &str,
        names: Vec<Option<&'static str>>,
        scene: &wgpu::Buffer,
        info: [u32; 4],
        writes: &[&'static str],
    ) -> Self {
        use wgpu::util::DeviceExt;
        let module = helio_core::shader::module_with(
            device,
            name,
            source,
            &[
                stream::SNIPPET,
                stream::SURFACE_SNIPPET,
                helio_frame_api::gbuffer::SURFACE_SNIPPET,
                light_samples::SNIPPET,
            ],
        );
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(name),
            layout: None,
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        });
        let reads: Vec<&'static str> = names.iter().flatten().copied().collect();
        Self {
            name,
            pipeline,
            info: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(name),
                contents: bytemuck::cast_slice(&info),
                usage: wgpu::BufferUsages::UNIFORM,
            }),
            scene: scene.clone(),
            names,
            reads: helio_frame_api::static_names(&reads),
            writes: helio_frame_api::static_names(writes),
        }
    }
}

impl RenderPass for TestPass {
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
        let mut buffers = Vec::new();
        for name in &self.names {
            match name {
                Some(name) => match ctx.registry.buffer_binding(name) {
                    Some(b) => buffers.push(Some(b.clone())),
                    None => return Ok(()),
                },
                None => buffers.push(None),
            }
        }
        let header = buffers[0].clone().unwrap();
        let mut entries = Vec::new();
        for (i, b) in buffers.iter().enumerate() {
            if let Some(b) = b {
                entries.push(wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: b.as_entire_binding(),
                });
            }
        }
        let n = buffers.len() as u32;
        entries.push(wgpu::BindGroupEntry {
            binding: n,
            resource: self.scene.as_entire_binding(),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: n + 1,
            resource: self.info.as_entire_binding(),
        });
        let group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(self.name),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut cmds = ctx.graphics_cmds();
        let mut pass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(self.name),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, Some(&group), &[]);
        pass.set_pipeline(&self.pipeline);
        pass.dispatch_workgroups_indirect(&header, 0);
        Ok(())
    }
}

fn tracer(scene: wgpu::Buffer) -> Tracer {
    Tracer {
        name: "test",
        supported: Arc::new(|_| true),
        trace: Arc::new(move |build, s: &Stream, id: TracerId| {
            let occlusion = s.class == RayClass::Occlusion;
            // Occlusion streams have no surfaces: bind the hits twice.
            let names = vec![
                Some(s.header),
                Some(s.rays),
                Some(s.hits),
                Some(s.surfaces.unwrap_or(s.hits)),
            ];
            let mut writes = vec![s.hits];
            writes.extend(s.surfaces);
            let pass = TestPass::new(
                build.env.device,
                "TestTracer",
                TRACER,
                "trace",
                names,
                &scene,
                [id.0, u32::from(occlusion), 0, 0],
                &writes,
            );
            build.graph.add_pass(Box::new(pass));
            Ok(())
        }),
        surface: Arc::new(|_, _, _| Ok(())),
    }
}

fn sampler(scene: wgpu::Buffer) -> LightSampler {
    LightSampler {
        name: "test",
        supported: Arc::new(|_| true),
        sample: Arc::new(move |build, batch: &LightBatch, id| {
            let names = vec![Some(batch.header), Some(batch.queries), Some(batch.samples)];
            let pass = TestPass::new(
                build.env.device,
                "TestSampler",
                SAMPLER,
                "sample",
                names,
                &scene,
                [id.0, 0, 0, 0],
                &[batch.samples],
            );
            build.graph.add_pass(Box::new(pass));
            Ok(())
        }),
    }
}

struct Input {
    gpu: &'static Gpu,
    camera: wgpu::Buffer,
    camera_data: GpuCameraUniforms,
    buffers: SceneBufferProjection,
    frame: u64,
}

impl SceneInput for Input {
    fn world_origin(&self) -> DVec3 {
        DVec3::ZERO
    }
    fn device(&self) -> &Arc<wgpu::Device> {
        &self.gpu.device
    }
    fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.gpu.queue
    }
    fn frame_count(&self) -> u64 {
        self.frame
    }
    fn camera(&self) -> &wgpu::Buffer {
        &self.camera
    }
    fn camera_data(&self) -> &GpuCameraUniforms {
        &self.camera_data
    }
    fn camera_generation(&self) -> u64 {
        0
    }
    fn scene_buffers(&self) -> &SceneBufferProjection {
        &self.buffers
    }
}

/// The camera at `eye` looking at `target`, 60 degrees, reversed infinite
/// depth.
fn camera(eye: Vec3, target: Vec3, up: Vec3) -> (GpuCameraUniforms, Mat4) {
    let view = glam::camera::rh::view::look_at_mat4(eye, target, up);
    let proj = glam::camera::rh::proj::directx::perspective_infinite_reverse(
        60f32.to_radians(),
        1.0,
        0.05,
    );
    let view_proj = proj * view;
    let mut c = GpuCameraUniforms::zeroed();
    c.view = view.to_cols_array();
    c.proj = proj.to_cols_array();
    c.view_proj = view_proj.to_cols_array();
    c.inv_view_proj = view_proj.inverse().to_cols_array();
    c.position_near = [eye.x, eye.y, eye.z, 0.05];
    (c, view_proj.inverse())
}

/// The image after `frames` frames (a path a pixel each) of `scene` seen
/// from `camera`: each pixel's mean, and the frames summed.
fn render(
    gpu: &'static Gpu,
    scene: Scene,
    camera: GpuCameraUniforms,
    depth: u32,
    frames: u32,
) -> (Vec<Vec3>, u32) {
    use wgpu::util::DeviceExt;
    let scene_buffer = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test scene"),
            contents: bytemuck::bytes_of(&scene),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let camera_buffer = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test camera"),
            contents: bytemuck::bytes_of(&camera),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::STORAGE,
        });
    let stats = helio_core::FrameStats::new(&gpu.device);
    let frame = FrameDesc {
        width: SIZE,
        height: SIZE,
        internal_width: SIZE,
        internal_height: SIZE,
        surface_format: wgpu::TextureFormat::Rgba8Unorm,
        xr: false,
    };
    let mut env = BuildEnv::new(
        &gpu.device,
        &gpu.queue,
        frame,
        &camera_buffer,
        &camera_buffer,
        &stats,
        true,
    )
    .with_setting(ReferenceSettings {
        resolution_scale: 1.0,
        paths_per_frame: SIZE * SIZE,
        max_depth: depth,
        roulette_depth: 3,
    })
    .with_setting(LightingSettings {
        debug_mode: REFERENCE_VIEW,
        ..Default::default()
    });
    stream::offer(&mut env, tracer(scene_buffer.clone()));
    light_samples::offer(&mut env, sampler(scene_buffer));
    let mut registry = NodeRegistry::new();
    FeatureSet::new()
        .with(ReferenceFeature)
        .register_into(&mut registry);
    let mut graph = RenderGraph::new(&gpu.device, &gpu.queue);
    let node = registry
        .get("helio.pass.reference")
        .expect("the node is registered");
    let params = node.effective_params(&Default::default());
    node.run(&mut NodeBuild {
        graph: &mut graph,
        env: &mut env,
        params: &params,
        node_id: "reference",
    })
    .expect("the reference builds");
    graph.lock(SIZE, SIZE);
    graph.set_debug_mode(REFERENCE_VIEW);
    let mut input = Input {
        gpu,
        camera: camera_buffer,
        camera_data: camera,
        buffers: SceneBufferProjection::empty(),
        frame: 0,
    };
    let texture = |format| {
        gpu.device
            .create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: SIZE,
                    height: SIZE,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default())
    };
    let (target, depth_view) = (
        texture(wgpu::TextureFormat::Rgba8Unorm),
        texture(wgpu::TextureFormat::Depth32Float),
    );
    for _ in 0..frames {
        let mut resources = helio_core::ResourceRegistry::empty();
        graph
            .execute_with_registry(&input, &target, &depth_view, &mut resources)
            .expect("the frame renders");
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        input.frame += 1;
    }
    let image = graph
        .find_pass::<ReferenceDisplay>()
        .and_then(ReferenceDisplay::image)
        .expect("the reference holds its image while its view is shown");
    let sums = read(gpu, &image.image);
    let means = sums
        .iter()
        .take((SIZE * SIZE) as usize)
        .map(|s| Vec3::new(s[0], s[1], s[2]) / s[3].max(1.0))
        .collect();
    (means, image.frames)
}

fn furnace(albedo: f32, emission: f32) -> Scene {
    Scene {
        lo: [-2.0, -1.5, -3.0, 0.0],
        hi: [2.5, 2.0, 1.5, 0.0],
        albedo: [albedo, albedo, albedo, 0.0],
        emission: [emission, emission, emission, 0.0],
        point: [0.0; 4],
        range: [1.0, 0.0, 0.0, 0.0],
        sun: [0.0; 4],
    }
}

fn room_camera() -> GpuCameraUniforms {
    camera(
        Vec3::new(0.3, 0.2, -0.4),
        Vec3::new(1.0, 0.5, -2.0),
        Vec3::Y,
    )
    .0
}

/// A closed white-ish room glowing everywhere: each depth adds a^k E.
#[test]
fn the_furnace_converges_to_its_analytic_radiance_at_every_depth() {
    let Some(gpu) = gpu() else { return };
    let (a, e) = (0.5f32, 1.0f32);
    for (depth, frames) in [(1u32, 8u32), (2, 64), (3, 64), (12, 256)] {
        let (image, summed) = render(gpu, furnace(a, e), room_camera(), depth, frames);
        assert_eq!(summed, frames);
        // The albedo as a surface row holds it (8 bits).
        let a8 = (a * 255.0).round() / 255.0;
        let expected = e * (1.0 - a8.powi(depth as i32)) / (1.0 - a8);
        let mean: f32 = image.iter().map(|c| c.x).sum::<f32>() / image.len() as f32;
        // Depths 1 to 3 have no roulette: every path carries exactly
        // a^k of each bounce's emission (a Lambertian sample's weight is its
        // albedo), so the mean is exact up to f32 and the packing. Depth 12 is a Monte
        // Carlo estimate: 1% over 1024 x 256 paths.
        let tolerance = if depth <= 3 { 1e-3 } else { 0.01 } * expected;
        println!("depth {depth}: mean {mean}, analytic {expected}");
        assert!(
            (mean - expected).abs() <= tolerance,
            "depth {depth}: {mean} against {expected}"
        );
        for c in &image {
            assert!(
                c.is_finite() && (c.x - c.y).abs() < 1e-4 && (c.x - c.z).abs() < 1e-4,
                "{c}"
            );
        }
    }
}

/// Ground of albedo rho under a point light and a sun, direct light only.
#[test]
fn direct_light_matches_the_analytic_irradiance() {
    let Some(gpu) = gpu() else { return };
    let rho = 153.0f32 / 255.0;
    let light = Vec3::new(0.4, 2.0, -0.3);
    let (intensity, range) = (30.0f32, 8.0f32);
    let sun = Vec3::new(0.3, 1.0, 0.5).normalize();
    let e_sun = 2.0f32;
    let scene = Scene {
        lo: [0.0; 4],
        hi: [0.0; 4],
        albedo: [rho, rho, rho, 1.0],
        emission: [0.0; 4],
        point: [light.x, light.y, light.z, intensity],
        range: [range, 0.0, 0.0, 0.0],
        sun: [sun.x, sun.y, sun.z, e_sun],
    };
    let eye = Vec3::new(0.0, 3.0, 0.0);
    let (camera, clip_to_frame) = camera(eye, Vec3::new(0.0, 0.0, -0.01), Vec3::NEG_Z);
    let frames = 512;
    let (image, _) = render(gpu, scene, camera, 1, frames);
    // The analytic radiance at the ground point under each pixel's centre
    // (the light varies smoothly over a pixel's jitter).
    let mut worst = 0.0f32;
    let mut sum_error = 0.0f32;
    for y in 0..SIZE {
        for x in 0..SIZE {
            let ndc = glam::Vec2::new(
                (x as f32 + 0.5) / SIZE as f32 * 2.0 - 1.0,
                1.0 - (y as f32 + 0.5) / SIZE as f32 * 2.0,
            );
            let near = clip_to_frame * ndc.extend(1.0).extend(1.0);
            let mid = clip_to_frame * ndc.extend(0.5).extend(1.0);
            let dir = (mid.truncate() / mid.w - near.truncate() / near.w).normalize();
            let p = eye + dir * (-eye.y / dir.y);
            let to_light = light - p;
            let d = to_light.length();
            let win = (1.0 - (d / range).powi(4)).max(0.0);
            let analytic = rho / std::f32::consts::PI
                * (intensity * win * (to_light.y / d) / (d * d) + e_sun * sun.y);
            let got = image[(y * SIZE + x) as usize].x;
            let error = (got - analytic).abs() / analytic;
            worst = worst.max(error);
            sum_error += error;
        }
    }
    let mean_error = sum_error / (SIZE * SIZE) as f32;
    println!("direct light: mean relative error {mean_error}, worst {worst}");
    // 512 samples a pixel, one of two lights each: a few percent a pixel,
    // well under 1% on average (the pixel's jitter adds its gradient's
    // second order).
    assert!(mean_error < 0.01, "mean relative error {mean_error}");
    assert!(worst < 0.08, "worst relative error {worst}");
}

/// Four times the paths, half the error.
#[test]
fn the_error_falls_as_one_over_the_square_root_of_the_samples() {
    let Some(gpu) = gpu() else { return };
    let (a, e) = (0.7f32, 1.0f32);
    let depth = 12;
    let a8 = (a * 255.0).round() / 255.0;
    let expected = e * (1.0 - a8.powi(depth as i32)) / (1.0 - a8);
    let rms = |frames| {
        let (image, _) = render(gpu, furnace(a, e), room_camera(), depth, frames);
        (image.iter().map(|c| (c.x - expected).powi(2)).sum::<f32>() / image.len() as f32).sqrt()
    };
    let (coarse, fine) = (rms(16), rms(64));
    let ratio = coarse / fine;
    println!("rms error {coarse} at 16 paths a pixel, {fine} at 64: ratio {ratio}");
    assert!((1.6..2.5).contains(&ratio), "ratio {ratio}");
}
