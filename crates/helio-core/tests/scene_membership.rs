//! The graph follows the scene: a pass whose content is SceneDB rows joins
//! the schedule when a row is live and leaves when none is, keeping every
//! other graph texture as it was; a resize keeps every pass and gives each
//! both resolutions.

use bytemuck::Zeroable;
use helio_core::graph::{ResourceBuilder, ResourceFormat, ResourceSize};
use helio_core::{
    BufferHandle, BufferKey, GpuCameraUniforms, PassContext, PrepareContext, RenderGraph,
    RenderPass, RenderSize, Result as HelioResult, SceneBufferProjection, SceneContent, SceneInput,
};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

const ROWS: &str = "test_rows";

/// A scene whose `test_rows` buffer the test writes.
struct RowScene {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    frame: u64,
    camera: wgpu::Buffer,
    camera_data: GpuCameraUniforms,
    rows: wgpu::Buffer,
    generation: u64,
    buffers: SceneBufferProjection,
}

impl RowScene {
    fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let camera = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Membership Camera"),
            size: std::mem::size_of::<GpuCameraUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let rows = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Membership Rows"),
            size: 64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut scene = Self {
            device,
            queue,
            frame: 0,
            camera,
            camera_data: GpuCameraUniforms::zeroed(),
            rows,
            generation: 0,
            buffers: SceneBufferProjection::empty(),
        };
        scene.write_rows(0);
        scene
    }

    /// Fills the four 16-byte rows with `value` (0 = no live row), as a
    /// SceneDB flush would, and reports new contents.
    fn write_rows(&mut self, value: u32) {
        self.queue
            .write_buffer(&self.rows, 0, bytemuck::cast_slice(&[value; 16]));
        self.generation += 1;
        self.buffers = SceneBufferProjection::from_handles([(
            BufferKey::of(ROWS),
            BufferHandle {
                buffer: self.rows.clone(),
                epoch: 0,
                row_bytes: 16,
                content_generation: self.generation,
            },
        )]);
    }
}

impl SceneInput for RowScene {
    fn world_origin(&self) -> glam::DVec3 {
        glam::DVec3::ZERO
    }
    fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }
    fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
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

#[derive(Default)]
struct Counters {
    prepared: AtomicU32,
    left: AtomicU32,
    resized: Mutex<Vec<RenderSize>>,
}

/// Always a member: writes `main` (internal size) and `presented` (output
/// size), and records the views it is given.
struct MainPass {
    counters: Arc<Counters>,
    views: Arc<Mutex<Vec<wgpu::TextureView>>>,
    sizes: Arc<Mutex<Vec<(wgpu::Extent3d, wgpu::Extent3d)>>>,
}

impl RenderPass for MainPass {
    fn name(&self) -> &'static str {
        "Main"
    }
    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.write_color(
            "main",
            ResourceFormat::Rgba8Unorm,
            ResourceSize::MatchSurface,
        );
        builder.write_color(
            "presented",
            ResourceFormat::Rgba8Unorm,
            ResourceSize::Output,
        );
    }
    fn on_resize(&mut self, _device: &wgpu::Device, size: RenderSize) {
        self.counters.resized.lock().unwrap().push(size);
    }
    fn prepare(&mut self, _ctx: &PrepareContext) -> HelioResult<()> {
        self.counters.prepared.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let view = ctx.resource_pool.get_view("main").expect("main").clone();
        self.views.lock().unwrap().push(view);
        let main = ctx.resource_pool.get_texture("main").expect("main").size();
        let presented = ctx
            .resource_pool
            .get_texture("presented")
            .expect("presented")
            .size();
        self.sizes.lock().unwrap().push((main, presented));
        Ok(())
    }
}

/// A member while `test_rows` holds a live row; writes its own texture.
struct ContentPass {
    counters: Arc<Counters>,
}

impl RenderPass for ContentPass {
    fn name(&self) -> &'static str {
        "Content"
    }
    fn is_member(&mut self, scene: &mut SceneContent<'_>) -> bool {
        scene.holds_rows(ROWS)
    }
    fn on_leave(&mut self, _device: &wgpu::Device) {
        self.counters.left.fetch_add(1, Ordering::Relaxed);
    }
    fn on_resize(&mut self, _device: &wgpu::Device, size: RenderSize) {
        self.counters.resized.lock().unwrap().push(size);
    }
    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.write_color(
            "content",
            ResourceFormat::Rgba8Unorm,
            ResourceSize::MatchSurface,
        );
    }
    fn prepare(&mut self, _ctx: &PrepareContext) -> HelioResult<()> {
        self.counters.prepared.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        assert!(
            ctx.resource_pool.get_view("content").is_some(),
            "a member's graph texture exists"
        );
        Ok(())
    }
}

struct Gpu {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

fn gpu() -> Option<Gpu> {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(helio_core::wgpu_instance_descriptor());
        let mut adapter = None;
        for force_fallback_adapter in [false, true] {
            if let Ok(found) = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter,
                    apply_limit_buckets: false,
                })
                .await
            {
                adapter = Some(found);
                break;
            }
        }
        let adapter = adapter?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Scene Membership Device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .ok()?;
        device.on_uncaptured_error(Arc::new(|error| {
            panic!("scene membership validation error: {error:?}");
        }));
        Some(Gpu {
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    })
}

fn frame_views(device: &wgpu::Device, size: [u32; 2]) -> (wgpu::TextureView, wgpu::TextureView) {
    let texture = |label, format| {
        device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: size[0],
                    height: size[1],
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default())
    };
    (
        texture("Membership Target", wgpu::TextureFormat::Rgba8Unorm),
        texture("Membership Depth", wgpu::TextureFormat::Depth32Float),
    )
}

/// Executes `frames` frames, waiting for the GPU after each so row readbacks
/// complete between them.
fn run(graph: &mut RenderGraph, scene: &mut RowScene, size: [u32; 2], frames: u32) {
    let (target, depth) = frame_views(&scene.device, size);
    for _ in 0..frames {
        graph.execute(&*scene, &target, &depth).expect("frame");
        scene
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("frame completes");
        scene.frame += 1;
    }
}

#[test]
fn a_pass_joins_while_its_rows_are_live_and_other_textures_stay() {
    let Some(gpu) = gpu() else {
        eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: scene membership");
        return;
    };
    let mut scene = RowScene::new(Arc::clone(&gpu.device), Arc::clone(&gpu.queue));
    let main = Arc::new(Counters::default());
    let content = Arc::new(Counters::default());
    let views = Arc::new(Mutex::new(Vec::new()));
    let mut graph = RenderGraph::new(&gpu.device, &gpu.queue);
    graph.add_pass(Box::new(MainPass {
        counters: Arc::clone(&main),
        views: Arc::clone(&views),
        sizes: Default::default(),
    }));
    graph.add_pass(Box::new(ContentPass {
        counters: Arc::clone(&content),
    }));
    graph.lock(32, 24);

    // No scene yet at lock: the content pass starts dormant, and empty rows
    // keep it there.
    assert_eq!(graph.member_names(), ["Main"]);
    assert_eq!(graph.dormant_count(), 1);
    assert!(
        graph.find_pass::<ContentPass>().is_some(),
        "dormant passes stay findable"
    );
    run(&mut graph, &mut scene, [32, 24], 4);
    assert_eq!(content.prepared.load(Ordering::Relaxed), 0);
    assert_eq!(graph.member_names(), ["Main"]);

    // A live row: the pass joins within a frame or two of its readback, and
    // the always-member's texture is the same one (bind groups over it stay
    // valid; no resize pulse).
    scene.write_rows(7);
    run(&mut graph, &mut scene, [32, 24], 4);
    assert_eq!(graph.member_names(), ["Main", "Content"]);
    assert!(content.prepared.load(Ordering::Relaxed) >= 2);
    assert!(
        main.resized.lock().unwrap().is_empty(),
        "joining is not a resize"
    );
    {
        let views = views.lock().unwrap();
        assert!(
            views.windows(2).all(|pair| pair[0] == pair[1]),
            "`main` kept its texture across the reschedule"
        );
    }

    // Every row cleared: it leaves, once.
    scene.write_rows(0);
    run(&mut graph, &mut scene, [32, 24], 4);
    assert_eq!(graph.member_names(), ["Main"]);
    assert_eq!(content.left.load(Ordering::Relaxed), 1);
    let prepared = content.prepared.load(Ordering::Relaxed);
    run(&mut graph, &mut scene, [32, 24], 2);
    assert_eq!(
        content.prepared.load(Ordering::Relaxed),
        prepared,
        "dormant: not prepared"
    );
    let views = views.lock().unwrap();
    assert!(views.windows(2).all(|pair| pair[0] == pair[1]));
}

#[test]
fn resize_keeps_passes_and_gives_both_resolutions() {
    let Some(gpu) = gpu() else {
        eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: scene membership resize");
        return;
    };
    let mut scene = RowScene::new(Arc::clone(&gpu.device), Arc::clone(&gpu.queue));
    let main = Arc::new(Counters::default());
    let content = Arc::new(Counters::default());
    let sizes = Arc::new(Mutex::new(Vec::new()));
    let mut graph = RenderGraph::new(&gpu.device, &gpu.queue);
    graph.add_pass(Box::new(MainPass {
        counters: Arc::clone(&main),
        views: Default::default(),
        sizes: Arc::clone(&sizes),
    }));
    graph.add_pass(Box::new(ContentPass {
        counters: Arc::clone(&content),
    }));
    graph.lock_sized(RenderSize::new([24, 16], [48, 32]));
    run(&mut graph, &mut scene, [48, 32], 1);

    let resized = RenderSize::new([30, 20], [60, 40]);
    graph.resize(resized);
    run(&mut graph, &mut scene, [60, 40], 1);

    // The same pass objects: prepare counted on across the resize.
    assert_eq!(main.prepared.load(Ordering::Relaxed), 2);
    // Scheduled and dormant passes both hear of it.
    assert_eq!(main.resized.lock().unwrap().as_slice(), &[resized]);
    assert_eq!(content.resized.lock().unwrap().as_slice(), &[resized]);
    let extent = |[width, height]: [u32; 2]| wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    assert_eq!(
        sizes.lock().unwrap().as_slice(),
        &[
            (extent([24, 16]), extent([48, 32])),
            (extent([30, 20]), extent([60, 40]))
        ],
        "internal textures follow the internal size, output ones the output"
    );
    assert_eq!(graph.render_size(), resized);
}
