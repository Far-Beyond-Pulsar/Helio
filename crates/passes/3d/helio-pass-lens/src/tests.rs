//! Execute the production lens WGSL through the real pass recording path.
//! GPU tests require an adapter and fail rather than silently skipping.

use super::*;
use wgpu::util::DeviceExt;

const SIZE: u32 = 128;
const NEAR: f32 = 0.1;
const FAR: f32 = 100.0;
const VIEW: u32 = 7;

/// Tests create their own devices; creating several concurrently hangs some
/// drivers (observed on NVIDIA Vulkan), so GPU tests run one at a time.
static GPU_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x7f_ffff;
    if value == 0.0 || exponent <= 0 {
        return sign;
    }
    if exponent >= 31 {
        return sign | 0x7bff;
    }
    sign | ((exponent as u16) << 10) | (mantissa >> 13) as u16
}

fn f16_to_f32(half: u16) -> f32 {
    let sign = if half & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = ((half >> 10) & 0x1f) as i32;
    let mantissa = (half & 0x3ff) as f32;
    match exponent {
        0 => sign * mantissa * 2f32.powi(-24),
        31 => sign * f32::INFINITY,
        _ => sign * (1.0 + mantissa / 1024.0) * 2f32.powi(exponent - 15),
    }
}

/// An image and the view distance (metres, `f32::INFINITY` for sky) of every
/// pixel, row-major.
struct Scene {
    color: Vec<[f32; 3]>,
    distance: Vec<f32>,
}

impl Scene {
    fn new(color: impl Fn(u32, u32) -> [f32; 3], distance: impl Fn(u32, u32) -> f32) -> Self {
        let pixels = || (0..SIZE).flat_map(|y| (0..SIZE).map(move |x| (x, y)));
        Self {
            color: pixels().map(|(x, y)| color(x, y)).collect(),
            distance: pixels().map(|(x, y)| distance(x, y)).collect(),
        }
    }

    /// A fine two-tone checker: every pixel differs from its neighbours, so
    /// any blur shows.
    fn checker(x: u32, y: u32) -> [f32; 3] {
        let v = if (x + y) % 2 == 0 { 0.2 } else { 2.0 };
        [v, v * 0.5, v * 2.0]
    }
}

struct Rig {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pass: LensPass,
    camera: wgpu::Buffer,
    _serial: std::sync::MutexGuard<'static, ()>,
}

/// One recorded frame's results.
struct Frame {
    output: Vec<u16>,
    coc: Vec<[f32; 4]>,
}

impl Frame {
    fn pixel(&self, x: u32, y: u32) -> [f32; 3] {
        let i = ((y * SIZE + x) * 4) as usize;
        [f16_to_f32(self.output[i]), f16_to_f32(self.output[i + 1]), f16_to_f32(self.output[i + 2])]
    }

    fn coc(&self, x: u32, y: u32) -> [f32; 4] {
        self.coc[(y * SIZE + x) as usize]
    }
}

fn bytes<T: bytemuck::Pod>(rows: &[T]) -> &[u8] {
    bytemuck::cast_slice(rows)
}

impl Rig {
    fn new() -> Self {
        let serial = GPU_SERIAL.lock().unwrap_or_else(|poison| poison.into_inner());
        let (device, queue) = pollster::block_on(async {
            let instance = wgpu::Instance::new(
                wgpu::InstanceDescriptor::new_without_display_handle_from_env(),
            );
            let adapter = instance
                .request_adapter(&Default::default())
                .await
                .expect("GPU adapter required for lens regressions");
            adapter.request_device(&Default::default()).await.unwrap()
        });
        let projection = glam::camera::rh::proj::directx::perspective(0.8, 1.0, NEAR, FAR);
        let camera = helio_core::GpuCameraUniforms::new(
            glam::Mat4::IDENTITY,
            projection,
            glam::Vec3::ZERO,
            NEAR,
            FAR,
            0,
            [0.0; 2],
            glam::Mat4::IDENTITY,
        )
        .with_view_id(VIEW);
        let camera = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lens test cameras"),
            contents: bytes(&[camera, camera]),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let pass = LensPass::new(&device, SIZE, SIZE);
        Self { device, queue, pass, camera, _serial: serial }
    }

    fn color_view(&self, scene: &Scene) -> wgpu::TextureView {
        let mut texels = vec![0u16; (SIZE * SIZE * 4) as usize];
        for (i, c) in scene.color.iter().enumerate() {
            for (k, v) in c.iter().enumerate() {
                texels[i * 4 + k] = f32_to_f16(*v);
            }
            texels[i * 4 + 3] = f32_to_f16(1.0);
        }
        self.device
            .create_texture_with_data(
                &self.queue,
                &wgpu::TextureDescriptor {
                    label: Some("lens test colour"),
                    size: wgpu::Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                },
                Default::default(),
                bytes(&texels),
            )
            .create_view(&Default::default())
    }

    /// The raw depth-buffer values of a perspective projection with `NEAR`
    /// and `FAR` (the pass reads them as plain floats).
    fn depth_view(&self, scene: &Scene) -> wgpu::TextureView {
        let raw: Vec<f32> = scene
            .distance
            .iter()
            .map(|&d| if d.is_finite() { FAR * (d - NEAR) / (d * (FAR - NEAR)) } else { 1.0 })
            .collect();
        self.device
            .create_texture_with_data(
                &self.queue,
                &wgpu::TextureDescriptor {
                    label: Some("lens test depth"),
                    size: wgpu::Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::R32Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
                Default::default(),
                bytes(&raw),
            )
            .create_view(&Default::default())
    }

    fn read_buffer(&self, buffer: &wgpu::Buffer, size: u64) -> Vec<u8> {
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lens test readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &readback, 0, size);
        self.queue.submit([encoder.finish()]);
        self.map(&readback)
    }

    fn map(&self, readback: &wgpu::Buffer) -> Vec<u8> {
        let (tx, rx) = std::sync::mpsc::channel();
        readback.slice(..).map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let data = readback.slice(..).get_mapped_range().unwrap().to_vec();
        data
    }

    fn read_texture(&self, view: &wgpu::TextureView) -> Vec<u16> {
        let (w, h) = (view.texture().width(), view.texture().height());
        let row_bytes = w * 8;
        assert_eq!(row_bytes % wgpu::COPY_BYTES_PER_ROW_ALIGNMENT, 0);
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lens test texture readback"),
            size: u64::from(row_bytes * h),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            view.texture().as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row_bytes),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.queue.submit([encoder.finish()]);
        bytemuck::cast_slice(&self.map(&readback)).to_vec()
    }

    /// Record the pass once. `lenses` and `elements` are the SceneDB rows,
    /// or `None` for a graph that has neither buffer.
    fn record(
        &mut self,
        scene: &Scene,
        lenses: Option<&[LensComponent]>,
        elements: Option<&[LensElementComponent]>,
    ) -> Frame {
        let color = self.color_view(scene);
        let depth = self.depth_view(scene);
        let buffer = |rows: &[u8]| {
            self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("lens test rows"),
                contents: rows,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let lenses = lenses.map(|rows| buffer(bytes(rows)));
        let elements = elements.map(|rows| buffer(bytes(rows)));
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.pass.record(
            &self.device,
            &mut CommandRecorder::from_encoder(&mut encoder),
            Some(&color),
            LensInputs {
                camera: Some(&self.camera),
                depth: Some(&depth),
                lenses: lenses.as_ref(),
                elements: elements.as_ref(),
            },
        );
        self.queue.submit([encoder.finish()]);
        let output = self.read_texture(&self.pass.targets.output);
        let coc = self
            .read_texture(&self.pass.targets.coc)
            .chunks_exact(4)
            .map(|t| [f16_to_f32(t[0]), f16_to_f32(t[1]), f16_to_f32(t[2]), f16_to_f32(t[3])])
            .collect();
        Frame { output, coc }
    }

    fn resolved(&self) -> ResolvedLens {
        let size = std::mem::size_of::<ResolvedLens>() as u64;
        bytemuck::pod_read_unaligned(&self.read_buffer(&self.pass.resolved, size))
    }

    fn args(&self) -> Vec<u32> {
        bytemuck::cast_slice(&self.read_buffer(&self.pass.args, ARGS_SIZE)).to_vec()
    }

    /// Frame time for autofocus easing.
    fn set_delta_time(&self, seconds: f32) {
        self.queue.write_buffer(&self.pass.frame, 0, bytes(&[seconds, 0.0, 0.0, 0.0]));
    }
}

/// The thin-lens circle of confusion in pixels for an `SIZE`-wide image.
fn expected_coc(lens: &LensComponent, focus: f32, distance: f32) -> f32 {
    let f = lens.focal_length_mm * 1.0e-3;
    let aperture = f / lens.f_number;
    let distance = if distance.is_finite() { distance } else { 1.0e5 };
    let on_sensor = aperture * (distance - focus).abs() / distance * f / (focus - f);
    on_sensor / (lens.sensor_width_mm * 1.0e-3) * SIZE as f32 * (distance - focus).signum()
}

fn lens() -> LensComponent {
    LensComponent::new(VIEW)
}

#[test]
fn component_rows_have_the_wgsl_strides() {
    assert_eq!(std::mem::size_of::<LensComponent>(), 96);
    assert_eq!(std::mem::offset_of!(LensComponent, view_id), 0);
    assert_eq!(std::mem::offset_of!(LensComponent, focal_length_mm), 16);
    assert_eq!(std::mem::offset_of!(LensComponent, focus_distance_m), 36);
    assert_eq!(std::mem::offset_of!(LensComponent, element_first), 56);
    assert_eq!(std::mem::offset_of!(LensComponent, field_curvature), 72);
    assert_eq!(std::mem::offset_of!(LensComponent, af_radius), 84);
    assert_eq!(std::mem::size_of::<LensElementComponent>(), 160);
    assert_eq!(std::mem::offset_of!(LensElementComponent, feather), 16);
    assert_eq!(std::mem::offset_of!(LensElementComponent, params), 32);
    assert_eq!(std::mem::offset_of!(LensElementComponent, region), 96);
    assert_eq!(std::mem::size_of::<ResolvedLens>(), 64);
    // A zero row is inert: never enabled, no element kind.
    let zero: LensComponent = bytemuck::Zeroable::zeroed();
    assert_eq!(zero.enabled, 0);
    let zero: LensElementComponent = bytemuck::Zeroable::zeroed();
    assert_eq!(zero.kind, element_kind::NONE);
}

#[test]
fn no_enabled_lens_for_the_view_passes_the_image_through_exactly() {
    let mut rig = Rig::new();
    // HDR values far above 1, so nothing here is clamped or tone mapped.
    let scene = Scene::new(
        |x, y| [x as f32 * 0.37, y as f32 * 3.1, 40.0 + (x ^ y) as f32],
        |x, _| 1.0 + x as f32 * 0.1,
    );
    let color = rig.color_view(&scene);
    let expected = rig.read_texture(&color);

    // A graph without the lens buffers.
    let frame = rig.record(&scene, None, None);
    assert_eq!(frame.output, expected, "missing buffers changed the image");
    let args = rig.args();
    assert_eq!(&args[..15], &[0; 15], "defocus dispatches must be empty with no lens");
    assert!(args[15] > 0, "the copy is still dispatched");
    let resolved = rig.resolved();
    assert_eq!(resolved, ResolvedLens { focus_distance_m: resolved.focus_distance_m, ..Default::default() });

    // A disabled row, and an enabled row belonging to another view.
    let mut disabled = lens();
    disabled.enabled = 0;
    let mut other = lens();
    other.view_id = VIEW + 1;
    let frame = rig.record(&scene, Some(&[disabled, other]), None);
    assert_eq!(frame.output, expected, "an unmatched lens changed the image");
    assert_eq!(rig.resolved().enabled, 0);
}

#[test]
fn resolve_matches_the_first_enabled_row_of_the_view() {
    let mut rig = Rig::new();
    let scene = Scene::new(Scene::checker, |_, _| 3.0);
    let mut other_view = lens();
    other_view.view_id = VIEW + 1;
    other_view.f_number = 9.0;
    let mut disabled = lens();
    disabled.enabled = 0;
    disabled.f_number = 8.0;
    let mut wanted = lens();
    wanted.focal_length_mm = 85.0;
    wanted.f_number = 1.8;
    wanted.sensor_width_mm = 24.0;
    wanted.sensor_height_mm = 0.0;
    wanted.focus_distance_m = 4.0;
    wanted.aperture_blades = 9;
    wanted.aperture_rotation = 0.25;
    wanted.max_coc_px = 12.0;
    wanted.quality = 1;
    wanted.element_first = 3;
    wanted.element_count = 2;
    let mut later = lens();
    later.f_number = 5.6;
    rig.record(&scene, Some(&[other_view, disabled, wanted, later]), None);
    let resolved = rig.resolved();
    assert_eq!(resolved.enabled, 1);
    assert_eq!(resolved.f_number, 1.8);
    assert_eq!(resolved.focal_length_mm, 85.0);
    assert_eq!(resolved.sensor_width_mm, 24.0);
    assert_eq!(resolved.sensor_height_mm, 16.0, "zero height derives 3:2");
    assert!((resolved.focus_distance_m - 4.0).abs() < 1.0e-4, "{resolved:?}");
    assert_eq!((resolved.aperture_blades, resolved.aperture_rotation), (9, 0.25));
    assert_eq!((resolved.max_coc_px, resolved.quality), (12.0, 1));
    assert_eq!((resolved.element_first, resolved.element_count), (3, 2));
    let args = rig.args();
    assert!(args[..18].iter().all(|&n| n > 0), "an enabled lens dispatches every stage: {args:?}");
}

#[test]
fn thin_lens_circle_of_confusion_matches_the_formula() {
    let mut rig = Rig::new();
    // Six depth bands from in front of the focus to the sky.
    let bands = [1.0, 2.0, 3.0, 5.0, 10.0, f32::INFINITY];
    let scene = Scene::new(Scene::checker, |x, _| bands[(x * 6 / SIZE) as usize]);
    let mut lens = lens();
    lens.focal_length_mm = 100.0;
    lens.f_number = 1.4;
    lens.focus_distance_m = 3.0;
    let frame = rig.record(&scene, Some(&[lens]), None);
    for (band, &distance) in bands.iter().enumerate() {
        let x = (band as u32 * SIZE + SIZE / 2) / 6;
        let expected = expected_coc(&lens, 3.0, distance);
        let [coc_a, coc_b, seam, transmission] = frame.coc(x, SIZE / 2);
        assert!(
            (coc_a - expected).abs() <= 0.01 * expected.abs() + 0.03,
            "depth {distance}: circle {coc_a}, expected {expected}"
        );
        assert_eq!((coc_b, seam, transmission), (coc_a, 0.0, 1.0), "no seam without elements");
    }
    // Near geometry is negative, far positive, the focal plane exactly sharp.
    assert!(frame.coc(10, 64)[0] < -10.0 && frame.coc(SIZE - 10, 64)[0] > 5.0);
    assert!(frame.coc(SIZE / 2 - 4, 64)[0].abs() < 0.03, "the focal plane is sharp");

    // The aperture scales every circle; the clamp bounds them.
    lens.f_number = 2.8;
    let frame = rig.record(&scene, Some(&[lens]), None);
    let expected = expected_coc(&lens, 3.0, 1.0);
    assert!((frame.coc(10, 64)[0] - expected).abs() < 0.01 * expected.abs() + 0.03);
    lens.f_number = 1.4;
    lens.max_coc_px = 6.0;
    let frame = rig.record(&scene, Some(&[lens]), None);
    assert_eq!(frame.coc(10, 64)[0], -6.0);
}

#[test]
fn depth_rendered_below_image_resolution_keeps_foreground_focus() {
    let mut rig = Rig::new();
    // The depth buffer is a quarter-size texture; the foreground column's
    // footprint straddles two texels and must take the nearer.
    let scene = Scene::new(Scene::checker, |x, _| if (60..68).contains(&x) { 1.0 } else { 9.0 });
    let color = rig.color_view(&scene);
    let raw: Vec<f32> = (0..32 * 32)
        .map(|i| {
            let d: f32 = if (15..17).contains(&(i % 32)) { 1.0 } else { 9.0 };
            FAR * (d - NEAR) / (d * (FAR - NEAR))
        })
        .collect();
    let depth = rig
        .device
        .create_texture_with_data(
            &rig.queue,
            &wgpu::TextureDescriptor {
                label: Some("lens test low-resolution depth"),
                size: wgpu::Extent3d { width: 32, height: 32, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            Default::default(),
            bytes(&raw),
        )
        .create_view(&Default::default());
    let mut lens = lens();
    lens.focus_distance_m = 1.0;
    lens.focal_length_mm = 100.0;
    lens.f_number = 1.4;
    let lenses = rig.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("lens test rows"),
        contents: bytes(&[lens]),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let mut encoder = rig.device.create_command_encoder(&Default::default());
    rig.pass.record(
        &rig.device,
        &mut CommandRecorder::from_encoder(&mut encoder),
        Some(&color),
        LensInputs {
            camera: Some(&rig.camera),
            depth: Some(&depth),
            lenses: Some(&lenses),
            elements: None,
        },
    );
    rig.queue.submit([encoder.finish()]);
    let coc = rig.read_texture(&rig.pass.targets.coc);
    let at = |x: u32| f16_to_f32(coc[((64 * SIZE + x) * 4) as usize]);
    assert!(at(62).abs() < 0.05, "foreground focus {}", at(62));
    assert!(at(20) > 3.0, "background blur {}", at(20));
}

#[test]
fn defocus_preserves_energy() {
    let mut rig = Rig::new();
    // A bright patch and a dimmer one on a dim floor, all on one plane far
    // behind the focus: a uniform circle of confusion of several pixels.
    let scene = Scene::new(
        |x, y| match (x, y) {
            (48..80, 48..80) => [8.0, 6.0, 4.0],
            (20..28, 90..98) => [3.0, 3.0, 12.0],
            _ => [0.05, 0.06, 0.07],
        },
        |_, _| f32::INFINITY,
    );
    let mut lens = lens();
    lens.focal_length_mm = 50.0;
    lens.f_number = 1.4;
    lens.focus_distance_m = 1.0;
    for quality in [0, 1] {
        lens.quality = quality;
        let frame = rig.record(&scene, Some(&[lens]), None);
        let expected = expected_coc(&lens, 1.0, f32::INFINITY);
        assert!(expected > 5.0 && expected < 8.0, "test geometry: circle {expected}");
        assert!((frame.coc(64, 64)[0] - expected).abs() < 0.2);
        for channel in 0..3 {
            let sum = |pixel: &dyn Fn(u32, u32) -> f32| -> f32 {
                (0..SIZE).flat_map(|y| (0..SIZE).map(move |x| (x, y))).map(|(x, y)| pixel(x, y)).sum()
            };
            let before = sum(&|x, y| scene.color[(y * SIZE + x) as usize][channel]);
            let after = sum(&|x, y| frame.pixel(x, y)[channel]);
            assert!(
                (after / before - 1.0).abs() < 0.03,
                "quality {quality} channel {channel}: energy {before} -> {after}"
            );
        }
        // And it really did blur: the patch edge spreads outward and the
        // patch's own corner is no longer at full strength.
        assert!(frame.pixel(45, 64)[0] > 0.5, "no spill outside the patch");
        assert!(frame.pixel(49, 49)[0] < 7.0, "the patch corner stayed sharp");
        assert!(frame.pixel(64, 64)[0] > 7.0, "the patch interior lost energy");
    }
}

#[test]
fn near_field_blur_covers_what_is_behind_it() {
    let mut rig = Rig::new();
    // A bright near square, out of focus, in front of a dim in-focus wall.
    let scene = Scene::new(
        |x, y| if (48..80).contains(&x) && (48..80).contains(&y) { [10.0; 3] } else { [0.5; 3] },
        |x, y| if (48..80).contains(&x) && (48..80).contains(&y) { 0.6 } else { 3.0 },
    );
    let mut lens = lens();
    lens.focal_length_mm = 50.0;
    lens.f_number = 1.4;
    lens.focus_distance_m = 3.0;
    let frame = rig.record(&scene, Some(&[lens]), None);
    assert!(frame.coc(64, 64)[0] < -5.0, "near square is blurred");
    assert!(frame.coc(10, 10)[0].abs() < 0.03, "the wall is in focus");
    // The blurred square bleeds over the sharp wall, and the wall's far side
    // is untouched.
    assert!(frame.pixel(45, 64)[0] > 1.0, "near blur must cover the wall");
    // Inside its own edge the blurred square fades into what is behind it.
    let inner = frame.pixel(49, 64)[0];
    assert!(inner > 3.0 && inner < 9.5, "inner edge {inner}");
    assert_eq!(frame.pixel(5, 5), [0.5; 3]);
    let total = |frame: &Frame| -> f32 {
        (0..SIZE).flat_map(|y| (0..SIZE).map(move |x| (x, y))).map(|(x, y)| frame.pixel(x, y)[0]).sum()
    };
    let before: f32 = scene.color.iter().map(|c| c[0]).sum();
    // The square's light spreads over wall it hides; the displaced wall
    // light is the loss, so the total stays within a few percent.
    assert!((total(&frame) / before - 1.0).abs() < 0.06, "{} vs {before}", total(&frame));
}

/// Pixels of a row whose seam weight exceeds `threshold`.
fn seam_width(frame: &Frame, y: u32) -> u32 {
    (0..SIZE).filter(|&x| frame.coc(x, y)[2] > 0.02).count() as u32
}

#[test]
fn split_diopter_focuses_each_half_and_the_seam_follows_the_aperture() {
    let mut rig = Rig::new();
    // The left half-frame is a plane 2 m away, the right one 6 m away, both
    // carrying fine detail. The lens focuses 6 m; a diopter over the left
    // half brings the focus to 2 m there.
    let scene = Scene::new(Scene::checker, |x, _| if x < SIZE / 2 { 2.0 } else { 6.0 });
    let power = 1.0 / 2.0 - 1.0 / 6.0;
    let diopter = LensElementComponent::diopter(
        power,
        LensRegion::HalfPlane { angle: std::f32::consts::PI, offset: 0.0 },
    )
    .with_mount_distance(0.5);
    let mut lens = lens();
    lens.focal_length_mm = 85.0;
    lens.f_number = 1.4;
    lens.focus_distance_m = 6.0;
    lens.element_count = 1;

    let wide = rig.record(&scene, Some(&[lens]), Some(&[diopter]));
    for x in (0..28).chain(100..SIZE) {
        let [coc_a, _, seam, _] = wide.coc(x, 64);
        assert!(coc_a.abs() < 0.1 && seam == 0.0, "column {x}: circle {coc_a}, seam {seam}");
        assert_eq!(wide.pixel(x, 64), scene.color[(64 * SIZE + x) as usize].map(|v| f16_to_f32(f32_to_f16(v))));
    }
    // Without the diopter the near plane is out of focus.
    lens.element_count = 0;
    let control = rig.record(&scene, Some(&[lens]), Some(&[diopter]));
    assert!(control.coc(10, 64)[0] < -4.0, "control circle {}", control.coc(10, 64)[0]);
    assert_ne!(control.pixel(10, 64), scene.color[(64 * SIZE + 10) as usize]);

    // The seam is the entrance pupil seen from the filter: A / e radians over
    // the sensor, so it narrows with the aperture.
    let widths: Vec<(f32, u32)> = [1.4, 5.6]
        .into_iter()
        .map(|f_number| {
            lens.element_count = 1;
            lens.f_number = f_number;
            let frame = rig.record(&scene, Some(&[lens]), Some(&[diopter]));
            // Both configurations meet at half weight in the middle.
            assert!(frame.coc(63, 64)[2] > 0.4 && frame.coc(63, 64)[2] <= 0.5);
            let aperture = 85.0e-3 / f_number;
            let feather = aperture / 0.5 * (85.0e-3 / 24.0e-3) * SIZE as f32;
            // Smoothstep: the weight is above 2% for 83% of the feather.
            let expected = 0.832 * feather;
            let width = seam_width(&frame, 64);
            assert!(
                (width as f32 - expected).abs() < 2.5,
                "f/{f_number}: seam {width} px, expected {expected}"
            );
            (f_number, width)
        })
        .collect();
    let ratio = widths[0].1 as f32 / widths[1].1 as f32;
    assert!((ratio - 4.0).abs() < 0.6, "seam widths {widths:?} do not scale with 1/N");

    // An authored feather replaces the physical one.
    lens.f_number = 1.4;
    let hard = diopter.with_feather(0.02);
    let frame = rig.record(&scene, Some(&[lens]), Some(&[hard]));
    assert!(seam_width(&frame, 64) <= 3, "authored feather ignored: {}", seam_width(&frame, 64));
}

#[test]
fn diopters_stack_across_many_regions() {
    let mut rig = Rig::new();
    // Three vertical thirds at 1.5, 3 and 9 m, each focused by the sum of the
    // diopters covering it.
    let scene = Scene::new(Scene::checker, |x, _| match x * 3 / SIZE {
        0 => 1.5,
        1 => 3.0,
        _ => 9.0,
    });
    let to_3m = 1.0 / 3.0 - 1.0 / 9.0;
    let to_1_5m = 1.0 / 1.5 - 1.0 / 3.0;
    let thirds = [
        // Covers x < 1/6 of the image height right of centre: left and middle.
        LensElementComponent::diopter(
            to_3m,
            LensRegion::HalfPlane { angle: std::f32::consts::PI, offset: -1.0 / 6.0 },
        )
        .with_feather(0.004),
        // Covers x < -1/6: the left third, which also has the first.
        LensElementComponent::diopter(
            to_1_5m,
            LensRegion::HalfPlane { angle: std::f32::consts::PI, offset: 1.0 / 6.0 },
        )
        .with_feather(0.004),
    ];
    let mut lens = lens();
    lens.focal_length_mm = 85.0;
    lens.f_number = 1.4;
    lens.focus_distance_m = 9.0;
    lens.element_count = 2;
    let frame = rig.record(&scene, Some(&[lens]), Some(&thirds));
    for x in [6, 20, 64, 90, 115, 124] {
        assert!(frame.coc(x, 64)[0].abs() < 0.1, "column {x}: circle {}", frame.coc(x, 64)[0]);
    }
    // Drop the second diopter: the left third is no longer in focus.
    lens.element_count = 1;
    let frame = rig.record(&scene, Some(&[lens]), Some(&thirds));
    assert!(frame.coc(20, 64)[0].abs() > 2.0, "{}", frame.coc(20, 64)[0]);
    assert!(frame.coc(64, 64)[0].abs() < 0.1);
}

#[test]
fn nested_regions_of_any_shape_focus_independently() {
    let mut rig = Rig::new();
    // A disc at 1.5 m inside a ring at 3 m inside a floor at 9 m.
    let radius = |x: u32, y: u32| {
        let q = [(x as f32 + 0.5) / SIZE as f32 - 0.5, (y as f32 + 0.5) / SIZE as f32 - 0.5];
        (q[0] * q[0] + q[1] * q[1]).sqrt()
    };
    let scene = Scene::new(Scene::checker, |x, y| match radius(x, y) {
        r if r < 0.15 => 1.5,
        r if r < 0.33 => 3.0,
        _ => 9.0,
    });
    let elements = [
        LensElementComponent::diopter(
            1.0 / 3.0 - 1.0 / 9.0,
            LensRegion::Ellipse { center: [0.0; 2], radii: [0.33, 0.33], rotation: 0.0 },
        )
        .with_feather(0.004),
        LensElementComponent::diopter(
            1.0 / 1.5 - 1.0 / 3.0,
            LensRegion::Polygon(&[[-0.15, -0.15], [0.15, -0.15], [0.15, 0.15], [-0.15, 0.15]]),
        )
        .with_feather(0.004),
    ];
    let mut lens = lens();
    lens.focal_length_mm = 85.0;
    lens.f_number = 1.4;
    lens.focus_distance_m = 9.0;
    lens.element_count = 2;
    let frame = rig.record(&scene, Some(&[lens]), Some(&elements));
    // The square covers the disc (and a little more) at the 1.5 m stop.
    for (x, y) in [(64, 64), (60, 70), (4, 4), (124, 64), (64, 100), (64, 30)] {
        let [coc, _, seam, _] = frame.coc(x, y);
        let r = radius(x, y);
        if (0.13..0.17).contains(&r) || (0.31..0.35).contains(&r) {
            continue;
        }
        assert!(coc.abs() < 0.1 && seam == 0.0, "({x}, {y}) r {r}: circle {coc}, seam {seam}");
    }
}

#[test]
fn every_region_kind_covers_what_it_describes() {
    let mut rig = Rig::new();
    // In-focus plane; a sweet spot adds 10 px of defocus outside its region.
    let scene = Scene::new(Scene::checker, |_, _| 3.0);
    let mut lens = lens();
    lens.focus_distance_m = 3.0;
    lens.element_count = 1;
    let triangle = [[-0.2, -0.2], [0.3, -0.1], [0.0, 0.3]];
    let reversed = [[0.0, 0.3], [0.3, -0.1], [-0.2, -0.2]];
    // (region, a pixel inside, a pixel outside)
    let cases: [(LensRegion, (u32, u32), (u32, u32)); 7] = [
        (LensRegion::FullFrame, (64, 64), (5, 5)),
        (LensRegion::HalfPlane { angle: 0.0, offset: 0.0 }, (100, 64), (20, 64)),
        (
            LensRegion::Ellipse { center: [0.1, 0.0], radii: [0.3, 0.1], rotation: 0.0 },
            (70, 64),
            (70, 20),
        ),
        (LensRegion::Polygon(&triangle), (64, 70), (20, 20)),
        (LensRegion::Polygon(&reversed), (64, 70), (20, 20)),
        (LensRegion::RadialBand { r0: 0.2, r1: 0.35, center: [0.0; 2] }, (64 + 33, 64), (64, 64)),
        (LensRegion::Mask, (64, 64), (64, 64)),
    ];
    for (i, (region, inside, outside)) in cases.into_iter().enumerate() {
        let element = LensElementComponent::sweet_spot(10.0, region).with_feather(0.0);
        let frame = rig.record(&scene, Some(&[lens]), Some(&[element]));
        let covered = i != 6;
        let at = |p: (u32, u32)| frame.coc(p.0, p.1)[0];
        if covered {
            assert!(at(inside).abs() < 0.1, "case {i}: inside {}", at(inside));
        }
        if i != 0 {
            assert!((at(outside) - 10.0).abs() < 0.1, "case {i}: outside {}", at(outside));
        }
    }
}

#[test]
fn tilt_and_field_curvature_shape_the_focus_field() {
    let mut rig = Rig::new();
    let scene = Scene::new(Scene::checker, |_, _| 1.0);
    let mut lens = lens();
    lens.focal_length_mm = 100.0;
    lens.f_number = 1.4;
    lens.focus_distance_m = 1.0;
    let flat = rig.record(&scene, Some(&[lens]), None);
    assert!(flat.coc(0, 0)[0].abs() < 0.05 && flat.coc(64, 64)[0].abs() < 0.05);

    // A tilted focal plane meets the wall nearer on one side: opposite
    // circles at the two edges, none on the axis.
    lens.tilt_angle = 0.6;
    lens.tilt_axis = 0.0;
    let tilted = rig.record(&scene, Some(&[lens]), None);
    assert!(tilted.coc(SIZE - 1, 64)[0] > 3.0, "{}", tilted.coc(SIZE - 1, 64)[0]);
    assert!(tilted.coc(0, 64)[0] < -3.0, "{}", tilted.coc(0, 64)[0]);
    assert!(tilted.coc(63, 64)[0].abs() < 0.8);
    // About a vertical axis the field varies vertically instead.
    lens.tilt_axis = std::f32::consts::FRAC_PI_2;
    let vertical = rig.record(&scene, Some(&[lens]), None);
    assert!(vertical.coc(64, 0)[0] > 3.0 || vertical.coc(64, 0)[0] < -3.0);
    assert!(vertical.coc(0, 64)[0].abs() < 1.0);

    // Field curvature shifts the focus with radius: sharp on the axis only.
    lens.tilt_angle = 0.0;
    lens.field_curvature = 1.0;
    let curved = rig.record(&scene, Some(&[lens]), None);
    assert!(curved.coc(64, 64)[0].abs() < 0.2);
    assert!(curved.coc(0, 0)[0].abs() > 5.0, "{}", curved.coc(0, 0)[0]);
}

#[test]
fn autofocus_tracks_the_depth_at_its_point_and_eases() {
    let mut rig = Rig::new();
    let mut lens = lens();
    lens.focus_mode = FOCUS_AUTO_POINT;
    lens.af_u = 0.25;
    lens.af_v = 0.5;
    // A wall at 5 m on the left, 2 m on the right: the point reads the left.
    let scene = Scene::new(Scene::checker, |x, _| if x < SIZE / 2 { 5.0 } else { 2.0 });
    rig.record(&scene, Some(&[lens]), None);
    let focus = rig.resolved().focus_distance_m;
    assert!((focus - 5.0).abs() < 0.05, "autofocus read {focus}");

    // Moving the point reads the other wall; with an ease rate it takes
    // time (in dioptres), and holds with no time passing.
    lens.af_u = 0.75;
    lens.af_speed = 10.0;
    rig.set_delta_time(0.0);
    rig.record(&scene, Some(&[lens]), None);
    assert!((rig.resolved().focus_distance_m - 5.0).abs() < 0.05, "moved without time");
    rig.set_delta_time(1.0 / 60.0);
    rig.record(&scene, Some(&[lens]), None);
    let eased = rig.resolved().focus_distance_m;
    // alpha = 1 - exp(-10 / 60), from 0.2 dpt toward 0.5 dpt.
    let alpha = 1.0 - (-10.0f32 / 60.0).exp();
    let expected = 1.0 / (0.2 + 0.3 * alpha);
    assert!((eased - expected).abs() < 0.02, "eased to {eased}, expected {expected}");
    // Without an ease rate the next read is immediate.
    lens.af_speed = 0.0;
    rig.record(&scene, Some(&[lens]), None);
    assert!((rig.resolved().focus_distance_m - 2.0).abs() < 0.02);

    // A point on the sky holds the last focus instead of racing to infinity.
    let sky = Scene::new(Scene::checker, |_, _| f32::INFINITY);
    rig.record(&sky, Some(&[lens]), None);
    assert!((rig.resolved().focus_distance_m - 2.0).abs() < 0.02);
}

#[test]
fn seam_darkening_dims_the_diopter_edge() {
    let mut rig = Rig::new();
    let scene = Scene::new(|_, _| [1.0; 3], |x, _| if x < SIZE / 2 { 2.0 } else { 6.0 });
    let diopter = LensElementComponent::diopter(
        1.0 / 2.0 - 1.0 / 6.0,
        LensRegion::HalfPlane { angle: std::f32::consts::PI, offset: 0.0 },
    )
    .with_feather(0.1)
    .with_seam_darkening(0.5);
    let mut lens = lens();
    lens.focus_distance_m = 6.0;
    lens.element_count = 1;
    let frame = rig.record(&scene, Some(&[lens]), Some(&[diopter]));
    let [.., centre] = frame.coc(63, 64);
    assert!((centre - 0.5).abs() < 0.08, "darkest at the seam centre: {centre}");
    assert_eq!(frame.coc(10, 64)[3], 1.0);
    assert!(frame.pixel(63, 64)[0] < 0.6 && frame.pixel(10, 64)[0] > 0.99);
}
