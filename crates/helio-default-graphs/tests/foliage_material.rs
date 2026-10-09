//! A foliage type's authored material reaches the frame (Pulsar-Native#1063).
//!
//! Grass is grown by the default graph's foliage passes from one type row and
//! rendered through the "Albedo Only" debug view, which shows the G-buffer
//! albedo without lighting. The same blades are rendered with a red, a blue
//! and no authored colour: the blades must take each authored colour, and the
//! type without one keeps the procedural green.
use std::sync::Arc;

use glam::Vec3;
use helio::{
    required_experimental_features, required_wgpu_features, required_wgpu_limits, Camera,
    RendererBuilder, RendererConfig,
};
use helio_default_graphs::build_default_graph_external_with_context;
use helio_pass_foliage_place::components::{
    FoliageLayerComponent, FoliageTypeComponent, FoliageWindComponent,
};
use helio_pass_foliage_place::{
    pack_kind_and_flags, FoliageKind, FoliageMaterial, GpuFoliageLayer, GpuFoliageType,
    FOLIAGE_FLAG_TWO_SIDED,
};
use pulsar_scenedb::gpu::{EngineGpuContext, GpuMirrorHandle, SceneGpuConfig, SceneGpuStore};

const SIZE: u32 = 128;
/// `DeferredLightPass`'s "Albedo Only" view.
const DEBUG_ALBEDO: u32 = 4;

struct Meadow {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    renderer: helio::Renderer,
    world: pulsar_scenedb::World,
    grass: pulsar_scenedb::Entity,
    target: wgpu::Texture,
}

fn meadow() -> Option<Meadow> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let Ok(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else {
        eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: foliage material");
        return None;
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: required_wgpu_features(adapter.features()),
        required_limits: required_wgpu_limits(adapter.limits()),
        experimental_features: required_experimental_features(adapter.features()),
        ..Default::default()
    }))
    .unwrap();
    let (device, queue) = (Arc::new(device), Arc::new(queue));
    let context = EngineGpuContext::new(Arc::clone(&device), Arc::clone(&queue));
    let mut store = SceneGpuStore::new(
        &context,
        SceneGpuConfig {
            classes: Vec::new(),
            tombstone_headroom: 0,
            max_cells_metadata: 0,
        },
    );
    helio_pass_gbuffer::MeshComponent::register_gpu_columns_growable(&mut store, 4, &device);
    helio_pass_gbuffer::MaterialComponent::register_gpu_columns_growable(&mut store, 4, &device);
    helio_pass_gbuffer::StaticObjectComponent::register_gpu_columns_growable(
        &mut store, 4, &device,
    );
    helio_pass_forward_lit::LightComponent::register_gpu_columns_growable(&mut store, 4, &device);
    FoliageTypeComponent::register_gpu_columns_growable(&mut store, 4, &device);
    FoliageLayerComponent::register_gpu_columns_growable(&mut store, 4, &device);
    FoliageWindComponent::register_gpu_columns_growable(&mut store, 4, &device);
    let mirror = GpuMirrorHandle::new(Arc::new(store), Arc::clone(&queue));

    // One entity carries the type, its layer and the wind, so each lands in
    // row 0 of its table.
    let mut world = pulsar_scenedb::World::new();
    world.attach_gpu_mirror(mirror.clone());
    let grass = world.spawn();
    let row = GpuFoliageType {
        density: 120.0,
        height_range: [0.4, 0.6],
        width_range: [0.03, 0.05],
        slope_range: [0.0, 1.0],
        wind_response: [0.0; 3],
        kind_and_flags: pack_kind_and_flags(FoliageKind::Blade, FOLIAGE_FLAG_TWO_SIDED),
        ..Default::default()
    };
    world.insert(grass, FoliageTypeComponent::from(row));
    world.insert(
        grass,
        FoliageLayerComponent::from(GpuFoliageLayer {
            bounds_min: [-20.0, -1.0, -20.0, 0.0],
            bounds_max: [20.0, 4.0, 20.0, 0.0],
        }),
    );
    world.insert(
        grass,
        FoliageWindComponent {
            direction_speed: [1.0, 0.0, 0.0, 0.0],
            gust: [0.0; 4],
            time_prev_time: [0.0; 2],
            _pad: [0.0; 2],
        },
    );

    let mut config = RendererConfig::new(SIZE, SIZE, wgpu::TextureFormat::Rgba8Unorm);
    config.enable_foliage = true;
    let mut renderer = RendererBuilder::new(config, mirror)
        .with_external_device()
        .with_pass_build_context(Box::new(build_default_graph_external_with_context))
        .build(
            Arc::clone(&device),
            Arc::clone(&queue),
            SIZE,
            SIZE,
            config.surface_format,
        );
    renderer.set_jitter_enabled(false);
    renderer.set_debug_mode(DEBUG_ALBEDO);
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("foliage material target"),
        size: wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: config.surface_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    Some(Meadow {
        device,
        queue,
        renderer,
        world,
        grass,
        target,
    })
}

impl Meadow {
    /// Renders with `material` on the type until placement settles, then
    /// returns the frame's RGBA8 pixels.
    fn frame(&mut self, material: Option<FoliageMaterial>) -> Vec<[u8; 4]> {
        {
            let mut row = self
                .world
                .get_mut::<FoliageTypeComponent>(self.grass)
                .unwrap();
            let mut ty = GpuFoliageType::from(*row);
            ty.set_material(material);
            *row = ty.into();
        }
        // Blades are about half a metre tall: stand 4 m away, 0.7 m up.
        let camera = Camera::perspective_look_at(
            Vec3::new(0.0, 0.7, 4.0),
            Vec3::new(0.0, 0.2, 0.0),
            Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            1.0,
            0.1,
            100.0,
        );
        let view = self.target.create_view(&Default::default());
        for _ in 0..24 {
            self.world.flush_gpu_mirror(&self.queue);
            self.renderer.render(&camera, &view).unwrap();
            self.device
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
        }

        let row = (SIZE * 4).div_ceil(256) * 256;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(row * SIZE),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            self.target.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(SIZE),
                },
            },
            self.target.size(),
        );
        self.queue.submit([encoder.finish()]);
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, |r| r.unwrap());
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        let data = buffer.slice(..).get_mapped_range().unwrap();
        (0..SIZE as usize)
            .flat_map(|y| {
                data[y * row as usize..][..SIZE as usize * 4]
                    .chunks(4)
                    .map(|p| [p[0], p[1], p[2], p[3]])
                    .collect::<Vec<_>>()
            })
            .collect()
    }
}

/// Mean RGB over `pixels` at `indices`.
fn mean(pixels: &[[u8; 4]], indices: &[usize]) -> [f32; 3] {
    let mut sum = [0.0f32; 3];
    for &i in indices {
        for c in 0..3 {
            sum[c] += f32::from(pixels[i][c]);
        }
    }
    sum.map(|s| s / indices.len() as f32 / 255.0)
}

/// `rgb` scaled to sum to one: the hue the debug view's shading leaves alone.
fn chromaticity([r, g, b]: [f32; 3]) -> [f32; 3] {
    let sum = (r + g + b).max(1e-6);
    [r / sum, g / sum, b / sum]
}

#[test]
fn authored_base_colour_reaches_the_grass() {
    let Some(mut meadow) = meadow() else { return };
    let red = FoliageMaterial {
        base_color: [0.8, 0.05, 0.05],
        roughness: 0.9,
        metallic: 0.0,
    };
    let blue = FoliageMaterial {
        base_color: [0.05, 0.05, 0.8],
        ..red
    };
    let red_frame = meadow.frame(Some(red));
    let blue_frame = meadow.frame(Some(blue));
    let procedural = meadow.frame(None);

    // Blades are where the two authored colours disagree; everything else
    // (the empty background) is identical in both frames.
    let blades: Vec<usize> = (0..red_frame.len())
        .filter(|&i| {
            (0..3).any(|c| (i32::from(red_frame[i][c]) - i32::from(blue_frame[i][c])).abs() > 24)
        })
        .collect();
    let coverage = blades.len() as f32 / red_frame.len() as f32;
    assert!(coverage > 0.05, "too few blades to judge: {coverage}");

    // The view darkens blades toward the root and per-blade tint varies each
    // channel around one, so the mean keeps the authored hue.
    for (material, frame) in [(red, &red_frame), (blue, &blue_frame)] {
        let got = chromaticity(mean(frame, &blades));
        let want = chromaticity(material.base_color);
        assert!(
            got.iter()
                .zip(want)
                .all(|(got, want)| (got - want).abs() < 0.05),
            "grass shows {got:?}, authored {want:?}"
        );
    }
    let [r, g, b] = mean(&procedural, &blades);
    assert!(
        g > r && g > b,
        "a type with no authored material must keep the procedural green: {:?}",
        [r, g, b]
    );
}
