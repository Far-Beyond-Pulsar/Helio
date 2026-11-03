//! Kinds' row joins in a camera-relative frame: the tables their passes
//! draw in the frame's space (water volumes, foliage layers) are placed
//! relative to the world origin, water rows carry the origin's world phase, and a moved
//! origin re-derives them. Needs a GPU adapter (lavapipe works); skips
//! without.

use std::sync::Arc;

use helio_pass_foliage_place::scene_rows::{FOLIAGE_LAYERS_KEY, FOLIAGE_SOURCE_ROW_BYTES};
use helio_pass_water_sim::scene_rows::{WATER_VOLUMES_KEY, WATER_VOLUME_SOURCE_ROW_BYTES};
use helio_scene_join::{
    BufferHandle, BufferKey, RowJoin, RowJoinKeys, SceneBufferProjection, SceneDerivation,
    SceneDerivationContext,
};

fn device() -> Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
    let instance = wgpu::Instance::new(helio_core::wgpu_instance_descriptor());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
    let limits = helio::required_wgpu_limits(adapter.limits());
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: limits,
        ..Default::default()
    }))
    .ok()?;
    Some((Arc::new(device), Arc::new(queue)))
}

const KEYS: RowJoinKeys = RowJoinKeys {
    owners: BufferKey::of("t_owners"),
    generations: BufferKey::of("t_generations"),
    hidden: BufferKey::of("t_hidden"),
    transforms: BufferKey::of("t_transforms"),
    mesh_bounds: BufferKey::of("t_mesh_bounds"),
    directional_light: BufferKey::of("directional_light"),
};
const WATER_VOLUMES: BufferKey = BufferKey::of("t_water_volumes");
const FOLIAGE: BufferKey = BufferKey::of("t_foliage");

/// The row joins of the kinds under test: water, foliage.
fn joins(device: &wgpu::Device) -> Vec<RowJoin> {
    vec![
        helio_pass_water_sim::scene_rows::scene_rows(
            device,
            KEYS,
            helio_pass_water_sim::WaterRowSources {
                water_volumes: WATER_VOLUMES,
                wind: BufferKey::of("t_wind"),
                water_hitboxes: BufferKey::of("t_water_hitboxes"),
            },
        ),
        helio_pass_foliage_place::scene_rows::scene_rows(
            device,
            KEYS,
            helio_pass_foliage_place::FoliageRowSources {
                foliage: FOLIAGE,
                wind: BufferKey::of("t_wind"),
            },
        ),
    ]
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Owner {
    index: u32,
    generation: u32,
    enabled: u32,
}

/// The frontend's `TransformRow`: the f64 position as an f32 pair.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Transform {
    position_hi: [f32; 3],
    position_lo: [f32; 3],
    rotation: [f32; 3],
    scale: [f32; 3],
}

impl Transform {
    /// At `position`, split as the frontend splits it: the nearest f32,
    /// then what that drops.
    fn at(position: [f64; 3], rotation: [f32; 3], scale: [f32; 3]) -> Self {
        let p = glam::DVec3::from_array(position);
        let hi = p.as_vec3();
        let lo = (p - hi.as_dvec3()).as_vec3();
        Self {
            position_hi: hi.to_array(),
            position_lo: lo.to_array(),
            rotation,
            scale,
        }
    }
}

const WATER_WORDS: usize = (WATER_VOLUME_SOURCE_ROW_BYTES / 4) as usize;
const FOLIAGE_WORDS: usize = (FOLIAGE_SOURCE_ROW_BYTES / 4) as usize;
/// Where a foliage source row's layer starts (after its 32-word type).
const FOLIAGE_LAYER_WORD: usize = 32;

/// One owner far from the world origin (6_371_002.5 is an f32, with 0.5 m
/// steps up there) holding a water volume (instance 1) of 4 x 2 x 6 m and a
/// foliage layer (instance 1) of half extent 8 m from 6_371_000 m to
/// 6_371_010 m up.
fn scene(device: &wgpu::Device, queue: &wgpu::Queue, far: Transform) -> SceneBufferProjection {
    let owner = Owner {
        index: 0,
        generation: 3,
        enabled: 1,
    };
    let none = Owner {
        index: 0,
        generation: 0,
        enabled: 0,
    };
    let mut water = vec![0u32; WATER_WORDS * 3];
    let mut foliage = vec![0u32; FOLIAGE_WORDS * 3];
    // A density (the join's gate), then the layer.
    foliage[FOLIAGE_WORDS] = 4.0f32.to_bits();
    for (word, value) in [8.0f32, 0.0, 6_371_000.0, 6_371_010.0]
        .into_iter()
        .enumerate()
    {
        foliage[FOLIAGE_WORDS + FOLIAGE_LAYER_WORD + word] = value.to_bits();
    }
    for (word, value) in [4.0f32, 2.0, 6.0, 0.5].into_iter().enumerate() {
        water[WATER_WORDS + word] = value.to_bits();
    }
    let buffers: Vec<(BufferKey, Vec<u8>, u64)> = vec![
        (
            KEYS.owners,
            bytemuck::cast_slice(&[none, owner, owner]).to_vec(),
            12,
        ),
        (KEYS.generations, bytemuck::cast_slice(&[3u32]).to_vec(), 4),
        (KEYS.hidden, bytemuck::cast_slice(&[0u32]).to_vec(), 4),
        (
            KEYS.transforms,
            bytemuck::bytes_of(&far).to_vec(),
            std::mem::size_of::<Transform>() as u64,
        ),
        (
            WATER_VOLUMES,
            bytemuck::cast_slice(&water).to_vec(),
            WATER_VOLUME_SOURCE_ROW_BYTES,
        ),
        (
            FOLIAGE,
            bytemuck::cast_slice(&foliage).to_vec(),
            FOLIAGE_SOURCE_ROW_BYTES,
        ),
    ];
    SceneBufferProjection::from_handles(buffers.into_iter().map(|(key, bytes, row_bytes)| {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (bytes.len() as u64).max(16),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer, 0, &bytes);
        (
            key,
            BufferHandle {
                buffer,
                epoch: 0,
                row_bytes,
                content_generation: 1,
            },
        )
    }))
}

fn read(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) -> Vec<f32> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: buffer.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
    queue.submit([encoder.finish()]);
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, |r| r.unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let out = bytemuck::cast_slice(&staging.slice(..).get_mapped_range().unwrap()).to_vec();
    staging.unmap();
    out
}

/// The first water row and foliage layer after a join at `origin`.
fn derive(
    joins: &mut [RowJoin],
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    inputs: &SceneBufferProjection,
    origin: glam::DVec3,
) -> (Vec<f32>, Vec<f32>) {
    let mut encoder = device.create_command_encoder(&Default::default());
    let mut projection = SceneBufferProjection::default();
    for join in joins.iter_mut() {
        let output = join.derive(
            &SceneDerivationContext {
                device,
                queue,
                inputs,
                world_origin: origin,
            },
            &mut encoder,
        );
        for (key, handle) in output.buffers {
            projection.insert(key, handle);
        }
    }
    queue.submit([encoder.finish()]);
    let water = read(
        device,
        queue,
        &projection.get(WATER_VOLUMES_KEY).unwrap().buffer,
    );
    let layers = read(
        device,
        queue,
        &projection.get(FOLIAGE_LAYERS_KEY).unwrap().buffer,
    );
    (water[..64].to_vec(), layers[..8].to_vec())
}

#[test]
fn camera_relative_frames_place_water_and_foliage_layers_relative_to_the_world_origin() {
    let Some((device, queue)) = device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let far_position = [1000.0, 6_371_002.371, -300.0];
    let far = Transform::at(far_position, [0.0; 3], [1.0; 3]);
    let inputs = scene(&device, &queue, far);
    let mut join = joins(&device);

    // The frame's origin at the world's: rows at the owner, no water phase.
    let (water, layer) = derive(&mut join[..], &device, &queue, &inputs, glam::DVec3::ZERO);
    assert_eq!(
        &layer,
        &[
            992.0,
            6_371_000.0,
            -308.0,
            0.0,
            1008.0,
            6_371_010.0,
            -292.0,
            0.0
        ],
        "the layer around its owner, at its world altitudes"
    );
    assert_eq!(&water[0..3], &[998.0, 6_371_001.5, -303.0], "water min");
    assert_eq!(&water[4..8], &[1002.0, 6_371_003.5, -297.0, 6_371_003.0]);
    assert_eq!(
        &water[60..62],
        &[0.0, 0.0],
        "water at the world's origin has no phase"
    );

    // An origin near the owner: the same rows, less the f64 origin, exact for
    // these small offsets where world positions are 0.5 m out.
    let origin = glam::DVec3::new(1000.25, 6_371_000.125, -290.0);
    let (water, layer) = derive(&mut join[..], &device, &queue, &inputs, origin);
    let offset = (glam::DVec3::from_array(far_position) - origin).as_vec3();
    assert!(
        (offset - glam::Vec3::new(-0.25, 2.246, -10.0))
            .abs()
            .max_element()
            < 1e-6,
        "{offset}"
    );
    let near =
        |a: &[f32], b: glam::Vec3| (glam::Vec3::from_slice(a) - b).abs().max_element() < 1e-5;
    assert!(
        near(&water[0..3], offset - glam::Vec3::new(2.0, 1.0, 3.0)),
        "{:?}",
        &water[0..3]
    );
    assert!(
        near(&water[4..7], offset + glam::Vec3::new(2.0, 1.0, 3.0)),
        "{:?}",
        &water[4..7]
    );
    assert!((water[7] - (offset.y + 0.5)).abs() < 1e-5, "surface height");
    // The waves tile the world every 270 m: 1000.25 and -290 into it.
    assert_eq!(&water[60..62], &[190.25, 250.0], "world phase");
    // The layer where placement tests its points: the frame's space, its
    // square around the owner exactly; its authored f32 altitudes keep
    // their own 0.5 m step up there.
    assert!(
        near(
            &[layer[0], 0.0, layer[2]],
            glam::Vec3::new(offset.x - 8.0, 0.0, offset.z - 8.0)
        ),
        "{:?}",
        &layer[0..3]
    );
    assert!(
        near(
            &[layer[4], 0.0, layer[6]],
            glam::Vec3::new(offset.x + 8.0, 0.0, offset.z + 8.0)
        ),
        "{:?}",
        &layer[4..7]
    );
    assert!(
        (layer[1] + 0.125).abs() <= 0.5 && (layer[5] - 9.875).abs() <= 0.5,
        "altitudes {} {}",
        layer[1],
        layer[5]
    );

    // A moved origin re-derives them, though no input row changed.
    let moved = origin + glam::DVec3::new(1024.0, 0.0, 0.0);
    let (water, layer) = derive(&mut join[..], &device, &queue, &inputs, moved);
    assert_eq!(layer[0], offset.x - 1024.0 - 8.0);
    assert_eq!(water[0], offset.x - 1024.0 - 2.0);
    assert_eq!(water[60], 134.25, "(1000.25 + 1024) mod 270");
}
