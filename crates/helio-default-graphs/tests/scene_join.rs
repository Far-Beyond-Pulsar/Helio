//! The scene join on the GPU: handcrafted input rows in, object/material/
//! light/billboard rows out, compared against the CPU math the join
//! replaces (`StaticObjectComponent::new` for meshes, the owner's rotation
//! of -Y for lights). Needs a GPU adapter (lavapipe works); skips without.

use std::sync::Arc;

use helio_default_graphs::scene_join::{
    BufferHandle, BufferKey, SceneBufferProjection, SceneDerivation, SceneDerivationContext,
    SceneDerivationOutput,
};
use helio_default_graphs::scene_join::{SceneJoin, SceneJoinKeys};

fn device() -> Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
    let limits = helio::required_wgpu_limits(adapter.limits());
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: limits,
        ..Default::default()
    }))
    .ok()?;
    Some((Arc::new(device), Arc::new(queue)))
}

const KEYS: SceneJoinKeys = SceneJoinKeys {
    owners: BufferKey::of("t_owners"),
    generations: BufferKey::of("t_generations"),
    hidden: BufferKey::of("t_hidden"),
    transforms: BufferKey::of("t_transforms"),
    vertex_handles: BufferKey::of("t_vertex_handles"),
    index_handles: BufferKey::of("t_index_handles"),
    mesh_bounds: BufferKey::of("t_bounds"),
    mesh_flags: BufferKey::of("t_flags"),
    section_handles: BufferKey::of("t_section_handles"),
    mesh_sections: BufferKey::of("t_sections"),
    light_sources: BufferKey::of("t_lights"),
};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Owner {
    index: u32,
    generation: u32,
    enabled: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Transform {
    position: [f32; 3],
    rotation: [f32; 3],
    scale: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Section {
    first_index: u32,
    index_count: u32,
    material_class: u32,
    graph_hash_lo: u32,
    graph_hash_hi: u32,
    pad: [u32; 3],
    material: helio::GpuMaterial,
}

struct Scene {
    buffers: Vec<(BufferKey, Vec<u8>, u64)>,
}

impl Scene {
    fn put<T: bytemuck::Pod>(&mut self, key: BufferKey, rows: &[T]) {
        self.buffers.retain(|(k, ..)| *k != key);
        self.buffers.push((
            key,
            bytemuck::cast_slice(rows).to_vec(),
            std::mem::size_of::<T>() as u64,
        ));
    }

    fn projection(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        generation: u64,
    ) -> SceneBufferProjection {
        SceneBufferProjection::from_handles(self.buffers.iter().map(|(key, bytes, row_bytes)| {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (bytes.len() as u64).max(16),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buffer, 0, bytes);
            (
                *key,
                BufferHandle {
                    buffer,
                    epoch: 0,
                    row_bytes: *row_bytes,
                    content_generation: generation,
                },
            )
        }))
    }
}

fn read<T: bytemuck::Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
) -> Vec<T> {
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

/// Run the join once over `scene`; returns the published buffers.
fn run(
    join: &mut SceneJoin,
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    scene: &Scene,
    generation: u64,
) -> SceneBufferProjection {
    let mut projection = scene.projection(device, queue, generation);
    let mut encoder = device.create_command_encoder(&Default::default());
    let output = join.derive(
        &SceneDerivationContext {
            device,
            queue,
            inputs: &projection,
        },
        &mut encoder,
    );
    queue.submit([encoder.finish()]);
    for (key, handle) in output.buffers {
        projection.insert(key, handle);
    }
    projection
}

fn material(base: f32) -> helio::GpuMaterial {
    helio::GpuMaterial {
        base_color: [base, 0.5, 0.25, 1.0],
        emissive: [0.0; 4],
        roughness_metallic: [0.5, 0.0, 1.5, 0.5],
        tex_base_color: u32::MAX,
        tex_normal: u32::MAX,
        tex_roughness: u32::MAX,
        tex_emissive: u32::MAX,
        tex_occlusion: u32::MAX,
        workflow: 0,
        flags: 0,
        material_class: 0,
        class_params: [0.0; 4],
    }
}

fn model_of(t: &Transform) -> glam::Mat4 {
    glam::Mat4::from_scale_rotation_translation(
        glam::Vec3::from_array(t.scale),
        glam::Quat::from_euler(
            glam::EulerRot::YXZ,
            t.rotation[1].to_radians(),
            t.rotation[0].to_radians(),
            t.rotation[2].to_radians(),
        ),
        glam::Vec3::from_array(t.position),
    )
}

fn close(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-4)
}

/// Entities: 0 = object (transform, generation 3), 1 = mesh instance of 0
/// with two sections, 2 = light instance of 0, 3 = a disabled mesh
/// instance of 0.
fn base_scene() -> Scene {
    let mut scene = Scene {
        buffers: Vec::new(),
    };
    let transform = Transform {
        position: [1.0, 2.0, 3.0],
        rotation: [30.0, 45.0, 10.0],
        scale: [2.0, 1.0, 0.5],
    };
    scene.put(
        KEYS.transforms,
        &[
            transform,
            Transform::zeroed_row(),
            Transform::zeroed_row(),
            Transform::zeroed_row(),
        ],
    );
    scene.put(KEYS.generations, &[3u32, 5, 7, 9]);
    scene.put(KEYS.hidden, &[0u32, 0, 0, 0]);
    let owner = |enabled| Owner {
        index: 0,
        generation: 3,
        enabled,
    };
    scene.put(
        KEYS.owners,
        &[
            Owner {
                index: 0,
                generation: 0,
                enabled: 0,
            },
            owner(1),
            owner(1),
            owner(0),
        ],
    );
    // Instances 1 and 3: vertices at 100.., indices at 200.., sections at
    // pool slots 0..2 and 2..3.
    scene.put(
        KEYS.vertex_handles,
        &[[0u32, 0], [100, 24], [0, 0], [100, 24]],
    );
    scene.put(
        KEYS.index_handles,
        &[[0u32, 0], [200, 36], [0, 0], [200, 36]],
    );
    scene.put(
        KEYS.mesh_bounds,
        &[
            [0.0f32; 4],
            [0.5, 0.0, 0.0, 1.5],
            [0.0; 4],
            [0.0, 0.0, 0.0, 1.0],
        ],
    );
    scene.put(KEYS.mesh_flags, &[0u32, helio::INSTANCE_FLAG_MOVABLE, 0, 0]);
    scene.put(KEYS.section_handles, &[[0u32, 0], [0, 2], [0, 0], [2, 1]]);
    let section = |first_index, index_count, base| Section {
        first_index,
        index_count,
        material_class: 0,
        graph_hash_lo: 0,
        graph_hash_hi: 0,
        pad: [0; 3],
        material: material(base),
    };
    scene.put(
        KEYS.mesh_sections,
        &[
            section(0, 12, 0.1),
            section(12, 24, 0.2),
            section(0, 36, 0.3),
        ],
    );
    let mut light = helio::GpuLight {
        position_range: [0.0, 0.0, 0.0, 10.0],
        direction_outer: [0.0, -1.0, 0.0, 0.5],
        color_intensity: [1.0, 0.5, 0.25, 100.0],
        ..Default::default()
    };
    light._pad = 1;
    scene.put(
        KEYS.light_sources,
        &[
            helio::GpuLight::default(),
            helio::GpuLight::default(),
            light,
            helio::GpuLight::default(),
        ],
    );
    scene
}

trait ZeroRow {
    fn zeroed_row() -> Self;
}

impl ZeroRow for Transform {
    fn zeroed_row() -> Self {
        bytemuck::Zeroable::zeroed()
    }
}

#[test]
fn placed_instances_are_joined_with_their_owner_and_nothing_else_is() {
    let Some((device, queue)) = device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut join = SceneJoin::new(&device, KEYS, true);
    let scene = base_scene();
    let out = run(&mut join, &device, &queue, &scene, 1);

    let objects: Vec<helio_pass_gbuffer::StaticObjectComponent> = read(
        &device,
        &queue,
        &out.get(BufferKey::of("static_objects")).unwrap().buffer,
    );
    let materials: Vec<helio::GpuMaterial> = read(
        &device,
        &queue,
        &out.get(BufferKey::of("materials")).unwrap().buffer,
    );
    let transform = Transform {
        position: [1.0, 2.0, 3.0],
        rotation: [30.0, 45.0, 10.0],
        scale: [2.0, 1.0, 0.5],
    };
    let model = model_of(&transform);
    let center = model.transform_point3(glam::Vec3::new(0.5, 0.0, 0.0));
    for (slot, (first, count, base)) in [(0u32, (0u32, 12u32, 0.1f32)), (1, (12, 24, 0.2))] {
        let expected = helio_pass_gbuffer::StaticObjectComponent::new(
            1,
            5 + 1,
            slot,
            5 + 1,
            model,
            [center.x, center.y, center.z, 1.5 * 2.0],
            count,
            200 + first,
            100,
            0,
            0,
            helio::INSTANCE_FLAG_MOVABLE,
        );
        let got = objects[slot as usize];
        assert_eq!(
            (
                got.mesh_slot,
                got.mesh_generation,
                got.material_slot,
                got.material_generation
            ),
            (1, 6, slot, 6),
            "section {slot} identity"
        );
        assert_eq!(
            (
                got.index_count,
                got.first_index,
                got.vertex_offset,
                got.flags
            ),
            (
                expected.index_count,
                expected.first_index,
                expected.vertex_offset,
                expected.flags
            ),
            "section {slot} draw range"
        );
        assert!(
            close(
                bytemuck::cast_slice(&got.transform),
                bytemuck::cast_slice(&expected.transform)
            ),
            "model matrix"
        );
        assert!(
            close(
                bytemuck::cast_slice(&got.normal_mat),
                bytemuck::cast_slice(&expected.normal_mat)
            ),
            "normal matrix"
        );
        assert!(
            close(&got.bounds, &expected.bounds),
            "bounds {:?} vs {:?}",
            got.bounds,
            expected.bounds
        );
        assert_eq!(
            materials[slot as usize].base_color[0], base,
            "section {slot} material"
        );
    }
    assert_eq!(
        objects[2].mesh_generation, 0,
        "a disabled instance draws nothing"
    );

    let lights: Vec<helio::GpuLight> = read(
        &device,
        &queue,
        &out.get(BufferKey::of("scene_lights")).unwrap().buffer,
    );
    let light = lights[2];
    assert!(close(&light.position_range, &[1.0, 2.0, 3.0, 10.0]));
    let direction = glam::Quat::from_euler(
        glam::EulerRot::YXZ,
        45f32.to_radians(),
        30f32.to_radians(),
        10f32.to_radians(),
    ) * -glam::Vec3::Y;
    assert!(
        close(&light.direction_outer[..3], &direction.to_array()),
        "light direction"
    );
    assert_eq!(
        light._pad, 0,
        "the enabled flag does not leak into the output"
    );
    assert_eq!(
        lights[1].color_intensity, [0.0; 4],
        "rows without a light stay dark"
    );
    let billboards: Vec<[f32; 12]> = read(
        &device,
        &queue,
        &out.get(BufferKey::of("billboard_instances"))
            .unwrap()
            .buffer,
    );
    assert!(
        close(&billboards[2][..3], &[1.0, 2.0, 3.0]),
        "billboard at the light"
    );
}

#[test]
fn hiding_the_owner_a_stale_generation_or_a_disabled_light_removes_the_rows() {
    let Some((device, queue)) = device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut join = SceneJoin::new(&device, KEYS, false);
    let mut scene = base_scene();
    scene.put(KEYS.hidden, &[1u32, 0, 0, 0]);
    let out = run(&mut join, &device, &queue, &scene, 1);
    let objects: Vec<helio_pass_gbuffer::StaticObjectComponent> = read(
        &device,
        &queue,
        &out.get(BufferKey::of("static_objects")).unwrap().buffer,
    );
    assert!(
        objects.iter().all(|o| o.mesh_generation == 0),
        "hidden owner"
    );
    let lights: Vec<helio::GpuLight> = read(
        &device,
        &queue,
        &out.get(BufferKey::of("scene_lights")).unwrap().buffer,
    );
    assert!(
        lights.iter().all(|l| l.color_intensity[3] == 0.0),
        "hidden owner"
    );
    assert!(
        out.get(BufferKey::of("billboard_instances")).is_none(),
        "billboards off"
    );

    let mut scene = base_scene();
    scene.put(KEYS.generations, &[4u32, 5, 7, 9]);
    let out = run(&mut join, &device, &queue, &scene, 2);
    let objects: Vec<helio_pass_gbuffer::StaticObjectComponent> = read(
        &device,
        &queue,
        &out.get(BufferKey::of("static_objects")).unwrap().buffer,
    );
    assert!(
        objects.iter().all(|o| o.mesh_generation == 0),
        "stale owner generation"
    );

    let mut scene = base_scene();
    let mut light = helio::GpuLight {
        color_intensity: [1.0; 4],
        ..Default::default()
    };
    light._pad = 0;
    scene.put(KEYS.light_sources, &[light, light, light, light]);
    let out = run(&mut join, &device, &queue, &scene, 3);
    let lights: Vec<helio::GpuLight> = read(
        &device,
        &queue,
        &out.get(BufferKey::of("scene_lights")).unwrap().buffer,
    );
    assert!(
        lights.iter().all(|l| l.color_intensity[3] == 0.0),
        "disabled light"
    );
}

#[test]
fn unchanged_inputs_record_nothing_and_keep_the_outputs() {
    let Some((device, queue)) = device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut join = SceneJoin::new(&device, KEYS, false);
    let scene = base_scene();
    let projection = scene.projection(&device, &queue, 7);
    let mut derive = || {
        let mut encoder = device.create_command_encoder(&Default::default());
        let output = join.derive(
            &SceneDerivationContext {
                device: &device,
                queue: &queue,
                inputs: &projection,
            },
            &mut encoder,
        );
        queue.submit([encoder.finish()]);
        output
    };
    let first = derive();
    assert!(first.recorded);
    let second = derive();
    assert!(!second.recorded, "same inputs: nothing to record");
    let generation = |output: &SceneDerivationOutput| {
        output
            .buffers
            .iter()
            .find(|(key, _)| *key == BufferKey::of("static_objects"))
            .map(|(_, h)| (h.epoch, h.content_generation))
    };
    assert_eq!(generation(&first), generation(&second), "outputs unchanged");
}
