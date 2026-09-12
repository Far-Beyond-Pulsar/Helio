use glam::{Mat4, Vec3};
use helio::{
    GpuLight, GpuMaterial, LightType, MaterialId, MeshId, MeshUpload, ObjectDescriptor,
    PackedVertex, Renderer, SceneDbHandle,
};
use pulsar_scenedb::{Entity, World};
use std::sync::Arc;

/// Creates a fresh SceneDB `SceneDb` with a GPU mirror already attached, and
/// returns the `SceneDbHandle` to hand to `RendererBuilder::new` -- SceneDB
/// is the sole scene authority, so no renderer in this workspace can be
/// built without one.
///
/// This demo cohort registers no `#[gpu]`-derived buffer classes up front:
/// `LightComponent` is `#[gpu(layout = packed)]`, which auto-registers its
/// `"scene_lights"` buffer (at `MAX_LIGHTS` capacity) on the first
/// `World::insert` of one — see that constant's doc.
pub fn new_scene_db_with_gpu_mirror(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
) -> pulsar_scenedb::SceneDb {
    let mut scene_db = pulsar_scenedb::SceneDb::new();
    let ctx = pulsar_scenedb::gpu::EngineGpuContext::new(device.clone(), queue.clone());
    let gpu_cfg = pulsar_scenedb::gpu::SceneGpuConfig {
        classes: Vec::new(),
        tombstone_headroom: 0,
        max_cells_metadata: 0,
    };
    let gpu_store = Arc::new(pulsar_scenedb::gpu::SceneGpuStore::new(&ctx, gpu_cfg));
    let mirror = pulsar_scenedb::gpu::GpuMirrorHandle::new(gpu_store, queue.clone());
    scene_db.world.attach_gpu_mirror(mirror);
    scene_db
}

/// The `SceneDbHandle` (`GpuMirrorHandle`) to pass to `RendererBuilder::new`
/// for a `SceneDb` created via [`new_scene_db_with_gpu_mirror`].
pub fn scene_db_handle(scene_db: &pulsar_scenedb::SceneDb) -> SceneDbHandle {
    scene_db
        .world
        .gpu_mirror()
        .cloned()
        .expect("new_scene_db_with_gpu_mirror always attaches a mirror")
}

pub fn make_material(
    base_color: [f32; 4],
    roughness: f32,
    metallic: f32,
    emissive: [f32; 3],
    emissive_strength: f32,
) -> GpuMaterial {
    GpuMaterial {
        base_color,
        emissive: [emissive[0], emissive[1], emissive[2], emissive_strength],
        roughness_metallic: [roughness, metallic, 1.5, 0.5],
        tex_base_color: GpuMaterial::NO_TEXTURE,
        tex_normal: GpuMaterial::NO_TEXTURE,
        tex_roughness: GpuMaterial::NO_TEXTURE,
        tex_emissive: GpuMaterial::NO_TEXTURE,
        tex_occlusion: GpuMaterial::NO_TEXTURE,
        workflow: 0,
        flags: 0,
        material_class: 0,
        class_params: [0.0; 4],
    }
}

pub fn directional_light(direction: [f32; 3], color: [f32; 3], intensity: f32) -> GpuLight {
    GpuLight {
        position_range: [0.0, 0.0, 0.0, f32::MAX],
        direction_outer: [direction[0], direction[1], direction[2], 0.0],
        color_intensity: [color[0], color[1], color[2], intensity],
        shadow_index: 0, // Enable shadows
        light_type: LightType::Directional as u32,
        inner_angle: 0.0,
        _pad: 0,
        ..Default::default()
    }
}

pub fn point_light(position: [f32; 3], color: [f32; 3], intensity: f32, range: f32) -> GpuLight {
    GpuLight {
        position_range: [position[0], position[1], position[2], range],
        direction_outer: [0.0, 0.0, -1.0, 0.0],
        color_intensity: [color[0], color[1], color[2], intensity],
        shadow_index: 0, // Enable shadows
        light_type: LightType::Point as u32,
        inner_angle: 0.0,
        _pad: 0,
        ..Default::default()
    }
}

pub fn spot_light(
    position: [f32; 3],
    direction: [f32; 3],
    color: [f32; 3],
    intensity: f32,
    range: f32,
    inner_angle: f32,
    outer_angle: f32,
) -> GpuLight {
    GpuLight {
        position_range: [position[0], position[1], position[2], range],
        direction_outer: [direction[0], direction[1], direction[2], outer_angle.cos()],
        color_intensity: [color[0], color[1], color[2], intensity],
        shadow_index: 0, // Enable shadows
        light_type: LightType::Spot as u32,
        inner_angle: inner_angle.cos(),
        _pad: 0,
        ..Default::default()
    }
}

// ── SceneDB-owned scene content ─────────────────────────────────────────────
//
// A demo's placed objects and lights are authored as *pass-owned* SceneDB
// components — `helio_pass_gbuffer::StaticObjectComponent` and
// `helio_pass_forward_lit::LightComponent` — not as example-local structs or
// entries in a Helio-owned `Vec`/arena. Each is `#[derive(SceneStore)]` +
// `#[gpu(layout = packed, ...)]`, exactly like the already-shipped
// `helio_pass_corona::CoronaEmitterComponent`: the pass that consumes a kind
// of scene content owns its schema and its generated GPU buffer, and a World
// row is the only authoritative record of "this exists."
//
// Resolving those rows into what the renderer actually draws still goes
// through `Renderer::place_static_object`/`submit_light_frame` each frame —
// not a shortcut, but the same "one dense per-frame resolve, real SceneDB
// authority underneath" shape `helio_component::LightComponent` +
// `HelioRenderer::rebuild_light_frame` already use in production (see
// `LightComponent`'s own doc for why a per-light scheduling problem, shadow-
// slot assignment, keeps this a CPU step rather than a raw GPU buffer bind).

pub use helio_pass_forward_lit::LightComponent;
pub use helio_pass_gbuffer::StaticObjectComponent;

pub fn spawn_object(
    world: &mut World,
    renderer: &mut Renderer,
    mesh: MeshId,
    material: MaterialId,
    transform: Mat4,
    radius: f32,
) -> helio::SceneResult<Entity> {
    spawn_object_with_movability(world, renderer, mesh, material, transform, radius, None)
}

pub fn spawn_object_with_movability(
    world: &mut World,
    renderer: &mut Renderer,
    mesh: MeshId,
    material: MaterialId,
    transform: Mat4,
    radius: f32,
    movability: Option<helio::Movability>,
) -> helio::SceneResult<Entity> {
    let bounds = [
        transform.w_axis.x,
        transform.w_axis.y,
        transform.w_axis.z,
        radius,
    ];
    // `place_static_object` is the render-facing projection, not the scene
    // authority; the projected instance's own `movability`/`groups` still
    // apply to how Helio draws it, they just aren't part of the SceneDB row
    // (which only needs mesh/material/transform/bounds to be re-derivable).
    let object_id = renderer.place_static_object(ObjectDescriptor {
        mesh,
        material,
        transform,
        bounds,
        flags: 0,
        groups: helio::GroupMask::NONE,
        movability,
        user_tag: 0,
    })?;
    // `StaticObjectComponent::new` resolves `mesh`/`material`'s draw
    // parameters via `renderer.mesh_slice`/`material_batch_key` -- both
    // handles were just accepted by `place_static_object` above, so they are
    // live asset-pool entries and this can't fail in practice; still handled
    // explicitly rather than unwrapped since it's a real `Option`.
    let Some(component) = StaticObjectComponent::new(renderer, mesh, material, transform, bounds, 0)
    else {
        return Err(helio::SceneError::InvalidHandle { resource: "mesh_or_material" });
    };
    let entity = world.spawn();
    world.insert(entity, component);
    world.insert(entity, ObjectRenderHandle(object_id));
    Ok(entity)
}

/// The renderer-side presentation handle for a spawned
/// `StaticObjectComponent`. Kept as its own component (not a field of
/// `StaticObjectComponent` itself) so the authoritative row stays plain,
/// re-derivable, `Pod` GPU data with no renderer handle mixed in.
#[derive(Clone, Copy, Debug)]
struct ObjectRenderHandle(helio::ObjectId);

/// Move a previously spawned object: updates both the SceneDB row (the
/// authoritative record) and the renderer's presentation projection.
pub fn update_object_transform(
    world: &mut World,
    renderer: &mut Renderer,
    entity: Entity,
    transform: Mat4,
) -> helio::SceneResult<()> {
    let Some(handle) = world.get::<ObjectRenderHandle>(entity).copied() else {
        return Err(helio::SceneError::InvalidHandle { resource: "object" });
    };
    let Some(mut object) = world.get_mut::<StaticObjectComponent>(entity) else {
        return Err(helio::SceneError::InvalidHandle { resource: "object" });
    };
    let bounds = [
        transform.w_axis.x,
        transform.w_axis.y,
        transform.w_axis.z,
        object.bounds[3],
    ];
    *object = object.with_transform(transform, bounds);
    drop(object);
    renderer.update_static_object_transform(handle.0, transform)
}

/// Despawn a previously spawned object: removes both the SceneDB rows and
/// the renderer's presentation projection.
pub fn despawn_object(
    world: &mut World,
    renderer: &mut Renderer,
    entity: Entity,
) -> helio::SceneResult<()> {
    if let Some(handle) = world.get::<ObjectRenderHandle>(entity).copied() {
        renderer.remove_static_object(handle.0)?;
    }
    world.despawn(entity);
    Ok(())
}

/// Spawn a light. Its `#[gpu]`-mirrored row uploads to the `"scene_lights"`
/// buffer automatically on insert; `helio_pass_forward_lit::ForwardLitPass`
/// resolves that buffer by key every frame on its own -- no renderer method,
/// no per-frame step to call from here at all.
pub fn spawn_light(world: &mut World, light: GpuLight) -> Entity {
    let entity = world.spawn();
    world.insert(entity, LightComponent::from(light));
    entity
}

/// Replace a previously spawned light's parameters in the World. SceneDB's
/// insert-time GPU mirroring uploads the change automatically; no per-frame
/// resubmission step exists to call anymore.
pub fn update_light(world: &mut World, entity: Entity, light: GpuLight) {
    if let Some(mut existing) = world.get_mut::<LightComponent>(entity) {
        *existing = LightComponent::from(light);
    }
}

/// Spawn a corona particle emitter into `slot`.
///
/// `slot` selects which of `helio_pass_corona`'s fixed `MAX_EMITTERS`
/// particle ranges this emitter owns (`0..MAX_EMITTERS`, see that pass's
/// module doc for why the layout is fixed rather than CPU-packed) — the
/// caller is responsible for giving each simultaneously-live emitter its own
/// slot. `emitter.particle_count`/`particle_offset` are overwritten to fit
/// that slot; `spawn_cursor` is left at whatever `emitter` carries (normally
/// `0` for a new emitter) since the pass owns advancing it from here via its
/// own transient `spawn_cursor_buf`, never through this SceneDB row again.
pub fn spawn_corona_emitter(
    world: &mut World,
    slot: u32,
    emitter: libhelio::GpuCoronaEmitter,
) -> Entity {
    let entity = world.spawn();
    world.insert(entity, corona_component_for_slot(slot, emitter));
    entity
}

/// Update a previously spawned corona emitter's authored parameters (e.g. a
/// moving emitter's transform). Never touches `spawn_cursor` in the GPU's
/// own transient buffer -- see `spawn_corona_emitter`'s doc.
pub fn update_corona_emitter(
    world: &mut World,
    entity: Entity,
    slot: u32,
    emitter: libhelio::GpuCoronaEmitter,
) {
    if let Some(mut existing) = world.get_mut::<helio_pass_corona::CoronaEmitterComponent>(entity)
    {
        *existing = corona_component_for_slot(slot, emitter);
    }
}

fn corona_component_for_slot(
    slot: u32,
    mut emitter: libhelio::GpuCoronaEmitter,
) -> helio_pass_corona::CoronaEmitterComponent {
    emitter.particle_count = emitter.particle_count.min(libhelio::CORONA_MAX_PARTICLES_PER_EMITTER);
    emitter.particle_offset = slot * libhelio::CORONA_MAX_PARTICLES_PER_EMITTER;
    helio_pass_corona::CoronaEmitterComponent::from(emitter)
}

/// Builds a cube mesh centred at `center` (mesh-local origin offset) with the
/// given `half_extent`.  The conventional usage is `[0.0, 0.0, 0.0]` with the
/// world position supplied via the transform passed to `insert_object`; the
/// `center` parameter exists for backwards compatibility only.
pub fn cube_mesh(center: [f32; 3], half_extent: f32) -> MeshUpload {
    box_mesh(center, [half_extent, half_extent, half_extent])
}

/// Builds a box mesh centred at `center` (mesh-local origin offset) with the
/// given `half_extents`.  The conventional usage is `[0.0, 0.0, 0.0]` with the
/// world position supplied via the transform passed to `insert_object`; the
/// `center` parameter exists for backwards compatibility only.
pub fn box_mesh(center: [f32; 3], half_extents: [f32; 3]) -> MeshUpload {
    let c = Vec3::from_array(center);
    let e = Vec3::from_array(half_extents);
    let corners = [
        c + Vec3::new(-e.x, -e.y, e.z),
        c + Vec3::new(e.x, -e.y, e.z),
        c + Vec3::new(e.x, e.y, e.z),
        c + Vec3::new(-e.x, e.y, e.z),
        c + Vec3::new(-e.x, -e.y, -e.z),
        c + Vec3::new(e.x, -e.y, -e.z),
        c + Vec3::new(e.x, e.y, -e.z),
        c + Vec3::new(-e.x, e.y, -e.z),
    ];
    let faces = [
        ([0, 1, 2, 3], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
        ([5, 4, 7, 6], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]),
        ([4, 0, 3, 7], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        ([1, 5, 6, 2], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
        ([3, 2, 6, 7], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),
        ([4, 5, 1, 0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]),
    ];
    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    for (face_index, (quad, normal, tangent)) in faces.iter().enumerate() {
        let base = (face_index * 4) as u32;
        let uv = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];
        for (i, corner_index) in quad.iter().enumerate() {
            vertices.push(PackedVertex::from_components(
                corners[*corner_index].to_array(),
                *normal,
                uv[i],
                *tangent,
                1.0,
            ));
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    MeshUpload { vertices, indices }
}

pub fn plane_mesh(center: [f32; 3], half_extent: f32) -> MeshUpload {
    let c = Vec3::from_array(center);
    let e = half_extent;
    let normal = [0.0, 1.0, 0.0];
    let tangent = [1.0, 0.0, 0.0];
    let positions = [
        c + Vec3::new(-e, 0.0, -e),
        c + Vec3::new(e, 0.0, -e),
        c + Vec3::new(e, 0.0, e),
        c + Vec3::new(-e, 0.0, e),
    ];
    let uvs = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let vertices = positions
        .into_iter()
        .zip(uvs)
        .map(|(position, uv)| {
            PackedVertex::from_components(position.to_array(), normal, uv, tangent, 1.0)
        })
        .collect();
    // Reverse triangle winding so the top-facing plane is front-facing (visible from above).
    let indices = vec![0, 2, 1, 0, 3, 2];
    MeshUpload { vertices, indices }
}

pub fn sphere_mesh(center: [f32; 3], radius: f32) -> MeshUpload {
    let center = Vec3::from_array(center);
    let lat_steps = 16;
    let lon_steps = 32;
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for i in 0..=lat_steps {
        let phi = std::f32::consts::PI * (i as f32 / lat_steps as f32);
        let y = phi.cos();
        let sin_phi = phi.sin();
        for j in 0..=lon_steps {
            let theta = 2.0 * std::f32::consts::PI * (j as f32 / lon_steps as f32);
            let x = sin_phi * theta.cos();
            let z = sin_phi * theta.sin();

            let position = center + Vec3::new(x, y, z) * radius;
            let normal = [x, y, z];
            let uv = [j as f32 / lon_steps as f32, i as f32 / lat_steps as f32];
            let tangent_vec = Vec3::new(-z, 0.0, x).normalize_or_zero();
            let tangent = tangent_vec.to_array();
            vertices.push(PackedVertex::from_components(
                position.to_array(),
                normal,
                uv,
                tangent,
                1.0,
            ));
        }
    }

    for i in 0..lat_steps {
        for j in 0..lon_steps {
            let a = (i * (lon_steps + 1) + j) as u32;
            let b = a + (lon_steps + 1) as u32;
            // CCW winding when viewed from outside (outward normals).
            indices.extend_from_slice(&[a, a + 1, b]);
            indices.extend_from_slice(&[b, a + 1, b + 1]);
        }
    }

    MeshUpload { vertices, indices }
}

/// Move/relight a previously spawned point light. Superseded
/// `update_point_light(renderer, LightId, ...)` (Helio's own light-arena
/// handle) — lights are SceneDB `World` rows now, keyed by `Entity`, and
/// [`rebuild_lights`] resubmits them wholesale each frame, so there is no
/// per-light renderer handle to update in place.
pub fn update_point_light(
    world: &mut World,
    entity: Entity,
    position: Vec3,
    color: [f32; 3],
    intensity: f32,
    range: f32,
) {
    update_light(
        world,
        entity,
        point_light(position.to_array(), color, intensity, range),
    );
}
