use std::path::Path;
use std::sync::Arc;

use glam::Vec3;
use helio::{
    Camera, GpuLight, GpuMaterial, LightType, Renderer, RendererBuilder, RendererConfig,
};
use helio_asset_compat::{load_scene_file_with_config, upload_scene_materials, LoadConfig};
use helio_pass_forward_lit::LightComponent;
use helio_pass_gbuffer::{MaterialComponent, MeshComponent, StaticObjectComponent};
use pulsar_scenedb::{Entity, SceneDb, World};
use pulsar_scenedb::gpu::{EngineGpuContext, GpuMirrorHandle, SceneGpuConfig, SceneGpuStore};
use thiserror::Error;

// ── Public types ──────────────────────────────────────────────────────────────

/// Which direction the camera looks at the model from.
#[derive(Debug, Clone, Copy, Default)]
pub enum ViewDirection {
    /// Slightly above and in front, rotated 45° — good general-purpose preview.
    #[default]
    Isometric,
    /// Straight from +Z (looking toward -Z).
    Front,
    /// Straight from -Z (looking toward +Z).
    Back,
    /// Straight from +X (looking toward -X).
    Right,
    /// Straight from -X (looking toward +X).
    Left,
    /// Straight from above +Y (looking toward -Y).
    Top,
    /// Straight from below -Y (looking toward +Y).
    Bottom,
}

/// Configuration for the snapshot.
pub struct SnapshotConfig {
    pub width: u32,
    pub height: u32,
    pub view: ViewDirection,
    /// Extra margin around the model (1.0 = exact fit, 1.2 = 20% breathing room).
    pub fit_margin: f32,
    /// Vertical field-of-view in degrees.
    pub fov_degrees: f32,
    /// Whether to flip the UV Y-axis when loading the model.
    pub flip_uv_y: bool,
}

impl Default for SnapshotConfig {
    fn default() -> Self {
        Self {
            width: 1024,
            height: 1024,
            view: ViewDirection::Isometric,
            fit_margin: 1.2,
            fov_degrees: 45.0,
            flip_uv_y: false,
        }
    }
}

#[derive(Debug, Error)]
pub enum SnapshotError {
    #[error("asset loading failed: {0}")]
    Asset(#[from] helio_asset_compat::AssetError),

    #[error("no geometry found in model")]
    EmptyModel,

    #[error("wgpu adapter not found — no GPU available for headless rendering")]
    NoAdapter,

    #[error("wgpu device error: {0}")]
    Device(#[from] wgpu::RequestDeviceError),

    #[error("render error: {0}")]
    Render(String),

    #[error("readback buffer mapping failed: {0}")]
    Readback(#[from] wgpu::BufferAsyncError),
}

// ── Entry point ───────────────────────────────────────────────────────────────

/// Load `model_path`, render one snapshot frame, and return an RGBA image.
///
/// The camera is placed automatically so the whole model fits in frame.
/// No window or event loop is required.
pub fn render_snapshot<P: AsRef<Path>>(
    model_path: P,
    config: SnapshotConfig,
) -> Result<image::RgbaImage, SnapshotError> {
    pollster::block_on(render_snapshot_async(model_path, config))
}

// ── Internals ─────────────────────────────────────────────────────────────────

async fn render_snapshot_async<P: AsRef<Path>>(
    model_path: P,
    cfg: SnapshotConfig,
) -> Result<image::RgbaImage, SnapshotError> {
    // ── 1. Load model ─────────────────────────────────────────────────────────
    let load_cfg = LoadConfig::default()
        .with_uv_flip(cfg.flip_uv_y)
        .with_merge_meshes(false);

    let scene = load_scene_file_with_config(model_path, load_cfg)?;

    // ── 2. Compute AABB over all mesh vertices ────────────────────────────────
    let (aabb_min, aabb_max) = compute_aabb(&scene)?;
    let center = (aabb_min + aabb_max) * 0.5;
    let half_extents = (aabb_max - aabb_min) * 0.5;
    let radius = half_extents.length().max(0.01);

    // ── 3. Initialise headless GPU ────────────────────────────────────────────
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        backend_options: wgpu::BackendOptions::default(),
        display: None,
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        })
        .await
        .map_err(|_| SnapshotError::NoAdapter)?;

    let (device, queue): (wgpu::Device, wgpu::Queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("helio-snapshot"),
            required_features: helio::required_wgpu_features(adapter.features()),
            required_limits: helio::required_wgpu_limits(adapter.limits()),
            ..Default::default()
        })
        .await?;

    let device = Arc::new(device);
    let queue = Arc::new(queue);

    // ── 4. Create offscreen render target ─────────────────────────────────────
    const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

    let target_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("snapshot-target"),
        size: wgpu::Extent3d {
            width: cfg.width,
            height: cfg.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target_texture.create_view(&wgpu::TextureViewDescriptor::default());

    // ── 5. Build Helio renderer ───────────────────────────────────────────────
    // Use new_with_external_device so the graph uses deferred (non-blocking)
    // GPU timestamp readback — we drive polling ourselves after the frame.
    let renderer_cfg = RendererConfig::new(cfg.width, cfg.height, FORMAT).with_render_scale(1.0);
    let mut scene_db = new_scene_db(&device, &queue);
    let mut renderer = RendererBuilder::new(renderer_cfg, scene_db_handle(&scene_db))
        .with_external_device()
        .with_pass_build_context(Box::new(helio_default_graphs::build_default_graph_external_with_context))
        .build(device.clone(), queue.clone(), cfg.width, cfg.height, FORMAT);

    // ── 6. Upload all meshes + materials via helio-asset-compat ──────────────
    let (mesh_ids, material_ids) = upload_scene_rows(&mut scene_db.world, &scene);

    // ── 7. Insert a fallback material for meshes with no material ─────────────
    let fallback_mat = {
        let entity = scene_db.world.spawn();
        scene_db.world.insert(entity, fallback_material());
        entity
    };

    // ── 8. Place a renderable object for each uploaded mesh ───────────────────
    for (i, mesh) in scene.meshes.iter().enumerate() {
        let mesh_id = match mesh_ids.get(i) {
            Some(&id) => id,
            None => continue,
        };
        let material_id = mesh.material_index.and_then(|i| material_ids.get(i).copied()).unwrap_or(fallback_mat);
        let transform = mesh.node_transform;
        let world_center = transform.transform_point3(Vec3::ZERO);

        insert_object(&mut scene_db.world, mesh_id, material_id, transform,
            [world_center.x, world_center.y, world_center.z, radius], radius)?;
    }

    // ── 9. Two-light rig: key (warm directional) + fill (cool fill) ───────────
    insert_light(&mut scene_db.world, GpuLight {
            position_range: [0.0, 0.0, 0.0, f32::MAX],
            direction_outer: [-0.5_f32.sqrt(), -0.5_f32.sqrt(), 0.0, 0.0],
            color_intensity: [1.0, 0.98, 0.95, 3.0],
            shadow_index: 0,
            light_type: LightType::Directional as u32,
            inner_angle: 0.0,
            _pad: 0,
            god_rays_enabled: 0,
            god_rays_density: 1.0,
            god_rays_weight: 0.6,
            god_rays_decay: 1.0,
            god_rays_exposure: 0.7,
            flare_enabled: 0,
            flare_type: 0,
            flare_intensity: 0.0,
            flare_scale: 0.0,
            flare_tint_r: 0.0,
            flare_tint_g: 0.0,
            flare_tint_b: 0.0,
            ies_profile_index: -1,
            light_function_index: -1,
            ies_angle_scale: 0.0,
            ies_angle_offset: 0.0,
        });
    insert_light(&mut scene_db.world, GpuLight {
            position_range: [0.0, 0.0, 0.0, f32::MAX],
            direction_outer: [0.5_f32.sqrt(), 0.5_f32.sqrt(), 0.0, 0.0],
            color_intensity: [0.5, 0.6, 0.8, 1.2],
            shadow_index: u32::MAX,
            light_type: LightType::Directional as u32,
            inner_angle: 0.0,
            _pad: 0,
            god_rays_enabled: 0,
            god_rays_density: 1.0,
            god_rays_weight: 0.6,
            god_rays_decay: 1.0,
            god_rays_exposure: 0.7,
            flare_enabled: 0,
            flare_type: 0,
            flare_intensity: 0.0,
            flare_scale: 0.0,
            flare_tint_r: 0.0,
            flare_tint_g: 0.0,
            flare_tint_b: 0.0,
            ies_profile_index: -1,
            light_function_index: -1,
            ies_angle_scale: 0.0,
            ies_angle_offset: 0.0,
        });
    scene_db.world.flush_gpu_mirror(&queue);

    // ── 10. Auto-place camera to frame the bounding sphere ────────────────────
    let camera = build_camera(center, radius, &cfg);

    // ── 11. Render one frame ──────────────────────────────────────────────────
    renderer
        .render(&camera, &target_view)
        .map_err(|e| SnapshotError::Render(e.to_string()))?;

    // Flush all submitted GPU work before we copy the texture to the staging buffer.
    // Because we used new_with_external_device the graph never blocks internally —
    // this single poll is the only synchronisation point we need.
    let _ = device.poll(wgpu::PollType::wait_indefinitely());

    // ── 12. Read pixels back to CPU ───────────────────────────────────────────
    readback_rgba(&device, &queue, &target_texture, cfg.width, cfg.height).await
}

// ── SceneDB authoring ─────────────────────────────────────────────────────────

fn new_scene_db(device: &Arc<wgpu::Device>, queue: &Arc<wgpu::Queue>) -> SceneDb {
    let mut scene_db = SceneDb::new();
    let ctx = EngineGpuContext::new(device.clone(), queue.clone());
    let config = SceneGpuConfig {
        classes: Vec::new(),
        tombstone_headroom: 0,
        max_cells_metadata: 0,
    };
    let mut store = SceneGpuStore::new(&ctx, config);
    MeshComponent::register_gpu_columns_growable(&mut store, 4096, device);
    MaterialComponent::register_gpu_columns_growable(&mut store, 4096, device);
    StaticObjectComponent::register_gpu_columns_growable(&mut store, 4096, device);
    LightComponent::register_gpu_columns_growable(&mut store, 256, device);
    let mirror = GpuMirrorHandle::new(Arc::new(store), queue.clone());
    scene_db.world.attach_gpu_mirror(mirror);
    scene_db
}

fn upload_scene_rows(
    world: &mut World,
    scene: &helio_asset_compat::ConvertedScene,
) -> (Vec<Entity>, Vec<Entity>) {
    let material_ids = upload_scene_materials(world, scene);
    let mesh_ids = scene
        .meshes
        .iter()
        .map(|mesh| {
            let entity = world.spawn();
            world.insert(
                entity,
                MeshComponent {
                    vertices: mesh.vertices.clone(),
                    indices: mesh.indices.clone(),
                },
            );
            entity
        })
        .collect();
    (mesh_ids, material_ids)
}

fn fallback_material() -> MaterialComponent {
    MaterialComponent::from(GpuMaterial {
        base_color: [0.7, 0.65, 0.55, 1.0],
        emissive: [0.0; 4],
        roughness_metallic: [0.6, 0.0, 1.5, 0.0],
        tex_base_color: GpuMaterial::NO_TEXTURE,
        tex_normal: GpuMaterial::NO_TEXTURE,
        tex_roughness: GpuMaterial::NO_TEXTURE,
        tex_emissive: GpuMaterial::NO_TEXTURE,
        tex_occlusion: GpuMaterial::NO_TEXTURE,
        workflow: 0,
        flags: 0,
        material_class: 0,
        class_params: [0.0; 4],
    })
}

fn insert_object(
    world: &mut World,
    mesh: Entity,
    material: Entity,
    transform: glam::Mat4,
    bounds: [f32; 4],
    _radius: f32,
) -> Result<Entity, SnapshotError> {
    let mirror = world
        .gpu_mirror()
        .cloned()
        .ok_or_else(|| SnapshotError::Render("SceneDB GPU mirror is missing".into()))?;
    let vertices = MeshComponent::vertices_gpu_handle(mirror.store(), mesh.index())
        .filter(|handle| handle.count != 0)
        .ok_or_else(|| SnapshotError::Render("mesh has no GPU vertex range".into()))?;
    let indices = MeshComponent::indices_gpu_handle(mirror.store(), mesh.index())
        .filter(|handle| handle.count != 0)
        .ok_or_else(|| SnapshotError::Render("mesh has no GPU index range".into()))?;
    let material_component = world
        .get::<MaterialComponent>(material)
        .ok_or_else(|| SnapshotError::Render("material entity is missing".into()))?;
    let object = StaticObjectComponent::new(
        mesh.index(),
        mesh.generation().wrapping_add(1),
        material.index(),
        material.generation().wrapping_add(1),
        transform,
        bounds,
        indices.count,
        indices.offset,
        vertices.offset as i32,
        material_component.material_class,
        0,
        3,
    );
    let entity = world.spawn();
    world.insert(entity, object);
    Ok(entity)
}

fn insert_light(world: &mut World, light: GpuLight) -> Entity {
    let entity = world.spawn();
    world.insert(entity, LightComponent::from(light));
    entity
}

fn scene_db_handle(scene_db: &SceneDb) -> helio::SceneDbHandle {
    scene_db
        .world
        .gpu_mirror()
        .cloned()
        .expect("SceneDB GPU mirror is attached during initialization")
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn compute_aabb(scene: &helio_asset_compat::ConvertedScene) -> Result<(Vec3, Vec3), SnapshotError> {
    let mut aabb_min = Vec3::splat(f32::MAX);
    let mut aabb_max = Vec3::splat(f32::MIN);

    for mesh in &scene.meshes {
        let t = mesh.node_transform;
        for v in &mesh.vertices {
            let world = t.transform_point3(Vec3::from(v.position));
            aabb_min = aabb_min.min(world);
            aabb_max = aabb_max.max(world);
        }
    }

    if aabb_min.x > aabb_max.x {
        return Err(SnapshotError::EmptyModel);
    }
    Ok((aabb_min, aabb_max))
}

fn build_camera(center: Vec3, radius: f32, cfg: &SnapshotConfig) -> Camera {
    let fov = cfg.fov_degrees.to_radians();
    let aspect = cfg.width as f32 / cfg.height as f32;

    // Distance so the bounding sphere fills the FOV, with the requested margin.
    let distance = (radius / (fov * 0.5).tan()) * cfg.fit_margin;

    let (view_dir, up) = view_dir_and_up(cfg.view);
    let eye = center - view_dir * distance;

    let near = (distance - radius * 1.05).max(0.01);
    let far = distance + radius * 2.0;

    Camera::perspective_look_at(eye, center, up, fov, aspect, near, far)
}

fn view_dir_and_up(dir: ViewDirection) -> (Vec3, Vec3) {
    match dir {
        ViewDirection::Isometric => (Vec3::new(1.0, 0.8, 1.0).normalize(), Vec3::Y),
        ViewDirection::Front => (Vec3::Z, Vec3::Y),
        ViewDirection::Back => (-Vec3::Z, Vec3::Y),
        ViewDirection::Right => (Vec3::X, Vec3::Y),
        ViewDirection::Left => (-Vec3::X, Vec3::Y),
        ViewDirection::Top => (Vec3::Y, Vec3::Z),
        ViewDirection::Bottom => (-Vec3::Y, -Vec3::Z),
    }
}

async fn readback_rgba(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
) -> Result<image::RgbaImage, SnapshotError> {
    // Row stride must be aligned to 256 bytes per wgpu spec.
    let bytes_per_row = align_up(width * 4, 256);

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("snapshot-staging"),
        size: (bytes_per_row * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("snapshot-readback"),
    });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);

    // Map and wait.
    let slice = staging.slice(..);
    let (tx, rx) = futures_channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.await.unwrap()?;

    // Strip the 256-byte row padding before building the image.
    let data = slice
        .get_mapped_range()
        .map_err(|_| SnapshotError::Render("readback buffer mapping failed".into()))?;
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);
    for row in 0..height {
        let start = (row * bytes_per_row) as usize;
        let end = start + (width * 4) as usize;
        pixels.extend_from_slice(&data[start..end]);
    }
    drop(data);
    staging.unmap();

    image::RgbaImage::from_raw(width, height, pixels)
        .ok_or_else(|| SnapshotError::Render("image buffer size mismatch".into()))
}

fn align_up(n: u32, align: u32) -> u32 {
    (n + align - 1) & !(align - 1)
}

/// Returns (key, fill, rim) directional light travel vectors derived from the
/// camera's own axes so the rig always illuminates the visible faces regardless
/// of ViewDirection.
fn camera_light_rig(camera: &Camera, target: Vec3) -> (Vec3, Vec3, Vec3) {
    let forward = (target - camera.position).normalize();
    let right = forward.cross(Vec3::Y).normalize();
    let up = right.cross(forward).normalize();

    // Key: slightly right + elevated, in the forward hemisphere
    let key_dir = (forward + right * 0.45 + up * 0.55).normalize();
    // Fill: mirrored left, lower elevation
    let fill_dir = (forward - right * 0.55 + up * 0.20).normalize();
    // Rim: from behind, adds depth separation
    let rim_dir = (-forward + up * 0.30).normalize();

    (key_dir, fill_dir, rim_dir)
}

// ── SnapshotBatch ─────────────────────────────────────────────────────────────

/// A persistent batch renderer — GPU and Helio are initialised once, then
/// [`render`](SnapshotBatch::render) can be called for thousands of models.
///
/// SceneDB rows for each model are despawned after readback, so there is no
/// authored-scene growth across models.
///
/// # Example
/// ```no_run
/// use helio_snapshot::{SnapshotBatch, SnapshotConfig};
///
/// let mut batch = SnapshotBatch::new(SnapshotConfig::default()).unwrap();
/// for path in &["a.fbx", "b.fbx", "c.fbx"] {
///     let img = batch.render(path).unwrap();
///     img.save(format!("{path}.png")).unwrap();
/// }
/// ```
pub struct SnapshotBatch {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    scene_db: SceneDb,
    renderer: Renderer,
    target_texture: wgpu::Texture,
    target_view: wgpu::TextureView,
    config: SnapshotConfig,
    live_entities: Vec<Entity>,
}

impl SnapshotBatch {
    /// Initialise GPU and Helio once.  All subsequent [`render`](Self::render)
    /// calls share this device/queue/renderer.
    pub fn new(config: SnapshotConfig) -> Result<Self, SnapshotError> {
        pollster::block_on(Self::new_async(config))
    }

    async fn new_async(config: SnapshotConfig) -> Result<Self, SnapshotError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            backend_options: wgpu::BackendOptions::default(),
            display: None,
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .map_err(|_| SnapshotError::NoAdapter)?;

        let (device, queue): (wgpu::Device, wgpu::Queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("helio-snapshot-batch"),
                required_features: helio::required_wgpu_features(adapter.features()),
                required_limits: helio::required_wgpu_limits(adapter.limits()),
                ..Default::default()
            })
            .await?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

        let target_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("batch-snapshot-target"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let target_view = target_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let renderer_cfg =
            RendererConfig::new(config.width, config.height, FORMAT).with_render_scale(1.0);
        let scene_db = new_scene_db(&device, &queue);
        let renderer = RendererBuilder::new(renderer_cfg, scene_db_handle(&scene_db))
            .with_external_device()
            .with_pass_build_context(Box::new(helio_default_graphs::build_default_graph_external_with_context))
            .build(device.clone(), queue.clone(), config.width, config.height, FORMAT);

        Ok(Self {
            device,
            queue,
            scene_db,
            renderer,
            target_texture,
            target_view,
            config,
            live_entities: Vec::new(),
        })
    }

    /// Render `model_path` and return an RGBA image.
    ///
    /// The scene is wiped after each call, so this can be called in a tight
    /// loop without any GPU memory growth.
    pub fn render<P: AsRef<Path>>(
        &mut self,
        model_path: P,
    ) -> Result<image::RgbaImage, SnapshotError> {
        pollster::block_on(self.render_async(model_path))
    }

    async fn render_async<P: AsRef<Path>>(
        &mut self,
        model_path: P,
    ) -> Result<image::RgbaImage, SnapshotError> {
        // ── Load + upload ─────────────────────────────────────────────────────
        let load_cfg = LoadConfig::default()
            .with_uv_flip(self.config.flip_uv_y)
            .with_merge_meshes(false);

        let scene = load_scene_file_with_config(model_path, load_cfg)?;

        let (aabb_min, aabb_max) = compute_aabb(&scene)?;
        let center = (aabb_min + aabb_max) * 0.5;
        let radius = ((aabb_max - aabb_min) * 0.5).length().max(0.01);
        let camera = build_camera(center, radius, &self.config);

        let (mesh_ids, material_ids) = upload_scene_rows(&mut self.scene_db.world, &scene);
        self.live_entities.extend(mesh_ids.iter().chain(material_ids.iter()).copied());
        let fallback_mat = self.scene_db.world.spawn();
        self.scene_db.world.insert(fallback_mat, fallback_material());
        self.live_entities.push(fallback_mat);

        for (i, mesh) in scene.meshes.iter().enumerate() {
            let Some(&mesh_id) = mesh_ids.get(i) else {
                continue;
            };
            let material_id = mesh.material_index.and_then(|i| material_ids.get(i).copied()).unwrap_or(fallback_mat);
            let transform = mesh.node_transform;
            let world_center = transform.transform_point3(Vec3::ZERO);

            let object = insert_object(&mut self.scene_db.world, mesh_id, material_id, transform,
                [world_center.x, world_center.y, world_center.z, radius], radius)?;
            self.live_entities.push(object);
        }

        // ── Camera-relative three-point light rig ─────────────────────────────
        let (key_dir, fill_dir, rim_dir) = camera_light_rig(&camera, center);
        for (dir, color, intensity, shadow) in [
            (key_dir, [1.00_f32, 0.97, 0.92], 3.5_f32, 0_u32),
            (fill_dir, [0.55, 0.65, 0.85], 1.2, u32::MAX),
            (rim_dir, [0.90, 0.95, 1.00], 0.8, u32::MAX),
        ] {
            let light = self.scene_db.world.spawn();
            self.scene_db.world.insert(light, LightComponent::from(GpuLight {
                    position_range: [0.0, 0.0, 0.0, f32::MAX],
                    direction_outer: [dir.x, dir.y, dir.z, 0.0],
                    color_intensity: [color[0], color[1], color[2], intensity],
                    shadow_index: shadow,
                    light_type: LightType::Directional as u32,
                    inner_angle: 0.0,
                    _pad: 0,
                    god_rays_enabled: 0,
                    god_rays_density: 1.0,
                    god_rays_weight: 0.6,
                    god_rays_decay: 1.0,
                    god_rays_exposure: 0.7,
                    flare_enabled: 0,
                    flare_type: 0,
                    flare_intensity: 0.0,
                    flare_scale: 0.0,
                    flare_tint_r: 0.0,
                    flare_tint_g: 0.0,
                    flare_tint_b: 0.0,
                    ies_profile_index: -1,
                    light_function_index: -1,
                    ies_angle_scale: 0.0,
                    ies_angle_offset: 0.0,
                }));
            self.live_entities.push(light);
        }

        self.scene_db.world.flush_gpu_mirror(&self.queue);

        // ── Render ────────────────────────────────────────────────────────────
        self.renderer
            .render(&camera, &self.target_view)
            .map_err(|e| SnapshotError::Render(e.to_string()))?;

        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

        // ── Readback ──────────────────────────────────────────────────────────
        let img = readback_rgba(
            &self.device,
            &self.queue,
            &self.target_texture,
            self.config.width,
            self.config.height,
        )
        .await?;

        // Remove the per-model rows from the authoritative SceneDB world.
        // Recreate the world/mirror on the next render so all GPU pools and
        // entity generations remain consistent without a renderer-side clear.
        for entity in self.live_entities.drain(..) {
            self.scene_db.world.despawn(entity);
        }
        self.scene_db.world.flush_gpu_mirror(&self.queue);

        Ok(img)
    }
}
