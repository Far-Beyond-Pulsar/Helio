//! Standalone virtual-geometry stress scene built from the colocated scan FBXs.
//!
//! From `crates/renderer/helio`, run `cargo vg-stress` (alias for `cargo run -p examples --features vg-stress --bin vg_stress --release`).
//! Controls: WASD move, Space/Shift rise/fall, left click grabs the mouse,
//! Escape releases it (or exits when released).

mod v3_demo_common;
mod vg_stress_loading;
/// Self-provisioning of the scan meshes (shared with `build.rs`).
mod vg_stress_prep;
mod vg_stress_data {
    pub mod dims;
    pub mod layout;
    pub mod materials;
    pub mod textures;
}

use helio::{
    Camera, MeshUpload, PackedVertex, Renderer, RendererBuilder, RendererConfig,
    required_experimental_features, required_wgpu_features, required_wgpu_limits,
};
use helio_core::{
    BufferHandle, BufferKey, SceneDerivation, SceneDerivationContext, SceneDerivationOutput,
};
use helio_default_graphs::build_default_graph_external_with_context;
use helio_vg_build::{BuildOptions, MeshInput};
use helio_vg_format::{
    GpuVgInstance, INST_CONE_CULL, INST_ENABLED, INST_LOD_SUPPRESS, INST_SHADOW_CASTER,
    VG_ASSET_READY_FLAG_READY,
};
use helio_vg_stream::{VgRuntime, VgeoBytes, VgeoSource, content_words};
use pulsar_scenedb::{Entity, SceneDb};
use v3_demo_common::{
    directional_light, make_material, new_scene_db_with_gpu_mirror, point_light, scene_db_handle,
    spawn_light, spawn_material, spawn_mesh, spawn_object, spawn_sky, spot_light,
};

use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

use bytemuck::Zeroable;
use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
    sync::{Arc, Mutex, RwLock},
};
use vg_stress_data::{
    layout::{LightKind, Placement},
    materials::Mat,
};

const VG_INSTANCES: BufferKey = BufferKey::of("vg_instances");
const VG_INSTANCE_COUNT: BufferKey = BufferKey::of("vg_instance_count");
const VG_ASSET_READY: BufferKey = BufferKey::of("vg_asset_ready");
const CLASSIC_SUPPRESSED: BufferKey = BufferKey::of("classic_suppressed");
const VG_FLAGS: u32 = INST_ENABLED | INST_SHADOW_CASTER | INST_LOD_SUPPRESS | INST_CONE_CULL;
/// The demo presents the scanned assets together on one metre-based stress
/// grid. Normalize their imported height to this scene height while
/// leaving the source mesh and its cached VG data unchanged.
const SCAN_SCENE_HEIGHT_METERS: f32 = 2.0;
/// Keep the classic fallback below SceneDB's 4 GiB var-len pool ceiling.
/// The full scan is still used to build VG and is preserved in the source
/// FBX; classic geometry is only a temporary fallback until VG roots arrive.
const MAX_CLASSIC_PROXY_VERTICES: usize = 300_000;

fn stable_content_id(vertices: &[PackedVertex], indices: &[u32]) -> u128 {
    let mut low = 0xcbf29ce484222325u64;
    let mut high = 0x84222325cbf29ce4u64;
    let bytes = bytemuck::cast_slice::<PackedVertex, u8>(vertices);
    for byte in bytes
        .iter()
        .copied()
        .chain(bytemuck::cast_slice::<u32, u8>(indices).iter().copied())
    {
        low = (low ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        high = (high ^ u64::from(byte).rotate_left(1)).wrapping_mul(0x9e3779b185ebca87);
    }
    (u128::from(high) << 64) | u128::from(low)
}

/// True for the prepared Stanford scans (`vg_stress_prep::scans::MODELS`), which
/// are normalised to [`SCAN_SCENE_HEIGHT_METERS`] on load.
fn is_scan_asset(path: &std::path::Path) -> bool {
    path.file_stem()
        .and_then(|s| s.to_str())
        .is_some_and(|stem| {
            vg_stress_prep::scans::MODELS
                .iter()
                .any(|m| m.name.eq_ignore_ascii_case(stem))
        })
}

fn load_or_build_vgeo_cache(
    content_id: u128,
    vertices: &[PackedVertex],
    indices: &[u32],
    cache_path: PathBuf,
) -> Result<(Arc<VgeoBytes>, bool), String> {
    if let Ok(bytes) = std::fs::read(&cache_path) {
        if let Ok(vgeo) = VgeoBytes::new(bytes) {
            if vgeo.header().content_id == content_words(content_id)
                && helio_vg_build::is_current_cook(&vgeo.header())
            {
                return Ok((Arc::new(vgeo), true));
            }
        }
    }

    // Called only on the loader worker after previews are published.
    let positions: Vec<[f32; 3]> = vertices.iter().map(|v| v.position).collect();
    let attributes = pack_vgeo_attributes(vertices);
    let mesh = MeshInput {
        positions: &positions,
        attributes: &attributes,
        attribute_stride: helio_vg_format::ATTR_STRIDE as usize,
        indices: &indices[..indices.len() / 3 * 3],
        tri_material: &[],
    };
    let bytes = helio_vg_build::build_vgeo(
        &mesh,
        &BuildOptions {
            content_id: content_words(content_id),
            resident_levels: 2,
        },
    )
    .map_err(|error| error.to_string())?;
    if let Some(parent) = cache_path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let temporary = cache_path.with_extension("vgeo.tmp");
    std::fs::write(&temporary, &bytes).map_err(|error| error.to_string())?;
    replace_cache_file(&temporary, &cache_path).map_err(|error| {
        let _ = std::fs::remove_file(&temporary);
        error
    })?;
    VgeoBytes::new(bytes)
        .map(|vgeo| (Arc::new(vgeo), false))
        .map_err(|error| error.to_string())
}

/// Replace a cooked cache on Windows too, where rename does not overwrite an
/// existing destination. Keep a backup until the new file is in place so a
/// failed install can restore the previous cache.
fn replace_cache_file(
    temporary: &std::path::Path,
    destination: &std::path::Path,
) -> Result<(), String> {
    if !destination.exists() {
        return std::fs::rename(temporary, destination).map_err(|error| error.to_string());
    }
    let mut backup_name = destination
        .file_name()
        .ok_or_else(|| "cache destination has no file name".to_string())?
        .to_os_string();
    backup_name.push(".bak");
    let backup = destination.with_file_name(backup_name);
    if backup.exists() {
        std::fs::remove_file(&backup).map_err(|error| error.to_string())?;
    }
    std::fs::rename(destination, &backup).map_err(|error| error.to_string())?;
    match std::fs::rename(temporary, destination) {
        Ok(()) => {
            let _ = std::fs::remove_file(backup);
            Ok(())
        }
        Err(error) => {
            let restore = std::fs::rename(&backup, destination);
            match restore {
                Ok(()) => Err(error.to_string()),
                Err(restore_error) => Err(format!(
                    "install failed ({error}); restoring previous cache also failed ({restore_error})"
                )),
            }
        }
    }
}

#[cfg(test)]
mod cache_file_tests {
    use super::replace_cache_file;

    #[test]
    fn replacing_existing_cache_keeps_new_file_and_cleans_backup() {
        let dir = std::env::temp_dir().join(format!(
            "vg-cache-replace-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let destination = dir.join("asset.vgeo");
        let temporary = dir.join("asset.vgeo.tmp");
        std::fs::write(&destination, b"old cook").unwrap();
        std::fs::write(&temporary, b"new cook").unwrap();

        replace_cache_file(&temporary, &destination).unwrap();

        assert_eq!(std::fs::read(&destination).unwrap(), b"new cook");
        assert!(!temporary.exists());
        assert!(!dir.join("asset.vgeo.bak").exists());
        std::fs::remove_dir_all(dir).unwrap();
    }
}

fn pack_vgeo_attributes(vertices: &[PackedVertex]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(vertices.len() * helio_vg_format::ATTR_STRIDE as usize);
    for vertex in vertices {
        let tangent = (vertex.tangent & 0x00ff_ffff)
            | ((if vertex.bitangent_sign < 0.0 {
                0x81u32
            } else {
                0x7fu32
            }) << 24);
        bytes.extend_from_slice(&vertex.normal.to_le_bytes());
        bytes.extend_from_slice(&tangent.to_le_bytes());
        for uv in [vertex.tex_coords0, vertex.tex_coords1] {
            bytes.extend_from_slice(&half::f16::from_f32(uv[0]).to_bits().to_le_bytes());
            bytes.extend_from_slice(&half::f16::from_f32(uv[1]).to_bits().to_le_bytes());
        }
    }
    bytes
}

/// Make a bounded classic fallback from evenly sampled source triangles.
/// Virtual geometry is built from the untouched full-resolution input; this
/// proxy only keeps the initial/failure rendering path within the mesh pool
/// ceiling while root pages stream in.
fn classic_proxy(vertices: &[PackedVertex], indices: &[u32], limit: usize) -> MeshUpload {
    if vertices.len() <= limit {
        return MeshUpload {
            vertices: vertices.to_vec(),
            indices: indices.to_vec(),
        };
    }

    let triangle_count = indices.len() / 3;
    let target_triangles = (limit / 3).max(1);
    let stride = triangle_count.div_ceil(target_triangles).max(1);
    let mut remap = std::collections::HashMap::<u32, u32>::with_capacity(limit);
    let mut proxy_vertices = Vec::with_capacity(limit);
    let mut proxy_indices = Vec::with_capacity(target_triangles * 3);

    for triangle in (0..triangle_count).step_by(stride) {
        let source = &indices[triangle * 3..triangle * 3 + 3];
        if source.iter().any(|&index| index as usize >= vertices.len()) {
            continue;
        }
        let new_vertices = source
            .iter()
            .filter(|&&index| !remap.contains_key(&index))
            .count();
        if proxy_vertices.len() + new_vertices > limit {
            continue;
        }
        for &source_index in source {
            let next_index = proxy_vertices.len() as u32;
            let index = *remap.entry(source_index).or_insert_with(|| {
                proxy_vertices.push(vertices[source_index as usize]);
                next_index
            });
            proxy_indices.push(index);
        }
    }

    MeshUpload {
        vertices: proxy_vertices,
        indices: proxy_indices,
    }
}

#[derive(Clone, Copy)]
struct VgTarget {
    vertex_offset: u32,
    vertex_count: u32,
    object_row: u32,
}

#[derive(Default)]
struct VgSceneTargets {
    targets: Vec<VgTarget>,
    object_rows: u32,
}

struct SceneVgDerivation {
    runtime: VgRuntime,
    enabled: bool,
    targets: Arc<Mutex<VgSceneTargets>>,
    instances: wgpu::Buffer,
    count: wgpu::Buffer,
    ready: wgpu::Buffer,
    classic: wgpu::Buffer,
    instance_capacity: u32,
    object_rows: u32,
    ready_capacity: u32,
    content_generation: u64,
    buffer_epoch: u64,
}

impl SceneVgDerivation {
    fn new(
        device: &wgpu::Device,
        runtime: VgRuntime,
        targets: Arc<Mutex<VgSceneTargets>>,
        enabled: bool,
    ) -> Self {
        let instance_capacity = 1;
        let object_rows = 1;
        let make = |label: &'static str, bytes: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(16),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        Self {
            runtime,
            enabled,
            targets,
            instances: make("VG Stress Instances", instance_capacity as u64 * 16),
            count: make("VG Stress Instance Count", 16),
            ready: make("VG Stress Asset Ready", 16),
            classic: make("VG Stress Classic Suppression", object_rows as u64 * 4),
            instance_capacity,
            object_rows,
            ready_capacity: 1,
            content_generation: 1,
            buffer_epoch: 1,
        }
    }

    fn handle(buffer: &wgpu::Buffer, row_bytes: u64, generation: u64, epoch: u64) -> BufferHandle {
        BufferHandle {
            buffer: buffer.clone(),
            epoch,
            row_bytes,
            content_generation: generation,
        }
    }
}

impl SceneDerivation for SceneVgDerivation {
    fn name(&self) -> &'static str {
        "VgStressScene"
    }

    fn derive(
        &mut self,
        ctx: &SceneDerivationContext<'_>,
        _encoder: &mut wgpu::CommandEncoder,
    ) -> SceneDerivationOutput {
        if self.enabled {
            self.runtime.service(ctx.device, ctx.queue);
        }
        let targets = self.targets.lock().expect("VG scene targets");
        let instance_count = targets.targets.len().max(1) as u32;
        let object_rows = targets.object_rows.max(1);
        let make = |label, bytes: u64| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(16),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        if instance_count > self.instance_capacity {
            self.instance_capacity = instance_count.next_power_of_two();
            self.instances = make("VG Stress Instances", self.instance_capacity as u64 * 16);
            self.buffer_epoch += 1;
        }
        if object_rows > self.object_rows {
            self.object_rows = object_rows.next_power_of_two();
            self.classic = make("VG Stress Classic Suppression", self.object_rows as u64 * 4);
            self.buffer_epoch += 1;
        }
        let (ready_rows, _) = self.runtime.asset_ready_rows(targets.targets.len().max(1));
        if ready_rows.len() > self.ready_capacity as usize {
            self.ready_capacity = ready_rows.len().next_power_of_two() as u32;
            self.ready = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("VG Stress Asset Ready"),
                size: self.ready_capacity as u64 * 16,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.buffer_epoch += 1;
        }

        let mut instances = Vec::with_capacity(self.instance_capacity as usize);
        let mut classic = vec![0u32; self.object_rows as usize];
        for target in targets.targets.iter().filter(|_| self.enabled) {
            let Some(asset) = ready_rows.iter().find(|row| {
                row.vertex_offset == target.vertex_offset
                    && row.vertex_count == target.vertex_count
                    && row.flags & VG_ASSET_READY_FLAG_READY != 0
            }) else {
                continue;
            };
            instances.push(GpuVgInstance {
                asset_slot: asset.asset_slot,
                flags: VG_FLAGS,
                lod_error_scale: 1.0,
                object_row: target.object_row,
            });
            if let Some(row) = classic.get_mut(target.object_row as usize) {
                *row = 1;
            }
        }
        let count = [instances.len() as u32, 0, 0, 0];
        instances.resize(self.instance_capacity as usize, GpuVgInstance::zeroed());
        ctx.queue
            .write_buffer(&self.instances, 0, bytemuck::cast_slice(&instances));
        ctx.queue
            .write_buffer(&self.count, 0, bytemuck::cast_slice(&count));
        ctx.queue
            .write_buffer(&self.ready, 0, bytemuck::cast_slice(&ready_rows));
        ctx.queue
            .write_buffer(&self.classic, 0, bytemuck::cast_slice(&classic));
        self.content_generation = self.content_generation.wrapping_add(1);

        SceneDerivationOutput {
            buffers: vec![
                (
                    VG_INSTANCES,
                    Self::handle(
                        &self.instances,
                        16,
                        self.content_generation,
                        self.buffer_epoch,
                    ),
                ),
                (
                    VG_INSTANCE_COUNT,
                    Self::handle(&self.count, 16, self.content_generation, self.buffer_epoch),
                ),
                (
                    VG_ASSET_READY,
                    Self::handle(&self.ready, 16, self.content_generation, self.buffer_epoch),
                ),
                (
                    CLASSIC_SUPPRESSED,
                    Self::handle(&self.classic, 4, self.content_generation, self.buffer_epoch),
                ),
            ],
            ..Default::default()
        }
    }
}

/// Print what the scene contains and what it will cost, before loading.
fn print_scene_report(
    layout: &vg_stress_data::layout::MuseumLayout,
    config: &vg_stress_prep::catalog::Config,
    root: &std::path::Path,
) {
    let (kit, statues, trees, total) = layout.triangle_report();
    let fmt = |t: u64| format!("{:.1}M", t as f64 / 1e6);
    println!(
        "[Museum] density {:.2} (VG_STRESS_DENSITY / --density: low, normal, high, insane or a number); trees: {}",
        config.density,
        if config.trees.is_empty() {
            "none".to_string()
        } else {
            config
                .trees
                .iter()
                .map(|t| t.id)
                .collect::<Vec<_>>()
                .join(", ")
        }
    );
    println!(
        "[Museum] expected source triangles: {} total = architecture {} + statues {} ({} statue instances) + trees {} ({} tree instances)",
        fmt(total),
        fmt(kit),
        fmt(statues),
        layout.statues.len(),
        fmt(trees),
        layout.tree_count()
    );
    let masked: u64 = layout
        .placements
        .iter()
        .filter(|p| p.asset.contains('#'))
        .map(|p| {
            let (path, variant) = vg_stress_loading::split_key(&p.asset);
            vg_stress_prep::catalog::TREES
                .iter()
                .find(|t| path.contains(t.id))
                .and_then(|t| t.variants.iter().find(|v| Some(v.mesh) == variant))
                .map_or(0, |v| v.masked_triangles)
        })
        .sum();
    println!(
        "[Museum] of the tree triangles {} are alpha-masked cards/leaves: classic path (60k-vertex stand-ins) until VG supports masked materials",
        fmt(masked)
    );
    let assets = root.join("assets/vg_stress");
    let mats = Mat::all(config);
    let jobs = vg_stress_data::textures::jobs(&assets, &mats, config);
    let tex = vg_stress_data::textures::estimate_bytes(&jobs);
    let download: u64 = vg_stress_prep::catalog::MARBLES
        .iter()
        .map(|m| vg_stress_prep::polyhaven::pinned_bytes(m.id))
        .chain(
            config
                .trees
                .iter()
                .map(|t| vg_stress_prep::polyhaven::pinned_bytes(t.id)),
        )
        .sum();
    println!(
        "[Museum] textures: {} uploads, about {:.2} GiB of GPU memory at quality {:?} (VG_STRESS_TEXTURES=low|balanced|max); Poly Haven downloads about {:.2} GB",
        jobs.len(),
        tex as f64 / (1u64 << 30) as f64,
        config.quality,
        download as f64 / 1e9
    );
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("Event loop error");
}

struct App {
    state: Option<AppState>,
}

struct AppState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    surface_format: wgpu::TextureFormat,
    renderer: Renderer,
    last_frame: std::time::Instant,

    cam_pos: glam::Vec3,
    cam_yaw: f32,
    cam_pitch: f32,
    keys: HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta: (f32, f32),

    scene_db: SceneDb,
    runtime: VgRuntime,
    loader: Option<vg_stress_loading::Loader>,
    placements: HashMap<String, Vec<Placement>>,
    materials: HashMap<Mat, Entity>,
    vg_ranges: HashMap<u128, (u32, u32)>,
    targets: Arc<Mutex<VgSceneTargets>>,
    loading_errors: usize,
    loaded_objects: usize,
    started: std::time::Instant,
    frames: u64,
    render_errors: u64,
    smoke_seconds: Option<f32>,
    debug_views: Vec<helio_core::DebugViewDescriptor>,
    debug_index: usize,
    scene_triangles: u64,
    loading_status: String,
    last_title: std::time::Instant,
}

impl App {
    fn new() -> Self {
        Self { state: None }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Helio – VG Museum | starting")
                        .with_visible(std::env::var_os("HELIO_VG_SMOKE_SECONDS").is_none())
                        .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32)),
                )
                .expect("window"),
        );

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::empty(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let surface = instance.create_surface(window.clone()).expect("surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .expect("adapter");

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Device"),
            required_features: required_wgpu_features(adapter.features()),
            required_limits: required_wgpu_limits(adapter.limits()),
            experimental_features: required_experimental_features(adapter.features()),
            ..Default::default()
        }))
        .expect("device");

        device.on_uncaptured_error(std::sync::Arc::new(|e: wgpu::Error| {
            panic!("[GPU UNCAPTURED ERROR] {:?}", e);
        }));
        let info = adapter.get_info();
        println!(
            "[WGPU] Backend: {:?}, Device: {}, Driver: {}",
            info.backend, info.name, info.driver
        );
        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);
        let size = window.inner_size();
        surface.configure(
            &device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                width: size.width,
                height: size.height,
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: caps.alpha_modes[0],
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
                color_space: wgpu::SurfaceColorSpace::Auto,
            },
        );

        let mut scene_db = new_scene_db_with_gpu_mirror(&device, &queue);
        let runtime = VgRuntime::default();
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let museum_config = vg_stress_prep::catalog::Config::from_env_and_args();
        let layout = vg_stress_data::layout::museum_layout_with(
            &vg_stress_data::dims::Dims::load(),
            &museum_config,
        );
        print_scene_report(&layout, &museum_config, &root);
        let textures = Arc::new(RwLock::new(pulsar_scenedb::gpu::TextureStore::new(1024)));
        let mirror = scene_db_handle(&scene_db)
            .with_texture_store(textures)
            .expect("museum texture store");
        scene_db.world.attach_gpu_mirror(mirror);
        let targets = Arc::new(Mutex::new(VgSceneTargets::default()));
        let mut materials = HashMap::new();
        for material in Mat::all(&museum_config) {
            let surface = material.surface();
            let entity = spawn_material(
                &mut scene_db.world,
                make_material(
                    surface.base_color,
                    surface.roughness,
                    surface.metallic,
                    surface.emissive_color,
                    surface.emissive_intensity,
                ),
            );
            materials.insert(material, entity);
        }
        spawn_sky(&mut scene_db.world, [1.0, 1.0, 1.0]);
        spawn_light(
            &mut scene_db.world,
            directional_light([0.35, -0.8, -0.3], [1.0, 0.94, 0.82], 3.0),
        );
        for light in &layout.lights {
            let direction = glam::Vec3::from_array(light.direction)
                .normalize_or_zero()
                .to_array();
            let gpu = match light.kind {
                LightKind::Point => {
                    point_light(light.position, light.color, light.intensity, light.range)
                }
                LightKind::Spot => spot_light(
                    light.position,
                    direction,
                    light.color,
                    light.intensity,
                    light.range,
                    light.inner_deg.to_radians(),
                    light.outer_deg.to_radians(),
                ),
                LightKind::Directional => {
                    directional_light(direction, light.color, light.intensity)
                }
            };
            spawn_light(&mut scene_db.world, gpu);
        }
        let mut placements: HashMap<String, Vec<Placement>> = HashMap::new();
        for placement in &layout.placements {
            placements
                .entry(placement.asset.clone())
                .or_default()
                .push(placement.clone());
        }
        let vg_supported = helio_default_graphs::virtual_geometry_supported(&device);
        if !vg_supported {
            log::warn!(
                "This device does not support Helio's virtual-geometry passes; rendering scans through the classic mesh path."
            );
        }
        let mut config = RendererConfig::new(size.width, size.height, format);
        if vg_supported {
            config = config.with_virtual_geometry(&runtime);
        }
        let derivation =
            SceneVgDerivation::new(&device, runtime.clone(), targets.clone(), vg_supported);
        let mut renderer = RendererBuilder::new(config, scene_db_handle(&scene_db))
            .with_pass_build_context(Box::new(build_default_graph_external_with_context))
            .with_scene_derivation(Box::new(derivation))
            .build(
                device.clone(),
                queue.clone(),
                size.width,
                size.height,
                format,
            );
        renderer.set_ambient([1.0, 0.95, 0.85], 0.05);
        renderer.set_clear_color([0.018, 0.022, 0.035, 1.0]);
        let mut debug_views = renderer.available_debug_views();
        debug_views.push(helio_core::DebugViewDescriptor {
            name: "Normal",
            debug_mode: 0,
            description: "Lit museum scene",
        });
        debug_views.sort_by_key(|v| v.debug_mode);
        debug_views.dedup_by_key(|v| v.debug_mode);
        println!(
            "[Museum controls] 0 normal, F3 VG triangles, F4 VG clusters, F5 LOD level, F6 triangle size, F7 raster tier, F8 page residency, F9 error vs tau, [ / ] every debug view (VG_DEBUG_VIEW=<id or key> picks one at startup), F1 list views"
        );
        for view in &debug_views {
            println!(
                "  {}: {} – {}",
                view.debug_mode, view.name, view.description
            );
        }

        // No source parsing or cooking takes place on the window thread.
        let loader =
            vg_stress_loading::start(root, &layout.placements, vg_supported, museum_config);
        println!(
            "[Museum] {} placements, {} exhibits; loading in background",
            layout.placements.len(),
            layout.statues.len()
        );
        self.state = Some(AppState {
            window,
            surface,
            device,
            queue,
            surface_format: format,
            renderer,
            last_frame: std::time::Instant::now(),
            cam_pos: glam::Vec3::new(0.0, 3.0, 34.0),
            cam_yaw: 0.0,
            cam_pitch: 0.04,
            keys: HashSet::new(),
            cursor_grabbed: false,
            mouse_delta: (0.0, 0.0),
            scene_db,
            runtime,
            loader: Some(loader),
            placements,
            materials,
            vg_ranges: HashMap::new(),
            targets,
            loading_errors: 0,
            loaded_objects: 0,
            started: std::time::Instant::now(),
            frames: 0,
            render_errors: 0,
            smoke_seconds: std::env::var("HELIO_VG_SMOKE_SECONDS")
                .ok()
                .and_then(|s| s.parse().ok()),
            debug_index: helio_vg_format::debug_view::from_env()
                .and_then(|view| debug_views.iter().position(|v| v.debug_mode == view.id()))
                .unwrap_or(0),
            debug_views,
            scene_triangles: 0,
            loading_status: "Loading museum".into(),
            last_title: std::time::Instant::now(),
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        ..
                    },
                ..
            } => {
                if state.cursor_grabbed {
                    state.cursor_grabbed = false;
                    let _ = state.window.set_cursor_grab(CursorGrabMode::None);
                    state.window.set_cursor_visible(true);
                } else {
                    event_loop.exit();
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ks,
                        physical_key: PhysicalKey::Code(key),
                        repeat,
                        ..
                    },
                ..
            } => match ks {
                ElementState::Pressed => {
                    if !repeat && state.debug_key(key) {
                        return;
                    }
                    state.keys.insert(key);
                }
                ElementState::Released => {
                    state.keys.remove(&key);
                }
            },
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                if !state.cursor_grabbed {
                    let ok = state
                        .window
                        .set_cursor_grab(CursorGrabMode::Confined)
                        .or_else(|_| state.window.set_cursor_grab(CursorGrabMode::Locked))
                        .is_ok();
                    if ok {
                        state.window.set_cursor_visible(false);
                        state.cursor_grabbed = true;
                    }
                }
            }
            WindowEvent::Resized(s) if s.width > 0 && s.height > 0 => {
                state.surface.configure(
                    &state.device,
                    &wgpu::SurfaceConfiguration {
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                        format: state.surface_format,
                        width: s.width,
                        height: s.height,
                        present_mode: wgpu::PresentMode::Fifo,
                        alpha_mode: wgpu::CompositeAlphaMode::Auto,
                        view_formats: vec![],
                        desired_maximum_frame_latency: 2,
                        color_space: wgpu::SurfaceColorSpace::Auto,
                    },
                );
                state.renderer.set_render_size(s.width, s.height);
            }
            WindowEvent::RedrawRequested => {
                let now = std::time::Instant::now();
                let dt = (now - state.last_frame).as_secs_f32();
                state.last_frame = now;
                state.render(dt);
                if state.smoke_complete() {
                    event_loop.exit();
                    return;
                }
                state.window.request_redraw();
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: winit::event::DeviceId, event: DeviceEvent) {
        let Some(state) = &mut self.state else { return };
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            if state.cursor_grabbed {
                state.mouse_delta.0 += dx as f32;
                state.mouse_delta.1 += dy as f32;
            }
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(s) = &mut self.state {
            if s.smoke_seconds.is_some() {
                // Hidden windows do not receive normal redraw events on Windows.
                event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
                let now = std::time::Instant::now();
                let dt = (now - s.last_frame).as_secs_f32();
                s.last_frame = now;
                if s.frames > 0 && s.frames % 30 == 0 {
                    s.debug_key(KeyCode::BracketRight);
                }
                s.render(dt);
                if s.smoke_complete() {
                    event_loop.exit();
                }
                return;
            }
            s.window.request_redraw();
        }
    }
}

impl AppState {
    fn smoke_complete(&self) -> bool {
        if !self
            .smoke_seconds
            .is_some_and(|seconds| self.started.elapsed().as_secs_f32() >= seconds)
        {
            return false;
        }
        println!(
            "[Museum smoke] {} frames, {} objects, {} VG assets, {} loading errors, {} render errors, loader active={}",
            self.frames,
            self.loaded_objects,
            self.vg_ranges
                .keys()
                .filter(|&&id| self.runtime.has_asset(id))
                .count(),
            self.loading_errors,
            self.render_errors,
            self.loader.is_some()
        );
        assert!(
            self.frames > 1
                && self.loaded_objects > 0
                && self.render_errors == 0
                && self.loading_errors == 0,
            "museum startup smoke failed"
        );
        true
    }

    fn debug_key(&mut self, key: KeyCode) -> bool {
        let mode = match key {
            KeyCode::Digit0 => Some(0),
            KeyCode::F3 => Some(20),
            KeyCode::F4 => Some(21),
            KeyCode::F5 => Some(helio_vg_format::debug_view::VgDebugView::LodLevel.id()),
            KeyCode::F6 => Some(helio_vg_format::debug_view::VgDebugView::TriangleSize.id()),
            KeyCode::F7 => Some(helio_vg_format::debug_view::VgDebugView::RasterTier.id()),
            KeyCode::F8 => Some(helio_vg_format::debug_view::VgDebugView::PageResidency.id()),
            KeyCode::F9 => Some(helio_vg_format::debug_view::VgDebugView::ErrorVsTau.id()),
            KeyCode::Digit1 => Some(4),
            KeyCode::Digit2 => Some(5),
            KeyCode::Digit3 => Some(6),
            KeyCode::Digit4 => Some(7),
            KeyCode::Digit5 => Some(8),
            KeyCode::Digit6 => Some(9),
            KeyCode::Digit7 => Some(10),
            KeyCode::Digit8 => Some(22),
            KeyCode::Digit9 => Some(23),
            KeyCode::BracketRight | KeyCode::PageDown => {
                self.debug_index = (self.debug_index + 1) % self.debug_views.len();
                None
            }
            KeyCode::BracketLeft | KeyCode::PageUp => {
                self.debug_index =
                    (self.debug_index + self.debug_views.len() - 1) % self.debug_views.len();
                None
            }
            KeyCode::F1 => {
                for view in &self.debug_views {
                    println!("{}: {} – {}", view.debug_mode, view.name, view.description);
                }
                return true;
            }
            _ => return false,
        };
        if let Some(mode) = mode {
            let mode = if self.debug_views[self.debug_index].debug_mode == mode {
                0
            } else {
                mode
            };
            if let Some(index) = self.debug_views.iter().position(|v| v.debug_mode == mode) {
                self.debug_index = index;
            } else {
                log::warn!("Debug mode {mode} is not available on this device");
                return true;
            }
        }
        let view = self.debug_views[self.debug_index];
        self.renderer.set_debug_mode(view.debug_mode);
        println!("[Museum view] {}: {}", view.name, view.description);
        self.update_title();
        true
    }

    fn update_title(&mut self) {
        let vg_triangles = self
            .renderer
            .find_pass::<helio_pass_vg_cull::VgCullEarlyPass>()
            .and_then(|pass| pass.selected_triangle_count());
        // The cull's existing async stats readback: no extra GPU stall.
        let vg_stats = self
            .renderer
            .find_pass::<helio_pass_vg_cull::VgCullEarlyPass>()
            .and_then(|pass| pass.stats_snapshot())
            .map(|stats| stats.summary())
            .unwrap_or_default();
        self.window.set_title(&format!(
            "Helio – VG Museum | {} | Scene tris: {} | VG selected: {} | {} | {}{}",
            self.debug_views[self.debug_index].name,
            self.scene_triangles,
            vg_triangles
                .map(|n| n.to_string())
                .unwrap_or_else(|| "pending".into()),
            vg_stats,
            self.loading_status,
            if self.loading_errors == 0 {
                String::new()
            } else {
                format!(" ({} errors)", self.loading_errors)
            }
        ));
        self.last_title = std::time::Instant::now();
    }

    fn poll_loader(&mut self) {
        use std::sync::mpsc::TryRecvError;
        use vg_stress_loading::Event;
        let start = std::time::Instant::now();
        while start.elapsed() < std::time::Duration::from_millis(3) {
            let Some(loader) = &self.loader else { break };
            let event = match loader.events.try_recv() {
                Ok(event) => event,
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.loader = None;
                    self.loading_errors += 1;
                    self.loading_status = "Loader stopped; see log".into();
                    break;
                }
            };
            match event {
                Event::Part(part) => {
                    let content = part
                        .meta
                        .content
                        .map(|w| u128::from(w[0]) | (u128::from(w[1]) << 64));
                    let mesh = spawn_mesh(&mut self.scene_db.world, part.mesh);
                    let mirror = scene_db_handle(&self.scene_db);
                    let range = helio_pass_gbuffer::MeshComponent::vertices_gpu_handle(
                        mirror.store(),
                        mesh.index(),
                    )
                    .expect("uploaded mesh vertex range");
                    if let Some(content) = content {
                        self.vg_ranges.insert(content, (range.offset, range.count));
                    }
                    let node = glam::Mat4::from_cols_array(&part.meta.transform);
                    let mut targets = self.targets.lock().expect("VG scene targets");
                    for placement in &self.placements[&part.asset] {
                        self.scene_triangles += part.meta.source_triangles;
                        let material = placement
                            .slots
                            .iter()
                            .find(|(slot, _)| {
                                slot == &part.meta.slot
                                    || part.meta.slot.strip_suffix(".001") == Some(slot.as_str())
                            })
                            .map(|(_, material)| *material)
                            .unwrap_or_else(|| Mat::for_kit_slot(&part.meta.slot));
                        let [pitch, yaw, roll] = placement.rotation.map(f32::to_radians);
                        let transform = glam::Mat4::from_scale_rotation_translation(
                            glam::Vec3::from_array(placement.scale),
                            glam::Quat::from_euler(glam::EulerRot::YXZ, yaw, pitch, roll),
                            glam::Vec3::from_array(placement.position.map(|v| v as f32)),
                        ) * node;
                        let scale = transform
                            .x_axis
                            .truncate()
                            .length()
                            .max(transform.y_axis.truncate().length())
                            .max(transform.z_axis.truncate().length());
                        let object = spawn_object(
                            &mut self.scene_db.world,
                            mesh,
                            self.materials[&material],
                            transform,
                            part.meta.radius * scale,
                        )
                        .expect("museum object mesh ranges");
                        targets.object_rows = targets.object_rows.max(object.index() + 1);
                        // Parts without a VG content id (alpha-masked tree cards
                        // for now) stay on the classic path.
                        if placement.virtual_geometry && part.meta.content.is_some() {
                            targets.targets.push(VgTarget {
                                vertex_offset: range.offset,
                                vertex_count: range.count,
                                object_row: object.index(),
                            });
                        }
                        self.loaded_objects += 1;
                    }
                    // At most one mesh upload and its instances per frame.
                    break;
                }
                Event::Vgeo(content, source) => {
                    if let Some(&(offset, count)) = self.vg_ranges.get(&content) {
                        self.runtime.request_asset(content, offset, count, source);
                    }
                    break;
                }
                Event::Texture {
                    bindings,
                    srgb,
                    width,
                    height,
                    mips,
                } => {
                    let mirror = scene_db_handle(&self.scene_db);
                    let textures = mirror.texture_store().expect("museum texture store");
                    let descriptor = wgpu::TextureDescriptor {
                        label: Some("Museum texture"),
                        size: wgpu::Extent3d {
                            width,
                            height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: mips.len() as u32,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: if srgb {
                            wgpu::TextureFormat::Rgba8UnormSrgb
                        } else {
                            wgpu::TextureFormat::Rgba8Unorm
                        },
                        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                        view_formats: &[],
                    };
                    let levels: Vec<&[u8]> = mips.iter().map(Vec::as_slice).collect();
                    match textures.write().expect("texture store").register_mips(
                        &self.device,
                        &self.queue,
                        &descriptor,
                        &levels,
                    ) {
                        Ok(slot) => {
                            for (material, channel) in bindings {
                                let Some(mut row) = self
                                    .scene_db
                                    .world
                                    .get_mut::<helio_pass_gbuffer::MaterialComponent>(
                                    self.materials[&material],
                                ) else {
                                    continue;
                                };
                                match channel {
                                    0 => {
                                        row.tex_base_color = slot;
                                        row.base_color = [1.0; 4];
                                    }
                                    1 => row.tex_normal = slot,
                                    _ => {
                                        // ARM: occlusion R, roughness G, metalness B.
                                        // Factors multiply the maps, so pass them through.
                                        row.tex_roughness = slot;
                                        row.tex_occlusion = slot;
                                        row.roughness_metallic[0] = 1.0;
                                        row.roughness_metallic[1] = 1.0;
                                    }
                                }
                            }
                        }
                        Err(error) => {
                            self.loading_errors += 1;
                            log::error!("Museum texture: {error:?}");
                        }
                    };
                    break;
                }
                Event::Status(status) => {
                    println!("[Museum] {status}");
                    self.loading_status = status;
                }
                Event::Error(error) => {
                    self.loading_errors += 1;
                    log::error!("Museum loading: {error}");
                }
                Event::Finished => {
                    self.loader = None;
                    self.loading_status = format!("Ready: {} objects", self.loaded_objects);
                    println!(
                        "[Museum] Loading complete: {} objects, {} errors",
                        self.loaded_objects, self.loading_errors
                    );
                    break;
                }
            }
        }
    }

    fn render(&mut self, dt: f32) {
        self.poll_loader();
        let dt = dt.min(0.1);
        const SPEED: f32 = 6.0;
        const SENS: f32 = 0.002;

        self.cam_yaw += self.mouse_delta.0 * SENS;
        self.cam_pitch = (self.cam_pitch - self.mouse_delta.1 * SENS).clamp(-1.4, 1.4);
        self.mouse_delta = (0.0, 0.0);

        let (sy, cy) = self.cam_yaw.sin_cos();
        let (sp, cp) = self.cam_pitch.sin_cos();
        let forward = glam::Vec3::new(sy * cp, sp, -cy * cp);
        let right = glam::Vec3::new(cy, 0.0, sy);

        if self.keys.contains(&KeyCode::KeyW) {
            self.cam_pos += forward * SPEED * dt;
        }
        if self.keys.contains(&KeyCode::KeyS) {
            self.cam_pos -= forward * SPEED * dt;
        }
        if self.keys.contains(&KeyCode::KeyA) {
            self.cam_pos -= right * SPEED * dt;
        }
        if self.keys.contains(&KeyCode::KeyD) {
            self.cam_pos += right * SPEED * dt;
        }
        if self.keys.contains(&KeyCode::Space) {
            self.cam_pos += glam::Vec3::Y * SPEED * dt;
        }
        if self.keys.contains(&KeyCode::ShiftLeft) {
            self.cam_pos -= glam::Vec3::Y * SPEED * dt;
        }

        let size = self.window.inner_size();
        let aspect = size.width as f32 / size.height.max(1) as f32;
        let camera = Camera::perspective_look_at(
            self.cam_pos,
            self.cam_pos + forward,
            glam::Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            aspect,
            0.1,
            250.0,
        );

        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            _ => return,
        };
        let view = output.texture.create_view(&Default::default());

        v3_demo_common::flush_scene_db(&self.scene_db, &self.queue);
        if let Err(e) = self.renderer.render(&camera, &view) {
            log::error!("Render: {:?}", e);
            self.render_errors += 1;
        }
        self.queue.present(output);
        self.frames += 1;
        if self.last_title.elapsed() >= std::time::Duration::from_millis(250) {
            self.update_title();
        }
        if self.frames == 1 {
            println!(
                "[Museum] First frame in {:.3}s; {} objects, loader active={}",
                self.started.elapsed().as_secs_f32(),
                self.loaded_objects,
                self.loader.is_some()
            );
        }
    }
}
