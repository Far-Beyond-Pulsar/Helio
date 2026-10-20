//! Foliage growing on the voxel planet through the whole default graph: the placement
//! pass asks the planet pass's ground (`chunks::ground`) and the blades it keeps stand on
//! the terrain the CPU planet has, on the ground their types allow, gone where it is
//! dug, still on a still camera.
use std::sync::{Arc, Mutex};

use glam::{DVec3, Vec3};
use helio::{required_experimental_features, required_wgpu_features, required_wgpu_limits, Camera, RendererBuilder, RendererConfig};
use helio_default_graphs::{build_default_graph_external_with_voxel_passes, VoxelPassFactory};
use helio_pass_foliage_place::components::FoliageTypeComponent;
use helio_pass_foliage_place::{
    pack_kind_and_flags, unpack_blade, FoliageKind, FoliagePlacePass, GpuBladeInstance, GpuFoliageTile, GpuFoliageType, GpuTileFrame, TileState,
};
use helio_pass_voxel_planet::chunks::pass::ChunkPass;
use helio_pass_voxel_planet::engine::{PlanetFrame, SharedPlanetFrame};
use helio_pass_voxel_planet::terrain::material;
use helio_pass_voxel_planet::{Brush, BrushOp, BrushShape, Cell, Planet, PlanetRecipe, RayHit};
use pulsar_scenedb::gpu::{EngineGpuContext, GpuMirrorHandle, SceneGpuConfig, SceneGpuStore};

const SIZE: [u32; 2] = [640, 360];

struct Harness {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    renderer: helio::Renderer,
    scene: pulsar_scenedb::SceneDb,
    source: SharedPlanetFrame,
    view: wgpu::TextureView,
}

/// A blade as placed: its type and its root in world (planet) coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Blade {
    kind: u8,
    root: DVec3,
}

impl Harness {
    fn new() -> Option<Self> {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
            let adapter = instance.request_adapter(&Default::default()).await.ok()?;
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    required_features: required_wgpu_features(adapter.features()),
                    required_limits: required_wgpu_limits(adapter.limits()),
                    experimental_features: required_experimental_features(adapter.features()),
                    ..Default::default()
                })
                .await
                .ok()?;
            let (device, queue) = (Arc::new(device), Arc::new(queue));
            let mut scene = pulsar_scenedb::SceneDb::new();
            let mut store = SceneGpuStore::new(&EngineGpuContext::new(device.clone(), queue.clone()), SceneGpuConfig { classes: Vec::new(), tombstone_headroom: 0, max_cells_metadata: 0 });
            helio_pass_sky::AtmosphereComponent::register_gpu_columns_growable(&mut store, 4, &device);
            helio_pass_gbuffer::MeshComponent::register_gpu_columns_growable(&mut store, 16, &device);
            helio_pass_gbuffer::MaterialComponent::register_gpu_columns_growable(&mut store, 16, &device);
            helio_pass_gbuffer::StaticObjectComponent::register_gpu_columns_growable(&mut store, 16, &device);
            helio_pass_forward_lit::LightComponent::register_gpu_columns_growable(&mut store, 16, &device);
            scene.world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::new(store), queue.clone()));
            // Grass on grass, and a type keyed to sand (never seen on grass).
            for (ground, density) in [(material::GRASS, 60.0), (material::SAND, 60.0)] {
                let t = GpuFoliageType {
                    density,
                    height_range: [0.1, 0.3],
                    width_range: [0.02, 0.03],
                    slope_range: [-1.0, 1.0],
                    lod_distances: [6.0, 12.0, 18.0, 22.0],
                    kind_and_flags: pack_kind_and_flags(FoliageKind::Block, 0),
                    ground_mask: 1 << ground,
                    ..GpuFoliageType::default()
                };
                let e = scene.world.spawn();
                scene.world.insert(e, FoliageTypeComponent::from(t));
            }
            let source: SharedPlanetFrame = Arc::new(Mutex::new(None));
            let pass_source = source.clone();
            let factory: VoxelPassFactory = Arc::new(move |_, _, _, _| Box::new(ChunkPass::new(pass_source.clone())));
            let mut config = RendererConfig::new(SIZE[0], SIZE[1], wgpu::TextureFormat::Rgba8Unorm);
            config.enable_foliage = true;
            config.foliage_blades_per_m2 = Some(120.0);
            config.foliage_ring_radius = Some(24.0);
            let mirror = scene.world.gpu_mirror().cloned().unwrap();
            let renderer = RendererBuilder::new(config, mirror)
                .with_external_device()
                .with_pass_build_context(Box::new(move |ctx| build_default_graph_external_with_voxel_passes(ctx, vec![factory.clone()])))
                .build(device.clone(), queue.clone(), SIZE[0], SIZE[1], config.surface_format);
            let target = device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d { width: SIZE[0], height: SIZE[1], depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: config.surface_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = target.create_view(&Default::default());
            Some(Self { device, queue, renderer, scene, source, view })
        })
    }

    /// `frames` frames of `planet` seen from `eye` looking along `forward`.
    fn run(&mut self, planet: &Arc<Planet>, eye: DVec3, forward: DVec3, frames: usize) {
        self.renderer.set_world_origin(Some(eye));
        let up = eye.normalize().as_vec3();
        let camera = Camera::perspective_look_at(Vec3::ZERO, forward.as_vec3(), up, std::f32::consts::FRAC_PI_4, SIZE[0] as f32 / SIZE[1] as f32, 0.05, 40_000_000.0);
        *self.source.lock().unwrap() = Some(PlanetFrame { eye, planet: planet.clone(), sun: up, picks: None, kept: None, export: None });
        for _ in 0..frames {
            self.scene.world.flush_gpu_mirror(&self.queue);
            self.renderer.render(&camera, &self.view).unwrap();
            self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        }
    }

    /// Until the planet pass is settled (at most `limit` frames), then `extra` more.
    fn settle(&mut self, planet: &Arc<Planet>, eye: DVec3, forward: DVec3, limit: usize, extra: usize) {
        for _ in 0..limit {
            self.run(planet, eye, forward, 1);
            if self.renderer.find_pass::<ChunkPass>().unwrap().settled() {
                break;
            }
        }
        self.run(planet, eye, forward, extra);
    }

    fn read(&self, buffer: &wgpu::Buffer) -> Vec<u8> {
        let size = buffer.size();
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        self.queue.submit([encoder.finish()]);
        staging.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let data = staging.slice(..).get_mapped_range().unwrap().to_vec();
        data
    }

    /// Every resident tile's blades, at world origin `origin`; and the resident tiles.
    fn blades(&self, origin: DVec3) -> (Vec<Blade>, usize) {
        let place = self.renderer.find_pass::<FoliagePlacePass>().unwrap();
        let tiles: Vec<GpuFoliageTile> = bytemuck::cast_slice(&self.read(&place.tile_table)).to_vec();
        let frames: Vec<GpuTileFrame> = bytemuck::cast_slice(&self.read(&place.tile_frames)).to_vec();
        let arena: Vec<GpuBladeInstance> = bytemuck::cast_slice(&self.read(&place.blade_arena)).to_vec();
        let mut out = Vec::new();
        let mut resident = 0;
        for (tile, frame) in tiles.iter().zip(&frames) {
            if tile.state() != Some(TileState::Resident) {
                continue;
            }
            resident += 1;
            for blade in &arena[tile.blade_offset as usize..(tile.blade_offset + tile.blade_count) as usize] {
                let p = unpack_blade(blade);
                let root = frame.point(p.tile_uv[0], p.tile_uv[1], tile.bounds_center + p.height_offset);
                out.push(Blade { kind: p.type_id, root: origin + root.as_dvec3() });
            }
        }
        (out, resident)
    }
}

/// Grassland: a land column whose surroundings are grass.
fn grassland(planet: &Planet) -> DVec3 {
    let grid = *planet.grid();
    let n = f64::from(grid.cells());
    for face in 0..6u8 {
        for t in 0..400 {
            let (a, b) = (0.3 + 0.4 * f64::from(t % 20) / 20.0, 0.3 + 0.4 * f64::from(t / 20) / 20.0);
            let (i, j) = ((a * n) as i32, (b * n) as i32);
            let grass = |di: i32, dj: i32| {
                let top = planet.terrain_top(face, i + di, j + dj);
                top > 20 && planet.terrain_material(Cell::new(face, i + di, j + dj, top - 1)) == material::GRASS
            };
            if (-12..=12).all(|d| grass(d * 10, 0) && grass(0, d * 10) && grass(d * 7, d * 7)) {
                return grid.direction(face, f64::from(i) + 0.5, f64::from(j) + 0.5);
            }
        }
    }
    panic!("no grassland");
}

/// Metres a root may lie from a voxel step's edge and stand on either column: the
/// probe locates cells by its tile's linearised cube-sphere map, the check by f64 rays.
const EDGE: f64 = 5.0e-5;

/// The surface under `root` (down its radial up), if it is within `tolerance` and faces up.
fn on_surface(planet: &Planet, root: DVec3, tolerance: f64) -> Option<RayHit> {
    let up = root.normalize();
    let hit = planet.raycast(root + up * 0.55, -up, 2.0)?;
    let miss = (0.55 - hit.distance).abs();
    (miss <= tolerance && hit.normal.dot(up) >= 0.5).then_some(hit)
}

/// Blades within `near` of `eye` against the CPU planet: (blades checked, roots off the
/// surface by more than `tolerance` m, on ground other than grass, of the sand type).
/// A root within [`EDGE`] of a step's edge stands on whichever column it touches.
fn check(planet: &Planet, blades: &[Blade], eye: DVec3, near: f64, tolerance: f64) -> (usize, usize, usize, usize) {
    let (mut checked, mut off, mut wrong_ground, mut sand, mut edges) = (0, 0, 0, 0, 0);
    for b in blades {
        if (b.root - eye).length() > near {
            continue;
        }
        checked += 1;
        sand += usize::from(b.kind == 1);
        let up = b.root.normalize();
        let side = up.any_orthonormal_vector();
        let hit = on_surface(planet, b.root, tolerance).or_else(|| {
            let other = up.cross(side);
            let edge = [side, -side, other, -other].into_iter().find_map(|d| on_surface(planet, b.root + d * EDGE, tolerance));
            edges += usize::from(edge.is_some());
            edge
        });
        let Some(hit) = hit else {
            off += 1;
            if off < 6 {
                let found = planet.raycast(b.root + up * 0.55, -up, 2.0).map(|h| 0.55 - h.distance);
                eprintln!("blade at {:?}: surface {found:?} m below", b.root);
            }
            continue;
        };
        wrong_ground += usize::from(planet.material(hit.cell) != material::GRASS);
    }
    eprintln!("{edges} roots on a step's edge");
    (checked, off, wrong_ground, sand)
}

#[test]
fn foliage_grows_on_the_voxel_ground() {
    let Some(mut h) = Harness::new() else {
        eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: foliage on voxels");
        return;
    };
    let mut planet = Planet::new(PlanetRecipe::default()).unwrap();
    let dir = grassland(&planet);
    let eye = planet.surface_point(dir, 1.7);
    let up = eye.normalize();
    let forward = (up.any_orthonormal_vector() - up * 0.6).normalize();
    let shared = Arc::new(planet.clone());
    h.settle(&shared, eye, forward, 600, 120);
    let (blades, resident) = h.blades(eye);
    // Within the level-0 range of the view (cells of a pixel or more) the ground is the
    // CPU's exactly; further, a coarser level's surface (a cell's height).
    let near = 0.7 * 0.1 / (2.0 * (std::f64::consts::FRAC_PI_8).tan() / f64::from(SIZE[1]));
    let (checked, off, wrong_ground, sand) = check(&planet, &blades, eye, near, 0.002);
    let empty_near = blades.is_empty();
    eprintln!("{} blades on {resident} tiles; within {near:.1} m: {checked} checked, {off} off the surface, {wrong_ground} not on grass, {sand} of the sand type", blades.len());
    assert!(!empty_near && checked > 2000, "{checked} blades near the eye");
    assert_eq!(off, 0, "blades off the surface");
    assert_eq!(wrong_ground + sand, 0, "blades off their ground");
    // Every tile around the eye holds blades (grassland all round).
    let coverage = {
        let mut cells = std::collections::HashSet::new();
        for b in &blades {
            let d = b.root - eye;
            if d.length() < 12.0 {
                let side = up.any_orthonormal_vector();
                let other = up.cross(side);
                cells.insert(((d.dot(side) / 2.0).floor() as i32, (d.dot(other) / 2.0).floor() as i32));
            }
        }
        cells.len()
    };
    eprintln!("2 m squares with blades within 12 m: {coverage}");
    assert!(coverage > 100, "bare ground near the eye: {coverage} squares with blades");

    // Still: the same blades frame after frame.
    h.run(&shared, eye, forward, 20);
    let (again, _) = h.blades(eye);
    assert_eq!(again, blades, "blades changed under a still camera");

    // A dig 3 m in front: the blades over it go within a few frames of its chunks.
    let ground = planet.surface_point(dir, 0.0) + (forward - up * forward.dot(up)).normalize() * 3.0;
    let radius = 1.2;
    planet.apply(Brush { center: ground.to_array(), radius, shape: BrushShape::Sphere, op: BrushOp::Remove, material: 0, height: 0.0 }).unwrap();
    let dug = Arc::new(planet.clone());
    let over = |blades: &[Blade]| blades.iter().filter(|b| (b.root - ground).length() < radius - 0.15 && (b.root - ground).dot(up) > -0.3).count();
    assert!(over(&blades) > 20, "{} blades over the dig site before it", over(&blades));
    // Frames until nothing stands over the hole, and until every blade near stands on
    // the dug ground again (the crater's rim re-placed).
    let (mut gone, mut settled) = (None, None);
    for frame in 1..=60 {
        h.run(&dug, eye, forward, 1);
        let (now, _) = h.blades(eye);
        if gone.is_none() && over(&now) == 0 {
            gone = Some(frame);
        }
        let (checked, off, wrong_ground, _) = check(&planet, &now, eye, near, 0.002);
        eprintln!("dig, frame {frame}: {} over the hole; {checked} checked, {off} off, {wrong_ground} off grass", over(&now));
        if gone.is_some() && off + wrong_ground == 0 {
            settled = Some(frame);
            break;
        }
    }
    eprintln!("dig: blades over it gone after {gone:?} frames, all on the ground after {settled:?}");
    assert!(gone.is_some_and(|f| f <= 8), "blades over the dig took {gone:?} frames to go");
    assert!(settled.is_some_and(|f| f <= 8), "blades off the dug ground for {settled:?} frames");
}
