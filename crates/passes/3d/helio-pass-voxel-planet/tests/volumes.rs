//! Volumes (docs/chunk-octree.md#volumes): worlds placed by f64 transforms
//! in one chunk world, against the CPU's casts in each world's own frame
//! (`Volumes`): a turned Earth far from the origin draws as in its own
//! frame, a moon and a ship-sized block draw beside a planet with the
//! nearest hit winning, a moving and spinning volume draws where it is each
//! frame, picks name the volume they hit, and a turned volume is lit as in
//! its own frame.
mod common;
use common::{gpu, Gpu};
use glam::{DQuat, DVec3, Mat4, Vec3};
use helio_pass_voxel_planet::chunks::{Capacity, ChunkWorld, ResidencyConfig, VolumeId};
use helio_pass_voxel_planet::engine::{PickHit, PickRequest, Picks, SharedPicks};
use helio_pass_voxel_planet::grid::Shape;
use helio_pass_voxel_planet::layers::TerrainLayers;
use helio_pass_voxel_planet::planet::Matter;
use helio_pass_voxel_planet::terrain::material;
use helio_pass_voxel_planet::{
    Brush, BrushOp, BrushShape, Cell, Planet, PlanetRecipe, VolumeTransform, Volumes, VoxelVolume,
};
use std::sync::{Arc, Mutex};

const FOV_Y: f32 = std::f32::consts::FRAC_PI_4;

/// As `tests/chunks.rs`'s: parallel tests fit a 12 GB GPU.
const CAPACITY: Capacity = Capacity {
    chunks: 16384,
    detail_pages: 16384,
    footprints: 1024,
    bricks: 65536,
    kept_bytes: 256 << 20,
    counters: false,
};

/// A camera in the world (eye in world coordinates, its axes the world's).
#[derive(Clone, Copy)]
struct Camera {
    eye: DVec3,
    forward: Vec3,
    up: Vec3,
    size: [u32; 2],
}

impl Camera {
    /// A camera at `eye` looking along `forward` in volume frame `t`
    /// (`up` there), placed in the world.
    fn placed(t: &VolumeTransform, eye: DVec3, forward: DVec3, up: DVec3, size: [u32; 2]) -> Self {
        Self {
            eye: t.to_world(eye),
            forward: t.dir_to_world(forward).normalize().as_vec3(),
            up: t.dir_to_world(up).normalize().as_vec3(),
            size,
        }
    }

    fn view_proj(&self) -> Mat4 {
        let up = if self.forward.dot(self.up).abs() > 0.99 {
            self.up.any_orthonormal_vector()
        } else {
            self.up
        };
        let view = glam::camera::rh::view::look_to_mat4(Vec3::ZERO, self.forward, up);
        let proj = helio_core::depth::perspective(
            FOV_Y,
            self.size[0] as f32 / self.size[1] as f32,
            0.05,
        );
        proj * view
    }

    /// The ray through pixel (x, y)'s centre (world).
    fn pixel_dir(&self, view_proj: Mat4, x: u32, y: u32) -> DVec3 {
        let ndc = glam::Vec4::new(
            (x as f32 + 0.5) / self.size[0] as f32 * 2.0 - 1.0,
            1.0 - (y as f32 + 0.5) / self.size[1] as f32 * 2.0,
            0.5,
            1.0,
        );
        let w = view_proj.inverse() * ndc;
        (w.truncate() / w.w).as_dvec3().normalize()
    }

    fn pixel_angle(&self) -> f64 {
        2.0 * f64::from((FOV_Y * 0.5).tan()) / f64::from(self.size[1])
    }

    fn config(&self) -> ResidencyConfig {
        let mut config = ResidencyConfig {
            footprints_per_frame: 512,
            generate_ms: None,
            ..Default::default()
        };
        config.plan.pixel_angle = self.pixel_angle();
        config
    }

    fn cone(&self) -> helio_pass_voxel_planet::chunks::residency::View {
        let tan_y = f64::from((FOV_Y * 0.5).tan());
        let tan_x = tan_y * f64::from(self.size[0]) / f64::from(self.size[1]);
        helio_pass_voxel_planet::chunks::residency::View {
            forward: self.forward.as_dvec3(),
            cos_half: (tan_x * tan_x + tan_y * tan_y).sqrt().atan().cos(),
        }
    }
}

/// A chunk world of several volumes and its frame counter.
struct Scene {
    world: ChunkWorld,
    ids: Vec<VolumeId>,
    frame: u64,
    /// The direction to the sun (world).
    sun: Vec3,
    last: Option<Mat4>,
}

impl Scene {
    fn new(gpu: &Gpu, volumes: &Volumes) -> Self {
        Self::with_capacity(gpu, volumes, CAPACITY)
    }

    fn with_capacity(gpu: &Gpu, volumes: &Volumes, capacity: Capacity) -> Self {
        let mut world = ChunkWorld::with_capacity(&gpu.device, &gpu.queue, capacity);
        let ids = volumes
            .volumes
            .iter()
            .map(|v| world.add_volume(v.id, &v.planet, v.transform))
            .collect();
        Self {
            world,
            ids,
            frame: 0,
            sun: Vec3::new(0.3, 0.8, 0.4).normalize(),
            last: None,
        }
    }

    /// The volumes' transforms now.
    fn place(&mut self, volumes: &Volumes) {
        for (id, v) in self.ids.iter().zip(&volumes.volumes) {
            self.world.set_transform(*id, v.transform);
        }
    }

    /// One frame: residency, the draw, `picks` copied; its residency.
    fn frame(
        &mut self,
        gpu: &Gpu,
        camera: &Camera,
        picks: Option<&SharedPicks>,
    ) -> helio_pass_voxel_planet::chunks::UpdateStats {
        let view_proj = camera.view_proj();
        let prev = self.last.replace(view_proj).unwrap_or(view_proj);
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        let mut recorder = helio_core::CommandRecorder::from_encoder(&mut encoder);
        let stats = self.world.update(
            &mut recorder,
            camera.eye,
            DVec3::ZERO,
            Some(camera.cone()),
            self.frame,
            &camera.config(),
        );
        self.world.draw(
            &mut recorder,
            self.frame,
            camera.eye,
            view_proj,
            prev,
            [glam::Vec2::ZERO; 2],
            0.05,
            self.sun,
            camera.size,
            None,
            None,
            None,
        );
        if let Some(picks) = picks {
            self.world.copy_picks(&mut recorder, picks);
        }
        drop(recorder);
        gpu.queue.submit([encoder.finish()]);
        self.world.submitted();
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        self.frame += 1;
        stats
    }

    /// Frames until residency settles; every pixel's hit then.
    fn settle(&mut self, gpu: &Gpu, camera: &Camera) -> Vec<Option<PickHit>> {
        let started = self.frame;
        loop {
            let stats = self.frame(gpu, camera, None);
            if self.world.settled() {
                break;
            }
            if (self.frame - started) % 500 == 0 {
                eprintln!(
                    "frame {}: volumes (id, tag, resident, drawn, waiting) {:?}; {stats:?}",
                    self.frame - started,
                    self.world.volume_stats()
                );
            }
            assert!(self.frame - started < 4000, "residency did not settle: {stats:?}");
        }
        eprintln!(
            "settled in {} frames: {} chunks drawn",
            self.frame - started,
            self.world.chunk_count()
        );
        self.world.read_hits()
    }
}

/// How drawn hits compare with the CPU's nearest over the volumes: pixels
/// compared, mismatched, and compared by volume (`Volumes::volumes` order).
#[derive(Debug, Default)]
struct Agreement {
    compared: usize,
    mismatched: usize,
    by_volume: Vec<usize>,
    /// Pixels whose drawn hit is in another volume than the CPU's (or none
    /// where it has one, or the reverse), of those checked coarsely.
    other_volume: usize,
    coarse: usize,
}

/// Drawn hits against CPU casts (`Volumes::raycast`, each in its world's
/// frame) at every second pixel. A hit within the level-0 range of its
/// volume (cells of a pixel or more) must be drawn at level 0 in the same
/// volume at the same cell (or its neighbour across an edge the ray runs
/// along) and distance; one past it but within the casts' reach (the
/// farthest volume's level-0 range), in the same volume within a few of the
/// drawn level's cells. Past the casts' reach a drawn hit must lie in the
/// bounds of the volume it names, along the ray.
fn compare(hits: &[Option<PickHit>], volumes: &Volumes, camera: &Camera) -> Agreement {
    let view_proj = camera.view_proj();
    let mut out = Agreement {
        by_volume: vec![0; volumes.volumes.len()],
        ..Default::default()
    };
    let near_of = |v: &VoxelVolume| 0.7 * v.planet.grid().voxel_size() / camera.pixel_angle();
    let reach = volumes.volumes.iter().map(near_of).fold(0.0, f64::max);
    for y in (0..camera.size[1]).step_by(2) {
        for x in (0..camera.size[0]).step_by(2) {
            let dir = camera.pixel_dir(view_proj, x, y);
            let cpu = volumes.raycast(camera.eye, dir, reach, Matter::All);
            let drawn = hits[(y * camera.size[0] + x) as usize];
            let Some(cpu) = cpu else {
                // Past the casts' reach: the named volume's bounds hold the
                // drawn hit.
                out.coarse += 1;
                let plausible = drawn.is_none_or(|h| {
                    h.distance >= reach * 0.99
                        && volumes.volumes.iter().filter(|v| v.id == h.volume).any(|v| {
                            let r = helio_pass_voxel_planet::volume::bounding_radius(&v.planet)
                                .unwrap_or(f64::INFINITY);
                            helio_pass_voxel_planet::volume::ray_ball(
                                camera.eye,
                                dir,
                                v.transform.position,
                                r,
                            )
                            .is_some_and(|(enter, leave)| {
                                h.distance >= enter - 1.0 && h.distance <= leave + 1.0
                            })
                        })
                });
                if !plausible && out.other_volume < 4 {
                    eprintln!("pixel {x},{y}: drawn {drawn:?} past the casts' reach");
                }
                out.other_volume += usize::from(!plausible);
                continue;
            };
            let v = &volumes.volumes[cpu.index];
            let planet = &v.planet;
            let voxel = planet.grid().voxel_size();
            if cpu.hit.distance >= near_of(v) {
                out.coarse += 1;
                let same = drawn.is_some_and(|h| {
                    let cell = voxel * f64::from(1u32 << h.level);
                    h.volume == cpu.id
                        && (h.distance - cpu.hit.distance).abs() <= 4.0 * cell + 1e-4 * cpu.hit.distance
                });
                if !same && out.other_volume < 4 {
                    eprintln!(
                        "pixel {x},{y}: drawn {drawn:?}, cpu volume {} at {} (coarse)",
                        cpu.id, cpu.hit.distance
                    );
                }
                out.other_volume += usize::from(!same);
                continue;
            }
            out.compared += 1;
            out.by_volume[cpu.index] += 1;
            let frame = if planet.grid().is_plane() {
                0.0
            } else {
                4.0 * 512.0 * voxel * voxel / planet.grid().radius()
            };
            // A ray along the edge between two solid cells meets both at
            // once: either is exact.
            let tie = |c: Cell| {
                c.face == cpu.hit.cell.face
                    && (c.i - cpu.hit.cell.i).abs()
                        + (c.j - cpu.hit.cell.j).abs()
                        + (c.k - cpu.hit.cell.k).abs()
                        == 1
                    && planet.solid(c)
            };
            let exact = drawn.is_some_and(|h| {
                h.volume == cpu.id
                    && h.level == 0
                    && (h.cell == cpu.hit.cell || tie(h.cell))
                    && (h.distance - cpu.hit.distance).abs()
                        <= 2e-3 + cpu.hit.distance * 1e-5 + frame
            });
            if !exact {
                out.mismatched += 1;
                if out.mismatched < 8 {
                    eprintln!(
                        "pixel {x},{y}: drawn {drawn:?}, cpu volume {} {:?} at {}",
                        cpu.id, cpu.hit.cell, cpu.hit.distance
                    );
                }
            }
        }
    }
    eprintln!("{out:?}");
    out
}

/// A land direction (surface above sea level) near `face` coordinates.
fn land(planet: &Planet, face: u8, fi: f64, fj: f64) -> DVec3 {
    let grid = planet.grid();
    let n = f64::from(grid.cells());
    for step in 0..400 {
        let a = fi + 0.013 * f64::from(step % 20);
        let b = fj + 0.017 * f64::from(step / 20);
        let (i, j) = ((a.fract() * n) as i32, (b.fract() * n) as i32);
        if planet.column_top(face, i, j, 0) > 40 {
            return grid.direction(face, f64::from(i) + 0.5, f64::from(j) + 0.5);
        }
    }
    panic!("no land found");
}

/// A transform far from the origin, turned about an oblique axis.
fn far_turned() -> VolumeTransform {
    VolumeTransform::new(
        DVec3::new(2.0e7, -1.3e7, 4.1e6),
        DQuat::from_axis_angle(DVec3::new(0.3, 0.8, -0.5).normalize(), 2.1),
    )
}

/// An Earth turned and moved 2.4e7 m from the origin draws its ground view
/// as an Earth at the origin does: level 0 at the same cells and distances
/// (the eye's f64 position, the frame's rotation and the chunks' frames
/// composed without losing the Earth's precision).
#[test]
fn a_turned_earth_far_from_the_origin_renders_exactly() {
    let Some(gpu) = gpu() else { return };
    let planet = Arc::new(Planet::new(PlanetRecipe::default()).unwrap());
    let t = far_turned();
    let dir = land(&planet, 4, 0.37, 0.61);
    let eye = planet.surface_point(dir, 1.7);
    let up = eye.normalize();
    let forward = (up.any_orthonormal_vector() - up * 0.35).normalize();
    let camera = Camera::placed(&t, eye, forward, up, [320, 180]);
    let volumes = Volumes::new(vec![VoxelVolume::new(3, planet, t)]);
    let mut scene = Scene::new(gpu, &volumes);
    let hits = scene.settle(gpu, &camera);
    let a = compare(&hits, &volumes, &camera);
    assert!(a.compared > 1000, "{a:?}");
    assert!(a.mismatched * 1000 <= a.compared, "{a:?}");
}

/// A ship-sized block: a 12.8 m plane of 0.1 m voxels, its ground a slab
/// 1.2 m thick, a cabin 2.4 m across on it.
fn block() -> Planet {
    let mut planet = Planet::new(PlanetRecipe {
        shape: Shape::Plane,
        plane_size_m: 12.8,
        terrain: TerrainLayers::flat_at(1.2).source(3),
        ..Default::default()
    })
    .unwrap();
    planet
        .apply(Brush {
            center: [0.0, 2.4, 0.0],
            radius: 1.2,
            shape: BrushShape::Cube,
            op: BrushOp::Add,
            material: material::PLANKS,
            height: 0.0,
        })
        .unwrap();
    planet
}

/// A moon 600 km across, 0.5 m voxels, the Moon's ground. (A sphere keeps
/// about 1024 cells across a face at its coarsest level, all its roots
/// resident: a 10 km moon with the Moon's kilometres of relief held 30 000
/// root chunks.)
fn moon() -> Planet {
    Planet::new(PlanetRecipe {
        radius_m: 300_000.0,
        voxel_size_m: 0.5,
        terrain: TerrainLayers::moon().source(11),
        ..Default::default()
    })
    .unwrap()
}

/// A planet, a moon high over its horizon and a block hovering a few
/// metres in front of the eye, each placed and turned: every pixel is the
/// nearest volume's hit (the CPU's, in each one's frame), the block's and
/// the ground's exactly.
#[test]
fn a_planet_a_moon_and_a_block_draw_as_one_scene() {
    let Some(gpu) = gpu() else { return };
    let earth = Arc::new(Planet::new(PlanetRecipe::default()).unwrap());
    let dir = land(&earth, 4, 0.37, 0.61);
    let eye = earth.surface_point(dir, 3.0);
    let up = eye.normalize();
    let across = up.any_orthonormal_vector();
    // Looking out over the ground, a little down.
    let forward = (across - up * 0.15).normalize();
    // The moon 1 800 km away, high over the horizon, turned.
    let moon_at = eye + (across + up * 0.35).normalize() * 1_800_000.0;
    let moon = VoxelVolume::new(
        2,
        Arc::new(moon()),
        VolumeTransform::new(moon_at, DQuat::from_rotation_z(0.4) * DQuat::from_rotation_x(1.1)),
    );
    // The block set into the ground 9 m ahead and 5 m to the right, turned
    // and leaning a little: its slab's top, sides and cabin show, and the
    // ground to its left whatever the ground's relief. (A plane's underside
    // is no world's: the CPU's casts clip a plane to its terrain's band, the
    // draw to its root chunks.)
    let right = forward.cross(up).normalize();
    let ground = earth.surface_point(eye + across * 9.0 + right * 5.0, 0.0);
    let block_at = ground - up * 1.0;
    let block = VoxelVolume::new(
        5,
        Arc::new(block()),
        VolumeTransform::new(
            block_at,
            DQuat::from_rotation_arc(DVec3::Y, (up + across * 0.05).normalize())
                * DQuat::from_rotation_y(0.6),
        ),
    );
    let volumes = Volumes::new(vec![
        VoxelVolume::new(1, earth, VolumeTransform::IDENTITY),
        moon,
        block,
    ]);
    let camera = Camera {
        eye,
        forward: forward.as_vec3(),
        up: up.as_vec3(),
        size: [320, 180],
    };
    // The horizon's chunks, the moon's and the block's: twice the slots.
    let mut scene = Scene::with_capacity(
        gpu,
        &volumes,
        Capacity {
            chunks: 2 * CAPACITY.chunks,
            ..CAPACITY
        },
    );
    let hits = scene.settle(gpu, &camera);
    let a = compare(&hits, &volumes, &camera);
    assert!(a.by_volume[0] > 500 && a.by_volume[2] > 500, "{a:?}");
    assert!(a.mismatched * 1000 <= a.compared, "{a:?}");
    // The moon's pixels (and the sky's) are the moon's (none) but at its
    // coarse silhouette.
    let moon_pixels = hits.iter().flatten().filter(|h| h.volume == 2).count();
    assert!(moon_pixels > 400, "{moon_pixels} moon pixels");
    assert!(a.other_volume * 50 <= a.coarse.max(1), "{a:?}");
}

/// A block flying and spinning past a camera that rides with it: each
/// frame its hits are where this frame's transform puts it (no frame of
/// lag), level 0 at its cells; then still, with the camera standing, its
/// hits after each move are its cells at its new place.
#[test]
fn a_moving_spinning_volume_renders_where_it_is_each_frame() {
    let Some(gpu) = gpu() else { return };
    let planet = Arc::new(block());
    // The camera 6 m off the block, looking at its cabin, in its frame.
    let eye = DVec3::new(4.5, 4.5, 4.5);
    let forward = (DVec3::new(0.0, 2.0, 0.0) - eye).normalize();
    let mut volumes = Volumes::new(vec![VoxelVolume::new(9, planet, VolumeTransform::IDENTITY)]);
    let mut scene = Scene::new(gpu, &volumes);
    let at = |frame: u32| {
        let s = f64::from(frame) * 0.05;
        VolumeTransform::new(
            DVec3::new(1.0e6 + 400.0 * s, 25.0 * s.sin(), -3.0e5 - 90.0 * s),
            DQuat::from_rotation_y(s * 1.3) * DQuat::from_rotation_x(s * 0.4),
        )
    };
    // Settled at the first place.
    volumes.volumes[0].transform = at(0);
    scene.place(&volumes);
    let camera = Camera::placed(&at(0), eye, forward, DVec3::Y, [256, 160]);
    let settled = scene.settle(gpu, &camera);
    let first = compare(&settled, &volumes, &camera);
    assert!(first.compared > 2000 && first.mismatched == 0, "{first:?}");
    // Flying: every frame's hits are the CPU's at that frame's place.
    for frame in 1..12 {
        volumes.volumes[0].transform = at(frame);
        scene.place(&volumes);
        let camera = Camera::placed(&at(frame), eye, forward, DVec3::Y, [256, 160]);
        scene.frame(gpu, &camera, None);
        let a = compare(&scene.world.read_hits(), &volumes, &camera);
        assert!(a.compared > 2000 && a.mismatched == 0, "frame {frame}: {a:?}");
    }
    // A standing camera: the block moved and turned under it.
    let standing = Camera::placed(&at(11), eye, forward, DVec3::Y, [256, 160]);
    for frame in [13, 17] {
        // Turned as at `frame`, moved so the camera's point of the block's
        // frame stays where it was: the block in view.
        let mut t = at(frame);
        t.position += standing.eye - t.to_world(eye);
        volumes.volumes[0].transform = t;
        scene.place(&volumes);
        let hits = scene.settle(gpu, &standing);
        let a = compare(&hits, &volumes, &standing);
        assert!(a.compared > 500 && a.mismatched * 200 <= a.compared, "frame {frame}: {a:?}");
    }
}

/// Picks name the volume they hit, at the hit the draw shows.
#[test]
fn picks_name_the_volume_they_hit() {
    let Some(gpu) = gpu() else { return };
    let earth = Arc::new(Planet::new(PlanetRecipe::default()).unwrap());
    let dir = land(&earth, 4, 0.37, 0.61);
    let eye = earth.surface_point(dir, 3.0);
    let up = eye.normalize();
    let across = up.any_orthonormal_vector();
    let forward = (across - up * 0.2).normalize();
    // The block on the ground ahead and to the right, the ground beside it.
    let block_at = eye + across * 14.0 + across.cross(up) * 5.0 - up * 2.5;
    let volumes = Volumes::new(vec![
        VoxelVolume::new(1, earth, VolumeTransform::IDENTITY),
        VoxelVolume::new(
            5,
            Arc::new(block()),
            VolumeTransform::new(block_at, DQuat::from_rotation_arc(DVec3::Y, up)),
        ),
    ]);
    let camera = Camera {
        eye,
        forward: forward.as_vec3(),
        up: up.as_vec3(),
        size: [256, 160],
    };
    let mut scene = Scene::new(gpu, &volumes);
    let hits = scene.settle(gpu, &camera);
    let picks: SharedPicks = Arc::new(Mutex::new(Picks::default()));
    // A pixel each volume shows, by the middle of the most it shows.
    let at: Vec<[f32; 2]> = [1u64, 5]
        .iter()
        .map(|&id| {
            let shown: Vec<usize> = (0..hits.len())
                .filter(|&n| hits[n].is_some_and(|h| h.volume == id))
                .collect();
            assert!(shown.len() > 100, "volume {id} shows {} pixels", shown.len());
            let n = shown[shown.len() / 2];
            [((n % 256) as f32 + 0.5) / 256.0, ((n / 256) as f32 + 0.5) / 160.0]
        })
        .collect();
    picks.lock().unwrap().requests = at
        .iter()
        .enumerate()
        .map(|(n, uv)| PickRequest { id: n as u64, uv: *uv })
        .collect();
    for _ in 0..6 {
        scene.frame(gpu, &camera, Some(&picks));
    }
    let results = picks.lock().unwrap().results.clone();
    assert_eq!(results.len(), at.len(), "{results:?}");
    let mut volumes_hit = std::collections::BTreeSet::new();
    for r in results {
        let [u, v] = at[r.id as usize];
        let (x, y) = (
            ((u * 256.0) as u32).min(255),
            ((v * 160.0) as u32).min(159),
        );
        let drawn = hits[(y * 256 + x) as usize];
        assert_eq!(r.hit.map(|h| (h.volume, h.cell)), drawn.map(|h| (h.volume, h.cell)), "pick {r:?}");
        if let Some(h) = r.hit {
            volumes_hit.insert(h.volume);
        }
    }
    assert_eq!(volumes_hit.into_iter().collect::<Vec<_>>(), vec![1, 5]);
}

/// Channel `channel` of an Rgba16Float target, row by row.
fn read_rgba16(gpu: &Gpu, texture: &wgpu::Texture, channel: usize) -> Vec<f32> {
    let [w, h] = [texture.width(), texture.height()];
    let row = (w * 8).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::from(row * h),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit([encoder.finish()]);
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, |r| r.unwrap());
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    let bytes = staging.slice(..).get_mapped_range().unwrap().to_vec();
    let half = |bits: u16| {
        let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
        let exponent = i32::from((bits >> 10) & 0x1f);
        let mantissa = f32::from(bits & 0x3ff);
        sign * match exponent {
            0 => mantissa / 1024.0 * 2f32.powi(-14),
            31 => f32::INFINITY,
            e => (1.0 + mantissa / 1024.0) * 2f32.powi(e - 15),
        }
    };
    (0..h)
        .flat_map(|y| (0..w).map(move |x| (y, x)))
        .map(|(y, x)| {
            let at = (y * row + x * 8) as usize + channel * 2;
            half(u16::from_le_bytes([bytes[at], bytes[at + 1]]))
        })
        .collect()
}

/// A volume turned and moved is lit as in its own frame: the same ground
/// view of a planet at the origin, sun `s`, and of the planet turned and
/// moved, sun turned with it, shows the same hits, the same sun (steady
/// shadows: every hit's own walks over the sun's disk) and normals turned
/// with it.
#[test]
fn a_turned_volume_is_lit_as_in_its_own_frame() {
    let Some(gpu) = gpu() else { return };
    let planet = Arc::new(Planet::new(PlanetRecipe::default()).unwrap());
    let dir = land(&planet, 2, 0.41, 0.33);
    let eye = planet.surface_point(dir, 2.0);
    let up = eye.normalize();
    let forward = (up.any_orthonormal_vector() - up * 0.4).normalize();
    // A low sun: long shadows.
    let sun = (up * 0.25 + up.any_orthonormal_vector().cross(up)).normalize();
    let mut images = Vec::new();
    for t in [VolumeTransform::IDENTITY, far_turned()] {
        let volumes = Volumes::new(vec![VoxelVolume::new(1, planet.clone(), t)]);
        let camera = Camera::placed(&t, eye, forward, up, [256, 160]);
        let mut scene = Scene::new(gpu, &volumes);
        scene.sun = t.dir_to_world(sun).as_vec3();
        scene.world.set_steady_shadows(true);
        let hits = scene.settle(gpu, &camera);
        let targets = scene.world.targets().unwrap();
        let lit = read_rgba16(gpu, &targets.sun, 0);
        let normals: Vec<DVec3> = (0..3)
            .map(|c| read_rgba16(gpu, &targets.normal, c))
            .collect::<Vec<_>>()
            .chain_normals();
        // Back into the volume's frame.
        let normals: Vec<DVec3> = normals.iter().map(|n| t.dir_to_local(*n)).collect();
        images.push((hits, lit, normals));
    }
    let (a, b) = (&images[0], &images[1]);
    let (mut hits, mut same_hit, mut same_sun, mut same_normal) = (0, 0, 0, 0);
    for i in 0..a.0.len() {
        let (Some(ha), Some(hb)) = (a.0[i], b.0[i]) else {
            continue;
        };
        hits += 1;
        same_hit += usize::from(ha.cell == hb.cell && (ha.distance - hb.distance).abs() < 1e-2);
        same_sun += usize::from((a.1[i] - b.1[i]).abs() < 0.05);
        same_normal += usize::from(a.2[i].dot(b.2[i]) > 0.999);
    }
    eprintln!("hits {hits}: same hit {same_hit}, sun {same_sun}, normal {same_normal}");
    assert!(hits > 10_000);
    assert!(same_hit * 1000 >= hits * 995, "{same_hit}/{hits}");
    assert!(same_sun * 1000 >= hits * 990, "{same_sun}/{hits}");
    assert!(same_normal * 1000 >= hits * 990, "{same_normal}/{hits}");
}

/// Three channels' rows as vectors.
trait ChainNormals {
    fn chain_normals(self) -> Vec<DVec3>;
}

impl ChainNormals for Vec<Vec<f32>> {
    fn chain_normals(self) -> Vec<DVec3> {
        (0..self[0].len())
            .map(|i| DVec3::new(f64::from(self[0][i]), f64::from(self[1][i]), f64::from(self[2][i])))
            .collect()
    }
}
