//! Sky visibility (`skylight`): terrain that sees less of the sky receives
//! less of its light. A trench floor is occluded, open ground is not, and the
//! setting turns the term off.
mod common;
use common::*;
use glam::{DVec3, Vec3};
use helio_pass_voxel_planet::engine::{PlanetRenderer, Settings};
use helio_pass_voxel_planet::{grid::Shape, layers::TerrainLayers, Brush, BrushOp, BrushShape, Planet, PlanetRecipe};
use std::sync::Arc;

/// The GBuffer's occlusion channel (ORM red, Rgba8Unorm) per pixel.
fn occlusion(gpu: &Gpu, target: &Target) -> Vec<f32> {
    let [w, h] = target.size;
    let row = (w * 4).div_ceil(256) * 256;
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::from(row * h),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        target.colors[2].as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) },
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    gpu.queue.submit([encoder.finish()]);
    buffer.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
    gpu.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let data = buffer.slice(..).get_mapped_range().unwrap();
    (0..h as usize)
        .flat_map(|y| (0..w as usize).map(move |x| (y, x)))
        .map(|(y, x)| f32::from(data[y * row as usize + x * 4]) / 255.0)
        .collect()
}

/// Mean occlusion over pixels whose terrain hit lies in `region` (height
/// above the flat ground, horizontal distance from the trench's axis).
fn mean_in(target: &Target, hits: &[Hit], ao: &[f32], eye: DVec3, forward: Vec3, ground_y: f64,
    region: impl Fn(f64, f64) -> bool) -> (f32, usize) {
    let camera = target.camera(forward, Vec3::Y);
    let (mut sum, mut count) = (0.0, 0);
    for y in 0..target.size[1] {
        for x in 0..target.size[0] {
            let index = (y * target.size[0] + x) as usize;
            if hits[index].status != 1 { continue; }
            let point = eye + pixel_dir(target, &camera, x, y) * f64::from(hits[index].t);
            if region(point.y - ground_y, point.x.abs()) {
                sum += ao[index];
                count += 1;
            }
        }
    }
    (sum / count.max(1) as f32, count)
}

#[test]
fn a_trench_floor_sees_less_sky_than_open_ground() {
    let Some(gpu) = gpu() else { return };
    let mut planet = Planet::new(PlanetRecipe {
        shape: Shape::Plane,
        plane_size_m: 1024.0,
        terrain: TerrainLayers::flat().source(7),
        ..Default::default()
    })
    .unwrap();
    let ground = planet.surface_point(DVec3::new(0.0, 0.0, 0.0), 0.0);
    // A trench along z: 3 m wide, 3 m deep, 30 m long.
    let mut z = -15.0;
    while z <= 15.0 {
        planet.apply(Brush { center: (ground + DVec3::new(0.0, -1.5, z)).to_array(), radius: 1.5,
            shape: BrushShape::Cube, op: BrushOp::Remove, material: 0 }).unwrap();
        z += 1.0;
    }
    let planet = Arc::new(planet);
    let size = [256, 192];
    let target = Target::new(&gpu, size);
    let eye = ground + DVec3::new(4.0, 14.0, -2.0);
    let forward = Vec3::new(-0.25, -1.0, 0.1).normalize();
    let mut results = Vec::new();
    for sky_occlusion in [true, false] {
        let settings = Settings { sky_occlusion, ..Settings::default() };
        let mut renderer = PlanetRenderer::new(&gpu.device, &gpu.queue, planet.clone(), settings, size);
        let frame = frame(&planet, eye);
        let mut n = 0;
        while n < 2000 {
            target.render(&gpu, &mut renderer, &frame, forward, n);
            n += 1;
            if renderer.settled() { break; }
        }
        target.render(&gpu, &mut renderer, &frame, forward, n);
        let hits = hits(&gpu, &renderer);
        let ao = occlusion(&gpu, &target);
        let floor = mean_in(&target, &hits, &ao, eye, forward, ground.y, |height, across| height < -2.5 && across < 0.5);
        let open = mean_in(&target, &hits, &ao, eye, forward, ground.y, |height, across| height.abs() < 0.2 && across > 8.0);
        eprintln!("sky occlusion {sky_occlusion}: trench floor {floor:?}, open ground {open:?}");
        assert!(floor.1 > 50 && open.1 > 500, "too few samples: floor {floor:?}, open {open:?}");
        results.push((floor.0, open.0));
    }
    let (with, without) = (results[0], results[1]);
    assert!(with.1 > 0.95, "open ground occluded: {with:?}");
    assert!((with.1 - without.1).abs() < 0.03, "open ground changed: {with:?} vs {without:?}");
    assert!(with.0 < 0.6, "trench floor sees the open sky: {with:?}");
    assert!(with.0 < without.0 - 0.2, "sky occlusion did not darken the trench: {with:?} vs {without:?}");
}
