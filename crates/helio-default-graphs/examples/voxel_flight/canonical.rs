//! Sparse exact ray oracle for fidelity diagnosis, independent of the resident
//! density approximation. A grid of single rays is not a supersampled image or
//! a silhouette/coverage acceptance test. Preserve raw differences, not a pass
//! flag that could disguise incorrect far geometry.
use glam::DVec3;
use helio_pass_tiny_voxel::engine::EngineVoxelFrame;
use std::{fs::File, io::Write, path::Path, time::Instant};

pub(super) fn requested(path: &Path) -> bool {
    std::env::var_os("HELIO_VOXEL_FLIGHT_CANONICAL").is_some()
        && matches!(
            path.file_stem().and_then(|s| s.to_str()),
            Some("200m" | "1km" | "orbit" | "returned-ground" | "orbital-edit" | "1m-base")
        )
}

/// Returns whether sampled coverage, first-cell and material all agree.
pub(super) fn save(hits: &[u8], size: [u32; 2], frame: &EngineVoxelFrame, path: &Path) -> bool {
    assert_eq!(hits.len(), size[0] as usize * size[1] as usize * 32);
    let start = Instant::now();
    let p = &frame.params;
    // Match the actual uploaded split origin, including its f32 fraction.
    let origin = DVec3::from_array(std::array::from_fn(|a| {
        (f64::from(p.origin[a]) + f64::from(p.fraction[a])) * 0.1
    }));
    let output = path.with_extension("canonical.csv");
    let mut csv = File::create(&output).unwrap();
    writeln!(csv, "x,y,width,height,voxel_size,origin_x,origin_y,origin_z,ray_x,ray_y,ray_z,gpu_status,gpu_level,gpu_x,gpu_y,gpu_z,gpu_material,canonical_at_gpu,gpu_distance,cpu_hit,cpu_x,cpu_y,cpu_z,cpu_material,cpu_distance,depth_error_m,pixel_footprint_m,depth_error_footprints,oracle_ms").unwrap();
    let mut mismatched_coverage = 0;
    let mut wrong_cell = 0;
    let mut wrong_material = 0;
    let mut far_surface_in_air = 0;
    eprintln!("VOXEL_CANONICAL_BEGIN capture={} rays=144", path.display());
    for gy in 0..9 {
        for gx in 0..16 {
            let x = ((2 * gx + 1) * size[0] / 32).min(size[0] - 1);
            let y = ((2 * gy + 1) * size[1] / 18).min(size[1] - 1);
            let offset = (x + y * size[0]) as usize * 32;
            let hit = &hits[offset..offset + 32];
            let uint =
                |word: usize| u32::from_le_bytes(hit[word * 4..word * 4 + 4].try_into().unwrap());
            let cell = [uint(0) as i32, uint(1) as i32, uint(2) as i32];
            let status = uint(3) & 3;
            assert!(status < 2, "invalid primary ray during canonical audit");
            let level = (uint(3) >> 2) & 31;
            let material = (uint(3) >> 8) & 3;
            // Hit.normal stores the original primary ray, not a reconstructed
            // face normal. Preserve its direction bits across both tracers.
            let ray = DVec3::from_array(std::array::from_fn(|a| {
                f64::from(f32::from_bits(uint(a + 4)))
            }));
            let gpu_distance = f64::from(f32::from_bits(uint(7))) * ray.length();
            let ray_start = Instant::now();
            let reference = frame.world.raycast(origin, ray, f64::from(p.settings[0]));
            let oracle_ms = ray_start.elapsed().as_secs_f64() * 1000.0;
            let (cpu_cell, cpu_distance) = reference.map_or(([0; 3], f64::NAN), |h| (h.0, h.2));
            let cpu_material = reference.map_or(0, |h| frame.world.material(h.0));
            let canonical_at_gpu = if status == 1 {
                frame.world.material(cell)
            } else {
                0
            };
            let depth_error = if status == 1 && reference.is_some() {
                (gpu_distance - cpu_distance).abs()
            } else {
                f64::NAN
            };
            let footprint = cpu_distance * 2.0 * f64::from(p.up[3]) / f64::from(size[1]);
            mismatched_coverage += usize::from((status == 1) != reference.is_some());
            wrong_cell += usize::from(status == 1 && reference.is_some() && cell != cpu_cell);
            wrong_material += usize::from(status == 1 && reference.is_some() && material != cpu_material);
            far_surface_in_air += usize::from(status == 1 && level > 0 && canonical_at_gpu == 0);
            writeln!(csv, "{x},{y},{},{},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{status},{level},{},{},{},{material},{canonical_at_gpu},{gpu_distance:.9},{},{},{},{},{cpu_material},{cpu_distance:.9},{depth_error:.9},{footprint:.9},{:.9},{oracle_ms:.6}",
                size[0], size[1], frame.world.voxel_size(), origin.x, origin.y, origin.z,
                ray.x, ray.y, ray.z, cell[0], cell[1], cell[2], usize::from(reference.is_some()),
                cpu_cell[0], cpu_cell[1], cpu_cell[2], depth_error / footprint).unwrap();
        }
        csv.flush().unwrap();
    }
    eprintln!("VOXEL_CANONICAL_END capture={} rays=144 coverage_mismatch={mismatched_coverage} different_first_cell={wrong_cell} material_mismatch={wrong_material} far_surface_in_canonical_air={far_surface_in_air} elapsed_ms={:.2}", path.display(), start.elapsed().as_secs_f64() * 1000.0);
    mismatched_coverage == 0 && wrong_cell == 0 && wrong_material == 0 && far_surface_in_air == 0
}
