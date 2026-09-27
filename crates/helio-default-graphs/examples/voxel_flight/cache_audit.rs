//! Replay capture disagreements against independent f64 grid-plane intervals.
//! Midpoints decide cell occupancy; no epsilon step can jump a narrow interval.
//! This diagnoses the differing samples, not every ray in a reference image.
use super::*;

fn word(bytes: &[u8], i: usize) -> u32 {
    u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap())
}
fn geometry(bytes: &[u8]) -> ([i32; 3], u32, u32, u32) {
    (
        std::array::from_fn(|a| word(bytes, a) as i32),
        word(bytes, 3) & 3,
        (word(bytes, 3) >> 8) & 3,
        (word(bytes, 3) >> 28) & 7,
    )
}

fn oracle(
    world: &World,
    anchor: [i32; 3],
    fraction: [f64; 3],
    ray: [f64; 3],
    maximum: f64,
) -> ([i32; 3], u32, u32, u32) {
    let step = i32::try_from(world.voxel_step()).unwrap();
    // Sort in canonical units. Multiplying each numerator by 0.1 before
    // division can spuriously split an exactly coincident edge in f64 too.
    let mut planes = vec![0.0, maximum * 10.0];
    for a in 0..3 {
        if ray[a] == 0.0 {
            continue;
        }
        let start = f64::from(anchor[a]) + fraction[a];
        let end = start + ray[a] * maximum * 10.0;
        let low = (start.min(end).floor() as i32).div_euclid(step) * step;
        let high = (start.max(end).ceil() as i32).div_euclid(step) * step;
        for plane in (low..=high).step_by(step as usize) {
            let t = (f64::from(plane - anchor[a]) - fraction[a]) / ray[a];
            if t > 0.0 && t < maximum * 10.0 {
                planes.push(t);
            }
        }
    }
    planes.sort_by(f64::total_cmp);
    planes.dedup();
    for interval in planes.windows(2) {
        let t = interval[0] + (interval[1] - interval[0]) * 0.5;
        let low: [i32; 3] = std::array::from_fn(|a| {
            // Split the integer address before the floating operation.
            let local = fraction[a] + ray[a] * t;
            (anchor[a] + local.floor() as i32).div_euclid(step) * step
        });
        let cell = world.sample_cell(low);
        let material = world.material(cell);
        if material == 0 {
            continue;
        }
        let near: [f64; 3] = std::array::from_fn(|a| {
            if ray[a] == 0.0 {
                return f64::NEG_INFINITY;
            }
            let edge = low[a] + if ray[a] < 0.0 { step } else { 0 };
            (f64::from(edge - anchor[a]) - fraction[a]) / ray[a]
        });
        let mut axis = 0;
        for a in 1..3 {
            if near[a] > near[axis] {
                axis = a;
            }
        }
        let face = 1 + axis as u32 * 2 + u32::from(ray[axis] > 0.0);
        return (cell, 1, material, face);
    }
    ([0; 3], 0, 0, 0)
}

pub fn run(control: &Path, candidate: &Path, output: &Path) {
    let poses = fs::read_to_string(control.join("poses.csv")).unwrap();
    assert_eq!(
        poses,
        fs::read_to_string(candidate.join("poses.csv")).unwrap()
    );
    fs::create_dir_all(output).unwrap();
    let mut report = fs::File::create(output.join("differing-rays.txt")).unwrap();
    let mut total = [0usize; 4]; // diagnosed, control wrong, candidate wrong, both wrong
    for line in poses.lines().skip(1) {
        let fields: Vec<_> = line.split(',').collect();
        let name = fields[0];
        let (case, light_move) = name.rsplit_once("-light").unwrap();
        let movement: usize = light_move.rsplit_once("-move").unwrap().1.parse().unwrap();
        let mut initial = World::default();
        initial.set_voxel_size(fields[10].parse().unwrap()).unwrap();
        let (_, world, eye, _) = surface_reference::cases(initial)
            .into_iter()
            .find(|r| r.0 == case)
            .unwrap();
        let eye = eye + DVec3::X * (movement as f64 * 0.025);
        for a in 0..3 {
            assert!((eye[a] - fields[a + 1].parse::<f64>().unwrap()).abs() < 1e-8);
        }
        let anchor = render_origin(eye);
        let fraction =
            std::array::from_fn(|a| f64::from((eye[a] / 0.1 - f64::from(anchor[a])) as f32));
        let a = fs::read(control.join(format!("{name}.samples.bin"))).unwrap();
        let b = fs::read(candidate.join(format!("{name}.samples.bin"))).unwrap();
        assert_eq!(a.len(), b.len());
        let samples: usize = fields[11].parse().unwrap();
        let pixels = a.len() / (samples * 80);
        assert_eq!(a.len(), samples * pixels * 80);
        let mut counts = [0usize; 4];
        for sample in 0..samples {
            for pixel in 0..pixels {
                let light = sample * pixels * 80 + pixel * 48;
                let offset = sample * pixels * 80 + pixels * 48 + pixel * 32;
                let (ha, hb) = (&a[offset..offset + 32], &b[offset..offset + 32]);
                assert_eq!(&ha[16..28], &hb[16..28], "ray bits differ");
                let (ga, gb) = (geometry(ha), geometry(hb));
                if ga == gb && a[light..light + 12] == b[light..light + 12] {
                    continue;
                }
                let ray = std::array::from_fn(|i| f64::from(f32::from_bits(word(ha, i + 4))));
                let distance = f64::from(f32::from_bits(word(ha, 7)))
                    .max(f64::from(f32::from_bits(word(hb, 7))));
                assert!(
                    distance.is_finite() && distance < 300.0,
                    "offline interval audit supports local patch rays below 300 m"
                );
                let reference = oracle(
                    &world,
                    anchor,
                    fraction,
                    ray,
                    distance + 2.0 * world.voxel_size(),
                );
                let wrong_a = ga != reference;
                let wrong_b = gb != reference;
                counts[0] += 1;
                counts[1] += usize::from(wrong_a);
                counts[2] += usize::from(wrong_b);
                counts[3] += usize::from(wrong_a && wrong_b);
                writeln!(report, "{name} sample={sample} pixel={pixel} ray={ray:?} control={ga:?} candidate={gb:?} oracle={reference:?} control_wrong={wrong_a} candidate_wrong={wrong_b}").unwrap();
            }
        }
        for i in 0..4 {
            total[i] += counts[i];
        }
        eprintln!("VOXEL_CACHE_AUDIT {name} diagnosed={} control_wrong={} candidate_wrong={} both_wrong={}", counts[0], counts[1], counts[2], counts[3]);
    }
    eprintln!(
        "VOXEL_CACHE_AUDIT_TOTAL diagnosed={} control_wrong={} candidate_wrong={} both_wrong={}",
        total[0], total[1], total[2], total[3]
    );
}

/// Independently replay every geometry disagreement in a recorded ground walk.
/// Reconstruct the exact benchmark eye expression; rays come from the GPU.
pub fn run_mesh_motion(control: &Path, candidate: &Path, output: &Path) {
    let metadata: serde_json::Value =
        serde_json::from_slice(&fs::read(control.join("benchmark.json")).unwrap()).unwrap();
    assert_eq!(metadata["fixture"], "ground-close");
    assert_eq!(metadata["recording"], true);
    assert_eq!(metadata["fixed_jitter"], true);
    let other: serde_json::Value =
        serde_json::from_slice(&fs::read(candidate.join("benchmark.json")).unwrap()).unwrap();
    for key in [
        "fixture",
        "recording",
        "fixed_jitter",
        "size",
        "voxel_size_m",
        "frames_per_stage",
    ] {
        assert_eq!(metadata[key], other[key]);
    }
    let mut world = World::default();
    world
        .set_voxel_size(metadata["voxel_size_m"].as_f64().unwrap())
        .unwrap();
    let eye = world.ground_spawn(0.0, 0.0, 3.0);
    let forward = Vec3::new(0.0, -0.8, -1.0).normalize();
    let right = forward.cross(Vec3::Y).normalize();
    let mut report = std::io::BufWriter::new(fs::File::create(output).unwrap());
    let mut total = [0usize; 4];
    for frame in 0..180 {
        let phase = frame as f64 / 179.0 * std::f64::consts::TAU;
        let offset =
            right.as_dvec3() * (phase.sin() * 0.5) + DVec3::Y * ((phase * 2.0).sin() * 0.1);
        let position = eye + offset;
        let anchor = render_origin(position);
        let fraction =
            std::array::from_fn(|a| f64::from((position[a] / 0.1 - f64::from(anchor[a])) as f32));
        let name = format!("motion-{frame:03}.hits.bin");
        let a = fs::read(control.join(&name)).unwrap();
        let b = fs::read(candidate.join(&name)).unwrap();
        assert_eq!(a.len(), b.len());
        for (pixel, (ha, hb)) in a.chunks_exact(32).zip(b.chunks_exact(32)).enumerate() {
            assert_eq!(&ha[16..28], &hb[16..28], "ray bits differ");
            let (ga, gb) = (geometry(ha), geometry(hb));
            if ga == gb {
                continue;
            }
            let ray = std::array::from_fn(|i| f64::from(f32::from_bits(word(ha, i + 4))));
            let distance =
                f64::from(f32::from_bits(word(ha, 7))).max(f64::from(f32::from_bits(word(hb, 7))));
            assert!(distance.is_finite() && distance < 300.0);
            let reference = oracle(
                &world,
                anchor,
                fraction,
                ray,
                distance + 2.0 * world.voxel_size(),
            );
            let wrong_a = ga != reference;
            let wrong_b = gb != reference;
            total[0] += 1;
            total[1] += usize::from(wrong_a);
            total[2] += usize::from(wrong_b);
            total[3] += usize::from(wrong_a && wrong_b);
            writeln!(report, "frame={frame} pixel={pixel} anchor={anchor:?} fraction={fraction:?} ray={ray:?} control={ga:?} candidate={gb:?} oracle={reference:?} control_wrong={wrong_a} candidate_wrong={wrong_b}").unwrap();
        }
    }
    eprintln!(
        "VOXEL_MESH_MOTION_ORACLE diagnosed={} control_wrong={} candidate_wrong={} both_wrong={}",
        total[0], total[1], total[2], total[3]
    );
}
