use helio_pass_tiny_voxel::landforms::{
    BoundedLandforms, LandformSnapshot, SnapshotId, VoxelEdit, VoxelField,
};
use sha2::{Digest, Sha256};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};
#[path = "fixtures/voxel_field_device.rs"]
mod gpu;
#[path = "fixtures/voxel_field_oracle.rs"]
mod oracle;

fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn random(seed: &mut u32) -> u32 {
    *seed = seed.wrapping_add(0x9e3779b9);
    let mut h = *seed;
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846ca68b);
    h ^ (h >> 16)
}
fn save(root: &Path, name: &str, bytes: &[u8]) -> serde_json::Value {
    std::fs::write(root.join(name), bytes).unwrap();
    serde_json::json!({"file":name,"bytes":bytes.len(),"sha256":hash(bytes)})
}
fn write(root: &Path, report: &serde_json::Value) {
    std::fs::write(
        root.join("results.json"),
        serde_json::to_vec_pretty(report).unwrap(),
    )
    .unwrap();
}

fn sites(cells: &[[i32; 4]]) -> Vec<[i32; 3]> {
    let mut result: Vec<_> = (0..32)
        .map(|i| {
            let c = cells[514 + i * 2016];
            [c[0], c[1], c[2]]
        })
        .collect();
    assert_eq!(
        result
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        32
    );
    result.extend((0..32).map(|i| [(i - 16) * 1024, 80_000_000, (i % 3 - 1) * 2048]));
    result
}
fn phases(base: Arc<BoundedLandforms>, sites: &[[i32; 3]]) -> Vec<(String, VoxelField)> {
    let empty = VoxelField::new(base, 419);
    let mut edits = Vec::new();
    for c in &sites[..32] {
        for (radius, material) in [(800, 3), (256, 0), (1, 2)] {
            edits.push(VoxelEdit::new(*c, radius, material).unwrap());
        }
    }
    let mut forward = empty.clone();
    for e in &edits {
        forward.push_edit(*e).unwrap();
    }
    let mut reverse = empty.clone();
    for e in edits.iter().rev() {
        reverse.push_edit(*e).unwrap();
    }
    let mut planet = forward.clone();
    planet
        .push_edit(VoxelEdit::new([0; 3], 200_000_000, 0).unwrap())
        .unwrap();
    for c in &sites[32..] {
        planet
            .push_edit(VoxelEdit::new(*c, 64, 3).unwrap())
            .unwrap();
    }
    vec![
        ("unedited".into(), empty),
        ("nested".into(), forward),
        ("reverse".into(), reverse),
        ("planet_delete_then_build".into(), planet),
    ]
}
fn prepare_cells(
    mut cells: Vec<[i32; 4]>,
    sites: &[[i32; 3]],
    field: &VoxelField,
) -> Vec<[i32; 4]> {
    let mut probes = Vec::new();
    for c in sites {
        for axis in 0..3 {
            for offset in [
                -401, -400, -129, -128, -33, -32, -1, 0, 1, 32, 33, 128, 129, 400, 401,
            ] {
                let mut p = *c;
                p[axis] += offset;
                probes.push([p[0], p[1], p[2], 0]);
            }
        }
    }
    // Keep the original invalid inputs. Add true full-field transition probes;
    // position generation uses floats, but all occupancy decisions use integers.
    for i in 0..128 {
        let p = cells[514 + i * 480];
        let v = [p[0] as f64 + 0.5, p[1] as f64 + 0.5, p[2] as f64 + 0.5];
        let length = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        let direction = v.map(|x| x / length);
        let base = field
            .landforms()
            .snapshot()
            .sample_cell([p[0], p[1], p[2]])
            .unwrap()
            .height_units;
        let mut lo = 127420000.0 + f64::from(base) - 20000.0;
        let mut hi = lo + 40000.0;
        for _ in 0..56 {
            let radius = (lo + hi) * 0.5;
            let c = direction.map(|d| ((d * radius - 1.0) * 0.5).floor() as i32);
            if oracle::sample(field, c)[5] != 0 {
                lo = radius;
            } else {
                hi = radius;
            }
        }
        let c = direction.map(|d| ((d * hi - 1.0) * 0.5).floor() as i32);
        for axis in 0..3 {
            for delta in -2..=2 {
                let mut q = c;
                q[axis] += delta;
                probes.push([q[0], q[1], q[2], 0]);
            }
        }
    }
    let start = cells.len() - probes.len();
    cells[start..].copy_from_slice(&probes);
    cells
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Region {
    low: [i32; 4],
    high: [i32; 4],
}
impl Region {
    fn minimum(self) -> [i32; 3] {
        [self.low[0], self.low[1], self.low[2]]
    }
    fn maximum(self) -> [i32; 3] {
        [self.high[0], self.high[1], self.high[2]]
    }
    fn valid(self) -> bool {
        (0..3).all(|a| {
            self.low[a] >= -100000000 && self.high[a] <= 100000000 && self.low[a] <= self.high[a]
        })
    }
}
fn around(c: [i32; 3], edge: i32) -> Region {
    let mut low = [0; 4];
    let mut high = [0; 4];
    for a in 0..3 {
        low[a] = (i64::from(c[a]) - i64::from(edge / 2))
            .clamp(-100000000, 100000001 - i64::from(edge)) as i32;
        high[a] = low[a] + edge - 1;
    }
    Region { low, high }
}
fn regions(cells: &[[i32; 4]], sites: &[[i32; 3]]) -> Vec<Region> {
    let mut result = Vec::with_capacity(2048);
    for i in 0..1024 {
        let c = cells[(i * 61 + 777) % cells.len()];
        result.push(around([c[0], c[1], c[2]], 1 << (i % 4)));
    }
    for i in 0..512 {
        let c = cells[(i * 89 + 319) % cells.len()];
        result.push(around([c[0], c[1], c[2]], 1 << (4 + i % 21)));
    }
    for i in 0..448 {
        let mut c = sites[i % sites.len()];
        if i / 64 % 2 != 0 {
            c[(i / 7) % 3] += [1, 32, 128, 400][i % 4];
        }
        result.push(around(c, [1, 2, 4, 8, 16, 64, 256][i % 7]));
    }
    for i in 0..64 {
        let mut r = around([0; 3], 4);
        let a = i % 3;
        match i % 4 {
            0 => r.low[a] = -100000001,
            1 => r.high[a] = 100000001,
            2 => {
                r.low[a] = 3;
                r.high[a] = 2;
            }
            _ => {
                r.low[a] = i32::MIN;
                r.high[a] = i32::MAX;
            }
        }
        result.push(r);
    }
    assert_eq!(result.len(), 2048);
    result
}
fn certificate(field: &VoxelField, r: Region) -> [u32; 4] {
    match field.classify(r.minimum(), r.maximum()) {
        None => [0; 4],
        Some(b) => [
            b.height_units[0] as u32,
            b.height_units[1] as u32,
            b.classification as u32,
            1,
        ],
    }
}
fn check_regions(field: &VoxelField, regions: &[Region], output: &[[u32; 4]]) -> serde_json::Value {
    let mut checked = 0u64;
    let mut exhaustive = 0u64;
    let mut invalid = 0;
    let mut errors = [0u64; 3];
    let mut distribution = [0u64; 3];
    let mut examples = Vec::new();
    let mut seed = 0x48413419;
    for (i, (r, b)) in regions.iter().zip(output).enumerate() {
        if !r.valid() {
            invalid += 1;
            errors[0] += u64::from(*b != [0; 4]);
            continue;
        }
        if b[3] != 1 || b[2] > 2 {
            errors[0] += 1;
            continue;
        }
        distribution[b[2] as usize] += 1;
        let side = r.high[0] - r.low[0] + 1;
        let mut check = |c: [i32; 3]| {
            checked += 1;
            // Public exact samples are independent of the interval algorithm;
            // their new arithmetic has its own wide-integer sample oracle gate.
            let height = field.height_units(c).unwrap();
            let material = field.sample_cell(c).unwrap().material;
            let height_bad = height < (b[0] as i32) || height > (b[1] as i32);
            let material_bad = (b[2] == 1 && material != 0) || (b[2] == 2 && material == 0);
            errors[1] += u64::from(height_bad);
            errors[2] += u64::from(material_bad);
            if (height_bad || material_bad) && examples.len() < 8 {
                examples.push(serde_json::json!({"region":i,"cell":c,"height":height,"material":material,"certificate":b}));
            }
        };
        if side <= 8 {
            for z in r.low[2]..=r.high[2] {
                for y in r.low[1]..=r.high[1] {
                    for x in r.low[0]..=r.high[0] {
                        check([x, y, z]);
                        exhaustive += 1;
                    }
                }
            }
        } else {
            for mask in 0..8 {
                check(std::array::from_fn(|a| {
                    if mask & (1 << a) == 0 {
                        r.low[a]
                    } else {
                        r.high[a]
                    }
                }));
            }
            for _ in 0..32 {
                check(std::array::from_fn(|a| {
                    r.low[a] + (random(&mut seed) % side as u32) as i32
                }));
            }
        }
    }
    serde_json::json!({"checked_cells":checked,"exhaustively_checked_cells":exhaustive,"invalid_regions":invalid,
        "errors_invalid_height_occupancy":errors,"classification_counts_mixed_air_solid":distribution,"examples":examples})
}

#[test]
fn canonical_integer_brush_boundaries_and_validation() {
    let atom = VoxelEdit::new([0; 3], 1, 3).unwrap();
    assert!(atom.contains([0; 3]));
    assert!(!atom.contains([1, 0, 0]));
    let adjacent = VoxelEdit::new([0; 3], 2, 0).unwrap();
    assert!(adjacent.contains([1, 0, 0]));
    assert!(!adjacent.contains([1, 1, 0]));
    let planet = VoxelEdit::new([0; 3], 200000000, 0).unwrap();
    assert!(planet.contains([100000000, 0, 0]));
    assert!(!planet.contains([100000001, 0, 0]));
    assert!(!planet.contains([i32::MIN, 0, 0]));
    for (c, r, m) in [
        ([i32::MIN, 0, 0], 1, 1),
        ([0; 3], 0, 1),
        ([0; 3], 254840001, 1),
        ([0; 3], 1, 4),
    ] {
        assert!(VoxelEdit::new(c, r, m).is_err());
    }
}

#[test]
#[ignore = "new volume, edit-order and certificate qualification; frozen HELIO_VOXEL_FIELD protocol required"]
fn new_voxel_field_matches_integer_oracle_and_edited_regions() {
    let root = PathBuf::from(std::env::var_os("HELIO_VOXEL_FIELD").expect("HELIO_VOXEL_FIELD"));
    assert!(root.join("protocol.json").is_file());
    let data = root.join("data");
    std::fs::create_dir(&data).expect("fresh evidence directory required");
    let previous = root.parent().unwrap().join("landform-sampling");
    let published: serde_json::Value =
        serde_json::from_slice(&std::fs::read(previous.join("results.json")).unwrap()).unwrap();
    assert_eq!(published["completed"], true);
    let started = Instant::now();
    let (gpu, info) = gpu::Device::new();
    let mut report = serde_json::json!({"device":info,"setup_ms":started.elapsed().as_secs_f64()*1000.0,
        "scope":"New volume revision 1, detail seed 419, integer edits and region certificates; no legacy terrain equivalence, engine or visual gate.","cases":[]});
    for source in published["cases"].as_array().unwrap() {
        let name = source["name"].as_str().unwrap();
        let id = source["published"]["id"].as_str().unwrap();
        let identity = SnapshotId(std::array::from_fn(|i| {
            u8::from_str_radix(&id[i * 2..i * 2 + 2], 16).unwrap()
        }));
        let snapshot = LandformSnapshot::load(
            &previous.join("data/published").join(format!("{id}.hlfm")),
            identity,
        )
        .unwrap();
        let input = source["artifacts"]
            .as_array()
            .unwrap()
            .iter()
            .find(|a| a["file"].as_str().unwrap().ends_with("-input.bin"))
            .unwrap();
        let bytes =
            std::fs::read(previous.join("data").join(input["file"].as_str().unwrap())).unwrap();
        assert_eq!(hash(&bytes), input["sha256"].as_str().unwrap());
        let cells: Vec<[i32; 4]> = bytes
            .chunks_exact(16)
            .map(|b| {
                std::array::from_fn(|i| i32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap()))
            })
            .collect();
        let sites = sites(&cells);
        let base = Arc::new(BoundedLandforms::new(snapshot));
        let phases = phases(base, &sites);
        let cells = prepare_cells(cells, &sites, &phases[0].1);
        let regions = regions(&cells, &sites);
        let input_files = [
            save(
                &data,
                &format!("{name}-cells.bin"),
                bytemuck::cast_slice(&cells),
            ),
            save(
                &data,
                &format!("{name}-regions.bin"),
                bytemuck::cast_slice(&regions),
            ),
        ];
        let mut forward_materials = Vec::new();
        for (phase, field) in phases {
            let reference: Vec<_> = cells
                .iter()
                .map(|c| oracle::sample(&field, [c[0], c[1], c[2]]))
                .collect();
            let public_errors = cells
                .iter()
                .zip(&reference)
                .filter(|(c, r)| !oracle::public_matches(&field, [c[0], c[1], c[2]], **r))
                .count();
            let started = Instant::now();
            let raw = gpu.run(&field, false, bytemuck::cast_slice(&cells), cells.len());
            let sample_wall = started.elapsed().as_secs_f64() * 1000.0;
            let sampled: Vec<[u32; 8]> =
                raw.chunks_exact(8).map(|r| r.try_into().unwrap()).collect();
            let sample_errors = reference
                .iter()
                .zip(&sampled)
                .filter(|(a, b)| a != b)
                .count();
            let mut word_errors = [0usize; 8];
            for (a, b) in reference.iter().zip(&sampled) {
                for i in 0..8 {
                    word_errors[i] += usize::from(a[i] != b[i]);
                }
            }
            let sample_examples:Vec<_>=reference.iter().zip(&sampled).enumerate().filter(|(_, (a,b))|a!=b).take(8)
                .map(|(i,(a,b))|serde_json::json!({"index":i,"cell":cells[i],"expected":a,"gpu":b})).collect();
            let overridden = sampled.iter().filter(|r| r[6] != 0).count();
            let mut order_changes = serde_json::Value::Null;
            if phase == "nested" {
                forward_materials = sampled.iter().map(|r| r[5]).collect::<Vec<_>>();
            }
            if phase == "reverse" {
                order_changes = serde_json::json!(sampled
                    .iter()
                    .zip(&forward_materials)
                    .filter(|(r, m)| r[5] != **m)
                    .count());
            }
            let cpu_bounds: Vec<_> = regions.iter().map(|r| certificate(&field, *r)).collect();
            let started = Instant::now();
            let raw = gpu.run(&field, true, bytemuck::cast_slice(&regions), regions.len());
            let bounds_wall = started.elapsed().as_secs_f64() * 1000.0;
            let gpu_bounds: Vec<[u32; 4]> =
                raw.chunks_exact(4).map(|r| r.try_into().unwrap()).collect();
            let bounds_errors = cpu_bounds
                .iter()
                .zip(&gpu_bounds)
                .filter(|(a, b)| a != b)
                .count();
            let bounds_examples:Vec<_>=cpu_bounds.iter().zip(&gpu_bounds).enumerate().filter(|(_, (a,b))|a!=b).take(8)
                .map(|(i,(a,b))|serde_json::json!({"index":i,"minimum":regions[i].minimum(),"maximum":regions[i].maximum(),"cpu":a,"gpu":b})).collect();
            let checks = check_regions(&field, &regions, &gpu_bounds);
            let artifacts: Vec<_> = [
                ("edits", bytemuck::cast_slice(field.edits())),
                ("oracle", bytemuck::cast_slice(&reference)),
                ("gpu", bytemuck::cast_slice(&sampled)),
                ("bounds-cpu", bytemuck::cast_slice(&cpu_bounds)),
                ("bounds-gpu", bytemuck::cast_slice(&gpu_bounds)),
            ]
            .into_iter()
            .map(|(kind, bytes)| save(&data, &format!("{name}-{phase}-{kind}.bin"), bytes))
            .collect();
            let result = serde_json::json!({"dataset":name,"snapshot_id":id,"phase":phase,"edits":field.edits().len(),"sample_records":cells.len(),
                "sample_gpu_errors":sample_errors,"sample_gpu_word_errors":word_errors,"public_cpu_errors":public_errors,"sample_examples":sample_examples,
                "overridden_cells":overridden,"reversed_order_material_changes":order_changes,"regions":regions.len(),"cpu_gpu_bounds_errors":bounds_errors,
                "bounds_examples":bounds_examples,"cell_checks":checks,"inputs":input_files,"artifacts":artifacts,
                "diagnostic_sample_wall_ms":sample_wall,"diagnostic_bounds_wall_ms":bounds_wall});
            report["cases"].as_array_mut().unwrap().push(result);
            write(&root, &report);
            println!("VOXEL_FIELD {name} {phase}: samples={} sample_errors={sample_errors} public_errors={public_errors} bounds_errors={bounds_errors} overridden={overridden} checks={checks}",cells.len());
            assert_eq!(
                sample_errors, 0,
                "GPU sample differs from independent new-field oracle"
            );
            assert_eq!(public_errors, 0, "public CPU field differs from oracle");
            assert_eq!(bounds_errors, 0, "CPU/GPU edited bounds differ");
            assert_eq!(checks["invalid_regions"], 64);
            assert!(checks["errors_invalid_height_occupancy"]
                .as_array()
                .unwrap()
                .iter()
                .all(|v| v == 0));
            assert!(checks["classification_counts_mixed_air_solid"]
                .as_array()
                .unwrap()
                .iter()
                .all(|v| v.as_u64().unwrap() > 0));
            if phase != "unedited" {
                assert!(overridden >= 32);
            }
            if phase == "reverse" {
                assert!(
                    order_changes.as_u64().unwrap() >= 32,
                    "edit-order negative control had no effect"
                );
            }
        }
    }
    report["completed"] = true.into();
    write(&root, &report);
}
