use helio_pass_tiny_voxel::landforms::VoxelField;

fn hash(c: [i64; 3], seed: u32) -> i128 {
    const MASK: u128 = 0xffff_ffff;
    let mut h = ((c[0] as u32 as u128 * 0x8da6b343)
        ^ (c[1] as u32 as u128 * 0xd8163841)
        ^ (c[2] as u32 as u128 * 0xcb1ab31f)
        ^ u128::from(seed))
        & MASK;
    h = ((h ^ (h >> 16)) * 0x7feb352d) & MASK;
    h = ((h ^ (h >> 15)) * 0x846ca68b) & MASK;
    ((h ^ (h >> 16)) & 65535) as i128
}
fn noise(c: [i32; 3], shift: u32, seed: u32) -> i128 {
    let size = 1i64 << shift;
    let offsets = [
        (0x9e3779b9u128, 0xa341316cu128),
        (0x85ebca6b, 0xc8013ea4),
        (0xc2b2ae35, 0xad90777d),
    ]
    .map(|(a, b)| ((u128::from(seed) * a) ^ b) as u32 as i64 & (size - 1));
    let p = std::array::from_fn::<_, 3, _>(|a| i64::from(c[a]) + offsets[a]);
    let lattice = p.map(|x| x.div_euclid(size));
    let weights = p.map(|x| {
        let t = i128::from(2 * x.rem_euclid(size) + 1) * (65536 / (2 * i128::from(size)));
        (t * t * (3 * 65536 - 2 * t)) / (65536i128.pow(2))
    });
    let mut values: Vec<i128> = (0..8)
        .map(|i| {
            hash(
                [
                    lattice[0] + (i & 1),
                    lattice[1] + ((i >> 1) & 1),
                    lattice[2] + ((i >> 2) & 1),
                ],
                seed,
            )
        })
        .collect();
    for w in weights {
        values = values
            .chunks_exact(2)
            .map(|p| (p[0] * (65536 - w) + p[1] * w).div_euclid(65536))
            .collect();
    }
    values[0]
}

pub fn sample(field: &VoxelField, c: [i32; 3]) -> [u32; 8] {
    if c.iter().any(|x| i64::from(*x).abs() > 100000000) {
        return [0; 8];
    }
    let seed = field.detail_seed();
    let raw = [
        noise(c, 14, seed ^ 73),
        noise(c, 10, seed ^ 191),
        noise(c, 7, seed ^ 311),
    ];
    let detail = ((raw[0] - 32768) * 8000).div_euclid(32768)
        + ((16384 - (raw[1] - 32768).abs()) * 480).div_euclid(32768)
        + ((raw[2] - 32768) * 40).div_euclid(32768);
    // The independent landform sampler was qualified separately; this oracle
    // independently computes the new detail, radius and edit predicates.
    let height = i128::from(
        field
            .landforms()
            .snapshot()
            .sample_cell(c)
            .unwrap()
            .height_units,
    ) + detail;
    let radius = (127420000 + height) as u128;
    let squared = c
        .iter()
        .map(|v| (i128::from(*v) * 2 + 1).unsigned_abs().pow(2))
        .sum::<u128>();
    let mut material = u32::from(squared <= radius * radius);
    let mut source = 0u32;
    // Forward replay, independent of the field's backward early-out query.
    for (i, e) in field.edits().iter().enumerate() {
        let squared = (0..3)
            .map(|a| {
                (i128::from(c[a]) - i128::from(e.cell()[a]))
                    .unsigned_abs()
                    .pow(2)
            })
            .sum::<u128>()
            * 4;
        if squared <= u128::from(e.radius_units()).pow(2) {
            material = e.material();
            source = i as u32 + 1;
        }
    }
    [
        raw[0] as u32,
        raw[1] as u32,
        raw[2] as u32,
        detail as i32 as u32,
        height as i32 as u32,
        material,
        source,
        1,
    ]
}

pub fn public_matches(field: &VoxelField, c: [i32; 3], expected: [u32; 8]) -> bool {
    match (field.height_units(c), field.sample_cell(c)) {
        (None, None) => expected == [0; 8],
        (Some(h), Some(s)) => {
            expected[7] == 1
                && expected[4] == h as u32
                && expected[5] == s.material
                && expected[6] == s.edit_source
        }
        _ => false,
    }
}
