use super::*;
use crate::landforms::FaceAddress;

fn recipe() -> LandformRecipe {
    LandformRecipe {
        revision: 1,
        seed: 419,
        iterations: 32,
        backend_revision: 1,
        rain_scale: 1.0,
        erosion_rate: 0.0005,
    }
}

fn fixture() -> (GlobalTopology, Vec<i32>, Vec<f32>) {
    let topology = GlobalTopology::new(4).unwrap();
    let units = topology
        .keys()
        .iter()
        .map(|k| k.0[0] * 111 - k.0[1] * 71 + k.0[2] * 99)
        .collect();
    let streams = vec![0.0; topology.nodes().len()];
    (topology, units, streams)
}

#[test]
fn snapshot_roundtrip_and_content_identity() {
    let (topology, units, streams) = fixture();
    let snapshot = LandformSnapshot::from_units(&topology, recipe(), &units, &streams).unwrap();
    let bytes = snapshot.encode();
    let decoded = LandformSnapshot::decode(&bytes, snapshot.id()).unwrap();
    assert_eq!(decoded.encode(), bytes);
    assert_eq!(decoded.recipe(), recipe());
    assert_eq!(decoded.resolution(), 4);
    assert_eq!(decoded.id().to_string().len(), 64);
    for field in 0..6 {
        let mut changed = recipe();
        match field {
            0 => changed.seed += 1,
            1 => changed.revision += 1,
            2 => changed.iterations += 1,
            3 => changed.backend_revision += 1,
            4 => changed.rain_scale += 0.1,
            _ => changed.erosion_rate += 0.01,
        }
        assert_ne!(
            snapshot.id(),
            LandformSnapshot::from_units(&topology, changed, &units, &streams)
                .unwrap()
                .id()
        );
    }
    let mut changed = units.clone();
    changed[0] += 1;
    assert_ne!(
        snapshot.id(),
        LandformSnapshot::from_units(&topology, recipe(), &changed, &streams)
            .unwrap()
            .id()
    );
    let mut changed = streams.clone();
    changed[0] = 1.0;
    assert_ne!(
        snapshot.id(),
        LandformSnapshot::from_units(&topology, recipe(), &units, &changed)
            .unwrap()
            .id()
    );
    // Zero is canonicalized during creation; a byte encoding with -0 is rejected.
    let mut negative_zero = recipe();
    negative_zero.rain_scale = -0.0;
    let mut positive_zero = negative_zero;
    positive_zero.rain_scale = 0.0;
    assert_eq!(
        LandformSnapshot::from_units(&topology, negative_zero, &units, &vec![-0.0; streams.len()])
            .unwrap()
            .id(),
        LandformSnapshot::from_units(&topology, positive_zero, &units, &streams)
            .unwrap()
            .id()
    );
}

// Rehash structural mutations so parser rejection cannot be explained by the
// outer checksum alone. The digest is identity/integrity, not a trust signature.
fn reseal(bytes: &mut [u8]) -> SnapshotId {
    let end = bytes.len() - 32;
    let digest: [u8; 32] = Sha256::digest(&bytes[..end]).into();
    bytes[end..].copy_from_slice(&digest);
    SnapshotId(digest)
}

#[test]
fn snapshot_rejects_corruption_and_noncanonical_data() {
    let (topology, units, streams) = fixture();
    let snapshot = LandformSnapshot::from_units(&topology, recipe(), &units, &streams).unwrap();
    let bytes = snapshot.encode();
    for length in [0, 7, 79, 80, bytes.len() - 1] {
        assert!(LandformSnapshot::decode(&bytes[..length], snapshot.id()).is_err());
    }
    assert!(LandformSnapshot::decode(&bytes, SnapshotId([0; 32])).is_err());
    for offset in [0, 8, 48, bytes.len() - 1] {
        let mut corrupt = bytes.clone();
        corrupt[offset] ^= 1;
        assert!(LandformSnapshot::decode(&corrupt, snapshot.id()).is_err());
    }
    let stream_start = 48 + snapshot.heights.len() * 4;
    for (offset, word) in [
        (8, 99u32),
        (12, 99),
        (16, 0),
        (16, 3),
        (16, 512),
        (20, 0),
        (32, 0),
        (36, f32::NAN.to_bits()),
        (40, (-1.0f32).to_bits()),
        (36, (-0.0f32).to_bits()),
        (44, 1),
        (48, (MAX_HEIGHT_UNITS + 1) as u32),
        (stream_start, f32::INFINITY.to_bits()),
        (stream_start, f32::NAN.to_bits()),
        (stream_start, (-0.0f32).to_bits()),
        (stream_start, (-1.0f32).to_bits()),
    ] {
        let mut corrupt = bytes.clone();
        corrupt[offset..offset + 4].copy_from_slice(&word.to_le_bytes());
        let id = reseal(&mut corrupt);
        assert!(
            LandformSnapshot::decode(&corrupt, id).is_err(),
            "accepted offset {offset}, word {word}"
        );
    }
    // First face corner has matching entries elsewhere; change only one copy.
    for offset in [48, stream_start] {
        let mut corrupt = bytes.clone();
        let word = if offset == 48 {
            snapshot.heights[0].wrapping_add(1) as u32
        } else {
            1.0f32.to_bits()
        };
        corrupt[offset..offset + 4].copy_from_slice(&word.to_le_bytes());
        let id = reseal(&mut corrupt);
        assert!(LandformSnapshot::decode(&corrupt, id)
            .err()
            .unwrap()
            .contains("boundary"));
    }
}

#[test]
fn snapshot_quantization_and_input_limits() {
    let (topology, units, streams) = fixture();
    let mut state = vec![[0.0, 0.0]; units.len()];
    // Exact binary halves of the 0.05m unit: 0.125*20 = 2.5.
    state[0][0] = 0.125;
    state[1][0] = -0.125;
    let snapshot = LandformSnapshot::from_state(&topology, recipe(), &state).unwrap();
    for (i, node) in topology.face_nodes().iter().enumerate() {
        assert_eq!(
            snapshot.heights[i],
            match node {
                0 => 3,
                1 => -3,
                _ => 0,
            }
        );
    }
    for height in [f32::NAN, f32::INFINITY, 2_000_001.0, -2_000_001.0] {
        state[0][0] = height;
        assert!(LandformSnapshot::from_state(&topology, recipe(), &state).is_err());
    }
    let mut bad = units.clone();
    bad[0] = i32::MIN;
    assert!(LandformSnapshot::from_units(&topology, recipe(), &bad, &streams).is_err());
    assert!(LandformSnapshot::from_units(&topology, recipe(), &units[..1], &streams).is_err());
}

#[test]
fn integer_cell_address_and_sphere_boundary() {
    let topology = GlobalTopology::new(16).unwrap();
    let zeroes = vec![0; topology.nodes().len()];
    let streams = vec![0.0; zeroes.len()];
    let sphere = LandformSnapshot::from_units(&topology, recipe(), &zeroes, &streams).unwrap();
    for (cell, face, fraction) in [
        ([0, 0, 0], 0, [65536, 65536]),
        ([-1, -1, -1], 1, [0, 0]),
        ([0, 1, 1], 2, [43690, 65536]),
        ([0, 0, 1], 4, [43690, 43690]),
    ] {
        let a = FaceAddress::from_cell(cell, 16).unwrap();
        assert_eq!(a.face, face);
        assert_eq!(a.fraction, fraction);
    }
    for n in [0, 3, 512] {
        assert!(FaceAddress::from_cell([0; 3], n).is_none());
    }
    for x in [i32::MIN, -100_000_001, 100_000_001, i32::MAX] {
        assert!(sphere.sample_cell([x, 0, 0]).is_none());
    }
    for axis in 0..3 {
        for negative in [false, true] {
            let mut inside = [0; 3];
            inside[axis] = 63_709_999;
            let mut outside = inside;
            outside[axis] += 1;
            if negative {
                inside[axis] = -inside[axis] - 1;
                outside[axis] = -outside[axis] - 1;
            }
            assert!(sphere.sample_cell(inside).unwrap().solid);
            assert!(!sphere.sample_cell(outside).unwrap().solid);
        }
    }
}
