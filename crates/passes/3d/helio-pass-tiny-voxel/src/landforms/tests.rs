use super::*;
use std::collections::VecDeque;

#[test]
fn shared_sphere_topology_closes_edges_and_corners() {
    assert_eq!(std::mem::size_of::<GpuNode>(), 112);
    for n in [1, 2, 4, 8, 16, 32] {
        let topology = GlobalTopology::new(n).unwrap();
        let reverse = GlobalTopology::with_face_order(n, [5, 4, 3, 2, 1, 0]).unwrap();
        assert_eq!(topology.keys.len(), (6 * n * n + 2) as usize);
        assert_eq!(topology.keys, reverse.keys);
        assert_eq!(topology.face_nodes, reverse.face_nodes);
        assert_eq!(
            bytemuck::cast_slice::<_, u8>(&topology.nodes),
            bytemuck::cast_slice::<_, u8>(&reverse.nodes)
        );
        assert_eq!(topology.areas, reverse.areas);
        assert!(topology.keys.windows(2).all(|w| w[0] < w[1]));
        let mut occurrences = vec![0; topology.keys.len()];
        for id in &topology.face_nodes {
            occurrences[*id as usize] += 1;
        }
        assert_eq!(occurrences.iter().filter(|c| **c == 3).count(), 8);
        assert_eq!(
            occurrences.iter().filter(|c| **c == 2).count(),
            12 * (n as usize - 1)
        );
        assert_eq!(
            occurrences.iter().filter(|c| **c == 1).count(),
            6 * (n as usize - 1).pow(2)
        );
        let mut reached = vec![false; topology.nodes.len()];
        reached[0] = true;
        let mut queue = VecDeque::from([0]);
        while let Some(i) = queue.pop_front() {
            let node = &topology.nodes[i];
            let corner = topology.keys[i].0.iter().all(|x| x.abs() == 1);
            assert_eq!(node.degree, if corner { 6 } else { 8 });
            assert!(node.rain.is_finite() && node.rain > 0.0);
            let adjacent = &node.neighbors[..node.degree as usize];
            assert!(adjacent.windows(2).all(|w| w[0] < w[1]));
            for slot in 0..node.degree as usize {
                let j = node.neighbors[slot] as usize;
                assert_ne!(i, j);
                let back = node.reverse[slot] as usize;
                assert!(back < topology.nodes[j].degree as usize);
                assert_eq!(topology.nodes[j].neighbors[back], i as u32);
                assert_eq!(topology.nodes[j].reverse[back], slot as u32);
                let inverse = node.inverse_distance[slot];
                assert!(inverse.is_finite() && inverse > 0.0);
                assert_eq!(
                    inverse.to_bits(),
                    topology.nodes[j].inverse_distance[back].to_bits()
                );
                if !reached[j] {
                    reached[j] = true;
                    queue.push_back(j);
                }
            }
        }
        assert!(reached.iter().all(|x| *x));
        let expected = 4.0 * std::f64::consts::PI * crate::world::RADIUS.powi(2);
        let total = topology.areas.iter().sum::<f64>();
        assert!(
            (total / expected - 1.0).abs() < 1e-11,
            "n={n} area={total} expected={expected}"
        );
    }
}

#[test]
fn nested_spherical_directions_keep_their_identity() {
    for n in [1, 2, 4, 8, 16] {
        let coarse = GlobalTopology::new(n).unwrap();
        let fine = GlobalTopology::new(n * 2).unwrap();
        for face in 0..6 {
            for v in 0..=n {
                for u in 0..=n {
                    let a = coarse.face_node(face, u, v).unwrap() as usize;
                    let b = fine.face_node(face, u * 2, v * 2).unwrap() as usize;
                    assert_eq!(coarse.keys[a], fine.keys[b]);
                    assert_eq!(coarse.directions[a], fine.directions[b]);
                }
            }
        }
    }
    for n in [0, 3, 512] {
        assert!(GlobalTopology::new(n).is_err());
    }
}
