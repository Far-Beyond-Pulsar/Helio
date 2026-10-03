//! Ordering of accepted GPU demand; this never decides terrain membership.
use crate::grid::{Grid, BRICK};
use glam::DVec3;

pub(crate) fn order_blocks(
    grid: &Grid, eye: DVec3, forward: DVec3, up: DVec3, projection_y: f32,
    size: [u32; 2], ground_radial: f64, blocks: Vec<(u32, u32)>, urgent_blocks: usize,
) -> Vec<(u32, u32)> {
    let projection = projection_basis(grid, eye, forward, up, projection_y, size, ground_radial);
    let urgent_blocks = urgent_blocks.min(blocks.len());
    let mut unique: Vec<_> = blocks.into_iter().enumerate()
        .map(|(index, key)| (key, index < urgent_blocks)).collect();
    unique.sort_unstable_by_key(|entry| entry.0);
    // An ordinary duplicate cannot downgrade a request from the urgent bank.
    let mut merged: Vec<((u32, u32), bool)> = Vec::with_capacity(unique.len());
    for (key, urgent) in unique {
        if let Some(last) = merged.last_mut().filter(|last| last.0 == key) {
            last.1 |= urgent;
        } else {
            merged.push((key, urgent));
        }
    }
    let mut ranked: Vec<_> = merged.into_iter().map(|(key, urgent)| {
        let point = block_center(grid, key);
        let malformed = point.is_none();
        let distance = point.map(|point| grid.ground_distance(eye, point))
            .filter(|distance| distance.is_finite()).unwrap_or(f64::INFINITY);
        let class = match (point, projection) {
            (Some(point), Some((forward, right, up, scale, aspect))) => {
                // The local ground radius is a cheap centre proxy, not a
                // terrain sample. Mountains and block edges can move it out
                // of the view; even those requests remain in the result.
                let delta = grid.at_radial(point, ground_radial) - eye;
                let depth = delta.dot(forward);
                if depth <= 0.0 { 2u8 }
                else if delta.dot(up).abs() * scale <= depth
                    && delta.dot(right).abs() * scale <= depth * aspect { 0 }
                else { 1 }
            }
            _ => 0,
        };
        (malformed, !urgent, class, distance, key)
    }).collect();
    ranked.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1))
        .then(a.2.cmp(&b.2)).then(a.3.total_cmp(&b.3)).then(a.4.cmp(&b.4)));
    ranked.into_iter().map(|entry| entry.4).collect()
}

fn projection_basis(
    grid: &Grid, eye: DVec3, forward: DVec3, up: DVec3, projection_y: f32,
    size: [u32; 2], ground_radial: f64,
) -> Option<(DVec3, DVec3, DVec3, f64, f64)> {
    if !eye.is_finite() || !ground_radial.is_finite()
        || (!grid.is_plane() && (ground_radial <= 0.0 || eye.length_squared() <= 0.0))
        || !projection_y.is_finite() || projection_y <= 0.0 || size.contains(&0)
        || !forward.is_finite() || !up.is_finite() { return None; }
    let forward = forward.try_normalize()?;
    let right = forward.cross(up).try_normalize()?;
    let up = right.cross(forward);
    Some((forward, right, up, f64::from(projection_y), f64::from(size[0]) / f64::from(size[1])))
}

fn block_center(grid: &Grid, key: (u32, u32)) -> Option<DVec3> {
    let face = ((key.0 >> 24) & 7) as u8;
    let level = key.0 >> 27;
    let i = key.0 & 0xffffff;
    let j = key.1 as i32;
    if !grid.faces().contains(&face) || level >= grid.levels() || i & 3 != 0 || j & 3 != 0 || j < 0 {
        return None;
    }
    let cells = i64::from(BRICK) << level;
    let columns = i64::from(grid.cells()) / cells;
    if i64::from(i) + 3 >= columns || i64::from(j) + 3 >= columns { return None; }
    let point = grid.ground_point(face, (f64::from(i) + 2.0) * cells as f64,
        (f64::from(j) + 2.0) * cells as f64);
    point.is_finite().then_some(point)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Shape;

    fn key(face: u8, level: u32, i: i32, j: i32) -> (u32, u32) {
        ((level << 27) | (u32::from(face) << 24) | i as u32, j as u32)
    }
    fn plane() -> Grid { Grid::plane(Shape::Plane, 1024.0, 0.1).unwrap() }
    fn plane_keys(grid: &Grid) -> [(u32, u32); 3] {
        let mid = (grid.origin_index() / BRICK) & !3;
        [key(2, 0, mid, mid - 32), key(2, 0, mid, mid + 4), key(2, 0, mid + 96, mid - 32)]
    }
    fn order(grid: &Grid, blocks: Vec<(u32, u32)>, urgent: usize) -> Vec<(u32, u32)> {
        order_blocks(grid, DVec3::new(0.0, 10.0, 0.0), DVec3::Z, DVec3::Y,
            1.0, [800, 800], 0.0, blocks, urgent)
    }

    #[test]
    fn urgent_precedes_even_visible_ordinary_and_duplicate_upgrades() {
        let grid = plane();
        let [front, behind, _] = plane_keys(&grid);
        assert_eq!(order(&grid, vec![behind, front, behind], 1), vec![behind, front]);
    }

    #[test]
    fn current_view_beats_nearer_previous_view_and_offscreen_centres() {
        let grid = plane();
        let [front, behind, outside] = plane_keys(&grid);
        assert_eq!(order(&grid, vec![behind, outside, front], 0), vec![front, outside, behind]);
        let rotated = order_blocks(&grid, DVec3::new(0.0, 10.0, 0.0), -DVec3::Z,
            DVec3::Y, 1.0, [800, 800], 0.0, vec![front, behind], 0);
        assert_eq!(rotated, vec![behind, front]);
    }

    #[test]
    fn invalid_projection_disables_view_classes_for_every_key() {
        let grid = plane();
        let [front, behind, _] = plane_keys(&grid);
        for (forward, up, projection, size, radial) in [
            (DVec3::Z, DVec3::Y, f32::NAN, [800, 800], 0.0),
            (DVec3::Z, DVec3::Y, 1.0, [0, 800], 0.0),
            (DVec3::Z, DVec3::Z, 1.0, [800, 800], 0.0),
            (DVec3::ZERO, DVec3::Y, 1.0, [800, 800], 0.0),
            (DVec3::Z, DVec3::Y, 1.0, [800, 800], f64::INFINITY),
        ] {
            assert_eq!(order_blocks(&grid, DVec3::new(0.0, 10.0, 0.0), forward, up,
                projection, size, radial, vec![front, behind], 0), vec![behind, front]);
        }
    }

    #[test]
    fn malformed_keys_are_retained_last_with_deterministic_ties() {
        let grid = plane();
        let [front, behind, _] = plane_keys(&grid);
        let malformed = [key(7, 0, 0, 0), key(2, 31, 0, 0), key(2, 0, 1, 0), key(2, 0, 0, -4)];
        let mut input = vec![malformed[0], behind, front];
        input.extend(malformed);
        let output = order(&grid, input, 1);
        assert_eq!(&output[..2], &[front, behind]);
        let mut expected = malformed.to_vec(); expected.sort_unstable();
        // The urgent malformed key stays ahead of the other malformed keys.
        expected.retain(|key| *key != malformed[0]); expected.insert(0, malformed[0]);
        assert_eq!(&output[2..], expected);
        assert_eq!(order(&grid, vec![behind, front], usize::MAX), vec![front, behind]);
    }

    #[test]
    fn equal_distance_ties_and_cube_seams_have_no_face_priority() {
        let flat = plane();
        let mid = (flat.origin_index() / BRICK) & !3;
        let left = key(2, 0, mid - 4, mid - 32);
        let right = key(2, 0, mid, mid - 32);
        assert_eq!(order(&flat, vec![right, left], 0), vec![left, right]);
        assert_eq!(order(&flat, vec![left, right], 0), vec![left, right]);
        let grid = Grid::new(3_000_000.0, 0.1).unwrap();
        let columns = grid.cells() / BRICK;
        let mid = (columns / 2) & !3;
        let a = key(0, 0, 0, mid);
        let b = key(4, 0, columns - 4, mid);
        let ca = block_center(&grid, a).unwrap();
        let cb = block_center(&grid, b).unwrap();
        let direction = (ca + cb).normalize();
        let eye = direction * (grid.radius() + 1000.0);
        let expected = order_blocks(&grid, eye, -direction, DVec3::Y, 1.0,
            [800, 800], grid.radius(), vec![a, b], 0);
        assert_eq!(expected, order_blocks(&grid, eye, -direction, DVec3::Y, 1.0,
            [800, 800], grid.radius(), vec![b, a, b], 0));
        assert!(ca.distance(cb) < 10.0);
        assert_eq!(expected.len(), 2);
    }
}
