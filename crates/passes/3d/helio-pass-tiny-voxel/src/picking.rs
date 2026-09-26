//! Exact interaction rays. Rendering may filter subpixel coverage; edits must
//! select an actual occupied cell using double-precision world coordinates.
#[cfg(test)]
use crate::world::RADIUS;
use crate::world::{World, VOXEL};
use glam::DVec3;

impl World {
    /// Cache exact voxels near excavated or constructed surfaces as well as
    /// the original terrain, including deep planetary craters.
    pub fn near_surface(&self, position: DVec3, range: f64) -> bool {
        if self.density(crate::world::render_origin(position)).abs() < range {
            return true;
        }
        let cell = crate::world::cell_of(position);
        let r = (range * 10.0).ceil() as i32;
        for i in self.region_edits(cell.map(|v| v - r), cell.map(|v| v + r)) {
            let e = self.edits[i];
            if (position.distance(crate::world::center(e.cell)) - e.canonical_radius()).abs()
                < range
            {
                return true;
            }
        }
        false
    }
    pub fn bounds(&self) -> (DVec3, DVec3) {
        let mut lo = DVec3::splat(-crate::world::procedural_outer_radius());
        let mut hi = -lo;
        if let Some((minimum, maximum)) = self.edit_index.bounds {
            lo = lo.min(DVec3::from_array(minimum.map(f64::from)) * VOXEL - DVec3::ONE);
            hi = hi.max(DVec3::from_array(maximum.map(f64::from)) * VOXEL + DVec3::ONE);
        }
        (lo, hi)
    }
    pub fn outer_radius(&self) -> f64 {
        let (lo, hi) = self.bounds();
        lo.abs().max(hi.abs()).length()
    }
    /// First occupied 10 cm cell, preceding air cell, and distance in metres.
    /// Clip against world bounds before converting positions to cell indices,
    /// so distant cameras do not overflow the integer voxel address space.
    pub fn raycast(
        &self,
        origin: DVec3,
        direction: DVec3,
        maximum: f64,
    ) -> Option<([i32; 3], [i32; 3], f64)> {
        if !origin.is_finite()
            || !direction.is_finite()
            || direction.length_squared() < 1e-20
            || maximum <= 0.0
        {
            return None;
        }
        let dir = direction.normalize();
        let (lo, hi) = self.bounds();
        let mut entry = 0.0_f64;
        let mut exit = maximum;
        for a in 0..3 {
            if dir[a].abs() < 1e-15 {
                if origin[a] < lo[a] || origin[a] > hi[a] {
                    return None;
                }
                continue;
            }
            let x = (lo[a] - origin[a]) / dir[a];
            let y = (hi[a] - origin[a]) / dir[a];
            entry = entry.max(x.min(y));
            exit = exit.min(x.max(y));
        }
        if exit < entry {
            return None;
        }
        self.cast_hit_impl(origin + dir * entry, dir, exit - entry, true)
            .map(|(cell, air, t)| (cell, air, t + entry))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::{center, Edit};
    #[test]
    fn precise_selection_and_single_voxel_destruction_across_scales_and_hemispheres() {
        for radial in [
            DVec3::X,
            -DVec3::X,
            DVec3::Y,
            -DVec3::Y,
            DVec3::Z,
            -DVec3::Z,
            DVec3::new(1.0, 2.0, -3.0).normalize(),
        ] {
            let world = World::default();
            let surface = world
                .raycast(radial * (RADIUS + 2_000_000.0), -radial, f64::INFINITY)
                .unwrap()
                .0;
            let target = center(surface);
            let reference = world.cast_hit(target + radial * 3.0, -radial, 6.0).unwrap();
            for distance in [
                3.0,
                100.0,
                10_000.0,
                300_000.0,
                8_000_000.0,
                1_000_000_000.0,
            ] {
                let origin = target + radial * distance;
                let hit = world.raycast(origin, -radial, f64::INFINITY).unwrap();
                assert_eq!(hit.0, reference.0, "radial={radial} distance={distance}");
                assert_eq!(world.material(hit.0), 1);
                assert_eq!(world.material(hit.1), 0);
                let mut edited = World::default();
                edited.edits.push(Edit {
                    cell: hit.0,
                    radius: 0.05,
                    material: 0,
                });
                edited.rebuild_edits();
                let next = edited.raycast(origin, -radial, f64::INFINITY).unwrap();
                assert_ne!(next.0, hit.0);
                assert!(next.2 > hit.2);
            }
        }
    }
    #[test]
    fn selection_reaches_remote_construction_and_rejects_sky() {
        let mut world = World::default();
        world.edits.push(Edit {
            cell: [0, -180_000_000, 0],
            radius: 1_000_000.0,
            material: 3,
        });
        world.rebuild_edits();
        let origin = DVec3::new(0.05, -1_000_000_000.0, 0.05);
        let hit = world.raycast(origin, DVec3::Y, f64::INFINITY).unwrap();
        assert_eq!(world.material(hit.0), 3);
        assert!(world.raycast(origin, -DVec3::Y, f64::INFINITY).is_none());
    }
    #[test]
    fn deep_planetary_excavations_activate_the_exact_near_cache() {
        let mut world = World::default();
        let c = [0, 63_710_000, 0];
        world.edits.push(Edit {
            cell: c,
            radius: 800_000.0,
            material: 0,
        });
        world.rebuild_edits();
        let floor = center(c) - DVec3::Y * 799_998.0;
        assert!(world.density(crate::world::cell_of(floor)).abs() > 700_000.0);
        assert!(world.near_surface(floor, 80.0));
        assert!(!world.near_surface(center(c) - DVec3::Y * 400_000.0, 80.0));
    }
}
